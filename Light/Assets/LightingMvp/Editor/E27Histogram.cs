using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using UnityEngine;
using Lm = LightingMvp.ImageJacobianLm;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            private sealed class E27Histogram
            {
                private readonly int bins, window, stride, nx, ny;
                private readonly float[] target, cdf, pos, mix;
                private readonly float[] lower = new float[3], upper = new float[3];
                private readonly double[] weights = new double[3];
                private readonly E23FrameObjective map;
                internal int Size => target.Length;
                internal double[] FrozenRange => lower.Concat(upper).Select(x => (double)x).ToArray();
                internal E27Histogram(float[] raw, E23FrameObjective objective, int w, int b)
                {
                    map = objective; window = w; bins = b; stride = w / 2;
                    nx = (E19FitW - window) / stride + 1;
                    ny = (E19FitH - window) / stride + 1;
                    target = new float[3 * nx * ny * 3 * bins]; cdf = new float[target.Length];
                    pos = new float[raw.Length]; mix = new float[raw.Length];
                    for (int c = 0; c < 3; ++c)
                    {
                        var v = Enumerable.Range(0, raw.Length / 3).Select(i => map.Transform(raw[i * 3 + c])).ToArray();
                        lower[c] = E24Quantile(v, (int)(.01 * (v.Length - 1)));
                        upper[c] = Math.Max(lower[c] + 1e-5f, E24Quantile(v, (int)(.99 * (v.Length - 1))));
                    }
                    Histogram(raw, target);
                    int block = target.Length / 3;
                    for (int v = 0; v < 3; ++v) weights[v] = 1 / Math.Sqrt(3 * (E21Dot(target, target, v * block, block) + 1e-12));
                }
                private void Histogram(float[] image, float[] dst)
                {
                    Array.Clear(dst, 0, dst.Length);
                    // The first/last bins include values outside the frozen target quantiles.
                    for (int i = 0; i < image.Length; ++i)
                    {
                        int c = i % 3;
                        float x = 1 + (bins - 3) * (map.Transform(image[i]) - lower[c]) / (upper[c] - lower[c]);
                        x = Math.Min(bins - 1, Math.Max(0, x));
                        pos[i] = (int)x; mix[i] = x - pos[i];
                    }
                    float inv = 1f / (window * window);
                    int k = 0;
                    for (int v = 0; v < 3; ++v)
                    for (int wy = 0; wy < ny; ++wy)
                    for (int wx = 0; wx < nx; ++wx)
                    {
                        int at = v * E23ChannelsPerView + (wy * stride * E19FitW + wx * stride) * 3;
                        for (int dy = 0; dy < window; ++dy)
                        for (int dx = 0; dx < window; ++dx)
                        for (int c = 0; c < 3; ++c)
                        {
                            int p = at + (dy * E19FitW + dx) * 3 + c;
                            int index = k + c * bins + (int)pos[p];
                            dst[index] += 1 - mix[p];
                            if (mix[p] != 0) dst[index + 1] += mix[p];
                        }
                        for (int c = 0; c < 3; ++c)
                        {
                            float cumulative = 0;
                            for (int b = 0; b < bins; ++b, ++k) { cumulative += dst[k]; dst[k] = cumulative * inv; }
                        }
                    }
                }
                internal double Evaluate(float[] image, float[] output, int offset, double factor)
                {
                    Histogram(image, cdf);
                    double f = 0; int block = cdf.Length / 3;
                    for (int i = 0; i < cdf.Length; ++i)
                    {
                        double z = (cdf[i] - target[i]) * weights[i / block] * factor;
                        if (output != null) output[offset + i] = (float)z;
                        f += z * z;
                    }
                    return f;
                }
            }

            private sealed partial class E27Trial
            {
                private void HistogramPolicy()
                {
                    // LOIR-inspired, not a reproduction of their estimator. Window = bin count here.
                    string policy = string.IsNullOrEmpty(arm.selected) ? arm.policy : arm.selected;
                    bool useRgb = !policy.StartsWith("hist", StringComparison.Ordinal);
                    bool useHist = policy != "rgb";
                    int w = policy.EndsWith("32", StringComparison.Ordinal) ? 32 : 16;
                    var hist = useHist ? new E27Histogram(target.packed, main, w, w) : null;
                    int offset = useRgb ? main.Size : 0, length = offset + (hist?.Size ?? 0);
                    double factor = useRgb && useHist ? Math.Sqrt(.5) : 1;
                    var rig = sample.initial; var rgb = new float[main.Size];
                    segment = "hist_fixed";
                    double before = Full(rig, "proxy_initial_full");
                    using var csv = Proposals();
                    var x0 = Local(rig);
                    var settings = new Lm.TrustSettings { Layout = layout, Iterations = 1, Budget = (arm.cap - 3) / 3, FixedProbeScale = true,
                        NormalMode = Lm.TrustNormalMode.LampScaledDense, ProbeAllowed = LegalLocal, CandidateAllowed = LegalLocal, FinalAllowed = LegalLocal };
                    bool LegalLocal(double[] x) => Legal(Expand(x, rig));
                    var sh = E27Mode(sample);
                    double Evaluate(float[] img, float[] residual)
                    {
                        long start = Stopwatch.GetTimestamp(); double f = 0;
                        if (useRgb) f = main.Evaluate(img, residual, factor: factor) * factor * factor;
                        if (useHist) f += hist.Evaluate(img, residual, offset, factor);
                        mapSeconds += Seconds(start);
                        return f;
                    }
                    if (arm.method == "C3")
                    {
                        var hard = new float[main.Size]; var off = new float[main.Size]; var raw = new float[main.Size];
                        var anchor = new float[length]; var plus = new float[length]; var minus = new float[length];
                        settings.PositionModel = new Lm.TrustPositionModel(2, 1, (q, x) =>
                        {
                            var pose = Expand(x, rig);
                            Capture(pose, LightShadows.Hard, E27Views, hard); Query("anchor", LightShadows.Hard, pose, E27Views);
                            Capture(pose, LightShadows.None, E27Views, off); Query("anchor", LightShadows.None, pose, E27Views);
                            Evaluate(hard, anchor);
                        }, (q, x, axis, d, stencil, column) =>
                        {
                            void Probe(double dx, float[] y)
                            {
                                var local = (double[])x.Clone(); local[axis] += dx; var pose = Expand(local, rig);
                                Capture(pose, LightShadows.None, E27Views, raw);
                                Query("xyz", LightShadows.None, pose, E27Views, axis: indices[axis], step: dx * layout.Axis(axis).Unit);
                                for (int i = 0; i < raw.Length; ++i) raw[i] = hard[i] + (raw[i] - off[i]);
                                Evaluate(raw, y);
                            }
                            Probe(d, plus);
                            if (stencil != 1) Probe(stencil == 3 ? -d : 2 * d, minus);
                            var (a, b) = E24Weights(d, stencil);
                            for (int i = 0; i < length; ++i) column[i] = (float)(a * (plus[i] - anchor[i]) + b * (stencil == 1 ? 0 : minus[i] - anchor[i]));
                        });
                    }
                    double fFixed = before;
                    bool tested = false;
                    settings.BuildModelAtZero = true;
                    settings.FinishAfterModel = j =>
                    {
                        var d = Lm.LampScaledTrustProposal(j.Gram, j.Gradient, x0, layout, LegalLocal, out int shrinks);
                        tested = Try(d, arm.policy, shrinks, null);
                        if (!tested)
                        {
                            bool LampLegal(double[] x, int owner) => h.room.Contains(Expand(x, rig).Skip(4 * sample.active[owner]).Take(4).ToArray());
                            foreach (bool cauchy in new[] { false, true })
                            {
                                var local = new int[layout.Lamps];
                                d = Lm.LampScaledTrustProposal(j.Gram, j.Gradient, x0, layout, LegalLocal, out shrinks,
                                    cauchy: cauchy, lampLegal: LampLegal, lampShrinks: local);
                                tested = Try(d, arm.policy + (cauchy ? "-lamp-cauchy" : "-lamp-lm"), shrinks, local);
                                if (tested) break;
                            }
                        }
                        return true;

                        bool Try(double[] step, string source, int shrunk, int[] perLamp)
                        {
                            var move = Expand(x0.Zip(step, (x, t) => x + t).ToArray(), rig);
                            bool legal = Legal(move); bool eval = legal && step.Any(t => t != 0);
                            if (eval) fFixed = Full(move, "proxy_fixed_full");
                            csv.WriteLine(string.Join(",", arm.id, source, "1", result.cameras, result.auditCameras, Flag(legal), Flag(eval), shrunk,
                                F(Lm.TrustPredicted(j.Gram, j.Gradient, step)), eval ? F(before - fFixed) : "", "", "",
                                T(step.Select((z, i) => z * layout.Axis(i).Unit).ToArray()), T(move), perLamp == null ? "" : string.Join(";", perLamp)));
                            return eval;
                        }
                    };
                    Run(x0, "hist_fixed");
                    if (!tested) { incomplete = true; result.reason = "no feasible, nonzero fixed proxy proposal"; }
                    if (policy.EndsWith("32", StringComparison.Ordinal))
                    { result.stop = "fixed_32px_sensitivity"; WriteHistogramConfig(); return; }
                    settings.FinishAfterModel = null;
                    settings.Iterations = 3;
                    settings.BuildModelAtZero = false;
                    settings.StallModelLimit = 3;
                    settings.ReserveRemainingModels = true;
                    settings.Budget = (arm.cap - result.cameras - 15) / 3;
                    settings.Trial = t =>
                    {
                        if (t.Accepted) Full(Expand(t.Theta, rig), "proxy_branch_accept");
                        decisions.WriteLine(string.Join(",", arm.id, result.cameras, segment, t.Iteration, t.Source, t.Status, Flag(t.Legal), Flag(t.Accepted),
                            F(t.Predicted), F(t.Actual), F(t.Ratio), t.Shrinks, F(t.CurrentLoss), F(bestLoss), T(t.Step), T(Expand(t.Theta, rig)), "", "", "", F(t.UnclippedPrediction), t.Constraint,
                            T(t.Step?.Select((d, i) => d * layout.Axis(i).Unit).ToArray()), F(t.ReferenceLoss)));
                    };
                    var search = Run(x0, "hist_branch");
                    if (search.Jacobians < 3 && search.Stop != "near_zero_residual")
                    { incomplete = true; result.reason = "unfulfilled three-model proxy branch"; }
                    bool improved = bestLoss < before - Math.Max(1e-14, 1e-9 * before);
                    result.explored = improved;
                    result.stop = search.Stop;
                    WriteHistogramConfig();
                    void WriteHistogramConfig() => File.WriteAllText(Path.Combine(dir, "histogram-config.csv"),
                        "policy,rgb,histogram,window,bins,stride,scaled_range,fixed_true_drop,best_true_drop\n" +
                        string.Join(",", policy, Flag(useRgb), Flag(useHist), useHist ? w : 0, useHist ? w : 0, useHist ? w / 2 : 0,
                            T(hist?.FrozenRange), F(before - fFixed), F(before - bestLoss)) + "\n");
                    Lm.TrustResult Run(double[] theta, string name)
                    {
                        segment = name;
                        settings.Jacobian = j => { ++result.models; if (result.models is 4 or 8 or 16) Checkpoint("model", result.models); };
                        return Lm.RunTrust(theta, length, (x, phase, r) =>
                        {
                            var pose = Expand(x, rig); Capture(pose, sh, E27Views, rgb); double f = Evaluate(rgb, r);
                            Query(phase, sh, pose, E27Views, useHist ? double.NaN : f, f); return f;
                        }, settings);
                    }
                }
            }
        }
    }
}

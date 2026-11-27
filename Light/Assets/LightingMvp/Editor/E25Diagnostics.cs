using System;
using System.IO;
using System.Linq;
using UnityEngine;
using Frame = LightingMvp.NativeRenderBackend.Frame;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            private sealed class E25LocalModel
            {
                internal float[] anchor, residual;
                internal float[][] cols;
                internal double[] steps;
                internal int[] stencils;
            }

            private (double d, int stencil) E25Stencil(double[] at, int axis, double physical, Func<double[], bool> legal = null)
            {
                legal ??= room.Contains;
                var moved = (double[])at.Clone();
                double h = physical / units[axis % 4];
                for (int n = 0; n < 8; ++n, h *= .5)
                {
                    moved[axis] = at[axis] + h; bool plus = legal(moved);
                    moved[axis] = at[axis] - h; bool minus = legal(moved);
                    if (!plus && !minus) continue;
                    if (plus && minus) return (h, 3);
                    double d = plus ? h : -h;
                    moved[axis] = at[axis] + 2 * d;
                    return (d, legal(moved) ? 2 : 1);
                }
                return (0, 0);
            }

            private E25LocalModel E25Local(double[] rig, int[] active, double step, E23FrameObjective objective, Action<double[], float[]> capture, Func<double[], bool> legal = null)
            {
                int n = 4 * active.Length, length = objective.Size;
                var m = new E25LocalModel { anchor = new float[length], residual = new float[length],
                    cols = Enumerable.Range(0, n).Select(_ => new float[length]).ToArray(), steps = new double[n], stencils = new int[n] };
                var plus = new float[length]; var minus = new float[length];
                capture(rig, m.anchor);
                objective.Evaluate(m.anchor, m.residual);
                for (int j = 0; j < n; ++j)
                {
                    int axis = 4 * active[j / 4] + j % 4;
                    var (d, stencil) = E25Stencil(rig, axis, j % 4 == 3 ? .04 : step, legal);
                    m.steps[j] = d * units[j % 4]; m.stencils[j] = stencil;
                    if (stencil == 0) continue;
                    var moved = (double[])rig.Clone();
                    moved[axis] = rig[axis] + d;
                    capture(moved, plus);
                    if (stencil != 1)
                    {
                        moved[axis] = rig[axis] + (stencil == 3 ? -d : 2 * d);
                        capture(moved, minus);
                    }
                    var (a, b) = E24Weights(d, stencil);
                    for (int i = 0; i < length; ++i)
                    {
                        float center = objective.Transform(m.anchor[i]);
                        double p = objective.Transform(plus[i]) - center;
                        double q = stencil == 1 ? 0 : objective.Transform(minus[i]) - center;
                        m.cols[j][i] = (float)(objective.weights[i / E23ChannelsPerView] * (a * p + b * q));
                    }
                }
                return m;
            }

            private static (double[,] gram, double[] gradient) E25Normal(float[][] cols, float[] r, int begin, int count)
            {
                var gram = new double[cols.Length, cols.Length]; var gradient = new double[cols.Length];
                for (int j = 0; j < cols.Length; ++j)
                {
                    gradient[j] = E21Dot(cols[j], r, begin, begin + count);
                    for (int k = 0; k <= j; ++k) gram[j, k] = gram[k, j] = E21Dot(cols[j], cols[k], begin, begin + count);
                }
                return (gram, gradient);
            }

            private static double E25Cos(double[] a, double[] b)
            {
                double den = Math.Sqrt(a.Sum(x => x * x) * b.Sum(x => x * x));
                return den <= 1e-30 ? double.NaN : a.Zip(b, (x, y) => x * y).Sum() / den;
            }

            private void E25WriteLocal(StreamWriter csv, string state, string mapping, double[] rig, int[] active, double step,
                E25LocalModel m, E25LocalModel baseline)
            {
                var toGt = active.SelectMany(lamp => Enumerable.Range(4 * lamp, 4)).Select(i => room.Truth[i] - rig[i]).ToArray();
                for (int v = -1; v < m.anchor.Length / E23ChannelsPerView; ++v)
                {
                    int start = v < 0 ? 0 : v * E23ChannelsPerView, end = v < 0 ? m.anchor.Length : start + E23ChannelsPerView;
                    var (g, gradient) = E25Normal(m.cols, m.residual, start, end - start);
                    var spec = ClassroomDiagnosis.GetSpectrum(g, m.cols.Length);
                    double[] corr = new double[m.cols.Length * m.cols.Length];
                    for (int j = 0; j < m.cols.Length; ++j)
                    for (int k = 0; k < m.cols.Length; ++k)
                        corr[j * m.cols.Length + k] = g[j, k] / Math.Sqrt(Math.Max(1e-60, g[j, j] * g[k, k]));
                    var ratios = new double[m.cols.Length]; var cos = new double[m.cols.Length];
                    for (int j = 0; j < m.cols.Length; ++j)
                    {
                        double reference = E21Dot(baseline.cols[j], baseline.cols[j], start, end), a = g[j, j];
                        ratios[j] = reference > 1e-30 ? Math.Sqrt(a / reference) : double.NaN;
                        cos[j] = reference * a > 1e-60 ? E21Dot(m.cols[j], baseline.cols[j], start, end) / Math.Sqrt(a * reference) : double.NaN;
                    }
                    csv.WriteLine(string.Join(",", state, mapping, v, F(step), "normalized_XYZ_and_logI", spec.rank, F(spec.condition), Series(spec.singular),
                        Series(spec.columnNorms), Series(corr), Series(gradient), F(E25Cos(gradient.Select(x => -x).ToArray(), toGt)),
                        Series(m.steps), string.Join(";", m.stencils), F(Math.Sqrt(E21Dot(m.residual, m.residual, start, end))), Series(ratios), Series(cos)));
                }
            }

            private void E25PathAndTrials(E21Trial prior, E23FrameObjective log, E23FrameObjective raw, Action<double[], float[]> capture)
            {
                using var path = Csv("loss-path.csv", "start,interpolation,point,t,legal,raw_loss,log_loss,raw_view_nrmse,log_view_nrmse,camera_renders,theta");
                using var trials = Csv("failed-lm-repeat.csv", "requested_position_m,step_fraction,repeat,legal,base_loss,trial_loss,predicted_decrease,actual_decrease,rho,camera_renders,theta");
                var buffer = new float[log.Size]; var r = new float[log.Size];
                foreach (var (label, start) in new[] { ("initial", prior.initial), ("e24_failed", prior.final) })
                foreach (bool power in new[] { false, true })
                for (int n = 0; n <= 40; ++n)
                {
                    double t = n / 40.0;
                    var rig = (double[])start.Clone();
                    for (int j = 0; j < (power ? 4 : 3); ++j) rig[20 + j] += t * (room.Truth[20 + j] - start[20 + j]);
                    bool legal = room.Contains(rig);
                    capture(rig, buffer);
                    var rawLoss = raw.Evaluate(buffer, r);
                    var rawView = Views();
                    var logLoss = log.Evaluate(buffer, r);
                    path.WriteLine(string.Join(",", label, power ? "xyz_logI" : "xyz_only", n, F(t), Flag(legal), F(rawLoss), F(logLoss), Series(rawView),
                        Series(Views()), E19FitViews, Series(rig)));
                }
                double[] Views() => Enumerable.Range(0, E19FitViews).Select(i => Math.Sqrt(E19FitViews * E21Dot(r, r,
                    i * E23ChannelsPerView, (i + 1) * E23ChannelsPerView))).ToArray();

                var signal = new E21Signal("diagnostic_off_Log", log.Size, (rig, residual) => { capture(rig, buffer); return log.Evaluate(buffer, residual); });
                foreach (double step in new[] { .005, .02, .05 })
                {
                    var diag = new E21Trial { id = "A1_failed_LM_" + F(step), probePolicy = "strict", positionStep = step };
                    var center = prior.final;
                    double[] delta = null;
                    double[,] gram = null; double[] gradient = null;
                    Solve(diag, "diagnostic", new[] { 5 }, center, signal, 24, null, iterations: 1, configure: (settings, _) =>
                    {
                        settings.NormalMode = ImageJacobianLm.TrustNormalMode.LampScaledDense;
                        var jac = settings.Jacobian; var trial = settings.Trial;
                        settings.Jacobian = j => { jac(j); gram = j.Gram; gradient = j.Gradient; };
                        settings.Trial = t => { trial(t); if (t.Legal && delta == null) delta = (double[])t.Step.Clone(); };
                    });
                    if (delta == null || gram == null) continue;
                    foreach (double fraction in new[] { .25, .5, 1.0 })
                    {
                        var pose = (double[])center.Clone();
                        for (int j = 0; j < delta.Length; ++j) pose[20 + j] += fraction * delta[j];
                        double prediction = E25Predicted(gram, gradient, delta.Select(x => x * fraction).ToArray());
                        bool legal = room.Contains(pose);
                        for (int rIndex = 1; rIndex <= 2; ++rIndex)
                        {
                            double baseLoss = signal.objective(center, null), trialLoss = legal ? signal.objective(pose, null) : double.NaN;
                            var actual = baseLoss - trialLoss;
                            trials.WriteLine(string.Join(",", F(step), F(fraction), rIndex, Flag(legal), F(baseLoss), F(trialLoss), F(prediction), F(actual),
                                F(prediction > 0 ? actual / prediction : double.NaN), legal ? 2 * E19FitViews : E19FitViews, Series(pose)));
                        }
                    }
                }
            }

            private static double E25Predicted(double[,] gram, double[] g, double[] delta)
            {
                double pred = 0;
                for (int j = 0; j < delta.Length; ++j)
                {
                    pred -= 2 * delta[j] * g[j];
                    for (int k = 0; k < delta.Length; ++k) pred -= delta[j] * gram[j, k] * delta[k];
                }
                return pred;
            }

            private void E25Hotspots(StreamWriter csv, string state, double[] rig, MbdRgbTarget target, MbdSurfaceGeometry geometry)
            {
                Apply(rig, false);
                for (int v = 0; v < E19FitViews; ++v)
                {
                    var a = low.Capture(v); var b = high.Capture(v); var ar = low.Capture(v); var br = high.Capture(v);
                    Analyze(a.Pixel, target.fit[v].Pixel, a.PixelCount, low.Width, low.Height, "native448", Math.Sqrt(NativeRenderBackend.Compare(ar, a)));
                    Analyze(b.Pixel, target.high[v].Pixel, b.PixelCount, high.Width, high.Height, "native896", Math.Sqrt(NativeRenderBackend.Compare(br, b)));
                    Analyze(i => BoxPixel(b, i), i => BoxPixel(target.high[v], i), a.PixelCount, low.Width, low.Height, "box896", double.NaN);

                    void Analyze(Func<int, Color> actual, Func<int, Color> gt, int count, int w, int h, string source, double repeat)
                    {
                        var error = new double[count]; int peak = 0;
                        for (int i = 0; i < count; ++i)
                        {
                            Color d = actual(i) - gt(i);
                            error[i] = (double)d.r * d.r + (double)d.g * d.g + (double)d.b * d.b;
                            if (error[i] > error[peak]) peak = i;
                        }
                        if (state == "e24_failed" && source == "native448")
                            SaveErrorMap(CapturePath("L6_hotspots"), "raw_error_v" + v, error, w, h, Math.Sqrt(target.fit[v].Energy / count));
                        var camera = room.Camera;
                        var oldPosition = camera.transform.position; var oldRotation = camera.transform.rotation;
                        float oldAspect = camera.aspect; camera.aspect = (float)w / h;
                        string renderer = "", mat = "", label = "", materialName = ""; bool boundary = false;
                        int x = peak % w, y = peak / w;
                        try
                        {
                            camera.transform.SetPositionAndRotation(room.Views[v], room.Rotations[v]);
                            if (error[peak] > 0 && Hit(x, y, out var hit))
                            {
                                renderer = hit.renderer.GetInstanceID().ToString(Invariant);
                                mat = hit.material.GetInstanceID().ToString(Invariant);
                                label = hit.renderer.name.Replace(',', '_');
                                materialName = hit.material.name.Replace(',', '_');
                                boundary = new[] { (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1) }.Any(p =>
                                    !Hit(p.Item1, p.Item2, out var other) || other.renderer != hit.renderer || other.material != hit.material ||
                                    Vector3.Dot(other.normal, hit.normal) < .9f || Math.Abs(other.distance - hit.distance) > .05f);
                            }
                        }
                        finally { camera.transform.SetPositionAndRotation(oldPosition, oldRotation); camera.aspect = oldAspect; }
                        bool Hit(int xx, int yy, out (Renderer renderer, Material material, Vector3 normal, float distance) hit)
                        {
                            ++rayQueries;
                            return geometry.TryFirstOpaqueHit(camera.ViewportPointToRay(new Vector3((xx + .5f) / w, (yy + .5f) / h)), camera.farClipPlane, out hit);
                        }
                        double max = Math.Sqrt(error[peak]), energy = error.Sum();
                        Array.Sort(error);
                        double top = error.Skip(count - (int)Math.Ceiling(.01 * count)).Sum() / Math.Max(1e-30, energy);
                        csv.WriteLine(string.Join(",", state, v, source, x, y, F(max), F(top), renderer, mat, label, materialName, Flag(boundary), F(repeat)));
                    }
                }
            }
        }
    }
}

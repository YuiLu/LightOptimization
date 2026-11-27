using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using Frame = LightingMvp.NativeRenderBackend.Frame;
using Mapping = LightingMvp.NativeRenderBackend.HdrMapping;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            private sealed class E24Anchor
            {
                internal string id;
                internal double[] theta;
                internal int[] lamps;
                internal bool failed;
            }

            private sealed class E24Direction
            {
                internal double startLoss, bestRepeatDrop;
                internal double[] physical;
            }

            private const string E24DirectionHeader = "state,failed,model,mapping,fraction,legal,loss_start,loss_1,loss_2,predicted,actual_1,actual_2,rho_1,rho_2,cos_h2,blocked,first_candidate_status,rejected,diagnostic_cameras,step,theta";

            private IEnumerable<E24Anchor> E24Anchors(E21StageSummary reference, bool classroom)
            {
                var start = E24Prior(reference, "C1");
                if (!classroom)
                {
                    var source = E24Read("source");
                    foreach (int seed in smoke ? new[] { 131 } : E23Seeds)
                    {
                        var raw = source.arms.Single(a => a.id == $"seed{seed}_near_on_B0");
                        var log = source.arms.Single(a => a.id == $"seed{seed}_near_on_B1");
                        var worst = E24Worst(log);
                        yield return new E24Anchor { id = $"seed{seed}_start", theta = raw.initial, lamps = worst };
                        yield return new E24Anchor { id = $"seed{seed}_log_failed", theta = log.final, lamps = worst, failed = !log.pass };
                    }
                }
                if (smoke) yield break;
                yield return new E24Anchor { id = start.scenario + "_start", theta = start.initial, lamps = E24Worst(start) };
                yield return new E24Anchor { id = start.scenario + "_log_failed", theta = start.final, lamps = E24Worst(start), failed = !start.pass };
                var b2 = E24Prior(reference, "B2");
                yield return new E24Anchor { id = start.scenario + "_B2_failed", theta = b2.final, lamps = E24Worst(b2), failed = !b2.pass };
            }

            private static int[] E24Worst(E21Trial trial) => Enumerable.Range(0, trial.lamps.Length)
                .OrderByDescending(j => Math.Max(trial.positionCm[j] / 5, trial.intensityRelative[j] / .05))
                .Take(2).Select(j => trial.lamps[j] - 1).OrderBy(j => j).ToArray();

            private static bool E24Better(E24Direction candidate, E24Direction reference) => candidate.bestRepeatDrop > 0 &&
                (reference.bestRepeatDrop <= 0 || candidate.bestRepeatDrop - reference.bestRepeatDrop >
                    Math.Max(.001 * reference.startLoss, .05 * reference.bestRepeatDrop));

            private void E24DirectionAudit(E21StageSummary reference, MbdRgbTarget hard, bool classroom)
            {
                using var vis = Csv("visibility-audit.csv", E24VisibilityHeader);
                using var atlas = E23AtlasCsv();
                using var mapping = Csv("mapping-fd.csv", "state,lamp,axis,requested_mm,physical_step,stencil,blocked,negative_proxy_fraction,negative_visible_proxy_fraction,cos_h2_hard,cos_h4_hard,cos_c3_h2,cos_c3_hard,cos_c4_h4,cos_c4_hard,h2_on_norm_ratio,h4_on_norm_ratio,h2_cos_20mm,h4_cos_20mm,c3_cos_20mm");
                using var stable = Csv("mask-stability.csv", "state,lamp,view,comparison,mean_abs_delta,rms_delta,changed_gt_01_fraction");
                using var direction = Csv("direction-probes.csv", E24DirectionHeader);
                using var observer = new E23AtlasObserver(lights);
                var anchors = E24Anchors(reference, classroom).ToArray();
                var byState = new Dictionary<string, Dictionary<string, E24Direction>>();
                foreach (var sample in anchors)
                {
                    if (!room.Contains(sample.theta)) throw new InvalidDataException("E23 checkpoint not legal with the local collider: " + sample.id);
                    E24AuditColumns(sample, hard, observer, vis, atlas, mapping, stable);
                    var rows = new Dictionary<string, E24Direction>();
                    byState.Add(sample.id, rows);
                    foreach (string model in new[] { "H2", "H0", "H1", "H4", "C1", "C3" })
                        rows.Add(model, E24ProbeDirection(sample, model, hard, direction, vis, rows.TryGetValue("H2", out var b2) ? b2 : null));
                    Debug.Log("[E24-DIR] " + sample.id + " recorded");
                }
                // A useful native shadow term is observed only when exact On beats the frozen mask on failed states
                // and the mask itself did not beat B2 there. The conditional diagnostic never changes the primary model.
                var failed = anchors.Where(x => x.failed).ToArray();
                var shadowCandidates = failed.Where(x =>
                    E24Better(byState[x.id]["H1"], byState[x.id]["H4"]) && !E24Better(byState[x.id]["H4"], byState[x.id]["H2"])).ToArray();
                bool startH5 = shadowCandidates.Length >= 2;
                bool startC4 = failed.Count(x => E24Better(byState[x.id]["H4"], byState[x.id]["H2"])) >= 2 &&
                    failed.Count(x => E24Better(byState[x.id]["C3"], byState[x.id]["H2"])) >= 2;
                if (startH5 || startC4)
                    foreach (var sample in failed)
                    {
                        var rows = byState[sample.id];
                        if (startH5)
                            foreach (var model in new[] { "H3", "H5" })
                                rows.Add(model, E24ProbeDirection(sample, model, hard, direction, vis, rows["H2"]));
                        if (startC4) rows.Add("C4", E24ProbeDirection(sample, "C4", hard, direction, vis, rows["H2"]));
                    }
                using var conditional = Csv("conditional-models.csv", "model,started,reason,eligible_anchor_count");
                conditional.WriteLine($"H5,{Flag(startH5)},on_better_than_frozen_mask_without_H4_gain,{shadowCandidates.Length}");
                int sharedWins = failed.Count(x => E24Better(byState[x.id]["H4"], byState[x.id]["H2"]) && E24Better(byState[x.id]["C3"], byState[x.id]["H2"]));
                conditional.WriteLine($"C4,{Flag(startC4)},both_H4_and_C3_pass,{sharedWins}");
            }

            private void E24AuditColumns(E24Anchor sample, MbdRgbTarget hard, E23AtlasObserver observer,
                StreamWriter visibility, StreamWriter atlas, StreamWriter mapping, StreamWriter stability)
            {
                var at = sample.theta;
                var objective = new E23FrameObjective(hard.fit, Mapping.SignedLog, e24HardScale);
                var h = new float[E23Channels]; var o = new float[E23Channels];
                var x = new float[E23Channels]; var y = new float[E23Channels];
                var h1 = new float[E23Channels]; var h2 = new float[E23Channels];
                var mask = new float[E23Channels / 3]; var alternate = new float[mask.Length];
                var work = new float[E24PixelsPerView];
                var layout0 = new E23Layout[E19FitViews];
                CaptureWithAtlas(at, "anchor", h, recordBase: true);
                E23CaptureRgb(at, LightShadows.None, o);
                foreach (int lamp in sample.lamps)
                {
                    int i = lamp * 4 + 3;
                    var standard = Intensity(.04, "l" + (lamp + 1) + "_004");
                    E24Mask(x, y, mask, work, E24RelativeThreshold, visibility, sample.id, "default", 0, lamp, standard.firstStep * units[3], standard.code);
                    foreach (float threshold in new[] { .0001f, .01f })
                    {
                        E24Mask(x, y, alternate, work, threshold, visibility, sample.id, "threshold", 0, lamp, standard.firstStep * units[3], standard.code);
                        MaskDelta("threshold_" + F(threshold));
                    }
                    var small = Intensity(.02, "l" + (lamp + 1) + "_002");
                    E24Mask(x, y, alternate, work, E24RelativeThreshold, visibility, sample.id, "intensity_step", 0, lamp, small.firstStep * units[3], small.code);
                    MaskDelta("logI_002_vs_004");
                    if (sample.failed) SaveMask(mask, lamp);
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        var measures = new List<(int mm, E23Stencil stencil, float[][] columns, double neg, double maskedNeg)>();
                        foreach (int mm in new[] { 5, 20, 50 })
                        {
                            var p = E23SelectStencil(at, lamp * 4 + axis, mm / 1000.0);
                            var cols = Enumerable.Range(0, 5).Select(_ => new float[E23Channels]).ToArray();
                            if (p.code == 0) { measures.Add((mm, p, cols, double.NaN, double.NaN)); continue; }
                            CaptureStencil(p, LightShadows.Hard, h1, h2);
                            CaptureStencil(p, LightShadows.None, x, y);
                            int neg = 0, maskedNeg = 0, terms = 0;
                            for (int k = 0; k < E23Channels; ++k)
                            {
                                double weight = objective.weights[k / E23ChannelsPerView], d = E23Derivative(h[k], Mapping.SignedLog, e24HardScale);
                                float a = x[k] - o[k], b = p.code == 1 ? 0 : y[k] - o[k];
                                float m = mask[k / 3];
                                float fa = h[k] + a, fb = h[k] + b, va = h[k] + m * a, vb = h[k] + m * b;
                                if (fa < 0) ++neg; if (va < 0) ++maskedNeg; ++terms;
                                if (p.code != 1) { if (fb < 0) ++neg; if (vb < 0) ++maskedNeg; ++terms; }
                                float f0 = objective.Transform(h[k]);
                                cols[0][k] = (float)(weight * (p.a * (objective.Transform(h1[k]) - f0) + p.b * (objective.Transform(h2[k]) - f0)));
                                cols[1][k] = (float)(weight * d * (p.a * a + p.b * b));
                                cols[2][k] = m * cols[1][k];
                                cols[3][k] = (float)(weight * (p.a * (objective.Transform(fa) - f0) + p.b * (objective.Transform(fb) - f0)));
                                cols[4][k] = (float)(weight * (p.a * (objective.Transform(va) - f0) + p.b * (objective.Transform(vb) - f0)));
                            }
                            measures.Add((mm, p, cols, (double)neg / terms, (double)maskedNeg / terms));
                        }
                        foreach (var m in measures)
                        {
                            var c = m.columns; var mid = measures[1].columns;
                            double ratio(int a, int b) => SafeRatio(Math.Sqrt(E21Dot(c[a], c[a])), Math.Sqrt(E21Dot(c[b], c[b])));
                            mapping.WriteLine(string.Join(",", sample.id, lamp + 1, "XYZ"[axis], m.mm, F(m.stencil.firstStep * units[axis]), m.stencil.code, m.stencil.blocked,
                                F(m.neg), F(m.maskedNeg), F(Cos(c[1], c[0])), F(Cos(c[2], c[0])), F(Cos(c[3], c[1])), F(Cos(c[3], c[0])), F(Cos(c[4], c[2])), F(Cos(c[4], c[0])),
                                F(ratio(1, 0)), F(ratio(2, 0)), F(Cos(c[1], mid[1])), F(Cos(c[2], mid[2])), F(Cos(c[3], mid[3]))));
                        }
                    }

                    E23Stencil Intensity(double step, string label)
                    {
                        var s = E23SelectStencil(at, i, step);
                        if (s.code == 0) throw new InvalidDataException("No legal intensity stencil for " + sample.id);
                        CaptureWithAtlas(s.first, label + "_a", h1);
                        if (s.second != null) CaptureWithAtlas(s.second, label + "_b", h2);
                        CaptureStencil(s, LightShadows.None, x, y);
                        for (int k = 0; k < h.Length; ++k)
                        {
                            double dh = s.a * (h1[k] - h[k]) + (s.code == 1 ? 0 : s.b * (h2[k] - h[k]));
                            y[k] = (float)(s.a * (x[k] - o[k]) + (s.code == 1 ? 0 : s.b * (y[k] - o[k])));
                            x[k] = (float)dh;
                        }
                        return s;
                    }

                    void MaskDelta(string comparison)
                    {
                        for (int v = 0; v < E19FitViews; ++v)
                        {
                            double e = 0, e2 = 0; int many = 0;
                            for (int k = v * E24PixelsPerView; k < (v + 1) * E24PixelsPerView; ++k)
                            {
                                double d = Math.Abs(mask[k] - alternate[k]); e += d; e2 += d * d; if (d > .1) ++many;
                            }
                            stability.WriteLine(string.Join(",", sample.id, lamp + 1, v, comparison, F(e / E24PixelsPerView),
                                F(Math.Sqrt(e2 / E24PixelsPerView)), F((double)many / E24PixelsPerView)));
                        }
                    }
                }

                void CaptureStencil(E23Stencil stencil, LightShadows mode, float[] first, float[] second)
                {
                    E23CaptureRgb(stencil.first, mode, first);
                    if (stencil.second != null) E23CaptureRgb(stencil.second, mode, second);
                    else Array.Clear(second, 0, second.Length);
                }

                void CaptureWithAtlas(double[] rig, string phase, float[] rgb, bool recordBase = false)
                {
                    Apply(rig, true);
                    for (int view = 0; view < E19FitViews; ++view)
                    {
                        var (frame, layout) = observer.Capture(() => low.Capture(view));
                        if (recordBase) layout0[view] = layout;
                        E23WriteLayout(atlas, "hard_intensity", sample.id, view, phase, layout, layout0[view]);
                        for (int k = view * E23ChannelsPerView, p = 0; p < frame.PixelCount; ++p)
                        {
                            Color c = frame.Pixel(p); rgb[k++] = c.r; rgb[k++] = c.g; rgb[k++] = c.b;
                        }
                    }
                }

                void SaveMask(float[] values, int lamp)
                {
                    int view = Enumerable.Range(0, E19FitViews).OrderBy(v =>
                        values.Skip(v * E24PixelsPerView).Take(E24PixelsPerView).Average(p => p)).First();
                    var tex = new Texture2D(E19FitW, E19FitH, TextureFormat.RGB24, false, true);
                    try
                    {
                        var colors = new Color32[E24PixelsPerView];
                        for (int p = 0; p < colors.Length; ++p)
                        {
                            byte g = (byte)Mathf.RoundToInt(values[view * E24PixelsPerView + p] * 255);
                            colors[p] = new Color32(g, g, g, 255);
                        }
                        tex.SetPixels32(colors); tex.Apply();
                        File.WriteAllBytes(Path.Combine(CapturePath("visibility"), $"{sample.id}_L{lamp + 1}_v{view}.png"), tex.EncodeToPNG());
                    }
                    finally { UnityEngine.Object.DestroyImmediate(tex); }
                }
            }

            private E24Direction E24ProbeDirection(E24Anchor state, string model, MbdRgbTarget hard,
                StreamWriter log, StreamWriter visibility, E24Direction baseline)
            {
                int[] active = E24AllLamps;
                var mapping = E24Raw(model) ? Mapping.Raw : Mapping.SignedLog;
                var objective = new E23FrameObjective(hard.fit, mapping, e24HardScale);
                var rgb = new float[E23Channels];
                var signal = new E21Signal(mapping.ToString(), E23Channels, (rig, r) =>
                {
                    E23CaptureRgb(rig, LightShadows.Hard, rgb);
                    return objective.Evaluate(rgb, r);
                });
                ImageJacobianLm.TrustJacobian jac = null;
                ImageJacobianLm.TrustTrial first = null;
                ImageJacobianLm.TrustResult result = null;
                double[] candidate = null;
                int before = low.RenderCount;
                var diag = new E21Trial { id = "dir_" + state.id + "_" + model, probePolicy = "strict", positionStep = .02 };
                Solve(diag, "direction", active, state.theta, signal, 160, null, iterations: 1, record: p => result = p,
                    configure: (settings, expand) =>
                    {
                        E24Configure(settings, expand, active, model, objective,
                            (q, rig, mode, phase, axis, step, buffer) =>
                            {
                                int begin = low.RenderCount;
                                E23CaptureRgb(rig, mode, buffer);
                                e24Model.WriteLine(string.Join(",", diag.id, q, phase, mode, axis, F(step), low.RenderCount - begin, Series(rig)));
                            }, diag.id, visibility);
                        var writeJac = settings.Jacobian; var writeTrial = settings.Trial;
                        settings.Jacobian = j => { jac = j; writeJac?.Invoke(j); };
                        settings.Trial = t =>
                        {
                            writeTrial?.Invoke(t);
                            if (first != null || !t.Legal || !(t.Predicted > 0)) return;
                            first = t; candidate = (double[])expand(t.Theta).Clone();
                        };
                    });
                if ((low.RenderCount - before) != result.Evaluations * E19FitViews)
                    throw new InvalidOperationException("E24 direction camera accounting differs.");
                int modelCameras = low.RenderCount - before;
                var d = state.theta.Select((x, i) => candidate == null ? 0 : candidate[i] - x).ToArray();
                var physical = d.Select((x, i) => x * units[i % 4]).ToArray();
                double cosine = baseline == null || mapping == Mapping.Raw ? double.NaN : SafeRatio(d.Select((x, i) => physical[i] * baseline.physical[i]).Sum(),
                    Math.Sqrt(physical.Sum(x => x * x) * baseline.physical.Sum(x => x * x)));
                double best = 0;
                foreach (double fraction in new[] { .25, .5, 1.0 })
                {
                    var p = state.theta.Select((x, i) => x + fraction * d[i]).ToArray();
                    bool legal = first != null && room.Contains(p);
                    double loss1 = legal ? signal.objective(p, null) : double.NaN;
                    double loss2 = legal ? signal.objective(p, null) : double.NaN;
                    double actual1 = result.InitialLoss - loss1, actual2 = result.InitialLoss - loss2;
                    double predicted = legal ? E24Predicted(jac, d, fraction) : double.NaN;
                    if (legal) best = Math.Max(best, Math.Min(actual1, actual2));
                    log.WriteLine(string.Join(",", state.id, Flag(state.failed), model, mapping, F(fraction), Flag(legal), F(result.InitialLoss), F(loss1), F(loss2), F(predicted),
                        F(actual1), F(actual2), F(SafeRatio(actual1, predicted)), F(SafeRatio(actual2, predicted)), F(cosine), diag.blocked, first?.Status ?? result.Stop,
                        diag.rejected, (fraction == .25 ? modelCameras : 0) + (legal ? 2 * E19FitViews : 0), Series(physical), Series(p)));
                }
                return new E24Direction { startLoss = result.InitialLoss, bestRepeatDrop = best, physical = physical };
            }

            private static double E24Predicted(ImageJacobianLm.TrustJacobian jac, double[] direction, double fraction)
            {
                if (jac.Gradient.Length != direction.Length)
                    throw new InvalidOperationException("E24 direction must use the full active parameterization.");
                double p = 0;
                for (int j = 0; j < direction.Length; ++j)
                {
                    p -= 2 * fraction * jac.Gradient[j] * direction[j];
                    for (int k = 0; k < direction.Length; ++k)
                        p -= fraction * fraction * direction[j] * jac.Gram[j, k] * direction[k];
                }
                return p;
            }
        }
    }
}

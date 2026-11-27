using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using Mapping = LightingMvp.NativeRenderBackend.HdrMapping;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            private sealed class E25Probes
            {
                internal double d;
                internal int stencil;
                internal readonly float[] hp = new float[E23Channels], hm = new float[E23Channels], op = new float[E23Channels], om = new float[E23Channels];
            }
            private sealed class E25LampInput
            {
                internal readonly float[] hard = new float[E23Channels], off = new float[E23Channels], residual = new float[E23Channels];
                internal readonly float[] mask = new float[E24PixelsPerView * E19FitViews], ih = new float[E23Channels], io = new float[E23Channels];
                internal readonly float[] floors = new float[E19FitViews];
                internal readonly bool[] weak = new bool[E24PixelsPerView * E19FitViews];
                internal readonly E25Probes[] points = Enumerable.Range(0, 4).Select(_ => new E25Probes()).ToArray();
            }
            [Serializable] private sealed class E25ArchivePose { public double[] theta; }

            private E25LampInput E25AuditedLamp(string label, double[] rig, int lamp, E23FrameObjective fit, double physical,
                StreamWriter maskStats, StreamWriter atlas)
            {
                var m = new E25LampInput();
                using var observer = atlas == null ? null : new E23AtlasObserver(lights);
                var layout0 = new E23Layout[E19FitViews];
                Hard(rig, m.hard, "anchor", true);
                E25Capture(rig, LightShadows.None, E25DefaultViews, m.off);
                fit.Evaluate(m.hard, m.residual);
                for (int axis = 0; axis < 4; ++axis)
                {
                    E25Probes p = m.points[axis];
                    int i = lamp * 4 + axis;
                    (p.d, p.stencil) = E25Stencil(rig, i, axis == 3 ? .04 : physical);
                    if (p.stencil == 0) continue;
                    var next = (double[])rig.Clone();
                    next[i] += p.d;
                    E25Capture(next, LightShadows.None, E25DefaultViews, p.op);
                    Hard(next, p.hp, axis == 3 ? "intensity_first" : null);
                    if (p.stencil == 1) continue;
                    next[i] = rig[i] + (p.stencil == 3 ? -p.d : 2 * p.d);
                    E25Capture(next, LightShadows.None, E25DefaultViews, p.om);
                    Hard(next, p.hm, axis == 3 ? "intensity_second" : null);
                }
                var intensity = m.points[3];
                if (intensity.stencil != 0)
                {
                    for (int i = 0; i < m.ih.Length; ++i)
                    {
                        m.ih[i] = (float)Difference(m.hard[i], intensity.hp[i], intensity.hm[i], intensity.d, intensity.stencil);
                        m.io[i] = (float)Difference(m.off[i], intensity.op[i], intensity.om[i], intensity.d, intensity.stencil);
                    }
                    var norm = new float[E24PixelsPerView];
                    E24Mask(m.ih, m.io, m.mask, norm, E24RelativeThreshold, maskStats, label, "audit", -1, lamp, intensity.d * units[3], intensity.stencil);
                    for (int v = 0; v < E19FitViews; ++v)
                    {
                        int start = v * E24PixelsPerView;
                        for (int p = 0; p < norm.Length; ++p)
                        {
                            int k = 3 * (start + p);
                            float x = m.io[k], y = m.io[k + 1], z = m.io[k + 2];
                            norm[p] = x * x + y * y + z * z;
                        }
                        double p95 = Math.Sqrt(E24Quantile(norm, (int)Math.Ceiling(.95 * norm.Length) - 1));
                        double floor = Math.Max(1e-12, E24RelativeThreshold * p95); m.floors[v] = (float)floor;
                        for (int p = start; p < start + norm.Length; ++p) m.weak[p] = ChannelEnergy(m.io, 3 * p) <= floor * floor;
                    }
                }
                else { Array.Fill(m.mask, 1); Array.Fill(m.weak, true); }
                return m;

                void Hard(double[] theta, float[] rgb, string phase, bool initial = false)
                {
                    if (observer == null || phase == null)
                    {
                        E25Capture(theta, LightShadows.Hard, E25DefaultViews, rgb);
                        return;
                    }
                    Apply(theta, true);
                    for (int v = 0; v < E19FitViews; ++v)
                    {
                        int offset = v * E23ChannelsPerView;
                        var layout = observer.Observe(() => low.CaptureRgb(v, rgb, offset));
                        if (initial) layout0[v] = layout;
                        E23WriteLayout(atlas, "hard_intensity", label + "_L" + (lamp + 1), v, phase, layout, layout0[v]);
                    }
                }
            }

            private static double ChannelEnergy(float[] x, int k) => (double)x[k] * x[k] + (double)x[k + 1] * x[k + 1] + (double)x[k + 2] * x[k + 2];
            private static double Difference(float h, float p, float m, double d, int stencil)
            {
                var (a, b) = E24Weights(d, stencil);
                return a * (p - h) + (stencil == 1 ? 0 : b * (m - h));
            }

            private void E25Audit(E21StageSummary saved, MbdRgbTarget hard, bool classroom)
            {
                var fit = new E23FrameObjective(hard.fit, Mapping.SignedLog, e24HardScale);
                var linear = new E23FrameObjective(hard.fit, Mapping.Raw, e24HardScale);
                var points = new List<(string label, double[] theta, int[] lamps)>();
                if (classroom)
                {
                    var reference = E25Read("a_classroom");
                    var l6 = E25L6(reference);
                    var historical = JsonUtility.FromJson<E25ArchivePose>(File.ReadAllText(Path.Combine(e25Inputs, "classroom_low_response.json")));
                    points.Add(("e24_seed509_classroom_near_log_failed", historical.theta, E24AllLamps));
                    points.Add(("e24_L6_off_oracle_initial", l6.initial, new[] { 5 }));
                    points.Add(("e24_L6_off_oracle_failed", l6.final, new[] { 5 }));
                }
                foreach (var a in saved.arms.Where(x => x.method == "C4" && (x.scenario is "dir911_20cm" or "dir1201_30cm" or "dir1301_10cm" or "dir1601_10cm")))
                    points.Add((a.scenario + "_initial", a.initial, E24AllLamps));
                if (!classroom)
                {
                    var far = E25Read("a_complex").arms.Single(a => a.id == "seed509_complex_far_A-log");
                    points.Add((far.scenario + "_initial", far.initial, E24AllLamps));
                }
                using var weak = Csv("visibility-response-audit.csv", "state,lamp,view,mapping,model,low_response_fraction,mask_lt_half_among_supported,hard_high_xyz_low_intensity_fraction,xyz_energy_full,xyz_energy_weak,xyz_energy_fraction,gradient_full_xyz,gradient_weak_xyz,gradient_norm_full,gradient_norm_weak,gram_norm_full,gram_norm_weak");
                using var mapping = Csv("mapping-fd-audit.csv", "state,lamp,view,axis,requested_mm,actual_m,stencil,c3_relative_error,c4_relative_error,c3_norm_ratio,c4_norm_ratio,c3_hard_cos,c4_hard_cos,c4_c3_cos,c4_error_sq_weak,c4_error_sq_total");
                using var direction = Csv("native-steps.csv", "state,lamps,model,position_mm,step_fraction,repeat,legal,base_loss,trial_loss,predicted,actual,rho,hard_fd_direction_cos_20mm,theta");
                using var atlas = E23AtlasCsv();
                using var cost = Csv("audit-cost.csv", "state,diagnostic_cameras");
                using var gate = Csv("soft-fallback-gate.csv", "state,c4_error_sq_weak_fraction,c4_xyz_sq_weak_fraction,hard_repeat_drop,c4_repeat_drop,enabled,soft_repeat_drop,soft_shadow_mask_fallback");
                foreach (var (label, rig, lamps) in points)
                {
                    int start = low.RenderCount;
                    double weakProxyError = 0, allProxyError = 0, weakProxyXYZ = 0, allProxyXYZ = 0;
                    if (!room.Contains(rig)) throw new InvalidDataException("Illegal archived E25 audit point: " + label);
                    foreach (int lamp in lamps)
                    {
                        var m = E25AuditedLamp(label, rig, lamp, fit, .02, e25Visibility, atlas);
                        var rawRes = new float[E23Channels]; linear.Evaluate(m.hard, rawRes);
                        foreach (var (objective, residual) in new[] { (fit, m.residual), (linear, rawRes) })
                        {
                            var models = new[] { "H1", "C3", "C4" };
                            var c = models.Select(_ => Enumerable.Range(0, 3).Select(__ => new float[E23Channels]).ToArray()).ToArray();
                            for (int axis = 0; axis < 3; ++axis)
                            {
                                var p = m.points[axis];
                                if (p.stencil == 0) continue;
                                float[] mapped = m.hard.Select(objective.Transform).ToArray();
                                for (int i = 0; i < E23Channels; ++i)
                                    c[0][axis][i] = (float)(objective.weights[i / E23ChannelsPerView] * Difference(mapped[i], objective.Transform(p.hp[i]),
                                        p.stencil == 1 ? 0 : objective.Transform(p.hm[i]), p.d, p.stencil));
                                E25ReferenceColumn(objective, m.hard, m.off, p.op, p.om, null, p.d, p.stencil, c[1][axis]);
                                E25ReferenceColumn(objective, m.hard, m.off, p.op, p.om, m.mask, p.d, p.stencil, c[2][axis]);
                            }
                            for (int view = 0; view < E19FitViews; ++view)
                            {
                                int first = view * E24PixelsPerView, begin = 3 * first, end = begin + E23ChannelsPerView;
                                int weakCount = m.weak.Skip(first).Take(E24PixelsPerView).Count(x => x);
                                int occluded = 0, strongPos = 0;
                                double positionPowerThreshold = Enumerable.Range(0, 3).Sum(dim => E21Dot(c[0][dim], c[0][dim], begin, end)) / E24PixelsPerView;
                                for (int p = first; p < first + E24PixelsPerView; ++p)
                                {
                                    if (!m.weak[p] && m.mask[p] < .5) ++occluded;
                                    if (m.weak[p] && positionPowerThreshold > 0 && ChannelEnergy(c[0][0], 3 * p) + ChannelEnergy(c[0][1], 3 * p) + ChannelEnergy(c[0][2], 3 * p) > positionPowerThreshold) ++strongPos;
                                }
                                for (int model = 0; model < models.Length; ++model)
                                {
                                    var full = E25Normal(c[model], residual, begin, end - begin);
                                    var gram = new double[3, 3]; var g = new double[3];
                                    for (int i = begin; i < end; ++i)
                                    {
                                        if (!m.weak[i / 3]) continue;
                                        for (int j = 0; j < 3; ++j)
                                        {
                                            double a = c[model][j][i]; g[j] += a * residual[i];
                                            for (int k = 0; k < 3; ++k) gram[j,k] += a * c[model][k][i];
                                        }
                                    }
                                    double e = full.gram[0,0] + full.gram[1,1] + full.gram[2,2], u = gram[0,0] + gram[1,1] + gram[2,2];
                                    weak.WriteLine(string.Join(",", label, lamp + 1, view, objective.mapping, model == 0 && objective.mapping == Mapping.Raw ? "HardRaw" : models[model],
                                        F((double)weakCount / E24PixelsPerView), F((double)occluded / Math.Max(1, E24PixelsPerView - weakCount)), F((double)strongPos / E24PixelsPerView),
                                        F(e), F(u), F(SafeRatio(u, e)), Series(full.gradient), Series(g), F(L2(full.gradient)), F(L2(g)), F(L2(full.gram.Cast<double>())), F(L2(gram.Cast<double>()))));
                                }
                                if (objective.mapping != Mapping.SignedLog) continue;
                                for (int axis = 0; axis < 3; ++axis)
                                {
                                    var p = m.points[axis];
                                    var h = c[0][axis]; var c3 = c[1][axis]; var c4 = c[2][axis];
                                    double xx = E21Dot(h, h, begin, end), yy = E21Dot(c3, c3, begin, end), zz = E21Dot(c4, c4, begin, end);
                                    double xy = E21Dot(h, c3, begin, end), xz = E21Dot(h, c4, begin, end), yz = E21Dot(c3, c4, begin, end);
                                    double err = 0, errWeak = 0, xyzWeak = 0;
                                    for (int i = begin; i < end; ++i)
                                    {
                                        double delta = (double)h[i] - c4[i], delta2 = delta * delta;
                                        err += delta2;
                                        if (!m.weak[i / 3]) continue;
                                        errWeak += delta2; xyzWeak += (double)c4[i] * c4[i];
                                    }
                                    weakProxyError += errWeak; allProxyError += err; weakProxyXYZ += xyzWeak; allProxyXYZ += zz;
                                    mapping.WriteLine(string.Join(",", label, lamp + 1, view, "XYZ"[axis], 20, F(p.d * units[axis]), p.stencil,
                                        F(RelativeColumnError(h, c3, begin, end, xx)), F(RelativeColumnError(h, c4, begin, end, xx)), F(SafeRatio(Math.Sqrt(yy), Math.Sqrt(xx))),
                                        F(SafeRatio(Math.Sqrt(zz), Math.Sqrt(xx))), F(SafeRatio(xy, Math.Sqrt(xx * yy))),
                                        F(SafeRatio(xz, Math.Sqrt(xx * zz))), F(SafeRatio(yz, Math.Sqrt(yy * zz))), F(errWeak), F(err)));
                                }
                                if (classroom && lamp == 5 && view == 2 && label == "e24_seed509_classroom_near_log_failed")
                                    E25AuditMaps(label, view, m, c[0]);
                            }
                        }
                    }
                    var result = E25AuditSteps(label, rig, lamps, fit, direction);
                    double errorFraction = SafeRatio(weakProxyError, allProxyError), energy = SafeRatio(weakProxyXYZ, allProxyXYZ);
                    double h1 = result.drop["H1"], c4Drop = result.drop["C4"];
                    bool eligible = errorFraction > .5 && energy > .25 && h1 > Math.Max(1e-12, .001 * result.baseLoss) && !(c4Drop > Math.Max(1e-12, .00001 * result.baseLoss));
                    double soft = eligible ? E25AuditSteps(label, rig, lamps, fit, direction, softOnly: true).drop["C4-soft"] : double.NaN;
                    gate.WriteLine(string.Join(",", label, F(errorFraction), F(energy), F(h1), F(c4Drop), Flag(eligible), F(soft), eligible ? "0.5" : ""));
                    cost.WriteLine($"{label},{low.RenderCount - start}");
                }
            }

            private static double RelativeColumnError(float[] x, float[] y, int start, int end, double power)
            {
                double sum = 0;
                for (int i = start; i < end; ++i) { double d = (double)x[i] - y[i]; sum += d * d; }
                return power > 1e-30 ? Math.Sqrt(sum / power) : double.NaN;
            }
            private static double L2(IEnumerable<double> x) => Math.Sqrt(x.Sum(v => v * v));

            private void E25AuditMaps(string label, int v, E25LampInput m, float[][] xyz)
            {
                int begin = v * E24PixelsPerView;
                string path = CapturePath("mask_" + label);
                Map("mask", p => m.mask[p]);
                Map("weak", p => m.weak[p] ? 1f : 0f);
                Map("off_intensity_response", p => Mathf.Log(1 + (float)Math.Sqrt(ChannelEnergy(m.io, 3 * p)) / m.floors[v]) / 8);
                Map("mapped_residual", p => Mathf.Log(1 + 1000f * (float)Math.Sqrt(ChannelEnergy(m.residual, 3 * p))) / 4);
                Map("hard_position_response", p => Mathf.Log(1 + 1000f * (float)Math.Sqrt(xyz.Sum(col => ChannelEnergy(col, 3 * p)))) / 4);
                void Map(string name, Func<int, float> value)
                {
                    var tex = new Texture2D(E19FitW, E19FitH, TextureFormat.RGB24, false, true);
                    try
                    {
                        var pixels = new Color[E24PixelsPerView];
                        for (int i = 0; i < pixels.Length; ++i) { float x = Mathf.Clamp01(value(begin + i)); pixels[i] = new Color(x, x, x, 1); }
                        tex.SetPixels(pixels); tex.Apply();
                        File.WriteAllBytes(Path.Combine(path, $"v{v}_{name}.png"), tex.EncodeToPNG());
                    }
                    finally { UnityEngine.Object.DestroyImmediate(tex); }
                }
            }

            private (double baseLoss, Dictionary<string, double> drop) E25AuditSteps(string label, double[] pose, int[] lamps,
                E23FrameObjective fit, StreamWriter table, bool softOnly = false)
            {
                var values = new float[fit.Size];
                var signal = new E21Signal("native_hard_log_diagnostic", fit.Size, (r, res) =>
                { E25Capture(r, LightShadows.Hard, E25DefaultViews, values); return fit.Evaluate(values, res); });
                double baseLoss = signal.objective(pose, null);
                double[] hard20 = null;
                var drops = new Dictionary<string, double>();
                var methods = softOnly ? new[] { ("C4-soft", .02) } : new[] { ("H1", .02), ("H1", .005), ("H1", .05), ("C3", .02), ("C4", .02) };
                foreach (var (method, step) in methods)
                {
                    var diagnostic = new E21Trial { id = label + "_" + method + "_" + F(step), probePolicy = "strict", positionStep = step };
                    var indices = lamps.SelectMany(i => Enumerable.Range(4 * i, 4)).ToArray();
                    double[] delta = null; double[,] gram = null; double[] gradient = null;
                    Solve(diagnostic, "diagnostic", lamps, pose, signal, 100, null, iterations: 1, configure: (settings, expand) =>
                    {
                        E24Configure(settings, expand, lamps, method, fit, (_, r, mode, phase, axis, h, rgb) => E25Capture(r, mode, E25DefaultViews, rgb), diagnostic.id);
                        var jOld = settings.Jacobian; var tOld = settings.Trial;
                        settings.Jacobian = j => { jOld(j); gram = j.Gram; gradient = j.Gradient; };
                        settings.Trial = t => { tOld(t); if (delta == null && t.Legal) delta = (double[])t.Step.Clone(); };
                    });
                    if (!drops.ContainsKey(method)) drops.Add(method, double.NegativeInfinity);
                    if (delta == null || gram == null) continue;
                    if (method == "H1" && step == .02) hard20 = delta;
                    foreach (double scale in new[] { .25, .5, 1.0 })
                    {
                        var take = (double[])pose.Clone();
                        for (int i = 0; i < indices.Length; ++i) take[indices[i]] += scale * delta[i];
                        bool legal = room.Contains(take);
                        double pred = E25Predicted(gram, gradient, delta.Select(x => x * scale).ToArray());
                        double minDrop = double.PositiveInfinity;
                        for (int rep = 1; rep <= 2; ++rep)
                        {
                            var loss = legal ? signal.objective(take, null) : double.NaN;
                            minDrop = Math.Min(minDrop, baseLoss - loss);
                            table.WriteLine(string.Join(",", label, LampList(lamps), method, F(1000 * step), F(scale), rep, Flag(legal), F(baseLoss), F(loss),
                                F(pred), F(baseLoss - loss), F(SafeRatio(baseLoss - loss, pred)), F(hard20 == null ? double.NaN : E25Cos(delta, hard20)), Series(take)));
                        }
                        if (legal) drops[method] = Math.Max(drops[method], minDrop);
                    }
                }
                return (baseLoss, drops);
            }
        }
    }
}

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
            private void E26ColorAudit(double[] truth, E26ColorTarget on, E26ColorTarget off, StreamWriter native)
            {
                int begin = low.RenderCount;
                using var csv = Csv("color-jacobian.csv", "shadows,lamp,gt_kelvin,requested_fd_mired,signed_fd_mired,stencil,temperature_col_l2_per_mired,intensity_col_l2_per_logI,temperature_intensity_cos,temperature_col_cos_vs_5mired,plus_minus_direction_cos,orthogonal_ratio,rank_j4,rank_j5,relative_spectral_cutoff,projection_relative_column_cutoff,repeat_noise_l2,per_step_signal_to_repeat_floor,plus_loss,minus_loss,atlas_unchanged");
                using var bounds = Csv("color-boundaries.csv", "shadows,lamp,bound_kelvin,temperature_step_mired,stencil,legal,derivative_l2_per_mired");
                const double qrRelative = 1e-6;
                using var atlas = new E23AtlasObserver(lights, observeColors: true);
                var baseLayouts = AuditNative("gt", truth, null);
                for (int lamp = 0; lamp < 3; ++lamp)
                {
                    var move = (double[])truth.Clone(); AddTemperature(move, lamp, 40);
                    var plusLayouts = AuditNative($"L{lamp + 1}_plus40", move, baseLayouts);
                    move = (double[])truth.Clone(); AddTemperature(move, lamp, -40);
                    var minusLayouts = AuditNative($"L{lamp + 1}_minus40", move, baseLayouts);
                    bool unchanged = Enumerable.Range(0, 3).All(v => plusLayouts[v].Signature == baseLayouts[v].Signature && minusLayouts[v].Signature == baseLayouts[v].Signature);
                    foreach (var config in new[] { on, off })
                    {
                        var mode = config.target.shadows ? LightShadows.Hard : LightShadows.None;
                        var fit = config.fit;
                        // Old four columns are native, mapped derivatives. Convert normalized axes to physical units.
                        var local = E25Local(truth, new[] { lamp }, .02, fit, (x, r) => ColorCapture(x, mode, r), ColorLegal);
                        var j4 = local.cols.Select((c, i) => c.Select(x => (float)(x / units[i])).ToArray()).ToArray();
                        var check = new float[fit.Size]; ColorCapture(truth, mode, check); fit.Evaluate(check, check);
                        double noise = L2(local.residual.Zip(check, (x, y) => (double)x - y));
                        var (basis, r4) = Basis(j4, qrRelative);
                        float[] refCol = null;
                        foreach (double step in new[] { 5d, 10d })
                        {
                            var probe = TemperatureColumn(truth, lamp, mode, fit, step);
                            var col = probe.column;
                            refCol ??= col;
                            var j5 = j4.Concat(new[] { col }).ToArray();
                            var spectrum = ClassroomDiagnosis.GetSpectrum(E25Normal(j5, local.residual, 0, fit.Size).gram, 5);
                            var residual = col.Select(x => (double)x).ToArray();
                            foreach (var q in basis) Remove(q, residual);
                            double norm = L2(col.Select(x => (double)x));
                            double orthogonal = L2(residual) / Math.Max(1e-30, norm);
                            double logI = L2(j4[3].Select(x => (double)x));
                            double forward = probe.plusLoss, backward = probe.minusLoss;
                            csv.WriteLine(string.Join(",", mode, lamp + 1, F(E26Kelvin[lamp]), F(step), F(probe.step), probe.stencil,
                                F(norm), F(logI), F(Cos(col, j4[3])), F(Cos(col, refCol)), F(probe.directionCos), F(orthogonal), r4, spectrum.rank,
                                F(spectrum.cutoff / Math.Max(1e-30, spectrum.eigen.Max())), F(qrRelative), F(noise), F(norm * step / Math.Max(1e-8, noise)), F(forward), F(backward), config.target.shadows ? Flag(unchanged).ToString(Invariant) : "n/a"));
                        }
                        foreach (double k in new[] { 2500d, 9000 })
                        {
                            var at = (double[])truth.Clone(); at[E26RigWidth + lamp] = k == 2500 ? 1 : 0;
                            var probe = TemperatureColumn(at, lamp, mode, fit, 5);
                            bounds.WriteLine(string.Join(",", mode, lamp + 1, F(k), F(probe.step), probe.stencil, Flag(ColorLegal(at)), F(L2(probe.column.Select(x => (double)x)))));
                        }
                    }
                }
                File.WriteAllText(Path.Combine(directory, "diagnostic-cameras.txt"), (low.RenderCount - begin).ToString(Invariant));

                E23Layout[] AuditNative(string phase, double[] rig, E23Layout[] baseline)
                {
                    ColorApply(rig, LightShadows.Hard);
                    return E25DefaultViews.Select(v =>
                    {
                        var layout = atlas.Observe(() => low.Capture(v));
                        for (int lamp = 0; lamp < lights.Length; ++lamp)
                        {
                            var lc = lights[lamp]; var c = layout.finalColors[lamp];
                            native.WriteLine(string.Join(",", phase, v, lamp + 1, F(lc.colorTemperature), F(1e6 / lc.colorTemperature), F(lc.intensity),
                                Series(new double[] { lc.color.r, lc.color.g, lc.color.b }), Series(new double[] { c.r, c.g, c.b }), layout.width, layout.height, layout.faces.Length,
                                baseline == null ? "1" : Flag(layout.Signature == baseline[v].Signature).ToString(Invariant)));
                        }
                        return layout;
                    }).ToArray();
                }
                static double Cos(float[] a, float[] b) => E21Dot(a, b, 0, a.Length) / Math.Max(1e-30, Math.Sqrt(E21Dot(a, a, 0, a.Length) * E21Dot(b, b, 0, b.Length)));
                static void Remove(double[] q, double[] c)
                {
                    double coeff = 0;
                    for (int i = 0; i < c.Length; ++i) coeff += q[i] * c[i];
                    for (int i = 0; i < c.Length; ++i) c[i] -= coeff * q[i];
                }
                static (double[][] basis, int rank) Basis(float[][] columns, double relativeCutoff)
                {
                    var rest = columns.Select(c => c.Select(x => (double)x).ToArray()).ToList();
                    double cutoff = relativeCutoff * rest.Max(L2);
                    var orthonormal = new System.Collections.Generic.List<double[]>();
                    while (rest.Count > 0)
                    {
                        var (q, norm) = rest.Select(c => (c, L2(c))).OrderByDescending(x => x.Item2).First();
                        if (!(norm > cutoff)) break;
                        rest.Remove(q);
                        for (int i = 0; i < q.Length; ++i) q[i] /= norm;
                        orthonormal.Add(q);
                        foreach (var c in rest) { Remove(q, c); Remove(q, c); }
                    }
                    return (orthonormal.ToArray(), orthonormal.Count);
                }
            }

            private (float[] column, double step, int stencil, double directionCos, double plusLoss, double minusLoss)
                TemperatureColumn(double[] rig, int lamp, LightShadows shadows, E23FrameObjective fit, double step)
            {
                int j = E26RigWidth + lamp;
                var probe = (double[])rig.Clone(); double h = step / MiredSpan;
                bool plus = rig[j] + h <= 1, minus = rig[j] - h >= 0;
                double d = plus ? h : -h;
                if (!plus && !minus) throw new ArgumentException("No legal temperature stencil.");
                int stencil = plus && minus ? 3 : rig[j] + 2 * d >= 0 && rig[j] + 2 * d <= 1 ? 2 : 1;
                var p = new float[fit.Size]; var m = new float[fit.Size]; var c = new float[fit.Size];
                ColorCapture(rig, shadows, c); double cf = fit.Evaluate(c, c);
                probe[j] = rig[j] + d; ColorCapture(probe, shadows, p); double pf = fit.Evaluate(p, p);
                double mf = double.NaN;
                if (stencil != 1) { probe[j] = rig[j] + (stencil == 3 ? -d : 2 * d); ColorCapture(probe, shadows, m); mf = fit.Evaluate(m, m); }
                var (a, b) = E24Weights(d * MiredSpan, stencil);
                var col = new float[fit.Size]; double pp = 0, mm = 0, pm = 0;
                for (int i = 0; i < col.Length; ++i)
                {
                    double u = (double)p[i] - c[i], v = stencil == 1 ? 0 : (double)m[i] - c[i];
                    col[i] = (float)(a * u + b * v); pp += u * u; mm += v * v; pm -= u * v;
                }
                return (col, d * MiredSpan, stencil, stencil == 3 && pp > 0 && mm > 0 ? pm / Math.Sqrt(pp * mm) : double.NaN, pf - cf, mf - cf);
            }

            private (string group, int[] lamps)[] E26ColorPairs(double[] truth, E26ColorTarget hard, E26ColorTarget off)
            {
                using var csv = Csv("color-coupling.csv", "lamp_a,lamp_b,hard_weighted_linear_rgb_cos_squared,off_weighted_linear_rgb_cos_squared,selected");
                int begin = low.RenderCount;
                var matrix = new System.Collections.Generic.Dictionary<bool, double[,] >();
                foreach (var target in new[] { hard, off })
                {
                    var mode = target.target.shadows ? LightShadows.Hard : LightShadows.None;
                    var col = new float[lights.Length][];
                    // Full scene remains on. The mixed native RGB logI response measures coupling
                    // without decomposing the target into isolated per-lamp renders.
                    for (int lamp = 0; lamp < lights.Length; ++lamp)
                    {
                        int axis = 4 * lamp + 3;
                        double step = .04 / units[3];
                        var moved = (double[])truth.Clone();
                        col[lamp] = new float[target.fit.Size]; var minus = new float[target.fit.Size];
                        moved[axis] += step; ColorCapture(moved, mode, col[lamp]);
                        moved[axis] -= 2 * step; ColorCapture(moved, mode, minus);
                        for (int i = 0; i < minus.Length; ++i) col[lamp][i] = (float)((col[lamp][i] - minus[i]) * target.fit.weights[i / E23ChannelsPerView]);
                    }
                    var h = E25Normal(col, new float[target.fit.Size], 0, target.fit.Size).gram;
                    var cos2 = new double[lights.Length, lights.Length];
                    for (int i = 0; i < lights.Length; ++i) for (int j = i + 1; j < lights.Length; ++j)
                        cos2[i, j] = h[i, j] * h[i, j] / Math.Max(1e-30, h[i, i] * h[j, j]);
                    matrix.Add(target.target.shadows, cos2);
                }
                var pairs = Enumerable.Range(0, lights.Length).SelectMany(i => Enumerable.Range(i + 1, lights.Length - i - 1).Select(j => (i, j))).ToArray();
                var sorted = pairs.OrderBy(p => matrix[true][p.i, p.j]).ToArray();
                var lowPair = sorted[0]; var highPair = sorted[sorted.Length - 1];
                foreach (var p in pairs) csv.WriteLine(string.Join(",", p.i + 1, p.j + 1, F(matrix[true][p.i, p.j]), F(matrix[false][p.i, p.j]),
                    p == lowPair ? "low" : p == highPair ? "high" : ""));
                File.WriteAllText(Path.Combine(directory, "diagnostic-cameras.txt"), (low.RenderCount - begin).ToString(Invariant));
                return new[] { ("low", new[] { lowPair.i, lowPair.j }), ("high", new[] { highPair.i, highPair.j }) };
            }
        }
    }
}

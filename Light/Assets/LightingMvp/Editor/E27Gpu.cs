using System;
using System.IO;
using System.Linq;
using UnityEngine;
using Lm = LightingMvp.ImageJacobianLm;
using Mapping = LightingMvp.NativeRenderBackend.HdrMapping;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            private sealed partial class E27Trial
            {
                private void GpuParity()
                {
                    using var csv = Open("gpu-parity.csv", "arm,input,dimension,views,stencil,axis,what,relative_l2,absolute_l2,step_cos,pass");
                    bool all = true;
                    using var gradient = Open("gradient-synthetic.csv", "arm,input,method,direction,task_dot,upgrad_nonnegative,finite");
                    foreach (var (name, g) in new[] {
                        ("collinear", new[] {new[] {1d, 0}, new[] {2d, 0}, new[] {3d, 0}}),
                        ("orthogonal", new[] {new[] {1d, 0, 0}, new[] {0d, 1, 0}, new[] {0d, 0, 1}}),
                        ("opposite", new[] {new[] {1d, 0}, new[] {-1d, 0}, new[] {0d, 0}}),
                        ("unbalanced", new[] {new[] {1000d, 1}, new[] {-1d, .02}, new[] {3d, -1}}),
                        ("zeros", new[] {new[] {0d, 0}, new[] {0d, 0}, new[] {0d, 0}}),
                        ("rank_deficient", new[] {new[] {1d, 2}, new[] {-2d, -4}, new[] {3d, 6}})
                    })
                    foreach (string policy in new[] { "G0", "G1", "G2", "G3" })
                    {
                        var v = E27Aggregate(policy, g); var products = g.Select(x => Dot(x, v)).ToArray();
                        bool finite = v.All(x => !double.IsNaN(x) && !double.IsInfinity(x));
                        bool feasible = products.Select((p, i) => p >= -1e-9 * Math.Max(1, L2(g[i]) * L2(v))).All(x => x);
                        gradient.WriteLine(string.Join(",", arm.id, name, policy, T(v), T(products), Flag(feasible), Flag(finite)));
                        all &= finite && (policy != "G3" || feasible);
                    }
                    foreach (var kind in new[] { "native", "synthetic_signed_hdr" })
                    foreach (int views in new[] { 1, 2, 3, 5 })
                    {
                        var ids = Enumerable.Range(0, views).ToArray();
                        int n = E23ChannelsPerView * views;
                        var hard = new float[n]; var off = new float[n]; var plus = new float[n]; var minus = new float[n]; var r = new float[n];
                        if (kind == "native")
                        {
                            Capture(sample.initial, LightShadows.Hard, ids, hard);
                            Capture(sample.initial, LightShadows.None, ids, off);
                            var gt = new float[n]; Capture(sample.truth, LightShadows.Hard, ids, gt);
                            new E23FrameObjective(gt, Mapping.SignedLog, target.scale).Evaluate(hard, r);
                        }
                        else
                            for (int i = 0; i < n; ++i)
                            { hard[i] = (float)((i % 29 - 14) * (i % 17 == 0 ? 17 : .025)); off[i] = (i % 19 - 9) * .073f; r[i] = (i % 33 - 16) * 1e-5f; }
                        var fit = new E23FrameObjective(hard, Mapping.SignedLog, target.scale);
                        var mapped = hard.Select(fit.Transform).ToArray();
                        using var gpu = new E25GpuColumns(fit); gpu.Anchor(hard, off);
                        foreach (int stencil in new[] { 1, 2, 3 })
                        {
                            var cpuBase = new float[6][]; var gpuBase = new float[6][];
                            for (int axis = 0; axis < 6; ++axis)
                            {
                                cpuBase[axis] = new float[n]; gpuBase[axis] = new float[n];
                                if (axis < 3)
                                {
                                    double d = .02 / h.units[axis];
                                    if (kind == "native")
                                    {
                                        // Same rendered probes go to both backends. For one-sided tests use the same base and stencil as the solver.
                                        var probe = (double[])sample.initial.Clone(); probe[axis] += d;
                                        Capture(probe, LightShadows.None, ids, plus);
                                        if (stencil != 1)
                                        {
                                            probe[axis] = sample.initial[axis] + (stencil == 2 ? 2 * d : -d);
                                            Capture(probe, LightShadows.None, ids, minus);
                                        }
                                    }
                                    else for (int i = 0; i < n; ++i)
                                    { plus[i] = off[i] + (i % (31 + axis) - 15) * .021f; minus[i] = off[i] - (i % (43 + axis) - 21) * .019f; }
                                    E25CachedColumn(fit, hard, mapped, off, plus, minus, null, d, stencil, cpuBase[axis]);
                                    gpu.Position(plus, minus, null, 0, d, stencil, gpuBase[axis]);
                                }
                                else
                                {
                                    // GPU never evaluates non-XYZ columns; use identical host residual derivatives on both sides.
                                    if (kind == "native" && axis == 3)
                                    {
                                        var probe = (double[])sample.initial.Clone(); probe[3] += .04 / h.units[3];
                                        Capture(probe, LightShadows.Hard, ids, plus);
                                        fit.Evaluate(plus, cpuBase[axis], factor: h.units[3] / .04);
                                    }
                                    else for (int i = 0; i < n; ++i) cpuBase[axis][i] = (i % (37 + axis) - 18) * .00001f;
                                    Array.Copy(cpuBase[axis], gpuBase[axis], n);
                                }
                            }
                            foreach (int width in new[] { 4, 5, 6 })
                                all &= CompareGpuNormal(csv, kind == "native" && width > 4 ? "native_xyz_synthetic_extras" : kind,
                                    ids.Length, stencil, cpuBase.Take(width).ToArray(), gpuBase.Take(width).ToArray(), r);
                        }
                    }
                    all &= GpuNativeColoredLayouts(csv);
                    result.stop = "fixed_gpu_parity";
                    if (!all) { result.status = "failed"; result.reason = "numeric parity tolerance exceeded"; }
                }

                private bool GpuNativeColoredLayouts(StreamWriter csv)
                {
                    // Non-XYZ derivatives remain on the host. Assemble full 5D/6D normal equations with
                    // different real URP probes for mired and u/v, not copies of fabricated columns.
                    bool all = true;
                    int rigWidth = 4 * h.room.Count;
                    foreach (int width in new[] { 5, 6 })
                    foreach (int views in new[] { 1, 2, 3, 5 })
                    foreach (int stencil in new[] { 1, 2, 3 })
                    {
                        var test = new E27Case { mode = width == 5 ? "white" : "colored" };
                        var extras = Enumerable.Repeat(.4, (width - 4) * h.room.Count).ToArray();
                        test.initial = sample.initial.Concat(extras).ToArray();
                        test.truth = sample.truth.Concat(extras).ToArray();
                        if (stencil != 3) test.initial[rigWidth] = 0; // lower mired or log-ratio bound
                        if (width == 6 && stencil != 3) test.initial[rigWidth + 1] = 1; // upper log-ratio bound
                        int[] ids = Enumerable.Range(0, views).ToArray();
                        int n = E23ChannelsPerView * views;
                        var hard = new float[n]; var off = new float[n]; var plus = new float[n];
                        var minus = new float[n]; var gt = new float[n]; var residual = new float[n];
                        Capture(test.truth, LightShadows.Hard, ids, gt, test);
                        Capture(test.initial, LightShadows.Hard, ids, hard, test);
                        Capture(test.initial, LightShadows.None, ids, off, test);
                        var fit = new E23FrameObjective(gt, Mapping.SignedLog, E25Scale(gt));
                        fit.Evaluate(hard, residual);
                        var mapped = hard.Select(fit.Transform).ToArray();
                        using var gpu = new E25GpuColumns(fit);
                        gpu.Anchor(hard, off);
                        var host = new float[width][]; var device = new float[width][];
                        var theta = test.initial.Take(4).Concat(test.initial.Skip(rigWidth).Take(width - 4)).ToArray();
                        for (int axis = 0; axis < width; ++axis)
                        {
                            int index = axis < 4 ? axis : rigWidth + axis - 4;
                            double step = axis < 3 ? .02 / h.units[axis] : axis == 3 ? .04 / h.units[3] : width == 5 ? 5 / MiredSpan : .03 / 6;
                            double center = test.initial[index];
                            bool p = center + step <= 1, m = center - step >= 0;
                            if (stencil != 3 && !p && m) step = -step;
                            var a = (double[])test.initial.Clone(); var b = (double[])test.initial.Clone();
                            bool legal = false;
                            for (int half = 0; half <= 10; ++half)
                            {
                                a[index] = center + step;
                                b[index] = center + (stencil == 2 ? 2 * step : -step);
                                bool Aligned() => Legal(a) && (stencil == 1 || Legal(b));
                                if (Aligned()) { legal = true; break; }
                                if (stencil != 3)
                                {
                                    step = -step;
                                    a[index] = center + step; b[index] = center + 2 * step;
                                    if (Aligned()) { legal = true; break; }
                                }
                                step *= .5;
                            }
                            if (!legal) throw new InvalidDataException($"No legal native GPU parity stencil: {width}D axis {axis}, stencil {stencil}.");
                            var mode = axis < 3 ? LightShadows.None : LightShadows.Hard;
                            Capture(a, mode, ids, plus, test);
                            if (stencil != 1) Capture(b, mode, ids, minus, test);
                            host[axis] = new float[n]; device[axis] = new float[n];
                            if (axis < 3)
                            {
                                E25CachedColumn(fit, hard, mapped, off, plus, minus, null, step, stencil, host[axis]);
                                gpu.Position(plus, minus, null, 0, step, stencil, device[axis]);
                            }
                            else
                            {
                                for (int i = 0; i < n; ++i)
                                    host[axis][i] = (float)(fit.weights[i / E23ChannelsPerView] * Difference(mapped[i],
                                        fit.Transform(plus[i]), stencil == 1 ? 0 : fit.Transform(minus[i]), step, stencil));
                                Array.Copy(host[axis], device[axis], n);
                            }
                        }
                        all &= CompareGpuNormal(csv, stencil == 3 ? "native_all_axes" : "native_all_axes_bound",
                            views, stencil, host, device, residual, theta);
                    }
                    return all;
                }

                private bool CompareGpuNormal(StreamWriter csv, string kind, int views, int stencil, float[][] cpu, float[][] dev,
                    float[] residual, double[] theta = null)
                {
                    int width = cpu.Length;
                    bool all = true;
                    for (int i = 0; i < width; ++i) Compare("column", i, cpu[i].Select(x => (double)x).ToArray(), dev[i].Select(x => (double)x).ToArray());
                    var a = E25Normal(cpu, residual, 0, residual.Length); var b = E25Normal(dev, residual, 0, residual.Length);
                    Compare("gram", -1, a.gram.Cast<double>().ToArray(), b.gram.Cast<double>().ToArray());
                    Compare("gradient", -1, a.gradient, b.gradient);
                    var axes = h.E26Axes().ToList();
                    if (width == 5) axes.Add(new Lm.TrustAxis(Lm.TrustCoordinate.Mired, MiredSpan, 5, 20, 50, MinMired));
                    if (width == 6) axes.AddRange(new[] {
                        new Lm.TrustAxis(Lm.TrustCoordinate.LogRedGreen, 6, .03, .25, .75, -3),
                        new Lm.TrustAxis(Lm.TrustCoordinate.LogBlueGreen, 6, .03, .25, .75, -3) });
                    var l = new Lm.TrustLayout(axes.ToArray()); var x = theta ?? Enumerable.Repeat(.5, width).ToArray();
                    Compare("step", -1, Lm.LampScaledTrustProposal(a.gram, a.gradient, x, l, null, out _),
                        Lm.LampScaledTrustProposal(b.gram, b.gradient, x, l, null, out _));
                    return all;

                    void Compare(string what, int axis, double[] aa, double[] bb)
                    {
                        double d = L2(aa.Zip(bb, (x0, x1) => x0 - x1)); double an = L2(aa), bn = L2(bb);
                        double cos = an > 1e-12 && bn > 1e-12 ? Cos(aa, bb) : double.NaN;
                        bool pass = what != "step" ? d <= Math.Max(1e-12, 1e-4 * an) :
                            an > 1e-12 && bn > 1e-12 ? cos >= .9999 : d <= 1e-12;
                        all &= pass;
                        csv.WriteLine(string.Join(",", arm.id, kind, width, views, stencil, axis, what, F(d / Math.Max(1e-12, an)), F(d), F(cos), Flag(pass)));
                    }
                }
            }
        }
    }
}

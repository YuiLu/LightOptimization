using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;
using Mapping = LightingMvp.NativeRenderBackend.HdrMapping;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            private sealed class E23Anchor
            {
                internal string name;
                internal int seed;
                internal bool failed, full;
                internal double[] rig;
                internal int[] selected;
            }

            private IEnumerable<E23Anchor> E23Anchors()
            {
                foreach (int seed in E23Seeds)
                    yield return new E23Anchor { name = $"seed{seed}_start", seed = seed, rig = E23Near(seed), selected = new[] { 4, 6 }, full = seed == 131 };
                foreach (int seed in E23Seeds)
                foreach (string loss in new[] { "B0", "B1" })
                {
                    var arm = E23SourceArm(seed, loss);
                    if (arm.pass) continue;
                    var selected = Enumerable.Range(0, arm.lamps.Length)
                        .OrderByDescending(j => Math.Max(arm.positionCm[j] / 5, arm.intensityRelative[j] / .05))
                        .Take(2).Select(j => arm.lamps[j] - 1).OrderBy(j => j).ToArray();
                    yield return new E23Anchor { name = $"seed{seed}_{(loss == "B0" ? "raw" : "log")}_failed", seed = seed,
                        rig = arm.final, failed = true, selected = selected, full = seed == 157 && loss == "B1" };
                }
                yield return new E23Anchor { name = "gt", seed = 0, rig = room.Truth, selected = new[] { 4, 6 } };
            }

            private readonly struct E23Stencil
            {
                internal readonly double[] first, second;
                internal readonly double firstStep, secondStep, a, b, c;
                internal readonly int code, blocked;
                internal E23Stencil(double[] first, double[] second, double firstStep, double secondStep, double a, double b, double c, int code, int blocked)
                { this.first = first; this.second = second; this.firstStep = firstStep; this.secondStep = secondStep; this.a = a; this.b = b; this.c = c; this.code = code; this.blocked = blocked; }
            }

            private E23Stencil E23SelectStencil(double[] rig, int axis, double requested)
            {
                double h = requested / units[axis % 4];
                int blocked = 0;
                for (int k = 0; k < 8; ++k, h *= .5)
                {
                    var p = Shift(h); var m = Shift(-h);
                    bool plus = room.Contains(p), minus = room.Contains(m);
                    blocked += (plus ? 0 : 1) + (minus ? 0 : 1);
                    if (plus && minus) return new E23Stencil(p, m, h, -h, .5 / h, -.5 / h, 0, 3, blocked);
                    if (plus || minus)
                    {
                        double d = plus ? h : -h;
                        var s = Shift(2 * d);
                        if (room.Contains(s)) return new E23Stencil(plus ? p : m, s, d, 2 * d, 2 / d, -.5 / d, -1.5 / d, 2, blocked);
                        return new E23Stencil(plus ? p : m, null, d, 0, 1 / d, 0, -1 / d, 1, blocked + 1);
                    }
                }
                return new E23Stencil(null, null, 0, 0, 0, 0, 0, 0, blocked);

                double[] Shift(double offset) { var x = (double[])rig.Clone(); x[axis] += offset; return x; }
            }

            private sealed class E23Column
            {
                internal float[] on, off, shadow;
                internal float[][] exact;
                internal E23Stencil stencil;
                internal double closure;
            }

            private void E23ShadowDecomposition()
            {
                var gt = E22Target(true);
                var rawTarget = new float[E23Channels];
                for (int v = 0; v < E19FitViews; ++v)
                for (int p = 0, k = v * E23ChannelsPerView; p < gt.fit[v].PixelCount; ++p)
                { var c = gt.fit[v].Pixel(p); rawTarget[k++] = c.r; rawTarget[k++] = c.g; rawTarget[k++] = c.b; }
                var all = E23Maps.Select(key => (key, objective: new E23FrameObjective(gt.fit, E23Mapping(key), e23Scale))).ToArray();
                using var groups = Csv("diagnostic-poses.csv", "pose,seed,failed,full_28d,selected,theta");
                using var columns = Csv("gradient-decomposition.csv", "pose,seed,mapping,lamp,axis,requested_step,coordinate_scale,first_step,second_step,stencil,blocked,on_rgb_norm,off_rgb_norm,shadow_rgb_norm,raw_closure_relative,on_chain_norm,off_chain_norm,shadow_chain_norm,exact_mapped_norm,chain_exact_cosine,chain_relative_error,g_on,g_off,g_shadow,g_exact,cross_20mm_on,cross_20mm_off,cross_20mm_shadow,cross_20mm_mapped");
                using var spectrum = Csv("decomposition-spectrum.csv", "pose,mapping,model,dimensions,lamps,region,rank,condition,singular_values,cross_gram_norm,gram_norm,gradient_norm,gradient_versus_exact_cosine,gradient_fraction,gram_fraction");
                using var grams = Csv("decomposition-gram.csv", "pose,mapping,model,region,axis,values");
                using var directions = Csv("direction-probes.csv", "pose,seed,failed,mapping,model,alpha,dimensions,length,iteration,legal,status,base_loss,predicted,actual,rho,decision,shrinks,theta,cameras,seconds");
                using var masks = Csv("response-regions.csv", "pose,view,region,pixels,overlap_with_raw1,reference_pixels,score_total");
                using var gains = Csv("mapping-derivatives.csv", "pose,mapping,view,scope,scale,channels,negative_fraction,saturated_fraction,near_zero_derivative_fraction,mean_derivative");
                using var costs = Csv("diagnostic-costs.csv", "pose,on_cameras,off_cameras,seconds");
                float[] centreOn = new float[E23Channels], centreOff = new float[E23Channels], candidate = new float[E23Channels];
                float[] pOn = new float[E23Channels], pOff = new float[E23Channels], mOn = new float[E23Channels], mOff = new float[E23Channels];
                foreach (var anchor in smoke ? E23Anchors().Where(x => x.name == "seed157_log_failed") : E23Anchors())
                {
                    if (!room.Contains(anchor.rig)) throw new InvalidOperationException("An archived E23 diagnostic pose is no longer legal.");
                    var timer = Stopwatch.StartNew();
                    int on = 0, off = 0;
                    Capture(anchor.rig, LightShadows.Hard, centreOn); Capture(anchor.rig, LightShadows.None, centreOff);
                    bool expandedMappings = !anchor.failed && (anchor.seed == 131 || anchor.seed == 157);
                    var objectives = expandedMappings ? all : all.Where(o => o.key == "C0" || o.key == "C1" || o.key == "C3").ToArray();
                    int[] paired = anchor.selected.SelectMany(l => Enumerable.Range(4 * l, 4)).ToArray();
                    int[] axes = anchor.full && !smoke ? Enumerable.Range(0, room.Count * 4).ToArray() : paired;
                    groups.WriteLine(string.Join(",", anchor.name, anchor.seed, Flag(anchor.failed), Flag(axes.Length == 28), LampList(anchor.selected), Series(anchor.rig)));
                    var residual = objectives.Select(o => new float[E23Channels]).ToArray();
                    var derivative = objectives.Select(o => new float[E23Channels]).ToArray();
                    var baseLoss = objectives.Select((o, m) => o.objective.Evaluate(centreOn, residual[m])).ToArray();
                    for (int m = 0; m < objectives.Length; ++m)
                    for (int v = 0; v < E19FitViews; ++v)
                    {
                        var obj = objectives[m].objective;
                        int start = v * E23ChannelsPerView, end = start + E23ChannelsPerView, negative = 0, saturated = 0, small = 0;
                        double total = 0;
                        for (int i = start; i < end; ++i)
                        {
                            float x = centreOn[i];
                            double slope = E23Derivative(x, obj.mapping, obj.scale);
                            derivative[m][i] = (float)(slope * obj.weights[v]);
                            if (x < 0) ++negative;
                            if (NativeRenderBackend.Display(x) >= .99f) ++saturated;
                            if (Math.Abs(slope) <= 1e-6) ++small;
                            total += slope;
                        }
                        gains.WriteLine(string.Join(",", anchor.name, objectives[m].key, v, "current_on", F(e23Scale), E23ChannelsPerView,
                            F(negative / (double)E23ChannelsPerView), F(saturated / (double)E23ChannelsPerView), F(small / (double)E23ChannelsPerView), F(total / E23ChannelsPerView)));
                    }

                    var central = axes.ToDictionary(axis => axis, axis => Read(axis, axis % 4 == 3 ? .04 : .02));
                    foreach (int axis in axes) Describe(axis, axis % 4 == 3 ? .04 : .02, central[axis], null);
                    if (!smoke)
                    foreach (double h in new[] { .005, .05 })
                    foreach (int axis in paired.Where(i => i % 4 != 3)) Describe(axis, h, Read(axis, h), central[axis]);

                    const int pixels = E19FitW * E19FitH;
                    var rawScore = new double[pixels * E19FitViews]; var shadowScore = new double[rawScore.Length]; var changeScore = new double[rawScore.Length];
                    for (int p = 0; p < rawScore.Length; ++p)
                    for (int ch = 0; ch < 3; ++ch)
                    {
                        int i = 3 * p + ch;
                        double d = centreOn[i] - rawTarget[i], s = centreOn[i] - centreOff[i];
                        rawScore[p] += d * d; shadowScore[p] += s * s;
                    }
                    foreach (int axis in axes.Where(j => j % 4 != 3))
                    {
                        var c = central[axis];
                        double length = Math.Abs(c.stencil.firstStep);
                        for (int p = 0; p < changeScore.Length; ++p)
                        {
                            int i = 3 * p;
                            changeScore[p] += length * length * ((double)c.shadow[i] * c.shadow[i] + (double)c.shadow[i + 1] * c.shadow[i + 1] + (double)c.shadow[i + 2] * c.shadow[i + 2]);
                        }
                    }
                    var regions = new Dictionary<string, List<int>> { ["raw_top_0_1"] = new List<int>(), ["raw_top_1"] = new List<int>(), ["raw_top_5"] = new List<int>(), ["shadow_union_top_1"] = new List<int>() };
                    for (int v = 0; v < E19FitViews; ++v)
                    {
                        int start = v * pixels;
                        var raw1 = Top(rawScore, .01, start, pixels);
                        var shadow = Top(shadowScore, .01, start, pixels).Union(Top(changeScore, .01, start, pixels)).OrderBy(i => i).ToArray();
                        Add("raw_top_0_1", Top(rawScore, .001, start, pixels), rawScore);
                        Add("raw_top_1", raw1, rawScore);
                        Add("raw_top_5", Top(rawScore, .05, start, pixels), rawScore);
                        Add("shadow_union_top_1", shadow, shadowScore);

                        void Add(string key, int[] sites, double[] score)
                        {
                            masks.WriteLine(string.Join(",", anchor.name, v, key, sites.Length, sites.Intersect(raw1).Count(), pixels, F(sites.Sum(i => score[i]))));
                            regions[key].AddRange(sites.Select(i => i * 3));
                        }
                    }

                    for (int m = 0; m < objectives.Length; ++m)
                    {
                        var item = objectives[m];
                        var exact = axes.Select(axis => central[axis].exact[m]).ToArray();
                        var native = Normals(exact, residual[m]);
                        RecordGram("B0", "full", native, native);
                        foreach (var zone in regions) RecordGram("B0", zone.Key, Normals(exact, residual[m], zone.Value.ToArray()), native);
                        if (item.key != "C0" && item.key != "C1" && item.key != "C3") continue;
                        Replay("B0", 1, native);
                        foreach (double alpha in new[] { 0d, .5d, 1d })
                        {
                            string method = alpha == 0 ? "B2" : alpha == .5 ? "B3" : "B1";
                            var blended = new float[axes.Length][];
                            for (int j = 0; j < axes.Length; ++j)
                            {
                                var c = central[axes[j]];
                                if (axes[j] % 4 == 3) { blended[j] = c.exact[m]; continue; }
                                var col = blended[j] = new float[E23Channels];
                                float mix = (float)alpha;
                                for (int i = 0; i < E23Channels; ++i)
                                    col[i] = derivative[m][i] * (alpha == 1 ? c.on[i] : c.off[i] + mix * c.shadow[i]);
                            }
                            var local = Normals(blended, residual[m]);
                            RecordGram(method, "full", local, native);
                            if (alpha != 1)
                                RecordGram(method, "shadow_union_top_1", Normals(blended, residual[m], regions["shadow_union_top_1"].ToArray()), local);
                            Replay(method, alpha, local);
                        }

                        void RecordGram(string model, string region, (double[,] h, double[] g) local, (double[,] h, double[] g) reference)
                        {
                            double h = 0, cross = 0, h0 = 0;
                            for (int i = 0; i < axes.Length; ++i)
                            {
                                for (int j = 0; j < axes.Length; ++j)
                                {
                                    h += local.h[i, j] * local.h[i, j]; h0 += reference.h[i, j] * reference.h[i, j];
                                    if (axes[i] / 4 != axes[j] / 4) cross += local.h[i, j] * local.h[i, j];
                                }
                                grams.WriteLine(string.Join(",", anchor.name, item.key, model, region, axes[i], Series(Enumerable.Range(0, axes.Length).Select(j => local.h[i, j]))));
                            }
                            var sp = ClassroomDiagnosis.GetSpectrum(local.h, axes.Length);
                            double g = Math.Sqrt(local.g.Sum(x => x * x)), g0 = Math.Sqrt(reference.g.Sum(x => x * x));
                            double cos = g > 0 && g0 > 0 ? local.g.Zip(reference.g, (x, y) => x * y).Sum() / (g * g0) : double.NaN;
                            spectrum.WriteLine(string.Join(",", anchor.name, item.key, model, axes.Length, LampList(axes.Select(i => i / 4).Distinct()), region,
                                sp.rank, F(sp.condition), Series(sp.singular), F(Math.Sqrt(cross)), F(Math.Sqrt(h)), F(g), F(cos), F(SafeRatio(g, g0)), F(SafeRatio(Math.Sqrt(h), Math.Sqrt(h0)))));
                        }

                        void Replay(string method, double alpha, (double[,] h, double[] g) model)
                        {
                            var small = axes.Select(i => anchor.rig[i]).ToArray();
                            var physical = axes.Select(i => units[i % 4]).ToArray();
                            var step = ImageJacobianLm.LampScaledTrustProposal(model.h, model.g, small, physical, x => room.Contains(Expand(x)), out int shrink);
                            foreach (double multiplier in new[] { .25, .5, 1d })
                            {
                                var delta = step.Select(v => v * multiplier).ToArray();
                                var proposed = Expand(small.Zip(delta, (x, d) => x + d).ToArray());
                                bool legal = room.Contains(proposed);
                                double prediction = -2 * model.g.Zip(delta, (g, d) => g * d).Sum();
                                for (int i = 0; i < delta.Length; ++i)
                                for (int j = 0; j < delta.Length; ++j) prediction -= delta[i] * model.h[i, j] * delta[j];
                                int repeats = anchor.failed && !smoke ? 2 : 1;
                                for (int repeat = 0; repeat < repeats; ++repeat)
                                {
                                    var sw = Stopwatch.StartNew(); int before = on;
                                    double actual = double.NaN;
                                    string status = !legal ? "illegal" : baseLoss[m] <= 1e-12 || delta.All(d => Math.Abs(d) < 1e-15) ? "zero" : "measured";
                                    if (status == "measured")
                                    {
                                        Capture(proposed, LightShadows.Hard, candidate);
                                        actual = baseLoss[m] - item.objective.Evaluate(candidate);
                                    }
                                    double rho = prediction > 1e-18 ? actual / prediction : double.NaN;
                                    bool accepted = actual > Math.Max(1e-14, 1e-9 * baseLoss[m]) && rho >= .02;
                                    directions.WriteLine(string.Join(",", anchor.name, anchor.seed, Flag(anchor.failed), item.key, method, F(alpha), axes.Length,
                                        F(multiplier), repeat, Flag(legal), status, F(baseLoss[m]), F(prediction), F(actual), F(rho), Flag(accepted), shrink,
                                        Series(proposed), on - before, F(sw.Elapsed.TotalSeconds)));
                                }
                            }

                            double[] Expand(double[] x)
                            {
                                var pose = (double[])anchor.rig.Clone();
                                for (int j = 0; j < axes.Length; ++j) pose[axes[j]] = x[j];
                                return pose;
                            }
                        }
                    }
                    costs.WriteLine(string.Join(",", anchor.name, on, off, F(timer.Elapsed.TotalSeconds)));
                    columns.Flush(); spectrum.Flush(); grams.Flush(); directions.Flush(); costs.Flush();
                    UnityEngine.Debug.Log($"[E23-A] {anchor.name} {axes.Length}D, Hard={on}, Off={off} cameras, {timer.Elapsed.TotalSeconds:F1}s");

                    void Capture(double[] rig, LightShadows kind, float[] output)
                    {
                        E23CaptureRgb(rig, kind, output);
                        if (kind == LightShadows.None) off += E19FitViews; else on += E19FitViews;
                    }

                    E23Column Read(int axis, double step)
                    {
                        var s = E23SelectStencil(anchor.rig, axis, step);
                        var data = new E23Column { on = new float[E23Channels], off = new float[E23Channels], shadow = new float[E23Channels],
                            exact = objectives.Select(_ => new float[E23Channels]).ToArray(), stencil = s };
                        if (s.code == 0) return data;
                        Capture(s.first, LightShadows.Hard, pOn); Capture(s.first, LightShadows.None, pOff);
                        if (s.second != null) { Capture(s.second, LightShadows.Hard, mOn); Capture(s.second, LightShadows.None, mOff); }
                        var otherOn = s.second == null ? centreOn : mOn; var otherOff = s.second == null ? centreOff : mOff;
                        double error = 0, energy = 0;
                        for (int i = 0; i < E23Channels; ++i)
                        {
                            // Subtract before scaling, including for the second-order one-sided formula.
                            double onRaw = s.a * ((double)pOn[i] - centreOn[i]) + s.b * ((double)otherOn[i] - centreOn[i]);
                            double offRaw = s.a * ((double)pOff[i] - centreOff[i]) + s.b * ((double)otherOff[i] - centreOff[i]);
                            data.on[i] = (float)onRaw; data.off[i] = (float)offRaw;
                            double s0 = (double)centreOn[i] - centreOff[i];
                            double directShadow = s.a * ((double)pOn[i] - pOff[i] - s0) + s.b * ((double)otherOn[i] - otherOff[i] - s0);
                            data.shadow[i] = (float)directShadow;
                            double diff = (double)data.on[i] - data.off[i] - data.shadow[i];
                            error += diff * diff; energy += onRaw * onRaw;
                        }
                        data.closure = SafeRatio(Math.Sqrt(error), Math.Sqrt(energy));
                        for (int m = 0; m < objectives.Length; ++m)
                        {
                            var obj = objectives[m].objective;
                            for (int v = 0; v < E19FitViews; ++v)
                            {
                                int start = v * E23ChannelsPerView, end = start + E23ChannelsPerView;
                                double w = obj.weights[v];
                                for (int i = start; i < end; ++i)
                                {
                                    float baseValue = obj.Transform(centreOn[i]);
                                    double x = obj.Transform(pOn[i]) - baseValue, y = obj.Transform(otherOn[i]) - baseValue;
                                    data.exact[m][i] = (float)(w * (s.a * x + s.b * y));
                                }
                            }
                        }
                        return data;
                    }

                    void Describe(int axis, double step, E23Column c, E23Column before)
                    {
                        double onRaw = Math.Sqrt(E21Dot(c.on, c.on)), offRaw = Math.Sqrt(E21Dot(c.off, c.off)), shadowRaw = Math.Sqrt(E21Dot(c.shadow, c.shadow));
                        for (int m = 0; m < objectives.Length; ++m)
                        {
                            var gain = derivative[m]; var r = residual[m]; var e = c.exact[m];
                            double u = 0, v = 0, z = 0, gOn = 0, gOff = 0, gShadow = 0, dot = 0, error = 0;
                            for (int i = 0; i < E23Channels; ++i)
                            {
                                double a = gain[i] * c.on[i], b = gain[i] * c.off[i], d = gain[i] * c.shadow[i];
                                u += a * a; v += b * b; z += d * d;
                                gOn += a * r[i]; gOff += b * r[i]; gShadow += d * r[i];
                                dot += a * e[i]; double diff = a - e[i]; error += diff * diff;
                            }
                            double en = Math.Sqrt(E21Dot(e, e));
                            columns.WriteLine(string.Join(",", anchor.name, anchor.seed, objectives[m].key, axis / 4 + 1, axis % 4, F(step), F(units[axis % 4]),
                                F(c.stencil.firstStep * units[axis % 4]), F(c.stencil.secondStep * units[axis % 4]), c.stencil.code, c.stencil.blocked,
                                F(onRaw), F(offRaw), F(shadowRaw), F(c.closure), F(Math.Sqrt(u)), F(Math.Sqrt(v)), F(Math.Sqrt(z)), F(en),
                                F(SafeRatio(dot, en * Math.Sqrt(u))), F(SafeRatio(Math.Sqrt(error), en)), F(gOn), F(gOff), F(gShadow), F(E21Dot(e, r)),
                                F(before == null ? double.NaN : Cos(c.on, before.on)), F(before == null ? double.NaN : Cos(c.off, before.off)),
                                F(before == null ? double.NaN : Cos(c.shadow, before.shadow)), F(before == null ? double.NaN : Cos(e, before.exact[m]))));
                        }
                    }
                }
            }

            private static int[] Top(double[] score, double fraction, int start, int length) => Enumerable.Range(start, length)
                .Where(i => score[i] > 0).OrderByDescending(i => score[i]).Take((int)Math.Ceiling(length * fraction)).OrderBy(i => i).ToArray();

            private static double E23Derivative(float x, Mapping mapping, double scale)
            {
                double m = Math.Abs((double)x);
                if (mapping == Mapping.Raw) return 1;
                if (mapping == Mapping.SignedLog) return 1 / (scale + m);
                if (mapping == Mapping.SignedSqrt) return .5 / Math.Sqrt(m + 1e-6);
                if (x <= 0) return 0;
                if (mapping == Mapping.ToneExp && NativeRenderBackend.Display(x) >= 1) return 0;
                double t = mapping == Mapping.ToneExp ? 1 - Math.Exp(-x) : x / (scale + x);
                double d = mapping == Mapping.ToneExp ? Math.Exp(-x) : scale / ((scale + x) * (scale + x));
                double transfer = t <= .0031308 ? 12.92 : 1.055 / 2.4 * Math.Pow(t, 1 / 2.4 - 1);
                return d * transfer;
            }

            private static double SafeRatio(double x, double y) => y > 0 ? x / y : double.NaN;
            private static double Cos(float[] a, float[] b) => SafeRatio(E21Dot(a, b), Math.Sqrt(E21Dot(a, a) * E21Dot(b, b)));

            private static (double[,] h, double[] g) Normals(float[][] j, float[] r, int[] region = null)
            {
                int n = j.Length;
                var h = new double[n, n]; var g = new double[n];
                Parallel.For(0, n, new ParallelOptions { MaxDegreeOfParallelism = Math.Min(Environment.ProcessorCount, 8) }, i =>
                {
                    g[i] = Dot(j[i], r);
                    for (int k = 0; k <= i; ++k) h[i, k] = h[k, i] = Dot(j[i], j[k]);
                });
                return (h, g);

                double Dot(float[] a, float[] b)
                {
                    if (region == null) return E21Dot(a, b);
                    double sum = 0;
                    foreach (int i in region) sum += (double)a[i] * b[i] + (double)a[i + 1] * b[i + 1] + (double)a[i + 2] * b[i + 2];
                    return sum;
                }
            }
        }
    }
}

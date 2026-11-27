using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using UnityEngine;
using Mapping = LightingMvp.NativeRenderBackend.HdrMapping;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            private sealed class E25Counter
            {
                internal long mapTicks, maskTicks;
                internal static long Now(E25Counter counter) => counter == null ? 0 : Stopwatch.GetTimestamp();
                internal double Map => (double)mapTicks / Stopwatch.Frequency;
                internal double Mask => (double)maskTicks / Stopwatch.Frequency;
            }

            private static void E25CachedColumn(E23FrameObjective fit, float[] hard, float[] mappedHard, float[] off,
                float[] p, float[] m, float[] mask, double d, int stencil, float[] output)
            {
                var (a, b) = E24Weights(d, stencil);
                int i = 0;
                foreach (double w in fit.weights)
                {
                    int last = i + E23ChannelsPerView;
                    if (stencil == 3)
                    {
                        for (; i < last; ++i)
                        {
                            float v = mask == null ? 1 : mask[i / 3];
                            float plus = fit.Transform(hard[i] + v * (p[i] - off[i]));
                            float minus = fit.Transform(hard[i] + v * (m[i] - off[i]));
                            output[i] = (float)(w * a * ((double)plus - minus));
                        }
                    }
                    else for (; i < last; ++i)
                    {
                        float v = mask == null ? 1 : mask[i / 3];
                        float center = mappedHard[i];
                        double plus = fit.Transform(hard[i] + v * (p[i] - off[i])) - center;
                        double minus = stencil == 1 ? 0 : fit.Transform(hard[i] + v * (m[i] - off[i])) - center;
                        output[i] = (float)(w * (a * plus + b * minus));
                    }
                }
            }

            private static void E25ReferenceColumn(E23FrameObjective fit, float[] hard, float[] off,
                float[] p, float[] m, float[] mask, double d, int stencil, float[] output)
            {
                var (a, b) = E24Weights(d, stencil);
                for (int i = 0; i < output.Length; ++i)
                {
                    float v = mask == null ? 1 : mask[i / 3], center = fit.Transform(hard[i]);
                    double plus = fit.Transform(hard[i] + v * (p[i] - off[i])) - center;
                    double minus = stencil == 1 ? 0 : fit.Transform(hard[i] + v * (m[i] - off[i])) - center;
                    output[i] = (float)(fit.weights[i / E23ChannelsPerView] * (a * plus + b * minus));
                }
            }

            partial void E25Engineering(string key, E21StageSummary saved, MbdRgbTarget hard)
            {
                switch (key)
                {
                    case "c-audit-complex": E25Audit(E25Read("e_complex"), hard, false); break;
                    case "c-audit-classroom": E25Audit(E25Read("e_classroom"), hard, true); break;
                    case "moment-classroom-1301": E25Moment(hard, "dir1301_10cm"); break;
                    case "moment-classroom-1601": E25Moment(hard, "dir1601_10cm"); break;
                    case "d-gpu-transfer-complex": E25GpuTransfer(hard); break;
                    case "d-gpu-recovery-131-complex": E25GpuRecovery(hard, "seed131_near_on", 3); break;
                    case "d-gpu-recovery-157-complex": E25GpuRecovery(hard, "seed157_near_on", 1); break;
                    case "d-gpu-recovery-1301-classroom": E25GpuRecovery(hard, "dir1301_10cm", 1); break;
                    case "d-gpu-recovery-1601-classroom": E25GpuRecovery(hard, "dir1601_10cm", 1); break;
                    case "d-fixed-complex": E25Fixed(hard); break;
                    case "d-replay-131-complex": E25Replay(hard, "seed131_near_on", 3); break;
                    case "d-replay-157-complex": E25Replay(hard, "seed157_near_on", 1); break;
                    case "d-replay-911-complex": E25Replay(hard, "dir911_20cm", 1); break;
                    case "d-replay-1301-classroom": E25Replay(hard, "dir1301_10cm", 1); break;
                    case "d-replay-1601-classroom": E25Replay(hard, "dir1601_10cm", 1); break;
                    default: throw new ArgumentException("Unknown E25 stage: " + key);
                }
            }

            private E21Trial E25DSource(string label)
            {
                bool developer = label.StartsWith("seed", StringComparison.Ordinal);
                return E25Read(developer ? "p_full" : label is "dir1601_10cm" or "dir1301_10cm" ? "e_classroom" : "e_complex").arms.Single(a => a.scenario == label && a.method == "C4");
            }

            private void E25Replay(MbdRgbTarget hard, string scenario, int repeats)
            {
                var start = E25DSource(scenario).initial;
                // One model rebuild per method to warm both capture and numeric paths. Its cost is separate.
                foreach (string id in new[] { "C4", "C4-cache" })
                    E25Optimize(scenario + "_warmup", id, E24AllLamps, start, hard, iterations: 1, saveImages: false);
                for (int rep = 1; rep <= repeats; ++rep)
                    foreach (string id in rep % 2 == 0 ? new[] { "C4-cache", "C4" } : new[] { "C4", "C4-cache" })
                        E25Optimize(scenario + "_rep" + rep, id, E24AllLamps, start, hard, saveImages: rep == 1);
            }

            private void E25Fixed(MbdRgbTarget hard)
            {
                using var table = Csv("numeric-parity.csv", "sample,method,quantity,coordinate,reference_norm,absolute_l2,relative_l2,pass,nonfinite");
                using var data = Csv("fixed-inputs.csv", "sample,file,sha256,shape,axis,stencil,physical_step,theta,weights,scale,units,lowest,highest,negative_channels,zero_channels");
                var a = E25DSource("seed131_near_on"); var b = E25DSource("dir911_20cm");
                var target = new E23FrameObjective(hard.fit, Mapping.SignedLog, e24HardScale);
                var fixtures = new[] { ("success", a.final, a.lamps.First() - 1), ("failure", b.final, b.lamps.First() - 1),
                    ("hdr_gt", room.Truth, Enumerable.Range(0, lights.Length).OrderByDescending(i => lights[i].intensity).First()) };
                foreach (var (label, pose, lamp) in fixtures)
                {
                    var inputs = E25AuditedLamp(label, pose, lamp, target, .02, null, null);
                    if (label == "hdr_gt")
                    {
                        for (int dim = 0; dim < 3; ++dim)
                        {
                            // If both same-direction points are legal, explicitly select the second-order one-sided stencil.
                            // Natural legal one-sided probes from all three sources still take precedence otherwise.
                            if (inputs.points[dim].stencil != 3) continue;
                            int axis = 4 * lamp + dim; double d = inputs.points[dim].d;
                            var probe = (double[])pose.Clone(); probe[axis] += 2 * d;
                            if (!room.Contains(probe)) continue;
                            E25Capture(probe, LightShadows.None, E25DefaultViews, inputs.points[dim].om);
                            E25Capture(probe, LightShadows.Hard, E25DefaultViews, inputs.points[dim].hm);
                            inputs.points[dim].stencil = 2;
                            break;
                        }
                    }
                    var path = Path.Combine(directory, label + "-fixed-f32.gz");
                    using (var file = File.Create(path))
                    using (var gz = new GZipStream(file, System.IO.Compression.CompressionLevel.Fastest))
                    using (var writer = new BinaryWriter(gz))
                    {
                        writer.Write(target.Size); writer.Write(inputs.points.Length);
                        Array(inputs.hard); Array(inputs.off); Array(inputs.residual); Array(inputs.mask);
                        foreach (var p in inputs.points)
                        {
                            writer.Write(p.stencil); writer.Write(p.d);
                            Array(p.hp); Array(p.hm); Array(p.op); Array(p.om);
                        }
                        void Array(float[] f) { writer.Write(f.Length); foreach (float value in f) writer.Write(value); }
                    }
                    string sha = E21Sha(path);
                    for (int dim = 0; dim < inputs.points.Length; ++dim)
                    {
                        var pp = inputs.points[dim];
                        data.WriteLine(string.Join(",", label, Path.GetFileName(path), sha, target.Size, 4 * lamp + dim, pp.stencil,
                            F(pp.d * units[dim]), Series(pose), Series(target.weights), F(target.scale), Series(units), F(inputs.hard.Min()), F(inputs.hard.Max()),
                            inputs.hard.Count(x => x < 0), inputs.hard.Count(x => x == 0)));
                    }
                    foreach (string method in new[] { "C3", "C4" })
                    {
                        var reference = new float[4][]; var cached = new float[4][];
                        var anchor = inputs.hard.Select(target.Transform).ToArray();
                        for (int dim = 0; dim < 4; ++dim)
                        {
                            var raw = inputs.points[dim];
                            var r = reference[dim] = new float[target.Size];
                            var c = cached[dim] = new float[target.Size];
                            if (raw.stencil != 0)
                            {
                                // Intensity uses native Hard probes. In this helper zero anchor delta is exact; use Hard directly below.
                                if (dim < 3)
                                {
                                    var mask = method == "C4" ? inputs.mask : null;
                                    E25ReferenceColumn(target, inputs.hard, inputs.off, raw.op, raw.om, mask, raw.d, raw.stencil, r);
                                    E25CachedColumn(target, inputs.hard, anchor, inputs.off, raw.op, raw.om, mask, raw.d, raw.stencil, c);
                                }
                                else
                                {
                                    var (x, y) = E24Weights(raw.d, raw.stencil);
                                    for (int i = 0; i < c.Length; ++i)
                                    {
                                        var h = anchor[i]; var p = target.Transform(raw.hp[i]); var m = raw.stencil == 1 ? 0 : target.Transform(raw.hm[i]);
                                        double w = target.weights[i / E23ChannelsPerView];
                                        r[i] = (float)(w * (x * (p - h) + y * (raw.stencil == 1 ? 0 : m - h)));
                                        c[i] = raw.stencil == 3 ? (float)(w * x * ((double)p - m)) : r[i];
                                    }
                                }
                            }
                            Parity("column", dim, r.Select(v => (double)v), c.Select(v => (double)v));
                        }
                        var p0 = E25Normal(reference, inputs.residual, 0, target.Size);
                        var p1 = E25Normal(cached, inputs.residual, 0, target.Size);
                        Parity("gram", -1, p0.gram.Cast<double>(), p1.gram.Cast<double>());
                        Parity("gradient", -1, p0.gradient, p1.gradient);
                        void Parity(string name, int dim, System.Collections.Generic.IEnumerable<double> p, System.Collections.Generic.IEnumerable<double> q)
                        {
                            double n2 = 0, d2 = 0; int bad = 0;
                            using var it = q.GetEnumerator();
                            foreach (double x in p)
                            {
                                if (!it.MoveNext()) throw new InvalidDataException("Parity dimension mismatch.");
                                double y = it.Current;
                                if (double.IsNaN(x) || double.IsInfinity(x) || double.IsNaN(y) || double.IsInfinity(y)) { ++bad; continue; }
                                n2 += x * x; d2 += (x - y) * (x - y);
                            }
                            double n = Math.Sqrt(n2), d = Math.Sqrt(d2), rel = d / Math.Max(n, 1e-12);
                            bool pass = bad == 0 && d <= Math.Max(1e-12, 1e-6 * n);
                            table.WriteLine(string.Join(",", label, method, name, dim, F(n), F(d), F(rel), Flag(pass), bad));
                        }
                    }
                    data.Flush(); table.Flush();
                }
                // Explicit signed/near-zero/saturated arithmetic cases supplement the real captures.
                var h0 = new float[E23ChannelsPerView]; var o0 = new float[h0.Length];
                var p0s = new float[h0.Length]; var m0 = new float[h0.Length];
                var signed = new[] { -1e5f, -4f, -1e-8f, 0f, 1e-8f, 4f, 1e5f };
                for (int i = 0; i < h0.Length; ++i)
                {
                    h0[i] = signed[i % 7]; o0[i] = signed[(i + 3) % 7];
                    p0s[i] = o0[i] + signed[(i + 2) % 7]; m0[i] = o0[i] - signed[(i + 5) % 7];
                }
                var f0 = new E23FrameObjective(h0, Mapping.SignedLog, e24HardScale);
                var anchor0 = h0.Select(f0.Transform).ToArray();
                foreach (int stencil in new[] { 1, 2, 3 })
                {
                    var a0 = new float[h0.Length]; var a1 = new float[h0.Length];
                    E25ReferenceColumn(f0, h0, o0, p0s, m0, null, .025, stencil, a0);
                    E25CachedColumn(f0, h0, anchor0, o0, p0s, m0, null, .025, stencil, a1);
                    var norm = Math.Sqrt(E21Dot(a0, a0));
                    var delta = L2(a0.Zip(a1, (x, y) => (double)x - y));
                    int bad = a1.Count(v => float.IsInfinity(v) || float.IsNaN(v));
                    table.WriteLine(string.Join(",", "synthetic_signed_stencil_" + stencil, "C3", "column", 0, F(norm), F(delta),
                        F(delta / Math.Max(1e-12, norm)), Flag(bad == 0 && delta <= Math.Max(1e-12, 1e-6 * norm)), bad));
                }
            }
        }
    }
}

using System;
using System.Collections.Generic;
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
            private static readonly double[][] e27Moderate = { new[] { 1d, .65, .40 }, new[] { .40, 1d, .65 }, new[] { .55, .40, 1d } };
            private static readonly double[][] e27High = { new[] { 1d, .12, .10 }, new[] { .10, 1d, .15 }, new[] { .12, .15, 1d } };
            private E27Case E27DerivedCase(string key, E27Job job)
            {
                if (e27Derived.TryGetValue(key, out var saved)) return saved;
                int count = room.Count, width = 4 * count;
                int[] active = Enumerable.Range(0, count).ToArray();
                var tokens = key.Split('-');
                string axes = tokens.Last(), mode = "original", scene = "complex";
                double[] xyz = room.Truth, truth, start;
                int seed = 5107, perturbIndex = 0;
                bool strong = false, temperatureStrong = false, intensityStrong = false;
                if (key.StartsWith("color-", StringComparison.Ordinal))
                {
                    mode = "colored";
                    var group = tokens[1];
                    perturbIndex = int.TryParse(tokens[2], out int i) ? i : 0;
                    seed = 5303 + 73 * perturbIndex;
                    strong = key.EndsWith("high", StringComparison.Ordinal) || group == "single" && perturbIndex >= 6;
                    if (group == "small" || group == "single")
                    {
                        active = new[] { new[] { 0, 2, 5 }[(perturbIndex % 6) / 2] };
                        if (group == "single") xyz = Frozen(perturbIndex % 2 == 0 ? "OK20" : "OK30").initial;
                    }
                    else if (group == "pair")
                    {
                        active = E27CoupledPairs(job)[perturbIndex / 2];
                        xyz = Frozen(perturbIndex % 2 == 0 ? "OK20" : "OK30").initial;
                    }
                    else if (group == "full")
                    {
                        seed = perturbIndex;
                        xyz = perturbIndex switch {
                            911 => Frozen("OK20").initial, 1201 => Frozen("OK30").initial,
                            509 => Frozen("CF-O").initial, 2003 or 2909 => Frozen("W28-" + perturbIndex).initial,
                            3109 or 4211 => E27LegalPose(active, perturbIndex, perturbIndex == 3109 ? .20 : .30, .22),
                            _ => throw new ArgumentException(key)
                        };
                    }
                    else if (group == "audit") axes = "xyziuv";
                    else return null;
                }
                else if (key.StartsWith("room-", StringComparison.Ordinal))
                {
                    scene = "room"; mode = axes.Contains('u') || tokens[1] == "colored" ? "colored" : "white";
                    perturbIndex = int.Parse(tokens[tokens[1] == "colored" ? 2 : 1], Invariant);
                    active = new[] { perturbIndex == 0 ? 0 : 3 };
                    seed += 43 * perturbIndex;
                    xyz = E27LegalPose(active, seed, perturbIndex == 0 ? .12 : .20, .18);
                }
                else if (key.StartsWith("class-white-", StringComparison.Ordinal))
                {
                    mode = "white"; scene = "classroom";
                    if (tokens[2] == "all") { xyz = Frozen("C13").initial; seed = 4583; }
                    else
                    {
                        active = new[] { int.Parse(tokens[2], Invariant) - 1 };
                        perturbIndex = int.Parse(tokens[3], Invariant);
                        seed = 6101 + 11 * active[0] + 53 * perturbIndex;
                        xyz = Frozen(perturbIndex == 0 ? "C13" : "C16").initial;
                    }
                }
                else if (key.StartsWith("complex-white-", StringComparison.Ordinal))
                {
                    mode = "white"; seed = int.Parse(tokens[2], Invariant);
                    xyz = seed == 911 ? Frozen("OK20").initial : seed == 1201 ? Frozen("OK30").initial : Frozen("W28-2003").initial;
                    temperatureStrong = true; intensityStrong = tokens.Contains("intensity");
                }
                else if (key == "temperature-6500")
                { mode = "white"; axes = "xyzit"; active = new[] { 2 }; }
                else if (key.StartsWith("new-", StringComparison.Ordinal))
                {
                    seed = int.Parse(tokens[1], Invariant); axes = "xyzi";
                    xyz = E27LegalPose(active, seed, seed == 3109 ? .20 : .30, .22);
                }
                else return null;
                if (xyz == null) throw new InvalidDataException("No legal sample for frozen direction seed " + seed);
                var gt = xyz.Take(width).ToArray(); Array.Copy(room.Truth, gt, width);
                truth = mode switch {
                    "white" => gt.Concat(Enumerable.Range(0, count).Select(i => EncodeMired(1e6 / E26Kelvin[i % E26Kelvin.Length]))).ToArray(),
                    "colored" => gt.Concat(Enumerable.Range(0, count).SelectMany(i => {
                        var rgb = (strong ? e27High : e27Moderate)[i % 3]; return new[] { (Math.Log(rgb[0] / rgb[1]) + 3) / 6, (Math.Log(rgb[2] / rgb[1]) + 3) / 6 };
                    })).ToArray(), _ => gt
                };
                start = (double[])truth.Clone();
                var rng = new System.Random(seed);
                foreach (int lamp in active)
                {
                    if (axes.Contains('x')) Array.Copy(xyz, 4 * lamp, start, 4 * lamp, 3);
                    if (axes.Contains('i'))
                    {
                        if (intensityStrong || key.StartsWith("color-small", StringComparison.Ordinal))
                            start[4 * lamp + 3] = E27BoundedDelta(truth[4 * lamp + 3], (intensityStrong ? .35 : .20) / LogPowerSpan, rng);
                        else if (!temperatureStrong) start[4 * lamp + 3] = xyz[4 * lamp + 3];
                    }
                    if (mode == "white" && (axes.Contains('t') || temperatureStrong))
                    {
                        // Keep the frozen draw stream identical in 35D and fixed-GT-temperature
                        // 28D. Only the applied temperature differs; every XYZ/I pair matches.
                        double shifted = E27BoundedDelta(truth[width + lamp],
                            (temperatureStrong ? 80 : perturbIndex % 2 == 0 ? 20 : 40) / MiredSpan, rng);
                        if (axes.Contains('t')) start[width + lamp] = shifted;
                    }
                    if (mode == "colored" && axes.Contains('u'))
                    {
                        int i = width + 2 * lamp; double magnitude = strong ? .8 : (lamp + perturbIndex) % 2 == 0 ? .2 : .4;
                        start[i] = E27BoundedDelta(truth[i], magnitude / 6, rng);
                        start[i + 1] = E27BoundedDelta(truth[i + 1], magnitude / 6, rng);
                    }
                }
                var derived = new E27Case { id = key, scene = scene, target = "Hard", mode = mode, axes = axes, initial = start, truth = truth, active = active };
                e27Derived.Add(key, derived);
                return derived;

                E27Case Frozen(string id) => job.cases.Single(c => c.id == id);
            }

            private static double E27BoundedDelta(double value, double step, System.Random rng)
            {
                for (int i = 0; i < 16; ++i) { double x = value + (rng.Next(2) == 0 ? -step : step); if (x >= 0 && x <= 1) return x; }
                throw new InvalidDataException("No feasible encoded offset of the preregistered amplitude.");
            }
            private double[] E27LegalPose(int[] active, int seed, double meters, double intensity)
            {
                var rng = new System.Random(seed); var x = (double[])room.Truth.Clone();
                foreach (int lamp in active)
                {
                    int k = 4 * lamp;
                    bool legal = false;
                    for (int trial = 0; trial < 128 && !legal; ++trial)
                    {
                        double z = 2 * rng.NextDouble() - 1, a = 2 * Math.PI * rng.NextDouble(), r = Math.Sqrt(1 - z * z);
                        var d = new[] { r * Math.Cos(a), z, r * Math.Sin(a) };
                        for (int j = 0; j < 3; ++j) x[k + j] = room.Truth[k + j] + d[j] * meters / units[j];
                        legal = x.Skip(k).Take(3).All(v => v >= 0 && v <= 1) && room.Contains(x);
                    }
                    if (!legal) return null;
                    x[k + 3] = E27BoundedDelta(room.Truth[k + 3], intensity / units[3], rng);
                }
                return x;
            }
            private int[][] e27Couples;
            private int[][] E27CoupledPairs(E27Job job)
            {
                if (e27Couples != null) return e27Couples;
                var color = E27DerivedCase("color-audit-moderate", job);
                var target = E27FreezeTarget(color, out _);
                int count = room.Count, width = 4 * count;
                var cols = new float[count][];
                for (int i = 0; i < count; ++i)
                {
                    var rig = (double[])color.truth.Clone(); rig[width + 2 * i + 1] += .03 / 6;
                    var rgb = new float[E23Channels]; E27Apply(rig, color, LightShadows.Hard);
                    for (int v = 0; v < 3; ++v) low.CaptureRgb(v, rgb, v * E23ChannelsPerView);
                    cols[i] = new float[E23Channels]; target.log.Evaluate(rgb, cols[i]);
                }
                var data = new List<(int a, int b, double dot)>();
                for (int i = 0; i < count; ++i)
                for (int j = i + 1; j < count; ++j)
                    data.Add((i, j, Math.Abs(E21Dot(cols[i], cols[j], 0, E23Channels)) /
                        Math.Sqrt(Math.Max(1e-60, E21Dot(cols[i], cols[i], 0, E23Channels) * E21Dot(cols[j], cols[j], 0, E23Channels)))));
                var sorted = data.OrderBy(x => x.dot).ToArray();
                using var csv = Csv("color-coupling.csv", "lamp_a,lamp_b,abs_blue_green_column_cos");
                foreach (var x in sorted) csv.WriteLine(string.Join(",", x.a + 1, x.b + 1, F(x.dot)));
                return e27Couples = new[] { new[] { sorted[0].a, sorted[0].b }, new[] { sorted.Last().a, sorted.Last().b } };
            }

            private sealed partial class E27Trial
            {
                private bool RunExtra()
                {
                    switch (arm.policy)
                    {
                        case "selected-exploration": return RunSearch(arm.selected);
                        case "color-audit": NativeColor(); return true;
                        case "temperature-audit": TemperatureAudit(); return true;
                        case "mixed-layout": LayoutAudit(); return true;
                        case "rgb": case "hist16": case "hist32": case "mix16": case "mix32": case "selected-histogram": HistogramPolicy(); return true;
                        default: return false;
                    }
                }
                private void NativeColor()
                {
                    using var table = Open("color-native.csv", "arm,shadows,state,view,lamp,effective_i,native_i,api_filter_rgb,visible_final_rgb,intended_linear_per_i,relative_l2,atlas_w,atlas_h,atlas_faces,atlas_same");
                    using var response = Open("color-response.csv", "arm,shadows,lamp,axis,fd,state,plus_direction_l2,minus_direction_l2,plus_minus_cos,repeat_raw_loss");
                    using var spectral = Open("color-jacobian.csv", "arm,shadows,lamp,fd_ratio,rank4,rank5,rank6,sv6,relative_orthogonal_u,relative_orthogonal_v,normalized_dot,steps,stencils,white_point");
                    using var atlas = new E23AtlasObserver(h.lights, observeColors: true);
                    int width = h.room.Count * 4;
                    bool valid = true;
                    var initialActive = sample.active;
                    foreach (var shadow in new[] { LightShadows.Hard, LightShadows.None })
                    {
                        var actual = new E27Case { mode = sample.mode, truth = sample.truth, target = shadow == LightShadows.Hard ? "Hard" : "Off" };
                        var native = h.E27FreezeTarget(actual, out int costs); result.auditCameras += costs;
                        h.E27Apply(sample.truth, sample, shadow);
                        var signatures = new string[3];
                        ObserveNative(sample.truth, "gt", true);
                        foreach (int lamp in new[] { 0, 1, 2 })
                        {
                            foreach (int axis in new[] { 4 * lamp + 3, width + 2 * lamp, width + 2 * lamp + 1 })
                            foreach (double sign in new[] { -1d, 1 })
                            {
                                var r = (double[])sample.truth.Clone(); r[axis] += sign * (axis < width ? .04 / LogPowerSpan : .03 / 6);
                                ObserveNative(r, $"L{lamp + 1}_axis{axis}_{sign}", false);
                            }
                            sample.active = new[] { lamp };
                            foreach (double fd in new[] { .03, .06 })
                            {
                                MakeLayout(.02, fd);
                                AuditRank(sample.truth, native, shadow, fd, lamp, "gt");
                            }
                            foreach (int coord in new[] { 0, 1 })
                            foreach (double boundary in new[] { 0d, 1d })
                            {
                                var r = (double[])sample.truth.Clone(); r[width + 2 * lamp + coord] = boundary;
                                MakeLayout(.02);
                                AuditRank(r, native, shadow, .03, lamp, "boundary_" + coord + "_" + boundary);
                            }
                        }
                        void ObserveNative(double[] pose, string state, bool original)
                        {
                            h.E27Apply(pose, sample, shadow);
                            for (int v = 0; v < 3; ++v)
                            {
                                var a = atlas.Observe(() => h.low.Capture(v)); ++result.auditCameras;
                                string signature = a.Signature.ToString();
                                if (original) signatures[v] = signature;
                                for (int lamp = 0; lamp < h.room.Count; ++lamp)
                                {
                                    var light = h.lights[lamp]; var rgb = a.finalColors[lamp];
                                    var c = E27Chroma(6 * pose[width + 2 * lamp] - 3, 6 * pose[width + 2 * lamp + 1] - 3);
                                    double intensity = h.room.PowerOf(pose, lamp);
                                    var wanted = c * (float)intensity;
                                    var observed = new Vector3(rgb.r, rgb.g, rgb.b);
                                    double error = (observed - wanted).magnitude / Math.Max(1e-20, wanted.magnitude);
                                    if (error > 1e-4) valid = false;
                                    table.WriteLine(string.Join(",", arm.id, shadow, state, v, lamp + 1, F(intensity), F(light.intensity),
                                        Series(new double[] { light.color.r, light.color.g, light.color.b }), Series(new double[] { rgb.r, rgb.g, rgb.b }),
                                        Series(new double[] { c.x, c.y, c.z }), F(error), a.width, a.height, a.faces.Length, Flag(signature == signatures[v])));
                                }
                            }
                        }
                    }
                    sample.active = initialActive;
                    result.stop = "native_color_audit";
                    if (!valid) { result.status = "failed"; result.reason = "VisibleLight.finalColor does not match luminance-normalized RGB within 1e-4"; }

                    void AuditRank(double[] pose, E27Target gt, LightShadows shadow, double fd, int lamp, string point)
                    {
                        int d = layout.Count;
                        var buffer = new float[E23Channels];
                        float[] center = null, latest = null;
                        var pos = new Dictionary<int, float[]>();
                        var obj = gt.log;
                        var settings = new Lm.TrustSettings { Layout = layout, Budget = 2 * d + 2, Iterations = 1, FixedProbeScale = true, BuildModelAtZero = true,
                            ProbeAllowed = x => Legal(Expand(x, pose)), CandidateAllowed = x => Legal(Expand(x, pose)), FinalAllowed = x => Legal(Expand(x, pose)),
                            FinishAfterModel = j =>
                            {
                                var n4 = ClassroomDiagnosis.GetSpectrum(j.Gram, 4);
                                var n5 = ClassroomDiagnosis.GetSpectrum(j.Gram, 5);
                                var g = (double[,])j.Gram.Clone();
                                for (int i = 0; i < d; ++i)
                                for (int k = 0; k < d; ++k) g[i, k] /= layout.Axis(i).Unit * layout.Axis(k).Unit;
                                var n6 = ClassroomDiagnosis.GetSpectrum(g, 6);
                                double Orthogonal(int at)
                                {
                                    var a = new double[4, 4]; var b = new double[4];
                                    var s = Enumerable.Range(0, 4).Select(i => Math.Sqrt(Math.Max(1e-30, g[i, i]))).ToArray();
                                    for (int i = 0; i < 4; ++i)
                                    {
                                        b[i] = g[i, at] / s[i]; for (int k = 0; k < 4; ++k) a[i, k] = g[i, k] / (s[i] * s[k]);
                                    }
                                    var w = E27Solve(a, (double[])b.Clone());
                                    return w == null ? double.NaN : Math.Sqrt(Math.Max(0, 1 - E27Dot(w, b) / Math.Max(1e-30, g[at, at])));
                                }
                                spectral.WriteLine(string.Join(",", arm.id, shadow, lamp + 1, F(fd), n4.rank, n5.rank, n6.rank, T(n6.singular),
                                    F(Orthogonal(4)), F(Orthogonal(5)), F(g[4, 5] / Math.Sqrt(Math.Max(1e-30, g[4, 4] * g[5, 5]))), T(j.PhysicalSteps), string.Join(";", j.Stencils), point));
                                return true;
                            } };
                        settings.Query = (_, phase, x, value, at, step) =>
                        {
                            if (phase == "initial") { center = (float[])latest.Clone(); return; }
                            if (at < 3 || phase == "fd_one_sided") return;
                            if (phase == "fd_plus") pos[at] = (float[])latest.Clone();
                            else if (phase == "fd_minus" && pos.TryGetValue(at, out var positive))
                            {
                                double aa = 0, bb = 0, ab = 0;
                                for (int k = 0; k < latest.Length; ++k)
                                { double a = positive[k] - center[k], b = center[k] - latest[k]; aa += a * a; bb += b * b; ab += a * b; }
                                response.WriteLine(string.Join(",", arm.id, shadow, lamp + 1, layout.Axis(at).Coordinate, F(fd), point,
                                    F(Math.Sqrt(aa)), F(Math.Sqrt(bb)), F(ab / Math.Sqrt(Math.Max(1e-60, aa * bb))), F(gt.repeat)));
                            }
                        };
                        segment = "color_column_audit";
                        Lm.RunTrust(Local(pose), E23Channels, (x, phase, residual) =>
                        {
                            var rig = Expand(x, pose); Capture(rig, shadow, E27Views, buffer);
                            double loss = obj.Evaluate(buffer, residual); latest = residual; Query(phase, shadow, rig, E27Views, loss); return loss;
                        }, settings);
                    }
                }
                private void TemperatureAudit()
                {
                    using var csv = Open("temperature-6500.csv", "arm,shadows,kelvin,requested_mired,side,actual_mired,final_rgb,normalized_pixel_l2,camera");
                    using var atlas = new E23AtlasObserver(h.lights, observeColors: true);
                    int lamp = sample.active[0], axis = 4 * h.room.Count + lamp;
                    foreach (var mode in new[] { LightShadows.Hard, LightShadows.None })
                    foreach (double k in new[] { 5500d, 6000, 6400, 6500, 6600, 7000 })
                    {
                        var rig = (double[])sample.truth.Clone(); rig[axis] = EncodeMired(1e6 / k);
                        h.E27Apply(rig, sample, mode); var reference = h.low.Capture(0); ++result.auditCameras;
                        foreach (double step in new[] { 0d, 1, 5, 10 })
                        foreach (int side in step != 0 ? new[] { -1, 1 } : new[] { 0 })
                        {
                            rig[axis] = EncodeMired(1e6 / k + step * side);
                            h.E27Apply(rig, sample, mode);
                            var info = atlas.Capture(() => h.low.Capture(0)); ++result.auditCameras;
                            var color = info.state.finalColors[lamp];
                            csv.WriteLine(string.Join(",", arm.id, mode, F(k), F(step), side, F(1e6 / h.lights[lamp].colorTemperature),
                                Series(new double[] { color.r, color.g, color.b }), F(Math.Sqrt(NativeRenderBackend.Compare(info.rgb, reference))), 0));
                        }
                    }
                    result.stop = "native_temperature_audit";
                }
                private void LayoutAudit()
                {
                    var axes = h.E26Axes();
                    var t = new Lm.TrustAxis(Lm.TrustCoordinate.Mired, MiredSpan, 5, 20, 50, MinMired);
                    var u = new Lm.TrustAxis(Lm.TrustCoordinate.LogRedGreen, 6, .03, .25, .75, -3);
                    var v = new Lm.TrustAxis(Lm.TrustCoordinate.LogBlueGreen, 6, .03, .25, .75, -3);
                    var l = new Lm.TrustLayout(axes, axes.Append(t).ToArray(), axes.Append(u).Append(v).ToArray(), new[] { u, v });
                    double[] x = Enumerable.Repeat(.5, l.Count).ToArray();
                    var settings = new Lm.TrustSettings { Budget = 360, Layout = l, CandidateAllowed = a => a.All(p => p >= 0 && p <= 1) };
                    var r = Lm.RunTrust(x, l.Count, (a, _, y) =>
                    { double sum = 0; for (int j = 0; j < a.Length; ++j) { double e = a[j] - .4; y[j] = (float)e; sum += e * e; } return sum; }, settings);
                    using var csv = Open("layout-parity.csv", "arm,widths,owners,offsets,final_max_error,pass");
                    double error = r.Best.Max(p => Math.Abs(p - .4));
                    bool success = l.Count == 17 && l.Owner(15) == 3 && l.Start(2) == 9 && l.Width(1) == 5 && error < 1e-4;
                    csv.WriteLine(string.Join(",", arm.id, "4;5;6;2", string.Join(";", Enumerable.Range(0, l.Count).Select(l.Owner)), "0;4;9;15", F(error), Flag(success)));
                    if (!success) result.status = "failed";
                    result.stop = r.Stop;
                }
            }
        }
    }
}

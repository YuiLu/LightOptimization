using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.Rendering;
using Frame = LightingMvp.NativeRenderBackend.Frame;
using Mapping = LightingMvp.NativeRenderBackend.HdrMapping;
using Lm = LightingMvp.ImageJacobianLm;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            private const double MinMired = 1e6 / 9000, MaxMired = 1e6 / 2500, MiredSpan = MaxMired - MinMired;
            private const int E26RigWidth = 28;
            private static readonly double[] E26Kelvin = { 3000, 4500, 6500, 3000, 4500, 6500, 4500 };
            private static readonly int[] E26NewSeeds = { 2003, 2909 };
            private sealed class E26ColorTarget
            {
                internal MbdRgbTarget target;
                internal E23FrameObjective fit;
            }
            [Serializable] private sealed class E26ColorSnapshot
            {
                public string arm, phase, group, method, layout;
                public int seed, cameras, models;
                public int[] lamps;
                public double[] start, final, truth, startMiredDelta, endMiredAbs, endKelvinAbs, endChroma;
                public double[] positionCm, intensityRelative;
                public bool pass;
            }

            partial void E26Color(string key)
            {
                var source20 = E26ArchivedCase(false, "911").initial;
                var source30 = E26ArchivedCase(false, "1201").initial;
                // Validate the E25 frozen rig before changing any native light color.
                var sourceFar = key == "d-joint-complex-509" ? E26ArchivedCase(false, "509").initial : null;
                var truth = room.Truth.Concat(E26Kelvin.Select(k => EncodeMired(1e6 / k))).ToArray();
                var old = lights.Select(l => (l.color, l.useColorTemperature, l.colorTemperature)).ToArray();
                bool oldGlobal = GraphicsSettings.lightsUseColorTemperature;
                if (QualitySettings.activeColorSpace != ColorSpace.Linear || !GraphicsSettings.lightsUseLinearIntensity)
                    throw new InvalidOperationException("E26-D requires Unity linear lighting.");
                GraphicsSettings.lightsUseColorTemperature = true;
                using var trialCsv = Csv("color-trials.csv", "arm,stage,sample,group,seed,method,layout,shadows,lamps,position_perturb_cm,intensity_perturb_relative,actual_mired_delta,max_cm,max_intensity_relative,max_mired,max_kelvin,max_chroma,legal,pass,cap_camera,opt_camera,models,first_pass_camera,first_pass_model,first_pass_s,fit_loss_start,fit_loss_final,held_raw,held_log,held_display,stop");
                using var queryCsv = Csv("query-log.csv", "arm,query,actual_camera,phase,shadows,axis,physical_step,cameras,theta");
                using var scoreCsv = Csv("view-metrics.csv", "arm,checkpoint,target,width,view,raw_nrmse,log_nrmse,display_nrmse");
                using var modelCsv = Csv("model-checkpoints.csv", "arm,model,actual_camera,best_loss,max_cm,max_intensity_relative,max_mired,legal,pass,theta");
                using var profile = Csv("color-profiling.csv", "arm,fit_views,cap_camera,opt_camera,hard_camera,off_camera,other_camera,opt_wall_s,render_s,readback_s,map_fd_s,mask_p95_s,gram_gradient_s,damped_solve_s,scale");
                using var nativeSettings = Csv("color-native.csv", "phase,view,lamp,kelvin,mired,intensity,filter_rgb,native_visible_final_rgb,atlas_width,atlas_height,atlas_faces,layout_matches_gt");
                var snapshots = new List<E26ColorSnapshot>();
                var targets = new Dictionary<bool, E26ColorTarget>();
                try
                {
                    if (!GraphicsSettings.lightsUseColorTemperature) throw new InvalidOperationException("Native temperature flag did not enable.");
                    var on = ColorTarget(true); var off = ColorTarget(false);
                    WriteJson(directory, "color-config.json", new E26ColorConfig { allowTemperature = GraphicsSettings.lightsUseColorTemperature, linearIntensity = GraphicsSettings.lightsUseLinearIntensity,
                        colorSpace = QualitySettings.activeColorSpace.ToString(), gtKelvin = E26Kelvin, minMired = MinMired, maxMired = MaxMired,
                        truth = truth, frozenIndependentSeeds = E26NewSeeds, scene = room.Describe(), widths = new[] { low.Width, high.Width } });
                    if (key == "d-small-complex")
                    {
                        E26ColorAudit(truth, on, off, nativeSettings);
                        for (int lamp = 0; lamp < 3; ++lamp)
                        foreach (double offset in new[] { -20d - (lamp % 2) * 20, 40d - (lamp % 2) * 20 })
                        {
                            string sample = $"L{lamp + 1}_{(offset < 0 ? "minus" : "plus")}{(int)Math.Abs(offset)}";
                            var active = new[] { lamp };
                            var start = (double[])truth.Clone();
                            AddTemperature(start, lamp, offset);
                            Fit("D1", sample, "single", 911, "H1", active, start, on, "mired");
                            start = (double[])start.Clone(); start[4 * lamp + 3] = source20[4 * lamp + 3];
                            Fit("D2", sample, "single", 911, "H1", active, start, on, "logI;mired");
                        }
                    }
                    else if (key == "d-single-complex")
                    {
                        foreach (int lamp in new[] { 0, 2, 5 })
                        foreach (int seed in new[] { 911, 1201 })
                        {
                            string sample = $"L{lamp + 1}_seed{seed}";
                            var active = new[] { lamp };
                            var baseRig = ColorStart(active, seed == 911 ? source20 : source30);
                            var temp = (double[])baseRig.Clone(); AddTemperature(temp, lamp, seed == 911 ? -(lamp == 2 ? 20 : 40) : lamp == 2 ? 40 : 20);
                            Fit("D3", sample, "single", seed, "C3", active, temp, on, "XYZ;logI;mired");
                            Fit("D3", sample, "single", seed, "C3", active, baseRig, on, "XYZ;logI");
                            Fit("D3", sample, "single", seed, "H1", active, temp, on, "XYZ;logI;mired");
                            Fit("D3", sample, "single", seed, "A-log", active, temp, off, "XYZ;logI;mired");
                            Fit("D3", sample, "single", seed, "A-log", active, baseRig, off, "XYZ;logI");
                        }
                    }
                    else if (key == "d-pair-complex")
                    {
                        var pairGroups = E26ColorPairs(truth, on, off);
                        foreach (var (group, lampIds) in pairGroups)
                        foreach (int seed in new[] { 911, 1201 })
                        {
                            var start = ColorStart(lampIds, seed == 911 ? source20 : source30);
                            string sample = $"{group}_seed{seed}";
                            var fixedTemp = (double[])start.Clone();
                            AddTemperature(start, lampIds[0], seed == 911 ? -20 : -40);
                            AddTemperature(start, lampIds[1], seed == 911 ? 40 : 20);
                            Fit("D4", sample, group, seed, "C3", lampIds, start, on, "XYZ;logI;mired");
                            Fit("D4", sample, group, seed, "C3", lampIds, fixedTemp, on, "XYZ;logI");
                        }
                    }
                    else if (key.StartsWith("d-joint-complex-", StringComparison.Ordinal))
                    {
                        int seed = int.Parse(key.Split('-').Last(), Invariant);
                        var active = E24AllLamps;
                        double[] pose = seed switch
                        {
                            911 => source20, 1201 => source30, 509 => sourceFar,
                            2003 or 2909 => IndependentPose(seed), _ => throw new ArgumentException(key)
                        };
                        var fixedTemp = ColorStart(active, pose); var temp = (double[])fixedTemp.Clone();
                        foreach (int lamp in active) AddTemperature(temp, lamp, ((lamp + seed) % 2 == 0 ? 1 : -1) * (lamp % 2 == 0 ? 40 : 20));
                        var color = Fit("D5", "seed" + seed, "joint", seed, "C3", active, temp, on, "XYZ;logI;mired");
                        Fit("D5", "seed" + seed, "joint", seed, "C3", active, fixedTemp, on, "XYZ;logI");
                        if (!color.pass && color.stop == "budget")
                            Fit("D5-budget", "seed" + seed, "joint", seed, "C3", active, temp, on, "XYZ;logI;mired", cap: 5907);
                    }
                    else throw new ArgumentException("Unknown E26 color stage " + key);
                    WriteJson(directory, "color-snapshots.json", new E26ColorSnapshotSet { samples = snapshots.ToArray() });
                }
                finally
                {
                    for (int i = 0; i < lights.Length; ++i)
                    {
                        var (filter, enabled, kelvin) = old[i];
                        lights[i].color = filter; lights[i].colorTemperature = kelvin; lights[i].useColorTemperature = enabled;
                    }
                    GraphicsSettings.lightsUseColorTemperature = oldGlobal;
                }
                return;

                E26ColorTarget ColorTarget(bool shadows)
                {
                    if (targets.TryGetValue(shadows, out var t)) return t;
                    ColorApply(truth, shadows ? LightShadows.Hard : LightShadows.None);
                    var frames = E25DefaultViews.Select(low.Capture).ToArray();
                    for (int i = 0; i < frames.Length; ++i)
                        if (NativeRenderBackend.Compare(low.Capture(i), frames[i]) > 1e-12) throw new InvalidOperationException("Color target is not repeatable.");
                    var highFrames = Enumerable.Range(0, room.Views.Length).Select(high.Capture).ToArray();
                    SaveFrames(highFrames, CapturePath(shadows ? "color_gt_hard" : "color_gt_off"), "gt", false);
                    var result = new MbdRgbTarget(frames, highFrames, shadows);
                    t = new E26ColorTarget { target = result, fit = new E23FrameObjective(frames, Mapping.SignedLog, E22Scale(result)) };
                    targets[shadows] = t;
                    return t;
                }

                double[] ColorStart(int[] active, double[] pose)
                {
                    var start = (double[])truth.Clone();
                    foreach (int lamp in active) Array.Copy(pose, 4 * lamp, start, 4 * lamp, 4);
                    if (!ColorLegal(start)) throw new InvalidDataException("Frozen color start is illegal.");
                    return start;
                }

                double[] IndependentPose(int seed)
                {
                    var prng = new System.Random(seed);
                    var pose = (double[])room.Truth.Clone();
                    for (int lamp = 0; lamp < lights.Length; ++lamp)
                    {
                        var center = room.PositionOf(truth, lamp);
                        double radius = seed == 2003 ? .20 : .30;
                        bool found = false;
                        for (int draw = 0; draw < 4096 && !found; ++draw)
                        {
                            var direction = new Vector3((float)(2 * prng.NextDouble() - 1), (float)(2 * prng.NextDouble() - 1), (float)(2 * prng.NextDouble() - 1));
                            if (direction.sqrMagnitude > 1 || direction.sqrMagnitude < 1e-6) continue;
                            var candidate = room.WithPosition(pose, lamp, center + direction.normalized * (float)radius);
                            if (room.Contains(candidate)) { pose = candidate; found = true; }
                        }
                        if (!found) throw new InvalidDataException("Independent frozen seed did not generate a legal start.");
                        pose[4 * lamp + 3] = Mathf.Clamp01((float)(pose[4 * lamp + 3] + (2 * prng.NextDouble() - 1) * .18 / units[3]));
                    }
                    return pose;
                }

                (bool pass, string stop) Fit(string phase, string sample, string group, int seed, string method, int[] active, double[] start,
                    E26ColorTarget target, string layoutName, int cap = E25Cap)
                {
                    var axes = new List<Lm.TrustAxis>();
                    if (layoutName.Contains("XYZ")) axes.AddRange(E26Axes().Take(3));
                    if (layoutName.Contains("logI")) axes.Add(E26Axes()[3]);
                    if (layoutName.Contains("mired")) axes.Add(new Lm.TrustAxis(Lm.TrustCoordinate.Mired, MiredSpan, 5, 20, 50, MinMired));
                    var layout = new Lm.TrustLayout(active.Select(_ => axes.ToArray()).ToArray());
                    int[] indices = Enumerable.Range(0, layout.Count).Select(i => layout.Axis(i).Coordinate == Lm.TrustCoordinate.Mired ?
                        E26RigWidth + active[layout.Owner(i)] : 4 * active[layout.Owner(i)] + (int)layout.Axis(i).Coordinate).ToArray();
                    var theta = indices.Select(i => start[i]).ToArray(); var rig = (double[])start.Clone();
                    double[] Expand(double[] x) { for (int i = 0; i < indices.Length; ++i) rig[indices[i]] = x[i]; return rig; }
                    bool Legal(double[] x) => ColorLegal(Expand(x));
                    string id = $"{phase}_{sample}_{method}_{layout.Count}D";
                    var timing = new Lm.TrustTiming(); var counters = new E25Counter();
                    var frame = new float[target.fit.Size];
                    int first = -1, firstModel = -1, cams = 0, hardCams = 0;
                    double firstTime = double.NaN, mapSeconds = 0, bestLoss = double.PositiveInfinity;
                    double[] best = start;
                    var mode = target.target.shadows ? LightShadows.Hard : LightShadows.None;
                    int renderStart = low.RenderCount + high.RenderCount;
                    double wallRender = low.RenderSeconds, wallRead = low.ReadbackSeconds;
                    var sw = Stopwatch.StartNew();
                    var settings = new Lm.TrustSettings { Layout = layout, Units = Enumerable.Range(0, layout.Count).Select(i => layout.Axis(i).Unit).ToArray(), Budget = cap / 3,
                        ProbeAllowed = Legal, CandidateAllowed = Legal, FinalAllowed = Legal,
                        FixedProbeScale = true, Timing = timing,
                        Trial = t =>
                        {
                            if (!t.Accepted) return;
                            var x = Expand(t.Theta);
                            if (t.CurrentLoss - t.Actual < bestLoss) { bestLoss = t.CurrentLoss - t.Actual; best = (double[])x.Clone(); }
                            ObservePass(x, t.Query * 3, t.Iteration);
                        },
                        Jacobian = j =>
                        {
                            if (j.Iteration is not (4 or 8 or 16)) return;
                            var e = ColorErrors(best, truth, active);
                            modelCsv.WriteLine(string.Join(",", id, j.Iteration, cams, F(bestLoss), F(e.cm.Max()), F(e.rel.Max()), F(e.mired.Max()), Flag(e.legal), Flag(Pass(e)), Series(best)));
                        },
                        Query = (q, action, x, loss, axis, step) =>
                        {
                            var full = Expand(x);
                            queryCsv.WriteLine(string.Join(",", id, q, cams, action, mode, axis < 0 ? -1 : indices[axis], F(step), 3, Series(full)));
                            if (action == "initial") { bestLoss = loss; ObservePass(full, q * 3, 0); }
                        }
                    };
                    E24Configure(settings, Expand, active, method, target.fit, (q, full, shadows, action, axis, step, rgb) =>
                    {
                        Capture(full, shadows, rgb);
                        queryCsv.WriteLine(string.Join(",", id, q, cams, action, shadows, axis, F(step), 3, Series(full)));
                    }, id, cachedMapping: method == "C3", metrics: counters);
                    var result = Lm.RunTrust(theta, frame.Length, (x, _, values) =>
                    {
                        Capture(Expand(x), mode, frame);
                        long map = Stopwatch.GetTimestamp(); double f = target.fit.Evaluate(frame, values); mapSeconds += Seconds(map); return f;
                    }, settings);
                    sw.Stop();
                    double renderTime = low.RenderSeconds - wallRender, readTime = low.ReadbackSeconds - wallRead;
                    if (cams != result.Evaluations * 3 || cams > cap || result.IllegalAccepted != 0) throw new InvalidOperationException("Color native query accounting diverged.");
                    var final = (double[])Expand(result.Best).Clone();
                    var err = ColorErrors(final, truth, active); var initial = ColorErrors(start, truth, active);
                    if (!err.legal) throw new InvalidOperationException("Color endpoint illegal.");
                    bool pass = Pass(err);
                    Score(start, "initial", false);
                    var held = Score(final, "final", (phase is "D3" or "D4") && pass);
                    trialCsv.WriteLine(string.Join(",", id, phase, sample, group, seed, method, layoutName, mode, LampList(active), Series(initial.cm),
                        Series(active.Select(i => room.PowerOf(start, i) / room.PowerOf(truth, i) - 1)),
                        Series(active.Select(i => (start[E26RigWidth + i] - truth[E26RigWidth + i]) * MiredSpan)),
                        F(err.cm.Max()), F(err.rel.Max()), F(err.mired.Max()), F(err.kelvin.Max()), F(err.chroma.Max()), Flag(err.legal), Flag(pass),
                        cap, cams, result.Jacobians, first, firstModel, F(firstTime), F(result.InitialLoss), F(result.BestLoss), Series(held).Replace(';', ','), result.Stop));
                    profile.WriteLine(string.Join(",", id, "0;1;2", cap, cams, hardCams, cams - hardCams, low.RenderCount + high.RenderCount - renderStart - cams,
                        F(sw.Elapsed.TotalSeconds), F(renderTime), F(readTime), F(counters.Map + mapSeconds), F(counters.Mask), F(timing.GramGradientSeconds), F(timing.DampedSolveSeconds), F(target.fit.scale)));
                    snapshots.Add(new E26ColorSnapshot { arm = id, phase = phase, group = group, seed = seed, method = method, layout = layoutName,
                        lamps = active, cameras = cams, models = result.Jacobians, start = start, final = final, truth = truth,
                        startMiredDelta = active.Select(i => (start[E26RigWidth + i] - truth[E26RigWidth + i]) * MiredSpan).ToArray(),
                        endMiredAbs = err.mired, endKelvinAbs = err.kelvin, endChroma = err.chroma, positionCm = err.cm, intensityRelative = err.rel, pass = pass });
                    UnityEngine.Debug.Log($"[E26] {id}: {err.cm.Max():F2} cm / {err.rel.Max():P2} / {err.mired.Max():F2} mired, {cams} images, {result.Stop}");
                    return (pass, result.Stop);

                    void Capture(double[] full, LightShadows shadows, float[] rgb)
                    {
                        int before = low.RenderCount; ColorCapture(full, shadows, rgb);
                        int added = low.RenderCount - before; cams += added;
                        if (shadows == LightShadows.Hard) hardCams += added;
                    }
                    void ObservePass(double[] full, int q, int model)
                    {
                        if (first >= 0 || !Pass(ColorErrors(full, truth, active))) return;
                        first = q; firstModel = model; firstTime = sw.Elapsed.TotalSeconds;
                    }
                    double[] Score(double[] full, string checkpoint, bool save)
                    {
                        ColorApply(full, mode);
                        for (int v = 0; v < 3; ++v) Write(low.Capture(v), target.target.fit[v], low.Width, v);
                        var held = new List<double[]>();
                        for (int v = 0; v < room.Views.Length; ++v)
                        {
                            var actual = high.Capture(v);
                            var metrics = Write(actual, target.target.high[v], high.Width, v);
                            if (v >= 3) held.Add(metrics);
                            if (save && (v == 0 || v == 3)) NativeRenderBackend.Save(actual, high.Width, high.Height, Path.Combine(CapturePath(id), $"final_v{v}"));
                        }
                        return Enumerable.Range(0, 3).Select(c => Math.Sqrt(held.Average(a => a[c] * a[c]))).ToArray();
                        double[] Write(Frame actual, Frame reference, int width, int v)
                        {
                            var metrics = E25FrameScores(actual, reference, target.fit.scale);
                            scoreCsv.WriteLine(string.Join(",", id, checkpoint, mode == LightShadows.None ? "off" : "hard", width, v, Series(metrics).Replace(';', ',')));
                            return metrics;
                        }
                    }
                }
            }

            [Serializable] private sealed class E26ColorConfig
            {
                public bool allowTemperature, linearIntensity;
                public string colorSpace;
                public double[] gtKelvin, truth;
                public double minMired, maxMired;
                public int[] frozenIndependentSeeds, widths;
                public E17Config scene;
            }

            [Serializable] private sealed class E26ColorSnapshotSet { public E26ColorSnapshot[] samples; }
            private static double EncodeMired(double mired) => (mired - MinMired) / MiredSpan;
            private static double Kelvin(double encoded) => 1e6 / (MinMired + MiredSpan * encoded);
            private static void AddTemperature(double[] rig, int lamp, double offset)
            {
                int i = E26RigWidth + lamp;
                double moved = rig[i] + offset / MiredSpan;
                if (!(moved >= 0 && moved <= 1)) throw new InvalidOperationException($"Unfrozen out-of-bounds mired request for L{lamp + 1}: {offset}");
                rig[i] = moved;
            }
            private bool ColorLegal(double[] rig) => rig.Length == E26RigWidth + room.Count && rig.All(x => x >= 0 && x <= 1) && room.Contains(rig.Take(E26RigWidth).ToArray());
            private void ColorApply(double[] rig, LightShadows mode)
            {
                Apply(rig, mode != LightShadows.None, mode);
                for (int lamp = 0; lamp < lights.Length; ++lamp)
                {
                    lights[lamp].color = Color.white;
                    lights[lamp].useColorTemperature = true;
                    lights[lamp].colorTemperature = (float)Kelvin(rig[E26RigWidth + lamp]);
                }
            }
            private void ColorCapture(double[] rig, LightShadows mode, float[] values)
            {
                ColorApply(rig, mode);
                for (int v = 0; v < 3; ++v) low.CaptureRgb(v, values, v * E23ChannelsPerView);
            }
            private (double[] cm, double[] rel, double[] mired, double[] kelvin, double[] chroma, bool legal) ColorErrors(double[] x, double[] gt, int[] active) =>
                (active.Select(i => 100d * Vector3.Distance(room.PositionOf(x, i), room.PositionOf(gt, i))).ToArray(),
                 active.Select(i => Math.Abs(room.PowerOf(x, i) / room.PowerOf(gt, i) - 1)).ToArray(),
                 active.Select(i => MiredSpan * Math.Abs(x[E26RigWidth + i] - gt[E26RigWidth + i])).ToArray(),
                 active.Select(i => Math.Abs(Kelvin(x[E26RigWidth + i]) - Kelvin(gt[E26RigWidth + i]))).ToArray(),
                 active.Select(i => Vector3.Distance(Chroma(Mathf.CorrelatedColorTemperatureToRGB((float)Kelvin(x[E26RigWidth + i]))),
                      Chroma(Mathf.CorrelatedColorTemperatureToRGB((float)Kelvin(gt[E26RigWidth + i]))))).Select(v => (double)v).ToArray(), ColorLegal(x));
            private static bool Pass((double[] cm, double[] rel, double[] mired, double[] kelvin, double[] chroma, bool legal) e) =>
                e.legal && e.cm.Max() <= 5 && e.rel.Max() <= .05 && e.mired.Max() <= 10;
            private static Vector3 Chroma(Color c) => new Vector3(c.r, c.g, c.b) / Mathf.Max(1e-20f, c.r + c.g + c.b);
        }
    }
}

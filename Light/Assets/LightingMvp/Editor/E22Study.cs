using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Profiling;
using Frame = LightingMvp.NativeRenderBackend.Frame;
using NormalMode = LightingMvp.ImageJacobianLm.TrustNormalMode;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private static void RunE22Stages(string task, string output)
        {
            if (!Application.isBatchMode) throw new InvalidOperationException("E22 uses an isolated batch project.");
            using var pipeline = new TestPipeline(classroom: true);
            using var room = new ComplexRoomSensitivityScene();
            using var study = new E21RoomStudy(room, task, Path.Combine(output, task.Replace('-', '_')));
            study.RunE22(task.EndsWith("-smoke", StringComparison.Ordinal) ? task.Substring(4, task.Length - 10) : task.Substring(4));
            study.Finish();
        }

        [Serializable] private sealed class E22Metrics
        {
            public string arm, solver, mapping, geometry, reusedFrom;
            public double scale, exposure = 1, optimizationSeconds, elapsedSeconds, firstPassSeconds = -1;
            public int newOptimizationCameras, optimizationCameras, validationCameras, olatCameras, firstPassCameras = -1, firstPassQuery = -1;
            public long peakWorkingBytes, peakGraphicsBytes;
            public double rawTrainStart, rawHighStart, rawHeldStart;
            public double displayTrainStart, displayHighStart, displayHeldStart, displayTrain, displayHigh, displayHeld;
            public double[] displayPerView;
        }

        [Serializable] private sealed class E22Supplement
        {
            public E22Metrics[] metrics;
            public int geometryPasses, olatCameras;
            public double geometrySeconds;
            public long processPeakWorkingBytes;
        }

        private sealed partial class E21RoomStudy
        {
            private readonly List<E22Metrics> e22Metrics = new List<E22Metrics>();
            private readonly Dictionary<bool, E22DisplayTarget> displayTargets = new Dictionary<bool, E22DisplayTarget>();
            private readonly HashSet<(bool shadows, string objective)> checkedGt = new HashSet<(bool, string)>();
            private StreamWriter e22Damping, e22Checks, e22ViewMetrics;
            private int olatFrames, geometryPasses;
            private double geometrySeconds;

            internal void RunE22(string key)
            {
                using (e22Damping = Csv("damping.csv", "arm,segment,query,lambda,diagonal_scale"))
                using (e22Checks = Csv("loss-checks.csv", "arm,checkpoint,scalar,residual_loss,scalar_without_buffer,absolute_difference"))
                using (e22ViewMetrics = Csv("rgb-views.csv", "arm,checkpoint,measurement,view,raw_nrmse,display_nrmse"))
                {
                    switch (key)
                    {
                        case "a8": E22SolvePairs(); break;
                        case "a28": E22SolveSeven(); break;
                        case "a-fd": E22AdaptiveStudy(); break;
                        case "b-screen": E22ToneScreen(); break;
                        case "b-off": E22ToneValidate(false, false, new[] { 131, 157, 733 }); break;
                        case "b-on": E22ToneValidate(true, false, new[] { 131, 157, 733 }); break;
                        case "b-confirm": E22ToneValidate(false, false, new[] { 181 }); break;
                        case "b-far": E22ToneValidate(true, true, new[] { 131, 157, 733 }); break;
                        case "b-switch": E22ToneSwitch(); break;
                        case "b-hotspots": E22Hotspots(); break;
                        case "c": E22GeometryValidate(false, false, new[] { 131, 157 }); break;
                        case "c-mean": E22GeometryValidate(true, false, new[] { 131, 157 }); break;
                        case "c-off-733": E22GeometryValidate(false, false, new[] { 733 }); break;
                        case "c-on": E22GeometryValidate(false, true, new[] { 131, 157, 733 }); break;
                        case "d-audit": E22OlatAudit(); break;
                        case "d-8d": E22StructuredOlat(false, false); break;
                        case "d-8d-on": E22StructuredOlat(false, true); break;
                        case "d-28d": E22StructuredOlat(true, false); break;
                        case "d-28d-on": E22StructuredOlat(true, true); break;
                        case "e-calibration": E22ShadowStudy(false); break;
                        case "e-recovery": E22ShadowStudy(true); break;
                        default: throw new ArgumentException("Unknown E22 stage " + key);
                    }
                }
                using var process = Process.GetCurrentProcess();
                WriteJson(directory, "metrics.json", new E22Supplement { metrics = e22Metrics.ToArray(),
                    geometryPasses = geometryPasses, geometrySeconds = geometrySeconds, processPeakWorkingBytes = E22Memory.Working(process, peak: true),
                    olatCameras = olatFrames });
            }

            private void E22SolvePairs()
            {
                var gt = E22Target(false);
                foreach (int[] active in new[] { new[] { 4, 6 }, new[] { 1, 6 } })
                {
                    string key = $"seed733_near_L{active[0] + 1}_L7";
                    foreach (var mode in new[] { NormalMode.Shared, NormalMode.LampDiagonal, NormalMode.LampScaledDense })
                        RunE22Arm(key, E22SolverKey(mode), active, E22Start(733, active), gt, NativeSignal(gt), 520, mode);
                }
            }

            private void E22SolveSeven()
            {
                var gt = E22Target(false);
                int[] active = room.Lamps;
                string challenger = Arg("-bbE22Solver", "A0");
                RunE22Arm("seed733_near_28d", "A0", active, E22Start(733, active), gt, NativeSignal(gt), 1575, NormalMode.Shared);
                if (challenger != "A0")
                    RunE22Arm("seed733_near_28d", challenger, active, E22Start(733, active), gt, NativeSignal(gt), 1575, E22Solver(challenger));
            }

            private static string E22SolverKey(NormalMode mode) => mode == NormalMode.Shared ? "A0" : mode == NormalMode.LampDiagonal ? "A1" : "A2";

            private static NormalMode E22Solver(string key) => key switch
            {
                "A0" => NormalMode.Shared,
                "A1" => NormalMode.LampDiagonal,
                "A2" => NormalMode.LampScaledDense,
                _ => throw new ArgumentException(key)
            };

            private double[] E22Start(int seed, int[] active, bool far = false)
            {
                var seeded = room.Start(seed);
                return Start(far ? seeded : E19Start(room, seeded, .45f), active);
            }

            private MbdRgbTarget E22Target(bool shadows)
            {
                bool fresh = !targets.ContainsKey(shadows);
                var gt = Target(shadows);
                if (!fresh) return gt;
                displayTargets.Add(shadows, new E22DisplayTarget(gt));
                Apply(room.Truth, shadows);
                using var report = Csv(shadows ? "repeat-on.csv" : "repeat-off.csv", "resolution,view,raw_nrmse,exposure,shadows");
                for (int v = 0; v < room.Views.Length; ++v)
                {
                    if (v < E19FitViews) Repeat(low, gt.fit[v], v);
                    Repeat(high, gt.high[v], v);
                }
                return gt;

                void Repeat(NativeRenderBackend backend, Frame reference, int view)
                {
                    double relative = Math.Sqrt(NativeRenderBackend.Compare(backend.Capture(view), reference));
                    report.WriteLine(string.Join(",", backend.Width, view, F(relative), 1, Flag(shadows)));
                    if (relative > 1e-6) throw new InvalidOperationException("E22 GT native replay drift.");
                }
            }

            private E21Trial RunE22Arm(string scenario, string method, int[] lamps, double[] initial, MbdRgbTarget gt,
                E21Signal signal, int cap, NormalMode mode, string mapping = "B0", double scale = 1, string geometry = "none", double step = .02,
                E21Trial reused = null, string reusedFrom = null, E21Signal tail = null,
                Action<ImageJacobianLm.TrustSettings, Func<double[], double[]>> extra = null, StreamWriter adaptive = null, bool nativeGate = false)
            {
                var timer = Stopwatch.StartNew();
                var id = $"{scenario}_{method}";
                int startCameras = low.RenderCount + high.RenderCount, startOlat = olatFrames;
                var m = new E22Metrics { arm = id, solver = E22SolverKey(mode), mapping = mapping, scale = scale,
                    geometry = geometry, reusedFrom = reusedFrom ?? "" };
                using var process = Process.GetCurrentProcess();
                SampleMemory();
                if (checkedGt.Add((gt.shadows, signal.name))) CheckSignal(id, "gt", signal, room.Truth);
                CheckSignal(id, "start", signal, initial);
                var displayGt = displayTargets[gt.shadows];
                Apply(initial, gt.shadows);
                var initLow = Enumerable.Range(0, E19FitViews).Select(low.Capture).ToArray();
                var initHigh = Enumerable.Range(0, room.Views.Length).Select(high.Capture).ToArray();
                (m.rawTrainStart, m.displayTrainStart) = Measure(initLow, gt.fit, displayGt.low, "initial", "448", 0, E19FitViews);
                (m.rawHighStart, m.displayHighStart) = Measure(initHigh, gt.high, displayGt.high, "initial", "896", 0, E19FitViews);
                (m.rawHeldStart, m.displayHeldStart) = Measure(initHigh, gt.high, displayGt.high, "initial", "896", E19FitViews, room.Views.Length);
                var arm = RunArm(scenario, method, lamps, initial, gt, signal, cap, step, nativeGate: nativeGate, partition: Optimize, validation: Validate);
                m.elapsedSeconds = timer.Elapsed.TotalSeconds;
                m.olatCameras = olatFrames - startOlat;
                m.validationCameras = low.RenderCount + high.RenderCount - startCameras - m.newOptimizationCameras - m.olatCameras;
                SampleMemory();
                e22Metrics.Add(m);
                return arm;

                double[] Optimize(E21Trial trial)
                {
                    if (reused != null)
                    {
                        if (!E22SamePose(initial, reused.initial) || trial.limit != reused.limit || trial.shadows != reused.shadows || !trial.lamps.SequenceEqual(reused.lamps))
                            throw new InvalidOperationException("E21 reuse does not match the E22 arm.");
                        trial.evaluations = reused.evaluations; trial.objectiveQueries = reused.objectiveQueries;
                        trial.jacobians = reused.jacobians; trial.accepted = reused.accepted; trial.rejected = reused.rejected;
                        trial.blocked = reused.blocked; trial.zeroColumns = reused.zeroColumns; trial.stop = reused.stop;
                        m.optimizationCameras = reused.objectiveQueries * E19FitViews;
                        m.optimizationSeconds = -1;
                        return (double[])reused.final.Clone();
                    }
                    int before = low.RenderCount + high.RenderCount, cal = olatFrames;
                    var fitTime = Stopwatch.StartNew();
                    int lowRho = 0;
                    int firstBudget = tail == null ? trial.limit : (int)Math.Floor(.60 * trial.limit);
                    var best = Solve(trial, tail == null ? "joint" : "compressed", lamps, initial, signal, firstBudget, nativeGate ? gt : null,
                        configure: (settings, expand) => Configure(settings, expand, tail == null ? "joint" : "compressed", 0));
                    if (tail != null)
                    {
                        int offset = trial.evaluations;
                        best = Solve(trial, "raw_tail", lamps, best, tail, trial.limit - firstBudget, null,
                            configure: (settings, expand) => Configure(settings, expand, "raw_tail", offset));
                    }
                    else if (adaptive != null && (trial.stop == "no_decreasing_step" || lowRho >= 3) && trial.limit - trial.evaluations >= 36)
                    {
                        double choice = E22ChooseFdScale(trial, best, lamps.Single(), signal, adaptive);
                        int offset = trial.evaluations;
                        best = Solve(trial, "resampled", lamps, best, signal, trial.limit - offset, null,
                            configure: (settings, expand) =>
                            {
                                Configure(settings, expand, "resampled", offset);
                                for (int i = 0; i < 3; ++i) settings.PhysicalSteps[i] = choice;
                            });
                    }
                    m.optimizationSeconds = fitTime.Elapsed.TotalSeconds;
                    m.newOptimizationCameras = low.RenderCount + high.RenderCount - before - (olatFrames - cal);
                    m.optimizationCameras = m.newOptimizationCameras;
                    return best;

                    void Configure(ImageJacobianLm.TrustSettings settings, Func<double[], double[]> expand, string segment, int offset)
                    {
                        settings.NormalMode = mode;
                        settings.Query += (q, phase, x, loss, axis, h) => { if ((q & 7) == 0) SampleMemory(); };
                        settings.Trial += t =>
                        {
                            e22Damping.WriteLine(string.Join(",", id, segment, offset + t.Query, Series(t.Damping), t.DampingScale == null ? "1" : Series(t.DampingScale)));
                            if (t.Accepted) lowRho = t.Ratio < .23 ? lowRho + 1 : 0;
                            if (!t.Accepted || m.firstPassQuery >= 0 || !Passes(expand(t.Theta), lamps)) return;
                            m.firstPassQuery = offset + t.Query;
                            m.firstPassCameras = low.RenderCount + high.RenderCount - before;
                            m.firstPassSeconds = fitTime.Elapsed.TotalSeconds;
                        };
                        extra?.Invoke(settings, expand);
                    }
                }

                void Validate(E21Trial trial, Frame[] finalHigh, MbdRgbTarget target, string path)
                {
                    CheckSignal(id, "final", signal, trial.final);
                    if (tail != null) CheckSignal(id, "final_raw_tail", tail, trial.final);
                    Apply(trial.final, gt.shadows);
                    var finalLow = Enumerable.Range(0, E19FitViews).Select(low.Capture).ToArray();
                    (_, m.displayTrain) = Measure(finalLow, gt.fit, displayGt.low, "final", "448", 0, E19FitViews);
                    (_, m.displayHigh) = Measure(finalHigh, gt.high, displayGt.high, "final", "896", 0, E19FitViews);
                    (_, m.displayHeld) = Measure(finalHigh, gt.high, displayGt.high, "final", "896", E19FitViews, room.Views.Length);
                    m.displayPerView = finalHigh.Select((f, v) => Math.Sqrt(DisplayLoss(f, displayGt.high[v]))).ToArray();
                    if (reused != null && Math.Abs(trial.nativeFinal - reused.nativeFinal) > 1e-8)
                        throw new InvalidOperationException("E21 reused endpoint disagrees with the independent E22 replay.");
                }

                (double raw, double display) Measure(Frame[] frames, Frame[] target, E22DisplayFrame[] display, string checkpoint, string resolution, int from, int to)
                {
                    double rawSum = 0, displaySum = 0;
                    for (int v = from; v < to; ++v)
                    {
                        double lr = NativeRenderBackend.Compare(frames[v], target[v]), ld = DisplayLoss(frames[v], display[v]);
                        rawSum += lr; displaySum += ld;
                        e22ViewMetrics.WriteLine(string.Join(",", id, checkpoint, resolution, v, F(Math.Sqrt(lr)), F(Math.Sqrt(ld))));
                    }
                    return (Math.Sqrt(rawSum / (to - from)), Math.Sqrt(displaySum / (to - from)));
                }

                void SampleMemory()
                {
                    m.peakWorkingBytes = Math.Max(m.peakWorkingBytes, E22Memory.Working(process));
                    m.peakGraphicsBytes = Math.Max(m.peakGraphicsBytes, Profiler.GetAllocatedMemoryForGraphicsDriver());
                }
            }

            private bool Passes(double[] rig, int[] lamps) => room.Contains(rig) && lamps.All(i =>
                Vector3.Distance(room.PositionOf(rig, i), room.PositionOf(room.Truth, i)) <= .05f &&
                Math.Abs(room.PowerOf(rig, i) / room.PowerOf(room.Truth, i) - 1) <= .05);

            private static bool E22SamePose(double[] a, double[] b) => a?.Length == b?.Length && a != null &&
                a.Zip(b, (x, y) => Math.Abs(x - y)).All(d => d <= 1e-12);

            private void CheckSignal(string id, string checkpoint, E21Signal signal, double[] rig)
            {
                var residual = new float[signal.size];
                double loss = signal.objective(rig, residual), scalar = signal.objective(rig, null), l2 = E21Dot(residual, residual);
                double difference = Math.Max(Math.Abs(loss - l2), Math.Abs(loss - scalar));
                if (difference > 2e-6 * Math.Max(1e-7, loss)) throw new InvalidOperationException("E22 residual norm differs from loss for " + id);
                e22Checks.WriteLine(string.Join(",", id, checkpoint, F(loss), F(l2), F(scalar), F(difference)));
            }

            private sealed class E22DisplayTarget
            {
                internal readonly E22DisplayFrame[] low, high;
                internal E22DisplayTarget(MbdRgbTarget gt)
                { low = gt.fit.Select(f => new E22DisplayFrame(f)).ToArray(); high = gt.high.Select(f => new E22DisplayFrame(f)).ToArray(); }
            }

            private sealed class E22DisplayFrame
            {
                internal readonly Vector3[] pixels;
                internal readonly double energy;
                internal E22DisplayFrame(Frame frame)
                {
                    pixels = Enumerable.Range(0, frame.PixelCount).Select(i => DisplayRgb(frame.Pixel(i))).ToArray();
                    foreach (var c in pixels) energy += (double)c.x * c.x + (double)c.y * c.y + (double)c.z * c.z;
                }
            }

            private static Vector3 DisplayRgb(Color rgb) => new Vector3(NativeRenderBackend.Display(rgb.r), NativeRenderBackend.Display(rgb.g), NativeRenderBackend.Display(rgb.b));

            private static double DisplayLoss(Frame frame, E22DisplayFrame reference)
            {
                double sum = 0;
                for (int p = 0; p < frame.PixelCount; ++p)
                {
                    var d = DisplayRgb(frame.Pixel(p)) - reference.pixels[p];
                    sum += (double)d.x * d.x + (double)d.y * d.y + (double)d.z * d.z;
                }
                return sum / Math.Max(1e-12, reference.energy);
            }
        }

        private static class E22Memory
        {
#if UNITY_EDITOR_WIN
            [StructLayout(LayoutKind.Sequential)]
            private struct Counters
            {
                internal uint size, faults;
                internal UIntPtr peakWorking, working, peakPagedPool, pagedPool, peakNonPagedPool, nonPagedPool, pageFile, peakPageFile;
            }

            [DllImport("kernel32.dll")] private static extern IntPtr GetCurrentProcess();
            [DllImport("psapi.dll")] [return: MarshalAs(UnmanagedType.Bool)]
            private static extern bool GetProcessMemoryInfo(IntPtr process, ref Counters counters, uint size);
#endif
            internal static long Working(Process process, bool peak = false)
            {
#if UNITY_EDITOR_WIN
                uint size = (uint)Marshal.SizeOf<Counters>();
                var counters = new Counters { size = size };
                if (GetProcessMemoryInfo(GetCurrentProcess(), ref counters, size))
                    return (long)(peak ? counters.peakWorking : counters.working).ToUInt64();
#endif
                process.Refresh();
                return peak ? process.PeakWorkingSet64 : process.WorkingSet64;
            }
        }
    }
}

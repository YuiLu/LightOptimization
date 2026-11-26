using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using UnityEngine;
using Debug = UnityEngine.Debug;
using Frame = LightingMvp.NativeRenderBackend.Frame;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        [Serializable] private sealed class MbdTrial
        {
            public string scene, key, method, objective, stop;
            public int[] lamps;
            public double[] initial, final, positionCm, intensityRelative, perViewNrmse;
            public bool passed, finalColliderLegal;
            public int objectiveQueries, nativeQueries, accepted, rejected, blocked, zeroColumns, shadowFaces, nativeRgbViews, surfaceViews,
                invalidBounds, nonfiniteBounds, infiniteRoots;
            public long estimatedMapBytes;
            public double nativeStart, nativeFinal, nativeHighTrain, nativeHeld, proxyStart, proxyFinal, proxyNrmseStart, proxyNrmseFinal,
                passSubmit, filterSubmit, forwardSync, backwardSync, seconds;
        }

        [Serializable] private sealed class MbdStudyResult
        {
            public string branch, commit, experiment, classroomSha256, e19SceneSha256;
            public E17Config scene;
            public double[] truth;
            public string[] trainingAndHeldViews;
            public int nativeRgbViews, surfaceViews, shadowFaces, surfaceSamples, renderers, alphaClipSlots;
            public long unityAllocatedBytes, managedBytes;
            public double seconds;
            public MbdTrial[] arms;
        }

        private sealed class MbdRgbTarget
        {
            internal readonly Frame[] fit, high;
            internal readonly bool shadows;
            internal MbdRgbTarget(Frame[] fit, Frame[] high, bool shadows) { this.fit = fit; this.high = high; this.shadows = shadows; }
        }

        private sealed class MbdSignal
        {
            internal readonly string name;
            internal readonly int size;
            internal readonly double targetEnergy;
            internal readonly Func<double[], float[], double> residual;
            internal MbdSignal(string name, int size, double targetEnergy, Func<double[], float[], double> residual)
            { this.name = name; this.size = size; this.targetEnergy = targetEnergy; this.residual = residual; }
        }

        private sealed partial class MbdStudy : IDisposable
        {
            private const int TrainCount = 3, RgbChannels = 448 * 256 * 3;
            private const int Seed = 131;
            private readonly ISensitivityScene scene;
            private readonly string dir;
            private readonly bool smoke;
            private readonly NativeRenderBackend fit, high;
            private readonly MbdSurfaceGeometry geometry;
            private readonly Light[] lights;
            private readonly double[] physicalSpan;
            private readonly List<MbdTrial> trials = new List<MbdTrial>();
            private readonly Dictionary<int, (PointMomentShadow map, Vector3 last, float[] values)> maps =
                new Dictionary<int, (PointMomentShadow, Vector3, float[])>();
            private readonly Dictionary<bool, MbdRgbTarget> targets = new Dictionary<bool, MbdRgbTarget>();
            private readonly StreamWriter queries, jacobians, probeCsv;
            private readonly Stopwatch clock = Stopwatch.StartNew();
            private Vector3[] mapSites;
            private int surfaceReads, sampleCount;

            internal MbdStudy(ISensitivityScene scene, string dir, bool smoke)
            {
                this.scene = scene; this.dir = dir; this.smoke = smoke;
                Directory.CreateDirectory(dir);
                fit = new NativeRenderBackend(scene.Camera, scene.Views, scene.Rotations, 448, 256, -1);
                high = new NativeRenderBackend(scene.Camera, scene.Views, scene.Rotations, 896, 512, -1);
                lights = Enumerable.Range(0, scene.Count).Select(l => GameObject.Find(scene.UnityName(l)).GetComponent<Light>()).ToArray();
                physicalSpan = new[] { (double)scene.Extents.x, scene.Extents.y, scene.Extents.z, LogPowerSpan };
                geometry = new MbdSurfaceGeometry(scene.Camera);
                queries = Csv("queries.csv", "case,method,kind,query,phase,theta,proxy_loss,native_rgb_loss,fd_axis_step,predicted_reduction,actual_reduction,accepted,since_stage_s");
                jacobians = Csv("jacobians.csv", "case,method,batch,query,rank,condition,singular_values,column_norms,stencils,signed_steps");
                probeCsv = Csv("gradient-probes.csv", "case,objective,scale,axis,physical_step,stencil,proxy_derivative,native_derivative,proxy_column_norm,native_column_norm,proxy_residual_alignment,native_residual_alignment,gradient_cosine,proxy_cosine_prev_scale,native_cosine_prev_scale");
            }

            private StreamWriter Csv(string file, string header)
            {
                var stream = new StreamWriter(Path.Combine(dir, file));
                stream.WriteLine(header);
                return stream;
            }

            private static string Values(IEnumerable<double> values) => string.Join(";", values.Select(F));
            private void Log(string key, string method, string kind, int q, string phase, double[] x = null,
                double proxy = double.NaN, double native = double.NaN, string step = "", double prediction = double.NaN,
                double actual = double.NaN, string accepted = "") =>
                queries.WriteLine(string.Join(",", key, method, kind, q, phase, x == null ? "" : Values(x),
                    F(proxy), F(native), step, F(prediction), F(actual), accepted, F(clock.Elapsed.TotalSeconds)));

            private void Apply(double[] theta, bool shadows, bool through = false)
            {
                if (!through) scene.Apply(theta);
                for (int i = 0; i < lights.Length; ++i)
                {
                    if (through)
                    {
                        lights[i].transform.position = scene.PositionOf(theta, i);
                        lights[i].intensity = (float)scene.PowerOf(theta, i);
                    }
                    lights[i].enabled = true;
                    scene.SetShadows(i, shadows);
                }
            }

            private MbdRgbTarget Target(bool shadows)
            {
                if (targets.TryGetValue(shadows, out var target)) return target;
                Apply(scene.Truth, shadows);
                for (int v = 0; v < TrainCount; ++v) fit.Capture(v);
                var frames = Enumerable.Range(0, TrainCount).Select(fit.Capture).ToArray();
                for (int v = 0; v < TrainCount; ++v)
                    if (NativeRenderBackend.Compare(fit.Capture(v), frames[v]) > 1e-12)
                        throw new InvalidOperationException("Native RGB GT is not reproducible.");
                var validation = Enumerable.Range(0, scene.Views.Length).Select(high.Capture).ToArray();
                target = new MbdRgbTarget(frames, validation, shadows);
                targets.Add(shadows, target);
                SaveNative(validation, CapturePath("gt_" + (shadows ? "on" : "off")), "native");
                return target;
            }

            private string CapturePath(string name)
            {
                string path = Path.Combine(dir, "captures", name);
                Directory.CreateDirectory(path);
                return path;
            }

            private static void SaveNative(Frame[] validation, string path, string label)
            {
                NativeRenderBackend.Save(validation[0], 896, 512, Path.Combine(path, label + "_v0"));
                NativeRenderBackend.Save(validation[3], 896, 512, Path.Combine(path, label + "_v3"));
            }

            private double NativeLoss(double[] x, MbdRgbTarget target, float[] residual, bool through = false)
            {
                Apply(x, target.shadows, through);
                double loss = 0;
                for (int v = 0; v < TrainCount; ++v)
                    loss += residual == null ? fit.Loss(v, target.fit[v]) / TrainCount : fit.Residual(v, target.fit[v], residual, v * RgbChannels, TrainCount);
                return loss;
            }

            private float[] Visibility(int lamp, double[] theta)
            {
                Vector3 position = scene.PositionOf(theta, lamp);
                if (!maps.TryGetValue(lamp, out var current))
                    current.map = new PointMomentShadow(geometry.DrawCasters, mapSites, MbdSettings(lights[lamp]));
                else if (position.Equals(current.last)) return current.values;
                var values = current.map.Render(position);
                maps[lamp] = (current.map, position, values);
                return values;
            }

            private double[] Combine(double[] baseTheta, IEnumerable<(int lamp, double[] from)> parts)
            {
                var result = (double[])baseTheta.Clone();
                foreach (var part in parts) Array.Copy(part.from, 4 * part.lamp, result, 4 * part.lamp, 4);
                return result;
            }

            private IEnumerable<(string key, int lamp, double[] start)> ClassroomStarts()
            {
                foreach (int lamp in smoke ? new[] { 4 } : new[] { 4, 5 })
                foreach (float distance in smoke ? new[] { 30f } : new[] { 10f, 30f })
                {
                    var start = E18LampStart(scene, lamp, distance, 0, out _);
                    yield return ($"L{lamp + 1}_{distance.ToString("F0", Invariant)}cm_d0", lamp, start);
                }
            }

            private MbdTrial Fit(string key, string method, int[] lamps, double[] start, MbdRgbTarget rgb,
                MbdSignal signal = null, bool nativeAcceptance = false, bool through = false, int budget = 350)
            {
                int cap = smoke ? Math.Min(80, budget) : budget;
                string path = CapturePath(key + "_" + method);
                var row = new MbdTrial { scene = scene.Name, key = key, method = method, lamps = lamps.Select(x => x + 1).ToArray(),
                    initial = (double[])start.Clone(), objective = signal?.name ?? "native_rgb", proxyStart = -1, proxyFinal = -1,
                    proxyNrmseStart = -1, proxyNrmseFinal = -1 };
                var watch = Stopwatch.StartNew();
                var momentBefore = MomentCost();
                int viewsBefore = fit.RenderCount + high.RenderCount, surfaceBefore = surfaceReads;
                row.nativeStart = NativeLoss(start, rgb, null, through);
                Log(key, method, "validation", 0, "initial_rgb", start, native: row.nativeStart);
                if (signal != null) row.proxyStart = signal.residual(start, null);
                int[] active = lamps.SelectMany(l => Enumerable.Range(4 * l, 4)).ToArray();
                var work = (double[])start.Clone();
                double[] current = active.Select(i => start[i]).ToArray();
                int n = active.Length;
                int residuals = signal?.size ?? TrainCount * RgbChannels;
                int batchOffset = 0;
                string signedStep = "";
                row.stop = "budget";
                if (!nativeAcceptance)
                {
                    var result = Solve(current, cap, Legal);
                    row.accepted = result.Accepted; row.rejected = result.Rejected;
                    row.stop = result.StopReason;
                    current = result.Best;
                }
                else
                {
                    double native = row.nativeStart, proxy = row.proxyStart;
                    double radius = .24, intensityRadius = .27;
                    row.stop = "proposal_budget";
                    while (cap - row.objectiveQueries >= 2 * n + 1)
                    {
                        double[] anchor = current;
                        var result = Solve(anchor, Math.Min(Math.Max(56, 3 * n + 8), cap - row.objectiveQueries), InTrustRegion);
                        var displacement = result.Best.Select((x, i) => x - anchor[i]).ToArray();
                        if (displacement.All(d => Math.Abs(d) < 1e-11))
                        {
                            row.stop = "no_proxy_proposal";
                            break;
                        }
                        bool accepted = false;
                        foreach (double fraction in new[] { 1d, .5, .25, .125 })
                        {
                            var candidate = anchor.Select((x, i) => x + fraction * displacement[i]).ToArray();
                            if (!Legal(candidate) || row.objectiveQueries >= cap) continue;
                            double[] theta = Expand(candidate);
                            double proposed = signal.residual(theta, null);
                            row.objectiveQueries++;
                            double measured = NativeLoss(theta, rgb, null);
                            row.nativeQueries++;
                            accepted = measured < native - Math.Max(1e-12, native * 1e-8);
                            Log(key, method, "native_acceptance", row.nativeQueries, "proposal", theta, proposed, measured,
                                F(fraction), proxy - proposed, native - measured, accepted ? "1" : "0");
                            if (accepted)
                            {
                                row.accepted++; current = candidate; native = measured; proxy = proposed;
                                if (fraction == 1) { radius = Math.Min(.42, radius * 1.25); intensityRadius = Math.Min(.5, intensityRadius * 1.25); }
                                break;
                            }
                            row.rejected++;
                        }
                        if (!accepted) { radius *= .5; intensityRadius *= .5; }
                        if (native <= 1e-12) { row.stop = "near_zero_native_residual"; break; }
                        if (radius < .006) { row.stop = "no_accepted_native_step"; break; }

                        bool InTrustRegion(double[] x)
                        {
                            if (!Legal(x)) return false;
                            for (int i = 0; i < n; i += 4)
                            {
                                double mx = (x[i] - anchor[i]) * physicalSpan[0], my = (x[i + 1] - anchor[i + 1]) * physicalSpan[1];
                                double mz = (x[i + 2] - anchor[i + 2]) * physicalSpan[2];
                                if (mx * mx + my * my + mz * mz > radius * radius ||
                                    Math.Abs(x[i + 3] - anchor[i + 3]) * physicalSpan[3] > intensityRadius) return false;
                            }
                            return true;
                        }
                    }
                }
                row.final = (double[])Expand(current).Clone();
                row.finalColliderLegal = scene.Contains(row.final);
                row.nativeFinal = NativeLoss(row.final, rgb, null, through);
                row.positionCm = lamps.Select(l => 100d * Vector3.Distance(scene.PositionOf(row.final, l), scene.PositionOf(scene.Truth, l))).ToArray();
                row.intensityRelative = lamps.Select(l => Math.Abs(scene.PowerOf(row.final, l) / scene.PowerOf(scene.Truth, l) - 1)).ToArray();
                row.passed = row.finalColliderLegal && row.positionCm.All(x => x <= 5) && row.intensityRelative.All(x => x <= .05);
                if (signal != null)
                {
                    row.proxyFinal = signal.residual(row.final, null);
                    row.proxyNrmseStart = Math.Sqrt(row.proxyStart / Math.Max(1e-12, signal.targetEnergy));
                    row.proxyNrmseFinal = Math.Sqrt(row.proxyFinal / Math.Max(1e-12, signal.targetEnergy));
                }
                Apply(row.final, rgb.shadows, through);
                var finals = Enumerable.Range(0, scene.Views.Length).Select(high.Capture).ToArray();
                var errors = finals.Select((f, v) => NativeRenderBackend.Compare(f, rgb.high[v])).ToArray();
                row.nativeHighTrain = Math.Sqrt(errors.Take(TrainCount).Average());
                row.nativeHeld = Math.Sqrt(errors.Skip(TrainCount).Average());
                row.perViewNrmse = errors.Select(Math.Sqrt).ToArray();
                SaveNative(finals, path, "best_native");
                Apply(start, rgb.shadows, through);
                NativeRenderBackend.Save(high.Capture(0), 896, 512, Path.Combine(path, "start_native_v0"));
                row.nativeRgbViews = fit.RenderCount + high.RenderCount - viewsBefore;
                row.surfaceViews = surfaceReads - surfaceBefore;
                var costs = MomentCost();
                row.shadowFaces = (int)(costs[0] - momentBefore[0]);
                row.invalidBounds = (int)(costs[1] - momentBefore[1]);
                row.nonfiniteBounds = (int)(costs[2] - momentBefore[2]);
                row.infiniteRoots = (int)(costs[3] - momentBefore[3]);
                row.passSubmit = costs[4] - momentBefore[4]; row.filterSubmit = costs[5] - momentBefore[5];
                row.forwardSync = costs[6] - momentBefore[6]; row.backwardSync = costs[7] - momentBefore[7];
                row.estimatedMapBytes = (long)costs[8];
                row.seconds = watch.Elapsed.TotalSeconds;
                Log(key, method, "validation", row.nativeQueries, "final_recheck", row.final, row.proxyFinal, row.nativeFinal);
                queries.Flush(); jacobians.Flush(); probeCsv.Flush();
                trials.Add(row);
                WriteJson(path, "trial.json", row);
                Debug.Log($"[MBD] {scene.Name} {key} {method}: RGB {Math.Sqrt(row.nativeStart):P2} -> {Math.Sqrt(row.nativeFinal):P2}, " +
                    $"{row.positionCm.Max():F2} cm / {row.intensityRelative.Max():P2}, {row.objectiveQueries} objective queries, {row.shadowFaces} faces, {row.stop}");
                return row;

                double[] Expand(double[] x)
                {
                    for (int i = 0; i < n; ++i) work[active[i]] = x[i];
                    return work;
                }
                bool Legal(double[] x) => through ? x.All(v => v >= 0 && v <= 1) : scene.Contains(Expand(x));

                ImageJacobianLm.Result Solve(double[] anchor, int budgetLeft, Func<double[], bool> allowed)
                {
                    var diag = new ImageJacobianLm.Diagnostics {
                        ProbeSteps = signal == null ? null : active.Select(i => i % 4 == 3 ? .04 / physicalSpan[3] : .015 / physicalSpan[i % 4]).ToArray(),
                        Probe = (q, j, h) => signedStep = $"{active[j]}={F(h * physicalSpan[active[j] % 4])}",
                        Jacobian = (batch, q, loss, used, stencil, gram) =>
                        {
                            var spectrum = ClassroomDiagnosis.GetSpectrum(gram, n);
                            row.zeroColumns += stencil.Count(x => x == 0);
                            jacobians.WriteLine(string.Join(",", key, method, batchOffset + batch, row.objectiveQueries,
                                spectrum.rank, F(spectrum.condition), Values(spectrum.singular), Values(spectrum.columnNorms),
                                string.Join(";", stencil), Values(used.Select((h, i) => h * physicalSpan[active[i] % 4]))));
                        },
                        Trial = (batch, q, damping, predicted, actual, ratio, bounds, shrinks, status) =>
                            Log(key, method, "lm_decision", row.objectiveQueries, status, prediction: predicted, actual: actual,
                                accepted: status == "accept" ? "1" : "0")
                    };
                    var result = ImageJacobianLm.Run(anchor, residuals, (x, phase, residual) =>
                    {
                        var theta = Expand(x);
                        ++row.objectiveQueries;
                        double value;
                        if (signal == null)
                        {
                            row.nativeQueries++;
                            value = NativeLoss(theta, rgb, residual, through);
                            Log(key, method, "objective", row.objectiveQueries, phase, theta, native: value, step: signedStep);
                        }
                        else
                        {
                            value = signal.residual(theta, residual);
                            Log(key, method, "objective", row.objectiveQueries, phase, theta, proxy: value, step: signedStep);
                        }
                        signedStep = "";
                        return value;
                    }, budgetLeft, Seed, allowed, diagnostics: diag);
                    row.blocked += result.BlockedProbes;
                    batchOffset += result.Jacobians;
                    return result;
                }
            }

            private double[] MomentCost() => new double[] {
                maps.Values.Sum(x => x.map.Faces), maps.Values.Sum(x => x.map.Degenerate), maps.Values.Sum(x => x.map.Nonfinite),
                maps.Values.Sum(x => x.map.LimitSamples), maps.Values.Sum(x => x.map.PassSubmitSeconds), maps.Values.Sum(x => x.map.FilterSubmitSeconds),
                maps.Values.Sum(x => x.map.ForwardSyncSeconds), maps.Values.Sum(x => x.map.BackwardSyncSeconds), maps.Values.Sum(x => x.map.EstimatedGpuBytes)
            };

            internal void Finish(string stage)
            {
                using (var csv = Csv("trials.csv", "scene,case,method,objective,lamps,passed,collider_legal,max_position_cm,max_intensity_relative,native_start_nrmse,native_train_nrmse,native_high_train_nrmse,native_held_nrmse,proxy_start,proxy_final,proxy_start_nrmse,proxy_final_nrmse,objective_queries,native_queries,native_rgb_views,native_surface_views,shadow_faces,accepted,rejected,blocked,zero_columns,invalid,nonfinite,infinite_roots,pass_submit_s,filter_submit_s,forward_sync_s,backward_sync_s,estimated_map_bytes,wall_s,stop"))
                    foreach (var r in trials)
                        csv.WriteLine(string.Join(",", r.scene, r.key, r.method, r.objective, string.Join(";", r.lamps), r.passed ? 1 : 0, r.finalColliderLegal ? 1 : 0,
                            F(r.positionCm.Max()), F(r.intensityRelative.Max()), F(Math.Sqrt(r.nativeStart)), F(Math.Sqrt(r.nativeFinal)), F(r.nativeHighTrain), F(r.nativeHeld),
                            F(r.proxyStart), F(r.proxyFinal), F(r.proxyNrmseStart), F(r.proxyNrmseFinal), r.objectiveQueries, r.nativeQueries, r.nativeRgbViews, r.surfaceViews,
                            r.shadowFaces, r.accepted, r.rejected, r.blocked, r.zeroColumns, r.invalidBounds, r.nonfiniteBounds, r.infiniteRoots,
                            F(r.passSubmit), F(r.filterSubmit), F(r.forwardSync), F(r.backwardSync), r.estimatedMapBytes, F(r.seconds), r.stop));
                WriteJson(dir, "summary.json", new MbdStudyResult { branch = Arg("-bbE20Branch", "blackbox"), commit = Arg("-bbE20BaseCommit", E20BaseCommit),
                    experiment = stage, scene = scene.Describe(), truth = scene.Truth, arms = trials.ToArray(), nativeRgbViews = fit.RenderCount + high.RenderCount,
                    surfaceViews = surfaceReads, shadowFaces = maps.Values.Sum(x => x.map.Faces), surfaceSamples = sampleCount, renderers = geometry.Renderers,
                    alphaClipSlots = geometry.AlphaClipSlots, classroomSha256 = HashAsset("Assets/ClassroomScale/room.bytes"),
                    e19SceneSha256 = HashAsset(NativeComplexRoomBuilder.ScenePath), trainingAndHeldViews = new[] { "train:0", "train:1", "train:2", "held:3", "held:4" },
                    unityAllocatedBytes = UnityEngine.Profiling.Profiler.GetTotalAllocatedMemoryLong(), managedBytes = GC.GetTotalMemory(false),
                    seconds = clock.Elapsed.TotalSeconds });
            }

            private static string HashAsset(string path)
            {
                if (!File.Exists(path)) return "";
                using var sha = System.Security.Cryptography.SHA256.Create();
                using var stream = File.OpenRead(path);
                return BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
            }

            public void Dispose()
            {
                Apply(scene.Truth, true);
                foreach (var entry in maps.Values) entry.map.Dispose();
                geometry.Dispose(); fit.Dispose(); high.Dispose();
                queries.Dispose(); jacobians.Dispose(); probeCsv.Dispose();
            }
        }
    }
}

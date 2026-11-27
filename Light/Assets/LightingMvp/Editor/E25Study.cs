using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using UnityEngine;
using Frame = LightingMvp.NativeRenderBackend.Frame;
using Mapping = LightingMvp.NativeRenderBackend.HdrMapping;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private const string E25Baseline = "fa060f204469a4ed733a8f622aeb34aaf1761795";

        private static void RunE25Stages(string task, string output)
        {
            if (!Application.isBatchMode) throw new InvalidOperationException("E25 is batch only.");
            string key = task.Substring(4);
            bool classroom = key.Contains("classroom");
            using var pipeline = new TestPipeline(classroom: true);
            using ISensitivityScene scene = classroom ? new ClassroomSensitivityScene(pipeline.Asset, null) : new ComplexRoomSensitivityScene();
            using var study = new E21RoomStudy(scene, task, Path.Combine(output, task.Replace('-', '_')),
                classroom ? ClassroomAssets + "/room.bytes" : NativeComplexRoomBuilder.ScenePath);
            study.RunE25(key, classroom);
            study.Finish();
        }

        private sealed partial class E21RoomStudy
        {
            private const int E25Cap = 4725, E25Checkpoint = 1560;
            private static readonly int[] E25DefaultViews = { 0, 1, 2 };
            private string e25Inputs;
            private StreamWriter e25Costs, e25Scores, e25Queries, e25Checkpoints, e25Visibility, e25Clip;

            private E21StageSummary E25Read(string key)
            {
                var saved = JsonUtility.FromJson<E21StageSummary>(File.ReadAllText(Path.Combine(e25Inputs, key + ".json")));
                var actual = room.Describe();
                var prior = saved.scene;
                if (saved.sceneHash != E21Sha(scenePath) || prior.unity != actual.unity || prior.urp != actual.urp ||
                    prior.renderer != actual.renderer || prior.graphics != actual.graphics || prior.colorSpace != actual.colorSpace ||
                    prior.width != actual.width || prior.height != actual.height || prior.checkWidth != actual.checkWidth || prior.checkHeight != actual.checkHeight ||
                    room.Count != 7 || !E22SamePose(saved.truth, room.Truth) || prior.gt.Length != actual.gt.Length ||
                    !prior.gt.Zip(actual.gt, (a, b) => a.type == b.type && a.color == b.color && a.range == b.range &&
                        a.intensity == b.intensity && a.innerSpotAngle == b.innerSpotAngle && a.spotAngle == b.spotAngle &&
                        (a.position - b.position).sqrMagnitude < 1e-12f && (a.direction - b.direction).sqrMagnitude < 1e-12f).All(x => x) ||
                    prior.viewPositions.Length != room.Views.Length || prior.viewRotations.Length != room.Rotations.Length ||
                    !prior.viewPositions.Zip(room.Views, (a, b) => (a - b).sqrMagnitude < 1e-12f).All(x => x) ||
                    !prior.viewRotations.Zip(room.Rotations, (a, b) => Math.Abs(Quaternion.Dot(a, b)) > .999999f).All(x => x))
                    throw new InvalidDataException("E25 E24 replay mismatch: " + key);
                return saved;
            }

            internal void RunE25(string key, bool classroom)
            {
                e25Inputs = Arg("-bbE25Inputs", "");
                var saved = E25Read(classroom ? "a_classroom" : "a_complex");
                var hard = Target(true);
                e24HardScale = E22Scale(hard);
                using (e25Costs = Csv("arm-cost.csv", "arm,fit_views,cap_camera,opt_camera,hard_camera,off_camera,other_camera,shadow_pass_cameras,shadow_face_slices,render_s,readback_s,process_cpu_s,opt_wall_s,gc0,gc1,gc2,main_thread_allocated_bytes,gpu_s,scale,scene_sha,map_fd_s,mask_p95_s,gram_gradient_s,damped_solve_s,gpu_columns,gpu_upload_s,gpu_submit_s,gpu_sync_readback_s"))
                using (e25Scores = Csv("view-metrics.csv", "arm,checkpoint,target,width,view,raw_nrmse,log_nrmse,display_nrmse"))
                using (e25Queries = Csv("model-queries.csv", "arm,query,actual_camera,phase,shadows,axis,physical_step,cameras,theta"))
                using (e25Checkpoints = Csv("checkpoints.csv", "arm,checkpoint,actual_camera,completed_models,loss,max_position_cm,max_intensity_relative,legal,pass,theta"))
                using (e25Visibility = Csv("mask-stats.csv", E24VisibilityHeader))
                using (e25Clip = Csv("trust-clipping.csv", "arm,iteration,actual_camera,lamp,proposed_xyz_m,radius_xyz_m,final_xyz_m,proposed_logI,radius_logI,final_logI,damping,damping_scale,collider_shrinks,predicted,actual,reason"))
                {
                    if (key.StartsWith("c-complex-", StringComparison.Ordinal) || key.StartsWith("c-classroom-", StringComparison.Ordinal) ||
                        key.StartsWith("h1-", StringComparison.Ordinal))
                    {
                        string scenario = key.Split('-').Last() switch
                        {
                            "911" or "smoke" => "dir911_20cm", "1201" => "dir1201_30cm", "1301" => "dir1301_10cm",
                            "1601" => "dir1601_10cm", "509" => "seed509_complex_far", _ => throw new ArgumentException(key)
                        };
                        var starts = scenario.StartsWith("seed", StringComparison.Ordinal) ? saved : E25Read(classroom ? "e_classroom" : "e_complex");
                        var start = starts.arms.Single(a => a.scenario == scenario && a.method == (scenario.StartsWith("seed", StringComparison.Ordinal) ? "A-log" : "H0")).initial;
                        bool h1Only = key.StartsWith("h1-", StringComparison.Ordinal);
                        foreach (string method in h1Only ? new[] { "H1" } : new[] { "H0", "C3", "C4" })
                            E25Optimize(scenario, method, E24AllLamps, start, hard);
                        if (classroom && !h1Only) E25Optimize(scenario, "A-log", E24AllLamps, start, Target(false));
                    }
                    else E25Additional(key, saved, hard);
                }
            }

            // Optional stages share the same optimizer, limits, accounting and immutable snapshot validation.
            partial void E25Additional(string key, E21StageSummary saved, MbdRgbTarget hard);

            private E21Trial E25Optimize(string scenario, string method, int[] active, double[] initial, MbdRgbTarget native,
                int[] views = null, NativeRenderBackend fitBackend = null, E23FrameObjective fit = null,
                Action<double[], float[]> customCapture = null, E21Signal surface = null, int iterations = int.MaxValue,
                Action<E21Trial, Frame[]> extra = null, bool saveImages = true,
                Action<ImageJacobianLm.TrustJacobian> modelObserver = null,
                Action<ImageJacobianLm.TrustTrial, double[]> acceptedObserver = null,
                Action<ImageJacobianLm.TrustSettings, Func<double[], double[]>> configureRun = null)
            {
                views ??= E25DefaultViews;
                fitBackend ??= low;
                bool useProxy = method is "C3" or "C4" or "C3-cache" or "C4-cache" or "C4-gpu" or "C3-gpu";
                var mapping = method is "H0" or "A-raw" ? Mapping.Raw : Mapping.SignedLog;
                double scale = native.shadows ? e24HardScale : E22Scale(native);
                fit ??= new E23FrameObjective(native.fit, mapping, scale);
                if ((surface == null && fit.Size != views.Length * E23ChannelsPerView) || (useProxy && !views.SequenceEqual(E25DefaultViews)))
                    throw new ArgumentException("E25 capture and objective view counts must agree; anchored models use the E24 three-view rig.");
                int queryCost = surface == null ? views.Length : 0;
                int capacity = surface == null ? E25Cap / queryCost : E25Cap / E19FitViews;
                int cameras = 0, on = 0, off = 0, shadowViews = 0, slices = 0, lastQ = 0, lastModel = 0;
                double checkpointLoss = double.NaN, latestLoss = double.NaN;
                var checkpoint = (double[])initial.Clone();
                string id = scenario + "_" + method;
                bool metered = false;
                var frame = new float[fit.Size];
                int otherStart = low.RenderCount + high.RenderCount + (fitBackend == low ? 0 : fitBackend.RenderCount);
                using var observer = native.shadows ? new E23AtlasObserver(lights) : null;
                using var gpu = method is "C4-gpu" or "C3-gpu" ? new E25GpuColumns(fit) : null;
                LightShadows mode = native.shadows ? LightShadows.Hard : LightShadows.None;
                var signal = surface ?? new E21Signal(fit.mapping.ToString(), fit.Size, (rig, residual) =>
                {
                    Capture(rig, mode, frame);
                    return fit.Evaluate(frame, residual);
                });
                var timing = new ImageJacobianLm.TrustTiming();
                var metrics = new E25Counter();
                double render = 0, read = 0, wall = 0, cpu = 0;
                int gc0 = 0, gc1 = 0, gc2 = 0;
                long alloc = 0;
                E25Score(id, "initial", native, initial);
                var arm = RunArm(scenario, method, active, initial, native, signal, capacity, .02, partition: trial =>
                {
                    var sw = Stopwatch.StartNew();
                    using var process = Process.GetCurrentProcess();
                    var cpuBefore = process.TotalProcessorTime;
                    var renderBefore = fitBackend.RenderSeconds;
                    var readBefore = fitBackend.ReadbackSeconds;
                    gc0 = GC.CollectionCount(0); gc1 = GC.CollectionCount(1); gc2 = GC.CollectionCount(2);
                    alloc = GC.GetAllocatedBytesForCurrentThread();
                    checkpointLoss = trial.initialLoss;
                    metered = true;
                    var final = Solve(trial, "joint", active, initial, signal, trial.limit, null, iterations: iterations,
                        configure: (settings, expand) =>
                        {
                            settings.Timing = timing;
                            settings.TraceProposalClipping = stage.StartsWith("e25-b-", StringComparison.Ordinal);
                            E24Configure(settings, expand, active, method.Replace("-cache", "").Replace("-gpu", ""), fit, ModelCapture, id, e25Visibility,
                                cachedMapping: gpu != null || method.EndsWith("-cache", StringComparison.Ordinal), metrics: metrics, gpu: gpu);
                            if (method == "A-log-diag") settings.NormalMode = ImageJacobianLm.TrustNormalMode.ParameterScaledDense;
                            var existingTrial = settings.Trial;
                            settings.Trial = t =>
                            {
                                existingTrial(t);
                                if (e25Clip != null && settings.TraceProposalClipping && t.UnclippedStep != null)
                                    for (int j = 0; j < active.Length; ++j)
                                    {
                                        int offset = 4 * j;
                                        double XYZ(double[] d) => Math.Sqrt(Enumerable.Range(0, 3).Sum(k => Math.Pow(d[offset + k] * units[k], 2)));
                                        e25Clip.WriteLine(string.Join(",", id, t.Iteration, cameras, active[j] + 1, F(XYZ(t.UnclippedStep)), F(XYZ(t.RadiusLimitedStep)),
                                            F(XYZ(t.Step)), F(t.UnclippedStep[offset + 3] * units[3]), F(t.RadiusLimitedStep[offset + 3] * units[3]),
                                            F(t.Step[offset + 3] * units[3]), F(t.Damping[0]), Series(t.DampingScale.Skip(offset).Take(4)),
                                            t.Shrinks, F(t.Predicted), F(t.Actual), t.Status));
                                    }
                                if (t.Accepted) acceptedObserver?.Invoke(t, expand(t.Theta));
                                if (t.Accepted && t.Query * queryCost <= E25Checkpoint)
                                {
                                    Array.Copy(expand(t.Theta), checkpoint, initial.Length);
                                    lastQ = t.Query; lastModel = t.Iteration;
                                    checkpointLoss = latestLoss;
                                }
                            };
                            var existingQuery = settings.Query;
                            settings.Query = (q, phase, x, loss, axis, offset) =>
                            {
                                existingQuery(q, phase, x, loss, axis, offset);
                                if (phase == "trial" || phase.StartsWith("views_", StringComparison.Ordinal) || phase is "cauchy" or "nonmonotone_recheck") latestLoss = loss;
                                e25Queries.WriteLine(string.Join(",", id, q, cameras, phase, surface == null ? mode.ToString() : "surface",
                                    axis < 0 ? -1 : 4 * active[axis / 4] + axis % 4, F(offset), queryCost, Series(expand(x))));
                            };
                            if (modelObserver != null)
                            {
                                var existingJac = settings.Jacobian;
                                settings.Jacobian = j => { existingJac(j); modelObserver(j); };
                            }
                            configureRun?.Invoke(settings, expand);
                        });
                    metered = false;
                    wall = sw.Elapsed.TotalSeconds; cpu = (process.TotalProcessorTime - cpuBefore).TotalSeconds;
                    render = fitBackend.RenderSeconds - renderBefore; read = fitBackend.ReadbackSeconds - readBefore;
                    gc0 = GC.CollectionCount(0) - gc0; gc1 = GC.CollectionCount(1) - gc1; gc2 = GC.CollectionCount(2) - gc2;
                    alloc = GC.GetAllocatedBytesForCurrentThread() - alloc;
                    return final;
                }, validation: (trial, frames, _, __) =>
                {
                    E25Score(id, "final", native, trial.final, frames);
                    if (!native.shadows) E25Score(id, "final", Target(true), trial.final);
                    extra?.Invoke(trial, frames);
                }, saveImages: saveImages);
                if (cameras != arm.evaluations * queryCost || cameras > E25Cap || arm.illegalAccepted != 0)
                    throw new InvalidOperationException("E25 solver and actual render accounting diverged: " + id);
                string checkpointName = surface == null ? "last_accepted_at_or_before_1560" : "last_accepted_surface_evaluation";
                E25CheckpointRow(id, checkpointName, lastQ * queryCost, lastModel, checkpointLoss, checkpoint, active);
                E25CheckpointRow(id, "final", cameras, arm.jacobians, arm.finalLoss, arm.final, active);
                int other = low.RenderCount + high.RenderCount + (fitBackend == low ? 0 : fitBackend.RenderCount) - otherStart - cameras;
                e25Costs.WriteLine(string.Join(",", id, string.Join(";", views), capacity * queryCost, cameras, on, off, other, shadowViews, slices,
                    F(render), F(read), F(cpu), F(wall), gc0, gc1, gc2, alloc == 0 ? "unavailable" : alloc.ToString(Invariant), "unavailable", F(fit.scale), E21Sha(scenePath), F(metrics.Map), F(metrics.Mask), F(timing.GramGradientSeconds), F(timing.DampedSolveSeconds), gpu?.Columns ?? 0, gpu == null ? "" : F(gpu.UploadSeconds),
                    gpu == null ? "" : F(gpu.SubmitSeconds), gpu == null ? "" : F(gpu.ReadbackSeconds)));
                e25Costs.Flush(); e25Queries.Flush(); e25Checkpoints.Flush(); e25Scores.Flush(); e25Visibility.Flush();
                return arm;

                void ModelCapture(int q, double[] rig, LightShadows shadows, string phase, int axis, double step, float[] values)
                {
                    int before = cameras;
                    Capture(rig, shadows, values);
                    e25Queries.WriteLine(string.Join(",", id, q, cameras, phase, shadows, axis, F(step), cameras - before, Series(rig)));
                }

                void Capture(double[] rig, LightShadows shadows, float[] values)
                {
                    int before = fitBackend.RenderCount;
                    if (customCapture != null) customCapture(rig, values);
                    else
                    {
                        Apply(rig, shadows != LightShadows.None, shadows);
                        for (int j = 0; j < views.Length; ++j)
                        {
                            int view = views[j], offset = j * E23ChannelsPerView;
                            if (metered && shadows == LightShadows.Hard)
                            {
                                int count = observer.Observe(() => fitBackend.CaptureRgb(view, values, offset)).faces.Length;
                                if (count > 0) ++shadowViews;
                                slices += count;
                            }
                            else fitBackend.CaptureRgb(view, values, offset);
                        }
                    }
                    if (!metered) return;
                    int used = fitBackend.RenderCount - before;
                    cameras += used;
                    if (shadows == LightShadows.Hard) on += used; else off += used;
                }
            }

            private void E25CheckpointRow(string id, string kind, int count, int models, double loss, double[] rig, int[] active)
            {
                var position = active.Max(i => 100 * Vector3.Distance(room.PositionOf(rig, i), room.PositionOf(room.Truth, i)));
                var intensity = active.Max(i => Math.Abs(room.PowerOf(rig, i) / room.PowerOf(room.Truth, i) - 1));
                bool legal = room.Contains(rig);
                e25Checkpoints.WriteLine(string.Join(",", id, kind, count, models, F(loss), F(position), F(intensity), Flag(legal),
                    Flag(legal && position <= 5 && intensity <= .05), Series(rig)));
            }

            private void E25Score(string id, string phase, MbdRgbTarget target, double[] rig, Frame[] alreadyHigh = null)
            {
                Apply(rig, target.shadows);
                double scale = E22Scale(target);
                for (int j = 0; j < E19FitViews; ++j) Write(low.Capture(j), target.fit[j], low.Width, j);
                for (int j = 0; j < room.Views.Length; ++j) Write(alreadyHigh == null ? high.Capture(j) : alreadyHigh[j], target.high[j], high.Width, j);
                void Write(Frame actual, Frame truth, int width, int view)
                {
                    var score = E25FrameScores(actual, truth, scale);
                    e25Scores.WriteLine(string.Join(",", id, phase, target.shadows ? "hard" : "off", width, view, Series(score).Replace(';', ',')));
                }
            }

            private static double[] E25FrameScores(Frame actual, Frame truth, double scale)
            {
                double e = 0, n = 0;
                for (int i = 0; i < truth.PixelCount; ++i)
                {
                    var a = actual.Pixel(i); var t = truth.Pixel(i);
                    for (int ch = 0; ch < 3; ++ch)
                    {
                        double x = NativeRenderBackend.Map(t[ch], Mapping.SignedLog, scale);
                        double d = NativeRenderBackend.Map(a[ch], Mapping.SignedLog, scale) - x;
                        e += d * d; n += x * x;
                    }
                }
                return new[] { Math.Sqrt(NativeRenderBackend.Compare(actual, truth)), Math.Sqrt(e / (n + 1e-12)),
                    Math.Sqrt(DisplayLoss(actual, new E22DisplayFrame(truth))) };
            }
        }
    }
}

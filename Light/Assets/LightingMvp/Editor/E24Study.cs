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
        private static void RunE24Stages(string task, string output)
        {
            if (!Application.isBatchMode) throw new InvalidOperationException("E24 is batch only.");
            var key = task.Substring("e24-".Length);
            var suffix = key.EndsWith("-smoke", StringComparison.Ordinal);
            if (suffix) key = key.Substring(0, key.Length - "-smoke".Length);
            bool classroom = key.EndsWith("-classroom", StringComparison.Ordinal);
            using var pipeline = new TestPipeline(classroom: true);
            using ISensitivityScene scene = classroom ? new ClassroomSensitivityScene(pipeline.Asset, null) : new ComplexRoomSensitivityScene();
            using var study = new E21RoomStudy(scene, task, Path.Combine(output, task.Replace('-', '_')),
                classroom ? ClassroomAssets + "/room.bytes" : NativeComplexRoomBuilder.ScenePath);
            study.RunE24(key, classroom);
            study.Finish();
        }

        private sealed partial class E21RoomStudy
        {
            private const int E24Screen = 520, E24Validate = 1575;
            private const string E24ReferenceCommit = "a89669d68ad6ce3927f7bb1e7bbe250e2100de9e";
            private string e24Inputs;
            // ClassroomSensitivityScene.Lamps is the four-lamp E17 display subset, not the full rig.
            private int[] E24AllLamps => Enumerable.Range(0, room.Count).ToArray();
            private double e24HardScale;
            private bool e24MatchedAsset;
            private StreamWriter e24Arm, e24View, e24Lamp, e24Model, e24Switch, e24OptimizationVisibility;

            private E21StageSummary E24Read(string key)
            {
                var record = JsonUtility.FromJson<E21StageSummary>(File.ReadAllText(Path.Combine(e24Inputs, key + ".json")));
                if (record?.arms == null || record.scene?.unity != Application.unityVersion || record.scene.urp != room.Describe().urp)
                    throw new InvalidDataException("Incompatible E24 reference " + key);
                return record;
            }
            private E21StageSummary E24Reference(bool classroom) => E24Read(classroom ? "classroom" : "far");
            private static E21Trial E24Prior(E21StageSummary reference, string method) => reference.arms.Single(a => a.method == method);

            internal void RunE24(string key, bool classroom)
            {
                e24Inputs = Arg("-bbE24Inputs", "");
                var reference = E24Reference(classroom);
                string assetHash = E21Sha(scenePath);
                if (room.Count != 7 || !E22SamePose(reference.truth, room.Truth))
                    throw new InvalidDataException("E24 GT differs from recorded E23 rig; snapshots are not interchangeable.");
                if (!reference.scene.viewPositions.Zip(room.Views, (a, b) => Vector3.Distance(a, b) < 1e-6f).All(x => x) ||
                    !reference.scene.viewRotations.Zip(room.Rotations, (a, b) => Math.Abs(Quaternion.Dot(a, b)) > .999999f).All(x => x))
                    throw new InvalidDataException("E24 camera poses differ from E23.");
                e24MatchedAsset = assetHash.Equals(reference.sceneHash, StringComparison.OrdinalIgnoreCase);
                if (!e24MatchedAsset && Arg("-bbE24AllowDifferentAssetHash", "0") != "1")
                    throw new InvalidDataException($"E23 scene SHA {reference.sceneHash}; local SHA {assetHash}. Exact replay is the default.");
                using (e24Arm = Csv("arm-cost.csv", "arm,model,objective,scale,cap_camera,on_camera,off_camera,opt_camera,other_camera,render_seconds,readback_seconds,cpu_seconds,opt_wall_seconds,asset_hash_match,hard_shadow_pass_cameras,hard_shadow_face_slices,gpu_seconds"))
                using (e24View = Csv("view-metrics.csv", "arm,checkpoint,target,width,view,energy,raw_nrmse,log_nrmse,display_nrmse"))
                using (e24Lamp = Csv("lamp-metrics.csv", "arm,lamp,start_cm,start_log_intensity,range,nearest_collider_m,final_cm,final_intensity_relative"))
                using (e24Model = Csv("model-queries.csv", "arm,query,phase,shadows,axis,signed_step,camera_renders,theta"))
                using (e24Switch = Csv("switch-log.csv", "arm,reason,used_camera,remaining_camera,from_model,to_model,theta"))
                using (e24OptimizationVisibility = Csv("optimization-visibility.csv", E24VisibilityHeader))
                {
                    var hard = Target(true);
                    e24HardScale = E22Scale(hard);
                    double priorScale = classroom ? .2584995627403259 : .38417816162109375;
                    var prior = E24Prior(reference, "C1");
                    var im = new float[E23Channels];
                    E23CaptureRgb(prior.initial, LightShadows.Hard, im);
                    var sameObjective = new E23FrameObjective(hard.fit, Mapping.SignedLog, priorScale);
                    WriteJson(directory, "provenance.json", new E24Provenance { referenceCommit = E24ReferenceCommit, referenceSceneHash = reference.sceneHash,
                        localSceneHash = assetHash, sceneHashMatch = e24MatchedAsset, hardScale = e24HardScale, referenceHardScale = priorScale,
                        referenceHardLogStartLoss = prior.initialLoss, localHardLogStartLossAtReferenceScale = sameObjective.Evaluate(im) });
                    switch (key)
                    {
                        case "a-complex": case "a-classroom": E24Off(reference); break;
                        case "bc-complex": case "bc-classroom": E24DirectionAudit(reference, hard, classroom); break;
                        case "p-small": E24ScreenRecovery(hard, false); break;
                        case "p-full": E24ScreenRecovery(hard, true); break;
                        case "d-complex": E24Refinement(hard); break;
                        case "e-complex": case "e-classroom": E24Sweep(reference, hard, classroom, false); break;
                        case "e50-complex": case "e50-classroom": E24Sweep(reference, hard, classroom, true); break;
                        default: throw new ArgumentException("Unknown E24 stage: " + key);
                    }
                }
            }

            [Serializable] private sealed class E24Provenance
            {
                public string referenceCommit, referenceSceneHash, localSceneHash;
                public bool sceneHashMatch;
                public double hardScale, referenceHardScale, referenceHardLogStartLoss, localHardLogStartLossAtReferenceScale;
            }

            private void E24Off(E21StageSummary reference)
            {
                var off = Target(false);
                var prior = E24Prior(reference, "C1");
                if (!reference.arms.All(a => E22SamePose(a.initial, prior.initial))) throw new InvalidDataException("E23 arms did not share the seed509 snapshot.");
                var log = E24Optimize(prior.scenario, "A-log", E24AllLamps, prior.initial, off, E24Validate);
                if (log.pass || smoke) return;
                E24Optimize(prior.scenario, "A-raw", E24AllLamps, prior.initial, off, E24Validate);
                int[] worst = Enumerable.Range(0, log.lamps.Length).OrderByDescending(j => Math.Max(log.positionCm[j] / 5, log.intensityRelative[j] / .05))
                    .Take(2).Select(j => log.lamps[j] - 1).ToArray();
                foreach (int i in worst)
                    E24Optimize(prior.scenario + "_oracle_L" + (i + 1), "A-log", new[] { i }, Start(prior.initial, new[] { i }), off, E24Validate);
            }

            private static bool E24Raw(string model) => model == "A-raw" || model == "H0" || model == "C1";

            private E21Trial E24Optimize(string scenario, string model, int[] active, double[] initial, MbdRgbTarget target, int cap, string tail = null)
            {
                Mapping mapping = E24Raw(model) ? Mapping.Raw : Mapping.SignedLog;
                double scale = target.shadows ? e24HardScale : E22Scale(target);
                string id = scenario + "_" + model;
                var objective = new E23FrameObjective(target.fit, mapping, scale);
                var currentObjective = objective;
                var rgb = new float[E23Channels];
                bool metered = false;
                int on = 0, off = 0, shadowCameras = 0, faceSlices = 0;
                using var observer = target.shadows ? new E23AtlasObserver(lights) : null;
                var signal = new E21Signal(mapping.ToString(), E23Channels, (rig, residual) =>
                {
                    Capture(rig, target.shadows ? LightShadows.Hard : LightShadows.None, rgb);
                    if (metered)
                    {
                        if (target.shadows) on += E19FitViews;
                        else off += E19FitViews;
                    }
                    return currentObjective.Evaluate(rgb, residual);
                });
                int begin = low.RenderCount + high.RenderCount;
                double render = 0, readback = 0, cpu = 0, wall = 0;
                E24Score(id, "initial", target, initial, scale);
                var arm = RunArm(scenario, model, active, initial, target, signal, cap, .02, partition: trial =>
                {
                    using var process = Process.GetCurrentProcess();
                    double startCpu = process.TotalProcessorTime.TotalSeconds;
                    double startRender = low.RenderSeconds, startRead = low.ReadbackSeconds;
                    var timer = Stopwatch.StartNew();
                    metered = true;
                    var end = Solve(trial, tail == null ? "joint" : "aux", active, initial, signal,
                        tail == null ? trial.limit : (int)Math.Floor(.75 * trial.limit), null, configure: (settings, expand) =>
                        {
                            E24Configure(settings, expand, active, model, objective, ModelCapture, id, e24OptimizationVisibility);
                            if (tail != null) settings.MaxConsecutiveCandidateRejects = 3;
                        });
                    if (tail != null)
                    {
                        int remaining = trial.limit - trial.evaluations;
                        string reason = trial.stop == "budget" ? "reserved_25pct" : trial.stop;
                        bool switchAllowed = trial.stop is "budget" or "no_decreasing_step" or "consecutive_candidate_rejects_shrunk";
                        bool enough = remaining >= 2 * active.Length * 4 + 2;
                        bool refine = switchAllowed && enough;
                        e24Switch.WriteLine(string.Join(",", id, switchAllowed && !enough ? "too_few_cameras" : reason,
                            trial.evaluations * E19FitViews, remaining * E19FitViews, model, refine ? tail : "", Series(end)));
                        e24Switch.Flush();
                        if (refine)
                        {
                            currentObjective = new E23FrameObjective(target.fit, E24Raw(tail) ? Mapping.Raw : Mapping.SignedLog, scale);
                            end = Solve(trial, "native_refine", active, end, signal, remaining, null,
                                configure: (settings, _) => settings.NormalMode = ImageJacobianLm.TrustNormalMode.LampScaledDense);
                            currentObjective = objective; // Public arm loss is always scored in its original mapping.
                        }
                    }
                    metered = false;
                    render = low.RenderSeconds - startRender; readback = low.ReadbackSeconds - startRead;
                    cpu = process.TotalProcessorTime.TotalSeconds - startCpu; wall = timer.Elapsed.TotalSeconds;
                    return end;
                }, validation: (trial, highFrames, _, __) =>
                {
                    E24Score(id, "final", target, trial.final, scale, highFrames);
                    if (!target.shadows) E24Score(id, "final", Target(true), trial.final, e24HardScale);
                });
                if (on + off != E19FitViews * arm.evaluations || on + off > E19FitViews * arm.limit)
                    throw new InvalidOperationException("E24 optimization camera counter disagrees with solver budget.");
                e24Arm.WriteLine(string.Join(",", id, model, mapping, F(scale), E19FitViews * arm.limit, on, off, on + off,
                    low.RenderCount + high.RenderCount - begin - on - off, F(render), F(readback), F(cpu), F(wall), Flag(e24MatchedAsset), shadowCameras, faceSlices, ""));
                for (int j = 0; j < active.Length; ++j)
                {
                    int i = active[j]; var position = room.PositionOf(initial, i);
                    e24Lamp.WriteLine(string.Join(",", id, i + 1, F(Vector3.Distance(position, room.PositionOf(room.Truth, i)) * 100),
                        F((initial[4 * i + 3] - room.Truth[4 * i + 3]) * LogPowerSpan), F(lights[i].range), F(E24Nearest(position)),
                        F(arm.positionCm[j]), F(arm.intensityRelative[j])));
                }
                e24Arm.Flush(); e24View.Flush(); e24Lamp.Flush(); e24Model.Flush();
                return arm;

                void ModelCapture(int q, double[] rig, LightShadows shadows, string phase, int axis, double step, float[] buffer)
                {
                    int beginQuery = low.RenderCount;
                    Capture(rig, shadows, buffer);
                    int count = low.RenderCount - beginQuery;
                    if (shadows == LightShadows.None) off += count; else on += count;
                    e24Model.WriteLine(string.Join(",", id, q, phase, shadows, axis, F(step), count, Series(rig)));
                }

                void Capture(double[] rig, LightShadows shadows, float[] buffer)
                {
                    if (!metered || shadows != LightShadows.Hard)
                    {
                        E23CaptureRgb(rig, shadows, buffer);
                        return;
                    }
                    Apply(rig, true);
                    for (int v = 0; v < E19FitViews; ++v)
                    {
                        int faces = observer.Observe(() => low.CaptureRgb(v, buffer, v * E23ChannelsPerView)).faces.Length;
                        if (faces > 0) ++shadowCameras;
                        faceSlices += faces;
                    }
                }
            }

            private static double E24Nearest(Vector3 point)
            {
                var nearby = Physics.OverlapSphere(point, 2, ~0, QueryTriggerInteraction.Ignore)
                    .Where(c => c is BoxCollider || c is SphereCollider || c is CapsuleCollider || c is MeshCollider m && m.convex).ToArray();
                return nearby.Length == 0 ? double.NaN : nearby.Min(c => (c.ClosestPoint(point) - point).magnitude);
            }

            private void E24Score(string id, string phase, MbdRgbTarget gt, double[] rig, double scale, Frame[] alreadyHigh = null)
            {
                Apply(rig, gt.shadows);
                for (int v = 0; v < E19FitViews; ++v) Score(low.Capture(v), gt.fit[v], low.Width, v);
                for (int v = 0; v < room.Views.Length; ++v) Score(alreadyHigh == null ? high.Capture(v) : alreadyHigh[v], gt.high[v], high.Width, v);

                void Score(Frame candidate, Frame reference, int width, int view)
                {
                    double logDiff = 0, logEnergy = 0;
                    for (int i = 0; i < reference.PixelCount; ++i)
                    {
                        var truth = reference.Pixel(i); var sample = candidate.Pixel(i);
                        for (int ch = 0; ch < 3; ++ch)
                        {
                            double x = NativeRenderBackend.Map(truth[ch], Mapping.SignedLog, scale);
                            double d = NativeRenderBackend.Map(sample[ch], Mapping.SignedLog, scale) - x;
                            logDiff += d * d; logEnergy += x * x;
                        }
                    }
                    e24View.WriteLine(string.Join(",", id, phase, gt.shadows ? "hard" : "off", width, view, F(reference.Energy),
                        F(Math.Sqrt(NativeRenderBackend.Compare(candidate, reference))), F(Math.Sqrt(logDiff / (logEnergy + 1e-12))),
                        F(Math.Sqrt(DisplayLoss(candidate, new E22DisplayFrame(reference))))));
                }
            }
        }
    }
}

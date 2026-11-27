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
        private static void RunE21RoomStages(string stage, string output)
        {
            if (!Application.isBatchMode) throw new InvalidOperationException("E21 uses an isolated batch project.");
            using var pipeline = new TestPipeline(classroom: true);
            using var room = new ComplexRoomSensitivityScene();
            using var study = new E21RoomStudy(room, stage, Path.Combine(output, stage.Replace('-', '_')));
            switch (stage[4])
            {
                case 'b': study.ProbePolicies(); break;
                case 'c': study.JointIrradiance(); break;
                case 'd': study.RgbHotspots(); break;
                case 'e': study.CompareProxies(); break;
                default: throw new ArgumentException(stage);
            }
            study.Finish();
        }

        [Serializable] private sealed class E21Trial
        {
            public string id, scenario, method, objective, stop, probePolicy;
            public int[] lamps;
            public bool shadows, pass, legal;
            public int limit, evaluations, objectiveQueries, nativeChecks, jacobians, accepted, rejected, illegalAccepted, blocked, zeroColumns,
                nativeCameraRenders, shadowFaces, surfaceRenders;
            public long rayQueries;
            public double positionStep, initialLoss, finalLoss, nativeInitial, nativeFinal, trainHigh, held, seconds;
            public double[] initial, final, positionCm, intensityRelative, nativePerView;
        }

        [Serializable] private sealed class E21StageSummary
        {
            public string stage, branch, commit, sceneHash;
            public E17Config scene;
            public double[] truth;
            public string[] views;
            public E21Trial[] arms;
            public int nativeCameraRenders, shadowFaces, surfaceRenders;
            public long rayQueries;
            public double seconds;
        }

        private sealed class E21Signal
        {
            internal readonly string name;
            internal readonly int size;
            internal readonly Func<double[], float[], double> objective;
            internal readonly int[] residualBlocks;
            internal E21Signal(string name, int size, Func<double[], float[], double> objective, int[] residualBlocks = null)
            { this.name = name; this.size = size; this.objective = objective; this.residualBlocks = residualBlocks; }
        }

        private sealed partial class E21RoomStudy : IDisposable
        {
            private readonly ISensitivityScene room;
            private readonly string stage, directory, scenePath;
            private readonly Func<double[], bool> inSurveyDomain;
            private readonly NativeRenderBackend low, high;
            private readonly Light[] lights;
            private readonly double[] units;
            private readonly bool smoke;
            private readonly Stopwatch clock = Stopwatch.StartNew();
            private readonly List<E21Trial> arms = new List<E21Trial>();
            private readonly Dictionary<bool, MbdRgbTarget> targets = new Dictionary<bool, MbdRgbTarget>();
            private readonly StreamWriter queryLog, jacobianLog, trialLog, gramLog;
            private int shadowFaces, surfaceRenders;
            private long rayQueries;

            internal E21RoomStudy(ISensitivityScene room, string stage, string directory, string scenePath = NativeComplexRoomBuilder.ScenePath)
            {
                this.room = room; this.stage = stage; this.directory = directory; this.scenePath = scenePath;
                inSurveyDomain = room is ComplexRoomSensitivityScene complex ? complex.InSurveyDomain : room.Contains;
                smoke = stage.EndsWith("-smoke", StringComparison.Ordinal);
                Directory.CreateDirectory(directory);
                units = new[] { (double)room.Extents.x, room.Extents.y, room.Extents.z, LogPowerSpan };
                low = new NativeRenderBackend(room.Camera, room.Views, room.Rotations, E19FitW, E19FitH, -1);
                high = new NativeRenderBackend(room.Camera, room.Views, room.Rotations, E19HighW, E19HighH, -1);
                lights = Enumerable.Range(0, room.Count).Select(i => GameObject.Find(room.UnityName(i)).GetComponent<Light>()).ToArray();
                queryLog = Csv("queries.csv", "arm,segment,query,phase,axis,signed_step,collider_legal,survey_legal,loss,theta,stage_s");
                jacobianLog = Csv("jacobians.csv", "arm,segment,iteration,query,lamps,rank,condition,singular_values,column_norms,physical_steps,stencils,gradient_theta,off_block_frobenius,off_block_relative,blocked,theta");
                gramLog = Csv("gram.csv", "arm,segment,iteration,row,values");
                trialLog = Csv("decisions.csv", "arm,segment,iteration,query,legal,accepted,status,shrinks,predicted,actual,rho,native_actual,position_radius_m,intensity_radius_log,damping,step_phys,predicted_by_lamp,actual_by_lamp");
            }

            private StreamWriter Csv(string file, string header)
            {
                var stream = new StreamWriter(Path.Combine(directory, file));
                stream.WriteLine(header);
                return stream;
            }

            private static string Series(IEnumerable<double> v) => string.Join(";", v.Select(F));
            private static string LampList(IEnumerable<int> v) => string.Join(";", v.Select(i => i + 1));
            private static int Flag(bool value) => value ? 1 : 0;
            private double[] Start(double[] mixed, int[] lamps)
            {
                var x = (double[])room.Truth.Clone();
                foreach (int lamp in lamps) Array.Copy(mixed, 4 * lamp, x, 4 * lamp, 4);
                return x;
            }

            private void Apply(double[] theta, bool shadows, LightShadows? modeOverride = null)
            {
                // E21's solver independently audits probe/candidate policies before rendering.
                var mode = modeOverride ?? (shadows ? (e23SoftTarget ? LightShadows.Soft : LightShadows.Hard) : LightShadows.None);
                e23ShadowSettings?.Invoke(mode);
                for (int i = 0; i < lights.Length; ++i)
                {
                    var lamp = lights[i];
                    lamp.transform.position = room.PositionOf(theta, i);
                    lamp.intensity = (float)room.PowerOf(theta, i);
                    lamp.enabled = true;
                    room.SetShadows(i, mode != LightShadows.None);
                    if (mode == LightShadows.Soft) lamp.shadows = LightShadows.Soft;
                }
            }

            private MbdRgbTarget Target(bool shadows)
            {
                if (targets.TryGetValue(shadows, out var existing)) return existing;
                Apply(room.Truth, shadows);
                for (int v = 0; v < E19FitViews; ++v) low.Capture(v);
                var fitFrames = Enumerable.Range(0, E19FitViews).Select(low.Capture).ToArray();
                for (int v = 0; v < E19FitViews; ++v)
                    if (NativeRenderBackend.Compare(low.Capture(v), fitFrames[v]) > 1e-12)
                        throw new InvalidOperationException("Native target repeat differs.");
                var highFrames = Enumerable.Range(0, room.Views.Length).Select(high.Capture).ToArray();
                var target = new MbdRgbTarget(fitFrames, highFrames, shadows);
                targets.Add(shadows, target);
                string path = CapturePath(shadows ? e23SoftTarget ? "gt_soft" : "gt_on" : "gt_off");
                SaveFrames(highFrames, path, "gt", stage[4] == 'd');
                return target;
            }

            private string CapturePath(string key)
            {
                string path = Path.Combine(directory, "captures", key);
                Directory.CreateDirectory(path);
                return path;
            }

            private static void SaveFrames(Frame[] frames, string path, string label, bool allFitViews)
            {
                for (int v = 0; v < frames.Length; ++v)
                    if (v == 0 || v == 3 || allFitViews && v < E19FitViews)
                        NativeRenderBackend.Save(frames[v], E19HighW, E19HighH, Path.Combine(path, $"{label}_v{v}"));
            }

            private double NativeLoss(MbdRgbTarget gt, double[] theta, float[] residual, bool highFit = false)
            {
                Apply(theta, gt.shadows);
                var backend = highFit ? high : low;
                var frames = highFit ? gt.high : gt.fit;
                double loss = 0;
                int channels = backend.Width * backend.Height * 3;
                for (int v = 0; v < E19FitViews; ++v)
                    loss += residual == null ? backend.Loss(v, frames[v]) / E19FitViews : backend.Residual(v, frames[v], residual, channels * v, E19FitViews);
                return loss;
            }

            private E21Signal NativeSignal(MbdRgbTarget gt, bool highFit = false) =>
                new E21Signal(highFit ? "native_l2_896" : "native_l2_448",
                    (highFit ? E19HighW * E19HighH : E19FitW * E19FitH) * E19FitViews * 3,
                    (x, r) => NativeLoss(gt, x, r, highFit));

            private E21Trial RunArm(string scenario, string method, int[] lamps, double[] initial, MbdRgbTarget gt, E21Signal signal,
                int cap, double step, string policy = "strict", bool nativeGate = false, bool independent = false,
                Func<E21Trial, double[]> partition = null, Action<E21Trial, Frame[], MbdRgbTarget, string> validation = null, bool saveImages = true)
            {
                int budget = smoke ? Math.Min(80, cap) : cap;
                var arm = new E21Trial { id = $"{scenario}_{method}", scenario = scenario, method = method, lamps = lamps.Select(i => i + 1).ToArray(),
                    initial = initial, objective = signal.name, probePolicy = policy, limit = budget, positionStep = step, shadows = gt.shadows };
                int renders = low.RenderCount + high.RenderCount, faces = shadowFaces, surfaces = surfaceRenders;
                long rays = rayQueries;
                var timer = Stopwatch.StartNew();
                arm.nativeInitial = NativeLoss(gt, initial, null);
                arm.initialLoss = signal.objective(initial, null);
                arm.final = partition == null ? Solve(arm, "joint", lamps, initial, signal, budget, nativeGate ? gt : null, independent) : partition(arm);
                arm.finalLoss = signal.objective(arm.final, null);
                arm.nativeFinal = NativeLoss(gt, arm.final, null);
                if (nativeGate && arm.nativeFinal > arm.nativeInitial + 1e-9)
                    throw new InvalidOperationException("The accepted native endpoint must improve or equal the start.");
                Apply(arm.final, gt.shadows);
                var frames = Enumerable.Range(0, room.Views.Length).Select(high.Capture).ToArray();
                arm.nativePerView = frames.Select((f, i) => NativeRenderBackend.Compare(f, gt.high[i])).Select(Math.Sqrt).ToArray();
                arm.trainHigh = Math.Sqrt(arm.nativePerView.Take(E19FitViews).Average(x => x * x));
                arm.held = Math.Sqrt(arm.nativePerView.Skip(E19FitViews).Average(x => x * x));
                arm.legal = room.Contains(arm.final);
                arm.positionCm = lamps.Select(i => (double)Vector3.Distance(room.PositionOf(room.Truth, i), room.PositionOf(arm.final, i)) * 100).ToArray();
                arm.intensityRelative = lamps.Select(i => Math.Abs(room.PowerOf(arm.final, i) / room.PowerOf(room.Truth, i) - 1)).ToArray();
                arm.pass = arm.legal && arm.positionCm.All(x => x <= 5) && arm.intensityRelative.All(x => x <= .05);
                string path = CapturePath(arm.id);
                if (saveImages) SaveFrames(frames, path, "best", stage[4] == 'd' && !stage.StartsWith("e22-", StringComparison.Ordinal));
                validation?.Invoke(arm, frames, gt, path);
                Apply(initial, gt.shadows);
                if (saveImages) NativeRenderBackend.Save(high.Capture(0), E19HighW, E19HighH, Path.Combine(path, "initial_v0"));
                arm.seconds = timer.Elapsed.TotalSeconds;
                arm.nativeCameraRenders = low.RenderCount + high.RenderCount - renders;
                arm.shadowFaces = shadowFaces - faces;
                arm.surfaceRenders = surfaceRenders - surfaces;
                arm.rayQueries = rayQueries - rays;
                if (!arm.legal || arm.evaluations > budget) throw new InvalidOperationException("Arm returned an illegal or over-budget result.");
                arms.Add(arm);
                WriteJson(path, "trial.json", arm);
                queryLog.Flush(); jacobianLog.Flush(); trialLog.Flush(); gramLog.Flush();
                Debug.Log($"[{stage.ToUpperInvariant()}] {arm.id}: {Math.Sqrt(arm.nativeInitial):P2} -> {Math.Sqrt(arm.nativeFinal):P2} RGB, " +
                    $"{arm.positionCm.Max():F2} cm / {arm.intensityRelative.Max():P2}, {arm.evaluations} evaluations, {arm.stop}");
                return arm;
            }

            private double[] Solve(E21Trial arm, string segment, int[] lamps, double[] start, E21Signal signal,
                int budget, MbdRgbTarget nativeGuard, bool independent = false, int iterations = int.MaxValue,
                Action<ImageJacobianLm.TrustResult> record = null,
                Action<ImageJacobianLm.TrustSettings, Func<double[], double[]>> configure = null)
            {
                int[] indices = lamps.SelectMany(i => Enumerable.Range(4 * i, 4)).ToArray();
                var full = (double[])start.Clone();
                double[] Expand(double[] x)
                {
                    for (int j = 0; j < indices.Length; ++j) full[indices[j]] = x[j];
                    return full;
                }
                bool Strict(double[] local) => room.Contains(Expand(local));
                bool Survey(double[] local) => inSurveyDomain(Expand(local));
                var startLocal = indices.Select(i => start[i]).ToArray();
                int queryOffset = arm.evaluations;
                var settings = new ImageJacobianLm.TrustSettings
                {
                    Budget = budget, Iterations = iterations, Units = indices.Select(i => units[i % 4]).ToArray(),
                    PhysicalSteps = indices.Select(i => i % 4 == 3 ? .04 : arm.positionStep).ToArray(),
                    ResidualBlocks = signal.residualBlocks, IndependentLampDamping = independent,
                    ProbeAllowed = arm.probePolicy == "strict" ? (Func<double[], bool>)Strict : Survey,
                    CandidateAllowed = arm.probePolicy == "full_through" ? (Func<double[], bool>)Survey : Strict,
                    FinalAllowed = Strict,
                    NativeAcceptance = nativeGuard == null ? null : (Func<double[], double>)(x => NativeLoss(nativeGuard, Expand(x), null)),
                    Query = (q, phase, x, loss, axis, offset) =>
                    {
                        var rig = Expand(x);
                        queryLog.WriteLine(string.Join(",", arm.id, segment, queryOffset + q, phase, axis < 0 ? -1 : indices[axis], F(offset),
                            Flag(room.Contains(rig)), Flag(inSurveyDomain(rig)), F(loss), Series(rig), F(clock.Elapsed.TotalSeconds)));
                    },
                    Jacobian = j =>
                    {
                        var spectrum = ClassroomDiagnosis.GetSpectrum(j.Gram, indices.Length);
                        double cross = 0, total = 0;
                        for (int i = 0; i < indices.Length; ++i)
                        {
                            for (int k = 0; k < indices.Length; ++k)
                            {
                                double v = j.Gram[i, k] * j.Gram[i, k];
                                total += v;
                                if (i / 4 != k / 4) cross += v;
                            }
                            gramLog.WriteLine(string.Join(",", arm.id, segment, j.Iteration, indices[i],
                                Series(Enumerable.Range(0, indices.Length).Select(k => j.Gram[i, k]))));
                        }
                        jacobianLog.WriteLine(string.Join(",", arm.id, segment, j.Iteration, queryOffset + j.Query, LampList(lamps), spectrum.rank,
                            F(spectrum.condition), Series(spectrum.singular), Series(spectrum.columnNorms), Series(j.PhysicalSteps),
                            string.Join(";", j.Stencils), Series(j.Gradient), F(Math.Sqrt(cross)), F(total > 0 ? Math.Sqrt(cross / total) : 0),
                            j.Blocked, Series(Expand(j.Theta))));
                    },
                    Trial = t =>
                    {
                        trialLog.WriteLine(string.Join(",", arm.id, segment, t.Iteration, queryOffset + t.Query, Flag(t.Legal),
                            Flag(t.Accepted), t.Status, t.Shrinks, F(t.Predicted), F(t.Actual), F(t.Ratio), F(t.NativeActual),
                            Series(t.PositionRadius), Series(t.IntensityRadius), Series(t.Damping), Series(t.Step.Select((x, i) => x * units[indices[i] % 4])),
                            Series(t.PredictedByLamp), Series(t.ActualByLamp)));
                    }
                };
                configure?.Invoke(settings, Expand);
                var result = ImageJacobianLm.RunTrust(startLocal, signal.size, (x, phase, r) => signal.objective(Expand(x), r), settings);
                arm.evaluations += result.Evaluations; arm.objectiveQueries += result.ObjectiveQueries; arm.nativeChecks += result.NativeChecks;
                arm.jacobians += result.Jacobians; arm.accepted += result.Accepted; arm.rejected += result.Rejected; arm.illegalAccepted += result.IllegalAccepted;
                arm.blocked += result.Blocked; arm.zeroColumns += result.ZeroColumns; arm.stop = result.Stop;
                record?.Invoke(result);
                return (double[])Expand(result.Best).Clone();
            }

            internal void Finish()
            {
                Apply(room.Truth, true);
                using (var csv = Csv("trials.csv", "arm,scenario,method,objective,lamps,shadows,policy,requested_step_m,budget,evaluations,objective_queries,native_checks,jacobians,accepted,rejected,illegal_accepted,blocked,zero_columns,native_render_count,shadow_faces,surface_renders,ray_queries,loss_start,loss_final,native_start_nrmse,native_train_nrmse,high_train_nrmse,held_nrmse,max_position_cm,max_intensity_relative,legal,pass,seconds,stop"))
                    foreach (var a in arms)
                        csv.WriteLine(string.Join(",", a.id, a.scenario, a.method, a.objective, string.Join(";", a.lamps), Flag(a.shadows), a.probePolicy,
                            F(a.positionStep), a.limit, a.evaluations, a.objectiveQueries, a.nativeChecks, a.jacobians, a.accepted, a.rejected, a.illegalAccepted,
                            a.blocked, a.zeroColumns, a.nativeCameraRenders, a.shadowFaces, a.surfaceRenders, a.rayQueries, F(a.initialLoss), F(a.finalLoss),
                            F(Math.Sqrt(a.nativeInitial)), F(Math.Sqrt(a.nativeFinal)), F(a.trainHigh), F(a.held), F(a.positionCm.Max()), F(a.intensityRelative.Max()),
                            Flag(a.legal), Flag(a.pass), F(a.seconds), a.stop));
                WriteJson(directory, "summary.json", new E21StageSummary { stage = stage,
                    branch = Arg("-bbE20Branch", "blackbox"), commit = Arg("-bbE20BaseCommit", "22c00f9"),
                    sceneHash = E21Sha(scenePath), scene = room.Describe(), truth = room.Truth,
                    views = new[] { "train:0", "train:1", "train:2", "held:3", "held:4" },
                    nativeCameraRenders = low.RenderCount + high.RenderCount, surfaceRenders = surfaceRenders, shadowFaces = shadowFaces,
                    rayQueries = rayQueries, arms = arms.ToArray(), seconds = clock.Elapsed.TotalSeconds });
            }

            public void Dispose()
            {
                Apply(room.Truth, true);
                low.Dispose(); high.Dispose();
                queryLog.Dispose(); jacobianLog.Dispose(); trialLog.Dispose(); gramLog.Dispose();
            }
        }
    }
}

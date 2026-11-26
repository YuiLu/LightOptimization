using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private const int E19FitW = 448, E19FitH = 256, E19HighW = 896, E19HighH = 512, E19FitViews = 3;

        [Serializable] private sealed class E19LampResult
        {
            public int lamp;
            public double startPositionErrorCm, bestPositionErrorCm, startIntensityRelative, bestIntensityRelative;
            public float truthPower, startPower, bestPower;
        }

        [Serializable] private sealed class E19Arm
        {
            public string scene, condition, start, stop;
            public int seed, budget, queries, renders, jacobians, accepted, rejected, blocked, shortened;
            public double initialLoss, bestLoss, recheckLoss, wallSeconds;
            public double startNrmse448, bestNrmse448, startNrmse896, bestNrmse896, startHeld896, bestHeld896;
            public double meanPositionErrorCm, maxPositionErrorCm, meanIntensityRelative, maxIntensityRelative;
            public bool passed;
            public double[] startTheta, bestTheta;
            public E19LampResult[] lamps;
        }

        [Serializable] private sealed class E19Run
        {
            public string scene, task, error;
            public int lamps, dimensions, budget, seedBase;
            public double elapsed;
            public E19Arm[] arms;
        }

        private static void RunJointRecovery(string task, string output)
        {
            if (!Application.isBatchMode) throw new InvalidOperationException("E19 opens transient scenes and is batch-only.");
            if (!UnityEngine.Rendering.GraphicsSettings.lightsUseLinearIntensity)
                throw new InvalidOperationException("Enable linear light intensity for calibrated color coordinates.");
            string caseName = task.StartsWith("e19-", StringComparison.Ordinal) ? task.Substring(4) : task;
            bool complex = caseName.StartsWith("complex", StringComparison.Ordinal);
            bool smoke = caseName.EndsWith("smoke", StringComparison.Ordinal);
            string dir = Path.Combine(output, task.Replace('-', '_'));
            Directory.CreateDirectory(dir);
            int budget = int.Parse(Arg("-bbE19Budget", smoke ? "240" : "900"), Invariant);
            int seedBase = int.Parse(Arg("-bbE19Seed", "131"), Invariant);
            using var pipeline = new TestPipeline(classroom: complex);
            using ISensitivityScene scene = complex ? (ISensitivityScene)new ComplexRoomSensitivityScene() : new RoomSensitivityScene();
            new JointRecoveryRunner(dir, scene, task, budget, seedBase, smoke).Run();
        }

        private sealed class JointRecoveryRunner
        {
            private readonly string dir, task;
            private readonly ISensitivityScene scene;
            private readonly int budget, seedBase;
            private readonly bool smoke;
            private readonly NativeRenderBackend fit, high;
            private readonly E19Run run = new E19Run();
            private readonly List<E19Arm> arms = new List<E19Arm>();
            private readonly System.Diagnostics.Stopwatch watch = System.Diagnostics.Stopwatch.StartNew();
            private StreamWriter armCsv, queryCsv;
            private int queryCount;
            private double[] lastCandidate;

            internal JointRecoveryRunner(string dir, ISensitivityScene scene, string task, int budget, int seedBase, bool smoke)
            {
                this.dir = dir; this.scene = scene; this.task = task; this.budget = budget; this.seedBase = seedBase; this.smoke = smoke;
                fit = new NativeRenderBackend(scene.Camera, scene.Views.Take(E19FitViews).ToArray(), scene.Rotations.Take(E19FitViews).ToArray(), E19FitW, E19FitH, -1);
                high = new NativeRenderBackend(scene.Camera, scene.Views, scene.Rotations, E19HighW, E19HighH, -1);
                run.scene = scene.Name; run.task = task; run.lamps = scene.Count; run.dimensions = scene.Truth.Length;
                run.budget = budget; run.seedBase = seedBase;
            }

            internal void Run()
            {
                try
                {
                    armCsv = new StreamWriter(Path.Combine(dir, "arms.csv"));
                    armCsv.WriteLine("scene,condition,start,seed,queries,renders,jacobians,accepted,rejected,stop,initialLoss,bestLoss,recheckLoss,startNrmse448,bestNrmse448,startNrmse896,bestNrmse896,startHeld896,bestHeld896,meanPositionErrorCm,maxPositionErrorCm,meanIntensityRelative,maxIntensityRelative,wallSeconds,passed");
                    queryCsv = new StreamWriter(Path.Combine(dir, "queries.csv"));
                    queryCsv.WriteLine("condition,start,seed,query,phase,loss");

                    var cfg = scene.Describe(); cfg.task = task; cfg.budget = budget; cfg.seed = seedBase;
                    WriteJson(dir, "scene.json", cfg);

                    if (task.EndsWith("-audit", StringComparison.Ordinal))
                    {
                        Audit();
                        run.arms = Array.Empty<E19Arm>();
                        run.elapsed = watch.Elapsed.TotalSeconds;
                        WriteJson(dir, "summary.json", run);
                        return;
                    }

                    string[] conditions = smoke ? new[] { "on" } : new[] { "on", "off" };
                    int[] seeds = smoke ? new[] { seedBase } : new[] { seedBase, seedBase + 602 };
                    float[] scales = smoke ? new[] { 1f } : new[] { 1f, .45f };
                    foreach (var condition in conditions)
                    foreach (var seed in seeds)
                    foreach (var scale in scales)
                    {
                        var arm = OneArm(condition, seed, scale);
                        arms.Add(arm);
                        armCsv.WriteLine(string.Join(",", scene.Name, condition, arm.start, seed.ToString(Invariant),
                            arm.queries.ToString(Invariant), arm.renders.ToString(Invariant), arm.jacobians.ToString(Invariant),
                            arm.accepted.ToString(Invariant), arm.rejected.ToString(Invariant), arm.stop,
                            F(arm.initialLoss), F(arm.bestLoss), F(arm.recheckLoss), F(arm.startNrmse448), F(arm.bestNrmse448),
                            F(arm.startNrmse896), F(arm.bestNrmse896), F(arm.startHeld896), F(arm.bestHeld896),
                            F(arm.meanPositionErrorCm), F(arm.maxPositionErrorCm), F(arm.meanIntensityRelative), F(arm.maxIntensityRelative),
                            F(arm.wallSeconds), arm.passed ? "true" : "false"));
                        armCsv.Flush();
                    }

                    run.arms = arms.ToArray();
                    run.elapsed = watch.Elapsed.TotalSeconds;
                    WriteJson(dir, "summary.json", run);
                    Debug.Log($"[E19] {scene.Name} {task}: finished in {run.elapsed:F1}s ({arms.Count} arms, {arms.Sum(a => a.queries)} queries)");
                }
                catch (Exception ex)
                {
                    run.error = ex.GetType().Name + ": " + ex.Message;
                    try { WriteJson(dir, "summary.json", run); } catch { }
                    throw;
                }
                finally { armCsv?.Dispose(); queryCsv?.Dispose(); fit.Dispose(); high.Dispose(); }
            }

            private void SetCondition(string condition)
            {
                scene.ResetShadows();
                for (int i = 0; i < scene.Count; ++i) scene.SetShadows(i, condition == "on");
            }

            private NativeRenderBackend.Frame[] CaptureSet(NativeRenderBackend backend, int views)
            {
                var frames = new NativeRenderBackend.Frame[views];
                for (int v = 0; v < views; ++v) frames[v] = backend.Capture(v);
                return frames;
            }

            private static double MeanViewNrmse(NativeRenderBackend.Frame[] rendered, NativeRenderBackend.Frame[] target, int views)
            {
                double sum = 0;
                for (int v = 0; v < views; ++v) sum += NativeRenderBackend.Compare(rendered[v], target[v]);
                return Math.Sqrt(sum / views);
            }

            private double[] ScaleStart(double[] far, float scale)
            {
                var truth = scene.Truth;
                var theta = new double[far.Length];
                for (int i = 0; i < theta.Length; ++i) theta[i] = Math.Max(0, Math.Min(1, truth[i] + scale * (far[i] - truth[i])));
                return scene.Contains(theta) ? theta : far;
            }

            private E19Arm OneArm(string condition, int seed, float scale)
            {
                var truth = scene.Truth;
                SetCondition(condition);
                scene.Apply(truth);
                var targetFit = CaptureSet(fit, E19FitViews);
                var targetHigh = CaptureSet(high, scene.Views.Length);
                NativeRenderBackend.Save(targetHigh[0], E19HighW, E19HighH, Path.Combine(dir, "gt-" + condition + "-v0-" + seed.ToString(Invariant)));

                var far = scene.Start(seed);
                var startTheta = scale >= 1f ? (double[])far.Clone() : ScaleStart(far, scale);
                string startLabel = scale >= 1f ? "far" : "near";
                Debug.Log($"[E19] {scene.Name} {condition} seed {seed} {startLabel}: start " + string.Join("/", Enumerable.Range(0, scene.Count)
                    .Select(i => ((scene.PositionOf(startTheta, i) - scene.PositionOf(truth, i)).magnitude * 100).ToString("F1", Invariant))) + " cm");

                var buffer = new float[E19FitW * E19FitH * 3 * E19FitViews];
                queryCount = 0;
                lastCandidate = null;
                double t0 = watch.Elapsed.TotalSeconds;
                double Evaluate(double[] theta, string phase, float[] res)
                {
                    if (!scene.Contains(theta)) throw new InvalidOperationException("Optimizer requested an illegal joint rig.");
                    scene.Apply(theta);
                    double loss = 0;
                    for (int v = 0; v < E19FitViews; ++v) loss += fit.Residual(v, targetFit[v], res, v * E19FitW * E19FitH * 3, E19FitViews);
                    ++queryCount;
                    queryCsv.WriteLine(string.Join(",", condition, startLabel, seed.ToString(Invariant), queryCount.ToString(Invariant), phase, F(loss)));
                    lastCandidate = (double[])theta.Clone();
                    return loss;
                }

                double startLoss = Evaluate(startTheta, "initial", buffer);
                startTheta = (double[])lastCandidate.Clone();
                var result = ImageJacobianLm.Run(startTheta, E19FitW * E19FitH * 3 * E19FitViews, Evaluate, budget, seed, scene.Contains, false, null);
                var bestTheta = (double[])result.Best.Clone();

                scene.Apply(startTheta);
                var startFit = CaptureSet(fit, E19FitViews);
                var startHigh = CaptureSet(high, scene.Views.Length);
                scene.Apply(bestTheta);
                var bestFit = CaptureSet(fit, E19FitViews);
                var bestHigh = CaptureSet(high, scene.Views.Length);
                double recheck = 0;
                for (int v = 0; v < E19FitViews; ++v) recheck += fit.Residual(v, targetFit[v], buffer, v * E19FitW * E19FitH * 3, E19FitViews);
                NativeRenderBackend.Save(bestHigh[0], E19HighW, E19HighH, Path.Combine(dir, "best-" + condition + "-" + seed.ToString(Invariant) + "-" + startLabel + "-v0"));

                var lamps = new E19LampResult[scene.Count];
                for (int i = 0; i < lamps.Length; ++i)
                {
                    var tp = scene.PositionOf(truth, i);
                    lamps[i] = new E19LampResult
                    {
                        lamp = i + 1,
                        truthPower = (float)scene.PowerOf(truth, i),
                        startPower = (float)scene.PowerOf(startTheta, i),
                        bestPower = (float)scene.PowerOf(bestTheta, i),
                        startPositionErrorCm = (scene.PositionOf(startTheta, i) - tp).magnitude * 100,
                        bestPositionErrorCm = (scene.PositionOf(bestTheta, i) - tp).magnitude * 100,
                        startIntensityRelative = Math.Abs(scene.PowerOf(startTheta, i) - scene.PowerOf(truth, i)) / scene.PowerOf(truth, i),
                        bestIntensityRelative = Math.Abs(scene.PowerOf(bestTheta, i) - scene.PowerOf(truth, i)) / scene.PowerOf(truth, i)
                    };
                }

                var arm = new E19Arm
                {
                    scene = scene.Name, condition = condition, start = startLabel, seed = seed, budget = budget,
                    queries = result.Queries, renders = result.Queries * E19FitViews, jacobians = result.Jacobians,
                    accepted = result.Accepted, rejected = result.Rejected, blocked = result.BlockedProbes, shortened = result.ShortenedTrials,
                    stop = result.StopReason, initialLoss = startLoss, bestLoss = result.BestLoss, recheckLoss = recheck,
                    startNrmse448 = MeanViewNrmse(startFit, targetFit, E19FitViews),
                    bestNrmse448 = MeanViewNrmse(bestFit, targetFit, E19FitViews),
                    startNrmse896 = MeanViewNrmse(startHigh.Take(E19FitViews).ToArray(), targetHigh.Take(E19FitViews).ToArray(), E19FitViews),
                    bestNrmse896 = MeanViewNrmse(bestHigh.Take(E19FitViews).ToArray(), targetHigh.Take(E19FitViews).ToArray(), E19FitViews),
                    startHeld896 = MeanViewNrmse(startHigh.Skip(E19FitViews).ToArray(), targetHigh.Skip(E19FitViews).ToArray(), scene.Views.Length - E19FitViews),
                    bestHeld896 = MeanViewNrmse(bestHigh.Skip(E19FitViews).ToArray(), targetHigh.Skip(E19FitViews).ToArray(), scene.Views.Length - E19FitViews),
                    meanPositionErrorCm = lamps.Average(l => l.bestPositionErrorCm), maxPositionErrorCm = lamps.Max(l => l.bestPositionErrorCm),
                    meanIntensityRelative = lamps.Average(l => l.bestIntensityRelative), maxIntensityRelative = lamps.Max(l => l.bestIntensityRelative),
                    passed = lamps.All(l => l.bestPositionErrorCm <= 5 && l.bestIntensityRelative <= .05),
                    startTheta = startTheta, bestTheta = bestTheta, lamps = lamps, wallSeconds = watch.Elapsed.TotalSeconds - t0
                };
                Debug.Log($"[E19] {scene.Name} {condition} {startLabel} seed {seed}: NRMSE {arm.startNrmse448:P1}->{arm.bestNrmse448:P1} (896 {arm.bestNrmse896:P2}, held {arm.bestHeld896:P2}), pos mean {arm.meanPositionErrorCm:F2} cm max {arm.maxPositionErrorCm:F2} cm, I mean {arm.meanIntensityRelative:P1} max {arm.maxIntensityRelative:P1}, {arm.stop}, {arm.queries}q, {arm.wallSeconds:F1}s, {(arm.passed ? "PASS" : "fail")}");
                return arm;
            }
            private void Audit()
            {
                SetCondition("on");
                scene.Apply(scene.Truth);
                var all = CaptureSet(fit, E19FitViews);
                var total = all.Select(f => f.Energy).ToArray();
                var lights = Enumerable.Range(1, scene.Count).Select(i => GameObject.Find($"Candidate Light {i:00}")?.GetComponent<Light>()).ToArray();
                if (lights.Any(l => !l)) throw new InvalidOperationException("Audit needs Candidate Light 01..0N in the open scene.");
                foreach (var l in lights) l.enabled = true;
                using (var share = new StreamWriter(Path.Combine(dir, "lamp-share.csv")))
                {
                    share.WriteLine("lamp,view0,view1,view2,mean");
                    for (int i = 0; i < scene.Count; ++i)
                    {
                        for (int j = 0; j < lights.Length; ++j) lights[j].enabled = j == i;
                        var only = CaptureSet(fit, E19FitViews);
                        var shares = Enumerable.Range(0, E19FitViews).Select(v => only[v].Energy / Math.Max(1e-12, total[v])).ToArray();
                        share.WriteLine(string.Join(",", "L" + (i + 1), string.Join(",", shares.Select(F)), F(shares.Average())));
                        Debug.Log($"[E19] audit {scene.Name} L{i + 1} energy share " + string.Join(" / ", shares.Select(s => s.ToString("P2", Invariant))));
                    }
                    for (int j = 0; j < lights.Length; ++j) lights[j].enabled = true;
                }
                using (var shadow = new StreamWriter(Path.Combine(dir, "shadow-difference.csv")))
                {
                    shadow.WriteLine("lamp,view0,view1,view2,mean");
                    for (int i = 0; i < scene.Count; ++i)
                    {
                        lights[i].shadows = LightShadows.None;
                        var off = CaptureSet(fit, E19FitViews);
                        lights[i].shadows = LightShadows.Hard;
                        var delta = Enumerable.Range(0, E19FitViews).Select(v => Math.Sqrt(NativeRenderBackend.Compare(off[v], all[v]))).ToArray();
                        shadow.WriteLine(string.Join(",", "L" + (i + 1), string.Join(",", delta.Select(F)), F(delta.Average())));
                        Debug.Log($"[E19] audit {scene.Name} L{i + 1} shadow-diff NRMSE " + string.Join(" / ", delta.Select(s => s.ToString("P2", Invariant))));
                    }
                }
                NativeRenderBackend.Save(all[0], E19FitW, E19FitH, Path.Combine(dir, "audit-gt-v0"));
            }
        }
    }
}




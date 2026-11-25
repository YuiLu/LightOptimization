using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private const string LbfgsName = "CentralLbfgs", LmName = "ImageJacobianLm";
        private static readonly int[] NewSeeds = { 131, 157, 181, 211, 241 };

        [Serializable] private sealed class NativeLamp
        {
            public string type;
            public Vector3 position, direction;
            public Color color;
            public float intensity, range, innerSpotAngle, spotAngle;
            public static NativeLamp Read(Light l) => new NativeLamp
            {
                type = l.type.ToString(), position = l.transform.position, direction = l.transform.forward,
                intensity = l.intensity, color = l.color, range = l.range,
                innerSpotAngle = l.type == LightType.Spot ? l.innerSpotAngle : 0, spotAngle = l.type == LightType.Spot ? l.spotAngle : 0
            };
        }

        [Serializable] private sealed class NextTrial
        {
            public string stage, cohort, method, strategy, stopReason, lightTypes;
            public int seed, dimensions, lightCount, fitResolution = FitSize, validationResolution = CheckSize;
            public int cameraBudget, trainingCameraRenders, sharedCameraRenders, validationCameraRenders, screenedRenders, localRenders;
            public int acceptedSteps, rejectedSteps, jacobians, completedLocalSolves;
            public double initialLoss, bestLoss, recapturedLoss, initialNrmse, bestNrmse, highInitialNrmse, highBestNrmse, improvement;
            public double initializationWallSeconds, trainingWallSeconds, validationWallSeconds, wallSeconds, renderSubmitSeconds, readbackSeconds, damping, pilotH;
            public bool passed;
            public double[] targetTheta, initialTheta, bestTheta;
            public NativeLamp[] targetRig, initialRig, bestRig;
        }

        [Serializable] private sealed class NextStage
        {
            public string stage, shader;
            public ViewInfo camera;
            public double wallSeconds, repeatLoss, repeatLoss512, sharedWallSeconds;
            public int totalCameraRenders, sharedCameraRenders;
            public NextTrial[] trials;
        }

        [Serializable] private sealed class Decisions
        {
            public string preferred;
            public int lbfgsPasses, lmPasses;
            public double lbfgsMedian, lmMedian, lbfgsWorst, lmWorst;
        }

        private sealed class LocalResult
        {
            public double[] Best;
            public string Stop;
            public double InitialLoss, Damping, H;
            public int Accepted, Rejected, Jacobians;
        }

        // All screen, restart and optimizer queries pass through this one ledger.
        private sealed class NextObjective
        {
            public readonly NativeRenderBackend Backend;
            public readonly QueryLog Log;
            public readonly Action<double[]> Apply;
            public readonly NativeRenderBackend.Frame Target;
            public readonly int Budget;
            public double BestLoss = double.PositiveInfinity;
            public double[] Best;
            public int Remaining => Budget - Log.Cameras;

            public NextObjective(NativeRenderBackend backend, QueryLog log, Action<double[]> apply, NativeRenderBackend.Frame target, int budget)
            { Backend = backend; Log = log; Apply = apply; Target = target; Budget = budget; }

            public double Evaluate(double[] theta, string phase, float[] residual = null)
            {
                if (Remaining < 1 || theta.Any(u => !(u >= 0 && u <= 1))) throw new InvalidOperationException("Invalid or over-budget native query.");
                double f = Log.Measure(Backend, phase, theta, () =>
                {
                    Apply(theta);
                    return residual == null ? Backend.Loss(0, Target) : Backend.Residual(0, Target, residual);
                });
                if (!(f >= 0) || double.IsInfinity(f)) throw new InvalidOperationException("Non-finite native RGB loss.");
                if (f < BestLoss) { BestLoss = f; Best = (double[])theta.Clone(); }
                return f;
            }

            public LocalResult Solve(string name, double[] start, int cap, int seed, string prefix = "")
            {
                cap = Math.Min(cap, Remaining);
                if (name == LmName)
                {
                    var s = ImageJacobianLm.Run(start, Target.PixelCount * 3,
                        (t, phase, residual) => Evaluate(t, prefix + phase, residual), cap, seed);
                    return new LocalResult { Best = s.Best, InitialLoss = s.InitialLoss, Stop = s.StopReason, H = s.PilotH, Damping = s.Damping,
                        Accepted = s.Accepted, Rejected = s.Rejected, Jacobians = s.Jacobians };
                }
                if (name != LbfgsName) throw new ArgumentException("Unrecognized local optimizer.");
                var result = BlackBoxOptimizer.Run(BlackBoxOptimizer.Method.CentralLbfgs, start,
                    (t, phase) => Evaluate(t, prefix + phase), cap, seed);
                return new LocalResult { Best = result.Best, InitialLoss = result.InitialLoss, Stop = result.StopReason, H = result.PilotH };
            }
        }

        private static void RunNext(string task, string output, Camera camera, ViewInfo view, GameObject room, Light[] lights,
            RecoveryLightState[] gt, BoxCollider block)
        {
            SelfTestLm();
            var colliders = new[] { block, GameObject.Find("Cornell Short Block")?.GetComponent<BoxCollider>() };
            if (!colliders[1]) throw new InvalidOperationException("Cornell short block collider is missing.");
            bool Free(Vector3 p) => colliders.All(c => Vector3.SqrMagnitude(c.ClosestPoint(p) - p) > .000001f);
            bool FreeRig(RecoveryRigCoordinates coords, double[] theta) => coords.Decode(theta).All(s => Free(s.position));
            void Only(int n) { for (int i = 0; i < lights.Length; ++i) { gt[i].Apply(lights[i]); lights[i].enabled = i < n; } }

            string method = null;
            if (task == "e4")
            {
                Only(1);
                var e1 = Points("e4_e1_lit", "URP/Lit", 1, 500, false);
                Only(3);
                var lit = Points("e4_e2_lit", "URP/Lit", 3, 2000, false);
                NextStage stylized;
                using (new StylizedMaterials(room)) stylized = Points("e4_e2_stylized", "Stylized Point Response", 3, 2000, false);
                var rows = new[] { e1, lit, stylized }.SelectMany(s => s.trials).ToArray();
                var a = rows.Where(t => t.method == LbfgsName).ToArray();
                var b = rows.Where(t => t.method == LmName).ToArray();
                var choice = new Decisions { lbfgsPasses = a.Count(t => t.passed), lmPasses = b.Count(t => t.passed),
                    lbfgsMedian = Median(a.Select(t => t.highBestNrmse)), lmMedian = Median(b.Select(t => t.highBestNrmse)),
                    lbfgsWorst = a.Max(t => t.highBestNrmse), lmWorst = b.Max(t => t.highBestNrmse) };
                // Prefer more passes unless the tail degrades >20%; otherwise require a meaningful tail gain at equal passes.
                choice.preferred = (choice.lmPasses > choice.lbfgsPasses && choice.lmWorst <= 1.2 * choice.lbfgsWorst ||
                    choice.lmPasses == choice.lbfgsPasses && choice.lmWorst <= .8 * choice.lbfgsWorst && choice.lmMedian <= 1.2 * choice.lbfgsMedian) ? LmName : LbfgsName;
                WriteJson(output, "e4_decision.json", choice);
                return;
            }

            var decisionPath = Path.Combine(output, "e4_decision.json");
            method = Arg("-bbLocal", File.Exists(decisionPath) ? JsonUtility.FromJson<Decisions>(File.ReadAllText(decisionPath)).preferred : "");
            if (method != LbfgsName && method != LmName) throw new InvalidOperationException("Run E4 first in this output directory or specify -bbLocal CentralLbfgs|ImageJacobianLm.");
            if (task == "e5")
            {
                Only(3);
                Seed47Diagnostic(camera, view, lights, gt, block, output, Free);
                Points("e5a_multistart", "URP/Lit", 3, 2000, true);
                Only(1);
                // Different positions AND strengths. The scanner never receives these fixture values.
                var targets = new[] { new[] { .23, .64, .30, .48 }, new[] { .76, .72, .31, .63 }, new[] { .61, .51, .77, .55 },
                    new[] { .29, .40, .73, .70 }, new[] { .71, .84, .58, .44 } };
                var one = lights.Take(1).ToArray();
                var coords = new RecoveryRigCoordinates(one, block, false);
                for (int i = 0; i < targets.Length; ++i)
                {
                    int seed = NewSeeds[i];
                    double[] truth = targets[i];
                    if (!FreeRig(coords, truth)) throw new InvalidOperationException("Cold-start fixture is not in free space.");
                    double[] Start(int s) => CandidateSet(1, 4, s, true, t => FreeRig(coords, t))[0];
                    RunNextStage(output, "e5b_cold_gt" + (i + 1), "URP/Lit", camera, view, one, coords.Apply, truth,
                        () => coords.Apply(truth), new[] { seed }, new[] { method }, new[] { "random", "screen48" }, 500,
                        _ => "cold_start", Start, s => CandidateSet(48, 4, s, true, t => FreeRig(coords, t)));
                }
                return;
            }
            Only(1);
            RunSpot(output, camera, view, lights, block, method);

            NextStage Points(string stage, string shader, int n, int budget, bool initialize)
            {
                var subset = lights.Take(n).ToArray();
                var goal = gt.Take(n).ToArray();
                var coords = new RecoveryRigCoordinates(subset, block, false);
                int[] seeds = initialize ? Seeds.Concat(NewSeeds).ToArray() : Seeds;
                float Distance(int s)
                {
                    int ix = Array.IndexOf(Seeds, s);
                    return ix >= 0 ? Distances[ix] : new[] { .33f, .44f, .51f, .63f, .72f }[Array.IndexOf(NewSeeds, s)];
                }
                return RunNextStage(output, stage, shader, camera, view, subset, coords.Apply, coords.Encode(goal),
                    () => { for (int j = 0; j < n; ++j) goal[j].Apply(subset[j]); }, seeds,
                    initialize ? new[] { method } : new[] { LbfgsName, LmName }, initialize ? new[] { "single", "multistart" } : new[] { "single" },
                    budget, s => Seeds.Contains(s) ? "original" : "held_out_starts", s => coords.Encode(coords.Perturb(goal, s, Distance(s)), true),
                    initialize ? (Func<int, double[][]>)(s => CandidateSet(48, 4 * n, s + 8123, false, t => FreeRig(coords, t))) : null);
            }
        }

        private static double Median(IEnumerable<double> items)
        {
            var a = items.OrderBy(x => x).ToArray();
            return (a[(a.Length - 1) / 2] + a[a.Length / 2]) / 2;
        }

        // Stratify normalized coordinates (not a claim of uniform physical-volume sampling).
        // Fixed cold-start intensity is the midpoint of the public log-intensity bounds, independent of GT.
        private static double[][] CandidateSet(int count, int dims, int seed, bool fixedPower, Func<double[], bool> admissible)
        {
            var random = new System.Random(seed);
            var result = new List<double[]>(count);
            for (int batch = 0; batch < 8 && result.Count < count; ++batch)
            {
                int n = count - result.Count;
                var z = Enumerable.Range(0, n).Select(i => new double[dims]).ToArray();
                for (int k = 0; k < dims; ++k)
                {
                    var strata = Enumerable.Range(0, n).ToArray();
                    for (int i = n - 1; i > 0; --i) { int j = random.Next(i + 1); (strata[i], strata[j]) = (strata[j], strata[i]); }
                    for (int i = 0; i < n; ++i) z[i][k] = fixedPower && k % 4 == 3 ? .5 : (strata[i] + random.NextDouble()) / n;
                }
                result.AddRange(z.Where(admissible));
            }
            if (result.Count != count) throw new InvalidOperationException("Could not sample the prescribed number of free-space starts.");
            return result.ToArray();
        }

        private static NextStage RunNextStage(string root, string stage, string shader, Camera camera, ViewInfo view,
            Light[] lights, Action<double[]> apply, double[] truth, Action installGt, int[] seeds, string[] methods,
            string[] strategies, int budget, Func<int, string> cohort, Func<int, double[]> initial, Func<int, double[][]> candidates = null)
        {
            var watch = Stopwatch.StartNew();
            string dir = Path.Combine(root, stage);
            var rows = new List<NextTrial>();
            var report = new NextStage { stage = stage, shader = shader, camera = view };
            using (var train = Backend(camera, new[] { view }, FitSize))
            using (var check = Backend(camera, new[] { view }, CheckSize))
            using (var shared = new QueryLog(dir, "shared_queries.csv", "shared", 0))
            {
                installGt();
                var targetRig = lights.Select(NativeLamp.Read).ToArray();
                var target = CaptureReference(train, "256", truth);
                var targetHigh = CaptureReference(check, "512", truth);
                report.repeatLoss = shared.Measure(train, "repeat", truth, () => train.Loss(0, target));
                report.repeatLoss512 = shared.Measure(check, "repeat_512", truth, () => check.Loss(0, targetHigh));
                if (Math.Max(report.repeatLoss, report.repeatLoss512) > 1e-10) throw new InvalidOperationException("Native Cornell target is not repeatable.");
                apply(truth);
                double encoded = shared.Measure(train, "coordinate_roundtrip", truth, () => train.Loss(0, target));
                if (encoded > 1e-5) throw new InvalidOperationException("GT coordinates do not reproduce the nominal target.");
                report.sharedCameraRenders = shared.Cameras;
                report.sharedWallSeconds = shared.WallSeconds;
                foreach (int seed in seeds)
                {
                    foreach (var method in methods)
                        foreach (var strategy in strategies)
                        {
                            var trialWatch = Stopwatch.StartNew();
                            // The generator is deterministic in seed. Charge preparation only to the policy using it.
                            var start = initial(seed);
                            var screening = strategy == "multistart" || strategy == "screen48";
                            var starts = screening ? candidates(seed) : null;
                            double preparation = trialWatch.Elapsed.TotalSeconds;
                            string trialDir = Path.Combine(dir, $"{seed}_{method}_{strategy}");
                            using (var log = new QueryLog(trialDir, "queries.csv", method, seed))
                            {
                                var objective = new NextObjective(train, log, apply, target, budget);
                                apply(start);
                                var row = new NextTrial { stage = stage, cohort = cohort(seed), method = method, strategy = strategy,
                                    lightTypes = string.Join("+", targetRig.Select(t => t.type)), lightCount = lights.Length, seed = seed,
                                    dimensions = start.Length, cameraBudget = budget, targetTheta = truth, initialTheta = start, targetRig = targetRig,
                                    initialRig = lights.Select(NativeLamp.Read).ToArray(), sharedCameraRenders = shared.Cameras, initializationWallSeconds = preparation };
                                // All methods start from the original/random anchor; use its first real measured loss for eta.
                                LocalResult last;
                                if (!screening)
                                {
                                    last = objective.Solve(method, start, budget, seed);
                                    row.initialLoss = last.InitialLoss;
                                }
                                else
                                {
                                    row.initialLoss = objective.Evaluate(start, "screen_anchor");
                                    var scores = starts.Select(x => (point: x, loss: objective.Evaluate(x, "screen_candidate"))).ToList();
                                    row.screenedRenders = log.Cameras;
                                    if (strategy == "multistart")
                                    {
                                        // Freeze globally before hold-out: refine original + best two non-GT starts, then spend the rest on the best actual query.
                                        var shortlist = scores.OrderBy(s => s.loss).Take(2).Select(s => s.point).Prepend(start).ToArray();
                                        for (int i = 0; i < shortlist.Length; ++i)
                                            Record(objective.Solve(method, shortlist[i], 300, seed, "short" + i + "_"));
                                    }
                                    last = objective.Solve(method, objective.Best, objective.Remaining, seed, "refine_");
                                }
                                Record(last);
                                row.pilotH = last.H; row.damping = last.Damping; row.stopReason = last.Stop;
                                row.bestTheta = objective.Best; row.bestLoss = objective.BestLoss;
                                row.initialNrmse = Math.Sqrt(row.initialLoss); row.bestNrmse = Math.Sqrt(row.bestLoss);
                                row.improvement = 1 - row.bestLoss / Math.Max(1e-20, row.initialLoss);
                                row.passed = row.improvement >= .9 && row.bestNrmse <= .05;
                                row.trainingCameraRenders = log.Cameras;
                                row.localRenders = row.trainingCameraRenders - row.screenedRenders;
                                row.trainingWallSeconds = trialWatch.Elapsed.TotalSeconds; row.renderSubmitSeconds = log.RenderSeconds; row.readbackSeconds = log.ReadbackSeconds;
                                var verification = Stopwatch.StartNew();
                                row.recapturedLoss = SaveVerification(train, target, row.bestTheta, "best_256");
                                row.highInitialNrmse = Math.Sqrt(SaveVerification(check, targetHigh, start, "initial_512"));
                                row.highBestNrmse = Math.Sqrt(SaveVerification(check, targetHigh, row.bestTheta, "best_512"));
                                row.bestRig = lights.Select(NativeLamp.Read).ToArray();
                                if (Math.Abs(row.recapturedLoss - row.bestLoss) > Math.Max(1e-10, row.bestLoss * 1e-6))
                                    throw new InvalidOperationException("Best queried loss was not reproduced.");
                                row.validationCameraRenders = log.Cameras - row.trainingCameraRenders;
                                row.validationWallSeconds = verification.Elapsed.TotalSeconds;
                                row.wallSeconds = trialWatch.Elapsed.TotalSeconds;
                                WriteJson(trialDir, "report.json", row);
                                rows.Add(row);
                                report.trials = rows.ToArray();
                                report.totalCameraRenders = train.RenderCount + check.RenderCount;
                                report.wallSeconds = watch.Elapsed.TotalSeconds;
                                WriteJson(dir, "report.json", report);
                                Debug.Log($"{stage} {seed} {method}/{strategy}: {row.initialNrmse:P2} -> {row.bestNrmse:P2} ({row.highBestNrmse:P2} at 512), {row.trainingCameraRenders} renders/{row.trainingWallSeconds:F2}s; pass={row.passed}");

                                void Record(LocalResult r)
                                {
                                    ++row.completedLocalSolves; row.acceptedSteps += r.Accepted; row.rejectedSteps += r.Rejected; row.jacobians += r.Jacobians;
                                }
                                double SaveVerification(NativeRenderBackend backend, NativeRenderBackend.Frame reference, double[] theta, string name)
                                {
                                    NativeRenderBackend.Frame frame = null;
                                    double loss = log.Measure(backend, "validate_" + name, theta, () =>
                                    { apply(theta); frame = backend.Capture(0); return NativeRenderBackend.Compare(frame, reference); });
                                    NativeRenderBackend.Save(frame, backend.Size, Path.Combine(trialDir, name));
                                    return loss;
                                }
                            }
                        }
                }
                NativeRenderBackend.Frame CaptureReference(NativeRenderBackend backend, string label, double[] theta)
                {
                    NativeRenderBackend.Frame frame = null;
                    shared.Measure(backend, "warmup_" + label, theta, () => { backend.Capture(0); return double.NaN; });
                    shared.Measure(backend, "target_" + label, theta, () => { frame = backend.Capture(0); return double.NaN; });
                    NativeRenderBackend.Save(frame, backend.Size, Path.Combine(dir, "target_" + label));
                    return frame;
                }
            }
            return report;
        }

        private static void SelfTestLm()
        {
            double[] goal = { .31, .75, 1.2, -.3 }, initial = { 1, 0, .5, .5 };
            int calls = 0;
            double observed = double.PositiveInfinity;
            var r = ImageJacobianLm.Run(initial, goal.Length, (x, _, residual) =>
            {
                ++calls;
                if (x.Any(v => !(v >= 0 && v <= 1))) throw new InvalidOperationException("Infeasible LM probe.");
                double loss = 0;
                for (int i = 0; i < goal.Length; ++i) { residual[i] = (float)((x[i] - goal[i]) * (i + 1)); loss += (double)residual[i] * residual[i]; }
                observed = Math.Min(observed, loss);
                return loss;
            }, 180, 17);
            if (calls != r.Queries || calls > 180 || r.BestLoss != observed || r.Accepted == 0 ||
                Math.Abs(r.Best[0] - goal[0]) > .001 || Math.Abs(r.Best[1] - goal[1]) > .001 || r.Best[2] != 1 || r.Best[3] != 0)
                throw new InvalidOperationException("Bounded LM self-test failed: " + r.BestLoss);
        }
    }
}

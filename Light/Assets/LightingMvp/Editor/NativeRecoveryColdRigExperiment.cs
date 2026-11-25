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
        private const double PassNrmse = .05, RequiredReduction = .90;
        private const int InsertionBudget = 1024, InsertLocalCap = 120;
        private const int MultiStartCount = 48, RestartCap = 300, VerificationCount = 3;
        private static readonly string[] ColdMethods = { "A", "B", "C", "D" };

        [Serializable] private sealed class ColdExperiment
        {
            public int budget, trainingResolution = FitSize, recaptureResolution = CheckSize, targets, seedsPerTarget;
            public int trainingRenders, sharedRenders, recaptureRenders;
            public double wallSeconds;
            public string screening;
            public ViewInfo trainingCamera, heldOutCamera;
            public ColdTarget[] cases;
        }

        [Serializable] private sealed class ColdTarget
        {
            public string target;
            public double[] targetTheta;
            public NativeLamp[] targetRig;
            public int sharedRenders;
            public double trainRepeatLoss, highRepeatLoss, holdRepeatLoss;
            public ColdTrial[] trials;
        }

        [Serializable] private sealed class ColdTrial
        {
            public string target, method, stopReason, commonStop, screening;
            public int seed, budget, trainingRenders, validationRenders, firstPassQuery = -1, firstFivePercentQuery = -1;
            public int insertionRenders, fullRigQueryStart, commonRenders, swapRenders, blockedProbes, shortenedTrials;
            public double wallSeconds, trainingSeconds, validationSeconds, renderSubmitSeconds, readbackSeconds;
            public double emptyLoss, initialLoss, generatedLoss, loss, repeatLoss, reduction, trainNrmse, initialHighNrmse, highNrmse, holdNrmse, commonLoss;
            public bool emptyBelowFivePercent, passesFivePercent, passed, structureFailure;
            public double[] initialTheta, generatedTheta, bestTheta;
            public NativeLamp[] initialRig, generatedRig, bestRig;
            public InsertionTrial[] insertions;
            public SwapTrial[] swaps;
        }

        [Serializable] private sealed class InsertionTrial
        {
            public int lightsBefore, renders, candidate = -1;
            public double baseLoss, insertedLoss, jointLoss, intensity;
            public bool accepted;
        }

        [Serializable] private sealed class SwapTrial
        {
            public int removedSlot, candidate = -1, renders, firstQuery, lastQuery;
            public double beforeLoss, removedLoss, proposalLoss, afterLoss, proposedDistance, finalDistance;
            public bool accepted;
            public double[] beforeTheta, afterTheta;
        }

        private sealed class ColdObjective
        {
            private readonly NativeRenderBackend backend;
            private readonly QueryLog log;
            private readonly StructureRig rig;
            internal readonly NativeRenderBackend.Frame Target;
            internal readonly ColdTrial Row;
            internal int Renders => log.Cameras;
            internal int Remaining => Row.budget - Renders;
            internal double[] Best;
            internal double BestLoss = double.PositiveInfinity;
            private bool trackFull;

            internal ColdObjective(NativeRenderBackend backend, QueryLog log, StructureRig rig, NativeRenderBackend.Frame target, ColdTrial row)
            { this.backend = backend; this.log = log; this.rig = rig; Target = target; Row = row; }

            internal void BeginFull()
            {
                if (trackFull) return;
                trackFull = true;
                Row.fullRigQueryStart = Renders + 1;
            }

            internal (NativeRenderBackend.Frame frame, double loss) Capture(double[] theta, string phase, bool compare = true)
            {
                Guard(theta);
                NativeRenderBackend.Frame frame = null;
                double loss = log.Measure(backend, phase, theta, () =>
                {
                    rig.Apply(theta);
                    frame = backend.Capture(0);
                    return compare ? NativeRenderBackend.Compare(frame, Target) : double.NaN;
                });
                if (compare) CheckLoss(loss);
                if (compare && trackFull && theta.Length == RigDofs) Observe(theta, loss, Renders);
                return (frame, loss);
            }

            internal double Evaluate(double[] theta, string phase, float[] residual = null)
            {
                Guard(theta);
                double loss = log.Measure(backend, phase, theta, () =>
                {
                    rig.Apply(theta);
                    return residual == null ? backend.Loss(0, Target) : backend.Residual(0, Target, residual);
                });
                CheckLoss(loss);
                if (trackFull && theta.Length == RigDofs) Observe(theta, loss, Renders);
                return loss;
            }

            internal ImageJacobianLm.Result Solve(double[] start, int cap, int seed, string phase)
            {
                int allowance = Math.Min(Remaining, cap);
                if (allowance < 1) throw new InvalidOperationException("No render allowance for LM.");
                var solve = ImageJacobianLm.Run(start, Target.PixelCount * 3,
                    (theta, p, buffer) => Evaluate(theta, phase + p, buffer), allowance, seed, rig.Domain.Contains);
                Row.blockedProbes += solve.BlockedProbes;
                Row.shortenedTrials += solve.ShortenedTrials;
                return solve;
            }

            private void Observe(double[] theta, double loss, int at)
            {
                if (loss < BestLoss) { BestLoss = loss; Best = (double[])theta.Clone(); }
                if (loss <= PassNrmse * PassNrmse)
                {
                    if (Row.firstFivePercentQuery < 0 || at < Row.firstFivePercentQuery) Row.firstFivePercentQuery = at;
                    if (!Row.emptyBelowFivePercent && Reduction(Row.emptyLoss, loss) >= RequiredReduction &&
                        (Row.firstPassQuery < 0 || at < Row.firstPassQuery)) Row.firstPassQuery = at;
                }
            }

            private static void CheckLoss(double loss)
            {
                if (!(loss >= 0) || double.IsInfinity(loss)) throw new InvalidOperationException("Non-finite native RGB loss.");
            }

            private void Guard(double[] theta)
            {
                if (Remaining < 1 || !rig.Domain.Contains(theta)) throw new InvalidOperationException("Invalid or over-budget E8 camera query.");
            }
        }

        private sealed class Pick
        {
            internal double[] Theta;
            internal int Index;
            internal double Power, Loss;
        }

        private static double Reduction(double empty, double loss) => 1 - loss / Math.Max(1e-20, empty);
        private static bool MaterialGain(double before, double after) => before - after > Math.Max(1e-10, 1e-5 * before);
        private static int ScreeningCost(bool olat) => 1 + VerificationCount + (olat ? 0 : PoolSize * PowerGrid.Length);

        private static Pick SelectAddition(ColdObjective o, double[] context, double[][] pool, Response[] bases,
            string label, out double baseLoss)
        {
            bool olat = bases != null;
            var current = o.Capture(context, label + "base");
            baseLoss = current.loss;
            if (context.Length == RigDofs - LampDofs) o.BeginFull();
            var shortlist = new List<(int index, double power, double score)>(pool.Length);
            if (olat)
            {
                var residual = Difference(o.Target, current.frame);
                for (int i = 0; i < pool.Length; ++i)
                {
                    var a = bases[i];
                    if (a.Energy <= NearSignal * NearSignal) continue;
                    double dot = RgbDot(a.Pixels, residual);
                    double w = Math.Max(PowerMin / Q0, Math.Min(PowerMax / Q0, dot / a.Energy));
                    double power = w * Q0;
                    double gain = PredictedGain(dot, a.Energy, power);
                    if (gain > 0) shortlist.Add((i, power, gain));
                }
            }
            else
            {
                for (int i = 0; i < pool.Length; ++i)
                {
                    double power = 0, loss = double.PositiveInfinity;
                    foreach (double q in PowerGrid)
                    {
                        double f = o.Evaluate(Join(context, WithPower(pool[i], q)), label + $"grid_{i}_{F(q)}");
                        if (f < loss) { power = q; loss = f; }
                    }
                    shortlist.Add((i, power, baseLoss - loss));
                }
            }
            var tested = shortlist.OrderByDescending(p => p.score).Take(VerificationCount).Select(p =>
            {
                var theta = Join(context, WithPower(pool[p.index], p.power));
                double loss = o.Evaluate(theta, label + $"verify_{p.index}");
                return new Pick { Theta = theta, Loss = loss, Index = p.index, Power = p.power };
            }).ToArray();
            return tested.OrderBy(p => p.Loss).FirstOrDefault();
        }

        private static double[] RemoveLamp(double[] theta, int index) => theta.Take(index * LampDofs)
            .Concat(theta.Skip((index + 1) * LampDofs)).ToArray();

        private static double[] RandomFullRig(FreeCornell domain, int seed)
        {
            var random = new System.Random(seed);
            var theta = new List<double>(RigDofs);
            while (theta.Count < RigDofs)
            {
                var lamp = Enumerable.Range(0, LampDofs).Select(i => random.NextDouble()).ToArray();
                if (domain.Contains(lamp)) theta.AddRange(lamp);
            }
            return theta.ToArray();
        }

        // Only the target factory and evaluation code use these authored positions.
        private static (string name, double[] theta)[] ColdTargets() => new[] {
            ("spread", Join(Join(Lamp(.68f, 1.55f, .67f, 1.85), Lamp(-.65f, 1.44f, -.62f, 1.3)), Lamp(.03f, .85f, -.77f, .85))),
            ("clustered", Join(Join(Lamp(-.65f, 1.63f, -.33f, 1.12), Lamp(-.30f, 1.72f, -.06f, 1.41)), Lamp(-.54f, 1.38f, -.11f, .92))),
            ("unbalanced", Join(Join(Lamp(.60f, 1.57f, -.54f, 4.8), Lamp(-.64f, 1.62f, .56f, .38)), Lamp(-.72f, .77f, -.55f, .16))),
            ("occluded", Join(Join(Lamp(-.59f, .49f, .79f, 1.65), Lamp(-.72f, .53f, -.33f, 1.07)), Lamp(.42f, .79f, -.41f, 1.20))),
            ("boundary", Join(Join(Lamp(.75f, .42f, .22f, .73), Lamp(-.78f, 1.81f, .83f, 1.90)), Lamp(.15f, 1.12f, .58f, 1.42)))
        };

        private static void RunColdRigs(string root, Camera camera, ViewInfo[] views, StructureRig rig, bool olat, int budget, bool smoke)
        {
            string output = Path.Combine(root, (smoke ? "e8_smoke_" : "e8_") + budget);
            Directory.CreateDirectory(output);
            var clock = Stopwatch.StartNew();
            var cases = ColdTargets();
            if (cases.Any(t => !rig.Domain.Contains(t.theta))) throw new InvalidOperationException("An E8 GT is outside the public domain.");
            var seeds = smoke ? StructureSeeds.Take(1) : StructureSeeds.AsEnumerable();
            var targetRows = new List<ColdTarget>();
            using (var training = Backend(camera, new[] { views[0] }, FitSize))
            using (var high = Backend(camera, views, CheckSize))
            {
                foreach (var target in smoke ? cases.Take(1) : cases.AsEnumerable())
                {
                    string dir = Path.Combine(output, target.name);
                    using (var log = new QueryLog(dir, "shared_queries.csv", "gt", 0))
                    {
                        var row = new ColdTarget { target = target.name, targetTheta = target.theta, targetRig = rig.Describe(target.theta) };
                        var trainTarget = Truth(training, 0, "train", out row.trainRepeatLoss);
                        var highTarget = Truth(high, 0, "high", out row.highRepeatLoss);
                        var holdTarget = Truth(high, 1, "hold", out row.holdRepeatLoss);
                        row.sharedRenders = log.Cameras;
                        if (new[] { row.trainRepeatLoss, row.highRepeatLoss, row.holdRepeatLoss }.Any(f => f > 1e-10))
                            throw new InvalidOperationException("E8 GT is not repeatable across capture requests.");
                        var trialRows = new List<ColdTrial>();
                        foreach (int seed in seeds)
                            foreach (string method in ColdMethods)
                            {
                                string trialDir = Path.Combine(dir, $"{seed}_{method}");
                                var trial = RunTrial(trialDir, rig, training, high, trainTarget, highTarget, holdTarget, target.name, seed, method, budget, olat);
                                trialRows.Add(trial);
                                Debug.Log($"E8 {target.name}/{seed}/{method}: {100 * trial.trainNrmse:F3}% train, {100 * trial.highNrmse:F3}% high, {100 * trial.holdNrmse:F3}% held-out, {trial.trainingRenders} queries, {trial.stopReason}");
                                // Keep reference checkpoints after every trial so a stopped batch is diagnosable.
                                row.trials = trialRows.ToArray();
                                WriteJson(dir, "target.json", row);
                            }
                        targetRows.Add(row);

                        NativeRenderBackend.Frame Truth(NativeRenderBackend b, int view, string label, out double repeat)
                        {
                            Capture(b, log, rig, target.theta, label + "_warmup", view);
                            var frame = Capture(b, log, rig, target.theta, label + "_gt", view);
                            var check = Capture(b, log, rig, target.theta, label + "_repeat", view, frame);
                            repeat = NativeRenderBackend.Compare(frame, check);
                            NativeRenderBackend.Save(frame, b.Width, Path.Combine(dir, label + "_gt"));
                            return frame;
                        }
                    }
                }
                var trials = targetRows.SelectMany(t => t.trials).ToArray();
                int trainCount = trials.Sum(t => t.trainingRenders), validation = trials.Sum(t => t.validationRenders),
                    shared = targetRows.Sum(t => t.sharedRenders);
                if (training.RenderCount + high.RenderCount != trainCount + validation + shared)
                    throw new InvalidOperationException("E8 camera render ledger is incomplete.");
                WriteJson(output, "e8.json", new ColdExperiment { budget = budget, trainingCamera = views[0], heldOutCamera = views[1],
                    targets = targetRows.Count, seedsPerTarget = seeds.Count(), cases = targetRows.ToArray(), screening = olat ? "OLAT" : "native_grid",
                    trainingRenders = trainCount, sharedRenders = shared, recaptureRenders = validation, wallSeconds = clock.Elapsed.TotalSeconds });
                Debug.Log($"E8 complete: {trials.Length} trials, {trainCount} training queries; {output}");
            }
        }

        private static ColdTrial RunTrial(string dir, StructureRig rig, NativeRenderBackend training, NativeRenderBackend high,
            NativeRenderBackend.Frame trainTarget, NativeRenderBackend.Frame highTarget, NativeRenderBackend.Frame holdTarget,
            string targetName, int seed, string method, int budget, bool olat)
        {
            var clock = Stopwatch.StartNew();
            var row = new ColdTrial { target = targetName, seed = seed, method = method, budget = budget,
                screening = method == "C" || method == "D" ? (olat ? "OLAT" : "native_grid") : "none" };
            using (var log = new QueryLog(dir, "queries.csv", method, seed))
            {
                var o = new ColdObjective(training, log, rig, trainTarget, row);
                var empty = Array.Empty<double>();
                var baseline = o.Capture(empty, "empty");
                row.emptyLoss = baseline.loss;
                row.emptyBelowFivePercent = baseline.loss <= PassNrmse * PassNrmse;
                var inserts = new List<InsertionTrial>();
                var swaps = new List<SwapTrial>();
                if (method == "A" || method == "B")
                {
                    row.initialTheta = RandomFullRig(rig.Domain, seed);
                    o.BeginFull();
                    row.initialLoss = o.Evaluate(row.initialTheta, "random_anchor");
                    if (method == "B")
                    {
                        // E5 allocation: the original anchor and two best screened alternatives get 300 queries each.
                        var pool = CandidateSet(MultiStartCount, RigDofs, seed + 9011, false, rig.Domain.Contains);
                        var starts = pool.Select((p, index) => (p, loss: o.Evaluate(p, $"rig_screen_{index}")))
                            .OrderBy(t => t.loss).Take(2).Select(t => t.p).Prepend(row.initialTheta).ToArray();
                        for (int i = 0; i < starts.Length; ++i) o.Solve(starts[i], RestartCap, seed, $"short{i}_");
                    }
                    row.stopReason = o.Solve(o.Best, o.Remaining, seed, "final_").StopReason;
                    row.bestTheta = o.Best; row.loss = o.BestLoss;
                }
                else
                {
                    // Deterministically replay and charge the same prefix to both C and D (including OLAT if used).
                    row.initialTheta = empty; row.initialLoss = baseline.loss;
                    var pool = StructurePool(rig.Domain, PoolSize, seed + 18713);
                    var bases = olat ? pool.Select((p, index) => new Response(o.Capture(p, $"dict_{index}", false).frame, baseline.frame)).ToArray() : null;
                    var current = empty;
                    double loss = baseline.loss;
                    for (int count = 0; count < 3; ++count)
                    {
                        int first = o.Renders;
                        if (first + ScreeningCost(olat) + InsertLocalCap > InsertionBudget)
                            throw new InvalidOperationException("Insertion schedule cannot fit the frozen cap.");
                        string prefix = $"insert{count + 1}_";
                        var pick = SelectAddition(o, current, pool, bases, prefix, out double currentLoss);
                        var insertion = new InsertionTrial { lightsBefore = count, baseLoss = currentLoss,
                            candidate = pick?.Index ?? -1, insertedLoss = pick?.Loss ?? currentLoss,
                            jointLoss = currentLoss, intensity = pick?.Power ?? 0 };
                        if (pick != null && MaterialGain(currentLoss, pick.Loss))
                        {
                            var local = o.Solve(pick.Theta, InsertLocalCap, seed, prefix + "relax_");
                            current = local.Best; loss = local.BestLoss;
                            insertion.accepted = true; insertion.jointLoss = loss;
                        }
                        else { loss = currentLoss; row.structureFailure = true; }
                        insertion.renders = o.Renders - first;
                        inserts.Add(insertion);
                        if (row.structureFailure) break;
                    }
                    row.generatedTheta = current; row.generatedLoss = loss;
                    row.insertionRenders = o.Renders;
                    if (row.structureFailure)
                    {
                        row.stopReason = "no_effective_insertion";
                        row.firstPassQuery = row.firstFivePercentQuery = -1;
                        row.bestTheta = current; row.loss = loss;
                    }
                    else
                    {
                        int reserve = budget / 4;
                        int oneLeast = 2 * RigDofs + 1;
                        var common = o.Solve(o.Best, o.Remaining - reserve, seed, "common_");
                        row.commonRenders = o.Renders;
                        row.commonLoss = o.BestLoss;
                        row.commonStop = common.StopReason;
                        row.stopReason = common.StopReason;
                        // A refinement that stopped because it used up this phase is not evidence of stagnation.
                        if (method == "D" && common.StopReason != "budget" && common.StopReason != "near_zero_residual")
                        {
                            int swapStart = o.Renders;
                            int perSwapLocal = olat ? 100 : 54; // grid: two complete scans + refinements fit in 500 renders.
                            // Iterate known lamp slots from this snapshot. An accepted replacement is appended;
                            // keep identity labels to avoid visiting the same original lamp twice after reordering.
                            var slotIds = new List<int> { 0, 1, 2 };
                            for (int attempt = 0; attempt < 3 && reserve - (o.Renders - swapStart) >= ScreeningCost(olat) + perSwapLocal; ++attempt)
                            {
                                int slot = slotIds.IndexOf(attempt);
                                var before = o.Best;
                                int begin = o.Renders;
                                var swap = new SwapTrial { removedSlot = slot, beforeLoss = o.BestLoss, beforeTheta = before, firstQuery = begin + 1 };
                                var pick = SelectAddition(o, RemoveLamp(before, slot), pool, bases, $"swap{attempt + 1}_", out swap.removedLoss);
                                if (pick != null)
                                {
                                    swap.candidate = pick.Index; swap.proposalLoss = pick.Loss;
                                    swap.proposedDistance = Vector3.Distance(FreeCornell.Position(before, slot * LampDofs), FreeCornell.Position(pick.Theta, 2 * LampDofs));
                                    var rel = o.Solve(pick.Theta, perSwapLocal, seed, $"swap{attempt + 1}_relax_");
                                    swap.afterTheta = rel.Best; swap.afterLoss = rel.BestLoss;
                                    swap.finalDistance = Vector3.Distance(FreeCornell.Position(before, slot * LampDofs), FreeCornell.Position(rel.Best, 2 * LampDofs));
                                    swap.accepted = rel.BestLoss < swap.beforeLoss && rig.Domain.Contains(rel.Best);
                                    if (swap.accepted) { slotIds.RemoveAt(slot); slotIds.Add(3 + attempt); }
                                }
                                else { swap.proposalLoss = swap.removedLoss; swap.afterTheta = before; swap.afterLoss = swap.beforeLoss; }
                                if (o.BestLoss != (swap.accepted ? swap.afterLoss : swap.beforeLoss))
                                    throw new InvalidOperationException("Swap acceptance disagrees with the best verified full rig.");
                                swap.renders = o.Renders - begin; swap.lastQuery = o.Renders;
                                swaps.Add(swap);
                                if (o.BestLoss <= 1e-12) break;
                            }
                            row.swapRenders = o.Renders - swapStart;
                            if (row.swapRenders > reserve) throw new InvalidOperationException("Swap exceeded its reserved budget.");
                        }
                        if (o.Remaining >= oneLeast && o.BestLoss > 1e-12)
                            row.stopReason = o.Solve(o.Best, o.Remaining, seed, "finish_").StopReason;
                        row.bestTheta = o.Best; row.loss = o.BestLoss;
                    }
                }
                row.trainingRenders = o.Renders; row.trainingSeconds = clock.Elapsed.TotalSeconds;
                row.renderSubmitSeconds = log.RenderSeconds; row.readbackSeconds = log.ReadbackSeconds;
                row.reduction = Reduction(row.emptyLoss, row.loss); row.trainNrmse = Math.Sqrt(row.loss);
                row.passesFivePercent = !row.structureFailure && row.trainNrmse <= PassNrmse;
                row.passed = row.passesFivePercent && !row.emptyBelowFivePercent && row.reduction >= RequiredReduction;
                row.insertions = inserts.ToArray(); row.swaps = swaps.ToArray();
                row.initialRig = rig.Describe(row.initialTheta);
                row.generatedRig = rig.Describe(row.generatedTheta ?? row.initialTheta);
                row.bestRig = rig.Describe(row.bestTheta);
            }
            double startVal = clock.Elapsed.TotalSeconds;
            using (var validation = new QueryLog(dir, "validation.csv", method, seed))
            {
                row.repeatLoss = Save(training, 0, row.bestTheta, trainTarget, "train_best");
                if (Math.Abs(row.repeatLoss - row.loss) > 1e-9) throw new InvalidOperationException("E8 optimal query could not be recaptured.");
                row.initialHighNrmse = Math.Sqrt(Save(high, 0, row.initialTheta, highTarget, "high_initial"));
                if (row.generatedTheta != null) Save(high, 0, row.generatedTheta, highTarget, "high_after_insertions");
                row.highNrmse = Math.Sqrt(Save(high, 0, row.bestTheta, highTarget, "high_best"));
                row.holdNrmse = Math.Sqrt(Save(high, 1, row.bestTheta, holdTarget, "heldout_best"));
                for (int i = 0; i < row.swaps.Length; ++i)
                {
                    var swap = row.swaps[i];
                    Save(high, 0, swap.beforeTheta, highTarget, $"swap{i + 1}_before");
                    Save(high, 0, swap.afterTheta, highTarget, $"swap{i + 1}_proposal");
                }
                row.validationRenders = validation.Cameras;

                double Save(NativeRenderBackend backend, int view, double[] theta, NativeRenderBackend.Frame target, string name)
                {
                    var captured = Capture(backend, validation, rig, theta, name, view, target);
                    NativeRenderBackend.Save(captured, backend.Width, Path.Combine(dir, name));
                    return NativeRenderBackend.Compare(captured, target);
                }
            }
            row.wallSeconds = clock.Elapsed.TotalSeconds; row.validationSeconds = row.wallSeconds - startVal;
            WriteJson(dir, "trial.json", row);
            return row;
        }
    }
}

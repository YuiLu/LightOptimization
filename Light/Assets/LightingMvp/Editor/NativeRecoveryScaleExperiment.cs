using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private static readonly double[] NoLights = Array.Empty<double>();
        private const string LegacyInsertion = "insertion", JointInsertion = "joint", JointSwap = "joint_swap";

        private sealed class ScaleRig
        {
            internal int Count;
            internal Action<double[]> Apply;
            internal Func<double[], bool> Contains;
            internal Func<double[], NativeLamp[]> Describe;
        }

        [Serializable] private sealed class ScaleStep
        {
            public string phase, reason;
            public int before, fromQuery, toQuery, candidate = -1, removedSlot = -1, positivePredictions;
            public double beforeLoss, verifiedLoss = -1, afterLoss, bestPredictedGain;
            public double[] relaxedWeights, weights, beforeTheta, afterTheta;
            public bool accepted;
        }

        [Serializable] private sealed class ScaleTrial
        {
            public string stage, target, method, stop, screening;
            public int seed, budget, pool, lamps, active, views, trainingRenders, validationRenders, firstPass = -1, bestQuery;
            public int jacobians, blockedProbes, shortenedTrials, swapRenders;
            public long lmBufferBytes, processWorkingSetBefore, processPeakWorkingSet, managedBytesBefore, managedBytesAfter;
            public double elapsedSeconds, trainingSeconds, renderSubmitSeconds, readbackSeconds, residualSeconds, jacobianSeconds, normalSeconds, linearSolveSeconds;
            public double emptyLoss, initialLoss, bestLoss, repeatedLoss, reduction, highNrmse, heldNrmse, highInitialNrmse, heldInitialNrmse;
            public double[] trainViewNrmse, highViewNrmse, heldViewNrmse, initialTheta, bestTheta;
            public NativeLamp[] initialRig, bestRig;
            public bool passed, structureFailure, emptyAlreadyPasses;
            public ScaleStep[] steps;
        }

        [Serializable] private sealed class ScaleRun
        {
            public string stage, unity, gpu, graphics, renderer, scene, recoveryProfile;
            public int trainingWidth, trainingHeight, validationWidth, validationHeight, sharedRenders, trainingRenders, validationRenders;
            public bool supportsOlat;
            public ViewInfo[] views;
            public ScaleTrial[] trials;
        }

        private sealed class ScaleObjective
        {
            private readonly NativeRenderBackend backend;
            private readonly QueryLog log;
            private readonly ScaleRig rig;
            internal readonly ScaleTrial Row;
            internal readonly NativeRenderBackend.Frame[] Target;
            internal readonly int[] Views;
            internal StreamWriter Screening;
            internal int Passes => Views.Length;
            internal int Renders => log.Cameras;
            internal int Remaining => Row.budget - Renders;
            internal int ImageSize => Target[0].PixelCount * 3;
            internal readonly double[] BestLoss;
            internal readonly int[] BestQuery;
            internal readonly double[][] Best;

            internal ScaleObjective(NativeRenderBackend backend, QueryLog log, ScaleRig rig,
                NativeRenderBackend.Frame[] targets, int[] views, ScaleTrial row)
            {
                this.backend = backend; this.log = log; this.rig = rig; Row = row;
                Views = views; Target = views.Select(i => targets[i]).ToArray();
                Best = new double[rig.Count + 1][];
                BestQuery = new int[rig.Count + 1];
                BestLoss = Enumerable.Repeat(double.PositiveInfinity, rig.Count + 1).ToArray();
            }

            internal (double loss, NativeRenderBackend.Frame[] frames) Capture(double[] theta, string phase, bool compare)
            {
                var frames = new NativeRenderBackend.Frame[Passes];
                double loss = Measure(theta, phase, () =>
                {
                    double sum = 0;
                    for (int i = 0; i < Passes; i++)
                    {
                        frames[i] = backend.Capture(Views[i]);
                        if (!compare) continue;
                        long t = Stopwatch.GetTimestamp();
                        sum += NativeRenderBackend.Compare(frames[i], Target[i]);
                        Row.residualSeconds += Since(t);
                    }
                    return compare ? sum / Passes : double.NaN;
                }, compare);
                return (loss, frames);
            }

            internal double Evaluate(double[] theta, string phase, float[] residual = null) => Measure(theta, phase, () =>
            {
                double sum = 0;
                for (int i = 0; i < Passes; i++)
                {
                    if (residual == null) sum += backend.Loss(Views[i], Target[i]) / Passes;
                    else sum += backend.Residual(Views[i], Target[i], residual, i * ImageSize, Passes);
                }
                return sum;
            }, true, true);

            private double Measure(double[] theta, string phase, Func<double> render, bool compare, bool inlineResidual = false)
            {
                if (Remaining < Passes || !rig.Contains(theta)) throw new InvalidOperationException("Illegal or over-budget scale query: " + phase);
                double loss = log.Measure(backend, phase, theta, () =>
                {
                    rig.Apply(theta);
                    long t = Stopwatch.GetTimestamp();
                    double before = backend.RenderSeconds + backend.ReadbackSeconds;
                    double result = render();
                    if (inlineResidual) Row.residualSeconds += Math.Max(0, Since(t) - (backend.RenderSeconds + backend.ReadbackSeconds - before));
                    return result;
                });
                if (compare)
                {
                    if (!(loss >= 0) || double.IsInfinity(loss)) throw new InvalidOperationException("Non-finite scale query.");
                    int n = theta.Length / LampDofs;
                    if (loss < BestLoss[n]) { BestLoss[n] = loss; Best[n] = (double[])theta.Clone(); BestQuery[n] = Renders; }
                    if (n == rig.Count && Row.firstPass < 0 && MeetsThreshold(Row.emptyLoss, loss)) Row.firstPass = Renders;
                }
                return loss;
            }

            internal float[] DifferenceNormalized(NativeRenderBackend.Frame[] a, NativeRenderBackend.Frame[] b)
            {
                var buffer = new float[ImageSize * Passes];
                for (int v = 0, k = 0; v < Passes; ++v)
                {
                    double factor = Passes == 1 ? 1 : 1 / Math.Sqrt(Passes * (Target[v].Energy + 1e-12));
                    for (int p = 0; p < a[v].PixelCount; ++p)
                    {
                        Color x = a[v].Pixel(p), y = b[v].Pixel(p);
                        buffer[k++] = (float)((x.r - y.r) * factor);
                        buffer[k++] = (float)((x.g - y.g) * factor);
                        buffer[k++] = (float)((x.b - y.b) * factor);
                    }
                }
                return buffer;
            }

            internal ImageJacobianLm.Result Solve(double[] from, int cap, int seed, string phase)
            {
                int units = Math.Min(cap, Remaining) / Passes;
                if (units < 1) throw new InvalidOperationException("No budget for scale LM.");
                Row.lmBufferBytes = Math.Max(Row.lmBufferBytes, sizeof(float) * (long)ImageSize * Passes * (from.Length + 3));
                var result = ImageJacobianLm.Run(from, ImageSize * Passes, (x, p, r) => Evaluate(x, phase + p, r),
                    units, seed, rig.Contains, profile: true);
                Row.jacobians += result.Jacobians; Row.blockedProbes += result.BlockedProbes; Row.shortenedTrials += result.ShortenedTrials;
                Row.jacobianSeconds += result.JacobianSeconds; Row.normalSeconds += result.NormalSeconds;
                Row.linearSolveSeconds += result.LinearSolveSeconds;
                return result;
            }
        }

        private static double Since(long ticks) => (Stopwatch.GetTimestamp() - ticks) / (double)Stopwatch.Frequency;
        private static bool MeetsThreshold(double emptyLoss, double loss) => emptyLoss > PassNrmse * PassNrmse &&
            loss <= PassNrmse * PassNrmse && Reduction(emptyLoss, loss) >= RequiredReduction;
        private static double[] SingleLamp(double[] rig, int lamp) => rig.Skip(lamp * LampDofs).Take(LampDofs).ToArray();

        private sealed class ScaleChoice
        {
            internal int Index;
            internal double Gain, Loss = double.PositiveInfinity;
            internal double[] Theta, Relaxed, Legal;
        }

        private static ScaleChoice ScaleSelect(ScaleObjective o, double[] context, double[][] pool, float[][] columns,
            NativeRenderBackend.Frame[] empty, bool joint, double multiviewEnergyScale, ScaleStep step, bool replacing = false)
        {
            int m = context.Length / LampDofs;
            var current = o.Capture(context, step.phase + "_base", true);
            var shortlist = new List<ScaleChoice>();
            if (columns == null)
            {
                for (int i = 0; i < pool.Length; i++)
                {
                    var best = new ScaleChoice { Index = i };
                    for (int g = 0; g < PowerGrid.Length; g++)
                    {
                        var x = Join(context, WithPower(pool[i], PowerGrid[g]));
                        double f = o.Evaluate(x, $"{step.phase}_grid_{i}_{g}");
                        if (f < best.Loss) { best.Theta = x; best.Loss = f; }
                    }
                    best.Gain = current.loss - best.Loss;
                    shortlist.Add(best);
                }
            }
            else if (!joint)
            {
                var residual = o.DifferenceNormalized(o.Target, current.frames);
                for (int i = 0; i < columns.Length; i++)
                {
                    var a = columns[i]; double energy = RgbDot(a, a);
                    if (energy <= NearSignal * NearSignal) continue;
                    double dot = RgbDot(a, residual);
                    double power = Math.Max(PowerMin, Math.Min(PowerMax, dot / energy));
                    double gain = PredictedGain(dot, energy, power) * multiviewEnergyScale;
                    if (gain > 0) shortlist.Add(new ScaleChoice { Index = i, Theta = Join(context, WithPower(pool[i], power)), Gain = gain, Legal = new[] { power } });
                    else WriteCandidate(i, gain, null, new[] { power }, double.NaN);
                }
            }
            else
            {
                var old = new float[m][];
                var target = o.DifferenceNormalized(o.Target, empty);
                var gram = new double[m + 1, m + 1];
                var rhs = new double[m + 1];
                var original = new double[m + 1];
                for (int j = 0; j < m; j++)
                {
                    var p = SingleLamp(context, j);
                    original[j] = PowerMin * Math.Exp(p[3] * LogPowerSpan);
                    old[j] = o.DifferenceNormalized(o.Capture(WithPower(p, Q0), $"{step.phase}_response_{j}", false).frames, empty);
                    rhs[j] = RgbDot(target, old[j]);
                    for (int k = 0; k <= j; k++) gram[j, k] = gram[k, j] = RgbDot(old[j], old[k]);
                }
                double powerLoss = RgbDot(target, target);
                for (int i = 0; i < pool.Length; i++)
                {
                    var a = columns[i]; gram[m, m] = RgbDot(a, a);
                    if (gram[m, m] <= NearSignal * NearSignal) continue;
                    rhs[m] = RgbDot(target, a);
                    for (int j = 0; j < m; j++) gram[m, j] = gram[j, m] = RgbDot(a, old[j]);
                    var relaxed = (double[])original.Clone();
                    double predicted = BoundedLightWeights.Solve(gram, rhs, m + 1, 0, PowerMax, relaxed);
                    double[] legal = relaxed;
                    if (relaxed.Any(w => w < PowerMin))
                    {
                        // Fixed-cardinality arm: force every slot to a legal active strength and re-evaluate the constrained optimum.
                        legal = (double[])relaxed.Clone();
                        predicted = BoundedLightWeights.Solve(gram, rhs, m + 1, PowerMin, PowerMax, legal);
                    }
                    double gain = current.loss - Math.Max(0, powerLoss - predicted) * multiviewEnergyScale;
                    if (gain <= 0) { WriteCandidate(i, gain, relaxed, legal, double.NaN); continue; }
                    var x = Join(context, pool[i]);
                    for (int j = 0; j <= m; j++) x[j * LampDofs + 3] = PowerCoordinate(legal[j]);
                    shortlist.Add(new ScaleChoice { Index = i, Theta = x, Gain = gain, Relaxed = relaxed, Legal = legal });
                }
            }
            step.positivePredictions = shortlist.Count(p => p.Gain > 0);
            step.bestPredictedGain = shortlist.Count == 0 ? 0 : shortlist.Max(p => p.Gain);
            var top = shortlist.OrderByDescending(p => p.Gain).Take(VerificationCount).ToArray();
            foreach (var candidate in top)
                candidate.Loss = o.Evaluate(candidate.Theta, $"{step.phase}_verify_{candidate.Index}");
            foreach (var candidate in shortlist)
                WriteCandidate(candidate.Index, candidate.Gain, candidate.Relaxed, candidate.Legal,
                    top.Contains(candidate) ? candidate.Loss : double.NaN);
            var pick = shortlist.OrderByDescending(p => p.Gain).Take(VerificationCount).OrderBy(p => p.Loss).FirstOrDefault();
            step.candidate = pick?.Index ?? -1; step.verifiedLoss = pick?.Loss ?? -1;
            step.relaxedWeights = pick?.Relaxed; step.weights = pick?.Legal;
            return pick != null && (replacing || MaterialGain(current.loss, pick.Loss)) ? pick : null;

            void WriteCandidate(int i, double gain, double[] relaxed, double[] legal, double measured) => o.Screening?.WriteLine(
                string.Join(",", step.phase, i, F(current.loss), F(gain), relaxed == null ? "" : string.Join(";", relaxed.Select(F)),
                    legal == null ? "" : string.Join(";", legal.Select(F)), double.IsNaN(measured) ? "" : F(measured),
                    string.Join(";", pool[i].Take(3).Select(F))));
        }

        private static double[] GenerateScale(ScaleObjective o, ScaleRig rig, double[] anchor, NativeRenderBackend.Frame[] empty,
            List<ScaleStep> steps, bool olat, bool e9)
        {
            var row = o.Row; int v = o.Passes;
            int fullCost = (2 * rig.Count * LampDofs + 1) * v;
            row.stop = "budget";
            if (row.method == "multi_start")
            {
                row.initialLoss = o.Evaluate(anchor, "random_anchor");
                int shortCap = Math.Min(3 * fullCost + 7 * v, row.budget / 9);
                int candidates = Math.Min(MultiStartCount, Math.Max(0, (o.Remaining - fullCost - 3 * shortCap) / v));
                var rest = CandidateSet(candidates, rig.Count * LampDofs, row.seed + 9011, false, rig.Contains, maxBatches: 32);
                var starts = rest.Select((p, i) => (p, f: o.Evaluate(p, $"rig_screen_{i}"))).OrderBy(p => p.f)
                    .Take(2).Select(p => p.p).Prepend(anchor).ToArray();
                for (int i = 0; i < starts.Length && shortCap >= v && o.Remaining > shortCap; i++) o.Solve(starts[i], shortCap, row.seed, $"restart{i}_");
                if (o.Remaining >= v) row.stop = o.Solve(o.Best[rig.Count], o.Remaining, row.seed, "finish_").StopReason;
                return o.Best[rig.Count];
            }
            bool joint = row.method != LegacyInsertion && olat, swapEnabled = row.method == JointSwap && olat;
            var pool = CandidateSet(row.pool, LampDofs, row.seed + 18713, true, rig.Contains).Select(p => WithPower(p, Q0)).ToArray();
            float[][] dictionary = null;
            if (olat && o.Remaining >= v * pool.Length)
                dictionary = pool.Select((p, i) => o.DifferenceNormalized(o.Capture(p, $"dict_{i}", false).frames, empty)).ToArray();
            if (olat && dictionary == null) throw new InvalidOperationException("OLAT dictionary cannot fit inside budget.");
            double dotScale = v == 1 ? 1 / (o.Target[0].Energy + 1e-12) : 1;
            int localCap = e9 ? InsertLocalCap : Math.Max(0, Math.Min(3 * fullCost, row.budget / (2 * rig.Count)));
            int reserve = row.budget / 4;
            int attempts = 0;
            double[] current = NoLights;
            double loss = row.emptyLoss;
            int ScreenCost(int n) => v * (1 + VerificationCount + (joint ? n : olat ? 0 : pool.Length * PowerGrid.Length));
            bool Swap()
            {
                int n = current.Length / LampDofs;
                int shortCap = Math.Max(v, (4 + 4 * n * LampDofs) * v);
                if (!swapEnabled || n < 2) return false;
                for (int slot = 0; slot < n && attempts < rig.Count; ++slot)
                {
                    int cap = ScreenCost(n - 1) + shortCap;
                    if (o.Remaining < cap || row.swapRenders + cap > reserve) return false;
                    var entry = new ScaleStep { phase = $"swap{++attempts}", before = n, fromQuery = o.Renders, beforeLoss = loss,
                        removedSlot = slot, beforeTheta = current };
                    var choice = ScaleSelect(o, RemoveLamp(current, slot), pool, dictionary, empty, true, dotScale, entry, replacing: true);
                    if (choice != null)
                    {
                        var lm = o.Solve(choice.Theta, shortCap, row.seed, entry.phase + "_relax_");
                        if (lm.BestLoss < loss) { current = lm.Best; loss = lm.BestLoss; entry.accepted = true; }
                    }
                    entry.reason = entry.accepted ? "native_gain" : "rollback"; entry.afterLoss = loss; entry.afterTheta = current;
                    entry.toQuery = o.Renders; row.swapRenders += entry.toQuery - entry.fromQuery; steps.Add(entry);
                    if (entry.accepted) return true;
                }
                return false;
            }
            int insertId = 0;
            while (current.Length < rig.Count * LampDofs)
            {
                int n = current.Length / LampDofs;
                // Preserve space for remaining grid scans in non-OLAT scenes rather than spending it on early LM.
                int scansLeft = Enumerable.Range(n + 1, rig.Count - n - 1).Sum(ScreenCost);
                var entry = new ScaleStep { phase = $"insert{++insertId}", before = n, fromQuery = o.Renders, beforeLoss = loss, beforeTheta = current };
                if (o.Remaining < ScreenCost(n)) { row.stop = "screen_budget"; break; }
                var choice = ScaleSelect(o, current, pool, dictionary, empty, joint, dotScale, entry);
                if (choice != null)
                {
                    current = choice.Theta; loss = choice.Loss; entry.accepted = true; entry.reason = "native_gain";
                    int cap = Math.Min(localCap, o.Remaining - scansLeft);
                    if (cap >= v)
                    {
                        var local = o.Solve(current, cap, row.seed, entry.phase + "_relax_");
                        current = local.Best; loss = local.BestLoss;
                    }
                }
                else entry.reason = entry.positivePredictions == 0 ? "no_predicted_gain" : "no_native_gain";
                entry.afterLoss = loss; entry.afterTheta = current; entry.toQuery = o.Renders; steps.Add(entry);
                if (!entry.accepted && !Swap()) { row.stop = entry.reason; break; }
            }
            if (current.Length == rig.Count * LampDofs)
            {
                // Match E8's two LM calls in Cornell; the final quarter is available to both arms.
                int commonCap = e9 || swapEnabled ? o.Remaining - reserve : o.Remaining;
                if (commonCap >= v)
                {
                    var common = o.Solve(o.Best[rig.Count], commonCap, row.seed, "common_");
                    current = o.Best[rig.Count]; loss = o.BestLoss[rig.Count]; row.stop = common.StopReason;
                    if (common.StopReason != "budget" && !MeetsThreshold(row.emptyLoss, loss)) Swap();
                }
                if (o.Remaining >= fullCost && o.BestLoss[rig.Count] > 1e-12)
                    row.stop = o.Solve(o.Best[rig.Count], o.Remaining, row.seed, "finish_").StopReason;
            }
            else row.structureFailure = true;
            return o.Best[current.Length / LampDofs];
        }

#if UNITY_EDITOR_WIN
        [StructLayout(LayoutKind.Sequential)]
        private struct WorkingSetCounters
        {
            public uint bytes, faults;
            public UIntPtr peak, current, quotaPeakPaged, quotaPaged, quotaPeakNonPaged, quotaNonPaged, pageFile, peakPageFile;
        }
        [DllImport("kernel32.dll")] private static extern IntPtr GetCurrentProcess();
        [DllImport("psapi.dll")] private static extern bool GetProcessMemoryInfo(IntPtr process, ref WorkingSetCounters counters, uint bytes);
#endif
        private static (long working, long peak) ReadScaleMemory()
        {
#if UNITY_EDITOR_WIN
            // Unity Mono on Windows can return 0 for Process.PeakWorkingSet64.
            var c = new WorkingSetCounters { bytes = (uint)Marshal.SizeOf<WorkingSetCounters>() };
            if (GetProcessMemoryInfo(GetCurrentProcess(), ref c, c.bytes)) return ((long)c.current.ToUInt64(), (long)c.peak.ToUInt64());
#endif
            using (var p = Process.GetCurrentProcess()) return (p.WorkingSet64, p.PeakWorkingSet64);
        }

        private static ScaleTrial RunScaleTrial(string dir, string stage, string target, ScaleRig rig,
            NativeRenderBackend train, NativeRenderBackend check, NativeRenderBackend.Frame[] gtTrain, NativeRenderBackend.Frame[] gtCheck,
            int[] views, int trainCandidates, string method, int seed, int budget, int pool, bool olat, double[] anchor = null)
        {
            var clock = Stopwatch.StartNew();
            var row = new ScaleTrial { stage = stage, target = target, method = method, seed = seed, budget = budget, pool = pool,
                lamps = rig.Count, views = views.Length, initialTheta = anchor ?? NoLights,
                screening = method == "known" || method == "multi_start" ? "none" : olat ? "olat" : "native_grid" };
            var steps = new List<ScaleStep>();
            row.processWorkingSetBefore = ReadScaleMemory().working;
            row.managedBytesBefore = GC.GetTotalMemory(false);
            using (var queries = new QueryLog(dir, "queries.csv", method, seed))
            {
                using var screening = row.screening == "none" ? null : new StreamWriter(Path.Combine(dir, "screening.csv"));
                screening?.WriteLine("phase,candidate,baseline_loss,gain,relaxed_weights,legal_weights,verified_loss,xyz_normalized");
                int start = train.RenderCount;
                var o = new ScaleObjective(train, queries, rig, gtTrain, views, row) { Screening = screening };
                var baseline = o.Capture(NoLights, "empty", true);
                row.emptyLoss = row.initialLoss = baseline.loss; row.emptyAlreadyPasses = baseline.loss <= PassNrmse * PassNrmse;
                if (method == "known")
                {
                    row.initialLoss = o.Evaluate(anchor, "anchor");
                    row.stop = o.Solve(anchor, o.Remaining, seed, "lm_").StopReason;
                    row.bestTheta = o.Best[rig.Count];
                }
                else row.bestTheta = GenerateScale(o, rig, anchor, baseline.frames, steps, olat, stage.StartsWith("e9", StringComparison.Ordinal));
                row.active = row.bestTheta.Length / LampDofs;
                row.bestLoss = o.BestLoss[row.active]; row.bestQuery = o.BestQuery[row.active];
                row.reduction = Reduction(row.emptyLoss, row.bestLoss);
                row.passed = !row.structureFailure && row.active == row.lamps && MeetsThreshold(row.emptyLoss, row.bestLoss);
                if (!row.passed) row.firstPass = -1;
                row.trainingRenders = queries.Cameras;
                row.renderSubmitSeconds = queries.RenderSeconds; row.readbackSeconds = queries.ReadbackSeconds;
                if (row.trainingRenders != train.RenderCount - start || row.trainingRenders > budget) throw new InvalidOperationException("Training ledger mismatch.");
            }
            row.trainingSeconds = clock.Elapsed.TotalSeconds;
            row.processPeakWorkingSet = ReadScaleMemory().peak;
            row.managedBytesAfter = GC.GetTotalMemory(false);
            row.steps = steps.ToArray(); row.initialRig = rig.Describe(row.initialTheta); row.bestRig = rig.Describe(row.bestTheta);
            using (var log = new QueryLog(dir, "recaptures.csv", method, seed))
            {
                row.trainViewNrmse = views.Select(i => Math.Sqrt(ViewCapture(train, log, rig, row.bestTheta, i, gtTrain[i], "train_best", null))).ToArray();
                row.repeatedLoss = row.trainViewNrmse.Average(x => x * x);
                if (Math.Abs(row.repeatedLoss - row.bestLoss) > 1e-9) throw new InvalidOperationException("Scale best query could not be recaptured.");
                var start = new double[gtCheck.Length]; var best = new double[gtCheck.Length];
                for (int v = 0; v < gtCheck.Length; ++v)
                {
                    start[v] = ViewCapture(check, log, rig, row.initialTheta, v, gtCheck[v], "check_initial",
                        row.method == "known" && (v == 0 || v == trainCandidates) ? Path.Combine(dir, $"initial_view{v}") : null);
                    best[v] = ViewCapture(check, log, rig, row.bestTheta, v, gtCheck[v], "check_best", Path.Combine(dir, $"best_view{v}"));
                }
                row.highNrmse = Math.Sqrt(views.Average(i => best[i])); row.highInitialNrmse = Math.Sqrt(views.Average(i => start[i]));
                row.heldNrmse = Math.Sqrt(best.Skip(trainCandidates).Average()); row.heldInitialNrmse = Math.Sqrt(start.Skip(trainCandidates).Average());
                row.highViewNrmse = best.Take(trainCandidates).Select(Math.Sqrt).ToArray(); row.heldViewNrmse = best.Skip(trainCandidates).Select(Math.Sqrt).ToArray();
                row.validationRenders = log.Cameras;
            }
            row.elapsedSeconds = clock.Elapsed.TotalSeconds;
            WriteJson(dir, "trial.json", row);
            Debug.Log($"{stage} {target}/{seed}/{method}/{views.Length}v: {row.active}/{row.lamps} lights, {Math.Sqrt(row.bestLoss):P4}, held {row.heldNrmse:P4}; {row.trainingRenders} queries/{row.trainingSeconds:F1}s, {row.stop}");
            return row;
        }

        private static double ViewCapture(NativeRenderBackend backend, QueryLog log, ScaleRig rig, double[] theta, int view,
            NativeRenderBackend.Frame target, string phase, string output, Action<NativeRenderBackend.Frame> receive = null)
        {
            NativeRenderBackend.Frame frame = null;
            double f = log.Measure(backend, phase + $"_v{view}", theta, () =>
            {
                rig.Apply(theta); frame = backend.Capture(view);
                return target == null ? double.NaN : NativeRenderBackend.Compare(frame, target);
            });
            if (output != null) NativeRenderBackend.Save(frame, backend.Width, backend.Height, output);
            receive?.Invoke(frame);
            return f;
        }

        private static ScaleRun MakeScaleRun(string stage, string scene, string renderer, bool olat, ViewInfo[] views,
            NativeRenderBackend train, NativeRenderBackend check) => new ScaleRun { stage = stage, scene = scene, renderer = renderer,
                unity = Application.unityVersion, gpu = SystemInfo.graphicsDeviceName, graphics = SystemInfo.graphicsDeviceType.ToString(),
                supportsOlat = olat, views = views, trainingWidth = train.Width, trainingHeight = train.Height,
                validationWidth = check.Width, validationHeight = check.Height };

        private static void FinishScaleRun(string dir, ScaleRun run, List<ScaleTrial> rows)
        {
            run.trials = rows.ToArray(); run.trainingRenders = rows.Sum(t => t.trainingRenders); run.validationRenders = rows.Sum(t => t.validationRenders);
            WriteJson(dir, "stage.json", run);
        }

        [Serializable] private sealed class FrozenProbe
        {
            public string target, method;
            public int seed, renders, budget;
            public ScaleStep result;
        }
        [Serializable] private sealed class FrozenProbeStage
        {
            public int sharedRenders, probeRenders;
            public FrozenProbe[] trials;
        }

        private static void RunFrozenE9Probe(string root, Camera camera, ViewInfo view, StructureRig cornell, bool olat)
        {
            BoundedLightWeights.CheckKnownCases();
            if (!olat) throw new InvalidOperationException("Frozen E9 probes require Cornell Lit OLAT gate.");
            string main = Path.Combine(root, "e9_2000"), dir = Path.Combine(root, "e9_frozen_probe");
            var run = JsonUtility.FromJson<ScaleRun>(File.ReadAllText(Path.Combine(main, "stage.json")));
            var failed = run.trials.Where(r => r.method == LegacyInsertion && r.pool == PoolSize && r.structureFailure).ToArray();
            var rig = new ScaleRig { Count = 3, Apply = cornell.Apply, Contains = cornell.Domain.Contains, Describe = cornell.Describe };
            var rows = new List<FrozenProbe>();
            int sharedRenders = 0;
            using (var backend = Backend(camera, new[] { view }, FitSize))
            {
                foreach (var input in failed)
                {
                    string targetDir = Path.Combine(dir, input.target, input.seed.ToString(Invariant));
                    var gt = JsonUtility.FromJson<ScaleTrial>(File.ReadAllText(Path.Combine(main, input.target, "ground_truth.json")));
                    var frame = new NativeRenderBackend.Frame[1];
                    using (var shared = new QueryLog(targetDir, "shared.csv", "shared", input.seed))
                    {
                        ViewCapture(backend, shared, rig, gt.bestTheta, 0, null, "warmup", null);
                        ViewCapture(backend, shared, rig, gt.bestTheta, 0, null, "target", null, f => frame[0] = f);
                        sharedRenders += shared.Cameras;
                    }
                    var stuck = input.steps.Last().beforeTheta;
                    foreach (string method in new[] { LegacyInsertion, JointInsertion })
                    {
                        string trialDir = Path.Combine(targetDir, method);
                        using var log = new QueryLog(trialDir, "queries.csv", method, input.seed);
                        using var screening = new StreamWriter(Path.Combine(trialDir, "screening.csv"));
                        screening.WriteLine("phase,candidate,baseline_loss,gain,relaxed_weights,legal_weights,verified_loss,xyz_normalized");
                        var o = new ScaleObjective(backend, log, rig, frame, new[] { 0 }, new ScaleTrial { budget = 2000 }) { Screening = screening };
                        var empty = o.Capture(NoLights, "empty", false).frames;
                        var pool = StructurePool(cornell.Domain, PoolSize, input.seed + 18713);
                        var dict = pool.Select((p, i) => o.DifferenceNormalized(o.Capture(p, $"dict_{i}", false).frames, empty)).ToArray();
                        var step = new ScaleStep { before = stuck.Length / LampDofs, beforeTheta = stuck, phase = "fixed_context",
                            fromQuery = o.Renders, beforeLoss = input.steps.Last().beforeLoss };
                        var choice = ScaleSelect(o, stuck, pool, dict, empty, method == JointInsertion, 1 / (frame[0].Energy + 1e-12), step);
                        step.accepted = choice != null;
                        step.afterLoss = choice?.Loss ?? step.beforeLoss;
                        step.afterTheta = choice?.Theta ?? stuck;
                        step.reason = step.accepted ? "native_gain" : step.positivePredictions > 0 ? "no_native_gain" : "no_predicted_gain";
                        step.toQuery = o.Renders;
                        rows.Add(new FrozenProbe { target = input.target, method = method, seed = input.seed, renders = o.Renders, budget = 2000, result = step });
                    }
                }
                if (backend.RenderCount != sharedRenders + rows.Sum(r => r.renders)) throw new InvalidOperationException("Frozen E9 ledger mismatch.");
            }
            WriteJson(dir, "probe.json", new FrozenProbeStage { sharedRenders = sharedRenders, probeRenders = rows.Sum(r => r.renders), trials = rows.ToArray() });
        }

        private static void RunJointCornell(string root, Camera camera, ViewInfo[] allViews, StructureRig cornell, bool olat, bool smoke)
        {
            BoundedLightWeights.CheckKnownCases();
            if (!olat) throw new InvalidOperationException("E9 requires the Cornell Lit OLAT gate.");
            var rig = new ScaleRig { Count = 3, Apply = cornell.Apply, Contains = cornell.Domain.Contains, Describe = cornell.Describe };
            var views = new[] { allViews[0], allViews[3] };
            int budget = int.Parse(Arg("-bbE9Budget", "2000"), Invariant);
            if (budget != 2000 && budget != 4000) throw new ArgumentException("E9 uses 2000 or 4000 paired camera renders.");
            string stage = smoke ? "e9_smoke" : "e9";
            string dir = Path.Combine(root, stage + "_" + budget);
            var targets = new[] {
                ("clustered", ColdTargets()[1].theta),
                ("tight_pair", Join(Join(Lamp(-.46f, 1.72f, -.27f, 2.4), Lamp(-.30f, 1.62f, -.17f, 1.55)), Lamp(.36f, 1.57f, .55f, .58))),
                ("weak_neighbor", Join(Join(Lamp(.48f, 1.67f, .25f, 5.7), Lamp(.35f, 1.52f, .40f, .35)), Lamp(-.53f, 1.59f, -.51f, .8)))
            };
            using (var training = Backend(camera, views.Take(1).ToArray(), FitSize))
            using (var high = Backend(camera, views, CheckSize))
            {
                var run = MakeScaleRun(stage, "Cornell Lit", "URP stock Forward", true, views, training, high);
                var rows = new List<ScaleTrial>();
                foreach (var (name, theta) in smoke ? targets.Take(1) : targets)
                {
                    var main = new NativeRenderBackend.Frame[1]; var checks = new NativeRenderBackend.Frame[2];
                    string targetDir = Path.Combine(dir, name);
                    using (var setup = new QueryLog(targetDir, "shared.csv", "shared", 0))
                    {
                        ViewCapture(training, setup, rig, theta, 0, null, "warmup", null);
                        ViewCapture(training, setup, rig, theta, 0, null, "target", null, f => main[0] = f);
                        for (int v = 0; v < 2; ++v)
                        {
                            int i = v;
                            ViewCapture(high, setup, rig, theta, i, null, "target", Path.Combine(targetDir, $"target_view{i}"), f => checks[i] = f);
                        }
                        run.sharedRenders += setup.Cameras;
                    }
                    WriteJson(targetDir, "ground_truth.json", new ScaleTrial { target = name, bestTheta = theta, bestRig = rig.Describe(theta) });
                    foreach (int seed in smoke ? new[] { 181 } : StructureSeeds)
                    {
                        var methods = new[] { LegacyInsertion, JointInsertion, JointSwap };
                        var comparison = methods.Select(m => Trial("main", seed, m, PoolSize)).ToArray();
                        rows.AddRange(comparison);
                        if (comparison.Any(r => !r.passed)) rows.AddRange(methods.Select(m => Trial("diagnostic256", seed, m, 256)));
                    }
                    ScaleTrial Trial(string part, int seed, string m, int pool) => RunScaleTrial(Path.Combine(targetDir, part, $"{seed}_{m}"),
                        stage + "_" + part, name, rig, training, high, main, checks, new[] { 0 }, 1, m, seed, budget, pool, true);
                }
                if (training.RenderCount + high.RenderCount != run.sharedRenders + rows.Sum(t => t.trainingRenders + t.validationRenders))
                    throw new InvalidOperationException("E9 stage ledger mismatch.");
                FinishScaleRun(dir, run, rows);
            }
        }
    }
}

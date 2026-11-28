using System;
using System.Linq;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace LightingMvp
{
    public static partial class ImageJacobianLm
    {
        public enum TrustNormalMode { Shared, LampDiagonal, LampScaledDense, ParameterScaledDense }

        public sealed class TrustSettings
        {
            public int Budget, Iterations = int.MaxValue;
            public int MaxConsecutiveCandidateRejects; // 0 preserves the default trust-region schedule.
            public double[] Units, PhysicalSteps;
            public int[] ResidualBlocks;
            public TrustLayout Layout;
            public TrustFallback Fallback;
            public int[] ViewBlocks;
            public bool FixedProbeScale;
            public bool ReserveRemainingModels; // Opt-in for predetermined short branch comparisons.
            // Optional physical, per-lamp position test. The original whole-rig check still runs.
            public Func<double[], int, bool> LampPositionAllowed;
            public bool IndependentLampDamping;
            public TrustNormalMode NormalMode;
            public double PositionRadius = .24, IntensityRadius = .28;
            public Func<double[], bool> ProbeAllowed, CandidateAllowed, FinalAllowed;
            public Func<double[], double> NativeAcceptance;
            public Action<double[], int, float[]> LogIntensityJacobian;
            public TrustPositionModel PositionModel;
            public Action<int, string, double[], double, int, double> Query;
            public Action<TrustJacobian> Jacobian;
            // Diagnostic observer receives borrowed columns, valid only during the callback.
            public Action<int, float[][]> AuditColumns;
            public Action<TrustTrial> Trial;
            public Func<TrustJacobian, bool> FinishAfterModel;
            public Func<TrustJacobian, float[][], float[], double[]> ProposalGradient;
            public int StallModelLimit;
            public Func<bool> StopAfterIteration;
            public bool BuildModelAtZero;
            public TrustTiming Timing;
            public bool TraceProposalClipping;
        }

        public sealed class TrustTiming
        {
            internal long normalTicks, solveTicks;
            public double GramGradientSeconds => (double)normalTicks / System.Diagnostics.Stopwatch.Frequency;
            public double DampedSolveSeconds => (double)solveTicks / System.Diagnostics.Stopwatch.Frequency;
        }

        /// <summary>Supplies XYZ columns on the solver's legal stencil. Costs are in objective evaluation equivalents.</summary>
        public sealed class TrustPositionModel
        {
            public readonly int AnchorCost, ProbeCost, IntensityProbeCost;
            internal readonly Action<int, double[]> Anchor;
            internal readonly Action<int, double[], int, double, int, float[]> Column;

            internal readonly Action<int, double[], int, double, int, float[]> IntensityColumn;

            public TrustPositionModel(int anchorCost, int probeCost, Action<int, double[]> anchor,
                Action<int, double[], int, double, int, float[]> column, int intensityProbeCost = 1,
                Action<int, double[], int, double, int, float[]> intensityColumn = null)
            {
                if (anchorCost < 0 || probeCost < 1 || intensityProbeCost < 1 || (intensityColumn == null && intensityProbeCost != 1)) throw new ArgumentException("Invalid position model costs.");
                AnchorCost = anchorCost;
                ProbeCost = probeCost;
                IntensityProbeCost = intensityProbeCost; IntensityColumn = intensityColumn;
                Anchor = anchor ?? throw new ArgumentNullException(nameof(anchor));
                Column = column ?? throw new ArgumentNullException(nameof(column));
            }
        }

        public sealed class TrustJacobian
        {
            public int Iteration, Query, Blocked;
            public double[] PhysicalSteps, Gradient, Theta;
            public int[] Stencils;
            public double[,] Gram;
        }

        public sealed class TrustTrial
        {
            public int Iteration, Query, Shrinks;
            public double Predicted, Actual, NativeActual, Ratio;
            public double[] Damping, PositionRadius, IntensityRadius, Step, PredictedByLamp, ActualByLamp;
            public double[] DampingScale, Theta, UnclippedStep, RadiusLimitedStep;
            public bool Accepted, Legal;
            public string Status, Source, Constraint;
            public int[] LampShrinks;
            public double CurrentLoss, ReferenceLoss, ReferenceRatio, BestLoss;
            public double UnclippedPrediction = double.NaN;
        }

        public sealed class TrustResult
        {
            public double[] Best;
            public double InitialLoss, BestLoss, LastLoss;
            public int Evaluations, ObjectiveQueries, NativeChecks, Jacobians, Accepted, Rejected, IllegalAccepted, Blocked, ZeroColumns;
            public string Stop = "budget";
        }

        /// <summary>Candidate and probe policies are separate. Only accepted, final-legal poses can become the return value.</summary>
        public static TrustResult RunTrust(double[] initial, int residualSize, ResidualQuery query, TrustSettings settings)
        {
            int n = initial.Length;
            var layout = settings.Layout ?? TrustLayout.Legacy(settings.Units, settings.PhysicalSteps);
            if (n != layout.Count || residualSize < 1 || settings.Budget < 2)
                throw new ArgumentException("Parameter layout, residual size and budget must agree.");
            var units = layout.Axes.Select(a => a.Unit).ToArray();
            int lamps = layout.Lamps;
            bool viewPool = settings.Fallback is TrustFallback.ViewPool or TrustFallback.ViewPairPool;
            if (settings.Fallback != TrustFallback.None && (settings.IndependentLampDamping || settings.NativeAcceptance != null))
                throw new ArgumentException("Recovery proposals require the single native objective.");
            if (viewPool && (settings.ViewBlocks == null || settings.ViewBlocks.Length != 3 ||
                settings.ViewBlocks.Any(v => v <= 0) || settings.ViewBlocks.Sum() != residualSize))
                throw new ArgumentException("The view pool requires three contiguous view blocks.");
            if (settings.ResidualBlocks != null && (settings.ResidualBlocks.Length != lamps || settings.ResidualBlocks.Any(x => x <= 0) ||
                settings.ResidualBlocks.Sum() != residualSize))
                throw new ArgumentException("Residual blocks must match the active lamps and residual size.");
            if (!Allowed(initial, settings.FinalAllowed)) throw new ArgumentException("Initial rig must be legal.");
            if (settings.IndependentLampDamping && settings.ResidualBlocks == null)
                throw new ArgumentException("Independent damping requires one independent residual block per lamp.");
            if (settings.IndependentLampDamping && settings.NormalMode != TrustNormalMode.Shared)
                throw new ArgumentException("Mixed RGB damping uses a global reduction test.");

            var result = new TrustResult { Best = (double[])initial.Clone() };
            var x = (double[])initial.Clone();
            var probe = new double[n];
            var center = new float[residualSize];
            var temporary = new float[residualSize];
            var trialResidual = new float[residualSize];
            var columns = Enumerable.Range(0, n).Select(_ => new float[residualSize]).ToArray();
            int groups = settings.IndependentLampDamping ? lamps : 1;
            var positionRadius = Enumerable.Repeat(settings.PositionRadius, groups).ToArray();
            var intensityRadius = Enumerable.Repeat(settings.IntensityRadius, groups).ToArray();
            var damping = new double[groups];
            var offsets = new int[lamps + 1];
            if (settings.ResidualBlocks != null)
                for (int i = 0; i < lamps; ++i) offsets[i + 1] = offsets[i] + settings.ResidualBlocks[i];
            double f = result.InitialLoss = result.BestLoss = Evaluate(x, "initial", center);
            double native = settings.NativeAcceptance == null ? f : CheckNative(x, "initial_native");
            double probeScale = 1;
            int consecutiveCandidateRejects = 0, stalledModels = 0;
            int checkCost = settings.NativeAcceptance == null ? 0 : 1;
            var history = new Queue<double>();
            history.Enqueue(f);
            int uphillInARow = 0, firstUphillModel = 0;
            int maxProbes = 2 * (n - (settings.LogIntensityJacobian == null ? 0 : layout.Axes.Count(a => a.IsIntensity)));
            var positionModel = settings.PositionModel;
            bool modelIntensity = positionModel?.IntensityColumn != null;
            if (modelIntensity && settings.LogIntensityJacobian != null)
                throw new ArgumentException("Only one intensity Jacobian provider may be used.");
            if (positionModel != null)
                maxProbes += positionModel.AnchorCost + 2 * layout.Axes.Count(a => a.IsPosition) * (positionModel.ProbeCost - 1) +
                    2 * layout.Axes.Count(a => a.IsIntensity) * (positionModel.IntensityProbeCost - 1);
            // Paired intensity captures are shared with the position model before building XYZ columns.
            var columnOrder = Enumerable.Range(0, n).ToArray();
            if (modelIntensity) columnOrder = columnOrder.OrderBy(j => !layout.Axes[j].IsIntensity).ToArray();

            while (result.Evaluations + maxProbes + 1 + checkCost <= settings.Budget && result.Jacobians < settings.Iterations)
            {
                if (f < 1e-12 && !settings.BuildModelAtZero) { result.Stop = "near_zero_residual"; break; }
                ++result.Jacobians;
                int blockedBefore = result.Blocked;
                var steps = new double[n];
                var stencils = new int[n];
                Array.Copy(x, probe, n);
                if (positionModel != null)
                {
                    positionModel.Anchor(result.Evaluations + 1, x);
                    result.Evaluations += positionModel.AnchorCost;
                }
                foreach (int j in columnOrder)
                {
                    var col = columns[j];
                    if (settings.LogIntensityJacobian != null && layout.Axes[j].IsIntensity)
                    {
                        settings.LogIntensityJacobian(x, layout.Owners[j], col);
                        stencils[j] = 4;
                        continue;
                    }
                    double h = layout.Axes[j].Step * probeScale / units[j];
                    for (int shrink = 0; shrink < 8; ++shrink, h *= .5)
                    {
                        probe[j] = x[j] + h; bool plus = CanProbe();
                        probe[j] = x[j] - h; bool minus = CanProbe();
                        if (!plus && !minus) continue;
                        bool central = plus && minus;
                        double d = plus ? h : -h;
                        bool second = false;
                        if (!central)
                        {
                            probe[j] = x[j] + 2 * d;
                            second = CanProbe();
                        }
                        bool intensity = layout.Axes[j].IsIntensity;
                        if (positionModel != null && (layout.Axes[j].IsPosition || (intensity && modelIntensity)))
                        {
                            int stencil = central ? 3 : second ? 2 : 1;
                            var build = intensity ? positionModel.IntensityColumn : positionModel.Column;
                            build(result.Evaluations + 1, x, j, d, stencil, col);
                            result.Evaluations += (stencil == 1 ? 1 : 2) * (intensity ? positionModel.IntensityProbeCost : positionModel.ProbeCost);
                            steps[j] = d * units[j]; stencils[j] = stencil;
                        }
                        else if (central)
                        {
                            probe[j] = x[j] + h; Evaluate(probe, "fd_plus", col, j, h);
                            probe[j] = x[j] - h; Evaluate(probe, "fd_minus", temporary, j, -h);
                            float scale = (float)(.5 / h);
                            for (int i = 0; i < residualSize; ++i) col[i] = (col[i] - temporary[i]) * scale;
                            steps[j] = h * units[j]; stencils[j] = 3;
                        }
                        else
                        {
                            probe[j] = x[j] + d; Evaluate(probe, "fd_one_sided", col, j, d);
                            if (second)
                            {
                                probe[j] = x[j] + 2 * d; Evaluate(probe, "fd_one_sided", temporary, j, 2 * d);
                                float scale = (float)(.5 / d);
                                for (int i = 0; i < residualSize; ++i) col[i] = (4 * col[i] - temporary[i] - 3 * center[i]) * scale;
                            }
                            else
                            {
                                float scale = (float)(1 / d);
                                for (int i = 0; i < residualSize; ++i) col[i] = (col[i] - center[i]) * scale;
                            }
                            steps[j] = d * units[j]; stencils[j] = second ? 2 : 1;
                        }
                        break;
                    }
                    if (stencils[j] == 0) Array.Clear(col, 0, col.Length);
                    probe[j] = x[j];
                }
                settings.AuditColumns?.Invoke(result.Jacobians, columns);
                long normalStart = settings.Timing == null ? 0 : System.Diagnostics.Stopwatch.GetTimestamp();
                var gradient = new double[n];
                var gram = new double[n, n];
                if (residualSize >= 300000 && n >= 8)
                    Parallel.For(0, n, new ParallelOptions { MaxDegreeOfParallelism = Math.Min(Environment.ProcessorCount, 8) }, NormalRow);
                else for (int j = 0; j < n; ++j) NormalRow(j);
                if (settings.Timing != null) settings.Timing.normalTicks += System.Diagnostics.Stopwatch.GetTimestamp() - normalStart;
                int zeros = 0;
                for (int j = 0; j < n; ++j) if (!(gram[j, j] > 1e-18)) ++zeros;
                result.ZeroColumns += zeros;
                TrustJacobian snapshot = null;
                if (settings.Jacobian != null || settings.FinishAfterModel != null || settings.ProposalGradient != null)
                    snapshot = new TrustJacobian { Iteration = result.Jacobians, Query = result.Evaluations, Theta = x,
                        PhysicalSteps = steps, Stencils = stencils, Gram = gram, Gradient = gradient, Blocked = result.Blocked - blockedBefore };
                settings.Jacobian?.Invoke(snapshot);
                if (settings.FinishAfterModel?.Invoke(snapshot) == true) { result.Stop = "audit_model"; break; }
                if (zeros == n) { result.Stop = "zero_jacobian"; break; }
                var directionGradient = settings.ProposalGradient?.Invoke(snapshot, columns, center) ?? gradient;
                var model = gram;
                double[] dampingScale = null;
                if (settings.NormalMode == TrustNormalMode.LampDiagonal)
                {
                    model = new double[n, n];
                    for (int i = 0; i < n; ++i)
                    for (int j = layout.Starts[layout.Owners[i]]; j < layout.Starts[layout.Owners[i]] + layout.Widths[layout.Owners[i]]; ++j)
                        model[i, j] = gram[i, j];
                }
                else if (settings.NormalMode == TrustNormalMode.LampScaledDense)
                    dampingScale = LampDampingScale(gram, layout);
                else if (settings.NormalMode == TrustNormalMode.ParameterScaledDense)
                {
                    var diag = Enumerable.Range(0, n).Select(i => gram[i, i]).ToArray();
                    double floor = Math.Max(1e-10, 1e-6 * diag.Max());
                    dampingScale = diag.Select(x => Math.Max(floor, x)).ToArray();
                }
                for (int group = 0; group < groups; ++group)
                {
                    double maxDiagonal = 0;
                    for (int j = groups == 1 ? 0 : layout.Starts[group]; j < (groups == 1 ? n : layout.Starts[group] + layout.Widths[group]); ++j)
                        maxDiagonal = Math.Max(maxDiagonal, gram[j, j]);
                    if (damping[group] == 0) damping[group] = dampingScale == null ? Math.Max(1e-10, .002 * maxDiagonal) : .002;
                }
                double modelRadius = positionRadius[0], modelIntensityRadius = intensityRadius[0], modelLambda = damping[0];
                bool poolUsed = false;
                var uphill = new List<(double[] theta, double[] delta, double predicted, double next, int shrinks)>();
                bool acceptedIteration = false;
                long reserved = settings.ReserveRemainingModels ?
                    (long)(settings.Iterations - result.Jacobians) * (maxProbes + 1 + checkCost) : 0;
                for (int attempt = 0; attempt < 7 && result.Evaluations + 1 + checkCost + reserved <= settings.Budget; ++attempt)
                {
                    var delta = groups == 1 ? TimedSolve(model, directionGradient, x, damping[0], dampingScale) : new double[n];
                    var unclipped = settings.TraceProposalClipping ? new double[n] : null;
                    for (int lamp = 0; lamp < lamps; ++lamp)
                    {
                        int group = groups == 1 ? 0 : lamp;
                        int k = layout.Starts[lamp], width = layout.Widths[lamp];
                        if (groups > 1)
                        {
                            var block = new double[width, width];
                            for (int i = 0; i < width; ++i)
                            for (int j = 0; j < width; ++j) block[i, j] = gram[k + i, k + j];
                            Array.Copy(TimedSolve(block, gradient.Skip(k).Take(width).ToArray(), x.Skip(k).Take(width).ToArray(),
                                damping[group], null), 0, delta, k, width);
                        }
                        if (unclipped != null) Array.Copy(delta, k, unclipped, k, width);
                        LimitLampStep(delta, layout, lamp, positionRadius[group], intensityRadius[group]);
                    }
                    var radiusLimited = settings.TraceProposalClipping ? (double[])delta.Clone() : null;
                    int[] lampShrinks = settings.LampPositionAllowed == null ? null : new int[lamps];
                    string constraint = "none";
                    int shrinks = BuildCandidate(x, delta, probe, layout, settings.CandidateAllowed, settings.LampPositionAllowed, lampShrinks, out constraint);
                    bool legal = Allowed(probe, settings.FinalAllowed);
                    bool trialAllowed = Allowed(probe, settings.CandidateAllowed);
                    var predictedByLamp = new double[lamps];
                    var actualByLamp = Enumerable.Repeat(double.NaN, lamps).ToArray();
                    for (int i = 0; i < n; ++i)
                    {
                        double p = -2 * gradient[i] * delta[i];
                        for (int j = 0; j < n; ++j) p -= delta[i] * gram[i, j] * delta[j];
                        predictedByLamp[layout.Owners[i]] += p;
                    }
                    double predicted = predictedByLamp.Sum();
                    double actual = double.NaN, ratio = double.NaN, nativeActual = double.NaN, next = f;
                    bool accepted = false;
                    string status = !trialAllowed ? "infeasible" : "nonpositive_prediction";
                    if (trialAllowed && predicted > 1e-18)
                    {
                        next = Evaluate(probe, "trial", trialResidual);
                        actual = f - next;
                        ratio = actual / predicted;
                        if (settings.ResidualBlocks != null)
                            for (int lamp = 0; lamp < lamps; ++lamp)
                                actualByLamp[lamp] = SliceNorm(center, offsets[lamp], offsets[lamp + 1]) -
                                    SliceNorm(trialResidual, offsets[lamp], offsets[lamp + 1]);
                        accepted = actual > Math.Max(1e-14, 1e-9 * f) && ratio >= .02;
                        status = accepted ? "accepted" : "objective_rejected";
                        if (accepted && settings.NativeAcceptance != null)
                        {
                            nativeActual = native - CheckNative(probe, "native_acceptance");
                            accepted = nativeActual > Math.Max(1e-12, 1e-8 * native);
                            if (!accepted) status = "native_rejected";
                        }
                    }
                    if (settings.Fallback == TrustFallback.Nonmonotone && legal && !accepted && actual <= 0 && predicted > 0)
                        uphill.Add(((double[])probe.Clone(), delta, predicted, next, shrinks));
                    settings.Trial?.Invoke(new TrustTrial { Iteration = result.Jacobians, Query = result.Evaluations, Shrinks = shrinks, Status = status,
                        Accepted = accepted, Legal = legal, Source = "full", Constraint = constraint, LampShrinks = lampShrinks, CurrentLoss = f, BestLoss = result.BestLoss, Predicted = predicted, Actual = actual, Ratio = ratio, NativeActual = nativeActual,
                        PositionRadius = positionRadius, IntensityRadius = intensityRadius, Damping = damping, Step = delta,
                        DampingScale = dampingScale, Theta = probe, UnclippedStep = unclipped, RadiusLimitedStep = radiusLimited,
                        UnclippedPrediction = unclipped == null ? double.NaN : TrustPredicted(gram, gradient, unclipped),
                        PredictedByLamp = predictedByLamp, ActualByLamp = actualByLamp });
                    for (int group = 0; group < groups; ++group)
                    {
                        double localRatio = groups == 1 ? ratio : predictedByLamp[group] > 0 ? actualByLamp[group] / predictedByLamp[group] : double.NaN;
                        if (!accepted || !(localRatio >= .23))
                        {
                            positionRadius[group] *= .62; intensityRadius[group] *= .62;
                            damping[group] *= 4;
                        }
                        else if (localRatio > .72)
                        {
                            positionRadius[group] = Math.Min(.70, positionRadius[group] * 1.38);
                            intensityRadius[group] = Math.Min(.75, intensityRadius[group] * 1.38);
                            damping[group] = Math.Max(1e-12, damping[group] * .40);
                        }
                    }
                    consecutiveCandidateRejects = status is "objective_rejected" or "native_rejected"
                        ? consecutiveCandidateRejects + 1 : 0;
                    if (!accepted)
                    {
                        ++result.Rejected;
                        if (settings.MaxConsecutiveCandidateRejects > 0 &&
                            consecutiveCandidateRejects >= settings.MaxConsecutiveCandidateRejects)
                        {
                            result.Stop = "consecutive_candidate_rejects_shrunk";
                            break;
                        }
                        if (viewPool && consecutiveCandidateRejects >= 2 && !poolUsed && TryFallback())
                        { acceptedIteration = true; break; }
                        continue;
                    }
                    ++result.Accepted;
                    if (!legal) ++result.IllegalAccepted;
                    Array.Copy(probe, x, n);
                    (center, trialResidual) = (trialResidual, center);
                    f = next;
                    if (settings.NativeAcceptance != null) native -= nativeActual;
                    if (legal && f < result.BestLoss)
                    {
                        firstUphillModel = 0;
                        result.BestLoss = f;
                        result.Best = (double[])x.Clone();
                    }
                    Remember(f, uphillStep: false);
                    acceptedIteration = true;
                    break;
                }
                if (result.Stop == "consecutive_candidate_rejects_shrunk") break;
                if (!acceptedIteration && !poolUsed && settings.Fallback != TrustFallback.None &&
                    (!viewPool || settings.FixedProbeScale || probeScale <= .125)) acceptedIteration = TryFallback();
                if (firstUphillModel > 0 && result.Jacobians - firstUphillModel >= 3)
                { result.Stop = "nonmonotone_rollback"; f = result.BestLoss; break; }
                if (settings.StopAfterIteration?.Invoke() == true) { result.Stop = "small_improvements"; break; }
                if (acceptedIteration) stalledModels = 0;
                if (!acceptedIteration)
                {
                    if (settings.StallModelLimit > 0 && ++stalledModels >= settings.StallModelLimit) { result.Stop = "stalled_models"; break; }
                    if (settings.StallModelLimit > 0 && settings.FixedProbeScale) continue;
                    if (result.Evaluations + maxProbes + 1 + checkCost > settings.Budget) break;
                    if (settings.FixedProbeScale || probeScale <= .125) { result.Stop = "no_decreasing_step"; break; }
                    probeScale *= .5;
                    for (int group = 0; group < groups; ++group)
                    {
                        positionRadius[group] = Math.Max(.012, positionRadius[group]);
                        intensityRadius[group] = Math.Max(.03, intensityRadius[group]);
                        damping[group] = 0;
                    }
                }

                bool TryFallback()
                {
                    poolUsed = true;
                    var proposals = new List<(string source, double[] theta, double[] delta, double predicted, int shrinks)>();
                    if (viewPool)
                    {
                        int[] masks = settings.Fallback == TrustFallback.ViewPairPool ? new[] { 1, 2, 4, 3, 5, 6 } : new[] { 1, 2, 4 };
                        foreach (int mask in masks)
                        {
                            var (h, g) = SubsetNormal(columns, center, settings.ViewBlocks, mask);
                            Add("views_" + mask, TimedSolve(h, g, x, modelLambda, LampDampingScale(h, layout)), 1);
                        }
                        proposals = proposals.OrderByDescending(p => p.predicted).ToList();
                        if (proposals.Count > 0 && proposals[0].predicted <= 0) proposals = proposals.Take(1).ToList();
                    }
                    else if (settings.Fallback == TrustFallback.Cauchy)
                    {
                        var d = gradient.Select((g, j) => -g / (dampingScale == null ? 1 : dampingScale[j])).ToArray();
                        double slope = d.Select((v, i) => -v * gradient[i]).Sum();
                        double curvature = d.Select((v, i) => v * d.Select((w, j) => gram[i, j] * w).Sum()).Sum();
                        if (slope > 0 && curvature > 0)
                            foreach (double fraction in new[] { 1d, .5, .25 }) Add("cauchy", d.Select(v => v * slope / curvature).ToArray(), fraction);
                    }
                    else if (settings.Fallback == TrustFallback.Dogleg)
                    {
                        // Gauss-Newton dogleg in the physical trust-radius coordinates.
                        var s = layout.Axes.Select(a => (a.IsPosition ? modelRadius * a.Radius / .24 :
                            Math.Min(a.MaximumRadius, modelIntensityRadius * a.Radius / .28)) / a.Unit).ToArray();
                        var z = gradient.Select((g, j) => -g * s[j]).ToArray();
                        var physicalCauchy = z.Select((v, j) => v * s[j]).ToArray();
                        double slope = z.Sum(v => v * v);
                        double curvature = 2 * slope - TrustPredicted(gram, gradient, physicalCauchy);
                        if (slope > 0 && curvature > 0)
                        {
                            var u = z.Select(v => v * slope / curvature).ToArray();
                            double Norm(double[] v) => Math.Sqrt(v.Sum(a => a * a));
                            double un = Norm(u);
                            var gn = TimedSolve(gram, gradient, x, 1e-12, dampingScale).Select((v, j) => v / s[j]).ToArray();
                            double[] Step()
                            {
                                if (un >= 1) return u.Select(v => v / un).ToArray();
                                if (Norm(gn) <= 1) return gn;
                                var b = gn.Select((v, j) => v - u[j]).ToArray();
                                double bb = b.Sum(v => v * v), ub = b.Select((v, j) => v * u[j]).Sum();
                                double t = (1 - un * un) / (Math.Sqrt(Math.Max(0, ub * ub + bb * (1 - un * un))) + ub);
                                return u.Select((v, j) => v + t * b[j]).ToArray();
                            }
                            Add("dogleg", Step().Select((v, j) => v * s[j]).ToArray(), 1);
                            Add("scaled_cauchy", u.Select((v, j) => v * s[j] / Math.Max(1, un)).ToArray(), 1);
                        }
                    }
                    else if (settings.Fallback == TrustFallback.Nonmonotone && uphillInARow < 2)
                    {
                        // Retest, against the bounded reference, only candidates that received a real render.
                        double reference = history.Max();
                        foreach (var p in uphill.OrderBy(u => u.next))
                        {
                            double rhoRef = (reference - p.next) / p.predicted;
                            bool take = p.next <= Math.Min(reference, 1.05 * result.BestLoss) && rhoRef >= .02;
                            Array.Copy(p.theta, probe, n);
                            if (!take) { Audit("nonmonotone", p.delta, p.predicted, p.next, p.shrinks, false, reference); continue; }
                            // A rejected trial's residual buffer may have been overwritten. Meter the recapture.
                            if (result.Evaluations >= settings.Budget) break;
                            double observed = Evaluate(probe, "nonmonotone_recheck", trialResidual);
                            take = observed <= Math.Min(reference, 1.05 * result.BestLoss) && (reference - observed) / p.predicted >= .02;
                            Audit("nonmonotone", p.delta, p.predicted, observed, p.shrinks, take, reference);
                            if (take) { Accept(observed, true); return true; }
                            ++result.Rejected;
                        }
                    }
                    foreach (var p in proposals.Take(3))
                    {
                        if (result.Evaluations >= settings.Budget) break;
                        Array.Copy(p.theta, probe, n);
                        double next = Evaluate(probe, p.source, trialResidual);
                        bool take = f - next > Math.Max(1e-14, 1e-9 * f);
                        Audit(p.source, p.delta, p.predicted, next, p.shrinks, take, f);
                        if (take) { Accept(next, false); return true; }
                        ++result.Rejected;
                    }
                    return false;

                    void Add(string source, double[] d, double fraction)
                    {
                        for (int lamp = 0; lamp < lamps; ++lamp) LimitLampStep(d, layout, lamp, modelRadius, modelIntensityRadius);
                        for (int j = 0; j < n; ++j) d[j] *= fraction;
                        var candidate = new double[n];
                        int shrinks = BuildCandidate(x, d, candidate, layout, settings.CandidateAllowed, settings.LampPositionAllowed, null, out _);
                        double predicted = TrustPredicted(gram, gradient, d);
                        if (!Allowed(candidate, settings.CandidateAllowed) || !Allowed(candidate, settings.FinalAllowed))
                        {
                            settings.Trial?.Invoke(new TrustTrial { Iteration = result.Jacobians, Query = result.Evaluations,
                                Status = "infeasible", Source = source, Theta = candidate, Step = d, Shrinks = shrinks,
                                CurrentLoss = f, Predicted = predicted, Actual = double.NaN, PositionRadius = positionRadius,
                                IntensityRadius = intensityRadius, Damping = damping });
                            return;
                        }
                        if (d.Any(v => v != 0)) proposals.Add((source, candidate, d, predicted, shrinks));
                    }
                    void Audit(string source, double[] d, double predicted, double next, int shrinks, bool accepted, double reference)
                    {
                        settings.Trial?.Invoke(new TrustTrial { Iteration = result.Jacobians, Query = result.Evaluations,
                            Source = source, Status = !accepted ? "fallback_rejected" : source == "nonmonotone" ? "nonmonotone_accepted" :
                                predicted <= 0 ? "accepted_nonpositive_prediction" : "fallback_accepted",
                            Legal = true, Accepted = accepted, Theta = probe, Step = d, Predicted = predicted, Actual = f - next,
                            Ratio = predicted > 0 ? (f - next) / predicted : double.NaN, CurrentLoss = f, ReferenceLoss = reference,
                            ReferenceRatio = predicted > 0 ? (reference - next) / predicted : double.NaN, BestLoss = result.BestLoss,
                            Shrinks = shrinks, PositionRadius = positionRadius, IntensityRadius = intensityRadius, Damping = damping });
                    }
                    void Accept(double next, bool isUphill)
                    {
                        ++result.Accepted; consecutiveCandidateRejects = 0;
                        Array.Copy(probe, x, n); (center, trialResidual) = (trialResidual, center); f = next;
                        if (f < result.BestLoss) { firstUphillModel = 0; result.BestLoss = f; result.Best = (double[])x.Clone(); }
                        Remember(f, isUphill);
                        // Rebuild around the observed point; no trust update from a proxy/reference rho.
                        positionRadius[0] = modelRadius; intensityRadius[0] = modelIntensityRadius; damping[0] = modelLambda;
                    }
                }

                void NormalRow(int i)
                {
                    gradient[i] = Dot(columns[i], center);
                    for (int j = 0; j <= i; ++j)
                        gram[i, j] = gram[j, i] = Dot(columns[i], columns[j]);
                }
            }
            if (result.Jacobians >= settings.Iterations && result.Stop == "budget") result.Stop = "iteration_cap";
            result.LastLoss = f;
            return result;

            void Remember(double loss, bool uphillStep)
            {
                if (settings.Fallback != TrustFallback.Nonmonotone) return;
                if (history.Count == 5) history.Dequeue();
                history.Enqueue(loss);
                uphillInARow = uphillStep ? uphillInARow + 1 : 0;
                if (uphillStep && firstUphillModel == 0) firstUphillModel = result.Jacobians;
            }

            bool CanProbe()
            {
                if (Allowed(probe, settings.ProbeAllowed)) return true;
                ++result.Blocked;
                return false;
            }
            double[] TimedSolve(double[,] normal, double[] g, double[] current, double lambda, double[] scale)
            {
                if (settings.Timing == null) return SolveDamped(normal, g, current, lambda, 0, scale);
                long begin = System.Diagnostics.Stopwatch.GetTimestamp();
                var step = SolveDamped(normal, g, current, lambda, 0, scale);
                settings.Timing.solveTicks += System.Diagnostics.Stopwatch.GetTimestamp() - begin;
                return step;
            }

            double Evaluate(double[] theta, string phase, float[] values, int axis = -1, double h = 0)
            {
                ++result.ObjectiveQueries;
                ++result.Evaluations;
                double loss = query(theta, phase, values);
                if (!(loss >= 0) || double.IsInfinity(loss)) throw new InvalidOperationException("Nonfinite residual objective.");
                settings.Query?.Invoke(result.Evaluations, phase, theta, loss, axis, axis < 0 ? 0 : h * units[axis]);
                return loss;
            }
            double CheckNative(double[] theta, string phase)
            {
                ++result.NativeChecks;
                ++result.Evaluations;
                double loss = settings.NativeAcceptance(theta);
                if (!(loss >= 0) || double.IsInfinity(loss)) throw new InvalidOperationException("Nonfinite native validation objective.");
                settings.Query?.Invoke(result.Evaluations, phase, theta, loss, -1, 0);
                return loss;
            }
        }

        /// <summary>The first A2 proposal, using the same damping, radii and feasibility backtracking as RunTrust.</summary>
        public static double[] LampScaledTrustProposal(double[,] gram, double[] gradient, double[] theta, double[] units,
            Func<double[], bool> legal, out int shrinks)
        {
            return LampScaledTrustProposal(gram, gradient, theta, TrustLayout.Legacy(units, units), legal, out shrinks);
        }

        public static double[] LampScaledTrustProposal(double[,] gram, double[] gradient, double[] theta, TrustLayout layout,
            Func<double[], bool> legal, out int shrinks, double fraction = 1, bool cauchy = false,
            Func<double[], int, bool> lampLegal = null, int[] lampShrinks = null)
        {
            var scale = LampDampingScale(gram, layout);
            var step = cauchy ? gradient.Select((g, i) => -g / scale[i]).ToArray() : SolveDamped(gram, gradient, theta, .002, 0, scale);
            if (cauchy)
            {
                double slope = -step.Select((v, i) => v * gradient[i]).Sum();
                double curvature = 2 * slope - TrustPredicted(gram, gradient, step);
                if (curvature > 0) for (int i = 0; i < step.Length; ++i) step[i] *= slope / curvature;
            }
            for (int lamp = 0; lamp < layout.Lamps; ++lamp) LimitLampStep(step, layout, lamp, .24, .28);
            for (int i = 0; i < step.Length; ++i) step[i] *= fraction;
            shrinks = BuildCandidate(theta, step, new double[theta.Length], layout, legal, lampLegal, lampShrinks, out _);
            return step;
        }

        public static double TrustPredicted(double[,] h, double[] g, double[] d)
        {
            double total = 0;
            for (int i = 0; i < d.Length; ++i)
            {
                double p = -2 * g[i] * d[i];
                for (int j = 0; j < d.Length; ++j) p -= d[i] * h[i, j] * d[j];
                total += p;
            }
            return total;
        }

        private static (double[,] h, double[] g) SubsetNormal(float[][] columns, float[] residual, int[] blocks, int mask)
        {
            int n = columns.Length, start = 0;
            var h = new double[n, n]; var g = new double[n];
            for (int v = 0; v < blocks.Length; ++v)
            {
                int end = start + blocks[v];
                if ((mask & (1 << v)) != 0)
                    for (int i = 0; i < n; ++i)
                    {
                        for (int k = start; k < end; ++k) g[i] += (double)columns[i][k] * residual[k];
                        for (int j = 0; j <= i; ++j)
                            for (int k = start; k < end; ++k) h[i, j] += (double)columns[i][k] * columns[j][k];
                    }
                start = end;
            }
            for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) h[j, i] = h[i, j];
            return (h, g);
        }

        private static double[] LampDampingScale(double[,] gram, TrustLayout layout)
        {
            var scale = new double[layout.Lamps];
            for (int i = 0; i < layout.Count; ++i) scale[layout.Owners[i]] = Math.Max(scale[layout.Owners[i]], gram[i, i]);
            double floor = Math.Max(1e-10, 1e-6 * scale.Max());
            return Enumerable.Range(0, layout.Count).Select(i => Math.Max(floor, scale[layout.Owners[i]])).ToArray();
        }

        private static void LimitLampStep(double[] delta, TrustLayout layout, int lamp, double positionRadius, double intensityRadius)
        {
            int from = layout.Starts[lamp], end = from + layout.Widths[lamp];
            double norm = 0;
            for (int i = from; i < end; ++i)
                if (layout.Axes[i].IsPosition) { double dx = delta[i] * layout.Axes[i].Unit; norm += dx * dx; }
            for (int i = from; i < end; ++i)
            {
                var axis = layout.Axes[i];
                if (axis.IsPosition) delta[i] *= Math.Min(1, (axis.Radius == .24 ? positionRadius : positionRadius * (axis.Radius / .24)) / Math.Max(1e-30, Math.Sqrt(norm)));
                else
                {
                    double radius = axis.IsIntensity ? (axis.Radius == .28 ? intensityRadius : intensityRadius * (axis.Radius / .28)) : Math.Min(axis.MaximumRadius, axis.Radius * intensityRadius / .28);
                    double limit = radius / axis.Unit;
                    delta[i] = BlackBoxOptimizer.Clamp(delta[i], -limit, limit);
                }
            }
        }

        private static int BuildCandidate(double[] theta, double[] delta, double[] candidate, TrustLayout layout,
            Func<double[], bool> legal, Func<double[], int, bool> lampLegal, int[] shrinksByLamp, out string reason)
        {
            reason = "none";
            if (lampLegal == null)
            {
                int shrinks = FormCandidate(theta, delta, candidate, legal);
                if (shrinks != 0) reason = "whole_rig";
                return shrinks;
            }
            for (int j = 0; j < theta.Length; ++j)
            {
                candidate[j] = BlackBoxOptimizer.Clamp(theta[j] + delta[j], 0, 1);
                delta[j] = candidate[j] - theta[j];
            }
            for (int lamp = 0; lamp < layout.Lamps; ++lamp)
            {
                if (lampLegal(candidate, lamp)) continue;
                int count = 0, from = layout.Starts[lamp], end = from + layout.Widths[lamp];
                reason = "lamp_position";
                do
                {
                    for (int j = from; j < end; ++j)
                        if (layout.Axes[j].IsPosition) candidate[j] = theta[j] + (delta[j] *= .5);
                    ++count;
                } while (count < 10 && !lampLegal(candidate, lamp));
                if (!lampLegal(candidate, lamp))
                {
                    for (int j = from; j < end; ++j)
                        if (layout.Axes[j].IsPosition) { candidate[j] = theta[j]; delta[j] = 0; }
                    ++count;
                }
                if (shrinksByLamp != null) shrinksByLamp[lamp] = count;
            }
            if (Allowed(candidate, legal)) return 0;
            reason = "rig_after_local";
            return FormCandidate(theta, delta, candidate, legal);
        }

        private static int FormCandidate(double[] theta, double[] delta, double[] candidate, Func<double[], bool> legal)
        {
            for (int j = 0; j < theta.Length; ++j)
            {
                candidate[j] = BlackBoxOptimizer.Clamp(theta[j] + delta[j], 0, 1);
                delta[j] = candidate[j] - theta[j];
            }
            int shrinks = 0;
            while (!Allowed(candidate, legal) && shrinks < 10)
            {
                for (int j = 0; j < theta.Length; ++j) candidate[j] = theta[j] + (delta[j] *= .5);
                ++shrinks;
            }
            return shrinks;
        }

        private static double SliceNorm(float[] residual, int from, int to)
        {
            double result = 0;
            for (int i = from; i < to; ++i) result += (double)residual[i] * residual[i];
            return result;
        }

        private static bool Allowed(double[] theta, Func<double[], bool> condition)
        {
            foreach (double value in theta) if (!(value >= 0 && value <= 1)) return false;
            return condition == null || condition(theta);
        }
    }
}

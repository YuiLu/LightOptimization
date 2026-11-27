using System;
using System.Linq;
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
            public bool IndependentLampDamping;
            public TrustNormalMode NormalMode;
            public double PositionRadius = .24, IntensityRadius = .28;
            public Func<double[], bool> ProbeAllowed, CandidateAllowed, FinalAllowed;
            public Func<double[], double> NativeAcceptance;
            public Action<double[], int, float[]> LogIntensityJacobian;
            public TrustPositionModel PositionModel;
            public Action<int, string, double[], double, int, double> Query;
            public Action<TrustJacobian> Jacobian;
            public Action<TrustTrial> Trial;
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
            public string Status;
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
            if (n <= 0 || n % 4 != 0 || residualSize < 1 || settings.Budget < 2 || settings.Units?.Length != n ||
                settings.PhysicalSteps?.Length != n || settings.Units.Any(u => !(u > 0)) || settings.PhysicalSteps.Any(h => !(h > 0)))
                throw new ArgumentException("Expected XYZ/log-intensity units, positive physical steps and a budget.");
            int lamps = n / 4;
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
            int consecutiveCandidateRejects = 0;
            int checkCost = settings.NativeAcceptance == null ? 0 : 1;
            int maxProbes = 2 * (n - (settings.LogIntensityJacobian == null ? 0 : lamps));
            var positionModel = settings.PositionModel;
            bool modelIntensity = positionModel?.IntensityColumn != null;
            if (modelIntensity && settings.LogIntensityJacobian != null)
                throw new ArgumentException("Only one intensity Jacobian provider may be used.");
            if (positionModel != null)
                maxProbes += positionModel.AnchorCost + 6 * lamps * (positionModel.ProbeCost - 1) +
                    2 * lamps * (positionModel.IntensityProbeCost - 1);
            // Paired intensity captures are shared with the position model before building XYZ columns.
            var columnOrder = Enumerable.Range(0, n).ToArray();
            if (modelIntensity) columnOrder = columnOrder.OrderBy(j => j % 4 != 3).ToArray();

            while (result.Evaluations + maxProbes + 1 + checkCost <= settings.Budget && result.Jacobians < settings.Iterations)
            {
                if (f < 1e-12) { result.Stop = "near_zero_residual"; break; }
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
                    if (settings.LogIntensityJacobian != null && j % 4 == 3)
                    {
                        settings.LogIntensityJacobian(x, j / 4, col);
                        stencils[j] = 4;
                        continue;
                    }
                    double h = settings.PhysicalSteps[j] * probeScale / settings.Units[j];
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
                        bool intensity = j % 4 == 3;
                        if (positionModel != null && (!intensity || modelIntensity))
                        {
                            int stencil = central ? 3 : second ? 2 : 1;
                            var build = intensity ? positionModel.IntensityColumn : positionModel.Column;
                            build(result.Evaluations + 1, x, j, d, stencil, col);
                            result.Evaluations += (stencil == 1 ? 1 : 2) * (intensity ? positionModel.IntensityProbeCost : positionModel.ProbeCost);
                            steps[j] = d * settings.Units[j]; stencils[j] = stencil;
                        }
                        else if (central)
                        {
                            probe[j] = x[j] + h; Evaluate(probe, "fd_plus", col, j, h);
                            probe[j] = x[j] - h; Evaluate(probe, "fd_minus", temporary, j, -h);
                            float scale = (float)(.5 / h);
                            for (int i = 0; i < residualSize; ++i) col[i] = (col[i] - temporary[i]) * scale;
                            steps[j] = h * settings.Units[j]; stencils[j] = 3;
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
                            steps[j] = d * settings.Units[j]; stencils[j] = second ? 2 : 1;
                        }
                        break;
                    }
                    if (stencils[j] == 0) Array.Clear(col, 0, col.Length);
                    probe[j] = x[j];
                }
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
                settings.Jacobian?.Invoke(new TrustJacobian { Iteration = result.Jacobians, Query = result.Evaluations, Theta = x,
                    PhysicalSteps = steps, Stencils = stencils, Gram = gram, Gradient = gradient, Blocked = result.Blocked - blockedBefore });
                if (zeros == n) { result.Stop = "zero_jacobian"; break; }
                var model = gram;
                double[] dampingScale = null;
                if (settings.NormalMode == TrustNormalMode.LampDiagonal)
                {
                    model = new double[n, n];
                    for (int i = 0; i < n; ++i)
                    for (int j = 4 * (i / 4); j < 4 * (i / 4 + 1); ++j)
                        model[i, j] = gram[i, j];
                }
                else if (settings.NormalMode == TrustNormalMode.LampScaledDense)
                    dampingScale = LampDampingScale(gram, n);
                else if (settings.NormalMode == TrustNormalMode.ParameterScaledDense)
                {
                    var diag = Enumerable.Range(0, n).Select(i => gram[i, i]).ToArray();
                    double floor = Math.Max(1e-10, 1e-6 * diag.Max());
                    dampingScale = diag.Select(x => Math.Max(floor, x)).ToArray();
                }
                for (int group = 0; group < groups; ++group)
                {
                    double maxDiagonal = 0;
                    for (int j = groups == 1 ? 0 : 4 * group; j < (groups == 1 ? n : 4 * (group + 1)); ++j)
                        maxDiagonal = Math.Max(maxDiagonal, gram[j, j]);
                    if (damping[group] == 0) damping[group] = dampingScale == null ? Math.Max(1e-10, .002 * maxDiagonal) : .002;
                }
                bool acceptedIteration = false;
                for (int attempt = 0; attempt < 7 && result.Evaluations + 1 + checkCost <= settings.Budget; ++attempt)
                {
                    var delta = groups == 1 ? TimedSolve(model, gradient, x, damping[0], dampingScale) : new double[n];
                    var unclipped = settings.TraceProposalClipping ? new double[n] : null;
                    for (int lamp = 0; lamp < lamps; ++lamp)
                    {
                        int group = groups == 1 ? 0 : lamp;
                        int k = 4 * lamp;
                        if (groups > 1)
                        {
                            var block = new double[4, 4];
                            for (int i = 0; i < 4; ++i)
                            for (int j = 0; j < 4; ++j) block[i, j] = gram[k + i, k + j];
                            Array.Copy(TimedSolve(block, gradient.Skip(k).Take(4).ToArray(), x.Skip(k).Take(4).ToArray(),
                                damping[group], null), 0, delta, k, 4);
                        }
                        if (unclipped != null) Array.Copy(delta, k, unclipped, k, 4);
                        LimitLampStep(delta, k, settings.Units, positionRadius[group], intensityRadius[group]);
                    }
                    var radiusLimited = settings.TraceProposalClipping ? (double[])delta.Clone() : null;
                    int shrinks = FormCandidate(x, delta, probe, settings.CandidateAllowed);
                    bool legal = Allowed(probe, settings.FinalAllowed);
                    bool trialAllowed = Allowed(probe, settings.CandidateAllowed);
                    var predictedByLamp = new double[lamps];
                    var actualByLamp = Enumerable.Repeat(double.NaN, lamps).ToArray();
                    for (int i = 0; i < n; ++i)
                    {
                        double p = -2 * gradient[i] * delta[i];
                        for (int j = 0; j < n; ++j) p -= delta[i] * gram[i, j] * delta[j];
                        predictedByLamp[i / 4] += p;
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
                    settings.Trial?.Invoke(new TrustTrial { Iteration = result.Jacobians, Query = result.Evaluations, Shrinks = shrinks, Status = status,
                        Accepted = accepted, Legal = legal, Predicted = predicted, Actual = actual, Ratio = ratio, NativeActual = nativeActual,
                        PositionRadius = positionRadius, IntensityRadius = intensityRadius, Damping = damping, Step = delta,
                        DampingScale = dampingScale, Theta = probe, UnclippedStep = unclipped, RadiusLimitedStep = radiusLimited,
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
                        result.BestLoss = f;
                        result.Best = (double[])x.Clone();
                    }
                    acceptedIteration = true;
                    break;
                }
                if (result.Stop == "consecutive_candidate_rejects_shrunk") break;
                if (!acceptedIteration)
                {
                    if (result.Evaluations + maxProbes + 1 + checkCost > settings.Budget) break;
                    if (probeScale <= .125) { result.Stop = "no_decreasing_step"; break; }
                    probeScale *= .5;
                    for (int group = 0; group < groups; ++group)
                    {
                        positionRadius[group] = Math.Max(.012, positionRadius[group]);
                        intensityRadius[group] = Math.Max(.03, intensityRadius[group]);
                        damping[group] = 0;
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
                settings.Query?.Invoke(result.Evaluations, phase, theta, loss, axis, axis < 0 ? 0 : h * settings.Units[axis]);
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
            var step = SolveDamped(gram, gradient, theta, .002, 0, LampDampingScale(gram, theta.Length));
            for (int k = 0; k < theta.Length; k += 4) LimitLampStep(step, k, units, .24, .28);
            var candidate = new double[theta.Length];
            shrinks = FormCandidate(theta, step, candidate, legal);
            return step;
        }

        private static double[] LampDampingScale(double[,] gram, int n)
        {
            var scale = new double[n / 4];
            for (int i = 0; i < n; ++i) scale[i / 4] = Math.Max(scale[i / 4], gram[i, i]);
            double floor = Math.Max(1e-10, 1e-6 * scale.Max());
            return Enumerable.Range(0, n).Select(i => Math.Max(floor, scale[i / 4])).ToArray();
        }

        private static void LimitLampStep(double[] delta, int k, double[] units, double positionRadius, double intensityRadius)
        {
            double dx = delta[k] * units[k], dy = delta[k + 1] * units[k + 1], dz = delta[k + 2] * units[k + 2];
            double limit = positionRadius / Math.Max(1e-30, Math.Sqrt(dx * dx + dy * dy + dz * dz));
            for (int j = 0; j < 3; ++j) delta[k + j] *= Math.Min(1, limit);
            double intensityLimit = intensityRadius / units[k + 3];
            delta[k + 3] = BlackBoxOptimizer.Clamp(delta[k + 3], -intensityLimit, intensityLimit);
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

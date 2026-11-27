using System;
using System.Linq;
using System.Threading.Tasks;

namespace LightingMvp
{
    public static partial class ImageJacobianLm
    {
        public enum TrustNormalMode { Shared, LampDiagonal, LampScaledDense }

        public sealed class TrustSettings
        {
            public int Budget, Iterations = int.MaxValue;
            public double[] Units, PhysicalSteps;
            public int[] ResidualBlocks;
            public bool IndependentLampDamping;
            public TrustNormalMode NormalMode;
            public double PositionRadius = .24, IntensityRadius = .28;
            public Func<double[], bool> ProbeAllowed, CandidateAllowed, FinalAllowed;
            public Func<double[], double> NativeAcceptance;
            public Action<double[], int, float[]> LogIntensityJacobian;
            public Action<int, string, double[], double, int, double> Query;
            public Action<TrustJacobian> Jacobian;
            public Action<TrustTrial> Trial;
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
            public double[] DampingScale, Theta;
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
            int checkCost = settings.NativeAcceptance == null ? 0 : 1;
            int maxProbes = 2 * (n - (settings.LogIntensityJacobian == null ? 0 : lamps));

            while (result.Evaluations + maxProbes + 1 + checkCost <= settings.Budget && result.Jacobians < settings.Iterations)
            {
                if (f < 1e-12) { result.Stop = "near_zero_residual"; break; }
                ++result.Jacobians;
                int blockedBefore = result.Blocked;
                var steps = new double[n];
                var stencils = new int[n];
                Array.Copy(x, probe, n);
                for (int j = 0; j < n; ++j)
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
                        if (plus && minus)
                        {
                            probe[j] = x[j] + h; Evaluate(probe, "fd_plus", col, j, h);
                            probe[j] = x[j] - h; Evaluate(probe, "fd_minus", temporary, j, -h);
                            float scale = (float)(.5 / h);
                            for (int i = 0; i < residualSize; ++i) col[i] = (col[i] - temporary[i]) * scale;
                            steps[j] = h * settings.Units[j]; stencils[j] = 3;
                        }
                        else
                        {
                            double d = plus ? h : -h;
                            probe[j] = x[j] + 2 * d; bool second = CanProbe();
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
                var gradient = new double[n];
                var gram = new double[n, n];
                if (residualSize >= 300000 && n >= 8)
                    Parallel.For(0, n, new ParallelOptions { MaxDegreeOfParallelism = Math.Min(Environment.ProcessorCount, 8) }, NormalRow);
                else for (int j = 0; j < n; ++j) NormalRow(j);
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
                {
                    var scale = new double[lamps];
                    for (int i = 0; i < n; ++i) scale[i / 4] = Math.Max(scale[i / 4], gram[i, i]);
                    double floor = Math.Max(1e-10, 1e-6 * scale.Max());
                    dampingScale = Enumerable.Range(0, n).Select(i => Math.Max(floor, scale[i / 4])).ToArray();
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
                    var delta = groups == 1 ? SolveDamped(model, gradient, x, damping[0], 0, dampingScale) : new double[n];
                    for (int lamp = 0; lamp < lamps; ++lamp)
                    {
                        int group = groups == 1 ? 0 : lamp;
                        int k = 4 * lamp;
                        if (groups > 1)
                        {
                            var block = new double[4, 4];
                            for (int i = 0; i < 4; ++i)
                            for (int j = 0; j < 4; ++j) block[i, j] = gram[k + i, k + j];
                            Array.Copy(SolveDamped(block, gradient.Skip(k).Take(4).ToArray(), x.Skip(k).Take(4).ToArray(),
                                damping[group], 0), 0, delta, k, 4);
                        }
                        double dx = delta[k] * settings.Units[k], dy = delta[k + 1] * settings.Units[k + 1], dz = delta[k + 2] * settings.Units[k + 2];
                        double limit = positionRadius[group] / Math.Max(1e-30, Math.Sqrt(dx * dx + dy * dy + dz * dz));
                        for (int j = 0; j < 3; ++j) delta[k + j] *= Math.Min(1, limit);
                        double intensityLimit = intensityRadius[group] / settings.Units[k + 3];
                        delta[k + 3] = BlackBoxOptimizer.Clamp(delta[k + 3], -intensityLimit, intensityLimit);
                    }
                    for (int j = 0; j < n; ++j)
                    {
                        probe[j] = BlackBoxOptimizer.Clamp(x[j] + delta[j], 0, 1);
                        delta[j] = probe[j] - x[j];
                    }
                    int shrinks = 0;
                    while (!Allowed(probe, settings.CandidateAllowed) && shrinks < 10)
                    {
                        for (int j = 0; j < n; ++j) probe[j] = x[j] + (delta[j] *= .5);
                        ++shrinks;
                    }
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
                        DampingScale = dampingScale, Theta = probe, PredictedByLamp = predictedByLamp, ActualByLamp = actualByLamp });
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
                    if (!accepted) { ++result.Rejected; continue; }
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

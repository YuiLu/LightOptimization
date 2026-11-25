using System;

namespace LightingMvp
{
    /// <summary>Finite-difference image Jacobian with a bounded, damped Gauss–Newton step.</summary>
    public static class ImageJacobianLm
    {
        public delegate double ResidualQuery(double[] theta, string phase, float[] residual);

        public sealed class Result
        {
            public double[] Best;
            public double InitialLoss, BestLoss = double.PositiveInfinity, PilotH, Damping;
            public int Queries, Jacobians, Accepted, Rejected, BlockedProbes, ShortenedTrials;
            public string StopReason = "budget";
        }

        public static Result Run(double[] initial, int residualSize, ResidualQuery query, int budget, int seed,
            Func<double[], bool> feasible = null)
        {
            if (initial == null || initial.Length == 0 || residualSize < 1 || query == null || budget < 1)
                throw new ArgumentException("Initial parameters, residual callback, and positive sizes/budget are required.");
            foreach (double u in initial)
                if (!(u >= 0 && u <= 1)) throw new ArgumentException("Parameters must be normalized to [0,1].");
            if (feasible != null && !feasible(initial)) throw new ArgumentException("Initial rig is not feasible.");
            int n = initial.Length;
            var result = new Result();
            var center = new float[residualSize];
            var bestResidual = new float[residualSize];
            var trial = new float[residualSize];
            var columns = new float[n][];
            for (int i = 0; i < n; ++i) columns[i] = new float[residualSize];
            var x = (double[])initial.Clone();
            var probe = new double[n];
            double f = result.InitialLoss = Evaluate(x, "initial", center);
            double h = result.PilotH = BlackBoxOptimizer.Pilot(x, f, new Random(seed), budget - result.Queries,
                (v, phase) => Evaluate(v, phase, trial), feasible == null ? null : (Func<double[], bool>)Legal);
            double mu = 0;
            while (budget - result.Queries >= 2 * n + 1)
            {
                if (result.BestLoss <= 1e-12) { result.StopReason = "near_zero_residual"; break; }
                if (result.BestLoss < f)
                {
                    // Differencing and rejected trial points still represent real evaluated rigs.
                    x = (double[])result.Best.Clone(); f = result.BestLoss;
                    Array.Copy(bestResidual, center, residualSize);
                }
                double step = Math.Max(.00125, h * Math.Min(1, Math.Pow(f / Math.Max(1e-20, result.InitialLoss), .25)));
                ++result.Jacobians;
                Array.Copy(x, probe, n);
                for (int i = 0; i < n; ++i)
                {
                    var col = columns[i];
                    if (feasible != null)
                        ConstrainedColumn(i, step, col);
                    else if (x[i] >= step && 1 - x[i] >= step)
                    {
                        probe[i] = x[i] + step; Evaluate(probe, "jac_plus", col);
                        probe[i] = x[i] - step; Evaluate(probe, "jac_minus", trial);
                        float scale = (float)(.5 / step);
                        for (int k = 0; k < residualSize; ++k) col[k] = (col[k] - trial[k]) * scale;
                    }
                    else
                    {
                        double dx = x[i] < .5 ? step : -step;
                        probe[i] = x[i] + dx; Evaluate(probe, "jac_one_sided", col);
                        probe[i] = x[i] + 2 * dx; Evaluate(probe, "jac_one_sided", trial);
                        float scale = (float)(.5 / dx);
                        for (int k = 0; k < residualSize; ++k) col[k] = (4 * col[k] - trial[k] - 3 * center[k]) * scale;
                    }
                    probe[i] = x[i];
                }
                var g = new double[n];
                var normal = new double[n, n];
                double diagonal = 0;
                for (int i = 0; i < n; ++i)
                {
                    g[i] = Dot(columns[i], center);
                    for (int j = 0; j <= i; ++j) normal[i, j] = normal[j, i] = Dot(columns[i], columns[j]);
                    diagonal = Math.Max(diagonal, normal[i, i]);
                }
                if (!(diagonal > 1e-18)) { result.StopReason = "zero_image_jacobian"; break; }
                double minMu = Math.Max(1e-12, diagonal * 1e-8);
                mu = Math.Max(minMu, mu == 0 ? diagonal * .002 : mu);
                bool accepted = false;
                double previousBest = result.BestLoss;
                for (int attempt = 0; attempt < 9 && result.Queries < budget; ++attempt)
                {
                    var delta = SolveDamped(normal, g, x, mu);
                    double max = 0;
                    for (int i = 0; i < n; ++i) max = Math.Max(max, Math.Abs(delta[i]));
                    double cap = max > .25 ? .25 / max : 1;
                    for (int i = 0; i < n; ++i)
                    {
                        probe[i] = BlackBoxOptimizer.Clamp(x[i] + delta[i] * cap, 0, 1);
                        delta[i] = probe[i] - x[i];
                    }
                    if (feasible != null && !Legal(probe))
                    {
                        // Shrink in parameter space; never substitute a penalty image for an illegal rig.
                        bool legal = false;
                        for (int shrink = 0; shrink < 12 && !legal; ++shrink)
                        {
                            for (int i = 0; i < n; ++i) probe[i] = x[i] + (delta[i] *= .5);
                            legal = Legal(probe);
                        }
                        if (!legal) { mu *= 8; continue; }
                        ++result.ShortenedTrials;
                    }
                    // The projected/damped candidate is judged by its actual feasible displacement.
                    double predicted = 0;
                    for (int i = 0; i < n; ++i)
                    {
                        predicted -= 2 * g[i] * delta[i];
                        for (int j = 0; j < n; ++j) predicted -= delta[i] * normal[i, j] * delta[j];
                    }
                    if (!(predicted > 1e-18)) { mu *= 8; continue; }
                    double next = Evaluate(probe, "lm_trial", trial);
                    double gain = (f - next) / predicted;
                    if (next < f && gain > 0)
                    {
                        Array.Copy(probe, x, n);
                        (trial, center) = (center, trial);
                        f = next;
                        accepted = true;
                        ++result.Accepted;
                        mu = Math.Max(minMu, mu * (gain > .75 ? .33 : gain < .25 ? 3 : 1));
                        break;
                    }
                    ++result.Rejected;
                    mu *= 5;
                }
                if (!accepted && result.BestLoss >= previousBest && result.BestLoss >= f)
                {
                    if (budget - result.Queries < 2 * n + 1) break;
                    if (step <= .00125) { result.StopReason = "no_decreasing_step"; break; }
                    h *= .5;
                    mu = 0;
                }
            }
            result.Damping = mu;
            return result;

            bool Legal(double[] v)
            {
                foreach (double u in v) if (!(u >= 0 && u <= 1)) { ++result.BlockedProbes; return false; }
                if (feasible(v)) return true;
                ++result.BlockedProbes;
                return false;
            }

            void ConstrainedColumn(int i, double hColumn, float[] col)
            {
                for (int shrink = 0; shrink < 10; ++shrink, hColumn *= .5)
                {
                    probe[i] = x[i] + hColumn; bool plus = Legal(probe);
                    probe[i] = x[i] - hColumn; bool minus = Legal(probe);
                    if (plus && minus)
                    {
                        probe[i] = x[i] + hColumn; Evaluate(probe, "jac_plus", col);
                        probe[i] = x[i] - hColumn; Evaluate(probe, "jac_minus", trial);
                        float scale = (float)(.5 / hColumn);
                        for (int k = 0; k < residualSize; ++k) col[k] = (col[k] - trial[k]) * scale;
                        return;
                    }
                    if (!plus && !minus) continue;
                    double dx = plus ? hColumn : -hColumn;
                    probe[i] = x[i] + 2 * dx;
                    bool secondOrder = Legal(probe);
                    probe[i] = x[i] + dx; Evaluate(probe, "jac_one_sided", col);
                    if (secondOrder)
                    {
                        probe[i] = x[i] + 2 * dx; Evaluate(probe, "jac_one_sided", trial);
                        float scale = (float)(.5 / dx);
                        for (int k = 0; k < residualSize; ++k) col[k] = (4 * col[k] - trial[k] - 3 * center[k]) * scale;
                    }
                    else
                    {
                        float scale = (float)(1 / dx);
                        for (int k = 0; k < residualSize; ++k) col[k] = (col[k] - center[k]) * scale;
                    }
                    return;
                }
                Array.Clear(col, 0, col.Length);
            }

            double Evaluate(double[] v, string phase, float[] buffer)
            {
                if (feasible != null && !Legal(v)) throw new InvalidOperationException("LM attempted an infeasible render.");
                if (++result.Queries > budget) throw new InvalidOperationException("Image Jacobian budget exceeded.");
                double loss = query(v, phase, buffer);
                if (!(loss >= 0) || double.IsInfinity(loss)) throw new InvalidOperationException("Expected finite nonnegative RGB loss.");
                if (loss < result.BestLoss)
                {
                    result.BestLoss = loss;
                    result.Best = (double[])v.Clone();
                    Array.Copy(buffer, bestResidual, residualSize);
                }
                return loss;
            }
        }

        private static double Dot(float[] a, float[] b)
        {
            double sum = 0, s1 = 0, s2 = 0, s3 = 0;
            int k = 0;
            for (; k + 3 < a.Length; k += 4)
            {
                sum += (double)a[k] * b[k]; s1 += (double)a[k + 1] * b[k + 1];
                s2 += (double)a[k + 2] * b[k + 2]; s3 += (double)a[k + 3] * b[k + 3];
            }
            for (; k < a.Length; ++k) sum += (double)a[k] * b[k];
            return sum + s1 + s2 + s3;
        }

        private static double[] SolveDamped(double[,] h, double[] g, double[] x, double mu)
        {
            int n = g.Length;
            var l = new double[n, n];
            var fixedAxis = new bool[n];
            var step = new double[n];
            for (int i = 0; i < n; ++i) fixedAxis[i] = x[i] == 0 && g[i] > 0 || x[i] == 1 && g[i] < 0;
            for (int i = 0; i < n; ++i)
                for (int j = 0; j <= i; ++j)
                {
                    double v = i == j ? h[i, i] + mu : fixedAxis[i] || fixedAxis[j] ? 0 : h[i, j];
                    for (int k = 0; k < j; ++k) v -= l[i, k] * l[j, k];
                    l[i, j] = i == j ? Math.Sqrt(v) : v / l[j, j];
                }
            for (int i = 0; i < n; ++i)
            {
                double v = fixedAxis[i] ? 0 : -g[i];
                for (int j = 0; j < i; ++j) v -= l[i, j] * step[j];
                step[i] = v / l[i, i];
            }
            for (int i = n - 1; i >= 0; --i)
            {
                double v = step[i];
                for (int j = i + 1; j < n; ++j) v -= l[j, i] * step[j];
                step[i] = v / l[i, i];
            }
            return step;
        }
    }
}

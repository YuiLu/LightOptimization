using System;
using System.Collections.Generic;

namespace LightingMvp
{
    /// <summary>Bounded numerical optimization; the objective is only a scalar callback.</summary>
    public static class BlackBoxOptimizer
    {
        public enum Method { CentralLbfgs, SpsaAdam }

        public sealed class Result
        {
            public double[] Best;
            public double InitialLoss, BestLoss, PilotH;
            public int Queries, Iterations;
            public string StopReason;
        }

        private sealed class Search
        {
            internal readonly Func<double[], string, double> Objective;
            internal readonly int Budget;
            internal readonly Result Result = new Result { BestLoss = double.PositiveInfinity, StopReason = "budget" };
            internal int Remaining => Budget - Result.Queries;

            internal Search(Func<double[], string, double> objective, int budget) { Objective = objective; Budget = budget; }

            internal double Evaluate(double[] x, string phase)
            {
                if (Remaining <= 0) throw new InvalidOperationException("Objective budget exceeded.");
                ++Result.Queries;
                double loss = Objective(x, phase);
                if (double.IsNaN(loss) || double.IsInfinity(loss) || loss < 0)
                    throw new InvalidOperationException("Expected a finite nonnegative loss.");
                if (loss < Result.BestLoss) { Result.BestLoss = loss; Result.Best = (double[])x.Clone(); }
                return loss;
            }
        }

        public static Result Run(Method method, double[] initial, Func<double[], string, double> objective, int budget, int seed)
        {
            if (initial == null || initial.Length == 0 || objective == null || budget < 1)
                throw new ArgumentException("Initial parameters, objective and a positive budget are required.");
            foreach (double u in initial)
                if (!(u >= 0 && u <= 1)) throw new ArgumentException("Parameters must be in [0,1].");
            var x = (double[])initial.Clone();
            var search = new Search(objective, budget);
            double f = search.Result.InitialLoss = search.Evaluate(x, "initial");
            var rng = new Random(seed);
            double h = search.Result.PilotH = Pilot(x, f, rng, search.Remaining, search.Evaluate);
            if (method == Method.CentralLbfgs) Lbfgs(search, x, f, h);
            else if (method == Method.SpsaAdam) Spsa(search, x, f, h, rng);
            else throw new ArgumentOutOfRangeException(nameof(method));
            return search.Result;
        }

        internal static double Pilot(double[] x, double f, Random rng, int remaining, Func<double[], string, double> evaluate)
        {
            if (remaining < 6 || InteriorRadius(x) < .001) return .03;
            var signs = Signs(x.Length, rng);
            double chosen = .03;
            double bestScore = double.PositiveInfinity;
            foreach (double candidate in new[] { .01, .03, .1 })
            {
                double h = Math.Min(candidate, InteriorRadius(x));
                double a = evaluate(Along(x, signs, h), "pilot_plus");
                double b = evaluate(Along(x, signs, -h), "pilot_minus");
                // Prefer slopes that are measurable, yet not dominated by the even (curvature) response.
                double odd = Math.Abs(a - b), even = Math.Abs(a + b - 2 * f);
                double score = (even + 1e-5 * Math.Max(f, 1e-8)) / (odd + 1e-15);
                if (score < bestScore) { bestScore = score; chosen = h; }
            }
            return chosen;
        }

        private static double[] FiniteGradient(Search search, double[] x, double f, double h)
        {
            var g = new double[x.Length];
            var probe = (double[])x.Clone();
            for (int i = 0; i < g.Length; ++i)
            {
                if (x[i] >= h && 1 - x[i] >= h)
                {
                    probe[i] = x[i] + h;
                    double plus = search.Evaluate(probe, "fd_plus");
                    probe[i] = x[i] - h;
                    g[i] = (plus - search.Evaluate(probe, "fd_minus")) / (2 * h);
                }
                else
                {
                    // Second-order one-sided stencil, not a clipped centered stencil.
                    double step = x[i] < .5 ? h : -h;
                    probe[i] = x[i] + step;
                    double a = search.Evaluate(probe, "fd_one_sided");
                    probe[i] = x[i] + 2 * step;
                    double b = search.Evaluate(probe, "fd_one_sided");
                    g[i] = (4 * a - b - 3 * f) / (2 * step);
                }
                probe[i] = x[i];
            }
            return g;
        }

        private static void Lbfgs(Search search, double[] x, double f, double h)
        {
            int d = x.Length;
            var sHistory = new List<double[]>();
            var yHistory = new List<double[]>();
            double[] previousX = null, previousG = null;
            // Start at the pilot scale and increase spatial precision as the measured residual decreases.
            while (search.Remaining >= 2 * d + 1)
            {
                ++search.Result.Iterations;
                double stepH = Math.Max(.00125, h * Math.Min(1, Math.Pow(f / Math.Max(1e-20, search.Result.InitialLoss), .25)));
                var g = FiniteGradient(search, x, f, stepH);
                if (previousG != null)
                {
                    var s = Along(x, previousX, -1);
                    var y = Along(g, previousG, -1);
                    if (Dot(s, y) > 1e-8 * Math.Sqrt(Dot(s, s) * Dot(y, y)))
                    {
                        if (sHistory.Count == 8) { sHistory.RemoveAt(0); yHistory.RemoveAt(0); }
                        sHistory.Add(s);
                        yHistory.Add(y);
                    }
                }
                var pg = (double[])g.Clone();
                for (int i = 0; i < d; ++i)
                    if ((x[i] == 0 && g[i] > 0) || (x[i] == 1 && g[i] < 0)) pg[i] = 0;
                var direction = Direction(pg, sHistory, yHistory);
                bool found = TryStep(search, x, f, g, direction, out var next, out var nextF);
                if (!found && search.Remaining > 0)
                {
                    for (int i = 0; i < d; ++i) direction[i] = -pg[i];
                    sHistory.Clear(); yHistory.Clear();
                    found = TryStep(search, x, f, g, direction, out next, out nextF);
                }
                if (!found)
                {
                    // If a real probe improved the loss, restart there; otherwise refine the stencil before giving up.
                    if (search.Result.BestLoss < f)
                    {
                        x = (double[])search.Result.Best.Clone(); f = search.Result.BestLoss;
                        previousX = previousG = null;
                        sHistory.Clear(); yHistory.Clear();
                        continue;
                    }
                    if (stepH <= .00125 || search.Remaining < 2 * d + 1)
                    {
                        if (search.Remaining >= 2 * d + 1) search.Result.StopReason = "no_decreasing_step";
                        break;
                    }
                    h *= .5;
                    previousX = previousG = null;
                    continue;
                }
                previousX = x; previousG = g;
                x = next; f = nextF;
            }
        }

        private static double[] Direction(double[] g, List<double[]> s, List<double[]> y)
        {
            var q = (double[])g.Clone();
            int count = s.Count;
            var alpha = new double[count];
            for (int i = count - 1; i >= 0; --i)
            {
                alpha[i] = Dot(s[i], q) / Dot(s[i], y[i]);
                AddScaled(q, y[i], -alpha[i]);
            }
            double gamma = count > 0 ? Dot(s[count - 1], y[count - 1]) / Dot(y[count - 1], y[count - 1]) : 1;
            for (int j = 0; j < q.Length; ++j) q[j] *= gamma;
            for (int i = 0; i < count; ++i)
                AddScaled(q, s[i], alpha[i] - Dot(y[i], q) / Dot(s[i], y[i]));
            for (int j = 0; j < q.Length; ++j) q[j] = -q[j];
            return q;
        }

        private static bool TryStep(Search search, double[] x, double f, double[] g, double[] d, out double[] next, out double loss)
        {
            next = new double[x.Length]; loss = f;
            double max = 0;
            foreach (double v in d) max = Math.Max(max, Math.Abs(v));
            if (max < 1e-15) return false;
            double step = Math.Min(1, .18 / max);
            for (int i = 0; i < 9 && search.Remaining > 0; ++i, step *= .5)
            {
                double descent = 0;
                for (int j = 0; j < x.Length; ++j)
                {
                    next[j] = Clamp(x[j] + step * d[j], 0, 1);
                    descent += g[j] * (next[j] - x[j]);
                }
                if (descent >= 0) return false;
                loss = search.Evaluate(next, "line_search");
                if (loss < f && loss <= f + 1e-4 * descent) return true;
            }
            return false;
        }

        private static void Spsa(Search search, double[] x, double f, double h, Random rng)
        {
            int d = x.Length, total = search.Remaining;
            var m = new double[d]; var v = new double[d]; var g = new double[d];
            // SPSA centers stay just inside the domain; probes use the same feasible radius in both directions.
            const double margin = .004;
            if (InteriorRadius(x) < margin && search.Remaining > 0)
            {
                for (int i = 0; i < d; ++i) x[i] = Clamp(x[i], margin, 1 - margin);
                f = search.Evaluate(x, "spsa_center");
            }
            int window = 0;
            double lastBest = search.Result.BestLoss;
            while (search.Remaining >= 3)
            {
                int t = ++search.Result.Iterations;
                double fraction = 1 - (double)search.Remaining / total;
                double dh = Math.Min(InteriorRadius(x), Math.Max(.002, h * Math.Pow(Math.Max(0, f / Math.Max(1e-20, search.Result.InitialLoss)), .25)));
                int pairs = Math.Min(d <= 4 ? 2 : 3, (search.Remaining - 1) / 2);
                Array.Clear(g, 0, d);
                for (int pair = 0; pair < pairs; ++pair)
                {
                    var signs = Signs(d, rng);
                    double a = search.Evaluate(Along(x, signs, dh), "spsa_plus");
                    double b = search.Evaluate(Along(x, signs, -dh), "spsa_minus");
                    AddScaled(g, signs, (a - b) / (2 * dh * pairs));
                }
                double lr = .038 * Math.Pow(Math.Max(.04, 1 - fraction), 1.15);
                for (int i = 0; i < d; ++i)
                {
                    m[i] = .85 * m[i] + .15 * g[i];
                    v[i] = .98 * v[i] + .02 * g[i] * g[i];
                    double delta = lr * (m[i] / (1 - Math.Pow(.85, t))) / (Math.Sqrt(v[i] / (1 - Math.Pow(.98, t))) + 1e-10);
                    x[i] = Clamp(x[i] - delta, margin, 1 - margin);
                }
                f = search.Evaluate(x, "adam_step");
                // Return to the best *queried* point after a stagnant window rather than following noisy momentum indefinitely.
                if (++window == 12)
                {
                    if (search.Result.BestLoss >= lastBest * .98 && search.Remaining > 0)
                    {
                        for (int i = 0; i < d; ++i) x[i] = Clamp(search.Result.Best[i], margin, 1 - margin);
                        f = search.Evaluate(x, "spsa_restart");
                        Array.Clear(m, 0, d); Array.Clear(v, 0, d);
                    }
                    lastBest = search.Result.BestLoss;
                    window = 0;
                }
            }
        }

        private static double InteriorRadius(double[] x)
        {
            double h = .25;
            foreach (double u in x) h = Math.Min(h, Math.Min(u, 1 - u));
            return h;
        }

        private static double[] Signs(int d, Random rng)
        {
            var result = new double[d];
            for (int i = 0; i < d; ++i) result[i] = rng.Next(2) == 0 ? -1 : 1;
            return result;
        }

        internal static double Clamp(double x, double a, double b) => Math.Min(b, Math.Max(a, x));
        private static double Dot(double[] a, double[] b) { double sum = 0; for (int i = 0; i < a.Length; ++i) sum += a[i] * b[i]; return sum; }
        private static void AddScaled(double[] a, double[] b, double h) { for (int i = 0; i < a.Length; ++i) a[i] += h * b[i]; }
        private static double[] Along(double[] x, double[] d, double h) { var next = (double[])x.Clone(); AddScaled(next, d, h); return next; }
    }
}

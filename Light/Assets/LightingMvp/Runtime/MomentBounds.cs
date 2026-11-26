using System;

namespace LightingMvp
{
    // n = 1 and n = 2 infinite-domain moment bounds and their reverse pass.
    // Worchel and Alexa, SIGGRAPH 2025, sections 4.3 and 5.1 (doi:10.1145/3730899).
    public static class MomentBounds
    {
        public enum Status { Success, InfiniteRoot, NotPositive, NoRealRoots, SingularWeights, NonFinite }

        public static Status Evaluate(ReadOnlySpan<double> input, double eta, double alpha, double beta,
            out double bound, Span<double> gradient, out double depthGradient)
        {
            int n = (input.Length - 1) / 2;
            if (input.Length != 3 && input.Length != 5) throw new ArgumentException("Expected three or five power moments.");
            if (gradient.Length < input.Length) throw new ArgumentException("The moment gradient buffer is too short.");
            if (!(alpha >= 0 && alpha <= 1 && beta >= 0 && beta <= 1))
                throw new ArgumentOutOfRangeException(nameof(alpha), "Moment bias and bound blend must be in [0,1].");
            gradient.Clear();
            bound = depthGradient = 0;
            if (!Finite(eta)) return Status.NonFinite;

            Span<double> moments = stackalloc double[5];
            for (int i = 0; i < input.Length; ++i)
            {
                double prior = i == 0 || n == 1 && i == 2 ? 1 : i % 2 == 0 ? .375 : 0;
                moments[i] = (1 - alpha) * input[i] + alpha * prior;
                if (!Finite(moments[i])) return Status.NonFinite;
            }
            Span<double> lower = stackalloc double[9];
            for (int i = 0; i <= n; ++i)
            {
                double normSquared = 0;
                for (int j = 0; j < i; ++j)
                {
                    double product = 0;
                    for (int k = 0; k < j; ++k) product += lower[3 * i + k] * lower[3 * j + k];
                    double l = (moments[i + j] - product) / lower[3 * j + j];
                    lower[3 * i + j] = l;
                    normSquared += l * l;
                }
                double diagonal = moments[2 * i] - normSquared;
                if (!Finite(diagonal)) return Status.NonFinite;
                if (diagonal <= 0) return Status.NotPositive;
                lower[3 * i + i] = Math.Sqrt(diagonal);
            }
            Span<double> coefficients = stackalloc double[3];
            Solve(lower, n, 1, eta, eta * eta, coefficients);
            Span<double> nodes = stackalloc double[3];
            nodes[0] = eta;
            if (n == 1) nodes[1] = -coefficients[0] / coefficients[1];
            else
            {
                double c0 = coefficients[0], c1 = coefficients[1], c2 = coefficients[2];
                double discriminant = c1 * c1 - 4 * c0 * c2;
                if (!Finite(discriminant)) return Status.NonFinite;
                if (discriminant < 0) return Status.NoRealRoots;
                double root = Math.Sqrt(discriminant);
                double q = -.5 * (c1 + (c1 < 0 ? -root : root));
                if (q == 0) return Status.SingularWeights;
                nodes[1] = c0 / q;
                nodes[2] = q / c2;
            }
            Span<int> order = stackalloc int[3];
            for (int i = 0; i <= n; ++i)
            {
                if (double.IsNaN(nodes[i])) return Status.NonFinite;
                order[i] = i;
            }
            for (int i = 0; i < n; ++i)
            for (int j = i + 1; j <= n; ++j)
                if (Math.Abs(nodes[order[i]]) > Math.Abs(nodes[order[j]]))
                { int swap = order[i]; order[i] = order[j]; order[j] = swap; }
            int size = n + 1;
            while (size > 1 && double.IsInfinity(nodes[order[size - 1]])) size--;

            // Björck–Pereyra applied to the finite support; an infinite node has zero mass.
            Span<double> sorted = stackalloc double[3];
            Span<double> weights = stackalloc double[3];
            Span<double> dual = stackalloc double[3];
            for (int i = 0; i < size; ++i)
            {
                sorted[i] = nodes[order[i]];
                for (int j = 0; j < i; ++j)
                    if (sorted[i] == sorted[j]) return Status.SingularWeights;
                weights[i] = moments[i];
                dual[i] = order[i] == 0 ? beta : sorted[i] < eta ? 1 : 0;
            }
            for (int k = 0; k < size - 1; ++k)
            for (int j = size - 1; j > k; --j)
            {
                weights[j] -= sorted[k] * weights[j - 1];
                dual[j] = (dual[j] - dual[j - 1]) / (sorted[j] - sorted[j - k - 1]);
            }
            for (int k = size - 2; k >= 0; --k)
            {
                for (int j = k + 1; j < size; ++j)
                {
                    weights[j] /= sorted[j] - sorted[j - k - 1];
                    weights[j - 1] -= weights[j];
                }
                for (int j = k; j < size - 1; ++j) dual[j] -= sorted[k] * dual[j + 1];
            }

            // V^T p = dL/dw. A finite root contributes through K(x)=0, and
            // H^(-T) dc propagates through the Hankel solve without differentiating a factorization.
            double dualQuadratic = size == 3 ? dual[2] : 0;
            double dualLinear = size >= 2 ? dual[1] : 0;
            double dc0 = 0, dc1 = 0, dc2 = 0;
            for (int i = 0; i < size; ++i)
            {
                double x = sorted[i], w = weights[i];
                double mask = order[i] == 0 ? beta : x < eta ? 1 : 0;
                bound += mask * w;
                double dp = dualLinear + 2 * dualQuadratic * x;
                if (order[i] == 0) depthGradient -= w * dp;
                else if (Math.Abs(w) > 1e-150)
                {
                    double dk = coefficients[1] + (n == 2 ? 2 * coefficients[2] * x : 0);
                    if (dk == 0) return Status.SingularWeights;
                    double multiplier = dp / dk * w;
                    dc0 += multiplier;
                    dc1 += multiplier * x;
                    if (n == 2) dc2 += multiplier * x * x;
                }
            }
            Span<double> rhsAdjoint = stackalloc double[3];
            Solve(lower, n, dc0, dc1, dc2, rhsAdjoint);
            depthGradient += rhsAdjoint[1] + (n == 2 ? 2 * eta * rhsAdjoint[2] : 0);
            for (int i = 0; i < size; ++i) gradient[i] = dual[i];
            for (int i = 0; i <= n; ++i)
            for (int j = 0; j <= n; ++j)
                gradient[i + j] -= rhsAdjoint[i] * coefficients[j];
            for (int i = 0; i < input.Length; ++i)
            {
                gradient[i] *= 1 - alpha;
                if (!Finite(gradient[i])) return Status.NonFinite;
            }
            if (!Finite(bound) || !Finite(depthGradient)) return Status.NonFinite;
            return size == n + 1 ? Status.Success : Status.InfiniteRoot;
        }

        private static void Solve(ReadOnlySpan<double> l, int n, double b0, double b1, double b2, Span<double> x)
        {
            double y0 = b0 / l[0];
            double y1 = (b1 - l[3] * y0) / l[4];
            x[2] = n == 2 ? (b2 - (l[6] * y0 + l[7] * y1)) / (l[8] * l[8]) : 0;
            x[1] = (y1 - (n == 2 ? l[7] * x[2] : 0)) / l[4];
            x[0] = (y0 - (l[3] * x[1] + (n == 2 ? l[6] * x[2] : 0))) / l[0];
        }

        private static bool Finite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
    }
}

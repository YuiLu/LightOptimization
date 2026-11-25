using System;

namespace LightingMvp.Editor
{
    // Small dense bounded least squares over Gram statistics, in normalized image space.
    // Coordinate minimization is monotone, handles nearly dependent columns and needs no regularizer.
    internal static class BoundedLightWeights
    {
        internal static double Solve(double[,] gram, double[] rhs, int n, double lower, double upper, double[] weights)
        {
            var gradient = new double[n];
            for (int i = 0; i < n; i++) weights[i] = Math.Max(lower, Math.Min(upper, weights[i]));
            for (int i = 0; i < n; i++)
            {
                gradient[i] = -rhs[i];
                for (int j = 0; j < n; j++) gradient[i] += gram[i, j] * weights[j];
            }
            for (int sweep = 0; sweep < 512; sweep++)
            {
                double largest = 0;
                for (int i = 0; i < n; i++)
                {
                    if (gram[i, i] <= 1e-22) continue;
                    double next = Math.Max(lower, Math.Min(upper, weights[i] - gradient[i] / gram[i, i]));
                    double delta = next - weights[i];
                    if (delta == 0) continue;
                    weights[i] = next;
                    for (int j = 0; j < n; j++) gradient[j] += gram[j, i] * delta;
                    largest = Math.Max(largest, Math.Abs(delta));
                }
                if (largest < 1e-8) break;
            }
            // Return ||b||^2 - ||b - A*w||^2, without constructing an image per candidate.
            double gain = 0;
            for (int i = 0; i < n; i++) gain += weights[i] * (rhs[i] - gradient[i]);
            return gain;
        }

        internal static void CheckKnownCases()
        {
            // Existing a=(1,0) at 1.3; candidate b=(2,1) has a negative dot with the residual of target (1,.5).
            // Changing both strengths makes the same position viable, including with a nonzero active floor.
            var gram = new double[,] { { 1, 2 }, { 2, 5 } };
            var rhs = new[] { 1.0, 2.5 };
            var weights = new[] { 1.3, 0.0 };
            var relaxed = Solve(gram, rhs, 2, 0, 9, weights);
            if (Math.Abs(relaxed - 1.25) > 1e-10 || weights[0] > 1e-6 || Math.Abs(weights[1] - .5) > 1e-6)
                throw new InvalidOperationException("Unexpected relaxed NNLS solution.");
            var bounded = Solve(gram, rhs, 2, .055, 9, weights);
            if (weights[0] < .055 || Math.Abs(weights[1] - .478) > 1e-6 || bounded >= relaxed || 1.25 - bounded >= .34)
                throw new InvalidOperationException("Unexpected active-lamp bounded NNLS solution.");
        }
    }
}

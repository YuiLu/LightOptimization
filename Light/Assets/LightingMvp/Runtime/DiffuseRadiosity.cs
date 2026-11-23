using System;
using UnityEngine;

namespace LightingMvp
{
    /// <summary>Fixed-geometry diffuse transport on Unity surface patches. Rebuild after geometry or albedo edits.</summary>
    public sealed class DiffuseRadiosity
    {
        private const float Epsilon = 1e-4f;
        private readonly int count;
        private readonly float[] factors;
        private readonly Vector3[] albedo;
        private readonly double[][] lu;

        public float MaxRawRowSum { get; }
        public float MaxRowSum { get; }
        public float MaxReciprocityError { get; }
        public int BlockedPairs { get; }
        public int RefinedPairs { get; }

        public DiffuseRadiosity(SurfaceSampleSet samples)
        {
            const float rowLimit = 0.999f;
            count = samples.Count;
            factors = new float[checked(count * count)];
            albedo = new Vector3[count];
            var rowSums = new float[count];
            var quadrature2 = BuildQuadrature(2);
            var quadrature4 = BuildQuadrature(4);
            Physics.SyncTransforms();
            for (int i = 0; i < count; i++)
            {
                Color color = samples.albedos[i];
                albedo[i] = new Vector3(Mathf.Clamp01(color.r), Mathf.Clamp01(color.g), Mathf.Clamp01(color.b));
                for (int j = i + 1; j < count; j++)
                {
                    Vector3 from = samples.positions[i], to = samples.positions[j];
                    Vector3 ni = samples.normals[i], nj = samples.normals[j];
                    Vector3 delta = to - from;
                    float distanceSquared = delta.sqrMagnitude;
                    float kernel = GeometricKernel(delta, distanceSquared, ni, nj);
                    if (kernel <= 0) continue;
                    float area = Mathf.Max(samples.areas[i], samples.areas[j]);
                    float centerFactor = area * kernel;
                    Vector3[][] quadrature = distanceSquared < 4 * area || centerFactor > 0.1f ? quadrature4 :
                        distanceSquared < 16 * area || centerFactor > 0.005f ? quadrature2 : null;
                    bool blocked = false;
                    if (quadrature != null)
                    {
                        RefinedPairs++;
                        kernel = 0;
                        Vector3[] pointsI = quadrature[i], pointsJ = quadrature[j];
                        foreach (Vector3 pointI in pointsI)
                        foreach (Vector3 pointJ in pointsJ)
                        {
                            Vector3 segment = pointJ - pointI;
                            float contribution = GeometricKernel(segment, segment.sqrMagnitude, ni, nj);
                            if (contribution == 0) continue;
                            if (IsOccluded(pointI, pointJ, ni, nj)) blocked = true;
                            else kernel += contribution;
                        }
                        kernel /= pointsI.Length * pointsJ.Length;
                    }
                    else if (IsOccluded(from, to, ni, nj)) { kernel = 0; blocked = true; }
                    if (blocked) BlockedPairs++;
                    float ij = kernel * samples.areas[j], ji = kernel * samples.areas[i];
                    factors[i * count + j] = ij;
                    factors[j * count + i] = ji;
                    rowSums[i] += ij;
                    rowSums[j] += ji;
                }
                MaxRawRowSum = Mathf.Max(MaxRawRowSum, rowSums[i]);
            }

            // Even with near-field quadrature, close pairs can be overestimated. A symmetric correction preserves
            // A_i F_ij = A_j F_ji and bounds every row, including pure-white materials.
            var boundedSums = new float[count];
            for (int i = 0; i < count; i++)
            {
                for (int j = i + 1; j < count; j++)
                {
                    float scale = Mathf.Min(1, rowLimit / Mathf.Max(1e-10f, Mathf.Max(rowSums[i], rowSums[j])));
                    float ij = factors[i * count + j] *= scale, ji = factors[j * count + i] *= scale;
                    boundedSums[i] += ij;
                    boundedSums[j] += ji;
                    MaxReciprocityError = Mathf.Max(MaxReciprocityError, Mathf.Abs(samples.areas[i] * ij - samples.areas[j] * ji));
                }
                MaxRowSum = Mathf.Max(MaxRowSum, boundedSums[i]);
            }

            lu = new double[3][];
            for (int c = 0; c < 3; c++)
            {
                var matrix = new double[factors.Length];
                for (int i = 0; i < count; i++)
                for (int j = 0; j < count; j++)
                    matrix[i * count + j] = (i == j ? 1.0 : 0.0) - (double)albedo[i][c] * factors[i * count + j];
                FactorInPlace(matrix);
                lu[c] = matrix;
            }

            Vector3[][] BuildQuadrature(int order)
            {
                if (samples.patches == null) return null;
                var points = new Vector3[count][];
                for (int i = 0; i < count; i++) points[i] = samples.patches[i].Quadrature(order);
                return points;
            }
        }

        private static float GeometricKernel(Vector3 delta, float distanceSquared, Vector3 ni, Vector3 nj)
        {
            if (distanceSquared <= 1e-12f) return 0;
            Vector3 direction = delta / Mathf.Sqrt(distanceSquared);
            float cosI = Vector3.Dot(ni, direction), cosJ = Vector3.Dot(nj, -direction);
            return cosI > 0 && cosJ > 0 ? cosI * cosJ / (Mathf.PI * distanceSquared) : 0;
        }

        private static bool IsOccluded(Vector3 from, Vector3 to, Vector3 ni, Vector3 nj)
        {
            float epsilon = Mathf.Min(Epsilon, (to - from).magnitude * 0.01f);
            Vector3 start = from + epsilon * ni, segment = to + epsilon * nj - start;
            float distance = segment.magnitude;
            return distance > 2 * epsilon && Physics.Raycast(start, segment / distance, distance - epsilon, ~0, QueryTriggerInteraction.Ignore);
        }

        /// <summary>B=(I-diag(rho) F)^-1 diag(rho/pi) E.</summary>
        public Vector3[] Evaluate(Vector3[] directIrradiance) => Solve(directIrradiance, false);

        /// <summary>Transpose of the complete mapping from direct irradiance to total outgoing radiance.</summary>
        public Vector3[] Backpropagate(Vector3[] outgoingGradient) => Solve(outgoingGradient, true);

        /// <summary>Applies one diffuse surface-to-surface reflection; used by the finite-bounce check.</summary>
        public Vector3[] Reflect(Vector3[] outgoingRadiance)
        {
            CheckLength(outgoingRadiance);
            var next = new Vector3[count];
            for (int i = 0; i < count; i++)
            {
                Vector3 incoming = Vector3.zero;
                for (int j = 0; j < count; j++) incoming += factors[i * count + j] * outgoingRadiance[j];
                next[i] = Vector3.Scale(albedo[i], incoming);
            }
            return next;
        }

        private void CheckLength(Vector3[] input)
        {
            if (input.Length != count) throw new ArgumentException("Field must use the same surface patches as the transport.");
        }

        private void FactorInPlace(double[] matrix)
        {
            // I-rho*F is strictly row diagonally dominant; this factorization needs no pivoting.
            for (int k = 0; k < count; k++)
            {
                int pivot = k * count;
                for (int i = k + 1; i < count; i++)
                {
                    int row = i * count;
                    double scale = matrix[row + k] /= matrix[pivot + k];
                    if (scale == 0) continue;
                    for (int j = k + 1; j < count; j++) matrix[row + j] -= scale * matrix[pivot + j];
                }
            }
        }

        private Vector3[] Solve(Vector3[] input, bool transpose)
        {
            CheckLength(input);
            var result = new Vector3[count];
            var rhs = new double[count];
            for (int c = 0; c < 3; c++)
            {
                double[] matrix = lu[c];
                for (int i = 0; i < count; i++) rhs[i] = transpose ? input[i][c] : (double)input[i][c] * albedo[i][c] / Math.PI;
                if (transpose)
                {
                    // U^T then L^T. Accumulate by matrix row to keep both solves contiguous.
                    for (int i = 0; i < count; i++)
                    {
                        int row = i * count;
                        double value = rhs[i] /= matrix[row + i];
                        for (int j = i + 1; j < count; j++) rhs[j] -= matrix[row + j] * value;
                    }
                    for (int i = count - 1; i >= 0; i--)
                        for (int j = 0; j < i; j++) rhs[j] -= matrix[i * count + j] * rhs[i];
                }
                else
                {
                    for (int i = 0; i < count; i++)
                        for (int j = 0; j < i; j++) rhs[i] -= matrix[i * count + j] * rhs[j];
                    for (int i = count - 1; i >= 0; i--)
                    {
                        for (int j = i + 1; j < count; j++) rhs[i] -= matrix[i * count + j] * rhs[j];
                        rhs[i] /= matrix[i * count + i];
                    }
                }
                for (int i = 0; i < count; i++) result[i][c] = (float)(transpose ? rhs[i] * albedo[i][c] / Math.PI : rhs[i]);
            }
            return result;
        }
    }
}

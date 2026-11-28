using System;
using System.IO;
using System.Linq;
using Lm = LightingMvp.ImageJacobianLm;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            private sealed partial class E27Trial
            {
                private double[] AggregateGradient(string policy, double[][] per)
                {
                    var scale = TrustScale();
                    var g = per.Select(v => v.Select((x, j) => x * scale[j]).ToArray()).ToArray();
                    var value = E27Aggregate(policy, g);
                    return value.Select((v, j) => v / scale[j]).ToArray();
                }
            }

            private static double E27Dot(double[] a, double[] b)
            { double x = 0; for (int i = 0; i < a.Length; ++i) x += a[i] * b[i]; return x; }
            private static double[] E27Aggregate(string policy, double[][] g)
            {
                int m = g.Length, n = g[0].Length;
                var sum = new double[n];
                foreach (var v in g) for (int j = 0; j < n; ++j) sum[j] += v[j];
                if (policy == "G0") return sum;
                if (policy == "G1")
                {
                    // PCGrad: project each task against the other original task gradients.
                    // Deterministic cyclic task order, frozen for the comparison.
                    Array.Clear(sum, 0, n);
                    for (int i = 0; i < m; ++i)
                    {
                        var v = (double[])g[i].Clone();
                        for (int k = 1; k < m; ++k)
                        {
                            var other = g[(i + k) % m]; double dot = E27Dot(v, other);
                            if (dot >= 0) continue;
                            double a = dot / E27Dot(other, other);
                            for (int j = 0; j < n; ++j) v[j] -= a * other[j];
                        }
                        for (int j = 0; j < n; ++j) sum[j] += v[j];
                    }
                    return sum;
                }
                if (policy == "G3")
                {
                    // Equal-preference UPGrad: project each task onto the dual cone and average.
                    var cone = g.Where(v => E27Dot(v, v) > 1e-60).Select(v => v.Select(x => x / Math.Sqrt(E27Dot(v, v))).ToArray()).ToArray();
                    Array.Clear(sum, 0, n);
                    foreach (var v in g)
                    {
                        double tol = 1e-10 * Math.Max(1e-30, Math.Sqrt(E27Dot(v, v)));
                        var best = new double[n]; double dBest = E27Dot(v, v);
                        for (int mask = 0; mask < (1 << cone.Length); ++mask)
                        {
                            var idx = Enumerable.Range(0, cone.Length).Where(i => (mask & (1 << i)) != 0).ToArray();
                            var a = new double[idx.Length, idx.Length]; var b = new double[idx.Length];
                            for (int i = 0; i < idx.Length; ++i)
                            {
                                b[i] = -E27Dot(cone[idx[i]], v);
                                for (int j = 0; j < idx.Length; ++j) a[i, j] = E27Dot(cone[idx[i]], cone[idx[j]]);
                            }
                            var w = E27Solve(a, b);
                            if (w == null || w.Any(x => x < -tol)) continue;
                            var p = (double[])v.Clone();
                            for (int i = 0; i < idx.Length; ++i)
                                for (int j = 0; j < n; ++j) p[j] += w[i] * cone[idx[i]][j];
                            if (cone.Any(q => E27Dot(q, p) < -tol)) continue;
                            double d = p.Select((x, j) => (x - v[j]) * (x - v[j])).Sum();
                            if (d <= dBest * (1 + 1e-12)) { best = p; dBest = d; }
                        }
                        for (int j = 0; j < n; ++j) sum[j] += best[j];
                    }
                    return sum.Select(v => v / m).ToArray();
                }
                // For the three camera gradients, minimize the CAGrad simplex objective over
                // their convex hull: three vertices, three scalar edge minima, and the affine
                // interior stationary point when it exists. A fixed iteration projected
                // descent was inaccurate on the norm-unbalanced synthetic case.
                if (m != 3) throw new ArgumentException("E27 CAGrad is registered for three views.");
                double norm = Math.Sqrt(E27Dot(sum, sum)) / m;
                if (norm <= 1e-30) return new double[n];
                const double c = .4;
                double max = g.Max(v => Math.Sqrt(E27Dot(v, v)));
                var a0 = sum.Select(v => v / (m * max)).ToArray();
                var scaled = g.Select(row => row.Select(v => v / max).ToArray()).ToArray();
                double cn = c * norm / max;
                var nearest = scaled[0]; double minimum = Score(nearest);
                double Score(double[] v) => E27Dot(a0, v) + cn * Math.Sqrt(E27Dot(v, v));
                void Try(double[] v)
                {
                    double f = Score(v);
                    if (f < minimum) { nearest = v; minimum = f; }
                }
                foreach (var v in scaled) Try(v);
                for (int i = 0; i < m; ++i)
                for (int j = i + 1; j < m; ++j)
                {
                    var x = scaled[i]; var y = scaled[j];
                    double[] At(double t) => x.Select((v, k) => (1 - t) * v + t * y[k]).ToArray();
                    const double ratio = .6180339887498948482;
                    double left = 0, right = 1, l = 1 - ratio, r = ratio;
                    double fl = Score(At(l)), fr = Score(At(r));
                    for (int k = 0; k < 72; ++k)
                    {
                        if (fl <= fr)
                        { right = r; r = l; fr = fl; l = right - ratio * (right - left); fl = Score(At(l)); }
                        else
                        { left = l; l = r; fl = fr; r = left + ratio * (right - left); fr = Score(At(r)); }
                    }
                    Try(At((left + right) * .5));
                }
                var v0 = scaled[0];
                var d1 = scaled[1].Select((v, i) => v - v0[i]).ToArray();
                var d2 = scaled[2].Select((v, i) => v - v0[i]).ToArray();
                double l1 = Math.Sqrt(E27Dot(d1, d1));
                if (l1 > 1e-14)
                {
                    var e1 = d1.Select(v => v / l1).ToArray(); double d2e1 = E27Dot(d2, e1);
                    var ortho = d2.Select((v, i) => v - d2e1 * e1[i]).ToArray();
                    double l2 = Math.Sqrt(E27Dot(ortho, ortho));
                    if (l2 > 1e-14)
                    {
                        var e2 = ortho.Select(v => v / l2).ToArray();
                        double aa = E27Dot(a0, e1), ab = E27Dot(a0, e2);
                        double va = E27Dot(v0, e1), vb = E27Dot(v0, e2);
                        double s2 = cn * cn - aa * aa - ab * ab;
                        if (s2 > 0)
                        {
                            var height = v0.Select((v, i) => v - va * e1[i] - vb * e2[i]).ToArray();
                            double f = Math.Sqrt(E27Dot(height, height) / s2);
                            double w2 = (-f * ab - vb) / l2, w1 = (-f * aa - va - w2 * d2e1) / l1;
                            if (w1 >= 0 && w2 >= 0 && w1 + w2 <= 1)
                                Try(v0.Select((v, i) => v + w1 * d1[i] + w2 * d2[i]).ToArray());
                        }
                    }
                }
                // Scale choice is kept fixed for this arm so even the preconditioned proposal
                // is compared consistently. Avoid multiplication by an unbounded lambda near zero.
                double nearestNorm = Math.Sqrt(E27Dot(nearest, nearest));
                return nearest.Select((v, j) => max * (nearestNorm * a0[j] + cn * v) /
                    Math.Max(1e-30, nearestNorm + cn)).ToArray();
            }
            private static double[] E27Solve(double[,] a, double[] rhs)
            {
                int n = rhs.Length;
                for (int i = 0; i < n; ++i)
                {
                    int k = i; for (int j = i + 1; j < n; ++j) if (Math.Abs(a[j, i]) > Math.Abs(a[k, i])) k = j;
                    if (Math.Abs(a[k, i]) <= 1e-12) return null;
                    (rhs[k], rhs[i]) = (rhs[i], rhs[k]);
                    for (int j = i; j < n; ++j) (a[k, j], a[i, j]) = (a[i, j], a[k, j]);
                    double d = a[i, i];
                    for (int j = i; j < n; ++j) a[i, j] /= d;
                    rhs[i] /= d;
                    for (int row = 0; row < n; ++row)
                    {
                        if (row == i) continue;
                        double t = a[row, i];
                        for (int j = i; j < n; ++j) a[row, j] -= t * a[i, j]; rhs[row] -= t * rhs[i];
                    }
                }
                return rhs;
            }

            // Rank-one/rank-mu CMA-ES, positive recombination weights, cumulative step size adaptation.
            // Actual bounded/feasible coordinates are supplied by the caller; no penalty losses are manufactured.
            private sealed class E27Cma
            {
                private readonly int n;
                private readonly Func<double> normal;
                private readonly double[] mean, pc, ps, weights, roots;
                private readonly double[,] cov, eigenvectors;
                private readonly double cc, cs, c1, cmu, ds, mueff, chi;
                private double sigma = 1;
                private int generation;
                internal int Lambda { get; }
                internal int Mu => weights.Length;
                internal E27Cma(int d, Func<double> gaussian)
                {
                    n = d; normal = gaussian; Lambda = 4 + (int)Math.Floor(3 * Math.Log(n));
                    weights = Enumerable.Range(1, Lambda / 2).Select(k => Math.Log(Lambda / 2d + .5) - Math.Log(k)).ToArray();
                    double sum = weights.Sum(); for (int k = 0; k < weights.Length; ++k) weights[k] /= sum;
                    mueff = 1 / E27Dot(weights, weights);
                    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n); cs = (mueff + 2) / (n + mueff + 5);
                    c1 = 2 / (Math.Pow(n + 1.3, 2) + mueff);
                    cmu = Math.Min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / (Math.Pow(n + 2, 2) + mueff));
                    ds = 1 + 2 * Math.Max(0, Math.Sqrt((mueff - 1) / (n + 1)) - 1) + cs;
                    chi = Math.Sqrt(n) * (1 - 1 / (4d * n) + 1 / (21d * n * n));
                    mean = new double[n]; ps = new double[n]; pc = new double[n]; roots = new double[n];
                    cov = new double[n, n]; eigenvectors = new double[n, n];
                    for (int i = 0; i < n; ++i) cov[i, i] = eigenvectors[i, i] = roots[i] = 1;
                }
                internal double[] Sample()
                {
                    var z = roots.Select(s => s * normal()).ToArray();
                    return mean.Select((v, i) => v + sigma * z.Select((x, j) => eigenvectors[i, j] * x).Sum()).ToArray();
                }
                internal void Update(double[][] samples)
                {
                    ++generation;
                    var y = samples.Select(x => x.Select((v, i) => (v - mean[i]) / sigma).ToArray()).ToArray();
                    var step = mean.Select((_, i) => weights.Select((w, k) => w * y[k][i]).Sum()).ToArray();
                    var eig = roots.Select((r, j) => step.Select((v, i) => eigenvectors[i, j] * v).Sum() / r).ToArray();
                    for (int i = 0; i < n; ++i)
                    {
                        mean[i] += sigma * step[i];
                        ps[i] = (1 - cs) * ps[i] + Math.Sqrt(cs * (2 - cs) * mueff) * eig.Select((v, j) => eigenvectors[i, j] * v).Sum();
                    }
                    double length = Math.Sqrt(E27Dot(ps, ps));
                    bool hsig = length / Math.Sqrt(1 - Math.Pow(1 - cs, 2 * generation)) / chi < 1.4 + 2d / (n + 1);
                    for (int i = 0; i < n; ++i) pc[i] = (1 - cc) * pc[i] + (hsig ? 1 : 0) * Math.Sqrt(cc * (2 - cc) * mueff) * step[i];
                    for (int i = 0; i < n; ++i)
                    for (int j = 0; j <= i; ++j)
                    {
                        double a = pc[i] * pc[j] + (hsig ? 0 : cc * (2 - cc) * cov[i, j]);
                        double b = weights.Select((w, k) => w * y[k][i] * y[k][j]).Sum();
                        cov[i, j] = cov[j, i] = (1 - c1 - cmu) * cov[i, j] + c1 * a + cmu * b;
                    }
                    sigma = Math.Max(1e-8, Math.Min(1e3, sigma * Math.Exp((cs / ds) * (length / chi - 1))));
                    Diagonalize();
                }
                private void Diagonalize()
                {
                    var a = (double[,])cov.Clone();
                    Array.Clear(eigenvectors, 0, eigenvectors.Length);
                    for (int i = 0; i < n; ++i) eigenvectors[i, i] = 1;
                    for (int sweep = 0; sweep < 18; ++sweep)
                    {
                        bool done = true;
                        for (int p = 0; p < n - 1; ++p)
                        for (int q = p + 1; q < n; ++q)
                        {
                            if (Math.Abs(a[p, q]) < 1e-13 * Math.Max(1e-16, Math.Max(Math.Abs(a[p, p]), Math.Abs(a[q, q])))) continue;
                            done = false;
                            double angle = .5 * Math.Atan2(2 * a[p, q], a[q, q] - a[p, p]), c = Math.Cos(angle), s = Math.Sin(angle);
                            for (int i = 0; i < n; ++i)
                            {
                                (a[i, p], a[i, q]) = (c * a[i, p] - s * a[i, q], s * a[i, p] + c * a[i, q]);
                                (eigenvectors[i, p], eigenvectors[i, q]) = (c * eigenvectors[i, p] - s * eigenvectors[i, q], s * eigenvectors[i, p] + c * eigenvectors[i, q]);
                            }
                            for (int j = 0; j < n; ++j) (a[p, j], a[q, j]) = (c * a[p, j] - s * a[q, j], s * a[p, j] + c * a[q, j]);
                        }
                        if (done) break;
                    }
                    for (int i = 0; i < n; ++i) roots[i] = Math.Sqrt(Math.Max(1e-18, a[i, i]));
                }
            }
        }
    }
}

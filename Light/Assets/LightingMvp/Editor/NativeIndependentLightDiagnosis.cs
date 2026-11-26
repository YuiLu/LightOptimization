using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E20Experiment
        {
            private sealed class MatrixProbe
            {
                public double[,] gram;
                public double[] thetaGradient;
            }

            private void RunDiagnosis()
            {
                var full = Target(-1, false);
                var oracle = scene.Lamps.Select(l => Target(l, false)).ToArray();
                int[] allAxes = Axes(scene.Lamps);
                using var gram = OpenCsv("column-matrices.csv", "anchor,scale,view,axis,gram_row,correlation_row");
                using var stable = OpenCsv("step-consistency.csv", "anchor,view,axis,from_position_m,to_position_m,from_log_intensity,to_log_intensity,cosine,norm_ratio");
                using var scans = OpenCsv("direction-scans.csv", "arm,seed,profile,direction,lamp,axis,unit,step,legal,centre_loss,loss,predicted_derivative,actual_forward_derivative,location");
                using var containment = OpenCsv("error-concentration.csv", "seed,profile,view,nrmse,peak_rgb_error,top1pct_energy_fraction,top01pct_energy_fraction,max_error_x,max_error_y");
                var caseResults = new List<(E20Start start, E20Arm joint, E20Arm[] single, MatrixProbe matrix)>();
                ProbeMatrices(full, "gt", 0, "gt", scene.Truth, allAxes, gram, stable);

                foreach (var start in starts)
                {
                    var joint = NativeArm("A3", full, start, scene.Lamps);
                    var one = new E20Arm[scene.Count];
                    foreach (int l in scene.Lamps)
                    {
                        one[l] = NativeArm("A1", full, start, new[] { l });
                        var isolated = NativeArm("oracle", oracle[l], start, new[] { l });
                        if (start.profile == "near" && !smoke)
                        {
                            if (!one[l].passed) ScanDirections(one[l], full, new[] { l }, scans);
                            if (!isolated.passed) ScanDirections(isolated, oracle[l], new[] { l }, scans);
                        }
                    }
                    var anchor = $"seed{start.seed}_{start.profile}";
                    var matrix = ProbeMatrices(full, anchor, start.seed, start.profile, start.theta, allAxes, gram, stable);
                    caseResults.Add((start, joint, one, matrix));
                    RecordConcentration(full, start, containment);
                    if (start.profile == "near" && !joint.passed && !smoke)
                        ScanDirections(joint, full, scene.Lamps, scans, matrix.thetaGradient);
                    gram.Flush(); stable.Flush(); containment.Flush(); scans.Flush();
                }

                var pairs = new List<(int a, int b, int evidence, double correlation)>();
                for (int a = 0; a < scene.Count; a++)
                for (int b = a + 1; b < scene.Count; b++)
                {
                    var near = caseResults.Where(c => c.start.profile == "near").ToArray();
                    var evidence = near.Where(c => c.single[a].passed && c.single[b].passed && (!Pass(c.joint.theta, a) || !Pass(c.joint.theta, b))).ToArray();
                    var comparison = evidence.Length == 0 ? near : evidence;
                    pairs.Add((a, b, evidence.Length, comparison.Max(c => CrossCorrelation(c.matrix.gram, a, b))));
                }
                var ordered = pairs.OrderByDescending(p => p.evidence).ThenByDescending(p => p.correlation).ThenBy(p => p.a).ThenBy(p => p.b).ToArray();
                var strongest = ordered[0];
                var selected = new[] { strongest, ordered.Last(p => p.evidence == strongest.evidence) }.Distinct().ToArray();
                using (var csv = OpenCsv("pair-selection.csv", "lamp_a,lamp_b,near_evidence_cases,max_abs_cross_correlation,selected"))
                {
                    foreach (var p in ordered)
                        csv.WriteLine(string.Join(",", p.a + 1, p.b + 1, p.evidence, F(p.correlation),
                            p.Equals(selected[0]) ? "strong" : selected.Contains(p) ? "comparison" : ""));
                }
                foreach (var p in selected)
                foreach (var start in starts)
                {
                    var lamps = new[] { p.a, p.b };
                    var row = NativeArm("A2", full, start, lamps);
                    if (!row.passed && start.profile == "near" && !smoke) ScanDirections(row, full, lamps, scans);
                }
            }

            private static double CrossCorrelation(double[,] gram, int a, int b)
            {
                double largest = 0;
                for (int i = 4 * a; i < 4 * a + 4; i++)
                for (int j = 4 * b; j < 4 * b + 4; j++)
                    if (gram[i, i] > 0 && gram[j, j] > 0)
                        largest = Math.Max(largest, Math.Abs(Cosine(gram[i, j], gram[i, i], gram[j, j])));
                return largest;
            }

            private MatrixProbe ProbeMatrices(E20Target target, string name, int seed, string profile, double[] theta, int[] active,
                StreamWriter matrix, StreamWriter stability)
            {
                SetMode(target.Shadows, target.Lamp);
                var watch = Stopwatch.StartNew();
                int n = active.Length, vCount = scene.Views.Length;
                var centre = new float[ViewChannels * vCount];
                int query = 0;
                var allowed = active.Select(i => i / 4).Distinct().ToArray();
                Measure(theta, "centre", centre, "");
                float[][] previous = null;
                MatrixProbe result = null;
                for (int scale = 0; scale < PositionSteps.Length; scale++)
                {
                    var columns = new float[n][];
                    var step = active.Select(i => i % 4 == 3 ? IntensitySteps[scale] : PositionSteps[scale]).ToArray();
                    var stencils = new int[n];
                    var other = new float[centre.Length];
                    for (int j = 0; j < n; j++)
                    {
                        int k = active[j];
                        double d = step[j] / units[k % 4];
                        var plus = (double[])theta.Clone(); plus[k] += d;
                        var minus = (double[])theta.Clone(); minus[k] -= d;
                        bool p = scene.Contains(plus), m = scene.Contains(minus);
                        var col = columns[j] = new float[centre.Length];
                        if (p) Measure(plus, $"L{k / 4 + 1}_{AxisNames[k % 4]}_plus", col, F(step[j]));
                        if (m) Measure(minus, $"L{k / 4 + 1}_{AxisNames[k % 4]}_minus", other, F(step[j]));
                        if (!(p || m)) continue;
                        stencils[j] = p && m ? 2 : p ? 1 : -1;
                        var left = p ? col : centre;
                        var right = m ? other : centre;
                        float inv = (float)(1 / (p && m ? 2 * step[j] : step[j]));
                        for (int kpx = 0; kpx < col.Length; kpx++) col[kpx] = (left[kpx] - right[kpx]) * inv;
                    }
                    var train = new double[n, n];
                    for (int v = 0; v < vCount; v++)
                    {
                        int offset = v * ViewChannels;
                        var perView = new double[n, n];
                        for (int i = 0; i < n; i++)
                        for (int j = 0; j <= i; j++)
                        {
                            var dot = DotSlice(columns[i], columns[j], offset, ViewChannels);
                            perView[i, j] = perView[j, i] = dot;
                            if (v < FitViews) { train[i, j] += dot; if (i != j) train[j, i] += dot; }
                        }
                        SaveGram(v, perView);
                        CompareScale(v, offset, ViewChannels);
                    }
                    SaveGram(-1, train);
                    CompareScale(-1, 0, FitViews * ViewChannels);
                    if (scale == 1)
                    {
                        result = new MatrixProbe { gram = train, thetaGradient = new double[n] };
                        for (int j = 0; j < n; j++)
                            result.thetaGradient[j] = 2 * units[active[j] % 4] * DotSlice(columns[j], centre, 0, FitViews * ViewChannels);
                    }
                    previous = columns;

                    void SaveGram(int v, double[,] values)
                    {
                        LogSpectrum(name, "physical_fd", seed, profile, scale, query, v, PositionSteps[scale], IntensitySteps[scale], step, stencils, values);
                        for (int i = 0; i < n; i++)
                            matrix.WriteLine(string.Join(",", name, scale, v, AxisLabel(active[i]),
                                Fields(Enumerable.Range(0, n).Select(j => values[i, j])),
                                Fields(Enumerable.Range(0, n).Select(j => Cosine(values[i, j], values[i, i], values[j, j])))));
                    }

                    void CompareScale(int v, int offset, int length)
                    {
                        if (previous == null) return;
                        for (int j = 0; j < n; j++)
                        {
                            double aa = DotSlice(previous[j], previous[j], offset, length);
                            double bb = DotSlice(columns[j], columns[j], offset, length);
                            double dot = DotSlice(previous[j], columns[j], offset, length);
                            stability.WriteLine(string.Join(",", name, v, AxisLabel(active[j]), F(PositionSteps[scale - 1]),
                                F(PositionSteps[scale]), F(IntensitySteps[scale - 1]), F(IntensitySteps[scale]),
                                Value(Cosine(dot, aa, bb)), aa > 0 ? F(Math.Sqrt(bb / aa)) : ""));
                        }
                    }
                }
                return result;

                void Measure(double[] x, string phase, float[] buffer, string steps)
                {
                    double loss = NativeResidual(target, x, buffer, vCount, "fd_diagnostics");
                    LogQuery(name, "diagnostic", ++query, phase, allowed, x, watch.Elapsed.TotalSeconds,
                        native: loss, steps: steps, cameraRenders: vCount);
                }
            }

            private void RecordConcentration(E20Target target, E20Start start, StreamWriter csv)
            {
                SetMode(target.Shadows, target.Lamp);
                scene.Apply(start.theta);
                var frames = Capture(fit, scene.Views.Length, "analysis");
                for (int v = 0; v < frames.Length; v++)
                {
                    var values = new double[frames[v].PixelCount];
                    int peak = 0;
                    for (int p = 0; p < values.Length; p++)
                    {
                        var delta = frames[v].Pixel(p) - target.Fit[v].Pixel(p);
                        values[p] = Sq20(delta.r) + Sq20(delta.g) + Sq20(delta.b);
                        if (values[p] > values[peak]) peak = p;
                    }
                    double total = values.Sum(), max = Math.Sqrt(values[peak]);
                    Array.Sort(values);
                    double Top(int fraction) => values.Skip(values.Length - Math.Max(1, values.Length / fraction)).Sum() / Math.Max(1e-30, total);
                    csv.WriteLine(string.Join(",", start.seed, start.profile, v, F(Math.Sqrt(NativeRenderBackend.Compare(frames[v], target.Fit[v]))),
                        F(max), F(Top(100)), F(Top(1000)), peak % E19FitW, peak / E19FitW));
                }
            }

            private void ScanDirections(E20Arm arm, E20Target target, int[] lamps, StreamWriter csv, double[] gradient = null)
            {
                SetMode(target.Shadows, target.Lamp);
                var watch = Stopwatch.StartNew();
                int[] active = Axes(lamps);
                var centreResidual = new float[FitViews * ViewChannels];
                var scratch = new float[centreResidual.Length];
                double centre = NativeResidual(target, arm.start, centreResidual, FitViews, "scans");
                bool measureGradient = gradient == null;
                if (measureGradient) gradient = new double[active.Length];
                string scan = arm.id + "_scan";
                int query = 0;
                for (int j = 0; j < active.Length; j++)
                {
                    int k = active[j];
                    bool strength = k % 4 == 3;
                    var scales = strength ? new[] { .01, .02, .05, .10, .20 } : new[] { .0025, .005, .01, .02, .05, .10, .20 };
                    if (measureGradient)
                    {
                        double h = strength ? .05 : .02;
                        double a = NativeAtAxis(k, h), b = NativeAtAxis(k, -h);
                        if (!double.IsNaN(a) && !double.IsNaN(b)) gradient[j] = ((a - b) / (2 * h)) * units[k % 4];
                        else if (!double.IsNaN(a)) gradient[j] = ((a - centre) / h) * units[k % 4];
                        else if (!double.IsNaN(b)) gradient[j] = ((centre - b) / h) * units[k % 4];
                    }
                    foreach (double h in scales)
                    foreach (double sign in new[] { -1.0, 1.0 })
                    {
                        var x = AtAxis(k, h * sign);
                        Record(x, "axis", k / 4 + 1, AxisNames[k % 4], strength ? "log" : "m", h * sign, gradient[j] / units[k % 4]);
                    }
                }
                double norm = Math.Sqrt(gradient.Sum(Sq20));
                if (norm > 0)
                {
                    foreach (double alpha in new[] { .0005, .00125, .0025, .005, .01, .02 })
                    foreach (double sign in new[] { -1.0, 1.0 })
                    {
                        var x = (double[])arm.start.Clone();
                        for (int j = 0; j < active.Length; j++) x[active[j]] -= alpha * sign * gradient[j] / norm;
                        Record(x, "negative_gradient", 0, "all", "theta", alpha * sign, -norm);
                    }
                }
                csv.Flush();

                double NativeAtAxis(int k, double h)
                {
                    var x = AtAxis(k, h);
                    return scene.Contains(x) ? Measure(x, "direction_model", h) : double.NaN;
                }
                double[] AtAxis(int k, double h)
                {
                    var x = (double[])arm.start.Clone();
                    x[k] += h / units[k % 4];
                    return x;
                }
                double Measure(double[] x, string phase, double h)
                {
                    double loss = NativeResidual(target, x, scratch, FitViews, "scans");
                    LogQuery(scan, "diagnostic", ++query, phase, lamps, x, watch.Elapsed.TotalSeconds,
                        native: loss, steps: F(h), cameraRenders: FitViews);
                    return loss;
                }
                void Record(double[] x, string direction, int lamp, string axis, string unit, double h, double prediction)
                {
                    bool legal = scene.Contains(x);
                    double loss = legal ? Measure(x, direction, h) : double.NaN;
                    csv.WriteLine(string.Join(",", arm.id, arm.seed, arm.profile, direction, lamp, axis, unit, F(h), legal ? 1 : 0,
                        F(centre), Value(loss), Value(prediction), Value((loss - centre) / h), Fields(x)));
                }
            }

            private static string AxisLabel(int i) => "L" + (i / 4 + 1) + "_" + AxisNames[i % 4];

            private static double DotSlice(float[] a, float[] b, int offset, int length)
            {
                double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                int end = offset + length, i = offset;
                for (; i + 3 < end; i += 4)
                {
                    s0 += (double)a[i] * b[i];
                    s1 += (double)a[i + 1] * b[i + 1];
                    s2 += (double)a[i + 2] * b[i + 2];
                    s3 += (double)a[i + 3] * b[i + 3];
                }
                for (; i < end; i++) s0 += (double)a[i] * b[i];
                return s0 + s1 + s2 + s3;
            }
        }
    }
}

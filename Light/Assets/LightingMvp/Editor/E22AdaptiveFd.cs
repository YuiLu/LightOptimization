using System;
using System.IO;
using System.Linq;
using NormalMode = LightingMvp.ImageJacobianLm.TrustNormalMode;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            private static readonly double[] E22FdSteps = { .005, .02, .05 };

            private void E22AdaptiveStudy()
            {
                var earlier = E22ReadE21("e21_b");
                using var history = Csv("adaptive-scale.csv", "arm,step_m,validation_axis,validation_step_m_or_log,predicted,actual,disagreement,selected");
                foreach (var sample in new[] { ("cross_geometry_L1", true, 0), ("seed157_far_L3", false, 2) })
                {
                    var target = E22Target(sample.Item2); var native = NativeSignal(target);
                    var start = earlier.arms.First(a => a.scenario == sample.Item1 + "_fd20mm" && a.method == "B0").initial;
                    var selected = new[] { sample.Item3 };
                    var fixedRuns = E22FdSteps.Select(h => RunE22Arm(sample.Item1, $"fixed_{h * 1000:F0}mm", selected, start, target, native,
                        420, NormalMode.Shared, step: h)).ToArray();
                    var failure = fixedRuns.FirstOrDefault(a => a.stop == "no_decreasing_step");
                    if (failure != null)
                        RunE22Arm(sample.Item1, $"adaptive_from_{failure.positionStep * 1000:F0}mm", selected, start, target, native,
                            420, NormalMode.Shared, step: failure.positionStep, adaptive: history);
                }
            }

            private double E22ChooseFdScale(E21Trial arm, double[] rig, int lamp, E21Signal signal, StreamWriter report)
            {
                int k = lamp * 4;
                var original = new float[signal.size]; var left = new float[signal.size]; var right = new float[signal.size];
                double centre = Evaluate(rig, original, "fd_scale_centre", -1, 0);
                var models = new (double[] g, double[] h)[E22FdSteps.Length];
                for (int z = 0; z < E22FdSteps.Length; ++z)
                {
                    var g = new double[4]; var a = new double[4];
                    for (int j = 0; j < 4; ++j)
                    {
                        double step = (j == 3 ? .04 : E22FdSteps[z]) / units[j];
                        for (int attempt = 0; attempt < 8; ++attempt, step *= .5)
                        {
                            var plus = Shift(j, step); var minus = Shift(j, -step);
                            bool legalP = room.Contains(plus), legalM = room.Contains(minus);
                            if (!legalP && !legalM) continue;
                            if (legalP) Evaluate(plus, left, "fd_scale_plus", k + j, step * units[j]); else Array.Copy(original, left, left.Length);
                            if (legalM) Evaluate(minus, right, "fd_scale_minus", k + j, -step * units[j]); else Array.Copy(original, right, right.Length);
                            float inv = (float)(1 / ((legalP && legalM ? 2 : 1) * step));
                            for (int i = 0; i < signal.size; ++i) left[i] = (left[i] - right[i]) * inv;
                            g[j] = E21Dot(left, original); a[j] = E21Dot(left, left);
                            break;
                        }
                    }
                    models[z] = (g, a);
                }

                // Use the same reserved, legal perturbation as an independent test of every local model.
                int axis = -1; double displacement = 0; double actual = 0;
                foreach (int j in Enumerable.Range(0, 3).OrderByDescending(i => Math.Abs(models[1].g[i]) / units[i]))
                {
                    if (models[1].h[j] <= 1e-20) continue;
                    double d = (j == 3 ? .013 : .008) * (models[1].g[j] > 0 ? -1 : 1) / units[j];
                    for (int shrink = 0; shrink < 6; ++shrink, d *= .5)
                    {
                        var candidate = Shift(j, d);
                        if (!room.Contains(candidate)) continue;
                        actual = centre - Evaluate(candidate, null, "fd_scale_validation", k + j, d * units[j]);
                        axis = j; displacement = d;
                        break;
                    }
                    if (axis >= 0) break;
                }
                if (axis < 0) return arm.positionStep;
                var predictions = models.Select(p => -2 * p.g[axis] * displacement - p.h[axis] * displacement * displacement).ToArray();
                var errors = predictions.Select(p => Math.Abs(p - actual) / Math.Max(1e-12, Math.Max(Math.Abs(p), Math.Abs(actual)))).ToArray();
                int winner = Enumerable.Range(0, errors.Length).OrderBy(i => errors[i]).ThenBy(i => Math.Abs(E22FdSteps[i] - arm.positionStep)).First();
                for (int i = 0; i < E22FdSteps.Length; ++i)
                    report.WriteLine(string.Join(",", arm.id, F(E22FdSteps[i]), axis, F(displacement * units[axis]), F(predictions[i]), F(actual), F(errors[i]), Flag(i == winner)));
                return E22FdSteps[winner];

                double[] Shift(int axis, double d) { var x = (double[])rig.Clone(); x[k + axis] += d; return x; }
                double Evaluate(double[] x, float[] residual, string phase, int axis, double step)
                {
                    if (arm.evaluations >= arm.limit || !room.Contains(x)) throw new InvalidOperationException("Adaptive FD audit violated Q or legality.");
                    double loss = signal.objective(x, residual);
                    ++arm.objectiveQueries; ++arm.evaluations;
                    queryLog.WriteLine(string.Join(",", arm.id, "scale_audit", arm.evaluations, phase, axis, F(step), 1, 1, F(loss), Series(x), F(clock.Elapsed.TotalSeconds)));
                    return loss;
                }
            }
        }
    }
}

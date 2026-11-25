using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;
using Frame = LightingMvp.NativeRenderBackend.Frame;
using Mapping = LightingMvp.NativeRenderBackend.HdrMapping;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class ClassroomDiagnosis
        {
            private static readonly Mapping[] LossArms = { Mapping.Raw, Mapping.SignedLog, Mapping.SignedSqrt };
            private static readonly Vector3 Metres = RoomMax - RoomMin;
            private const double WorldStep = .02, LogIntensityStep = .05;
            private const int BlockQuota = 153; // 51 whole three-view queries; fixed regardless of lamp or round.

            private DiagnosisTrial Trial(string profile, int seed, string method, int lamp, Mapping mapping, double[] start,
                int[] axes, int budget, bool block = false, bool nnls = false, bool physical = false, bool diagonal = false)
            {
                string key = $"{profile}_{seed}_{method}_{lamp}_{mapping}";
                string folder = Path.Combine(dir, key);
                var row = new DiagnosisTrial { key = key, stage = task, profile = profile, method = method, lamp = lamp,
                    seed = seed, mapping = mapping.ToString(), dimensions = axes.Length, budget = budget, start = start,
                    predictedNrmse = -1, initialContributionNrmse = -1, finalContributionNrmse = -1 };
                var watch = Stopwatch.StartNew();
                int channels = train.Width * train.Height * 3, residualSize = 3 * channels;
                int calls = 0;
                double bestLoss = double.PositiveInfinity, bestRawSquared = double.PositiveInfinity;
                double lastRaw = 0;
                double[] best = null;
                bool track = false;
                using (var log = new QueryLog(folder, "queries.csv", method, seed))
                using (var progress = new StreamWriter(Path.Combine(folder, "progress.csv")))
                using (var jac = new StreamWriter(Path.Combine(folder, "jacobians.csv")))
                using (var steps = new StreamWriter(Path.Combine(folder, "lm-steps.csv")))
                {
                    progress.WriteLine("camera,phase,objective_sq,raw_sq,best_objective_sq,raw_at_best_sq,best_raw_sq");
                    jac.WriteLine("call,batch,camera,centre_sq,axes,normalized_step,metres_or_log_step,stencil,column_norm,eigenvalues,rank,eigen_cutoff,condition_j,gram_upper");
                    steps.WriteLine("call,batch,camera,mu,predicted,actual,gain_ratio,bound_hits,shrinks,status");
                    var scratch = new float[residualSize];
                    Measure(NoLights, "empty", scratch);
                    track = true;
                    row.initialObjective = Math.Sqrt(Measure(start, "anchor", scratch));
                    row.initialRaw = Math.Sqrt(lastRaw);
                    row.initialAlreadyPasses = MeetsThreshold(gate.emptyLoss, lastRaw);
                    if (nnls)
                    {
                        FitPower(folder, start);
                        row.stop = "bounded_solve";
                    }
                    else if (block)
                    {
                        int sweep = 0;
                        row.stop = "budget";
                        while (budget - log.Cameras >= 3 * (2 * LampDofs + 2))
                        {
                            double before = bestLoss;
                            int first = log.Cameras;
                            for (int slot = 0; slot < 7 && budget - log.Cameras >= 3 * (2 * LampDofs + 2); ++slot)
                            {
                                Solve((double[])best.Clone(), LampAxes(slot, true), Math.Min(BlockQuota, budget - log.Cameras), $"round{sweep}_lamp{slot}_");
                                if (bestLoss <= 1e-12) break;
                            }
                            ++sweep;
                            if (bestLoss <= 1e-12) { row.stop = "near_zero_residual"; break; }
                            if (first == log.Cameras || before - bestLoss <= Math.Max(1e-12, before * 1e-8)) { row.stop = "no_block_progress"; break; }
                        }
                    }
                    else row.stop = Solve(start, axes, budget - log.Cameras, "lm_").StopReason;
                    row.best = best;
                    row.finalObjective = Math.Sqrt(bestLoss);
                    row.bestRaw = Math.Sqrt(bestRawSquared);
                    row.trainingRenders = log.Cameras;
                    row.renderSeconds = log.RenderSeconds; row.readbackSeconds = log.ReadbackSeconds;
                    if (log.Cameras > budget) throw new InvalidOperationException("Trial exceeded camera budget.");

                    double Measure(double[] x, string phase, float[] residual)
                    {
                        if (log.Cameras + 3 > budget || !fixture.Rig.Contains(x)) throw new InvalidOperationException("Illegal diagnosis query " + phase);
                        double raw = 0;
                        double loss = log.Measure(train, phase, x, () =>
                        {
                            fixture.Rig.Apply(x);
                            double before = train.RenderSeconds + train.ReadbackSeconds;
                            long t = Stopwatch.GetTimestamp();
                            double cost = 0;
                            foreach (int v in DiagnosisViews)
                            {
                                if (mapping == Mapping.Raw)
                                {
                                    double f = train.Residual(v, target[v], residual, v * channels, 3);
                                    raw += f; cost += f;
                                }
                                else
                                {
                                    cost += train.MappedResidual(v, maps[mapping][v], residual, v * channels, 3, out double f);
                                    raw += f;
                                }
                            }
                            row.residualSeconds += Math.Max(0, Since(t) - (train.RenderSeconds + train.ReadbackSeconds - before));
                            return cost;
                        });
                        lastRaw = raw;
                        if (track)
                        {
                            if (loss < bestLoss) { bestLoss = loss; best = (double[])x.Clone(); row.bestQuery = log.Cameras; row.finalRaw = Math.Sqrt(raw); }
                            if (raw < bestRawSquared) { bestRawSquared = raw; row.bestRawQuery = log.Cameras; }
                            if (row.firstRawPass < 0 && MeetsThreshold(gate.emptyLoss, raw)) row.firstRawPass = log.Cameras;
                        }
                        progress.WriteLine(string.Join(",", log.Cameras, phase, F(loss), F(raw), track ? F(bestLoss) : "", track ? F(Sq(row.finalRaw)) : "", track ? F(bestRawSquared) : ""));
                        return loss;
                    }

                    ImageJacobianLm.Result Solve(double[] anchor, int[] active, int cap, string prefix)
                    {
                        ++calls;
                        var full = (double[])anchor.Clone();
                        int first = log.Cameras;
                        double[] Expand(double[] partial)
                        {
                            for (int i = 0; i < active.Length; ++i) full[active[i]] = partial[i];
                            return full;
                        }
                        var physicalSteps = active.Select(i => i % 4 == 3 ? LogIntensityStep / LogPowerSpan : WorldStep / Metres[i % 4]).ToArray();
                        var diag = new ImageJacobianLm.Diagnostics { ProbeSteps = physical ? physicalSteps : null, DiagonalDamping = diagonal };
                        diag.Jacobian = (batch, q, value, dx, stencil, h) =>
                        {
                            long tick = Stopwatch.GetTimestamp();
                            var sp = GetSpectrum(h, active.Length);
                            var upper = Enumerable.Range(0, active.Length).SelectMany(i => Enumerable.Range(i, active.Length - i).Select(j => h[i, j]));
                            jac.WriteLine(string.Join(",", calls, batch, first + 3 * q, F(value), string.Join(";", active), JoinFields(dx),
                                JoinFields(dx.Select((x, i) => x * (active[i] % 4 == 3 ? LogPowerSpan : Metres[active[i] % 4]))), string.Join(";", stencil),
                                JoinFields(sp.columnNorms), JoinFields(sp.eigen), sp.rank, F(sp.cutoff), F(sp.condition), JoinFields(upper)));
                            row.diagnosticSeconds += Since(tick);
                        };
                        diag.Trial = (batch, q, mu, predicted, actual, ratio, hits, shrinks, status) =>
                            steps.WriteLine(string.Join(",", calls, batch, first + 3 * q, F(mu), Finite(predicted), Finite(actual), Finite(ratio), hits, shrinks, status));
                        row.lmBufferBytes = Math.Max(row.lmBufferBytes, sizeof(float) * (long)residualSize * (active.Length + 3));
                        var result = ImageJacobianLm.Run(active.Select(i => anchor[i]).ToArray(), residualSize,
                            (x, phase, buffer) => Measure(Expand(x), prefix + phase, buffer), cap / 3, seed,
                            x => fixture.Rig.Contains(Expand(x)), profile: true, diagnostics: diag);
                        if (log.Cameras != first + 3 * result.Queries) throw new InvalidOperationException("LM query accounting mismatch.");
                        row.jacobians += result.Jacobians; row.accepted += result.Accepted; row.rejected += result.Rejected;
                        row.blockedProbes += result.BlockedProbes; row.shortenedTrials += result.ShortenedTrials;
                        row.jacobianSeconds += result.JacobianSeconds; row.normalSeconds += result.NormalSeconds; row.linearSolveSeconds += result.LinearSolveSeconds;
                        return result;
                    }
                    void FitPower(string destination, double[] initial)
                    {
                        const int n = 7;
                        var gram = new double[n, n]; var rhs = new double[n]; var columns = new float[n][];
                        // No mapped OLAT; NNLS is used only with the raw objective at fixed Oracle positions.
                        var b = Difference(target, empty);
                        for (int l = 0; l < n; ++l)
                        {
                            int index = l;
                            var unit = WithPower(SingleLamp(initial, l), Q0);
                            var a = DiagnosisViews.Select(v => Capture(train, log, unit, v, "unit_" + index)).ToArray();
                            columns[l] = Difference(a, empty);
                            rhs[l] = RgbDot(b, columns[l]);
                            for (int k = 0; k <= l; ++k) gram[l, k] = gram[k, l] = RgbDot(columns[l], columns[k]);
                        }
                        var powers = axes.Select(i => Power(initial[i])).ToArray();
                        double gain = BoundedLightWeights.Solve(gram, rhs, n, PowerMin, PowerMax, powers);
                        row.predictedNrmse = Math.Sqrt(Math.Max(0, RgbDot(b, b) - gain));
                        WriteJson(destination, "response-spectrum.json", GetSpectrum(gram, n));
                        var fitted = (double[])initial.Clone();
                        for (int i = 0; i < n; ++i) fitted[axes[i]] = PowerCoordinate(powers[i]);
                        Measure(fitted, "nnls_verify", scratch);
                    }
                }
                row.trainingSeconds = watch.Elapsed.TotalSeconds;
                row.processPeakWorkingSet = ReadScaleMemory().peak;
                Recapture(row, folder);
                row.elapsed = watch.Elapsed.TotalSeconds;
                rows.Add(row);
                WriteJson(folder, "trial.json", row);
                return row;
            }

            private float[] Difference(Frame[] a, Frame[] b)
            {
                var rgb = new float[3 * 3 * a[0].PixelCount];
                int k = 0;
                foreach (int v in DiagnosisViews)
                {
                    double scale = 1 / Math.Sqrt(3 * (target[v].Energy + DiagnosisEpsilon));
                    for (int p = 0; p < a[v].PixelCount; ++p)
                    {
                        var x = a[v].Pixel(p); var y = b[v].Pixel(p);
                        rgb[k++] = (float)((x.r - y.r) * scale); rgb[k++] = (float)((x.g - y.g) * scale); rgb[k++] = (float)((x.b - y.b) * scale);
                    }
                }
                return rgb;
            }

            private void Recapture(DiagnosisTrial row, string folder)
            {
                var f0 = new double[3]; var f1 = new double[3];
                row.displayPerView = new double[3]; row.displayHighPerView = new double[5]; row.highPerView = new double[5];
                row.initialBrightness = new Brightness[3]; row.finalBrightness = new Brightness[3];
                using (var log = new QueryLog(folder, "recaptures.csv", row.method, row.seed))
                {
                    foreach (int v in DiagnosisViews)
                    {
                        var start = Capture(train, log, row.start, v, "train_initial");
                        var best = Capture(train, log, row.best, v, "train_best");
                        f0[v] = NativeRenderBackend.Compare(start, target[v]);
                        f1[v] = NativeRenderBackend.Compare(best, target[v]);
                        row.displayPerView[v] = DisplayNrmse(best, target[v]);
                        row.initialBrightness[v] = BrightnessStats(start, target[v]); row.finalBrightness[v] = BrightnessStats(best, target[v]);
                    }
                    for (int v = 0; v < 5; ++v)
                    {
                        var best = Capture(high, log, row.best, v, "check_best");
                        row.highPerView[v] = Math.Sqrt(NativeRenderBackend.Compare(best, targetHi[v]));
                        row.displayHighPerView[v] = DisplayNrmse(best, targetHi[v]);
                        if (row.lamp < 0 && row.seed == 131 && (v == 0 || v == 3) && (task == "e14" || task == "e14-large" || task == "e15-budget"))
                            NativeRenderBackend.Save(best, high.Width, high.Height, Path.Combine(folder, "best_view" + v));
                    }
                    row.recaptures = log.Cameras;
                }
                row.trainPerView = f1.Select(Math.Sqrt).ToArray(); row.recapturedRaw = Rms(f1);
                row.highNrmse = Rms(row.highPerView.Take(3).Select(Sq)); row.heldNrmse = Rms(row.highPerView.Skip(3).Select(Sq));
                if (Math.Abs(row.recapturedRaw - row.finalRaw) > Math.Max(1, row.finalRaw) * 1e-6 || Math.Abs(Rms(f0) - row.initialRaw) > Math.Max(1, row.initialRaw) * 1e-6)
                    throw new InvalidOperationException("Trial raw recapture differs from the objective query.");
                row.reduction = Reduction(gate.emptyLoss, Sq(row.recapturedRaw)); row.passed = MeetsThreshold(gate.emptyLoss, Sq(row.recapturedRaw));
                row.positionErrorMetres = Enumerable.Range(0, 7).Select(i => (double)Vector3.Distance(ClassroomFixture.Position(truth, 4 * i), ClassroomFixture.Position(row.best, 4 * i))).ToArray();
                row.intensityError = Enumerable.Range(0, 7).Select(i => Math.Abs(Power(truth[4 * i + 3]) - Power(row.best[4 * i + 3]))).ToArray();
                row.intensityRelativeError = row.intensityError.Select((e, i) => e / Power(truth[4 * i + 3])).ToArray();
                if (row.lamp < 0) return;
                var e = contribution[row.lamp];
                row.initialContributionPerView = DiagnosisViews.Select(v => e[v] > ContributionFloor ? Math.Sqrt(f0[v] / e[v]) : -1).ToArray();
                row.finalContributionPerView = DiagnosisViews.Select(v => e[v] > ContributionFloor ? Math.Sqrt(f1[v] / e[v]) : -1).ToArray();
                row.contributionViews = e.Count(x => x > ContributionFloor);
                if (row.contributionViews == 0) return;
                row.initialContributionNrmse = Rms(row.initialContributionPerView.Where(x => x >= 0).Select(Sq));
                row.finalContributionNrmse = Rms(row.finalContributionPerView.Where(x => x >= 0).Select(Sq));
                row.contributionPass = row.finalContributionNrmse <= .05;
            }

            private static string Finite(double x) => double.IsNaN(x) || double.IsInfinity(x) ? "" : F(x);
            private static double DisplayNrmse(Frame f, Frame target)
            {
                double sq = 0, energy = 0;
                for (int p = 0; p < f.PixelCount; ++p)
                {
                    var a = f.Pixel(p); var b = target.Pixel(p);
                    Add(a.r, b.r); Add(a.g, b.g); Add(a.b, b.b);
                }
                return Math.Sqrt(sq / (energy + DiagnosisEpsilon));
                void Add(float a, float b)
                {
                    double x = NativeRenderBackend.Display(a), y = NativeRenderBackend.Display(b);
                    sq += Sq(x - y); energy += y * y;
                }
            }
            private static double PixelError(Color a, Color b) => Sq(a.r - b.r) + Sq(a.g - b.g) + Sq(a.b - b.b);
            private static Brightness BrightnessStats(Frame f, Frame target)
            {
                var rgb = new float[3 * f.PixelCount];
                var errors = new double[f.PixelCount];
                var stat = new Brightness();
                double peakError = -1;
                for (int p = 0, k = 0; p < f.PixelCount; ++p)
                {
                    var a = f.Pixel(p); var b = target.Pixel(p);
                    rgb[k++] = a.r; rgb[k++] = a.g; rgb[k++] = a.b;
                    stat.negativeChannels += (a.r < 0 ? 1 : 0) + (a.g < 0 ? 1 : 0) + (a.b < 0 ? 1 : 0);
                    double e = errors[p] = PixelError(a, b);
                    if (e > peakError) { peakError = e; stat.maxErrorPixel = p; }
                    stat.errorEnergy += e;
                }
                Array.Sort(errors); Array.Sort(rgb);
                stat.peak = rgb[rgb.Length - 1]; stat.p50 = P(.5); stat.p90 = P(.9); stat.p99 = P(.99); stat.p999 = P(.999);
                stat.top01 = E(.001); stat.top1 = E(.01); stat.top5 = E(.05);
                return stat;
                double P(double p) => rgb[(int)Math.Floor(p * (rgb.Length - 1))];
                double E(double p) => stat.errorEnergy <= 0 ? 0 : errors.Skip(errors.Length - Math.Max(1, (int)(p * errors.Length))).Sum() / stat.errorEnergy;
            }
        }
    }
}

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using UnityEngine;
using Frame = LightingMvp.NativeRenderBackend.Frame;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E20Experiment
        {
            private const double MsmStep = .02;
            private static readonly double[] MsmProbeSteps = { .0025, .005, .01, .02, .05 };
            private static readonly double[] NativeDirectionSteps = { .005, .01, .02, .04, .08 };

            [Serializable] private sealed class MomentGate
            {
                public int lamp, trials, nearRecoveries, nearCount, directionsPassing, directionCount;
                public string rule;
                public bool expandToSeven;
            }

            private sealed class ShadowFeature
            {
                internal readonly ProxyGeometry.Receiver[] samples;
                internal readonly float[] target;
                private readonly double[] norms;
                internal readonly int fitSize;

                internal ShadowFeature(ProxyGeometry.Receiver[] receivers, float[] visibility, int views)
                {
                    samples = receivers; target = visibility;
                    fitSize = samples.Count(s => s.View < FitViews);
                    norms = new double[views];
                    for (int i = 0; i < samples.Length; i++) norms[samples[i].View] += (double)target[i] * target[i];
                    for (int v = 0; v < views; v++)
                        norms[v] = Math.Max(norms[v], 1e-8) * (v < FitViews ? FitViews : views - FitViews);
                }

                internal double Loss(float[] visibility, float[] residual = null, bool held = false)
                {
                    double loss = 0;
                    int at = 0;
                    for (int i = 0; i < visibility.Length; i++)
                    {
                        int v = samples[i].View;
                        if ((v >= FitViews) != held) continue;
                        double error = visibility[i] - target[i];
                        loss += error * error / norms[v];
                        if (residual != null) residual[at++] = (float)(error / Math.Sqrt(norms[v]));
                    }
                    return loss;
                }

                internal void Save(string path, string caption, float[] visibility)
                {
                    SaveReceiverFeature(samples, visibility.Select(x => new Vector3(x, x, x)).ToArray(), path, caption, visibility: true);
                    if (ReferenceEquals(visibility, target)) return;
                    var error = visibility.Select((x, i) => Mathf.Abs(x - target[i])).ToArray();
                    SaveReceiverFeature(samples, error.Select(x => new Vector3(x, x * .22f, 0)).ToArray(),
                        path, caption + "_error", visibility: true);
                }
            }

            private void RunMomentStudy()
            {
                using var geometry = new ProxyGeometry(scene.Camera);
                var samples = geometry.Capture(scene, dir);
                var nativeOn = Target(-1, true);
                var nativeOff = Target(-1, false);
                int lamp = SelectMomentLamp(nativeOn);
                using var settingsCsv = OpenCsv("moment-settings.csv", "lamp,setting,face_size,msaa_requested,msaa_actual,blur_radius,near_m,far_m,texture,solver,moment_bias,depth_bias_m,leak_bias,median_texel_at_gt_m");
                using var metrics = OpenCsv("visibility-diagnostics.csv", "lamp,setting,anchor,seed,profile,min,max,mean,shadowed_lt_half,unshadowed_ge_half,degenerate,nonfinite,seam_mean,seam_max");
                using var settingsCosts = OpenCsv("moment-costs.csv", "lamp,setting,face_renders,solver_degenerate_samples,solver_nonfinite_samples,seconds");
                using var scans = OpenCsv("moment-fd.csv", "lamp,setting,seed,profile,axis,step_m,stencil,centre_loss,minus_loss,plus_loss,derivative,cosine_to_previous_scale,gradient_norm");
                using var nativeScans = OpenCsv("moment-native-direction.csv", "lamp,setting,seed,profile,shadows,step_m,centre_native_loss,minus_native_loss,plus_native_loss,plus_minus_derivative,positive_lowers_native,msm_predicted_slope,dir_x,dir_y,dir_z,gt_alignment");
                var configs = new List<MomentSettings>();
                var preferred = RunLamp(lamp, extended: !smoke);
                var gate = new MomentGate { lamp = lamp + 1, trials = preferred.recovery.Count,
                    nearRecoveries = preferred.recovery.Count(x => x.profile == "near" && x.passed),
                    nearCount = preferred.recovery.Count(x => x.profile == "near"),
                    directionsPassing = preferred.directionPasses, directionCount = starts.Length,
                    rule = "Recover 5 cm or better on every Near seed and beat both centre and negative direction in at least three alpha values on every start (native shadow On)." };
                gate.expandToSeven = !smoke && gate.nearCount > 0 && gate.nearRecoveries == gate.nearCount && gate.directionCount == gate.directionsPassing;
                WriteJson(dir, "gate.json", gate);
                if (gate.expandToSeven)
                    foreach (int other in scene.Lamps.Where(x => x != lamp)) RunLamp(other, extended: false);
                WriteJson(dir, "settings.json", new MomentSettingsSet { cases = configs.ToArray() });

                (List<E20Arm> recovery, int directionPasses) RunLamp(int selected, bool extended)
                {
                    var recovery = new List<E20Arm>();
                    int directionPasses = 0;
                    var soloOff = Target(selected, false);
                    var soloOn = Target(selected, true);
                    var dark = CaptureDark();
                    var all = MomentConfigs(lights[selected], extended);
                    foreach (var cfg in all)
                    {
                        var watch = Stopwatch.StartNew();
                        using var shadow = new PointMomentShadow(geometry, samples, cfg);
                        configs.Add(cfg);
                        bool primary = cfg == all[0];
                        var gt = scene.PositionOf(scene.Truth, selected);
                        var visible = shadow.Render(gt);
                        if (visible.Max() - visible.Min() < .01f || shadow.LastNonfinite > 0)
                            throw new InvalidOperationException($"MSM {cfg.label} L{selected + 1} reference visibility is invalid or constant.");
                        var feature = new ShadowFeature(samples, visible, scene.Views.Length);
                        double texel = samples.Select(s => (double)Vector3.Distance(gt, s.Position) * 2 / cfg.size)
                            .OrderBy(x => x).ElementAt(samples.Length / 2);
                        settingsCsv.WriteLine(string.Join(",", selected + 1, cfg.label, cfg.size, cfg.requestedMsaa, cfg.actualMsaa, cfg.blurRadius,
                            F(cfg.near), F(cfg.far), "R32G32B32A32_SFloat", cfg.variance ? "VSM" : "4MSM",
                            F(cfg.momentBias), F(cfg.depthBiasMetres), F(cfg.leakBias), F(texel)));
                        string settingFolder = CapturePath($"moment_L{selected + 1}_{cfg.label}");
                        LogVisibility(selected, cfg, "gt", 0, "", visible, shadow, metrics);
                        RecordNativeVisibilityMatch(selected, cfg, samples, visible, soloOn.Fit, soloOff.Fit, dark, settingFolder);
                        if (primary)
                        {
                            feature.Save(settingFolder, "gt_visibility", visible);
                            shadow.SaveFaces(settingFolder, "gt");
                        }
                        foreach (var start in starts)
                        {
                            var x = IsolatedStart(start.theta, new[] { selected });
                            var local = ProbeMoment(start, selected, cfg, feature, shadow, x, scans, metrics);
                            if (!primary) continue;
                            if (CheckMomentDirection(start, selected, cfg, local, nativeOn, nativeOff, nativeScans))
                                directionPasses++;
                            var fit = RecoverMoment(start, selected, cfg, feature, shadow, nativeOn, local);
                            recovery.Add(fit);
                            NativeArm("M0", nativeOn, start, new[] { selected });
                            AcceptMomentProposal(start, selected, cfg, feature, shadow, nativeOn, fit);
                        }
                        settingsCosts.WriteLine(string.Join(",", selected + 1, cfg.label, shadow.Faces, shadow.Degenerate, shadow.Nonfinite, F(watch.Elapsed.TotalSeconds)));
                        settingsCsv.Flush(); metrics.Flush(); settingsCosts.Flush(); scans.Flush(); nativeScans.Flush();
                    }
                    return (recovery, directionPasses);
                }
            }

            [Serializable] private sealed class MomentSettingsSet { public MomentSettings[] cases; }

            private static MomentSettings[] MomentConfigs(Light light, bool extended)
            {
                MomentSettings Variant(string label, int size, int msaa, int radius, bool variance = false) => new MomentSettings {
                    label = label, size = size, requestedMsaa = msaa, blurRadius = radius, variance = variance,
                    near = Mathf.Max(.02f, light.shadowNearPlane), far = light.range,
                    momentBias = .001f, depthBiasMetres = .012f, leakBias = .06f };
                var primary = Variant("msm128_4x_k5", 128, 4, 2);
                return extended ? new[] { primary, Variant("msm128_1x_k5", 128, 1, 2), Variant("msm128_8x_k5", 128, 8, 2),
                    Variant("msm64_4x_k5", 64, 4, 2), Variant("msm256_4x_k5", 256, 4, 2),
                    Variant("msm128_4x_k3", 128, 4, 1), Variant("msm128_4x_k9", 128, 4, 4),
                    Variant("vsm128_4x_k5", 128, 4, 2, true) } : new[] { primary };
            }

            private int SelectMomentLamp(E20Target nativeOn)
            {
                SetMode(true, -1);
                scene.Apply(scene.Truth);
                using var csv = OpenCsv("lamp-selection.csv", "lamp,all_train_shadow_delta_nrmse,per_view");
                var energy = new double[scene.Count];
                foreach (int l in scene.Lamps)
                {
                    scene.SetShadows(l, false);
                    var frames = Capture(fit, FitViews, "moment_lamp_selection");
                    var view = frames.Select((f, v) => Math.Sqrt(NativeRenderBackend.Compare(f, nativeOn.Fit[v]))).ToArray();
                    energy[l] = Math.Sqrt(view.Select(Sq20).Average());
                    csv.WriteLine(string.Join(",", l + 1, F(energy[l]), Fields(view)));
                    scene.SetShadows(l, true);
                }
                return Enumerable.Range(0, scene.Count).OrderByDescending(l => energy[l]).First();
            }

            private static void LogVisibility(int lamp, MomentSettings cfg, string label, int seed, string profile, float[] values,
                PointMomentShadow shadow, StreamWriter csv)
            {
                int occluded = values.Count(x => x < .5f);
                csv.WriteLine(string.Join(",", lamp + 1, cfg.label, label, seed, profile, F(values.Min()), F(values.Max()), F(values.Average()),
                    occluded, values.Length - occluded, shadow.LastDegenerate, shadow.LastNonfinite, F(shadow.LastSeamMean), F(shadow.LastSeamMax)));
            }

            private static void RecordNativeVisibilityMatch(int lamp, MomentSettings cfg, ProxyGeometry.Receiver[] samples, float[] visibility,
                Frame[] on, Frame[] off, Frame[] dark, string output)
            {
                using var csv = new StreamWriter(Path.Combine(output, "native-visibility-reference.csv"));
                csv.WriteLine("lamp,setting,view,meaningfully_lit_samples,native_mean,msm_mean,weighted_rmse,native_shadowed_fraction,msm_shadowed_fraction");
                for (int v = 0; v < on.Length; v++)
                {
                    double weight = 0, native = 0, msm = 0, sq = 0;
                    int n = 0, nShadow = 0, mShadow = 0;
                    for (int i = 0; i < samples.Length; i++)
                    {
                        var s = samples[i];
                        if (s.View != v) continue;
                        var a = ProxyGeometry.Rgb(off[v].Pixel(s.Pixel) - dark[v].Pixel(s.Pixel));
                        var b = ProxyGeometry.Rgb(on[v].Pixel(s.Pixel) - dark[v].Pixel(s.Pixel));
                        double energy = a.sqrMagnitude;
                        if (energy <= 1e-6) continue;
                        double expected = Mathf.Clamp01((float)(Vector3.Dot(a, b) / energy));
                        double p = visibility[i];
                        weight += energy; native += expected * energy; msm += p * energy; sq += energy * Sq20(p - expected);
                        n++;
                        if (expected < .5) nShadow++;
                        if (p < .5) mShadow++;
                    }
                    if (n != 0)
                        csv.WriteLine(string.Join(",", lamp + 1, cfg.label, v, n, F(native / weight), F(msm / weight), F(Math.Sqrt(sq / weight)),
                            F(nShadow / (double)n), F(mShadow / (double)n)));
                }
            }

            private Vector3 ProbeMoment(E20Start start, int lamp, MomentSettings cfg, ShadowFeature feature, PointMomentShadow shadow,
                double[] x, StreamWriter csv, StreamWriter metrics)
            {
                var watch = Stopwatch.StartNew();
                int query = 0;
                var lamps = new[] { lamp };
                string key = $"momfd_{cfg.label}_L{lamp + 1}_{start.seed}_{start.profile}";
                var centreVis = shadow.Render(scene.PositionOf(x, lamp));
                double centre = feature.Loss(centreVis);
                LogQuery(key, "diagnostic", ++query, "centre", lamps, x, watch.Elapsed.TotalSeconds, msm: centre, faces: 6);
                LogVisibility(lamp, cfg, "initial", start.seed, start.profile, centreVis, shadow, metrics);
                Vector3 previous = Vector3.zero, nominal = Vector3.zero;
                foreach (double h in MsmProbeSteps)
                {
                    var gradient = Vector3.zero;
                    var data = new (double minus, double plus, int stencil)[3];
                    for (int a = 0; a < 3; a++)
                    {
                        var plus = (double[])x.Clone(); plus[4 * lamp + a] += h / units[a];
                        var minus = (double[])x.Clone(); minus[4 * lamp + a] -= h / units[a];
                        bool p = scene.Contains(plus), m = scene.Contains(minus);
                        double lp = p ? Measure(plus, AxisNames[a] + "_plus", h) : double.NaN;
                        double lm = m ? Measure(minus, AxisNames[a] + "_minus", -h) : double.NaN;
                        double d = p && m ? (lp - lm) / (2 * h) : p ? (lp - centre) / h : m ? (centre - lm) / h : 0;
                        gradient[a] = (float)d;
                        data[a] = (lm, lp, p && m ? 2 : p ? 1 : m ? -1 : 0);
                    }
                    double cosine = Cosine(Vector3.Dot(gradient, previous), gradient.sqrMagnitude, previous.sqrMagnitude);
                    for (int a = 0; a < 3; a++)
                        csv.WriteLine(string.Join(",", lamp + 1, cfg.label, start.seed, start.profile, AxisNames[a], F(h),
                            data[a].stencil, F(centre), Value(data[a].minus), Value(data[a].plus), F(gradient[a]), Value(cosine), F(gradient.magnitude)));
                    previous = gradient;
                    if (h == MsmStep) nominal = gradient;
                }
                return nominal;

                double Measure(double[] theta, string phase, double step)
                {
                    var visible = shadow.Render(scene.PositionOf(theta, lamp));
                    double loss = feature.Loss(visible);
                    LogQuery(key, "diagnostic", ++query, phase, lamps, theta, watch.Elapsed.TotalSeconds, msm: loss, steps: F(step), faces: 6);
                    return loss;
                }
            }

            private bool CheckMomentDirection(E20Start start, int lamp, MomentSettings cfg, Vector3 derivative,
                E20Target nativeOn, E20Target nativeOff, StreamWriter csv)
            {
                if (derivative.sqrMagnitude < 1e-20) return false;
                Vector3 d = -derivative.normalized;
                var x = IsolatedStart(start.theta, new[] { lamp });
                var p = scene.PositionOf(x, lamp);
                var gtDirection = (scene.PositionOf(scene.Truth, lamp) - p).normalized;
                var scratch = new float[FitViews * ViewChannels];
                var clock = Stopwatch.StartNew();
                int supported = 0;
                foreach (var context in new[] { nativeOn, nativeOff })
                {
                    SetMode(context.Shadows, -1);
                    string id = $"moment_native_{context.Name}_L{lamp + 1}_{start.seed}_{start.profile}";
                    int q = 0;
                    double centre = Measure(x, "centre", 0);
                    foreach (double alpha in NativeDirectionSteps)
                    {
                        var a = scene.WithPosition(x, lamp, p + (float)alpha * d);
                        var b = scene.WithPosition(x, lamp, p - (float)alpha * d);
                        double plus = scene.Contains(a) ? Measure(a, "plus", alpha) : double.NaN;
                        double minus = scene.Contains(b) ? Measure(b, "minus", -alpha) : double.NaN;
                        bool lowers = plus < centre - Math.Max(1e-12, centre * 1e-8) && plus < minus;
                        if (context.Shadows && lowers) supported++;
                        csv.WriteLine(string.Join(",", lamp + 1, cfg.label, start.seed, start.profile, context.Shadows ? "on" : "off",
                            F(alpha), F(centre), Value(minus), Value(plus), Value((plus - minus) / (2 * alpha)), lowers ? 1 : 0,
                            F(-derivative.magnitude), F(d.x), F(d.y), F(d.z), F(Vector3.Dot(d, gtDirection))));
                    }
                    double Measure(double[] theta, string phase, double h)
                    {
                        double loss = NativeResidual(context, theta, scratch, FitViews, "moment_native_diagnostics");
                        LogQuery(id, "diagnostic", ++q, phase, new[] { lamp }, theta, clock.Elapsed.TotalSeconds,
                            native: loss, steps: F(h), predicted: -derivative.magnitude, cameraRenders: FitViews);
                        return loss;
                    }
                }
                return supported >= 3;
            }

            private E20Arm RecoverMoment(E20Start start, int lamp, MomentSettings cfg, ShadowFeature feature, PointMomentShadow shadow,
                E20Target nativeOn, Vector3 initialDirection)
            {
                int cap = Math.Min(budget, 180);
                var lamps = new[] { lamp };
                var row = NewArm("MSM_XYZ", nativeOn, start, lamps, cap);
                row.proxyBudget = cap;
                row.start[4 * lamp + 3] = scene.Truth[4 * lamp + 3];
                var scratch = (double[])row.start.Clone();
                var watch = Stopwatch.StartNew();
                SetMode(true, -1);
                row.nativeInitial = NativeResidual(nativeOn, row.start, new float[FitViews * ViewChannels], FitViews, "validation");
                row.validationRenders = FitViews;
                LogQuery(row.id, "validation", 0, "initial_native", lamps, row.start, watch.Elapsed.TotalSeconds,
                    native: row.nativeInitial, cameraRenders: FitViews);
                int faces = shadow.Faces, nonfinite = shadow.Nonfinite;
                string probeStep = "";
                double[] candidate = null;
                var options = new ImageJacobianLm.Diagnostics {
                    ProbeSteps = units.Take(3).Select(u => MsmStep / u).ToArray(),
                    Probe = (q, axis, h) => probeStep = AxisLabel(4 * lamp + axis) + "=" + F(h * units[axis]),
                    Jacobian = (batch, q, loss, dx, stencil, gram) =>
                        LogSpectrum(row.id, "moment_proxy", row.seed, row.profile, batch, q, -1, MsmStep, double.NaN,
                            dx.Select((h, axis) => h * units[axis]).ToArray(), stencil, gram),
                    Trial = (batch, q, damping, predicted, actual, ratio, bounds, shrinks, status) =>
                        LogQuery(row.id, "proxy_decision", row.proxyQueries, status, lamps, status == "accept" || status == "reject" ? candidate : null,
                            watch.Elapsed.TotalSeconds, predicted: -predicted, accept: status == "accept" ? "1" : "0")
                };
                var result = ImageJacobianLm.Run(row.start.Skip(4 * lamp).Take(3).ToArray(), feature.fitSize, (x, phase, r) =>
                {
                    var theta = Expand(x);
                    var vis = shadow.Render(scene.PositionOf(theta, lamp));
                    double loss = feature.Loss(vis, r);
                    if (phase == "lm_trial") candidate = (double[])theta.Clone();
                    LogQuery(row.id, "proxy", ++row.proxyQueries, phase, lamps, theta, watch.Elapsed.TotalSeconds, msm: loss, steps: probeStep, faces: 6);
                    probeStep = "";
                    return loss;
                }, cap, row.seed, x => scene.Contains(Expand(x)), diagnostics: options);
                row.theta = (double[])Expand(result.Best).Clone();
                row.initial = result.InitialLoss; row.best = result.BestLoss;
                row.queries = result.Queries; row.stop = result.StopReason;
                row.accepted = result.Accepted; row.rejected = result.Rejected; row.jacobians = result.Jacobians;
                row.blocked = result.BlockedProbes; row.shortened = result.ShortenedTrials;
                row.shadowFaces = shadow.Faces - faces; row.nonfinite = shadow.Nonfinite - nonfinite;
                row.seconds = watch.Elapsed.TotalSeconds;
                ValidateNative(row, nativeOn, save: true);
                var path = CapturePath(row.id);
                var initial = shadow.Render(scene.PositionOf(row.start, lamp));
                feature.Save(path, "initial_visibility", initial);
                var best = shadow.Render(scene.PositionOf(row.theta, lamp));
                feature.Save(path, "best_visibility", best);
                shadow.SaveFaces(path, "best");
                row.recheck = feature.Loss(best);
                row.proxyHeldNrmse = Math.Sqrt(feature.Loss(best, held: true));
                using (var csv = new StreamWriter(Path.Combine(path, "recheck.csv")))
                {
                    csv.WriteLine("setting,initial_gradient_norm,train_feature_nrmse,held_feature_nrmse");
                    csv.WriteLine(string.Join(",", cfg.label, F(initialDirection.magnitude), F(Math.Sqrt(row.recheck)), F(row.proxyHeldNrmse)));
                }
                AddArm(row);
                return row;

                double[] Expand(double[] x)
                {
                    Array.Copy(x, 0, scratch, 4 * lamp, 3);
                    return scratch;
                }
            }

            private void AcceptMomentProposal(E20Start start, int lamp, MomentSettings cfg, ShadowFeature feature, PointMomentShadow shadow,
                E20Target target, E20Arm xyz)
            {
                const int cap = 15;
                var lamps = new[] { lamp };
                var row = NewArm("MSM_4D", target, start, lamps, cap);
                row.nativeBudget = 9; row.proxyBudget = 6;
                var watch = Stopwatch.StartNew();
                int f0 = shadow.Faces, n0 = shadow.Nonfinite;
                SetMode(true, -1);
                var r = new float[FitViews * ViewChannels];
                var hi = new float[r.Length]; var lo = new float[r.Length];
                double anchor = Native(row.start, "centre", r);
                row.nativeInitial = anchor;
                row.initial = Moment(row.start);
                row.best = row.initial;
                row.theta = (double[])row.start.Clone();
                double best = anchor;
                int strength = 4 * lamp + 3;
                var update = (double[])xyz.theta.Clone();
                update[strength] = row.start[strength];
                Native(update, "xyz_proposal_at_initial_intensity", r);
                var a = (double[])update.Clone(); var b = (double[])update.Clone();
                a[strength] = Math.Min(1, a[strength] + .05 / LogPowerSpan);
                b[strength] = Math.Max(0, b[strength] - .05 / LogPowerSpan);
                Native(a, "log_intensity_plus", hi); Native(b, "log_intensity_minus", lo);
                float inverse = (float)(1 / (a[strength] - b[strength]));
                for (int k = 0; k < hi.Length; k++) hi[k] = (hi[k] - lo[k]) * inverse;
                double dh = -RgbDot(hi, r) / Math.Max(1e-18, RgbDot(hi, hi));
                dh = Math.Max(-.15, Math.Min(.15, dh));
                update[strength] = Math.Max(0, Math.Min(1, row.start[strength] + dh));
                for (int i = 0; i < 5; i++)
                {
                    double step = Math.Pow(.5, i);
                    var x = (double[])row.start.Clone();
                    for (int k = 4 * lamp; k < 4 * lamp + 4; k++) x[k] += step * (update[k] - x[k]);
                    if (!scene.Contains(x)) continue;
                    double proxy = Moment(x);
                    double n = Native(x, "joint_xyz_log_intensity_trial", r);
                    bool accept = n < best - Math.Max(1e-12, best * 1e-8);
                    LogQuery(row.id, "decision", row.nativeQueries, "joint_trial", lamps, x, watch.Elapsed.TotalSeconds,
                        native: n, msm: proxy, steps: F(step), predicted: proxy - row.initial, actual: n - best, accept: accept ? "1" : "0");
                    if (accept) { row.theta = x; best = n; row.best = proxy; row.accepted++; }
                    else row.rejected++;
                }
                row.queries = row.proxyQueries + row.nativeQueries;
                row.shadowFaces = shadow.Faces - f0; row.nonfinite = shadow.Nonfinite - n0;
                row.stop = row.accepted > 0 ? "best_native_accepted_proposal" : "native_rejected_all_proposals";
                row.seconds = watch.Elapsed.TotalSeconds;
                ValidateNative(row, target, save: true);
                var actual = shadow.Render(scene.PositionOf(row.theta, lamp));
                row.recheck = feature.Loss(actual);
                row.proxyHeldNrmse = Math.Sqrt(feature.Loss(actual, held: true));
                if (Math.Abs(best - row.nativeBest) > Math.Max(1e-8, best * 1e-6))
                    throw new InvalidOperationException("MSM 4D candidate did not replay its native loss.");
                AddArm(row);

                double Moment(double[] x)
                {
                    double value = feature.Loss(shadow.Render(scene.PositionOf(x, lamp)));
                    LogQuery(row.id, "proxy", ++row.proxyQueries, cfg.label, lamps, x, watch.Elapsed.TotalSeconds, msm: value, faces: 6);
                    return value;
                }
                double Native(double[] x, string phase, float[] buffer)
                {
                    double value = NativeResidual(target, x, buffer, FitViews, "native_optimization");
                    row.cameraRenders += FitViews;
                    LogQuery(row.id, "optimizer", ++row.nativeQueries, phase, lamps, x, watch.Elapsed.TotalSeconds, native: value, cameraRenders: FitViews);
                    return value;
                }
            }
        }
    }
}

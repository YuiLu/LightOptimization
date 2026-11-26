using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using UnityEngine;
using Frame = LightingMvp.NativeRenderBackend.Frame;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed class DirectFeature
        {
            private readonly ProxyGeometry.Receiver[] samples;
            private readonly Vector3 lightColor;
            private readonly float inverseRangeSquared;
            private readonly Vector3[] target;
            private readonly double[] denominators;
            internal int FitLength { get; }

            internal DirectFeature(ProxyGeometry.Receiver[] receivers, Light light, Frame[] isolated, Frame[] dark)
            {
                samples = receivers;
                lightColor = ProxyGeometry.Rgb(light.color.linear);
                inverseRangeSquared = 1 / (light.range * light.range);
                target = samples.Select(s => ProxyGeometry.Rgb(isolated[s.View].Pixel(s.Pixel) - dark[s.View].Pixel(s.Pixel))).ToArray();
                var energies = new double[isolated.Length];
                for (int i = 0; i < target.Length; i++) energies[samples[i].View] += target[i].sqrMagnitude;
                double floor = Math.Max(1e-12, energies.Take(E19FitViews).Max() * 1e-4);
                denominators = energies.Select(x => Math.Max(floor, x)).ToArray();
                FitLength = 3 * samples.Count(s => s.View < E19FitViews);
            }

            private Vector3 Predict(ProxyGeometry.Receiver receiver, Vector3 position, double intensity)
            {
                var v = position - receiver.Position;
                float d2 = Mathf.Max(6.10352e-5f, v.sqrMagnitude);
                float f = d2 * inverseRangeSquared;
                float fade = Mathf.Max(0, 1 - f * f);
                float incident = Mathf.Max(0, Vector3.Dot(v, receiver.Normal)) * fade * fade / (d2 * Mathf.Sqrt(d2));
                return Vector3.Scale(receiver.Diffuse, lightColor) * (float)(intensity * incident);
            }

            internal double Loss(Vector3 position, double intensity, float[] buffer, bool held = false)
            {
                double loss = 0;
                int k = 0;
                int views = held ? denominators.Length - E19FitViews : E19FitViews;
                for (int i = 0; i < samples.Length; i++)
                {
                    var s = samples[i];
                    if ((s.View >= E19FitViews) != held) continue;
                    var error = Predict(s, position, intensity) - target[i];
                    double weight = 1 / (views * denominators[s.View]);
                    loss += (double)error.x * error.x * weight + (double)error.y * error.y * weight + (double)error.z * error.z * weight;
                    if (buffer == null) continue;
                    float inv = (float)Math.Sqrt(weight);
                    buffer[k++] = error.x * inv; buffer[k++] = error.y * inv; buffer[k++] = error.z * inv;
                }
                return loss;
            }

            internal void Save(string dir, string label, Vector3 p, double intensity, bool observed = false)
            {
                var values = observed ? target : samples.Select(s => Predict(s, p, intensity)).ToArray();
                SaveReceiverFeature(samples, values, dir, label, visibility: false);
            }
        }

        private static void SaveReceiverFeature(ProxyGeometry.Receiver[] samples, Vector3[] values, string dir, string label, bool visibility)
        {
            using (var stream = new BinaryWriter(File.Create(Path.Combine(dir, label + ".f32"))))
                foreach (var v in values) { stream.Write(v.x); stream.Write(v.y); stream.Write(v.z); }
            var texture = new Texture2D(ProxyGeometry.GridWidth, ProxyGeometry.GridHeight, TextureFormat.RGB24, false, true);
            try
            {
                int count = ProxyGeometry.GridWidth * ProxyGeometry.GridHeight;
                for (int v = 0; v < samples[samples.Length - 1].View + 1; v++)
                {
                    var pixels = new Color32[count];
                    for (int i = 0; i < samples.Length; i++)
                    {
                        if (samples[i].View != v) continue;
                        var f = values[i];
                        pixels[samples[i].Grid] = visibility ? new Color(f.x, f.y, f.z) :
                            new Color(NativeRenderBackend.Display(f.x), NativeRenderBackend.Display(f.y), NativeRenderBackend.Display(f.z));
                    }
                    texture.SetPixels32(pixels); texture.Apply();
                    File.WriteAllBytes(Path.Combine(dir, $"{label}_v{v}.png"), texture.EncodeToPNG());
                }
            }
            finally { UnityEngine.Object.DestroyImmediate(texture); }
        }

        private sealed partial class E20Experiment
        {
            private Frame[] CaptureDark()
            {
                SetMode(false, 0);
                lights[0].enabled = false;
                scene.Apply(scene.Truth);
                var dark = Capture(fit, scene.Views.Length, "oracle_dark");
                using var csv = OpenCsv("oracle-dark.csv", "view,mean_rgb_energy,peak_rgb");
                for (int v = 0; v < dark.Length; v++)
                {
                    csv.WriteLine(string.Join(",", v, F(dark[v].Energy / dark[v].PixelCount), F(dark[v].Peak)));
                    if (dark[v].Peak > .02f) throw new InvalidOperationException("Direct-light target has unrelated illumination.");
                }
                return dark;
            }

            private void RunDirectProxy()
            {
                using var geometry = new ProxyGeometry(scene.Camera);
                var samples = geometry.Capture(scene, dir);
                var full = Target(-1, false);
                var dark = CaptureDark();
                string baseline = Arg("-bbE20B0Source", "");
                using var fitBefore = OpenCsv("forward-match.csv", "lamp,train_proxy_nrmse_at_gt,held_proxy_nrmse_at_gt");
                using var directions = OpenCsv("direct-fd.csv", "seed,profile,lamp,scale,axis,step,stencil,proxy_derivative,native_derivative,direction_cosine");
                using var references = OpenCsv("b0-reference.csv", "seed,profile,lamp,baseline_arm,baseline_scene_sha256");
                var current = JsonUtility.FromJson<E20Configuration>(File.ReadAllText(Path.Combine(dir, "config.json")));
                if (baseline.Length != 0)
                {
                    var prior = JsonUtility.FromJson<E20Configuration>(File.ReadAllText(Path.Combine(baseline, "config.json")));
                    if (prior.baseCommit != current.baseCommit || prior.sceneSha256 != current.sceneSha256 || prior.queryBudget != budget ||
                        prior.scene.unity != current.scene.unity || prior.scene.urp != current.scene.urp || prior.scene.graphics != current.scene.graphics)
                        throw new InvalidDataException("B0 reference must use the same E19 scene, render backend and native query budget.");
                }
                foreach (int l in smoke ? new[] { 0 } : scene.Lamps)
                {
                    var oracle = Target(l, false);
                    var feature = new DirectFeature(samples, lights[l], oracle.Fit, dark);
                    var pos = scene.PositionOf(scene.Truth, l);
                    var strength = scene.PowerOf(scene.Truth, l);
                    fitBefore.WriteLine(string.Join(",", l + 1, F(Math.Sqrt(feature.Loss(pos, strength, null))),
                        F(Math.Sqrt(feature.Loss(pos, strength, null, held: true)))));
                    feature.Save(CapturePath($"direct_L{l + 1}"), "gt_observed_direct", pos, strength, observed: true);
                    feature.Save(CapturePath($"direct_L{l + 1}"), "gt_predicted_direct", pos, strength);
                    foreach (var start in starts)
                    {
                        if (baseline.Length == 0) NativeArm("B0", full, start, new[] { l });
                        else
                        {
                            string key = $"A1_{start.seed}_{start.profile}_L{l + 1}";
                            var a1 = JsonUtility.FromJson<E20Arm>(File.ReadAllText(Path.Combine(baseline, key + ".json")));
                            var expected = IsolatedStart(start.theta, new[] { l });
                            bool sameStart = a1.start?.Length == expected.Length &&
                                a1.start.Zip(expected, (a, b) => Math.Abs(a - b)).All(delta => delta <= 1e-12);
                            if (a1.method != "A1" || a1.condition != full.Name || a1.budget != budget ||
                                !sameStart)
                                throw new InvalidDataException("B0 and E20-A1 controls differ: " + key);
                            references.WriteLine(string.Join(",", start.seed, start.profile, l + 1, a1.id, current.sceneSha256));
                        }
                        CompareDirectDerivatives(full, start, l, feature, directions);
                        ProxyArm("B1", full, start, l, feature, false);
                        ProxyArm("B2", full, start, l, feature, true);
                    }
                }
            }

            private void CompareDirectDerivatives(E20Target full, E20Start start, int lamp, DirectFeature feature, StreamWriter csv)
            {
                SetMode(false, -1);
                var x = IsolatedStart(start.theta, new[] { lamp });
                var scratch = new float[FitViews * ViewChannels];
                var watch = Stopwatch.StartNew();
                string key = $"direct_fd_{start.seed}_{start.profile}_L{lamp + 1}";
                var lamps = new[] { lamp };
                int q = 0;
                var centre = Measure(x, "centre", 0);
                for (int s = 0; s < PositionSteps.Length; s++)
                {
                    var data = new (double step, int stencil, double proxy, double rgb)[4];
                    for (int a = 0; a < 4; a++)
                    {
                        double h = a == 3 ? IntensitySteps[s] : PositionSteps[s];
                        var plus = (double[])x.Clone(); plus[4 * lamp + a] += h / units[a];
                        var minus = (double[])x.Clone(); minus[4 * lamp + a] -= h / units[a];
                        bool p = scene.Contains(plus), m = scene.Contains(minus);
                        if (!(p || m)) { data[a] = (h, 0, double.NaN, double.NaN); continue; }
                        var a1 = p ? Measure(plus, AxisNames[a] + "_plus", h) : centre;
                        var b1 = m ? Measure(minus, AxisNames[a] + "_minus", -h) : centre;
                        double delta = p && m ? 2 * h : h;
                        data[a] = (h, p && m ? 2 : p ? 1 : -1, (a1.direct - b1.direct) / delta, (a1.native - b1.native) / delta);
                    }
                    var legal = data.Where(d => d.stencil != 0).ToArray();
                    double cosine = Cosine(legal.Sum(d => d.proxy * d.rgb), legal.Sum(d => Sq20(d.proxy)), legal.Sum(d => Sq20(d.rgb)));
                    for (int a = 0; a < 4; a++)
                    {
                        var d = data[a];
                        csv.WriteLine(string.Join(",", start.seed, start.profile, lamp + 1, s, AxisNames[a], F(d.step),
                            d.stencil, Value(d.proxy), Value(d.rgb), Value(cosine)));
                    }
                }

                (double native, double direct) Measure(double[] theta, string phase, double h)
                {
                    double n = NativeResidual(full, theta, scratch, FitViews, "direct_fd_diagnostics");
                    double p = feature.Loss(scene.PositionOf(theta, lamp), scene.PowerOf(theta, lamp), null);
                    LogQuery(key, "diagnostic", ++q, phase, lamps, theta, watch.Elapsed.TotalSeconds, native: n, direct: p,
                        steps: F(h), cameraRenders: FitViews);
                    return (n, p);
                }
            }

            private void ProxyArm(string name, E20Target full, E20Start start, int lamp, DirectFeature feature, bool acceptNative)
            {
                int proxyBudget = Math.Min(budget, acceptNative ? 600 : 400);
                var lamps = new[] { lamp };
                var row = NewArm(name, full, start, lamps, proxyBudget);
                row.nativeBudget = acceptNative ? budget : 0;
                row.proxyBudget = proxyBudget;
                var timer = Stopwatch.StartNew();
                var fullScratch = (double[])row.start.Clone();
                var nativeScratch = new float[FitViews * ViewChannels];
                var current = (double[])row.start.Clone();
                SetMode(false, -1);
                double native = EvaluateNative(current, "initial", double.NaN);
                row.nativeInitial = native;
                row.initial = Loss(current, null);
                double proxy = row.initial;
                var steps = units.Select((u, j) => (j == 3 ? .05 : .02) / u).ToArray();
                row.stop = "proxy_budget";
                double radius = .18, logRadius = .22;
                while (proxyBudget - row.proxyQueries >= 16)
                {
                    var anchor = current;
                    int cap = acceptNative ? Math.Min(40, proxyBudget - row.proxyQueries) : proxyBudget - row.proxyQueries;
                    string probeStep = "";
                    double[] candidateForDecision = null;
                    var diagnostics = new ImageJacobianLm.Diagnostics {
                        ProbeSteps = steps,
                        Probe = (q, axis, h) => probeStep = AxisLabel(4 * lamp + axis) + "=" + F(h * units[axis]),
                        Jacobian = (batch, queriesNow, value, dx, stencil, gram) =>
                            LogSpectrum(row.id, "direct_proxy", row.seed, row.profile, batch, row.proxyQueries, -1, .02, .05,
                                dx.Select((h, j) => h * units[j]).ToArray(), stencil, gram),
                        Trial = (batch, q, damping, predicted, actual, ratio, bound, shrinks, status) =>
                            LogQuery(row.id, "proxy_decision", row.proxyQueries, status, lamps,
                                status == "accept" || status == "reject" ? candidateForDecision : null,
                                timer.Elapsed.TotalSeconds, predicted: -predicted, accept: status == "accept" ? "1" : "0")
                    };
                    var local = ImageJacobianLm.Run(anchor.Skip(4 * lamp).Take(4).ToArray(), feature.FitLength,
                        (x, phase, r) =>
                        {
                            var all = Expand(x);
                            double f = Loss(all, r);
                            if (phase == "lm_trial") candidateForDecision = (double[])all.Clone();
                            LogQuery(row.id, "proxy", ++row.proxyQueries, phase, lamps, all, timer.Elapsed.TotalSeconds, direct: f, steps: probeStep);
                            probeStep = "";
                            return f;
                        }, cap, row.seed, x =>
                        {
                            var all = Expand(x);
                            return scene.Contains(all) && (!acceptNative ||
                                (Vector3.Distance(scene.PositionOf(all, lamp), scene.PositionOf(anchor, lamp)) <= radius &&
                                Math.Abs(x[3] - anchor[4 * lamp + 3]) * LogPowerSpan <= logRadius));
                        }, diagnostics: diagnostics);
                    row.jacobians += local.Jacobians; row.blocked += local.BlockedProbes; row.shortened += local.ShortenedTrials;
                    var candidate = (double[])Expand(local.Best).Clone();
                    if (!acceptNative)
                    {
                        current = candidate; proxy = local.BestLoss; row.stop = local.StopReason;
                        row.accepted = local.Accepted; row.rejected = local.Rejected;
                        break;
                    }
                    if (proxy - local.BestLoss <= 1e-12)
                    {
                        row.stop = "no_decreasing_proxy_proposal";
                        break;
                    }
                    bool accepted = false;
                    for (int step = 0; step < 5 && row.nativeQueries < budget; step++)
                    {
                        double fraction = Math.Pow(.5, step);
                        var trial = (double[])current.Clone();
                        for (int c = 4 * lamp; c < 4 * lamp + 4; c++) trial[c] += fraction * (candidate[c] - current[c]);
                        if (!scene.Contains(trial)) continue;
                        double trialProxy = Loss(trial, null);
                        double actual = EvaluateNative(trial, "trust_trial", trialProxy);
                        accepted = actual < native - Math.Max(1e-12, 1e-8 * native);
                        LogQuery(row.id, "decision", row.nativeQueries, "trust_trial", lamps, trial, timer.Elapsed.TotalSeconds,
                            native: actual, direct: trialProxy, steps: Fields(new[] { radius, logRadius, fraction }),
                            predicted: trialProxy - proxy, actual: actual - native, accept: accepted ? "1" : "0");
                        if (accepted)
                        {
                            current = trial; native = actual; proxy = trialProxy; row.accepted++;
                            if (step == 0) { radius = Math.Min(.40, radius * 1.4); logRadius = Math.Min(.65, logRadius * 1.4); }
                            break;
                        }
                        row.rejected++;
                    }
                    if (native <= 1e-12) { row.stop = "near_zero_native_residual"; break; }
                    if (!accepted) { radius *= .5; logRadius *= .5; }
                    if (radius < .006) { row.stop = "native_rejects_proxy_direction"; break; }
                    if (row.nativeQueries >= budget) { row.stop = "native_budget"; break; }
                }
                row.theta = current; row.best = proxy; row.recheck = Loss(current, null);
                row.proxyHeldNrmse = Math.Sqrt(Loss(current, null, held: true));
                row.queries = row.proxyQueries + row.nativeQueries;
                row.seconds = timer.Elapsed.TotalSeconds;
                ValidateNative(row, full, save: true);
                if (acceptNative && Math.Abs(row.nativeBest - native) > Math.Max(1e-8, native * 1e-6))
                    throw new InvalidOperationException("The direct trial's native acceptance failed to replay.");
                var path = CapturePath(row.id);
                feature.Save(path, "initial_direct", scene.PositionOf(row.start, lamp), scene.PowerOf(row.start, lamp));
                feature.Save(path, "best_direct", scene.PositionOf(row.theta, lamp), scene.PowerOf(row.theta, lamp));
                AddArm(row);

                double[] Expand(double[] x)
                {
                    Array.Copy(x, 0, fullScratch, 4 * lamp, 4);
                    return fullScratch;
                }
                double Loss(double[] theta, float[] r, bool held = false) =>
                    feature.Loss(scene.PositionOf(theta, lamp), scene.PowerOf(theta, lamp), r, held);
                double EvaluateNative(double[] theta, string phase, double value)
                {
                    double f = NativeResidual(full, theta, nativeScratch, FitViews, acceptNative ? "native_optimization" : "validation");
                    if (acceptNative) { row.nativeQueries++; row.cameraRenders += FitViews; }
                    else row.validationRenders += FitViews;
                    LogQuery(row.id, acceptNative ? "optimizer" : "validation", row.nativeQueries, phase, lamps, theta,
                        timer.Elapsed.TotalSeconds, native: f, direct: value, cameraRenders: FitViews);
                    return f;
                }
            }
        }
    }
}

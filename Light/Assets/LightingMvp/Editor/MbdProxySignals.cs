using System;
using System.IO;
using System.Linq;
using UnityEngine;
using Frame = LightingMvp.NativeRenderBackend.Frame;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class MbdStudy
        {
            internal void Recovery()
            {
                using var receiverGeometry = new ProxyGeometry(scene.Camera, includeTextured: true);
                var sites = receiverGeometry.Capture(scene, dir).Where(s => s.View < TrainCount).ToArray();
                mapSites = sites.Select(s => s.Position).ToArray();
                sampleCount = sites.Length;
                var target = Target(true);
                var dark = SoloFrames(scene.Truth, -1, false);
                using var match = Csv("forward-match.csv", "lamp,proxy_nrmse_at_gt,unshadowed_energy,shadowed_energy");
                foreach (var group in ClassroomStarts().GroupBy(s => s.lamp))
                {
                    int lamp = group.Key;
                    var hard = SoloFrames(scene.Truth, lamp, true);
                    var observed = sites.Select(s => ProxyGeometry.Rgb(hard[s.View].Pixel(s.Pixel) - dark[s.View].Pixel(s.Pixel))).ToArray();
                    var off = SoloFrames(scene.Truth, lamp, false);
                    var unshadowed = sites.Select(s => ProxyGeometry.Rgb(off[s.View].Pixel(s.Pixel) - dark[s.View].Pixel(s.Pixel))).ToArray();
                    var energy = new double[TrainCount];
                    for (int i = 0; i < sites.Length; ++i) energy[sites[i].View] += observed[i].sqrMagnitude;
                    double floor = Math.Max(1e-10, energy.Max() * 1e-4);
                    var normalizer = energy.Select(e => 1 / Math.Sqrt(TrainCount * Math.Max(e, floor))).ToArray();
                    double observedEnergy = energy.Select((e, i) => e * normalizer[i] * normalizer[i]).Sum();
                    if (!(observedEnergy > 0)) throw new InvalidOperationException("The isolated native shadowed RGB target is zero.");
                    var signal = new MbdSignal("mbd_times_unshadowed_native_rgb", 3 * sites.Length, observedEnergy,
                        (x, residual) =>
                        {
                            var frames = SoloFrames(x, lamp, false);
                            var visibility = Visibility(lamp, x);
                            double loss = 0;
                            for (int i = 0; i < sites.Length; ++i)
                            {
                                var site = sites[i];
                                var rgb = ProxyGeometry.Rgb(frames[site.View].Pixel(site.Pixel) - dark[site.View].Pixel(site.Pixel));
                                Vector3 difference = (rgb * visibility[i] - observed[i]) * (float)normalizer[site.View];
                                loss += WriteResidual(residual, 3 * i, difference);
                            }
                            return loss;
                        });
                    match.WriteLine(string.Join(",", lamp + 1, F(Math.Sqrt(signal.residual(scene.Truth, null) / observedEnergy)),
                        F(unshadowed.Sum(p => (double)p.sqrMagnitude)), F(energy.Sum())));
                    string featurePath = CapturePath("L" + (lamp + 1) + "_gt_camera_samples");
                    SaveReceiverFeature(sites, observed, featurePath, "native_hard_rgb", false);
                    SaveReceiverFeature(sites, unshadowed, featurePath, "native_off_rgb", false);
                    foreach (var start in group)
                    {
                        ProbeAlignment(start.key, lamp, start.start, signal, target);
                        Fit(start.key, "C0", new[] { lamp }, start.start, target);
                        Fit(start.key, "C1", new[] { lamp }, start.start, target, signal, budget: 240);
                        Fit(start.key, "C2", new[] { lamp }, start.start, target, signal, nativeAcceptance: true, budget: 260);
                    }
                }
            }

            private Frame[] SoloFrames(double[] theta, int slot, bool shadows)
            {
                Apply(theta, false);
                for (int l = 0; l < lights.Length; ++l) lights[l].enabled = l == slot;
                if (slot >= 0) scene.SetShadows(slot, shadows);
                return Enumerable.Range(0, TrainCount).Select(fit.Capture).ToArray();
            }

            internal void Irradiance(bool complexRoom, MbdTrial joint = null)
            {
                var sites = geometry.Samples(smoke ? 936 : complexRoom ? 4096 : 3072, dir);
                mapSites = sites.Select(v => v.position).ToArray();
                sampleCount = sites.Length;
                int width = 256, height = (sites.Length * lights.Length + width - 1) / width;
                using var capture = new NativeIrradianceCapture(scene.Camera, width, height);
                capture.SetSurfaceSamples(mapSites, sites.Select(v => v.normal).ToArray(), geometry.Bounds);
                int[] all = Enumerable.Range(0, lights.Length).ToArray();
                Color[] Read(double[] theta, int[] active, bool shadows)
                {
                    Apply(theta, shadows);
                    surfaceReads++;
                    return capture.ReadSurfaces(active.Select(i => lights[i]).ToArray());
                }
                var off = Read(scene.Truth, all, false);
                if (off.Take(sites.Length * lights.Length).Any(c => c.a < .99f))
                    throw new InvalidOperationException("Unshadowed URP probe capture did not cover the fixed surface samples.");
                if (joint != null)
                {
                    RunSeparation(joint);
                    return;
                }
                var on = Read(scene.Truth, all, true);
                var gtMbd = all.Select(i => Visibility(i, scene.Truth)).ToArray();
                using (var table = Csv("surface-gt.csv", "lamp,sample,gt_unshadowed_r,gt_unshadowed_g,gt_unshadowed_b,gt_hard_r,gt_hard_g,gt_hard_b,native_hard_visibility,mbd_visibility"))
                for (int l = 0; l < lights.Length; ++l)
                for (int s = 0, k = l * sites.Length; s < sites.Length; ++s, ++k)
                    table.WriteLine(string.Join(",", l + 1, s, Xyz(ProxyGeometry.Rgb(off[k])), Xyz(ProxyGeometry.Rgb(on[k])), F(on[k].a), F(gtMbd[l][s])));

                var nativeOn = Target(true);
                var nativeOff = Target(false);
                MbdSignal Signal(int[] active, int model) => SurfaceSignal(sites, active, model, model == 0 ? off : on, Read);
                void RunSeparation(MbdTrial baseline)
                {
                    var expected = E19Start(scene, scene.Start(733), .45f);
                    if (baseline.scene != scene.Name || baseline.method != "D0" || baseline.lamps.Length != lights.Length ||
                        expected.Length != baseline.initial.Length || expected.Zip(baseline.initial, (a, b) => Math.Abs(a - b)).Any(d => d > 1e-12))
                        throw new InvalidDataException("The E19 D0 joint reference differs from the isolated-lamp study.");
                    using var selection = Csv("separation-selection.csv", "lamp,selected_for,joint_position_cm,joint_intensity_relative,joint_stop");
                    if (baseline.passed) return;
                    int position = Array.IndexOf(baseline.positionCm, baseline.positionCm.Max());
                    int intensity = Array.IndexOf(baseline.intensityRelative, baseline.intensityRelative.Max());
                    var target = Target(false);
                    foreach (int index in new[] { position, intensity }.Distinct())
                    {
                        int lamp = baseline.lamps[index] - 1;
                        string criterion = position == intensity ? "position_and_intensity" : index == position ? "position" : "intensity";
                        selection.WriteLine(string.Join(",", lamp + 1, criterion, F(baseline.positionCm[index]), F(baseline.intensityRelative[index]), baseline.stop));
                        var single = new[] { lamp };
                        var start = Combine(scene.Truth, new[] { (lamp, expected) });
                        Fit($"seed733_near_L{lamp + 1}", "E_D0_single", single, start, target,
                            SurfaceSignal(sites, single, 0, off, Read), budget: 225);
                    }
                }
                using (var match = Csv("forward-match.csv", "lamp,off_model_nrmse_at_gt,native_hard_repeat_nrmse_at_gt,mbd_model_nrmse_at_gt"))
                    foreach (int l in all)
                        match.WriteLine(string.Join(",", l + 1, string.Join(",", Enumerable.Range(0, 3).Select(m =>
                        {
                            var signal = Signal(new[] { l }, m);
                            return F(Math.Sqrt(signal.residual(scene.Truth, null) / signal.targetEnergy));
                        }))));

                if (complexRoom)
                {
                    var start = E19Start(scene, scene.Start(733), .45f);
                    Compare("seed733_near_all", all, start, 520, false);
                }
                else
                {
                    var starts = ClassroomStarts().ToArray();
                    foreach (var entry in starts) Compare(entry.key, new[] { entry.lamp }, entry.start, 225, entry.key.Contains("30cm"));
                    if (!smoke)
                    {
                        var far = starts.Where(e => e.key.Contains("30cm")).ToArray();
                        Compare("L5_L6_30cm_d0", far.Select(x => x.lamp).ToArray(),
                            Combine(scene.Truth, far.Select(s => (s.lamp, s.start))), 320, false);
                    }
                }

                void Compare(string key, int[] group, double[] start, int budget, bool diagnostics)
                {
                    var d0 = Signal(group, 0);
                    var d1 = Signal(group, 1);
                    var d2 = Signal(group, 2);
                    if (diagnostics)
                    {
                        ProbeAlignment(key, group[0], start, d0, nativeOff);
                        ProbeAlignment(key, group[0], start, d2, nativeOn);
                    }
                    Fit(key, "D0", group, start, nativeOff, d0, budget: budget);
                    Fit(key, "D1", group, start, nativeOn, d1, budget: budget);
                    Fit(key, "D2", group, start, nativeOn, d2, budget: budget);
                    Fit(key, "D3", group, start, nativeOn, d2, nativeAcceptance: true, budget: budget);
                }
            }

            private MbdSignal SurfaceSignal(MbdSurface[] sites, int[] active, int model, Color[] target,
                Func<double[], int[], bool, Color[]> nativeRead)
            {
                double meanArea = 1 / (sites.Sum(s => (double)s.area) * active.Length);
                var observed = active.SelectMany(l => Enumerable.Range(l * sites.Length, sites.Length).Select(k => ProxyGeometry.Rgb(target[k]))).ToArray();
                var weights = sites.Select(s => (float)Math.Sqrt(s.area * meanArea)).ToArray();
                double norm = observed.Select((v, i) => (v * weights[i % sites.Length]).sqrMagnitude).Sum(v => (double)v);
                if (!(norm > 1e-12)) throw new InvalidOperationException("No measurable GT direct irradiance.");
                var colors = lights.Select(l => ProxyGeometry.Rgb(l.color.linear)).ToArray();
                var ranges = lights.Select(l => l.range).ToArray();
                return new MbdSignal(model == 0 ? "direct_irradiance_off" : model == 1 ?
                    "native_hard_irradiance" : "mbd_times_direct_irradiance", 3 * observed.Length, norm, (theta, residual) =>
                {
                    var measured = model == 1 ? nativeRead(theta, active, true) : null;
                    double loss = 0;
                    for (int index = 0; index < active.Length; ++index)
                    {
                        int lamp = active[index];
                        var center = scene.PositionOf(theta, lamp);
                        double intensity = scene.PowerOf(theta, lamp);
                        var visibility = model == 2 ? Visibility(lamp, theta) : null;
                        for (int i = 0, k = index * sites.Length; i < sites.Length; ++i, ++k)
                        {
                            Vector3 predicted = measured == null ? colors[lamp] * IncidentIrradiance(sites[i], center, intensity, ranges[lamp]) :
                                ProxyGeometry.Rgb(measured[k]);
                            if (visibility != null) predicted *= visibility[i];
                            loss += WriteResidual(residual, k * 3, (predicted - observed[k]) * weights[i]);
                        }
                    }
                    return loss;
                });
            }

            internal static float IncidentIrradiance(MbdSurface sample, Vector3 p, double intensity, float range)
            {
                var delta = p - sample.position;
                float squared = Mathf.Max(6.10352e-5f, delta.sqrMagnitude);
                float fraction = squared / (range * range);
                float fade = Mathf.Max(0, 1 - fraction * fraction);
                return (float)intensity * Mathf.Clamp01(Vector3.Dot(delta, sample.normal) / Mathf.Sqrt(squared)) * fade * fade / squared;
            }

            private static double WriteResidual(float[] buffer, int i, Vector3 v)
            {
                if (buffer != null) { buffer[i] = v.x; buffer[i + 1] = v.y; buffer[i + 2] = v.z; }
                return (double)v.x * v.x + (double)v.y * v.y + (double)v.z * v.z;
            }

            private void ProbeAlignment(string key, int lamp, double[] x, MbdSignal signal, MbdRgbTarget target)
            {
                var native0 = new float[TrainCount * RgbChannels];
                var proxy0 = new float[signal.size];
                var nativeMinus = new float[native0.Length]; var nativePlus = new float[native0.Length];
                var proxyMinus = new float[proxy0.Length]; var proxyPlus = new float[proxy0.Length];
                double initialNative = NativeLoss(x, target, native0);
                double initialProxy = signal.residual(x, proxy0);
                int query = 0;
                Log(key, signal.name, "probe", ++query, "centre", x, initialProxy, initialNative);
                var previousProxy = new double[4]; var previousNative = new double[4];
                foreach (double metres in new[] { .005, .02, .05 })
                {
                    var gp = new double[4]; var gn = new double[4];
                    var fields = new string[4][];
                    for (int a = 0; a < 4; ++a)
                    {
                        double h = a == 3 ? (metres == .005 ? .02 : metres == .02 ? .05 : .10) : metres;
                        var minus = (double[])x.Clone(); var plus = (double[])x.Clone();
                        minus[4 * lamp + a] -= h / physicalSpan[a]; plus[4 * lamp + a] += h / physicalSpan[a];
                        bool p = scene.Contains(plus), m = scene.Contains(minus);
                        double np = initialNative, nm = initialNative;
                        double lp = p ? Measure(plus, nativePlus, proxyPlus, h, a, out np) : initialProxy;
                        double ln = m ? Measure(minus, nativeMinus, proxyMinus, -h, a, out nm) : initialProxy;
                        double denominator = (m && p ? 2 : 1) * h;
                        gp[a] = (lp - ln) / denominator;
                        var pv = Column(p ? proxyPlus : proxy0, m ? proxyMinus : proxy0, proxy0, denominator, initialProxy);
                        var nv = Column(p ? nativePlus : native0, m ? nativeMinus : native0, native0, denominator, initialNative);
                        gn[a] = (np - nm) / denominator;
                        fields[a] = new[] { a.ToString(Invariant), F(h), m && p ? "central" : p ? "plus" : m ? "minus" : "blocked",
                            F(gp[a]), F(gn[a]), F(pv.norm), F(nv.norm), F(pv.alignment), F(nv.alignment) };
                    }
                    foreach (var row in fields)
                        probeCsv.WriteLine(string.Join(",", key, signal.name, F(metres), string.Join(",", row), F(GradientCosine(gn, gp)),
                            F(GradientCosine(gp, previousProxy)), F(GradientCosine(gn, previousNative))));
                    previousProxy = gp; previousNative = gn;
                }

                double Measure(double[] theta, float[] nr, float[] pr, double step, int axis, out double native)
                {
                    native = NativeLoss(theta, target, nr);
                    double proxy = signal.residual(theta, pr);
                    Log(key, signal.name, "probe", ++query, step > 0 ? "plus" : "minus", theta, proxy, native, $"{axis}={F(step)}");
                    return proxy;
                }
            }

            private static (double gradient, double norm, double alignment) Column(float[] p, float[] m, float[] start, double delta, double initial)
            {
                double projection = 0, normSq = 0;
                for (int i = 0; i < p.Length; ++i)
                {
                    double change = ((double)p[i] - m[i]) / delta;
                    projection += change * start[i]; normSq += change * change;
                }
                return (2 * projection, Math.Sqrt(normSq), normSq > 0 && initial > 0 ? projection / Math.Sqrt(normSq * initial) : double.NaN);
            }

            private static double GradientCosine(double[] a, double[] b)
            {
                double na = a.Sum(x => x * x), nb = b.Sum(x => x * x);
                return na > 0 && nb > 0 ? a.Select((x, i) => x * b[i]).Sum() / Math.Sqrt(na * nb) : double.NaN;
            }
        }
    }
}

using System;
using System.IO;
using System.Linq;
using UnityEngine;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        [Serializable] private sealed class HardStartReport
        {
            public int seed = 47, cameraRenders;
            public bool gtInsideBounds;
            public bool[] gtInFreeSpace, initialInFreeSpace;
            public double[] targetTheta, initialTheta, gradient, pixelJacobianColumnNorm, halfIntensityRelativeRgb;
            public double targetRepeatLoss, initialLoss, wallSeconds;
            public NativeLamp[] targetRig, initialRig;
        }

        private static void Seed47Diagnostic(Camera camera, ViewInfo view, Light[] lights, RecoveryLightState[] gt, BoxCollider block,
            string output, Func<Vector3, bool> free)
        {
            string dir = Path.Combine(output, "e5a_seed47_diagnostic");
            var coords = new RecoveryRigCoordinates(lights, block, false);
            var row = new HardStartReport { targetTheta = coords.Encode(gt),
                initialTheta = coords.Encode(coords.Perturb(gt, 47, Distances[Array.IndexOf(Seeds, 47)]), true),
                gtInsideBounds = true, gtInFreeSpace = gt.Select(s => free(s.position)).ToArray() };
            row.initialInFreeSpace = coords.Decode(row.initialTheta).Select(s => free(s.position)).ToArray();
            using (var backend = Backend(camera, new[] { view }, FitSize))
            using (var log = new QueryLog(dir, "queries.csv", "seed47_diagnostic", 47))
            {
                for (int i = 0; i < lights.Length; ++i) gt[i].Apply(lights[i]);
                row.targetRig = lights.Select(NativeLamp.Read).ToArray();
                NativeRenderBackend.Frame target = null, initial = null;
                log.Measure(backend, "warmup", row.targetTheta, () => { backend.Capture(0); return double.NaN; });
                log.Measure(backend, "target", row.targetTheta, () => { target = backend.Capture(0); return double.NaN; });
                row.targetRepeatLoss = log.Measure(backend, "repeat", row.targetTheta, () => backend.Loss(0, target));
                if (row.targetRepeatLoss > 1e-10) throw new InvalidOperationException("Nonrepeatable seed 47 diagnostic.");
                coords.Apply(row.initialTheta);
                row.initialRig = lights.Select(NativeLamp.Read).ToArray();
                row.initialLoss = log.Measure(backend, "initial", row.initialTheta, () =>
                { initial = backend.Capture(0); return NativeRenderBackend.Compare(initial, target); });
                row.halfIntensityRelativeRgb = lights.Select((l, i) =>
                {
                    float intensity = l.intensity;
                    try
                    {
                        l.intensity *= .5f;
                        return Math.Sqrt(log.Measure(backend, "half_intensity_" + i, null, () => backend.Loss(0, initial)));
                    }
                    finally { l.intensity = intensity; }
                }).ToArray();
                var x = (double[])row.initialTheta.Clone(); // Original anchors are clipped to [.105,.895].
                var plus = new float[target.PixelCount * 3]; var minus = new float[plus.Length];
                row.gradient = new double[x.Length]; row.pixelJacobianColumnNorm = new double[x.Length];
                const double h = .01;
                for (int i = 0; i < x.Length; ++i)
                {
                    x[i] = row.initialTheta[i] + h;
                    double p = log.Measure(backend, "diagnostic_plus", x, () => { coords.Apply(x); return backend.Residual(0, target, plus); });
                    x[i] = row.initialTheta[i] - h;
                    double m = log.Measure(backend, "diagnostic_minus", x, () => { coords.Apply(x); return backend.Residual(0, target, minus); });
                    x[i] = row.initialTheta[i];
                    row.gradient[i] = (p - m) / (2 * h);
                    double norm = 0;
                    for (int k = 0; k < plus.Length; ++k) { double v = (plus[k] - minus[k]) / (2 * h); norm += v * v; }
                    row.pixelJacobianColumnNorm[i] = Math.Sqrt(norm);
                }
                NativeRenderBackend.Save(target, FitSize, Path.Combine(dir, "target"));
                NativeRenderBackend.Save(initial, FitSize, Path.Combine(dir, "initial"));
                row.cameraRenders = log.Cameras; row.wallSeconds = log.WallSeconds;
                WriteJson(dir, "report.json", row);
            }
        }

        [Serializable] private sealed class SpotSanity
        {
            public NativeLamp target;
            public double repeatedLoss, positionLoss, intensityLoss, directionLoss, noShadowsLoss, boundaryLoss, awayLoss, awayPowerLoss, awayEnergyRatio, wallSeconds;
            public int cameraRenders;
            public bool passed;
        }

        private static void RunSpot(string output, Camera camera, ViewInfo view, Light[] lights, BoxCollider block, string method)
        {
            var spot = lights[0]; var point = lights[1];
            var spotCoords = new RecoveryRigCoordinates(new[] { spot }, block, false);
            var pointCoords = new RecoveryRigCoordinates(new[] { point }, block, false);
            var pTruth = pointCoords.Encode(new[] { RecoveryLightState.Read(point) });
            var reference = Quaternion.LookRotation(new Vector3(-.12f, -.64f, 1.05f), Vector3.up);
            var aim = new SpotAimCoordinates(reference, 46);
            var uv = aim.Encode(Quaternion.LookRotation(new Vector3(.18f, -.70f, 1.06f), Vector3.up) * Vector3.forward);
            double[] truth = { .46, .65, .24, .65, uv.x, uv.y };
            spot.type = LightType.Spot;
            spot.range = 5;
            spot.spotAngle = 64;
            spot.innerSpotAngle = 34;
            void Apply(double[] theta)
            {
                spotCoords.Apply(theta);
                spot.transform.rotation = aim.Decode(theta[4], theta[5]);
            }
            void ApplyDirection(double[] theta) => Apply(new[] { truth[0], truth[1], truth[2], truth[3], theta[0], theta[1] });
            double[] Shift(int seed, bool directionOnly)
            {
                int i = Array.IndexOf(Seeds, seed);
                var starts = new[] { new[] { .20, .70 }, new[] { .84, .70 }, new[] { .27, .18 } };
                if (directionOnly) return starts[i];
                var positions = new[] { new[] { .33, .75, .33, .49 }, new[] { .63, .53, .18, .77 }, new[] { .32, .72, .42, .53 } };
                return positions[i].Concat(starts[i]).ToArray();
            }

            string sanityDir = Path.Combine(output, "e6_spot_sanity");
            using (var backend = Backend(camera, new[] { view }, FitSize))
            using (var log = new QueryLog(sanityDir, "queries.csv", "spot_sanity", 0))
            {
                Apply(truth);
                var report = new SpotSanity { target = NativeLamp.Read(spot) };
                NativeRenderBackend.Frame target = null;
                log.Measure(backend, "warmup", truth, () => { backend.Capture(0); return double.NaN; });
                log.Measure(backend, "target", truth, () => { target = backend.Capture(0); return double.NaN; });
                report.repeatedLoss = Probe("repeat", _ => { });
                report.positionLoss = Probe("position", t => t[0] += .12);
                report.intensityLoss = Probe("intensity", t => t[3] -= .14);
                report.directionLoss = Probe("direction", t => t[4] -= .14);
                report.boundaryLoss = Probe("boundary", t => { t[4] = 0; t[5] = 1; });
                Apply(truth);
                var shadow = spot.shadows;
                try
                {
                    spot.shadows = LightShadows.None;
                    report.noShadowsLoss = log.Measure(backend, "no_shadows", truth, () => backend.Loss(0, target));
                }
                finally { spot.shadows = shadow; }
                // The Cornell front is open. Aim straight out through it, rather than merely reversing
                // the target direction (which can still illuminate its ceiling or side walls).
                spot.transform.rotation = Quaternion.LookRotation(Vector3.back, Vector3.up);
                NativeRenderBackend.Frame away = null;
                report.awayLoss = log.Measure(backend, "away_from_room", null, () =>
                { away = backend.Capture(0); return NativeRenderBackend.Compare(away, target); });
                spot.intensity *= .5f;
                report.awayEnergyRatio = away.Energy / (target.Energy + 1e-12);
                report.awayPowerLoss = log.Measure(backend, "away_half_power", null, () =>
                    backend.Loss(0, away) * ((away.Energy + 1e-12) / (target.Energy + 1e-12)));
                Apply(truth);
                report.cameraRenders = log.Cameras; report.wallSeconds = log.WallSeconds;
                report.passed = report.repeatedLoss <= 1e-10 && new[] { report.positionLoss, report.intensityLoss, report.directionLoss,
                    report.noShadowsLoss, report.awayLoss, report.boundaryLoss }.All(j => j > 1e-6 && !double.IsInfinity(j));
                WriteJson(sanityDir, "report.json", report);
                if (!report.passed) throw new InvalidOperationException("Native Spot does not pass the position/intensity/aim/visibility checks.");
                double Probe(string name, Action<double[]> edit)
                {
                    var t = (double[])truth.Clone(); edit(t); Apply(t);
                    if (Math.Abs(spot.transform.forward.sqrMagnitude - 1) > 1e-5 || t.Any(x => !(x >= 0 && x <= 1)))
                        throw new InvalidOperationException("Invalid Spot aim or probe.");
                    return log.Measure(backend, name, t, () => backend.Loss(0, target));
                }
            }
            var seeds = Seeds.Take(3).ToArray();
            var one = new[] { spot };
            RunNextStage(output, "e6_spot_direction", "URP/Lit", camera, view, one, ApplyDirection, truth.Skip(4).ToArray(), () => Apply(truth),
                seeds, new[] { method }, new[] { "single" }, 1000, _ => "spot_smoke", s => Shift(s, true));
            var joint = RunNextStage(output, "e6_spot_joint", "URP/Lit", camera, view, one, Apply, truth, () => Apply(truth),
                seeds, new[] { method }, new[] { "single" }, 1000, _ => "spot_smoke", s => Shift(s, false));
            if (!joint.trials.All(t => t.passed)) return;
            point.enabled = true;
            var mixed = truth.Concat(pTruth).ToArray();
            void ApplyMixed(double[] t) { Apply(t); pointCoords.Apply(t.Skip(6).ToArray()); }
            RunNextStage(output, "e6_spot_point_mixed", "URP/Lit", camera, view, new[] { spot, point }, ApplyMixed, mixed, () => ApplyMixed(mixed),
                new[] { 17 }, new[] { method }, new[] { "single" }, 2000, _ => "mixed_smoke", s => Shift(s, false).Concat(new[] { .64, .53, .38, .63 }).ToArray());
        }
    }
}

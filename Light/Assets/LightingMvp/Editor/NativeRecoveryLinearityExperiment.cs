using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private const int StructureSchema = 1, LampDofs = 4, RigDofs = 12, PoolSize = 64;
        private const double PowerMin = .055, PowerMax = 9, Q0 = 1, NearSignal = 1e-6;
        private const float XzLimit = .91f, FloorLimit = .15f, CeilingLimit = 1.94f, Clearance = .045f, PointRange = 4f;
        private static readonly double LogPowerSpan = Math.Log(PowerMax / PowerMin);
        private static readonly double[] PowerGrid = { .35, 1, 3 };
        private static readonly int[] StructureSeeds = { 131, 157, 181 };

        // The domain is authored geometry, not GT. Work in Cartesian coordinates for LM;
        // test both oriented boxes in local space, expanded by world-axis clearance.
        private sealed class FreeCornell
        {
            private readonly (Matrix4x4 inverse, Vector3 center, Vector3 halfSize)[] boxes;
            internal FreeCornell(params BoxCollider[] blocks)
            {
                boxes = blocks.Select(b =>
                {
                    var s = b.transform.lossyScale;
                    return (b.transform.worldToLocalMatrix, b.center, .5f * b.size + new Vector3(
                        Clearance / Mathf.Abs(s.x), Clearance / Mathf.Abs(s.y), Clearance / Mathf.Abs(s.z)));
                }).ToArray();
            }

            internal static Vector3 Position(double[] theta, int offset = 0) => new Vector3(
                Mathf.Lerp(-XzLimit, XzLimit, (float)theta[offset]),
                Mathf.Lerp(FloorLimit, CeilingLimit, (float)theta[offset + 1]),
                Mathf.Lerp(-XzLimit, XzLimit, (float)theta[offset + 2]));

            internal bool Contains(double[] theta)
            {
                if (theta == null || theta.Length > RigDofs || theta.Length % LampDofs != 0) return false;
                foreach (double u in theta) if (!(u >= 0 && u <= 1)) return false;
                for (int i = 0; i < theta.Length; i += LampDofs)
                {
                    Vector3 p = Position(theta, i);
                    foreach (var box in boxes)
                    {
                        Vector3 q = box.inverse.MultiplyPoint3x4(p) - box.center;
                        if (Mathf.Abs(q.x) <= box.halfSize.x && Mathf.Abs(q.y) <= box.halfSize.y && Mathf.Abs(q.z) <= box.halfSize.z)
                            return false;
                    }
                }
                return true;
            }
        }

        private sealed class StructureRig
        {
            private readonly Light[] lights;
            internal readonly FreeCornell Domain;
            internal StructureRig(Light[] lights, FreeCornell domain)
            {
                this.lights = lights;
                Domain = domain;
                foreach (var light in lights)
                {
                    light.color = Color.white;
                    light.range = PointRange;
                }
            }

            internal void Apply(double[] theta)
            {
                if (!Domain.Contains(theta)) throw new InvalidOperationException("Attempted to submit an infeasible Cornell rig.");
                for (int i = 0; i < lights.Length; ++i)
                {
                    int offset = i * LampDofs;
                    var light = lights[i];
                    light.enabled = offset < theta.Length;
                    if (!light.enabled) continue;
                    light.transform.position = FreeCornell.Position(theta, offset);
                    light.intensity = (float)(PowerMin * Math.Exp(theta[offset + 3] * LogPowerSpan));
                }
            }

            internal NativeLamp[] Describe(double[] theta)
            {
                Apply(theta);
                return lights.Take(theta.Length / LampDofs).Select(NativeLamp.Read).ToArray();
            }
        }

        private static double[] Lamp(float x, float y, float z, double power) => new[] {
            ((double)x + XzLimit) / (2 * XzLimit), ((double)y - FloorLimit) / (CeilingLimit - FloorLimit),
            ((double)z + XzLimit) / (2 * XzLimit), PowerCoordinate(power) };
        private static double PowerCoordinate(double power) => Math.Log(power / PowerMin) / LogPowerSpan;
        private static double[] WithPower(double[] lamp, double power) => new[] { lamp[0], lamp[1], lamp[2], PowerCoordinate(power) };
        private static double[] Join(double[] rig, double[] lamp) => rig.Concat(lamp).ToArray();
        private static double[][] StructurePool(FreeCornell domain, int count, int seed) =>
            CandidateSet(count, LampDofs, seed, true, domain.Contains).Select(p => WithPower(p, Q0)).ToArray();

        private static float[] Difference(NativeRenderBackend.Frame a, NativeRenderBackend.Frame b)
        {
            var result = new float[a.PixelCount * 3];
            for (int i = 0, k = 0; i < a.PixelCount; ++i)
            {
                Color p = a.Pixel(i), q = b.Pixel(i);
                result[k++] = p.r - q.r; result[k++] = p.g - q.g; result[k++] = p.b - q.b;
            }
            return result;
        }

        private static double RgbDot(float[] a, float[] b)
        {
            double sum = 0;
            for (int i = 0; i < a.Length; ++i) sum += (double)a[i] * b[i];
            return sum;
        }

        private sealed class Response
        {
            internal readonly float[] Pixels;
            internal readonly double Energy;
            internal Response(NativeRenderBackend.Frame light, NativeRenderBackend.Frame empty)
            { Pixels = Difference(light, empty); Energy = RgbDot(Pixels, Pixels); }
        }

        private static NativeRenderBackend.Frame Capture(NativeRenderBackend backend, QueryLog log, StructureRig rig,
            double[] theta, string phase, int view = 0, NativeRenderBackend.Frame target = null)
        {
            NativeRenderBackend.Frame frame = null;
            log.Measure(backend, phase, theta, () =>
            {
                rig.Apply(theta);
                frame = backend.Capture(view);
                return target == null ? double.NaN : NativeRenderBackend.Compare(frame, target);
            });
            return frame;
        }

        private static void RunStructureExperiments(string task, string root, Camera camera, ViewInfo[] views, GameObject room,
            Light[] lights, BoxCollider tall)
        {
            var shortBlock = GameObject.Find("Cornell Short Block")?.GetComponent<BoxCollider>();
            if (!shortBlock || !shortBlock.enabled || !tall.enabled) throw new InvalidOperationException("Both Cornell box colliders are required.");
            var rig = new StructureRig(lights, new FreeCornell(tall, shortBlock));
            if (task == "e7") { RunLinearity(root, camera, views[0], room, rig); return; }
            SelfTestLm();
            SelfTestFeasibleLm();
            string e7Path = Path.Combine(root, "e7.json");
            if (!File.Exists(e7Path)) throw new InvalidOperationException("Run E7 in the same output directory before E8.");
            var e7 = JsonUtility.FromJson<LinearityExperiment>(File.ReadAllText(e7Path));
            if (e7.schema != StructureSchema || e7.resolution != FitSize || e7.renderer != SystemInfo.graphicsDeviceType.ToString() ||
                e7.camera.position != views[0].position || e7.camera.rotation != views[0].rotation || e7.camera.fov != views[0].fov)
                throw new InvalidOperationException("E7 capture protocol does not match E8.");
            bool olat = e7.materials.Single(m => m.shader == "Lit").supportsOlat;
            if (!int.TryParse(Arg("-bbE8Budget", "2000"), out int budget) || (budget != 2000 && budget != 4000))
                throw new ArgumentException("-bbE8Budget must be 2000 or 4000.");
            RunColdRigs(root, camera, views.Take(2).ToArray(), rig, olat, budget, task == "e8-smoke");
        }

        private static void SelfTestFeasibleLm()
        {
            // Two feasible half-planes meeting at a corner. The optimum is in the legal interior;
            // pilot and central differences at the start must not evaluate through either wall.
            static bool Legal(double[] x) => x[0] >= .5 && x[1] <= .5;
            double[] initial = { .5, .5 };
            int calls = 0, oneSided = 0;
            var result = ImageJacobianLm.Run(initial, 2, (x, phase, r) =>
            {
                if (!Legal(x)) throw new InvalidOperationException("Illegal synthetic LM query.");
                ++calls;
                if (phase == "jac_one_sided") ++oneSided;
                r[0] = (float)(x[0] - .73); r[1] = (float)(x[1] - .29);
                return (double)r[0] * r[0] + (double)r[1] * r[1];
            }, 100, 17, Legal);
            if (result.BestLoss > 1e-10 || calls != result.Queries || calls > 100 || oneSided == 0 || result.BlockedProbes == 0)
                throw new InvalidOperationException("Geometry-constrained LM self-test failed.");

            // The unconstrained Gauss–Newton direction crosses into a forbidden region.
            static bool HalfPlane(double[] x) => x[0] >= .55;
            var clipped = ImageJacobianLm.Run(new[] { .63 }, 1, (x, _, r) =>
            {
                if (!HalfPlane(x)) throw new InvalidOperationException("Illegal synthetic trial step.");
                r[0] = (float)(x[0] - .3);
                return (double)r[0] * r[0];
            }, 100, 17, HalfPlane);
            if (clipped.ShortenedTrials == 0 || clipped.Best[0] < .55 || clipped.Best[0] > .551)
                throw new InvalidOperationException("Geometry-constrained LM backtracking self-test failed.");
        }

        [Serializable] private sealed class LinearityExperiment
        {
            public int schema = StructureSchema, resolution = FitSize;
            public string unity = Application.unityVersion, renderer = SystemInfo.graphicsDeviceType.ToString(), gpu = SystemInfo.graphicsDeviceName;
            public ViewInfo camera;
            public LinearityMaterial[] materials;
        }

        [Serializable] private sealed class LinearityMaterial
        {
            public string shader;
            public bool supportsOlat;
            public int cameraRenders;
            public double wallSeconds, emptyRepeatLoss;
            public ErrorSummary intensity, additivity;
            public RankCheck[] rankings;
        }

        [Serializable] private sealed class ErrorSummary
        {
            public int samples, nearZero, nonzero;
            public double median, worst;
        }

        [Serializable] private sealed class RankCheck
        {
            public int contextLights, sampleCount, predictedTop, measuredTop;
            public bool measuredTopInPredictedTop3;
            public double spearman, top3Regret;
        }

        private static void RunLinearity(string root, Camera camera, ViewInfo view, GameObject room, StructureRig rig)
        {
            var lit = Measure("Lit");
            LinearityMaterial stylized;
            using (new StylizedMaterials(room)) stylized = Measure("Stylized");
            WriteJson(root, "e7.json", new LinearityExperiment { camera = view, materials = new[] { lit, stylized } });
            Debug.Log($"E7 complete: Lit OLAT={lit.supportsOlat}, Stylized OLAT={stylized.supportsOlat}; {root}");

            LinearityMaterial Measure(string material)
            {
                using (var backend = Backend(camera, new[] { view }, FitSize))
                using (var log = new QueryLog(root, $"e7_{material}_queries.csv", material, 0))
                using (var errors = new StreamWriter(Path.Combine(root, $"e7_{material}_error.csv")))
                using (var ranks = new StreamWriter(Path.Combine(root, $"e7_{material}_ranking.csv")))
                {
                    errors.WriteLine("test,group,variant,real_increment_l2,prediction_delta_l2,relative_l2_error,near_zero");
                    ranks.WriteLine("context_lights,candidate,predicted_gain,measured_gain,predicted_power,measured_power");
                    var emptyTheta = Array.Empty<double>();
                    Take(emptyTheta, "warmup");
                    var empty = Take(emptyTheta, "empty");
                    double repeat = NativeRenderBackend.Compare(Take(emptyTheta, "empty_repeat"), empty);
                    if (repeat > 1e-10) throw new InvalidOperationException("E7 camera is not repeatable.");
                    var strengths = new List<double>(); var sums = new List<double>();
                    var cases = new[] {
                        new[] { Lamp(-.62f, 1.52f, -.69f, Q0), Lamp(.67f, 1.42f, .64f, Q0), Lamp(.03f, 1.81f, -.11f, Q0) },
                        new[] { Lamp(-.64f, .44f, .80f, Q0), Lamp(.63f, .76f, -.48f, Q0), Lamp(-.70f, .52f, -.47f, Q0) }
                    };
                    for (int g = 0; g < cases.Length; ++g)
                    {
                        var set = cases[g];
                        var bases = set.Select((l, i) => new Response(Take(l, $"olat_{g}_{i}"), empty)).ToArray();
                        for (int i = 0; i < set.Length; ++i)
                            foreach (double w in new[] { .25, .5, 1, 2 })
                                Record(strengths, "intensity", $"{g}:{i}", F(w),
                                    Difference(Take(WithPower(set[i], w * Q0), $"intensity_{g}_{i}_{F(w)}"), empty),
                                    new[] { bases[i].Pixels }, w);
                        foreach (var ids in new[] { new[] { 0, 1 }, new[] { 0, 2 }, new[] { 1, 2 }, new[] { 0, 1, 2 } })
                        {
                            string name = string.Join("+", ids);
                            Record(sums, "additivity", g.ToString(Invariant), name,
                                Difference(Take(ids.SelectMany(i => set[i]).ToArray(), $"sum_{g}_{name}"), empty),
                                ids.Select(i => bases[i].Pixels).ToArray(), 1);
                        }
                    }

                    // Independent holdout lighting conditions for ranking, not the E8 targets.
                    var rankPool = StructurePool(rig.Domain, 24, 9409);
                    var dictionary = rankPool.Select((p, i) => new Response(Take(p, $"rank_olat_{i}"), empty)).ToArray();
                    var checks = new List<RankCheck>();
                    var scanLamps = new[] { Lamp(-.72f, .62f, .72f, .8), Lamp(.71f, 1.10f, -.32f, 1.4) };
                    var wanted = new[] { Lamp(.67f, 1.69f, .59f, 1.37), Lamp(-.75f, .74f, -.21f, 1.72), Lamp(.18f, 1.15f, -.39f, .48) };
                    for (int n = 0; n <= 2; ++n)
                    {
                        var context = scanLamps.Take(n).SelectMany(p => p).ToArray();
                        var target = Take(wanted.Take(n + 1).SelectMany(p => p).ToArray(), $"rank_target_{n}");
                        var current = Take(context, $"rank_context_{n}");
                        var residual = Difference(target, current);
                        double currentLoss = NativeRenderBackend.Compare(current, target);
                        var predicted = new double[rankPool.Length]; var actual = new double[rankPool.Length];
                        for (int i = 0; i < rankPool.Length; ++i)
                        {
                            var a = dictionary[i];
                            double dot = RgbDot(a.Pixels, residual);
                            // Same intensity grid for the predicted and rendered ranking, isolating response model error.
                            var predict = PowerGrid.Select(p => (p, gain: PredictedGain(dot, a.Energy, p))).OrderByDescending(t => t.gain).First();
                            var measured = PowerGrid.Select(p => (p, gain: currentLoss - NativeRenderBackend.Compare(
                                Take(Join(context, WithPower(rankPool[i], p)), $"rank_{n}_{i}_{F(p)}"), target)))
                                .OrderByDescending(t => t.gain).First();
                            predicted[i] = predict.gain / (target.Energy + 1e-12);
                            actual[i] = measured.gain;
                            ranks.WriteLine($"{n},{i},{F(predicted[i])},{F(actual[i])},{F(predict.p)},{F(measured.p)}");
                        }
                        int[] order = Order(predicted), realOrder = Order(actual);
                        double top = actual[realOrder[0]];
                        checks.Add(new RankCheck { contextLights = n, sampleCount = rankPool.Length,
                            predictedTop = order[0], measuredTop = realOrder[0], measuredTopInPredictedTop3 = order.Take(3).Contains(realOrder[0]),
                            spearman = Spearman(predicted, actual),
                            top3Regret = (top - order.Take(3).Max(i => actual[i])) / Math.Max(1e-12, Math.Abs(top)) });
                    }
                    var si = Summarize(strengths); var sa = Summarize(sums);
                    bool usable = Passes(si) && Passes(sa) && checks.All(c => c.spearman >= .90 && c.top3Regret <= .01);
                    if (log.Cameras != backend.RenderCount) throw new InvalidOperationException("E7 query ledger is incomplete.");
                    return new LinearityMaterial { shader = material, cameraRenders = log.Cameras, wallSeconds = log.WallSeconds,
                        emptyRepeatLoss = repeat, intensity = si, additivity = sa, rankings = checks.ToArray(), supportsOlat = usable };

                    NativeRenderBackend.Frame Take(double[] p, string phase) => Capture(backend, log, rig, p, phase);
                    void Record(List<double> items, string test, string group, string variant, float[] actual, float[][] basis, double w)
                    {
                        double signal = RgbDot(actual, actual), error = 0;
                        for (int i = 0; i < actual.Length; ++i)
                        {
                            double prediction = 0;
                            foreach (var a in basis) prediction += a[i];
                            double d = actual[i] - w * prediction;
                            error += d * d;
                        }
                        double norm = Math.Sqrt(signal), err = Math.Sqrt(error);
                        bool near = norm <= NearSignal;
                        items.Add(near ? double.NaN : err / norm);
                        errors.WriteLine($"{test},{group},{variant},{F(norm)},{F(err)},{(near ? "" : F(err / norm))},{(near ? 1 : 0)}");
                    }
                }
            }

            ErrorSummary Summarize(List<double> values)
            {
                var finite = values.Where(x => !double.IsNaN(x)).ToArray();
                return new ErrorSummary { samples = values.Count, nearZero = values.Count - finite.Length, nonzero = finite.Length,
                    median = finite.Length == 0 ? -1 : Median(finite), worst = finite.Length == 0 ? -1 : finite.Max() };
            }
            bool Passes(ErrorSummary s) => s.nonzero > 0 && s.median <= .02 && s.worst <= .05;
        }

        private static double PredictedGain(double dot, double energy, double power) => 2 * (power / Q0) * dot - Math.Pow(power / Q0, 2) * energy;
        private static int[] Order(double[] values) => Enumerable.Range(0, values.Length).OrderByDescending(i => values[i]).ToArray();

        private static double Spearman(double[] p, double[] q)
        {
            var a = Ranks(p); var b = Ranks(q);
            double mean = (a.Length - 1) / 2.0, dot = 0, aa = 0, bb = 0;
            for (int i = 0; i < a.Length; ++i)
            {
                double x = a[i] - mean, y = b[i] - mean;
                dot += x * y; aa += x * x; bb += y * y;
            }
            return aa > 0 && bb > 0 ? dot / Math.Sqrt(aa * bb) : 0;

            double[] Ranks(double[] values)
            {
                var sorted = Order(values); var ranks = new double[values.Length];
                for (int i = 0, j; i < sorted.Length; i = j)
                {
                    j = i + 1;
                    while (j < sorted.Length && values[sorted[j]] == values[sorted[i]]) ++j;
                    for (int k = i; k < j; ++k) ranks[sorted[k]] = (i + j - 1) / 2.0;
                }
                return ranks;
            }
        }
    }
}

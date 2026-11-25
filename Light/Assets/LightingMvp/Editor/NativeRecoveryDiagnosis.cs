using System;
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
        private static readonly int[] DiagnosisViews = { 0, 1, 2 };
        private const double ContributionFloor = 1e-10, DiagnosisEpsilon = 1e-12;
        private const int ShortDiagnosisBudget = 1000, FullDiagnosisBudget = 6000;

        [Serializable] private sealed class DiagnosisGate
        {
            public string baseline = "96002e457fdc65d0373d761858a4ad307fee5149";
            public bool passed;
            public int negativeTargetChannels, sharedRenders, easyLamp = -1, hardLamp = -1, singleLampPasses;
            public double logScale, emptyLoss, maxScalarVectorGap;
            public double[] replayTrain, replayHigh, repeatTrain, repeatHigh, gtTheta, logTargetEnergy, sqrtTargetEnergy;
        }
        [Serializable] private sealed class DiagnosisTrial
        {
            public string key, stage, profile, method, mapping, stop;
            public int seed, lamp, budget, dimensions, trainingRenders, recaptures, bestQuery, bestRawQuery, firstRawPass = -1;
            public int jacobians, accepted, rejected, blockedProbes, shortenedTrials, contributionViews;
            public bool passed, initialAlreadyPasses, contributionPass;
            public double initialRaw, initialObjective, finalRaw, finalObjective, bestRaw, recapturedRaw, reduction, heldNrmse, highNrmse;
            public double initialContributionNrmse, finalContributionNrmse, predictedNrmse;
            public double elapsed, trainingSeconds, renderSeconds, readbackSeconds, residualSeconds, jacobianSeconds, normalSeconds, linearSolveSeconds, diagnosticSeconds;
            public long processPeakWorkingSet, lmBufferBytes;
            public double[] start, best, trainPerView, highPerView, displayPerView, displayHighPerView, initialContributionPerView, finalContributionPerView;
            public double[] positionErrorMetres, intensityError, intensityRelativeError;
            public Brightness[] initialBrightness, finalBrightness;
        }
        [Serializable] private sealed class DiagnosisStage
        {
            public string stage;
            public bool smoke;
            public int sharedRenders, trainingRenders, recaptures;
            public DiagnosisTrial[] trials;
        }
        [Serializable] private sealed class Brightness
        {
            public double peak, p50, p90, p99, p999, errorEnergy, top01, top1, top5;
            public int negativeChannels, maxErrorPixel;
        }
        [Serializable] private sealed class Spectrum
        {
            public int rank;
            public double cutoff, condition;
            public double[] eigen, singular, columnNorms;
        }

        private static void RunClassroomDiagnosis(string task, string output)
        {
            if (!Application.isBatchMode) throw new InvalidOperationException("Classroom diagnosis opens a transient scene and is batch-only.");
            string source = Arg("-bbScaleSource", "");
            if (string.IsNullOrWhiteSpace(source)) throw new ArgumentException("-bbScaleSource must contain the saved e10 and e10_near runs.");
            bool smoke = task.EndsWith("-smoke", StringComparison.Ordinal);
            var name = task.Replace('-', '_');
            string dir = Path.Combine(output, name);
            Directory.CreateDirectory(dir);
            int seed = int.Parse(Arg("-bbDiagnosisSeed", "131"), Invariant);
            if (!StructureSeeds.Contains(seed)) throw new ArgumentException("The diagnosis seed must be 131, 157 or 181.");
            using var pipeline = new TestPipeline(classroom: true);
            using var fixture = new ClassroomFixture(pipeline.Asset);
            using var train = ClassroomBackend(fixture, false);
            using var high = ClassroomBackend(fixture, true);
            using var session = new ClassroomDiagnosis(output, dir, source, task, fixture, train, high);
            session.Run(seed, smoke);
        }

        private sealed partial class ClassroomDiagnosis : IDisposable
        {
            private readonly string root, dir, task;
            private readonly ClassroomFixture fixture;
            private readonly NativeRenderBackend train, high;
            private readonly QueryLog shared;
            private readonly DiagnosisGate gate = new DiagnosisGate();
            private readonly Frame[] target = new Frame[5], targetHi = new Frame[5];
            private readonly Frame[] empty = new Frame[3];
            private readonly double[][] contribution = new double[7][];
            private readonly Dictionary<Mapping, NativeRenderBackend.MappedTarget[]> maps = new Dictionary<Mapping, NativeRenderBackend.MappedTarget[]>();
            private readonly Dictionary<string, ScaleRun> baseline = new Dictionary<string, ScaleRun>();
            private readonly List<DiagnosisTrial> rows = new List<DiagnosisTrial>();
            private readonly double[] truth;
            private readonly RoomConfig saved;

            internal ClassroomDiagnosis(string root, string dir, string source, string task, ClassroomFixture fixture, NativeRenderBackend train, NativeRenderBackend high)
            {
                this.root = root; this.dir = dir; this.task = task; this.fixture = fixture; this.train = train; this.high = high;
                saved = Load<RoomConfig>(Path.Combine(source, "e10", "scene.json"));
                var nearConfig = Load<RoomConfig>(Path.Combine(source, "e10_near", "scene.json"));
                if (JsonUtility.ToJson(saved) != JsonUtility.ToJson(nearConfig)) throw new InvalidDataException("The two E10 scenes differ.");
                saved.gpu = fixture.Config.gpu; saved.graphics = fixture.Config.graphics;
                if (JsonUtility.ToJson(saved) != JsonUtility.ToJson(fixture.Config)) throw new InvalidDataException("Classroom geometry, rendering, cameras or authored GT differ from E10.");
                truth = saved.gt.SelectMany(l => ClassroomFixture.Encode(l.position, l.intensity)).ToArray();
                if (saved.lamps != 7 || !fixture.Rig.Contains(truth)) throw new InvalidDataException("Invalid saved seven-light rig.");
                foreach (string x in new[] { "e10", "e10_near" }) baseline[x] = Load<ScaleRun>(Path.Combine(source, x, "stage.json"));
                foreach (string x in baseline.Keys) foreach (int s in StructureSeeds)
                    if (Prior(x, s).initialTheta.Length != truth.Length || !fixture.Rig.Contains(Prior(x, s).initialTheta))
                        throw new InvalidDataException("Missing or illegal original E10 initialization.");
                shared = new QueryLog(dir, "shared.csv", "diagnosis_shared", 0);
            }

            private ScaleTrial Prior(string profile, int seed) => baseline[profile].trials.Single(r => r.method == "known" && r.views == 3 && r.seed == seed);
            private static T Load<T>(string path) => JsonUtility.FromJson<T>(File.ReadAllText(path));
            private static double Rms(IEnumerable<double> squared) => Math.Sqrt(squared.Average());
            private static double Power(double coordinate) => PowerMin * Math.Exp(coordinate * LogPowerSpan);
            private static int[] LampAxes(int lamp, bool strength) => Enumerable.Range(lamp * LampDofs, strength ? LampDofs : 3).ToArray();
            private static int[] AllAxes() => Enumerable.Range(0, 7 * LampDofs).ToArray();
            private double[] Mix(string profile, int seed, int[] axes)
            {
                var start = (double[])truth.Clone(); var p = Prior(profile, seed).initialTheta;
                foreach (int i in axes) start[i] = p[i];
                return start;
            }
            private Frame Capture(NativeRenderBackend backend, QueryLog log, double[] x, int v, string phase, int omit = -1)
            {
                Frame f = null;
                log.Measure(backend, phase, x, () => {
                    fixture.Rig.Apply(x); if (omit >= 0) fixture.Omit(omit);
                    f = backend.Capture(v); return double.NaN;
                });
                return f;
            }

            internal void Run(int seed, bool smoke)
            {
                WriteJson(dir, "scene.json", fixture.Config);
                RunGate();
                if (!gate.passed) throw new InvalidOperationException("E13a GT replay/residual consistency failed; see replay.json.");
                if (task == "e13" || task == "e13-smoke") RunOracle(seed, smoke);
                else
                {
                    var prior = Load<DiagnosisGate>(Path.Combine(root, "e13", "oracle.json"));
                    if (!prior.passed || prior.easyLamp < 0 || prior.hardLamp < 0 || !prior.gtTheta.SequenceEqual(truth))
                        throw new InvalidDataException("Full E13 must run before subsequent diagnosis stages.");
                    if (task == "e13-position") RunPositionOracle(seed, smoke);
                    else RunComparisons(prior, seed, smoke);
                }
                var stage = new DiagnosisStage { stage = task, smoke = smoke, sharedRenders = shared.Cameras,
                    trainingRenders = rows.Sum(r => r.trainingRenders), recaptures = rows.Sum(r => r.recaptures), trials = rows.ToArray() };
                if (train.RenderCount + high.RenderCount != stage.sharedRenders + stage.trainingRenders + stage.recaptures)
                    throw new InvalidOperationException("Classroom diagnosis camera counters do not match the ledgers.");
                WriteJson(dir, "stage.json", stage);
            }

            private void RunGate()
            {
                gate.gtTheta = truth;
                gate.replayTrain = new double[5]; gate.replayHigh = new double[5]; gate.repeatTrain = new double[5]; gate.repeatHigh = new double[5];
                for (int v = 0; v < 5; ++v)
                {
                    Capture(train, shared, fixture.Truth, v, "warmup");
                    target[v] = Capture(train, shared, fixture.Truth, v, "target_train");
                    targetHi[v] = Capture(high, shared, fixture.Truth, v, "target_check");
                    fixture.Rig.Apply(truth); fixture.VerifyActiveRig(saved.gt);
                    gate.replayTrain[v] = Math.Sqrt(NativeRenderBackend.Compare(Capture(train, shared, truth, v, "replay_train"), target[v]));
                    gate.replayHigh[v] = Math.Sqrt(NativeRenderBackend.Compare(Capture(high, shared, truth, v, "replay_check"), targetHi[v]));
                    gate.repeatTrain[v] = Math.Sqrt(NativeRenderBackend.Compare(Capture(train, shared, truth, v, "repeat_train"), target[v]));
                    gate.repeatHigh[v] = Math.Sqrt(NativeRenderBackend.Compare(Capture(high, shared, truth, v, "repeat_check"), targetHi[v]));
                }
                var positive = new List<float>(3 * 3 * train.Width * train.Height);
                foreach (var f in target.Take(3)) for (int p = 0; p < f.PixelCount; ++p)
                {
                    Color c = f.Pixel(p); Add(c.r); Add(c.g); Add(c.b);
                }
                void Add(float x) { if (x > 0) positive.Add(x); else if (x < 0) ++gate.negativeTargetChannels; }
                positive.Sort();
                if (positive.Count == 0) throw new InvalidDataException("There are no positive training target channels.");
                gate.logScale = Math.Max(1e-6, positive[(int)Math.Floor(.9 * (positive.Count - 1))]);
                foreach (var mapping in new[] { Mapping.SignedLog, Mapping.SignedSqrt })
                    maps[mapping] = target.Take(3).Select(f => new NativeRenderBackend.MappedTarget(f, mapping, gate.logScale)).ToArray();
                gate.logTargetEnergy = maps[Mapping.SignedLog].Select(f => f.Energy).ToArray();
                gate.sqrtTargetEnergy = maps[Mapping.SignedSqrt].Select(f => f.Energy).ToArray();
                var x0 = Prior("e10_near", 131).initialTheta;
                foreach (int views in new[] { 1, 3 })
                {
                    var residual = new float[views * train.Width * train.Height * 3];
                    double scalar = 0, raw = 0;
                    for (int v = 0; v < views; ++v)
                    {
                        var f = Capture(train, shared, x0, v, "raw_scalar_check");
                        raw += NativeRenderBackend.Compare(f, target[v]) / views;
                        int n = v;
                        scalar += shared.Measure(train, "raw_vector_check", x0, () => train.Residual(n, target[n], residual, n * f.PixelCount * 3, views));
                    }
                    CheckNorm(scalar, raw, residual);
                }
                foreach (var mapping in maps.Keys)
                {
                    var residual = new float[3 * train.Width * train.Height * 3];
                    double scalar = 0, expected = 0;
                    foreach (int v in DiagnosisViews)
                    {
                        var f = Capture(train, shared, x0, v, "mapped_scalar_check");
                        var t = maps[mapping][v];
                        double sq = 0;
                        for (int i = 0; i < f.PixelCount; ++i)
                        {
                            Color a = f.Pixel(i), b = target[v].Pixel(i);
                            sq += Sq(NativeRenderBackend.Map(a.r, mapping, gate.logScale) - NativeRenderBackend.Map(b.r, mapping, gate.logScale));
                            sq += Sq(NativeRenderBackend.Map(a.g, mapping, gate.logScale) - NativeRenderBackend.Map(b.g, mapping, gate.logScale));
                            sq += Sq(NativeRenderBackend.Map(a.b, mapping, gate.logScale) - NativeRenderBackend.Map(b.b, mapping, gate.logScale));
                        }
                        expected += sq / (3 * (t.Energy + DiagnosisEpsilon));
                        scalar += shared.Measure(train, "mapped_vector_check", x0, () => train.MappedResidual(v, t, residual, v * f.PixelCount * 3, 3, out _));
                    }
                    CheckNorm(scalar, expected, residual);
                }
                for (int v = 0; v < 3; ++v) empty[v] = Capture(train, shared, NoLights, v, "empty");
                gate.emptyLoss = DiagnosisViews.Average(v => NativeRenderBackend.Compare(empty[v], target[v]));
                for (int l = 0; l < fixture.Rig.Count; ++l)
                {
                    contribution[l] = DiagnosisViews.Select(v => NativeRenderBackend.Compare(Capture(train, shared, truth, v, "gt_leave_out_" + l, l), target[v])).ToArray();
                }
                gate.sharedRenders = shared.Cameras;
                gate.passed = gate.replayTrain.Concat(gate.replayHigh).Concat(gate.repeatTrain).Concat(gate.repeatHigh).Max() <= 1e-5 && gate.maxScalarVectorGap <= 1e-6;
                WriteJson(dir, "replay.json", gate);

                void CheckNorm(double scalar, double independent, float[] r)
                {
                    double norm = RgbDot(r, r);
                    double gap = Math.Max(Math.Abs(norm - scalar), Math.Abs(independent - scalar)) / Math.Max(1, scalar);
                    gate.maxScalarVectorGap = Math.Max(gate.maxScalarVectorGap, gap);
                }
            }

            public void Dispose() => shared.Dispose();
            private static double Sq(double x) => x * x;
            private static string JoinFields(IEnumerable<double> x) => string.Join(";", x.Select(F));
            private static Spectrum GetSpectrum(double[,] gram, int n)
            {
                var h = (double[,])gram.Clone();
                var norms = Enumerable.Range(0, n).Select(i => Math.Sqrt(Math.Max(0, h[i, i]))).ToArray();
                double max = norms.Max();
                double tolerance = Math.Max(1e-28, max * max * 1e-13);
                for (int sweep = 0; sweep < 40; ++sweep)
                {
                    double off = 0;
                    for (int p = 0; p < n; ++p) for (int q = p + 1; q < n; ++q)
                    {
                        double z = h[p, q]; off = Math.Max(off, Math.Abs(z));
                        if (Math.Abs(z) <= tolerance) continue;
                        double tau = (h[q, q] - h[p, p]) / (2 * z);
                        double t = (tau < 0 ? -1 : 1) / (Math.Abs(tau) + Math.Sqrt(1 + tau * tau));
                        double c = 1 / Math.Sqrt(1 + t * t), s = c * t;
                        for (int k = 0; k < n; ++k) if (k != p && k != q)
                        {
                            double x = h[k, p], y = h[k, q];
                            h[k, p] = h[p, k] = c * x - s * y;
                            h[k, q] = h[q, k] = s * x + c * y;
                        }
                        h[p, p] -= t * z; h[q, q] += t * z; h[q, p] = h[p, q] = 0;
                    }
                    if (off <= tolerance) break;
                }
                var eigen = Enumerable.Range(0, n).Select(i => Math.Max(0, h[i, i])).OrderByDescending(v => v).ToArray();
                double cutoff = Math.Max(1e-24, eigen[0] * 1e-10);
                int rank = eigen.Count(v => v > cutoff);
                return new Spectrum { cutoff = cutoff, columnNorms = norms, eigen = eigen, singular = eigen.Select(Math.Sqrt).ToArray(), rank = rank,
                    condition = rank == n ? Math.Sqrt(eigen[0] / eigen[n - 1]) : -1 };
            }
        }
    }
}

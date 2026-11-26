using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using Debug = UnityEngine.Debug;
using Frame = LightingMvp.NativeRenderBackend.Frame;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private const string E20BaseCommit = "60285c50866b27160efa2b9b46808d39bdfb9f82";

        private static void RunIndependentLightStudy(string task, string output)
        {
            if (!Application.isBatchMode) throw new InvalidOperationException("E20 opens the E19 scene and is batch only.");
            if (!AssetDatabase.LoadAssetAtPath<SceneAsset>(NativeComplexRoomBuilder.ScenePath))
                throw new FileNotFoundException("Build the E19 complex room with NativeComplexRoomBuilder.BuildMenu first.");
            int budget = int.Parse(Arg("-bbE20Budget", task.EndsWith("-smoke", StringComparison.Ordinal) ? "160" : "2000"), Invariant);
            if (budget < 100 || budget > 2000) throw new ArgumentException("E20 uses at most the E19 2000-query budget.");
            string defaultSeeds = task.EndsWith("-smoke", StringComparison.Ordinal) ? "733" : task == "e20-a" ? "131,733,157,181" : "131,733";
            int[] seeds = Arg("-bbE20Seeds", defaultSeeds)
                .Split(',').Select(s => int.Parse(s, Invariant)).Distinct().ToArray();
            string dir = Path.Combine(output, task.Replace('-', '_'));
            Directory.CreateDirectory(dir);
            using var pipeline = new TestPipeline(classroom: true);
            using var scene = new ComplexRoomSensitivityScene();
            using var experiment = new E20Experiment(task, dir, scene, seeds, budget);
            experiment.Run();
        }

        [Serializable] private sealed class E20Start
        {
            public string profile;
            public int seed;
            public bool nearFallback;
            public double[] theta, positionCm, intensityRelative;
        }

        [Serializable] private sealed class E20Configuration
        {
            public string branch, baseCommit, sceneSha256, stage, selectionRule;
            public E17Config scene;
            public E20Start[] starts;
            public int[] train, held, seeds;
            public int queryBudget, nativeMsaa, nativeShadowAtlas, nativeMediumShadowTile;
            public double[] truth, fdPositionMetres, fdLogIntensity;
            public double rankRelativeSigma, positionPassCm, powerPassRelative;
        }

        [Serializable] private sealed class E20Arm
        {
            public string id, method, condition, profile, stop;
            public int seed, budget, queries, nativeBudget, proxyBudget, nativeQueries, cameraRenders, validationRenders, proxyQueries, shadowFaces, accepted, rejected, blocked, shortened, jacobians, nonfinite;
            public int[] lamps;
            public bool passed;
            public double initial, best, recheck, nativeInitial, nativeBest, trainHighNrmse, heldNrmse, proxyHeldNrmse, seconds;
            public double[] start, theta, positionCm, intensityRelative, perViewNrmse;
        }

        [Serializable] private sealed class E20Summary
        {
            public string stage;
            public int cameraRenders, optimizationCameraRenders, validationCameraRenders, sharedAndDiagnosticRenders;
            public double seconds;
            public E20Arm[] arms;
        }

        private sealed class E20Target
        {
            public readonly string Name;
            public readonly int Lamp;
            public readonly bool Shadows;
            public readonly Frame[] Fit, High;

            public E20Target(string name, int lamp, bool shadows, Frame[] fit, Frame[] high)
            { Name = name; Lamp = lamp; Shadows = shadows; Fit = fit; High = high; }
        }

        private sealed partial class E20Experiment : IDisposable
        {
            private const int FitViews = E19FitViews, ViewChannels = E19FitW * E19FitH * 3;
            private const double PositionPass = 5, IntensityPass = .05;
            private static readonly string[] AxisNames = { "x", "y", "z", "log_intensity" };
            private static readonly double[] PositionSteps = { .005, .02, .05 }, IntensitySteps = { .02, .05, .10 };
            private readonly string task, dir;
            private readonly ComplexRoomSensitivityScene scene;
            private readonly NativeRenderBackend fit, high;
            private readonly Light[] lights;
            private readonly Color ambient;
            private readonly float ambientIntensity, reflectionIntensity;
            private readonly E20Start[] starts;
            private readonly int budget;
            private readonly StreamWriter queries, spectra, stages;
            private readonly Dictionary<string, int> counters = new Dictionary<string, int>();
            private readonly List<E20Arm> arms = new List<E20Arm>();
            private readonly double[] units;
            private readonly bool smoke;
            private int rowId;

            internal E20Experiment(string task, string dir, ComplexRoomSensitivityScene scene, int[] seeds, int budget)
            {
                this.task = task; this.dir = dir; this.scene = scene; this.budget = budget;
                smoke = task.EndsWith("-smoke", StringComparison.Ordinal);
                lights = scene.Lamps.Select(l => GameObject.Find(scene.UnityName(l)).GetComponent<Light>()).ToArray();
                ambient = RenderSettings.ambientLight;
                ambientIntensity = RenderSettings.ambientIntensity;
                reflectionIntensity = RenderSettings.reflectionIntensity;
                units = new[] { (double)scene.Extents.x, scene.Extents.y, scene.Extents.z, LogPowerSpan };
                starts = seeds.SelectMany(SeedStarts).ToArray();
                fit = new NativeRenderBackend(scene.Camera, scene.Views, scene.Rotations, E19FitW, E19FitH, -1);
                high = new NativeRenderBackend(scene.Camera, scene.Views, scene.Rotations, E19HighW, E19HighH, -1);
                queries = OpenCsv("queries.csv", "row,arm,kind,query,phase,active_lamps,theta,native_rgb_loss,direct_proxy_loss,msm_proxy_loss,free_space_penalty,step,proxy_predicted_change,native_actual_change,accepted,native_camera_renders,shadow_faces,elapsed_s");
                spectra = OpenCsv("jacobians.csv", "arm,phase,seed,profile,batch,query,view,position_step_m,log_intensity_step,physical_steps,stencils,rank,eigen_cutoff,condition,singular_values,column_norms,lamp_response_norms");
                stages = OpenCsv("arms.csv", "arm,method,condition,seed,profile,lamps,queries,budget,native_budget,proxy_budget,native_queries,camera_renders,validation_renders,proxy_queries,shadow_faces,initial,best,recheck,native_initial,native_best,train_high_nrmse,held_nrmse,proxy_held_nrmse,position_cm,intensity_relative,accepted,rejected,blocked,shortened,jacobians,nonfinite,seconds,pass,stop");
                using var sha = SHA256.Create();
                using var sceneFile = File.OpenRead(NativeComplexRoomBuilder.ScenePath);
                var config = GraphicsSettings.currentRenderPipeline as UnityEngine.Rendering.Universal.UniversalRenderPipelineAsset;
                WriteJson(dir, "config.json", new E20Configuration {
                    branch = Arg("-bbE20Branch", "blackbox"), baseCommit = Arg("-bbE20BaseCommit", E20BaseCommit),
                    sceneSha256 = BitConverter.ToString(sha.ComputeHash(sceneFile)).Replace("-", "").ToLowerInvariant(),
                    stage = task, scene = scene.Describe(), starts = starts, seeds = seeds, truth = scene.Truth,
                    train = Enumerable.Range(0, FitViews).ToArray(), held = Enumerable.Range(FitViews, scene.Views.Length - FitViews).ToArray(),
                    queryBudget = budget, nativeMsaa = config.msaaSampleCount, nativeShadowAtlas = config.additionalLightsShadowmapResolution,
                    nativeMediumShadowTile = config.additionalLightsShadowResolutionTierMedium, fdPositionMetres = PositionSteps,
                    fdLogIntensity = IntensitySteps, rankRelativeSigma = 1e-5, positionPassCm = PositionPass, powerPassRelative = IntensityPass,
                    selectionRule = "Prefer pairs with a failed joint Near lamp but both isolated fits passing; order by maximum absolute cross-lamp correlation in the 2 cm Jacobian. Compare the weakest pair of the same class." });
            }

            internal void Run()
            {
                var timer = Stopwatch.StartNew();
                if (task.StartsWith("e20-direct", StringComparison.Ordinal)) RunDirectProxy();
                else if (task.StartsWith("e20-msm", StringComparison.Ordinal)) RunMomentStudy();
                else RunDiagnosis();
                int measured = fit.RenderCount + high.RenderCount;
                int optimize = arms.Sum(a => a.cameraRenders), validate = arms.Sum(a => a.validationRenders);
                if (counters.Values.Sum() != measured || optimize != Counted("native_optimization") || validate != Counted("validation"))
                    throw new InvalidOperationException("E20 camera ledgers disagree.");
                WriteJson(dir, "summary.json", new E20Summary { stage = task, arms = arms.ToArray(), seconds = timer.Elapsed.TotalSeconds,
                    cameraRenders = measured, optimizationCameraRenders = optimize, validationCameraRenders = validate,
                    sharedAndDiagnosticRenders = measured - optimize - validate });
                using var csv = OpenCsv("capture-counts.csv", "purpose,camera_renders");
                foreach (var row in counters) csv.WriteLine(row.Key + "," + row.Value.ToString(Invariant));
                Debug.Log($"[E20] {task} completed {arms.Count} arms, {measured} native camera renders, {timer.Elapsed.TotalSeconds:F1}s");
            }

            private IEnumerable<E20Start> SeedStarts(int seed)
            {
                var far = scene.Start(seed);
                foreach (bool near in new[] { true, false })
                {
                    var x = E19Start(scene, far, near ? .45f : 1f);
                    yield return new E20Start { seed = seed, profile = near ? "near" : "far", nearFallback = near && ReferenceEquals(x, far),
                        theta = x, positionCm = scene.Lamps.Select(l => PositionCm(x, l)).ToArray(), intensityRelative = scene.Lamps.Select(l => PowerError(x, l)).ToArray() };
                }
            }

            private double PositionCm(double[] theta, int lamp) => 100 * Vector3.Distance(scene.PositionOf(theta, lamp), scene.PositionOf(scene.Truth, lamp));
            private double PowerError(double[] theta, int lamp) => Math.Abs(scene.PowerOf(theta, lamp) / scene.PowerOf(scene.Truth, lamp) - 1);
            private bool Pass(double[] theta, int lamp) => PositionCm(theta, lamp) <= PositionPass && PowerError(theta, lamp) <= IntensityPass;
            private int Counted(string purpose) => counters.TryGetValue(purpose, out int n) ? n : 0;
            private void Count(string purpose, int n) => counters[purpose] = Counted(purpose) + n;
            private static string Fields(IEnumerable<double> values) => string.Join(";", values.Select(Value));
            private static string Ids(IEnumerable<int> ids) => string.Join(";", ids.Select(i => (i + 1).ToString(Invariant)));
            private static string Value(double value) => double.IsNaN(value) || double.IsInfinity(value) ? "" : F(value);
            private static double Cosine(double dot, double aa, double bb) => aa > 0 && bb > 0 ? Math.Max(-1, Math.Min(1, dot / Math.Sqrt(aa * bb))) : double.NaN;
            private static double Sq20(double x) => x * x;

            private StreamWriter OpenCsv(string file, string fields)
            {
                var csv = new StreamWriter(Path.Combine(dir, file));
                csv.WriteLine(fields);
                return csv;
            }

            private void SetMode(bool shadows, int lamp)
            {
                for (int l = 0; l < lights.Length; l++)
                {
                    lights[l].enabled = lamp < 0 || lamp == l;
                    scene.SetShadows(l, shadows);
                }
                RenderSettings.ambientLight = lamp < 0 ? ambient : Color.black;
                RenderSettings.ambientIntensity = lamp < 0 ? ambientIntensity : 0;
                RenderSettings.reflectionIntensity = lamp < 0 ? reflectionIntensity : 0;
            }

            private E20Target Target(int lamp, bool shadows)
            {
                string name = (lamp < 0 ? "full" : $"oracle_L{lamp + 1}") + (shadows ? "_on" : "_off");
                SetMode(shadows, lamp);
                scene.Apply(scene.Truth);
                Capture(fit, scene.Views.Length, "target_warmup");
                var target = new E20Target(name, lamp, shadows, Capture(fit, scene.Views.Length, "targets"), Capture(high, scene.Views.Length, "targets"));
                var repeat = Capture(fit, scene.Views.Length, "target_repeat");
                double max = repeat.Select((f, i) => NativeRenderBackend.Compare(f, target.Fit[i])).Max();
                if (max > 1e-12) throw new InvalidOperationException($"E20 {name} target repeats differ: squared NRMSE {max:G8}");
                var path = CapturePath(name);
                SaveFrames(target.High, path, "gt");
                return target;
            }

            private Frame[] Capture(NativeRenderBackend backend, int views, string purpose)
            {
                var frames = Enumerable.Range(0, views).Select(backend.Capture).ToArray();
                Count(purpose, views);
                return frames;
            }

            private double NativeResidual(E20Target target, double[] x, float[] residual, int views, string purpose)
            {
                scene.Apply(x);
                double loss = 0;
                for (int v = 0; v < views; v++)
                {
                    double viewLoss = fit.Residual(v, target.Fit[v], residual, v * ViewChannels, FitViews);
                    if (v < FitViews) loss += viewLoss;
                }
                Count(purpose, views);
                return loss;
            }

            private string CapturePath(string name)
            {
                string folder = Path.Combine(dir, "captures", name);
                Directory.CreateDirectory(folder);
                return folder;
            }

            private static void SaveFrames(Frame[] frames, string folder, string label)
            {
                for (int v = 0; v < frames.Length; v++)
                    NativeRenderBackend.Save(frames[v], E19HighW, E19HighH, Path.Combine(folder, $"{label}_v{v}"));
            }

            private static int[] Axes(int[] lamps) =>
                lamps.SelectMany(l => Enumerable.Range(l * LampDofs, LampDofs)).ToArray();

            private double[] IsolatedStart(double[] start, int[] lamps)
            {
                var x = (double[])scene.Truth.Clone();
                foreach (int l in lamps) Array.Copy(start, 4 * l, x, 4 * l, 4);
                return x;
            }

            private void LogQuery(string arm, string kind, int query, string phase, int[] lamps, double[] x, double elapsed,
                double native = double.NaN, double direct = double.NaN, double msm = double.NaN, string steps = "",
                double predicted = double.NaN, double actual = double.NaN, string accept = "", int cameraRenders = 0, int faces = 0)
            {
                queries.WriteLine(string.Join(",", ++rowId, arm, kind, query, phase, Ids(lamps), x == null ? "" : Fields(x),
                    Value(native), Value(direct), Value(msm), "", steps, Value(predicted), Value(actual), accept,
                    cameraRenders, faces, Value(elapsed)));
            }

            private Spectrum LogSpectrum(string arm, string phase, int seed, string profile, int batch, int query, int view,
                double positionStep, double logStep, double[] steps, int[] stencil, double[,] gram)
            {
                int n = gram.GetLength(0);
                var spectrum = ClassroomDiagnosis.GetSpectrum(gram, n);
                double[] lampNorms = Enumerable.Range(0, (n + 3) / 4).Select(l =>
                    Math.Sqrt(spectrum.columnNorms.Skip(l * 4).Take(4).Sum(Sq20))).ToArray();
                spectra.WriteLine(string.Join(",", arm, phase, seed, profile, batch, query, view, Value(positionStep), Value(logStep),
                    Fields(steps), string.Join(";", stencil), spectrum.rank, Value(spectrum.cutoff),
                    spectrum.condition < 0 ? "inf" : Value(spectrum.condition), Fields(spectrum.singular), Fields(spectrum.columnNorms), Fields(lampNorms)));
                return spectrum;
            }

            private E20Arm NewArm(string method, E20Target target, E20Start start, int[] lamps, int maxQueries)
            {
                return new E20Arm { id = $"{method}_{start.seed}_{start.profile}_L{Ids(lamps).Replace(';', '-')}",
                    method = method, condition = target.Name, seed = start.seed, profile = start.profile,
                    lamps = lamps.Select(l => l + 1).ToArray(), budget = maxQueries, start = IsolatedStart(start.theta, lamps) };
            }

            private E20Arm NativeArm(string method, E20Target target, E20Start start, int[] lamps)
            {
                var row = NewArm(method, target, start, lamps, budget);
                row.nativeBudget = budget;
                SetMode(target.Shadows, target.Lamp);
                var timer = Stopwatch.StartNew();
                var active = Axes(lamps);
                int calls = 0;
                var full = (double[])row.start.Clone();
                double[] Expand(double[] x)
                {
                    for (int j = 0; j < x.Length; j++) full[active[j]] = x[j];
                    return full;
                }
                double[] last = null;
                string probeStep = "", batchSteps = "";
                var diag = new ImageJacobianLm.Diagnostics
                {
                    Probe = (q, j, h) => probeStep = AxisLabel(active[j]) + "=" + F(h * units[active[j] % 4]),
                    Jacobian = (batch, q, loss, dx, stencil, gram) =>
                    {
                        var metres = dx.Select((h, j) => h * units[active[j] % 4]).ToArray();
                        batchSteps = Fields(metres);
                        LogSpectrum(row.id, "solver_normalized", row.seed, row.profile, batch, q, -1, double.NaN, double.NaN, metres, stencil, gram);
                    },
                    Trial = (batch, q, damping, predicted, actual, ratio, hits, shrinks, status) =>
                        LogQuery(row.id, "decision", q, status, lamps, status == "accept" || status == "reject" ? last : null,
                            timer.Elapsed.TotalSeconds, steps: batchSteps, predicted: -predicted, actual: -actual,
                            accept: status == "accept" ? "1" : "0")
                };
                int before = fit.RenderCount;
                var result = ImageJacobianLm.Run(active.Select(i => row.start[i]).ToArray(), FitViews * ViewChannels,
                    (x, phase, r) =>
                    {
                        var theta = Expand(x);
                        double loss = NativeResidual(target, theta, r, FitViews, "native_optimization");
                        ++calls;
                        if (phase == "lm_trial") last = (double[])theta.Clone();
                        LogQuery(row.id, "optimizer", calls, phase, lamps, theta, timer.Elapsed.TotalSeconds,
                            native: loss, steps: probeStep, cameraRenders: FitViews);
                        probeStep = "";
                        return loss;
                    }, budget, row.seed, x => scene.Contains(Expand(x)), diagnostics: diag);
                if (calls != result.Queries || fit.RenderCount - before != calls * FitViews)
                    throw new InvalidOperationException("E20 LM query accounting mismatch.");
                row.theta = (double[])Expand(result.Best).Clone();
                row.initial = row.nativeInitial = result.InitialLoss;
                row.best = result.BestLoss;
                row.queries = row.nativeQueries = calls;
                row.cameraRenders = calls * FitViews;
                row.stop = result.StopReason; row.accepted = result.Accepted; row.rejected = result.Rejected;
                row.blocked = result.BlockedProbes; row.shortened = result.ShortenedTrials; row.jacobians = result.Jacobians;
                row.seconds = timer.Elapsed.TotalSeconds;
                ValidateNative(row, target, save: true);
                row.recheck = row.nativeBest;
                if (Math.Abs(row.recheck - row.best) > Math.Max(1e-8, row.best * 1e-6))
                    throw new InvalidOperationException("Native best loss did not match its replay: " + row.id);
                AddArm(row);
                return row;
            }

            private void ValidateNative(E20Arm row, E20Target target, bool save)
            {
                SetMode(target.Shadows, target.Lamp);
                row.nativeBest = NativeResidual(target, row.theta, new float[FitViews * ViewChannels], FitViews, "validation");
                row.validationRenders += FitViews;
                scene.Apply(row.theta);
                var best = Capture(high, scene.Views.Length, "validation");
                row.validationRenders += best.Length;
                var errors = best.Select((f, v) => NativeRenderBackend.Compare(f, target.High[v])).ToArray();
                row.trainHighNrmse = Math.Sqrt(errors.Take(FitViews).Average());
                row.heldNrmse = Math.Sqrt(errors.Skip(FitViews).Average());
                row.perViewNrmse = errors.Select(Math.Sqrt).ToArray();
                row.positionCm = row.lamps.Select(l => PositionCm(row.theta, l - 1)).ToArray();
                row.intensityRelative = row.lamps.Select(l => PowerError(row.theta, l - 1)).ToArray();
                row.passed = row.lamps.All(l => Pass(row.theta, l - 1));
                if (save)
                {
                    scene.Apply(row.start);
                    var initial = Capture(high, scene.Views.Length, "validation");
                    row.validationRenders += initial.Length;
                    var path = CapturePath(row.id);
                    SaveFrames(initial, path, "initial");
                    SaveFrames(best, path, "best");
                    for (int v = 0; v < best.Length; v++) SaveRgbError(best[v], target.High[v], Path.Combine(path, $"best_error_v{v}.png"));
                }
            }

            private static void SaveRgbError(Frame a, Frame b, string file)
            {
                var pixels = new Color32[a.PixelCount];
                double normalizer = Math.Sqrt(b.Energy / b.PixelCount) + 1e-10;
                for (int p = 0; p < pixels.Length; p++)
                {
                    var d = a.Pixel(p) - b.Pixel(p);
                    float value = (float)(Math.Log10(1 + Math.Sqrt(Sq20(d.r) + Sq20(d.g) + Sq20(d.b)) / normalizer) / 2);
                    pixels[p] = new Color(Mathf.Clamp01(value * 3), Mathf.Clamp01(value * 2 - .25f), Mathf.Clamp01(value));
                }
                var t = new Texture2D(E19HighW, E19HighH, TextureFormat.RGB24, false, true);
                try { t.SetPixels32(pixels); t.Apply(); File.WriteAllBytes(file, t.EncodeToPNG()); }
                finally { UnityEngine.Object.DestroyImmediate(t); }
            }

            private void AddArm(E20Arm a)
            {
                arms.Add(a);
                WriteJson(dir, a.id + ".json", a);
                stages.WriteLine(string.Join(",", a.id, a.method, a.condition, a.seed, a.profile, string.Join(";", a.lamps), a.queries, a.budget,
                    a.nativeBudget, a.proxyBudget, a.nativeQueries, a.cameraRenders, a.validationRenders, a.proxyQueries, a.shadowFaces, F(a.initial), F(a.best),
                    F(a.recheck), F(a.nativeInitial), F(a.nativeBest), F(a.trainHighNrmse), F(a.heldNrmse), F(a.proxyHeldNrmse),
                    Fields(a.positionCm), Fields(a.intensityRelative), a.accepted, a.rejected, a.blocked, a.shortened, a.jacobians,
                    a.nonfinite, F(a.seconds), a.passed ? "1" : "0", a.stop));
                stages.Flush(); queries.Flush(); spectra.Flush();
                Debug.Log($"[E20] {a.id}: {a.queries} queries, RGB {Math.Sqrt(a.nativeInitial):P3} → {Math.Sqrt(a.nativeBest):P3}, " +
                          $"max {a.positionCm.Max():F2} cm / {a.intensityRelative.Max():P2}, {a.stop}");
            }

            public void Dispose()
            {
                SetMode(true, -1);
                scene.Apply(scene.Truth);
                queries.Dispose(); spectra.Dispose(); stages.Dispose(); fit.Dispose(); high.Dispose();
            }
        }
    }
}

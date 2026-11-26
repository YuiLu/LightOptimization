using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using UnityEditor.SceneManagement;
using UnityEngine;
using Frame = LightingMvp.NativeRenderBackend.Frame;

namespace LightingMvp.Editor
{
    /// <summary>E17: shadow ablation, multi-scale local sensitivity and LM trajectory diagnosis.</summary>
    public static partial class NativeRecoveryExperiment
    {
        private const int E17Width = 448, E17Height = 256, E17CheckWidth = 896, E17CheckHeight = 512;
        private static readonly float[] E17PositionStepsCm = { .5f, 1f, 2f, 5f };
        private static readonly float[] E17IntensitySteps = { .02f, .05f, .1f };

        [Serializable] private sealed class E17Config
        {
            public string scene, task, unity, urp, renderer, gpu, graphics, colorSpace, startSource;
            public int lamps, trainingViews, checkViews, width, height, checkWidth, checkHeight, budget, seed;
            public bool bakedGi, realtimeGi;
            public double noiseRms, noiseHigh, noiseThreshold, ambient;
            public Vector3 boundsMin, boundsMax;
            public Vector3[] viewPositions;
            public Quaternion[] viewRotations;
            public NativeLamp[] gt;
        }
        [Serializable] private sealed class E17Shadow
        {
            public int lamp;
            public double[] energy, startEnergy, affected, overlapOn, overlapOff, maxDiff;
            public double meanEnergy, meanStartEnergy, shadowChange, meanAffected, meanOverlapOn, meanOverlapOff;
        }
        [Serializable] private sealed class E17Trial
        {
            public string scene, shadows, stop;
            public int lamp, seed, budget, queries, jacobians, accepted, rejected, blocked, shortened, trainRenders;
            public bool passed;
            public double initialNrmse, finalNrmse, bestNrmse, reduction, checkNrmse, heldNrmse;
            public double positionErrorCm, relativeIntensityError, wallSeconds;
            public double[] start, best, positionErrorMetres;
        }
        [Serializable] private sealed class E17Probe
        {
            public string scene, kind, shadows, state;
            public int lamp, axis;
            public double step, norm, activeRatio, alignment, consistency, normRatio;
        }
        [Serializable] private sealed class E17Run
        {
            public string scene, task, error, startSource;
            public int lamps, trainingViews, seed, budget;
            public double noiseRms, noiseHigh, noiseThreshold, elapsed;
            public E17Shadow[] shadows;
            public E17Trial[] trials;
            public E17Probe[] probes;
        }

        private interface ISensitivityScene : IDisposable
        {
            string Name { get; }
            string RendererName { get; }
            int Count { get; }
            int[] Lamps { get; }
            double[] Truth { get; }
            Vector3[] Views { get; }
            Quaternion[] Rotations { get; }
            Camera Camera { get; }
            Vector3 Extents { get; }
            string StartSource { get; }
            double[] Start(int seed);
            void Apply(double[] theta);
            void SetShadows(int lamp, bool shadows);
            void ResetShadows();
            bool SupportsFixtureShadows { get; }
            void SetFixtureShadows(int lamp, bool casting);
            void SetShadowStrength(int lamp, float strength);
            Vector3 PositionOf(double[] theta, int lamp);
            double PowerOf(double[] theta, int lamp);
            bool Contains(double[] theta);
            E17Config Describe();
            string SourceName(int lamp);
            string UnityName(int lamp);
            string[] FixtureRenderers(int lamp);
            int[] FixtureTriangleCounts();
            Bounds FixtureBounds(int lamp);
            double[] WithPosition(double[] theta, int lamp, Vector3 position);
        }

        private static void RunSensitivity(string task, string output)
        {
            if (!Application.isBatchMode) throw new InvalidOperationException("E17 opens transient scenes and is batch-only.");
            if (!UnityEngine.Rendering.GraphicsSettings.lightsUseLinearIntensity)
                throw new InvalidOperationException("Enable linear light intensity for calibrated color coordinates.");
            bool room = task.StartsWith("e17-room", StringComparison.Ordinal);
            bool smoke = task.EndsWith("-smoke", StringComparison.Ordinal);
            string dir = Path.Combine(output, task.Replace('-', '_'));
            Directory.CreateDirectory(dir);
            int seed = int.Parse(Arg("-bbE17Seed", "131"), Invariant);
            int budget = int.Parse(Arg("-bbE17Budget", smoke ? "160" : "2000"), Invariant);
            if (budget < 40 || budget > 8000) throw new ArgumentException("-bbE17Budget must be 40-8000 optimizer queries (three training views each).");
            string savedStart = Arg("-bbE17Start", "");
            using var pipeline = new TestPipeline(classroom: !room);
            using ISensitivityScene scene = room ? (ISensitivityScene)new RoomSensitivityScene() : new ClassroomSensitivityScene(pipeline.Asset, savedStart);
            new SensitivityRunner(dir, scene, budget, seed, smoke).Run();
        }

        [Serializable] private sealed class E17SavedTrial { public string key, method; public int seed, lamp; public double[] start; }

        private sealed class ClassroomSensitivityScene : ISensitivityScene
        {
            private readonly ClassroomFixture fixture;
            private readonly string savedStartRoot;
            private bool savedStartUsed;
            internal ClassroomSensitivityScene(UnityEngine.Rendering.Universal.UniversalRenderPipelineAsset pipeline, string savedStartRoot)
            {
                fixture = new ClassroomFixture(pipeline);
                this.savedStartRoot = string.IsNullOrEmpty(savedStartRoot) ? null : savedStartRoot;
            }
            public string Name => "classroom";
            public string RendererName => fixture.Config.renderer;
            public int Count => fixture.Rig.Count;
            public int[] Lamps => new[] { 0, 3, 4, 5 };
            public double[] Truth => fixture.Truth;
            public Vector3[] Views => fixture.Config.views.Select(v => v.position).ToArray();
            public Quaternion[] Rotations => fixture.Config.views.Select(v => v.rotation).ToArray();
            public Camera Camera => fixture.Camera;
            public Vector3 Extents => RoomMax - RoomMin;
            public string StartSource => savedStartUsed ? "e13d_saved_near" : "generated_far";
            public double[] Start(int seed)
            {
                if (savedStartRoot == null) return fixture.Perturb(seed, near: false);
                var theta = (double[])fixture.Truth.Clone();
                foreach (int lamp in Lamps)
                {
                    string path = Path.Combine(savedStartRoot, "e10_near_" + seed.ToString(Invariant) + "_single_position_strength_" + lamp.ToString(Invariant) + "_Raw", "trial.json");
                    if (!File.Exists(path)) { Debug.LogWarning("[E17] missing saved E13d start " + path + "; using a generated far perturbation."); return fixture.Perturb(seed, near: false); }
                    var saved = JsonUtility.FromJson<E17SavedTrial>(File.ReadAllText(path));
                    if (saved?.start == null || saved.start.Length != Count * LampDofs || saved.lamp != lamp)
                    { Debug.LogWarning("[E17] invalid saved E13d start " + path + "; using a generated far perturbation."); return fixture.Perturb(seed, near: false); }
                    Array.Copy(saved.start, lamp * LampDofs, theta, lamp * LampDofs, LampDofs);
                }
                if (!fixture.Rig.Contains(theta)) { Debug.LogWarning("[E17] saved E13d start outside the search domain; using a generated far perturbation."); return fixture.Perturb(seed, near: false); }
                savedStartUsed = true;
                Debug.Log("[E17] classroom start: E13d saved near perturbation, seed " + seed.ToString(Invariant));
                return theta;
            }
            public void Apply(double[] theta) => fixture.Rig.Apply(theta);
            public void SetShadows(int lamp, bool shadows) => fixture.SetShadows(lamp, shadows);
            public void ResetShadows() => fixture.ResetShadows();
            public bool SupportsFixtureShadows => fixture.SupportsFixtureShadows;
            public void SetFixtureShadows(int lamp, bool casting) => fixture.SetFixtureShadows(lamp, casting);
            public void SetShadowStrength(int lamp, float strength) => fixture.SetShadowStrength(lamp, strength);
            public string SourceName(int lamp) => fixture.Config.original[lamp].name;
            public string UnityName(int lamp) => "Classroom point " + (lamp + 1).ToString(Invariant);
            public string[] FixtureRenderers(int lamp) { var list = fixture.FixtureRendererNames(lamp); return list.Count == 0 ? Array.Empty<string>() : list.ToArray(); }
            public int[] FixtureTriangleCounts() => fixture.FixtureTriangles == null ? new int[Count] : (int[])fixture.FixtureTriangles.Clone();
            public Bounds FixtureBounds(int lamp) => fixture.FixtureBounds(lamp);
            public double[] WithPosition(double[] theta, int lamp, Vector3 position)
            {
                var x = (double[])theta.Clone(); int k = lamp * LampDofs;
                var moved = ClassroomFixture.Encode(position, PowerMin * Math.Exp(theta[k + 3] * LogPowerSpan));
                Array.Copy(moved, 0, x, k, 3);
                return x;
            }
            public Vector3 PositionOf(double[] theta, int lamp) => ClassroomFixture.Position(theta, lamp * LampDofs);
            public double PowerOf(double[] theta, int lamp) => PowerMin * Math.Exp(theta[lamp * LampDofs + 3] * LogPowerSpan);
            public bool Contains(double[] theta) => fixture.Rig.Contains(theta);
            public E17Config Describe() => new E17Config
            {
                scene = Name, unity = fixture.Config.unity, urp = fixture.Config.urp, renderer = RendererName,
                gpu = fixture.Config.gpu, graphics = fixture.Config.graphics, colorSpace = fixture.Config.colorSpace,
                lamps = Count, trainingViews = 3, checkViews = Views.Length, width = E17Width, height = E17Height,
                checkWidth = E17CheckWidth, checkHeight = E17CheckHeight, boundsMin = RoomMin, boundsMax = RoomMax,
                bakedGi = false, realtimeGi = false, ambient = fixture.Config.ambientLinear, gt = fixture.Config.gt,
                viewPositions = Views, viewRotations = Rotations
            };
            public void Dispose() => fixture.Dispose();
        }

        private sealed class RoomSensitivityScene : ISensitivityScene
        {
            private static readonly Vector3 MinBounds = new Vector3(-3.75f, .25f, -3.75f), MaxBounds = new Vector3(3.75f, 2.9f, 3.75f);
            private readonly Light[] lights;
            private readonly Camera camera;
            private readonly double[] truth;
            private static readonly Vector3[] Eyes =
            {
                new Vector3(2.95f, 1.55f, 3.10f), new Vector3(-2.90f, 1.60f, 2.90f), new Vector3(3.05f, 1.50f, -2.85f),
                new Vector3(-3.00f, 1.45f, -3.00f), new Vector3(0f, 1.70f, 3.55f)
            };
            private static readonly Vector3[] Look =
            {
                new Vector3(-.2f, 1.15f, -.2f), new Vector3(.4f, 1.10f, -.4f), new Vector3(-.4f, 1.15f, .5f),
                new Vector3(.6f, 1.20f, .6f), new Vector3(0f, 1.10f, -1.5f)
            };

            internal RoomSensitivityScene()
            {
                var scene = EditorSceneManager.OpenScene("Assets/Scenes/UrpNativeRoom.unity", OpenSceneMode.Single);
                lights = Enumerable.Range(1, 7).Select(i => GameObject.Find($"Candidate Light {i:00}")?.GetComponent<Light>()).ToArray();
                if (lights.Any(l => !l)) throw new InvalidOperationException("Assets/Scenes/UrpNativeRoom.unity is missing Candidate Light 01..07.");
                camera = GameObject.Find("Validation Camera")?.GetComponent<Camera>();
                if (!camera) throw new InvalidOperationException("Validation Camera missing from the native room scene.");
                if (scene.isDirty) throw new InvalidOperationException("The native room scene has unsaved changes in the editor.");
                truth = lights.SelectMany(l => Encode(l.transform.position, l.intensity)).ToArray();
                if (!Contains(truth)) throw new InvalidOperationException("Authored native room rig is outside the search domain.");
            }

            public string Name => "native_room";
            public string RendererName => "URP stock Forward";
            public int Count => lights.Length;
            public int[] Lamps => new[] { 0, 3, 4, 5 };
            public double[] Truth => truth;
            public Vector3[] Views => Eyes;
            public Quaternion[] Rotations => Eyes.Select((e, i) => Quaternion.LookRotation((Look[i] - e).normalized, Vector3.up)).ToArray();
            public Camera Camera => camera;
            public Vector3 Extents => MaxBounds - MinBounds;
            public string StartSource => "scene_seeded";

            private static double[] Encode(Vector3 p, double power) => new[] {
                ((double)p.x - MinBounds.x) / (MaxBounds.x - MinBounds.x), ((double)p.y - MinBounds.y) / (MaxBounds.y - MinBounds.y),
                ((double)p.z - MinBounds.z) / (MaxBounds.z - MinBounds.z), PowerCoordinate(power) };
            public static Vector3 Decode(double[] x, int k) => new Vector3(
                Mathf.Lerp(MinBounds.x, MaxBounds.x, (float)x[k]), Mathf.Lerp(MinBounds.y, MaxBounds.y, (float)x[k + 1]),
                Mathf.Lerp(MinBounds.z, MaxBounds.z, (float)x[k + 2]));
            private static double Power(double q) => PowerMin * Math.Exp(q * LogPowerSpan);

            public double[] Start(int seed)
            {
                var rng = new System.Random(seed); var theta = (double[])truth.Clone();
                for (int i = 0; i < Count; ++i)
                {
                    int k = i * LampDofs;
                    var p = Decode(truth, k); double q = Power(truth[k + 3]);
                    var offset = .50f + .18f * (float)rng.NextDouble();
                    q = Math.Max(PowerMin, Math.Min(PowerMax, q * (rng.NextDouble() < .5 ? .48 : 1.85)));
                    bool placed = false;
                    for (int attempt = 0; attempt < 4096 && !placed; ++attempt)
                    {
                        if (attempt > 0 && attempt % 256 == 0) offset = Math.Max(.08f, offset * .8f);
                        var d = new Vector3((float)rng.NextDouble() - .5f, (float)rng.NextDouble() - .5f, (float)rng.NextDouble() - .5f).normalized;
                        var lamp = Encode(p + offset * d, q);
                        if (!Contains(lamp)) continue;
                        Array.Copy(lamp, 0, theta, k, LampDofs); placed = true;
                    }
                    if (!placed) throw new InvalidOperationException("No admissible perturbed native-room pose for light " + (i + 1) + ".");
                    Debug.Log($"[E17] native_room perturb L{i + 1}: {offset:F3} m");
                }
                return theta;
            }

            public void Apply(double[] theta)
            {
                if (!Contains(theta)) throw new InvalidOperationException("Illegal native-room rig submitted.");
                for (int i = 0; i < Count; ++i)
                {
                    int k = i * LampDofs;
                    lights[i].transform.position = Decode(theta, k);
                    lights[i].intensity = (float)Power(theta[k + 3]);
                }
            }
            public void SetShadows(int lamp, bool shadows) => lights[lamp].shadows = shadows ? LightShadows.Hard : LightShadows.None;
            public void ResetShadows() { foreach (var lamp in lights) { lamp.shadows = LightShadows.Hard; lamp.shadowStrength = 1; } }
            public bool SupportsFixtureShadows => false;
            public void SetFixtureShadows(int lamp, bool casting) { }
            public void SetShadowStrength(int lamp, float strength) => lights[lamp].shadowStrength = Mathf.Clamp01(strength);
            public string SourceName(int lamp) => "Candidate Light " + (lamp + 1).ToString("00", Invariant);
            public string UnityName(int lamp) => "Candidate Light " + (lamp + 1).ToString("00", Invariant);
            public string[] FixtureRenderers(int lamp) => Array.Empty<string>();
            public int[] FixtureTriangleCounts() => new int[Count];
            public Bounds FixtureBounds(int lamp) => default;
            public double[] WithPosition(double[] theta, int lamp, Vector3 position)
            {
                var x = (double[])theta.Clone(); int k = lamp * LampDofs;
                var moved = Encode(position, Power(theta[k + 3]));
                Array.Copy(moved, 0, x, k, 3);
                return x;
            }
            public Vector3 PositionOf(double[] theta, int lamp) => Decode(theta, lamp * LampDofs);
            public double PowerOf(double[] theta, int lamp) => Power(theta[lamp * LampDofs + 3]);
            public bool Contains(double[] theta)
            {
                if (theta == null || theta.Length % LampDofs != 0 || theta.Length > Count * LampDofs || theta.Any(u => !(u >= 0 && u <= 1))) return false;
                for (int i = 0; i < theta.Length; i += LampDofs)
                {
                    var p = Decode(theta, i);
                    if (Physics.CheckSphere(p, .06f, ~0, QueryTriggerInteraction.Ignore)) return false;
                    if ((Mathf.Abs(p.x) > 3.6f) || (Mathf.Abs(p.z) > 3.6f) || p.y < .3f || p.y > 2.85f) return false;
                }
                return true;
            }
            public E17Config Describe() => new E17Config
            {
                scene = Name, unity = Application.unityVersion, urp = UnityEditor.PackageManager.PackageInfo.FindForAssembly(typeof(UnityEngine.Rendering.Universal.UniversalRenderPipeline).Assembly)?.version,
                renderer = RendererName, gpu = SystemInfo.graphicsDeviceName, graphics = SystemInfo.graphicsDeviceType.ToString(),
                colorSpace = QualitySettings.activeColorSpace.ToString(), lamps = Count, trainingViews = 3, checkViews = Eyes.Length,
                width = E17Width, height = E17Height, checkWidth = E17CheckWidth, checkHeight = E17CheckHeight,
                boundsMin = MinBounds, boundsMax = MaxBounds, bakedGi = false, realtimeGi = false,
                ambient = RenderSettings.ambientLight.linear.r,
                gt = lights.Select(NativeLamp.Read).ToArray(), viewPositions = Eyes, viewRotations = Rotations
            };
            public void Dispose() { }
        }

        private sealed class SensitivityRunner
        {
            private readonly string dir;
            private readonly ISensitivityScene scene;
            private readonly NativeRenderBackend train, check;
            private readonly int budget, seed;
            private readonly bool smoke;
            private readonly E17Run run = new E17Run();
            private readonly List<E17Trial> trials = new List<E17Trial>();
            private readonly List<E17Probe> probes = new List<E17Probe>();
            private readonly List<E17Shadow> shadows = new List<E17Shadow>();
            private readonly System.Diagnostics.Stopwatch watch = System.Diagnostics.Stopwatch.StartNew();
            private readonly int trainViews = 3, checkViews = 5;
            private double noiseRms, threshold;
            private StreamWriter trace, jacTrace, probeCsv, summaryCsv, queryCsv, gramCsv;

            internal SensitivityRunner(string dir, ISensitivityScene scene, int budget, int seed, bool smoke)
            {
                this.dir = dir; this.scene = scene; this.budget = budget; this.seed = seed; this.smoke = smoke;
                train = new NativeRenderBackend(scene.Camera, scene.Views.Take(trainViews).ToArray(), scene.Rotations.Take(trainViews).ToArray(), E17Width, E17Height, -1);
                check = new NativeRenderBackend(scene.Camera, scene.Views, scene.Rotations, E17CheckWidth, E17CheckHeight, -1);
                run.scene = scene.Name; run.task = Path.GetFileName(dir); run.seed = seed; run.budget = budget; run.lamps = scene.Count; run.trainingViews = trainViews;
                trace = new StreamWriter(Path.Combine(dir, "trials-steps.csv"));
                trace.WriteLine("scene,lamp,shadows,batch,query,mu,predicted,actual,ratio,status");
                jacTrace = new StreamWriter(Path.Combine(dir, "jacobians.csv"));
                jacTrace.WriteLine("scene,lamp,shadows,batch,query,loss,column_norm,normalized_step,stencil");
                probeCsv = new StreamWriter(Path.Combine(dir, "probes.csv"));
                probeCsv.WriteLine("scene,state,lamp,shadows,kind,axis,step,norm,activeRatio,alignment,consistency,normRatio");
                gramCsv = new StreamWriter(Path.Combine(dir, "gram.csv"));
                gramCsv.WriteLine("scene,lamp,shadows,state,rank,condition,minEigen,maxEigen,correlation,normRatio");
                summaryCsv = new StreamWriter(Path.Combine(dir, "trials.csv"));
                summaryCsv.WriteLine("scene,lamp,shadows,seed,budget,queries,trainRenders,jacobians,accepted,rejected,blocked,shortened,stop,initialNrmse,finalNrmse,bestNrmse,reduction,checkNrmse,heldNrmse,positionErrorCm,relativeIntensityError,wallSeconds,passed");
                queryCsv = new StreamWriter(Path.Combine(dir, "queries.csv"));
                queryCsv.WriteLine("scene,lamp,shadows,query,phase,x,y,z,power,loss");
            }

            internal void Run()
            {
                try
                {
                    scene.ResetShadows();
                    scene.Apply(scene.Truth);
                    for (int v = 0; v < trainViews; ++v) train.Capture(v);
                    var a = CaptureTrain(); var b = CaptureTrain();
                    noiseRms = RepeatNoise(a, b);
                    threshold = Math.Max(3 * noiseRms, 1e-7);
                    var ha = CaptureCheck(); var hb = CaptureCheck();
                    double noiseHigh = RepeatNoise(ha, hb);
                    run.noiseRms = noiseRms; run.noiseHigh = noiseHigh; run.noiseThreshold = threshold;
                    var cfg = scene.Describe(); cfg.task = run.task; cfg.budget = budget; cfg.seed = seed;
                    cfg.noiseRms = noiseRms; cfg.noiseHigh = noiseHigh; cfg.noiseThreshold = threshold;

                    double[] start = scene.Start(seed);
                    run.startSource = scene.StartSource; cfg.startSource = scene.StartSource;
                    WriteJson(dir, "scene.json", cfg);
                    Debug.Log($"[E17] {scene.Name}: repeat noise {noiseRms:G4}, high {noiseHigh:G4}, threshold {threshold:G4}, start {scene.StartSource}");
                    foreach (int lamp in scene.Lamps)
                    {
                        var onT = ArmTargets(lamp, true); var onC = CheckTargets(lamp, true);
                        var offT = ArmTargets(lamp, false); var offC = CheckTargets(lamp, false);
                        var shadow = ShadowStats(lamp, true, onT, offT, start);
                        shadows.Add(shadow);
                        SaveHeatmap(onT[0], offT[0], Path.Combine(dir, $"shadow-diff-l{lamp + 1}.png"));
                        NativeRenderBackend.Save(onT[0], E17Width, E17Height, Path.Combine(dir, $"target-l{lamp + 1}-on"));
                        NativeRenderBackend.Save(offT[0], E17Width, E17Height, Path.Combine(dir, $"target-l{lamp + 1}-off"));
                        Debug.Log($"[E17] {scene.Name} L{lamp + 1}: shadow RMS {shadow.meanEnergy:G4}, moved {shadow.shadowChange:F2}x, affected {shadow.meanAffected:P2}, overlap {shadow.meanOverlapOn:P2}");

                        Arm(lamp, true, onT, onC, start, true);
                        Arm(lamp, false, offT, offC, start, true);
                    }

                    run.shadows = shadows.ToArray(); run.trials = trials.ToArray(); run.probes = probes.ToArray();
                    run.elapsed = watch.Elapsed.TotalSeconds;
                    WriteJson(dir, "summary.json", run);
                    WriteProbeCsv();
                    Debug.Log($"[E17] {scene.Name}: finished {trials.Count} arms in {run.elapsed:F1}s");
                }
                catch (Exception ex)
                {
                    run.error = ex.GetType().Name + ": " + ex.Message;
                    try { WriteJson(dir, "summary.json", run); } catch { }
                    throw;
                }
                finally
                {
                    trace.Dispose(); jacTrace.Dispose(); probeCsv.Dispose(); summaryCsv.Dispose(); queryCsv.Dispose(); gramCsv.Dispose();
                }
            }

            private void Arm(int lamp, bool shadowsOn, Frame[] target, Frame[] checkTarget, double[] start, bool withProbes)
            {
                int k = lamp * LampDofs;
                var sub = new double[LampDofs];
                Array.Copy(start, k, sub, 0, LampDofs);
                var full = (double[])scene.Truth.Clone();
                var initial = (double[])sub.Clone();
                double[] mid = null, best = null;
                int queries = 0;
                int calls = 0;
                var t0 = watch.Elapsed.TotalSeconds;

                SetArmShadows(lamp, shadowsOn);
                ApplySub(full, sub, k);

                var diagnostics = new ImageJacobianLm.Diagnostics
                {
                    Jacobian = (batch, query, loss, effective, stencil, normal) =>
                    {
                        int n = normal.GetLength(0);
                        var norms = new string[n]; var steps = new string[n]; var kinds = new string[n];
                        for (int i = 0; i < n; ++i)
                        {
                            norms[i] = F(Math.Sqrt(Math.Max(0, normal[i, i])));
                            steps[i] = F(effective[i]); kinds[i] = stencil[i].ToString(Invariant);
                        }
                        jacTrace.WriteLine(string.Join(",", scene.Name, (lamp + 1).ToString(Invariant), shadowsOn ? "on" : "off",
                            batch.ToString(Invariant), query.ToString(Invariant), F(loss), string.Join(";", norms), string.Join(";", steps), string.Join(";", kinds)));
                    },
                    Trial = (batch, query, mu, predicted, actual, ratio, boundHits, shrinks, status) =>
                    {
                        trace.WriteLine(string.Join(",", scene.Name, (lamp + 1).ToString(Invariant), shadowsOn ? "on" : "off",
                            batch.ToString(Invariant), query.ToString(Invariant), F(mu), Finite(predicted), Finite(actual), Finite(ratio), status));
                    }
                };

                double Evaluate(double[] candidate, string phase, float[] buffer)
                {
                    ApplySub(full, candidate, k);
                    if (!scene.Contains(full)) throw new InvalidOperationException("Optimizer requested an illegal rig.");
                    scene.Apply(full);
                    double loss = 0;
                    for (int v = 0; v < trainViews; ++v) loss += train.Residual(v, target[v], buffer, v * E17Width * E17Height * 3, trainViews);
                    ++queries; ++calls;
                    var qp = scene.PositionOf(full, lamp);
                    queryCsv.WriteLine(string.Join(",", scene.Name, (lamp + 1).ToString(Invariant), shadowsOn ? "on" : "off",
                        queries.ToString(Invariant), phase, F(qp.x), F(qp.y), F(qp.z), F(scene.PowerOf(full, lamp)), F(loss)));
                    if (mid == null && queries >= Math.Max(2, budget / 2)) mid = (double[])candidate.Clone();
                    if ((calls & 127) == 0) Debug.Log($"[E17] {scene.Name} L{lamp + 1} {(shadowsOn ? "on" : "off")}: q={queries}, loss={loss:G4}");
                    return loss;
                }

                bool Feasible(double[] candidate)
                {
                    for (int i = 0; i < LampDofs; ++i) if (!(candidate[i] >= 0 && candidate[i] <= 1)) return false;
                    ApplySub(full, candidate, k);
                    return scene.Contains(full);
                }

                var result = ImageJacobianLm.Run(sub, E17Width * E17Height * 3 * trainViews, Evaluate, budget, seed, Feasible, false, diagnostics);
                best = (double[])result.Best.Clone();
                if (mid == null) mid = (double[])best.Clone();

                ApplySub(full, best, k);
                scene.Apply(full);
                var finalTrain = CaptureTrain();
                double initialNrmse = Math.Sqrt(result.InitialLoss);
                double finalNrmse = Nrmse(finalTrain, target);
                double bestNrmse = Math.Sqrt(result.BestLoss);

                scene.Apply(full);
                var high = CaptureCheck();
                var err = new double[checkViews];
                for (int v = 0; v < checkViews; ++v) err[v] = NativeRenderBackend.Compare(high[v], checkTarget[v]);
                double checkNrmse = Math.Sqrt(err.Take(3).Average());
                double heldNrmse = Math.Sqrt(err.Skip(3).Average());
                NativeRenderBackend.Save(high[0], E17CheckWidth, E17CheckHeight, Path.Combine(dir, $"best-{scene.Name}-l{lamp + 1}-{(shadowsOn ? "on" : "off")}"));

                var bestPos = scene.PositionOf(full, lamp); var truthPos = scene.PositionOf(scene.Truth, lamp);
                var posErr = bestPos - truthPos;
                double posCm = posErr.magnitude * 100;
                double relI = Math.Abs(scene.PowerOf(full, lamp) - scene.PowerOf(scene.Truth, lamp)) / scene.PowerOf(scene.Truth, lamp);
                var row = new E17Trial
                {
                    scene = scene.Name, lamp = lamp + 1, shadows = shadowsOn ? "on" : "off", seed = seed, budget = budget,
                    queries = result.Queries, jacobians = result.Jacobians, accepted = result.Accepted, rejected = result.Rejected, trainRenders = result.Queries * trainViews,
                    blocked = result.BlockedProbes, shortened = result.ShortenedTrials, stop = result.StopReason,
                    initialNrmse = initialNrmse, finalNrmse = finalNrmse, bestNrmse = bestNrmse,
                    reduction = initialNrmse > 0 ? 1 - Sq(finalNrmse) / Sq(initialNrmse) : 0,
                    checkNrmse = checkNrmse, heldNrmse = heldNrmse, positionErrorCm = posCm, relativeIntensityError = relI,
                    wallSeconds = watch.Elapsed.TotalSeconds - t0, passed = posCm <= 5 && relI <= .05,
                    start = initial, best = best, positionErrorMetres = new[] { (double)posErr.x, (double)posErr.y, (double)posErr.z }
                };
                trials.Add(row);
                summaryCsv.WriteLine(string.Join(",", row.scene, row.lamp.ToString(Invariant), row.shadows, row.seed.ToString(Invariant),
                    row.budget.ToString(Invariant), row.queries.ToString(Invariant), row.trainRenders.ToString(Invariant), row.jacobians.ToString(Invariant), row.accepted.ToString(Invariant),
                    row.rejected.ToString(Invariant), row.blocked.ToString(Invariant), row.shortened.ToString(Invariant), row.stop,
                    F(row.initialNrmse), F(row.finalNrmse), F(row.bestNrmse), F(row.reduction), F(row.checkNrmse), F(row.heldNrmse),
                    F(row.positionErrorCm), F(row.relativeIntensityError), F(row.wallSeconds), row.passed ? "1" : "0"));
                Debug.Log($"[E17] {scene.Name} L{lamp + 1} {(shadowsOn ? "on" : "off")}: NRMSE {initialNrmse:P1} -> {finalNrmse:P1}, pos {posCm:F2} cm, I {relI:P1}, {result.StopReason}, {row.wallSeconds:F0}s");

                if (!withProbes) return;
                foreach (var state in new[] { ("initial", initial), ("mid", mid), ("best", best) })
                {
                    SetArmShadows(lamp, shadowsOn);
                    Probe(state.Item1, lamp, shadowsOn, state.Item2, target);
                }
            }

            private void Probe(string state, int lamp, bool shadowsOn, double[] sub, Frame[] target)
            {
                ProbeGram(state, lamp, shadowsOn, sub);
                int k = lamp * LampDofs;
                var full = (double[])scene.Truth.Clone();
                ApplySub(full, sub, k);
                scene.Apply(full);
                var frames = CaptureTrain();
                double energy = 0, rsq = 0;
                int channels = E17Width * E17Height * 3;
                var r = new double[channels * trainViews];
                for (int v = 0; v < trainViews; ++v)
                {
                    var f = frames[v]; var t = target[v]; energy += t.Energy;
                    for (int p = 0, c = v * channels; p < f.PixelCount; ++p)
                    {
                        var x = f.Pixel(p); var y = t.Pixel(p);
                        double a = x.r - y.r, b = x.g - y.g, c2 = x.b - y.b;
                        r[c++] = a; r[c++] = b; r[c++] = c2; rsq += a * a + b * b + c2 * c2;
                    }
                }
                double rnorm = Math.Sqrt(rsq);
                var extents = scene.Extents;

                for (int axis = 0; axis < LampDofs; ++axis)
                {
                    double[] previous = null; double previousNorm = 0;
                    bool logAxis = axis == 3;
                    int count = logAxis ? E17IntensitySteps.Length : E17PositionStepsCm.Length;
                    for (int s = 0; s < count; ++s)
                    {
                        double physical = logAxis ? E17IntensitySteps[s] : E17PositionStepsCm[s] / 100.0;
                        double hNorm = logAxis ? physical / LogPowerSpan : physical / AxisExtent(extents, axis);
                        var column = Column(lamp, sub, axis, hNorm, out double usedPhysical, out bool ok);
                        if (!ok)
                        {
                            probes.Add(new E17Probe { scene = scene.Name, state = state, lamp = lamp + 1, shadows = shadowsOn ? "on" : "off",
                                kind = logAxis ? "log_intensity" : "position", axis = axis, step = physical, norm = -1 });
                            continue;
                        }
                        double norm = Math.Sqrt(column.Sum(x => x * x) / energy) / usedPhysical;
                        double active = column.Count(x => Math.Abs(x) > threshold) / (double)column.Length;
                        double dot = 0, csq = 0;
                        for (int i = 0; i < column.Length; ++i) { dot += column[i] * r[i]; csq += column[i] * column[i]; }
                        double alignment = rnorm > 0 && csq > 0 ? dot / (Math.Sqrt(csq) * rnorm) : 0;
                        double consistency = 0, normRatio = 0;
                        if (previous != null)
                        {
                            double pdot = 0, pn = 0;
                            for (int i = 0; i < column.Length; ++i) { pdot += column[i] * previous[i]; pn += previous[i] * previous[i]; }
                            consistency = pn > 0 && csq > 0 ? pdot / (Math.Sqrt(csq) * Math.Sqrt(pn)) : 0;
                            normRatio = previousNorm > 0 ? norm / previousNorm : 0;
                        }
                        probes.Add(new E17Probe { scene = scene.Name, state = state, lamp = lamp + 1, shadows = shadowsOn ? "on" : "off",
                            kind = logAxis ? "log_intensity" : "position", axis = axis, step = physical, norm = norm,
                            activeRatio = active, alignment = alignment, consistency = consistency, normRatio = normRatio });
                        previous = column; previousNorm = norm;
                    }
                }
            }

            private void ProbeGram(string state, int lamp, bool shadowsOn, double[] sub)
            {
                var extents = scene.Extents;
                var columns = new double[LampDofs][]; var norms = new double[LampDofs];
                for (int axis = 0; axis < LampDofs; ++axis)
                {
                    bool logAxis = axis == 3;
                    double physical = logAxis ? E17IntensitySteps[0] : E17PositionStepsCm[0] / 100.0;
                    double hNorm = logAxis ? physical / LogPowerSpan : physical / AxisExtent(extents, axis);
                    var column = Column(lamp, sub, axis, hNorm, out _, out bool ok);
                    if (!ok)
                    {
                        gramCsv.WriteLine(string.Join(",", scene.Name, (lamp + 1).ToString(Invariant), shadowsOn ? "on" : "off", state, "0", "", "", "", "", ""));
                        return;
                    }
                    columns[axis] = column;
                    double sq = 0; foreach (double x in column) sq += x * x;
                    norms[axis] = Math.Sqrt(sq);
                    if (!(norms[axis] > 0)) norms[axis] = double.Epsilon;
                }
                var gram = new double[LampDofs, LampDofs];
                for (int i = 0; i < LampDofs; ++i)
                for (int j = 0; j < LampDofs; ++j)
                {
                    double dot = 0;
                    for (int q = 0; q < columns[i].Length; ++q) dot += columns[i][q] * columns[j][q];
                    gram[i, j] = dot / (norms[i] * norms[j]);
                }
                var eigen = SymmetricEigenvalues(gram);
                double max = eigen.Max(), min = eigen.Min();
                int rank = eigen.Count(e => e > 1e-6 * Math.Max(max, 1e-12));
                double normRatio = norms.Max() / Math.Max(double.Epsilon, norms.Min());
                double condition = min > 1e-9 ? Math.Sqrt(max / min) * normRatio : double.PositiveInfinity;
                double correlation = 0;
                for (int i = 0; i < LampDofs; ++i) for (int j = i + 1; j < LampDofs; ++j) correlation = Math.Max(correlation, Math.Abs(gram[i, j]));
                gramCsv.WriteLine(string.Join(",", scene.Name, (lamp + 1).ToString(Invariant), shadowsOn ? "on" : "off", state,
                    rank.ToString(Invariant), Finite(condition), F(min), F(max), F(correlation), F(normRatio)));
            }

            private static double[] SymmetricEigenvalues(double[,] source)
            {
                int n = source.GetLength(0);
                var m = (double[,])source.Clone();
                for (int sweep = 0; sweep < 64; ++sweep)
                {
                    double off = 0;
                    for (int i = 0; i < n - 1; ++i) for (int j = i + 1; j < n; ++j) off += m[i, j] * m[i, j];
                    if (off < 1e-22) break;
                    for (int i = 0; i < n - 1; ++i)
                    for (int j = i + 1; j < n; ++j)
                    {
                        if (Math.Abs(m[i, j]) < 1e-18) continue;
                        double theta = (m[j, j] - m[i, i]) / (2 * m[i, j]);
                        double t = (theta >= 0 ? 1 : -1) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1));
                        double c = 1 / Math.Sqrt(t * t + 1), s = t * c;
                        for (int k = 0; k < n; ++k)
                        {
                            double aki = m[k, i], akj = m[k, j];
                            m[k, i] = c * aki - s * akj; m[k, j] = s * aki + c * akj;
                        }
                        for (int k = 0; k < n; ++k)
                        {
                            double aik = m[i, k], ajk = m[j, k];
                            m[i, k] = c * aik - s * ajk; m[j, k] = s * aik + c * ajk;
                        }
                    }
                }
                var d = new double[n];
                for (int i = 0; i < n; ++i) d[i] = m[i, i];
                return d;
            }

            private double[] Column(int lamp, double[] sub, int axis, double hNorm, out double usedPhysical, out bool ok)
            {
                int k = lamp * LampDofs;
                var full = (double[])scene.Truth.Clone();
                var plus = (double[])sub.Clone(); var minus = (double[])sub.Clone();
                usedPhysical = axis == 3 ? hNorm * LogPowerSpan : hNorm * AxisExtent(scene.Extents, axis);
                for (int shrink = 0; shrink < 6; ++shrink)
                {
                    plus[axis] = sub[axis] + hNorm; minus[axis] = sub[axis] - hNorm;
                    ApplySub(full, plus, k); bool plusOk = scene.Contains(full);
                    ApplySub(full, minus, k); bool minusOk = scene.Contains(full);
                    if (plusOk && minusOk)
                    {
                        ApplySub(full, plus, k); scene.Apply(full); var pf = CaptureTrain();
                        ApplySub(full, minus, k); scene.Apply(full); var mf = CaptureTrain();
                        int channels = E17Width * E17Height * 3;
                        var column = new double[channels * trainViews];
                        double scale = .5 / usedPhysical;
                        for (int v = 0; v < trainViews; ++v)
                        {
                            var p = pf[v]; var m = mf[v];
                            for (int i = 0, c = v * channels; i < p.PixelCount; ++i)
                            {
                                var x = p.Pixel(i); var y = m.Pixel(i);
                                column[c++] = (x.r - y.r) * scale; column[c++] = (x.g - y.g) * scale; column[c++] = (x.b - y.b) * scale;
                            }
                        }
                        ok = true; return column;
                    }
                    hNorm *= .5;
                    usedPhysical = axis == 3 ? hNorm * LogPowerSpan : hNorm * AxisExtent(scene.Extents, axis);
                }
                ok = false; return null;
            }

            private static double AxisExtent(Vector3 e, int axis) => axis == 0 ? e.x : axis == 1 ? e.y : e.z;

            private Frame[] ArmTargets(int lamp, bool shadowsOn)
            {
                scene.Apply(scene.Truth); SetArmShadows(lamp, shadowsOn);
                return CaptureTrain();
            }
            private Frame[] CheckTargets(int lamp, bool shadowsOn)
            {
                scene.Apply(scene.Truth); SetArmShadows(lamp, shadowsOn);
                return CaptureCheck();
            }

            private E17Shadow ShadowStats(int lamp, bool shadowsOn, Frame[] on, Frame[] off, double[] start)
            {
                var s = new E17Shadow { lamp = lamp + 1, energy = new double[trainViews], startEnergy = new double[trainViews], affected = new double[trainViews], maxDiff = new double[trainViews], overlapOn = new double[trainViews], overlapOff = new double[trainViews] };
                var armState = (double[])scene.Truth.Clone();
                ApplySub(armState, start, lamp * LampDofs);
                var renderOn = RenderAt(lamp, true, armState);
                var renderOff = RenderAt(lamp, false, armState);
                for (int v = 0; v < trainViews; ++v)
                {
                    double diffSq = 0, targetEnergy = 0, startSq = 0, startEnergy = 0; int affected = 0, overlappingOn = 0, overlappingOff = 0;
                    var a = on[v]; var b = off[v];
                    for (int p = 0; p < a.PixelCount; ++p)
                    {
                        var x = a.Pixel(p); var y = b.Pixel(p);
                        double d0 = x.r - y.r, d1 = x.g - y.g, d2 = x.b - y.b;
                        double m = Math.Sqrt(d0 * d0 + d1 * d1 + d2 * d2);
                        diffSq += d0 * d0 + d1 * d1 + d2 * d2; targetEnergy += x.r * x.r + x.g * x.g + x.b * x.b;
                        s.maxDiff[v] = Math.Max(s.maxDiff[v], m);
                        var p0 = renderOn[v].Pixel(p); var p1 = renderOff[v].Pixel(p);
                        double e0 = p0.r - p1.r, e1 = p0.g - p1.g, e2 = p0.b - p1.b;
                        startSq += e0 * e0 + e1 * e1 + e2 * e2; startEnergy += p0.r * p0.r + p0.g * p0.g + p0.b * p0.b;
                        if (m > threshold)
                        {
                            ++affected;
                            if (Math.Abs(p0.r - x.r) > threshold || Math.Abs(p0.g - x.g) > threshold || Math.Abs(p0.b - x.b) > threshold) ++overlappingOn;
                            if (Math.Abs(p1.r - y.r) > threshold || Math.Abs(p1.g - y.g) > threshold || Math.Abs(p1.b - y.b) > threshold) ++overlappingOff;
                        }
                    }
                    s.energy[v] = Math.Sqrt(diffSq / Math.Max(1e-12, targetEnergy));
                    s.startEnergy[v] = Math.Sqrt(startSq / Math.Max(1e-12, startEnergy));
                    s.affected[v] = affected / (double)a.PixelCount;
                    s.overlapOn[v] = affected > 0 ? overlappingOn / (double)affected : 0;
                    s.overlapOff[v] = affected > 0 ? overlappingOff / (double)affected : 0;
                }
                s.meanEnergy = s.energy.Average(); s.meanStartEnergy = s.startEnergy.Average();
                s.shadowChange = s.meanEnergy > 1e-12 ? s.meanStartEnergy / s.meanEnergy : 0;
                s.meanAffected = s.affected.Average();
                s.meanOverlapOn = s.overlapOn.Average(); s.meanOverlapOff = s.overlapOff.Average();
                return s;
            }

            private void SetArmShadows(int lamp, bool shadowsOn) { scene.ResetShadows(); scene.SetShadows(lamp, shadowsOn); }

            private Frame[] RenderAt(int lamp, bool shadowsOn, double[] theta)
            {
                scene.Apply(theta); SetArmShadows(lamp, shadowsOn);
                return CaptureTrain();
            }

            private static void ApplySub(double[] full, double[] sub, int k) => Array.Copy(sub, 0, full, k, LampDofs);

            private Frame[] CaptureTrain() => Enumerable.Range(0, trainViews).Select(train.Capture).ToArray();
            private Frame[] CaptureCheck() => Enumerable.Range(0, checkViews).Select(check.Capture).ToArray();

            private static double Sq(double x) => x * x;

            private static string Finite(double x) => double.IsNaN(x) || double.IsInfinity(x) ? "" : F(x);

            private static double Nrmse(Frame[] a, Frame[] b) { var e = new double[a.Length]; for (int i = 0; i < a.Length; ++i) e[i] = NativeRenderBackend.Compare(a[i], b[i]); return Math.Sqrt(e.Average()); }

            private static double RepeatNoise(Frame[] a, Frame[] b)
            {
                double sum = 0; long n = 0;
                for (int v = 0; v < a.Length; ++v)
                    for (int p = 0; p < a[v].PixelCount; ++p)
                    {
                        var x = a[v].Pixel(p); var y = b[v].Pixel(p);
                        sum += Sq(x.r - y.r) + Sq(x.g - y.g) + Sq(x.b - y.b); n += 3;
                    }
                return Math.Sqrt(sum / Math.Max(1, n));
            }

            private void WriteProbeCsv()
            {
                foreach (var p in probes)
                    probeCsv.WriteLine(string.Join(",", p.scene, p.state, p.lamp.ToString(Invariant), p.shadows, p.kind, p.axis.ToString(Invariant),
                        F(p.step), F(p.norm), F(p.activeRatio), F(p.alignment), F(p.consistency), F(p.normRatio)));
            }

            private static void SaveHeatmap(Frame a, Frame b, string path)
            {
                var tex = new Texture2D(E17Width, E17Height, TextureFormat.RGB24, false, false);
                try
                {
                    var px = new Color32[a.PixelCount]; double peak = 0;
                    for (int i = 0; i < a.PixelCount; ++i)
                    {
                        var x = a.Pixel(i); var y = b.Pixel(i);
                        peak = Math.Max(peak, Math.Sqrt(Sq(x.r - y.r) + Sq(x.g - y.g) + Sq(x.b - y.b)));
                    }
                    peak = Math.Max(peak, 1e-6);
                    for (int i = 0; i < a.PixelCount; ++i)
                    {
                        var x = a.Pixel(i); var y = b.Pixel(i);
                        double d = Math.Sqrt(Sq(x.r - y.r) + Sq(x.g - y.g) + Sq(x.b - y.b)) / peak;
                        double s = Math.Sqrt(Math.Min(1, d));
                        px[i] = new Color32((byte)(255 * Math.Min(1, 1.5 * s)), (byte)(255 * Math.Max(0, 1.5 * s - .5)), (byte)(255 * Math.Max(0, s - .45) * 1.8), 255);
                    }
                    tex.SetPixels32(px); tex.Apply();
                    File.WriteAllBytes(path, tex.EncodeToPNG());
                }
                finally { UnityEngine.Object.DestroyImmediate(tex); }
            }
        }
    }
}

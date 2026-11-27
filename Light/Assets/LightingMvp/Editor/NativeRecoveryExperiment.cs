using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.SceneManagement;
using Debug = UnityEngine.Debug;
using Object = UnityEngine.Object;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private const int FitSize = 256, CheckSize = 512;
        private const string PipelinePath = "Assets/Settings/URP-HighFidelity.asset";
        private const string RendererPath = "Assets/Settings/URP-HighFidelity-Renderer.asset";
        private const string ShaderPath = "Assets/LightingMvp/Shaders/StylizedPointResponse.shader";
        private static readonly int[] Seeds = { 17, 31, 47, 73, 109 };
        private static readonly float[] Distances = { .28f, .34f, .41f, .63f, .74f };
        private static readonly CultureInfo Invariant = CultureInfo.InvariantCulture;
        private static readonly BlackBoxOptimizer.Method[] Methods = { BlackBoxOptimizer.Method.CentralLbfgs, BlackBoxOptimizer.Method.SpsaAdam };

        [Serializable] private sealed class ViewInfo { public Vector3 position; public Quaternion rotation; public float fov; }
        [Serializable] private sealed class SanityReport
        {
            public string shader;
            public double repeatMax, positionPlus, positionMinus, intensityPlus, intensityMinus, shadowsOff, lightsOff;
            public float peak, brightPeak;
            public RecoveryLightState targetLight;
            public bool passed;
            public int cameraRenders;
            public double wallSeconds, renderSubmitSeconds, readbackSeconds;
        }
        [Serializable] private sealed class TrialReport
        {
            public string stage, shader, method, stopReason;
            public int seed, dimensions, trainingViews, cameraBudget, trainingCameraRenders, objectiveQueries, iterations;
            public int sharedCameraRenders, validationCameraRenders, totalCameraRenders;
            public double h, initialLoss, bestLoss, recapturedLoss, initialNrmse, bestNrmse, improvement;
            public double highInitialNrmse, highBestNrmse, heldOutInitialNrmse = -1, heldOutBestNrmse = -1;
            public double trainingWallSeconds, renderSubmitSeconds, readbackSeconds, optimizerAndBookkeepingSeconds;
            public double validationWallSeconds, validationRenderSubmitSeconds, validationReadbackSeconds;
            public bool passed;
            public float nominalPositionOffset;
            public double[] initialTheta, bestTheta;
            public RecoveryLightState[] targetRig, initialRig, bestRig;
            public float[] positionOffsets, positionErrors, linearPowerRelativeErrors;
        }
        [Serializable] private sealed class StageReport
        {
            public string stage, shader;
            public ViewInfo[] views;
            public int fitResolution = FitSize, validationResolution = CheckSize;
            public int allCameraRenders;
            public double wallSeconds, renderSubmitSeconds, readbackSeconds, sharedWallSeconds;
            public TrialReport[] trials;
        }

        private sealed class TestPipeline : IDisposable
        {
            private readonly RenderPipelineAsset originalGraphics = GraphicsSettings.defaultRenderPipeline;
            private readonly RenderPipelineAsset originalQuality = QualitySettings.renderPipeline;
            private readonly UniversalRenderPipelineAsset data;
            private readonly ScriptableRendererData renderer;

            internal UniversalRenderPipelineAsset Asset => data;

            internal TestPipeline(bool classroom = false)
            {
                var source = AssetDatabase.LoadAssetAtPath<UniversalRenderPipelineAsset>(PipelinePath);
                var sourceRenderer = AssetDatabase.LoadAssetAtPath<UniversalRendererData>(RendererPath);
                if (!source || !sourceRenderer) throw new FileNotFoundException("The Cornell reference URP assets are missing.");
                data = Object.Instantiate(source);
                renderer = Object.Instantiate(sourceRenderer);
                try
                {
                    data.hideFlags = renderer.hideFlags = HideFlags.HideAndDontSave;
                    data.name = "Black-box Cornell reference URP (no renderer features)";
                    // Stock Cornell renderer, not StarRail's renderer. The source assets stay unchanged.
                    var serializedRenderer = new SerializedObject(renderer);
                    serializedRenderer.FindProperty("m_RendererFeatures").ClearArray();
                    serializedRenderer.FindProperty("m_RendererFeatureMap").ClearArray();
                    if (classroom) serializedRenderer.FindProperty("m_RenderingMode").intValue = (int)RenderingMode.ForwardPlus;
                    serializedRenderer.ApplyModifiedPropertiesWithoutUndo();
                    var serializedPipeline = new SerializedObject(data);
                    var list = serializedPipeline.FindProperty("m_RendererDataList");
                    list.arraySize = 1;
                    list.GetArrayElementAtIndex(0).objectReferenceValue = renderer;
                    serializedPipeline.FindProperty("m_DefaultRendererIndex").intValue = 0;
                    if (classroom)
                    {
                        serializedPipeline.FindProperty("m_AdditionalLightsShadowmapResolution").intValue = 2048;
                        serializedPipeline.FindProperty("m_AdditionalLightsShadowResolutionTierMedium").intValue = 256;
                    }
                    serializedPipeline.ApplyModifiedPropertiesWithoutUndo();
                    if (classroom) { data.name = "Black-box Classroom reference URP Forward+"; data.shadowDistance = 22; }
                    data.msaaSampleCount = 1;
                    data.renderScale = 1;
                    data.hdrColorBufferPrecision = HDRColorBufferPrecision._64Bits;
                    if (!data.supportsAdditionalLightShadows || data.additionalLightsRenderingMode != LightRenderingMode.PerPixel)
                        throw new InvalidOperationException("Cornell reference URP needs per-pixel additional lights and realtime shadows.");
                    GraphicsSettings.defaultRenderPipeline = data;
                    QualitySettings.renderPipeline = data;
                }
                catch { Dispose(); throw; }
            }

            public void Dispose()
            {
                QualitySettings.renderPipeline = originalQuality;
                GraphicsSettings.defaultRenderPipeline = originalGraphics;
                Object.DestroyImmediate(data);
                Object.DestroyImmediate(renderer);
            }
        }

        private sealed class QueryLog : IDisposable
        {
            private readonly StreamWriter writer;
            private readonly Stopwatch wall = Stopwatch.StartNew();
            private readonly int seed;
            private readonly string method;
            internal int Cameras { get; private set; }
            internal double WallSeconds => wall.Elapsed.TotalSeconds;
            internal double RenderSeconds { get; private set; }
            internal double ReadbackSeconds { get; private set; }

            internal QueryLog(string directory, string file, string method, int seed)
            {
                Directory.CreateDirectory(directory);
                writer = new StreamWriter(Path.Combine(directory, file)) { AutoFlush = true };
                writer.WriteLine("seed,method,phase,loss,camera_renders,cumulative_camera_renders,elapsed_s,query_s,render_submit_s,readback_s,theta");
                this.seed = seed;
                this.method = method;
            }

            internal double Measure(NativeRenderBackend backend, string phase, double[] theta, Func<double> query)
            {
                int first = backend.RenderCount;
                double r0 = backend.RenderSeconds, b0 = backend.ReadbackSeconds;
                double start = WallSeconds;
                double loss = query();
                double end = WallSeconds;
                int count = backend.RenderCount - first;
                double render = backend.RenderSeconds - r0, read = backend.ReadbackSeconds - b0;
                Cameras += count; RenderSeconds += render; ReadbackSeconds += read;
                writer.WriteLine(string.Join(",", seed.ToString(Invariant), method, phase, double.IsNaN(loss) ? "" : F(loss),
                    count.ToString(Invariant), Cameras.ToString(Invariant), F(end), F(end - start), F(render), F(read),
                    theta == null ? "" : string.Join(";", theta.Select(F))));
                return loss;
            }

            public void Dispose() => writer.Dispose();
        }

        private sealed class StylizedMaterials : IDisposable
        {
            private readonly Dictionary<Renderer, Material[]> original = new Dictionary<Renderer, Material[]>();
            private readonly Dictionary<Material, Material> clones = new Dictionary<Material, Material>();

            internal StylizedMaterials(GameObject room)
            {
                var shader = AssetDatabase.LoadAssetAtPath<Shader>(ShaderPath);
                if (!shader || !shader.isSupported || ShaderUtil.ShaderHasError(shader)) throw new InvalidOperationException("Stylized Point Response failed to compile.");
                try
                {
                    foreach (var renderer in room.GetComponentsInChildren<MeshRenderer>())
                    {
                        var materials = original[renderer] = renderer.sharedMaterials;
                        renderer.sharedMaterials = materials.Select(Clone).ToArray();
                    }
                }
                catch { Dispose(); throw; }

                Material Clone(Material source)
                {
                    if (clones.TryGetValue(source, out var result)) return result;
                    result = new Material(shader) { name = source.name + " (stylized experiment)", hideFlags = HideFlags.HideAndDontSave };
                    clones.Add(source, result);
                    result.SetColor("_BaseColor", source.GetColor("_BaseColor"));
                    result.SetFloat("_Cull", source.GetFloat("_Cull"));
                    return result;
                }
            }

            public void Dispose()
            {
                foreach (var entry in original) if (entry.Key) entry.Key.sharedMaterials = entry.Value;
                foreach (var material in clones.Values) Object.DestroyImmediate(material);
            }
        }

        [MenuItem("Lighting MVP/Research/Run Cornell RGB E0 + E1 (one seed)")]
        public static void RunQuick() => Run("smoke", DefaultOutput(), 1);

        public static void BatchRun()
        {
            try
            {
                Run(Arg("-bbCase", "smoke"), Arg("-bbOutput", DefaultOutput()), int.Parse(Arg("-bbSeeds", "1"), Invariant));
                EditorApplication.Exit(0);
            }
            catch (Exception e) { Debug.LogException(e); EditorApplication.Exit(2); }
        }

        private static string Arg(string name, string fallback)
        {
            var args = Environment.GetCommandLineArgs();
            int index = Array.IndexOf(args, name);
            return index < 0 ? fallback : args[index + 1];
        }

        private static string DefaultOutput() => Path.Combine(LightingExperimentPaths.OutputRoot, "blackbox_rgb", DateTime.Now.ToString("yyyyMMdd_HHmmss", Invariant));
        private static string F(double v) => v.ToString("G17", Invariant);
        private static void WriteJson<T>(string directory, string file, T obj) => File.WriteAllText(Path.Combine(directory, file), JsonUtility.ToJson(obj, true));

        private static void Run(string test, string output, int seedCount)
        {
            if (test == "e20-mbd-reference")
            {
                RunMomentBoundsReference(output);
                return;
            }
            if (EditorApplication.isPlayingOrWillChangePlaymode) throw new InvalidOperationException("Run the experiment in Edit mode.");
            if (seedCount < 1 || seedCount > Seeds.Length) throw new ArgumentOutOfRangeException(nameof(seedCount));
            if (!GraphicsSettings.lightsUseLinearIntensity) throw new InvalidOperationException("Enable linear light intensity for calibrated color coordinates.");
            test = test.ToLowerInvariant();
            if (test.StartsWith("e23-", StringComparison.Ordinal)) { RunE23Stages(test, output); return; }
            if (test.StartsWith("e22-", StringComparison.Ordinal)) { RunE22Stages(test, output); return; }
            if (test == "e21-a" || test == "e21-a-smoke") { RunE21GradientStages(test, output); return; }
            if (test == "e21-b" || test == "e21-b-smoke" || test == "e21-c" || test == "e21-c-smoke" ||
                test == "e21-d" || test == "e21-d-smoke" || test == "e21-e" || test == "e21-e-smoke")
            { RunE21RoomStages(test, output); return; }
            if (test.StartsWith("e20-mbd-", StringComparison.Ordinal)) { RunMbdStage(test.Substring(8), output); return; }
            if (test.StartsWith("e17-", StringComparison.Ordinal)) { RunSensitivity(test, output); return; }
            if (test.StartsWith("e18-", StringComparison.Ordinal)) { RunShadowStudy(test, output); return; }
            if (test.StartsWith("e19-", StringComparison.Ordinal)) { RunJointRecovery(test, output); return; }
            if (test == "e20-a" || test == "e20-a-smoke" || test == "e20-direct" || test == "e20-direct-smoke" ||
                test == "e20-msm" || test == "e20-msm-smoke")
            { RunIndependentLightStudy(test, output); return; }
            if (new[] { "e13-smoke", "e13", "e13-position", "e14", "e14-smoke", "e14-large", "e15", "e15-smoke", "e15-budget", "e15-block" }.Contains(test))
            { RunClassroomDiagnosis(test, output); return; }
            if (test == "e10" || test == "e10-smoke" || test == "e10-near" || test == "e11" || test == "e11-smoke")
            { RunClassroomScale(test, output); return; }
            if (!new[] { "e0", "smoke", "all", "e1", "e2", "e2-fixed", "e2-lit", "e2-stylized", "e2-stylized-fixed", "e3", "diagnose", "zerograds", "zerograds-stylized", "e4", "e5", "e6", "e7", "e8", "e8-smoke", "e9", "e9-smoke", "e9-probe" }.Contains(test))
                throw new ArgumentException("Unknown black-box case: " + test);
            SelfTest();
            var scene = SceneManager.GetActiveScene();
            if (scene.path != LightingExperimentPaths.CornellScene)
            {
                if (!Application.isBatchMode && !EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
                scene = EditorSceneManager.OpenScene(LightingExperimentPaths.CornellScene);
            }
            var room = GameObject.Find("Cornell Box - 2m open front");
            var camera = GameObject.Find("Validation Camera")?.GetComponent<Camera>();
            var block = GameObject.Find("Cornell Tall Block")?.GetComponent<BoxCollider>();
            if (!room || !camera || !block) throw new InvalidOperationException("Expected the saved two-block Cornell scene and validation camera.");
            Lightmapping.TryGetLightingSettings(out var settings);
            if (settings && settings.realtimeGI) throw new InvalidOperationException("Disable realtime GI before running the deterministic Cornell experiment.");
            Directory.CreateDirectory(output);
            using (new TestPipeline())
            using (var placement = new CornellSceneRig(scene))
            {
                var lights = Enumerable.Range(0, placement.Count).Select(i => placement[i]).ToArray();
                if (lights.Length != 3 || lights.Any(l => l.type != LightType.Point || !l.isActiveAndEnabled || l.shadows == LightShadows.None ||
                    l.lightmapBakeType != LightmapBakeType.Realtime || l.useColorTemperature))
                    throw new InvalidOperationException("The saved Cornell rig needs three enabled realtime shadowed RGB Point Lights.");
                if (scene.GetRootGameObjects().SelectMany(g => g.GetComponentsInChildren<Light>()).Any(l => l.isActiveAndEnabled && !lights.Contains(l)))
                    throw new InvalidOperationException("The reference Cornell scene must not contain other active lights.");
                var gt = lights.Select(RecoveryLightState.Read).ToArray();
                var views = BuildViews(camera);
                if (test == "e7" || test == "e8" || test == "e8-smoke" || test == "e9" || test == "e9-smoke" || test == "e9-probe")
                {
                    RunStructureExperiments(test, output, camera, views, room, lights, block);
                    return;
                }
                if (test == "e4" || test == "e5" || test == "e6")
                {
                    RunNext(test, output, camera, views[0], room, lights, gt, block);
                    return;
                }
                if (test == "e0" || test == "smoke" || test == "all")
                {
                    lights[1].enabled = lights[2].enabled = false;
                    bool valid = Sanity(camera, views[0], lights[0], "URP/Lit", Path.Combine(output, "e0"));
                    lights[1].enabled = lights[2].enabled = true;
                    if (!valid) throw new InvalidOperationException("E0 failed. See renderer diagnostics in the output folder.");
                }
                if (test == "smoke" || test == "all" || test == "e1")
                {
                    lights[1].enabled = lights[2].enabled = false;
                    var report = Fit("e1", "URP/Lit", false, 1, false, 500, 1);
                    lights[1].enabled = lights[2].enabled = true;
                    if (test == "all" && !StagePassed(report, seedCount)) throw new InvalidOperationException("E1 did not meet the staged gate; reports retained.");
                }
                bool failedStage = false;
                if (test == "all" || test == "e2" || test == "e2-fixed" || test == "e2-lit")
                {
                    var fixedLit = Fit("e2_lit_fixed", "URP/Lit", false, 1, false, 2000);
                    if (test != "e2-fixed")
                    {
                        if (StagePassed(fixedLit, seedCount)) Fit("e2_lit_rgb", "URP/Lit", true, 1, false, 2000);
                        else { failedStage = true; Debug.LogWarning("Fixed-color Lit gate not met: skipped Lit full RGB."); }
                    }
                }
                if (test == "all" || test == "e2" || test == "e2-fixed" || test == "e2-stylized" || test == "e2-stylized-fixed")
                {
                    using (new StylizedMaterials(room))
                    {
                        lights[1].enabled = lights[2].enabled = false;
                        bool valid = Sanity(camera, views[0], lights[0], "Stylized Point Response", Path.Combine(output, "e0_stylized"));
                        lights[1].enabled = lights[2].enabled = true;
                        if (!valid) throw new InvalidOperationException("Stylized RGB does not pass the renderer response check.");
                        var fixedToon = Fit("e2_stylized_fixed", "Stylized Point Response", false, 1, false, 2000);
                        if (test != "e2-fixed" && test != "e2-stylized-fixed")
                        {
                            if (StagePassed(fixedToon, seedCount)) Fit("e2_stylized_rgb", "Stylized Point Response", true, 1, false, 2000);
                            else { failedStage = true; Debug.LogWarning("Fixed-color stylized gate not met: skipped stylized full RGB."); }
                        }
                    }
                }
                if (failedStage) throw new InvalidOperationException("Fixed-color recovery gate not met; reports retained.");
                if (test == "all" || test == "e3")
                {
                    Fit("e3_lit_single", "URP/Lit", true, 1, true, 6000);
                    Fit("e3_lit_multi", "URP/Lit", true, 3, true, 6000);
                }

                if (test == "diagnose")
                {
                    // Only the shadow setting changes; the rig, starts, solver seeds and budgets match E1/E2.
                    var shadows = lights.Select(l => l.shadows).ToArray();
                    try
                    {
                        foreach (var light in lights) light.shadows = LightShadows.None;
                        lights[1].enabled = lights[2].enabled = false;
                        Fit("diagnostic_e1_no_shadows", "URP/Lit (native shadows off)", false, 1, false, 500, 1);
                        lights[1].enabled = lights[2].enabled = true;
                        Fit("diagnostic_e2_fixed_no_shadows", "URP/Lit (native shadows off)", false, 1, false, 2000);
                    }
                    finally { for (int i = 0; i < lights.Length; ++i) lights[i].shadows = shadows[i]; }
                }

                if (test == "zerograds" || test == "zerograds-stylized")
                {
                    var proxy = new ZeroGradsTrial(Arg("-bbPython", ""), Arg("-bbZeroGradsAdapter",
                        Path.GetFullPath(Path.Combine(Application.dataPath, "../../mvp/blackbox_zerograds.py"))), Arg("-bbZeroGradsModels", ""));
                    if (test == "zerograds")
                    {
                        lights[1].enabled = lights[2].enabled = false;
                        Fit("zg_e1", "URP/Lit", false, 1, false, 500, 1, proxy);
                        lights[1].enabled = lights[2].enabled = true;
                        Fit("zg_e2_lit_fixed", "URP/Lit", false, 1, false, 2000, proxy: proxy);
                    }
                    else
                    {
                        using (new StylizedMaterials(room))
                            Fit("zg_e2_stylized_rgb", "Stylized Point Response", true, 1, false, 2000, proxy: proxy);
                    }
                }

                StageReport Fit(string stage, string shader, bool color, int viewsUsed, bool validation, int budget, int nLights = 3, ZeroGradsTrial proxy = null) =>
                    Recover(stage, shader, camera, views, lights.Take(nLights).ToArray(), gt.Take(nLights).ToArray(), block,
                        color, viewsUsed, validation, budget, seedCount, output, proxy);
            }
            Debug.Log("Black-box RGB experiment complete: " + output);
        }

        private static ViewInfo[] BuildViews(Camera camera)
        {
            var p = camera.transform.position;
            var aim = new Vector3(0, .98f, .45f);
            var positions = new[] { p, p + new Vector3(-.90f, .10f, .42f), p + new Vector3(.90f, .19f, .42f),
                p + new Vector3(-.40f, -.46f, .54f), p + new Vector3(.52f, .63f, .55f) };
            return positions.Select((eye, i) => new ViewInfo { position = eye, fov = camera.fieldOfView,
                rotation = i == 0 ? camera.transform.rotation : Quaternion.LookRotation(aim - eye, Vector3.up) }).ToArray();
        }

        private static NativeRenderBackend Backend(Camera camera, ViewInfo[] views, int resolution) =>
            new NativeRenderBackend(camera, views.Select(v => v.position).ToArray(), views.Select(v => v.rotation).ToArray(), resolution, 0);

        private static bool Sanity(Camera camera, ViewInfo view, Light light, string shader, string directory)
        {
            using (var backend = Backend(camera, new[] { view }, FitSize))
            using (var log = new QueryLog(directory, "queries.csv", "sanity", 0))
            {
                var baseState = RecoveryLightState.Read(light);
                LightShadows shadows = light.shadows;
                var report = new SanityReport { shader = shader, targetLight = baseState };
                NativeRenderBackend.Frame target = null;
                try
                {
                    log.Measure(backend, "warmup", PhysicalCoordinates(), () => { backend.Capture(0); return double.NaN; });
                    log.Measure(backend, "target", PhysicalCoordinates(), () => { target = backend.Capture(0); return double.NaN; });
                    report.peak = target.Peak;
                    for (int i = 0; i < 5; ++i) report.repeatMax = Math.Max(report.repeatMax, Loss("repeat"));
                    light.transform.position = baseState.position + new Vector3(.25f, .12f, .06f);
                    report.positionPlus = Loss("position_plus");
                    light.transform.position = baseState.position - new Vector3(.25f, .12f, .06f);
                    report.positionMinus = Loss("position_minus");
                    baseState.Apply(light);
                    light.intensity *= 1.35f;
                    report.intensityPlus = Loss("intensity_plus");
                    light.intensity = baseState.intensity / 1.35f;
                    report.intensityMinus = Loss("intensity_minus");
                    baseState.Apply(light);
                    light.shadows = LightShadows.None;
                    report.shadowsOff = Loss("shadows_off");
                    light.shadows = shadows;
                    light.intensity = 0;
                    report.lightsOff = Loss("light_off");
                    light.intensity = baseState.intensity * 24;
                    log.Measure(backend, "hdr_probe", PhysicalCoordinates(), () => { report.brightPeak = backend.Capture(0).Peak; return double.NaN; });
                    baseState.Apply(light);
                    report.repeatMax = Math.Max(report.repeatMax, Loss("repeat_after_changes"));
                    report.cameraRenders = log.Cameras;
                    report.wallSeconds = log.WallSeconds;
                    report.renderSubmitSeconds = log.RenderSeconds;
                    report.readbackSeconds = log.ReadbackSeconds;
                    double effect = new[] { report.positionPlus, report.positionMinus, report.intensityPlus, report.intensityMinus, report.shadowsOff, report.lightsOff }.Min();
                    report.passed = report.peak > .001 && report.brightPeak > 1.01 && effect > Math.Max(1e-6, report.repeatMax * 100);
                    WriteJson(directory, "report.json", report);
                    NativeRenderBackend.Save(target, FitSize, Path.Combine(directory, "target"));
                    Debug.Log($"Black-box E0 ({shader}): pass={report.passed} max repeat={report.repeatMax:G5}, min response={effect:G5}, HDR peak={report.brightPeak:G5}");
                    return report.passed;
                }
                finally { baseState.Apply(light); light.shadows = shadows; }

                double Loss(string phase) => log.Measure(backend, phase, PhysicalCoordinates(), () => backend.Loss(0, target));

                double[] PhysicalCoordinates()
                {
                    var p = light.transform.position;
                    var c = light.color;
                    // E0 vectors are raw [x,y,z,intensity,r,g,b,shadows], not optimizer coordinates.
                    return new[] { (double)p.x, p.y, p.z, light.intensity, c.r, c.g, c.b, (int)light.shadows };
                }
            }
        }

        private static bool StagePassed(StageReport report, int seeds) => Methods.Any(m => report.trials.Count(t => t.method == m.ToString() && t.passed) >= Math.Ceiling(.8 * seeds));

        private static StageReport Recover(string stage, string shader, Camera camera, ViewInfo[] poses, Light[] lights, RecoveryLightState[] gt,
            BoxCollider block, bool fitColor, int viewCount, bool heldOut, int cameraBudget, int seedCount, string output, ZeroGradsTrial proxy)
        {
            var clock = Stopwatch.StartNew();
            string directory = Path.Combine(output, stage);
            for (int i = 0; i < lights.Length; ++i) gt[i].Apply(lights[i]);
            var coords = new RecoveryRigCoordinates(lights, block, fitColor);
            double[] truth = coords.Encode(gt);
            var selectedPoses = heldOut ? poses : poses.Take(viewCount).ToArray();
            int[] trainingViews = Enumerable.Range(0, viewCount).ToArray();
            int[] highViews = heldOut ? trainingViews.Concat(new[] { 3, 4 }).ToArray() : trainingViews;
            var trials = new List<TrialReport>();
            var report = new StageReport { stage = stage, shader = shader, views = selectedPoses };
            using (var training = Backend(camera, selectedPoses, FitSize))
            using (var validation = Backend(camera, selectedPoses, CheckSize))
            using (var setup = new QueryLog(directory, "shared_queries.csv", "shared", 0))
            {
                var target = new NativeRenderBackend.Frame[selectedPoses.Length];
                var targetHigh = new NativeRenderBackend.Frame[selectedPoses.Length];
                foreach (int view in trainingViews)
                {
                    setup.Measure(training, "warmup_v" + view, truth, () => { training.Capture(view); return double.NaN; });
                    setup.Measure(training, "target_v" + view, truth, () => { target[view] = training.Capture(view); return double.NaN; });
                    NativeRenderBackend.Save(target[view], FitSize, Path.Combine(directory, "target_v" + view));
                }
                foreach (int view in highViews)
                {
                    setup.Measure(validation, "warmup_512_v" + view, truth, () => { validation.Capture(view); return double.NaN; });
                    setup.Measure(validation, "target_512_v" + view, truth, () => { targetHigh[view] = validation.Capture(view); return double.NaN; });
                    NativeRenderBackend.Save(targetHigh[view], CheckSize, Path.Combine(directory, "target_512_v" + view));
                }
                coords.Apply(truth);
                double representationError = setup.Measure(training, "coordinate_roundtrip", truth, () => trainingViews.Average(v => training.Loss(v, target[v])));
                if (representationError > 1e-5) throw new InvalidOperationException("GT light reparameterization unexpectedly changes RGB: " + representationError);
                report.sharedWallSeconds = setup.WallSeconds;
                for (int k = 0; k < seedCount; ++k)
                {
                    int seed = Seeds[k];
                    double[] initial = coords.Encode(coords.Perturb(gt, seed, Distances[k]), true);
                    foreach (string method in proxy == null ? Methods.Select(m => m.ToString()) : new[] { "ZeroGrads" })
                    {
                        string dir = Path.Combine(directory, seed + "_" + method);
                        using (var log = new QueryLog(dir, "queries.csv", method, seed))
                        {
                            double Objective(double[] theta, string phase) => log.Measure(training, phase, theta, () =>
                            {
                                coords.Apply(theta);
                                return trainingViews.Average(v => training.Loss(v, target[v]));
                            });
                            int budget = cameraBudget / viewCount;
                            var result = proxy != null ? proxy.Run(initial, Objective, budget, seed) :
                                BlackBoxOptimizer.Run((BlackBoxOptimizer.Method)Enum.Parse(typeof(BlackBoxOptimizer.Method), method), initial, Objective, budget, seed);
                            var row = new TrialReport
                            {
                                stage = stage, shader = shader, method = method, seed = seed, dimensions = initial.Length,
                                trainingViews = viewCount, cameraBudget = cameraBudget, trainingCameraRenders = log.Cameras,
                                objectiveQueries = result.Queries, iterations = result.Iterations, stopReason = result.StopReason,
                                initialLoss = result.InitialLoss, bestLoss = result.BestLoss, h = result.PilotH,
                                initialNrmse = Math.Sqrt(result.InitialLoss), bestNrmse = Math.Sqrt(result.BestLoss),
                                improvement = 1 - result.BestLoss / Math.Max(1e-20, result.InitialLoss),
                                initialTheta = initial, bestTheta = result.Best, targetRig = gt,
                                initialRig = coords.Decode(initial), bestRig = coords.Decode(result.Best), nominalPositionOffset = Distances[k],
                                trainingWallSeconds = log.WallSeconds, renderSubmitSeconds = log.RenderSeconds, readbackSeconds = log.ReadbackSeconds,
                                sharedCameraRenders = setup.Cameras
                            };
                            row.passed = row.improvement >= .9 && row.bestNrmse <= .05;
                            row.optimizerAndBookkeepingSeconds = Math.Max(0, row.trainingWallSeconds - row.renderSubmitSeconds - row.readbackSeconds);
                            row.positionOffsets = row.initialRig.Select((l, j) => Vector3.Distance(l.position, gt[j].position)).ToArray();
                            row.positionErrors = row.bestRig.Select((l, j) => Vector3.Distance(l.position, gt[j].position)).ToArray();
                            row.linearPowerRelativeErrors = row.bestRig.Select((l, j) => (Power(l) - Power(gt[j])).magnitude / Power(gt[j]).magnitude).ToArray();
                            int before = log.Cameras;
                            double vr = log.RenderSeconds, vb = log.ReadbackSeconds, vt = log.WallSeconds;
                            Images(training, target, trainingViews, initial, "initial", dir, log);
                            row.recapturedLoss = Images(training, target, trainingViews, result.Best, "best", dir, log).Average();
                            double[] iHigh = Images(validation, targetHigh, highViews, initial, "initial_512", dir, log);
                            double[] bHigh = Images(validation, targetHigh, highViews, result.Best, "best_512", dir, log);
                            row.highInitialNrmse = Math.Sqrt(iHigh.Take(viewCount).Average());
                            row.highBestNrmse = Math.Sqrt(bHigh.Take(viewCount).Average());
                            if (heldOut)
                            {
                                row.heldOutInitialNrmse = Math.Sqrt(iHigh.Skip(viewCount).Average());
                                row.heldOutBestNrmse = Math.Sqrt(bHigh.Skip(viewCount).Average());
                            }
                            row.validationCameraRenders = log.Cameras - before;
                            row.totalCameraRenders = log.Cameras + setup.Cameras;
                            row.validationWallSeconds = log.WallSeconds - vt;
                            row.validationRenderSubmitSeconds = log.RenderSeconds - vr;
                            row.validationReadbackSeconds = log.ReadbackSeconds - vb;
                            WriteJson(dir, "report.json", row);
                            trials.Add(row);
                            report.trials = trials.ToArray();
                            report.allCameraRenders = training.RenderCount + validation.RenderCount;
                            report.wallSeconds = clock.Elapsed.TotalSeconds;
                            report.renderSubmitSeconds = training.RenderSeconds + validation.RenderSeconds;
                            report.readbackSeconds = training.ReadbackSeconds + validation.ReadbackSeconds;
                            WriteJson(directory, "report.json", report);
                            Debug.Log($"{stage} {seed} {method}: {row.initialNrmse:P2} -> {row.bestNrmse:P2} ({row.highBestNrmse:P2} at 512), {row.trainingCameraRenders} renders/{row.trainingWallSeconds:F2}s; pass={row.passed}");
                        }
                    }
                }
            }
            return report;

            double[] Images(NativeRenderBackend backend, NativeRenderBackend.Frame[] references, int[] views, double[] theta, string name, string dir, QueryLog log)
            {
                var loss = new double[views.Length];
                coords.Apply(theta);
                for (int i = 0; i < views.Length; ++i)
                {
                    int view = views[i];
                    NativeRenderBackend.Frame frame = null;
                    loss[i] = log.Measure(backend, "validate_" + name + "_v" + view, theta, () =>
                    {
                        frame = backend.Capture(view);
                        return NativeRenderBackend.Compare(frame, references[view]);
                    });
                    NativeRenderBackend.Save(frame, backend.Size, Path.Combine(dir, name + "_v" + view));
                }
                return loss;
            }
        }

        private static Vector3 Power(RecoveryLightState s) => new Vector3(s.color.linear.r, s.color.linear.g, s.color.linear.b) * s.intensity;

        private static void SelfTest()
        {
            foreach (var method in Methods)
            {
                int calls = 0;
                double best = double.PositiveInfinity;
                var x = new[] { 0.0, .5, 1.0, .9 };
                var target = new[] { .25, .5, .62, .37 };
                var result = BlackBoxOptimizer.Run(method, x, (v, _) =>
                {
                    calls++;
                    if (v.Any(t => !(t >= 0 && t <= 1))) throw new InvalidOperationException("Optimizer emitted an infeasible probe.");
                    double loss = v.Select((t, i) => (i + 1) * Math.Pow(t - target[i], 2)).Sum();
                    best = Math.Min(best, loss);
                    return loss;
                }, 500, 17);
                if (calls != result.Queries || calls > 500 || result.BestLoss != best || best > 1e-3 || x[0] != 0 || x[2] != 1)
                    throw new InvalidOperationException("Black-box numerical self-test failed for " + method + ": " + best);
            }
        }
    }
}

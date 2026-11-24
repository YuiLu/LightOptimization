using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.SceneManagement;

namespace LightingMvp.Editor
{
    // Editor workflow shared by the production window and the native integration test.
    internal sealed class BlackBoxLightingSession
    {
        internal readonly struct LightState
        {
            public readonly Vector3 Position;
            public readonly float Intensity;
            public readonly Color Color;
            public readonly bool Enabled;
            private readonly Vector3? LocalPosition;

            public LightState(Vector3 position, float intensity, Color color, bool enabled)
            { Position = position; Intensity = intensity; Color = color; Enabled = enabled; LocalPosition = null; }
            public LightState(Light light) : this(light.transform.position, light.intensity, light.color, light.enabled) { LocalPosition = light.transform.localPosition; }

            public void Write(Light light, bool restoreEnabled = false)
            {
                if (!light) return;
                if (restoreEnabled && LocalPosition.HasValue)
                {
                    if (!light.transform.localPosition.Equals(LocalPosition.Value)) light.transform.localPosition = LocalPosition.Value;
                }
                else if (!light.transform.position.Equals(Position)) light.transform.position = Position;
                if (light.intensity != Intensity) light.intensity = Intensity;
                if (!light.color.Equals(Color)) light.color = Color;
                if (restoreEnabled && light.enabled != Enabled) light.enabled = Enabled;
            }
        }

        internal sealed class Image
        {
            public readonly NativeRenderBackend.Frame Frame;
            public readonly int Width, Height;
            public Image(NativeRenderBackend backend)
            { Width = backend.Width; Height = backend.Height; Frame = backend.Capture(0); }

            public Texture2D Preview(bool readable = false)
            {
                var texture = new Texture2D(Width, Height, TextureFormat.RGB24, false, false)
                    { hideFlags = HideFlags.HideAndDontSave };
                var rgb = new Color32[Frame.PixelCount];
                for (int i = 0; i < rgb.Length; ++i)
                {
                    Color c = Frame.Pixel(i);
                    rgb[i] = new Color(Display(c.r), Display(c.g), Display(c.b), 1);
                }
                texture.SetPixels32(rgb);
                texture.Apply(false, !readable);
                return texture;
            }

            private static float Display(float linear) => Mathf.LinearToGammaSpace(1 - Mathf.Exp(-Mathf.Max(0, linear)));
        }

        internal sealed class Keyframe
        {
            public readonly LightState[] Lights;
            public readonly int Query;
            public readonly double Loss;
            public Keyframe(LightState[] lights, int query, double loss) { Lights = lights; Query = query; Loss = loss; }
        }

        internal sealed class Fit
        {
            public BlackBoxOptimizer.Result Search;
            public Image[] Initial, Best;
            public double RecapturedLoss, Seconds;
            public int TrainingRenders, CheckRenders;
            public string[] UnobservedLights;
            public readonly List<Keyframe> History = new List<Keyframe>();
            public LightState[] BestLights => History[History.Count - 1].Lights;
        }

        private sealed class Coordinates
        {
            private readonly LightState[] start;
            private readonly float extent;
            private readonly double logRange;
            private readonly bool fitColor;
            private int Stride => fitColor ? 6 : 4;

            public Coordinates(LightState[] start, float extent, float intensityFactor, bool fitColor)
            {
                if (!(extent > 0 && extent <= 1e6f) || !(intensityFactor > 1 && intensityFactor <= 1000))
                    throw new ArgumentException("Choose a positive position extent and an intensity factor between 1 and 1000.");
                this.start = start; this.extent = extent; logRange = Math.Log(intensityFactor); this.fitColor = fitColor;
                if (fitColor && start.Any(s => !ValidColor(s.Color)))
                    throw new InvalidOperationException("Fit color requires nonblack RGB in [0, 1]. Disable color fitting for other light colors.");
            }

            public double[] Initial()
            {
                var theta = Enumerable.Repeat(.5, start.Length * Stride).ToArray();
                for (int i = 0; fitColor && i < start.Length; ++i)
                {
                    Color.RGBToHSV(start[i].Color, out _, out float saturation, out _);
                    theta[i * Stride + 5] = saturation;
                }
                return theta;
            }

            public LightState[] Decode(double[] theta)
            {
                var rig = new LightState[start.Length];
                for (int i = 0; i < start.Length; ++i)
                {
                    int k = i * Stride;
                    var s = start[i];
                    var offset = new Vector3((float)(theta[k] * 2 - 1), (float)(theta[k + 1] * 2 - 1), (float)(theta[k + 2] * 2 - 1));
                    var color = s.Color;
                    if (fitColor)
                    {
                        Color.RGBToHSV(color, out float hue, out float saturation, out float value);
                        if (theta[k + 4] != .5 || theta[k + 5] != saturation)
                            color = WithChromaticity(color, hue + (float)(theta[k + 4] - .5), (float)theta[k + 5], value);
                    }
                    rig[i] = new LightState(s.Position + offset * extent,
                        (float)(s.Intensity * Math.Exp((theta[k + 3] * 2 - 1) * logRange)), color, s.Enabled);
                }
                return rig;
            }
        }

        private sealed class CaptureSet : IDisposable
        {
            private readonly NativeRenderBackend[] backends;
            private string renderError;
            public int Count => backends.Sum(item => item.RenderCount);

            public CaptureSet(Camera[] views, int longEdge)
            {
                backends = new NativeRenderBackend[views.Length];
                try
                {
                    for (int i = 0; i < views.Length; ++i)
                    {
                        float aspect = views[i].aspect;
                        if (!(aspect > .05f && aspect < 20)) throw new InvalidOperationException("Unsupported camera aspect ratio.");
                        int width = aspect >= 1 ? longEdge : Mathf.RoundToInt(longEdge * aspect);
                        int height = aspect <= 1 ? longEdge : Mathf.RoundToInt(longEdge / aspect);
                        backends[i] = new NativeRenderBackend(views[i], width, height);
                    }
                    Application.logMessageReceived += OnLog;
                }
                catch { Dispose(); throw; }
            }

            private void OnLog(string message, string stack, LogType type)
            {
                if (renderError == null && (type == LogType.Error || type == LogType.Exception || type == LogType.Assert)) renderError = message;
            }

            public Image[] Read() => Enumerable.Range(0, backends.Length).Select(Read).ToArray();
            public Image Read(int view)
            {
                var image = new Image(backends[view]);
                CheckError();
                return image;
            }

            public double Loss(Image[] target)
            {
                double j = 0;
                for (int i = 0; i < backends.Length; ++i) j += backends[i].Loss(0, target[i].Frame);
                CheckError();
                return j / backends.Length;
            }

            private void CheckError()
            {
                if (renderError != null) throw new InvalidOperationException("The active Unity renderer reported an error: " + renderError);
            }

            public void Dispose()
            {
                Application.logMessageReceived -= OnLog;
                foreach (var backend in backends) backend?.Dispose();
            }
        }

        private const double RepeatTolerance = 1e-8; // 0.01% relative RGB RMSE
        private readonly Camera[] views;
        private readonly Light[] lights;
        private readonly string[] cameraSignatures;
        private readonly RenderPipelineAsset pipeline;
        private readonly int sceneHandle, resolution;
        public Image[] Target { get; }
        public int TargetRenders { get; }
        public int ViewCount => views.Length;
        public int LightCount => lights.Length;
        public string ViewName(int id) => views[id] ? views[id].name : "Missing camera";
        public string LightName(int id) => lights[id] ? lights[id].name : "Missing light";

        public BlackBoxLightingSession(Camera[] views, Light[] lights, int resolution)
        {
            ValidateSetup(views, lights);
            if (resolution < 64 || resolution > 1024) throw new ArgumentOutOfRangeException(nameof(resolution));
            this.views = (Camera[])views.Clone(); this.lights = (Light[])lights.Clone(); this.resolution = resolution;
            sceneHandle = SceneManager.GetActiveScene().handle;
            pipeline = GraphicsSettings.currentRenderPipeline;
            cameraSignatures = this.views.Select(CameraSignature).ToArray();
            using (var capture = new CaptureSet(this.views, resolution))
            {
                Target = capture.Read();
                if (Target.Any(t => t.Frame.Energy <= 1e-10)) throw new InvalidOperationException("A target view is black. Check its camera framing and lighting.");
                RequireRepeatable(capture.Read(), Target);
                TargetRenders = capture.Count;
            }
        }

        public Fit Solve(BlackBoxOptimizer.Method method, bool fitColor, float extent, float intensityFactor,
            int renderBudget, int seed, Action<int, int, string, double> beforeQuery = null)
        {
            CheckSetup();
            var clock = Stopwatch.StartNew();
            var baseline = ReadRig();
            var coordinates = new Coordinates(baseline, extent, intensityFactor, fitColor);
            var initial = coordinates.Initial();
            int queries = renderBudget / ViewCount;
            int minimum = method == BlackBoxOptimizer.Method.CentralLbfgs ? 8 + 2 * initial.Length : 10;
            if (queries < minimum) throw new ArgumentException($"Set a budget of at least {minimum * ViewCount} camera renders for this rig and method.");
            var fit = new Fit();
            using (var capture = new CaptureSet(views, resolution))
            {
                try
                {
                    beforeQuery?.Invoke(0, renderBudget, "Checking repeatability and light response", double.PositiveInfinity);
                    fit.Initial = capture.Read();
                    double measuredInitial = Compare(fit.Initial, Target);
                    var unobserved = new List<string>();
                    for (int i = 0; i < lights.Length; ++i)
                    {
                        lights[i].intensity = baseline[i].Intensity * .5f;
                        if (capture.Loss(fit.Initial) < RepeatTolerance * 10) unobserved.Add(lights[i].name);
                        baseline[i].Write(lights[i]);
                    }
                    if (unobserved.Count == lights.Length)
                        throw new InvalidOperationException("None of the selected lights measurably affect these views. Check camera framing, light layers/range, and the scene shader's Point Light support.");
                    fit.UnobservedLights = unobserved.ToArray();
                    RequireRepeatable(capture.Read(), fit.Initial);
                    int setup = capture.Count;
                    double best = double.PositiveInfinity;
                    fit.Search = BlackBoxOptimizer.Run(method, initial, (theta, phase) =>
                    {
                        int used = capture.Count - setup;
                        beforeQuery?.Invoke(used, renderBudget, phase, best);
                        var rig = coordinates.Decode(theta);
                        WriteRig(rig);
                        double loss = capture.Loss(Target);
                        if (phase == "initial" && !SameLoss(loss, measuredInitial))
                            throw new InvalidOperationException("Initial RGB loss drifted. Disable temporal effects or fix scene time before fitting.");
                        if (loss < best)
                        {
                            best = loss;
                            fit.History.Add(new Keyframe(rig, used / ViewCount + 1, loss));
                        }
                        return loss;
                    }, queries, seed);
                    fit.TrainingRenders = capture.Count - setup;
                    beforeQuery?.Invoke(fit.TrainingRenders, renderBudget, "Re-capturing best rig", best);
                    WriteRig(fit.BestLights);
                    fit.Best = capture.Read();
                    fit.RecapturedLoss = Compare(fit.Best, Target);
                    RequireRepeatable(capture.Read(), fit.Best);
                    if (!SameLoss(fit.RecapturedLoss, fit.Search.BestLoss))
                        throw new InvalidOperationException("The best queried RGB loss did not reproduce. Disable history-dependent effects or fix scene time and try again.");
                    fit.CheckRenders = capture.Count - fit.TrainingRenders;
                }
                finally { WriteRig(baseline, true); }
            }
            fit.Seconds = clock.Elapsed.TotalSeconds;
            return fit;
        }

        public Image Replay(Keyframe keyframe, int view, int isolatedLight)
        {
            CheckSetup();
            var baseline = ReadRig();
            // Size only this view. Offscreen replay never leaves the recorded state in the scene.
            using (var capture = new CaptureSet(new[] { views[view] }, resolution))
            {
                try
                {
                    WriteRig(keyframe.Lights);
                    if (isolatedLight >= 0)
                        for (int i = 0; i < lights.Length; ++i) if (i != isolatedLight) lights[i].enabled = false;
                    return capture.Read(0);
                }
                finally { WriteRig(baseline, true); }
            }
        }

        public void Perturb(float distance, bool changeColor, int seed)
        {
            CheckSetup();
            if (!(distance > 0 && distance < 1e6f)) throw new ArgumentOutOfRangeException(nameof(distance));
            var rng = new System.Random(seed);
            var rig = ReadRig();
            for (int i = 0; i < rig.Length; ++i)
            {
                Vector3 v;
                do { v = new Vector3(Signed(rng), Signed(rng), Signed(rng)); }
                while (v.sqrMagnitude < .01f || v.sqrMagnitude > 1);
                float logScale = (.36f + .55f * (float)rng.NextDouble()) * (rng.Next(2) == 0 ? -1 : 1);
                var s = rig[i];
                Color color = s.Color;
                if (changeColor && ValidColor(color))
                {
                    Color.RGBToHSV(color, out float hue, out float saturation, out float value);
                    color = WithChromaticity(color, hue + .2f + .2f * Signed(rng), saturation > .4f ? saturation * .45f : .65f, value);
                }
                rig[i] = new LightState(s.Position + distance * v.normalized, s.Intensity * Mathf.Exp(logScale), color, s.Enabled);
            }
            Apply(rig, "Perturb lighting rig");
        }

        public void Apply(LightState[] rig, string label = "Apply RGB optimized rig")
        {
            CheckSetup();
            Undo.IncrementCurrentGroup();
            int group = Undo.GetCurrentGroup();
            Undo.SetCurrentGroupName(label);
            var objects = lights.SelectMany(l => new UnityEngine.Object[] { l, l.transform }).ToArray();
            Undo.RecordObjects(objects, label);
            WriteRig(rig);
            foreach (var obj in objects) PrefabUtility.RecordPrefabInstancePropertyModifications(obj);
            Undo.CollapseUndoOperations(group);
            SceneView.RepaintAll();
        }

        private LightState[] ReadRig() => lights.Select(l => new LightState(l)).ToArray();
        private void WriteRig(LightState[] rig, bool restoreEnabled = false)
        {
            for (int i = 0; i < lights.Length; ++i) rig[i].Write(lights[i], restoreEnabled);
        }

        private void CheckSetup()
        {
            ValidateSetup(views, lights);
            if (sceneHandle != SceneManager.GetActiveScene().handle || pipeline != GraphicsSettings.currentRenderPipeline ||
                !views.Select(CameraSignature).SequenceEqual(cameraSignatures))
                throw new InvalidOperationException("The scene, pipeline, or camera changed. Capture the target again.");
        }

        private static void ValidateSetup(Camera[] views, Light[] lights)
        {
            if (Resources.FindObjectsOfTypeAll<LightingDebugWindow>().Any(window => window.HasScenePreview))
                throw new InvalidOperationException("Close the legacy field/replay window or turn off its scene preview before capturing RGB.");
            if (views.Length == 0 || views.Distinct().Count() != views.Length || lights.Length == 0 || lights.Distinct().Count() != lights.Length)
                throw new ArgumentException("Select at least one camera and one unique realtime Point Light; remove duplicate entries.");
            foreach (Camera c in views)
            {
                if (!c || !c.gameObject.scene.IsValid() || !c.gameObject.scene.isLoaded || !c.gameObject.activeInHierarchy || c.cameraType != CameraType.Game || !c.allowHDR ||
                    c.rect != new Rect(0, 0, 1, 1) || c.stereoEnabled || c.allowDynamicResolution)
                    throw new InvalidOperationException($"Use an active HDR Game camera with a full viewport, no XR, and dynamic resolution disabled ({(c ? c.name + $": {c.cameraType}, HDR={c.allowHDR}, viewport={c.rect}, stereo={c.stereoEnabled}, dynamic={c.allowDynamicResolution}, scene loaded={c.gameObject.scene.isLoaded}" : "missing camera")}).");
                if (c.TryGetComponent(out UniversalAdditionalCameraData data))
                {
                    if (data.renderType != CameraRenderType.Base || (data.cameraStack != null && data.cameraStack.Count > 0))
                        throw new InvalidOperationException("This MVP supports standalone Base cameras, not camera stacks.");
                    if (data.antialiasing == AntialiasingMode.TemporalAntiAliasing)
                        throw new InvalidOperationException("Disable temporal AA on the capture camera.");
                }
            }
            foreach (Light l in lights)
            {
                if (!l || !l.gameObject.scene.IsValid() || !l.gameObject.scene.isLoaded || !l.isActiveAndEnabled ||
                    l.type != LightType.Point || l.lightmapBakeType != LightmapBakeType.Realtime || !(l.intensity >= 1e-6f && l.intensity <= 1e14f))
                    throw new InvalidOperationException("Select enabled realtime Point Lights with finite positive intensity.");
                if (l.GetComponentInChildren<Camera>(true) || l.GetComponentInChildren<Renderer>(true) || l.GetComponentsInChildren<Light>(true).Length != 1)
                    throw new InvalidOperationException($"'{l.name}' must be a standalone light transform; moving it must not move geometry, a camera, or another light.");
            }
        }

        private static string CameraSignature(Camera c) => EditorJsonUtility.ToJson(c) + "|" + c.aspect.ToString("R") + "|" +
            c.projectionMatrix.ToString("F8") + "|" + c.worldToCameraMatrix.ToString("F8") + "|" +
            (c.TryGetComponent(out UniversalAdditionalCameraData data) ? EditorJsonUtility.ToJson(data) : "");
        private static double Compare(Image[] a, Image[] b) => a.Select((image, i) => NativeRenderBackend.Compare(image.Frame, b[i].Frame)).Average();
        private static void RequireRepeatable(Image[] observed, Image[] reference)
        {
            double max = observed.Select((image, i) => NativeRenderBackend.Compare(image.Frame, reference[i].Frame)).Max();
            if (!(max <= RepeatTolerance))
                throw new InvalidOperationException($"Same-rig RGB is not repeatable (worst view: {Math.Sqrt(max):P3} NRMSE). Disable TAA/temporal effects (including custom renderer features) or freeze animated scene state.");
        }
        private static bool SameLoss(double a, double b) => Math.Abs(a - b) <= Math.Max(1e-8, Math.Max(a, b) * .002);
        private static float Signed(System.Random rng) => (float)(rng.NextDouble() * 2 - 1);
        private static bool ValidColor(Color color) => color.r >= 0 && color.r <= 1 && color.g >= 0 && color.g <= 1 &&
            color.b >= 0 && color.b <= 1 && color.maxColorComponent > 0;
        private static Color WithChromaticity(Color original, float hue, float saturation, float value)
        {
            var color = Color.HSVToRGB(Mathf.Repeat(hue, 1), saturation, value);
            color.a = original.a;
            return color;
        }
    }
}

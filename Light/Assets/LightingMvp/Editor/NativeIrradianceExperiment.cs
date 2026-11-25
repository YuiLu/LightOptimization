using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.SceneManagement;

namespace LightingMvp.Editor
{
    public static class NativeIrradianceExperiment
    {
        private const int Width = 800, Height = 600;
        private const float ProxyAttenuation = 1e-4f;
        private static readonly Vector3[] PositionJitter = { new Vector3(.47f, -.34f, .28f), new Vector3(-.26f, -.23f, -.61f), new Vector3(-.43f, .36f, .17f) };
        private static readonly float[] IntensityJitter = { .43f, 1.92f, .57f };
        private static readonly Vector3[] ColorJitter = { new Vector3(.35f, .98f, 1.65f), new Vector3(1.7f, .46f, .71f), new Vector3(.65f, 1.65f, .38f) };
        internal static string OutputRoot => Path.GetFullPath(Path.Combine(Application.dataPath, "../../results/unity_mvp/native_irradiance"));

        [Serializable] private sealed class Pose
        {
            public Vector3 position;
            public Quaternion rotation;
            public Matrix4x4 view, projection;
            public int width, height, coveredPixels, litPixels, matchedSamples;
        }
        [Serializable] private sealed class Fit
        {
            public string name;
            public float proxyInitialNrmse, proxyFittedNrmse, proxyAfterRgbClampNrmse, recapturedNrmse;
            public float[] recapturedPixelNrmseByView;
            public int clampedChannels;
            public float visibleLightColorMappingNrmse;
            public LightSpec[] fittedUnityLights;
        }
        [Serializable] private sealed class Report
        {
            public string scene, pipeline, unityVersion, graphicsDevice;
            public string[] lightNames, lightShadows;
            public LightSpec[] gtUnityLights, initialUnityLights;
            public int sampleCount, matchedSamples;
            public float sampleCoverage, areaCoverage, unityInitialNrmse, nativeGtRepeatNrmse, nativeShadowSignalNrmse;
            public float proxyAtGtWithoutVisibilityNrmse, proxyAtGtWithVisibilityNrmse, unshadowedProxyAtGtVsUnshadowedUnityNrmse;
            public float gtLightColorMappingNrmse, initialLightColorMappingNrmse, pngWhitePoint;
            public float[] unityInitialPixelNrmseByView;
            public Pose[] cameras;
            public Fit[] fits;
        }

        public static void RunCornell()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode) throw new InvalidOperationException("Run the native experiment in Edit mode.");
            LightingDebugWindow.EndCornellPreviews();
            if (SceneManager.GetActiveScene().path != CornellBoxMvpEditor.ScenePath)
            {
                if (!Application.isBatchMode && !EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
                EditorSceneManager.OpenScene(CornellBoxMvpEditor.ScenePath);
            }
            Run(SceneManager.GetActiveScene());
        }

        public static void BatchRun()
        {
            try { RunCornell(); EditorApplication.Exit(0); }
            catch (Exception e) { Debug.LogException(e); EditorApplication.Exit(2); }
        }

        private static void Run(Scene scene)
        {
            if (Enumerable.Range(0, SceneManager.sceneCount).Count(i => SceneManager.GetSceneAt(i).isLoaded) != 1)
                throw new InvalidOperationException("Open Cornell Box by itself before running the experiment.");
            var objects = scene.GetRootGameObjects();
            var lights = objects.SelectMany(o => o.GetComponentsInChildren<Light>(true))
                .Where(l => l.name == "Candidate Light" || l.name == "Candidate Light 2" || l.name == "Candidate Light 3")
                .OrderBy(l => l.name, StringComparer.Ordinal).ToArray();
            if (lights.Length != 3 || lights.Select(l => l.name).Distinct().Count() != 3 ||
                lights.Any(l => l.type != LightType.Point || !l.isActiveAndEnabled || l.lightmapBakeType != LightmapBakeType.Realtime || l.useColorTemperature || l.intensity <= 0))
                throw new InvalidOperationException("Expected the three saved active realtime Cornell point lights with RGB color and positive intensity. Stop replay/solo first.");
            var otherLights = objects.SelectMany(o => o.GetComponentsInChildren<Light>(false)).Where(l => l.enabled && !lights.Contains(l)).ToArray();
            var source = objects.SelectMany(o => o.GetComponentsInChildren<Camera>(false)).Single(c => c.name == "Validation Camera");
            var room = objects.Single(o => o.name == "Cornell Box - 2m open front");
            var allSamples = SurfaceSampleSet.FromRenderers(room.GetComponentsInChildren<MeshRenderer>(), 18);
            var gtLights = lights.Select(ReadLight).ToArray();
            var initialLights = Perturb(gtLights);
            var shadows = lights.Select(l => l.shadows).ToArray();
            var report = new Report
            {
                scene = scene.path, pipeline = GraphicsSettings.currentRenderPipeline.name,
                unityVersion = Application.unityVersion, graphicsDevice = SystemInfo.graphicsDeviceType.ToString(),
                lightNames = lights.Select(l => l.name).ToArray(), lightShadows = shadows.Select(s => s.ToString()).ToArray(),
                gtUnityLights = gtLights, initialUnityLights = initialLights, sampleCount = allSamples.Count
            };
            Directory.CreateDirectory(OutputRoot);
            var captures = new List<NativeIrradianceCapture>();
            try
            {
                foreach (var light in otherLights) light.enabled = false;
                captures.Add(new NativeIrradianceCapture(source, Width, Height));
                // Additional views through the opening improve coverage of floors and occlusion boundaries.
                AddView(new Vector3(-.72f, 1.46f, -2.65f), new Vector3(.12f, .78f, .22f));
                AddView(new Vector3(.73f, 1.47f, -2.65f), new Vector3(-.12f, .78f, .22f));

                var gt = ReadFrames(captures, "gt");
                report.pngWhitePoint = gt[0].WhitePoint;
                if (gt.Any(f => f.Covered == 0 || f.Lit == 0)) throw new InvalidOperationException("GT capture has no covered, lit receivers.");
                var projections = new List<IrradianceSampleProjection>();
                var sampleIds = new List<int>();
                report.cameras = captures.Select((cap, index) => Describe(cap.ProjectionCamera, gt[index])).ToArray();
                for (int i = 0; i < allSamples.Count; i++)
                {
                    float best = 0;
                    IrradianceSampleProjection match = default;
                    for (int v = 0; v < captures.Count; v++)
                    {
                        if (IrradianceSampleProjection.TryMatch(v, captures[v].ProjectionCamera, gt[v], allSamples.positions[i], allSamples.normals[i], out var candidate, out float score) && score > best)
                        { match = candidate; best = score; }
                    }
                    if (best == 0) continue;
                    projections.Add(match);
                    sampleIds.Add(i);
                    report.cameras[match.CameraIndex].matchedSamples++;
                }
                var samples = Subset(allSamples, sampleIds);
                report.matchedSamples = samples.Count;
                report.sampleCoverage = samples.Count / (float)allSamples.Count;
                report.areaCoverage = samples.areas.Sum() / allSamples.areas.Sum();
                if (report.sampleCoverage < .20f || report.areaCoverage < .20f) throw new InvalidOperationException("Too few surface samples pass capture depth registration.");
                Vector3[] target = Project(gt, gt, projections);
                if (!target.Any(v => v.sqrMagnitude > 1e-10f)) throw new InvalidOperationException("No matching sample received direct light.");
                report.nativeGtRepeatNrmse = Score(Project(ReadFrames(captures), gt, projections), target, samples);
                if (report.nativeGtRepeatNrmse > .005f) throw new InvalidOperationException("Repeating the GT capture is not stable.");

                for (int i = 0; i < lights.Length; i++) lights[i].shadows = LightShadows.None;
                var gtNoShadow = ReadFrames(captures, "gt_unshadowed", report.pngWhitePoint);
                for (int i = 0; i < lights.Length; i++) lights[i].shadows = shadows[i];
                Vector3[] targetNoShadow = Project(gtNoShadow, gt, projections);
                report.nativeShadowSignalNrmse = Score(targetNoShadow, target, samples);

                Apply(lights, initialLights);
                var initial = ReadFrames(captures, "initial", report.pngWhitePoint);
                Vector3[] nativeInitial = Project(initial, gt, projections);
                report.unityInitialNrmse = Score(nativeInitial, target, samples);
                report.unityInitialPixelNrmseByView = PixelNrmseByView(initial, gt);
                if (report.unityInitialNrmse < .1f) throw new InvalidOperationException("The initial perturbation did not materially change the captured field.");

                // Native culling's finalColor is the exact RGB power seen by URP. Check the conversion
                // separately, so a units/color-space mistake cannot look like an attenuation residual.
                var gtProxy = MakeProxy(lights, gtLights, gt[0], out report.gtLightColorMappingNrmse);
                var initialProxy = MakeProxy(lights, initialLights, initial[0], out report.initialLightColorMappingNrmse);
                var gtNoMask = AnalyticLighting.Evaluate(gtProxy, samples);
                var gtExactMask = AnalyticLighting.Evaluate(gtProxy, samples, UnityVisibility.Evaluate(gtProxy, samples));
                report.proxyAtGtWithoutVisibilityNrmse = Score(gtNoMask, target, samples);
                report.proxyAtGtWithVisibilityNrmse = Score(gtExactMask, target, samples);
                report.unshadowedProxyAtGtVsUnshadowedUnityNrmse = Score(gtNoMask, targetNoShadow, samples);

                var options = new OptimizerOptions { steps = 230, lbfgsIterations = 120, intensity = false, color = true, positionRate = .02f };
                var fitFields = new Vector3[2][];
                var recapturedFields = new Vector3[2][];
                report.fits = new Fit[2];
                for (int k = 0; k < report.fits.Length; k++)
                {
                    VisibilityProvider visibility = k == 0 ? null : (VisibilityProvider)UnityVisibility.Evaluate;
                    string name = k == 0 ? "no_visibility" : "collider_visibility";
                    var optimization = LightRigOptimizer.Optimize(initialProxy, target, samples, options, visibility);
                    fitFields[k] = AnalyticLighting.Evaluate(optimization.optimized, samples, visibility?.Invoke(optimization.optimized, samples));
                    var fitted = ExportLights(optimization.optimized, out int clamped);
                    Apply(lights, fitted);
                    var post = ReadFrames(captures, "optimized_" + name, report.pngWhitePoint);
                    recapturedFields[k] = Project(post, gt, projections);
                    LightSpec[] imported = MakeProxy(lights, fitted, post[0], out float lightMapping);
                    report.fits[k] = new Fit
                    {
                        name = name, fittedUnityLights = fitted, clampedChannels = clamped, visibleLightColorMappingNrmse = lightMapping,
                        proxyInitialNrmse = optimization.initialNrmse, proxyFittedNrmse = optimization.finalNrmse,
                        proxyAfterRgbClampNrmse = Score(AnalyticLighting.Evaluate(imported, samples, visibility?.Invoke(imported, samples)), target, samples),
                        recapturedNrmse = Score(recapturedFields[k], target, samples),
                        recapturedPixelNrmseByView = PixelNrmseByView(post, gt)
                    };
                }

                WriteSamples(samples, projections, sampleIds, target, targetNoShadow, nativeInitial, fitFields, recapturedFields);
                File.WriteAllText(Path.Combine(OutputRoot, "report.json"), JsonUtility.ToJson(report, true));
                Debug.Log($"Native irradiance: {samples.Count}/{allSamples.Count} matches, URP initial NRMSE {report.unityInitialNrmse:P2}; " +
                    string.Join(", ", report.fits.Select(f => $"{f.name}: proxy {f.proxyFittedNrmse:P2}, URP recapture {f.recapturedNrmse:P2}")) + ". " + OutputRoot);
            }
            finally
            {
                foreach (var capture in captures) capture.Dispose();
                Apply(lights, gtLights);
                for (int i = 0; i < lights.Length; i++) lights[i].shadows = shadows[i];
                foreach (var light in otherLights) light.enabled = true;
                SceneView.RepaintAll();
            }

            void AddView(Vector3 position, Vector3 lookAt)
            {
                var go = new GameObject("Auxiliary irradiance pose", typeof(Camera)) { hideFlags = HideFlags.HideAndDontSave };
                try
                {
                    var view = go.GetComponent<Camera>();
                    view.CopyFrom(source);
                    view.enabled = false;
                    view.transform.SetPositionAndRotation(position, Quaternion.LookRotation(lookAt - position, Vector3.up));
                    view.ResetWorldToCameraMatrix();
                    if (source.TryGetComponent<UniversalAdditionalCameraData>(out var sourceData))
                        EditorUtility.CopySerialized(sourceData, view.GetUniversalAdditionalCameraData());
                    captures.Add(new NativeIrradianceCapture(view, Width, Height));
                }
                finally { UnityEngine.Object.DestroyImmediate(go); }
            }
        }

        private static NativeIrradianceCapture.Frame[] ReadFrames(List<NativeIrradianceCapture> captures, string label = null, float white = 0)
        {
            var frames = new NativeIrradianceCapture.Frame[captures.Count];
            for (int i = 0; i < captures.Count; i++)
            {
                frames[i] = captures[i].Read();
                if (label == null) continue;
                string stem = Path.Combine(OutputRoot, label + "_view" + i);
                captures[i].SaveRaw(stem + ".exr");
                frames[i].SavePreview(stem + ".png", white > 0 ? white : frames[0].WhitePoint);
            }
            return frames;
        }

        private static Pose Describe(Camera camera, NativeIrradianceCapture.Frame frame) => new Pose
        {
            position = camera.transform.position, rotation = camera.transform.rotation, view = camera.worldToCameraMatrix,
            projection = camera.projectionMatrix, width = frame.Width, height = frame.Height, coveredPixels = frame.Covered, litPixels = frame.Lit
        };
        private static SurfaceSampleSet Subset(SurfaceSampleSet src, List<int> ids) => new SurfaceSampleSet
        {
            positions = ids.Select(i => src.positions[i]).ToArray(), normals = ids.Select(i => src.normals[i]).ToArray(),
            areas = ids.Select(i => src.areas[i]).ToArray(), albedos = ids.Select(i => src.albedos[i]).ToArray(), names = ids.Select(i => src.names[i]).ToArray()
        };
        private static Vector3[] Project(NativeIrradianceCapture.Frame[] frames, NativeIrradianceCapture.Frame[] gt, List<IrradianceSampleProjection> projections) =>
            projections.Select(p => p.Read(frames[p.CameraIndex], gt[p.CameraIndex])).ToArray();
        private static float[] PixelNrmseByView(NativeIrradianceCapture.Frame[] observed, NativeIrradianceCapture.Frame[] gt)
        {
            var nrmse = new float[gt.Length];
            for (int view = 0; view < gt.Length; view++)
            {
                double error = 0, energy = 0;
                for (int p = 0; p < gt[view].Pixels.Length; p++)
                {
                    Color predicted = observed[view].Pixels[p], target = gt[view].Pixels[p];
                    if ((target.a > 0) != (predicted.a > 0) || Mathf.Abs(target.a - predicted.a) > .001f)
                        throw new InvalidOperationException("Geometry moved between the native captures.");
                    if (target.a <= 0) continue;
                    Vector3 reference = IrradianceSampleProjection.Rgb(target);
                    error += (IrradianceSampleProjection.Rgb(predicted) - reference).sqrMagnitude;
                    energy += reference.sqrMagnitude;
                }
                nrmse[view] = (float)Math.Sqrt(error / Math.Max(energy, 1e-12));
            }
            return nrmse;
        }
        private static float Score(Vector3[] prediction, Vector3[] target, SurfaceSampleSet samples) => AnalyticLighting.FieldNrmse(prediction, target, samples);
        private static LightSpec ReadLight(Light l) => new LightSpec { position = l.transform.position, intensity = l.intensity, color = l.color, range = l.range };
        private static void Apply(Light[] lights, LightSpec[] specs)
        {
            for (int i = 0; i < lights.Length; i++)
            {
                lights[i].transform.position = specs[i].position;
                lights[i].intensity = specs[i].intensity;
                lights[i].color = specs[i].color;
            }
        }
        private static LightSpec[] Perturb(LightSpec[] gt)
        {
            var start = LightSpec.CloneArray(gt);
            for (int i = 0; i < start.Length; i++)
            {
                var l = start[i];
                l.position += PositionJitter[i];
                l.intensity *= IntensityJitter[i];
                var c = l.color;
                var scale = ColorJitter[i];
                l.color = new Color(Mathf.Clamp01(c.r * scale.x), Mathf.Clamp01(c.g * scale.y), Mathf.Clamp01(c.b * scale.z), c.a);
            }
            return start;
        }
        private static LightSpec[] MakeProxy(Light[] lights, LightSpec[] specs, NativeIrradianceCapture.Frame frame, out float mappingError)
        {
            var result = LightSpec.CloneArray(specs);
            float error = 0, energy = 0;
            for (int i = 0; i < result.Length; i++)
            {
                if (!frame.VisibleLightColors.TryGetValue(lights[i].GetInstanceID(), out var power))
                    throw new InvalidOperationException($"{lights[i].name} is outside the native camera light list.");
                Vector3 observed = IrradianceSampleProjection.Rgb(power);
                Vector3 predicted = specs[i].intensity * IrradianceSampleProjection.Rgb(specs[i].color.linear);
                error += (observed - predicted).sqrMagnitude;
                energy += observed.sqrMagnitude;
                result[i].attenuation = ProxyAttenuation;
                result[i].intensity = 1 / ProxyAttenuation;
                result[i].color = new Color(power.r, power.g, power.b, 1);
            }
            mappingError = Mathf.Sqrt(error / Mathf.Max(energy, 1e-12f));
            if (mappingError > .005f) throw new InvalidOperationException($"Unity intensity × linear RGB does not match visible light power ({mappingError:P2}).");
            return result;
        }
        private static LightSpec[] ExportLights(LightSpec[] proxy, out int clamped)
        {
            var result = LightSpec.CloneArray(proxy);
            clamped = 0;
            foreach (var l in result)
            {
                Color p = l.color * (l.intensity * l.attenuation);
                if (!float.IsFinite(p.r) || !float.IsFinite(p.g) || !float.IsFinite(p.b)) throw new InvalidOperationException("Non-finite optimized light power.");
                clamped += (p.r < 0 ? 1 : 0) + (p.g < 0 ? 1 : 0) + (p.b < 0 ? 1 : 0);
                // RGB power is identifiable, RGB and intensity separately are not. Use max linear RGB as intensity.
                l.intensity = Mathf.Max(0, Mathf.Max(p.r, Mathf.Max(p.g, p.b)));
                var unit = l.intensity > 0 ? new Color(Mathf.Max(0, p.r) / l.intensity, Mathf.Max(0, p.g) / l.intensity, Mathf.Max(0, p.b) / l.intensity) : Color.white;
                l.color = unit.gamma;
            }
            return result;
        }
        private static void WriteSamples(SurfaceSampleSet samples, List<IrradianceSampleProjection> projections, List<int> ids, Vector3[] gt,
            Vector3[] unshadowed, Vector3[] initial, Vector3[][] fits, Vector3[][] recaptures)
        {
            using (var writer = new StreamWriter(Path.Combine(OutputRoot, "registered_samples.csv")))
            {
                writer.Write("sample_id,surface,x,y,z,nx,ny,nz,area,view,pixel_x,pixel_y,eye_depth");
                foreach (string label in new[] { "gt", "gt_unshadowed", "initial", "proxy_no_visibility", "proxy_collider_visibility", "recapture_no_visibility", "recapture_collider_visibility" })
                    writer.Write($",{label}_r,{label}_g,{label}_b");
                writer.WriteLine();
                for (int i = 0; i < samples.Count; i++)
                {
                    var p = projections[i];
                    writer.Write($"{ids[i]},\"{samples.names[i].Replace("\"", "\"\"")}\",{Csv(samples.positions[i])},{Csv(samples.normals[i])},");
                    writer.WriteLine(FormattableString.Invariant($"{samples.areas[i]:R},{p.CameraIndex},{p.Pixel.x:R},{p.Pixel.y:R},{p.EyeDepth:R},{Csv(gt[i])},{Csv(unshadowed[i])},{Csv(initial[i])},{Csv(fits[0][i])},{Csv(fits[1][i])},{Csv(recaptures[0][i])},{Csv(recaptures[1][i])}"));
                }
            }
        }
        private static string Csv(Vector3 v) => FormattableString.Invariant($"{v.x:R},{v.y:R},{v.z:R}");
    }
}

using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using Object = UnityEngine.Object;

namespace LightingMvp.Editor
{
    public static class BlackBoxLightingSmoke
    {
        public static void BatchCornell() => Batch(false);
        public static void BatchMar7thRoom() => Batch(true);
        public static void BatchMixedLm() => Batch(false, true);

        private static void Batch(bool room, bool mixed = false)
        {
            int exit = 2;
            var previousPipeline = QualitySettings.renderPipeline;
            GameObject temporary = null;
            VolumeProfile temporalOverride = null;
            bool starRailTaaDisabled = false;
            try
            {
                string path = room ? "Assets/Unity_StarRail_CRP_Sample/Scenes/Mar7thRoom/Mar7thRoom.unity" : CornellBoxMvpEditor.ScenePath;
                EditorSceneManager.OpenScene(path);
                temporary = new GameObject("Black-box workflow smoke setup");
                Camera[] views;
                Light[] lights;
                if (room)
                {
                    // The sample's gameplay cameras are not all framed for an Edit-mode still.
                    // Use a fixed interior view, plus temporary realtime point lights, without changing the renderer.
                    views = new[] { AddCamera(temporary, new Vector3(0, 1.35f, 0), Quaternion.Euler(0, 180, 0)) };
                    lights = new[] { AddLight(temporary, new Vector3(-.5f, 1.7f, -.65f), new Color(1, .62f, .42f)),
                        AddLight(temporary, new Vector3(.7f, 1.1f, -1.2f), new Color(.45f, .72f, 1)) };
                }
                else
                {
                    // Use the existing Unity Forward asset as-is, including its renderer features.
                    // The plugin doesn't select a pipeline; Cornell's local project also contains StarRail's pipeline.
                    var forward = AssetDatabase.LoadAssetAtPath<UniversalRenderPipelineAsset>("Assets/Settings/URP-HighFidelity.asset");
                    if (!forward) throw new InvalidOperationException("Missing Cornell smoke pipeline asset.");
                    QualitySettings.renderPipeline = forward;
                    var source = GameObject.Find("Validation Camera").GetComponent<Camera>();
                    var second = AddCamera(temporary, source.transform.position + Vector3.right * .35f, Quaternion.Euler(0, -5, 0));
                    views = new[] { source, second };
                    lights = new[] { "Candidate Light", "Candidate Light 2", "Candidate Light 3" }.Select(n => GameObject.Find(n).GetComponent<Light>()).ToArray();
                    if (mixed)
                    {
                        lights[1].enabled = lights[2].enabled = false;
                        var spot = AddLight(temporary, new Vector3(-.45f, 1.76f, -.56f), new Color(1, .78f, .5f));
                        spot.type = LightType.Spot;
                        spot.spotAngle = 67; spot.innerSpotAngle = 33;
                        spot.transform.rotation = Quaternion.LookRotation(new Vector3(.25f, -.62f, 1), Vector3.up);
                        lights = new[] { lights[0], spot };
                    }
                }
                if (room)
                {
                    // Pin the temporal state for this regression even if two frames happen to match.
                    // Preserve StarRail's deferred renderer, shaders and all other renderer features.
                    var temporalType = TypeCache.GetTypesDerivedFrom<VolumeComponent>()
                        .Single(t => t.FullName == "Unity_StarRail_CRP_Sample.TemporalAA");
                    temporalOverride = ScriptableObject.CreateInstance<VolumeProfile>();
                    temporalOverride.Add(temporalType, true).parameters.OfType<BoolParameter>().Single().Override(false);
                    var volume = temporary.AddComponent<Volume>();
                    volume.isGlobal = true;
                    volume.priority = 10000;
                    volume.sharedProfile = temporalOverride;
                    starRailTaaDisabled = true;
                    Debug.Log("Mar7thRoom smoke explicitly disabled the StarRail Temporal AA volume override.");
                }
                var pipeline = GraphicsSettings.currentRenderPipeline;
                string pipelineName = pipeline.name;
                int beginCameraCount = 0;
                void CountCamera(ScriptableRenderContext context, Camera camera)
                {
                    if (Array.IndexOf(views, camera) >= 0) ++beginCameraCount;
                }
                RenderPipelineManager.beginCameraRendering += CountCamera;
                try { Verify(room, mixed, starRailTaaDisabled, pipelineName, views, lights, () => beginCameraCount); }
                finally { RenderPipelineManager.beginCameraRendering -= CountCamera; }
                Require(GraphicsSettings.currentRenderPipeline == pipeline, "The plugin changed the active pipeline.");
                exit = 0;
            }
            catch (Exception e) { Debug.LogException(e); }
            finally
            {
                if (temporary) Object.DestroyImmediate(temporary);
                if (temporalOverride) Object.DestroyImmediate(temporalOverride);
                if (!room) QualitySettings.renderPipeline = previousPipeline;
            }
            EditorApplication.Exit(exit);
        }

        private static void Verify(bool room, bool mixed, bool starRailTaaDisabled, string pipeline, Camera[] views, Light[] lights, Func<int> cameraCallbacks)
        {
            var viewTargets = views.Select(c => c.targetTexture).ToArray();
            var session = new BlackBoxLightingSession(views, lights, 160);
            Require(session.TargetRenders == views.Length * 3 && cameraCallbacks() == session.TargetRenders, "The actual scene cameras must run for every request.");
            bool color = !room && !mixed;
            var method = mixed ? BlackBoxLightingSession.Method.ImageLm : BlackBoxLightingSession.Method.CentralLbfgs;
            session.Perturb(room || mixed ? .22f : .27f, color, 17);
            var start = lights.Select(l => new BlackBoxLightingSession.LightState(l)).ToArray();
            var unrelated = lights.Select(l => new { l.range, l.shadows, l.cullingMask, l.renderingLayerMask, l.bounceIntensity, l.useColorTemperature, l.colorTemperature, l.type, l.spotAngle, l.innerSpotAngle }).ToArray();
            bool canceled = false;
            try
            {
                session.Solve(method, color, .63f, 6, 536 * views.Length, 17,
                    (used, _, __, ___) => { if (used >= views.Length * 3) throw new OperationCanceledException(); });
            }
            catch (OperationCanceledException) { canceled = true; }
            Require(canceled && Same(lights, start), "Cancel failed to restore the starting lights.");
            // Also test restore on an unexpected exception, after at least one actual probe.
            bool faulted = false;
            try
            {
                session.Solve(method, color, .63f, 6, 536 * views.Length, 17,
                    (used, _, __, ___) => { if (used >= views.Length * 3) throw new ArithmeticException(); });
            }
            catch (ArithmeticException) { faulted = true; }
            Require(faulted && Same(lights, start), "Error failed to restore the starting lights.");
            int previous = cameraCallbacks();
            var fit = session.Solve(method, color, .63f, 6, 536 * views.Length, 17);
            Require(cameraCallbacks() - previous == fit.TrainingRenders + fit.CheckRenders && fit.TrainingRenders <= 536 * views.Length &&
                fit.TrainingRenders == fit.Queries * views.Length, "The camera render budget is incorrect.");
            Require(fit.InitialLoss > .001 && fit.RecapturedLoss < fit.InitialLoss * .65, "The native RGB fit did not reduce a meaningful perturbation.");
            Require(fit.History.Count > 1 && Same(lights, start), "Solve failed to restore the starting lights.");
            if (mixed)
            {
                int spot = Array.FindIndex(lights, l => l.type == LightType.Spot);
                Require(spot >= 0 && Quaternion.Angle(start[spot].Aim.Value, fit.BestLights[spot].Aim.Value) > .1f,
                    "The mixed solve did not move the Spot aim.");
            }
            var last = fit.History.Last();
            var all = session.Replay(last, 0, -1);
            Require(NativeRenderBackend.Compare(all.Frame, fit.Best[0].Frame) <= 1e-8, "All-lights replay does not match the fit.");
            var solo = session.Replay(last, 0, 0);
            Require(NativeRenderBackend.Compare(solo.Frame, all.Frame) > 1e-8 && Same(lights, start), "Selected-light replay did not isolate and restore the rig.");
            session.Apply(fit.BestLights);
            Require(Same(lights, fit.BestLights) && !Same(lights, start), "Apply did not write the complete best rig.");
            Require(lights.Select(l => new { l.range, l.shadows, l.cullingMask, l.renderingLayerMask, l.bounceIntensity, l.useColorTemperature, l.colorTemperature, l.type, l.spotAngle, l.innerSpotAngle }).SequenceEqual(unrelated),
                "Apply changed unrelated Light properties.");
            Undo.FlushUndoRecordObjects();
            Undo.PerformUndo();
            Require(Same(lights, start), "Undo did not restore the complete starting rig.");
            Require(views.Select(c => c.targetTexture).SequenceEqual(viewTargets), "Camera targets were not restored.");
            var report = new SmokeReport { pipeline = pipeline, method = method.ToString(),
                lightTypes = string.Join("+", lights.Select(l => l.type)), viewCount = views.Length, lightCount = lights.Length,
                width = session.Target[0].Width, height = session.Target[0].Height, fitsColor = color, explicitlyDisabledStarRailTaa = starRailTaaDisabled,
                initialNrmse = (float)Math.Sqrt(fit.InitialLoss), recapturedNrmse = (float)Math.Sqrt(fit.RecapturedLoss),
                wallSeconds = (float)fit.Seconds, trainingRenders = fit.TrainingRenders, checkRenders = fit.CheckRenders + session.TargetRenders,
                keyframes = fit.History.Count, passed = true };
            string json = JsonUtility.ToJson(report, true);
            string output = Path.GetFullPath(Path.Combine(Application.dataPath, "../../results/unity_mvp/blackbox_plugin"));
            Directory.CreateDirectory(output);
            string stem = Path.Combine(output, room ? "mar7throom" : mixed ? "cornell_mixed_lm" : "cornell");
            File.WriteAllText(stem + ".json", json);
            SavePreview(session.Target[0], stem + "_target.png");
            SavePreview(fit.Initial[0], stem + "_initial.png");
            SavePreview(fit.Best[0], stem + "_best.png");
            Debug.Log("Black-box plugin smoke passed: " + json);
        }

        private static void SavePreview(BlackBoxLightingSession.Image image, string path)
        {
            var preview = image.Preview(readable: true);
            try { File.WriteAllBytes(path, preview.EncodeToPNG()); }
            finally { Object.DestroyImmediate(preview); }
        }

        [Serializable]
        private sealed class SmokeReport
        {
            public string pipeline, method, lightTypes;
            public bool passed, fitsColor, explicitlyDisabledStarRailTaa;
            public int viewCount, lightCount, width, height, trainingRenders, checkRenders, keyframes;
            public float initialNrmse, recapturedNrmse, wallSeconds;
        }

        private static bool Same(Light[] lights, BlackBoxLightingSession.LightState[] states) => lights.Select((l, i) =>
            l.transform.position.Equals(states[i].Position) && l.intensity == states[i].Intensity && l.color.Equals(states[i].Color) && l.enabled == states[i].Enabled &&
            (!states[i].Aim.HasValue || Quaternion.Angle(l.transform.rotation, states[i].Aim.Value) <= .01f)).All(v => v);
        private static void Require(bool condition, string message) { if (!condition) throw new InvalidOperationException(message); }

        private static Camera AddCamera(GameObject root, Vector3 position, Quaternion rotation)
        {
            var go = new GameObject("Black-box verification camera", typeof(Camera));
            go.transform.SetParent(root.transform);
            var camera = go.GetComponent<Camera>();
            camera.enabled = false;
            camera.aspect = 4f / 3f;
            camera.fieldOfView = 70;
            camera.nearClipPlane = .06f;
            camera.allowHDR = true;
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = Color.black;
            camera.transform.SetPositionAndRotation(position, rotation);
            return camera;
        }

        private static Light AddLight(GameObject root, Vector3 position, Color color)
        {
            var go = new GameObject("Black-box validation point light", typeof(Light));
            go.transform.SetParent(root.transform); go.transform.position = position;
            var light = go.GetComponent<Light>();
            light.type = LightType.Point;
            light.lightmapBakeType = LightmapBakeType.Realtime;
            light.intensity = 8; light.range = 5; light.color = color; light.shadows = LightShadows.Soft;
            return light;
        }
    }
}

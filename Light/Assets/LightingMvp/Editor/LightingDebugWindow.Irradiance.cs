using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace LightingMvp.Editor
{
    public sealed partial class LightingDebugWindow
    {
        private const string Mar7thRoom = "Assets/Unity_StarRail_CRP_Sample/Scenes/Mar7thRoom/Mar7thRoom.unity";
        private static Vector3 Rgb(Color c) => new Vector3(c.r, c.g, c.b);
        private static readonly string[] DisplayModes = { "Direct RGB", "Eye depth" };
        [SerializeField] private bool useSceneView = true;
        [SerializeField] private Camera source;
        [SerializeField] private int mode;
        [SerializeField] private float exposure;
        [SerializeField] private bool probeLight;
        private NativeIrradianceCapture.Frame frame;
        private Texture2D image;
        private string captureLabel, error;

        private void ClearIrradiance()
        {
            frame = null;
            error = null;
            ReleasePreview(ref image);
        }

        private void DrawIrradiance()
        {
            EditorGUILayout.LabelField("Active scene", SceneManager.GetActiveScene().name);
            useSceneView = EditorGUILayout.Toggle("Use Scene view pose", useSceneView);
            if (!useSceneView) source = (Camera)EditorGUILayout.ObjectField("Camera", source, typeof(Camera), true);
            probeLight = EditorGUILayout.Toggle("Add temporary test light", probeLight);
            if (GUILayout.Button("Capture irradiance")) CaptureIrradiance();
            if (error != null) EditorGUILayout.HelpBox(error, MessageType.Warning);
            if (frame == null) return;

            EditorGUILayout.LabelField("Captured from", captureLabel);
            EditorGUILayout.LabelField($"{frame.Lit:N0} lit / {frame.Covered:N0} covered pixels ({frame.Width} × {frame.Height})");
            EditorGUI.BeginChangeCheck();
            mode = GUILayout.Toolbar(mode, DisplayModes);
            if (mode == 0) exposure = EditorGUILayout.Slider("Exposure (EV)", exposure, -8, 8);
            if (EditorGUI.EndChangeCheck()) UpdateImage();
            if (frame.Covered == 0) EditorGUILayout.HelpBox("No supported opaque geometry in this view.", MessageType.Warning);
            else if (frame.Lit == 0) EditorGUILayout.HelpBox("No direct illumination in this view. Baked lighting and custom emissive or stylized terms are not sampled.", MessageType.Info);
            DrawPreview(image);
        }

        private void CaptureIrradiance()
        {
            try
            {
                Camera view = useSceneView ? SceneView.lastActiveSceneView?.camera : source;
                if (!view) throw new InvalidOperationException("Open a Scene view or select a scene camera first.");
                float aspect = Mathf.Clamp(view.aspect, .25f, 4f);
                int width = Mathf.RoundToInt(960 * Mathf.Min(1, aspect));
                int height = Mathf.RoundToInt(960 / Mathf.Max(1, aspect));
                using (var capture = new NativeIrradianceCapture(view, width, height))
                    frame = probeLight ? ReadWithProbe(view, capture) : capture.Read();
                captureLabel = (useSceneView ? "Scene view" : view.name) + " / " + SceneManager.GetActiveScene().name + (probeLight ? " (test light)" : "");
                error = null;
                UpdateImage();
            }
            catch (Exception e) { ClearIrradiance(); error = e.Message; Debug.LogException(e); }
        }

        private void UpdateImage()
        {
            ReleasePreview(ref image);
            float white = frame.WhitePoint / Mathf.Pow(2, exposure);
            if (mode != 0) white = frame.Pixels.Max(p => p.a);
            image = frame.MakePreview(white, mode != 0);
        }

        private static NativeIrradianceCapture.Frame ReadWithProbe(Camera view, NativeIrradianceCapture capture)
        {
            var go = new GameObject("Temporary irradiance test light", typeof(Light)) { hideFlags = HideFlags.HideAndDontSave };
            try
            {
                var light = go.GetComponent<Light>();
                light.type = LightType.Point;
                light.lightmapBakeType = LightmapBakeType.Realtime;
                light.intensity = 12;
                light.range = 20;
                light.color = new Color(1, .91f, .78f);
                light.cullingMask = view.cullingMask;
                light.shadows = LightShadows.None;
                go.transform.position = view.transform.position + view.transform.forward * .5f + view.transform.right * .35f;
                return capture.Read();
            }
            finally { DestroyImmediate(go); }
        }

        // Verify the actual room from a fixed, repeatable interior view. The saved gameplay
        // cameras are not all framed for an Edit-mode still (one is controlled at runtime).
        public static void BatchMar7thRoomIrradiance()
        {
            GameObject pose = null;
            try
            {
                EditorSceneManager.OpenScene(Mar7thRoom);
                pose = new GameObject("Mar7thRoom validation pose", typeof(Camera)) { hideFlags = HideFlags.HideAndDontSave };
                var camera = pose.GetComponent<Camera>();
                camera.enabled = false;
                camera.aspect = 4f / 3f;
                camera.fieldOfView = 75;
                camera.nearClipPlane = .06f;
                camera.transform.SetPositionAndRotation(new Vector3(0, 1.35f, 0), Quaternion.Euler(0, 180, 0));
                string output = Path.Combine(LightingExperimentPaths.OutputRoot, "native_irradiance");
                Directory.CreateDirectory(output);
                string stem = Path.Combine(output, "mar7th_room_interior");
                using (var capture = new NativeIrradianceCapture(camera, 800, 600))
                {
                    var original = capture.Read();
                    capture.SaveRaw(stem + ".exr");
                    original.SavePreview(stem + ".png", original.WhitePoint);
                    var test = ReadWithProbe(camera, capture);
                    float error = 0, energy = 0;
                    for (int p = 0; p < test.Pixels.Length; p++)
                    {
                        if (Mathf.Abs(test.Pixels[p].a - original.Pixels[p].a) > .001f)
                            throw new InvalidOperationException("Adding the test light unexpectedly changed Mar7thRoom receiver depth.");
                        Vector3 lit = Rgb(test.Pixels[p]);
                        Vector3 baseline = Rgb(original.Pixels[p]);
                        error += (lit - baseline).sqrMagnitude;
                        energy += lit.sqrMagnitude;
                    }
                    float delta = Mathf.Sqrt(error / Mathf.Max(energy, 1e-12f));
                    string line = FormattableString.Invariant($"{original.Width},{original.Height},{original.Covered},{original.Lit},{test.Lit},{delta:R}");
                    File.WriteAllText(Path.Combine(output, "mar7th_room.csv"),
                        "width,height,covered_pixels,lit_pixels,lit_pixels_with_test_light,relative_test_light_delta\n" + line + "\n");
                    if (original.Covered < original.Pixels.Length / 2 || original.Lit < 1000 || delta < .01f)
                        throw new InvalidOperationException("Mar7thRoom interior has insufficient native direct-light coverage or test-light response.");
                    test.SavePreview(stem + "_test_light.png", original.WhitePoint);
                    capture.SaveRaw(stem + "_test_light.exr");
                    Debug.Log("Mar7thRoom interior capture: " + line);
                }
            }
            catch (Exception e) { Debug.LogException(e); EditorApplication.Exit(2); return; }
            finally { if (pose) DestroyImmediate(pose); }
            EditorApplication.Exit(0);
        }
    }
}

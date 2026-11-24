using System;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace LightingMvp.Editor
{
    public sealed class BlackBoxLightingWindow : EditorWindow
    {
        private static readonly string[] Methods = { "Central FD + L-BFGS", "SPSA + Adam" };
        private static readonly int[] Resolutions = { 128, 256, 512, 1024 };
        [SerializeField] private Camera[] views = { null };
        [SerializeField] private Light[] lights = { null };
        [SerializeField] private int resolution = 256, method, budget = 2000, seed = 17;
        [SerializeField] private float extent = .75f, intensityFactor = 8, perturbDistance = .3f;
        [SerializeField] private bool fitColor;
        private BlackBoxLightingSession session;
        private BlackBoxLightingSession.Fit fit;
        private Texture2D targetImage, initialImage, bestImage, replayImage;
        private int view, keyframe, isolation;
        private bool playing;
        private double nextFrame;
        private string status;
        private MessageType severity;
        private Vector2 scroll;

        [MenuItem("Lighting MVP/Optimize Lighting (Unity RGB)", false, 0)]
        public static void Open() => GetWindow<BlackBoxLightingWindow>("RGB Lighting Optimizer").Show();

        private void OnEnable()
        {
            minSize = new Vector2(370, 360);
            EditorSceneManager.activeSceneChangedInEditMode += SceneChanged;
            EditorSceneManager.sceneClosed += SceneClosed;
            EditorApplication.playModeStateChanged += PlayModeChanged;
            EditorApplication.update += TickReplay;
        }

        private void OnDisable()
        {
            EditorSceneManager.activeSceneChangedInEditMode -= SceneChanged;
            EditorSceneManager.sceneClosed -= SceneClosed;
            EditorApplication.playModeStateChanged -= PlayModeChanged;
            EditorApplication.update -= TickReplay;
            ClearTarget();
        }

        private void SceneChanged(Scene _, Scene __) => ClearTarget();
        private void SceneClosed(Scene _) => ClearTarget();
        private void PlayModeChanged(PlayModeStateChange state) { if (state == PlayModeStateChange.ExitingEditMode) ClearTarget(); }

        private void OnGUI()
        {
            scroll = EditorGUILayout.BeginScrollView(scroll);
            EditorGUILayout.HelpBox("Match Unity's own final RGB. Capture the reference rig, move/recolor the lights, then fit the perturbed rig. Only Apply writes the optimized rig to the scene.", MessageType.Info);
            using (new EditorGUI.DisabledScope(EditorApplication.isPlayingOrWillChangePlaymode))
            {
                DrawTargetControls();
                if (session != null)
                {
                    DrawFitControls();
                    DrawResult();
                }
            }
            if (!string.IsNullOrEmpty(status)) EditorGUILayout.HelpBox(status, severity);
            EditorGUILayout.EndScrollView();
        }

        private void DrawTargetControls()
        {
            EditorGUILayout.LabelField("1. Capture reference", EditorStyles.boldLabel);
            EditorGUI.BeginChangeCheck();
            DrawObjects(ref views, "Cameras", 4);
            DrawObjects(ref lights, "Realtime Point Lights", 8);
            resolution = EditorGUILayout.IntPopup("Render long edge", resolution, Resolutions.Select(n => n.ToString()).ToArray(), Resolutions);
            if (EditorGUI.EndChangeCheck()) ClearTarget();
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Use main camera") && Camera.main)
            { views = new[] { Camera.main }; ClearTarget(); }
            using (new EditorGUI.DisabledScope(!Selection.gameObjects.Any(go => go.GetComponent<Light>())))
                if (GUILayout.Button("Use selected lights"))
                {
                    lights = Selection.gameObjects.Select(go => go.GetComponent<Light>()).Where(l => l).Distinct().Take(8).ToArray();
                    ClearTarget();
                }
            EditorGUILayout.EndHorizontal();
            if (GUILayout.Button(session == null ? "Capture current rig as target" : "Replace target with current rig"))
                RunAction(() =>
                {
                    var captured = new BlackBoxLightingSession(views, lights, resolution);
                    ClearTarget();
                    session = captured;
                    RefreshComparison();
                    Report($"Target captured and repeated ({session.TargetRenders} camera renders). Edit the lights or use Seeded perturbation.");
                });
        }

        private static void DrawObjects<T>(ref T[] list, string label, int max) where T : Component
        {
            int count = Mathf.Clamp(EditorGUILayout.IntField(label, list.Length), 1, max);
            if (count != list.Length) Array.Resize(ref list, count);
            for (int i = 0; i < list.Length; ++i)
                list[i] = (T)EditorGUILayout.ObjectField($"    {i + 1}", list[i], typeof(T), true);
        }

        private void DrawFitControls()
        {
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("2. Fit the current rig to the target", EditorStyles.boldLabel);
            method = EditorGUILayout.Popup("Optimizer", method, Methods);
            fitColor = EditorGUILayout.Toggle("Fit hue and saturation", fitColor);
            extent = EditorGUILayout.FloatField(new GUIContent("Position ± per axis", "World units around each starting light position."), extent);
            intensityFactor = EditorGUILayout.FloatField(new GUIContent("Intensity factor", "Log-space bounds: start / factor to start × factor."), intensityFactor);
            budget = Mathf.Clamp(EditorGUILayout.IntField("Training camera renders", budget), 1, 20000);
            seed = EditorGUILayout.IntField("Seed", seed);
            perturbDistance = EditorGUILayout.FloatField("Perturb distance", perturbDistance);
            if (GUILayout.Button("Seeded perturbation (Undoable)"))
                RunAction(() =>
                {
                    session.Perturb(perturbDistance, fitColor, seed);
                    ClearFit();
                    Report("Perturbation applied. Check the scene, then start optimization.");
                });
            if (GUILayout.Button("Start / re-run optimization")) RunAction(Solve);
        }

        private void Solve()
        {
            ClearFit();
            double nextProgress = 0;
            try
            {
                fit = session.Solve((BlackBoxOptimizer.Method)method, fitColor, extent, intensityFactor, budget, seed,
                    (used, max, phase, loss) =>
                    {
                        double now = EditorApplication.timeSinceStartup;
                        if (now < nextProgress && used < max) return;
                        nextProgress = now + .12;
                        string detail = double.IsInfinity(loss) ? "" : $" | Best NRMSE {Math.Sqrt(loss):P2}";
                        if (EditorUtility.DisplayCancelableProgressBar("Unity RGB lighting optimization", $"{phase}: {used:N0}/{max:N0} training renders{detail}", (float)used / max))
                            throw new OperationCanceledException();
                    });
                RefreshComparison();
                string ignored = fit.UnobservedLights.Length == 0 ? "" : " Little initial RGB response: " + string.Join(", ", fit.UnobservedLights) + ".";
                Report("Best rig re-captured. The starting scene lights were restored." + ignored,
                    fit.UnobservedLights.Length == 0 ? MessageType.Info : MessageType.Warning);
            }
            finally
            {
                EditorUtility.ClearProgressBar();
                SceneView.RepaintAll();
            }
        }

        private void DrawResult()
        {
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("3. Inspect and apply", EditorStyles.boldLabel);
            int chosen = EditorGUILayout.Popup("View", view, Enumerable.Range(0, session.ViewCount).Select(session.ViewName).ToArray());
            if (chosen != view)
            {
                view = chosen;
                RefreshComparison();
                if (replayImage) RunAction(UpdateReplay);
            }
            EditorGUILayout.BeginHorizontal();
            DrawImage("Target", targetImage);
            if (fit != null) { DrawImage("Before", initialImage); DrawImage("Best re-capture", bestImage); }
            EditorGUILayout.EndHorizontal();
            if (fit == null) return;
            EditorGUILayout.LabelField("RGB NRMSE (all views)", $"{Math.Sqrt(fit.Search.InitialLoss):P2} → {Math.Sqrt(fit.RecapturedLoss):P2}");
            if (fit.Search.InitialLoss > 0)
                EditorGUILayout.LabelField("Relative squared-error decrease", $"{1 - fit.RecapturedLoss / fit.Search.InitialLoss:P2}");
            EditorGUILayout.LabelField("Camera renders", $"{fit.TrainingRenders:N0} train + {session.TargetRenders + fit.CheckRenders:N0} capture/check");
            EditorGUILayout.LabelField("Solve wall time / stop", $"{fit.Seconds:F2} s / {fit.Search.StopReason}");
            using (new EditorGUI.DisabledScope(fit.Search.BestLoss >= fit.Search.InitialLoss))
                if (GUILayout.Button("Apply best rig to scene (Undoable)")) RunAction(() =>
                {
                    session.Apply(fit.BestLights);
                    Report("Applied the best rig. Undo restores the previous lights.");
                });
            DrawReplay();
        }

        private void DrawReplay()
        {
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Replay best-so-far states", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox("Isolate shows one fitted light at a time; other scene lighting stays visible. Scores always use all lights.", MessageType.None);
            EditorGUI.BeginChangeCheck();
            isolation = EditorGUILayout.Popup("Preview lighting", isolation,
                new[] { "All lights" }.Concat(Enumerable.Range(0, session.LightCount).Select(i => "Selected light only: " + session.LightName(i))).ToArray());
            keyframe = EditorGUILayout.IntSlider("Recorded state", keyframe, 0, fit.History.Count - 1);
            if (EditorGUI.EndChangeCheck()) RunAction(UpdateReplay);
            var entry = fit.History[keyframe];
            EditorGUILayout.LabelField($"Query {entry.Query} / all-lights NRMSE {Math.Sqrt(entry.Loss):P2}");
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Preview frame")) RunAction(UpdateReplay);
            using (new EditorGUI.DisabledScope(fit.History.Count < 2))
                if (GUILayout.Button(playing ? "Pause" : "Play"))
                {
                    playing = !playing;
                    if (playing)
                    {
                        if (keyframe == fit.History.Count - 1) keyframe = 0;
                        RunAction(UpdateReplay);
                        nextFrame = EditorApplication.timeSinceStartup + .25;
                    }
                }
            EditorGUILayout.EndHorizontal();
            if (replayImage) DrawImage("Replay", replayImage);
        }

        private void TickReplay()
        {
            if (!playing || fit == null || EditorApplication.timeSinceStartup < nextFrame) return;
            keyframe = Math.Min(keyframe + 1, fit.History.Count - 1);
            RunAction(UpdateReplay);
            playing &= keyframe < fit.History.Count - 1;
            nextFrame = EditorApplication.timeSinceStartup + .25;
            Repaint();
        }

        private static void DrawImage(string label, Texture2D image)
        {
            EditorGUILayout.BeginVertical();
            EditorGUILayout.LabelField(label, EditorStyles.centeredGreyMiniLabel);
            var area = GUILayoutUtility.GetRect(60, 180, GUILayout.ExpandWidth(true));
            if (image) EditorGUI.DrawPreviewTexture(area, image, null, ScaleMode.ScaleToFit);
            EditorGUILayout.EndVertical();
        }

        private void UpdateReplay() => Replace(ref replayImage, session.Replay(fit.History[keyframe], view, isolation - 1).Preview());
        private void RefreshComparison()
        {
            Replace(ref targetImage, session.Target[view].Preview());
            Replace(ref initialImage, fit?.Initial[view].Preview());
            Replace(ref bestImage, fit?.Best[view].Preview());
        }

        private void RunAction(Action action)
        {
            try { action(); }
            catch (OperationCanceledException) { Report("Canceled. Starting lights restored."); }
            catch (Exception e) { playing = false; Report(e.Message, MessageType.Error); }
        }

        private void Report(string message, MessageType type = MessageType.Info)
        { status = message; severity = type; Repaint(); }

        private static void Replace(ref Texture2D texture, Texture2D next = null)
        { if (texture) DestroyImmediate(texture); texture = next; }

        private void ClearFit()
        {
            fit = null; playing = false; keyframe = 0; isolation = 0;
            Replace(ref initialImage); Replace(ref bestImage); Replace(ref replayImage);
        }

        private void ClearTarget()
        {
            ClearFit(); session = null; view = 0; status = null;
            Replace(ref targetImage); Repaint();
        }
    }
}

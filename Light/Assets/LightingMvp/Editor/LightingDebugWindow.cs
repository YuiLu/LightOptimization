using System;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace LightingMvp.Editor
{
    [InitializeOnLoad]
    public sealed partial class LightingDebugWindow : EditorWindow
    {
        private enum Tab { Irradiance, Replay }
        private const double FrameInterval = .25;
        private static readonly string[] Tabs = { "Irradiance", "Replay" };

        private static BlackBoxLightingSession replaySession;
        private static IReadOnlyList<BlackBoxLightingSession.Keyframe> history;

        [SerializeField] private Tab tab;
        private Vector2 scroll;
        private Texture2D replayImage;
        private string replayError;
        private int cameraIndex, frameIndex, isolatedLight;
        private bool playing;
        private double nextFrame;

        static LightingDebugWindow()
        {
            EditorSceneManager.activeSceneChangedInEditMode += (_, __) => ClearSceneData();
            EditorSceneManager.sceneClosed += _ => ClearSceneData();
            EditorApplication.playModeStateChanged += state =>
            {
                if (state == PlayModeStateChange.ExitingEditMode) ClearSceneData();
            };
        }

        [MenuItem("Lighting MVP/Debug", false, 1)]
        public static void Open() => GetWindow<LightingDebugWindow>("Lighting Debug").Show();

        internal static void SetRecording(BlackBoxLightingSession session, IReadOnlyList<BlackBoxLightingSession.Keyframe> recording)
        {
            replaySession = session;
            history = recording;
            foreach (var window in Resources.FindObjectsOfTypeAll<LightingDebugWindow>())
                window.ResetReplay();
        }

        internal static void OpenReplay(BlackBoxLightingSession session, IReadOnlyList<BlackBoxLightingSession.Keyframe> recording, int view)
        {
            if (session != replaySession || recording != history) SetRecording(session, recording);
            var window = GetWindow<LightingDebugWindow>("Lighting Debug");
            window.tab = Tab.Replay;
            window.scroll = Vector2.zero;
            window.playing = false;
            window.cameraIndex = view;
            window.RenderReplayFrame();
            window.Show();
        }

        private static void ClearSceneData()
        {
            SetRecording(null, null);
            foreach (var window in Resources.FindObjectsOfTypeAll<LightingDebugWindow>())
                window.ClearIrradiance();
        }

        private void OnEnable()
        {
            minSize = new Vector2(360, 320);
            EditorApplication.update += AdvanceReplay;
        }

        private void OnDisable()
        {
            EditorApplication.update -= AdvanceReplay;
            ClearIrradiance();
            ResetReplay();
        }

        private void OnGUI()
        {
            var selected = (Tab)GUILayout.Toolbar((int)tab, Tabs);
            if (selected != tab)
            {
                tab = selected;
                scroll = Vector2.zero;
                playing = false;
            }

            using (var scope = new EditorGUILayout.ScrollViewScope(scroll))
            {
                scroll = scope.scrollPosition;
                using (new EditorGUI.DisabledScope(EditorApplication.isPlayingOrWillChangePlaymode))
                {
                    if (tab == Tab.Irradiance) DrawIrradiance();
                    else DrawReplay();
                }
            }
        }

        private void DrawReplay()
        {
            if (history == null)
            {
                EditorGUILayout.HelpBox("Run an optimization to record a replay.", MessageType.Info);
                return;
            }

            EditorGUI.BeginChangeCheck();
            cameraIndex = EditorGUILayout.Popup("Camera", cameraIndex,
                Enumerable.Range(0, replaySession.ViewCount).Select(replaySession.ViewName).ToArray());
            isolatedLight = EditorGUILayout.Popup("Lights", isolatedLight,
                new[] { "All lights" }.Concat(Enumerable.Range(0, replaySession.LightCount).Select(replaySession.LightName)).ToArray());
            frameIndex = EditorGUILayout.IntSlider("Recorded frame", frameIndex, 0, history.Count - 1);
            if (EditorGUI.EndChangeCheck())
            {
                playing = false;
                RenderReplayFrame();
            }

            var frame = history[frameIndex];
            EditorGUILayout.LabelField($"Frame {frameIndex + 1}/{history.Count} · Query {frame.Query}");
            EditorGUILayout.LabelField("All-lights NRMSE", $"{Math.Sqrt(frame.Loss):P2}");
            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button("Render frame")) RenderReplayFrame();
                using (new EditorGUI.DisabledScope(history.Count < 2))
                    if (GUILayout.Button(playing ? "Pause" : "Play"))
                    {
                        playing = !playing;
                        if (playing)
                        {
                            if (frameIndex == history.Count - 1) frameIndex = 0;
                            RenderReplayFrame();
                            nextFrame = EditorApplication.timeSinceStartup + FrameInterval;
                        }
                    }
            }

            if (replayError != null) EditorGUILayout.HelpBox(replayError, MessageType.Warning);
            DrawPreview(replayImage);
        }

        private void AdvanceReplay()
        {
            if (!playing || tab != Tab.Replay || history == null || EditorApplication.timeSinceStartup < nextFrame) return;
            frameIndex = Math.Min(frameIndex + 1, history.Count - 1);
            RenderReplayFrame();
            playing &= frameIndex < history.Count - 1;
            nextFrame = EditorApplication.timeSinceStartup + FrameInterval;
        }

        private void RenderReplayFrame()
        {
            ReleasePreview(ref replayImage);
            try
            {
                replayImage = replaySession.Replay(history[frameIndex], cameraIndex, isolatedLight - 1).Preview();
                replayError = null;
            }
            catch (Exception e)
            {
                playing = false;
                replayError = e.Message;
            }
            Repaint();
        }

        private void ResetReplay()
        {
            playing = false;
            cameraIndex = frameIndex = isolatedLight = 0;
            replayError = null;
            ReleasePreview(ref replayImage);
            Repaint();
        }

        private static void DrawPreview(Texture2D preview)
        {
            if (!preview) return;
            var area = GUILayoutUtility.GetAspectRect((float)preview.width / preview.height, GUILayout.ExpandWidth(true));
            EditorGUI.DrawPreviewTexture(area, preview, null, ScaleMode.ScaleToFit);
        }

        private static void ReleasePreview(ref Texture2D preview)
        {
            if (preview) DestroyImmediate(preview);
            preview = null;
        }
    }
}

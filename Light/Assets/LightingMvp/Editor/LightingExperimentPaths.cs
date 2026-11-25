using System.IO;
using UnityEngine;

namespace LightingMvp.Editor
{
    internal static class LightingExperimentPaths
    {
        internal const string CornellScene = "Assets/Scenes/CornellBoxMvp.unity";
        internal static string OutputRoot => Path.GetFullPath(Path.Combine(Application.dataPath, "../../results/unity_mvp"));
    }
}

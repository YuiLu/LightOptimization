using System;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace LightingMvp.Editor
{
    /// <summary>Existing Cornell lights, including inactive slots, with exact preview restoration.</summary>
    internal sealed class CornellSceneRig : IDisposable
    {
        private readonly Light[] lights;
        private readonly string[] originals;
        private readonly Vector3[] positions;
        private readonly Quaternion[] rotations;
        private readonly bool[] active;

        public CornellSceneRig(Scene scene)
        {
            lights=scene.GetRootGameObjects().SelectMany(go=>go.GetComponentsInChildren<Light>(true))
                .Where(light=>light.name=="Candidate Light"||light.name.StartsWith("Candidate Light ",StringComparison.Ordinal))
                .OrderBy(light=>light.name,StringComparer.Ordinal).ToArray();
            originals=lights.Select(light=>EditorJsonUtility.ToJson(light)).ToArray();
            positions=lights.Select(light=>light.transform.localPosition).ToArray();
            rotations=lights.Select(light=>light.transform.localRotation).ToArray();
            active=lights.Select(light=>light.gameObject.activeSelf).ToArray();
        }

        public int Count=>lights.Length;
        public Light this[int index]=>lights[index];

        public void Dispose()
        {
            for(int i=0;i<Count;i++)
            {
                if(lights[i]==null)continue;
                EditorJsonUtility.FromJsonOverwrite(originals[i],lights[i]);
                lights[i].transform.localPosition=positions[i];
                lights[i].transform.localRotation=rotations[i];
                lights[i].gameObject.SetActive(active[i]);
            }
        }
    }
}

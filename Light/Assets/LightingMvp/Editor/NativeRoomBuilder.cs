using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.SceneManagement;

namespace LightingMvp.Editor
{
    /// <summary>
    /// Builds a self-contained URP test room from Unity primitives: no imported
    /// geometry, no textures, no custom shaders, no baked GI. Seven realtime
    /// shadowed Point Lights named "Candidate Light 01..07" so the existing
    /// recovery rigs can drive them unchanged.
    /// </summary>
    internal static class NativeRoomBuilder
    {
        internal const string ScenePath = "Assets/Scenes/UrpNativeRoom.unity";
        private const string AssetFolder = "Assets/Scenes/UrpNativeRoom";
        private const string LightingPath = AssetFolder + "/UrpNativeRoomLighting.lighting";

        [MenuItem("Tools/Lighting MVP/Build Native URP Room")]
        public static void BuildMenu() => Build();

        public static void Build()
        {
            Directory.CreateDirectory(AssetFolder);
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            var lighting = AssetDatabase.LoadAssetAtPath<LightingSettings>(LightingPath);
            if (!lighting)
            {
                lighting = new LightingSettings();
                lighting.bakedGI = false;
                lighting.realtimeGI = false;
                lighting.autoGenerate = false;
                AssetDatabase.CreateAsset(lighting, LightingPath);
            }
            Lightmapping.lightingSettings = lighting;
            Lightmapping.bakedGI = false;
            Lightmapping.realtimeGI = false;

            RenderSettings.skybox = null;
            RenderSettings.ambientMode = AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.035f, 0.035f, 0.04f);
            RenderSettings.fog = false;
            RenderSettings.reflectionIntensity = 0.15f;

            var shader = Shader.Find("Universal Render Pipeline/Lit");
            if (!shader) throw new InvalidOperationException("URP Lit shader not found; open this project with the URP pipeline active.");
            var wall = Mat(shader, "RoomWall", new Color(0.68f, 0.66f, 0.62f), 0.08f);
            var floor = Mat(shader, "RoomFloor", new Color(0.42f, 0.36f, 0.30f), 0.25f);
            var wood = Mat(shader, "RoomWood", new Color(0.50f, 0.36f, 0.24f), 0.18f);
            var dark = Mat(shader, "RoomDark", new Color(0.15f, 0.15f, 0.16f), 0.12f);
            var fabric = Mat(shader, "RoomFabric", new Color(0.30f, 0.34f, 0.38f), 0.05f);

            var root = new GameObject("Urp Native Room").transform;

            Box(root, "Floor", new Vector3(0, -0.1f, 0), new Vector3(7.8f, 0.2f, 7.8f), floor);
            Box(root, "Ceiling", new Vector3(0, 3.1f, 0), new Vector3(7.8f, 0.2f, 7.8f), wall);
            Box(root, "Wall North", new Vector3(0, 1.5f, 3.9f), new Vector3(7.8f, 3f, 0.2f), wall);
            Box(root, "Wall South", new Vector3(0, 1.5f, -3.9f), new Vector3(7.8f, 3f, 0.2f), wall);
            Box(root, "Wall West", new Vector3(-3.9f, 1.5f, 0), new Vector3(0.2f, 3f, 7.8f), wall);
            Box(root, "Wall East", new Vector3(3.9f, 1.5f, 0), new Vector3(0.2f, 3f, 7.8f), wall);
            Box(root, "Desk", new Vector3(0.9f, 0.72f, 1.2f), new Vector3(1.8f, 0.08f, 0.9f), wood);
            Box(root, "Desk Leg West", new Vector3(0.12f, 0.35f, 1.2f), new Vector3(0.08f, 0.7f, 0.8f), dark);
            Box(root, "Desk Leg East", new Vector3(1.68f, 0.35f, 1.2f), new Vector3(0.08f, 0.7f, 0.8f), dark);
            Box(root, "Chair Seat", new Vector3(-0.35f, 0.44f, 0.95f), new Vector3(0.5f, 0.08f, 0.5f), fabric);
            Box(root, "Chair Back", new Vector3(-0.35f, 0.78f, 0.72f), new Vector3(0.5f, 0.62f, 0.08f), fabric);
            Box(root, "Cabinet", new Vector3(-3.15f, 0.9f, 1.4f), new Vector3(0.5f, 1.8f, 2.2f), wood);
            Box(root, "Partition", new Vector3(0.2f, 1.05f, 0.05f), new Vector3(0.15f, 2.1f, 2.6f), wall);
            Box(root, "Shelf", new Vector3(3.1f, 0.9f, -1.6f), new Vector3(1.2f, 1.8f, 0.5f), dark);
            Box(root, "Pedestal", new Vector3(-1.6f, 0.35f, -1.9f), new Vector3(0.6f, 0.7f, 0.6f), fabric);

            var lights = new[]
            {
                PointLight(root, 1, new Vector3(0f, 2.82f, 0f), new Color(1f, 0.95f, 0.88f), 6.0f, 9f),
                PointLight(root, 2, new Vector3(-2.4f, 2.82f, 1.8f), new Color(1f, 0.97f, 0.93f), 4.5f, 7f),
                PointLight(root, 3, new Vector3(2.4f, 2.82f, -1.6f), new Color(0.92f, 0.96f, 1f), 4.5f, 7f),
                PointLight(root, 4, new Vector3(0f, 2.82f, 2.6f), new Color(1f, 0.93f, 0.84f), 3.2f, 6f),
                PointLight(root, 5, new Vector3(-3.55f, 2.1f, 0.4f), new Color(1f, 0.9f, 0.8f), 2.2f, 4f),
                PointLight(root, 6, new Vector3(0.9f, 1.18f, 1.2f), new Color(1f, 0.86f, 0.7f), 1.6f, 3f),
                PointLight(root, 7, new Vector3(-2.2f, 1.05f, -2.6f), new Color(0.95f, 0.95f, 1f), 2.0f, 4f),
            };

            var cameraGo = new GameObject("Validation Camera");
            cameraGo.transform.SetPositionAndRotation(new Vector3(2.95f, 1.55f, 3.1f), Quaternion.LookRotation(new Vector3(-3.1f, -0.5f, -3.1f).normalized, Vector3.up));
            var camera = cameraGo.AddComponent<Camera>();
            camera.fieldOfView = 55f;
            camera.nearClipPlane = 0.05f;
            camera.farClipPlane = 40f;
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(0.012f, 0.012f, 0.015f);
            camera.allowHDR = true;
            camera.allowMSAA = false;
            var cameraData = cameraGo.AddComponent<UniversalAdditionalCameraData>();
            cameraData.renderPostProcessing = false;
            cameraData.antialiasing = AntialiasingMode.None;
            cameraData.renderShadows = true;

            Directory.CreateDirectory(Path.GetDirectoryName(ScenePath) ?? "Assets/Scenes");
            EditorSceneManager.SaveScene(scene, ScenePath);
            AssetDatabase.SaveAssets();
            Debug.Log($"[NativeRoom] Built {ScenePath}: {lights.Length} realtime Point Lights, 1 camera, baked GI off.");
        }

        [MenuItem("Tools/Lighting MVP/Render Native Room Preview")]
        public static void PreviewMenu() => Preview();

        public static void Preview()
        {
            var scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
            var camera = GameObject.Find("Validation Camera")?.GetComponent<Camera>();
            if (!camera) throw new InvalidOperationException("Validation Camera missing from " + ScenePath);
            const int width = 896, height = 512;
            var rt = new RenderTexture(width, height, 24, RenderTextureFormat.ARGBHalf) { name = "NativeRoomPreview" };
            var previous = camera.targetTexture;
            camera.targetTexture = rt;
            camera.Render();
            camera.targetTexture = previous;
            var active = RenderTexture.active;
            RenderTexture.active = rt;
            var tex = new Texture2D(width, height, TextureFormat.RGBAFloat, false, true);
            tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            tex.Apply();
            RenderTexture.active = active;
            string outPath = Environment.GetEnvironmentVariable("BB_ROOM_PREVIEW") ?? "native-room-preview.png";
            File.WriteAllBytes(outPath, tex.EncodeToPNG());
            UnityEngine.Object.DestroyImmediate(tex);
            rt.Release();
            UnityEngine.Object.DestroyImmediate(rt);
            Debug.Log($"[NativeRoom] Preview {width}x{height} -> {outPath}");
        }

        private static Material Mat(Shader shader, string name, Color color, float smoothness)
        {
            string path = $"{AssetFolder}/{name}.mat";
            var material = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (!material)
            {
                material = new Material(shader) { name = name };
                AssetDatabase.CreateAsset(material, path);
            }
            material.shader = shader;
            material.SetColor("_BaseColor", color);
            material.SetFloat("_Smoothness", smoothness);
            material.SetFloat("_Metallic", 0f);
            EditorUtility.SetDirty(material);
            return material;
        }

        private static void Box(Transform parent, string name, Vector3 position, Vector3 size, Material material)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = name;
            go.transform.SetParent(parent, false);
            go.transform.localPosition = position;
            go.transform.localScale = size;
            GameObjectUtility.SetStaticEditorFlags(go, 0);
            var renderer = go.GetComponent<MeshRenderer>();
            renderer.sharedMaterial = material;
            renderer.shadowCastingMode = ShadowCastingMode.On;
            renderer.receiveShadows = true;
        }

        private static Light PointLight(Transform parent, int index, Vector3 position, Color color, float intensity, float range)
        {
            var go = new GameObject($"Candidate Light {index:00}");
            go.transform.SetParent(parent, false);
            go.transform.localPosition = position;
            var light = go.AddComponent<Light>();
            light.type = LightType.Point;
            light.color = color;
            light.intensity = intensity;
            light.range = range;
            light.shadows = LightShadows.Hard;
            light.shadowStrength = 1f;
            light.lightmapBakeType = LightmapBakeType.Realtime;
            light.useColorTemperature = false;
            light.renderMode = LightRenderMode.Auto;
            return light;
        }
    }
}





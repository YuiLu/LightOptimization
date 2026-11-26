using System;
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
    /// Builds a Classroom-scale URP room from Unity primitives: deep occluders,
    /// hanging baffles, columns, shelving and seven shadow-casting lamp fixtures.
    /// Stock URP Lit materials, no textures, no GI, seven realtime Point Lights
    /// named "Candidate Light 01..07" so the existing recovery rigs drive them.
    /// </summary>
    internal static class NativeComplexRoomBuilder
    {
        internal const string ScenePath = "Assets/Scenes/UrpNativeComplexRoom.unity";
        private const string AssetFolder = "Assets/Scenes/UrpNativeComplexRoom";
        private const string LightingPath = AssetFolder + "/UrpNativeComplexRoomLighting.lighting";

        internal static readonly Vector3[] Eyes =
        {
            new Vector3(3.40f, 1.55f, 2.95f), new Vector3(-3.45f, 1.55f, 2.90f), new Vector3(3.45f, 1.50f, -2.90f),
            new Vector3(-3.40f, 1.50f, -2.95f), new Vector3(0f, 1.65f, 3.35f)
        };
        internal static readonly Vector3[] Look =
        {
            new Vector3(-.6f, 1.05f, -.8f), new Vector3(.8f, 1.05f, -.6f), new Vector3(-.7f, 1.10f, .9f),
            new Vector3(.7f, 1.10f, .8f), new Vector3(0f, 1.00f, -1.8f)
        };

        [MenuItem("Tools/Lighting MVP/Build Native Complex Room")]
        public static void BuildMenu() => Build();

        public static void Build()
        {
            Directory.CreateDirectory(AssetFolder);
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            var lighting = AssetDatabase.LoadAssetAtPath<LightingSettings>(LightingPath);
            if (!lighting)
            {
                lighting = new LightingSettings { bakedGI = false, realtimeGI = false, autoGenerate = false };
                AssetDatabase.CreateAsset(lighting, LightingPath);
            }
            Lightmapping.lightingSettings = lighting;
            Lightmapping.bakedGI = false;
            Lightmapping.realtimeGI = false;
            RenderSettings.skybox = null;
            RenderSettings.ambientMode = AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.030f, 0.030f, 0.035f);
            RenderSettings.fog = false;
            RenderSettings.reflectionIntensity = 0.12f;

            var shader = Shader.Find("Universal Render Pipeline/Lit");
            if (!shader) throw new InvalidOperationException("URP Lit shader not found; open this project with the URP pipeline active.");
            var wall = Mat(shader, "ComplexWall", new Color(0.66f, 0.64f, 0.60f), 0.06f);
            var floor = Mat(shader, "ComplexFloor", new Color(0.34f, 0.30f, 0.26f), 0.22f);
            var wood = Mat(shader, "ComplexWood", new Color(0.46f, 0.33f, 0.22f), 0.16f);
            var dark = Mat(shader, "ComplexDark", new Color(0.13f, 0.13f, 0.14f), 0.10f);
            var fabric = Mat(shader, "ComplexFabric", new Color(0.26f, 0.30f, 0.35f), 0.04f);
            var metal = Mat(shader, "ComplexMetal", new Color(0.55f, 0.55f, 0.57f), 0.55f);

            var root = new GameObject("Urp Native Complex Room").transform;

            // Shell: 9.2 x 7.4 x 3.0 m.
            Box(root, "Floor", new Vector3(0, -0.1f, 0), new Vector3(9.2f, 0.2f, 7.4f), floor);
            Box(root, "Ceiling", new Vector3(0, 3.1f, 0), new Vector3(9.2f, 0.2f, 7.4f), wall);
            Box(root, "Wall North", new Vector3(0, 1.5f, 3.7f), new Vector3(9.2f, 3f, 0.2f), wall);
            Box(root, "Wall South", new Vector3(0, 1.5f, -3.7f), new Vector3(9.2f, 3f, 0.2f), wall);
            Box(root, "Wall West", new Vector3(-4.6f, 1.5f, 0), new Vector3(0.2f, 3f, 7.4f), wall);
            Box(root, "Wall East", new Vector3(4.6f, 1.5f, 0), new Vector3(0.2f, 3f, 7.4f), wall);

            // Structure that breaks the light field: beams, columns, baffles, partitions.
            for (int i = -1; i <= 1; ++i) Box(root, "Ceiling Beam " + (i + 2), new Vector3(0, 2.62f, i * 2.35f), new Vector3(9.2f, 0.30f, 0.26f), dark);
            Box(root, "Column West", new Vector3(-3.05f, 1.3f, 0.05f), new Vector3(0.42f, 2.6f, 0.42f), wall);
            Box(root, "Column East", new Vector3(3.05f, 1.3f, 0.05f), new Vector3(0.42f, 2.6f, 0.42f), wall);
            Box(root, "Partition", new Vector3(-1.95f, 0.675f, -0.15f), new Vector3(2.9f, 1.35f, 0.14f), wall);
            Box(root, "Partition Cap", new Vector3(-1.95f, 1.39f, -0.15f), new Vector3(3.0f, 0.08f, 0.24f), wood);
            Box(root, "Hanging Baffle A", new Vector3(0.15f, 1.92f, 0.25f), new Vector3(1.3f, 0.42f, 0.06f), fabric);
            Box(root, "Hanging Baffle B", new Vector3(-2.40f, 1.95f, -2.00f), new Vector3(1.0f, 0.36f, 0.06f), fabric);

            // Deep furniture: shelves, cabinets, radiator, window frames.
            Box(root, "Shelf North West", new Vector3(-3.35f, 1.0f, 3.30f), new Vector3(1.7f, 2.0f, 0.48f), wood);
            Box(root, "Shelf North East", new Vector3(3.35f, 1.0f, 3.30f), new Vector3(1.7f, 2.0f, 0.48f), wood);
            Box(root, "Shelf East", new Vector3(4.25f, 0.95f, 0.55f), new Vector3(0.42f, 1.9f, 2.2f), wood);
            Box(root, "Cabinet West", new Vector3(-4.25f, 0.80f, -1.85f), new Vector3(0.44f, 1.6f, 1.9f), dark);
            Box(root, "Radiator", new Vector3(1.05f, 0.35f, 3.52f), new Vector3(1.8f, 0.6f, 0.18f), metal);
            Box(root, "Window Frame North", new Vector3(1.05f, 1.95f, 3.60f), new Vector3(2.0f, 1.3f, 0.10f), dark);
            Box(root, "Door Frame West", new Vector3(-4.50f, 1.05f, -3.05f), new Vector3(0.10f, 2.1f, 1.0f), dark);

            // Six desks with chairs, plus a teacher desk.
            Vector3[] desks = { new Vector3(-2.75f, 0, -1.75f), new Vector3(0, 0, -1.75f), new Vector3(2.75f, 0, -1.75f),
                new Vector3(-2.75f, 0, 1.35f), new Vector3(0, 0, 1.35f), new Vector3(2.75f, 0, 1.35f) };
            for (int i = 0; i < desks.Length; ++i)
            {
                Desk(root, "Desk " + (i + 1), desks[i], wood, dark);
                Chair(root, "Chair " + (i + 1), desks[i] + new Vector3(0, 0, 0.85f), fabric, dark);
            }
            Desk(root, "Teacher Desk", new Vector3(-1.2f, 0, -3.05f), wood, dark);
            Chair(root, "Teacher Chair", new Vector3(-1.2f, 0, -3.55f), fabric, dark);

            var lights = new[]
            {
                PointLight(root, 1, new Vector3(0f, 2.28f, 0f), new Color(1f, 0.95f, 0.87f), 2.4f, 7.0f),
                PointLight(root, 2, new Vector3(-2.30f, 2.28f, -1.50f), new Color(1f, 0.97f, 0.92f), 2.8f, 6.6f),
                PointLight(root, 3, new Vector3(2.30f, 2.28f, -1.50f), new Color(0.93f, 0.96f, 1f), 2.8f, 6.6f),
                PointLight(root, 4, new Vector3(-2.30f, 2.28f, 1.50f), new Color(1f, 0.93f, 0.83f), 2.8f, 6.6f),
                PointLight(root, 5, new Vector3(2.30f, 2.28f, 1.50f), new Color(1f, 0.94f, 0.86f), 2.8f, 6.6f),
                PointLight(root, 6, new Vector3(-3.70f, 2.15f, -1.10f), new Color(1f, 0.90f, 0.80f), 2.0f, 4.6f),
                PointLight(root, 7, new Vector3(3.70f, 2.15f, 1.10f), new Color(0.95f, 0.93f, 1f), 2.0f, 4.6f),
            };
            for (int i = 0; i < lights.Length; ++i) Fixture(root, i + 1, lights[i], metal);

            var cameraGo = new GameObject("Validation Camera");
            cameraGo.transform.SetPositionAndRotation(Eyes[0], Quaternion.LookRotation((Look[0] - Eyes[0]).normalized, Vector3.up));
            var camera = cameraGo.AddComponent<Camera>();
            camera.fieldOfView = 55f;
            camera.nearClipPlane = 0.05f;
            camera.farClipPlane = 40f;
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(0.010f, 0.010f, 0.013f);
            camera.allowHDR = true;
            camera.allowMSAA = false;
            var cameraData = cameraGo.AddComponent<UniversalAdditionalCameraData>();
            cameraData.renderPostProcessing = false;
            cameraData.antialiasing = AntialiasingMode.None;
            cameraData.renderShadows = true;

            Directory.CreateDirectory(Path.GetDirectoryName(ScenePath) ?? "Assets/Scenes");
            EditorSceneManager.SaveScene(scene, ScenePath);
            AssetDatabase.SaveAssets();
            Debug.Log($"[ComplexRoom] Built {ScenePath}: {lights.Length} realtime Point Lights with fixtures, baked GI off.");
        }

        [MenuItem("Tools/Lighting MVP/Render Native Complex Room Preview")]
        public static void PreviewMenu() => Preview();

        public static void Preview()
        {
            var scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
            var camera = GameObject.Find("Validation Camera")?.GetComponent<Camera>();
            if (!camera) throw new InvalidOperationException("Validation Camera missing from " + ScenePath);
            const int width = 896, height = 512;
            var original = camera.transform;
            Vector3 p0 = camera.transform.position; Quaternion r0 = camera.transform.rotation;
            string prefix = Environment.GetEnvironmentVariable("BB_ROOM_PREVIEW") ?? "native-complex-room";
            for (int v = 0; v < Eyes.Length; ++v)
            {
                camera.transform.SetPositionAndRotation(Eyes[v], Quaternion.LookRotation((Look[v] - Eyes[v]).normalized, Vector3.up));
                var rt = new RenderTexture(width, height, 24, RenderTextureFormat.ARGBHalf) { name = "ComplexRoomPreview" };
                var previous = camera.targetTexture;
                camera.targetTexture = rt;
                camera.Render();
                camera.targetTexture = previous;
                var active = RenderTexture.active;
                RenderTexture.active = rt;
                var tex = new Texture2D(width, height, TextureFormat.RGBAFloat, false, true);
                tex.ReadPixels(new UnityEngine.Rect(0, 0, width, height), 0, 0);
                tex.Apply();
                RenderTexture.active = active;
                File.WriteAllBytes(prefix + "-view" + v + ".png", tex.EncodeToPNG());
                UnityEngine.Object.DestroyImmediate(tex);
                rt.Release();
                UnityEngine.Object.DestroyImmediate(rt);
            }
            Debug.Log($"[ComplexRoom] Preview {width}x{height} x{Eyes.Length} -> {prefix}-view*.png ({scene.name}, camera {p0} {r0})");
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

        private static void Box(Transform parent, string name, Vector3 position, Vector3 size, Material material, bool collider = true)
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
            if (!collider) UnityEngine.Object.DestroyImmediate(go.GetComponent<Collider>());
        }

        private static void Cylinder(Transform parent, string name, Vector3 position, float radius, float height, Material material)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            go.name = name;
            go.transform.SetParent(parent, false);
            go.transform.localPosition = position;
            go.transform.localScale = new Vector3(radius * 2f, height * .5f, radius * 2f);
            GameObjectUtility.SetStaticEditorFlags(go, 0);
            var renderer = go.GetComponent<MeshRenderer>();
            renderer.sharedMaterial = material;
            renderer.shadowCastingMode = ShadowCastingMode.On;
            renderer.receiveShadows = true;
            UnityEngine.Object.DestroyImmediate(go.GetComponent<Collider>());
        }

        private static void Desk(Transform parent, string name, Vector3 origin, Material top, Material legs)
        {
            Box(parent, name + " Top", origin + new Vector3(0, .75f, 0), new Vector3(1.7f, .06f, .8f), top, false);
            Box(parent, name + " Leg W", origin + new Vector3(-.78f, .36f, 0), new Vector3(.06f, .72f, .72f), legs);
            Box(parent, name + " Leg E", origin + new Vector3(.78f, .36f, 0), new Vector3(.06f, .72f, .72f), legs);
            Box(parent, name + " Modesty", origin + new Vector3(0, .55f, -.32f), new Vector3(1.6f, .28f, .05f), top, false);
        }

        private static void Chair(Transform parent, string name, Vector3 origin, Material seat, Material frame)
        {
            Box(parent, name + " Seat", origin + new Vector3(0, .45f, 0), new Vector3(.5f, .06f, .5f), seat, false);
            Box(parent, name + " Back", origin + new Vector3(0, .78f, .24f), new Vector3(.5f, .62f, .07f), seat, false);
            Box(parent, name + " Post", origin + new Vector3(0, .22f, 0), new Vector3(.08f, .44f, .08f), frame);
            Box(parent, name + " Foot", origin + new Vector3(0, .03f, 0), new Vector3(.42f, .06f, .42f), frame);
        }

        private static void Fixture(Transform parent, int index, Light light, Material metal)
        {
            var p = light.transform.localPosition;
            Cylinder(parent, $"Fixture L{index:00} Shade", p + new Vector3(0, .16f, 0), .18f, .07f, metal);
            Cylinder(parent, $"Fixture L{index:00} Stem", p + new Vector3(0, .42f, 0), .015f, .48f, metal);
            Cylinder(parent, $"Fixture L{index:00} Ring", p + new Vector3(0, .11f, 0), .185f, .02f, metal);
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


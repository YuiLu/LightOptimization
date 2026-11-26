using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace LightingMvp.Editor
{
    // Synchronous Editor capture using the source camera's live URP renderer. No renderer asset is edited.
    internal sealed class NativeIrradianceCapture : IDisposable
    {
        private readonly Camera camera;
        private readonly RenderTexture buffer, destination;
        private readonly Texture2D readback;
        private readonly CaptureFeature feature;
        private readonly UniversalAdditionalCameraData cameraData;

        private static readonly FieldInfo FeaturesField = typeof(ScriptableRenderer).GetField(
            "m_RendererFeatures", BindingFlags.Instance | BindingFlags.NonPublic);
        private static readonly PropertyInfo RendererMode = typeof(UniversalRenderer).GetProperty(
            "renderingModeActual", BindingFlags.Instance | BindingFlags.NonPublic);

        public NativeIrradianceCapture(Camera source, int width, int height)
        {
            if (!source) throw new ArgumentNullException(nameof(source));
            if (QualitySettings.activeColorSpace != ColorSpace.Linear || !(GraphicsSettings.currentRenderPipeline is UniversalRenderPipelineAsset))
                throw new InvalidOperationException("Capture requires a linear-color URP project.");
            if (!SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.ARGBFloat) || !SystemInfo.SupportsTextureFormat(TextureFormat.RGBAFloat))
                throw new InvalidOperationException("Capture requires a graphics device with RGBA32F render and readback support.");
            var shader = AssetDatabase.LoadAssetAtPath<Shader>("Assets/LightingMvp/Shaders/IrradianceCapture.shader");
            if (!shader || !shader.isSupported) throw new InvalidOperationException("IrradianceCapture shader is unavailable on this device.");

            camera = new GameObject("Irradiance capture camera", typeof(Camera)) { hideFlags = HideFlags.HideAndDontSave }.GetComponent<Camera>();
            try
            {
                camera.CopyFrom(source);
                camera.overrideSceneCullingMask = source.overrideSceneCullingMask;
                camera.transform.SetPositionAndRotation(source.transform.position, source.transform.rotation);
                camera.cameraType = CameraType.Game;
                camera.enabled = false;
                camera.stereoTargetEye = StereoTargetEyeMask.None;
                camera.allowMSAA = false;
                camera.allowDynamicResolution = false;
                camera.rect = new Rect(0, 0, 1, 1);
                camera.aspect = source.aspect;
                var data = cameraData = camera.GetUniversalAdditionalCameraData();
                if (source.TryGetComponent<UniversalAdditionalCameraData>(out var originalData))
                {
                    var index = new SerializedObject(originalData).FindProperty("m_RendererIndex");
                    if (index != null) data.SetRenderer(index.intValue);
                    data.renderShadows = originalData.renderShadows;
                    data.volumeLayerMask = originalData.volumeLayerMask;
                    data.volumeTrigger = originalData.volumeTrigger;
                }
                data.renderPostProcessing = false;
                data.allowXRRendering = false;
                buffer = new RenderTexture(width, height, 24, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear)
                    { name = "Native irradiance RGB + eye depth", hideFlags = HideFlags.HideAndDontSave };
                destination = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear)
                    { name = "Native capture camera target", hideFlags = HideFlags.HideAndDontSave };
                if (!buffer.Create() || !destination.Create()) throw new InvalidOperationException("Could not allocate native capture targets.");
                readback = new Texture2D(width, height, TextureFormat.RGBAFloat, false, true) { hideFlags = HideFlags.HideAndDontSave };
                feature = ScriptableObject.CreateInstance<CaptureFeature>();
                feature.hideFlags = HideFlags.HideAndDontSave;
                feature.Initialize(camera, buffer, shader);
            }
            catch { Dispose(); throw; }
        }

        public Frame Read() => new Frame(buffer.width, buffer.height, ReadPixels());

        internal void SetSurfaceSamples(Vector3[] positions, Vector3[] normals, Bounds bounds)
        {
            feature.Pass.SetSurfaceSamples(positions, normals);
            float radius = bounds.extents.magnitude + .5f;
            camera.orthographic = true;
            camera.aspect = buffer.width / (float)buffer.height;
            camera.orthographicSize = radius * Mathf.Max(1, 1 / camera.aspect);
            camera.nearClipPlane = .03f;
            camera.farClipPlane = 2 * radius + 1;
            cameraData.renderShadows = true;
            camera.transform.SetPositionAndRotation(bounds.center + Vector3.up * radius, Quaternion.LookRotation(Vector3.down, Vector3.forward));
            camera.ResetProjectionMatrix();
            camera.ResetWorldToCameraMatrix();
        }

        internal Color[] ReadSurfaces(Light[] lights)
        {
            feature.Pass.SurfaceLights = lights;
            try { return ReadPixels(); }
            finally { feature.Pass.SurfaceLights = null; }
        }

        private Color[] ReadPixels()
        {
            var request = new UniversalRenderPipeline.SingleCameraRequest { destination = destination };
            if (!RenderPipeline.SupportsRenderRequest(camera, request))
                throw new InvalidOperationException("The current URP renderer does not accept single-camera requests.");
            feature.Pass.Executed = false;
            RenderTexture previous = RenderTexture.active;
            RenderTexture previousCameraTarget = camera.targetTexture;
            List<ScriptableRendererFeature> activeFeatures = null;
            var removed = new List<(int index, ScriptableRendererFeature item)>();
            RenderPipelineManager.beginCameraRendering += InstallPass;
            try
            {
                RenderPipeline.SubmitRenderRequest(camera, request);
                if (!feature.Pass.Executed) throw new InvalidOperationException("The native capture pass did not execute.");
                RenderTexture.active = buffer;
                readback.ReadPixels(new Rect(0, 0, buffer.width, buffer.height), 0, 0, false);
                return readback.GetRawTextureData<Color>().ToArray();
            }
            finally
            {
                RenderPipelineManager.beginCameraRendering -= InstallPass;
                if (activeFeatures != null)
                {
                    activeFeatures.Remove(feature);
                    for (int i = removed.Count - 1; i >= 0; i--)
                        activeFeatures.Insert(removed[i].index, removed[i].item);
                }
                camera.targetTexture = previousCameraTarget;
                RenderTexture.active = previous;
            }

            void InstallPass(ScriptableRenderContext context, Camera renderingCamera)
            {
                if (renderingCamera != camera || activeFeatures != null) return;
                // The first request can (re)create the pipeline and its renderer. Install the feature
                // on the live renderer here, not on a renderer cached beforehand. Enqueue later, after Clear().
                if (!(cameraData.scriptableRenderer is UniversalRenderer renderer))
                    throw new InvalidOperationException("Capture requires the URP Universal Renderer.");
                string mode = RendererMode?.GetValue(renderer)?.ToString();
                if (mode != "Forward" && mode != "ForwardPlus")
                    throw new InvalidOperationException("Capture supports URP Forward and Forward+; current renderer mode: " + (mode ?? "unknown"));
                activeFeatures = FeaturesField?.GetValue(renderer) as List<ScriptableRendererFeature>;
                if (activeFeatures == null) throw new InvalidOperationException("Cannot install the temporary capture pass in this URP version (tested on URP 14).");

                // The StarRail custom deferred feature applies its own shading/TAA and retains buffers
                // for each camera. Do not run it for disposable sensor cameras. Native URP lighting,
                // including its light/shadow setup, still runs in the same renderer and pipeline.
                for (int i = activeFeatures.Count - 1; i >= 0; i--)
                {
                    var item = activeFeatures[i];
                    if (item && item.GetType().FullName == "Unity_StarRail_CRP_Sample.CRPDeferredRenderingRendererFeature")
                    { removed.Add((i, item)); activeFeatures.RemoveAt(i); }
                }
                activeFeatures.Add(feature);
            }
        }

        public void SaveRaw(string path) => File.WriteAllBytes(path, readback.EncodeToEXR(Texture2D.EXRFlags.OutputAsFloat));

        public void Dispose()
        {
            if (feature) { feature.Dispose(); UnityEngine.Object.DestroyImmediate(feature); }
            if (camera) UnityEngine.Object.DestroyImmediate(camera.gameObject);
            if (buffer) UnityEngine.Object.DestroyImmediate(buffer);
            if (destination) UnityEngine.Object.DestroyImmediate(destination);
            if (readback) UnityEngine.Object.DestroyImmediate(readback);
        }

        internal sealed class Frame
        {
            public readonly int Width, Height;
            public readonly Color[] Pixels;
            public readonly int Covered, Lit;
            public readonly float WhitePoint;

            public Frame(int width, int height, Color[] pixels)
            {
                Width = width; Height = height; Pixels = pixels;
                var values = new List<float>();
                foreach (var pixel in pixels)
                {
                    if (!float.IsFinite(pixel.a) || !float.IsFinite(pixel.r) || !float.IsFinite(pixel.g) || !float.IsFinite(pixel.b))
                        throw new InvalidOperationException("Non-finite result in the URP capture.");
                    if (pixel.a <= 0) continue;
                    Covered++;
                    float value = Mathf.Max(pixel.r, Mathf.Max(pixel.g, pixel.b));
                    if (value > 1e-6f) { Lit++; values.Add(value); }
                }
                if (values.Count > 0) { values.Sort(); WhitePoint = values[(values.Count - 1) * 99 / 100]; }
                else WhitePoint = 1;
            }

            public Texture2D MakePreview(float white, bool depth = false)
            {
                var image = new Texture2D(Width, Height, TextureFormat.RGB24, false, false) { hideFlags = HideFlags.HideAndDontSave };
                var pixels = new Color[Pixels.Length];
                float gain = 1 / Mathf.Max(white, 1e-8f);
                for (int i = 0; i < pixels.Length; i++)
                {
                    Color p = Pixels[i];
                    pixels[i] = p.a <= 0 ? new Color(0.065f, 0.065f, 0.065f) : depth
                        ? new Color(1, 1, 1) * Mathf.Clamp01(p.a * gain)
                        : new Color(Mathf.Clamp01(p.r * gain), Mathf.Clamp01(p.g * gain), Mathf.Clamp01(p.b * gain)).gamma;
                }
                image.SetPixels(pixels);
                image.Apply(false);
                return image;
            }

            public void SavePreview(string path, float white)
            {
                var image = MakePreview(white);
                try { File.WriteAllBytes(path, image.EncodeToPNG()); }
                finally { UnityEngine.Object.DestroyImmediate(image); }
            }
        }

        private sealed class CaptureFeature : ScriptableRendererFeature
        {
            private Camera camera;
            public CapturePass Pass { get; private set; }
            public override void Create() { }
            protected override void Dispose(bool disposing) => Pass?.Release();
            public void Initialize(Camera source, RenderTexture target, Shader shader)
            {
                camera = source;
                Pass = new CapturePass(target, shader);
            }
            public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData data)
            {
                if (data.cameraData.camera == camera) renderer.EnqueuePass(Pass);
            }
        }

        private sealed class CapturePass : ScriptableRenderPass
        {
            private static readonly List<ShaderTagId> Tags = new List<ShaderTagId>
            {
                new ShaderTagId("UniversalForward"), new ShaderTagId("UniversalForwardOnly"), new ShaderTagId("SRPDefaultUnlit"),
                new ShaderTagId("UniversalGBuffer"), new ShaderTagId("SceneForward"), new ShaderTagId("SceneGBuffer"),
                new ShaderTagId("SceneGBufferSSR"), new ShaderTagId("SSSForward"), new ShaderTagId("SSSGBuffer"),
                new ShaderTagId("UnLitForward"), new ShaderTagId("UnLitGBuffer"), new ShaderTagId("CharacterGBuffer")
            };
            private readonly RTHandle target;
            private readonly Shader shader;
            private readonly int width, height;
            private ComputeBuffer positions, normals;
            private Material surfaceMaterial;
            private int count;
            internal Light[] SurfaceLights;
            public bool Executed;

            public CapturePass(RenderTexture buffer, Shader overrideShader)
            {
                target = RTHandles.Alloc(new RenderTargetIdentifier(buffer), "Native irradiance");
                width = buffer.width; height = buffer.height;
                shader = overrideShader;
                renderPassEvent = RenderPassEvent.BeforeRenderingOpaques;
            }
            public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData data)
            {
                ConfigureTarget(target);
                ConfigureClear(ClearFlag.All, Color.clear);
            }
            public override void Execute(ScriptableRenderContext context, ref RenderingData data)
            {
                if (SurfaceLights != null)
                {
                    DrawSamples(context, ref data);
                    Executed = true;
                    return;
                }
                var draw = CreateDrawingSettings(Tags, ref data, data.cameraData.defaultOpaqueSortFlags);
                // Keep the material's raster culling mode (for example Mar7thRoom's two-sided room meshes).
                draw.overrideShader = shader;
                draw.overrideShaderPassIndex = 0;
                var filter = new FilteringSettings(RenderQueueRange.opaque, data.cameraData.camera.cullingMask);
                context.DrawRenderers(data.cullResults, ref draw, ref filter);
                Executed = true;
            }

            internal void SetSurfaceSamples(Vector3[] points, Vector3[] directions)
            {
                if (positions != null || points.Length == 0 || points.Length != directions.Length || points.Length > width * height)
                    throw new ArgumentException("Invalid fixed surface sample set.");
                count = points.Length;
                positions = new ComputeBuffer(count, 12);
                normals = new ComputeBuffer(count, 12);
                positions.SetData(points); normals.SetData(directions);
                surfaceMaterial = new Material(shader) { hideFlags = HideFlags.HideAndDontSave };
            }

            private void DrawSamples(ScriptableRenderContext context, ref RenderingData data)
            {
                if (count == 0 || SurfaceLights.Length * count > width * height)
                    throw new InvalidOperationException("Surface irradiance does not fit its render target.");
                using var cmd = new CommandBuffer { name = "Native per-light surface irradiance" };
                cmd.SetViewport(new Rect(0, 0, width, height));
                cmd.SetGlobalBuffer("_SurfacePositions", positions);
                cmd.SetGlobalBuffer("_SurfaceNormals", normals);
                cmd.SetGlobalVector("_SurfaceDimensions", new Vector4(width, height, SystemInfo.graphicsUVStartsAtTop ? -1 : 1, 0));
                var visible = data.lightData.visibleLights;
                int main = data.lightData.mainLightIndex;
                for (int lamp = 0; lamp < SurfaceLights.Length; ++lamp)
                {
                    int found = -1;
                    for (int index = 0; index < visible.Length; ++index)
                        if (visible[index].light == SurfaceLights[lamp]) { found = index; break; }
                    if (found < 0 || found == main)
                        throw new InvalidOperationException("Fixed surface capture needs each requested point light in the native additional-light list.");
                    cmd.SetGlobalInt("_SurfaceLightIndex", found - (main >= 0 && main < found ? 1 : 0));
                    cmd.SetGlobalInt("_SurfaceOffset", lamp * count);
                    cmd.DrawProcedural(Matrix4x4.identity, surfaceMaterial, 1, MeshTopology.Points, count, 1);
                }
                context.ExecuteCommandBuffer(cmd);
            }

            public void Release()
            {
                positions?.Dispose(); normals?.Dispose();
                if (surfaceMaterial) UnityEngine.Object.DestroyImmediate(surfaceMaterial);
                target.Release();
            }
        }
    }
}

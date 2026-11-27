using System;
using System.Diagnostics;
using System.IO;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using Object = UnityEngine.Object;

namespace LightingMvp
{
    /// <summary>Final camera RGB, with the active URP renderer and actual scene materials.</summary>
    public sealed class NativeRenderBackend : IDisposable
    {
        public sealed class Frame
        {
            internal readonly Color[] Pixels;
            public readonly double Energy;
            public int PixelCount => Pixels.Length;
            public Color Pixel(int index) => Pixels[index];
            public readonly float Peak;

            internal Frame(NativeArray<Color> source)
            {
                Pixels = source.ToArray();
                foreach (Color c in Pixels)
                {
                    Energy += (double)c.r * c.r + (double)c.g * c.g + (double)c.b * c.b;
                    Peak = Mathf.Max(Peak, Mathf.Max(c.r, Mathf.Max(c.g, c.b)));
                }
                if (double.IsNaN(Energy) || double.IsInfinity(Energy))
                    throw new InvalidOperationException("Non-finite final RGB in camera target.");
            }
        }

        private readonly Camera[] cameras;
        private readonly bool ownsCameras;
        private readonly RenderTexture destination;
        private readonly Texture2D readback;
        private readonly RenderPipelineAsset pipeline;
        public int RenderCount { get; private set; }
        public double RenderSeconds { get; private set; }
        public double ReadbackSeconds { get; private set; }

        public NativeRenderBackend(Camera source, Vector3[] positions, Quaternion[] rotations, int size, int rendererIndex = -1)
            : this(source, positions, rotations, size, size, rendererIndex) { }

        public NativeRenderBackend(Camera source, Vector3[] positions, Quaternion[] rotations, int width, int height, int rendererIndex)
        {
            if (!source || positions == null || rotations == null || positions.Length == 0 || positions.Length != rotations.Length || width < 1 || height < 1)
                throw new ArgumentException("Invalid offscreen camera configuration.");
            pipeline = RequirePipeline();
            ownsCameras = true;
            cameras = new Camera[positions.Length];
            try
            {
                destination = new RenderTexture(width, height, 24, GraphicsFormat.R32G32B32A32_SFloat)
                    { name = "Black-box final linear HDR RGB", hideFlags = HideFlags.HideAndDontSave };
                if (!destination.Create()) throw new InvalidOperationException("Could not create linear HDR target.");
                readback = new Texture2D(width, height, TextureFormat.RGBAFloat, false, true) { hideFlags = HideFlags.HideAndDontSave };
                for (int i = 0; i < positions.Length; ++i)
                {
                    var go = new GameObject("Black-box view " + i) { hideFlags = HideFlags.HideAndDontSave };
                    var camera = cameras[i] = go.AddComponent<Camera>();
                    camera.CopyFrom(source);
                    camera.enabled = false;
                    camera.targetTexture = destination;
                    camera.rect = new Rect(0, 0, 1, 1);
                    camera.aspect = (float)width / height;
                    camera.ResetProjectionMatrix();
                    camera.ResetWorldToCameraMatrix();
                    camera.clearFlags = CameraClearFlags.SolidColor;
                    camera.backgroundColor = Color.black;
                    camera.allowHDR = true;
                    camera.allowMSAA = false;
                    camera.allowDynamicResolution = false;
                    camera.transform.SetPositionAndRotation(positions[i], rotations[i]);
                    var data = go.AddComponent<UniversalAdditionalCameraData>();
                    data.SetRenderer(rendererIndex);
                    data.renderShadows = true;
                    data.renderPostProcessing = false;
                    data.antialiasing = AntialiasingMode.None;
                    data.allowXRRendering = false;
                    data.allowHDROutput = false;
                    data.dithering = false;
                }
            }
            catch { Dispose(); throw; }
        }

        // Interactive path: render the actual scene camera so its renderer, volumes,
        // custom passes and render callbacks see the same camera/identity.
        public NativeRenderBackend(Camera source, int width, int height)
        {
            if (!source || width < 1 || height < 1) throw new ArgumentException("Invalid scene camera configuration.");
            pipeline = RequirePipeline();
            cameras = new[] { source };
            try
            {
                destination = new RenderTexture(width, height, 24, GraphicsFormat.R32G32B32A32_SFloat)
                    { name = "Black-box final linear RGB", hideFlags = HideFlags.HideAndDontSave };
                if (!destination.Create()) throw new InvalidOperationException("Could not create linear HDR target.");
                readback = new Texture2D(width, height, TextureFormat.RGBAFloat, false, true) { hideFlags = HideFlags.HideAndDontSave };
            }
            catch { Dispose(); throw; }
        }

        private static RenderPipelineAsset RequirePipeline()
        {
            var pipeline = GraphicsSettings.currentRenderPipeline;
            if (!(pipeline is UniversalRenderPipelineAsset urp) || !urp.supportsHDR || QualitySettings.activeColorSpace != ColorSpace.Linear)
                throw new InvalidOperationException("RGB recovery requires Linear color space and HDR-enabled URP.");
            if (!SystemInfo.IsFormatSupported(GraphicsFormat.R32G32B32A32_SFloat, FormatUsage.Render) ||
                !SystemInfo.SupportsTextureFormat(TextureFormat.RGBAFloat))
                throw new NotSupportedException("Full float HDR render and CPU readback are required.");
            return pipeline;
        }

        public int Width => destination.width;
        public int Height => destination.height;
        public int Size => Width;
        public Frame Capture(int view) => new Frame(Read(view));

        public double Loss(int view, Frame target)
        {
            if (target == null || target.Pixels.Length != Width * Height)
                throw new ArgumentException("Target resolution must match the capture resolution.");
            var rendered = Read(view);
            double error = 0;
            for (int i = 0; i < rendered.Length; ++i)
            {
                Color a = rendered[i], b = target.Pixels[i];
                double r = a.r - b.r, g = a.g - b.g, blue = a.b - b.b;
                error += r * r + g * g + blue * blue;
            }
            return error / (target.Energy + 1e-12);
        }

        // Concatenate views at caller-provided offsets, with 1/sqrt(viewCount) weighting.
        // Store the residual in float, but return the same double RGB MSE used by Loss.
        public double Residual(int view, Frame target, float[] residual, int offset = 0, int viewCount = 1)
        {
            int count = checked(Width * Height * 3);
            if (target == null || target.PixelCount * 3 != count || residual == null ||
                offset < 0 || offset > residual.Length - count || viewCount < 1)
                throw new ArgumentException("Target and residual buffer must match the capture size.");
            var pixels = Read(view);
            double denominator = (target.Energy + 1e-12) * viewCount;
            double scale = 1 / Math.Sqrt(denominator), energy = 0;
            for (int i = 0, k = offset; i < pixels.Length; ++i)
            {
                Color a = pixels[i], b = target.Pixels[i];
                double r = a.r - b.r, g = a.g - b.g, blue = a.b - b.b;
                energy += r * r + g * g + blue * blue;
                residual[k++] = (float)(r * scale);
                residual[k++] = (float)(g * scale);
                residual[k++] = (float)(blue * scale);
            }
            return energy / denominator;
        }

        public enum HdrMapping { Raw, SignedLog, SignedSqrt, ToneExp, ToneReinhard }

        public sealed class MappedTarget
        {
            internal readonly Frame Raw;
            internal readonly float[] Rgb;
            internal readonly double LogScale;
            public double Energy { get; }
            internal readonly HdrMapping Mapping;

            public MappedTarget(Frame target, HdrMapping mapping, double logScale)
            {
                if (target == null || mapping == HdrMapping.Raw || !(logScale > 0)) throw new ArgumentException("Invalid mapped HDR target.");
                Raw = target; Mapping = mapping; LogScale = logScale;
                Rgb = new float[target.PixelCount * 3];
                for (int i = 0, k = 0; i < target.PixelCount; ++i)
                {
                    var c = target.Pixels[i];
                    Rgb[k++] = Map(c.r, mapping, logScale);
                    Rgb[k++] = Map(c.g, mapping, logScale);
                    Rgb[k++] = Map(c.b, mapping, logScale);
                }
                foreach (float value in Rgb) Energy += (double)value * value;
            }
        }

        public static float Map(float value, HdrMapping mapping, double logScale)
        {
            if (mapping == HdrMapping.Raw || value == 0) return value;
            if (mapping == HdrMapping.ToneExp) return Display(value);
            if (mapping == HdrMapping.ToneReinhard)
                return value <= 0 ? 0 : Mathf.LinearToGammaSpace((float)(value / (logScale + value)));
            double x = Math.Abs((double)value), y;
            if (mapping == HdrMapping.SignedLog)
            {
                x /= logScale;
                // Avoid cancellation near zero; error is below the float rounding scale here.
                y = x < 1e-4 ? x * (1 + x * (-.5 + x / 3)) : Math.Log(1 + x);
            }
            else y = x / (Math.Sqrt(x + 1e-6) + .001); // sqrt(x + eps) - sqrt(eps)
            return (float)(value < 0 ? -y : y);
        }

        public double MappedResidual(int view, MappedTarget target, float[] residual, int offset, int viewCount, out double rawLoss)
        {
            int count = checked(Width * Height * 3);
            if (target?.Rgb.Length != count || offset < 0 || residual != null && offset > residual.Length - count || viewCount < 1)
                throw new ArgumentException("Invalid mapped HDR residual dimensions.");
            var pixels = Read(view);
            double denominator = (target.Energy + 1e-12) * viewCount;
            double scale = 1 / Math.Sqrt(denominator), mapped = 0, raw = 0;
            for (int i = 0, k = 0; i < pixels.Length; ++i)
            {
                Color a = pixels[i], b = target.Raw.Pixels[i];
                double r = a.r - b.r, g = a.g - b.g, blue = a.b - b.b;
                raw += r * r + g * g + blue * blue;
                r = Map(a.r, target.Mapping, target.LogScale) - target.Rgb[k];
                g = Map(a.g, target.Mapping, target.LogScale) - target.Rgb[k + 1];
                blue = Map(a.b, target.Mapping, target.LogScale) - target.Rgb[k + 2];
                if (residual != null)
                {
                    residual[offset + k] = (float)(r * scale);
                    residual[offset + k + 1] = (float)(g * scale);
                    residual[offset + k + 2] = (float)(blue * scale);
                }
                k += 3;
                mapped += r * r + g * g + blue * blue;
            }
            rawLoss = raw / ((target.Raw.Energy + 1e-12) * viewCount);
            return mapped / denominator;
        }

        private NativeArray<Color> Read(int view)
        {
            if (pipeline != GraphicsSettings.currentRenderPipeline) throw new InvalidOperationException("The URP pipeline changed during recovery.");
            var camera = cameras[view];
            var request = new UniversalRenderPipeline.SingleCameraRequest { destination = destination };
            if (!RenderPipeline.SupportsRenderRequest(camera, request))
                throw new InvalidOperationException("This URP renderer does not support single camera render requests.");
            var previous = RenderTexture.active;
            var previousTarget = camera.targetTexture;
            try
            {
                var clock = Stopwatch.StartNew();
                ++RenderCount;
                RenderPipeline.SubmitRenderRequest(camera, request);
                RenderSeconds += clock.Elapsed.TotalSeconds;
                clock.Restart();
                RenderTexture.active = destination;
                readback.ReadPixels(new Rect(0, 0, Width, Height), 0, 0, false);
                var data = readback.GetRawTextureData<Color>();
                ReadbackSeconds += clock.Elapsed.TotalSeconds;
                return data;
            }
            finally
            {
                // URP restores targetTexture on success; also restore it when a renderer feature throws.
                if (camera && camera.targetTexture != previousTarget) camera.targetTexture = previousTarget;
                RenderTexture.active = previous;
            }
        }

        public static double Compare(Frame rendered, Frame target)
        {
            if (rendered.Pixels.Length != target.Pixels.Length) throw new ArgumentException("Mismatched frame sizes.");
            double error = 0;
            for (int i = 0; i < rendered.Pixels.Length; ++i)
            {
                Color a = rendered.Pixels[i], b = target.Pixels[i];
                double r = a.r - b.r, g = a.g - b.g, blue = a.b - b.b;
                error += r * r + g * g + blue * blue;
            }
            return error / (target.Energy + 1e-12);
        }

        public static void Save(Frame frame, int size, string path) => Save(frame, size, size, path);

        public static void Save(Frame frame, int width, int height, string path)
        {
            var raw = new Texture2D(width, height, TextureFormat.RGBAFloat, false, true);
            var preview = new Texture2D(width, height, TextureFormat.RGB24, false, false);
            try
            {
                raw.SetPixels(frame.Pixels);
                File.WriteAllBytes(path + ".exr", raw.EncodeToEXR(Texture2D.EXRFlags.OutputAsFloat | Texture2D.EXRFlags.CompressZIP));
                var rgb = new Color32[frame.Pixels.Length];
                for (int i = 0; i < rgb.Length; ++i)
                {
                    // Fixed exposure / display mapping across all views, states and methods. Loss uses unmodified float RGB.
                    Color c = frame.Pixels[i];
                    rgb[i] = new Color(Display(c.r), Display(c.g), Display(c.b), 1);
                }
                preview.SetPixels32(rgb);
                File.WriteAllBytes(path + ".png", preview.EncodeToPNG());
            }
            finally { Destroy(raw); Destroy(preview); }
        }

        public static float Display(float value) => Mathf.LinearToGammaSpace(1 - Mathf.Exp(-Mathf.Max(0, value)));

        public void Dispose()
        {
            if (ownsCameras) foreach (Camera camera in cameras) if (camera) Destroy(camera.gameObject);
            if (destination) destination.Release();
            Destroy(destination);
            Destroy(readback);
        }

        private static void Destroy(Object obj)
        {
            if (!obj) return;
            if (Application.isPlaying) Object.Destroy(obj);
            else Object.DestroyImmediate(obj);
        }
    }
}

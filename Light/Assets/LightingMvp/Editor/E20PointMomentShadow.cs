using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using Object = UnityEngine.Object;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        [Serializable] private sealed class MomentSettings
        {
            public string label;
            public int size, requestedMsaa, actualMsaa, blurRadius;
            public float near, far, momentBias, depthBiasMetres, leakBias;
            public bool variance, mbd;
            public float overestimationWeight;
        }

        private sealed class PointMomentShadow : IDisposable
        {
            private const string ComputePath = "Assets/LightingMvp/Shaders/E20MomentShadow.compute";
            private static readonly Vector3[] Directions = { Vector3.right, Vector3.left, Vector3.up, Vector3.down, Vector3.forward, Vector3.back };
            private static readonly Vector3[] Up = { Vector3.up, Vector3.up, Vector3.back, Vector3.forward, Vector3.up, Vector3.up };
            private static readonly string[] FaceNames = { "px", "nx", "py", "ny", "pz", "nz" };
            private readonly ComputeShader compute;
            private readonly Action<CommandBuffer> drawCasters;
            private readonly MomentSettings settings;
            private readonly RenderTexture face, resolved, raw, temporary, filtered;
            private readonly ComputeBuffer receiverBuffer, results;
            private readonly Vector4[] pixels;
            private readonly Vector4[] basesForward, basesRight, basesUp;
            private readonly float cubeYSign;
            private readonly Matrix4x4 projection;
            private readonly int surfaceCount, blurKernel, visibilityKernel, adjointKernel;
            internal int Faces { get; private set; }
            internal int Degenerate { get; private set; }
            internal int Nonfinite { get; private set; }
            internal int LastDegenerate { get; private set; }
            internal int LastNonfinite { get; private set; }
            internal float LastSeamMean { get; private set; }
            internal float LastSeamMax { get; private set; }
            internal int LimitSamples { get; private set; }
            internal double PassSubmitSeconds { get; private set; }
            internal double FilterSubmitSeconds { get; private set; }
            internal double ForwardSyncSeconds { get; private set; }
            internal double BackwardSyncSeconds { get; private set; }
            internal double ProfilePassSeconds { get; private set; }
            internal double ProfileFilterSeconds { get; private set; }
            internal double ProfileForwardSeconds { get; private set; }
            internal long EstimatedGpuBytes => (long)settings.size * settings.size *
                (16 * 6 * 3 + 20 * settings.actualMsaa + (resolved ? 16 : 0)) + 32L * pixels.Length;

            internal PointMomentShadow(ProxyGeometry drawGeometry, ProxyGeometry.Receiver[] samples, MomentSettings config)
                : this(drawGeometry.DrawCasters, samples.Select(s => s.Position).ToArray(), config) { }

            internal PointMomentShadow(Action<CommandBuffer> draw, Vector3[] samples, MomentSettings config)
            {
                drawCasters = draw; settings = config; surfaceCount = samples.Length;
                if (!SystemInfo.supportsComputeShaders || !SystemInfo.supports2DArrayTextures ||
                    !SystemInfo.IsFormatSupported(ProxyGeometry.Format, FormatUsage.LoadStore))
                    throw new NotSupportedException("MSM requires compute and float texture-array load/store.");
                var asset = AssetDatabase.LoadAssetAtPath<ComputeShader>(ComputePath);
                if (!asset) throw new FileNotFoundException(ComputePath);
                var errors = ShaderUtil.GetComputeShaderMessages(asset).Where(m => m.severity.ToString() == "Error").ToArray();
                if (errors.Length != 0)
                    throw new InvalidOperationException("MSM compute compilation failed: " + errors[0].message);
                compute = Object.Instantiate(asset);
                blurKernel = compute.FindKernel("BlurCube"); visibilityKernel = compute.FindKernel("EvaluateVisibility");
                adjointKernel = config.mbd ? compute.FindKernel("EvaluateMbdPositionAdjoint") : -1;
                var regular = Matrix4x4.Perspective(90, 1, settings.near, settings.far);
                projection = GL.GetGPUProjectionMatrix(regular, true);
                cubeYSign = Mathf.Sign(projection.m11 / regular.m11) * (SystemInfo.graphicsUVStartsAtTop ? -1 : 1);
                basesForward = Directions.Select(v => (Vector4)v).ToArray();
                basesUp = Up.Select(v => (Vector4)v).ToArray();
                basesRight = Directions.Select((v, i) => (Vector4)Vector3.Cross(Up[i], v)).ToArray();
                var points = samples.Select(s => (Vector4)s).ToList();
                AddSeamSamples(points);
                pixels = new Vector4[points.Count];
                try
                {
                    var desc = new RenderTextureDescriptor(settings.size, settings.size, ProxyGeometry.Format, 24)
                    { msaaSamples = settings.requestedMsaa, sRGB = false };
                    settings.actualMsaa = SystemInfo.GetRenderTextureSupportedMSAASampleCount(desc);
                    settings.actualMsaa = Math.Max(1, settings.actualMsaa);
                    desc.msaaSamples = settings.actualMsaa;
                    desc.bindMS = settings.actualMsaa > 1;
                    face = MakeTexture(desc, "E20 moment face");
                    if (settings.actualMsaa > 1)
                    {
                        desc.msaaSamples = 1; desc.bindMS = false; desc.depthBufferBits = 0;
                        resolved = MakeTexture(desc, "E20 resolved moment face");
                    }
                    var cube = new RenderTextureDescriptor(settings.size, settings.size, ProxyGeometry.Format, 0)
                    { dimension = TextureDimension.Tex2DArray, volumeDepth = 6, enableRandomWrite = true, sRGB = false };
                    raw = MakeTexture(cube, "E20 radial moment array");
                    temporary = MakeTexture(cube, "E20 horizontal moment filter");
                    filtered = MakeTexture(cube, "E20 filtered radial moment array");
                    receiverBuffer = new ComputeBuffer(points.Count, sizeof(float) * 4);
                    results = new ComputeBuffer(points.Count, sizeof(float) * 4);
                    receiverBuffer.SetData(points);
                }
                catch { Dispose(); throw; }
            }

            private static RenderTexture MakeTexture(RenderTextureDescriptor desc, string name)
            {
                var texture = new RenderTexture(desc) { name = name, filterMode = FilterMode.Point, wrapMode = TextureWrapMode.Clamp,
                    hideFlags = HideFlags.HideAndDontSave };
                if (texture.Create()) return texture;
                Object.DestroyImmediate(texture);
                throw new InvalidOperationException("Could not create " + name);
            }

            private static void AddSeamSamples(List<Vector4> points)
            {
                for (int a = 0; a < 3; a++)
                for (int b = a + 1; b < 3; b++)
                foreach (int signA in new[] { -1, 1 })
                foreach (int signB in new[] { -1, 1 })
                foreach (float edgeOffset in new[] { -.75f, -.45f, -.15f, .15f, .45f, .75f })
                foreach (float radius in new[] { .30f, .50f, .70f })
                {
                    var d = Vector3.zero;
                    d[a] = signA; d[b] = signB; d[3 - a - b] = edgeOffset;
                    d = d.normalized * radius;
                    points.Add(new Vector4(d.x, d.y, d.z, 1));
                }
            }

            internal float[] Render(Vector3 position, bool profile = false) =>
                Render(position, position, position, position, profile);

            internal float[] Render(Vector3 projectionOrigin, Vector3 casterDepthOrigin, Vector3 uvOrigin, Vector3 receiverDepthOrigin, bool profile = false)
            {
                var timer = Stopwatch.StartNew();
                using (var draw = new CommandBuffer { name = "E20 six point-light moment faces" })
                {
                    draw.SetGlobalVector(ProxyGeometry.LightId, new Vector4(casterDepthOrigin.x, casterDepthOrigin.y, casterDepthOrigin.z, 1 / settings.far));
                    for (int i = 0; i < Directions.Length; i++)
                    {
                        var rotation = Quaternion.LookRotation(Directions[i], Up[i]);
                        draw.SetRenderTarget(face);
                        draw.SetViewport(new Rect(0, 0, settings.size, settings.size));
                        draw.ClearRenderTarget(true, true, Color.white);
                        draw.SetGlobalMatrix(ProxyGeometry.ViewProjectionId, projection * ProxyGeometry.View(projectionOrigin, rotation));
                        drawCasters(draw);
                        if (resolved) draw.ResolveAntiAliasedSurface(face, resolved);
                        draw.CopyTexture(resolved ? resolved : face, 0, 0, raw, i, 0);
                    }
                    Graphics.ExecuteCommandBuffer(draw);
                }
                Faces += 6;
                PassSubmitSeconds += timer.Elapsed.TotalSeconds;
                profile &= SystemInfo.supportsAsyncGPUReadback;
                if (profile)
                {
                    WaitForTexture(raw);
                    ProfilePassSeconds = timer.Elapsed.TotalSeconds;
                }
                timer.Restart();
                compute.SetVectorArray("_CubeForward", basesForward);
                compute.SetVectorArray("_CubeRight", basesRight);
                compute.SetVectorArray("_CubeUp", basesUp);
                compute.SetFloat("_CubeYSign", cubeYSign);
                compute.SetInt("_Size", settings.size);
                compute.SetInt("_Radius", settings.blurRadius);
                if (settings.blurRadius > 0)
                {
                    Blur(raw, temporary, false);
                    Blur(temporary, filtered, true);
                }
                FilterSubmitSeconds += timer.Elapsed.TotalSeconds;
                if (profile && settings.blurRadius > 0)
                {
                    WaitForTexture(filtered);
                    ProfileFilterSeconds = timer.Elapsed.TotalSeconds;
                }
                timer.Restart();
                var measured = Query(uvOrigin, receiverDepthOrigin);
                if (profile) ProfileForwardSeconds = timer.Elapsed.TotalSeconds;
                return measured;
            }

            private static void WaitForTexture(RenderTexture texture)
            {
                var request = AsyncGPUReadback.Request(texture, 0, 0, 1, 0, 1, 5, 1);
                request.WaitForCompletion();
                if (request.hasError) throw new InvalidOperationException("Synchronized cubemap timing readback failed.");
            }

            internal float[] Query(Vector3 position) => Query(position, position);

            internal float[] Query(Vector3 uvOrigin, Vector3 receiverDepthOrigin)
            {
                var timer = Stopwatch.StartNew();
                BindQuery(visibilityKernel, uvOrigin, receiverDepthOrigin);
                compute.Dispatch(visibilityKernel, (pixels.Length + 127) / 128, 1, 1);
                results.GetData(pixels);
                ForwardSyncSeconds += timer.Elapsed.TotalSeconds;
                var visibility = new float[surfaceCount];
                LastNonfinite = LastDegenerate = 0;
                LastSeamMean = LastSeamMax = 0;
                for (int i = 0; i < pixels.Length; i++)
                {
                    var p = pixels[i];
                    bool finite = !(float.IsNaN(p.x) || float.IsInfinity(p.x));
                    if (settings.mbd)
                    {
                        if (!finite || p.y >= 5) LastNonfinite++;
                        else if (p.y >= 2) LastDegenerate++;
                        else if (p.y >= 1) LimitSamples++;
                    }
                    else
                    {
                        if (!finite || p.y >= 1.5f) LastNonfinite++;
                        if (p.y >= .5f && p.y < 1.5f) LastDegenerate++;
                    }
                    if (i < surfaceCount) visibility[i] = finite ? Mathf.Clamp01(p.x) : 1;
                    else
                    {
                        LastSeamMean += p.z / (pixels.Length - surfaceCount);
                        LastSeamMax = Mathf.Max(LastSeamMax, p.z);
                    }
                }
                Nonfinite += LastNonfinite;
                Degenerate += LastDegenerate;
                return visibility;
            }

            internal Vector4[] FixedMapPositionAdjoint(Vector3 position)
            {
                if (adjointKernel < 0) throw new InvalidOperationException("This map does not use MBD.");
                var timer = Stopwatch.StartNew();
                BindQuery(adjointKernel, position, position);
                compute.Dispatch(adjointKernel, (pixels.Length + 127) / 128, 1, 1);
                results.GetData(pixels);
                BackwardSyncSeconds += timer.Elapsed.TotalSeconds;
                var adjoint = new Vector4[surfaceCount];
                Array.Copy(pixels, adjoint, surfaceCount);
                return adjoint;
            }

            private void BindQuery(int kernel, Vector3 position, Vector3 receiverDepthOrigin)
            {
                compute.SetTexture(kernel, "_MomentsIn", settings.blurRadius > 0 ? filtered : raw);
                compute.SetVector("_LightAndRange", new Vector4(position.x, position.y, position.z, settings.far));
                compute.SetVector("_ReceiverDepthOrigin", receiverDepthOrigin);
                compute.SetVector("_ClipAndBias", new Vector4(settings.near, settings.depthBiasMetres, 0, 0));
                compute.SetVector("_MomentAndLeakBias", new Vector4(settings.momentBias, settings.leakBias, 0, 0));
                compute.SetInt("_UseVariance", settings.variance ? 1 : 0);
                compute.SetInt("_UseMbd", settings.mbd ? 1 : 0);
                compute.SetFloat("_MbdBlend", settings.overestimationWeight);
                compute.SetInt("_Count", pixels.Length);
                compute.SetBuffer(kernel, "_Receivers", receiverBuffer);
                compute.SetBuffer(kernel, "_Visibility", results);
            }

            private void Blur(RenderTexture source, RenderTexture destination, bool vertical)
            {
                compute.SetTexture(blurKernel, "_MomentsIn", source);
                compute.SetTexture(blurKernel, "_MomentsOut", destination);
                compute.SetInt("_Vertical", vertical ? 1 : 0);
                int groups = (settings.size + 7) / 8;
                compute.Dispatch(blurKernel, groups, groups, 6);
            }

            internal void SaveFaces(string folder, string label)
            {
                int size = settings.size;
                var buffer = RenderTexture.GetTemporary(size, size, 0, ProxyGeometry.Format);
                var cache = new Texture2D(size, size, TextureFormat.RGBAFloat, false, true);
                var preview = new Texture2D(size, size, TextureFormat.RGB24, false, true);
                try
                {
                    foreach (var pair in new[] { (texture: raw, tag: "raw"), (texture: settings.blurRadius > 0 ? filtered : raw, tag: "filtered") })
                    for (int i = 0; i < 6; i++)
                    {
                        Graphics.CopyTexture(pair.texture, i, 0, buffer, 0, 0);
                        var moments = ProxyGeometry.Read(buffer, cache, size, size);
                        string name = Path.Combine(folder, $"{label}_{pair.tag}_{FaceNames[i]}");
                        using (var dump = new BinaryWriter(File.Create(name + ".f32")))
                            foreach (var m in moments) { dump.Write(m.r); dump.Write(m.g); dump.Write(m.b); dump.Write(m.a); }
                        preview.SetPixels(moments.Select(m => settings.mbd ? new Color(.5f + .5f * m.r, m.g, .5f + .5f * m.b) :
                            new Color(m.r, m.g, m.b)).ToArray());
                        preview.Apply();
                        File.WriteAllBytes(name + ".png", preview.EncodeToPNG());
                    }
                }
                finally
                {
                    RenderTexture.ReleaseTemporary(buffer);
                    Object.DestroyImmediate(cache);
                    Object.DestroyImmediate(preview);
                }
            }

            public void Dispose()
            {
                receiverBuffer?.Dispose(); results?.Dispose();
                foreach (var t in new[] { face, resolved, raw, temporary, filtered })
                {
                    if (!t) continue;
                    t.Release(); Object.DestroyImmediate(t);
                }
                if (compute) Object.DestroyImmediate(compute);
            }
        }
    }
}

using System;
using System.Collections.Generic;
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
        private sealed class ProxyGeometry : IDisposable
        {
            private const string PathToShader = "Assets/LightingMvp/Shaders/E20ProxyGeometry.shader";
            internal const int SampleStride = 4;
            internal const int GridWidth = E19FitW / SampleStride, GridHeight = E19FitH / SampleStride;
            internal static readonly int ViewProjectionId = Shader.PropertyToID("_E20ViewProjection");
            internal static readonly int LightId = Shader.PropertyToID("_E20LightPositionInvRange");
            internal static readonly GraphicsFormat Format = GraphicsFormat.R32G32B32A32_SFloat;
            private readonly Material caster;
            private readonly bool includeTextured;
            private readonly List<Material> owned = new List<Material>();
            private readonly List<(MeshRenderer renderer, int slot, Material material)> models =
                new List<(MeshRenderer renderer, int slot, Material material)>();

            internal readonly struct Receiver
            {
                internal readonly Vector3 Position, Normal, Diffuse;
                internal readonly int View, Pixel, Grid;

                internal Receiver(Vector3 position, Vector3 normal, Vector3 diffuse, int view, int pixel, int grid)
                { Position = position; Normal = normal; Diffuse = diffuse; View = view; Pixel = pixel; Grid = grid; }
            }

            internal ProxyGeometry(Camera camera, bool includeTextured = false)
            {
                this.includeTextured = includeTextured;
                var shader = AssetDatabase.LoadAssetAtPath<Shader>(PathToShader);
                if (!shader || !shader.isSupported || ShaderUtil.ShaderHasError(shader))
                    throw new InvalidOperationException("The E20 geometry and moment shader must compile.");
                if (!SystemInfo.IsFormatSupported(Format, FormatUsage.Render))
                    throw new InvalidOperationException("E20 requires floating-point render targets.");
                var lookup = new Dictionary<Material, Material>();
                caster = NewMaterial(shader);
                try
                {
                    foreach (var renderer in Object.FindObjectsOfType<MeshRenderer>().OrderBy(r => r.name, StringComparer.Ordinal))
                    {
                        if (!renderer.enabled || (camera.cullingMask & (1 << renderer.gameObject.layer)) == 0) continue;
                        var filter = renderer.GetComponent<MeshFilter>();
                        if (!filter || !filter.sharedMesh) continue;
                        var materials = renderer.sharedMaterials;
                        for (int slot = 0; slot < filter.sharedMesh.subMeshCount; slot++)
                        {
                            var source = materials[Math.Min(slot, materials.Length - 1)];
                            bool supported = source && source.shader.name == "Universal Render Pipeline/Lit" &&
                                source.renderQueue <= (int)RenderQueue.GeometryLast && !source.IsKeywordEnabled("_ALPHATEST_ON");
                            if (includeTextured && !supported) continue;
                            if (!supported || !includeTextured && (source.IsKeywordEnabled("_NORMALMAP") ||
                                source.IsKeywordEnabled("_EMISSION") || (source.GetTexture("_BaseMap") is Texture map && map != Texture2D.whiteTexture)))
                                throw new InvalidOperationException($"Unsupported E20 surface {renderer.name}: expected E19 opaque untextured URP Lit.");
                            if (!lookup.TryGetValue(source, out var material))
                            {
                                material = NewMaterial(shader);
                                material.SetColor("_BaseColor", source.GetColor("_BaseColor"));
                                material.SetFloat("_Metallic", source.GetFloat("_Metallic"));
                                if (includeTextured) material.SetFloat("_Cull", source.GetFloat("_Cull"));
                                lookup.Add(source, material);
                            }
                            models.Add((renderer, slot, material));
                        }
                    }
                    if (models.Count == 0) throw new InvalidOperationException("No E19 geometry found for the proxies.");
                }
                catch { Dispose(); throw; }
            }

            private Material NewMaterial(Shader shader)
            {
                var material = new Material(shader) { hideFlags = HideFlags.HideAndDontSave };
                owned.Add(material);
                return material;
            }

            internal Receiver[] Capture(ISensitivityScene scene, string dir)
            {
                if (SystemInfo.supportedRenderTargetCount < 3) throw new NotSupportedException("Receiver sampling requires three MRT attachments.");
                var receivers = new List<Receiver>();
                var buffers = new RenderTexture[3];
                var readback = new Texture2D(E19FitW, E19FitH, TextureFormat.RGBAFloat, false, true);
                try
                {
                    for (int i = 0; i < buffers.Length; i++) buffers[i] = RenderTexture.GetTemporary(E19FitW, E19FitH, i == 0 ? 24 : 0, Format);
                    var targets = buffers.Select(r => new RenderTargetIdentifier(r)).ToArray();
                    var views = scene.Views;
                    var rotations = scene.Rotations;
                    var camera = scene.Camera;
                    float aspect = E19FitW / (float)E19FitH;
                    using var csv = new StreamWriter(Path.Combine(dir, "receiver-views.csv"));
                    csv.WriteLine("view,samples,grid,vertical_flip,max_projection_error_pixels");
                    for (int v = 0; v < views.Length; v++)
                    {
                        var eye = views[v];
                        var rotation = rotations[v];
                        var projection = Matrix4x4.Perspective(camera.fieldOfView, aspect, camera.nearClipPlane, camera.farClipPlane);
                        var view = View(eye, rotation);
                        using (var draw = new CommandBuffer { name = "E20 fixed camera receiver attributes" })
                        {
                            draw.SetRenderTarget(targets, targets[0]);
                            draw.SetViewport(new Rect(0, 0, E19FitW, E19FitH));
                            draw.ClearRenderTarget(true, true, Color.clear);
                            draw.SetGlobalMatrix(ViewProjectionId, GL.GetGPUProjectionMatrix(projection, true) * view);
                            foreach (var model in models) draw.DrawRenderer(model.renderer, model.material, model.slot, 0);
                            Graphics.ExecuteCommandBuffer(draw);
                        }
                        var pos = Read(buffers[0], readback, E19FitW, E19FitH);
                        var normals = Read(buffers[1], readback, E19FitW, E19FitH);
                        var diffuse = Read(buffers[2], readback, E19FitW, E19FitH);
                        var screen = projection * view;
                        // Readback origin depends on the graphics API and the render-to-texture projection.
                        // Verify registration in native camera pixel coordinates using the visible world positions.
                        double directError = 0, flippedError = 0;
                        for (int p = 103; p < pos.Length; p += 227)
                        {
                            if (pos[p].a < .5f) continue;
                            var q = ProjectPixel(screen, pos[p]);
                            directError += Math.Abs(q.y - p / E19FitW);
                            flippedError += Math.Abs(q.y - (E19FitH - 1 - p / E19FitW));
                        }
                        bool flip = flippedError < directError;
                        double maxError = 0;
                        int start = receivers.Count;
                        for (int j = 0; j < GridHeight; j++)
                        for (int i = 0; i < GridWidth; i++)
                        {
                            int x = i * SampleStride + SampleStride / 2, y = j * SampleStride + SampleStride / 2;
                            int rawY = flip ? E19FitH - 1 - y : y;
                            int index = rawY * E19FitW + x;
                            if (pos[index].a < .5f) continue;
                            var q = ProjectPixel(screen, pos[index]);
                            maxError = Math.Max(maxError, Math.Max(Math.Abs(q.x - x), Math.Abs(q.y - y)));
                            receivers.Add(new Receiver(Rgb(pos[index]), Rgb(normals[index]), Rgb(diffuse[index]), v,
                                y * E19FitW + x, j * GridWidth + i));
                        }
                        int count = receivers.Count - start;
                        if (count < (includeTextured ? 64 : GridWidth * GridHeight / 2) || maxError > .05)
                            throw new InvalidOperationException($"E20 receiver coverage or registration failed at view {v}: {count} sites, {maxError:F3}px.");
                        csv.WriteLine(string.Join(",", v, count, GridWidth * GridHeight, flip ? 1 : 0, F(maxError)));
                    }
                }
                finally
                {
                    foreach (var buffer in buffers) if (buffer) RenderTexture.ReleaseTemporary(buffer);
                    Object.DestroyImmediate(readback);
                }
                using var samples = new StreamWriter(Path.Combine(dir, "receivers.csv"));
                samples.WriteLine("index,view,pixel,grid,x,y,z,nx,ny,nz,diffuse_r,diffuse_g,diffuse_b");
                for (int i = 0; i < receivers.Count; i++)
                {
                    var r = receivers[i];
                    Vector3 p = r.Position, n = r.Normal, d = r.Diffuse;
                    samples.WriteLine(string.Join(",", i, r.View, r.Pixel, r.Grid, F(p.x), F(p.y), F(p.z), F(n.x), F(n.y), F(n.z), F(d.x), F(d.y), F(d.z)));
                }
                return receivers.ToArray();
            }

            internal void DrawCasters(CommandBuffer draw)
            {
                foreach (var model in models)
                    if (model.renderer.shadowCastingMode != ShadowCastingMode.Off)
                        draw.DrawRenderer(model.renderer, caster, model.slot, 1);
            }

            internal static Matrix4x4 View(Vector3 eye, Quaternion rotation) =>
                Matrix4x4.Scale(new Vector3(1, 1, -1)) * Matrix4x4.TRS(eye, rotation, Vector3.one).inverse;

            private static Vector2 ProjectPixel(Matrix4x4 transform, Color p)
            {
                var clip = transform * new Vector4(p.r, p.g, p.b, 1);
                return new Vector2((clip.x / clip.w + 1) * E19FitW * .5f - .5f, (clip.y / clip.w + 1) * E19FitH * .5f - .5f);
            }

            internal static Vector3 Rgb(Color color) => new Vector3(color.r, color.g, color.b);

            internal static Color[] Read(RenderTexture texture, Texture2D cache, int width, int height)
            {
                var last = RenderTexture.active;
                try
                {
                    RenderTexture.active = texture;
                    cache.ReadPixels(new Rect(0, 0, width, height), 0, 0, false);
                    return cache.GetRawTextureData<Color>().ToArray();
                }
                finally { RenderTexture.active = last; }
            }

            public void Dispose()
            {
                foreach (var material in owned) if (material) Object.DestroyImmediate(material);
            }
        }
    }
}

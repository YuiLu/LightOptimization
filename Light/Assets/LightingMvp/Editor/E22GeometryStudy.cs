using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using Frame = LightingMvp.NativeRenderBackend.Frame;
using HdrMapping = LightingMvp.NativeRenderBackend.HdrMapping;
using Object = UnityEngine.Object;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            private sealed class E22GeometryView
            {
                internal readonly Vector3[] position = new Vector3[E19FitW * E19FitH];
                internal readonly Vector3[] normal = new Vector3[E19FitW * E19FitH];
                internal readonly int[] renderer = new int[E19FitW * E19FitH];
                internal readonly int[] material = new int[E19FitW * E19FitH];
                internal readonly bool[] reliable = new bool[E19FitW * E19FitH];
                internal readonly bool[] boundary = new bool[E19FitW * E19FitH];
            }

            private sealed class E22GeometryData
            {
                internal E22GeometryView[] views;
                internal MeshRenderer[] renderers;
                internal Material[] materials;
            }

            private E22GeometryData CaptureE22Geometry()
            {
                var sw = Stopwatch.StartNew();
                var shader = AssetDatabase.LoadAssetAtPath<Shader>("Assets/LightingMvp/Shaders/E20ProxyGeometry.shader");
                if (!shader || !shader.isSupported || ShaderUtil.ShaderHasError(shader))
                    throw new InvalidOperationException("E22 requires the compiled geometry pass.");
                var visible = Object.FindObjectsOfType<MeshRenderer>().Where(r => r.enabled && (room.Camera.cullingMask & (1 << r.gameObject.layer)) != 0)
                    .OrderBy(r => r.name, StringComparer.Ordinal).ThenBy(r => r.transform.position.x).ThenBy(r => r.transform.position.y)
                    .ThenBy(r => r.transform.position.z).ToArray();
                var materials = new List<Material>();
                var overrides = new Dictionary<Material, Material>();
                var draws = new List<(MeshRenderer renderer, int slot, Material raster, int rendererId, int materialId)>();
                var buffers = new RenderTexture[2];
                Texture2D readback = null;
                var result = new E22GeometryView[E19FitViews];
                try
                {
                    for (int r = 0; r < visible.Length; ++r)
                    {
                        var renderer = visible[r];
                        var filter = renderer.GetComponent<MeshFilter>();
                        var slots = renderer.sharedMaterials;
                        if (!filter || !filter.sharedMesh || slots.Length == 0 || renderer.shadowCastingMode == ShadowCastingMode.ShadowsOnly) continue;
                        for (int sub = 0; sub < filter.sharedMesh.subMeshCount; ++sub)
                        {
                            var source = slots[Math.Min(sub, slots.Length - 1)];
                            if (!source || source.shader.name != "Universal Render Pipeline/Lit" || source.renderQueue > (int)RenderQueue.GeometryLast ||
                                source.IsKeywordEnabled("_ALPHATEST_ON")) continue;
                            if (!overrides.TryGetValue(source, out var raster))
                            {
                                raster = new Material(shader) { hideFlags = HideFlags.HideAndDontSave };
                                raster.SetInt("_Cull", source.HasProperty("_Cull") ? source.GetInt("_Cull") : (int)CullMode.Back);
                                overrides.Add(source, raster);
                                materials.Add(source);
                            }
                            draws.Add((renderer, sub, raster, r + 1, materials.IndexOf(source) + 1));
                        }
                    }
                    using (var file = Csv("geometry-renderers.csv", "renderer,name"))
                        for (int i = 0; i < visible.Length; ++i) file.WriteLine($"{i + 1},{E22CsvName(visible[i].name)}");
                    using (var file = Csv("geometry-materials.csv", "material,name,metallic,smoothness"))
                        for (int i = 0; i < materials.Count; ++i)
                            file.WriteLine(string.Join(",", i + 1, E22CsvName(materials[i].name), F(materials[i].GetFloat("_Metallic")), F(materials[i].GetFloat("_Smoothness"))));
                    using var registration = Csv("geometry-registration.csv",
                        "view,pixels,native_covered,geometry_covered,reliable,depth_outliers,flip_geometry,flip_depth,max_reprojection_px,max_valid_depth_error_m,boundary_pixels");
                    for (int i = 0; i < 2; ++i)
                        buffers[i] = RenderTexture.GetTemporary(E19FitW, E19FitH, i == 0 ? 24 : 0, ProxyGeometry.Format);
                    var targets = buffers.Select(rt => new RenderTargetIdentifier(rt)).ToArray();
                    readback = new Texture2D(E19FitW, E19FitH, TextureFormat.RGBAFloat, false, true);
                    using var depthCapture = new NativeIrradianceCapture(room.Camera, E19FitW, E19FitH, depthOnly: true);
                    bool[] wasEnabled = lights.Select(l => l.enabled).ToArray();
                    foreach (var light in lights) light.enabled = false;
                    try
                    {
                        for (int v = 0; v < E19FitViews; ++v)
                        {
                            var eye = room.Views[v]; var rotation = room.Rotations[v];
                            var view = ProxyGeometry.View(eye, rotation);
                            var projection = Matrix4x4.Perspective(room.Camera.fieldOfView, (float)E19FitW / E19FitH, room.Camera.nearClipPlane, room.Camera.farClipPlane);
                            using (var draw = new CommandBuffer { name = "E22 dense geometry registration" })
                            {
                                draw.SetRenderTarget(targets, targets[0]);
                                draw.SetViewport(new Rect(0, 0, E19FitW, E19FitH));
                                draw.ClearRenderTarget(true, true, Color.clear);
                                draw.SetGlobalMatrix(ProxyGeometry.ViewProjectionId, GL.GetGPUProjectionMatrix(projection, true) * view);
                                foreach (var item in draws)
                                {
                                    draw.SetGlobalVector("_E22GeometryIds", new Vector4(item.rendererId, item.materialId, 0, 0));
                                    draw.DrawRenderer(item.renderer, item.raster, item.slot, 2);
                                }
                                Graphics.ExecuteCommandBuffer(draw);
                            }
                            var positions = ProxyGeometry.Read(buffers[0], readback, E19FitW, E19FitH);
                            var normals = ProxyGeometry.Read(buffers[1], readback, E19FitW, E19FitH);
                            var depth = depthCapture.ReadViewDepth(eye, rotation);
                            geometryPasses += 2;
                            var map = result[v] = new E22GeometryView();
                            var screen = projection * view;
                            double direct = 0, flip = 0;
                            for (int i = 103; i < positions.Length; i += 227)
                            {
                                if (positions[i].a < .5f) continue;
                                double y = Project(positions[i]).y;
                                direct += Math.Abs(y - i / E19FitW);
                                flip += Math.Abs(y - (E19FitH - 1 - i / E19FitW));
                            }
                            bool flipGeometry = flip < direct;
                            int depthDirect = 0, depthFlipped = 0;
                            for (int i = 53; i < positions.Length; i += 71)
                            {
                                var p = positions[flipGeometry ? FlipY(i) : i];
                                if (p.a < .5f) continue;
                                float expected = -view.MultiplyPoint3x4(ProxyGeometry.Rgb(p)).z;
                                if (CompatibleDepth(expected, depth[i].r)) ++depthDirect;
                                if (CompatibleDepth(expected, depth[FlipY(i)].r)) ++depthFlipped;
                            }
                            bool flipDepth = depthFlipped > depthDirect;
                            int native = 0, geometry = 0, reliable = 0, misses = 0;
                            double maxProjection = 0, maxDepth = 0;
                            for (int i = 0; i < positions.Length; ++i)
                            {
                                int source = flipGeometry ? FlipY(i) : i;
                                float d = depth[flipDepth ? FlipY(i) : i].r;
                                if (d > 0) ++native;
                                var p = positions[source];
                                if (p.a < .5f) continue;
                                ++geometry;
                                var q = Project(p);
                                float px = Mathf.Max(Mathf.Abs(q.x - i % E19FitW), Mathf.Abs(q.y - i / E19FitW));
                                maxProjection = Math.Max(maxProjection, px);
                                map.position[i] = ProxyGeometry.Rgb(p);
                                map.normal[i] = ProxyGeometry.Rgb(normals[source]);
                                map.renderer[i] = Mathf.RoundToInt(p.a);
                                map.material[i] = Mathf.RoundToInt(normals[source].a);
                                float reference = -view.MultiplyPoint3x4(map.position[i]).z;
                                if (!CompatibleDepth(reference, d)) { ++misses; continue; }
                                if (px > .075f || map.normal[i].sqrMagnitude < .5f) continue;
                                ++reliable;
                                map.reliable[i] = true;
                                maxDepth = Math.Max(maxDepth, Math.Abs(reference - d));
                            }
                            for (int i = 0; i < positions.Length; ++i)
                            {
                                if (!map.reliable[i]) continue;
                                int x = i % E19FitW, y = i / E19FitW;
                                map.boundary[i] = x > 0 && Break(i, i - 1) || x + 1 < E19FitW && Break(i, i + 1) ||
                                    y > 0 && Break(i, i - E19FitW) || y + 1 < E19FitH && Break(i, i + E19FitW);
                            }
                            registration.WriteLine(string.Join(",", v, positions.Length, native, geometry, reliable, misses, Flag(flipGeometry), Flag(flipDepth),
                                F(maxProjection), F(maxDepth), map.boundary.Count(b => b)));

                            Vector2 Project(Color p)
                            {
                                var q = screen * new Vector4(p.r, p.g, p.b, 1);
                                return new Vector2((q.x / q.w + 1) * E19FitW * .5f - .5f, (q.y / q.w + 1) * E19FitH * .5f - .5f);
                            }
                            bool Break(int a, int b) => !map.reliable[b] || map.renderer[a] != map.renderer[b] || map.material[a] != map.material[b] ||
                                Vector3.Dot(map.normal[a], map.normal[b]) < .9f || (map.position[a] - map.position[b]).sqrMagnitude > .06f * .06f;
                        }
                    }
                    finally { for (int i = 0; i < lights.Length; ++i) lights[i].enabled = wasEnabled[i]; }
                }
                finally
                {
                    foreach (var buffer in buffers) if (buffer) RenderTexture.ReleaseTemporary(buffer);
                    if (readback) Object.DestroyImmediate(readback);
                    foreach (var material in overrides.Values) Object.DestroyImmediate(material);
                    geometrySeconds += sw.Elapsed.TotalSeconds;
                }
                return new E22GeometryData { views = result, renderers = visible, materials = materials.ToArray() };

                static int FlipY(int index) => (E19FitH - 1 - index / E19FitW) * E19FitW + index % E19FitW;
                static bool CompatibleDepth(float expected, float measured) => measured > 0 && Math.Abs(expected - measured) <= Math.Max(.001f, .001f * expected);
            }

            private static string E22CsvName(string name) => name.Replace(',', '_').Replace('\r', ' ').Replace('\n', ' ');

            private sealed class E22PatchView
            {
                internal readonly int[] pixelGroup;
                internal readonly int[] sizes;
                internal readonly float[] weights;
                internal E22PatchView(E22GeometryView view)
                {
                    var groups = new Dictionary<(int r, int m, Vector3Int cell, Vector3Int normal), List<int>>();
                    for (int i = 0; i < view.position.Length; ++i)
                    {
                        if (!view.reliable[i]) continue;
                        var p = view.position[i] * 4; var n = view.normal[i] * 4;
                        var key = (view.renderer[i], view.material[i], Vector3Int.FloorToInt(p), Vector3Int.RoundToInt(n));
                        if (!groups.TryGetValue(key, out var pixels)) groups.Add(key, pixels = new List<int>());
                        pixels.Add(i);
                    }
                    var patches = groups.Values.Where(v => v.Count >= 4).ToArray();
                    sizes = patches.Select(p => p.Count).ToArray();
                    pixelGroup = Enumerable.Repeat(-1, view.position.Length).ToArray();
                    weights = Enumerable.Repeat(1f, view.position.Length).ToArray();
                    if (patches.Length == 0) return;
                    var unscaled = sizes.Select(n => 1 / Math.Sqrt(n)).ToArray();
                    int weightedPixels = sizes.Sum();
                    // Normalize over patch-covered pixels, while retaining exactly unit weight elsewhere.
                    // Projection onto [.25,4] and mean one is monotone in the common scale.
                    double lo = 0, hi = 4 / unscaled.Min();
                    for (int iter = 0; iter < 32; ++iter)
                    {
                        double gain = (lo + hi) * .5, total = 0;
                        for (int j = 0; j < sizes.Length; ++j) total += sizes[j] * Bound(gain * unscaled[j]);
                        if (total < weightedPixels) lo = gain; else hi = gain;
                    }
                    for (int g = 0; g < patches.Length; ++g)
                    foreach (int i in patches[g])
                    {
                        pixelGroup[i] = g;
                        weights[i] = (float)Bound((lo + hi) * .5 * unscaled[g]);
                    }

                    static double Bound(double value) => Math.Max(.25, Math.Min(4, value));
                }
            }

            // Frame-to-residual evaluation also serves the offline hotspot Jacobian audit.
            private sealed class E22FrameLoss
            {
                internal readonly int size;
                private readonly bool huber;
                private readonly HdrMapping mode;
                private readonly double scale, delta, normalizer, meanNormalizer;
                private readonly Vector3[] reference, groupResidual;
                private readonly E22PatchView patches;

                internal E22FrameLoss(Frame target, string key, double scale, E22PatchView patches = null, bool mean = false)
                {
                    this.patches = patches; this.scale = scale;
                    huber = key == "B5"; mode = E22Mapping(key);
                    reference = Enumerable.Range(0, target.PixelCount).Select(i => Transform(target.Pixel(i))).ToArray();
                    double energy = 0;
                    for (int i = 0; i < reference.Length; ++i) energy += Norm(reference[i]) * (patches?.weights[i] ?? 1);
                    bool hasMeans = mean && patches?.sizes.Length > 0;
                    normalizer = (hasMeans ? .8 : 1) / (E19FitViews * Math.Max(1e-12, energy));
                    delta = Math.Max(1e-8, .25 * Math.Sqrt(target.Energy / target.PixelCount));
                    if (hasMeans)
                    {
                        groupResidual = new Vector3[patches.sizes.Length];
                        for (int i = 0; i < reference.Length; ++i) if (patches.pixelGroup[i] >= 0) groupResidual[patches.pixelGroup[i]] += reference[i];
                        double patchEnergy = 0;
                        for (int p = 0; p < groupResidual.Length; ++p) patchEnergy += Norm(groupResidual[p] / patches.sizes[p]);
                        meanNormalizer = .2 / (E19FitViews * Math.Max(1e-12, patchEnergy));
                    }
                    size = (target.PixelCount + (groupResidual?.Length ?? 0)) * 3;
                }

                internal double Evaluate(Frame image, float[] output, int offset, double[] costs = null)
                {
                    double sum = 0;
                    if (groupResidual != null) Array.Clear(groupResidual, 0, groupResidual.Length);
                    for (int i = 0; i < reference.Length; ++i)
                    {
                        var d = Transform(image.Pixel(i)) - reference[i];
                        double square = Norm(d);
                        double cost = huber && square > delta * delta ? 2 * delta * Math.Sqrt(square) - delta * delta : square;
                        double norm = normalizer * (patches?.weights[i] ?? 1);
                        if (output != null) Put(d, square > 0 ? norm * cost / square : norm, offset + 3 * i);
                        if (costs != null) costs[i] = cost * norm;
                        if (groupResidual != null && patches.pixelGroup[i] >= 0) groupResidual[patches.pixelGroup[i]] += d;
                        sum += cost * norm;
                    }
                    if (groupResidual == null) return sum;
                    for (int i = 0; i < groupResidual.Length; ++i)
                    {
                        var d = groupResidual[i] / patches.sizes[i];
                        if (output != null) Put(d, meanNormalizer, offset + 3 * (reference.Length + i));
                        sum += Norm(d) * meanNormalizer;
                    }
                    return sum;

                    void Put(Vector3 rgb, double gain, int k)
                    {
                        float s = (float)Math.Sqrt(gain);
                        output[k] = rgb.x * s; output[k + 1] = rgb.y * s; output[k + 2] = rgb.z * s;
                    }
                }

                private Vector3 Transform(Color p) => mode == HdrMapping.Raw ? ProxyGeometry.Rgb(p) : new Vector3(
                    NativeRenderBackend.Map(p.r, mode, scale), NativeRenderBackend.Map(p.g, mode, scale), NativeRenderBackend.Map(p.b, mode, scale));
            }

            private static double Norm(Vector3 rgb) => (double)rgb.x * rgb.x + (double)rgb.y * rgb.y + (double)rgb.z * rgb.z;

            private void E22GeometryValidate(bool mean, bool shadows, int[] seeds)
            {
                var data = CaptureE22Geometry();
                var patches = data.views.Select(v => new E22PatchView(v)).ToArray();
                using (var csv = Csv("patch-weights.csv", "view,patches,reliable_pixels,weighted_pixels,fallback_pixels,min_weight,p50_weight,p95_weight,max_weight,mean_weight"))
                    for (int v = 0; v < patches.Length; ++v)
                    {
                        var p = patches[v]; var w = p.weights.OrderBy(x => x).ToArray();
                        csv.WriteLine(string.Join(",", v, p.sizes.Length, data.views[v].reliable.Count(x => x), p.sizes.Sum(), w.Length - p.sizes.Sum(),
                            F(w[0]), F(w[w.Length / 2]), F(w[(int)(.95 * w.Length)]), F(w[w.Length - 1]), F(w.Average(x => (double)x))));
                    }
                var gt = E22Target(shadows); double scale = E22Scale(gt);
                string mapping = Arg("-bbE22Mapping", "B0");
                var views = gt.fit.Select((f, v) => new E22FrameLoss(f, mapping, scale, patches[v], mean)).ToArray();
                var signal = new E21Signal(mapping + (mean ? "_patch_mean" : "_patch_weight"), views.Sum(v => v.size), (rig, residual) =>
                {
                    Apply(rig, shadows);
                    double sum = 0;
                    for (int v = 0, offset = 0; v < views.Length; offset += views[v].size, ++v)
                        sum += views[v].Evaluate(low.Capture(v), residual, offset);
                    return sum;
                });
                foreach (int seed in smoke ? seeds.Take(1) : seeds)
                    RunE22Arm($"seed{seed}_near_{(shadows ? "on" : "off")}", mean ? "C2" : "C1", room.Lamps, E22Start(seed, room.Lamps),
                        gt, signal, 1575, E22Solver(Arg("-bbE22Solver", "A0")), mapping, scale, mean ? "patch_mean" : "patch_weight");
            }
        }
    }
}

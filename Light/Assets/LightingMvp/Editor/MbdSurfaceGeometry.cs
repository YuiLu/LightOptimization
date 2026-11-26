using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using Object = UnityEngine.Object;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private readonly struct MbdSurface
        {
            internal readonly Vector3 position, normal, albedo;
            internal readonly int material;
            internal readonly float area;
            internal MbdSurface(Vector3 p, Vector3 n, Vector3 albedo, int material, float area)
            { position = p; normal = n; this.albedo = albedo; this.material = material; this.area = area; }
        }

        private sealed class MbdSurfaceGeometry : IDisposable
        {
            private readonly List<(MeshRenderer renderer, Mesh mesh, int slot, Material source, Material caster)> draws =
                new List<(MeshRenderer, Mesh, int, Material, Material)>();
            private readonly Dictionary<(Material material, ShadowCastingMode mode), Material> materials =
                new Dictionary<(Material, ShadowCastingMode), Material>();
            private RayTriangle[] rayTriangles;
            private RayNode rayTree;
            internal Bounds Bounds { get; private set; }
            internal int Renderers => draws.Select(x => x.renderer).Distinct().Count();
            internal int AlphaClipSlots => draws.Count(r => r.source.IsKeywordEnabled("_ALPHATEST_ON"));

            internal MbdSurfaceGeometry(Camera camera)
            {
                var shader = AssetDatabase.LoadAssetAtPath<Shader>("Assets/LightingMvp/Shaders/MbdMomentCaster.shader");
                if (!shader || ShaderUtil.ShaderHasError(shader)) throw new InvalidOperationException("MBD caster shader unavailable.");
                try
                {
                    foreach (var r in Object.FindObjectsOfType<MeshRenderer>().OrderBy(r => r.name, StringComparer.Ordinal))
                    {
                        if (!r.enabled || (camera.cullingMask & (1 << r.gameObject.layer)) == 0) continue;
                        var mesh = r.GetComponent<MeshFilter>()?.sharedMesh;
                        var original = r.sharedMaterials;
                        if (!mesh || original.Length == 0) continue;
                        for (int i = 0; i < mesh.subMeshCount; ++i)
                        {
                            var source = original[Math.Min(i, original.Length - 1)];
                            if (!source || source.renderQueue > (int)RenderQueue.GeometryLast) continue;
                            if (draws.Count == 0) Bounds = r.bounds;
                            else { var bounds = Bounds; bounds.Encapsulate(r.bounds); Bounds = bounds; }
                            var key = (source, r.shadowCastingMode);
                            if (!materials.TryGetValue(key, out Material material))
                            {
                                material = new Material(shader) { hideFlags = HideFlags.HideAndDontSave };
                                materials.Add(key, material);
                                material.SetInt("_Cull", r.shadowCastingMode == ShadowCastingMode.TwoSided ? 0 :
                                    source.HasProperty("_Cull") ? source.GetInt("_Cull") : 2);
                                if (source.IsKeywordEnabled("_ALPHATEST_ON") && source.HasProperty("_BaseMap"))
                                {
                                    material.SetFloat("_AlphaClip", 1);
                                    material.SetTexture("_BaseMap", source.GetTexture("_BaseMap"));
                                    material.SetTextureOffset("_BaseMap", source.GetTextureOffset("_BaseMap"));
                                    material.SetTextureScale("_BaseMap", source.GetTextureScale("_BaseMap"));
                                    material.SetFloat("_Cutoff", source.GetFloat("_Cutoff"));
                                    material.SetColor("_BaseColor", source.GetColor("_BaseColor"));
                                }
                            }
                            draws.Add((r, mesh, i, source, material));
                        }
                    }
                    if (draws.Count == 0) throw new InvalidOperationException("No render meshes in the MBD scene.");
                }
                catch { Dispose(); throw; }
            }

            internal void DrawCasters(CommandBuffer draw)
            {
                foreach (var item in draws)
                    if (item.renderer.enabled && item.renderer.shadowCastingMode != ShadowCastingMode.Off)
                        draw.DrawRenderer(item.renderer, item.caster, item.slot, 0);
            }

            private readonly struct Triangle
            {
                internal readonly Vector3 a, b, c, na, nb, nc, color;
                internal readonly int material;
                internal readonly double end;
                internal Triangle(Vector3 a, Vector3 b, Vector3 c, Vector3 na, Vector3 nb, Vector3 nc, Vector3 color, int material, double end)
                { this.a = a; this.b = b; this.c = c; this.na = na; this.nb = nb; this.nc = nc; this.color = color; this.material = material; this.end = end; }
            }

            internal MbdSurface[] Samples(int count, string directory)
            {
                var triangles = new List<Triangle>();
                var ids = new Dictionary<Material, int>();
                double area = 0;
                foreach (var item in draws)
                {
                    if (item.renderer.shadowCastingMode == ShadowCastingMode.ShadowsOnly) continue;
                    if (!ids.ContainsKey(item.source)) ids.Add(item.source, ids.Count);
                    var tint = item.source.HasProperty("_BaseColor") ? item.source.GetColor("_BaseColor").linear : Color.white;
                    var vertices = item.mesh.vertices;
                    var normals = item.mesh.normals;
                    var indices = item.mesh.GetTriangles(item.slot);
                    var toWorld = item.renderer.localToWorldMatrix;
                    var normalMatrix = toWorld.inverse.transpose;
                    for (int i = 0; i < indices.Length; i += 3)
                    {
                        int ai = indices[i], bi = indices[i + 1], ci = indices[i + 2];
                        Vector3 a = toWorld.MultiplyPoint3x4(vertices[ai]), b = toWorld.MultiplyPoint3x4(vertices[bi]), c = toWorld.MultiplyPoint3x4(vertices[ci]);
                        Vector3 rawNormal = Vector3.Cross(b - a, c - a);
                        float size = .5f * rawNormal.magnitude;
                        if (size < 1e-10f) continue;
                        area += size;
                        var na = normals.Length == vertices.Length ? normalMatrix.MultiplyVector(normals[ai]) : rawNormal;
                        var nb = normals.Length == vertices.Length ? normalMatrix.MultiplyVector(normals[bi]) : rawNormal;
                        var nc = normals.Length == vertices.Length ? normalMatrix.MultiplyVector(normals[ci]) : rawNormal;
                        triangles.Add(new Triangle(a, b, c, na.normalized, nb.normalized, nc.normalized, ProxyGeometry.Rgb(tint), ids[item.source], area));
                    }
                }
                if (!(area > 0)) throw new InvalidOperationException("No nondegenerate surface triangles.");
                var samples = new MbdSurface[count];
                int t = 0;
                using (var csv = new StreamWriter(Path.Combine(directory, "surface-samples.csv")))
                {
                    csv.WriteLine("sample,x,y,z,nx,ny,nz,area_m2,material,albedo_r,albedo_g,albedo_b");
                    var rng = new System.Random(613);
                    for (int i = 0; i < count; ++i)
                    {
                        double quantile = (i + .5) * area / count;
                        while (t < triangles.Count - 1 && triangles[t].end < quantile) ++t;
                        var v = triangles[t];
                        float s = Mathf.Sqrt((float)rng.NextDouble()), w2 = s * (float)rng.NextDouble(), w1 = s - w2, w0 = 1 - s;
                        Vector3 position = v.a * w0 + v.b * w1 + v.c * w2;
                        Vector3 normal = (v.na * w0 + v.nb * w1 + v.nc * w2).normalized;
                        samples[i] = new MbdSurface(position, normal, v.color, v.material, (float)(area / count));
                        csv.WriteLine(string.Join(",", i, Xyz(position), Xyz(normal), F(area / count), v.material, Xyz(v.color)));
                    }
                }
                using (var csv = new StreamWriter(Path.Combine(directory, "surface-materials.csv")))
                {
                    csv.WriteLine("material,name,alpha_clip");
                    foreach (var p in ids) csv.WriteLine(string.Join(",", p.Value, p.Key.name.Replace(',', '_'), p.Key.IsKeywordEnabled("_ALPHATEST_ON") ? 1 : 0));
                }
                return samples;
            }

            internal void OpenRayReference()
            {
                if (rayTriangles != null) return;
                var triangles = new List<RayTriangle>();
                foreach (var item in draws)
                {
                    if (item.renderer.shadowCastingMode == ShadowCastingMode.Off) continue;
                    var indices = item.mesh.GetTriangles(item.slot);
                    var toWorld = item.renderer.localToWorldMatrix;
                    var vertices = item.mesh.vertices.Select(toWorld.MultiplyPoint3x4).ToArray();
                    for (int i = 0; i < indices.Length; i += 3)
                        triangles.Add(new RayTriangle(vertices[indices[i]], vertices[indices[i + 1]], vertices[indices[i + 2]]));
                }
                rayTriangles = triangles.ToArray();
                if (rayTriangles.Length != 0) rayTree = new RayNode(rayTriangles, 0, rayTriangles.Length);
            }

            internal float RayVisibility(Vector3 receiver, Vector3 light)
            {
                var offset = light - receiver;
                float distance = offset.magnitude;
                if (distance <= .004f) return 1;
                var direction = offset / distance;
                var ray = new Ray(receiver + direction * .002f, direction);
                return rayTree != null && rayTree.Intersects(rayTriangles, ray, distance - .004f) ? 0 : 1;
            }

            internal float FilteredRayVisibility(Vector3 receiver, Vector3 light, float angularRadius, int count)
            {
                if (count == 1) return RayVisibility(receiver, light);
                var direction = (light - receiver).normalized;
                var right = Vector3.Cross(Mathf.Abs(direction.y) < .9f ? Vector3.up : Vector3.right, direction).normalized;
                var up = Vector3.Cross(direction, right);
                float radius = angularRadius * Vector3.Distance(receiver, light);
                int visible = 0;
                for (int i = 0; i < count; ++i)
                {
                    float theta = 2.39996323f * i;
                    float r = radius * Mathf.Sqrt((i + .5f) / count);
                    var sample = receiver + r * (Mathf.Cos(theta) * right + Mathf.Sin(theta) * up);
                    visible += (int)RayVisibility(sample, light);
                }
                return visible / (float)count;
            }

            internal float FilteredRadialRayVisibility(Vector3 receiver, Vector3 light, float angularRadius, int count,
                float near, float far, float depthBias)
            {
                var delta = receiver - light;
                float distance = delta.magnitude;
                if (rayTree == null || distance <= near || distance >= far) return 1;
                var forward = delta / distance;
                var right = Vector3.Cross(Mathf.Abs(forward.y) < .9f ? Vector3.up : Vector3.right, forward).normalized;
                var up = Vector3.Cross(forward, right);
                float threshold = distance - depthBias;
                int visible = 0;
                for (int i = 0; i < count; ++i)
                {
                    float theta = 2.39996323f * i;
                    float radius = angularRadius * Mathf.Sqrt((i + .5f) / count);
                    var direction = (forward + radius * (Mathf.Cos(theta) * right + Mathf.Sin(theta) * up)).normalized;
                    // Keep the receiver's radial depth while querying nearby light rays. Clip along each cube face's axis.
                    float dominant = Mathf.Max(Mathf.Abs(direction.x), Mathf.Max(Mathf.Abs(direction.y), Mathf.Abs(direction.z)));
                    float offset = near / dominant;
                    if (threshold <= offset || !rayTree.Intersects(rayTriangles, new Ray(light + offset * direction, direction), threshold - offset))
                        ++visible;
                }
                return visible / (float)count;
            }

            public void Dispose()
            {
                foreach (var m in materials.Values) if (m) Object.DestroyImmediate(m);
            }

            private readonly struct RayTriangle
            {
                internal readonly Vector3 a, ab, ac;
                internal readonly Bounds bounds;
                internal RayTriangle(Vector3 a, Vector3 b, Vector3 c)
                {
                    this.a = a; ab = b - a; ac = c - a;
                    bounds = new Bounds(a, Vector3.zero);
                    bounds.Encapsulate(b); bounds.Encapsulate(c);
                }

                internal bool Intersects(Ray ray, float length)
                {
                    // Double-sided geometric intersection, independent of optimisation colliders.
                    Vector3 p = Vector3.Cross(ray.direction, ac);
                    float determinant = Vector3.Dot(ab, p);
                    if (Mathf.Abs(determinant) < 1e-9f) return false;
                    float inverse = 1 / determinant;
                    Vector3 t = ray.origin - a;
                    float u = Vector3.Dot(t, p) * inverse;
                    if (u < 0 || u > 1) return false;
                    Vector3 q = Vector3.Cross(t, ab);
                    float v = Vector3.Dot(ray.direction, q) * inverse;
                    if (v < 0 || u + v > 1) return false;
                    float hit = Vector3.Dot(ac, q) * inverse;
                    return hit > .00001f && hit < length;
                }
            }

            private sealed class RayNode
            {
                private readonly Bounds bounds;
                private readonly int first, count;
                private readonly RayNode left, right;

                internal RayNode(RayTriangle[] triangles, int first, int count)
                {
                    this.first = first; this.count = count;
                    bounds = triangles[first].bounds;
                    for (int i = first + 1; i < first + count; ++i) bounds.Encapsulate(triangles[i].bounds);
                    if (count <= 8) return;
                    var span = bounds.size;
                    int axis = span.x >= span.y && span.x >= span.z ? 0 : span.y >= span.z ? 1 : 2;
                    Array.Sort(triangles, first, count, Comparer<RayTriangle>.Create((a, b) => a.bounds.center[axis].CompareTo(b.bounds.center[axis])));
                    int middle = count / 2;
                    left = new RayNode(triangles, first, middle);
                    right = new RayNode(triangles, first + middle, count - middle);
                }

                internal bool Intersects(RayTriangle[] triangles, Ray ray, float length)
                {
                    if (!bounds.IntersectRay(ray, out float distance) || distance >= length) return false;
                    if (left != null) return left.Intersects(triangles, ray, length) || right.Intersects(triangles, ray, length);
                    for (int i = first; i < first + count; ++i)
                        if (triangles[i].Intersects(ray, length)) return true;
                    return false;
                }
            }
        }

        private static string Xyz(Vector3 v) => string.Join(",", F(v.x), F(v.y), F(v.z));
    }
}

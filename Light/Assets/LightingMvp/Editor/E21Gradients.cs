using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using Object = UnityEngine.Object;
using Debug = UnityEngine.Debug;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private static readonly string[] E21GradientMethods = { "frozen_adjoint", "fixed_fd", "full_fd", "ray_soft_fd",
            "ray_hard_fd", "native_hard_fd", "uv_only_fd", "receiver_depth_fd", "caster_radial_fd", "caster_projection_fd", "map_only_fd" };

        [Serializable] private sealed class E21GradientRun
        {
            public string branch, commit, unity, urp, gpu, graphics, classroomHash, fullGradientDecision;
            public int cases, rayTaps, cubeSize, msaa, blur;
            public double seconds;
        }

        private static string E21Sha(string path)
        {
            using var sha = System.Security.Cryptography.SHA256.Create();
            using var stream = File.OpenRead(path);
            return BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
        }

        private static void RunE21GradientStages(string task, string output)
        {
            bool smoke = task.EndsWith("-smoke", StringComparison.Ordinal);
            string directory = Path.Combine(output, task.Replace('-', '_'));
            Directory.CreateDirectory(directory);
            var clock = Stopwatch.StartNew();
            using var pipeline = new TestPipeline(classroom: true);
            using var summary = new StreamWriter(Path.Combine(directory, "gradient-summary.csv"));
            using var comparisons = new StreamWriter(Path.Combine(directory, "gradient-cosines.csv"));
            using var costs = new StreamWriter(Path.Combine(directory, "gradient-costs.csv"));
            summary.WriteLine("case,step_m,axis,method,norm,mean_abs,nonzero,receivers");
            comparisons.WriteLine("case,step_m,axis,left,right,cosine,rmse");
            costs.WriteLine("case,receivers,map_faces,native_surface_renders,adjoint_invalid,invalid_bound_queries,nonfinite_bound_queries,limit_queries,forward_sync_s,backward_sync_s,seconds");
            var source = new Vector3(.18f, 2.35f, -.12f);
            int grid = smoke ? 20 : 42, rayTaps = 17;
            var cases = smoke ? new[] { "triangle" } : new[] { "triangle", "box", "two_boxes_thin_wall" };
            foreach (string label in cases)
            {
                EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                var root = new GameObject("E21 " + label);
                var material = new Material(Shader.Find("Universal Render Pipeline/Lit"));
                Mesh triangle = null;
                try
                {
                    var camera = new GameObject("Calibration camera").AddComponent<Camera>();
                    camera.transform.SetParent(root.transform);
                    camera.enabled = false;
                    var light = new GameObject("Calibration point").AddComponent<Light>();
                    light.transform.SetParent(root.transform);
                    light.transform.position = source;
                    light.type = LightType.Point;
                    light.shadows = LightShadows.Hard;
                    light.range = 7;
                    light.intensity = 2;
                    light.shadowNearPlane = .055f;
                    light.shadowBias = .08f;
                    light.shadowNormalBias = .25f;
                    var info = new SerializedObject(light.GetUniversalAdditionalLightData());
                    info.FindProperty("m_AdditionalLightsShadowResolutionTier").intValue = UniversalAdditionalLightData.AdditionalLightsShadowResolutionTierMedium;
                    info.ApplyModifiedPropertiesWithoutUndo();
                    Box(new Vector3(0, -.045f, 0), new Vector3(6, .09f, 6)).shadowCastingMode = ShadowCastingMode.Off;
                    if (label == "triangle")
                    {
                        triangle = new Mesh { name = "Calibration triangle" };
                        triangle.vertices = new[] { new Vector3(-.61f, 1.25f, -.42f), new Vector3(.53f, 1.1f, -.33f), new Vector3(.12f, 1.35f, .61f) };
                        triangle.triangles = new[] { 0, 2, 1 };
                        triangle.RecalculateNormals();
                        triangle.RecalculateBounds();
                        var caster = new GameObject("Triangle", typeof(MeshFilter), typeof(MeshRenderer));
                        caster.transform.SetParent(root.transform);
                        caster.GetComponent<MeshFilter>().sharedMesh = triangle;
                        caster.GetComponent<MeshRenderer>().sharedMaterial = material;
                    }
                    else if (label == "box") Box(new Vector3(.08f, 1.2f, .02f), new Vector3(.55f, .24f, .70f));
                    else
                    {
                        Box(new Vector3(-.18f, 1.42f, .13f), new Vector3(.57f, .13f, .59f));
                        Box(new Vector3(.41f, .65f, -.26f), new Vector3(.75f, .12f, .95f));
                        Box(new Vector3(.76f, .87f, .15f), new Vector3(.018f, 1.72f, 1.90f));
                    }
                    var positions = new Vector3[grid * grid];
                    for (int z = 0; z < grid; z++)
                    for (int x = 0; x < grid; x++)
                        positions[z * grid + x] = new Vector3((x + .5f) * 5.4f / grid - 2.7f, 0, (z + .5f) * 5.4f / grid - 2.7f);
                    using var geometry = new MbdSurfaceGeometry(camera);
                    Measure(label, camera, geometry, light, positions, Enumerable.Repeat(Vector3.up, positions.Length).ToArray(), grid);
                }
                finally
                {
                    Object.DestroyImmediate(root);
                    if (triangle) Object.DestroyImmediate(triangle);
                    Object.DestroyImmediate(material);
                }

                MeshRenderer Box(Vector3 position, Vector3 scale)
                {
                    var box = GameObject.CreatePrimitive(PrimitiveType.Cube);
                    box.transform.SetParent(root.transform);
                    box.transform.position = position;
                    box.transform.localScale = scale;
                    var renderer = box.GetComponent<MeshRenderer>();
                    renderer.sharedMaterial = material;
                    return renderer;
                }
            }
            if (!smoke)
            {
                using var classroom = new ClassroomSensitivityScene(pipeline.Asset, null);
                classroom.Apply(classroom.Truth);
                var light = GameObject.Find(classroom.UnityName(4)).GetComponent<Light>();
                var samples = E18LampStart(classroom, 4, 30f, 0, out _);
                classroom.Apply(samples);
                using var geometry = new MbdSurfaceGeometry(classroom.Camera);
                string path = Path.Combine(directory, "classroom_L5");
                Directory.CreateDirectory(path);
                var sites = geometry.Samples(1280, path);
                Measure("classroom_L5", classroom.Camera, geometry, light, sites.Select(s => s.position).ToArray(),
                    sites.Select(s => s.normal).ToArray(), 0);
            }
            WriteJson(directory, "summary.json", new E21GradientRun {
                branch = Arg("-bbE20Branch", "blackbox"), commit = Arg("-bbE20BaseCommit", "22c00f9"),
                unity = Application.unityVersion, urp = UnityEditor.PackageManager.PackageInfo.FindForAssembly(typeof(UniversalRenderPipeline).Assembly)?.version,
                gpu = SystemInfo.graphicsDeviceName, graphics = SystemInfo.graphicsDeviceType.ToString(),
                classroomHash = smoke ? "" : E21Sha("Assets/ClassroomScale/room.bytes"), cases = cases.Length + (smoke ? 0 : 1),
                rayTaps = rayTaps, cubeSize = 128, msaa = 4, blur = 2, fullGradientDecision = "rerender_fd", seconds = clock.Elapsed.TotalSeconds
            });

            void Measure(string label, Camera camera, MbdSurfaceGeometry geometry, Light light, Vector3[] receivers, Vector3[] normals, int side)
            {
                var timer = Stopwatch.StartNew();
                geometry.OpenRayReference();
                int width = side > 0 ? side : 128, height = (receivers.Length + width - 1) / width;
                using var native = new NativeIrradianceCapture(camera, width, height);
                native.SetSurfaceSamples(receivers, normals, geometry.Bounds);
                var lights = new[] { light };
                Vector3 anchor = light.transform.position;
                var settings = MbdSettings(light);
                using var map = new PointMomentShadow(geometry.DrawCasters, receivers, settings);
                var gt = map.Render(anchor);
                var nativeStart = ReadNative(anchor);
                int nativeRenders = 1;
                var hard = Ray(anchor, 1);
                if (hard.Min() == hard.Max() || gt.Max() - gt.Min() < .01f)
                    throw new InvalidOperationException("No measurable calibration shadow in " + label);
                var adj = map.FixedMapPositionAdjoint(anchor);
                string path = Path.Combine(directory, label);
                Directory.CreateDirectory(path);
                if (side > 0)
                {
                    SaveVisibilityPng(path, "mbd", gt, side);
                    SaveVisibilityPng(path, "native", nativeStart, side);
                    SaveVisibilityPng(path, "ray_hard", hard, side);
                }
                using var points = new StreamWriter(Path.Combine(path, "receivers.csv"));
                points.WriteLine("sample,x,y,z,nx,ny,nz,mbd,native,ray_hard,adjoint_status");
                for (int i = 0; i < receivers.Length; ++i)
                    points.WriteLine(string.Join(",", i, Xyz(receivers[i]), Xyz(normals[i]), F(gt[i]), F(nativeStart[i]), F(hard[i]), F(adj[i].w)));
                using var raw = new StreamWriter(Path.Combine(path, "derivatives.csv"));
                raw.WriteLine("step_m,axis,sample," + string.Join(",", E21GradientMethods) + ",linear_sum_defect,map_projection_radial_defect");
                foreach (float h in smoke ? new[] { .01f } : new[] { .00025f, .005f, .01f, .02f })
                for (int axis = 0; axis < 3; ++axis)
                {
                    var delta = Vector3.zero;
                    delta[axis] = h;
                    var plus = anchor + delta; var minus = anchor - delta;
                    map.Render(anchor);
                    var fixedFd = Difference(map.Query(plus), map.Query(minus));
                    var uvOnly = Difference(map.Query(plus, anchor), map.Query(minus, anchor));
                    var receiverDepth = Difference(map.Query(anchor, plus), map.Query(anchor, minus));
                    var radialOnly = Difference(map.Render(anchor, plus, anchor, anchor), map.Render(anchor, minus, anchor, anchor));
                    var projectionOnly = Difference(map.Render(plus, anchor, anchor, anchor), map.Render(minus, anchor, anchor, anchor));
                    var mapOnly = Difference(map.Render(plus, plus, anchor, anchor), map.Render(minus, minus, anchor, anchor));
                    var full = Difference(map.Render(plus), map.Render(minus));
                    var raySoft = Difference(Ray(plus, rayTaps), Ray(minus, rayTaps));
                    var rayHard = Difference(Ray(plus, 1), Ray(minus, 1));
                    var nativeFd = Difference(ReadNative(plus), ReadNative(minus));
                    nativeRenders += 2;
                    var data = new[] { adj.Select(x => x[axis]).ToArray(), fixedFd, full, raySoft, rayHard, nativeFd, uvOnly, receiverDepth,
                        radialOnly, projectionOnly, mapOnly };
                    for (int i = 0; i < receivers.Length; ++i)
                        raw.WriteLine(string.Join(",", F(h), axis, i, string.Join(",", data.Select(x => F(x[i]))),
                            F(full[i] - uvOnly[i] - receiverDepth[i] - mapOnly[i]), F(mapOnly[i] - radialOnly[i] - projectionOnly[i])));
                    for (int j = 0; j < data.Length; ++j)
                    {
                        var x = data[j];
                        double xx = E21Dot(x, x);
                        summary.WriteLine(string.Join(",", label, F(h), axis, E21GradientMethods[j], F(Math.Sqrt(xx)),
                            F(x.Average(y => Math.Abs((double)y))), x.Count(y => y != 0), x.Length));
                        foreach (int reference in new[] { 1, 2, 3, 5 })
                        {
                            if (j == reference) continue;
                            var y = data[reference];
                            double yy = E21Dot(y, y), xy = E21Dot(x, y);
                            comparisons.WriteLine(string.Join(",", label, F(h), axis, E21GradientMethods[j], E21GradientMethods[reference],
                                F(xx > 0 && yy > 0 ? xy / Math.Sqrt(xx * yy) : double.NaN), F(Math.Sqrt(Math.Max(0, xx + yy - 2 * xy) / x.Length))));
                        }
                    }
                    float[] Difference(float[] a, float[] b) => a.Select((x, i) => (x - b[i]) / (2 * h)).ToArray();
                }
                light.transform.position = anchor;
                costs.WriteLine(string.Join(",", label, receivers.Length, map.Faces, nativeRenders, adj.Count(v => v.w >= 2), map.Degenerate, map.Nonfinite,
                    map.LimitSamples, F(map.ForwardSyncSeconds), F(map.BackwardSyncSeconds), F(timer.Elapsed.TotalSeconds)));
                summary.Flush(); comparisons.Flush(); costs.Flush();
                Debug.Log($"[E21-A] {label}, {receivers.Length} receivers, {map.Faces} faces in {timer.Elapsed.TotalSeconds:F1}s.");

                float[] ReadNative(Vector3 position)
                {
                    light.transform.position = position;
                    return native.ReadSurfaces(lights).Take(receivers.Length).Select(c => c.a).ToArray();
                }
                float[] Ray(Vector3 position, int taps) => receivers.Select(p =>
                    geometry.FilteredRayVisibility(p, position, 3.15f / settings.size, taps)).ToArray();
            }
        }

        private static double E21Dot(float[] a, float[] b, int start = 0, int end = -1)
        {
            double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            if (end < 0) end = a.Length;
            int k = start;
            for (; k + 3 < end; k += 4)
            {
                s0 += (double)a[k] * b[k]; s1 += (double)a[k + 1] * b[k + 1];
                s2 += (double)a[k + 2] * b[k + 2]; s3 += (double)a[k + 3] * b[k + 3];
            }
            for (; k < end; ++k) s0 += (double)a[k] * b[k];
            return s0 + s1 + s2 + s3;
        }
    }
}

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
using UnityEngine.SceneManagement;
using Debug = UnityEngine.Debug;
using Object = UnityEngine.Object;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private static MomentSettings MbdSettings(Light light, int size = 128, int msaa = 4, int radius = 2, float blend = .15f) =>
            new MomentSettings { label = $"r{size}_msaa{msaa}_radius{radius}_beta{blend.ToString("F2", Invariant)}",
                mbd = true, size = size, requestedMsaa = msaa, blurRadius = radius, near = Mathf.Max(.02f, light.shadowNearPlane),
                far = light.range, momentBias = .001f, depthBiasMetres = .012f, overestimationWeight = blend };

        private static void RunMbdStage(string task, string output)
        {
            if (!Application.isBatchMode) throw new InvalidOperationException("MBD runs in an isolated Unity batch project.");
            string dir = Path.Combine(output, task.Replace("-", "_"));
            Directory.CreateDirectory(dir);
            bool smoke = task.EndsWith("-smoke", StringComparison.Ordinal);
            string stage = smoke ? task.Substring(0, task.Length - 6) : task;
            using var pipeline = new TestPipeline(classroom: true);
            if (stage == "shadow") RunMbdShadowBenchmark(dir, smoke);
            else if (stage == "recovery")
            {
                using var scene = new ClassroomSensitivityScene(pipeline.Asset, null);
                using var study = new MbdStudy(scene, dir, smoke);
                study.Recovery();
                study.Finish(stage);
            }
            else if (stage == "irradiance")
            {
                string classroomDir = Path.Combine(dir, "classroom");
                using (var scene = new ClassroomSensitivityScene(pipeline.Asset, null))
                using (var study = new MbdStudy(scene, classroomDir, smoke))
                {
                    study.Irradiance(false);
                    study.Finish(stage);
                }
                if (!smoke)
                {
                    using var room = new ComplexRoomSensitivityScene();
                    using var study = new MbdStudy(room, Path.Combine(dir, "e19"), smoke);
                    study.Irradiance(true);
                    study.Finish(stage);
                }
                WriteJson(dir, "summary.json", new MbdShadowSummary { cases = smoke ? 1 : 2 });
            }
            else if (stage == "separation")
            {
                string jointFile = Path.Combine(output, "irradiance", "e19", "captures", "seed733_near_all_D0", "trial.json");
                var joint = JsonUtility.FromJson<MbdTrial>(File.ReadAllText(jointFile));
                using var room = new ComplexRoomSensitivityScene();
                using var study = new MbdStudy(room, dir, smoke: false);
                study.Irradiance(true, joint);
                study.Finish(stage);
            }
            else if (stage == "topology")
            {
                using var room = new ComplexRoomSensitivityScene();
                using var study = new MbdStudy(room, dir, smoke);
                study.Topology();
                study.Finish(stage);
            }
            else throw new ArgumentException("Unknown MBD scene stage: " + task);
        }

        [Serializable] private sealed class MbdShadowSummary
        {
            public string version, urp, gpu, graphics, commit, branch;
            public int cases, settings, grid, rayTaps;
            public long unityAllocatedBytes;
            public double seconds;
        }

        private static void RunMbdShadowBenchmark(string dir, bool smoke)
        {
            var clock = Stopwatch.StartNew();
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var root = new GameObject("MBD calibration fixtures");
            var shader = Shader.Find("Universal Render Pipeline/Lit");
            var material = new Material(shader);
            var camera = new GameObject("Calibration camera").AddComponent<Camera>();
            camera.enabled = false;
            var light = new GameObject("Calibration point light").AddComponent<Light>();
            light.type = LightType.Point;
            light.shadows = LightShadows.Hard;
            light.shadowNearPlane = .055f;
            light.shadowBias = .08f;
            light.shadowNormalBias = .25f;
            light.intensity = 2;
            light.range = 7;
            var lightData = new SerializedObject(light.GetUniversalAdditionalLightData());
            lightData.FindProperty("m_AdditionalLightsShadowResolutionTier").intValue = UniversalAdditionalLightData.AdditionalLightsShadowResolutionTierMedium;
            lightData.ApplyModifiedPropertiesWithoutUndo();
            var source = new Vector3(.18f, 2.35f, -.12f);
            light.transform.position = source;
            const int taps = 33;
            int side = smoke ? 32 : 64;
            const float spacing = .085f;
            var receiver = new Vector3[side * side];
            var normal = Enumerable.Repeat(Vector3.up, receiver.Length).ToArray();
            for (int z = 0; z < side; ++z)
            for (int x = 0; x < side; ++x)
                receiver[z * side + x] = new Vector3((x - (side - 1) * .5f) * spacing, 0, (z - (side - 1) * .5f) * spacing);
            Box("Receiver", new Vector3(0, -.045f, 0), new Vector3(6, .09f, 6), root.transform).shadowCastingMode = ShadowCastingMode.Off;
            var conditions = new Dictionary<string, (Vector3 p, Vector3 s)[]> {
                ["empty"] = Array.Empty<(Vector3, Vector3)>(),
                ["one_box"] = new[] { (new Vector3(.08f, 1.2f, .02f), new Vector3(.55f, .24f, .70f)) },
                ["two_layers"] = new[] { (new Vector3(-.18f, 1.42f, .13f), new Vector3(.57f, .13f, .59f)),
                    (new Vector3(.41f, .65f, -.26f), new Vector3(.75f, .12f, .95f)) },
                ["thin_wall"] = new[] { (new Vector3(.59f, .87f, .2f), new Vector3(.018f, 1.72f, 1.96f)) },
                ["small_occluder"] = new[] { (new Vector3(-.09f, 1.25f, .11f), new Vector3(.063f, .077f, .063f)) },
                ["multilayer"] = new[] { (new Vector3(-.50f, .66f, .15f), new Vector3(.72f, 1.20f, .40f)),
                    (new Vector3(.38f, 1.43f, .26f), new Vector3(1.52f, .09f, .26f)),
                    (new Vector3(.88f, .82f, -.55f), new Vector3(.15f, 1.40f, .82f)),
                    (new Vector3(-.06f, .54f, -.95f), new Vector3(.96f, .20f, .32f)) },
                ["cube_seam"] = new[] { (new Vector3(1.36f, 1.17f, -.05f), new Vector3(.22f, .27f, .80f)),
                    (new Vector3(-.50f, .97f, 1.25f), new Vector3(.67f, .32f, .24f)) }
            };
            int settings = 0;
            using var overview = new StreamWriter(Path.Combine(dir, "shadow-metrics.csv"));
            using var derivatives = new StreamWriter(Path.Combine(dir, "position-derivatives.csv"));
            overview.WriteLine("scene,setting,face_size,msaa,filter_radius,bias,depth_bias_m,beta,near_m,far_m,shadow_faces,native_surface_views,mbd_mae_ray_hard,native_mae_ray_hard,mbd_mae_filtered_ray,native_mae_filtered_ray,mbd_boundary_mae,native_boundary_mae,mbd_leak_ratio,native_leak_ratio,interior_shadow_samples,hard_shadow_samples,max_seam,invalid,nonfinite,infinite_roots,pass_submit_s,filter_submit_s,forward_sync_s,backward_sync_s,profile_pass_s,profile_filter_s,profile_forward_s,estimated_gpu_bytes,wall_s");
            derivatives.WriteLine("scene,setting,sample,axis,fixed_fd_step_m,full_fd_step_m,fixed_map_adjoint,fixed_map_fd,fixed_map_fd_at_full_step,full_mbd_fd,full_native_fd,full_ray_fd,adjoint_status");
            try
            {
                foreach (var condition in conditions.Where(x => !smoke || x.Key == "empty" || x.Key == "one_box"))
                {
                    var casters = new GameObject("Calibration occluders");
                    casters.transform.SetParent(root.transform);
                    foreach (var item in condition.Value) Box(condition.Key, item.p, item.s, casters.transform);
                    Physics.SyncTransforms();
                    try
                    {
                        using var geometry = new MbdSurfaceGeometry(camera);
                        geometry.OpenRayReference();
                        using var native = new NativeIrradianceCapture(camera, side, side);
                        native.SetSurfaceSamples(receiver, normal, geometry.Bounds);
                        float[] ReadNative(Vector3 pos)
                        {
                            light.transform.position = pos;
                            return native.ReadSurfaces(new[] { light }).Select(c => c.a).Take(receiver.Length).ToArray();
                        }
                        var nativeGt = ReadNative(source);
                        var hard = receiver.Select(p => geometry.RayVisibility(p, source)).ToArray();
                        if (condition.Key == "empty" && (nativeGt.Min() < .9f || hard.Min() < .99f))
                            throw new InvalidOperationException("Unoccluded fixture did not return visible receivers in the native/ray passes.");
                        var variants = new List<MomentSettings> { MbdSettings(light) };
                        if (!smoke && (condition.Key == "two_layers" || condition.Key == "cube_seam"))
                            variants.AddRange(new[] { MbdSettings(light, 64, 1), MbdSettings(light, 256), MbdSettings(light, radius: 0),
                                MbdSettings(light, radius: 4), MbdSettings(light, blend: 0), MbdSettings(light, blend: .5f) });
                        for (int setting = 0; setting < variants.Count; ++setting)
                        {
                            var cfg = variants[setting];
                            var watch = Stopwatch.StartNew();
                            using var shadow = new PointMomentShadow(geometry.DrawCasters, receiver, cfg);
                            var soft = shadow.Render(source, profile: true);
                            if (condition.Key == "empty" && soft.Min() < .9f)
                                throw new InvalidOperationException("Unoccluded MBD fixture returned a substantial shadow.");
                            var angularRadius = (cfg.blurRadius + 1) * 1.05f / cfg.size;
                            var filtered = receiver.Select(p => geometry.FilteredRayVisibility(p, source, angularRadius, taps)).ToArray();
                            var edge = Enumerable.Range(0, receiver.Length).Where(i => filtered[i] > .02f && filtered[i] < .98f).ToArray();
                            var dark = Enumerable.Range(0, receiver.Length).Where(i => hard[i] < .5f && filtered[i] < .01f).ToArray();
                            double Difference(float[] a, float[] b, int[] ids = null) =>
                                ids == null ? a.Select((v, i) => (double)Mathf.Abs(v - b[i])).Average() :
                                ids.Length == 0 ? double.NaN : ids.Average(i => (double)Mathf.Abs(a[i] - b[i]));
                            double Leaks(float[] values) => dark.Length == 0 ? double.NaN : dark.Count(i => values[i] > .5f) / (double)dark.Length;
                            string path = Path.Combine(dir, condition.Key, cfg.label);
                            Directory.CreateDirectory(path);
                            using (var samples = new StreamWriter(Path.Combine(path, "visibility.csv")))
                            {
                                samples.WriteLine("sample,x,y,z,ray_hard,ray_filtered,native_hard,mbd");
                                for (int i = 0; i < receiver.Length; ++i)
                                    samples.WriteLine(string.Join(",", i, Xyz(receiver[i]), F(hard[i]), F(filtered[i]), F(nativeGt[i]), F(soft[i])));
                            }
                            if (setting == 0)
                            {
                                SaveVisibilityPng(path, "ray_hard", hard, side);
                                SaveVisibilityPng(path, "ray_filtered", filtered, side);
                                SaveVisibilityPng(path, "native_hard", nativeGt, side);
                                SaveVisibilityPng(path, "mbd", soft, side);
                                if (condition.Key == "cube_seam") shadow.SaveFaces(path, "gt");
                            }
                            float seam = shadow.LastSeamMax;
                            int nativeViews = 1;
                            if (setting == 0 && condition.Key != "empty")
                            {
                                var adjoint = shadow.FixedMapPositionAdjoint(source);
                                const float h = .00025f, fullH = .01f;
                                for (int a = 0; a < 3; ++a)
                                {
                                    var axis = Vector3.zero; axis[a] = 1;
                                    var fixedPlus = shadow.Query(source + h * axis);
                                    var fixedMinus = shadow.Query(source - h * axis);
                                    var p = source + fullH * axis;
                                    var m = source - fullH * axis;
                                    var fixedWidePlus = shadow.Query(p);
                                    var fixedWideMinus = shadow.Query(m);
                                    var positive = shadow.Render(p);
                                    var negative = shadow.Render(m);
                                    var nativePositive = ReadNative(p);
                                    var nativeNegative = ReadNative(m);
                                    nativeViews += 2;
                                    for (int i = 0; i < receiver.Length; ++i)
                                    {
                                        float rp = geometry.FilteredRayVisibility(receiver[i], p, angularRadius, taps);
                                        float rm = geometry.FilteredRayVisibility(receiver[i], m, angularRadius, taps);
                                        derivatives.WriteLine(string.Join(",", condition.Key, cfg.label, i, a, F(h), F(fullH), F(adjoint[i][a]),
                                            F((fixedPlus[i] - fixedMinus[i]) / (2 * h)), F((fixedWidePlus[i] - fixedWideMinus[i]) / (2 * fullH)),
                                            F((positive[i] - negative[i]) / (2 * fullH)),
                                            F((nativePositive[i] - nativeNegative[i]) / (2 * fullH)), F((rp - rm) / (2 * fullH)), (int)adjoint[i].w));
                                    }
                                    if (a < 2) shadow.Render(source);
                                }
                                light.transform.position = source;
                            }
                            // Count the full pipeline cost; synchronized forward time also waits for queued passes and filters.
                            overview.WriteLine(string.Join(",", condition.Key, cfg.label, cfg.size, cfg.actualMsaa, cfg.blurRadius, F(cfg.momentBias),
                                F(cfg.depthBiasMetres), F(cfg.overestimationWeight), F(cfg.near), F(cfg.far), shadow.Faces, nativeViews,
                                F(Difference(soft, hard)), F(Difference(nativeGt, hard)), F(Difference(soft, filtered)), F(Difference(nativeGt, filtered)),
                                F(Difference(soft, filtered, edge)), F(Difference(nativeGt, filtered, edge)), F(Leaks(soft)), F(Leaks(nativeGt)),
                                dark.Length, hard.Count(x => x < .5f), F(seam), shadow.Degenerate, shadow.Nonfinite, shadow.LimitSamples,
                                F(shadow.PassSubmitSeconds), F(shadow.FilterSubmitSeconds), F(shadow.ForwardSyncSeconds), F(shadow.BackwardSyncSeconds),
                                F(shadow.ProfilePassSeconds), F(shadow.ProfileFilterSeconds), F(shadow.ProfileForwardSeconds),
                                shadow.EstimatedGpuBytes, F(watch.Elapsed.TotalSeconds)));
                            settings++;
                        }
                        Debug.Log("[MBD] shadow " + condition.Key + ", native shadow sites " + nativeGt.Count(v => v < .5f));
                    }
                    finally { Object.DestroyImmediate(casters); }
                }
                WriteJson(dir, "summary.json", new MbdShadowSummary { version = Application.unityVersion, gpu = SystemInfo.graphicsDeviceName,
                    graphics = SystemInfo.graphicsDeviceType.ToString(), branch = Arg("-bbE20Branch", "blackbox"), commit = Arg("-bbE20BaseCommit", E20BaseCommit),
                    urp = UnityEditor.PackageManager.PackageInfo.FindForAssembly(typeof(UniversalRenderPipeline).Assembly)?.version,
                    cases = smoke ? 2 : conditions.Count, settings = settings, grid = side, rayTaps = taps, seconds = clock.Elapsed.TotalSeconds,
                    unityAllocatedBytes = UnityEngine.Profiling.Profiler.GetTotalAllocatedMemoryLong() });
            }
            finally
            {
                Object.DestroyImmediate(light.gameObject); Object.DestroyImmediate(camera.gameObject);
                Object.DestroyImmediate(root); Object.DestroyImmediate(material);
            }

            MeshRenderer Box(string name, Vector3 p, Vector3 size, Transform parent)
            {
                var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
                go.name = name;
                go.transform.SetParent(parent, false);
                go.transform.SetPositionAndRotation(p, Quaternion.identity);
                go.transform.localScale = size;
                var renderer = go.GetComponent<MeshRenderer>();
                renderer.sharedMaterial = material;
                return renderer;
            }
        }

        private static void SaveVisibilityPng(string dir, string name, float[] values, int side)
        {
            var texture = new Texture2D(side, side, TextureFormat.RGB24, false, true);
            try
            {
                texture.SetPixels(values.Select(v => new Color(v, v, v)).ToArray());
                texture.Apply();
                File.WriteAllBytes(Path.Combine(dir, name + ".png"), texture.EncodeToPNG());
            }
            finally { Object.DestroyImmediate(texture); }
        }
    }
}

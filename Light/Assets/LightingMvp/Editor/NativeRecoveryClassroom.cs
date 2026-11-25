using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using Debug = UnityEngine.Debug;
using Object = UnityEngine.Object;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private const string ClassroomAssets = "Assets/ClassroomScale";
        private const string ClassroomMode = "URP stock Forward+";
        private const int ClassroomWidth = 448, ClassroomCheckWidth = 896; // 7:4, same at fit and recapture
        private const float ClassroomRange = 5.25f, SceneClearance = .018f;
        private static readonly Vector3 RoomMin = new Vector3(-5.33f, 1.06f, -4.40f), RoomMax = new Vector3(3.48f, 2.86f, 3.03f);

        [Serializable] private sealed class SourceRoom
        {
            public int version, triangles;
            public string source, blender, sourceUrl;
            public SourceLight[] lamps, excluded;
            public SourceMaterial[] materials;
        }
        [Serializable] private sealed class SourceLight { public string name, type; public float watts; public float[] position; }
        [Serializable] private sealed class SourceMaterial { public int id; public string name, texture; public float[] color; public float metallic, smoothness, emissive; }

        [Serializable] private sealed class RoomConfig
        {
            public string unity, urp, renderer, gpu, graphics, colorSpace, powerMapping, source, sourceUrl, roomSha256;
            public int lamps, triangles, materials, textures, atlas, shadowTile, maxRequestedFaces, theoreticalTiles, nominalPerObjectLimit;
            public int width, height, checkWidth, checkHeight;
            public float range, shadowDistance, colliderClearance, near, far, exposure, ambientLinear;
            public Vector3 boundsMin, boundsMax;
            public NativeLamp[] gt;
            public SourceLight[] original, omittedNonPoint;
            public ViewInfo[] views;
        }

        private sealed class ClassroomFixture : IDisposable
        {
            private const int CollisionLayer = 29;
            internal readonly RoomConfig Config;
            internal readonly ScaleRig Rig;
            internal readonly Camera Camera;
            internal readonly double[] Truth;
            private readonly List<Object> owned = new List<Object>();
            private readonly Light[] lights;

            internal ClassroomFixture(UniversalRenderPipelineAsset pipeline)
            {
                var descriptor = AssetDatabase.LoadAssetAtPath<TextAsset>(ClassroomAssets + "/room.json");
                var meshFile = AssetDatabase.LoadAssetAtPath<TextAsset>(ClassroomAssets + "/room.bytes");
                if (!descriptor || !meshFile) throw new FileNotFoundException("Convert Blender Classroom with mvp/export_classroom_scale.py into " + ClassroomAssets);
                var source = JsonUtility.FromJson<SourceRoom>(descriptor.text);
                if (source.version != 1 || source.lamps.Length < 2 || source.lamps.Length > 10 || source.lamps.Any(l => l.type != "POINT"))
                    throw new InvalidOperationException("Invalid small authored Point rig.");
                EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                try
                {
                    RenderSettings.skybox = null;
                    RenderSettings.ambientMode = AmbientMode.Flat;
                    RenderSettings.ambientLight = new Color(.20f, .20f, .20f, 1);
                    RenderSettings.reflectionIntensity = 0;
                    RenderSettings.fog = false;
                    Lightmapping.giWorkflowMode = Lightmapping.GIWorkflowMode.OnDemand;
                    var shader = Shader.Find("Universal Render Pipeline/Lit");
                    if (!shader || !shader.isSupported) throw new InvalidOperationException("URP/Lit not available.");
                    var root = New("Classroom source geometry");
                    using var reader = new BinaryReader(new MemoryStream(meshFile.bytes));
                    if (reader.ReadUInt32() != 0x53434C45 || reader.ReadInt32() != source.materials.Length)
                        throw new InvalidDataException("Classroom mesh/material schema mismatch.");
                    for (int i = 0; i < source.materials.Length; i++)
                    {
                        var def = source.materials[i];
                        var mat = Own(new Material(shader) { name = def.name });
                        mat.SetColor("_BaseColor", new Color(def.color[0], def.color[1], def.color[2], 1));
                        mat.SetFloat("_Metallic", def.metallic);
                        mat.SetFloat("_Smoothness", def.smoothness);
                        if (!string.IsNullOrEmpty(def.texture))
                        {
                            var texture = AssetDatabase.LoadAssetAtPath<Texture2D>(ClassroomAssets + "/" + def.texture);
                            if (!texture) throw new FileNotFoundException("Classroom material texture " + def.texture);
                            mat.SetTexture("_BaseMap", texture);
                        }
                        if (def.emissive > 0)
                        {
                            mat.EnableKeyword("_EMISSION"); mat.SetColor("_EmissionColor", Color.white * def.emissive);
                            mat.globalIlluminationFlags = MaterialGlobalIlluminationFlags.None;
                        }
                        int count = reader.ReadInt32(), indices = reader.ReadInt32();
                        if (def.id != i || count <= 0 || indices <= 0 || indices % 3 != 0 || 32L * count + 4L * indices > reader.BaseStream.Length - reader.BaseStream.Position)
                            throw new InvalidDataException("Invalid Classroom mesh size.");
                        var verts = new Vector3[count]; var normals = new Vector3[count]; var uv = new Vector2[count];
                        for (int j = 0; j < count; j++)
                        {
                            verts[j] = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
                            normals[j] = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
                            uv[j] = new Vector2(reader.ReadSingle(), reader.ReadSingle());
                        }
                        var triangles = new int[indices];
                        for (int j = 0; j < indices; j++) triangles[j] = reader.ReadInt32();
                        var mesh = Own(new Mesh { name = def.name, indexFormat = count > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16 });
                        mesh.vertices = verts; mesh.normals = normals; mesh.uv = uv; mesh.triangles = triangles;
                        var go = new GameObject(def.name) { layer = CollisionLayer };
                        go.transform.SetParent(root.transform, false);
                        go.AddComponent<MeshFilter>().sharedMesh = mesh;
                        var renderer = go.AddComponent<MeshRenderer>();
                        renderer.sharedMaterial = mat; renderer.lightProbeUsage = LightProbeUsage.Off; renderer.reflectionProbeUsage = ReflectionProbeUsage.Off;
                        // Non-convex render mesh handles clearance; the separate architectural domain excludes the solid walls.
                        go.AddComponent<MeshCollider>().sharedMesh = mesh;
                    }
                    if (reader.BaseStream.Position != reader.BaseStream.Length) throw new InvalidDataException("Trailing Classroom mesh data.");
                    Physics.SyncTransforms();
                    Camera = New("Classroom camera template").AddComponent<Camera>();
                    Camera.enabled = false; Camera.fieldOfView = 58; Camera.nearClipPlane = .07f; Camera.farClipPlane = 35;
                    lights = source.lamps.Select((l, i) =>
                    {
                        var lamp = New($"Classroom point {i + 1}").AddComponent<Light>();
                        lamp.type = LightType.Point; lamp.range = ClassroomRange; lamp.color = Color.white; lamp.useColorTemperature = false;
                        lamp.lightmapBakeType = LightmapBakeType.Realtime; lamp.shadows = LightShadows.Hard;
                        lamp.shadowBias = .12f; lamp.shadowNormalBias = .40f; lamp.shadowNearPlane = .05f;
                        var lightData = new SerializedObject(lamp.GetUniversalAdditionalLightData());
                        lightData.FindProperty("m_AdditionalLightsShadowResolutionTier").intValue = UniversalAdditionalLightData.AdditionalLightsShadowResolutionTierMedium;
                        lightData.ApplyModifiedPropertiesWithoutUndo();
                        lamp.enabled = false;
                        return lamp;
                    }).ToArray();
                    Rig = new ScaleRig { Count = lights.Length, Apply = Apply, Contains = Contains, Describe = Describe };
                    Truth = source.lamps.SelectMany(l => Encode(new Vector3(l.position[0], l.position[1], l.position[2]), Math.Min(6, l.watts * 1.1))).ToArray();
                    if (!Contains(Truth))
                        throw new InvalidOperationException("Authored lamp outside scene domain: " + string.Join(" ", Enumerable.Range(0, lights.Length).Where(i => !Contains(SingleLamp(Truth, i))).Select(i => (i + 1).ToString())));
                    var eyes = new[] { new Vector3(2.85f, 1.38f, -4.04f), new Vector3(-1.95f, 1.62f, 2.79f), new Vector3(-4.73f, 1.49f, -.65f),
                        new Vector3(2.98f, 1.52f, 2.54f), new Vector3(-4.59f, 1.85f, 2.77f) };
                    var look = new[] { new Vector3(.33f, 1.32f, .91f), new Vector3(.64f, 1.08f, -2.23f), new Vector3(-3.58f, 1.50f, 2.62f),
                        new Vector3(-1.10f, 1.20f, -.56f), new Vector3(-3.66f, 1.36f, -1.6f) };
                    string hash;
                    using (var sha = System.Security.Cryptography.SHA256.Create()) hash = BitConverter.ToString(sha.ComputeHash(meshFile.bytes)).Replace("-", "").ToLowerInvariant();
                    Config = new RoomConfig { unity = Application.unityVersion, urp = UnityEditor.PackageManager.PackageInfo.FindForAssembly(typeof(UniversalRenderPipeline).Assembly)?.version,
                        source = source.source, sourceUrl = source.sourceUrl, roomSha256 = hash, renderer = ClassroomMode, gpu = SystemInfo.graphicsDeviceName,
                        graphics = SystemInfo.graphicsDeviceType.ToString(), colorSpace = QualitySettings.activeColorSpace.ToString(),
                        lamps = lights.Length, triangles = source.triangles, materials = source.materials.Length, textures = source.materials.Where(m => !string.IsNullOrEmpty(m.texture)).Select(m => m.texture).Distinct().Count(),
                        width = ClassroomWidth, height = FitSize, checkWidth = ClassroomCheckWidth, checkHeight = CheckSize, atlas = pipeline.additionalLightsShadowmapResolution,
                        shadowTile = pipeline.additionalLightsShadowResolutionTierMedium, shadowDistance = pipeline.shadowDistance,
                        maxRequestedFaces = lights.Length * 6, theoreticalTiles = (int)Math.Pow(pipeline.additionalLightsShadowmapResolution / pipeline.additionalLightsShadowResolutionTierMedium, 2),
                        nominalPerObjectLimit = pipeline.maxAdditionalLightsCount, range = ClassroomRange, colliderClearance = SceneClearance,
                        near = Camera.nearClipPlane, far = Camera.farClipPlane, exposure = 1, ambientLinear = RenderSettings.ambientLight.linear.r,
                        boundsMin = RoomMin, boundsMax = RoomMax, powerMapping = "min(6, 1.1 * Blender watts), white, 5.25 m; not a photometric conversion",
                        gt = Describe(Truth), original = source.lamps, omittedNonPoint = source.excluded,
                        views = eyes.Select((e, i) => new ViewInfo { position = e, rotation = Quaternion.LookRotation(look[i] - e), fov = Camera.fieldOfView }).ToArray() };
                    if (Config.maxRequestedFaces > Config.theoreticalTiles) throw new InvalidOperationException("Classroom shadow requests exceed configured atlas capacity.");
                }
                catch { Dispose(); throw; }
            }

            private T Own<T>(T obj) where T : Object { owned.Add(obj); return obj; }
            private GameObject New(string name) => Own(new GameObject(name));
            private static Vector3 Position(double[] x, int i) => new Vector3(
                Mathf.Lerp(RoomMin.x, RoomMax.x, (float)x[i]), Mathf.Lerp(RoomMin.y, RoomMax.y, (float)x[i + 1]), Mathf.Lerp(RoomMin.z, RoomMax.z, (float)x[i + 2]));
            private static double[] Encode(Vector3 p, double power) => new[] {
                ((double)p.x - RoomMin.x) / (RoomMax.x - RoomMin.x), ((double)p.y - RoomMin.y) / (RoomMax.y - RoomMin.y),
                ((double)p.z - RoomMin.z) / (RoomMax.z - RoomMin.z), PowerCoordinate(power) };

            private bool Contains(double[] x)
            {
                if (x == null || x.Length % LampDofs != 0 || x.Length > lights.Length * LampDofs || x.Any(u => !(u >= 0 && u <= 1))) return false;
                for (int i = 0; i < x.Length; i += LampDofs)
                {
                    Vector3 p = Position(x, i);
                    // The corridor and classroom are separate public volumes; the shared partition is not free space.
                    if (p.x > -2.74f && p.x < -2.20f) return false;
                    // Lights above the furniture plane, except the tall coat rack, radiator and wall-mounted solids.
                    if ((p.x > 3.18f && p.y < 1.37f) || (p.x < -1.89f && p.x > -2.20f && p.z > 2.5f && p.y < 1.9f)) return false;
                    if (Physics.CheckSphere(p, SceneClearance, 1 << CollisionLayer, QueryTriggerInteraction.Ignore)) return false;
                }
                return true;
            }

            private void Apply(double[] x)
            {
                if (!Contains(x)) throw new InvalidOperationException("Attempted to submit an illegal Classroom rig.");
                for (int i = 0; i < lights.Length; i++)
                {
                    int k = i * LampDofs; var lamp = lights[i];
                    lamp.enabled = k < x.Length;
                    if (!lamp.enabled) continue;
                    lamp.transform.position = Position(x, k);
                    lamp.intensity = (float)(PowerMin * Math.Exp(x[k + 3] * LogPowerSpan));
                }
            }

            private NativeLamp[] Describe(double[] x) { Apply(x); return lights.Take(x.Length / LampDofs).Select(NativeLamp.Read).ToArray(); }

            internal double[] Perturb(int seed, bool near)
            {
                var rng = new System.Random(seed); var start = NoLights;
                for (int i = 0; i < Rig.Count; i++)
                {
                    var t = SingleLamp(Truth, i); var p = Position(t, 0); double q = PowerMin * Math.Exp(t[3] * LogPowerSpan);
                    var offset = near ? .09f + .06f * (float)rng.NextDouble() : .50f + .18f * (float)rng.NextDouble();
                    // Both camera arms start from the identical rig.
                    q = Math.Max(PowerMin, Math.Min(PowerMax, q * (rng.NextDouble() < .5 ? near ? .65 : .48 : near ? 1.48 : 1.85)));
                    for (int attempt = 0; ; attempt++)
                    {
                        if (attempt == 128) throw new InvalidOperationException("No admissible perturbed Classroom pose.");
                        var d = new Vector3((float)rng.NextDouble() - .5f, (float)rng.NextDouble() - .5f, (float)rng.NextDouble() - .5f).normalized;
                        var lamp = Encode(p + offset * d, q);
                        if (!Contains(lamp) || (near && Physics.SphereCast(p, SceneClearance, d, out _, offset,
                            1 << CollisionLayer, QueryTriggerInteraction.Ignore))) continue;
                        start = Join(start, lamp); break;
                    }
                }
                return start;
            }

            public void Dispose() { for (int i = owned.Count - 1; i >= 0; i--) if (owned[i]) Object.DestroyImmediate(owned[i]); }
        }

        [Serializable] private sealed class RoomGate
        {
            public int n, selectedViews, checkedViews, cameras, budget;
            public bool oneOlat, multiOlat, stable, allLampsVisible, proceed;
            public int[] contextCounts;
            public double[] repeatNrmse, positionResponse, powerResponse, oneLinearity, multiLinearity, oneAdditivity, multiAdditivity;
            public RoomObserved[] observability;
            public RoomRank[] ranking;
        }
        [Serializable] private sealed class RoomObserved { public int lamp; public double[] marginalNrmse; }
        [Serializable] private sealed class RoomRank { public int views, existing, bestNative, bestPredicted; public double spearman, top3Miss; }

        private static NativeRenderBackend ClassroomBackend(ClassroomFixture fixture, bool hi) => new NativeRenderBackend(fixture.Camera,
            fixture.Config.views.Select(v => v.position).ToArray(), fixture.Config.views.Select(v => v.rotation).ToArray(),
            hi ? ClassroomCheckWidth : ClassroomWidth, hi ? CheckSize : FitSize, -1);

        private static void RunClassroomScale(string task, string output)
        {
            if (!Application.isBatchMode) throw new InvalidOperationException("Classroom benchmark is batch-only (opens a transient scene).");
            BoundedLightWeights.CheckKnownCases();
            bool e10 = task.StartsWith("e10", StringComparison.Ordinal), smoke = task.EndsWith("smoke", StringComparison.Ordinal);
            string gateProfile = e10 ? (task == "e10-near" ? "e10_near" : "e10") : Arg("-bbScaleGate", "e10");
            if (gateProfile != "e10" && gateProfile != "e10_near") throw new ArgumentException("-bbScaleGate must be e10 or e10_near.");
            string tag = !e10 && gateProfile == "e10_near" ? (smoke ? "e11_near_smoke" : "e11_near") : task.Replace('-', '_');
            string dir = Path.Combine(output, tag);
            Directory.CreateDirectory(dir);
            using (var pipeline = new TestPipeline(classroom: true))
            using (var fixture = new ClassroomFixture(pipeline.Asset))
            using (var training = ClassroomBackend(fixture, false))
            using (var check = ClassroomBackend(fixture, true))
            {
                WriteJson(dir, "scene.json", fixture.Config);
                var rig = fixture.Rig;
                int budget = int.Parse(Arg("-bbScaleBudget", smoke ? "1200" : "6000"), Invariant);
                if (budget < 1000 || budget > 20000) throw new ArgumentException("Classroom paired camera budget must be 1000–20000.");
                var views = fixture.Config.views;
                var frames = new NativeRenderBackend.Frame[views.Length]; var high = new NativeRenderBackend.Frame[views.Length];
                var run = MakeScaleRun(task, fixture.Config.source, ClassroomMode, false, views, training, check);
                run.recoveryProfile = gateProfile;
                RoomGate gate;
                using (var setup = new QueryLog(dir, "shared.csv", "shared", 0))
                {
                    for (int v = 0; v < views.Length; v++)
                    {
                        int i = v;
                        ViewCapture(training, setup, rig, fixture.Truth, i, null, "warmup", null);
                        ViewCapture(training, setup, rig, fixture.Truth, i, null, "target", null, f => frames[i] = f);
                        ViewCapture(check, setup, rig, fixture.Truth, i, null, "target", Path.Combine(dir, $"target_view{i}"), f => high[i] = f);
                    }
                    gate = e10 ? CheckRoom(dir, fixture, training, setup, frames, budget) : null;
                    run.sharedRenders = setup.Cameras;
                }
                var rows = new List<ScaleTrial>();
                if (e10)
                {
                    WriteJson(dir, "gate.json", gate);
                    if (!gate.stable || !gate.allLampsVisible) throw new InvalidOperationException("E10 render repeatability/observability gate failed; see gate.json.");
                    foreach (int seed in smoke ? StructureSeeds.Take(1) : StructureSeeds)
                    {
                        var anchor = fixture.Perturb(seed, near: task == "e10-near");
                        foreach (int n in new[] { 1, 3 }) rows.Add(Trial("known", n, seed, 0, false, anchor));
                    }
                    var selected = rows.GroupBy(r => r.views).OrderByDescending(g => g.Count(r => r.passed)).ThenBy(g => g.Max(r => r.heldNrmse))
                        .ThenBy(g => g.Average(r => r.heldNrmse)).ThenBy(g => g.Key).First();
                    gate.selectedViews = selected.Key;
                    gate.proceed = !smoke && selected.Count(r => r.passed) >= 2 && selected.All(r => r.heldNrmse <= r.heldInitialNrmse + .005);
                    run.supportsOlat = selected.Key == 1 ? gate.oneOlat : gate.multiOlat;
                    WriteJson(dir, "gate.json", gate);
                }
                else
                {
                    var upstream = Path.Combine(output, gateProfile);
                    string gatePath = Path.Combine(upstream, "gate.json");
                    if (!File.Exists(gatePath)) throw new InvalidOperationException("Run full paired E10 in the same output directory before E11.");
                    gate = JsonUtility.FromJson<RoomGate>(File.ReadAllText(gatePath));
                    var prior = JsonUtility.FromJson<RoomConfig>(File.ReadAllText(Path.Combine(upstream, "scene.json")));
                    if (JsonUtility.ToJson(prior) != JsonUtility.ToJson(fixture.Config) || !gate.stable || !gate.allLampsVisible ||
                        gate.n != rig.Count || gate.selectedViews != 1 && gate.selectedViews != 3)
                        throw new InvalidOperationException("E11 configuration differs from the E10 gate.");
                    if (!smoke && !gate.proceed) throw new InvalidOperationException("Full E11 is gated on E10 known-rig recovery; only E11 smoke is available.");

                    int pool = int.Parse(Arg("-bbScalePool", "128"), Invariant);
                    if (pool < 8 || pool > 256 || (!smoke && pool < 128)) throw new ArgumentException("Invalid classroom candidate pool.");
                    int n = gate.selectedViews;
                    run.supportsOlat = n == 1 ? gate.oneOlat : gate.multiOlat;
                    string improved = run.supportsOlat ? SelectE9Arm(output) : null;
                    if (!smoke)
                    {
                        string screen = Path.Combine(output, gateProfile == "e10_near" ? "e11_near_smoke" : "e11_smoke", "stage.json");
                        if (!File.Exists(screen)) throw new InvalidOperationException("Run E11 smoke at the authored scale before the full matrix.");
                        var pilot = JsonUtility.FromJson<ScaleRun>(File.ReadAllText(screen));
                        if (pilot.supportsOlat != run.supportsOlat || pilot.trials.Any(r => r.lamps != rig.Count || r.views != n || r.budget != budget || r.pool != pool)
                            || !pilot.trials.Any(r => r.method != "multi_start" && !r.structureFailure))
                            throw new InvalidOperationException("E11 smoke did not clear the matching load/structural check.");
                    }
                    foreach (int seed in smoke ? StructureSeeds.Take(1) : StructureSeeds)
                    {
                        var rng = new System.Random(seed);
                        double[] anchor = NoLights;
                        while (anchor.Length < rig.Count * LampDofs)
                        {
                            var candidate = Enumerable.Range(0, LampDofs).Select(_ => rng.NextDouble()).ToArray();
                            if (rig.Contains(candidate)) anchor = Join(anchor, candidate);
                        }
                        rows.Add(Trial("multi_start", n, seed, pool, false, anchor));
                        rows.Add(Trial(LegacyInsertion, n, seed, pool, run.supportsOlat));
                        if (improved != null) rows.Add(Trial(improved, n, seed, pool, true));
                    }
                }
                if (training.RenderCount + check.RenderCount != run.sharedRenders + rows.Sum(r => r.trainingRenders + r.validationRenders))
                    throw new InvalidOperationException("Classroom camera counters do not match stage ledger.");
                FinishScaleRun(dir, run, rows);
                ScaleTrial Trial(string method, int n, int seed, int pool, bool olat, double[] anchor = null) => RunScaleTrial(
                    Path.Combine(dir, $"{seed}_{method}_{n}v"), task, "Classroom", rig, training, check, frames, high,
                    Enumerable.Range(0, n).ToArray(), 3, method, seed, budget, pool, olat, anchor);
            }
        }

        private static RoomGate CheckRoom(string dir, ClassroomFixture fixture, NativeRenderBackend backend, QueryLog log,
            NativeRenderBackend.Frame[] target, int budget)
        {
            var gate = new RoomGate { n = fixture.Rig.Count, checkedViews = 3, budget = budget };
            var rig = fixture.Rig; int n = rig.Count; int start = log.Cameras;
            var anchors = Enumerable.Range(0, n).Select(i => SingleLamp(fixture.Truth, i)).ToArray();
            var empty = Frames(NoLights, "check_empty");
            gate.repeatNrmse = target.Select((f, i) => Math.Sqrt(ViewCapture(backend, log, rig, fixture.Truth, i, f, "repeat", null))).ToArray();
            var singles = anchors.Select((a, i) => Frames(WithPower(a, Q0), $"check_unit{i}")).ToArray();
            var shift = (double[])fixture.Truth.Clone();
            bool legalProbe = false;
            foreach (double h in new[] { .06, .03, .015, .0075 })
            {
                foreach (int axis in new[] { 1, 0, 2 })
                {
                    foreach (int sign in new[] { -1, 1 })
                    {
                        shift[axis] += sign * h;
                        if (rig.Contains(shift)) { legalProbe = true; break; }
                        shift[axis] = fixture.Truth[axis];
                    }
                    if (legalProbe) break;
                }
                if (legalProbe) break;
            }
            if (!legalProbe) throw new InvalidOperationException("No legal response probe.");
            var position = Frames(shift, "check_position");
            var dim = (double[])fixture.Truth.Clone(); dim[3] = PowerCoordinate(.55);
            var intensity = Frames(dim, "check_intensity");
            gate.positionResponse = position.Select((f, i) => Math.Sqrt(NativeRenderBackend.Compare(f, target[i]))).ToArray();
            gate.powerResponse = intensity.Select((f, i) => Math.Sqrt(NativeRenderBackend.Compare(f, target[i]))).ToArray();
            gate.observability = anchors.Select((_, j) => new RoomObserved { lamp = j + 1,
                marginalNrmse = Frames(RemoveLamp(fixture.Truth, j), $"check_marginal{j}")
                    .Select((f, i) => Math.Sqrt(NativeRenderBackend.Compare(f, target[i]))).ToArray() }).ToArray();
            gate.stable = gate.repeatNrmse.All(x => x <= 1e-6) && gate.positionResponse.Max() >= 1e-4 && gate.powerResponse.Max() >= 1e-4;
            gate.allLampsVisible = gate.observability.All(o => o.marginalNrmse.Take(3).Max() >= 1e-3);
            var bases = singles.Select(f => Delta(f, empty)).ToArray();
            var line1 = new List<double>(); var line3 = new List<double>();
            var add1 = new List<double>(); var add3 = new List<double>();
            foreach (int j in new[] { 0, n / 2, n - 1 }.Distinct())
                foreach (double p in new[] { .25, .5, 1, 2 })
                    Errors(Delta(Frames(WithPower(anchors[j], p), $"check_linear_{j}_{F(p)}"), empty), ScaleBasis(j, p), line1, line3);
            foreach (int count in new[] { 1, Math.Min(5, n), n }.Distinct())
            {
                var ids = count == 1 ? new[] { 0 } : Enumerable.Range(0, count - 1).Append(n - 1).ToArray();
                var theta = ids.SelectMany(i => WithPower(anchors[i], Q0)).ToArray();
                var delta = Delta(Frames(theta, $"check_superpose{count}"), empty);
                var predicted = EmptyRgb();
                foreach (int j in ids) for (int v = 0; v < 3; v++) for (int k = 0; k < predicted[v].Length; k++) predicted[v][k] += bases[j][v][k];
                Errors(delta, predicted, add1, add3);
            }
            var pool = CandidateSet(16, LampDofs, 19273, true, rig.Contains).Select(p => WithPower(p, Q0)).ToArray();
            var dictionary = pool.Select((p, i) => Delta(Frames(p, $"check_dict{i}"), empty)).ToArray();
            gate.contextCounts = new[] { 1, Math.Min(5, n - 1), n - 1 }.Distinct().ToArray();
            var ranks = new List<RoomRank>();
            using (var rankLog = new StreamWriter(Path.Combine(dir, "e10_ranking.csv")))
            {
                rankLog.WriteLine("existing,candidate,optimal_predicted_score_1v,optimal_native_score_1v,optimal_predicted_score_3v,optimal_native_score_3v");
                foreach (int count in gate.contextCounts)
                {
                    var context = anchors.Take(count).SelectMany(a => a).ToArray();
                    var current = Frames(context, $"check_context{count}");
                    var residual = Delta(target.Take(3).ToArray(), current);
                    var old = current.Select((f, i) => NativeRenderBackend.Compare(f, target[i])).ToArray();
                    var p1 = new double[pool.Length]; var p3 = new double[pool.Length]; var m1 = new double[pool.Length]; var m3 = new double[pool.Length];
                    for (int i = 0; i < pool.Length; i++)
                    {
                        var dots = dictionary[i].Select((a, v) => RgbDot(a, residual[v]) / (target[v].Energy + 1e-12)).ToArray();
                        var energy = dictionary[i].Select((a, v) => RgbDot(a, a) / (target[v].Energy + 1e-12)).ToArray();
                        p1[i] = p3[i] = m1[i] = m3[i] = double.NegativeInfinity;
                        foreach (double p in PowerGrid)
                        {
                            var actual = Frames(Join(context, WithPower(pool[i], p)), $"check_grid{count}_{i}_{F(p)}")
                                .Select((f, v) => old[v] - NativeRenderBackend.Compare(f, target[v])).ToArray();
                            p1[i] = Math.Max(p1[i], PredictedGain(dots[0], energy[0], p));
                            p3[i] = Math.Max(p3[i], PredictedGain(dots.Average(), energy.Average(), p));
                            m1[i] = Math.Max(m1[i], actual[0]); m3[i] = Math.Max(m3[i], actual.Average());
                        }
                        rankLog.WriteLine(string.Join(",", count, i, F(p1[i]), F(m1[i]), F(p3[i]), F(m3[i])));
                    }
                    ranks.Add(Rank(1, count, p1, m1)); ranks.Add(Rank(3, count, p3, m3));
                }
            }
            gate.oneLinearity = line1.ToArray(); gate.multiLinearity = line3.ToArray(); gate.oneAdditivity = add1.ToArray(); gate.multiAdditivity = add3.ToArray();
            gate.ranking = ranks.ToArray();
            gate.oneOlat = PassErrors(line1) && PassErrors(add1) && ranks.Where(r => r.views == 1).All(PassRank);
            gate.multiOlat = PassErrors(line3) && PassErrors(add3) && ranks.Where(r => r.views == 3).All(PassRank);
            gate.cameras = log.Cameras - start;
            Debug.Log($"E10 gate: {n} lights, stable {gate.stable}, observed {gate.allLampsVisible}, OLAT 1v={gate.oneOlat}/3v={gate.multiOlat}, {gate.cameras} captures");
            return gate;

            NativeRenderBackend.Frame[] Frames(double[] theta, string phase)
            {
                var frames = new NativeRenderBackend.Frame[3];
                for (int v = 0; v < 3; v++) { int i = v; ViewCapture(backend, log, rig, theta, i, null, phase, null, f => frames[i] = f); }
                return frames;
            }
            float[][] Delta(NativeRenderBackend.Frame[] a, NativeRenderBackend.Frame[] b) => a.Select((f, v) => Difference(f, b[v])).ToArray();
            float[][] EmptyRgb() => Enumerable.Range(0, 3).Select(v => new float[target[v].PixelCount * 3]).ToArray();
            float[][] ScaleBasis(int i, double p) => bases[i].Select(a => a.Select(x => (float)(x * p)).ToArray()).ToArray();

            void Errors(float[][] actual, float[][] predicted, List<double> one, List<double> multi)
            {
                double err = 0, norm = 0;
                for (int v = 0; v < 3; v++)
                {
                    double e = 0;
                    for (int k = 0; k < actual[v].Length; k++) { double d = (double)actual[v][k] - predicted[v][k]; e += d * d; }
                    double a = RgbDot(actual[v], actual[v]);
                    if (v == 0 && a > 1e-12) one.Add(Math.Sqrt(e / a));
                    double z = target[v].Energy + 1e-12; err += e / z; norm += a / z;
                }
                if (norm > 1e-12) multi.Add(Math.Sqrt(err / norm));
            }
            bool PassErrors(List<double> e) => e.Count > 0 && Median(e) <= .02 && e.Max() <= .05;
            bool PassRank(RoomRank r) => r.spearman >= .9 && r.top3Miss <= .01;
            RoomRank Rank(int v, int count, double[] predicted, double[] native)
            {
                var order = Order(predicted); int winner = Order(native)[0];
                var gain = native[winner];
                return new RoomRank { views = v, existing = count, bestNative = winner, bestPredicted = order[0], spearman = Spearman(predicted, native),
                    top3Miss = Math.Max(0, gain - order.Take(3).Max(i => native[i])) / Math.Max(1e-9, Math.Abs(gain)) };
            }
        }

        private static string SelectE9Arm(string output)
        {
            var path = Path.Combine(output, "e9_2000", "stage.json");
            if (!File.Exists(path)) throw new InvalidOperationException("Run full E9 before comparing cold-start methods in E11.");
            var groups = JsonUtility.FromJson<ScaleRun>(File.ReadAllText(path)).trials.Where(r => r.pool == 64).GroupBy(r => r.method)
                .OrderByDescending(g => g.Count(r => r.passed)).ThenBy(g => g.Max(r => r.bestLoss)).ThenBy(g => g.Average(r => r.trainingRenders)).ToArray();
            return groups[0].Key == LegacyInsertion ? null : groups[0].Key;
        }
    }
}

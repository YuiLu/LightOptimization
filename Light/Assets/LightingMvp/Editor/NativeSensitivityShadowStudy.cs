using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using Frame = LightingMvp.NativeRenderBackend.Frame;

namespace LightingMvp.Editor
{
    /// <summary>E18: fixture self-shadowing, resolution consistency, oracle loss paths and shadow-aware optimizers.</summary>
    public static partial class NativeRecoveryExperiment
    {
        private const int E18FitW = 448, E18FitH = 256, E18HighW = 896, E18HighH = 512;
        private static readonly int[] E18Lamps = { 0, 3, 4, 5 };
        private static readonly Vector3[] E18Directions =
        {
            new Vector3(1, 0, 0), new Vector3(-1, 0, 0), new Vector3(0, 0, 1),
            new Vector3(0, 0, -1), new Vector3(0, 1, 0), new Vector3(.7071f, .45f, .45f)
        };

        private static double[] E18LampStart(ISensitivityScene scene, int lamp, float distanceCm, int direction, out float usedCm)
        {
            var origin = scene.PositionOf(scene.Truth, lamp);
            var unit = E18Directions[direction % E18Directions.Length].normalized;
            for (float scale = 1; scale >= .3f; scale *= .82f)
            {
                var candidate = scene.WithPosition(scene.Truth, lamp, origin + unit * (distanceCm / 100f * scale));
                if (!scene.Contains(candidate)) continue;
                usedCm = distanceCm * scale;
                return candidate;
            }
            for (int k = 1; k < 12; ++k)
            {
                var rotated = Quaternion.Euler(0, 30 * k, 0) * unit;
                var candidate = scene.WithPosition(scene.Truth, lamp, origin + rotated * (distanceCm / 100f));
                if (!scene.Contains(candidate)) continue;
                usedCm = distanceCm;
                return candidate;
            }
            throw new InvalidOperationException($"No feasible {distanceCm} cm start for L{lamp + 1}.");
        }

        [Serializable] private sealed class E18MapRow
        {
            public int slot, fixtureTriangles;
            public string label, sourceName, unityName, type, fixtureRenderers;
            public float watts, range, intensity;
            public Vector3 position, fixtureMin, fixtureMax;
        }
        [Serializable] private sealed class E18Condition
        {
            public string condition;
            public double shadowRms, startShadowRms, affected, startAffected, fixtureRms, fixtureAffected, selfShadowRms, targetEnergy, fixtureOverlap;
            public double meanColumn, conditionNumber, residualAlignment;
            public double[] columnNorm, stepConsistency;
            public float[] stepCm;
        }
        [Serializable] private sealed class E18Ablation
        {
            public int lamp;
            public double startPositionCm, startIntensityError;
            public E18Condition[] conditions;
        }
        [Serializable] private sealed class E18Resolution
        {
            public int lamp;
            public string condition, parameters;
            public double nrmse448, nrmse896, targetMismatch, renderMismatch, nrmse896Downsampled;
            public double shadowSignal448, shadowSignal896;
            public string perView448, perView896, perViewXTarget, perViewRender;
        }
        [Serializable] private sealed class E18Path
        {
            public int lamp;
            public string shadows;
            public double distanceCm, direction, startPositionCm, startIntensityError;
            public double[] loss448, t;
            public bool passed;
            public double positionErrorCm, relativeIntensityError;
        }
        [Serializable] private sealed class E18Checkpoint
        {
            public double fraction, loss, nrmse, condition, consistency;
            public int query;
            public double[] x, columnNorm, stepCm;
        }
        [Serializable] private sealed class E18Arm
        {
            public string method, condition;
            public int lamp, budget, queries, renders, accepted, rejected, jacobians;
            public double initialNrmse448, finalNrmse448, bestNrmse448, globalNrmse448, nrmseFull448, nrmse896, heldNrmse896, positionErrorCm, relativeIntensityError, wallSeconds;
            public string viewNrmse448;
            public bool passed;
            public string stop;
            public double[] start, best;
            public string[] stages;
            public double[] stageLoss;
            public E18Checkpoint[] checkpoints;
        }
        [Serializable] private sealed class E18Run
        {
            public string scene, task, error, startSource;
            public int seed, budget;
            public double elapsed;
            public int[] fixtureSplit;
            public E18MapRow[] map;
            public E18Ablation[] ablations;
            public E18Resolution[] resolutions;
            public E18Path[] paths;
            public E18Arm[] arms;
        }

        private static void RunShadowStudy(string task, string output)
        {
            if (!Application.isBatchMode) throw new InvalidOperationException("E18 opens transient scenes and is batch-only.");
            if (!UnityEngine.Rendering.GraphicsSettings.lightsUseLinearIntensity)
                throw new InvalidOperationException("Enable linear light intensity for calibrated color coordinates.");
            bool room = task.StartsWith("e18-room", StringComparison.Ordinal);
            string caseName = task.StartsWith("e18-room-", StringComparison.Ordinal) ? task.Substring(9)
                : task.StartsWith("e18-", StringComparison.Ordinal) ? task.Substring(4) : task;
            string dir = Path.Combine(output, task.Replace('-', '_'));
            Directory.CreateDirectory(dir);
            int budget = int.Parse(Arg("-bbE18Budget", "800"), Invariant);
            int seed = int.Parse(Arg("-bbE18Seed", "131"), Invariant);
            using var pipeline = new TestPipeline(classroom: !room);
            using ISensitivityScene scene = room ? (ISensitivityScene)new RoomSensitivityScene() : new ClassroomSensitivityScene(pipeline.Asset, null);
            new ShadowStudyRunner(dir, scene, task, budget, seed).Run();
        }

        private sealed class RgbImage
        {
            internal readonly int Width, Height;
            internal readonly float[] Data;
            internal readonly double Energy;
            private RgbImage(int w, int h, float[] data)
            {
                Width = w; Height = h; Data = data;
                double e = 0;
                for (int i = 0; i < data.Length; i += 3) e += (double)data[i] * data[i] + (double)data[i + 1] * data[i + 1] + (double)data[i + 2] * data[i + 2];
                Energy = e;
            }
            internal static RgbImage From(Frame frame, int w, int h)
            {
                var data = new float[w * h * 3];
                for (int i = 0, k = 0; i < w * h; ++i)
                {
                    var c = frame.Pixel(i);
                    data[k++] = c.r; data[k++] = c.g; data[k++] = c.b;
                }
                return new RgbImage(w, h, data);
            }
            internal RgbImage Box(int factor)
            {
                int w = Width / factor, h = Height / factor;
                var data = new float[w * h * 3];
                for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                {
                    double r = 0, g = 0, b = 0;
                    for (int dy = 0; dy < factor; ++dy)
                    for (int dx = 0; dx < factor; ++dx)
                    {
                        int s = ((y * factor + dy) * Width + x * factor + dx) * 3;
                        r += Data[s]; g += Data[s + 1]; b += Data[s + 2];
                    }
                    int k = (y * w + x) * 3, n = factor * factor;
                    data[k] = (float)(r / n); data[k + 1] = (float)(g / n); data[k + 2] = (float)(b / n);
                }
                return new RgbImage(w, h, data);
            }
            internal double Residual(RgbImage target, float[] buffer, int offset)
            {
                double error = 0;
                double denominator = target.Energy + 1e-12;
                double scale = 1 / Math.Sqrt(denominator);
                for (int i = 0; i < Data.Length; ++i)
                {
                    double d = Data[i] - target.Data[i];
                    error += d * d;
                    buffer[offset + i] = (float)(d * scale);
                }
                return error / denominator;
            }
            internal static RgbImage Average(List<RgbImage> images)
            {
                var data = new float[images[0].Data.Length];
                foreach (var image in images)
                    for (int i = 0; i < data.Length; ++i) data[i] += image.Data[i];
                for (int i = 0; i < data.Length; ++i) data[i] /= images.Count;
                return new RgbImage(images[0].Width, images[0].Height, data);
            }
            internal static double Nrmse(RgbImage a, RgbImage b)
            {
                if (a.Width != b.Width || a.Height != b.Height) throw new ArgumentException("Mismatched image sizes.");
                double error = 0;
                for (int i = 0; i < a.Data.Length; ++i) { double d = a.Data[i] - b.Data[i]; error += d * d; }
                return Math.Sqrt(error / (b.Energy + 1e-12) / 3);
            }
        }

        private sealed class ShadowStudyRunner
        {
            private readonly string dir, task, caseName;
            private readonly ISensitivityScene scene;
            private readonly int seed, budget;
            private readonly NativeRenderBackend fit, high;
            private readonly E18Run run = new E18Run();
            private readonly System.Diagnostics.Stopwatch watch = System.Diagnostics.Stopwatch.StartNew();
            private readonly List<E18MapRow> map = new List<E18MapRow>();
            private readonly List<E18Ablation> ablations = new List<E18Ablation>();
            private readonly List<E18Resolution> resolutions = new List<E18Resolution>();
            private readonly List<E18Path> paths = new List<E18Path>();
            private readonly List<E18Arm> arms = new List<E18Arm>();
            private readonly Dictionary<string, Frame[]> lowTargets = new Dictionary<string, Frame[]>();
            private readonly Dictionary<string, Frame[]> highTargets = new Dictionary<string, Frame[]>();
            private StreamWriter armCsv, checkpointCsv, traceCsv, queryCsv, pathCsv;
            private double[] lastColumnNorm = new double[LampDofs], lastSteps = new double[LampDofs];
            private float[] e18ColumnA, e18ColumnB;
            private bool e18HasPreviousColumn;
            private double lastCondition = 0, lastConsistency = 1;
            private int queryCount, renderCount, thresholdQueries;
            private double nextFraction;
            private List<E18Checkpoint> activeCheckpoints;
            private bool room => scene.Name != "classroom";

            internal ShadowStudyRunner(string dir, ISensitivityScene scene, string task, int budget, int seed)
            {
                this.dir = dir; this.scene = scene; this.task = task; this.budget = budget; this.seed = seed;
                caseName = task.StartsWith("e18-room-", StringComparison.Ordinal) ? task.Substring(9)
                    : task.StartsWith("e18-", StringComparison.Ordinal) ? task.Substring(4) : task;
                fit = new NativeRenderBackend(scene.Camera, scene.Views, scene.Rotations, E18FitW, E18FitH, -1);
                high = new NativeRenderBackend(scene.Camera, scene.Views, scene.Rotations, E18HighW, E18HighH, -1);
                run.seed = seed; run.budget = budget; run.scene = scene.Name; run.task = task;
            }

            internal void Run()
            {
                try
                {
                    armCsv = new StreamWriter(Path.Combine(dir, "arms.csv"));
                    armCsv.WriteLine("method,lamp,condition,startCondition,budget,queries,renders,jacobians,accepted,rejected,stop,initialMeanViewNrmse,finalMeanViewNrmse,bestMeanViewNrmse,globalNrmse448,viewNrmse448,nrmseFull448,nrmse896,heldNrmse896,positionErrorCm,relativeIntensityError,wallSeconds,passed");
                    checkpointCsv = new StreamWriter(Path.Combine(dir, "checkpoints.csv"));
                    checkpointCsv.WriteLine("method,lamp,condition,fraction,query,loss,nrmse,condition,consistency,x0,x1,x2,x3,step_cm,column_norm");
                    traceCsv = new StreamWriter(Path.Combine(dir, "trace.csv"));
                    traceCsv.WriteLine("method,lamp,condition,batch,query,mu,predicted,actual,ratio,status");
                    queryCsv = new StreamWriter(Path.Combine(dir, "queries.csv"));
                    queryCsv.WriteLine("method,lamp,condition,query,phase,x,y,z,power,loss");
                    pathCsv = new StreamWriter(Path.Combine(dir, "path-loss.csv"));
                    pathCsv.WriteLine("lamp,shadows,distanceCm,direction,t,loss");

                    scene.ResetShadows();
                    ResetFixtures();
                    scene.Apply(scene.Truth);
                    var cfg = scene.Describe(); cfg.task = task; cfg.budget = budget; cfg.seed = seed;
                    run.startSource = "fixed_distance_directions";
                    WriteJson(dir, "scene.json", cfg);
                    BuildMap();
                    if (caseName == "map") MapProbe();
                    else if (caseName == "0") Ablation();
                    else if (caseName == "a") Resolution();
                    else if (caseName == "c") Paths();
                    else if (caseName == "b") Methods(new[] { 0, 5, 4 });
                    else if (caseName == "d") RoomContrast();
                    else if (caseName == "diag") Consistency();
                    else if (caseName == "arm1") OneArm();
                    else if (caseName == "views") ViewAudit();
                    else if (caseName == "all")
                    {
                        Ablation(); Resolution(); Paths();
                        Methods(new[] { 0, 5, 4 });
                        if (room) RoomContrast();
                    }
                    else throw new ArgumentException("Unknown E18 case: " + task);

                    run.map = map.ToArray(); run.ablations = ablations.ToArray(); run.resolutions = resolutions.ToArray();
                    run.paths = paths.ToArray(); run.arms = arms.ToArray();
                    run.fixtureSplit = scene.FixtureTriangleCounts();
                    run.elapsed = watch.Elapsed.TotalSeconds;
                    WriteJson(dir, "summary.json", run);
                    Debug.Log($"[E18] {scene.Name}: {task} finished in {run.elapsed:F1}s ({arms.Count} arms, {ablations.Count} ablations)");
                }
                catch (Exception ex)
                {
                    run.error = ex.GetType().Name + ": " + ex.Message;
                    try { WriteJson(dir, "summary.json", run); } catch { }
                    throw;
                }
                finally
                {
                    armCsv?.Dispose(); checkpointCsv?.Dispose(); traceCsv?.Dispose(); queryCsv?.Dispose(); pathCsv?.Dispose();
                    fit.Dispose(); high.Dispose();
                }
            }

            private void ResetFixtures()
            {
                for (int i = 0; i < scene.Count; ++i) scene.SetFixtureShadows(i, true);
            }

            private void SetCondition(int lamp, string condition)
            {
                scene.ResetShadows(); ResetFixtures();
                if (condition == "fixture_off")
                {
                    if (!scene.SupportsFixtureShadows) throw new InvalidOperationException("Scene has no fixture caster split.");
                    scene.SetFixtureShadows(lamp, false);
                }
                else if (condition == "light_shadow_off") scene.SetShadows(lamp, false);
                else if (condition != "full") throw new ArgumentException("Unknown shadow condition: " + condition);
            }

            private Frame[] Targets(int lamp, string condition, bool highRes)
            {
                string key = lamp + ":" + condition;
                var cache = highRes ? highTargets : lowTargets;
                if (cache.TryGetValue(key, out var cached)) return cached;
                SetCondition(lamp, condition);
                scene.Apply(scene.Truth);
                var frames = new Frame[scene.Views.Length];
                var backend = highRes ? high : fit;
                for (int v = 0; v < frames.Length; ++v) frames[v] = backend.Capture(v);
                cache[key] = frames;
                return frames;
            }

            private Frame[] Targets(int lamp, string condition) => Targets(lamp, condition, false);

            private void BuildMap()
            {
                var gt = scene.Describe().gt;
                var split = scene.FixtureTriangleCounts();
                for (int i = 0; i < scene.Count; ++i)
                {
                    var bounds = scene.FixtureBounds(i);
                    bool hasBounds = bounds.size != Vector3.zero;
                    map.Add(new E18MapRow
                    {
                        slot = i, label = "L" + (i + 1), sourceName = scene.SourceName(i), unityName = scene.UnityName(i),
                        type = gt[i].type, position = scene.PositionOf(scene.Truth, i), intensity = (float)scene.PowerOf(scene.Truth, i),
                        range = gt[i].range, fixtureTriangles = split.Length > i ? split[i] : 0,
                        fixtureRenderers = string.Join("|", scene.FixtureRenderers(i)),
                        fixtureMin = hasBounds ? bounds.min : Vector3.zero, fixtureMax = hasBounds ? bounds.max : Vector3.zero
                    });
                    Debug.Log($"[E18] {scene.Name} {map[map.Count - 1].label}: {scene.SourceName(i)} at {map[map.Count - 1].position:F3}, fixture triangles {map[map.Count - 1].fixtureTriangles}");
                }
                using (var csv = new StreamWriter(Path.Combine(dir, "lamp-map.csv")))
                {
                    csv.WriteLine("label,slot,source_name,unity_name,type,x,y,z,intensity,range,fixture_triangles,fixture_min_x,fixture_min_y,fixture_min_z,fixture_max_x,fixture_max_y,fixture_max_z,fixture_renderers");
                    foreach (var r in map)
                        csv.WriteLine(string.Join(",", r.label, r.slot.ToString(Invariant), Csv(r.sourceName), Csv(r.unityName), r.type,
                            F(r.position.x), F(r.position.y), F(r.position.z), F(r.intensity), F(r.range), r.fixtureTriangles.ToString(Invariant),
                            F(r.fixtureMin.x), F(r.fixtureMin.y), F(r.fixtureMin.z), F(r.fixtureMax.x), F(r.fixtureMax.y), F(r.fixtureMax.z), Csv(r.fixtureRenderers)));
                }
                DrawLampMap();
            }

            private static string Csv(string value) => "\"" + (value ?? "").Replace("\"", "\"\"") + "\"";

            private static readonly Color32[] LampColors =
            {
                new Color32(239, 68, 68, 255), new Color32(249, 115, 22, 255), new Color32(234, 179, 8, 255), new Color32(34, 197, 94, 255),
                new Color32(59, 130, 246, 255), new Color32(168, 85, 247, 255), new Color32(236, 72, 153, 255)
            };

            private void DrawLampMap()
            {
                int w = 1024, h = 720;
                var cfg = scene.Describe();
                var center = (cfg.boundsMin + cfg.boundsMax) * .5f;
                var top = Panel(w, h, new Vector3(center.x, cfg.boundsMax.y - .015f, center.z), Quaternion.Euler(90, 0, 0), true);
                var eye = scene.Views[0]; var rot = scene.Rotations[0];
                var perspective = Panel(w, h, eye, rot, false);
                var combined = new Color32[w * h * 2];
                Array.Copy(top, 0, combined, 0, top.Length);
                Array.Copy(perspective, 0, combined, w * h, perspective.Length);
                var tex = new Texture2D(w, h * 2, TextureFormat.RGB24, false, false);
                try
                {
                    tex.SetPixels32(combined); tex.Apply();
                    File.WriteAllBytes(Path.Combine(dir, "classroom-lamp-map.png"), tex.EncodeToPNG());
                }
                finally { UnityEngine.Object.DestroyImmediate(tex); }
            }

            private Color32[] Panel(int w, int h, Vector3 eye, Quaternion rotation, bool orthographic)
            {
                var cfg = scene.Describe();
                var go = new GameObject("E18 lamp map camera") { hideFlags = HideFlags.HideAndDontSave };
                try
                {
                    var cam = go.AddComponent<Camera>();
                    cam.CopyFrom(scene.Camera);
                    cam.enabled = false; cam.orthographic = orthographic; cam.transform.SetPositionAndRotation(eye, rotation);
                    cam.aspect = (float)w / h; cam.ResetProjectionMatrix();
                    if (orthographic)
                    {
                        float zExtent = cfg.boundsMax.z - cfg.boundsMin.z, xExtent = cfg.boundsMax.x - cfg.boundsMin.x;
                        cam.orthographicSize = Mathf.Max(zExtent * .5f, xExtent * .5f / cam.aspect) * 1.06f;
                        cam.nearClipPlane = .02f; cam.farClipPlane = 8;
                    }
                    Color32[] pixels;
                    using (var backend = new NativeRenderBackend(cam, new[] { eye }, new[] { rotation }, w, h, -1))
                    {
                        scene.Apply(scene.Truth); ResetFixtures(); scene.ResetShadows();
                        var frame = backend.Capture(0);
                        pixels = new Color32[frame.PixelCount];
                        for (int i = 0; i < pixels.Length; ++i)
                        {
                            var c = frame.Pixel(i);
                            pixels[i] = new Color32(ToByte(NativeRenderBackend.Display(c.r)), ToByte(NativeRenderBackend.Display(c.g)), ToByte(NativeRenderBackend.Display(c.b)), 255);
                        }
                    }
                    var ov = new Color32[w * h];
                    foreach (var row in map)
                    {
                        var color = LampColors[row.slot % LampColors.Length];
                        Vector2 anchor;
                        if (row.fixtureMax != Vector3.zero)
                        {
                            var min = row.fixtureMin; var max = row.fixtureMax;
                            var corners = new[]
                            {
                                new Vector3(min.x, min.y, min.z), new Vector3(max.x, min.y, min.z), new Vector3(max.x, min.y, max.z), new Vector3(min.x, min.y, max.z),
                                new Vector3(min.x, max.y, min.z), new Vector3(max.x, max.y, min.z), new Vector3(max.x, max.y, max.z), new Vector3(min.x, max.y, max.z)
                            };
                            var uv = corners.Select(p => cam.WorldToViewportPoint(p)).ToArray();
                            int[] edges = { 0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7 };
                            for (int e = 0; e < edges.Length; e += 2)
                            {
                                if (uv[edges[e]].z <= 0 || uv[edges[e + 1]].z <= 0) continue;
                                Line(ov, w, h, ToScreen(uv[edges[e]], w, h), ToScreen(uv[edges[e + 1]], w, h), color);
                            }
                            anchor = ToScreen(uv.Aggregate(Vector3.zero, (a, b) => a + b) / 8f, w, h);
                        }
                        else
                        {
                            var uv = cam.WorldToViewportPoint(row.position);
                            if (uv.z <= 0) continue;
                            anchor = ToScreen(uv, w, h);
                            Box(ov, w, h, anchor, 9, color);
                        }
                        Text(ov, w, h, anchor + new Vector2(12, -18), row.label, color, 3);
                    }
                    for (int y = 0; y < h; ++y)
                    {
                        int src = y * w, dst = (h - 1 - y) * w;
                        for (int x = 0; x < w; ++x) { var o = ov[src + x]; if (o.a != 0) pixels[dst + x] = o; }
                    }
                    return pixels;
                }
                finally { UnityEngine.Object.DestroyImmediate(go); }
            }

            private static byte ToByte(float v) => (byte)Mathf.Clamp(Mathf.RoundToInt(v * 255), 0, 255);
            private static Vector2 ToScreen(Vector3 vp, int w, int h) => new Vector2(vp.x * w, (1 - vp.y) * h);

            private static void Pixel(Color32[] px, int w, int h, int x, int y, Color32 c)
            {
                if (x < 0 || y < 0 || x >= w || y >= h) return;
                px[y * w + x] = c;
            }
            private static void Box(Color32[] px, int w, int h, Vector2 c, int r, Color32 color)
            {
                for (int i = -r; i <= r; ++i) { Pixel(px, w, h, (int)c.x + i, (int)c.y - r, color); Pixel(px, w, h, (int)c.x + i, (int)c.y + r, color); Pixel(px, w, h, (int)c.x - r, (int)c.y + i, color); Pixel(px, w, h, (int)c.x + r, (int)c.y + i, color); }
            }
            private static void Line(Color32[] px, int w, int h, Vector2 a, Vector2 b, Color32 color)
            {
                int x0 = (int)a.x, y0 = (int)a.y, x1 = (int)b.x, y1 = (int)b.y;
                int dx = Math.Abs(x1 - x0), sx = x0 < x1 ? 1 : -1, dy = -Math.Abs(y1 - y0), sy = y0 < y1 ? 1 : -1, err = dx + dy;
                int guard = dx + Math.Abs(dy) + 2;
                while (guard-- > 0 && (x0 != x1 || y0 != y1))
                {
                    Pixel(px, w, h, x0, y0, color);
                    int e2 = 2 * err;
                    if (e2 >= dy) { err += dy; x0 += sx; }
                    if (e2 <= dx) { err += dx; y0 += sy; }
                }
                Pixel(px, w, h, x1, y1, color);
            }

            private static readonly string[] GlyphL = { "10000", "10000", "10000", "10000", "10000", "10000", "11111" };
            private static readonly string[][] Glyphs =
            {
                new[] { "01110","10001","10011","10101","11001","10001","01110" },
                new[] { "00100","01100","00100","00100","00100","00100","01110" },
                new[] { "01110","10001","00001","00010","00100","01000","11111" },
                new[] { "11111","00010","00100","00010","00001","10001","01110" },
                new[] { "00010","00110","01010","10010","11111","00010","00010" },
                new[] { "11111","10000","11110","00001","00001","10001","01110" },
                new[] { "00110","01000","10000","11110","10001","10001","01110" },
                new[] { "11111","00001","00010","00100","01000","01000","01000" }
            };
            private static string[] Glyph(char c) => c == 'L' ? GlyphL : c >= '0' && c <= '7' ? Glyphs[c - '0'] : null;
            private static void Text(Color32[] px, int w, int h, Vector2 at, string text, Color32 color, int scale)
            {
                int x = (int)at.x, y = (int)at.y;
                foreach (char c in text)
                {
                    var glyph = Glyph(c);
                    if (glyph == null) { x += 6 * scale; continue; }
                    for (int row = 0; row < 7; ++row)
                        for (int col = 0; col < 5; ++col)
                            if (glyph[row][col] == '1')
                                for (int dy = 0; dy < scale; ++dy)
                                    for (int dx = 0; dx < scale; ++dx)
                                        Pixel(px, w, h, x + col * scale + dx, y + row * scale + dy, color);
                    x += 6 * scale;
                }
            }

            private double[] LampStart(int lamp, float distanceCm, int direction, out float usedCm) =>
                E18LampStart(scene, lamp, distanceCm, direction, out usedCm);

            private E18Condition ConditionStats(int lamp, string condition, double[] start, float[][] reference, out float[][] own)
            {
                var stats = new E18Condition { condition = condition, columnNorm = new double[LampDofs], stepConsistency = new double[LampDofs], stepCm = new float[LampDofs] };
                var fullTarget = Targets(lamp, "full");
                var ownTarget = Targets(lamp, condition);
                var offTarget = Targets(lamp, "light_shadow_off");
                var views = scene.Views.Length;
                double sumEnergy = 0, sumAffected = 0, sumFixture = 0, sumFixtureAffected = 0, sumSelf = 0, sumOverlap = 0;
                var rect = ProjectedRect(scene.FixtureBounds(lamp), 0);
                for (int v = 0; v < views; ++v)
                {
                    var a = fullTarget[v]; var b = offTarget[v];
                    double diff = 0, energy = 0, peak = 0; int affected = 0, inFixture = 0;
                    for (int p = 0; p < a.PixelCount; ++p)
                    {
                        var x = a.Pixel(p); var y = b.Pixel(p);
                        double d0 = x.r - y.r, d1 = x.g - y.g, d2 = x.b - y.b;
                        double m = Math.Sqrt(d0 * d0 + d1 * d1 + d2 * d2);
                        diff += d0 * d0 + d1 * d1 + d2 * d2; energy += x.r * x.r + x.g * x.g + x.b * x.b;
                        peak = Math.Max(peak, m);
                    }
                    double cutoff = Math.Max(1e-5, .01 * peak);
                    for (int p = 0; p < a.PixelCount; ++p)
                    {
                        var x = a.Pixel(p); var y = b.Pixel(p);
                        double m = Math.Sqrt(Sq(x.r - y.r) + Sq(x.g - y.g) + Sq(x.b - y.b));
                        if (m <= cutoff) continue;
                        ++affected;
                        int px = p % E18FitW, py = p / E18FitW;
                        if (v == 0 && rect.Contains(new Vector2(px / (float)E18FitW, 1 - py / (float)E18FitH))) ++inFixture;
                    }
                    sumEnergy += Math.Sqrt(diff / (energy + 1e-12));
                    sumAffected += affected / (double)a.PixelCount;
                    sumOverlap += affected > 0 ? inFixture / (double)affected : 0;
                    if (condition == "fixture_off")
                    {
                        double fdiff = 0, fenergy = 0;
                        for (int p = 0; p < b.PixelCount; ++p)
                        {
                            var x = ownTarget[v].Pixel(p); var y = offTarget[v].Pixel(p);
                            fdiff += Sq(x.r - y.r) + Sq(x.g - y.g) + Sq(x.b - y.b);
                            fenergy += x.r * x.r + x.g * x.g + x.b * x.b;
                        }
                        sumFixture += Math.Sqrt(fdiff / (fenergy + 1e-12));
                        int fa = 0;
                        for (int p = 0; p < b.PixelCount; ++p)
                        {
                            var x = ownTarget[v].Pixel(p); var y = offTarget[v].Pixel(p);
                            if (Math.Sqrt(Sq(x.r - y.r) + Sq(x.g - y.g) + Sq(x.b - y.b)) > cutoff) ++fa;
                        }
                        sumFixtureAffected += fa / (double)b.PixelCount;
                    }
                    double sdiff = 0;
                    for (int p = 0; p < ownTarget[v].PixelCount; ++p)
                    {
                        var x = ownTarget[v].Pixel(p); var y = b.Pixel(p);
                        sdiff += Sq(x.r - y.r) + Sq(x.g - y.g) + Sq(x.b - y.b);
                    }
                    sumSelf += Math.Sqrt(sdiff / (energy + 1e-12));
                }
                stats.shadowRms = sumEnergy / views;
                stats.affected = sumAffected / views;
                stats.fixtureRms = condition == "fixture_off" ? sumFixture / views : 0;
                stats.fixtureAffected = condition == "fixture_off" ? sumFixtureAffected / views : 0;
                stats.selfShadowRms = sumSelf / views;
                stats.fixtureOverlap = sumOverlap / views;

                // Rest-state compare with the start pose: the shadow signal the optimizer actually sees.
                int k = lamp * LampDofs;
                var startSub = new double[LampDofs]; Array.Copy(start, k, startSub, 0, LampDofs);
                var columns = new float[LampDofs][];
                var extents = scene.Extents;
                for (int axis = 0; axis < LampDofs; ++axis)
                {
                    bool logAxis = axis == 3;
                    double physical = logAxis ? .05 : .01;
                    double hNorm = logAxis ? physical / LogPowerSpan : physical / AxisExtent(extents, axis);
                    columns[axis] = Column(lamp, startSub, axis, hNorm, out double used, out bool ok);
                    stats.stepCm[axis] = (float)(used * 100);
                    if (!ok) continue;
                    double sq = 0; foreach (float x in columns[axis]) sq += (double)x * x;
                    stats.columnNorm[axis] = Math.Sqrt(sq);
                }
                var gram = new double[LampDofs, LampDofs];
                for (int i = 0; i < LampDofs; ++i)
                for (int j = 0; j < LampDofs; ++j)
                {
                    if (columns[i] == null || columns[j] == null) { gram[i, j] = 0; continue; }
                    double dot = 0;
                    for (int q = 0; q < columns[i].Length; ++q) dot += (double)columns[i][q] * columns[j][q];
                    gram[i, j] = dot / Math.Max(1e-30, stats.columnNorm[i] * stats.columnNorm[j]);
                }
                var eigen = Eigenvalues(gram);
                double max = eigen.Max(), min = eigen.Min();
                stats.conditionNumber = min > 1e-9 ? Math.Sqrt(max / min) * (stats.columnNorm.Max() / Math.Max(1e-30, stats.columnNorm.Min())) : double.PositiveInfinity;
                stats.meanColumn = stats.columnNorm.Average();

                // Residual alignment at the start pose against this condition's target.
                var state = (double[])scene.Truth.Clone(); Array.Copy(startSub, 0, state, k, LampDofs);
                scene.Apply(state);
                var rendered = Enumerable.Range(0, 3).Select(fit.Capture).ToArray();
                int channels = E18FitW * E18FitH * 3; var residual = new float[channels * 3];
                double rsq = 0;
                for (int v = 0; v < 3; ++v)
                {
                    var f = rendered[v]; var t = ownTarget[v];
                    for (int p = 0, c = v * channels; p < f.PixelCount; ++p)
                    {
                        var x = f.Pixel(p); var y = t.Pixel(p);
                        double d0 = x.r - y.r, d1 = x.g - y.g, d2 = x.b - y.b;
                        residual[c++] = (float)d0; residual[c++] = (float)d1; residual[c++] = (float)d2;
                        rsq += d0 * d0 + d1 * d1 + d2 * d2;
                    }
                }
                double rnorm = Math.Sqrt(rsq), align = 0; int count = 0;
                for (int axis = 0; axis < LampDofs; ++axis)
                {
                    if (columns[axis] == null) continue;
                    double dot = 0, csq = 0;
                    for (int q = 0; q < residual.Length; ++q) { dot += (double)residual[q] * columns[axis][q]; csq += (double)columns[axis][q] * columns[axis][q]; }
                    if (csq > 0 && rnorm > 0) align += Math.Abs(dot) / (Math.Sqrt(csq) * rnorm);
                    ++count;
                }
                stats.residualAlignment = count > 0 ? align / count : 0;
                own = columns;
                if (reference != null)
                {
                    for (int axis = 0; axis < LampDofs; ++axis)
                    {
                        if (columns[axis] == null || reference[axis] == null) continue;
                        double dot = 0, a2 = 0, b2 = 0;
                        for (int q = 0; q < columns[axis].Length; ++q)
                        {
                            dot += (double)columns[axis][q] * reference[axis][q];
                            a2 += (double)columns[axis][q] * columns[axis][q];
                            b2 += (double)reference[axis][q] * reference[axis][q];
                        }
                        stats.stepConsistency[axis] = a2 > 0 && b2 > 0 ? dot / Math.Sqrt(a2 * b2) : 0;
                    }
                }
                return stats;
            }

            private float[] Column(int lamp, double[] sub, int axis, double hNorm, out double usedPhysical, out bool ok)
            {
                int k = lamp * LampDofs;
                var full = (double[])scene.Truth.Clone();
                var plus = (double[])sub.Clone(); var minus = (double[])sub.Clone();
                var extents = scene.Extents;
                usedPhysical = axis == 3 ? hNorm * LogPowerSpan : hNorm * AxisExtent(extents, axis);
                for (int shrink = 0; shrink < 6; ++shrink)
                {
                    plus[axis] = sub[axis] + hNorm; minus[axis] = sub[axis] - hNorm;
                    Array.Copy(plus, 0, full, k, LampDofs); bool plusOk = scene.Contains(full);
                    Array.Copy(minus, 0, full, k, LampDofs); bool minusOk = scene.Contains(full);
                    if (plusOk && minusOk)
                    {
                        Array.Copy(plus, 0, full, k, LampDofs); scene.Apply(full); var pf = Enumerable.Range(0, 3).Select(fit.Capture).ToArray();
                        Array.Copy(minus, 0, full, k, LampDofs); scene.Apply(full); var mf = Enumerable.Range(0, 3).Select(fit.Capture).ToArray();
                        int channels = E18FitW * E18FitH * 3;
                        var column = new float[channels * 3];
                        float scale = (float)(.5 / usedPhysical);
                        for (int v = 0; v < 3; ++v)
                        {
                            var p = pf[v]; var m = mf[v];
                            for (int i = 0, c = v * channels; i < p.PixelCount; ++i)
                            {
                                var a = p.Pixel(i); var b = m.Pixel(i);
                                column[c++] = (a.r - b.r) * scale; column[c++] = (a.g - b.g) * scale; column[c++] = (a.b - b.b) * scale;
                            }
                        }
                        ok = true; return column;
                    }
                    hNorm *= .5;
                    usedPhysical = axis == 3 ? hNorm * LogPowerSpan : hNorm * AxisExtent(extents, axis);
                }
                ok = false; return null;
            }

            private static double AxisExtent(Vector3 e, int axis) => axis == 0 ? e.x : axis == 1 ? e.y : e.z;
            private static double Sq(double x) => x * x;

            private void MapProbe()
            {
                var conditions = scene.SupportsFixtureShadows ? new[] { "full", "fixture_off", "light_shadow_off" } : new[] { "full", "light_shadow_off" };
                using var csv = new StreamWriter(Path.Combine(dir, "shadow-metrics.csv"));
                csv.WriteLine("lamp,condition,shadowRms,affected,fixtureRms,fixtureAffected,selfShadowRms,fixtureOverlap,meanColumn,conditionNumber,residualAlignment,consistency");
                foreach (int lamp in LampsToRun())
                {
                    var start = LampStart(lamp, 30, 0, out float used);
                    float[][] reference = null;
                    foreach (string condition in conditions)
                    {
                        var stats = ConditionStats(lamp, condition, start, reference, out var own);
                        if (condition == "full") reference = own;
                        csv.WriteLine(string.Join(",", (lamp + 1).ToString(Invariant), condition, F(stats.shadowRms), F(stats.affected), F(stats.fixtureRms),
                            F(stats.fixtureAffected), F(stats.selfShadowRms), F(stats.fixtureOverlap), F(stats.meanColumn), Finite(stats.conditionNumber),
                            F(stats.residualAlignment), F(stats.stepConsistency.Average(v => Math.Abs(v)))));
                        Debug.Log($"[E18] {scene.Name} L{lamp + 1} {condition}: shadow {stats.shadowRms:G4}, selfShadow {stats.selfShadowRms:G4}, affected {stats.affected:P2}, fixture {stats.fixtureRms:G4}, cond {stats.conditionNumber:G3}, align {stats.residualAlignment:F3}");
                    }
                }
            }

            private IEnumerable<int> LampsToRun() => E18Lamps.Where(l => l < scene.Count);

            private void Ablation()
            {
                var conditions = scene.SupportsFixtureShadows ? new[] { "full", "fixture_off", "light_shadow_off" } : new[] { "full", "light_shadow_off" };
                foreach (int lamp in LampsToRun())
                {
                    var start = LampStart(lamp, 30, 0, out float used);
                    var row = new E18Ablation { lamp = lamp + 1, startPositionCm = used, startIntensityError = RelativeIntensity(lamp, start) };
                    var stats = new List<E18Condition>();
                    float[][] reference = null;
                    foreach (string condition in conditions)
                    {
                        var s = ConditionStats(lamp, condition, start, reference, out var own);
                        if (condition == "full") reference = own;
                        stats.Add(s);
                        ArmDirect("direct_lm", lamp, condition, start, budget, "ablation");
                    }
                    row.conditions = stats.ToArray();
                    ablations.Add(row);
                }
                Debug.Log($"[E18] {scene.Name}: ablation finished ({ablations.Count} lamps x {conditions.Length} conditions)");
            }

            private double RelativeIntensity(int lamp, double[] theta) =>
                Math.Abs(scene.PowerOf(theta, lamp) - scene.PowerOf(scene.Truth, lamp)) / scene.PowerOf(scene.Truth, lamp);

            private Rect ProjectedRect(Bounds bounds, int view)
            {
                if (bounds.size == Vector3.zero) return new Rect(0, 0, 0, 0);
                var go = new GameObject("E18 projection probe") { hideFlags = HideFlags.HideAndDontSave };
                try
                {
                    var cam = go.AddComponent<Camera>();
                    cam.CopyFrom(scene.Camera); cam.enabled = false;
                    cam.transform.SetPositionAndRotation(scene.Views[view], scene.Rotations[view]);
                    cam.aspect = (float)E18FitW / E18FitH; cam.ResetProjectionMatrix();
                    float minX = float.MaxValue, minY = float.MaxValue, maxX = -float.MaxValue, maxY = -float.MaxValue;
                    for (int i = 0; i < 8; ++i)
                    {
                        var corner = new Vector3((i & 1) == 0 ? bounds.min.x : bounds.max.x, (i & 2) == 0 ? bounds.min.y : bounds.max.y, (i & 4) == 0 ? bounds.min.z : bounds.max.z);
                        var vp = cam.WorldToViewportPoint(corner);
                        if (vp.z <= 0) continue;
                        minX = Mathf.Min(minX, vp.x); minY = Mathf.Min(minY, vp.y); maxX = Mathf.Max(maxX, vp.x); maxY = Mathf.Max(maxY, vp.y);
                    }
                    if (minX > maxX) return new Rect(0, 0, 0, 0);
                    return Rect.MinMaxRect(minX, minY, maxX, maxY);
                }
                finally { UnityEngine.Object.DestroyImmediate(go); }
            }

            private static double[] Eigenvalues(double[,] source)
            {
                int n = source.GetLength(0);
                var m = (double[,])source.Clone();
                for (int sweep = 0; sweep < 64; ++sweep)
                {
                    double off = 0;
                    for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) off += m[i, j] * m[i, j];
                    if (off < 1e-24) break;
                    for (int p = 0; p < n; ++p)
                    for (int q = p + 1; q < n; ++q)
                    {
                        if (Math.Abs(m[p, q]) < 1e-18) continue;
                        double theta = (m[q, q] - m[p, p]) / (2 * m[p, q]);
                        double t = Math.Sign(theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1));
                        if (theta == 0) t = 1;
                        double c = 1 / Math.Sqrt(t * t + 1), s = t * c;
                        for (int k = 0; k < n; ++k)
                        {
                            double a = m[k, p], b = m[k, q];
                            m[k, p] = c * a - s * b; m[k, q] = s * a + c * b;
                        }
                        for (int k = 0; k < n; ++k)
                        {
                            double a = m[p, k], b = m[q, k];
                            m[p, k] = c * a - s * b; m[q, k] = s * a + c * b;
                        }
                    }
                }
                var d = new double[n];
                for (int i = 0; i < n; ++i) d[i] = Math.Max(0, m[i, i]);
                return d;
            }

            private void Resolution()
            {
                var jobs = new[] { (lamp: 0, condition: "light_shadow_off"), (lamp: 5, condition: "light_shadow_off"), (lamp: 4, condition: "full") };
                foreach (var job in jobs)
                {
                    if (job.lamp >= scene.Count) continue;
                    var start = LampStart(job.lamp, 30, 0, out _);
                    var lowTarget = Targets(job.lamp, job.condition);
                    var highTarget = Targets(job.lamp, job.condition, true);
                    var lowGt = RgbImage.From(lowTarget[0], E18FitW, E18FitH);
                    var highGt = RgbImage.From(highTarget[0], E18HighW, E18HighH);
                    double targetMismatch = RgbImage.Nrmse(highGt.Box(2), lowGt);
                    var fullLow = RgbImage.From(Targets(job.lamp, "full")[0], E18FitW, E18FitH);
                    var offLow = RgbImage.From(Targets(job.lamp, "light_shadow_off")[0], E18FitW, E18FitH);
                    var fullHigh = RgbImage.From(Targets(job.lamp, "full", true)[0], E18HighW, E18HighH);
                    var offHigh = RgbImage.From(Targets(job.lamp, "light_shadow_off", true)[0], E18HighW, E18HighH);
                    double signal448 = RgbImage.Nrmse(fullLow, offLow), signal896 = RgbImage.Nrmse(fullHigh, offHigh);
                    Debug.Log($"[E18] {scene.Name} L{job.lamp + 1} {job.condition}: 448/896 target mismatch {targetMismatch:G4}, shadow signal {signal448:G4}/{signal896:G4}");

                    var arm448 = ArmDirect("resolution_448", job.lamp, job.condition, start, 400, "resolution");
                    ResolutionRow(job.lamp, job.condition, "start", start, targetMismatch, signal448, signal896);
                    ResolutionRow(job.lamp, job.condition, "best448", WithSub(start, job.lamp, arm448.best), targetMismatch, signal448, signal896);
                    var arm896 = ArmDirect("resolution_896", job.lamp, job.condition, start, 300, "resolution", true);
                    ResolutionRow(job.lamp, job.condition, "best896", WithSub(start, job.lamp, arm896.best), targetMismatch, signal448, signal896);
                }
            }

            private static double[] WithSub(double[] theta, int lamp, double[] sub)
            {
                var copy = (double[])theta.Clone(); Array.Copy(sub, 0, copy, lamp * LampDofs, LampDofs); return copy;
            }

            private void ResolutionRow(int lamp, string condition, string parameters, double[] theta, double targetMismatch, double signal448, double signal896)
            {
                var lowTarget = Targets(lamp, condition);
                var highTarget = Targets(lamp, condition, true);
                SetCondition(lamp, condition);
                scene.Apply(theta);
                var low = Enumerable.Range(0, 3).Select(fit.Capture).ToArray();
                var highFrames = Enumerable.Range(0, 3).Select(high.Capture).ToArray();
                var lowImage = RgbImage.From(low[0], E18FitW, E18FitH);
                var highImage = RgbImage.From(highFrames[0], E18HighW, E18HighH);
                var lowTargetImage = RgbImage.From(lowTarget[0], E18FitW, E18FitH);
                var highTargetImage = RgbImage.From(highTarget[0], E18HighW, E18HighH);
                var v448 = new double[3]; var v896 = new double[3]; var vXTarget = new double[3]; var vRender = new double[3];
                for (int v = 0; v < 3; ++v)
                {
                    var lv = RgbImage.From(low[v], E18FitW, E18FitH);
                    var hv = RgbImage.From(highFrames[v], E18HighW, E18HighH);
                    var lt = RgbImage.From(lowTarget[v], E18FitW, E18FitH);
                    var ht = RgbImage.From(highTarget[v], E18HighW, E18HighH);
                    v448[v] = RgbImage.Nrmse(lv, lt);
                    v896[v] = RgbImage.Nrmse(hv, ht);
                    vXTarget[v] = RgbImage.Nrmse(ht.Box(2), lt);
                    vRender[v] = RgbImage.Nrmse(hv.Box(2), lv);
                }
                resolutions.Add(new E18Resolution
                {
                    lamp = lamp + 1, condition = condition, parameters = parameters,
                    nrmse448 = RgbImage.Nrmse(lowImage, lowTargetImage),
                    nrmse896 = RgbImage.Nrmse(highImage, highTargetImage),
                    nrmse896Downsampled = RgbImage.Nrmse(highImage.Box(2), lowTargetImage),
                    renderMismatch = RgbImage.Nrmse(highImage.Box(2), lowImage),
                    targetMismatch = targetMismatch, shadowSignal448 = signal448, shadowSignal896 = signal896,
                    perView448 = string.Join(";", v448.Select(x => x.ToString(Invariant))),
                    perView896 = string.Join(";", v896.Select(x => x.ToString(Invariant))),
                    perViewXTarget = string.Join(";", vXTarget.Select(x => x.ToString(Invariant))),
                    perViewRender = string.Join(";", vRender.Select(x => x.ToString(Invariant)))
                });
                Debug.Log("[E18] resolution row L" + (lamp + 1) + " " + condition + " " + parameters + ": 448 " + string.Join("/", v448.Select(x => x.ToString("G4"))) + " | 896 " + string.Join("/", v896.Select(x => x.ToString("G4"))) + " | xTarget " + string.Join("/", vXTarget.Select(x => x.ToString("G4"))) + " | render " + string.Join("/", vRender.Select(x => x.ToString("G4"))));
            }

            private void MaybeCheckpoint(double[] candidate, double loss)
            {
                if (activeCheckpoints == null) return;
                double fraction = queryCount / (double)Math.Max(1, thresholdQueries);
                while (nextFraction <= 1.0001 && fraction >= nextFraction)
                {
                    var cp = new E18Checkpoint
                    {
                        fraction = nextFraction, query = queryCount, loss = loss, nrmse = Math.Sqrt(Math.Max(0, loss)),
                        condition = lastCondition, consistency = lastConsistency,
                        x = (double[])candidate.Clone(), columnNorm = (double[])lastColumnNorm.Clone(), stepCm = (double[])lastSteps.Clone()
                    };
                    activeCheckpoints.Add(cp);
                    checkpointCsv.WriteLine(string.Join(",", lastMethod, (lastLamp + 1).ToString(Invariant), lastConditionName, F(cp.fraction), cp.query.ToString(Invariant),
                        F(cp.loss), F(cp.nrmse), Finite(cp.condition), F(cp.consistency), F(cp.x[0]), F(cp.x[1]), F(cp.x[2]), F(cp.x[3]),
                        string.Join(";", cp.stepCm.Select(v => F(v))), string.Join(";", cp.columnNorm.Select(v => F(v)))));
                    nextFraction += .25;
                }
            }

            private string lastMethod = "", lastConditionName = "";
            private int lastLamp;

            private void BeginArm(string method, int lamp, string condition, int totalBudget)
            {
                lastMethod = method; lastLamp = lamp; lastConditionName = condition;
                queryCount = 0; renderCount = 0; thresholdQueries = totalBudget; nextFraction = .25;
                e18HasPreviousColumn = false; lastConsistency = 1; lastCondition = 0;
                lastSteps = new double[LampDofs]; lastColumnNorm = new double[LampDofs];
                activeCheckpoints = new List<E18Checkpoint>();
            }

            private ImageJacobianLm.Diagnostics Diagnostics(bool trace)
            {
                return new ImageJacobianLm.Diagnostics
                {
                    Jacobian = (batch, query, loss, effective, stencil, normal) =>
                    {
                        if (effective != null) lastSteps = (double[])effective.Clone();
                        lastCondition = ConditionOf(normal);
                        if (trace) traceCsv.WriteLine(string.Join(",", lastMethod, (lastLamp + 1).ToString(Invariant), lastConditionName,
                            batch.ToString(Invariant), query.ToString(Invariant), "", "", F(loss), "", "jacobian"));
                    },
                    Columns = (batch, query, columns) =>
                    {
                        int size = columns[0].Length;
                        int total = LampDofs * size;
                        if (e18ColumnA == null || e18ColumnA.Length != total)
                        {
                            e18ColumnA = new float[total]; e18ColumnB = new float[total]; e18HasPreviousColumn = false;
                        }
                        for (int axis = 0; axis < LampDofs; ++axis)
                        {
                            Array.Copy(columns[axis], 0, e18ColumnA, axis * size, size);
                            double sq = 0; foreach (float v in columns[axis]) sq += (double)v * v;
                            lastColumnNorm[axis] = Math.Sqrt(sq);
                        }
                        if (e18HasPreviousColumn)
                        {
                            double sum = 0; int n = 0;
                            for (int axis = 0; axis < LampDofs; ++axis)
                            {
                                int offset = axis * size;
                                double dot = 0, a2 = 0, b2 = 0;
                                for (int i = 0; i < size; ++i)
                                {
                                    dot += (double)e18ColumnA[offset + i] * e18ColumnB[offset + i];
                                    a2 += (double)e18ColumnA[offset + i] * e18ColumnA[offset + i];
                                    b2 += (double)e18ColumnB[offset + i] * e18ColumnB[offset + i];
                                }
                                if (a2 > 0 && b2 > 0) { sum += dot / Math.Sqrt(a2 * b2); ++n; }
                            }
                            if (n > 0) lastConsistency = sum / n;
                        }
                        var swap = e18ColumnB; e18ColumnB = e18ColumnA; e18ColumnA = swap; e18HasPreviousColumn = true;
                    },
                    Trial = (batch, query, mu, predicted, actual, ratio, boundHits, shrinks, status) =>
                    {
                        if (trace) traceCsv.WriteLine(string.Join(",", lastMethod, (lastLamp + 1).ToString(Invariant), lastConditionName,
                            batch.ToString(Invariant), query.ToString(Invariant), F(mu), Finite(predicted), Finite(actual), Finite(ratio), status));
                    }
                };
            }

            private static double ConditionOf(double[,] normal)
            {
                int n = normal.GetLength(0);
                var gram = new double[n, n]; var diag = new double[n];
                for (int i = 0; i < n; ++i) diag[i] = Math.Sqrt(Math.Max(0, normal[i, i]));
                for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    gram[i, j] = diag[i] > 0 && diag[j] > 0 ? normal[i, j] / (diag[i] * diag[j]) : 0;
                var eigen = Eigenvalues(gram);
                double max = eigen.Max(), min = eigen.Min();
                double ratio = diag.Max() / Math.Max(1e-300, diag.Min());
                return min > 1e-10 ? Math.Sqrt(max / min) * ratio : double.PositiveInfinity;
            }


            // ---------------- E18 shared helpers ----------------
            private Frame[] CaptureViews(NativeRenderBackend backend, int count) =>
                Enumerable.Range(0, count).Select(backend.Capture).ToArray();

            private static double FrameNrmse(Frame[] rendered, Frame[] target)
            {
                if (rendered.Length != target.Length) throw new ArgumentException("Frame counts differ.");
                double error = 0, energy = 0;
                for (int v = 0; v < rendered.Length; ++v)
                {
                    if (rendered[v].PixelCount != target[v].PixelCount) throw new ArgumentException("Frame sizes differ.");
                    for (int i = 0; i < rendered[v].PixelCount; ++i)
                    {
                        var a = rendered[v].Pixel(i); var b = target[v].Pixel(i);
                        double r = a.r - b.r, g = a.g - b.g, blue = a.b - b.b;
                        error += r * r + g * g + blue * blue;
                        energy += b.r * b.r + b.g * b.g + b.b * b.b;
                    }
                }
                return Math.Sqrt(error / (energy + 1e-12) / 3);
            }

            private static void FillImage(Frame frame, float[] data)
            {
                for (int i = 0, k = 0; i < frame.PixelCount; ++i)
                {
                    var c = frame.Pixel(i); data[k++] = c.r; data[k++] = c.g; data[k++] = c.b;
                }
            }

            private static double ArrayEnergy(float[] data)
            {
                double e = 0; for (int i = 0; i < data.Length; ++i) e += (double)data[i] * data[i]; return e;
            }

            // Separable running-sum box blur, O(N) in the radius.
            private static void BoxBlur(float[] src, float[] tmp, float[] dst, int w, int h, int radius)
            {
                if (radius <= 0) { if (!ReferenceEquals(src, dst)) Array.Copy(src, dst, src.Length); return; }
                int window = radius * 2 + 1;
                for (int y = 0; y < h; ++y)
                for (int c = 0; c < 3; ++c)
                {
                    int row = y * w * 3 + c;
                    double sum = 0;
                    for (int x = -radius; x <= radius; ++x) sum += src[row + Mathf.Clamp(x, 0, w - 1) * 3];
                    for (int x = 0; x < w; ++x)
                    {
                        tmp[row + x * 3] = (float)(sum / window);
                        int add = Mathf.Clamp(x + radius + 1, 0, w - 1), sub = Mathf.Clamp(x - radius, 0, w - 1);
                        sum += src[row + add * 3] - src[row + sub * 3];
                    }
                }
                for (int x = 0; x < w; ++x)
                for (int c = 0; c < 3; ++c)
                {
                    double sum = 0;
                    for (int y = -radius; y <= radius; ++y) sum += tmp[(Mathf.Clamp(y, 0, h - 1) * w + x) * 3 + c];
                    for (int y = 0; y < h; ++y)
                    {
                        dst[(y * w + x) * 3 + c] = (float)(sum / window);
                        int add = Mathf.Clamp(y + radius + 1, 0, h - 1), sub = Mathf.Clamp(y - radius, 0, h - 1);
                        sum += tmp[(add * w + x) * 3 + c] - tmp[(sub * w + x) * 3 + c];
                    }
                }
            }

            private double[] TryLampStart(int lamp, float distanceCm, int direction, out float usedCm)
            {
                try { return LampStart(lamp, distanceCm, direction, out usedCm); }
                catch (InvalidOperationException) { usedCm = 0; return null; }
            }

            private E18Arm FinishArm(string method, int lamp, string condition, double[] initial, double[] best,
                double initialLoss, double bestLoss, int queries, int renders, int jacobians, int acceptedSteps, int rejectedSteps,
                string stop, double t0, string[] stages, double[] stageLoss)
            {
                int k = lamp * LampDofs;
                var full = (double[])scene.Truth.Clone();
                Array.Copy(best, 0, full, k, LampDofs);

                var target = Targets(lamp, condition);
                SetCondition(lamp, condition);
                scene.Apply(full);
                var finalFrames = CaptureViews(fit, 3);
                double globalNrmse = FrameNrmse(finalFrames, target.Take(3).ToArray());

                int viewSize = E18FitW * E18FitH * 3;
                var recheckBuffer = new float[viewSize * 3];
                var recheck = CaptureViews(fit, 3);
                double repeatDelta = FrameNrmse(finalFrames, recheck);
                double recheckLoss = 0;
                var viewLoss = new double[3];
                for (int v = 0; v < 3; ++v) { viewLoss[v] = fit.Residual(v, target[v], recheckBuffer, v * viewSize, 3); recheckLoss += viewLoss[v]; }
                var viewNrmse = Enumerable.Range(0, 3).Select(v => Math.Sqrt(Math.Max(0, viewLoss[v]))).ToArray();
                double finalNrmse = Math.Sqrt(Math.Max(0, recheckLoss / 3));
                Debug.Log("[E18] recheck L" + (lamp + 1) + " " + method + " " + condition + ": bestLoss " + bestLoss.ToString("G6") + ", recheckLoss " + recheckLoss.ToString("G6") + ", meanViewNrmse " + finalNrmse.ToString("G6") + ", globalNrmse " + globalNrmse.ToString("G6") + ", repeatDelta " + repeatDelta.ToString("G3")
                    + ", viewNrmse " + string.Join("/", viewNrmse.Select(x => x.ToString("G4")))
                    + ", targetEnergy " + string.Join("/", Enumerable.Range(0, 3).Select(v => target[v].Energy.ToString("G5"))) + ", renderEnergy " + string.Join("/", Enumerable.Range(0, 3).Select(v => recheck[v].Energy.ToString("G5"))));
                var highTarget = Targets(lamp, condition, true);
                scene.Apply(full);
                var highFrames = CaptureViews(high, scene.Views.Length);
                double nrmse896 = FrameNrmse(highFrames, highTarget);
                double held = scene.Views.Length > 3
                    ? FrameNrmse(highFrames.Skip(3).ToArray(), highTarget.Skip(3).ToArray()) : nrmse896;

                SetCondition(lamp, "full");
                scene.Apply(full);
                var fullFrames = CaptureViews(fit, 3);
                var fullTarget = Targets(lamp, "full");
                double nrmseFull = FrameNrmse(fullFrames, fullTarget.Take(3).ToArray());

                var bestPos = scene.PositionOf(full, lamp); var truthPos = scene.PositionOf(scene.Truth, lamp);
                double posCm = (bestPos - truthPos).magnitude * 100;
                double truthPower = scene.PowerOf(scene.Truth, lamp);
                double relI = Math.Abs(scene.PowerOf(full, lamp) - truthPower) / truthPower;
                var arm = new E18Arm
                {
                    method = method, condition = condition, lamp = lamp + 1, budget = thresholdQueries,
                    queries = queries, renders = renders, jacobians = jacobians, accepted = acceptedSteps, rejected = rejectedSteps,
                    initialNrmse448 = Math.Sqrt(Math.Max(0, initialLoss) / 3), finalNrmse448 = finalNrmse,
                    bestNrmse448 = Math.Sqrt(Math.Max(0, bestLoss) / 3), globalNrmse448 = globalNrmse,
                    viewNrmse448 = string.Join(";", viewNrmse.Select(v => v.ToString(Invariant))),
                    nrmseFull448 = nrmseFull, nrmse896 = nrmse896, heldNrmse896 = held,
                    positionErrorCm = posCm, relativeIntensityError = relI, stop = stop,
                    wallSeconds = watch.Elapsed.TotalSeconds - t0, passed = posCm <= 5 && relI <= .05,
                    stages = stages, stageLoss = stageLoss, checkpoints = activeCheckpoints.ToArray(),
                    start = (double[])initial.Clone(), best = (double[])best.Clone()
                };
                armCsv.WriteLine(string.Join(",", arm.method, arm.lamp.ToString(Invariant), arm.condition, condition,
                    arm.budget.ToString(Invariant), arm.queries.ToString(Invariant), arm.renders.ToString(Invariant), arm.jacobians.ToString(Invariant),
                    arm.accepted.ToString(Invariant), arm.rejected.ToString(Invariant), arm.stop, F(arm.initialNrmse448), F(arm.finalNrmse448),
                    F(arm.bestNrmse448), F(arm.globalNrmse448), arm.viewNrmse448, F(arm.nrmseFull448), F(arm.nrmse896), F(arm.heldNrmse896), F(arm.positionErrorCm),
                    F(arm.relativeIntensityError), F(arm.wallSeconds), arm.passed ? "1" : "0"));
                arms.Add(arm);
                Debug.Log($"[E18] {scene.Name} L{lamp + 1} {method} {condition}: NRMSE {arm.initialNrmse448:P1}->{arm.bestNrmse448:P1} (full {arm.nrmseFull448:P2}, held896 {arm.heldNrmse896:P2}), pos {arm.positionErrorCm:F2} cm, I {arm.relativeIntensityError:P1}, {arm.stop}, {arm.queries}q/{arm.renders}r, {arm.wallSeconds:F1}s");
                return arm;
            }

            private E18Arm ArmDirect(string method, int lamp, string condition, double[] start, int armBudget, string phase, bool highRes = false)
            {
                int k = lamp * LampDofs;
                var sub = new double[LampDofs]; Array.Copy(start, k, sub, 0, LampDofs);
                var initial = (double[])sub.Clone();
                var backend = highRes ? high : fit;
                int w = highRes ? E18HighW : E18FitW, h = highRes ? E18HighH : E18FitH;
                int viewSize = w * h * 3;
                int residualSize = viewSize * 3;
                var target = Targets(lamp, condition, highRes);
                SetCondition(lamp, condition);
                BeginArm(method, lamp, condition, armBudget);
                var full = (double[])scene.Truth.Clone();
                var buffer = new float[residualSize];
                double[] lastCandidate = null, accepted = (double[])initial.Clone();
                double lastLoss = double.PositiveInfinity, acceptedLoss = double.PositiveInfinity;
                double t0 = watch.Elapsed.TotalSeconds;
                var diag = Diagnostics(!highRes);
                var trialLog = diag.Trial;
                diag.Trial = (batch, query, mu, predicted, actual, ratio, boundHits, shrinks, status) =>
                {
                    trialLog?.Invoke(batch, query, mu, predicted, actual, ratio, boundHits, shrinks, status);
                    if (status == "accept" && lastCandidate != null) { accepted = (double[])lastCandidate.Clone(); acceptedLoss = lastLoss; }
                };
                if (highRes) diag.Columns = null;

                double Evaluate(double[] candidate, string ph, float[] res)
                {
                    Array.Copy(candidate, 0, full, k, LampDofs);
                    if (!scene.Contains(full)) throw new InvalidOperationException("Optimizer requested an illegal rig.");
                    scene.Apply(full);
                    double loss = 0;
                    for (int v = 0; v < 3; ++v) loss += backend.Residual(v, target[v], res, v * viewSize, 3);
                    ++queryCount; ++renderCount;
                    lastCandidate = (double[])candidate.Clone(); lastLoss = loss;
                    var qp = scene.PositionOf(full, lamp);
                    queryCsv.WriteLine(string.Join(",", method, (lamp + 1).ToString(Invariant), condition,
                        queryCount.ToString(Invariant), ph, F(qp.x), F(qp.y), F(qp.z), F(scene.PowerOf(full, lamp)), F(loss)));
                    MaybeCheckpoint(accepted, acceptedLoss);
                    return loss;
                }
                bool Feasible(double[] candidate)
                {
                    for (int i = 0; i < LampDofs; ++i) if (!(candidate[i] >= 0 && candidate[i] <= 1)) return false;
                    Array.Copy(candidate, 0, full, k, LampDofs);
                    return scene.Contains(full);
                }
                double initialLoss = Evaluate(initial, "initial", buffer);
                accepted = (double[])initial.Clone(); acceptedLoss = initialLoss;
                var result = ImageJacobianLm.Run(sub, residualSize, Evaluate, armBudget, seed, Feasible, false, diag);
                return FinishArm(method, lamp, condition, initial, (double[])result.Best.Clone(), result.InitialLoss, result.BestLoss,
                    result.Queries, result.Queries * 3, result.Jacobians, result.Accepted, result.Rejected, result.StopReason, t0,
                    new[] { "direct" }, new[] { Math.Sqrt(result.BestLoss / 3) });
            }

            private E18Arm ArmMultiScale(int lamp, double[] start, int armBudget, string phase)
            {
                int k = lamp * LampDofs;
                var sub = new double[LampDofs]; Array.Copy(start, k, sub, 0, LampDofs);
                var initial = (double[])sub.Clone();
                int w = E18FitW, h = E18FitH, viewSize = w * h * 3, residualSize = viewSize * 3;
                var target = Targets(lamp, "full");
                SetCondition(lamp, "full");
                int stageBudget = Math.Max(60, armBudget / 3);
                BeginArm("multiscale_lm", lamp, "full", stageBudget * 3);
                var full = (double[])scene.Truth.Clone();
                var buffer = new float[residualSize];
                var scratch = new float[viewSize]; var tmp = new float[viewSize]; var blurred = new float[viewSize];
                var blurredTargets = new float[3][]; var energies = new double[3];
                int radius = 0;
                double[] lastCandidate = null, accepted = (double[])initial.Clone();
                double lastLoss = double.PositiveInfinity, acceptedLoss = double.PositiveInfinity;
                double initialLoss = double.NaN, bestLoss = double.NaN;
                double t0 = watch.Elapsed.TotalSeconds;
                var diag = Diagnostics(false);
                var trialLog = diag.Trial;
                diag.Trial = (batch, query, mu, predicted, actual, ratio, boundHits, shrinks, status) =>
                {
                    trialLog?.Invoke(batch, query, mu, predicted, actual, ratio, boundHits, shrinks, status);
                    if (status == "accept" && lastCandidate != null) { accepted = (double[])lastCandidate.Clone(); acceptedLoss = lastLoss; }
                };

                double Evaluate(double[] candidate, string ph, float[] res)
                {
                    Array.Copy(candidate, 0, full, k, LampDofs);
                    if (!scene.Contains(full)) throw new InvalidOperationException("Optimizer requested an illegal rig.");
                    scene.Apply(full);
                    double loss = 0;
                    for (int v = 0; v < 3; ++v)
                    {
                        FillImage(fit.Capture(v), scratch);
                        BoxBlur(scratch, tmp, blurred, w, h, radius);
                        double err = 0, scale = 1 / Math.Sqrt(energies[v] + 1e-12);
                        for (int i = 0; i < viewSize; ++i)
                        {
                            double d = blurred[i] - blurredTargets[v][i];
                            res[v * viewSize + i] = (float)(d * scale); err += d * d;
                        }
                        loss += err / (energies[v] + 1e-12);
                    }
                    ++queryCount; ++renderCount;
                    lastCandidate = (double[])candidate.Clone(); lastLoss = loss;
                    var qp = scene.PositionOf(full, lamp);
                    queryCsv.WriteLine(string.Join(",", "multiscale_lm", (lamp + 1).ToString(Invariant), "full",
                        queryCount.ToString(Invariant), ph + "@b" + radius, F(qp.x), F(qp.y), F(qp.z), F(scene.PowerOf(full, lamp)), F(loss)));
                    MaybeCheckpoint(accepted, acceptedLoss);
                    return loss;
                }
                bool Feasible(double[] candidate)
                {
                    for (int i = 0; i < LampDofs; ++i) if (!(candidate[i] >= 0 && candidate[i] <= 1)) return false;
                    Array.Copy(candidate, 0, full, k, LampDofs);
                    return scene.Contains(full);
                }

                var stages = new List<string>(); var stageLoss = new List<double>();
                int queries = 0, jacobians = 0, acceptedSteps = 0, rejectedSteps = 0; string stop = "budget";
                var current = (double[])initial.Clone();
                foreach (int r in new[] { 4, 2, 0 })
                {
                    radius = r;
                    for (int v = 0; v < 3; ++v)
                    {
                        blurredTargets[v] = new float[viewSize];
                        FillImage(target[v], blurredTargets[v]);
                        BoxBlur(blurredTargets[v], tmp, blurredTargets[v], w, h, r);
                        energies[v] = ArrayEnergy(blurredTargets[v]);
                    }
                    var stage = ImageJacobianLm.Run(current, residualSize, Evaluate, stageBudget, seed + r, Feasible, false, diag);
                    if (double.IsNaN(initialLoss)) initialLoss = stage.InitialLoss;
                    queries += stage.Queries; jacobians += stage.Jacobians; acceptedSteps += stage.Accepted; rejectedSteps += stage.Rejected;
                    stop = stage.StopReason;
                    stages.Add("blur" + r); stageLoss.Add(Math.Sqrt(stage.BestLoss / 3));
                    bestLoss = stage.BestLoss;
                    current = (double[])stage.Best.Clone();
                }
                return FinishArm("multiscale_lm", lamp, "full", initial, current, initialLoss, bestLoss,
                    queries, queries * 3, jacobians, acceptedSteps, rejectedSteps, stop, t0, stages.ToArray(), stageLoss.ToArray());
            }

            private E18Arm ArmSmoothing(int lamp, double[] start, int armBudget, string phase)
            {
                int k = lamp * LampDofs;
                var sub = new double[LampDofs]; Array.Copy(start, k, sub, 0, LampDofs);
                var initial = (double[])sub.Clone();
                int viewSize = E18FitW * E18FitH * 3, residualSize = viewSize * 3;
                var target = Targets(lamp, "full");
                SetCondition(lamp, "full");
                BeginArm("randomized_smoothing", lamp, "full", armBudget);
                var full = (double[])scene.Truth.Clone();
                var buffer = new float[residualSize]; var sample = new float[residualSize];
                var rng = new System.Random(seed);
                double[] lastCandidate = null, accepted = (double[])initial.Clone();
                double lastLoss = double.PositiveInfinity, acceptedLoss = double.PositiveInfinity;
                double t0 = watch.Elapsed.TotalSeconds;
                var diag = Diagnostics(false);
                var trialLog = diag.Trial;
                diag.Trial = (batch, query, mu, predicted, actual, ratio, boundHits, shrinks, status) =>
                {
                    trialLog?.Invoke(batch, query, mu, predicted, actual, ratio, boundHits, shrinks, status);
                    if (status == "accept" && lastCandidate != null) { accepted = (double[])lastCandidate.Clone(); acceptedLoss = lastLoss; }
                };

                double Evaluate(double[] candidate, string ph, float[] res)
                {
                    double fraction = Math.Min(1, queryCount / (double)Math.Max(1, armBudget));
                    double posSigma = .010 * (1 - fraction) + .0005, logSigma = .04 * (1 - fraction) + .002;
                    Array.Clear(res, 0, res.Length);
                    double sampleLoss = 0; int used = 0;
                    for (int s = 0; s < 3; ++s)
                    {
                        var p = (double[])candidate.Clone();
                        for (int axis = 0; axis < LampDofs; ++axis)
                        {
                            double sigma = axis == 3 ? logSigma : posSigma;
                            p[axis] = Math.Min(1, Math.Max(0, p[axis] + (rng.NextDouble() * 2 - 1) * sigma));
                        }
                        Array.Copy(p, 0, full, k, LampDofs);
                        if (!scene.Contains(full)) Array.Copy(candidate, 0, full, k, LampDofs);
                        scene.Apply(full);
                        double l = 0;
                        for (int v = 0; v < 3; ++v) l += fit.Residual(v, target[v], sample, v * viewSize, 3);
                        for (int i = 0; i < res.Length; ++i) res[i] += sample[i];
                        sampleLoss += l; ++used;
                    }
                    for (int i = 0; i < res.Length; ++i) res[i] /= used;
                    double loss = sampleLoss / used;
                    ++queryCount; renderCount += used;
                    lastCandidate = (double[])candidate.Clone(); lastLoss = loss;
                    var qp = scene.PositionOf(full, lamp);
                    queryCsv.WriteLine(string.Join(",", "randomized_smoothing", (lamp + 1).ToString(Invariant), "full",
                        queryCount.ToString(Invariant), ph, F(qp.x), F(qp.y), F(qp.z), F(scene.PowerOf(full, lamp)), F(loss)));
                    MaybeCheckpoint(accepted, acceptedLoss);
                    return loss;
                }
                bool Feasible(double[] candidate)
                {
                    for (int i = 0; i < LampDofs; ++i) if (!(candidate[i] >= 0 && candidate[i] <= 1)) return false;
                    Array.Copy(candidate, 0, full, k, LampDofs);
                    return scene.Contains(full);
                }
                double initialLoss = Evaluate(initial, "initial", buffer);
                accepted = (double[])initial.Clone(); acceptedLoss = initialLoss;
                var result = ImageJacobianLm.Run(sub, residualSize, Evaluate, armBudget, seed, Feasible, false, diag);
                return FinishArm("randomized_smoothing", lamp, "full", initial, (double[])result.Best.Clone(), result.InitialLoss, result.BestLoss,
                    result.Queries, renderCount, result.Jacobians, result.Accepted, result.Rejected, result.StopReason, t0,
                    new[] { "sigma_decay" }, new[] { Math.Sqrt(result.BestLoss / 3) });
            }

            private E18Arm ArmContinuation(int lamp, double[] start, int armBudget, string phase)
            {
                int k = lamp * LampDofs;
                var sub = new double[LampDofs]; Array.Copy(start, k, sub, 0, LampDofs);
                var initial = (double[])sub.Clone();
                int viewSize = E18FitW * E18FitH * 3, residualSize = viewSize * 3;
                var alphas = new[] { 0.0, 0.25, 0.5, 0.75, 1.0 };
                int stageBudget = Math.Max(80, armBudget / alphas.Length);
                BeginArm("shadow_continuation", lamp, "full", stageBudget * alphas.Length);
                var full = (double[])scene.Truth.Clone();
                var buffer = new float[residualSize];
                var stageTarget = new Frame[3];
                double[] lastCandidate = null, accepted = (double[])initial.Clone();
                double lastLoss = double.PositiveInfinity, acceptedLoss = double.PositiveInfinity;
                double initialLoss = double.NaN, bestLoss = double.NaN;
                double t0 = watch.Elapsed.TotalSeconds;
                var diag = Diagnostics(false);
                var trialLog = diag.Trial;
                diag.Trial = (batch, query, mu, predicted, actual, ratio, boundHits, shrinks, status) =>
                {
                    trialLog?.Invoke(batch, query, mu, predicted, actual, ratio, boundHits, shrinks, status);
                    if (status == "accept" && lastCandidate != null) { accepted = (double[])lastCandidate.Clone(); acceptedLoss = lastLoss; }
                };

                double Evaluate(double[] candidate, string ph, float[] res)
                {
                    Array.Copy(candidate, 0, full, k, LampDofs);
                    if (!scene.Contains(full)) throw new InvalidOperationException("Optimizer requested an illegal rig.");
                    scene.Apply(full);
                    double loss = 0;
                    for (int v = 0; v < 3; ++v) loss += fit.Residual(v, stageTarget[v], res, v * viewSize, 3);
                    ++queryCount; ++renderCount;
                    lastCandidate = (double[])candidate.Clone(); lastLoss = loss;
                    var qp = scene.PositionOf(full, lamp);
                    queryCsv.WriteLine(string.Join(",", "shadow_continuation", (lamp + 1).ToString(Invariant), "full",
                        queryCount.ToString(Invariant), ph, F(qp.x), F(qp.y), F(qp.z), F(scene.PowerOf(full, lamp)), F(loss)));
                    MaybeCheckpoint(accepted, acceptedLoss);
                    return loss;
                }
                bool Feasible(double[] candidate)
                {
                    for (int i = 0; i < LampDofs; ++i) if (!(candidate[i] >= 0 && candidate[i] <= 1)) return false;
                    Array.Copy(candidate, 0, full, k, LampDofs);
                    return scene.Contains(full);
                }

                var stages = new List<string>(); var stageLoss = new List<double>();
                int queries = 0, jacobians = 0, acceptedSteps = 0, rejectedSteps = 0; string stop = "budget";
                var current = (double[])initial.Clone();
                for (int s = 0; s < alphas.Length; ++s)
                {
                    ResetFixtures(); scene.ResetShadows();
                    for (int i = 0; i < scene.Count; ++i) scene.SetShadowStrength(i, (float)alphas[s]);
                    scene.Apply(scene.Truth);
                    for (int v = 0; v < 3; ++v) stageTarget[v] = fit.Capture(v);
                    var stage = ImageJacobianLm.Run(current, residualSize, Evaluate, stageBudget, seed + s, Feasible, false, diag);
                    if (double.IsNaN(initialLoss)) initialLoss = stage.InitialLoss;
                    queries += stage.Queries; jacobians += stage.Jacobians; acceptedSteps += stage.Accepted; rejectedSteps += stage.Rejected;
                    stop = stage.StopReason;
                    stages.Add("alpha" + alphas[s].ToString("0.00", Invariant)); stageLoss.Add(Math.Sqrt(stage.BestLoss / 3));
                    bestLoss = stage.BestLoss;
                    current = (double[])stage.Best.Clone();
                }
                ResetFixtures(); scene.ResetShadows();
                return FinishArm("shadow_continuation", lamp, "full", initial, current, initialLoss, bestLoss,
                    queries, queries * 3, jacobians, acceptedSteps, rejectedSteps, stop, t0, stages.ToArray(), stageLoss.ToArray());
            }

            private void Methods(int[] lamps)
            {
                foreach (int lamp in lamps)
                {
                    if (lamp >= scene.Count) continue;
                    var start = TryLampStart(lamp, 30, 0, out _);
                    if (start == null) { Debug.LogWarning($"[E18] no feasible start for L{lamp + 1}; skipping methods."); continue; }
                    ArmDirect("direct_lm", lamp, "full", start, budget, "methods");
                    ArmMultiScale(lamp, start, Math.Max(300, budget), "methods");
                    ArmSmoothing(lamp, start, Math.Min(500, Math.Max(300, budget)), "methods");
                    ArmContinuation(lamp, start, Math.Max(400, budget), "methods");
                }
            }

            private void PathRow(int lamp, float distance, int direction, float usedCm, double[] start, bool shadowsOn, int budget)
            {
                string condition = shadowsOn ? "full" : "light_shadow_off";
                var target = Targets(lamp, condition);
                SetCondition(lamp, condition);
                var truth = scene.Truth;
                int viewSize = E18FitW * E18FitH * 3;
                var buffer = new float[viewSize * 3];
                var t = new double[11]; var loss = new double[11];
                for (int s = 0; s <= 10; ++s)
                {
                    double u = s / 10.0;
                    var theta = new double[truth.Length];
                    for (int i = 0; i < theta.Length; ++i) theta[i] = (1 - u) * start[i] + u * truth[i];
                    t[s] = u;
                    try
                    {
                        scene.Apply(theta);
                        double l = 0;
                        for (int v = 0; v < 3; ++v) l += fit.Residual(v, target[v], buffer, v * viewSize, 3);
                        loss[s] = l;
                    }
                    catch (InvalidOperationException) { loss[s] = -1; }
                }
                var row = new E18Path
                {
                    lamp = lamp + 1, shadows = shadowsOn ? "on" : "off", distanceCm = usedCm, direction = direction,
                    startPositionCm = usedCm, startIntensityError = RelativeIntensity(lamp, start), t = t, loss448 = loss
                };
                if (budget > 0)
                {
                    var arm = ArmDirect("path_lm", lamp, condition, start, budget, "path");
                    row.passed = arm.passed; row.positionErrorCm = arm.positionErrorCm; row.relativeIntensityError = arm.relativeIntensityError;
                }
                paths.Add(row);
                for (int s = 0; s <= 10; ++s)
                    pathCsv.WriteLine(string.Join(",", (lamp + 1).ToString(Invariant), row.shadows, F(row.distanceCm), direction.ToString(Invariant), F(t[s]), Finite(loss[s])));
            }

            private void Paths()
            {
                int pathBudget = int.Parse(Arg("-bbE18PathBudget", "350"), Invariant);
                foreach (int lamp in new[] { 4, 5 }.Where(l => l < scene.Count))
                foreach (float distance in new[] { 10f, 30f, 60f })
                for (int direction = 0; direction < 4; ++direction)
                {
                    var start = TryLampStart(lamp, distance, direction, out float used);
                    if (start == null) continue;
                    PathRow(lamp, distance, direction, used, start, true, pathBudget);
                    PathRow(lamp, distance, direction, used, start, false, pathBudget);
                }
            }

            private void OneArm()
            {
                var start = LampStart(0, 30, 0, out float used);
                Debug.Log($"[E18] one arm start {used} cm");
                ArmDirect("direct_lm", 0, "full", start, 250, "one");
            }

            private void ViewAudit()
            {
                using var csv = new StreamWriter(Path.Combine(dir, "view-audit.csv"));
                csv.WriteLine("lamp,view,energy,moveNrmse,shadowNrmse,globalMoveNrmse,globalShadowNrmse");
                foreach (int lamp in LampsToRun())
                {
                    if (lamp == 6) continue;
                    var target = Targets(lamp, "full", true);
                    var start = LampStart(lamp, 30, 0, out _);
                    SetCondition(lamp, "full"); scene.Apply(scene.Truth);
                    var baseFrames = CaptureViews(high, scene.Views.Length);
                    SetCondition(lamp, "full"); scene.Apply(start);
                    var movedFrames = CaptureViews(high, scene.Views.Length);
                    SetCondition(lamp, "light_shadow_off"); scene.Apply(scene.Truth);
                    var offFrames = CaptureViews(high, scene.Views.Length);
                    double moveGlobal = FrameNrmse(movedFrames, baseFrames);
                    double shadowGlobal = FrameNrmse(offFrames, baseFrames);
                    for (int v = 0; v < scene.Views.Length; ++v)
                    {
                        double move = FrameNrmse(new[] { movedFrames[v] }, new[] { baseFrames[v] });
                        double shadow = FrameNrmse(new[] { offFrames[v] }, new[] { baseFrames[v] });
                        csv.WriteLine(string.Join(",", (lamp + 1).ToString(Invariant), v.ToString(Invariant), F(baseFrames[v].Energy), F(move), F(shadow), F(moveGlobal), F(shadowGlobal)));
                        Debug.Log($"[E18] view audit L{lamp + 1} v{v}: energy {baseFrames[v].Energy:G5}, move30cm {move:P2}, shadow {shadow:P2}");
                    }
                }
            }

            private void Consistency()
            {
                using var csv = new StreamWriter(Path.Combine(dir, "consistency.csv"));
                csv.WriteLine("lamp,case,energy1,energy2,energy3,nrmse12,nrmse23,lossCapture,lossResidual,energyTarget");
                foreach (int lamp in new[] { 0 })
                {
                    var truth = scene.Truth;
                    var probe = scene.WithPosition(truth, lamp, scene.PositionOf(truth, lamp) + new Vector3(.6f, 0f, 0f));
                    int viewSize = E18FitW * E18FitH * 3;
                    var buffer = new float[viewSize];
                    SetCondition(lamp, "full"); scene.Apply(truth);
                    var g1 = fit.Capture(0); var g2 = fit.Capture(0); var g3 = fit.Capture(0);
                    SetCondition(lamp, "full"); scene.Apply(probe);
                    var p1 = fit.Capture(0); var p2 = fit.Capture(0); var p3 = fit.Capture(0);
                    SetCondition(lamp, "full"); scene.Apply(truth);
                    var w1 = fit.Capture(0); var w2 = fit.Capture(0);
                    double lossCapture = fit.Loss(0, g3);
                    double lossResidual = fit.Residual(0, g3, buffer, 0, 1);
                    NativeRenderBackend.Save(g1, E18FitW, E18FitH, Path.Combine(dir, "diag-g1"));
                    NativeRenderBackend.Save(w2, E18FitW, E18FitH, Path.Combine(dir, "diag-w2"));
                    NativeRenderBackend.Save(p1, E18FitW, E18FitH, Path.Combine(dir, "diag-p1"));
                    Debug.Log($"[E18] diag gt energies {g1.Energy:G5}/{g2.Energy:G5}/{g3.Energy:G5} nrmse12 {FrameNrmse(new[]{g1}, new[]{g2}):G4} nrmse23 {FrameNrmse(new[]{g2}, new[]{g3}):G4} | probe {p1.Energy:G5}/{p2.Energy:G5}/{p3.Energy:G5} nrmse {FrameNrmse(new[]{p1}, new[]{p2}):G4}/{FrameNrmse(new[]{p2}, new[]{p3}):G4} | truth re-set {w1.Energy:G5}/{w2.Energy:G5} | lossCapture {lossCapture:G5} lossResidual {lossResidual:G5} targetEnergy {g3.Energy:G5}");
                    csv.WriteLine(string.Join(",", "1", "warmup", F(g1.Energy), F(g2.Energy), F(g3.Energy), F(FrameNrmse(new[]{g1}, new[]{g2})), F(FrameNrmse(new[]{g2}, new[]{g3})), F(lossCapture), F(lossResidual), F(g3.Energy)));
                }
            }
            private void RoomContrast()            {
                int roomBudget = int.Parse(Arg("-bbE18RoomBudget", "350"), Invariant);
                foreach (int lamp in E18Lamps.Where(l => l < scene.Count))
                foreach (float distance in new[] { 10f, 30f, 60f })
                for (int direction = 0; direction < 2; ++direction)
                {
                    var start = TryLampStart(lamp, distance, direction, out _);
                    if (start == null) continue;
                    ArmDirect("room_direct", lamp, "full", start, roomBudget, "room");
                    ArmDirect("room_direct", lamp, "light_shadow_off", start, roomBudget, "room");
                }
                foreach (int lamp in E18Lamps.Where(l => l < scene.Count))
                {
                    var start = TryLampStart(lamp, 60, 0, out float used);
                    if (start == null) continue;
                    PathRow(lamp, 60, 0, used, start, true, 0);
                    PathRow(lamp, 60, 0, used, start, false, 0);
                }
            }

            private static string Finite(double v) => double.IsNaN(v) || double.IsInfinity(v) ? "" : F(v);
        }
    }
}

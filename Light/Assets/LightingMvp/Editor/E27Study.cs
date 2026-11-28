using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using UnityEngine;
using UnityEngine.Rendering;
using Lm = LightingMvp.ImageJacobianLm;
using Mapping = LightingMvp.NativeRenderBackend.HdrMapping;
using Frame = LightingMvp.NativeRenderBackend.Frame;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        [Serializable] private sealed class E27Job { public E27Case[] cases; public E27Arm[] arms; public string source_sha256; }
        [Serializable] private sealed class E27Case
        {
            public string id, scene, mode, target, axes, initial_sha256, truth_sha256;
            public int[] active;
            public double[] initial, truth;
        }
        [Serializable] private sealed class E27Arm
        {
            public string id, stage, @case, method, policy, selected;
            public int seed, cap, step_mm, branch_models, repeat;
            public bool direct, gpu;
        }
        [Serializable] private sealed class E27Result
        {
            public string id, stage, @case, method, policy, stop, status, reason, sourceSha256, targetSha256, config;
            public int seed, cap, cameras, models, auditCameras, acceptanceCameras, firstPassCamera = -1, firstPassModel = -1;
            public double firstPassSeconds = -1, seconds, loss, initialLoss, maxCm, maxRelative, maxColor, heldRaw, heldLog, heldDisplay;
            public bool pass, legal, explored, triggered;
            public double[] initial, final, truth, positionCm, intensityRelative, colorError, chromaL2, chromaAngleDegrees;
        }

        private static void RunE27Stages(string output)
        {
            if (!Application.isBatchMode) throw new InvalidOperationException("E27 is batch only.");
            var job = JsonUtility.FromJson<E27Job>(File.ReadAllText(Arg("-bbE27Job", "")));
            if (job.arms.Length == 0) return;
            using var pipeline = new TestPipeline(classroom: true);
            foreach (var group in job.arms.GroupBy(a => job.cases.FirstOrDefault(c => c.id == a.@case)?.scene ??
                         (a.@case.StartsWith("class-white", StringComparison.Ordinal) ? "classroom" :
                          a.@case.StartsWith("room-", StringComparison.Ordinal) ? "room" : "complex")))
            {
                using ISensitivityScene scene = group.Key == "classroom" ? new ClassroomSensitivityScene(pipeline.Asset, null) :
                    group.Key == "room" ? new RoomSensitivityScene() : new ComplexRoomSensitivityScene();
                var path = group.Key == "classroom" ? ClassroomAssets + "/room.bytes" :
                    group.Key == "room" ? "Assets/Scenes/UrpNativeRoom.unity" : NativeComplexRoomBuilder.ScenePath;
                using var study = new E21RoomStudy(scene, "e27-run", Path.Combine(output, "scene", group.Key), path);
                study.E27Group(group, job, output);
            }
        }

        private sealed partial class E21RoomStudy
        {
            private static readonly int[] E27Views = { 0, 1, 2 };
            private readonly Dictionary<string, E27Target> e27Targets = new Dictionary<string, E27Target>();
            private readonly Dictionary<string, E27Case> e27Derived = new Dictionary<string, E27Case>();
            private sealed class E27Target
            {
                internal Frame[] low, high;
                internal float[] packed;
                internal double scale, repeat;
                internal string hash;
                internal E23FrameObjective log;
            }

            internal void E27Group(IEnumerable<E27Arm> arms, E27Job job, string output)
            {
                e25Inputs = Arg("-bbE25Inputs", "");
                if (room is ClassroomSensitivityScene) E25Read("a_classroom");
                else if (room is ComplexRoomSensitivityScene) E25Read("a_complex");
                var original = lights.Select(l => (l.color, l.useColorTemperature, l.colorTemperature)).ToArray();
                bool oldGlobal = GraphicsSettings.lightsUseColorTemperature;
                try
                {
                    if (QualitySettings.activeColorSpace != ColorSpace.Linear || !GraphicsSettings.lightsUseLinearIntensity)
                        throw new InvalidOperationException("E27 requires linear URP lighting.");
                    GraphicsSettings.lightsUseColorTemperature = true;
                    foreach (var arm in arms)
                    {
                        string dir = Path.Combine(output, "arms", arm.id);
                        Directory.CreateDirectory(dir);
                        if (File.Exists(Path.Combine(dir, "result.json"))) continue;
                        Restore();
                        UnityEngine.Debug.Log("E27 START " + arm.id);
                        E27Case sample;
                        try { sample = job.cases.FirstOrDefault(c => c.id == arm.@case) ?? E27DerivedCase(arm.@case, job); }
                        catch (Exception ex) when (ex is InvalidDataException || ex is ArgumentException)
                        {
                            WriteJson(dir, "result.json", new E27Result { id = arm.id, stage = arm.stage, @case = arm.@case,
                                method = arm.method, policy = arm.policy, seed = arm.seed, cap = arm.cap, status = "failed",
                                reason = "frozen input: " + ex.Message, sourceSha256 = job.source_sha256 });
                            continue;
                        }
                        if (sample == null)
                        {
                            WriteJson(dir, "result.json", new E27Result { id = arm.id, stage = arm.stage, @case = arm.@case, method = arm.method,
                                policy = arm.policy, seed = arm.seed, cap = arm.cap, status = "not_run", reason = "derived input not implemented", sourceSha256 = job.source_sha256 });
                            continue;
                        }
                        if (arm.method == "Off" && sample.target != "Off")
                            sample = new E27Case { id = sample.id, scene = sample.scene, mode = sample.mode, target = "Off", active = sample.active,
                                axes = sample.axes, initial = sample.initial, truth = sample.truth };
                        var trial = new E27Trial(this, arm, sample, dir, job.source_sha256);
                        try { using (trial) trial.Run(); }
                        catch (Exception exception)
                        {
                            UnityEngine.Debug.LogWarning($"E27 FAIL {arm.id}: {exception}");
                            trial.Fail(exception);
                        }
                        UnityEngine.Debug.Log("E27 END " + arm.id);
                    }
                }
                finally { GraphicsSettings.lightsUseColorTemperature = oldGlobal; Restore(); }

                void Restore()
                {
                    for (int i = 0; i < lights.Length; ++i)
                    {
                        lights[i].color = original[i].color; lights[i].useColorTemperature = original[i].useColorTemperature;
                        lights[i].colorTemperature = original[i].colorTemperature;
                    }
                }
            }

            private static LightShadows E27Mode(E27Case c) => c.target == "Off" ? LightShadows.None : LightShadows.Hard;
            private static Vector3 E27Chroma(double u, double v)
            {
                var c = new Vector3((float)Math.Exp(u), 1, (float)Math.Exp(v));
                return c / Vector3.Dot(c, new Vector3(.2126f, .7152f, .0722f));
            }
            private void E27Apply(double[] rig, E27Case sample, LightShadows mode)
            {
                Apply(rig, mode != LightShadows.None, mode);
                int width = room.Count * 4;
                for (int i = 0; i < lights.Length && sample.mode != "original"; ++i)
                {
                    var light = lights[i];
                    if (sample.mode == "white")
                    {
                        light.color = Color.white; light.useColorTemperature = true;
                        light.colorTemperature = (float)Kelvin(rig[width + i]);
                    }
                    else
                    {
                        var c = E27Chroma(6 * rig[width + 2 * i] - 3, 6 * rig[width + 2 * i + 1] - 3);
                        float maximum = Mathf.Max(c.x, Mathf.Max(c.y, c.z));
                        var filter = new Color(c.x / maximum, c.y / maximum, c.z / maximum, 1);
                        light.useColorTemperature = false;
                        // VisibleLight converts Light.color to linear when linear intensity is enabled.
                        light.color = filter.gamma;
                        light.intensity = (float)(room.PowerOf(rig, i) * maximum);
                    }
                }
            }

            private E27Target E27FreezeTarget(E27Case sample, out int captures)
            {
                string key = sample.mode + ":" + sample.target + ":" + Series(sample.truth);
                captures = 0;
                if (e27Targets.TryGetValue(key, out var saved)) return saved;
                E27Apply(sample.truth, sample, E27Mode(sample));
                foreach (int view in E27Views) low.Capture(view);
                var lf = E27Views.Select(low.Capture).ToArray();
                double repeat = E27Views.Select(v => NativeRenderBackend.Compare(low.Capture(v), lf[v])).Max();
                if (repeat > 1e-12) throw new InvalidDataException("Native target repeat differs.");
                var hf = Enumerable.Range(0, room.Views.Length).Select(high.Capture).ToArray();
                captures = 9 + hf.Length;
                var packed = new float[E23Channels];
                for (int view = 0, k = 0; view < lf.Length; ++view)
                    for (int i = 0; i < lf[view].PixelCount; ++i)
                    { var p = lf[view].Pixel(i); packed[k++] = p.r; packed[k++] = p.g; packed[k++] = p.b; }
                var bytes = new byte[4 * packed.Length];
                Buffer.BlockCopy(packed, 0, bytes, 0, bytes.Length);
                using var sha = SHA256.Create();
                string hex = BitConverter.ToString(sha.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant();
                double scale = E25Scale(packed);
                var target = new E27Target { low = lf, high = hf, packed = packed, hash = hex, scale = scale, repeat = repeat,
                    log = new E23FrameObjective(packed, Mapping.SignedLog, scale) };
                e27Targets.Add(key, target);
                return target;
            }

            private sealed partial class E27Trial : IDisposable
            {
                private readonly E21RoomStudy h;
                private readonly E27Arm arm;
                private readonly E27Case sample;
                private readonly string dir;
                private readonly E27Result result;
                private readonly StreamWriter query, decisions, checkpoints, profiling, scores;
                private readonly Stopwatch watch = new Stopwatch();
                private readonly E25Counter counters = new E25Counter();
                private readonly Lm.TrustTiming timing = new Lm.TrustTiming();
                private readonly HashSet<int> writtenCaps = new HashSet<int>();
                private readonly System.Random random;
                private Lm.TrustLayout layout;
                private int[] indices;
                private double[] best;
                private double bestLoss = double.PositiveInfinity;
                private E27Target target;
                private E23FrameObjective main;
                private int hardCamera, offCamera, scoreCamera;
                private double render0, read0, optSeconds, mapSeconds, orchestrationSeconds, gpuUpload, gpuSubmit, gpuSync;
                private int gpuColumns, gc0, gc1, gc2;
                private string segment;
                private double optimizeStart;
                private bool audit, incomplete;
                private readonly int[] capLines = { 1560, 3150, 4725, 5907, 7086, 9450 };

                internal E27Trial(E21RoomStudy host, E27Arm a, E27Case c, string directory, string sources)
                {
                    h = host; arm = a; sample = c; dir = directory; random = new System.Random(a.seed);
                    result = new E27Result { id = a.id, stage = a.stage, @case = a.@case, method = a.method, policy = a.policy, cap = a.cap,
                        seed = a.seed, status = "completed", initial = c.initial, truth = c.truth, sourceSha256 = sources };
                    query = Open("query-log.csv", "arm,cameras,diagnostic_cameras,segment,phase,shadows,axis,physical_step,views,full_f,branch_f,best_f,theta");
                    decisions = Open("branch-trace.csv", "arm,cameras,segment,model,source,status,legal,accepted,predicted,actual,rho,shrinks,branch_before,best_f,step_normalized,theta,unclipped_step,radius_step,lamp_shrinks,unclipped_prediction,constraint,actual_physical_step,current_reference_f");
                    checkpoints = Open("checkpoints.csv", "arm,kind,checkpoint,cameras,models,best_f,cm,relative,color,legal,pass,theta");
                    profiling = Open("profiling.csv", "arm,cameras,hard,off,diagnostic,acceptance,render_s,readback_s,mapping_fd_s,gram_gradient_s,solve_s,orchestration_s,opt_wall_s,total_s,gpu_columns,gpu_upload_s,gpu_submit_s,gpu_sync_s,gpu_exclusive_s,gc0,gc1,gc2");
                    scores = Open("view-metrics.csv", "arm,state,width,role,view,raw_nrmse,log_nrmse,display_nrmse");
                }

                private StreamWriter Open(string name, string columns)
                { var s = new StreamWriter(Path.Combine(dir, name)); s.WriteLine(columns); return s; }
                private static string T(double[] x) => x == null ? "" : Series(x);
                private void MakeLayout(double step, double colorStep = .03)
                {
                    var baseAxes = h.E26Axes(step);
                    var blocks = sample.active.Select(_ => sample.axes.Select(c => c switch {
                        'x' => baseAxes[0], 'y' => baseAxes[1], 'z' => baseAxes[2], 'i' => baseAxes[3],
                        't' => new Lm.TrustAxis(Lm.TrustCoordinate.Mired, MiredSpan, 5, 20, 50, MinMired),
                        'u' => new Lm.TrustAxis(Lm.TrustCoordinate.LogRedGreen, 6, colorStep, .25, .75, -3),
                        'v' => new Lm.TrustAxis(Lm.TrustCoordinate.LogBlueGreen, 6, colorStep, .25, .75, -3),
                        _ => throw new InvalidDataException("Unknown active axis " + c)
                    }).ToArray()).ToArray();
                    layout = new Lm.TrustLayout(blocks);
                    int width = 4 * h.room.Count;
                    indices = Enumerable.Range(0, layout.Count).Select(i => {
                        int owner = sample.active[layout.Owner(i)];
                        var axis = layout.Axis(i).Coordinate;
                        return axis switch { Lm.TrustCoordinate.Mired => width + owner,
                            Lm.TrustCoordinate.LogRedGreen => width + 2 * owner,
                            Lm.TrustCoordinate.LogBlueGreen => width + 2 * owner + 1, _ => 4 * owner + (int)axis };
                    }).ToArray();
                }
                private double[] Local(double[] rig) => indices.Select(i => rig[i]).ToArray();
                private double[] Expand(double[] x, double[] anchor = null)
                {
                    var rig = (double[])(anchor ?? sample.initial).Clone();
                    for (int i = 0; i < indices.Length; ++i) rig[indices[i]] = x[i];
                    return rig;
                }
                private bool Legal(double[] rig) => rig.All(v => v >= 0 && v <= 1) && h.room.Contains(rig.Take(4 * h.room.Count).ToArray());
                private (double[] cm, double[] intensity, double[] chroma, bool legal, bool pass) Errors(double[] rig)
                {
                    int width = 4 * h.room.Count; var gt = sample.truth;
                    var pos = sample.active.Select(i => 100d * Vector3.Distance(h.room.PositionOf(rig, i), h.room.PositionOf(gt, i))).ToArray();
                    var power = sample.active.Select(i => Math.Abs(h.room.PowerOf(rig, i) / h.room.PowerOf(gt, i) - 1)).ToArray();
                    var color = sample.active.Select(i => sample.mode == "white" ? MiredSpan * Math.Abs(rig[width + i] - gt[width + i]) :
                        sample.mode == "colored" ? 6 * Math.Max(Math.Abs(rig[width + 2 * i] - gt[width + 2 * i]), Math.Abs(rig[width + 2 * i + 1] - gt[width + 2 * i + 1])) : 0).ToArray();
                    bool legal = Legal(rig);
                    return (pos, power, color, legal, legal && pos.Max() <= 5 && power.Max() <= .05 && color.Max() <= (sample.mode == "white" ? 10 : .05));
                }

                internal void Run()
                {
                    watch.Start();
                    target = h.E27FreezeTarget(sample, out int targetCameras);
                    result.auditCameras += targetCameras;
                    result.targetSha256 = target.hash;
                    result.config = $"{sample.scene}:{sample.target}:{sample.mode}:{sample.axes}";
                    WriteJson(dir, "scene.json", h.room.Describe());
                    File.WriteAllText(Path.Combine(dir, "target.csv"), "sha256,scale,repeat_raw_loss,target_cameras,gpu,driver\n" +
                        string.Join(",", target.hash, F(target.scale), F(target.repeat), targetCameras, SystemInfo.graphicsDeviceName.Replace(',', ' '), SystemInfo.graphicsDeviceVersion.Replace(',', ' ')) + "\n");
                    MakeLayout(arm.step_mm > 0 ? arm.step_mm * .001 : .02);
                    FreezeReceipt();
                    if (!Legal(sample.initial)) { Fail(new InvalidDataException("Frozen initial is not collider-legal.")); return; }
                    main = arm.method == "H0" ? new E23FrameObjective(target.packed, Mapping.Raw, target.scale) : target.log;
                    gc0 = GC.CollectionCount(0); gc1 = GC.CollectionCount(1); gc2 = GC.CollectionCount(2);
                    audit = arm.policy is "audit" or "color-audit" or "temperature-audit" or "gpu-parity" or "mixed-layout";
                    if (arm.policy == "gpu-timed") Warmup();
                    render0 = h.low.RenderSeconds; read0 = h.low.ReadbackSeconds;
                    double clock0 = optimizeStart = watch.Elapsed.TotalSeconds;
                    if (!RunPolicy())
                    { result.status = "not_run"; result.reason = "policy not implemented"; }
                    optSeconds = watch.Elapsed.TotalSeconds - clock0;
                    double render = h.low.RenderSeconds - render0, readback = h.low.ReadbackSeconds - read0;
                    if (result.status == "not_run") { CommitResult(); return; }
                    if (incomplete) result.status = "incomplete";
                    if (best != null)
                    {
                        result.final = best; result.loss = bestLoss;
                        var e = Errors(best);
                        result.positionCm = e.cm; result.intensityRelative = e.intensity; result.colorError = e.chroma;
                        if (sample.mode == "colored")
                        {
                            int width = 4 * h.room.Count;
                            var color = sample.active.Select(lamp => {
                                int k = width + 2 * lamp;
                                var a = E27Chroma(6 * best[k] - 3, 6 * best[k + 1] - 3);
                                var b = E27Chroma(6 * sample.truth[k] - 3, 6 * sample.truth[k + 1] - 3);
                                return (Vector3.Distance(a, b), Vector3.Angle(a, b));
                            }).ToArray();
                            result.chromaL2 = color.Select(v => (double)v.Item1).ToArray();
                            result.chromaAngleDegrees = color.Select(v => (double)v.Item2).ToArray();
                        }
                        result.maxCm = e.cm.Max(); result.maxRelative = e.intensity.Max(); result.maxColor = e.chroma.Max();
                        result.pass = e.pass; result.legal = e.legal;
                        if (!audit)
                        {
                            foreach (int cap in capLines.Where(c => c <= arm.cap && (c <= 4725 || c == arm.cap)))
                                if (writtenCaps.Add(cap)) Checkpoint("camera", cap);
                            bool save = arm.stage == "0" || arm.direct && arm.seed is 0 or 1701 || arm.@case.Contains("color-full-911");
                            Score(sample.initial, "initial", save); Score(best, "final", save);
                            if (save) for (int v = 0; v <= 3; v += 3)
                                NativeRenderBackend.Save(target.high[v], E19HighW, E19HighH, Path.Combine(dir, "gt-v" + v));
                        }
                    }
                    result.acceptanceCameras = scoreCamera;
                    result.seconds = watch.Elapsed.TotalSeconds;
                    if (result.cameras > arm.cap) throw new InvalidDataException("Budget overshoot.");
                    profiling.WriteLine(string.Join(",", arm.id, result.cameras, hardCamera, offCamera, result.auditCameras, scoreCamera,
                        F(render), F(readback), F(mapSeconds + counters.Map), F(timing.GramGradientSeconds), F(timing.DampedSolveSeconds),
                        F(orchestrationSeconds), F(optSeconds), F(result.seconds), gpuColumns, F(gpuUpload), F(gpuSubmit), F(gpuSync), "",
                        GC.CollectionCount(0) - gc0, GC.CollectionCount(1) - gc1, GC.CollectionCount(2) - gc2));
                    CommitResult();
                }
                private void Warmup()
                {
                    using var gpu = arm.gpu ? new E25GpuColumns(main) : null;
                    var work = new float[E23Channels]; var off = new float[E23Channels]; var y = new float[E23Channels];
                    for (int k = 0; k < 3; ++k)
                    {
                        bool mode = audit; audit = true;
                        Capture(sample.initial, E27Mode(sample), E27Views, work);
                        Capture(sample.initial, LightShadows.None, E27Views, off);
                        audit = mode;
                        main.Evaluate(work, y);
                        if (gpu != null) { gpu.Anchor(work, off); gpu.Position(off, work, null, 0, .02, 3, y); }
                        else E25CachedColumn(main, work, work.Select(main.Transform).ToArray(), off, off, work, null, .02, 3, y);
                    }
                }
                [Serializable] private sealed class InputReceipt
                {
                    public string id, targetSha256, initialFloat64Sha256, truthFloat64Sha256;
                    public int[] active;
                    public double[] initial, truth;
                }
                private void FreezeReceipt()
                {
                    using var sha = SHA256.Create();
                    string Hex(double[] x)
                    {
                        var bytes = new byte[8 * x.Length]; Buffer.BlockCopy(x, 0, bytes, 0, bytes.Length);
                        return BitConverter.ToString(sha.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant();
                    }
                    var receipt = new InputReceipt { id = arm.id, targetSha256 = target.hash, initialFloat64Sha256 = Hex(sample.initial),
                        truthFloat64Sha256 = Hex(sample.truth), active = sample.active, initial = sample.initial, truth = sample.truth };
                    var path = Path.Combine(dir, "frozen-render-inputs.json");
                    if (!File.Exists(path)) WriteJson(dir, "frozen-render-inputs.json", receipt);
                    else if (JsonUtility.ToJson(receipt, true) != File.ReadAllText(path)) throw new InvalidDataException("Frozen arm inputs changed before retry.");
                }
                private void CommitResult()
                {
                    var pending = Path.Combine(dir, "result.pending.json");
                    File.WriteAllText(pending, JsonUtility.ToJson(result, true));
                    File.Move(pending, Path.Combine(dir, "result.json"));
                }
                internal void Fail(Exception e)
                {
                    result.status = "failed"; result.reason = e.GetType().Name; result.stop = e.Message;
                    result.seconds = watch.Elapsed.TotalSeconds;
                    CommitResult();
                }
                public void Dispose() { query.Dispose(); decisions.Dispose(); checkpoints.Dispose(); profiling.Dispose(); scores.Dispose(); }

                private void Score(double[] rig, string label, bool save)
                {
                    h.E27Apply(rig, sample, E27Mode(sample));
                    double rh = 0, lh = 0, dh = 0; int held = 0;
                    for (int v = 0; v < target.low.Length; ++v)
                    {
                        var f = h.low.Capture(v); ++scoreCamera;
                        scores.WriteLine(string.Join(",", arm.id, label, E19FitW, "fit_retake", v,
                            T(E25FrameScores(f, target.low[v], target.scale)).Replace(';', ',')));
                    }
                    for (int v = 0; v < target.high.Length; ++v)
                    {
                        var frame = h.high.Capture(v); ++scoreCamera;
                        var m = E25FrameScores(frame, target.high[v], target.scale);
                        scores.WriteLine(string.Join(",", arm.id, label, E19HighW, v < 3 ? "train_retake" : "held", v, T(m).Replace(';', ',')));
                        if (v >= 3) { rh += m[0] * m[0]; lh += m[1] * m[1]; dh += m[2] * m[2]; ++held; }
                        if (save && v is 0 or 3) NativeRenderBackend.Save(frame, E19HighW, E19HighH, Path.Combine(dir, label + "-v" + v));
                    }
                    if (label == "final" && held > 0)
                    { result.heldRaw = Math.Sqrt(rh / held); result.heldLog = Math.Sqrt(lh / held); result.heldDisplay = Math.Sqrt(dh / held); }
                }

                private void Capture(double[] rig, LightShadows shadows, int[] views, float[] rgb, E27Case input = null)
                {
                    if (!audit)
                    {
                        int after = result.cameras + views.Length;
                        if (after > arm.cap) throw new InvalidOperationException("E27 actual camera cap exhausted.");
                        foreach (int cap in capLines.Where(c => c <= arm.cap && (c <= 4725 || c == arm.cap) && after > c))
                            if (writtenCaps.Add(cap)) Checkpoint("camera", cap);
                        result.cameras = after;
                        if (shadows == LightShadows.Hard) hardCamera += views.Length; else offCamera += views.Length;
                    }
                    else result.auditCameras += views.Length;
                    h.E27Apply(rig, input ?? sample, shadows);
                    for (int k = 0; k < views.Length; ++k) h.low.CaptureRgb(views[k], rgb, k * E23ChannelsPerView);
                }
                private void Query(string phase, LightShadows mode, double[] rig, int[] views, double f = double.NaN, double branch = double.NaN, int axis = -1, double step = 0) =>
                    query.WriteLine(string.Join(",", arm.id, result.cameras, result.auditCameras, segment, phase, mode, axis, F(step), string.Join(";", views), F(f), F(branch), F(bestLoss), T(rig)));
                private void Observe(double[] rig, double f)
                {
                    if (best == null) result.initialLoss = f;
                    if (f < bestLoss && Legal(rig)) { bestLoss = f; best = (double[])rig.Clone(); }
                    if (!audit && result.firstPassCamera < 0 && Errors(rig).pass)
                    { result.firstPassCamera = result.cameras; result.firstPassModel = result.models; result.firstPassSeconds = watch.Elapsed.TotalSeconds - optimizeStart; }
                }
                private void Checkpoint(string kind, int value)
                {
                    if (best == null) return;
                    var e = Errors(best);
                    checkpoints.WriteLine(string.Join(",", arm.id, kind, value, result.cameras, result.models,
                        F(bestLoss), F(e.cm.Max()), F(e.intensity.Max()), F(e.chroma.Max()), Flag(e.legal), Flag(e.pass), T(best)));
                }
                private E23FrameObjective Objective(int[] views, double[] weight = null)
                {
                    if (views.SequenceEqual(E27Views) && weight == null) return main;
                    var obj = new E23FrameObjective(views.Select(v => target.low[v]).ToArray(), main.mapping, main.scale);
                    // Preserve the original full F per-view normalizers; subset weights are not recomputed.
                    for (int i = 0; i < views.Length; ++i) obj.weights[i] = main.weights[views[i]] * Math.Sqrt(weight?[i] ?? 1);
                    return obj;
                }
                private double Full(double[] rig, string phase, bool observe = true)
                {
                    var buf = new float[E23Channels];
                    var mode = E27Mode(sample);
                    Capture(rig, mode, E27Views, buf);
                    double f = main.Evaluate(buf);
                    if (observe) Observe(rig, f);
                    Query(phase, mode, rig, E27Views, f);
                    return f;
                }
            }
        }
    }
}

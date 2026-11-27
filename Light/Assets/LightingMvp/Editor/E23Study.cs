using System;
using System.IO;
using System.Linq;
using UnityEngine;
using Frame = LightingMvp.NativeRenderBackend.Frame;
using Mapping = LightingMvp.NativeRenderBackend.HdrMapping;
using NormalMode = LightingMvp.ImageJacobianLm.TrustNormalMode;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private static void RunE23Stages(string task, string output)
        {
            if (!Application.isBatchMode) throw new InvalidOperationException("E23 requires an isolated Unity batch project.");
            using var pipeline = new TestPipeline(classroom: true);
            string key = task.Substring(4);
            if (key.EndsWith("-smoke", StringComparison.Ordinal)) key = key.Substring(0, key.Length - 6);
            bool classroom = key == "f-classroom";
            using ISensitivityScene room = classroom ? new ClassroomSensitivityScene(pipeline.Asset, null) : new ComplexRoomSensitivityScene();
            using var study = new E21RoomStudy(room, task, Path.Combine(output, task.Replace('-', '_')),
                classroom ? ClassroomAssets + "/room.bytes" : NativeComplexRoomBuilder.ScenePath);
            study.RunE23(key);
            study.Finish();
        }

        private sealed partial class E21RoomStudy
        {
            private static readonly int[] E23Seeds = { 131, 157, 733 };
            private static readonly string[] E23Maps = { "C0", "C1", "C2", "C3", "C4" };
            private const int E23ChannelsPerView = E19FitW * E19FitH * 3;
            private const int E23Channels = E23ChannelsPerView * E19FitViews;
            private E21StageSummary e23Source;
            private double e23Scale;
            private StreamWriter e23ArmConfig, e23LampConfig;

            internal void RunE23(string key)
            {
                if (key != "f-classroom")
                {
                    string input = Arg("-bbE23Inputs", "");
                    e23Source = E22ReadBaseline(Path.Combine(input, "on-summary.json"));
                    var recorded = JsonUtility.FromJson<E22Supplement>(File.ReadAllText(Path.Combine(input, "on-metrics.json")));
                    var scales = recorded.metrics.Select(m => m.scale).Where(s => s > 0).Distinct().ToArray();
                    if (scales.Length != 1) throw new InvalidDataException("E22 Shadow On needs one recorded exposure scale.");
                    e23Scale = scales[0];
                }
                using (e23ArmConfig = Csv("arm-config.csv", "arm,seed,cap_cameras,mapping,jacobian,alpha,on_cameras,off_cameras,soft_cameras,olat_cameras,original_cameras,new_optimization_cameras,optimization_seconds,reused_from"))
                using (e23LampConfig = Csv("lamp-displacements.csv", "arm,lamp,position_start_cm,log_intensity_start,position_final_cm,intensity_final_relative"))
                    WithRoomMetrics(() =>
                    {
                        switch (key)
                        {
                            case "c-screen": E23Tone(false, false); break;
                            case "c-mix": E23Tone(false, true); break;
                            case "c-validate": E23Tone(true, false); break;
                            case "c-rgb": E23RgbDiagnostics(); break;
                            case "a": E23ShadowDecomposition(); break;
                            case "b-small": E23Hybrid(false); break;
                            case "b-full": E23Hybrid(true); break;
                            case "f-near": E23Neighbourhood(false); break;
                            case "f-best": E23Neighbourhood(true); break;
                            case "f-far": E23Generalization(false); break;
                            case "f-classroom": E23Generalization(true); break;
                            case "d-atlas": E23AtlasStudy(); break;
                            case "e-soft": E23SoftStudy(); break;
                            default: throw new ArgumentException("Unknown E23 stage " + key);
                        }
                    });
            }

            private E21Trial E23SourceArm(int seed, string mapping) => e23Source.arms.Single(a => a.id == $"seed{seed}_near_on_{mapping}");

            private double[] E23Near(int seed) => (double[])E23SourceArm(seed, "B0").initial.Clone();

            private static Mapping E23Mapping(string key) => E22Mapping("B" + key.Substring(1));

            private E21Signal E23Signal(MbdRgbTarget gt, string key) => key == "C5"
                ? E23PairedSignal(gt) : E22RgbSignal(gt, "B" + key.Substring(1), e23Scale);

            private void E23Tone(bool validate, bool mixture)
            {
                var gt = E22Target(true);
                string[] paths = validate ? new[] { "C0", "C1", Arg("-bbE23Mapping", "C3") }.Distinct().ToArray()
                    : mixture ? new[] { "C5" } : E23Maps;
                foreach (int seed in smoke ? E23Seeds.Take(1) : validate ? E23Seeds : E23Seeds.Take(2))
                foreach (string mapping in paths)
                {
                    var old = validate && !smoke && (mapping == "C0" || mapping == "C1") ? E23SourceArm(seed, "B" + mapping.Substring(1)) : null;
                    RunE23Arm($"seed{seed}_near_on", mapping, seed, room.Lamps, E23Near(seed), gt, mapping,
                        validate ? 1575 : 520, prior: old, reuse: old == null ? null : "E22-B-On/" + old.id);
                }
            }

            private E21Trial RunE23Arm(string scenario, string name, int seed, int[] active, double[] initial, MbdRgbTarget gt,
                string mapping, int budget, E21Trial prior = null, string reuse = null, E21Signal objective = null,
                Action<ImageJacobianLm.TrustSettings, Func<double[], double[]>> configure = null,
                Func<(int off, int soft)> modelCameras = null, string jacobian = "on_fd", double alpha = 1, string nativeMode = "hard")
            {
                var arm = RunE22Arm(scenario, name, active, initial, gt, objective ?? E23Signal(gt, mapping), budget,
                    NormalMode.LampScaledDense, mapping, e23Scale, reused: prior, reusedFrom: reuse, extra: configure);
                var measure = e22Metrics.Last();
                var counts = modelCameras?.Invoke() ?? (0, 0);
                int off = counts.Item1, soft = counts.Item2, original = measure.optimizationCameras;
                if (nativeMode == "soft") soft = original;
                if (nativeMode == "off") off = original;
                int on = original - off - soft - measure.olatCameras;
                if (on < 0 || original > arm.limit * E19FitViews) throw new InvalidOperationException("E23 optimization camera budget exceeded.");
                if (original != arm.evaluations * E19FitViews) throw new InvalidOperationException("E23 evaluation count disagrees with the native camera counter.");
                e23ArmConfig.WriteLine(string.Join(",", arm.id, seed, arm.limit * E19FitViews, mapping, jacobian, F(alpha), on, off, soft,
                    measure.olatCameras, original, measure.newOptimizationCameras, F(measure.optimizationSeconds), reuse ?? ""));
                for (int j = 0; j < active.Length; ++j)
                {
                    int lamp = active[j];
                    double cm = 100 * Vector3.Distance(room.PositionOf(initial, lamp), room.PositionOf(room.Truth, lamp));
                    double logI = (initial[4 * lamp + 3] - room.Truth[4 * lamp + 3]) * LogPowerSpan;
                    e23LampConfig.WriteLine(string.Join(",", arm.id, lamp + 1, F(cm), F(logI), F(arm.positionCm[j]), F(arm.intensityRelative[j])));
                }
                return arm;
            }

            private E21Signal E23PairedSignal(MbdRgbTarget gt)
            {
                var raw = new E23FrameObjective(gt.fit, Mapping.Raw, e23Scale);
                var logged = new E23FrameObjective(gt.fit, Mapping.SignedLog, e23Scale);
                var scratch = new float[E23Channels];
                return new E21Signal("native_raw_log_equal_parallel_448", 2 * E23Channels, (rig, residual) =>
                {
                    E23CaptureRgb(rig, LightShadows.Hard, scratch);
                    return .5 * raw.Evaluate(scratch, residual, factor: Math.Sqrt(.5)) + .5 * logged.Evaluate(scratch, residual, E23Channels, Math.Sqrt(.5));
                });
            }

            private void E23RgbDiagnostics()
            {
                var gt = E22Target(true);
                using var table = Csv("mapping-stats.csv", "pose,view,mapping,scale,raw_negative_fraction,display_saturated_fraction,mapping_saturated_fraction,near_zero_derivative_fraction,mean_derivative");
                using var costs = Csv("diagnostic-costs.csv", "pose,on_cameras,off_cameras");
                foreach (var anchor in E23Anchors().Where(a => !a.failed || a.name == "seed157_log_failed"))
                {
                    Apply(anchor.rig, true);
                    var frames = Enumerable.Range(0, E19FitViews).Select(low.Capture).ToArray();
                    for (int v = 0; v < frames.Length; ++v)
                    foreach (string key in E23Maps)
                    {
                        var mapping = E23Mapping(key);
                        bool bounded = mapping == Mapping.ToneExp || mapping == Mapping.ToneReinhard;
                        int negative = 0, displaySaturated = 0, saturated = 0, small = 0;
                        double sum = 0;
                        for (int p = 0; p < frames[v].PixelCount; ++p)
                        {
                            var rgb = frames[v].Pixel(p);
                            for (int ch = 0; ch < 3; ++ch)
                            {
                                float x = rgb[ch];
                                double slope = E23Derivative(x, mapping, e23Scale);
                                if (x < 0) ++negative;
                                if (NativeRenderBackend.Display(x) >= .99f) ++displaySaturated;
                                if (bounded && NativeRenderBackend.Map(x, mapping, e23Scale) >= .99f) ++saturated;
                                if (Math.Abs(slope) <= 1e-6) ++small;
                                sum += slope;
                            }
                        }
                        double n = 3 * frames[v].PixelCount;
                        table.WriteLine(string.Join(",", anchor.name, v, key, F(e23Scale), F(negative / n), F(displaySaturated / n),
                            F(saturated / n), F(small / n), F(sum / n)));
                    }
                    bool retain = anchor.name == "seed131_start" || anchor.name == "seed157_log_failed";
                    if (retain)
                    {
                        string path = CapturePath(anchor.name);
                        NativeRenderBackend.Save(frames[0], E19FitW, E19FitH, Path.Combine(path, "hard_v0"));
                        Apply(anchor.rig, false);
                        NativeRenderBackend.Save(low.Capture(0), E19FitW, E19FitH, Path.Combine(path, "off_v0"));
                    }
                    costs.WriteLine(string.Join(",", anchor.name, E19FitViews, Flag(retain)));
                }
            }

            private void E23CaptureRgb(double[] rig, LightShadows shadows, float[] values)
            {
                Apply(rig, shadows != LightShadows.None, shadows);
                for (int v = 0; v < E19FitViews; ++v) low.CaptureRgb(v, values, v * E23ChannelsPerView);
            }

            private sealed class E23FrameObjective
            {
                internal readonly Mapping mapping;
                internal readonly double scale;
                private readonly float[] target = new float[E23Channels];
                internal readonly double[] weights = new double[E19FitViews];
                internal E23FrameObjective(Frame[] frames, Mapping mapping, double scale)
                {
                    this.mapping = mapping; this.scale = scale;
                    for (int v = 0; v < E19FitViews; ++v)
                    {
                        double power = 0;
                        for (int i = 0, k = v * E23ChannelsPerView; i < frames[v].PixelCount; ++i)
                        {
                            Color p = frames[v].Pixel(i);
                            for (int ch = 0; ch < 3; ++ch)
                            {
                                float t = Transform(p[ch]); target[k++] = t;
                                power += (double)t * t;
                            }
                        }
                        weights[v] = 1 / Math.Sqrt(E19FitViews * (power + 1e-12));
                    }
                }

                internal float Transform(float x) => NativeRenderBackend.Map(x, mapping, scale);

                internal double Evaluate(float[] rgb, float[] residual = null, int offset = 0, double factor = 1)
                {
                    double loss = 0;
                    int k = 0;
                    foreach (double weight in weights)
                    {
                        double e = 0;
                        int end = k + E23ChannelsPerView;
                        for (; k < end; ++k)
                        {
                            double d = Transform(rgb[k]) - target[k];
                            if (residual != null) residual[offset + k] = (float)(d * weight * factor);
                            e += d * d;
                        }
                        loss += e * weight * weight;
                    }
                    return loss;
                }
            }
        }
    }
}

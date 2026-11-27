using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using HdrMapping = LightingMvp.NativeRenderBackend.HdrMapping;
using NormalMode = LightingMvp.ImageJacobianLm.TrustNormalMode;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            private static readonly string[] E22Mappings = { "B0", "B1", "B2", "B3", "B4", "B5" };
            private readonly Dictionary<bool, double> e22Scales = new Dictionary<bool, double>();

            private double E22Scale(MbdRgbTarget target)
            {
                if (e22Scales.TryGetValue(target.shadows, out var scale)) return scale;
                var positives = new List<float>(target.fit.Length * target.fit[0].PixelCount * 3);
                foreach (var frame in target.fit)
                for (int i = 0; i < frame.PixelCount; ++i)
                {
                    var p = frame.Pixel(i);
                    if (p.r > 0) positives.Add(p.r);
                    if (p.g > 0) positives.Add(p.g);
                    if (p.b > 0) positives.Add(p.b);
                }
                if (positives.Count == 0) throw new InvalidOperationException("HDR target contains no positive channels.");
                positives.Sort();
                scale = Math.Max(1e-8, positives[(int)Math.Ceiling(.90 * positives.Count) - 1]);
                e22Scales.Add(target.shadows, scale);
                return scale;
            }

            private static HdrMapping E22Mapping(string key) => key switch
            {
                "B0" or "B5" => HdrMapping.Raw,
                "B1" => HdrMapping.SignedLog,
                "B2" => HdrMapping.SignedSqrt,
                "B3" => HdrMapping.ToneExp,
                "B4" => HdrMapping.ToneReinhard,
                _ => throw new ArgumentException(key)
            };

            private E21Signal E22RgbSignal(MbdRgbTarget target, string key, double scale)
            {
                if (key == "B0") return NativeSignal(target);
                if (key == "B5") return RobustSignal(target, RgbObjective.Huber);
                var mode = E22Mapping(key);
                var mapped = target.fit.Select(f => new NativeRenderBackend.MappedTarget(f, mode, scale)).ToArray();
                int channels = target.fit[0].PixelCount * 3;
                return new E21Signal("native_" + mode.ToString().ToLowerInvariant() + "_448", channels * E19FitViews, (x, residual) =>
                {
                    Apply(x, target.shadows);
                    double sum = 0;
                    for (int v = 0; v < E19FitViews; ++v)
                        sum += low.MappedResidual(v, mapped[v], residual, v * channels, E19FitViews, out _);
                    return sum;
                });
            }

            private void E22ToneScreen()
            {
                var gt = E22Target(false);
                var mode = E22Solver(Arg("-bbE22Solver", "A0"));
                var scale = E22Scale(gt);
                var signals = E22Mappings.Select(k => E22RgbSignal(gt, k, scale)).ToArray();
                E21StageSummary old = mode == NormalMode.Shared && !smoke ? E22ReadE21("e21_d") : null;
                foreach (int seed in smoke ? new[] { 131 } : new[] { 131, 157 })
                for (int m = 0; m < E22Mappings.Length; ++m)
                {
                    string key = E22Mappings[m], scenario = $"seed{seed}_near_off";
                    string original = key == "B0" ? $"seed{seed}_near_L2_448" : key == "B5" ? $"seed{seed}_near_Huber_448" : "";
                    var prior = old?.arms.SingleOrDefault(r => r.id == original && r.objective == signals[m].name);
                    RunE22Arm(scenario, key, room.Lamps, E22Start(seed, room.Lamps), gt, signals[m], 520, mode, key, scale,
                        reused: prior, reusedFrom: prior == null ? null : "E21-D/" + original);
                }
            }

            private void E22ToneValidate(bool shadows, bool far, int[] seeds)
            {
                var gt = E22Target(shadows);
                var mode = E22Solver(Arg("-bbE22Solver", "A0"));
                var scale = E22Scale(gt);
                string selected = Arg("-bbE22Mapping", "B0");
                string[] mappings = selected == "B0" ? new[] { "B0" } : new[] { "B0", selected };
                var signals = mappings.Select(k => E22RgbSignal(gt, k, scale)).ToArray();
                var seven = !shadows && !far && !smoke && seeds.Contains(733) ? E22ReadOwn("e22_a28") : null;
                foreach (int seed in smoke ? seeds.Take(1) : seeds)
                for (int m = 0; m < mappings.Length; ++m)
                {
                    string key = mappings[m];
                    string scenario = $"seed{seed}_{(far ? "far" : "near")}_{(shadows ? "on" : "off")}";
                    var prior = seed == 733 && key == "B0" ? seven?.arms.SingleOrDefault(r => r.method == E22SolverKey(mode)) : null;
                    RunE22Arm(scenario, key, room.Lamps, E22Start(seed, room.Lamps, far), gt, signals[m], 1575, mode, key, scale,
                        reused: prior, reusedFrom: prior == null ? null : "E22-A28/" + prior.id);
                }
            }

            private void E22ToneSwitch()
            {
                int seed = int.Parse(Arg("-bbE22Seed", "157"), Invariant);
                bool shadows = Arg("-bbE22Shadows", "off") == "on";
                string key = Arg("-bbE22Mapping", "B0");
                if (key == "B0") throw new ArgumentException("The scheduled loss arm requires a compressed or robust front objective.");
                var gt = E22Target(shadows);
                double scale = E22Scale(gt);
                RunE22Arm($"seed{seed}_near_{(shadows ? "on" : "off")}", key + "_then_raw", room.Lamps, E22Start(seed, room.Lamps), gt,
                    E22RgbSignal(gt, key, scale), 1575, E22Solver(Arg("-bbE22Solver", "A0")), key, scale, tail: NativeSignal(gt));
            }

            private E21StageSummary E22ReadE21(string source)
            {
                string path = Path.Combine(Arg("-bbE22Inputs", ""), source + ".json");
                return E22ReadBaseline(path);
            }

            private E21StageSummary E22ReadOwn(string source)
            {
                string path = Path.Combine(Path.GetDirectoryName(directory), source, "summary.json");
                return File.Exists(path) ? E22ReadBaseline(path) : null;
            }

            private E21StageSummary E22ReadBaseline(string path)
            {
                var baseline = JsonUtility.FromJson<E21StageSummary>(File.ReadAllText(path));
                if (baseline.sceneHash != E21Sha(NativeComplexRoomBuilder.ScenePath) || !E22SamePose(baseline.truth, room.Truth) ||
                    baseline.scene.unity != Application.unityVersion || baseline.scene.urp != room.Describe().urp)
                    throw new InvalidDataException("E22 comparison source does not match scene, rig or renderer versions: " + path);
                return baseline;
            }
        }
    }
}

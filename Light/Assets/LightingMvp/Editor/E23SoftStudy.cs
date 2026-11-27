using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using Frame = LightingMvp.NativeRenderBackend.Frame;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            private bool e23SoftTarget;
            private Action<LightShadows> e23ShadowSettings;

            private void E23SoftStudy()
            {
                const string mapping = "C0";
                var hard = E22Target(true);
                var hardDisplay = displayTargets[true];
                int[] seeds = smoke ? new[] { 157 } : E23Seeds.Take(2).ToArray();
                var anchors = E23Anchors().Where(a => a.name == "gt" || seeds.Contains(a.seed) && (a.name.EndsWith("_start") || a.name.EndsWith("_log_failed"))).ToArray();
                using var atlas = new E23AtlasObserver(lights);
                using var layout = E23AtlasCsv();
                using var compare = Csv("soft-vs-hard.csv", "pose,view,hard_config_raw_delta,soft_to_original_hard_raw,soft_to_original_hard_display,hard_layout_preserved,soft_layout_preserved");
                using var hardReport = Csv("hard-recheck.csv", "arm,resolution,view,raw_nrmse,display_nrmse");
                using var modelLog = E23ModelLog();
                var original = anchors.Select(a =>
                {
                    Apply(a.rig, true);
                    return Enumerable.Range(0, E19FitViews).Select(v =>
                    {
                        var obs = atlas.Capture(() => low.Capture(v));
                        E23WriteLayout(layout, "original_hard", a.name, v, "base", obs.state);
                        return obs;
                    }).ToArray();
                }).ToArray();
                var pipeline = (UniversalRenderPipelineAsset)GraphicsSettings.currentRenderPipeline;
                var property = new SerializedObject(pipeline);
                property.FindProperty("m_SoftShadowsSupported").boolValue = true;
                property.ApplyModifiedPropertiesWithoutUndo();
                var originalRequests = lights.Select(light => ((int)light.shadowResolution, light.GetUniversalAdditionalLightData().additionalLightsShadowResolutionTier)).ToArray();
                foreach (var light in lights) light.GetUniversalAdditionalLightData().softShadowQuality = SoftShadowQuality.Medium;
                bool fixedResolution = false;
                e23ShadowSettings = mode =>
                {
                    bool useFixed = mode == LightShadows.Soft;
                    if (useFixed == fixedResolution) return;
                    if (useFixed) E23FixFaceSize();
                    else for (int i = 0; i < lights.Length; ++i)
                        E23SetFaceRequest(lights[i], originalRequests[i].Item2, originalRequests[i].Item1);
                    fixedResolution = useFixed;
                };
                File.WriteAllText(Path.Combine(directory, "soft-config.json"),
                    "{\"quality\":\"Medium\",\"face_request\":256,\"atlas_request\":2048,\"pipeline_bias\":[" +
                    F(pipeline.shadowDepthBias) + "," + F(pipeline.shadowNormalBias) + "],\"per_light_bias\":[" +
                    string.Join(",", lights.Select(l => "[" + F(l.shadowBias) + "," + F(l.shadowNormalBias) + "]")) + "]}\n");

                for (int index = 0; index < anchors.Length; ++index)
                for (int v = 0; v < E19FitViews; ++v)
                {
                    var a = anchors[index];
                    var prev = original[index][v];
                    Apply(a.rig, true);
                    var current = atlas.Capture(() => low.Capture(v));
                    Apply(a.rig, true, LightShadows.Soft);
                    var soft = atlas.Capture(() => low.Capture(v));
                    E23WriteLayout(layout, "hard_soft_support_enabled", a.name, v, "base", current.state, prev.state);
                    E23WriteLayout(layout, "fixed_soft_medium", a.name, v, "base", soft.state, current.state);
                    double delta = Math.Sqrt(NativeRenderBackend.Compare(current.rgb, prev.rgb));
                    if (delta > 1e-6) throw new InvalidOperationException("Enabling Soft support changed Hard radiance.");
                    if (soft.state.faces.Length != 42 || soft.state.faces.Any(face => face.size != 256 || face.requested != 256))
                        throw new InvalidOperationException("Soft did not allocate 42 faces at 256 pixels per face.");
                    compare.WriteLine(string.Join(",", a.name, v, F(delta), F(Math.Sqrt(NativeRenderBackend.Compare(soft.rgb, prev.rgb))),
                        F(Math.Sqrt(DisplayLoss(soft.rgb, new E22DisplayFrame(prev.rgb)))),
                        Flag(prev.state.Signature == current.state.Signature), Flag(soft.state.Signature == current.state.Signature)));
                }
                E23SoftColumns(anchors);

                var oldScreen = smoke ? null : E22ReadBaseline(Path.Combine(Path.GetDirectoryName(directory), "e23_c_screen", "summary.json"));
                foreach (int seed in seeds)
                {
                    string label = $"seed{seed}_near";
                    var start = E23Near(seed);
                    var prior = oldScreen?.arms.Single(a => a.id == $"seed{seed}_near_on_C0");
                    var e0 = RunE23Arm(label, "E0", seed, room.Lamps, start, hard, mapping,
                        520, prior: prior, reuse: prior == null ? null : "E23-C-SCREEN/" + prior.id);
                    var e2 = E23RunProxyArm(label, "E2", seed, room.Lamps, start, hard, mapping, 520, modelLog, LightShadows.Soft, 0, "soft_xyz");
                    Recheck(e0); Recheck(e2);
                }
                e23SoftTarget = true;
                targets.Remove(true);
                displayTargets.Remove(true);
                checkedGt.Clear();
                var softGt = E22Target(true);
                foreach (int seed in seeds)
                {
                    var e1 = RunE23Arm($"seed{seed}_near", "E1", seed, room.Lamps, E23Near(seed), softGt, mapping, 520,
                        jacobian: "soft_fd", nativeMode: "soft");
                    Recheck(e1);
                }
                e23SoftTarget = false;

                void Recheck(E21Trial arm)
                {
                    bool saved = e23SoftTarget;
                    e23SoftTarget = false;
                    try
                    {
                        Apply(arm.final, true);
                        Measure(low, hard.fit, hardDisplay.low);
                        Measure(high, hard.high, hardDisplay.high);
                    }
                    finally { e23SoftTarget = saved; }

                    void Measure(NativeRenderBackend backend, Frame[] target, E22DisplayFrame[] display)
                    {
                        for (int view = 0; view < target.Length; ++view)
                        {
                            var shot = backend.Capture(view);
                            hardReport.WriteLine(string.Join(",", arm.id, backend.Width, view, F(Math.Sqrt(NativeRenderBackend.Compare(shot, target[view]))),
                                F(Math.Sqrt(DisplayLoss(shot, display[view])))));
                            if (saved && view == 0 && backend == high)
                                NativeRenderBackend.Save(shot, backend.Width, backend.Height, Path.Combine(CapturePath(arm.id), "best_original_hard_v0"));
                        }
                    }
                }
            }

            private void E23SoftColumns(E23Anchor[] anchors)
            {
                using var log = Csv("soft-fd.csv", "pose,axis,step_mm,stencil,actual_first_mm,hard_norm,soft_norm,hard_vs_soft_cos,hard_step_vs_20_cos,soft_step_vs_20_cos");
                var first = new float[E23Channels]; var second = new float[E23Channels];
                var hard = new float[E23Channels]; var soft = new float[E23Channels];
                var hard20 = new float[E23Channels]; var soft20 = new float[E23Channels];
                var hardBase = new float[E23Channels]; var softBase = new float[E23Channels];
                foreach (var a in anchors)
                {
                    E23CaptureRgb(a.rig, LightShadows.Hard, hardBase);
                    E23CaptureRgb(a.rig, LightShadows.Soft, softBase);
                    foreach (int lamp in a.selected)
                    for (int d = 0; d < 3; ++d)
                    {
                        int axis = 4 * lamp + d;
                        foreach (int mm in new[] { 20, 5, 50 })
                        {
                            E23Stencil s = E23SelectStencil(a.rig, axis, .001 * mm);
                            if (s.code == 0) continue;
                            Column(LightShadows.Hard, hardBase, hard);
                            Column(LightShadows.Soft, softBase, soft);
                            if (mm == 20) { Array.Copy(hard, hard20, hard.Length); Array.Copy(soft, soft20, soft.Length); }
                            log.WriteLine(string.Join(",", a.name, axis, mm, s.code, F(s.firstStep * units[d] * 1000),
                                F(Math.Sqrt(E21Dot(hard, hard))), F(Math.Sqrt(E21Dot(soft, soft))),
                                F(Cos(hard, soft)), F(Cos(hard, hard20)), F(Cos(soft, soft20))));

                            void Column(LightShadows mode, float[] baseline, float[] result)
                            {
                                E23CaptureRgb(s.first, mode, first);
                                if (s.second != null) E23CaptureRgb(s.second, mode, second);
                                for (int i = 0; i < result.Length; ++i)
                                    result[i] = (float)(s.a * first[i] + s.b * second[i] + s.c * baseline[i]);
                            }
                        }
                    }
                }
            }
        }
    }
}

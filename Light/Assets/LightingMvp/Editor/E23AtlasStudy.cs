using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using Frame = LightingMvp.NativeRenderBackend.Frame;
using Object = UnityEngine.Object;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            private sealed class E23Layout
            {
                internal int width, height;
                internal int[] visible;
                internal (int lamp, int face, int size, int x, int y, int requested, float depth, float normal, float guard)[] faces;
                internal string Signature => width + "x" + height + "|" + string.Join(";", visible) + "|" + string.Join(";", faces.Select(f =>
                    $"{f.lamp}:{f.face}:{f.size}:{f.x}:{f.y}:{F(f.depth)}:{F(f.normal)}:{F(f.guard)}"));
            }

            private sealed class E23AtlasObserver : IDisposable
            {
                private const BindingFlags Fields = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;
                private static readonly FieldInfo features = typeof(ScriptableRenderer).GetField("m_RendererFeatures", Fields);
                private static readonly FieldInfo shadowPass = typeof(UniversalRenderer).GetField("m_AdditionalLightsShadowCasterPass", Fields);
                private readonly E23AtlasFeature feature;
                private readonly Light[] lamps;
                private List<ScriptableRendererFeature> attached;
                private E23Layout result;

                internal E23AtlasObserver(Light[] lights)
                {
                    lamps = lights;
                    feature = ScriptableObject.CreateInstance<E23AtlasFeature>();
                    feature.hideFlags = HideFlags.HideAndDontSave;
                    feature.observe = Read;
                }

                internal (Frame rgb, E23Layout state) Capture(Func<Frame> render)
                {
                    if (attached != null) throw new InvalidOperationException("An E23 atlas observation is already in progress.");
                    result = null;
                    RenderPipelineManager.beginCameraRendering += Attach;
                    try
                    {
                        var frame = render();
                        return (frame, result ?? throw new InvalidOperationException("The URP atlas observer did not execute."));
                    }
                    finally
                    {
                        RenderPipelineManager.beginCameraRendering -= Attach;
                        attached?.Remove(feature); attached = null;
                    }
                }

                private void Attach(ScriptableRenderContext context, Camera camera)
                {
                    if (attached != null) return;
                    var renderer = camera.GetUniversalAdditionalCameraData().scriptableRenderer;
                    if (!(renderer is UniversalRenderer)) throw new InvalidOperationException("Atlas audit needs the URP Universal Renderer.");
                    attached = features?.GetValue(renderer) as List<ScriptableRendererFeature>;
                    if (attached == null) throw new InvalidOperationException("Could not observe the active renderer features in URP 14.");
                    attached.Add(feature);
                }

                private void Read(ref RenderingData data)
                {
                    var camera = data.cameraData.camera;
                    var renderer = camera.GetUniversalAdditionalCameraData().scriptableRenderer;
                    var pass = shadowPass?.GetValue(renderer) ?? throw new InvalidOperationException("Missing additional-light shadow pass.");
                    var type = pass.GetType();
                    var visible = data.lightData.visibleLights;
                    var shadow = data.shadowData;
                    int[] present = Enumerable.Range(0, visible.Length).Select(i => Array.IndexOf(lamps, visible[i].light) + 1).Where(l => l > 0).OrderBy(l => l).ToArray();
                    var layout = new E23Layout { visible = present };
                    var slices = type.GetField("m_AdditionalLightsShadowSlices", Fields)?.GetValue(pass) as ShadowSliceData[];
                    var plane = type.GetField("m_GlobalShadowSliceIndexToPerLightShadowSliceIndex", Fields)?.GetValue(pass) as List<int>;
                    var map = type.GetField("m_ShadowSliceToAdditionalLightIndex", Fields)?.GetValue(pass) as List<int>;
                    var indices = type.GetField("m_AdditionalLightIndexToVisibleLightIndex", Fields)?.GetValue(pass) as int[];
                    var guard = type.GetMethod("GetPointLightShadowFrustumFovBiasInDegrees", BindingFlags.NonPublic | BindingFlags.Static);
                    if (map == null || indices == null || plane == null || guard == null) throw new InvalidOperationException("URP 14 atlas metadata is unavailable.");
                    var faces = new List<(int lamp, int face, int size, int x, int y, int requested, float depth, float normal, float guard)>();
                    if (present.Length > 0 && slices != null)
                    for (int i = 0; i < Math.Min(map.Count, slices.Length); ++i)
                    {
                        int index = indices[map[i]];
                        if (index < 0 || index >= visible.Length) continue;
                        var visibleLight = visible[index];
                        var lamp = visibleLight.light;
                        int id = Array.IndexOf(lamps, lamp) + 1;
                        if (id <= 0 || !lamp || !lamp.enabled || lamp.shadows == LightShadows.None || slices[i].resolution <= 0) continue;
                        var slice = slices[i];
                        var bias = ShadowUtils.GetShadowBias(ref visibleLight, index, ref shadow, slice.projectionMatrix, slice.resolution);
                        float angle = (float)guard.Invoke(null, new object[] { slice.resolution, lamp.shadows == LightShadows.Soft });
                        faces.Add((id, plane[i], slice.resolution, slice.offsetX, slice.offsetY, shadow.resolution[index], bias.x, bias.y, angle));
                    }
                    layout.faces = faces.OrderBy(x => x.lamp).ThenBy(x => x.face).ToArray();
                    if (layout.faces.Length > 0)
                    {
                        var atlas = E22Atlas.Snapshot(camera);
                        layout.width = atlas.Item1; layout.height = atlas.Item2;
                    }
                    result = layout;
                }

                public void Dispose() => Object.DestroyImmediate(feature);
            }

            private sealed class E23AtlasFeature : ScriptableRendererFeature
            {
                internal delegate void Observe(ref RenderingData data);
                internal Observe observe;
                private E23AtlasPass pass;
                public override void Create() => pass = new E23AtlasPass(this);
                public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData) => renderer.EnqueuePass(pass);

                private sealed class E23AtlasPass : ScriptableRenderPass
                {
                    private readonly E23AtlasFeature feature;
                    internal E23AtlasPass(E23AtlasFeature feature) { this.feature = feature; renderPassEvent = RenderPassEvent.BeforeRenderingOpaques; }
                    public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData) => feature.observe(ref renderingData);
                }
            }

            private StreamWriter E23AtlasCsv() => Csv("atlas-layout.csv", "config,pose,view,probe,lamp,face,visible,atlas_width,atlas_height,faces,requested,resolution,x,y,depth_bias,normal_bias,guard_deg,identity_matches_base");

            private void E23WriteLayout(StreamWriter log, string cfg, string pose, int view, string phase, E23Layout layout, E23Layout basis = null)
            {
                string[] common = { cfg, pose, view.ToString(Invariant), phase };
                string suffix = basis == null ? "" : Flag(layout.Signature == basis.Signature).ToString(Invariant);
                for (int lamp = 1; lamp <= lights.Length; ++lamp)
                {
                    var rows = layout.faces.Where(f => f.lamp == lamp).ToArray();
                    if (rows.Length == 0)
                        log.WriteLine(string.Join(",", common) + "," + string.Join(",", lamp, -1, Flag(layout.visible.Contains(lamp)), layout.width, layout.height, layout.faces.Length, 0, 0, 0, 0, 0, 0, 0, suffix));
                    else foreach (var row in rows)
                        log.WriteLine(string.Join(",", common) + "," + string.Join(",", lamp, row.face, 1, layout.width, layout.height, layout.faces.Length,
                            row.requested, row.size, row.x, row.y, F(row.depth), F(row.normal), F(row.guard), suffix));
                }
            }

            private void E23FixFaceSize()
            {
                foreach (var lamp in lights)
                    E23SetFaceRequest(lamp, UniversalAdditionalLightData.AdditionalLightsShadowResolutionTierCustom, 256);
            }

            private static void E23SetFaceRequest(Light lamp, int tier, int resolution)
            {
                using var native = new SerializedObject(lamp);
                // This is the integer field used by the URP Inspector's Custom tier.
                native.FindProperty("m_Shadows.m_Resolution").intValue = resolution;
                native.ApplyModifiedPropertiesWithoutUndo();
                using var urp = new SerializedObject(lamp.GetUniversalAdditionalLightData());
                urp.FindProperty("m_AdditionalLightsShadowResolutionTier").intValue = tier;
                urp.ApplyModifiedPropertiesWithoutUndo();
            }

            [Serializable] private sealed class E23AtlasGate
            {
                public int defaultStates, defaultChanges, fixedStates, fixedChanges, fixedNon256, retainedTests, olatRenders;
                public bool retentionPassed, linearPassed, columnsPassed;
                public double floor, median, worst;
                public string stop;
            }

            private void E23AtlasStudy()
            {
                var gt = E22Target(true);
                var display = displayTargets[true];
                using var observer = new E23AtlasObserver(lights);
                using var layout = E23AtlasCsv();
                using var compare = Csv("atlas-rgb.csv", "pose,view,fixed_to_original_raw,fixed_to_original_display,state_raw_to_orig_gt,state_display_to_orig_gt");
                using var allocation = Csv("olat-retention.csv", "pose,view,floor,case,visible,faces,layout_matches_full,remaining_floor_fraction,cameras");
                using var errors = Csv("olat-errors.csv", "pose,view,floor,case,relative_l2,max_pixel_rgb_error,hotspot_relative_l2");
                using var derivative = Csv("olat-strength-columns.csv", "pose,view,lamp,cosine,relative_l2");
                var anchors = (smoke ? E23Anchors().Where(a => a.name == "seed733_start" || a.name == "gt") : E23Anchors()).ToArray();
                var before = new Dictionary<(string,int), Frame>();
                var gate = new E23AtlasGate { stop = "retention_failed", floor = -1, median = -1, worst = -1 };
                foreach (bool fixedSize in new[] { false, true })
                {
                    if (fixedSize) E23FixFaceSize();
                    string cfg = fixedSize ? "fixed_request256" : "original";
                    foreach (var anchor in anchors)
                    for (int v = 0; v < E19FitViews; ++v)
                    {
                        var root = Render(anchor.rig, v);
                        E23WriteLayout(layout, cfg, anchor.name, v, "base", root.state);
                        if (fixedSize && (root.state.faces.Length == 0 || root.state.faces.Any(s => s.requested != 256)))
                            throw new InvalidOperationException("The renderer did not receive the fixed 256 face request.");
                        if (!fixedSize) before.Add((anchor.name, v), root.rgb);
                        else
                        {
                            var original = before[(anchor.name, v)];
                            compare.WriteLine(string.Join(",", anchor.name, v, F(Math.Sqrt(NativeRenderBackend.Compare(root.rgb, original))),
                                F(Math.Sqrt(DisplayLoss(root.rgb, new E22DisplayFrame(original)))), F(Math.Sqrt(NativeRenderBackend.Compare(root.rgb, gt.fit[v]))),
                                F(Math.Sqrt(DisplayLoss(root.rgb, display.low[v])))));
                            if (root.state.faces.Length != 42 || root.state.faces.Any(s => s.size != 256)) ++gate.fixedNon256;
                        }
                        foreach (double h in new[] { .005, .02 })
                        for (int axis = 4 * anchor.selected[0]; axis < 4 * anchor.selected[0] + 3; ++axis)
                        {
                            var stencil = E23SelectStencil(anchor.rig, axis, h);
                            Test(stencil.first, stencil.firstStep); Test(stencil.second, stencil.secondStep);
                            void Test(double[] shifted, double offset)
                            {
                                if (shifted == null) return;
                                var state = Render(shifted, v).state;
                                E23WriteLayout(layout, cfg, anchor.name, v, $"a{axis}_h{Math.Round(h*1000)}_d{F(offset * units[axis%4])}", state, root.state);
                                bool same = state.Signature == root.state.Signature;
                                if (fixedSize) { ++gate.fixedStates; if (!same) ++gate.fixedChanges; if (state.faces.Length != 42 || state.faces.Any(s => s.size != 256)) ++gate.fixedNon256; }
                                else { ++gate.defaultStates; if (!same) ++gate.defaultChanges; }
                            }
                        }
                    }
                }

                var olatPoses = anchors.Where(x => x.name == "gt" || x.name == "seed733_start").ToArray();
                int startCameras = low.RenderCount;
                foreach (double floor in new[] { 0, .0001 })
                {
                    bool valid = gate.fixedNon256 == 0;
                    foreach (var anchor in olatPoses)
                    for (int v = 0; v < E19FitViews; ++v)
                    {
                        var full = Render(anchor.rig, v);
                        var dark = Render(anchor.rig, v, lamps => { foreach (var lamp in lamps) lamp.enabled = false; });
                        var faint = Expose(anchor, v, floor, Array.Empty<int>(), Array.Empty<double>());
                        double relative = IncrementFraction(faint.rgb, dark.rgb, full.rgb);
                        Retention("floor", faint.state, relative);
                        Retention("L5", Expose(anchor, v, floor, new[] { 4 }, new[] { 1d }).state, relative);
                        Retention("L5_L7", Expose(anchor, v, floor, new[] { 4, 6 }, new[] { room.PowerOf(anchor.rig, 4), room.PowerOf(anchor.rig, 6) }).state, relative);

                        void Retention(string sample, E23Layout state, double remaining)
                        {
                            bool same = full.state.Signature == state.Signature;
                            valid &= same && remaining <= .001;
                            ++gate.retainedTests;
                            allocation.WriteLine(string.Join(",", anchor.name, v, F(floor), sample, state.visible.Length, state.faces.Length, Flag(same), F(remaining), 1));
                            E23WriteLayout(layout, "olat_retention", anchor.name, v, $"{F(floor)}_{sample}", state, full.state);
                        }
                    }
                    if (!valid) continue;
                    gate.retentionPassed = true; gate.floor = floor;
                    Audit(floor);
                    break;
                }
                gate.olatRenders = low.RenderCount - startCameras;
                olatFrames += gate.olatRenders;
                WriteJson(directory, "atlas-gate.json", gate);
                UnityEngine.Debug.Log($"[E23-D] default {gate.defaultChanges}/{gate.defaultStates} layouts changed, fixed {gate.fixedChanges}/{gate.fixedStates}, OLAT {gate.stop}");

                (Frame rgb, E23Layout state) Render(double[] rig, int v, Action<Light[]> change = null)
                {
                    Apply(rig, true); change?.Invoke(lights);
                    return observer.Capture(() => low.Capture(v));
                }

                (Frame rgb, E23Layout state) Expose(E23Anchor anchor, int view, double floor, int[] active, double[] power) => Render(anchor.rig, view, current =>
                {
                    foreach (var lamp in current) lamp.intensity = (float)floor;
                    for (int i = 0; i < active.Length; ++i) current[active[i]].intensity = (float)power[i];
                });

                static double IncrementFraction(Frame img, Frame empty, Frame full)
                {
                    double numerator = 0, denominator = 0;
                    for (int i = 0; i < img.PixelCount; ++i)
                    {
                        numerator += Norm(ProxyGeometry.Rgb(img.Pixel(i) - empty.Pixel(i)));
                        denominator += Norm(ProxyGeometry.Rgb(full.Pixel(i) - empty.Pixel(i)));
                    }
                    return Math.Sqrt(numerator / Math.Max(1e-24, denominator));
                }

                void Audit(double floor)
                {
                    var relative = new List<double>();
                    bool allStable = true, columnsGood = true;
                    foreach (var anchor in olatPoses)
                    for (int v = 0; v < E19FitViews; ++v)
                    {
                        var full = Render(anchor.rig, v); int[] allLamps = room.Lamps;
                        var empty = Measured("floor", Array.Empty<int>(), Array.Empty<double>());
                        var unit = allLamps.Select(i => Increment(Measured("unit_L" + (i + 1), new[] { i }, new[] { 1d }), empty)).ToArray();
                        foreach (int lamp in allLamps)
                        foreach (double a in new[] { .5, 1, 2 }) Check($"L{lamp + 1}_{a}", new[] { lamp }, new[] { a });
                        var pair = new[] { 4, 6 };
                        Check("L5_L7", pair, pair.Select(i => room.PowerOf(anchor.rig, i)).ToArray());
                        Check("all_7", allLamps, allLamps.Select(i => room.PowerOf(anchor.rig, i)).ToArray());
                        foreach (int lamp in pair)
                        {
                            double a = room.PowerOf(anchor.rig, lamp), h = .04;
                            var p = (double[])anchor.rig.Clone(); var m = (double[])anchor.rig.Clone();
                            p[4*lamp+3] += h / LogPowerSpan; m[4*lamp+3] -= h / LogPowerSpan;
                            var pos = Render(p, v); var neg = Render(m, v);
                            allStable &= pos.state.Signature == full.state.Signature && neg.state.Signature == full.state.Signature;
                            var fd = Increment(pos.rgb, neg.rgb); var native = unit[lamp];
                            double dot = E21Dot(fd,native), ff = E21Dot(fd,fd), nn = E21Dot(native,native);
                            double factor = 2 * h * a / (1-floor);
                            double cosine = dot / Math.Max(1e-24,Math.Sqrt(ff*nn));
                            double nrmse = Math.Sqrt(Math.Max(0,ff + factor*factor*nn - 2*factor*dot) / Math.Max(1e-24,ff));
                            columnsGood &= cosine >= .995 && nrmse <= .05;
                            derivative.WriteLine(string.Join(",", anchor.name,v,lamp+1,F(cosine),F(nrmse)));
                        }

                        Frame Measured(string sample, int[] active, double[] power)
                        {
                            var measured = Expose(anchor, v, floor, active, power);
                            allStable &= measured.state.Signature == full.state.Signature;
                            E23WriteLayout(layout, "olat_audit", anchor.name,v,sample,measured.state,full.state);
                            return measured.rgb;
                        }

                        void Check(string tag, int[] active, double[] power)
                        {
                            var target = Increment(Measured(tag, active, power), empty);
                            int n = target.Length / 3;
                            var signal = new double[n]; var error = new double[n];
                            for (int i = 0; i < target.Length; ++i)
                            {
                                double estimate = 0;
                                for (int j = 0; j < active.Length; ++j) estimate += (power[j] - floor) / (1 - floor) * unit[active[j]][i];
                                double d = target[i] - estimate;
                                signal[i/3] += (double)target[i] * target[i]; error[i/3] += d * d;
                            }
                            double worst = Math.Sqrt(error.Max());
                            double nrmse = Math.Sqrt(error.Sum() / Math.Max(1e-24, signal.Sum()));
                            var hot = Top(signal, .01, 0, n);
                            relative.Add(nrmse);
                            errors.WriteLine(string.Join(",", anchor.name, v, F(floor), tag, F(nrmse), F(worst), F(Math.Sqrt(hot.Sum(i=>error[i])/Math.Max(1e-24,hot.Sum(i=>signal[i]))))));
                        }
                    }
                    gate.median = Median(relative); gate.worst = relative.Max();
                    gate.linearPassed = allStable && gate.median <= .02 && gate.worst <= .05;
                    gate.columnsPassed = gate.linearPassed && columnsGood;
                    gate.stop = !allStable ? "layout_changed_during_audit" : !gate.linearPassed ? "linearity_threshold" : !gate.columnsPassed ? "intensity_columns_threshold" : "audit_passed";
                }
            }
        }
    }
}

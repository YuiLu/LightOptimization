using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            private void E22ShadowStudy(bool recover)
            {
                using var common = new MbdSurfaceGeometry(room.Camera);
                using var commonUnsigned = new MbdSurfaceGeometry(room.Camera, signedMoments: false);
                using var legacy = new ProxyGeometry(room.Camera);
                var sites = common.Samples(smoke ? 512 : 1536, directory);
                var points = sites.Select(s => s.position).ToArray();
                int width = 256, height = (sites.Length * lights.Length + width - 1) / width;
                using var capture = new NativeIrradianceCapture(room.Camera, width, height);
                capture.SetSurfaceSamples(points, sites.Select(s => s.normal).ToArray(), common.Bounds);
                var watch = Stopwatch.StartNew();
                var truth = Native(room.Truth);
                var layout = E22Atlas.Snapshot(room.Camera);
                int[] slices = layout.Item4.Split(';').Select(s => int.TryParse(s.Split(':').Last(), out int n) ? n : 0).Where(n => n > 0).Distinct().ToArray();
                if (slices.Length != 1) throw new InvalidOperationException("E22 calibration needs a known common native point shadow resolution.");
                int nativeSize = slices[0];
                var package = typeof(UniversalRenderer).Assembly.GetType("UnityEngine.Rendering.Universal.Internal.AdditionalLightsShadowCasterPass");
                var guard = package?.GetMethod("GetPointLightShadowFrustumFovBiasInDegrees", BindingFlags.NonPublic | BindingFlags.Static);
                if (guard == null) throw new InvalidOperationException("Cannot resolve the URP 14 point-light guard angle.");
                float nativeGuard = (float)guard.Invoke(null, new object[] { nativeSize, false });
                var pipeline = (UniversalRenderPipelineAsset)GraphicsSettings.currentRenderPipeline;
                using var settings = Csv("shadow-settings.csv", "lamp,method,near,far,receiver_bias_m,caster_bias_m,cube_size,msaa,blur,native_face_size,native_width,native_height,native_slices,native_guard_deg,casters,alpha_clip_slots");
                using var measures = Csv("shadow-calibration.csv", "lamp,pose,axis,signed_step_m,method,visibility_mae,weighted_visibility_mae,direct_nrmse,seconds");
                using var directions = Csv("shadow-direction.csv", "lamp,axis,step_m,method,visibility_cosine,direct_cosine,native_visibility_norm,proxy_visibility_norm");
                using var atlas = Csv("shadow-native-atlas.csv", "lamp,pose,width,height,slices,layout,seconds");
                using var status = Csv("shadow-validity.csv", "lamp,method,faces,invalid,nonfinite,limit,forward_sync_s,pass_submit_s,blur_submit_s");
                atlas.WriteLine(string.Join(",", 0, "gt", layout.Item1, layout.Item2, layout.Item3, layout.Item4, F(watch.Elapsed.TotalSeconds)));
                SaveSurfaceOracle("oracle-native-gt.csv", sites.Length, truth);
                var selected = smoke ? new[] { 0 } : new[] { 0, 4, 5 };
                string[] kinds = { "e21_msm", "e21_mbd", "matched_msm", "matched_mbd" };
                foreach (int lamp in selected)
                {
                    var casterLight = lights[lamp];
                    var maps = new Dictionary<string, PointMomentShadow>();
                    var cache = new Dictionary<string, (Vector3 position, float[] visibility)>();
                    try
                    {
                        foreach (var kind in recover ? kinds.Skip(2) : kinds)
                        {
                            bool mbd = kind.EndsWith("mbd", StringComparison.Ordinal), matched = kind.StartsWith("matched", StringComparison.Ordinal);
                            var cfg = MbdSettings(casterLight);
                            cfg.mbd = mbd;
                            cfg.leakBias = mbd ? 0 : .06f;
                            if (matched)
                            {
                                float nativeDepthBias = casterLight.GetUniversalAdditionalLightData().usePipelineSettings ? pipeline.shadowDepthBias : casterLight.shadowBias;
                                cfg.casterDepthBias = -nativeDepthBias * Mathf.Tan((90 + nativeGuard) * Mathf.Deg2Rad * .5f) * casterLight.range / nativeSize;
                                cfg.depthBiasMetres = 0;
                            }
                            Action<CommandBuffer> draw;
                            if (mbd) draw = common.DrawCasters;
                            else if (matched) draw = commonUnsigned.DrawCasters;
                            else draw = legacy.DrawCasters;
                            maps.Add(kind, new PointMomentShadow(draw, points, cfg));
                            settings.WriteLine(string.Join(",", lamp + 1, kind, F(cfg.near), F(cfg.far), F(cfg.depthBiasMetres), F(cfg.casterDepthBias), cfg.size,
                                cfg.actualMsaa, cfg.blurRadius, nativeSize, layout.Item1, layout.Item2, layout.Item3, F(nativeGuard),
                                mbd || matched ? common.Renderers : -1, mbd || matched ? common.AlphaClipSlots : -1));
                        }
                        if (recover)
                        {
                            var gt = E22Target(true);
                            var active = new[] { lamp };
                            foreach (bool far in new[] { false, true })
                            {
                                var start = E22Start(733, active, far);
                                string scenario = $"seed733_{(far ? "far" : "near")}_L{lamp + 1}";
                                RunE22Arm(scenario, "Native", active, start, gt, NativeSignal(gt), 350, E22Solver("A0"));
                                foreach (var kind in maps.Keys)
                                {
                                    var signal = FieldSignal(sites, active, truth, 1, kind + "_native_hard", normalizeLamp: true,
                                        visibility: (_, position) => Visibility(kind, position));
                                    RunE22Arm(scenario, kind, active, start, gt, signal, 350, E22Solver("A0"), nativeGate: true);
                                }
                            }
                        }
                        else
                        {
                            Analyze("gt", -1, 0, room.Truth, truth);
                            for (int axis = 0; axis < 3; ++axis)
                            {
                                const double h = .02;
                                var a = (double[])room.Truth.Clone(); var b = (double[])room.Truth.Clone();
                                a[4 * lamp + axis] += h / units[axis]; b[4 * lamp + axis] -= h / units[axis];
                                if (!room.Contains(a) || !room.Contains(b)) continue;
                                var plus = At("plus", a, axis, h); var minus = At("minus", b, axis, -h);
                                var vg = Difference(plus["native"].v, minus["native"].v); var dg = Difference(plus["native"].d, minus["native"].d);
                                foreach (var kind in kinds)
                                {
                                    var v = Difference(plus[kind].v, minus[kind].v); var d = Difference(plus[kind].d, minus[kind].d);
                                    directions.WriteLine(string.Join(",", lamp + 1, axis, F(h), kind, F(Cosine(v, vg)), F(Cosine(d, dg)),
                                        F(Math.Sqrt(E21Dot(vg, vg)) / (2 * h)), F(Math.Sqrt(E21Dot(v, v)) / (2 * h))));
                                }
                            }
                        }
                        foreach (var pair in maps)
                        {
                            var map = pair.Value;
                            status.WriteLine(string.Join(",", lamp + 1, pair.Key, map.Faces, map.Degenerate, map.Nonfinite, map.LimitSamples,
                                F(map.ForwardSyncSeconds), F(map.PassSubmitSeconds), F(map.FilterSubmitSeconds)));
                        }
                    }
                    finally { foreach (var map in maps.Values) map.Dispose(); }

                    Dictionary<string, (float[] v, float[] d)> At(string key, double[] rig, int axis, double delta)
                    {
                        var timer = Stopwatch.StartNew(); var native = Native(rig);
                        var check = E22Atlas.Snapshot(room.Camera);
                        atlas.WriteLine(string.Join(",", lamp + 1, key + axis, check.Item1, check.Item2, check.Item3, check.Item4, F(timer.Elapsed.TotalSeconds)));
                        return Analyze(key, axis, delta, rig, native);
                    }

                    Dictionary<string, (float[] v, float[] d)> Analyze(string pose, int axis, double step, double[] rig, Color[] observed)
                    {
                        var position = room.PositionOf(rig, lamp);
                        double power = room.PowerOf(rig, lamp);
                        var native = (v: new float[sites.Length], d: new float[3 * sites.Length]);
                        var clear = new Vector3[sites.Length];
                        var color = ProxyGeometry.Rgb(casterLight.color.linear);
                        for (int i = 0, index = lamp * sites.Length; i < sites.Length; ++i, ++index)
                        {
                            var f = observed[index]; native.v[i] = f.a;
                            native.d[3 * i] = f.r; native.d[3 * i + 1] = f.g; native.d[3 * i + 2] = f.b;
                            clear[i] = color * MbdStudy.IncidentIrradiance(sites[i], position, power, casterLight.range);
                        }
                        double energy = E21Dot(native.d, native.d);
                        var result = new Dictionary<string, (float[] v, float[] d)> { ["native"] = native };
                        foreach (var kind in kinds)
                        {
                            var timer = Stopwatch.StartNew(); var v = Visibility(kind, position); var direct = new float[v.Length * 3];
                            double mae = 0, weighted = 0, weight = 0, err = 0;
                            for (int i = 0; i < v.Length; ++i)
                            {
                                double siteWeight = Norm(clear[i]);
                                var p = v[i] * clear[i];
                                direct[3 * i] = p.x; direct[3 * i + 1] = p.y; direct[3 * i + 2] = p.z;
                                mae += Math.Abs(v[i] - native.v[i]); weighted += siteWeight * Math.Abs(v[i] - native.v[i]); weight += siteWeight;
                                err += Norm(p - new Vector3(native.d[3 * i], native.d[3 * i + 1], native.d[3 * i + 2]));
                            }
                            measures.WriteLine(string.Join(",", lamp + 1, pose, axis, F(step), kind, F(mae / v.Length), F(weighted / Math.Max(1e-12, weight)),
                                F(Math.Sqrt(err / Math.Max(1e-12, energy))), F(timer.Elapsed.TotalSeconds)));
                            result.Add(kind, (v, direct));
                        }
                        return result;
                    }

                    float[] Visibility(string kind, Vector3 position)
                    {
                        if (cache.TryGetValue(kind, out var old) && old.position.Equals(position)) return old.visibility;
                        var map = maps[kind]; int before = map.Faces;
                        var result = map.Render(position);
                        shadowFaces += map.Faces - before;
                        cache[kind] = (position, result);
                        return result;
                    }
                }

                Color[] Native(double[] rig)
                {
                    Apply(rig, true); ++surfaceRenders;
                    return capture.ReadSurfaces(lights);
                }

                static float[] Difference(float[] a, float[] b) => a.Select((x, i) => x - b[i]).ToArray();
                static double Cosine(float[] a, float[] b)
                {
                    double denom = Math.Sqrt(E21Dot(a, a) * E21Dot(b, b));
                    return denom > 0 ? E21Dot(a, b) / denom : double.NaN;
                }
            }
        }
    }
}

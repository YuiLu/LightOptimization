using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            internal void CompareProxies()
            {
                const int size = 128, msaa = 4, blur = 2, rayTaps = 13;
                const float angularRadius = 3.15f / size;
                using var geometry = new MbdSurfaceGeometry(room.Camera);
                using var legacyGeometry = new ProxyGeometry(room.Camera);
                geometry.OpenRayReference();
                var samples = geometry.Samples(smoke ? 512 : 1536, directory);
                var points = samples.Select(s => s.position).ToArray();
                var train = Target(true);
                var off = SurfaceTruth(geometry, samples, false);
                var hard = SurfaceTruth(geometry, samples, true);
                SaveSurfaceOracle("oracle-off.csv", samples.Length, off);
                SaveSurfaceOracle("oracle-native-hard.csv", samples.Length, hard);
                var raySoft = (Color[])off.Clone();
                var selfSoft = (Color[])off.Clone();
                var maps = new Dictionary<(string kind, int lamp), PointMomentShadow>();
                var cache = new Dictionary<(string kind, int lamp), (Vector3 position, float[] values)>();
                int[] selected = smoke ? new[] { 0 } : new[] { 0, 4, 5 };
                var raySettings = selected.ToDictionary(lamp => lamp, lamp => MbdSettings(lights[lamp], size, msaa, blur));
                using var mapConfig = Csv("proxy-config.csv", "method,lamp,cube_size,msaa,blur_radius,ray_taps,angular_filter,near_m,far_m,moment_bias,depth_bias_m,blend_or_leak");
                using var audit = Csv("proxy-direction.csv", "scenario,method,axis,h,stencil,proxy_centre,proxy_plus,proxy_minus,proxy_grad,native_grad,full_gradient_cosine,gt_proxy_loss");
                using var sampler = Csv("proxy-target-fit.csv", "lamp,method,gt_loss");
                using var validity = Csv("visibility-status.csv", "kind,lamp,faces,invalid,nonfinite,root_limits,forward_sync_s");
                try
                {
                    foreach (int lamp in selected)
                    {
                        var gt = room.PositionOf(room.Truth, lamp);
                        var ray = Visibility("ray", lamp, gt);
                        var mbd = Visibility("mbd", lamp, gt);
                        for (int i = 0, k = lamp * samples.Length; i < samples.Length; ++i, ++k)
                        {
                            raySoft[k] = new Color(off[k].r * ray[i], off[k].g * ray[i], off[k].b * ray[i], ray[i]);
                            selfSoft[k] = new Color(off[k].r * mbd[i], off[k].g * mbd[i], off[k].b * mbd[i], mbd[i]);
                        }
                    }
                    SaveSurfaceOracle("oracle-ray-soft.csv", samples.Length, raySoft, selected);
                    SaveSurfaceOracle("oracle-mbd-self-soft.csv", samples.Length, selfSoft, selected);
                    var seeded = room.Start(733);
                    var near = E19Start(room, seeded, .45f);
                    foreach (int lamp in selected)
                    {
                        var active = new[] { lamp };
                        E21Signal Proxy(string name, Color[] oracle, string kind = null) =>
                            FieldSignal(samples, active, oracle, 1, name, normalizeLamp: true,
                                visibility: kind == null ? (Func<int, Vector3, float[]>)null : (i, pos) => Visibility(kind, i, pos));
                        var methods = new[] {
                            ("Native_RGB", NativeSignal(train)),
                            ("MSM_hard", Proxy("msm_to_native_hard", hard, "msm")),
                            ("MBD_hard", Proxy("mbd_to_native_hard", hard, "mbd")),
                            ("Ray_soft", Proxy("ray_soft_to_native_hard", hard, "ray")),
                            ("D0_direct", Proxy("direct_to_unshadowed_oracle", off)),
                            ("MBD_ray_soft", Proxy("mbd_to_matched_ray_soft", raySoft, "mbd")),
                            ("MBD_self_soft", Proxy("mbd_to_exact_model_soft", selfSoft, "mbd"))
                        };
                        foreach (var method in methods)
                            sampler.WriteLine(string.Join(",", lamp + 1, method.Item1, F(method.Item2.objective(room.Truth, null))));
                        foreach (var choice in smoke ? new[] { ("near", near) } : new[] { ("near", near), ("far", seeded) })
                        {
                            string key = $"seed733_{choice.Item1}_L{lamp + 1}";
                            var start = Start(choice.Item2, active);
                            var nativeFd = Probe(measure: x => NativeLoss(train, x, null), lamp, start);
                            foreach (var method in methods)
                            {
                                bool native = method.Item1 == "Native_RGB";
                                var fd = native ? nativeFd : Probe(x => method.Item2.objective(x, null), lamp, start);
                                double cosine = Cos(fd.gradient, nativeFd.gradient);
                                double gtLoss = method.Item2.objective(room.Truth, null);
                                for (int axis = 0; axis < 4; ++axis)
                                    audit.WriteLine(string.Join(",", key, method.Item1, axis, F(axis == 3 ? .04 : .02), fd.stencil[axis],
                                        F(fd.centre), F(fd.plus[axis]), F(fd.minus[axis]), F(fd.gradient[axis]), F(nativeFd.gradient[axis]), F(cosine), F(gtLoss)));
                                RunArm(key, method.Item1, active, start, train, method.Item2, 350, .02, nativeGate: !native);
                            }
                        }
                    }
                    foreach (var pair in maps)
                    {
                        var map = pair.Value;
                        validity.WriteLine(string.Join(",", pair.Key.kind, pair.Key.lamp + 1, map.Faces, map.Degenerate, map.Nonfinite,
                            map.LimitSamples, F(map.ForwardSyncSeconds)));
                    }
                }
                finally { foreach (var map in maps.Values) map.Dispose(); }

                float[] Visibility(string kind, int lamp, Vector3 position)
                {
                    var key = (kind, lamp);
                    if (cache.TryGetValue(key, out var saved) && saved.position.Equals(position)) return saved.values;
                    float[] values;
                    if (kind == "ray")
                    {
                        var cfg = raySettings[lamp];
                        if (!cache.ContainsKey(key))
                            mapConfig.WriteLine(string.Join(",", "radial_ray", lamp + 1, 0, 0, 0, rayTaps, F(angularRadius),
                                F(cfg.near), F(cfg.far), 0, F(cfg.depthBiasMetres), 0));
                        rayQueries += (long)points.Length * rayTaps;
                        values = points.Select(point => geometry.FilteredRadialRayVisibility(point, position, angularRadius, rayTaps,
                            cfg.near, cfg.far, cfg.depthBiasMetres)).ToArray();
                    }
                    else
                    {
                        bool useMbd = kind == "mbd";
                        if (!maps.TryGetValue(key, out var map))
                        {
                            var cfg = MbdSettings(lights[lamp], size, msaa, blur);
                            cfg.mbd = useMbd;
                            cfg.leakBias = useMbd ? 0 : .06f;
                            Action<UnityEngine.Rendering.CommandBuffer> draw = useMbd ?
                                (Action<UnityEngine.Rendering.CommandBuffer>)geometry.DrawCasters : legacyGeometry.DrawCasters;
                            map = new PointMomentShadow(draw, points, cfg);
                            maps.Add(key, map);
                            mapConfig.WriteLine(string.Join(",", kind, lamp + 1, size, cfg.actualMsaa, blur, 0, 0, F(cfg.near), F(cfg.far),
                                F(cfg.momentBias), F(cfg.depthBiasMetres), F(useMbd ? cfg.overestimationWeight : cfg.leakBias)));
                        }
                        int before = map.Faces;
                        values = map.Render(position);
                        shadowFaces += map.Faces - before;
                    }
                    cache[key] = (position, values);
                    return values;
                }
            }

            private (double centre, double[] plus, double[] minus, double[] gradient, int[] stencil) Probe(
                Func<double[], double> measure, int lamp, double[] theta)
            {
                double centre = measure(theta);
                var plus = new double[4]; var minus = new double[4]; var gradient = new double[4]; var stencils = new int[4];
                for (int axis = 0; axis < 4; ++axis)
                {
                    double h = axis == 3 ? .04 : .02;
                    var p = (double[])theta.Clone(); var m = (double[])theta.Clone();
                    p[4 * lamp + axis] += h / units[axis]; m[4 * lamp + axis] -= h / units[axis];
                    bool legalP = room.Contains(p), legalM = room.Contains(m);
                    plus[axis] = legalP ? measure(p) : centre;
                    minus[axis] = legalM ? measure(m) : centre;
                    gradient[axis] = (plus[axis] - minus[axis]) / (legalP && legalM ? 2 * h : h);
                    stencils[axis] = legalP && legalM ? 2 : legalP ? 1 : legalM ? -1 : 0;
                }
                return (centre, plus, minus, gradient, stencils);
            }

            private static double Cos(double[] a, double[] b)
            {
                double denominator = Math.Sqrt(a.Sum(x => x * x) * b.Sum(x => x * x));
                return denominator > 0 ? a.Zip(b, (x, y) => x * y).Sum() / denominator : double.NaN;
            }
        }
    }
}

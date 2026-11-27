using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using Frame = LightingMvp.NativeRenderBackend.Frame;
using Mapping = LightingMvp.NativeRenderBackend.HdrMapping;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            partial void E25Additional(string key, E21StageSummary saved, MbdRgbTarget hard)
            {
                switch (key)
                {
                    case "a-classroom": E25Classroom(saved); break;
                    case "a-views-classroom": E25Views(saved, false); break;
                    case "a-views7-classroom": E25Views(saved, true); break;
                    case "b-complex": E25Complex(saved); break;
                    case "b-diag-complex":
                        var prior = saved.arms.Single(a => a.id == "seed509_complex_far_A-log");
                        E25Optimize(prior.scenario, "A-log-diag", E24AllLamps, prior.initial, Target(false));
                        break;
                    default: E25Engineering(key, saved, hard); break;
                }
            }

            partial void E25Engineering(string key, E21StageSummary saved, MbdRgbTarget hard);

            private void E25Capture(double[] rig, LightShadows mode, int[] views, float[] values, int only = -1)
            {
                Apply(rig, mode == LightShadows.Hard, mode);
                if (only != -1)
                    for (int i = 0; i < lights.Length; ++i) lights[i].enabled = i == only;
                for (int i = 0; i < views.Length; ++i) low.CaptureRgb(views[i], values, i * E23ChannelsPerView);
            }

            private static double E25Scale(float[] linear)
            {
                var positive = linear.Where(v => v > 0 && !float.IsInfinity(v)).ToArray();
                if (positive.Length == 0) throw new InvalidDataException("No positive energy in the E25 target.");
                return Math.Max(1e-8, E24Quantile(positive, (int)Math.Ceiling(.90 * positive.Length) - 1));
            }

            private E21Trial E25L6(E21StageSummary saved) => saved.arms.Single(a => a.id == "seed509_classroom_near_oracle_L6_A-log");

            private void E25Classroom(E21StageSummary saved)
            {
                var prior = E25L6(saved);
                int[] active = { 5 };
                var off = Target(false);
                double scale = E22Scale(off);
                void Mixed(double[] rig, float[] rgb) => E25Capture(rig, LightShadows.None, E25DefaultViews, rgb);
                var gt = new float[E23Channels]; Mixed(room.Truth, gt);
                var bg = new float[E23Channels]; E25Capture(room.Truth, LightShadows.None, E25DefaultViews, bg, -2);
                void Single(double[] rig, float[] rgb)
                {
                    E25Capture(rig, LightShadows.None, E25DefaultViews, rgb, 5);
                    for (int i = 0; i < rgb.Length; ++i) rgb[i] -= bg[i];
                }
                var singleGt = new float[E23Channels]; Single(room.Truth, singleGt);
                double singleScale = E25Scale(singleGt);
                var a0Fit = new E23FrameObjective(gt, Mapping.SignedLog, scale);
                var a1Fit = new E23FrameObjective(singleGt, Mapping.SignedLog, singleScale);
                var rawFit = new E23FrameObjective(gt, Mapping.Raw, scale);
                var rawA1Fit = new E23FrameObjective(singleGt, Mapping.Raw, singleScale);
                using var ap = Csv("observation-audit.csv", "state,view,superposition_relative_rgb,raw_delta_disagreement_relative,mixed_raw_weight,olat_raw_weight,mixed_log_weight,olat_log_weight,raw_gradient_cosine,log_gradient_cosine,mixed_log_gradient,olat_log_gradient");
                using var local = Csv("local-fd.csv", "state,mapping,view,position_step_m,units,rank,condition,singular_values,column_norms,column_correlation,gradient,gradient_to_gt_cosine,actual_steps,stencils,residual_l2,relative_column_norm_20mm,column_cosine_20mm");
                using var hotspot = Csv("hotspots.csv", "state,view,source,x,y,peak_rgb,top1pct_energy,renderer_id,material_id,renderer_name,material_name,opaque_mesh_geometry_boundary,repeat_nrmse");
                using var geom = new MbdSurfaceGeometry(room.Camera);
                geom.OpenRayReference(includeNonCasters: true);
                var sum = (float[])bg.Clone(); var tmp = new float[E23Channels];
                for (int lamp = 0; lamp < room.Count; ++lamp)
                {
                    E25Capture(room.Truth, LightShadows.None, E25DefaultViews, tmp, lamp);
                    for (int i = 0; i < sum.Length; ++i) sum[i] += tmp[i] - bg[i];
                }
                foreach (var (label, rig) in new[] { ("initial", prior.initial), ("e24_failed", prior.final), ("gt", room.Truth) })
                {
                    var log20 = E25Local(rig, active, .02, a0Fit, Mixed);
                    var a1 = E25Local(rig, active, .02, a1Fit, Single);
                    var raw0 = E25Local(rig, active, .02, rawFit, Mixed);
                    var raw1 = E25Local(rig, active, .02, rawA1Fit, Single);
                    for (int view = 0; view < E19FitViews; ++view)
                    {
                        int begin = view * E23ChannelsPerView, length = E23ChannelsPerView;
                        double super = 0, gtRgb = 0, error = 0, delta = 0;
                        for (int i = begin; i < begin + length; ++i)
                        {
                            double d0 = log20.anchor[i] - gt[i], d1 = a1.anchor[i] - singleGt[i];
                            error += (d0 - d1) * (d0 - d1); delta += d0 * d0;
                            double d = sum[i] - gt[i]; super += d * d; gtRgb += (double)gt[i] * gt[i];
                        }
                        var g0 = E25Normal(log20.cols, log20.residual, begin, length).gradient;
                        var g1 = E25Normal(a1.cols, a1.residual, begin, length).gradient;
                        ap.WriteLine(string.Join(",", label, view, F(Math.Sqrt(super / Math.Max(1e-30, gtRgb))), F(Math.Sqrt(error / Math.Max(1e-30, delta))),
                            F(rawFit.weights[view]), F(rawA1Fit.weights[view]), F(a0Fit.weights[view]), F(a1Fit.weights[view]),
                            F(E25Cos(E25Normal(raw0.cols, raw0.residual, begin, length).gradient, E25Normal(raw1.cols, raw1.residual, begin, length).gradient)),
                            F(E25Cos(g0, g1)), Series(g0), Series(g1)));
                    }
                    foreach (double step in new[] { .005, .02, .05 })
                    {
                        var measure = step == .02 ? log20 : E25Local(rig, active, step, a0Fit, Mixed);
                        E25WriteLocal(local, label, "SignedLog", rig, active, step, measure, log20);
                        E25WriteLocal(local, label, "Raw", rig, active, step, step == .02 ? raw0 : E25Local(rig, active, step, rawFit, Mixed), raw0);
                    }
                    E25Hotspots(hotspot, label, rig, off, geom);
                }
                E25PathAndTrials(prior, a0Fit, rawFit, Mixed);
                var isolated = E25Optimize(prior.scenario, "A1-olat", active, prior.initial, off, fit: a1Fit, customCapture: Single);
                var sites = geom.Samples(4096, directory);
                var observed = SurfaceTruth(geom, sites, false);
                SaveSurfaceOracle("oracle-off.csv", sites.Length, observed, active);
                var oracleSignal = FieldSignal(sites, active, observed, 1, "D0_fixed_surfaces", true);
                var solved = E25Optimize(prior.scenario, "A2-D0", active, prior.initial, off, surface: oracleSignal);
                using var gate = Csv("a2-gate.csv", "a1_pass,a2_pass,launched_visible,reason,total_surface_samples,visible_samples,gt_surface_captures,visibility_rays");
                if (solved.pass && !isolated.pass)
                {
                    var camera = room.Camera;
                    var visible = new List<int>();
                    var oldPosition = camera.transform.position; var oldRotation = camera.transform.rotation;
                    var oldAspect = camera.aspect; camera.aspect = (float)E19FitW / E19FitH;
                    long begin = rayQueries;
                    try
                    {
                        for (int i = 0; i < sites.Length; ++i)
                        {
                            var p = sites[i].position;
                            foreach (int v in E25DefaultViews)
                            {
                                camera.transform.SetPositionAndRotation(room.Views[v], room.Rotations[v]);
                                var uv = camera.WorldToViewportPoint(p);
                                if (uv.z <= camera.nearClipPlane || uv.z >= camera.farClipPlane || uv.x <= 0 || uv.x >= 1 || uv.y <= 0 || uv.y >= 1 ||
                                    Vector3.Dot(sites[i].normal, room.Views[v] - p) <= 0) continue;
                                ++rayQueries;
                                if (geom.RayVisibility(p, room.Views[v]) > .5f) { visible.Add(i); break; }
                            }
                        }
                    }
                    finally { camera.transform.SetPositionAndRotation(oldPosition, oldRotation); camera.aspect = oldAspect; }
                    var selected = visible.Select(i => sites[i]).ToArray();
                    var observations = E24AllLamps.SelectMany(l => visible.Select(i => observed[l * sites.Length + i])).ToArray();
                    File.WriteAllLines(Path.Combine(directory, "visible-sample-indices.txt"), visible.Select(i => i.ToString(Invariant)));
                    gate.WriteLine($"0,1,1,triggered,{sites.Length},{selected.Length},1,{rayQueries - begin}");
                    E25Optimize(prior.scenario, "A2-visible", active, prior.initial, off,
                        surface: FieldSignal(selected, active, observations, 1, "D0_visible_opaque_geometry", true));
                }
                else gate.WriteLine($"{Flag(isolated.pass)},{Flag(solved.pass)},0,requires_A2_pass_and_A1_fail,{sites.Length},,1,0");
            }

            private void E25Views(E21StageSummary saved, bool seven)
            {
                // Frozen before viewing any E25 outcomes. Both are independent of views 0–4.
                var eye = new[] { new Vector3(.75f, 1.38f, -3.1f), new Vector3(-3.85f, 1.52f, 1.35f) };
                var look = new[] { new Vector3(-1.35f, 1.15f, .9f), new Vector3(.7f, 1.15f, -1.25f) };
                var rotate = eye.Select((p, i) => Quaternion.LookRotation(look[i] - p, Vector3.up)).ToArray();
                using var held = new NativeRenderBackend(room.Camera, eye, rotate, E19HighW, E19HighH, -1);
                using var fitCam = new NativeRenderBackend(room.Camera, room.Views, room.Rotations, E19FitW, E19FitH, -1);
                using var table = Csv("view-ablation.csv", "arm,comparison,fit_views,checkpoint,camera_renders,models,view,target,raw_nrmse,log_nrmse,display_nrmse,max_position_cm,max_intensity_relative,legal,pass");
                using var config = Csv("held-camera.csv", "id,x,y,z,qx,qy,qz,qw,width,height,fov,near,far");
                for (int i = 0; i < eye.Length; ++i)
                    config.WriteLine(string.Join(",", "held_cam_" + (i + 5), F(eye[i].x), F(eye[i].y), F(eye[i].z), F(rotate[i].x), F(rotate[i].y), F(rotate[i].z), F(rotate[i].w),
                        held.Width, held.Height, F(room.Camera.fieldOfView), F(room.Camera.nearClipPlane), F(room.Camera.farClipPlane)));
                var old = E25L6(saved);
                var source = seven ? E25Read("e_classroom").arms.Single(a => a.scenario == "dir1301_10cm" && a.method == "H0") : old;
                var active = seven ? E24AllLamps : new[] { 5 };
                var off = Target(false);
                var scoreScale = E22Scale(off);
                Apply(room.Truth, false);
                var check = Enumerable.Range(0, 2).Select(held.Capture).ToArray();
                for (int i = 0; i < check.Length; ++i)
                    NativeRenderBackend.Save(check[i], held.Width, held.Height, Path.Combine(CapturePath("gt_off"), "held_cam_" + (i + 5)));
                Frame[] targets = Enumerable.Range(0, room.Views.Length).Select(fitCam.Capture).ToArray();
                Apply(room.Truth, true);
                var checkHard = Enumerable.Range(0, 2).Select(held.Capture).ToArray();
                var records = new List<(E21Trial trial, int[] views, List<(int n, int count, double[] rig)> states)>();
                var views = new[] { new[] { 0 }, new[] { 1 }, new[] { 2 }, E25DefaultViews, new[] { 0, 1, 2, 3, 4 } };
                foreach (int[] group in seven ? views.Skip(3) : views.AsEnumerable())
                {
                    var trace = new List<(int n, int count, double[] rig)> { (0, 0, source.initial) };
                    string id = source.scenario + (group.Length == 1 ? "_single_v" + group[0] : "_V" + group.Length);
                    var fit = new E23FrameObjective(group.Select(i => targets[i]).ToArray(), Mapping.SignedLog, scoreScale);
                    var trial = E25Optimize(id, "A-log", active, source.initial, off, group, fitCam, fit,
                        acceptedObserver: (t, rig) => trace.Add((t.Iteration, t.Query * group.Length, (double[])rig.Clone())));
                    records.Add((trial, group, trace));
                }
                int common = records.Min(a => a.trial.jacobians);
                foreach (var (trial, group, trace) in records)
                {
                    Score(trial, group, "equal_camera_cap", trial.evaluations * group.Length, trial.jacobians, trial.final);
                    var stop = trace.Last(p => p.n <= common);
                    Score(trial, group, "equal_completed_model_cap", stop.count, common, stop.rig);
                }
                var v3 = records.Single(a => a.views.Length == 3).trial;
                var v5 = records.Single(a => a.views.Length == 5).trial;
                bool improve = !v3.pass && (v5.pass || (v5.positionCm.Max() <= .5 * v3.positionCm.Max() && v5.intensityRelative.Max() <= .5 * v3.intensityRelative.Max()));
                using var gate = Csv("view-gate.csv", "v3_pass,v5_pass,v3_max_cm,v5_max_cm,v3_max_intensity,v5_max_intensity,expand_seven_lamps,rule,independent_verification_cameras");
                gate.WriteLine(string.Join(",", Flag(v3.pass), Flag(v5.pass), F(v3.positionCm.Max()), F(v5.positionCm.Max()), F(v3.intensityRelative.Max()),
                    F(v5.intensityRelative.Max()), Flag(!seven && improve), "V5 passes or halves both max errors while V3 fails", held.RenderCount));

                void Score(E21Trial trial, int[] group, string phase, int used, int models, double[] rig)
                {
                    Apply(rig, false);
                    var pos = active.Max(i => 100 * Vector3.Distance(room.PositionOf(rig, i), room.PositionOf(room.Truth, i)));
                    var pow = active.Max(i => Math.Abs(room.PowerOf(rig, i) / room.PowerOf(room.Truth, i) - 1));
                    bool legal = room.Contains(rig);
                    for (int v = 0; v < check.Length; ++v)
                    {
                        var f = held.Capture(v);
                        Row("held_cam_" + (5 + v), f, check[v]);
                        if (phase == "equal_camera_cap" && group.Length >= 3)
                            NativeRenderBackend.Save(f, held.Width, held.Height, Path.Combine(CapturePath(trial.id), "held_cam_" + (v + 5)));
                    }
                    foreach (int v in group) Row("train_" + v, fitCam.Capture(v), targets[v]);
                    Apply(rig, true);
                    for (int v = 0; v < checkHard.Length; ++v) Row("held_cam_" + (5 + v), held.Capture(v), checkHard[v], true);
                    void Row(string label, Frame f, Frame t, bool shadows = false)
                    {
                        var vals = E25FrameScores(f, t, shadows ? e24HardScale : scoreScale);
                        table.WriteLine(string.Join(",", trial.id, phase, string.Join(";", group), "final", used, models, label, shadows ? "hard" : "off",
                            F(vals[0]), F(vals[1]), F(vals[2]), F(pos), F(pow), Flag(legal), Flag(legal && pos <= 5 && pow <= .05)));
                    }
                }
            }

            private void E25Complex(E21StageSummary saved)
            {
                var prior = saved.arms.Single(a => a.id == "seed509_complex_far_A-log");
                var off = Target(false);
                var all = E24AllLamps;
                var fit = new E23FrameObjective(off.fit, Mapping.SignedLog, E22Scale(off));
                var model = E25Local(prior.initial, all, .02, fit, (r, rgb) => E25Capture(r, LightShadows.None, E25DefaultViews, rgb));
                var norm = E25Normal(model.cols, model.residual, 0, fit.Size);
                var pairs = new List<(int a, int b, double cross)>();
                using var pairLog = Csv("cross-lamp.csv", "lamp_a,lamp_b,normalized_gram_frobenius,raw_gram_frobenius,max_abs_column_cosine,selected");
                for (int a = 0; a < all.Length; ++a)
                for (int b = a + 1; b < all.Length; ++b)
                {
                    double aa = 0, bb = 0, ab = 0, cc = 0;
                    for (int j = 0; j < 4; ++j)
                    for (int k = 0; k < 4; ++k)
                    {
                        int x = 4 * a + j, y = 4 * b + k;
                        double[] v = { norm.gram[x, 4 * a + k], norm.gram[y, 4 * b + j], norm.gram[x, y] };
                        aa += v[0] * v[0]; bb += v[1] * v[1]; ab += v[2] * v[2];
                        cc = Math.Max(cc, Math.Abs(v[2]) / Math.Sqrt(Math.Max(1e-60, norm.gram[x,x] * norm.gram[y,y])));
                    }
                    double cross = Math.Sqrt(ab / Math.Sqrt(Math.Max(1e-60, aa * bb)));
                    pairs.Add((a, b, cross));
                    pairLog.WriteLine(string.Join(",", a + 1, b + 1, F(cross), F(Math.Sqrt(ab)), F(cc), "see_gate"));
                }
                var ranked = pairs.OrderBy(p => p.cross).ThenBy(p => p.a).ThenBy(p => p.b).ToArray();
                using var local = Csv("local-fd.csv", "state,mapping,view,position_step_m,units,rank,condition,singular_values,column_norms,column_correlation,gradient,gradient_to_gt_cosine,actual_steps,stencils,residual_l2,relative_column_norm_20mm,column_cosine_20mm");
                E25WriteLocal(local, "far_joint_start", "SignedLog", prior.initial, all, .02, model, model);
                int successes = 0;
                using var lamps = Csv("complex-lamps.csv", "lamp,source,opt_camera,start_cm,final_cm,final_intensity_relative,pass,stop,theta_initial,theta_final");
                foreach (int i in all)
                {
                    var trial = saved.arms.FirstOrDefault(a => a.id == $"{prior.scenario}_oracle_L{i + 1}_A-log");
                    bool reused = trial != null;
                    trial ??= E25Optimize(prior.scenario + "_oracle_L" + (i + 1), "A-log", new[] { i }, Start(prior.initial, new[] { i }), off);
                    if (trial.pass) ++successes;
                    lamps.WriteLine(string.Join(",", i + 1, reused ? "E24" : "E25", trial.evaluations * 3,
                        F(100 * Vector3.Distance(room.PositionOf(trial.initial, i), room.PositionOf(room.Truth, i))), F(trial.positionCm[0]),
                        F(trial.intensityRelative[0]), Flag(trial.pass), trial.stop, Series(trial.initial), Series(trial.final)));
                }
                using var gate = Csv("b-gate.csv", "passed,launched_pairs,lowest,highest,diagonal_damping");
                string Pair((int a, int b, double cross) t) => $"L{t.a + 1}+L{t.b + 1}";
                gate.WriteLine(string.Join(",", successes, Flag(successes >= 4), Pair(ranked.First()), Pair(ranked.Last()), "require_observed_scale_mechanism"));
                if (successes < 4) return;
                foreach (var (name, pair) in new[] { ("lowest", ranked.First()), ("highest", ranked.Last()) })
                {
                    int[] active = { pair.a, pair.b };
                    E25Optimize(prior.scenario + "_" + name + "_" + Pair(pair), "A-log", active, Start(prior.initial, active), off);
                }
            }
        }
    }
}

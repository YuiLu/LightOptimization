using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;
using Frame = LightingMvp.NativeRenderBackend.Frame;
using Mapping = LightingMvp.NativeRenderBackend.HdrMapping;
using Lm = LightingMvp.ImageJacobianLm;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private static void RunE26Stages(string task, string output)
        {
            if (!Application.isBatchMode) throw new InvalidOperationException("E26 is batch only.");
            string key = task.Substring(4);
            bool classroom = key.Contains("classroom");
            using var pipeline = new TestPipeline(classroom: true);
            using ISensitivityScene scene = classroom ? new ClassroomSensitivityScene(pipeline.Asset, null) : new ComplexRoomSensitivityScene();
            using var study = new E21RoomStudy(scene, task, Path.Combine(output, task.Replace('-', '_')),
                classroom ? ClassroomAssets + "/room.bytes" : NativeComplexRoomBuilder.ScenePath);
            study.RunE26(key, classroom);
            study.Finish();
        }

        private sealed partial class E21RoomStudy
        {
            private static readonly int[] E26L6 = { 5 };
            private StreamWriter e26Proposals, e26Constraints, e26Trace, e26Models, e26Pass;

            internal void RunE26(string key, bool classroom)
            {
                e25Inputs = Arg("-bbE25Inputs", "");
                E25Read(classroom ? "a_classroom" : "a_complex");
                if (key.StartsWith("d-", StringComparison.Ordinal)) { E26Color(key); return; }
                var hard = Target(true);
                e24HardScale = E22Scale(hard);
                using (e25Costs = Csv("profiling.csv", "arm,fit_views,cap_camera,opt_camera,hard_camera,off_camera,other_camera,shadow_pass_cameras,shadow_face_slices,render_s,readback_s,process_cpu_s,opt_wall_s,gc0,gc1,gc2,main_thread_allocated_bytes,gpu_s,scale,scene_sha,map_fd_s,mask_p95_s,gram_gradient_s,damped_solve_s,gpu_columns,gpu_upload_s,gpu_submit_s,gpu_sync_readback_s"))
                using (e25Scores = Csv("view-metrics.csv", "arm,checkpoint,target,width,view,raw_nrmse,log_nrmse,display_nrmse"))
                using (e25Queries = Csv("query-log.csv", "arm,query,actual_camera,phase,shadows,axis,physical_step,cameras,theta"))
                using (e25Checkpoints = Csv("checkpoints.csv", "arm,checkpoint,actual_camera,completed_models,loss,max_position_cm,max_intensity_relative,legal,pass,theta"))
                using (e25Visibility = Csv("mask-stats.csv", E24VisibilityHeader))
                using (e26Proposals = Csv("proposal-audit.csv", "arm,model,actual_camera,source,status,legal,accepted,predicted,actual,rho,backtracks,loss_before,delta_normalized,theta"))
                using (e26Constraints = Csv("constraint-backtrack.csv", "arm,model,actual_camera,lamp,proposed_xyz_m,radius_xyz_m,actual_xyz_m,proposed_logI,radius_logI,actual_logI,lamp_backtracks,rig_backtracks,constraint,status,predicted,actual"))
                using (e26Trace = Csv("nonmonotone-trace.csv", "arm,model,actual_camera,source,status,current,reference,best,predicted,actual,rho_current,rho_reference,accepted"))
                using (e26Models = Csv("model-checkpoints.csv", "arm,model,actual_camera,best_loss,max_cm,max_intensity_relative,legal,pass,theta"))
                using (e26Pass = Csv("first-pass.csv", "arm,cameras,model,elapsed_s,max_cm,max_intensity_relative"))
                {
                    if (key == "a-diagnostic-classroom") E26ViewDiagnostic();
                    else if (key is "a-pool-classroom" or "a-pairs-classroom")
                    {
                        var v3 = E26Archive("a-views-classroom").arms.Single(a => a.scenario.EndsWith("_V3", StringComparison.Ordinal));
                        if (key == "a-pool-classroom") E26Arm("L6_seed509_A0", "A-log", E26L6, v3.initial, Target(false));
                        E26Arm("L6_seed509_" + (key == "a-pool-classroom" ? "A1" : "A2"), "A-log", E26L6, v3.initial, Target(false),
                            fallback: key == "a-pool-classroom" ? Lm.TrustFallback.ViewPool : Lm.TrustFallback.ViewPairPool);
                    }
                    else if (key.StartsWith("a-expand-", StringComparison.Ordinal) || key.StartsWith("b-expand-", StringComparison.Ordinal))
                    {
                        string seed = key.Split('-').Last();
                        var source = E26ArchivedCase(classroom, seed);
                        var pool = Arg("-bbE26Pool", "single") == "pair" ? Lm.TrustFallback.ViewPairPool : Lm.TrustFallback.ViewPool;
                        bool usePool = key.StartsWith("a-expand-", StringComparison.Ordinal);
                        foreach (bool shadow in ShadowModes())
                        {
                            string method = shadow ? "C3" : "A-log";
                            E26Arm(source.scenario + "_control", method, E24AllLamps, source.initial, Target(shadow));
                            E26Arm(source.scenario + "_treatment", method, E24AllLamps, source.initial, Target(shadow),
                                fallback: usePool ? pool : Lm.TrustFallback.None, local: !usePool);
                        }
                        bool[] ShadowModes() => classroom && usePool ? new[] { false, true } : new[] { !classroom };
                    }
                    else if (key == "b-backtrack-complex")
                    {
                        var initial = E26ArchivedCase(false, "509").initial;
                        E26Arm("Far_seed509_B0", "A-log", E24AllLamps, initial, Target(false), clip: true);
                        E26Arm("Far_seed509_B1", "A-log", E24AllLamps, initial, Target(false), local: true, clip: true);
                    }
                    else if (key is "b-fd-complex" or "b-fd-classroom") E26FixedB(classroom);
                    else if (key.StartsWith("s-", StringComparison.Ordinal))
                    {
                        bool l6 = key == "s-l6-classroom";
                        var source = l6 ? E26Archive("a-views-classroom").arms.Single(a => a.scenario.EndsWith("_V3", StringComparison.Ordinal))
                            : E26ArchivedCase(classroom, classroom ? "1301" : "509");
                        var start = l6 ? source.final : source.initial;
                        var active = l6 ? E26L6 : E24AllLamps;
                        bool shadows = classroom && !l6;
                        var target = Target(shadows); string method = shadows ? "C3" : "A-log";
                        E26Arm(source.scenario + "_S0", method, active, start, target);
                        E26Arm(source.scenario + "_S0fixed", method, active, start, target, fixedStep: true);
                        E26Arm(source.scenario + "_S1", method, active, start, target, fallback: Lm.TrustFallback.Cauchy, fixedStep: true);
                        E26Arm(source.scenario + "_S2", method, active, start, target, fallback: Lm.TrustFallback.Nonmonotone, fixedStep: true);
                    }
                    else if (key == "c-numeric-complex") { E25GpuTransfer(hard); E26GpuNumeric(); }
                    else if (key.StartsWith("c-columns-", StringComparison.Ordinal)) E26CpuColumns(classroom, hard);
                    else if (key.StartsWith("c-replay-", StringComparison.Ordinal))
                    {
                        var source = E26ArchivedCase(classroom, key.Split('-').Last());
                        E26Arm(source.scenario + "_legacy-layout", "C3", E24AllLamps, source.initial, hard, explicitLayout: false, save: false);
                        E26Arm(source.scenario + "_explicit-layout", "C3", E24AllLamps, source.initial, hard, save: !classroom);
                    }
                    else if (key == "c-gpu-complex")
                    {
                        var source = E26Archive("d-gpu-recovery-131-complex").arms.Single(a => a.scenario.EndsWith("_rep1", StringComparison.Ordinal));
                        foreach (string method in new[] { "C3-cache", "C3-gpu" })
                            E26Arm("seed131_warmup", method, E24AllLamps, source.initial, hard, iterations: 1, save: false);
                        for (int r = 1; r <= 3; ++r)
                            foreach (string method in (r == 2 ? new[] { "C3-gpu", "C3-cache" } : new[] { "C3-cache", "C3-gpu" }))
                                E26Arm("seed131_rep" + r, method, E24AllLamps, source.initial, hard, save: r == 1);
                    }
                    else throw new ArgumentException("Unknown E26 stage " + key);
                }
            }

            partial void E26Color(string key);

            private E21StageSummary E26Archive(string key) => E25Read("e25-" + key);
            private E21Trial E26ArchivedCase(bool classroom, string seed) => E26Archive($"c-{(classroom ? "classroom" : "complex")}-{seed}").arms.Single(a => a.method == "C3");

            private Lm.TrustAxis[] E26Axes(double step = .02)
            {
                var min = room.PositionOf(new double[4], 0);
                return Enumerable.Range(0, 4).Select(k => new Lm.TrustAxis((Lm.TrustCoordinate)k, units[k],
                    k == 3 ? .04 : step, k == 3 ? .28 : .24, k == 3 ? .75 : .70, k == 3 ? Math.Log(PowerMin) : min[k])).ToArray();
            }

            private E21Trial E26Arm(string scenario, string method, int[] active, double[] initial, MbdRgbTarget target, int[] views = null,
                Lm.TrustFallback fallback = Lm.TrustFallback.None, bool local = false, bool fixedStep = false, bool clip = false,
                bool explicitLayout = true, int iterations = int.MaxValue, bool save = true, Action<int, float[][]> columnAudit = null)
            {
                views ??= E25DefaultViews;
                int count = views.Length;
                var fit = new E23FrameObjective(views.Select(v => target.fit[v]).ToArray(), Mapping.SignedLog, E22Scale(target));
                string id = scenario + "_" + method;
                var watch = new System.Diagnostics.Stopwatch();
                bool seenPass = false;
                double best = double.PositiveInfinity;
                double[] bestRig = initial;
                var replay = new List<(int model, int cameras, double loss, double[] rig)>();
                var arm = E25Optimize(scenario, method, active, initial, target, views: views, fit: fit, iterations: iterations, saveImages: save,
                    modelObserver: j =>
                    {
                        if (j.Iteration is 4 or 8 or 16) replay.Add((j.Iteration, j.Query * count, best, (double[])bestRig.Clone()));
                    },
                    configureRun: (settings, expand) =>
                    {
                        watch.Start();
                        if (explicitLayout) settings.Layout = new Lm.TrustLayout(active.Select(_ => E26Axes()).ToArray());
                        settings.Fallback = fallback;
                        settings.AuditColumns = columnAudit;
                        settings.FixedProbeScale = fixedStep;
                        settings.TraceProposalClipping = clip;
                        if (fallback is Lm.TrustFallback.ViewPool or Lm.TrustFallback.ViewPairPool)
                            settings.ViewBlocks = Enumerable.Repeat(E23ChannelsPerView, count).ToArray();
                        if (local) settings.LampPositionAllowed = (x, j) => room.Contains(expand(x).Skip(4 * active[j]).Take(4).ToArray());
                        var beforeQuery = settings.Query;
                        settings.Query = (q, phase, x, loss, axis, offset) =>
                        {
                            beforeQuery(q, phase, x, loss, axis, offset);
                            if (phase == "initial") { best = loss; Check(x, q, 0); }
                        };
                        var prior = settings.Trial;
                        settings.Trial = t =>
                        {
                            prior(t);
                            var rig = expand(t.Theta);
                            e26Proposals.WriteLine(string.Join(",", id, t.Iteration, t.Query * count, t.Source, t.Status, Flag(t.Legal), Flag(t.Accepted), F(t.Predicted), F(t.Actual), F(t.Ratio), t.Shrinks, F(t.CurrentLoss), Series(t.Step), Series(rig)));
                            if (fallback == Lm.TrustFallback.Nonmonotone && t.Source == "nonmonotone")
                                e26Trace.WriteLine(string.Join(",", id, t.Iteration, t.Query * count, t.Source, t.Status, F(t.CurrentLoss), F(t.ReferenceLoss), F(t.BestLoss), F(t.Predicted), F(t.Actual), F(t.Ratio), F(t.ReferenceRatio), Flag(t.Accepted)));
                            if (t.Accepted)
                            {
                                if (t.CurrentLoss - t.Actual < best) { best = t.CurrentLoss - t.Actual; bestRig = (double[])rig.Clone(); }
                                Check(t.Theta, t.Query, t.Iteration);
                            }
                            if (clip && t.UnclippedStep != null)
                                for (int j = 0; j < active.Length; ++j)
                                {
                                    int k = 4 * j;
                                    double XYZ(double[] d) => L2(Enumerable.Range(0, 3).Select(i => d[k + i] * units[i]));
                                    e26Constraints.WriteLine(string.Join(",", id, t.Iteration, t.Query * count, active[j] + 1,
                                        F(XYZ(t.UnclippedStep)), F(XYZ(t.RadiusLimitedStep)), F(XYZ(t.Step)), F(t.UnclippedStep[k + 3] * units[3]),
                                        F(t.RadiusLimitedStep[k + 3] * units[3]), F(t.Step[k + 3] * units[3]), t.LampShrinks?[j] ?? 0, t.Shrinks, t.Constraint, t.Status, F(t.Predicted), F(t.Actual)));
                                }
                        };
                        // Offline scorer: never changes the solver or terminates a run.
                        void Check(double[] x, int q, int model)
                        {
                            if (seenPass) return;
                            var rig = expand(x);
                            var (cm, rel, legal) = E26Errors(active, rig);
                            if (!legal || cm > 5 || rel > .05) return;
                            seenPass = true;
                            e26Pass.WriteLine(string.Join(",", id, q * count, model, F(watch.Elapsed.TotalSeconds), F(cm), F(rel)));
                        }
                    });
                foreach (var p in replay)
                {
                    var (cm, rel, legal) = E26Errors(active, p.rig);
                    e26Models.WriteLine(string.Join(",", id, p.model, p.cameras, F(p.loss), F(cm), F(rel), Flag(legal), Flag(legal && cm <= 5 && rel <= .05), Series(p.rig)));
                }
                return arm;
            }

            private (double cm, double rel, bool legal) E26Errors(int[] active, double[] rig) =>
                (active.Max(i => 100d * Vector3.Distance(room.PositionOf(rig, i), room.PositionOf(room.Truth, i))),
                 active.Max(i => Math.Abs(room.PowerOf(rig, i) / room.PowerOf(room.Truth, i) - 1)), room.Contains(rig));

            private void E26ViewDiagnostic()
            {
                var archive = E26Archive("a-views-classroom");
                var v3 = archive.arms.Single(a => a.scenario.EndsWith("_V3", StringComparison.Ordinal));
                var off = Target(false);
                var fit = new E23FrameObjective(off.fit, Mapping.SignedLog, E22Scale(off));
                using var gradient = Csv("view-gradient.csv", "state,position_fd_m,subset,loss,rank,singular_normalized,column_norms_normalized,gradient_normalized,gram_normalized,rank_physical,singular_physical,column_norms_physical,gradient_physical,cancel_normalized,cancel_physical,cos_normalized_v01_v02_v12,cos_physical_v01_v02_v12,sum_gram_relative,sum_gradient_relative");
                using var steps = Csv("fixed-proposals.csv", "state,position_fd_m,source,scale,legal,backtracks,predicted,actual,view_before,view_after,view_change,delta_phys,diag_cameras,theta");
                int diagBegin = low.RenderCount;
                var layout = new Lm.TrustLayout(E26Axes());
                foreach (var (name, rig) in new[] { ("original", v3.initial), ("v3_failed", v3.final), ("v2_endpoint_oracle", archive.arms.Single(a => a.scenario.EndsWith("_single_v2", StringComparison.Ordinal)).final), ("gt_oracle", room.Truth) })
                foreach (double step in name == "v3_failed" ? new[] { .005, .02, .05 } : new[] { .02 })
                {
                    var m = E25Local(rig, E26L6, step, fit, Capture);
                    var normals = Enumerable.Range(0, 3).Select(v => E25Normal(m.cols, m.residual, v * E23ChannelsPerView, E23ChannelsPerView)).ToArray();
                    var full = E25Normal(m.cols, m.residual, 0, fit.Size);
                    var normalized = normals.Select(x => x.gradient).ToArray();
                    var phys = normalized.Select(x => x.Select((g, i) => g / units[i]).ToArray()).ToArray();
                    var initialLosses = ViewLoss(m.residual);
                    var theta = rig.Skip(20).Take(4).ToArray();
                    var buffer = new float[fit.Size]; var rgb = new float[fit.Size];
                    var sumH = normals.Select(n => n.gram.Cast<double>().ToArray()).ToArray();
                    double gramError = L2(full.gram.Cast<double>().Select((x, i) => x - sumH.Sum(h => h[i]))) / Math.Max(1e-30, L2(full.gram.Cast<double>()));
                    double gError = L2(full.gradient.Select((x, i) => x - normalized.Sum(g => g[i]))) / Math.Max(1e-30, L2(full.gradient));
                    if (gramError > 1e-8 || gError > 1e-8) throw new InvalidDataException("View normals did not reconstruct the full model.");
                    foreach (int subset in new[] { 7, 1, 2, 4, 3, 5, 6, 0 })
                    {
                        var g = new double[4]; var h = new double[4, 4]; double loss = 0;
                        for (int v = 0; v < 3; ++v)
                            if (((subset == 0 ? 7 : subset) & (1 << v)) != 0)
                            {
                                loss += initialLosses[v];
                                for (int i = 0; i < 4; ++i)
                                {
                                    g[i] += normals[v].gradient[i];
                                    for (int j = 0; j < 4; ++j) h[i, j] += normals[v].gram[i, j];
                                }
                            }
                        if (subset is 7 or 1 or 2 or 4)
                        {
                            var hp = new double[4, 4];
                            for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) hp[i, j] = h[i, j] / (units[i] * units[j]);
                            var sn = ClassroomDiagnosis.GetSpectrum(h, 4); var sp = ClassroomDiagnosis.GetSpectrum(hp, 4);
                            gradient.WriteLine(string.Join(",", name, F(step), subset, F(loss), sn.rank, Series(sn.singular), Series(Enumerable.Range(0, 4).Select(i => Math.Sqrt(h[i, i]))), Series(g), Series(h.Cast<double>()),
                                sp.rank, Series(sp.singular), Series(Enumerable.Range(0, 4).Select(i => Math.Sqrt(hp[i, i]))), Series(g.Select((x, i) => x / units[i])),
                                F(Cancel(normalized)), F(Cancel(phys)), Series(Cos(normalized)), Series(Cos(phys)), F(gramError), F(gError)));
                        }
                        foreach (double fraction in new[] { 1d, .5, .25 })
                        {
                            int captureStart = low.RenderCount;
                            var delta = Lm.LampScaledTrustProposal(h, g, theta, layout, x => room.Contains(Expand(x)), out int shrinks, fraction, cauchy: subset == 0);
                            var trial = Expand(theta.Zip(delta, (x, d) => x + d).ToArray());
                            bool legal = room.Contains(trial);
                            double predict = Lm.TrustPredicted(full.gram, full.gradient, delta);
                            double[] after = Enumerable.Repeat(double.NaN, 3).ToArray();
                            if (legal) { Capture(trial, rgb); fit.Evaluate(rgb, buffer); after = ViewLoss(buffer); }
                            steps.WriteLine(string.Join(",", name, F(step), subset == 0 ? "cauchy" : "views_" + subset, F(fraction), Flag(legal), shrinks, F(predict), F(initialLosses.Sum() - after.Sum()),
                                Series(initialLosses), Series(after), Series(initialLosses.Zip(after, (a, b) => a - b)), Series(delta.Select((d, i) => d * units[i])), low.RenderCount - captureStart, Series(trial)));
                        }
                    }
                    double[] Expand(double[] x) { var copy = (double[])rig.Clone(); Array.Copy(x, 0, copy, 20, 4); return copy; }
                }
                File.WriteAllText(Path.Combine(directory, "diagnostic-cameras.txt"), (low.RenderCount - diagBegin).ToString(Invariant));
                foreach (var pair in new[] { new[] { 0, 1 }, new[] { 0, 2 }, new[] { 1, 2 } })
                    E26Arm("L6_seed509_V" + string.Join("", pair), "A-log", E26L6, v3.initial, off, views: pair, save: false);
                void Capture(double[] x, float[] r) => E25Capture(x, LightShadows.None, E25DefaultViews, r);
                static double Cancel(double[][] gs) => L2(Enumerable.Range(0, 4).Select(i => gs.Sum(a => a[i]))) / Math.Max(1e-30, gs.Sum(L2));
                static double[] Cos(double[][] gs) => new[] { E25Cos(gs[0], gs[1]), E25Cos(gs[0], gs[2]), E25Cos(gs[1], gs[2]) };
                static double[] ViewLoss(float[] r) => Enumerable.Range(0, 3).Select(v => E21Dot(r, r, v * E23ChannelsPerView, (v + 1) * E23ChannelsPerView)).ToArray();
            }

            private void E26CpuColumns(bool classroom, MbdRgbTarget hard)
            {
                using var csv = Csv("cpu-columns.csv", "arm,seed,model,column,relative_l2,absolute_l2,max_absolute,threshold,pass");
                var seeds = classroom ? new[] { "1301" } : new[] { "911", "1201", "509" };
                var cases = seeds.Select(seed => (seed, prior: E26ArchivedCase(classroom, seed))).ToArray();
                foreach (var (seed, prior) in cases)
                {
                    float[][] first = null;
                    E26Arm(prior.scenario + "_columns_implicit", "C3", E24AllLamps, prior.initial, hard, explicitLayout: false, iterations: 1, save: false,
                        columnAudit: (_, cols) => first = cols.Select(col => (float[])col.Clone()).ToArray());
                    E26Arm(prior.scenario + "_columns_explicit", "C3", E24AllLamps, prior.initial, hard, iterations: 1, save: false,
                        columnAudit: (model, cols) =>
                        {
                            for (int j = 0; j < cols.Length; ++j)
                            {
                                double error = 0, norm = 0, maximum = 0;
                                for (int i = 0; i < cols[j].Length; ++i)
                                {
                                    double a = first[j][i], delta = a - cols[j][i];
                                    error += delta * delta; norm += a * a; maximum = Math.Max(maximum, Math.Abs(delta));
                                }
                                double relative = Math.Sqrt(error / Math.Max(norm, 1e-24));
                                csv.WriteLine(string.Join(",", prior.id, seed, model, j, F(relative), F(Math.Sqrt(error)), F(maximum), "1e-6", Flag(relative <= 1e-6)));
                            }
                        });
                }
            }

            private void E26FixedB(bool classroom)
            {
                var prior = classroom ? E26ArchivedCase(true, "1301") : E25Read("a_complex").arms.Single(a => a.id == "seed509_complex_far_A-log");
                var target = Target(classroom);
                var fit = new E23FrameObjective(target.fit, Mapping.SignedLog, E22Scale(target));
                using var csv = Csv("stall-fd.csv", "arm,seed,state,method,xyz_fd_m,gradient_norm,gradient_cos_to_20mm,effective_rank,source,step_scale,legal,backtracks,predicted,actual,view_before,view_after,diag_cameras,theta");
                int begin = low.RenderCount;
                var rig = prior.final;
                if (!room.Contains(rig)) throw new InvalidDataException("Frozen B2 failure must be legal.");
                var layout = new Lm.TrustLayout(E24AllLamps.Select(_ => E26Axes()).ToArray());
                double[] baseline = null;
                var rgb = new float[fit.Size]; var residual = new float[fit.Size];
                foreach (double fd in new[] { .02, .005, .05 })
                {
                    var m = E25Local(rig, E24AllLamps, fd, fit, Capture);
                    var (h, g) = E25Normal(m.cols, m.residual, 0, fit.Size);
                    baseline ??= g;
                    var spectrum = ClassroomDiagnosis.GetSpectrum(h, rig.Length);
                    var before = ViewLoss(m.residual);
                    foreach (bool cauchy in new[] { false, true })
                    foreach (double fraction in new[] { 1d, .5, .25 })
                    {
                        int current = low.RenderCount;
                        var delta = Lm.LampScaledTrustProposal(h, g, rig, layout, room.Contains, out int shrinks, fraction, cauchy);
                        var moved = rig.Zip(delta, (x, d) => x + d).ToArray();
                        bool legal = room.Contains(moved);
                        double[] after = Enumerable.Repeat(double.NaN, 3).ToArray();
                        if (legal) { Capture(moved, rgb); fit.Evaluate(rgb, residual); after = ViewLoss(residual); }
                        csv.WriteLine(string.Join(",", prior.id, classroom ? 1301 : 509, "archived_failed_endpoint", classroom ? "H1" : "A-log", F(fd), F(L2(g)),
                            F(E25Cos(g, baseline)), spectrum.rank, cauchy ? "cauchy" : "full", F(fraction), Flag(legal), shrinks,
                            F(Lm.TrustPredicted(h, g, delta)), F(before.Sum() - after.Sum()), Series(before), Series(after), low.RenderCount - current, Series(moved)));
                    }
                }
                File.WriteAllText(Path.Combine(directory, "diagnostic-cameras.txt"), (low.RenderCount - begin).ToString(Invariant));
                void Capture(double[] x, float[] r) => E25Capture(x, classroom ? LightShadows.Hard : LightShadows.None, E25DefaultViews, r);
                static double[] ViewLoss(float[] r) => Enumerable.Range(0, 3).Select(v => E21Dot(r, r, v * E23ChannelsPerView, (v + 1) * E23ChannelsPerView)).ToArray();
            }

            private void E26GpuNumeric()
            {
                using var csv = Csv("gpu-fixed-stencils.csv", "views,stencil,quantity,coordinate,relative_l2,absolute_l2,step_cosine,pass");
                foreach (int views in new[] { 1, 2, 3, 5 })
                {
                    int n = views * E23ChannelsPerView;
                    float[] hard = new float[n], off = new float[n], p = new float[n], m = new float[n], residual = new float[n];
                    for (int i = 0; i < n; ++i)
                    {
                        hard[i] = (float)((i % 29 - 14) * (i % 17 == 0 ? 17d : .025));
                        off[i] = (float)((i % 19 - 9) * .073); residual[i] = (float)((i % 33 - 16) * 1e-5);
                    }
                    var fit = new E23FrameObjective(hard, Mapping.SignedLog, .1);
                    using var gpu = new E25GpuColumns(fit);
                    gpu.Anchor(hard, off);
                    foreach (int stencil in new[] { 1, 2, 3 })
                    {
                        var cpu = new float[4][]; var device = new float[4][];
                        for (int j = 0; j < 4; ++j)
                        {
                            for (int i = 0; i < n; ++i)
                            {
                                p[i] = off[i] + (float)((i % (31 + j) - 15) * .021);
                                m[i] = off[i] - (float)((i % (43 + j) - 21) * .019);
                            }
                            cpu[j] = new float[n]; device[j] = new float[n];
                            if (j == 3) { Array.Copy(residual, cpu[j], n); Array.Copy(residual, device[j], n); continue; }
                            var mapped = hard.Select(fit.Transform).ToArray();
                            E25CachedColumn(fit, hard, mapped, off, p, m, null, .01, stencil, cpu[j]);
                            gpu.Position(p, m, null, 0, .01, stencil, device[j]);
                            Compare("column", j, cpu[j].Select(x => (double)x), device[j].Select(x => (double)x));
                        }
                        var a = E25Normal(cpu, residual, 0, n); var b = E25Normal(device, residual, 0, n);
                        Compare("gram", -1, a.gram.Cast<double>(), b.gram.Cast<double>());
                        Compare("gradient", -1, a.gradient, b.gradient);
                        var x = Enumerable.Repeat(.5, 4).ToArray();
                        Compare("step", -1, Lm.LampScaledTrustProposal(a.gram, a.gradient, x, units, null, out _),
                            Lm.LampScaledTrustProposal(b.gram, b.gradient, x, units, null, out _));
                        void Compare(string what, int j, IEnumerable<double> aa, IEnumerable<double> bb)
                        {
                            var left = aa.ToArray(); var right = bb.ToArray();
                            double error = L2(left.Zip(right, (u, v) => u - v)); double norm = L2(left);
                            double cos = what == "step" ? E25Cos(left, right) : double.NaN;
                            bool ok = what == "step" && norm > 1e-12 ? cos >= .9999 : error <= Math.Max(1e-12, 1e-4 * norm);
                            csv.WriteLine(string.Join(",", views, stencil, what, j, F(error / Math.Max(norm, 1e-12)), F(error), F(cos), Flag(ok)));
                        }
                    }
                }
            }
        }
    }
}

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using UnityEngine;
using Lm = LightingMvp.ImageJacobianLm;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            private sealed partial class E27Trial
            {
                private sealed class FixedModel
                {
                    internal Lm.TrustJacobian jac;
                    internal double[] losses;
                    internal double fullLoss;
                    internal (double[,] gram, double[] gradient)[] views;
                }
                private FixedModel fixedModel;

                private bool RunPolicy()
                {
                    if (arm.policy == "audit") { FixedAudit(); return true; }
                    if (arm.policy == "gpu-parity") { GpuParity(); return true; }
                    if (arm.policy is "A2" or "A3" or "A4" or "A5" or "A6" or "A7" or "G4" or "M2" or "M3" or "X1" or "X2" or "X3" or "X5")
                        return RunSearch();
                    if (arm.policy is "color-audit" or "temperature-audit" or "mixed-layout" or "rgb" or "hist16" or "hist32" or "mix16" or "mix32" or "selected-histogram" or "selected-exploration" or "gram-gpu")
                        return RunExtra();
                    var l = arm.policy == "single-view" ? RunLm(sample.initial, arm.method, arm.cap, "L6-view0", limit: 24, views: new[] { 0 }) :
                        RunLm(sample.initial, arm.method, arm.cap, "main", stopping: false);
                    result.stop = l?.Stop ?? "insufficient_budget";
                    return true;
                }

                private Lm.TrustResult RunLm(double[] seed, string method, int budget, string label, int limit = int.MaxValue,
                    int[] views = null, double[] viewWeight = null, bool modelOnly = false, bool stopping = false,
                    Lm.TrustFallback localFallback = Lm.TrustFallback.None, bool completeBranchModels = false)
                {
                    views ??= E27Views; int count = views.Length;
                    bool shared = count == 3 && viewWeight == null;
                    int scoringReserve = shared || modelOnly ? 0 : (2 + limit) * 3;
                    int available = audit ? budget : Math.Min(budget, arm.cap - result.cameras);
                    int queryBudget = (available - scoringReserve) / count;
                    int minModel = 2 * layout.Count + 2 + (method is "C3" or "C4" ? 2 : 0);
                    if (queryBudget < minModel) return null;
                    segment = label;
                    var obj = Objective(views, viewWeight);
                    var sh = E27Mode(sample);
                    var frame = new float[obj.Size];
                    var units = Enumerable.Range(0, layout.Count).Select(i => layout.Axis(i).Unit).ToArray();
                    var steps = Enumerable.Range(0, layout.Count).Select(i => layout.Axis(i).Step).ToArray();
                    int entry = result.cameras;
                    int entryAudit = result.auditCameras;
                    var x0 = Local(seed);
                    double[] ExpandSeed(double[] x) => Expand(x, seed);
                    bool LocalLegal(double[] x) => Legal(ExpandSeed(x));
                    var fallback = localFallback != Lm.TrustFallback.None ? localFallback : arm.policy switch {
                        "A1" => Lm.TrustFallback.ViewPool, "X4" => Lm.TrustFallback.Nonmonotone, _ => Lm.TrustFallback.None };
                    var settings = new Lm.TrustSettings
                    {
                        Budget = queryBudget, Iterations = limit, Layout = arm.policy == "legacy" ? null : layout, Units = units,
                        PhysicalSteps = steps, Timing = timing, FixedProbeScale = arm.policy is not ("legacy" or "adaptive" or "A1" or "X4") && arm.stage != "0",
                        ProbeAllowed = LocalLegal, CandidateAllowed = LocalLegal, FinalAllowed = LocalLegal,
                        Fallback = fallback, TraceProposalClipping = true,
                        StallModelLimit = stopping ? 2 : completeBranchModels ? limit : 0,
                        ReserveRemainingModels = completeBranchModels,
                        Query = (q, phase, x, f, axis, step) => Query(phase, sh, ExpandSeed(x), views,
                            shared ? f : double.NaN, shared ? double.NaN : f, axis < 0 ? -1 : indices[axis], step)
                    };
                    if (fallback == Lm.TrustFallback.ViewPool) settings.ViewBlocks = Enumerable.Repeat(E23ChannelsPerView, count).ToArray();
                    if (arm.policy == "B1")
                        settings.LampPositionAllowed = (x, j) => h.room.Contains(ExpandSeed(x).Skip(4 * sample.active[j]).Take(4).ToArray());
                    if (arm.policy is "G1" or "G2" or "G3") settings.ProposalGradient = (j, columns, residual) =>
                        AggregateGradient(arm.policy, ViewModels(columns, residual, false).Select(v => v.gradient).ToArray());
                    float[][] borrowed = null;
                    if (modelOnly) settings.AuditColumns = (_, cols) => borrowed = cols;
                    settings.Jacobian = j =>
                    {
                        ++result.models;
                        if (result.models is 4 or 8 or 16) Checkpoint("model", result.models);
                        if (modelOnly)
                        {
                            var r = new float[obj.Size]; double centerLoss = obj.Evaluate(frameAtCenter, r);
                            fixedModel = new FixedModel { jac = j, views = ViewModels(borrowed, r, true), losses = ViewLoss(r), fullLoss = centerLoss };
                        }
                    };
                    if (modelOnly) { settings.BuildModelAtZero = true; settings.FinishAfterModel = _ => true; }
                    int small = 0;
                    if (stopping) settings.StopAfterIteration = () => small >= 3;
                    settings.Trial = t =>
                    {
                        var rig = ExpandSeed(t.Theta);
                        if (t.Accepted) small = t.Actual / Math.Max(t.CurrentLoss, 1e-30) < .0001 ? small + 1 : 0;
                        if (!shared && t.Accepted) Full(rig, "branch_accept_full");
                        decisions.WriteLine(string.Join(",", arm.id, result.cameras, label, t.Iteration, t.Source, t.Status, Flag(t.Legal), Flag(t.Accepted),
                            F(t.Predicted), F(t.Actual), F(t.Ratio), t.Shrinks, F(t.CurrentLoss), F(bestLoss), T(t.Step), T(rig), T(t.UnclippedStep),
                            T(t.RadiusLimitedStep), t.LampShrinks == null ? "" : string.Join(";", t.LampShrinks), F(t.UnclippedPrediction), t.Constraint,
                            T(t.Step?.Select((d, i) => d * layout.Axis(i).Unit).ToArray()), F(t.ReferenceLoss)));
                    };
                    using var gpu = arm.gpu && method == "C3" ? new E25GpuColumns(obj) : null;
                    h.E24Configure(settings, ExpandSeed, sample.active, method, obj, (_, rig, mode, phase, axis, d, y) =>
                    {
                        Capture(rig, mode, views, y); Query(phase, mode, rig, views, axis: axis, step: d);
                    }, arm.id, cachedMapping: method == "C3", metrics: counters, gpu: gpu);
                    if (!shared) Full(seed, "branch_start_full");
                    var run = Lm.RunTrust(x0, obj.Size, (x, phase, y) =>
                    {
                        var rig = ExpandSeed(x);
                        Capture(rig, sh, views, frame);
                        if (modelOnly && phase == "initial") frameAtCenter = (float[])frame.Clone();
                        long start = Stopwatch.GetTimestamp();
                        double f = obj.Evaluate(frame, y);
                        mapSeconds += Seconds(start);
                        if (shared && IsCandidate(phase)) Observe(rig, f);
                        return f;
                    }, settings);
                    if (gpu != null)
                    { gpuColumns += gpu.Columns; gpuUpload += gpu.UploadSeconds; gpuSubmit += gpu.SubmitSeconds; gpuSync += gpu.ReadbackSeconds; }
                    frameAtCenter = null;
                    if (shared && run.Evaluations * count != (audit ? result.auditCameras - entryAudit : result.cameras - entry))
                        throw new InvalidDataException("Native ledger disagrees with solver.");
                    if (!shared) Full(ExpandSeed(run.Best), "branch_end_full");
                    return run;
                }
                private float[] frameAtCenter;
                private static bool IsCandidate(string phase) => !(phase.StartsWith("fd", StringComparison.Ordinal) || phase.StartsWith("probe", StringComparison.Ordinal));
                private static double[] ViewLoss(float[] residual) => Enumerable.Range(0, residual.Length / E23ChannelsPerView)
                    .Select(v => E21Dot(residual, residual, v * E23ChannelsPerView, (v + 1) * E23ChannelsPerView)).ToArray();
                private static (double[,] gram, double[] gradient)[] ViewModels(float[][] j, float[] r, bool gram) => Enumerable.Range(0, 3).Select(v =>
                    gram ? E25Normal(j, r, v * E23ChannelsPerView, E23ChannelsPerView) :
                    (null, j.Select(col => E21Dot(col, r, v * E23ChannelsPerView, (v + 1) * E23ChannelsPerView)).ToArray())).ToArray();
                private (double[,] gram, double[] gradient) Subset(FixedModel model, int mask)
                {
                    int n = layout.Count; var g = new double[n]; var m = new double[n, n];
                    for (int v = 0; v < 3; ++v)
                        if ((mask & (1 << v)) != 0)
                            for (int i = 0; i < n; ++i)
                            {
                                g[i] += model.views[v].gradient[i];
                                for (int j = 0; j < n; ++j) m[i, j] += model.views[v].gram[i, j];
                            }
                    return (m, g);
                }
                private bool Proposal(double[] seed, FixedModel model, int mask, double fraction, StreamWriter table, bool cauchy = false, double[] overrideG = null, string name = null)
                {
                    var x = Local(seed); var normals = Subset(model, mask);
                    var delta = Lm.LampScaledTrustProposal(normals.gram, overrideG ?? normals.gradient, x, layout, t => Legal(Expand(t, seed)), out int shrinks, fraction, cauchy);
                    string source = name ?? "V" + mask;
                    if (TestDelta(seed, model, delta, source, fraction, shrinks, table)) return true;
                    bool diagnosticSubset = audit && arm.stage == "A1" && name == null;
                    bool onlineSubset = !audit && arm.policy == "A2" && result.cameras + 3 <= arm.cap;
                    if (!diagnosticSubset && !onlineSubset) return false;
                    // Same subset and normal equations. Salvage the proposal without moving a blocked
                    // lamp across a collider, using the existing E26 per-lamp constraint fallback.
                    bool LampLegal(double[] t, int owner) => h.room.Contains(Expand(t, seed).Skip(4 * sample.active[owner]).Take(4).ToArray());
                    foreach (bool useCauchy in new[] { false, true })
                    {
                        var perLamp = new int[layout.Lamps];
                        var repaired = Lm.LampScaledTrustProposal(normals.gram, overrideG ?? normals.gradient, x, layout,
                            t => Legal(Expand(t, seed)), out int whole, fraction, useCauchy, LampLegal, perLamp);
                        if (TestDelta(seed, model, repaired, source + (useCauchy ? "-lamp-cauchy" : "-lamp-lm"), fraction, whole, table, perLamp))
                            return true;
                    }
                    return false;
                }
                private bool TestDelta(double[] seed, FixedModel model, double[] delta, string source, double fraction, int shrinks, StreamWriter table, int[] lampShrinks = null)
                {
                    var x = Local(seed); var move = Expand(x.Zip(delta, (v, d) => v + d).ToArray(), seed);
                    bool legal = Legal(move); bool evaluated = legal && delta.Any(d => d != 0) && (audit || result.cameras + 3 <= arm.cap);
                    double[] after = null; double actual = double.NaN;
                    if (evaluated)
                    {
                        var f = new float[E23Channels]; var r = new float[E23Channels];
                        Capture(move, E27Mode(sample), E27Views, f); double loss = main.Evaluate(f, r);
                        after = ViewLoss(r); actual = model.fullLoss - loss;
                        Observe(move, loss); Query(source, E27Mode(sample), move, E27Views, loss);
                    }
                    double pred = Lm.TrustPredicted(model.jac.Gram, model.jac.Gradient, delta);
                    table.WriteLine(string.Join(",", arm.id, source, F(fraction), result.cameras, result.auditCameras, Flag(legal), Flag(evaluated), shrinks,
                        F(pred), F(actual), T(model.losses), T(after), T(delta.Select((d, i) => d * layout.Axis(i).Unit).ToArray()), T(move),
                        lampShrinks == null ? "" : string.Join(";", lampShrinks)));
                    return evaluated;
                }
                private StreamWriter Proposals() => Open("fixed-proposals.csv", "arm,source,fraction,cameras,audit_cameras,legal,evaluated,shrinks,predicted,actual,previous_views,next_views,step_physical,theta,lamp_shrinks");
                private void FixedAudit()
                {
                    RunLm(sample.initial, arm.method, arm.cap, "fixed", limit: 1, modelOnly: true);
                    var model = fixedModel;
                    if (model == null) { incomplete = true; result.stop = "no_fixed_model"; return; }
                    var rebuilt = Subset(model, 7);
                    double gerr = L2(rebuilt.gradient.Zip(model.jac.Gradient, (a, b) => a - b)) / Math.Max(1e-30, L2(model.jac.Gradient));
                    double herr = L2(rebuilt.gram.Cast<double>().Zip(model.jac.Gram.Cast<double>(), (a, b) => a - b)) /
                                  Math.Max(1e-30, L2(model.jac.Gram.Cast<double>()));
                    if (gerr > 1e-7 || herr > 1e-7 || Math.Abs(model.losses.Sum() - model.fullLoss) > 1e-7)
                        throw new InvalidDataException("Per-view model/score no longer reconstructs full F.");
                    using var view = Open("view-gradient.csv", "arm,physical_fd_mm,coordinates,view,rank,singular,columns,gradient,gram,gradient_retention,pair_cos,stencil,physical_steps");
                    using var table = Proposals();
                    double[] axis = Enumerable.Range(0, layout.Count).Select(i => layout.Axis(i).Unit).ToArray();
                    double[] scales = TrustScale();
                    foreach (var setting in new[] { ("normalized", axis.Select(_ => 1d).ToArray()),
                        ("physical", axis.Select(u => 1 / u).ToArray()), ("trust", scales) })
                    {
                        var per = model.views.Select(v => v.gradient.Select((g, j) => g * setting.Item2[j]).ToArray()).ToArray();
                        var sum = per[0].Select((a, j) => a + per[1][j] + per[2][j]);
                        double keep = L2(sum) / Math.Max(1e-30, per.Sum(L2));
                        var cos = new[] { Cos(per[0], per[1]), Cos(per[0], per[2]), Cos(per[1], per[2]) };
                        for (int m = 0; m <= 3; ++m)
                        {
                            var normal = Subset(model, m == 3 ? 7 : 1 << m);
                            int n = layout.Count;
                            for (int i = 0; i < n; ++i)
                            for (int j = 0; j < n; ++j) normal.gram[i, j] *= setting.Item2[i] * setting.Item2[j];
                            var spec = ClassroomDiagnosis.GetSpectrum(normal.gram, n);
                            view.WriteLine(string.Join(",", arm.id, arm.step_mm, setting.Item1, m, spec.rank, T(spec.singular), T(spec.columnNorms),
                                T(normal.gradient.Select((g, j) => g * setting.Item2[j]).ToArray()), Series(normal.gram.Cast<double>()),
                                F(keep), T(cos), string.Join(";", model.jac.Stencils), T(model.jac.PhysicalSteps)));
                        }
                    }
                    for (int mask = 1; mask <= 7; ++mask)
                    {
                        bool rendered = false;
                        foreach (double f in mask == 7 ? new[] { 1d } : new[] { 1d, .5, .25 })
                            rendered |= Proposal(sample.initial, model, mask, f, table);
                        if (!rendered) { incomplete = true; result.reason = "one or more diagnostic subsets had no legal nonzero proposal"; }
                    }
                    foreach (string g in new[] { "G0", "G1", "G2", "G3" })
                    {
                        var a = AggregateGradient(g, model.views.Select(v => v.gradient).ToArray());
                        var direction = a.Select((z, i) => -z * scales[i] * scales[i]).ToArray();
                        double max = direction.Select((d, i) => Math.Abs(d / scales[i])).Max();
                        if (max > 0) for (int i = 0; i < direction.Length; ++i) direction[i] /= max;
                        for (int i = 0; i < direction.Length; ++i) direction[i] *= .25;
                        ClipToLegal(Local(sample.initial), direction, sample.initial, out int shrinks);
                        TestDelta(sample.initial, model, direction, g + "-pure", 1, shrinks, table);
                        Proposal(sample.initial, model, 7, 1, table, overrideG: a, name: g + "-preconditioned");
                    }
                    if (sample.id == "L6-S" && arm.step_mm == 20) Explore(sample.initial, arm.cap, "X1");
                    result.stop = "fixed_diagnostic";
                }
                private double[] TrustScale() => Enumerable.Range(0, layout.Count).Select(i => layout.Axis(i).Radius / layout.Axis(i).Unit).ToArray();
                private static double Cos(double[] a, double[] b) => Dot(a, b) / Math.Max(1e-30, L2(a) * L2(b));
                private static double Dot(double[] a, double[] b) { double s = 0; for (int i = 0; i < a.Length; ++i) s += a[i] * b[i]; return s; }
                private void ClipToLegal(double[] x, double[] d, double[] ori, out int shrinks)
                {
                    for (shrinks = 0; shrinks <= 16; ++shrinks)
                    {
                        if (Legal(Expand(x.Zip(d, (a, b) => a + b).ToArray(), ori))) return;
                        for (int i = 0; i < d.Length; ++i) d[i] *= .5;
                    }
                    Array.Clear(d, 0, d.Length);
                }



            }

        }
    }
}

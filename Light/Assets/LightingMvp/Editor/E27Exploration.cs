using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using Lm = LightingMvp.ImageJacobianLm;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            private sealed partial class E27Trial
            {
                private bool RunSearch(string selected = null)
                {
                    string p = selected ?? arm.policy;
                    bool switching = p is "M2" or "M3";
                    bool randomSearch = p is "X1" or "X2" or "X3" or "X5";
                    bool direct = arm.direct || sample.id == "L6-S";
                    int minimum = ModelCost(arm.method, 3), left = arm.cap;
                    int reserve = p == "X5" ? arm.cap : switching ? Math.Max((int)(arm.cap * .35), 3 * (ModelCost("H1", 3) + 20)) : (int)(arm.cap * (randomSearch ? .40 : .35));
                    var incumbent = sample.initial;
                    if (!direct && p != "X5")
                    {
                        var mainRun = RunLm(incumbent, arm.method, left - reserve, "incumbent", stopping: true);
                        if (mainRun == null) { incomplete = true; result.stop = "not_enough_primary_budget"; return true; }
                        if (mainRun.Stop == "near_zero_residual") { result.stop = mainRun.Stop; result.reason = "near-zero incumbent; online trigger not needed"; return true; }
                        incumbent = best;
                        left = arm.cap - result.cameras;
                    }
                    if (best == null) { segment = "fixed_start"; Full(incumbent, "incumbent_full"); left = arm.cap - result.cameras; }
                    result.triggered = true;
                    double before = bestLoss;
                    long now = Stopwatch.GetTimestamp();
                    if (switching)
                    {
                        using var csv = Open("model-switch.csv", "arm,start_camera,end_camera,h1_models,h1_stop,start_f,end_f,used_dogleg,theta");
                        int cam = result.cameras, models = result.models;
                        var fit = RunLm(incumbent, "H1", left, "H1-switch", limit: 3, localFallback: p == "M3" ? Lm.TrustFallback.Dogleg : Lm.TrustFallback.None);
                        csv.WriteLine(string.Join(",", arm.id, cam, result.cameras, result.models - models, fit?.Stop,
                            F(before), F(bestLoss), Flag(p == "M3"), T(best)));
                        if (fit == null) { incomplete = true; result.reason = "no complete H1 model"; }
                    }
                    else if (p == "A2")
                    {
                        RunLm(incumbent, arm.method, left, "proper-subsets-model", limit: 1, modelOnly: true);
                        if (fixedModel == null) { incomplete = true; result.reason = "missing subset model"; }
                        else
                        {
                            using var proposals = Proposals();
                            for (int mask = 1; mask < 7; ++mask)
                                if (!Proposal(incumbent, fixedModel, mask, 1, proposals))
                                { incomplete = true; result.reason = "one or more subsets had no nonzero feasible proposal"; }
                        }
                    }
                    else if (randomSearch)
                        Explore(incumbent, Math.Min(left, p == "X5" ? left : reserve), p);
                    else
                        Branch(incumbent, direct ? Math.Min(left, Math.Max(reserve, 3 * (ModelCost(arm.method, 2) + 18) * 3)) : Math.Min(left, reserve), p);
                    orchestrationSeconds += Seconds(now); // Inclusive exploration/branch wall time, reported separately.
                    result.explored = bestLoss < before - Math.Max(1e-14, 1e-9 * before);
                    left = arm.cap - result.cameras;
                    var finish = p != "X5" && left >= minimum ? RunLm(best, arm.method, left, "refine") : null;
                    result.stop = finish?.Stop ?? (p == "X5" ? "cma_budget" : "insufficient_remaining_model_budget");
                    return true;
                }

                private int ModelCost(string method, int views) => (2 * layout.Count + 2 + (method is "C3" or "C4" ? 2 : 0)) * views;
                private void Branch(double[] origin, int budget, string policy)
                {
                    int requested = arm.branch_models > 0 ? arm.branch_models : policy is "A4" or "A6" ? 8 : 3;
                    int[][] groups = policy switch {
                        "A5" or "A6" => new[] { new[] { 0, 1 }, new[] { 0, 2 }, new[] { 1, 2 } },
                        "A7" or "G4" => new[] { E27Views, E27Views, E27Views },
                        _ => new[] { new[] { 0 }, new[] { 1 }, new[] { 2 } }
                    };
                    int per = budget / groups.Length;
                    int estimate = (ModelCost(arm.method, groups[0].Length) + (policy == "A7" ? 0 : 3));
                    int common = Math.Min(requested, Math.Max(0, (per - 6) / estimate));
                    using var table = Open("branches.csv", "arm,branch,cameras,views,view_weights,requested_models,common_cap,actual_models,branch_loss,global_best,stop,theta");
                    if (common == 0) { incomplete = true; result.reason = "reserved budget cannot fund all branches"; return; }
                    for (int k = 0; k < groups.Length; ++k)
                    {
                        int c = result.cameras;
                        double[] w = null;
                        if (policy == "G4")
                        {
                            w = E27Views.Select(_ => -Math.Log(Math.Max(1e-15, random.NextDouble()))).ToArray();
                            double s = w.Sum(); for (int i = 0; i < w.Length; ++i) w[i] *= 3 / s;
                        }
                        var l = RunLm(origin, arm.method, per, policy + "_branch_" + k, limit: common, views: groups[k], viewWeight: w,
                            completeBranchModels: true);
                        table.WriteLine(string.Join(",", arm.id, k, result.cameras - c, string.Join(";", groups[k]), T(w), requested, common,
                            l?.Jacobians ?? 0, l == null ? "" : F(l.BestLoss), F(bestLoss), l?.Stop, l == null ? "" : T(Expand(l.Best, origin))));
                        if (l == null || l.Jacobians < common && l.Stop != "near_zero_residual")
                        { incomplete = true; result.reason = "one or more branches did not construct its common model allocation"; }
                    }
                }

                private void Explore(double[] origin, int budget, string policy)
                {
                    int cap = Math.Min(arm.cap, result.cameras + budget);
                    segment = policy + "_explore";
                    double reference = bestLoss;
                    using var csv = Open("exploration-trace.csv", "arm,source,attempt,cameras,full_f,incumbent_f,best_f,legal,discarded_draws,max_physical_xyz_step,theta");
                    if (policy == "X1")
                    {
                        int d = layout.Count;
                        for (int j = 0; j < d && result.cameras + 3 <= cap; ++j)
                            for (int k = 0; k < 4 && result.cameras + 3 <= cap; ++k)
                            {
                                var v = new double[d]; v[j] = k % 2 == 0 ? 1 : -1;
                                Evaluate("coordinate", 4 * j + k, Move(origin, v, k >= 2), 0);
                            }
                        var basis = new List<double[]>();
                        for (int j = 0; j < d && result.cameras + 3 <= cap; ++j)
                        {
                            var v = Enumerable.Range(0, d).Select(_ => Gaussian()).ToArray();
                            foreach (var q in basis) { double a = Dot(v, q); for (int i = 0; i < d; ++i) v[i] -= a * q[i]; }
                            double n = L2(v); if (n < 1e-12) continue;
                            for (int i = 0; i < d; ++i) v[i] /= n;
                            basis.Add(v);
                            for (int sign = 0; sign < 2 && result.cameras + 3 <= cap; ++sign)
                                Evaluate("orthogonal", 2 * j + sign, Move(origin, v.Select(q => (sign == 0 ? 1 : -1) * q).ToArray(), j % 2 == 1), 0);
                        }
                        if (basis.Count < Math.Min(8, d)) { incomplete = true; result.reason = "polling budget did not cover minimum orthogonal directions"; }
                    }
                    else if (policy == "X2")
                    {
                        int per = (cap - result.cameras) / 6;
                        int count = per < ModelCost(arm.method, 3) + 3 ? 3 : 6;
                        per = (cap - result.cameras) / count;
                        int legalSeeds = 0;
                        for (int k = 0; k < count && result.cameras + 3 <= cap; ++k)
                        {
                            double[] draw = null; int tries = 0;
                            for (; tries < 16; ++tries)
                            {
                                draw = Move(origin, Enumerable.Range(0, layout.Count).Select(_ => 2 * random.NextDouble() - 1).ToArray(), k % 2 == 1);
                                if (Legal(draw)) break;
                            }
                            segment = "basin_" + k;
                            if (!Evaluate("basin_seed", k, draw, tries).HasValue) continue;
                            ++legalSeeds;
                            var l = RunLm(draw, arm.method, Math.Min(per - 3, cap - result.cameras), segment, limit: 3);
                            if (l == null) { incomplete = true; result.reason = "legal basin seed without full joint model"; }
                        }
                        if (legalSeeds < 3) { incomplete = true; result.reason = "fewer than three legal basin seeds"; }
                    }
                    else
                    {
                        var state = new E27Cma(layout.Count, Gaussian);
                        var scales = ExploreUnits(false);
                        var x0 = Local(origin);
                        int gen = 0;
                        for (; (policy == "X5" || gen < 12) && result.cameras + 3 <= cap; ++gen)
                        {
                            var valid = new List<(double[] x, double score)>();
                            for (int k = 0; k < state.Lambda && result.cameras + 3 <= cap; ++k)
                            {
                                double[] x = null, rig = null; int tries = 0;
                                for (; tries < 16; ++tries)
                                {
                                    x = state.Sample();
                                    var local = x.Select((v, j) => Math.Min(1, Math.Max(0, x0[j] + v * scales[j]))).ToArray();
                                    for (int j = 0; j < local.Length; ++j) x[j] = (local[j] - x0[j]) / scales[j];
                                    rig = Expand(local, origin);
                                    if (Legal(rig)) break;
                                }
                                double? value = Evaluate("cma_generation_" + gen, k, rig, tries);
                                if (value.HasValue) valid.Add((x, value.Value));
                            }
                            if (valid.Count >= state.Mu) state.Update(valid.OrderBy(t => t.score).Take(state.Mu).Select(t => t.x).ToArray());
                            if (valid.Count == 0 && result.cameras + 3 <= cap)
                            { incomplete = true; result.reason = "no legal CMA samples in a generation"; break; }
                        }
                    }
                    double? Evaluate(string source, int attempt, double[] rig, int discarded)
                    {
                        bool legal = rig != null && Legal(rig); double? value = null;
                        if (legal && result.cameras + 3 <= cap) value = Full(rig, source);
                        double move = rig == null ? 0 : sample.active.Max(i => UnityEngine.Vector3.Distance(h.room.PositionOf(rig, i), h.room.PositionOf(origin, i)));
                        csv.WriteLine(string.Join(",", arm.id, source, attempt, result.cameras, value.HasValue ? F(value.Value) : "", F(reference), F(bestLoss),
                            Flag(legal), discarded, F(move), T(rig)));
                        return value;
                    }
                }

                private double Gaussian() => Math.Sqrt(-2 * Math.Log(Math.Max(1e-15, random.NextDouble()))) * Math.Cos(2 * Math.PI * random.NextDouble());
                private double[] ExploreUnits(bool wide) => Enumerable.Range(0, layout.Count).Select(i => {
                    var a = layout.Axis(i); return (a.IsPosition ? (wide ? .30 : .10) : a.Coordinate == Lm.TrustCoordinate.Mired ?
                        (wide ? 40 : 20) : (wide ? .35 : .15)) / a.Unit;
                }).ToArray();
                private double[] Move(double[] origin, double[] direction, bool wide)
                {
                    var x = Local(origin); var s = ExploreUnits(wide);
                    for (int i = 0; i < x.Length; ++i) x[i] = Math.Min(1, Math.Max(0, x[i] + direction[i] * s[i]));
                    return Expand(x, origin);
                }
            }
        }
    }
}

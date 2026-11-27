using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using UnityEngine;
using Frame = LightingMvp.NativeRenderBackend.Frame;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            private void E22StructuredOlat(bool seven, bool requestedShadows)
            {
                var auditFile = Path.Combine(Path.GetDirectoryName(directory), "e22_d_audit", "olat-gate.json");
                var audit = JsonUtility.FromJson<E22LinearityReport>(File.ReadAllText(auditFile));
                using var columns = Csv("analytic-columns.csv", "shadows,lamp,step_log,norm_fd,norm_analytic,cosine,relative_error,camera_fd,seconds_fd,camera_olat,seconds_olat");
                using var cost = Csv("analytic-capture-cost.csv", "shadows,phase,lamp,camera_renders,seconds,cached");
                using var nnls = Csv("nnls-at-known-start-positions.csv", "shadows,active,objective_at_input_powers,objective_after_fit,mixed_recapture,olat_cameras,powers");
                var mode = E22Solver(Arg("-bbE22Solver", "A2"));
                int[] active = seven ? room.Lamps : new[] { 4, 6 };
                int cap = seven ? 1575 : 520;
                string sourceStage = seven ? "e22_a28" : "e22_a8";
                var prior = !smoke ? E22ReadOwn(sourceStage) : null;
                var expected = requestedShadows ? "on" : "off";
                BoundedLightWeights.CheckKnownCases();
                foreach (var condition in audit.states.Where(p => p.pass && p.shadows == requestedShadows))
                {
                    bool shadows = condition.shadows;
                    var gt = E22Target(shadows); var native = NativeSignal(gt);
                    var initial = E22Start(733, active);
                    string baseline = seven ? "seed733_near_28d" : "seed733_near_L5_L7";
                    string scenario = baseline + "_" + expected;
                    var reference = shadows ? null : prior?.arms.SingleOrDefault(a => a.scenario == baseline && a.method == E22SolverKey(mode));
                    RunE22Arm(scenario, "D0", active, initial, gt, native, cap, mode,
                        reused: reference, reusedFrom: reference == null ? null : sourceStage + "/" + reference.id);
                    var clock = Stopwatch.StartNew();
                    var dark = Enumerable.Range(0, E19FitViews).Select(v => CaptureOlat(v, initial, shadows, Array.Empty<int>(), Array.Empty<double>())).ToArray();
                    cost.WriteLine(string.Join(",", Flag(shadows), "empty", 0, E19FitViews, F(clock.Elapsed.TotalSeconds), 0));
                    var pos = new Vector3[lights.Length];
                    var cache = new float[lights.Length][];
                    var ana = new float[native.size]; var plus = new float[native.size]; var minus = new float[native.size];
                    bool goodColumns = true;
                    foreach (int lamp in active)
                    {
                        int before = olatFrames; clock.Restart();
                        Column(initial, lamp, ana);
                        double timeOlat = clock.Elapsed.TotalSeconds;
                        int olatCount = olatFrames - before, nativeBefore = low.RenderCount;
                        clock.Restart();
                        double h = .04 / LogPowerSpan;
                        var xp = (double[])initial.Clone(); var xm = (double[])initial.Clone();
                        xp[4 * lamp + 3] += h; xm[4 * lamp + 3] -= h;
                        if (!room.Contains(xp) || !room.Contains(xm)) throw new InvalidOperationException("D2 audit intensity perturbation is outside calibration.");
                        native.objective(xp, plus); native.objective(xm, minus);
                        for (int i = 0; i < plus.Length; ++i) plus[i] = (float)((plus[i] - minus[i]) / (2 * h));
                        double fdNorm = Math.Sqrt(E21Dot(plus, plus)), anaNorm = Math.Sqrt(E21Dot(ana, ana));
                        double dot = E21Dot(plus, ana);
                        double cosine = dot / Math.Max(1e-24, fdNorm * anaNorm);
                        double relative = Math.Sqrt(Math.Max(0, fdNorm * fdNorm + anaNorm * anaNorm - 2 * dot)) / Math.Max(1e-12, fdNorm);
                        goodColumns &= cosine >= .95 && relative <= .10;
                        columns.WriteLine(string.Join(",", Flag(shadows), lamp + 1, F(.04), F(fdNorm), F(anaNorm), F(cosine), F(relative),
                            low.RenderCount - nativeBefore, F(clock.Elapsed.TotalSeconds), olatCount, F(timeOlat)));
                    }
                    // Check the bounded intensity model once at the shared positions. This does not feed the joint solver.
                    AuditBounded(initial);
                    if (!goodColumns)
                    {
                        File.WriteAllText(Path.Combine(directory, "hybrid-skipped.txt"), "Initial intensity columns missed cosine 0.95 or the 10% relative error tolerance.\n");
                        continue;
                    }
                    Array.Clear(cache, 0, cache.Length);
                    RunE22Arm(scenario, "D1", active, initial, gt, native, cap, mode,
                        extra: (settings, expand) => settings.LogIntensityJacobian = (x, localLamp, output) => Column(expand(x), active[localLamp], output));

                    void Column(double[] rig, int lamp, float[] output)
                    {
                        var response = Unit(rig, lamp);
                        float scale = (float)(room.PowerOf(rig, lamp) * LogPowerSpan);
                        for (int j = 0; j < output.Length; ++j) output[j] = scale * response[j];
                    }

                    float[] Unit(double[] rig, int lamp)
                    {
                        var lightPosition = room.PositionOf(rig, lamp);
                        if (cache[lamp] != null && pos[lamp].Equals(lightPosition))
                        {
                            cost.WriteLine(string.Join(",", Flag(shadows), "unit", lamp + 1, 0, 0, 1));
                            return cache[lamp];
                        }
                        var timer = Stopwatch.StartNew();
                        var result = new float[native.size];
                        int offset = 0;
                        for (int v = 0; v < E19FitViews; ++v)
                        {
                            var raw = Increment(CaptureOlat(v, rig, shadows, new[] { lamp }, new[] { 1d }), dark[v]);
                            float weight = (float)(1 / Math.Sqrt(E19FitViews * (gt.fit[v].Energy + 1e-12)));
                            for (int i = 0; i < raw.Length; ++i) result[offset++] = raw[i] * weight;
                        }
                        cache[lamp] = result; pos[lamp] = lightPosition;
                        cost.WriteLine(string.Join(",", Flag(shadows), "unit", lamp + 1, E19FitViews, F(timer.Elapsed.TotalSeconds), 0));
                        return result;
                    }

                    void AuditBounded(double[] rig)
                    {
                        int n = active.Length, before = olatFrames;
                        int[] inactive = room.Lamps.Except(active).ToArray();
                        var fixedPowers = inactive.Select(l => room.PowerOf(rig, l)).ToArray();
                        var residual = new float[native.size]; int start = 0;
                        for (int v = 0; v < E19FitViews; ++v)
                        {
                            var background = CaptureOlat(v, rig, shadows, inactive, fixedPowers);
                            var goal = Increment(gt.fit[v], background);
                            float weight = (float)(1 / Math.Sqrt(E19FitViews * (gt.fit[v].Energy + 1e-12)));
                            for (int i = 0; i < goal.Length; ++i) residual[start++] = goal[i] * weight;
                        }
                        var basis = active.Select(l => Unit(rig, l)).ToArray();
                        var gram = new double[n, n]; var rhs = new double[n];
                        for (int i = 0; i < n; ++i)
                        {
                            rhs[i] = E21Dot(basis[i], residual);
                            for (int j = 0; j <= i; ++j) gram[i, j] = gram[j, i] = E21Dot(basis[i], basis[j]);
                        }
                        double[] w = active.Select(l => room.PowerOf(rig, l)).ToArray();
                        double baseLoss = ModelLoss(w);
                        BoundedLightWeights.Solve(gram, rhs, n, PowerMin, PowerMax, w);
                        var replay = (double[])rig.Clone();
                        for (int i = 0; i < n; ++i) replay[4 * active[i] + 3] = PowerCoordinate(w[i]);
                        double measured = native.objective(replay, null);
                        nnls.WriteLine(string.Join(",", Flag(shadows), LampList(active), F(baseLoss), F(ModelLoss(w)), F(measured), olatFrames - before, Series(w)));

                        double ModelLoss(double[] a)
                        {
                            double value = E21Dot(residual, residual);
                            for (int i = 0; i < n; ++i)
                            {
                                value -= 2 * a[i] * rhs[i];
                                for (int j = 0; j < n; ++j) value += a[i] * gram[i, j] * a[j];
                            }
                            return value;
                        }
                    }
                }
            }
        }
    }
}

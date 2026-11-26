using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            internal void JointIrradiance()
            {
                using var geometry = new MbdSurfaceGeometry(room.Camera);
                var sites = geometry.Samples(smoke ? 1024 : 4096, directory);
                var observed = SurfaceTruth(geometry, sites, false);
                SaveSurfaceOracle("oracle-off.csv", sites.Length, observed);
                var native = Target(false);
                var shared = E19Start(room, room.Start(733), .45f);
                var all = Enumerable.Range(0, room.Count).ToArray();
                var groups = new[] { all, new[] { 4, 6 }, new[] { 1, 6 } };
                using var blockLog = Csv("block-schedule.csv", "arm,segment,lamp,queries,loss_before,loss_after,stop");
                using var scaling = Csv("oracle-energy.csv", "lamp,area_mean_target_energy");
                using var nativeCross = Csv("native-pair-cross.csv", "case,axis_a,axis_b,gram,cosine,step_m,log_step");
                double area = sites.Sum(s => (double)s.area);
                foreach (int lamp in all)
                {
                    double energy = 0;
                    for (int i = 0, k = lamp * sites.Length; i < sites.Length; ++i, ++k)
                        energy += sites[i].area * (double)ProxyGeometry.Rgb(observed[k]).sqrMagnitude / area;
                    scaling.WriteLine(string.Join(",", lamp + 1, F(energy)));
                }
                foreach (var lamps in smoke ? groups.Skip(1).Take(1) : groups)
                {
                    var start = Start(shared, lamps);
                    string group = "seed733_near_" + string.Join("_", lamps.Select(i => $"L{i + 1}"));
                    if (lamps.Length == 2) RecordNativeCross(nativeCross, group, lamps, start, native);
                    foreach (int budget in lamps.Length == 7 ? new[] { 520, 1575 } : new[] { 520 })
                    {
                        string key = group + "_q" + budget;
                        var signal = FieldSignal(sites, lamps, observed, lamps.Length, "D0_area", normalizeLamp: false);
                        var equal = FieldSignal(sites, lamps, observed, lamps.Length, "D0_lamp_normalized", normalizeLamp: true);
                        var singles = lamps.Select(l => FieldSignal(sites, new[] { l }, observed, lamps.Length, "D0_area", normalizeLamp: false)).ToArray();
                        RunArm(key, "C0_independent4D", lamps, start, native, signal, budget, .015,
                            partition: arm => Partition(arm, start, lamps, singles, false));
                        RunArm(key, "C1_cyclic_BCD", lamps, start, native, signal, budget, .015,
                            partition: arm => Partition(arm, start, lamps, singles, true));
                        RunArm(key, "C2_shared_LM", lamps, start, native, signal, budget, .015);
                        RunArm(key, "C3_per_lamp_trust", lamps, start, native, signal, budget, .015, independent: true);
                        RunArm(key, "C4_lamp_normalized", lamps, start, native, equal, budget, .015);
                    }
                }

                double[] Partition(E21Trial arm, double[] theta, int[] lamps, E21Signal[] signals, bool cyclic)
                {
                    if (!cyclic)
                    {
                        for (int i = 0; i < lamps.Length; ++i)
                        {
                            int cap = arm.limit / lamps.Length + (i < arm.limit % lamps.Length ? 1 : 0);
                            theta = Block(theta, i, $"L{lamps[i] + 1}", cap, int.MaxValue, out _);
                        }
                        arm.stop = "independent_quota";
                        return theta;
                    }

                    var converged = new bool[lamps.Length];
                    int sweep = 0;
                    while (arm.limit - arm.evaluations >= 10)
                    {
                        ++sweep;
                        bool progress = false;
                        for (int i = 0; i < lamps.Length && arm.limit - arm.evaluations >= 10; ++i)
                        {
                            if (converged[i]) continue;
                            var next = Block(theta, i, $"s{sweep}_L{lamps[i] + 1}", Math.Min(16, arm.limit - arm.evaluations), 1, out double best);
                            progress |= !next.SequenceEqual(theta);
                            converged[i] = best < 1e-12;
                            theta = next;
                        }
                        if (!progress)
                        {
                            arm.stop = converged.All(v => v) ? "all_blocks_converged" : "stationary_sweep";
                            return theta;
                        }
                    }
                    arm.stop = converged.All(v => v) ? "all_blocks_converged" : "partition_budget";
                    return theta;

                    double[] Block(double[] current, int i, string segment, int cap, int iterations, out double finalLoss)
                    {
                        ImageJacobianLm.TrustResult measured = null;
                        var next = Solve(arm, segment, new[] { lamps[i] }, current, signals[i], cap, null, iterations: iterations, record: r => measured = r);
                        finalLoss = measured.BestLoss;
                        blockLog.WriteLine(string.Join(",", arm.id, segment, lamps[i] + 1, measured.Evaluations, F(measured.InitialLoss),
                            F(finalLoss), measured.Stop));
                        return next;
                    }
                }
            }

            private Color[] SurfaceTruth(MbdSurfaceGeometry geometry, MbdSurface[] sites, bool shadows)
            {
                int width = 256, height = (sites.Length * lights.Length + width - 1) / width;
                using var capture = new NativeIrradianceCapture(room.Camera, width, height);
                capture.SetSurfaceSamples(sites.Select(s => s.position).ToArray(), sites.Select(s => s.normal).ToArray(), geometry.Bounds);
                Apply(room.Truth, shadows);
                ++surfaceRenders;
                var data = capture.ReadSurfaces(lights);
                if (!shadows && data.Take(sites.Length * lights.Length).Any(c => c.a < .99f))
                    throw new InvalidOperationException("Incomplete unshadowed world-surface capture.");
                return data;
            }

            private void SaveSurfaceOracle(string file, int sampleCount, Color[] data, IEnumerable<int> lamps = null)
            {
                using var csv = Csv(file, "lamp,sample,r,g,b,visibility");
                foreach (int lamp in lamps ?? Enumerable.Range(0, room.Count))
                for (int i = 0, k = lamp * sampleCount; i < sampleCount; ++i, ++k)
                    csv.WriteLine(string.Join(",", lamp + 1, i, Xyz(ProxyGeometry.Rgb(data[k])), F(data[k].a)));
            }

            private E21Signal FieldSignal(MbdSurface[] sites, int[] active, Color[] observed, int countScale, string name,
                bool normalizeLamp, Func<int, Vector3, float[]> visibility = null)
            {
                double area = sites.Sum(s => (double)s.area);
                var positions = new Vector3[active.Length];
                var powers = new double[active.Length];
                var scales = new double[active.Length];
                var colors = active.Select(l => ProxyGeometry.Rgb(lights[l].color.linear)).ToArray();
                var ranges = active.Select(l => lights[l].range).ToArray();
                for (int group = 0; group < active.Length; ++group)
                {
                    int offset = active[group] * sites.Length;
                    double norm = 0;
                    for (int i = 0; i < sites.Length; ++i)
                        norm += sites[i].area * (double)ProxyGeometry.Rgb(observed[offset + i]).sqrMagnitude / area;
                    if (!(norm > 1e-12)) throw new InvalidOperationException("Degenerate irradiance oracle for lamp " + (active[group] + 1));
                    scales[group] = Math.Sqrt(1 / (area * countScale * (normalizeLamp ? norm : 1)));
                }
                var areaRoots = sites.Select(s => Math.Sqrt(s.area)).ToArray();
                var blocks = Enumerable.Repeat(3 * sites.Length, active.Length).ToArray();
                return new E21Signal(name, blocks.Sum(), (theta, residual) =>
                {
                    double value = 0;
                    for (int group = 0; group < active.Length; ++group)
                    {
                        positions[group] = room.PositionOf(theta, active[group]);
                        powers[group] = room.PowerOf(theta, active[group]);
                        var v = visibility?.Invoke(active[group], positions[group]);
                        int offset = active[group] * sites.Length;
                        for (int i = 0; i < sites.Length; ++i)
                        {
                            float direct = MbdStudy.IncidentIrradiance(sites[i], positions[group], powers[group], ranges[group]);
                            var prediction = colors[group] * (direct * (v == null ? 1 : v[i]));
                            var difference = (prediction - ProxyGeometry.Rgb(observed[offset + i])) * (float)(areaRoots[i] * scales[group]);
                            value += (double)difference.x * difference.x + (double)difference.y * difference.y + (double)difference.z * difference.z;
                            if (residual != null)
                            {
                                int j = 3 * (group * sites.Length + i);
                                residual[j] = difference.x; residual[j + 1] = difference.y; residual[j + 2] = difference.z;
                            }
                        }
                    }
                    return value;
                }, blocks);
            }

            private void RecordNativeCross(StreamWriter csv, string key, int[] lamps, double[] start, MbdRgbTarget native)
            {
                int size = NativeSignal(native).size;
                var axes = lamps.SelectMany(i => Enumerable.Range(4 * i, 4)).ToArray();
                var columns = new float[axes.Length][];
                var centre = new float[size]; var scratch = new float[size];
                NativeLoss(native, start, centre);
                for (int i = 0; i < axes.Length; ++i)
                {
                    int axis = axes[i];
                    var p = (double[])start.Clone(); var m = (double[])start.Clone();
                    double h = (axis % 4 == 3 ? .04 : .015) / units[axis % 4];
                    p[axis] += h; m[axis] -= h;
                    columns[i] = new float[size];
                    bool plus = room.Contains(p), minus = room.Contains(m);
                    if (!plus && !minus) continue;
                    var left = centre; var right = centre;
                    if (plus) { NativeLoss(native, p, columns[i]); left = columns[i]; }
                    if (minus) { NativeLoss(native, m, scratch); right = scratch; }
                    double denominator = plus && minus ? 2 * h : h;
                    for (int j = 0; j < size; ++j) columns[i][j] = (float)((left[j] - right[j]) / denominator);
                }
                for (int a = 0; a < 4; ++a)
                for (int b = 4; b < 8; ++b)
                {
                    double gram = E21Dot(columns[a], columns[b]), norm = Math.Sqrt(E21Dot(columns[a], columns[a]) * E21Dot(columns[b], columns[b]));
                    csv.WriteLine(string.Join(",", key, axes[a], axes[b], F(gram), F(norm > 0 ? gram / norm : double.NaN), F(.015), F(.04)));
                }
            }
        }
    }
}

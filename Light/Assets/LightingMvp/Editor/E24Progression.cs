using System;
using System.Linq;
using UnityEngine;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            private static string[] E24Selected()
            {
                var selected = Arg("-bbE24Models", "").Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries).Distinct().ToArray();
                if (selected.Any(m => m is not ("H4" or "H5" or "C1" or "C3" or "C4")))
                    throw new ArgumentException("Unexpected E24 model in -bbE24Models.");
                return selected;
            }

            private void E24ScreenRecovery(MbdRgbTarget hard, bool full)
            {
                var selected = E24Selected();
                if (selected.Length == 0) throw new ArgumentException("E24 recovery requires a model selected by the previous stage.");
                var source = E24Read(full ? "source" : "small");
                foreach (int seed in full ? E23Seeds : new[] { 131, 157 })
                {
                    var prior = full ? source.arms.Single(a => a.id == $"seed{seed}_near_on_B0") :
                        source.arms.Single(a => a.id.StartsWith($"seed{seed}_log_failed_L", StringComparison.Ordinal) && a.method == "B2");
                    int[] active = prior.lamps.Select(i => i - 1).ToArray();
                    foreach (string model in (full || selected.Contains("C1") ? new[] { "H0", "H2" } : new[] { "H2" }).Concat(selected))
                        E24Optimize(prior.scenario, model, active, prior.initial, hard, full ? E24Validate : E24Screen);
                }
            }

            private void E24Refinement(MbdRgbTarget hard)
            {
                string aux = E24Selected().FirstOrDefault();
                if (aux == null) throw new ArgumentException("E24 refinement requires a qualifying auxiliary model.");
                var source = E24Read("source");
                var cases = E23Seeds.Select(seed => source.arms.Single(a => a.id == $"seed{seed}_near_on_B0"))
                    .Select(a => (a.scenario, a.initial)).ToList();
                cases.Add(("dir793_5cm", E24Read("best").arms.Single(a => a.scenario == "dir793_5cm").initial));
                foreach (var (label, start) in cases)
                {
                    E24Optimize(label, "H0", E24AllLamps, start, hard, E24Validate);
                    E24Optimize(label, aux, E24AllLamps, start, hard, E24Validate);
                    E24Optimize(label + "_D2", aux, E24AllLamps, start, hard, E24Validate, "H1");
                    E24Optimize(label + "_D3", aux, E24AllLamps, start, hard, E24Validate, "H0");
                }
            }

            private void E24Sweep(E21StageSummary reference, MbdRgbTarget hard, bool classroom, bool far)
            {
                using var sweep = Csv("displacement-sweep.csv", "scene,scenario,seed,requested_cm,log_amplitude,lamp,actual_cm,lamp_range,draws,rejected,eligible,cause,direction_x,direction_y,direction_z");
                var methods = new[] { "H0", "H2" }.Concat(E24Selected().Take(1)).ToArray();
                if (classroom && !far)
                {
                    var recorded = E24Prior(reference, "C1");
                    foreach (string model in methods)
                        E24Optimize(recorded.scenario, model, E24AllLamps, recorded.initial, hard, E24Validate);
                }
                if (!classroom)
                {
                    // Load the actual archived 10 cm direction; do not re-seed its original sampler.
                    var history = E24Read("near");
                    foreach (int seed in new[] { 271, 793 })
                    {
                        var a = history.arms.Single(t => t.scenario == $"dir{seed}_10cm" && t.method == "C0");
                        var dir = E24AllLamps.Select(i => (room.PositionOf(a.initial, i) - room.PositionOf(room.Truth, i)).normalized).ToArray();
                        foreach (int cm in far ? new[] { 50 } : new[] { 20, 30, 50 })
                        {
                            bool test = !far && seed == 793 && cm < 50; // old directions: audit both, run at most one
                            TrySweep(seed, dir, new int[room.Count], cm, .05, test);
                        }
                    }
                }
                int[] seeds = classroom ? new[] { 1301, 1601 } : new[] { 911, 1201 };
                var radii = classroom ? far ? new[] { 10, 20, 30, 50 } : new[] { 10, 20, 30 } : new[] { 20, 30, 50 };
                foreach (int seed in seeds)
                {
                    var dirs = E24Directions(seed, radii, out var tries);
                    if (far) TrySweep(seed, dirs, tries, 50, .05, seed == seeds[0]);
                    else foreach (int cm in radii.Where(x => x < 50)) TrySweep(seed, dirs, tries, cm, .05, true);
                    if (!classroom && !far && seed == seeds[0]) TrySweep(seed, dirs, tries, 30, .25, true);
                }

                void TrySweep(int seed, Vector3[] dirs, int[] draws, int cm, double logI, bool test)
                {
                    string scenario = $"dir{seed}_{cm}cm" + (logI > .1 ? "_strong" : "");
                    var rig = (double[])room.Truth.Clone();
                    for (int i = 0; i < room.Count; ++i)
                    {
                        rig = room.WithPosition(rig, i, room.PositionOf(room.Truth, i) + dirs[i] * (cm / 100f));
                        rig[4 * i + 3] += (i % 2 == 0 ? logI : -logI) / LogPowerSpan;
                    }
                    bool legal = dirs.All(v => v != Vector3.zero) && room.Contains(rig);
                    for (int i = 0; i < room.Count; ++i)
                        sweep.WriteLine(string.Join(",", room.Name, scenario, seed, cm, F(logI), i + 1,
                            F(100 * Vector3.Distance(room.PositionOf(rig, i), room.PositionOf(room.Truth, i))), F(lights[i].range), draws[i],
                            draws[i] == 0 ? 0 : dirs[i] == Vector3.zero ? draws[i] : draws[i] - 1, Flag(legal),
                            !legal ? "strict_collider_or_bounds" : test ? "queued" : "checked_only", F(dirs[i].x), F(dirs[i].y), F(dirs[i].z)));
                    sweep.Flush();
                    if (!legal || !test) return;
                    foreach (var model in methods) E24Optimize(scenario, model, E24AllLamps, rig, hard, far ? E24Validate : E24Screen);
                }
            }

            private Vector3[] E24Directions(int seed, int[] radii, out int[] draws)
            {
                var rng = new System.Random(seed);
                var taken = new Vector3[room.Count];
                draws = new int[room.Count];
                for (int i = 0; i < taken.Length; ++i)
                {
                    var center = room.PositionOf(room.Truth, i);
                    for (int attempt = 0; attempt < 512; ++attempt)
                    {
                        ++draws[i];
                        var v = new Vector3((float)rng.NextDouble() - .5f, (float)rng.NextDouble() - .5f, (float)rng.NextDouble() - .5f);
                        if (v.sqrMagnitude < 1e-7f || v.sqrMagnitude > .25f) continue;
                        v.Normalize();
                        if (!radii.All(cm => room.Contains(room.WithPosition(room.Truth, i, center + v * (cm / 100f))))) continue;
                        taken[i] = v; break;
                    }
                }
                return taken;
            }
        }
    }
}

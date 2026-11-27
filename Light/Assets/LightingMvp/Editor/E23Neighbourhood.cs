using System;
using System.Linq;
using UnityEngine;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            private static readonly int[] E23DirectionSeeds = { 271, 793 };
            private static readonly int[] E23RadiiCm = { 2, 5, 10 };

            private void E23Neighbourhood(bool bestOnly)
            {
                var gt = E22Target(true);
                int[] active = room.Lamps;
                using var log = bestOnly ? E23ModelLog() : null;
                foreach (int seed in smoke ? E23DirectionSeeds.Take(1) : E23DirectionSeeds)
                {
                    var vectors = E23Directions(seed);
                    foreach (int cm in smoke ? new[] { 5 } : E23RadiiCm)
                        Compare($"dir{seed}_{cm}cm", seed, E23SmallRig(vectors, cm, false));
                    if (!smoke && seed == E23DirectionSeeds[0])
                        Compare($"dir{seed}_5cm_position_only", seed, E23SmallRig(vectors, 5, true));
                }

                void Compare(string scenario, int seed, double[] start)
                {
                    if (bestOnly)
                        E23RunProxyArm(scenario, "B2", seed, active, start, gt, "C1", 520, log, LightShadows.None, 0, "off_xyz");
                    else
                        foreach (string mapping in new[] { "C0", "C1" })
                            RunE23Arm(scenario, mapping, seed, active, start, gt, mapping, 520);
                }
            }

            private void E23Generalization(bool classroom)
            {
                const int seed = 509;
                int[] active = Enumerable.Range(0, room.Count).ToArray();
                if (active.Length != 7) throw new InvalidOperationException("E23 generalization needs the existing seven-light benchmark.");
                var gt = E22Target(true);
                if (classroom) e23Scale = E22Scale(gt);
                var start = room.Start(seed);
                if (classroom) start = E19Start(room, start, .45f);
                string scenario = classroom ? "seed509_classroom_near" : "seed509_complex_far";
                using var log = E23ModelLog();
                foreach (string mapping in new[] { "C0", "C1" })
                    RunE23Arm(scenario, mapping, seed, active, start, gt, mapping, 1575);
                E23RunProxyArm(scenario, "B2", seed, active, start, gt, "C1", 1575, log, LightShadows.None, 0, "off_xyz");
            }

            private Vector3[] E23Directions(int seed)
            {
                var rng = new System.Random(seed);
                var chosen = new Vector3[room.Count];
                for (int lamp = 0; lamp < room.Count; ++lamp)
                {
                    var at = room.PositionOf(room.Truth, lamp);
                    for (int attempt = 0; attempt < 256; ++attempt)
                    {
                        var test = new Vector3((float)(rng.NextDouble() - .5), (float)(rng.NextDouble() - .5), (float)(rng.NextDouble() - .5));
                        if (test.sqrMagnitude > .25f || test.sqrMagnitude < 1e-7f) continue;
                        test.Normalize();
                        if (!E23RadiiCm.All(cm => room.Contains(room.WithPosition(room.Truth, lamp, at + test * (cm / 100f))))) continue;
                        chosen[lamp] = test;
                        break;
                    }
                    if (chosen[lamp] == Vector3.zero) throw new InvalidOperationException($"Could not find a common legal 2/5/10 cm direction for L{lamp + 1}.");
                }
                return chosen;
            }

            private double[] E23SmallRig(Vector3[] directions, int cm, bool pure)
            {
                var rig = (double[])room.Truth.Clone();
                for (int lamp = 0; lamp < room.Count; ++lamp)
                {
                    rig = room.WithPosition(rig, lamp, room.PositionOf(room.Truth, lamp) + directions[lamp] * (cm / 100f));
                    if (!pure) rig[4 * lamp + 3] += (lamp % 2 == 0 ? .05 : -.05) / LogPowerSpan;
                }
                if (!room.Contains(rig)) throw new InvalidOperationException("E23 generated a small-neighbourhood pose outside Strict feasibility.");
                return rig;
            }
        }
    }
}

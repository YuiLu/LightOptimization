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
            internal void ProbePolicies()
            {
                var topology = JsonUtility.FromJson<MbdStudyResult>(File.ReadAllText(Arg("-bbE21PriorTopology", "")));
                var oldA = JsonUtility.FromJson<E20Summary>(File.ReadAllText(Arg("-bbE21PriorA", "")));
                if (topology.scene.scene != room.Name || topology.truth.Length != room.Truth.Length ||
                    topology.truth.Zip(room.Truth, (x, y) => Math.Abs(x - y)).Any(d => d > 1e-12))
                    throw new InvalidDataException("E20 topology reference does not match E21 scene parameters.");
                var origins = new List<(string label, int lamp, double[] x, bool shadows)>();
                foreach (var key in new[] { "seed131_near_L6", "seed733_far_L1", "cross_geometry_L1" })
                {
                    var prior = topology.arms.First(a => a.key == key && a.method == "StrictLocal");
                    int lamp = prior.lamps.Single() - 1;
                    origins.Add((key, lamp, Start(prior.initial, new[] { lamp }), key != "seed131_near_L6"));
                }
                var far = oldA.arms.First(a => a.id == "A1_157_far_L3" && !a.passed);
                origins.Add(("seed157_far_L3", 2, Start(far.start, new[] { 2 }), false));
                var legacy = topology.arms.First(a => a.key == "seed131_near_L6" && a.method == "StrictLocal");
                using var probes = Csv("legacy-endpoint-probes.csv", "case,source,axis,step_m,sign,collider_legal,survey_legal,native_loss");
                var off = Target(false);
                MeasureLegacy("seed131_near_L6", "E20_StrictLocal_final", 5, legacy.final);
                MeasureLegacy("seed157_far_L3", "E20_A1_final", 2, far.theta);
                foreach (var item in smoke ? origins.Take(1) : origins)
                {
                    var gt = Target(item.shadows);
                    foreach (double step in smoke ? new[] { .02 } : new[] { .005, .02, .05 })
                    {
                        string name = item.label + "_fd" + (int)Math.Round(step * 1000) + "mm";
                        foreach (var policy in new[] { ("B0", "strict"), ("B1", "probe_through"), ("B2", "full_through") })
                            RunArm(name, policy.Item1, new[] { item.lamp }, item.x, gt, NativeSignal(gt), 420, step, policy.Item2);
                    }
                }

                void MeasureLegacy(string label, string source, int lamp, double[] theta)
                {
                    probes.WriteLine(string.Join(",", label, source, -1, 0, 0, Flag(room.Contains(theta)), Flag(room.InSurveyDomain(theta)), F(NativeLoss(off, theta, null))));
                    foreach (double step in new[] { .005, .02, .05 })
                    for (int axis = 0; axis < 3; ++axis)
                    foreach (int sign in new[] { -1, 1 })
                    {
                        var x = (double[])theta.Clone();
                        x[4 * lamp + axis] += sign * step / units[axis];
                        bool inside = room.InSurveyDomain(x);
                        probes.WriteLine(string.Join(",", label, source, axis, F(step), sign, Flag(room.Contains(x)), Flag(inside),
                            F(inside ? NativeLoss(off, x, null) : double.NaN)));
                    }
                }
            }
        }
    }
}

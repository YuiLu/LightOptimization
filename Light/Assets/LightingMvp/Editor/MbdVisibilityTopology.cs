using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Object = UnityEngine.Object;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class MbdStudy
        {
            internal void Topology()
            {
                var fixedSites = geometry.Samples(smoke ? 768 : 2304, dir);
                mapSites = fixedSites.Select(x => x.position).ToArray();
                sampleCount = mapSites.Length;
                geometry.OpenRayReference();
                var colliders = Object.FindObjectsOfType<Collider>().Where(c => c.enabled && !c.isTrigger).ToArray();
                var on = Target(true);
                var off = Target(false);
                var cases = new List<(string label, int lamp, double[] x, bool primaryShadows)>();
                var near = E19Start(scene, scene.Start(131), .45f);
                cases.Add(("seed131_near_L6", 5, Combine(scene.Truth, new[] { (5, near) }), false));
                if (!smoke)
                {
                    var far = scene.Start(733);
                    cases.Add(("seed733_far_L1", 0, Combine(scene.Truth, new[] { (0, far) }), true));
                    var across = FindBlockedPath();
                    cases.Add(("cross_geometry_L" + (across.lamp + 1), across.lamp, across.start, true));
                }

                using var path = Csv("paths.csv", "case,lamp,t,x,y,z,distance_to_gt_cm,collider_legal,overlap_6cm,nearest_collider,nearest_geometry_m,native_on_loss,native_off_loss,hard_signature_distance,mbd_signature_distance,mbd_invalid");
                using var endpoints = Csv("endpoints.csv", "case,lamp,origin_cube_side,first_blocked_t,last_blocked_t,t,endpoint_legal,native_start_loss,endpoint_native_loss,max_path_native_loss,better_endpoint,peak_over_start,hard_distance,mbd_distance");
                using var signatures = Csv("visibility-signatures.csv", "case,t,sample,weight,hard_gt,hard,mbd_gt,mbd");
                foreach (var item in cases)
                {
                    int lamp = item.lamp;
                    var initial = item.x;
                    var truth = scene.PositionOf(scene.Truth, lamp);
                    var target = item.primaryShadows ? on : off;
                    var weights = fixedSites.Select(s => s.area * IncidentIrradiance(s, truth, 1, lights[lamp].range)).ToArray();
                    double weightSum = weights.Sum(w => (double)w);
                    if (!(weightSum > 0)) throw new InvalidOperationException("The visibility signature has no illuminated GT support.");
                    var gtHard = mapSites.Select(p => geometry.RayVisibility(p, truth)).ToArray();
                    var gtSoft = Visibility(lamp, scene.Truth);
                    int steps = smoke ? 12 : 40;
                    var samples = new (double t, bool legal, double native, double hard, double soft)[steps + 1];
                    for (int k = 0; k <= steps; ++k)
                    {
                        double t = k / (double)steps;
                        var x = initial.Select((u, i) => u + t * (scene.Truth[i] - u)).ToArray();
                        var p = scene.PositionOf(x, lamp);
                        var nearest = NearestGeometry(p);
                        int overlaps = Physics.OverlapSphere(p, .06f, ~0, QueryTriggerInteraction.Ignore).Length;
                        double rgbOn = NativeLoss(x, on, null, through: true), rgbOff = NativeLoss(x, off, null, through: true);
                        bool legal = scene.Contains(x);
                        var mbd = Visibility(lamp, x);
                        double hardDistance = 0, mbdDistance = 0;
                        for (int s = 0; s < mapSites.Length; ++s)
                        {
                            float hard = geometry.RayVisibility(mapSites[s], p);
                            hardDistance += weights[s] * Math.Abs(hard - gtHard[s]);
                            mbdDistance += weights[s] * Math.Abs(mbd[s] - gtSoft[s]);
                            if (k == 0 || k == steps / 2 || k == steps)
                                signatures.WriteLine(string.Join(",", item.label, F(t), s, F(weights[s]), F(gtHard[s]), F(hard), F(gtSoft[s]), F(mbd[s])));
                        }
                        hardDistance /= weightSum; mbdDistance /= weightSum;
                        samples[k] = (t, legal, item.primaryShadows ? rgbOn : rgbOff, hardDistance, mbdDistance);
                        path.WriteLine(string.Join(",", item.label, lamp + 1, F(t), Xyz(p), F(100 * Vector3.Distance(p, truth)),
                            legal ? 1 : 0, overlaps, nearest.name, F(nearest.distance), F(rgbOn), F(rgbOff), F(hardDistance), F(mbdDistance),
                            maps[lamp].map.LastDegenerate));
                        Log(item.label, "path_on", "path", k, "render_through", x, native: rgbOn);
                        Log(item.label, "path_off", "path", k, "render_through", x, native: rgbOff);
                    }
                    var blocked = samples.Where(s => !s.legal).ToArray();
                    if (blocked.Length == 0)
                        endpoints.WriteLine(string.Join(",", item.label, lamp + 1, "unblocked_control", "", "", "", "", F(samples[0].native), "", "", "", "", "", ""));
                    else
                    {
                        var jumps = samples.Where(s => s.t > blocked.Last().t && s.t < 1 && s.legal).ToArray();
                        foreach (var endpoint in jumps)
                        {
                            double peak = samples.Where(s => s.t <= endpoint.t).Max(s => s.native);
                            endpoints.WriteLine(string.Join(",", item.label, lamp + 1, "opposite_side", F(blocked[0].t), F(blocked.Last().t),
                                F(endpoint.t), 1, F(samples[0].native), F(endpoint.native), F(peak), endpoint.native < samples[0].native ? 1 : 0,
                                F(peak / Math.Max(1e-12, samples[0].native)), F(endpoint.hard), F(endpoint.soft)));
                        }
                    }
                    Fit(item.label, "StrictLocal", new[] { lamp }, initial, target, budget: 600);
                    Fit(item.label, "RenderThrough", new[] { lamp }, initial, target, through: true, budget: 600);
                    path.Flush(); endpoints.Flush(); signatures.Flush();
                }

                (int lamp, double[] start) FindBlockedPath()
                {
                    var axes = new List<Vector3>();
                    for (int a = -1; a <= 1; ++a)
                    for (int b = -1; b <= 1; ++b)
                    for (int c = -1; c <= 1; ++c)
                        if (a != 0 || b != 0 || c != 0) axes.Add(new Vector3(a, b, c).normalized);
                    // Select by geometry and displacement before any RGB or visibility evaluation.
                    foreach (float distance in new[] { .22f, .28f, .34f, .42f, .52f, .63f, .78f, .96f, 1.25f })
                    foreach (int lamp in Enumerable.Range(0, scene.Count))
                    foreach (var axis in axes)
                    {
                        Vector3 gt = scene.PositionOf(scene.Truth, lamp);
                        var start = scene.WithPosition(scene.Truth, lamp, gt + distance * axis);
                        if (!scene.Contains(start)) continue;
                        // An interior point with a zero solid distance distinguishes an actual crossing from a clearance shell.
                        for (int j = 3; j < 18; ++j)
                        {
                            Vector3 p = gt + (distance * j / 20) * axis;
                            if (colliders.Any(c => c is BoxCollider && (c.ClosestPoint(p) - p).sqrMagnitude < 1e-10f)) return (lamp, start);
                        }
                    }
                    throw new InvalidOperationException("No admissible diagnostic endpoints straddled complex-room geometry.");
                }

                (string name, float distance) NearestGeometry(Vector3 position)
                {
                    float nearest = float.PositiveInfinity;
                    string name = "";
                    foreach (var col in colliders)
                    {
                        var closest = col is MeshCollider m && !m.convex ? col.bounds.ClosestPoint(position) : col.ClosestPoint(position);
                        float distance = Vector3.Distance(closest, position);
                        if (distance >= nearest) continue;
                        nearest = distance; name = col.name.Replace(',', '_');
                    }
                    return (name, nearest);
                }
            }
        }
    }
}

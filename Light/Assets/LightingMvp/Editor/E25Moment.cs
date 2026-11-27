using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using UnityEngine;
using Mapping = LightingMvp.NativeRenderBackend.HdrMapping;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            private void E25Moment(MbdRgbTarget hard, string scenario)
            {
                const int lamp = 5; // fixed Classroom L6 attribution arm, not a selection based on the target.
                var start = E25Read("e_classroom").arms.Single(a => a.scenario == scenario && a.method == "H0").initial;
                var fit = new E23FrameObjective(hard.fit, Mapping.SignedLog, e24HardScale);
                var timer = Stopwatch.StartNew();
                int camerasBefore = low.RenderCount, passesBefore = geometryPasses, facesBefore = shadowFaces;
                var native = E25AuditedLamp(scenario, start, lamp, fit, .02, null, null);
                var registered = CaptureE22Geometry();
                var positions = new List<Vector3>(); var pixels = new List<int>();
                for (int v = 0; v < registered.views.Length; ++v)
                {
                    var view = registered.views[v];
                    for (int p = 0; p < view.reliable.Length; ++p)
                    {
                        if (!view.reliable[p]) continue;
                        positions.Add(view.position[p]); pixels.Add(v * E24PixelsPerView + p);
                    }
                }
                using var gate = Csv("moment-gate.csv", "scenario,lamp,registered_pixels,off_energy_coverage,hard_response_relative_error,passes_multiplication,fused_tested,shadow_faces,registration_passes,diagnostic_cameras,seconds,reason");
                if (positions.Count == 0)
                {
                    Gate(0, double.NaN, false, "no_native_registered_geometry");
                    return;
                }
                using var geometry = new MbdSurfaceGeometry(room.Camera);
                var config = MbdSettings(lights[lamp]);
                using var shadow = new PointMomentShadow(geometry.DrawCasters, positions.ToArray(), config);
                var center = Visibility(start);
                using var multiplicative = Csv("moment-multiplication.csv", "scenario,lamp,view,registered_pixels,off_energy_coverage,hard_relative_error,cosine,actual_shadow_faces");
                double errTotal = 0, hTotal = 0, oTotal = 0, regTotal = 0;
                int covered = 0;
                for (int view = 0; view < E19FitViews; ++view)
                {
                    int begin = view * E24PixelsPerView, end = begin + E24PixelsPerView;
                    double h = 0, o = 0, co = 0, e = 0, modelPower = 0, corr = 0;
                    int n = 0;
                    for (int p = begin; p < end; ++p) o += ChannelEnergy(native.io, p * 3);
                    while (covered < pixels.Count && pixels[covered] < end)
                    {
                        int p = pixels[covered], k = p * 3;
                        float v = center[covered++]; ++n;
                        for (int ch = k; ch < k + 3; ++ch)
                        {
                            double a = native.ih[ch], b = native.io[ch], q = v * b;
                            h += a * a; co += b * b; e += (a - q) * (a - q); modelPower += q * q; corr += q * a;
                        }
                    }
                    multiplicative.WriteLine(string.Join(",", scenario, lamp + 1, view, n, F(SafeRatio(co, o)), F(Math.Sqrt(SafeRatio(e, h))),
                        F(SafeRatio(corr, Math.Sqrt(h * modelPower))), shadow.Faces));
                    hTotal += h; oTotal += o; regTotal += co; errTotal += e;
                }
                double coverage = SafeRatio(regTotal, oTotal), error = Math.Sqrt(SafeRatio(errTotal, hTotal));
                // A poor fit between native On/Off intensity responses and Moment visibility rules out the product model.
                if (!(coverage >= .5 && error <= .2) || shadow.Nonfinite != 0)
                {
                    Gate(coverage, error, false, "requires_registered_half_off_energy_Hard_intensity_relative_error_at_most_0.2_and_finite_moments");
                    return;
                }
                string[] names = { "H1", "C3", "C4", "C4-MBD-delta" };
                var models = names.Select(_ => Enumerable.Range(0, 4).Select(__ => new float[fit.Size]).ToArray()).ToArray();
                var index = Enumerable.Repeat(-1, E19FitViews * E24PixelsPerView).ToArray();
                for (int p = 0; p < pixels.Count; ++p) index[pixels[p]] = p;
                var current = Enumerable.Range(4 * lamp, 4).Select(i => start[i]).ToArray();
                for (int axis = 0; axis < 4; ++axis)
                {
                    var probe = native.points[axis];
                    if (probe.stencil == 0) continue;
                    for (int i = 0; i < fit.Size; ++i)
                        models[0][axis][i] = Col(i, probe.hp[i], probe.hm[i], probe);
                    if (axis == 3)
                    {
                        for (int j = 1; j < models.Length; ++j) Array.Copy(models[0][axis], models[j][axis], fit.Size);
                        continue;
                    }
                    E25ReferenceColumn(fit, native.hard, native.off, probe.op, probe.om, null, probe.d, probe.stencil, models[1][axis]);
                    E25ReferenceColumn(fit, native.hard, native.off, probe.op, probe.om, native.mask, probe.d, probe.stencil, models[2][axis]);
                    var moved = (double[])start.Clone();
                    moved[4 * lamp + axis] += probe.d;
                    var vp = Visibility(moved);
                    float[] vm = null;
                    if (probe.stencil != 1)
                    {
                        moved[4 * lamp + axis] = start[4 * lamp + axis] + (probe.stencil == 3 ? -probe.d : 2 * probe.d);
                        vm = Visibility(moved);
                    }
                    for (int i = 0; i < fit.Size; ++i)
                    {
                        int p = index[i / 3];
                        if (p < 0) { models[3][axis][i] = models[2][axis][i]; continue; }
                        var contribution = native.io[i] / LogPowerSpan;
                        float a = Proxy(vp[p], probe.op[i]);
                        float b = vm == null ? 0 : Proxy(vm[p], probe.om[i]);
                        models[3][axis][i] = Col(i, a, b, probe);
                        float Proxy(float v, float off) => (float)(native.hard[i] + v * (off - native.off[i]) + (v - center[p]) * contribution);
                    }
                }
                var rgb = new float[fit.Size];
                double anchorLoss = fit.Evaluate(native.hard);
                using var compare = Csv("moment-direction.csv", "scenario,lamp,model,fraction,repeat,shrinks,predicted,actual,rho,legal,camera_renders,shadow_faces");
                for (int j = 0; j < names.Length; ++j)
                {
                    var (g, grad) = E25Normal(models[j], native.residual, 0, fit.Size);
                    double[] DeltaPose(double[] selected)
                    {
                        var rig = (double[])start.Clone(); Array.Copy(selected, 0, rig, 4 * lamp, 4); return rig;
                    }
                    var step = ImageJacobianLm.LampScaledTrustProposal(g, grad, current, units, q => room.Contains(DeltaPose(q)), out int shrinks);
                    foreach (double scale in new[] { .25, .5, 1d })
                    {
                        var proposed = DeltaPose(current.Zip(step, (x, d) => x + d * scale).ToArray());
                        double predicted = E25Predicted(g, grad, step.Select(x => x * scale).ToArray());
                        bool legal = room.Contains(proposed);
                        for (int rep = 1; rep <= 2; ++rep)
                        {
                            double actual = double.NaN;
                            if (legal) { E25Capture(proposed, LightShadows.Hard, E25DefaultViews, rgb); actual = anchorLoss - fit.Evaluate(rgb); }
                            compare.WriteLine(string.Join(",", scenario, lamp + 1, names[j], F(scale), rep, shrinks, F(predicted), F(actual),
                                F(SafeRatio(actual, predicted)), Flag(legal), legal ? E19FitViews : 0, names[j] == "C4-MBD-delta" ? shadow.Faces : 0));
                        }
                    }
                }
                Gate(coverage, error, true, "matching_native_candidate_recaptures");

                float[] Visibility(double[] pose)
                {
                    int before = shadow.Faces;
                    var v = shadow.Render(room.PositionOf(pose, lamp));
                    shadowFaces += shadow.Faces - before;
                    return v;
                }
                float Col(int i, float a, float b, E25Probes p) => (float)(fit.weights[i / E23ChannelsPerView] *
                    Difference(fit.Transform(native.hard[i]), fit.Transform(a), p.stencil == 1 ? 0 : fit.Transform(b), p.d, p.stencil));
                void Gate(double cov, double err, bool test, string reason) => gate.WriteLine(string.Join(",", scenario, lamp + 1, pixels.Count,
                    F(cov), F(err), Flag(test), Flag(test), shadowFaces - facesBefore, geometryPasses - passesBefore, low.RenderCount - camerasBefore,
                    F(timer.Elapsed.TotalSeconds), reason));
            }
        }
    }
}

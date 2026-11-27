using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;
using Frame = LightingMvp.NativeRenderBackend.Frame;
using Object = UnityEngine.Object;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            private static readonly double[] E22TopFractions = { .001, .01, .05 };

            private void E22Hotspots()
            {
                var screen = E22ReadOwn("e22_b_screen") ?? throw new InvalidDataException("Run b-screen before hotspot diagnostics.");
                var geometry = CaptureE22Geometry();
                var gt = E22Target(false); double scale = E22Scale(gt);
                int n = E19FitW * E19FitH;
                var maps = E22Mappings.ToDictionary(key => key, key => gt.fit.Select(f => new E22FrameLoss(f, key, scale)).ToArray());
                using var hotspots = Csv("hotspots.csv", "checkpoint,view,scheme,loss,top_0_1pct,top_1pct,top_5pct,peak_pixel,peak_value,target_rgb_energy,candidate_rgb_energy");
                using var mappingStats = Csv("mapping-stats.csv", "checkpoint,view,scheme,s,exposure,negative_channel_fraction,fixed_display_saturated_0_99,fixed_display_dark_0_01,transform_gain_p90,hotspot_gain_median");
                using var locations = Csv("hotspot-surfaces.csv", "checkpoint,view,pixel,top_masks,reliable,renderer,material,world_x,world_y,world_z,nx,ny,nz,boundary,nearest_lamp,nearest_m,raw_pixel_sq_error,display_pixel_sq_error,footprint_sq_error,specular_delta_rgb,shadow_delta_rgb,marginal_rgb_norms");
                using var jacobians = Csv("hotspot-jacobians.csv", "checkpoint,scheme,region,rank,condition,singular_values,gradient,gradient_fraction,gradient_cosine,gram_fraction,actual_step_m_or_log");
                using var grams = Csv("hotspot-gram.csv", "checkpoint,scheme,region,row,values");
                using var costs = Csv("diagnostic-costs.csv", "checkpoint,purpose,camera_renders,seconds");
                string firstFailure = screen.arms.FirstOrDefault(a => !a.pass)?.id;
                var gtPath = CapturePath("hotspot_gt");
                for (int v = 0; v < E19FitViews; ++v)
                {
                    NativeRenderBackend.Save(gt.fit[v], E19FitW, E19FitH, Path.Combine(gtPath, $"gt_v{v}"));
                    RecordDisplayStats("gt", "B0", v, gt.fit[v], null, Array.Empty<int>());
                }
                foreach (var start in screen.arms.GroupBy(a => a.scenario).Select(g => g.First()))
                    Analyze(start.scenario + "_start", start.initial, E22Mappings, !smoke);
                foreach (var arm in smoke ? screen.arms.Take(2) : screen.arms)
                    Analyze(arm.id, arm.final, arm.method == "B0" ? new[] { "B0" } : new[] { "B0", arm.method }, arm.id == firstFailure && !smoke);

                void Analyze(string id, double[] rig, string[] modes, bool audit)
                {
                    var folder = CapturePath(id);
                    var frames = Measure("mixed_448", () => CaptureRig(rig, false, low, E19FitViews));
                    var details = Measure("mixed_896", () => CaptureRig(rig, false, high, E19FitViews));
                    var shadowed = Measure("mixed_shadow_on", () => CaptureRig(rig, true, low, E19FitViews));
                    var specularOff = Measure("specular_off", () => CaptureWithoutSpecular(rig, geometry.renderers));
                    var ablations = Measure("leave_one_lamp_out", () => room.Lamps.Select(lamp => CaptureWithoutLamp(rig, lamp)).ToArray());
                    var rawMasks = new byte[E19FitViews][];
                    for (int v = 0; v < E19FitViews; ++v)
                    {
                        NativeRenderBackend.Save(frames[v], E19FitW, E19FitH, Path.Combine(folder, $"rgb_v{v}"));
                        var raw = new double[n]; var display = new double[n];
                        double referenceEnergy = gt.fit[v].Energy;
                        for (int p = 0; p < n; ++p)
                        {
                            raw[p] = Norm(ProxyGeometry.Rgb(frames[v].Pixel(p) - gt.fit[v].Pixel(p)));
                            display[p] = Norm(DisplayRgb(frames[v].Pixel(p)) - displayTargets[false].low[v].pixels[p]);
                        }
                        var masks = new Dictionary<string, byte[]>();
                        foreach (string mode in modes)
                        {
                            var errors = new double[n];
                            maps[mode][v].Evaluate(frames[v], null, 0, errors);
                            masks.Add(mode, Record(id, mode, v, errors, referenceEnergy, frames[v].Energy, folder));
                        }
                        rawMasks[v] = masks["B0"];
                        var shown = displayTargets[false].low[v];
                        double shownCurrent = 0;
                        for (int p = 0; p < n; ++p) shownCurrent += Norm(DisplayRgb(frames[v].Pixel(p)));
                        Record(id, "display", v, display, shown.energy, shownCurrent, null);
                        SaveErrorMap(folder, $"raw_error_v{v}", raw, E19FitW, E19FitH, Math.Sqrt(referenceEnergy / n));
                        SaveErrorMap(folder, $"display_error_v{v}", display, E19FitW, E19FitH, Math.Sqrt(shown.energy / n));
                        var top = Enumerable.Range(0, n).Where(p => rawMasks[v][p] >= 2).ToArray();
                        foreach (string mode in modes) RecordDisplayStats(id, mode, v, frames[v], gt.fit[v], top);
                        var map = geometry.views[v];
                        for (int p = 0; p < n; ++p)
                        {
                            if (masks.All(x => x.Value[p] == 0)) continue;
                            bool known = map.reliable[p];
                            var world = map.position[p]; var normal = map.normal[p];
                            int nearest = -1; double distance = -1;
                            if (known)
                            {
                                distance = double.PositiveInfinity;
                                foreach (int lamp in room.Lamps)
                                {
                                    double d = Vector3.Distance(world, room.PositionOf(rig, lamp));
                                    if (d < distance) { distance = d; nearest = lamp; }
                                }
                            }
                            var rgb = frames[v].Pixel(p);
                            var footprint = BoxPixel(details[v], p) - BoxPixel(gt.high[v], p);
                            var marginal = ablations.Select(a => Math.Sqrt(Norm(ProxyGeometry.Rgb(rgb - a[v].Pixel(p)))));
                            locations.WriteLine(string.Join(",", id, v, p, string.Join(";", masks.Where(k => k.Value[p] != 0).Select(k => k.Key + ":" + k.Value[p])),
                                Flag(known), known ? map.renderer[p] : 0, known ? map.material[p] : 0,
                                Xyz(known ? world : Vector3.zero), Xyz(known ? normal : Vector3.zero), Flag(known && map.boundary[p]), nearest + 1, F(distance),
                                F(raw[p]), F(display[p]), F(Norm(ProxyGeometry.Rgb(footprint))), F(Math.Sqrt(Norm(ProxyGeometry.Rgb(rgb - specularOff[v].Pixel(p))))),
                                F(Math.Sqrt(Norm(ProxyGeometry.Rgb(rgb - shadowed[v].Pixel(p))))), Series(marginal)));
                        }
                    }
                    if (audit) Measure("shared_fd_jacobian", () => { AuditHotspotJacobian(id, rig, frames, rawMasks, maps, jacobians, grams); return true; });
                    UnityEngine.Debug.Log($"[E22-B-HOTSPOTS] Diagnosed {id}");

                    T Measure<T>(string purpose, Func<T> call)
                    {
                        int before = low.RenderCount + high.RenderCount;
                        var timer = Stopwatch.StartNew(); var result = call();
                        costs.WriteLine(string.Join(",", id, purpose, low.RenderCount + high.RenderCount - before, F(timer.Elapsed.TotalSeconds)));
                        costs.Flush();
                        return result;
                    }
                }

                byte[] Record(string id, string scheme, int view, double[] values, double energy, double candidate, string folder)
                {
                    var order = Enumerable.Range(0, n).ToArray();
                    double[] sorted = (double[])values.Clone();
                    Array.Sort(sorted, order);
                    var mask = new byte[n];
                    var totals = new double[3];
                    double sum = values.Sum();
                    for (int j = 0; j < E22TopFractions.Length; ++j)
                    for (int i = 0; i < (int)Math.Ceiling(n * E22TopFractions[j]); ++i)
                    {
                        int k = n - 1 - i;
                        totals[j] += sorted[k];
                        if (sorted[k] > 0 && mask[order[k]] == 0) mask[order[k]] = (byte)(3 - j);
                    }
                    hotspots.WriteLine(string.Join(",", id, view, scheme, F(sum), string.Join(",", totals.Select(x => F(sum == 0 ? 0 : x / sum))),
                        order[n - 1], F(sorted[n - 1]), F(energy), F(candidate)));
                    if (folder != null) SaveHotspotMask(folder, $"{scheme}_top_v{view}", mask);
                    return mask;
                }

                void RecordDisplayStats(string id, string key, int view, Frame frame, Frame reference, int[] top)
                {
                    int negative = 0, saturated = 0, dark = 0;
                    for (int p = 0; p < n; ++p)
                    {
                        var rgb = frame.Pixel(p);
                        for (int ch = 0; ch < 3; ++ch)
                        {
                            float val = rgb[ch], shown = NativeRenderBackend.Display(val);
                            if (val < 0) ++negative;
                            if (shown >= .99f) ++saturated;
                            if (shown <= .01f) ++dark;
                        }
                    }
                    var mode = E22Mapping(key);
                    double derivative = Slope((float)scale), robustDelta = reference == null ? 0 : .25 * Math.Sqrt(reference.Energy / n);
                    var gain = top.Select(p =>
                    {
                        var rgb = frame.Pixel(p);
                        if (key == "B5") return Math.Min(1, robustDelta / Math.Max(1e-12, Math.Sqrt(Norm(ProxyGeometry.Rgb(rgb - reference.Pixel(p))))));
                        return Slope(Mathf.Max(rgb.r, Mathf.Max(rgb.g, rgb.b))) / Math.Max(1e-12, derivative);
                    }).ToArray();
                    mappingStats.WriteLine(string.Join(",", id, view, key, F(scale), 1, F(negative / (3d * n)), F(saturated / (3d * n)), F(dark / (3d * n)),
                        F(derivative), F(gain.Length == 0 ? -1 : Median(gain))));

                    double Slope(float at)
                    {
                        float h = Mathf.Max(1e-6f, Mathf.Abs(at) * 1e-3f);
                        return (NativeRenderBackend.Map(at + h, mode, scale) - NativeRenderBackend.Map(at - h, mode, scale)) / (2d * h);
                    }
                }
            }

            private Frame[] CaptureRig(double[] rig, bool shadows, NativeRenderBackend backend, int views)
            {
                Apply(rig, shadows);
                return Enumerable.Range(0, views).Select(backend.Capture).ToArray();
            }

            private Frame[] CaptureWithoutLamp(double[] rig, int lamp)
            {
                Apply(rig, false);
                lights[lamp].enabled = false;
                try { return Enumerable.Range(0, E19FitViews).Select(low.Capture).ToArray(); }
                finally { lights[lamp].enabled = true; }
            }

            private Frame[] CaptureWithoutSpecular(double[] rig, MeshRenderer[] renderers)
            {
                var originals = renderers.Select(r => r.sharedMaterials).ToArray();
                var copies = new Dictionary<Material, Material>();
                try
                {
                    for (int i = 0; i < renderers.Length; ++i)
                    {
                        var mats = (Material[])originals[i].Clone();
                        for (int slot = 0; slot < mats.Length; ++slot)
                        {
                            var source = mats[slot];
                            if (!source || !source.HasProperty("_SpecularHighlights")) continue;
                            if (!copies.TryGetValue(source, out var clone))
                            {
                                clone = new Material(source) { hideFlags = HideFlags.HideAndDontSave };
                                clone.SetFloat("_SpecularHighlights", 0); clone.EnableKeyword("_SPECULARHIGHLIGHTS_OFF");
                                copies.Add(source, clone);
                            }
                            mats[slot] = clone;
                        }
                        renderers[i].sharedMaterials = mats;
                    }
                    return CaptureRig(rig, false, low, E19FitViews);
                }
                finally
                {
                    for (int i = 0; i < renderers.Length; ++i) renderers[i].sharedMaterials = originals[i];
                    foreach (var material in copies.Values) Object.DestroyImmediate(material);
                }
            }

            private static void SaveHotspotMask(string folder, string label, byte[] mask)
            {
                var texture = new Texture2D(E19FitW, E19FitH, TextureFormat.RGB24, false, true);
                try
                {
                    var palette = new[] { new Color32(15, 19, 30, 255), new Color32(87, 149, 217, 255), new Color32(255, 199, 78, 255), new Color32(248, 70, 56, 255) };
                    texture.SetPixels32(mask.Select(k => palette[k]).ToArray()); texture.Apply();
                    File.WriteAllBytes(Path.Combine(folder, label + ".png"), texture.EncodeToPNG());
                }
                finally { Object.DestroyImmediate(texture); }
            }

            private void AuditHotspotJacobian(string id, double[] rig, Frame[] centre, byte[][] mask, Dictionary<string, E22FrameLoss[]> maps,
                StreamWriter log, StreamWriter matrices)
            {
                int axes = rig.Length, pixels = centre[0].PixelCount, size = E19FitViews * 3 * pixels;
                var stencil = new (Frame[] p, Frame[] m, double a, double b, double c, double physical)[axes];
                for (int j = 0; j < axes; ++j)
                {
                    double h = (j % 4 == 3 ? .04 : .02) / units[j % 4];
                    for (int attempt = 0; attempt < 8; ++attempt, h *= .5)
                    {
                        var p = Shift(j, h); var m = Shift(j, -h);
                        bool plus = room.Contains(p), minus = room.Contains(m);
                        if (!plus && !minus) continue;
                        if (plus && minus)
                            stencil[j] = (CaptureRig(p, false, low, E19FitViews), CaptureRig(m, false, low, E19FitViews), .5 / h, -.5 / h, 0, h * units[j % 4]);
                        else
                        {
                            double d = plus ? h : -h;
                            var second = Shift(j, 2 * d); bool usable = room.Contains(second);
                            stencil[j] = (CaptureRig(plus ? p : m, false, low, E19FitViews), usable ? CaptureRig(second, false, low, E19FitViews) : null,
                                usable ? 2 / d : 1 / d, usable ? -.5 / d : 0, usable ? -1.5 / d : -1 / d, d * units[j % 4]);
                        }
                        break;
                    }
                }
                var regions = new int[3][];
                for (int threshold = 1; threshold <= 3; ++threshold)
                    regions[3 - threshold] = Enumerable.Range(0, E19FitViews * pixels).Where(i => mask[i / pixels][i % pixels] >= threshold).Select(i => 3 * i).ToArray();
                var columns = Enumerable.Range(0, axes).Select(i => new float[size]).ToArray();
                var residual = new float[size]; var extra = new float[size];
                foreach (var method in maps)
                {
                    Write(centre, residual);
                    for (int j = 0; j < axes; ++j)
                    {
                        var s = stencil[j]; var col = columns[j];
                        if (s.p == null) { Array.Clear(col, 0, col.Length); continue; }
                        Write(s.p, col);
                        if (s.m != null) Write(s.m, extra);
                        float a = (float)s.a, b = (float)s.b, c = (float)s.c;
                        for (int i = 0; i < size; ++i) col[i] = a * col[i] + b * extra[i] + c * residual[i];
                    }
                    var g = new double[4, axes]; var h = Enumerable.Range(0, 4).Select(_ => new double[axes, axes]).ToArray();
                    Parallel.For(0, axes, new ParallelOptions { MaxDegreeOfParallelism = Math.Min(Environment.ProcessorCount, 8) }, j =>
                    {
                        g[0, j] = E21Dot(columns[j], residual);
                        for (int r = 1; r < 4; ++r) g[r, j] = Masked(columns[j], residual, regions[r - 1]);
                        for (int k = 0; k <= j; ++k)
                        {
                            h[0][j, k] = h[0][k, j] = E21Dot(columns[j], columns[k]);
                            for (int r = 1; r < 4; ++r) h[r][j, k] = h[r][k, j] = Masked(columns[j], columns[k], regions[r - 1]);
                        }
                    });
                    double fullG = Math.Sqrt(Enumerable.Range(0, axes).Sum(j => g[0, j] * g[0, j]));
                    double fullH = Math.Sqrt(h[0].Cast<double>().Sum(x => x * x));
                    for (int r = 0; r < 4; ++r)
                    {
                        string region = r == 0 ? "full" : new[] { "raw_top_0_1", "raw_top_1", "raw_top_5" }[r - 1];
                        var sp = ClassroomDiagnosis.GetSpectrum(h[r], axes);
                        double ng = Math.Sqrt(Enumerable.Range(0, axes).Sum(j => g[r, j] * g[r, j]));
                        double nh = Math.Sqrt(h[r].Cast<double>().Sum(x => x * x));
                        double cosine = fullG * ng > 0 ? Enumerable.Range(0, axes).Sum(j => g[r, j] * g[0, j]) / (fullG * ng) : double.NaN;
                        log.WriteLine(string.Join(",", id, method.Key, region, sp.rank, F(sp.condition), Series(sp.singular), Series(Enumerable.Range(0, axes).Select(j => g[r, j])),
                            F(ng / Math.Max(1e-18, fullG)), F(cosine), F(nh / Math.Max(1e-18, fullH)), Series(stencil.Select(s => s.physical))));
                        for (int j = 0; j < axes; ++j)
                            matrices.WriteLine(string.Join(",", id, method.Key, region, j, Series(Enumerable.Range(0, axes).Select(k => h[r][j, k]))));
                    }

                    void Write(Frame[] f, float[] result)
                    {
                        for (int v = 0; v < E19FitViews; ++v) method.Value[v].Evaluate(f[v], result, v * pixels * 3);
                    }
                }

                double[] Shift(int j, double h) { var x = (double[])rig.Clone(); x[j] += h; return x; }
                static double Masked(float[] a, float[] b, int[] region)
                {
                    double sum = 0;
                    foreach (int i in region) sum += (double)a[i] * b[i] + (double)a[i + 1] * b[i + 1] + (double)a[i + 2] * b[i + 2];
                    return sum;
                }
            }
        }
    }
}

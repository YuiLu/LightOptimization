using System;
using System.IO;
using System.Linq;
using UnityEngine;
using Object = UnityEngine.Object;
using Frame = LightingMvp.NativeRenderBackend.Frame;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            private enum RgbObjective { Huber, Charbonnier, Footprint }

            internal void RgbHotspots()
            {
                var target = Target(false);
                var lamps = Enumerable.Range(0, room.Count).ToArray();
                using var hotspots = Csv("hotspots.csv",
                    "scenario,method,checkpoint,measurement,view,role,pixels,squared_nrmse,nrmse,top_0_1pct,top_1pct,top_5pct,peak_x,peak_y,peak_abs_rgb");
                using var thresholds = Csv("rgb-objective-config.csv", "view,gt_rms_rgb,delta_rgb,delta_relative");
                for (int v = 0; v < E19FitViews; ++v)
                {
                    double rms = Math.Sqrt(target.fit[v].Energy / target.fit[v].PixelCount);
                    thresholds.WriteLine(string.Join(",", v, F(rms), F(rms * .25), F(.25)));
                }
                foreach (int seed in smoke ? new[] { 131 } : new[] { 131, 157 })
                {
                    string key = $"seed{seed}_near";
                    var start = E19Start(room, room.Start(seed), .45f);
                    Apply(start, false);
                    var initialHigh = Enumerable.Range(0, room.Views.Length).Select(high.Capture).ToArray();
                    Diagnose(key, "start", "initial", initialHigh, target, CapturePath(key + "_start"));
                    SaveFrames(initialHigh, CapturePath(key + "_start"), "initial", true);
                    var modes = new[] { ("L2_448", NativeSignal(target)),
                        ("Huber_448", RobustSignal(target, RgbObjective.Huber)),
                        ("Charbonnier_448", RobustSignal(target, RgbObjective.Charbonnier)),
                        ("L2_896", NativeSignal(target, highFit: true)),
                        ("Footprint_2x2", RobustSignal(target, RgbObjective.Footprint)) };
                    foreach (var mode in smoke ? modes.Take(2) : modes)
                        RunArm(key, mode.Item1, lamps, start, target, mode.Item2, 520, .02,
                            validation: (a, frames, gt, path) => Diagnose(key, a.method, "best", frames, gt, path));
                }

                void Diagnose(string scenario, string method, string checkpoint, Frame[] highFrames, MbdRgbTarget gt, string path)
                {
                    for (int v = 0; v < highFrames.Length; ++v)
                    {
                        int view = v;
                        Analyze(highFrames[v].Pixel, gt.high[v].Pixel, highFrames[v].PixelCount, E19HighW, E19HighH, gt.high[v].Energy,
                            "native_896", writeMap: v == 0 || v == 1 || v == 3);
                        if (v >= E19FitViews) continue;
                        var image = low.Capture(v);
                        Analyze(image.Pixel, gt.fit[v].Pixel, image.PixelCount, E19FitW, E19FitH, gt.fit[v].Energy, "native_448", false);
                        double energy = 0;
                        for (int i = 0; i < image.PixelCount; ++i)
                        {
                            var color = BoxPixel(gt.high[v], i);
                            energy += (double)color.r * color.r + (double)color.g * color.g + (double)color.b * color.b;
                        }
                        Analyze(i => BoxPixel(highFrames[v], i), i => BoxPixel(gt.high[v], i), image.PixelCount, E19FitW, E19FitH,
                            energy, "box_896_to_448", false);

                        void Analyze(Func<int, Color> current, Func<int, Color> reference, int pixels, int width, int height,
                            double energy, string name, bool writeMap)
                        {
                            var errors = new double[pixels];
                            double total = 0;
                            int peak = 0;
                            for (int i = 0; i < pixels; ++i)
                            {
                                var difference = current(i) - reference(i);
                                double value = (double)difference.r * difference.r + (double)difference.g * difference.g + (double)difference.b * difference.b;
                                errors[i] = value;
                                total += value;
                                if (value > errors[peak]) peak = i;
                            }
                            double peakValue = Math.Sqrt(errors[peak]);
                            if (writeMap) SaveErrorMap(path, $"error_v{view}", errors, width, height, Math.Sqrt(energy / pixels));
                            Array.Sort(errors);
                            double Fraction(double part)
                            {
                                if (total == 0) return 0;
                                int count = (int)Math.Ceiling(pixels * part);
                                double tail = 0;
                                for (int i = pixels - count; i < pixels; ++i) tail += errors[i];
                                return tail / total;
                            }
                            double loss = total / Math.Max(1e-12, energy);
                            hotspots.WriteLine(string.Join(",", scenario, method, checkpoint, name, view, view < E19FitViews ? "train" : "held", pixels,
                                F(loss), F(Math.Sqrt(loss)), F(Fraction(.001)), F(Fraction(.01)), F(Fraction(.05)), peak % width, peak / width, F(peakValue)));
                        }
                    }
                    hotspots.Flush();
                }
            }

            private E21Signal RobustSignal(MbdRgbTarget target, RgbObjective model)
            {
                int n = E19FitW * E19FitH;
                var references = new Color[E19FitViews][];
                var norms = new double[E19FitViews];
                var delta = new double[E19FitViews];
                bool footprint = model == RgbObjective.Footprint;
                for (int v = 0; v < E19FitViews; ++v)
                {
                    references[v] = new Color[n];
                    double energy = 0;
                    for (int i = 0; i < n; ++i)
                    {
                        Color c = references[v][i] = footprint ? BoxPixel(target.high[v], i) : target.fit[v].Pixel(i);
                        energy += (double)c.r * c.r + (double)c.g * c.g + (double)c.b * c.b;
                    }
                    norms[v] = 1 / (E19FitViews * Math.Max(1e-12, energy));
                    delta[v] = Math.Max(1e-8, .25 * Math.Sqrt(energy / n));
                }
                return new E21Signal(footprint ? "native_l2_footprint_2x2" : "native_" + model.ToString().ToLowerInvariant() + "_448", E19FitViews * 3 * n,
                    (theta, residual) =>
                    {
                        Apply(theta, target.shadows);
                        double value = 0;
                        for (int v = 0; v < E19FitViews; ++v)
                        {
                            var frame = (footprint ? high : low).Capture(v);
                            double breakpoint = delta[v], thresholdSquared = breakpoint * breakpoint;
                            for (int i = 0, k = v * n * 3; i < n; ++i, k += 3)
                            {
                                var d = (footprint ? BoxPixel(frame, i) : frame.Pixel(i)) - references[v][i];
                                double square = (double)d.r * d.r + (double)d.g * d.g + (double)d.b * d.b;
                                double cost = square;
                                if (model == RgbObjective.Huber && square > thresholdSquared)
                                    cost = 2 * breakpoint * Math.Sqrt(square) - thresholdSquared;
                                else if (model == RgbObjective.Charbonnier)
                                    cost = 2 * square / (Math.Sqrt(1 + square / thresholdSquared) + 1);
                                value += cost * norms[v];
                                if (residual != null)
                                {
                                    float gain = square > 0 ? (float)Math.Sqrt(norms[v] * (cost / square)) : (float)Math.Sqrt(norms[v]);
                                    residual[k] = d.r * gain; residual[k + 1] = d.g * gain; residual[k + 2] = d.b * gain;
                                }
                            }
                        }
                        return value;
                    });
            }

            private static Color BoxPixel(Frame highFrame, int localIndex)
            {
                int x = 2 * (localIndex % E19FitW);
                int y = 2 * (localIndex / E19FitW);
                int i = y * E19HighW + x;
                return .25f * (highFrame.Pixel(i) + highFrame.Pixel(i + 1) + highFrame.Pixel(i + E19HighW) + highFrame.Pixel(i + E19HighW + 1));
            }

            private static void SaveErrorMap(string directory, string name, double[] values, int width, int height, double rms)
            {
                var texture = new Texture2D(width, height, TextureFormat.RGB24, false, true);
                try
                {
                    var colors = new Color[values.Length];
                    for (int i = 0; i < values.Length; ++i)
                    {
                        float s = Mathf.Clamp01((float)Math.Log10(1 + Math.Sqrt(values[i]) / Math.Max(1e-8, rms)) / 3);
                        colors[i] = new Color(s, s * s, s * s * s);
                    }
                    texture.SetPixels(colors);
                    texture.Apply();
                    File.WriteAllBytes(Path.Combine(directory, name + ".png"), texture.EncodeToPNG());
                }
                finally { Object.DestroyImmediate(texture); }
            }
        }
    }
}

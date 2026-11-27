using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using Frame = LightingMvp.NativeRenderBackend.Frame;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            [Serializable] private sealed class E22Linearity
            {
                public bool shadows, pass;
                public int comparisons, nearZero, cameras;
                public double median, worst, seconds;
            }

            private void E22OlatAudit()
            {
                using var errors = Csv("olat-linearity.csv", "shadows,pose,view,test,active,factor,signal_norm,error_norm,relative_error,hotspot_relative_error,near_zero,powers");
                using var atlas = Csv("olat-atlas.csv", "shadows,pose,view,phase,active,width,height,slices,offset_resolution,submit_s,readback_s");
                using var spectra = Csv("olat-spectrum.csv", "shadows,pose,view,rank,condition,singular_values");
                using var poses = Csv("olat-poses.csv", "pose,theta");
                int[] all = room.Lamps;
                var cases = new[] { ("gt", room.Truth), ("seed733_near", E22Start(733, all)), ("seed131_near", E22Start(131, all)) };
                foreach (var item in cases) poses.WriteLine(string.Join(",", item.Item1, Series(item.Item2)));
                var byShadow = new List<E22Linearity>();
                foreach (bool shadows in new[] { false, true })
                {
                    int before = olatFrames, small = 0;
                    var timer = Stopwatch.StartNew();
                    var finite = new List<double>();
                    foreach (var item in smoke ? cases.Take(1) : cases)
                    {
                        string pose = item.Item1;
                        var rig = item.Item2;
                        for (int v = 0; v < (smoke ? 1 : room.Views.Length); ++v)
                        {
                            var empty = Render("empty", Array.Empty<int>(), Array.Empty<double>());
                            var response = new float[lights.Length][];
                            foreach (int lamp in all)
                            {
                                var unit = Render("unit", new[] { lamp }, new[] { 1d });
                                response[lamp] = Increment(unit, empty);
                            }
                            var gram = new double[lights.Length, lights.Length];
                            for (int l = 0; l < lights.Length; ++l)
                            for (int k = 0; k <= l; ++k)
                                gram[l, k] = gram[k, l] = E21Dot(response[l], response[k]);
                            var spectrum = ClassroomDiagnosis.GetSpectrum(gram, lights.Length);
                            spectra.WriteLine(string.Join(",", Flag(shadows), pose, v, spectrum.rank, F(spectrum.condition), Series(spectrum.singular)));
                            foreach (int lamp in all)
                            foreach (double factor in new[] { .5, 1, 2 })
                                Compare("intensity", new[] { lamp }, new[] { factor }, factor);
                            foreach (var mixture in new[] { new[] { 4, 6 }, new[] { 1, 6 }, all })
                                Compare("mixture", mixture, mixture.Select(l => room.PowerOf(rig, l)).ToArray(), 1);

                            Frame Render(string phase, int[] selected, double[] powers)
                            {
                                double submit = low.RenderSeconds, copy = low.ReadbackSeconds;
                                var frame = CaptureOlat(v, rig, shadows, selected, powers);
                                var layout = shadows && selected.Length > 0 ? E22Atlas.Snapshot(room.Camera) : (0, 0, 0, "");
                                atlas.WriteLine(string.Join(",", Flag(shadows), pose, v, phase, LampList(selected), layout.Item1, layout.Item2,
                                    layout.Item3, layout.Item4, F(low.RenderSeconds - submit), F(low.ReadbackSeconds - copy)));
                                return frame;
                            }

                            void Compare(string test, int[] selected, double[] weights, double factor)
                            {
                                var actual = Increment(Render(test + "_" + F(factor), selected, weights), empty);
                                var prediction = new float[actual.Length];
                                for (int k = 0; k < selected.Length; ++k)
                                {
                                    var r = response[selected[k]]; float w = (float)weights[k];
                                    for (int i = 0; i < prediction.Length; ++i) prediction[i] += w * r[i];
                                }
                                int n = actual.Length / 3;
                                var signal = new double[n]; var error = new double[n];
                                double ss = 0, ee = 0;
                                for (int p = 0, k = 0; p < n; ++p, k += 3)
                                {
                                    double x = actual[k] - prediction[k], y = actual[k + 1] - prediction[k + 1], z = actual[k + 2] - prediction[k + 2];
                                    signal[p] = (double)actual[k] * actual[k] + (double)actual[k + 1] * actual[k + 1] + (double)actual[k + 2] * actual[k + 2];
                                    error[p] = x * x + y * y + z * z; ss += signal[p]; ee += error[p];
                                }
                                Array.Sort(signal, error);
                                double localSignal = 0, localError = 0;
                                for (int p = n - (int)Math.Ceiling(.01 * n); p < n; ++p) { localSignal += signal[p]; localError += error[p]; }
                                double norm = Math.Sqrt(ss), relative = Math.Sqrt(ee / Math.Max(1e-24, ss));
                                bool near = norm <= NearSignal;
                                if (near) ++small; else finite.Add(relative);
                                errors.WriteLine(string.Join(",", Flag(shadows), pose, v, test, LampList(selected), F(factor), F(norm), F(Math.Sqrt(ee)),
                                    F(relative), F(Math.Sqrt(localError / Math.Max(1e-24, localSignal))), Flag(near), Series(weights)));
                            }
                        }
                    }
                    double median = finite.Count == 0 ? -1 : Median(finite), worst = finite.Count == 0 ? -1 : finite.Max();
                    var outcome = new E22Linearity { shadows = shadows, comparisons = finite.Count, nearZero = small,
                        median = median, worst = worst, pass = finite.Count > 0 && median <= .02 && worst <= .05,
                        cameras = olatFrames - before, seconds = timer.Elapsed.TotalSeconds };
                    byShadow.Add(outcome);
                    UnityEngine.Debug.Log($"[E22-D-AUDIT] shadows={shadows}, {finite.Count} comparisons, median={median:P3}, max={worst:P3}, pass={outcome.pass}");
                }
                WriteJson(directory, "olat-gate.json", new E22LinearityReport { states = byShadow.ToArray() });
            }

            [Serializable] private sealed class E22LinearityReport { public E22Linearity[] states; }

            private Frame CaptureOlat(int view, double[] rig, bool shadows, int[] selected, double[] powers)
            {
                Apply(rig, shadows);
                foreach (var lamp in lights) lamp.enabled = false;
                for (int i = 0; i < selected.Length; ++i)
                {
                    if (powers[i] < PowerMin || powers[i] > PowerMax) throw new InvalidOperationException("Illegal OLAT intensity.");
                    lights[selected[i]].intensity = (float)powers[i];
                    lights[selected[i]].enabled = true;
                }
                ++olatFrames;
                return low.Capture(view);
            }

            private static float[] Increment(Frame f, Frame baseline)
            {
                var data = new float[f.PixelCount * 3];
                for (int i = 0, k = 0; i < f.PixelCount; ++i)
                {
                    var delta = f.Pixel(i) - baseline.Pixel(i);
                    data[k++] = delta.r; data[k++] = delta.g; data[k++] = delta.b;
                }
                return data;
            }

            private static class E22Atlas
            {
                private const BindingFlags Members = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
                private static readonly FieldInfo Pass = typeof(UniversalRenderer).GetField("m_AdditionalLightsShadowCasterPass", Members);

                internal static (int, int, int, string) Snapshot(Camera sceneCamera)
                {
                    if (!(sceneCamera.GetUniversalAdditionalCameraData().scriptableRenderer is UniversalRenderer renderer)) return (-1, -1, -1, "");
                    var pass = Pass?.GetValue(renderer);
                    if (pass == null) return (-1, -1, -1, "");
                    var type = pass.GetType();
                    var handle = type.GetField("m_AdditionalLightsShadowmapHandle", Members)?.GetValue(pass) as RTHandle;
                    var active = type.GetField("m_ShadowSliceToAdditionalLightIndex", Members)?.GetValue(pass) as List<int>;
                    var slices = type.GetField("m_AdditionalLightsShadowSlices", Members)?.GetValue(pass) as Array;
                    if (active == null || slices == null) return (handle?.rt?.width ?? -1, handle?.rt?.height ?? -1, -1, "");
                    var element = slices.GetType().GetElementType();
                    var x = element.GetField("offsetX", Members); var y = element.GetField("offsetY", Members); var size = element.GetField("resolution", Members);
                    var layout = new string[Math.Min(active.Count, slices.Length)];
                    for (int i = 0; i < layout.Length; ++i)
                    {
                        var slice = slices.GetValue(i);
                        layout[i] = $"{active[i]}:{x?.GetValue(slice)}:{y?.GetValue(slice)}:{size?.GetValue(slice)}";
                    }
                    return (handle?.rt?.width ?? 0, handle?.rt?.height ?? 0, active.Count, string.Join(";", layout));
                }
            }
        }
    }
}

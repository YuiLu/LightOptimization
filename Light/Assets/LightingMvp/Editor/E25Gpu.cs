using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using Mapping = LightingMvp.NativeRenderBackend.HdrMapping;
using Object = UnityEngine.Object;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class E21RoomStudy
        {
            // Opt-in D2 path for the existing CPU-capture/CPU-normal-equation architecture.
            // The native shaders still produce every probe; this kernel only assembles XYZ columns.
            private sealed class E25GpuColumns : IDisposable
            {
                private readonly ComputeShader shader;
                private readonly ComputeBuffer hard, off, plus, minus, mask, output, weights;
                private readonly int kernel, size;
                private int maskLamp = -1;
                internal int Columns { get; private set; }
                internal double UploadSeconds { get; private set; }
                internal double SubmitSeconds { get; private set; }
                internal double ReadbackSeconds { get; private set; }

                internal E25GpuColumns(E23FrameObjective fit)
                {
                    if (!SystemInfo.supportsComputeShaders) throw new NotSupportedException("D2 requires compute shaders.");
                    var asset = AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/LightingMvp/Shaders/E25AnchoredFd.compute");
                    if (!asset) throw new InvalidOperationException("Missing E25 Anchored FD kernel.");
                    shader = Object.Instantiate(asset);
                    size = fit.Size;
                    try
                    {
                        kernel = shader.FindKernel("AnchoredFd");
                        hard = new ComputeBuffer(size, sizeof(float)); off = new ComputeBuffer(size, sizeof(float));
                        plus = new ComputeBuffer(size, sizeof(float)); minus = new ComputeBuffer(size, sizeof(float));
                        mask = new ComputeBuffer(size / 3, sizeof(float)); output = new ComputeBuffer(size, sizeof(float));
                        shader.SetBuffer(kernel, "_Hard", hard); shader.SetBuffer(kernel, "_Off", off);
                        shader.SetBuffer(kernel, "_Plus", plus); shader.SetBuffer(kernel, "_Minus", minus);
                        shader.SetBuffer(kernel, "_Mask", mask); shader.SetBuffer(kernel, "_Column", output);
                        shader.SetFloat("_Scale", (float)fit.scale); shader.SetInt("_Length", size);
                        shader.SetInt("_ViewChannels", E23ChannelsPerView);
                        weights = new ComputeBuffer(fit.weights.Length, sizeof(float));
                        weights.SetData(fit.weights.Select(w => (float)w).ToArray()); shader.SetBuffer(kernel, "_Weights", weights);
                    }
                    catch { Dispose(); throw; }
                }

                internal void Anchor(float[] h, float[] o)
                {
                    maskLamp = -1;
                    long now = Stopwatch.GetTimestamp(); hard.SetData(h); off.SetData(o); UploadSeconds += Seconds(now);
                }

                internal void Position(float[] p, float[] m, float[] v, int lamp, double d, int stencil, float[] result)
                {
                    long now = Stopwatch.GetTimestamp();
                    plus.SetData(p);
                    if (stencil != 1) minus.SetData(m);
                    if (v != null && maskLamp != lamp) { mask.SetData(v); maskLamp = lamp; }
                    UploadSeconds += Seconds(now);
                    now = Stopwatch.GetTimestamp();
                    var (a, b) = E24Weights(d, stencil);
                    shader.SetFloat("_A", (float)a); shader.SetFloat("_B", (float)b); shader.SetInt("_Stencil", stencil);
                    shader.SetInt("_Masked", Flag(v != null));
                    shader.Dispatch(kernel, (size + 255) / 256, 1, 1);
                    SubmitSeconds += Seconds(now);
                    now = Stopwatch.GetTimestamp(); output.GetData(result); ReadbackSeconds += Seconds(now);
                    ++Columns;
                }

                public void Dispose()
                {
                    hard?.Dispose(); off?.Dispose(); plus?.Dispose(); minus?.Dispose(); mask?.Dispose(); output?.Dispose(); weights?.Dispose();
                    if (shader) Object.DestroyImmediate(shader);
                }
            }

            private void E25GpuRecovery(MbdRgbTarget hard, string scenario, int repeats)
            {
                var initial = E25DSource(scenario).initial;
                E25Optimize(scenario + "_warmup", "C4-gpu", E24AllLamps, initial, hard, iterations: 1, saveImages: false);
                for (int i = 1; i <= repeats; ++i)
                    E25Optimize(scenario + "_rep" + i, "C4-gpu", E24AllLamps, initial, hard, saveImages: i == 1);
            }

            private void E25GpuTransfer(MbdRgbTarget hard)
            {
                using var timings = Csv("gpu-transfer.csv", "sample,model,rep,residual_channels,columns,upload_s,submit_s,sync_readback_s,full_gpu_path_wall_s,cpu_cached_wall_s,exclusive_gpu_s");
                using var parity = Csv("gpu-parity.csv", "sample,model,quantity,coordinate,relative_l2,absolute_l2,step_cosine,relative_step_norm_diff,pass,nonfinite");
                using var status = Csv("gpu-status.csv", "status,reason");
                if (!SystemInfo.supportsComputeShaders) { status.WriteLine("unavailable,compute_not_supported"); return; }
                const string path = "Assets/LightingMvp/Shaders/E25AnchoredFd.compute";
                var asset = AssetDatabase.LoadAssetAtPath<ComputeShader>(path);
                if (!asset) throw new FileNotFoundException(path);
                var errors = ShaderUtil.GetComputeShaderMessages(asset).Where(m => m.severity.ToString() == "Error").ToArray();
                if (errors.Length != 0) throw new InvalidDataException("D2 compute did not compile: " + errors[0].message);
                var compute = Object.Instantiate(asset);
                try
                {
                    int k = compute.FindKernel("AnchoredFd");
                    var fit = new E23FrameObjective(hard.fit, Mapping.SignedLog, e24HardScale);
                    var success = E25DSource("seed131_near_on");
                    var failure = E25DSource("dir911_20cm");
                    using var hardBuf = new ComputeBuffer(fit.Size, sizeof(float));
                    using var offBuf = new ComputeBuffer(fit.Size, sizeof(float));
                    using var plusBuf = new ComputeBuffer(fit.Size, sizeof(float));
                    using var minusBuf = new ComputeBuffer(fit.Size, sizeof(float));
                    using var maskBuf = new ComputeBuffer(fit.Size / 3, sizeof(float));
                    using var outBuf = new ComputeBuffer(fit.Size, sizeof(float));
                    compute.SetBuffer(k, "_Hard", hardBuf); compute.SetBuffer(k, "_Off", offBuf);
                    compute.SetBuffer(k, "_Plus", plusBuf); compute.SetBuffer(k, "_Minus", minusBuf);
                    compute.SetBuffer(k, "_Mask", maskBuf); compute.SetBuffer(k, "_Column", outBuf);
                    compute.SetFloat("_Scale", (float)fit.scale);
                    using var viewWeights = new ComputeBuffer(fit.weights.Length, sizeof(float));
                viewWeights.SetData(fit.weights.Select(w => (float)w).ToArray()); compute.SetBuffer(k, "_Weights", viewWeights);
                    compute.SetInt("_Length", fit.Size); compute.SetInt("_ViewChannels", E23ChannelsPerView);
                    foreach (var (label, pose, lamp) in new[] { ("success", success.final, 0), ("failure", failure.final, 0), ("hdr_gt", room.Truth, 0) })
                    {
                        var data = E25AuditedLamp(label, pose, lamp, fit, .02, null, null);
                        if (label == "hdr_gt")
                        {
                            for (int dim = 0; dim < 3; ++dim)
                            {
                                var p = data.points[dim];
                                if (p.stencil != 3) continue;
                                int axis = lamp * 4 + dim;
                                var moved = (double[])pose.Clone(); moved[axis] += 2 * p.d;
                                if (!room.Contains(moved)) continue;
                                E25Capture(moved, LightShadows.None, E25DefaultViews, p.om); p.stencil = 2;
                                break;
                            }
                        }
                        var g = Enumerable.Range(0, 4).Select(_ => new float[fit.Size]).ToArray();
                        var c = Enumerable.Range(0, 4).Select(_ => new float[fit.Size]).ToArray();
                        var mapped = data.hard.Select(fit.Transform).ToArray();
                        var rawIntensity = data.points[3];
                        if (rawIntensity.stencil != 0)
                            for (int i = 0; i < fit.Size; ++i)
                            {
                                double x = Difference(mapped[i], fit.Transform(rawIntensity.hp[i]), rawIntensity.stencil == 1 ? 0 : fit.Transform(rawIntensity.hm[i]),
                                    rawIntensity.d, rawIntensity.stencil);
                                g[3][i] = c[3][i] = (float)(x * fit.weights[i / E23ChannelsPerView]);
                            }
                        foreach (bool mask in new[] { false, true })
                        {
                            string name = mask ? "C4" : "C3";
                            compute.SetInt("_Masked", Flag(mask));
                            for (int r = 0; r <= 5; ++r)
                            {
                                // Rep zero warms both paths; fixed arrays and entry points are unchanged across repetitions.
                                var ticks = Stopwatch.GetTimestamp();
                                for (int j = 0; j < 3; ++j)
                                {
                                    var p = data.points[j]; if (p.stencil == 0) continue;
                                    E25CachedColumn(fit, data.hard, mapped, data.off, p.op, p.om, mask ? data.mask : null, p.d, p.stencil, c[j]);
                                }
                                double cpu = Seconds(ticks), up = 0, sub = 0, read = 0;
                                long all = Stopwatch.GetTimestamp();
                                ticks = Stopwatch.GetTimestamp();
                                hardBuf.SetData(data.hard); offBuf.SetData(data.off); maskBuf.SetData(data.mask);
                                up += Seconds(ticks);
                                int n = 0;
                                for (int j = 0; j < 3; ++j)
                                {
                                    var p = data.points[j]; if (p.stencil == 0) continue; ++n;
                                    ticks = Stopwatch.GetTimestamp(); plusBuf.SetData(p.op); minusBuf.SetData(p.om); up += Seconds(ticks);
                                    ticks = Stopwatch.GetTimestamp();
                                    var (a, b) = E24Weights(p.d, p.stencil);
                                    compute.SetFloat("_A", (float)a); compute.SetFloat("_B", (float)b); compute.SetInt("_Stencil", p.stencil);
                                    compute.Dispatch(k, (fit.Size + 255) / 256, 1, 1); sub += Seconds(ticks);
                                    ticks = Stopwatch.GetTimestamp(); outBuf.GetData(g[j]); read += Seconds(ticks);
                                }
                                double full = Seconds(all);
                                if (r > 0) timings.WriteLine(string.Join(",", label, name, r, fit.Size, n, F(up), F(sub), F(read), F(full), F(cpu), "unavailable"));
                            }
                            for (int j = 0; j < 4; ++j) Compare("column", j, c[j].Select(x => (double)x).ToArray(), g[j].Select(x => (double)x).ToArray());
                            var cpuNormal = E25Normal(c, data.residual, 0, fit.Size);
                            var gpuNormal = E25Normal(g, data.residual, 0, fit.Size);
                            Compare("gram", -1, cpuNormal.gram.Cast<double>().ToArray(), gpuNormal.gram.Cast<double>().ToArray());
                            Compare("gradient", -1, cpuNormal.gradient, gpuNormal.gradient);
                            var x0 = pose.Skip(4 * lamp).Take(4).ToArray();
                            bool Legal(double[] x)
                            {
                                var rig = (double[])pose.Clone(); Array.Copy(x, 0, rig, 4 * lamp, 4); return room.Contains(rig);
                            }
                            var p0 = ImageJacobianLm.LampScaledTrustProposal(cpuNormal.gram, cpuNormal.gradient, x0, units, Legal, out _);
                            var p1 = ImageJacobianLm.LampScaledTrustProposal(gpuNormal.gram, gpuNormal.gradient, x0, units, Legal, out _);
                            Compare("step", -1, p0, p1);

                            void Compare(string what, int coordinate, double[] cpu, double[] gpu)
                            {
                                int bad = gpu.Count(x => double.IsNaN(x) || double.IsInfinity(x));
                                double absolute = L2(cpu.Zip(gpu, (x, y) => x - y)), reference = L2(cpu);
                                double relative = absolute / Math.Max(reference, 1e-12);
                                double cos = what == "step" && reference > 1e-12 ? E25Cos(cpu, gpu) : double.NaN;
                                bool pass = bad == 0 && (what == "step" ? (reference <= 1e-12 ? absolute <= 1e-12 : cos >= .9999) : absolute <= Math.Max(1e-12, 1e-4 * reference));
                                parity.WriteLine(string.Join(",", label, name, what, coordinate, F(relative), F(absolute), F(cos),
                                    what == "step" ? F(Math.Abs(L2(gpu) - reference) / Math.Max(1e-12, reference)) : "", Flag(pass), bad));
                            }
                        }
                    }
                    status.WriteLine("completed,CPU_readback_architecture_roundtrip_benchmark_only");
                }
                finally { Object.DestroyImmediate(compute); }
            }
            private static double Seconds(long tick) => (double)(Stopwatch.GetTimestamp() - tick) / Stopwatch.Frequency;
        }
    }
}

using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private static void RunMomentBoundsReference(string output)
        {
            Directory.CreateDirectory(output);
            var records = File.ReadLines(Arg("-bbMbdVectors", Path.Combine(output, "reference-input.csv"))).Skip(1)
                .Select(line => line.Split(',')).ToArray();
            if (records.Length == 0) throw new InvalidDataException("No MBD reference vectors.");
            int count = records.Length;
            var moments = new Vector4[count];
            var parameters = new Vector4[count];
            var orders = new int[count];
            var cpu = new string[count];
            var grad = new double[5];
            var values = new double[5];
            var watch = Stopwatch.StartNew();
            for (int i = 0; i < count; ++i)
            {
                string[] row = records[i];
                int order = orders[i] = int.Parse(row[2], Invariant);
                for (int k = 0; k < 5; ++k) values[k] = double.Parse(row[6 + k], Invariant);
                double eta = double.Parse(row[3], Invariant);
                double bias = double.Parse(row[4], Invariant);
                double blend = double.Parse(row[5], Invariant);
                var state = MomentBounds.Evaluate(values.AsSpan(0, 2 * order + 1), eta, bias, blend, out double bound, grad, out double depth);
                cpu[i] = string.Join(",", (int)state, F(bound), string.Join(",", grad.Select(F)), F(depth));
                parameters[i] = new Vector4((float)values[0], (float)eta, (float)bias, (float)blend);
                moments[i] = new Vector4((float)values[1], (float)values[2], (float)values[3], (float)values[4]);
            }
            double cpuSeconds = watch.Elapsed.TotalSeconds;
            var shader = AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/LightingMvp/Shaders/MomentBoundsReference.compute");
            if (!shader || ShaderUtil.GetComputeShaderMessages(shader).Any(m => m.severity.ToString() == "Error"))
                throw new InvalidOperationException("Moment bounds reference compute shader did not compile.");
            using var momentBuffer = new ComputeBuffer(count, 16);
            using var parameterBuffer = new ComputeBuffer(count, 16);
            using var orderBuffer = new ComputeBuffer(count, 4);
            using var forward = new ComputeBuffer(count, 8);
            using var powers = new ComputeBuffer(count, 16);
            using var other = new ComputeBuffer(count, 16);
            momentBuffer.SetData(moments); parameterBuffer.SetData(parameters); orderBuffer.SetData(orders);
            shader.SetInt("_Count", count);
            int f = shader.FindKernel("ForwardBounds"), b = shader.FindKernel("BackwardBounds");
            foreach (int kernel in new[] { f, b })
            {
                shader.SetBuffer(kernel, "_Moments", momentBuffer);
                shader.SetBuffer(kernel, "_Parameters", parameterBuffer);
                shader.SetBuffer(kernel, "_Orders", orderBuffer);
            }
            shader.SetBuffer(f, "_Bounds", forward);
            shader.SetBuffer(b, "_PowerGradient", powers); shader.SetBuffer(b, "_OtherGradient", other);
            var fb = new Vector2[count];
            var mf = new Vector4[count];
            var of = new Vector4[count];
            int groups = (count + 63) / 64;
            shader.Dispatch(f, groups, 1, 1); forward.GetData(fb);
            shader.Dispatch(b, groups, 1, 1); other.GetData(of);
            watch.Restart();
            shader.Dispatch(f, groups, 1, 1); forward.GetData(fb);
            double forwardSeconds = watch.Elapsed.TotalSeconds;
            watch.Restart();
            shader.Dispatch(b, groups, 1, 1); powers.GetData(mf); other.GetData(of);
            double backwardSeconds = watch.Elapsed.TotalSeconds;

            using (var file = new StreamWriter(Path.Combine(output, "unity-reference.csv")))
            {
                file.WriteLine("id,cpu_status,cpu_bound,cpu_m0,cpu_m1,cpu_m2,cpu_m3,cpu_m4,cpu_eta,gpu_status,gpu_adjoint_status,gpu_bound,gpu_m0,gpu_m1,gpu_m2,gpu_m3,gpu_m4,gpu_eta");
                for (int i = 0; i < count; ++i)
                {
                    var m = mf[i]; var g = of[i];
                    file.WriteLine(string.Join(",", records[i][0], cpu[i], (int)fb[i].y, (int)g.z,
                        F(fb[i].x), F(g.x), F(m.x), F(m.y), F(m.z), F(m.w), F(g.y)));
                }
            }
            using (var file = new StreamWriter(Path.Combine(output, "reference-timing.csv")))
            {
                file.WriteLine("samples,unity,cpu_forward_and_reverse_including_csv_format_seconds,gpu_forward_dispatch_readback_seconds,gpu_recomputed_primal_and_reverse_dispatch_readback_seconds,memory_bytes");
                file.WriteLine(string.Join(",", count, Application.unityVersion, F(cpuSeconds), F(forwardSeconds), F(backwardSeconds), count * 76));
            }
            Debug.Log("[MBD] reference: " + count + " CPU/GPU cases");
        }
    }
}

using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

namespace LightingMvp.Editor
{
    /// <summary>Optional diagnostic. A child process receives scalar responses, never the GT rig.</summary>
    internal sealed class ZeroGradsTrial
    {
        [Serializable] private sealed class Input { public double[] initial; public int seed, remaining; }
        private static readonly CultureInfo Invariant = CultureInfo.InvariantCulture;
        private readonly string python, script, models;

        internal ZeroGradsTrial(string python, string script, string models)
        {
            this.python = RequireFile(python);
            this.script = RequireFile(script);
            this.models = RequireFile(models);
        }

        private static string RequireFile(string path)
        {
            if (string.IsNullOrWhiteSpace(path) || !File.Exists(path) || path.Contains("\""))
                throw new FileNotFoundException("ZeroGrads diagnostic needs real paths for Python, the bridge and the official models.py.", path);
            return Path.GetFullPath(path);
        }

        internal BlackBoxOptimizer.Result Run(double[] initial, Func<double[], string, double> objective, int budget, int seed)
        {
            double loss = objective(initial, "initial");
            var result = new BlackBoxOptimizer.Result { InitialLoss = loss, BestLoss = loss, Best = (double[])initial.Clone(), Queries = 1, StopReason = "budget" };
            var error = new StringBuilder();
            var start = new ProcessStartInfo
            {
                FileName = python, Arguments = $"-u \"{script}\" --models \"{models}\"",
                UseShellExecute = false, CreateNoWindow = true,
                RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true
            };
            using (var process = new Process { StartInfo = start })
            {
                process.ErrorDataReceived += (_, e) => { if (e.Data != null) lock (error) { error.AppendLine(e.Data); } };
                if (!process.Start()) throw new InvalidOperationException("Could not launch the ZeroGrads Python process.");
                try
                {
                    process.BeginErrorReadLine();
                    process.StandardInput.WriteLine(JsonUtility.ToJson(new Input { initial = initial, seed = seed, remaining = budget - 1 }));
                    process.StandardInput.Flush();
                    while (true)
                    {
                        var pending = process.StandardOutput.ReadLineAsync();
                        if (!pending.Wait(TimeSpan.FromSeconds(90))) throw new TimeoutException("No response from the ZeroGrads process.");
                        string line = pending.Result;
                        if (line == null) throw new EndOfStreamException("ZeroGrads exited before completing the trial.");
                        string[] parts = line.Split(' ');
                        if (parts.Length == 2 && parts[0] == "D")
                        {
                            result.Iterations = int.Parse(parts[1], Invariant);
                            if (!process.WaitForExit(10000) || process.ExitCode != 0) throw new InvalidOperationException("ZeroGrads did not exit successfully.");
                            return result;
                        }
                        if (parts.Length != initial.Length + 2 || parts[0] != "Q" || (parts[1] != "proxy_probe" && parts[1] != "proxy_step") || result.Queries >= budget)
                            throw new InvalidDataException("Invalid or over-budget ZeroGrads query.");
                        double[] x = parts.Skip(2).Select(v => double.Parse(v, Invariant)).ToArray();
                        if (x.Any(v => !(v >= 0 && v <= 1))) throw new InvalidDataException("ZeroGrads query outside the normalized domain.");
                        loss = objective(x, parts[1]);
                        ++result.Queries;
                        if (double.IsNaN(loss) || double.IsInfinity(loss) || loss < 0) throw new InvalidOperationException("Invalid Unity RGB loss.");
                        if (loss < result.BestLoss) { result.Best = x; result.BestLoss = loss; }
                        process.StandardInput.WriteLine(loss.ToString("G17", Invariant));
                        process.StandardInput.Flush();
                    }
                }
                catch (Exception ex)
                {
                    if (!process.HasExited) process.Kill();
                    process.WaitForExit(5000);
                    lock (error) { throw new InvalidOperationException("ZeroGrads diagnostic failed. " + error, ex); }
                }
            }
        }
    }
}

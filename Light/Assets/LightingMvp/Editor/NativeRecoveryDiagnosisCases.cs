using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;
using Frame = LightingMvp.NativeRenderBackend.Frame;
using Mapping = LightingMvp.NativeRenderBackend.HdrMapping;
using Object = UnityEngine.Object;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        private sealed partial class ClassroomDiagnosis
        {
            private void RunOracle(int smokeSeed, bool smoke)
            {
                var power = Enumerable.Range(0, 7).Select(i => i * 4 + 3).ToArray();
                foreach (int seed in smoke ? new[] { smokeSeed } : StructureSeeds)
                {
                    var start = Mix("e10_near", seed, power);
                    Trial("e10_near", seed, "strength_lm", -1, Mapping.Raw, start, power, ShortDiagnosisBudget);
                    Trial("e10_near", seed, "strength_nnls", -1, Mapping.Raw, start, power, ShortDiagnosisBudget, nnls: true);
                }
                int pairedSeed = smoke ? smokeSeed : 131;
                foreach (int l in smoke ? new[] { 0, 6 } : Enumerable.Range(0, 7))
                foreach (bool strength in new[] { false, true })
                {
                    var axes = LampAxes(l, strength);
                    Trial("e10_near", pairedSeed, strength ? "single_position_strength" : "single_position", l, Mapping.Raw,
                        Mix("e10_near", pairedSeed, axes), axes, ShortDiagnosisBudget);
                }
                var single = rows.Where(x => x.method == "single_position_strength").OrderBy(x => x.finalContributionNrmse).ThenBy(x => x.lamp).ToArray();
                gate.easyLamp = single.First().lamp; gate.hardLamp = single.Last().lamp;
                gate.singleLampPasses = single.Count(x => x.contributionPass);
                if (!smoke)
                {
                    foreach (var l in new[] { gate.easyLamp, gate.hardLamp }.Distinct())
                    foreach (var pair in new (string profile, int seed)[] { ("e10_near", 157), ("e10_near", 181), ("e10", 131) })
                    {
                        var axes = LampAxes(l, true);
                        Trial(pair.profile, pair.seed, "single_position_strength", l, Mapping.Raw, Mix(pair.profile, pair.seed, axes), axes, ShortDiagnosisBudget);
                    }
                    RecordHotspots();
                    ScanAxes();
                }
                gate.sharedRenders = shared.Cameras;
                WriteJson(dir, "oracle.json", gate);
            }

            private void RunPositionOracle(int seed, bool smoke)
            {
                var axes = Enumerable.Range(0, 28).Where(i => i % 4 != 3).ToArray();
                foreach (string profile in new[] { "e10_near", "e10" })
                foreach (int s in smoke ? new[] { seed } : StructureSeeds)
                    Trial(profile, s, "position_oracle", -1, Mapping.Raw, Mix(profile, s, axes), axes, FullDiagnosisBudget);
            }

            private DiagnosisStage Stage(string stage) => Load<DiagnosisStage>(Path.Combine(root, stage.Replace('-', '_'), "stage.json"));
            private void RunComparisons(DiagnosisGate oracle, int seed, bool smoke)
            {
                var axes = AllAxes();
                if (task.StartsWith("e14", StringComparison.Ordinal))
                {
                    string profile = task == "e14-large" ? "e10" : "e10_near";
                    if (task == "e14-large" && BestCompression(Stage("e14")) == Mapping.Raw)
                        throw new InvalidOperationException("Large paired E14 requires stable improvement in the near E14 ablation.");
                    if (task != "e14-large")
                        foreach (int l in new[] { oracle.hardLamp, oracle.easyLamp }.Distinct())
                        foreach (var mapping in LossArms)
                            Trial(profile, seed, "single_loss", l, mapping, Mix(profile, seed, LampAxes(l, true)), LampAxes(l, true), ShortDiagnosisBudget);
                    foreach (int s in smoke ? new[] { seed } : StructureSeeds)
                    foreach (var mapping in LossArms)
                        Trial(profile, s, "full_loss", -1, mapping, Prior(profile, s).initialTheta, axes, FullDiagnosisBudget);
                }
                else
                {
                    var loss = BestCompression(Stage("e14"));
                    if (task == "e15-budget")
                    {
                        var cases = Stage("e15").trials.Where(x => x.lamp < 0 && x.seed == seed && !x.passed).ToArray();
                        if (cases.Length == 0) throw new InvalidOperationException("Requested E15 budget seed has no failing E15 case.");
                        var selected = cases.OrderBy(x => x.recapturedRaw).ThenBy(x => x.method, StringComparer.Ordinal).First();
                        Trial(selected.profile, seed, selected.method, -1, loss, selected.start, axes, 20000,
                            physical: selected.method != "full_lm", diagonal: selected.method == "world_diag_lm");
                    }
                    else if (task == "e15-block")
                    {
                        if (oracle.singleLampPasses < 4) throw new InvalidOperationException("Block escalation requires at least four recoverable single lamps in E13d.");
                        foreach (int s in StructureSeeds)
                        {
                            var start = Prior("e10_near", s).initialTheta;
                            Trial("e10_near", s, "full_lm", -1, loss, start, axes, FullDiagnosisBudget);
                            Trial("e10_near", s, "block_lm", -1, loss, start, axes, FullDiagnosisBudget, block: true);
                        }
                    }
                    else
                    {
                        foreach (string method in new[] { "full_lm", "world_lm", "world_diag_lm" })
                            Trial("e10_near", seed, method, oracle.hardLamp, loss, Mix("e10_near", seed, LampAxes(oracle.hardLamp, true)),
                                LampAxes(oracle.hardLamp, true), ShortDiagnosisBudget, physical: method != "full_lm", diagonal: method == "world_diag_lm");
                        foreach (int s in smoke ? new[] { seed } : StructureSeeds)
                        foreach (string method in new[] { "full_lm", "world_lm", "world_diag_lm" })
                            Trial("e10_near", s, method, -1, loss, Prior("e10_near", s).initialTheta, axes, FullDiagnosisBudget,
                                physical: method != "full_lm", diagonal: method == "world_diag_lm");
                    }
                }
            }

            // Frozen before inspecting E14: at least 15% less raw error in two of three pairs,
            // with the paired held-out aggregate no worse by more than 2 percentage points.
            private static Mapping BestCompression(DiagnosisStage data)
            {
                var full = data.trials.Where(r => r.method == "full_loss").ToArray();
                var raw = full.Where(r => r.mapping == "Raw").ToDictionary(r => r.seed);
                if (raw.Count != 3) throw new InvalidDataException("Full E14 near matrix must precede the E15 comparison.");
                var arm = full.Where(r => r.mapping != "Raw").GroupBy(r => r.mapping)
                    .Where(g => g.Count(r => r.recapturedRaw <= .85 * raw[r.seed].recapturedRaw && r.heldNrmse <= raw[r.seed].heldNrmse + .02) >= 2)
                    .OrderBy(g => Median(g.Select(r => r.recapturedRaw))).ThenBy(g => g.Key, StringComparer.Ordinal).FirstOrDefault();
                return arm == null ? Mapping.Raw : (Mapping)Enum.Parse(typeof(Mapping), arm.Key);
            }

            [Serializable] private sealed class HotspotCapture
            {
                public string rig, collider;
                public int view, width, height, x, y;
                public bool colliderHit;
                public Vector3 hit;
                public Brightness target, rendered;
                public double[] lampToHitMetres, omitRawNrmse, omitTop1ErrorDropFraction;
            }
            [Serializable] private sealed class HotspotData { public HotspotCapture[] captures; }
            private void RecordHotspots()
            {
                var prior = Prior("e10", 131);
                var list = new List<HotspotCapture>();
                foreach (var pair in new (string label, double[] x)[] { ("initial", prior.initialTheta), ("best", prior.bestTheta) })
                for (int v = 0; v < 5; ++v)
                {
                    var t = targetHi[v];
                    var f = Capture(high, shared, pair.x, v, "hotspot_" + pair.label);
                    var stats = BrightnessStats(f, t);
                    var mask = Enumerable.Range(0, t.PixelCount).Select(p => (p, error: PixelError(f.Pixel(p), t.Pixel(p))))
                        .OrderByDescending(p => p.error).Take(t.PixelCount / 100).ToArray();
                    double baseError = mask.Sum(a => a.error);
                    var row = new HotspotCapture { rig = pair.label, view = v, width = high.Width, height = high.Height,
                        x = stats.maxErrorPixel % high.Width, y = stats.maxErrorPixel / high.Width,
                        target = BrightnessStats(t, t), rendered = stats, omitRawNrmse = new double[7], omitTop1ErrorDropFraction = new double[7] };
                    var pose = fixture.Config.views[v];
                    float tan = Mathf.Tan(.5f * pose.fov * Mathf.Deg2Rad);
                    Vector3 ray = pose.rotation * new Vector3((2f * (row.x + .5f) / high.Width - 1) * tan * high.Width / high.Height,
                        (2f * (row.y + .5f) / high.Height - 1) * tan, 1);
                    row.colliderHit = fixture.RaycastGeometry(pose.position, ray.normalized, out var hit);
                    if (row.colliderHit)
                    {
                        row.collider = hit.collider.name; row.hit = hit.point;
                        row.lampToHitMetres = Enumerable.Range(0, 7).Select(l => (double)Vector3.Distance(hit.point, ClassroomFixture.Position(pair.x, 4 * l))).ToArray();
                    }
                    for (int l = 0; l < 7; ++l)
                    {
                        var omitted = Capture(high, shared, pair.x, v, $"hotspot_{pair.label}_omit_{l}", l);
                        row.omitRawNrmse[l] = Math.Sqrt(NativeRenderBackend.Compare(omitted, t));
                        row.omitTop1ErrorDropFraction[l] = baseError > 0 ? 1 - mask.Sum(a => PixelError(omitted.Pixel(a.p), t.Pixel(a.p))) / baseError : 0;
                    }
                    list.Add(row);
                    if (v == 0)
                    {
                        NativeRenderBackend.Save(f, high.Width, high.Height, Path.Combine(dir, $"large131_{pair.label}"));
                        SaveHeatmap(f, t, high.Width, high.Height, Path.Combine(dir, $"large131_heat_{pair.label}.png"));
                    }
                }
                NativeRenderBackend.Save(targetHi[0], high.Width, high.Height, Path.Combine(dir, "large131_target"));
                WriteJson(dir, "hotspots.json", new HotspotData { captures = list.ToArray() });
            }

            private static void SaveHeatmap(Frame f, Frame target, int w, int h, string path)
            {
                var image = new Texture2D(w, h, TextureFormat.RGB24, false, true);
                try
                {
                    var paint = new Color32[f.PixelCount];
                    double normal = target.Energy / f.PixelCount + DiagnosisEpsilon;
                    for (int p = 0; p < paint.Length; ++p)
                    {
                        // Fixed scale: floor=0, mid=15x mean GT pixel energy, ceiling=1e4x.
                        float intensity = (float)Math.Max(0, Math.Min(1, Math.Log10(1 + PixelError(f.Pixel(p), target.Pixel(p)) / normal) / 4));
                        paint[p] = new Color(Mathf.Clamp01(intensity * 2), Mathf.Clamp01(intensity * 2 - 1), Mathf.Clamp01(intensity * .6f), 1);
                    }
                    image.SetPixels32(paint); image.Apply(); File.WriteAllBytes(path, image.EncodeToPNG());
                }
                finally { Object.DestroyImmediate(image); }
            }

            private void ScanAxes()
            {
                using var csv = new StreamWriter(Path.Combine(dir, "axis-curves.csv"));
                csv.WriteLine("lamp,context,axis,metres,legal,raw_sq,view0_sq,view1_sq,view2_sq,cumulative_shared");
                foreach (int l in new[] { gate.hardLamp, gate.easyLamp }.Distinct())
                foreach (var anchor in new (string label, double[] x)[] { ("gt", truth), ("near_single_start", Mix("e10_near", 131, LampAxes(l, true))), ("near_full_final", Prior("e10_near", 131).bestTheta) })
                for (int axis = 0; axis < 3; ++axis)
                foreach (double metres in new[] { -.05, -.02, -.01, -.005, 0, .005, .01, .02, .05 })
                {
                    var x = (double[])anchor.x.Clone(); x[l * 4 + axis] += metres / Metres[axis];
                    bool legal = fixture.Rig.Contains(x);
                    var value = legal ? DiagnosisViews.Select(v => NativeRenderBackend.Compare(Capture(train, shared, x, v, "axis_curve"), target[v])).ToArray() : null;
                    csv.WriteLine(string.Join(",", l, anchor.label, axis, F(metres), legal ? 1 : 0, legal ? F(value.Average()) : "",
                        legal ? F(value[0]) : "", legal ? F(value[1]) : "", legal ? F(value[2]) : "", shared.Cameras));
                }
            }
        }
    }
}

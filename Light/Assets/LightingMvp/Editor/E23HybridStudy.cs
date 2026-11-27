using System;
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
            private StreamWriter E23ModelLog() => Csv("model-queries.csv",
                "arm,query,phase,shadows,axis,signed_step_m,camera_renders,theta");

            private void E23Hybrid(bool full)
            {
                string mapping = Arg("-bbE23Mapping", "C1");
                var gt = E22Target(true);
                using var log = E23ModelLog();
                foreach (int seed in smoke ? new[] { 157 } : full ? E23Seeds : new[] { 157, 131 })
                {
                    int[] active = full ? room.Lamps : seed == 157 ? new[] { 4, 6 } : new[] { 1, 6 };
                    var initial = full ? E23Near(seed) : (double[])room.Truth.Clone();
                    if (!full)
                    {
                        var failed = E23SourceArm(seed, "B1").final;
                        foreach (int lamp in active) Array.Copy(failed, 4 * lamp, initial, 4 * lamp, 4);
                    }
                    string label = full ? $"seed{seed}_near_on" : $"seed{seed}_log_failed_L{active[0]+1}L{active[1]+1}";
                    foreach (string model in new[] { "B0", "B1", "B2", "B3" })
                    {
                        E21Trial prior = full && (mapping == "C0" || mapping == "C1") && !smoke && model == "B0"
                            ? E23SourceArm(seed, "B" + mapping.Substring(1)) : null;
                        if (model == "B0")
                        {
                            RunE23Arm(label, model, seed, active, initial, gt, mapping, full ? 1575 : 520,
                                prior: prior, reuse: prior == null ? null : "E22-B-On/" + prior.id);
                            continue;
                        }
                        double alpha = model == "B1" ? 1 : model == "B2" ? 0 : .5;
                        string jacobian = model == "B1" ? "chain_on_xyz" : model == "B2" ? "off_xyz" : "blend_xyz";
                        E23RunProxyArm(label, model, seed, active, initial, gt, mapping, full ? 1575 : 520, log, LightShadows.None, alpha, jacobian);
                    }
                }
            }

            private E21Trial E23RunProxyArm(string scenario, string name, int seed, int[] active, double[] initial, MbdRgbTarget gt,
                string mapping, int cap, StreamWriter log, LightShadows proxy, double alpha, string jacobian)
            {
                var objective = new E23FrameObjective(gt.fit, E23Mapping(mapping), e23Scale);
                int off = 0, soft = 0;
                return RunE23Arm(scenario, name, seed, active, initial, gt, mapping, cap, jacobian: jacobian, alpha: alpha,
                    modelCameras: () => (off, soft),
                    configure: (settings, expand) =>
                    {
                        string arm = $"{scenario}_{name}";
                        bool hardColumns = alpha > 0;
                        bool proxyColumns = alpha < 1;
                        bool hardAnchor = hardColumns || objective.mapping != Mapping.Raw;
                        int anchorCost = Flag(hardAnchor) + Flag(proxyColumns), probeCost = Flag(hardColumns) + Flag(proxyColumns);
                        var hardBase = hardAnchor ? new float[E23Channels] : null;
                        var proxyBase = proxyColumns ? new float[E23Channels] : null;
                        var weights = new double[E23Channels];
                        var first = new float[E23Channels];
                        var second = new float[E23Channels];

                        settings.PositionModel = new ImageJacobianLm.TrustPositionModel(anchorCost, probeCost, Anchor, Column);

                        void Anchor(int q, double[] local)
                        {
                            var rig = expand(local);
                            if (hardAnchor) Capture(q++, rig, LightShadows.Hard, "anchor", -1, 0, hardBase);
                            if (proxyColumns) Capture(q, rig, proxy, "anchor", -1, 0, proxyBase);
                            for (int i = 0; i < weights.Length; ++i)
                            {
                                double d = objective.mapping == Mapping.Raw ? 1 : E23Derivative(hardBase[i], objective.mapping, objective.scale);
                                weights[i] = objective.weights[i / E23ChannelsPerView] * d;
                            }
                        }

                        void Column(int q, double[] local, int axis, double d, int stencil, float[] result)
                        {
                            var shifted = (double[])local.Clone();
                            int coordinate = 4 * active[axis / 4] + axis % 4;
                            double next = stencil == 3 ? -d : 2 * d;
                            double a = (stencil == 3 ? .5 : stencil == 2 ? 2 : 1) / d;
                            double b = stencil == 1 ? 0 : -.5 / d;
                            double c = stencil == 3 ? 0 : (stencil == 2 ? -1.5 : -1) / d;
                            if (hardColumns) Append(LightShadows.Hard, hardBase, alpha, false);
                            if (proxyColumns) Append(proxy, proxyBase, 1 - alpha, hardColumns);

                            void Append(LightShadows shadows, float[] center, double strength, bool add)
                            {
                                shifted[axis] = local[axis] + d;
                                Capture(q++, expand(shifted), shadows, "probe", coordinate, d * settings.Units[axis], first);
                                if (stencil != 1)
                                {
                                    shifted[axis] = local[axis] + next;
                                    Capture(q++, expand(shifted), shadows, "probe", coordinate, next * settings.Units[axis], second);
                                }
                                for (int i = 0; i < result.Length; ++i)
                                {
                                    double derivative = a * first[i] + b * second[i] + c * center[i];
                                    float contribution = (float)(weights[i] * strength * derivative);
                                    result[i] = add ? result[i] + contribution : contribution;
                                }
                            }
                        }

                        void Capture(int q, double[] rig, LightShadows shadows, string phase, int axis, double step, float[] buffer)
                        {
                            int before = low.RenderCount;
                            E23CaptureRgb(rig, shadows, buffer);
                            int renders = low.RenderCount - before;
                            if (shadows == LightShadows.None) off += renders;
                            if (shadows == LightShadows.Soft) soft += renders;
                            if (renders != E19FitViews) throw new InvalidOperationException("E23 model rendered an unexpected number of views.");
                            log.WriteLine(string.Join(",", arm, q, phase, shadows, axis, F(step), renders, Series(rig)));
                        }
                    });
            }
        }
    }
}

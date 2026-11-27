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
            private delegate void E24Capture(int query, double[] rig, LightShadows mode, string phase, int axis, double step, float[] rgb);
            private const string E24VisibilityHeader = "state,purpose,query,lamp,view,physical_step,stencil,relative_threshold,p95_off,absolute_threshold,epsilon,hard_l2,off_l2,low_pct,unreliable_pct,negative_channel_pct,raw_negative_pct,raw_above_one_pct,raw_mean,raw_min,raw_max,fit_relative,mean_cosine,mask_mean,mask_lt_05_pct";
            private const int E24PixelsPerView = E23ChannelsPerView / 3;
            private const float E24RelativeThreshold = .001f;
            private static bool E24Proxy(string model) => model is "H2" or "H4" or "H5" or "H3" or "C1" or "C3" or "C4" or "C4-soft";

            private void E24Configure(ImageJacobianLm.TrustSettings settings, Func<double[], double[]> expand, int[] active,
                string model, E23FrameObjective objective, E24Capture capture, string state, StreamWriter audit = null, bool cachedMapping = false, E25Counter metrics = null, E25GpuColumns gpu = null)
            {
                settings.NormalMode = ImageJacobianLm.TrustNormalMode.LampScaledDense;
                if (!E24Proxy(model)) return;
                bool masked = model is "H4" or "H5" or "C4" or "C4-soft", anchored = model is "C3" or "C4" or "C4-soft";
                bool hardPosition = model is "H3" or "H5", offPosition = model != "H3";
                bool hardAnchor = model != "C1";
                if (cachedMapping && !anchored) throw new ArgumentException("Cached mapping requires C3 or C4.");
                var hardMapped = cachedMapping ? new float[E23Channels] : null;
                var probeRig = cachedMapping ? new double[settings.Layout?.Count ?? settings.Units.Length] : null;
                var hard = new float[E23Channels]; var off = new float[E23Channels];
                var plus = new float[E23Channels]; var minus = new float[E23Channels];
                var hardA = hardPosition || masked ? new float[E23Channels] : null;
                var hardB = hardA == null ? null : new float[E23Channels];
                var weights = cachedMapping ? null : new double[E23Channels];
                var mask = masked ? active.Select(_ => new float[E23Channels / 3]).ToArray() : null;
                var norms = masked ? new float[E24PixelsPerView] : null;
                settings.PositionModel = new ImageJacobianLm.TrustPositionModel(Flag(hardAnchor) + Flag(offPosition),
                    Flag(hardPosition) + Flag(offPosition), Anchor, Position, masked ? 2 : 1,
                    masked ? (Action<int, double[], int, double, int, float[]>)Intensity : null);

                void Anchor(int q, double[] x)
                {
                    if (hardAnchor) capture(q++, expand(x), LightShadows.Hard, "anchor", -1, 0, hard);
                    if (offPosition) capture(q, expand(x), LightShadows.None, "anchor", -1, 0, off);
                    long begin = E25Counter.Now(metrics);
                    gpu?.Anchor(hard, off);
                    if (hardMapped != null)
                        for (int i = 0; i < hardMapped.Length; ++i) hardMapped[i] = objective.Transform(hard[i]);
                    if (weights != null)
                    for (int i = 0; i < weights.Length; ++i)
                        weights[i] = objective.weights[i / E23ChannelsPerView] * (objective.mapping == Mapping.Raw || anchored ? 1 : E23Derivative(hard[i], objective.mapping, objective.scale));
                    if (masked) foreach (var v in mask) Array.Fill(v, model == "C4-soft" ? .5f : 1f);
                    if (metrics != null) metrics.mapTicks += E25Counter.Now(metrics) - begin;
                }

                void Position(int q, double[] x, int j, double d, int stencil, float[] column)
                {
                    var (a, b) = E24Weights(d, stencil);
                    if (offPosition) Probe(ref q, x, j, d, stencil, LightShadows.None, "xyz", plus, minus);
                    if (hardPosition) Probe(ref q, x, j, d, stencil, LightShadows.Hard, "xyz", hardA, hardB);
                    int owner = settings.Layout?.Owner(j) ?? j / 4;
                    var v = masked ? mask[owner] : null;
                    long begin = E25Counter.Now(metrics);
                    if (gpu != null)
                    {
                        gpu.Position(plus, minus, v, owner, d, stencil, column);
                        if (metrics != null) metrics.mapTicks += E25Counter.Now(metrics) - begin;
                        return;
                    }
                    if (cachedMapping)
                    {
                        E25CachedColumn(objective, hard, hardMapped, off, plus, minus, v, d, stencil, column);
                        if (metrics != null) metrics.mapTicks += E25Counter.Now(metrics) - begin;
                        return;
                    }
                    for (int i = 0; i < column.Length; ++i)
                    {
                        float k = v == null ? 1 : v[i / 3];
                        double result;
                        if (anchored)
                        {
                            float center = objective.Transform(hard[i]);
                            double p = objective.Transform(hard[i] + k * (plus[i] - off[i])) - center;
                            double m = stencil == 1 ? 0 : objective.Transform(hard[i] + k * (minus[i] - off[i])) - center;
                            result = a * p + b * m;
                        }
                        else
                        {
                            double s = offPosition ? k * (a * (plus[i] - off[i]) + b * (minus[i] - off[i])) : 0;
                            double h = hardPosition ? a * (hardA[i] - hard[i]) + b * (hardB[i] - hard[i]) : 0;
                            result = model == "H3" ? h : model == "H5" ? .5 * (s + h) : s;
                        }
                        column[i] = (float)(weights[i] * result);
                    }
                    if (metrics != null) metrics.mapTicks += E25Counter.Now(metrics) - begin;
                }

                void Intensity(int q, double[] x, int j, double d, int stencil, float[] column)
                {
                    int query = q;
                    var (a, b) = E24Weights(d, stencil);
                    Probe(ref q, x, j, d, stencil, LightShadows.Hard, "intensity", hardA, hardB);
                    Probe(ref q, x, j, d, stencil, LightShadows.None, "intensity", plus, minus);
                    long begin = E25Counter.Now(metrics);
                    for (int i = 0; i < column.Length; ++i)
                    {
                        float h = hardMapped == null ? objective.Transform(hard[i]) : hardMapped[i];
                        float p0 = objective.Transform(hardA[i]), m0 = stencil == 1 ? 0 : objective.Transform(hardB[i]);
                        double value = cachedMapping && stencil == 3 ? a * ((double)p0 - m0) : a * (p0 - h) + b * (stencil == 1 ? 0 : m0 - h);
                        column[i] = (float)(objective.weights[i / E23ChannelsPerView] * value);
                        hardA[i] = (float)(a * (hardA[i] - hard[i]) + b * (hardB[i] - hard[i]));
                        plus[i] = (float)(a * (plus[i] - off[i]) + b * (minus[i] - off[i]));
                    }
                    if (metrics != null) metrics.mapTicks += E25Counter.Now(metrics) - begin;
                    begin = E25Counter.Now(metrics);
                    E24Mask(hardA, plus, mask[settings.Layout?.Owner(j) ?? j / 4], norms, E24RelativeThreshold, audit, state, "model", query, active[settings.Layout?.Owner(j) ?? j / 4], d * (settings.Layout?.Axis(j).Unit ?? settings.Units[j]), stencil,
                        fallback: model == "C4-soft" ? .5f : 1f);
                    if (metrics != null) metrics.maskTicks += E25Counter.Now(metrics) - begin;
                }

                void Probe(ref int q, double[] x, int j, double d, int stencil, LightShadows shadows, string phase, float[] p, float[] m)
                {
                    var shifted = probeRig ?? (double[])x.Clone();
                    if (probeRig != null) Array.Copy(x, shifted, x.Length);
                    int coordinate = active[settings.Layout?.Owner(j) ?? j / 4] * 4 + (settings.Layout == null ? j % 4 : (int)settings.Layout.Axis(j).Coordinate);
                    shifted[j] += d;
                    capture(q++, expand(shifted), shadows, phase, coordinate, d * (settings.Layout?.Axis(j).Unit ?? settings.Units[j]), p);
                    if (stencil == 1) return;
                    double next = stencil == 3 ? -d : 2 * d;
                    shifted[j] = x[j] + next;
                    capture(q++, expand(shifted), shadows, phase, coordinate, next * (settings.Layout?.Axis(j).Unit ?? settings.Units[j]), m);
                }
            }

            private static (double a, double b) E24Weights(double d, int stencil) =>
                ((stencil == 3 ? .5 : stencil == 2 ? 2 : 1) / d, stencil == 1 ? 0 : -.5 / d);

            private static float E24Quantile(float[] x, int rank)
            {
                int left = 0, right = x.Length - 1;
                while (left < right)
                {
                    int i = left, j = right; float pivot = x[(left + right) / 2];
                    while (i <= j)
                    {
                        while (i <= right && x[i] < pivot) ++i;
                        while (j >= left && x[j] > pivot) --j;
                        if (i <= j) { (x[i], x[j]) = (x[j], x[i]); ++i; --j; }
                    }
                    if (rank <= j) right = j;
                    else if (rank >= i) left = i;
                    else return x[rank];
                }
                return x[left];
            }

            private static void E24Mask(float[] h, float[] o, float[] mask, float[] norm, float relative,
                StreamWriter audit, string state, string purpose, int seq, int lamp, double step, int stencil, float fallback = 1f)
            {
                for (int view = 0; view < E19FitViews; ++view)
                {
                    int begin = view * E24PixelsPerView;
                    for (int p = 0, k = 3 * begin; p < norm.Length; ++p, k += 3)
                        norm[p] = o[k] * o[k] + o[k + 1] * o[k + 1] + o[k + 2] * o[k + 2];
                    double p95 = Math.Sqrt(E24Quantile(norm, (int)Math.Ceiling(.95 * norm.Length) - 1));
                    double floor = Math.Max(1e-12, relative * p95), min2 = floor * floor, epsilon = Math.Max(1e-30, min2 * 1e-6);
                    int weak = 0, bad = 0, neg = 0, under = 0, over = 0, ratios = 0, cosCount = 0, covered = 0;
                    double ch2 = 0, co2 = 0, residual2 = 0, ratioSum = 0, rMin = double.PositiveInfinity, rMax = double.NegativeInfinity, cs = 0, sum = 0;
                    for (int p = begin, k = 3 * begin; p < begin + norm.Length; ++p, k += 3)
                    {
                        double a = h[k], b = h[k + 1], c = h[k + 2], x = o[k], y = o[k + 1], z = o[k + 2];
                        double hh = a * a + b * b + c * c, oo = x * x + y * y + z * z, dot = a * x + b * y + c * z;
                        double raw = dot / (oo + epsilon), clamp = Math.Max(0, Math.Min(1, raw)), v = fallback;
                        ch2 += hh; co2 += oo;
                        if (a < -floor || b < -floor || c < -floor || x < -floor || y < -floor || z < -floor) ++neg;
                        // Compare the unclamped ratio and color alignment before deciding if this is a usable scalar model.
                        double clippedError = (a - clamp * x) * (a - clamp * x) + (b - clamp * y) * (b - clamp * y) + (c - clamp * z) * (c - clamp * z);
                        residual2 += clippedError;
                        if (oo <= min2) ++weak;
                        else
                        {
                            ++ratios; ratioSum += raw; rMin = Math.Min(rMin, raw); rMax = Math.Max(rMax, raw);
                            if (raw < 0) ++under;
                            if (raw > 1) ++over;
                            double cosine = hh <= min2 ? 1 : dot / Math.Sqrt(hh * oo);
                            if (hh > min2) { cs += cosine; ++cosCount; }
                            if (hh > min2 && (cosine < .95 || clippedError > .04 * hh || raw < -.05 || raw > 1.05)) ++bad;
                            else v = clamp;
                        }
                        mask[p] = (float)v; sum += v; if (v < .5) ++covered;
                    }
                    double n = norm.Length;
                    audit?.WriteLine(string.Join(",", state, purpose, seq, lamp + 1, view, F(step), stencil, F(relative), F(p95), F(floor), F(epsilon),
                        F(Math.Sqrt(ch2)), F(Math.Sqrt(co2)), F(weak / n), F(bad / n), F(neg / n), F((double)under / Math.Max(1, ratios)), F((double)over / Math.Max(1, ratios)),
                        F(ratios == 0 ? double.NaN : ratioSum / ratios), F(ratios == 0 ? double.NaN : rMin), F(ratios == 0 ? double.NaN : rMax),
                        F(Math.Sqrt(residual2 / (ch2 + epsilon))), F(cosCount == 0 ? double.NaN : cs / cosCount), F(sum / n), F(covered / n)));
                }
            }
        }
    }
}

using System;
using System.Linq;
using UnityEngine;

namespace LightingMvp.Editor
{
    [Serializable]
    internal struct RecoveryLightState
    {
        public Vector3 position;
        public float intensity;
        public Color color;

        public static RecoveryLightState Read(Light light) => new RecoveryLightState
            { position = light.transform.position, intensity = light.intensity, color = light.color };

        public void Apply(Light light)
        {
            light.transform.position = position;
            light.intensity = intensity;
            light.color = color;
        }
    }

    /// <summary>Normalized box coordinates, with a geometry-derived floor under the lights.</summary>
    internal sealed class RecoveryRigCoordinates
    {
        private readonly Light[] lights;
        private readonly Color[] fixedColors;
        private readonly BoxCollider tallBlock;
        private readonly bool fitColor;
        private const float Xz = .91f, Ceiling = 1.94f, ChromaLimit = 2.6f;
        private float MinPower => fitColor ? .018f : .055f;
        private const float MaxPower = 9;
        private int Stride => fitColor ? 6 : 4;

        internal RecoveryRigCoordinates(Light[] lights, BoxCollider tallBlock, bool fitColor)
        {
            this.lights = lights;
            this.tallBlock = tallBlock;
            this.fitColor = fitColor;
            fixedColors = lights.Select(l => l.color).ToArray();
        }

        private float Floor(float x, float z)
        {
            Vector3 p = tallBlock.transform.InverseTransformPoint(new Vector3(x, tallBlock.transform.position.y, z)) - tallBlock.center;
            float dx = Mathf.Abs(p.x) - .5f * tallBlock.size.x, dz = Mathf.Abs(p.z) - .5f * tallBlock.size.z;
            // Tall block is vertical. Fade its exclusion footprint outside the collider to retain a continuous map.
            float w = Mathf.SmoothStep(1, 0, (Mathf.Max(dx, dz) - .025f) / .18f);
            return Mathf.Lerp(.65f, tallBlock.bounds.max.y + .045f, w);
        }

        internal double[] Encode(RecoveryLightState[] rig, bool clamp = false)
        {
            var theta = new double[lights.Length * Stride];
            for (int i = 0; i < lights.Length; ++i)
            {
                int o = i * Stride;
                var s = rig[i];
                theta[o] = (s.position.x + Xz) / (2 * Xz);
                float floor = Floor(s.position.x, s.position.z);
                theta[o + 1] = (s.position.y - floor) / (Ceiling - floor);
                theta[o + 2] = (s.position.z + Xz) / (2 * Xz);
                float power = s.intensity;
                if (fitColor)
                {
                    Color c = s.color.linear;
                    power *= c.r + c.g + c.b;
                    theta[o + 4] = .5 + Math.Log(c.r / c.g) / (2 * ChromaLimit);
                    theta[o + 5] = .5 + Math.Log(c.b / c.g) / (2 * ChromaLimit);
                }
                theta[o + 3] = Math.Log(power / MinPower) / Math.Log(MaxPower / MinPower);
            }
            for (int i = 0; i < theta.Length; ++i)
            {
                if (clamp) theta[i] = Math.Max(.105, Math.Min(.895, theta[i]));
                else if (!(theta[i] >= 0 && theta[i] <= 1)) throw new InvalidOperationException("Rig is outside the prescribed recovery domain.");
            }
            return theta;
        }

        internal RecoveryLightState[] Decode(double[] theta)
        {
            var rig = new RecoveryLightState[lights.Length];
            for (int i = 0; i < rig.Length; ++i) rig[i] = DecodeLight(theta, i);
            return rig;
        }

        private RecoveryLightState DecodeLight(double[] theta, int i)
        {
            int o = i * Stride;
            float x = Mathf.Lerp(-Xz, Xz, (float)theta[o]), z = Mathf.Lerp(-Xz, Xz, (float)theta[o + 2]);
            Color color = fixedColors[i];
            if (fitColor)
            {
                double r = Math.Exp((2 * theta[o + 4] - 1) * ChromaLimit);
                double b = Math.Exp((2 * theta[o + 5] - 1) * ChromaLimit);
                double sum = r + 1 + b;
                color = new Color((float)(r / sum), (float)(1 / sum), (float)(b / sum), fixedColors[i].a).gamma;
            }
            return new RecoveryLightState
            {
                position = new Vector3(x, Mathf.Lerp(Floor(x, z), Ceiling, (float)theta[o + 1]), z),
                intensity = (float)(MinPower * Math.Exp(theta[o + 3] * Math.Log(MaxPower / MinPower))),
                color = color
            };
        }

        internal void Apply(double[] theta)
        {
            for (int i = 0; i < lights.Length; ++i) DecodeLight(theta, i).Apply(lights[i]);
        }

        internal RecoveryLightState[] Perturb(RecoveryLightState[] target, int seed, float distance)
        {
            var random = new System.Random(seed);
            var rig = (RecoveryLightState[])target.Clone();
            for (int i = 0; i < rig.Length; ++i)
            {
                // Deterministically vary direction and increase scale for the prespecified hard starts.
                double angle = 2 * Math.PI * random.NextDouble();
                rig[i].position += distance * new Vector3((float)Math.Cos(angle), .8f * ((float)random.NextDouble() - .5f), (float)Math.Sin(angle));
                rig[i].intensity *= (float)Math.Exp((random.Next(2) == 0 ? -1 : 1) * (.45 + distance * random.NextDouble()));
                double dr = (2 * random.NextDouble() - 1) * (1.2 + distance), db = (2 * random.NextDouble() - 1) * (1.2 + distance);
                if (!fitColor) continue;
                Color c = rig[i].color.linear;
                float originalSum = c.r + c.g + c.b;
                c.r *= (float)Math.Exp(dr); c.b *= (float)Math.Exp(db);
                float sum = c.r + c.g + c.b;
                c.r /= sum; c.g /= sum; c.b /= sum;
                rig[i].color = c.gamma;
                rig[i].intensity *= originalSum;
            }
            return rig;
        }
    }
}

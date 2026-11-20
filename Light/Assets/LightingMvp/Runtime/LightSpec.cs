using System;
using UnityEngine;

namespace LightingMvp
{
    public enum LightKind { Point = 0, Spot = 1 }

    [Serializable]
    public sealed class LightSpec
    {
        public Vector3 position;
        public float intensity = 2.0f;
        public Color color = Color.white;
        public LightKind kind = LightKind.Point;
        public Vector3 direction = Vector3.down;
        public float range = 100.0f;
        public float attenuation = 1.0e-4f;
        public float innerAngle = 25.0f;
        public float outerAngle = 45.0f;

        public LightSpec Clone() { return (LightSpec)MemberwiseClone(); }
        public static LightSpec[] CloneArray(LightSpec[] source)
        {
            var result = new LightSpec[source.Length];
            for (int i = 0; i < source.Length; i++) result[i] = source[i].Clone();
            return result;
        }
    }
}

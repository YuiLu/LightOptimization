using System;
using UnityEngine;

namespace LightingMvp
{
    /// <summary>Two normalized tangent-plane coordinates on a bounded patch of the sphere.</summary>
    public readonly struct SpotAimCoordinates
    {
        private readonly Vector3 forward, right, up;
        private readonly float slope;

        public SpotAimCoordinates(Quaternion reference, float halfAxisDegrees)
        {
            if (!(halfAxisDegrees > 0 && halfAxisDegrees < 85)) throw new ArgumentOutOfRangeException(nameof(halfAxisDegrees));
            forward = reference * Vector3.forward;
            right = reference * Vector3.right;
            up = reference * Vector3.up;
            slope = Mathf.Tan(halfAxisDegrees * Mathf.Deg2Rad);
        }

        public Quaternion Decode(double x, double y)
        {
            var direction = (forward + right * ((float)(2 * x - 1) * slope) + up * ((float)(2 * y - 1) * slope)).normalized;
            // Transport the reference frame by the minimum rotation; roll is not an optimization axis.
            return Quaternion.LookRotation(direction, Quaternion.FromToRotation(forward, direction) * up);
        }

        public Vector2 Encode(Vector3 direction)
        {
            float z = Vector3.Dot(direction, forward);
            if (!(z > 0)) throw new ArgumentException("Spot aim is outside its reference hemisphere.");
            var uv = new Vector2(.5f + Vector3.Dot(direction, right) / (2 * z * slope), .5f + Vector3.Dot(direction, up) / (2 * z * slope));
            if (!(uv.x >= 0 && uv.x <= 1 && uv.y >= 0 && uv.y <= 1)) throw new ArgumentException("Spot aim is outside its normalized domain.");
            return uv;
        }
    }
}

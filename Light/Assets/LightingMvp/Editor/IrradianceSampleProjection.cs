using System;
using UnityEngine;

namespace LightingMvp.Editor
{
    // Match projected samples at subpixel positions. Every contributing tap must lie on
    // the receiver's local tangent plane, not on the foreground or the background.
    internal readonly struct IrradianceSampleProjection
    {
        public readonly int CameraIndex;
        public readonly Vector2 Pixel;
        public readonly float EyeDepth;
        private readonly float wx, wy;
        private readonly int a, b, c, d;

        private IrradianceSampleProjection(int view, Vector2 pixel, float depth, int width)
        {
            CameraIndex = view; Pixel = pixel; EyeDepth = depth;
            int x = Mathf.FloorToInt(pixel.x), y = Mathf.FloorToInt(pixel.y);
            a = y * width + x; b = a + 1; c = a + width; d = c + 1;
            wx = pixel.x - x; wy = pixel.y - y;
        }

        public static bool TryMatch(int view, Camera camera, NativeIrradianceCapture.Frame frame, Vector3 position, Vector3 normal,
            out IrradianceSampleProjection match, out float score)
        {
            match = default;
            score = 0;
            Vector3 vp = camera.WorldToViewportPoint(position);
            if (vp.z <= camera.nearClipPlane || vp.z >= camera.farClipPlane) return false;
            var p = new Vector2(vp.x * frame.Width - .5f, vp.y * frame.Height - .5f);
            if (p.x < 0 || p.x >= frame.Width - 1 || p.y < 0 || p.y >= frame.Height - 1) return false;
            Ray central = camera.ViewportPointToRay(vp);
            score = -Vector3.Dot(central.direction, normal);
            if (score < .18f) return false;

            int x = Mathf.FloorToInt(p.x), y = Mathf.FloorToInt(p.y);
            for (int j = y; j <= y + 1; j++)
            for (int i = x; i <= x + 1; i++)
            {
                float depth = frame.Pixels[j * frame.Width + i].a;
                if (depth <= 0) return false;
                Ray ray = camera.ViewportPointToRay(new Vector3((i + .5f) / frame.Width, (j + .5f) / frame.Height));
                float denominator = Vector3.Dot(ray.direction, normal);
                if (denominator > -.1f) return false;
                Vector3 receiver = ray.GetPoint(Vector3.Dot(position - ray.origin, normal) / denominator);
                float expected = Vector3.Dot(receiver - camera.transform.position, camera.transform.forward);
                // Depth tolerance accounts for rasterization/float precision, not receiver visibility along light rays.
                if (Mathf.Abs(expected - depth) > .003f + .0005f * depth) return false;
            }
            match = new IrradianceSampleProjection(view, p, vp.z, frame.Width);
            return true;
        }

        public Vector3 Read(NativeIrradianceCapture.Frame frame, NativeIrradianceCapture.Frame reference)
        {
            Verify(a); Verify(b); Verify(c); Verify(d);
            return Vector3.LerpUnclamped(Vector3.LerpUnclamped(Rgb(frame.Pixels[a]), Rgb(frame.Pixels[b]), wx),
                Vector3.LerpUnclamped(Rgb(frame.Pixels[c]), Rgb(frame.Pixels[d]), wx), wy);

            void Verify(int index)
            {
                if (frame.Pixels[index].a <= 0 || Mathf.Abs(frame.Pixels[index].a - reference.Pixels[index].a) > .0005f)
                    throw new InvalidOperationException("Receiver depth changed between captures; comparison would not use common samples.");
            }
        }

        public static Vector3 Rgb(Color color) => new Vector3(color.r, color.g, color.b);
    }
}

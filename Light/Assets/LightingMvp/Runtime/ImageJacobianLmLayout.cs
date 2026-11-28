using System;
using System.Collections.Generic;
using System.Linq;

namespace LightingMvp
{
    public static partial class ImageJacobianLm
    {
        public enum TrustCoordinate { X, Y, Z, LogIntensity, Mired, LogRedGreen, LogBlueGreen }

        /// <summary>Physical bounds [Minimum, Minimum + Unit] are encoded as [0,1].</summary>
        public readonly struct TrustAxis
        {
            public readonly TrustCoordinate Coordinate;
            public readonly double Unit, Step, Radius, MaximumRadius, Minimum;
            public TrustAxis(TrustCoordinate kind, double unit, double step, double radius, double maximumRadius, double minimum = 0)
            {
                if (!(unit > 0 && step > 0 && radius > 0 && maximumRadius >= radius) ||
                    double.IsInfinity(unit + step + radius + maximumRadius) || double.IsNaN(minimum) || double.IsInfinity(minimum))
                    throw new ArgumentException("Invalid physical axis.");
                Coordinate = kind; Unit = unit; Step = step; Radius = radius; MaximumRadius = maximumRadius; Minimum = minimum;
            }
            public bool IsPosition => Coordinate <= TrustCoordinate.Z;
            public bool IsIntensity => Coordinate == TrustCoordinate.LogIntensity;
        }

        public sealed class TrustLayout
        {
            internal readonly TrustAxis[] Axes;
            internal readonly int[] Owners, Starts, Widths;
            public int Count => Axes.Length;
            public int Lamps => Starts.Length;
            public TrustAxis Axis(int j) => Axes[j];
            public int Owner(int j) => Owners[j];
            public int Start(int lamp) => Starts[lamp];
            public int Width(int lamp) => Widths[lamp];

            public TrustLayout(params TrustAxis[][] blocks)
            {
                if (blocks == null || blocks.Length == 0) throw new ArgumentException("Expected lamp blocks.");
                var axes = new List<TrustAxis>(); var owners = new List<int>();
                Starts = new int[blocks.Length]; Widths = new int[blocks.Length];
                for (int i = 0; i < blocks.Length; ++i)
                {
                    var b = blocks[i];
                    if (b == null || b.Length == 0 || b.Any(a => !(a.Unit > 0)) ||
                        b.Select(a => a.Coordinate).Distinct().Count() != b.Length || b.Count(a => a.IsPosition) is not (0 or 3))
                        throw new ArgumentException("Each lamp must declare unique coordinates and either all XYZ or none.");
                    Starts[i] = axes.Count; Widths[i] = b.Length;
                    axes.AddRange(b); owners.AddRange(Enumerable.Repeat(i, b.Length));
                }
                Axes = axes.ToArray(); Owners = owners.ToArray();
            }

            internal static TrustLayout Legacy(double[] units, double[] steps)
            {
                if (units == null || steps == null || units.Length == 0 || units.Length % 4 != 0 || steps.Length != units.Length)
                    throw new ArgumentException("Expected legacy XYZ/log-intensity blocks.");
                var blocks = new TrustAxis[units.Length / 4][];
                for (int lamp = 0; lamp < blocks.Length; ++lamp)
                    blocks[lamp] = Enumerable.Range(0, 4).Select(k => new TrustAxis((TrustCoordinate)k,
                        units[4 * lamp + k], steps[4 * lamp + k], k < 3 ? .24 : .28, k < 3 ? .70 : .75)).ToArray();
                return new TrustLayout(blocks);
            }
        }

        public enum TrustFallback { None, ViewPool, ViewPairPool, Cauchy, Nonmonotone, Dogleg }
    }
}

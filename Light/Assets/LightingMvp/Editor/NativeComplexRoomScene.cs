using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;

namespace LightingMvp.Editor
{
    public static partial class NativeRecoveryExperiment
    {
        /// <summary>
        /// Classroom-scale controlled scene: seven realtime point lights with
        /// shadow-casting fixtures inside a cluttered URP room.
        /// </summary>
        private sealed class ComplexRoomSensitivityScene : ISensitivityScene
        {
            private static readonly Vector3 MinBounds = new Vector3(-4.35f, .25f, -3.50f), MaxBounds = new Vector3(4.35f, 2.85f, 3.50f);
            private readonly Light[] lights;
            private readonly Camera camera;
            private readonly double[] truth;
            private readonly string[][] fixtureRenderers;
            private readonly int[] fixtureTriangles;
            private readonly Bounds[] fixtureBounds;
            private static readonly string ScenePath = NativeComplexRoomBuilder.ScenePath;

            internal ComplexRoomSensitivityScene()
            {
                var scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
                lights = Enumerable.Range(1, 7).Select(i => GameObject.Find($"Candidate Light {i:00}")?.GetComponent<Light>()).ToArray();
                if (lights.Any(l => !l)) throw new InvalidOperationException(ScenePath + " is missing Candidate Light 01..07. Run NativeComplexRoomBuilder.Build first.");
                camera = GameObject.Find("Validation Camera")?.GetComponent<Camera>();
                if (!camera) throw new InvalidOperationException("Validation Camera missing from the native complex room scene.");
                if (scene.isDirty) throw new InvalidOperationException("The native complex room scene has unsaved changes in the editor.");
                truth = lights.SelectMany(l => Encode(l.transform.position, l.intensity)).ToArray();
                if (!Contains(truth)) throw new InvalidOperationException("Authored complex-room rig is outside the search domain.");

                fixtureRenderers = new string[lights.Length][];
                fixtureTriangles = new int[lights.Length];
                fixtureBounds = new Bounds[lights.Length];
                for (int i = 0; i < lights.Length; ++i)
                {
                    string prefix = $"Fixture L{i + 1:00} ";
                    var renderers = UnityEngine.Object.FindObjectsOfType<MeshRenderer>(true)
                        .Where(r => r.name.StartsWith(prefix, StringComparison.Ordinal)).ToArray();
                    fixtureRenderers[i] = renderers.Select(r => r.name).ToArray();
                    fixtureTriangles[i] = renderers.Sum(r => r.GetComponent<MeshFilter>()?.sharedMesh ? r.GetComponent<MeshFilter>().sharedMesh.triangles.Length / 3 : 0);
                    if (renderers.Length == 0) { fixtureBounds[i] = default; continue; }
                    var b = renderers[0].bounds;
                    foreach (var r in renderers.Skip(1)) b.Encapsulate(r.bounds);
                    fixtureBounds[i] = b;
                }
            }

            public string Name => "native_complex_room";
            public string RendererName => "URP stock Forward+ (Classroom-equivalent)";
            public int Count => lights.Length;
            public int[] Lamps => Enumerable.Range(0, Count).ToArray();
            public double[] Truth => truth;
            public Vector3[] Views => NativeComplexRoomBuilder.Eyes;
            public Quaternion[] Rotations => Views.Select((e, i) => Quaternion.LookRotation((NativeComplexRoomBuilder.Look[i] - e).normalized, Vector3.up)).ToArray();
            public Camera Camera => camera;
            public Vector3 Extents => MaxBounds - MinBounds;
            public string StartSource => "scene_seeded_joint";

            private static double[] Encode(Vector3 p, double power) => new[] {
                ((double)p.x - MinBounds.x) / (MaxBounds.x - MinBounds.x), ((double)p.y - MinBounds.y) / (MaxBounds.y - MinBounds.y),
                ((double)p.z - MinBounds.z) / (MaxBounds.z - MinBounds.z), PowerCoordinate(power) };

            private static Vector3 Decode(double[] x, int k) => new Vector3(
                Mathf.Lerp(MinBounds.x, MaxBounds.x, (float)x[k]), Mathf.Lerp(MinBounds.y, MaxBounds.y, (float)x[k + 1]),
                Mathf.Lerp(MinBounds.z, MaxBounds.z, (float)x[k + 2]));

            private static double Power(double q) => PowerMin * Math.Exp(q * LogPowerSpan);

            public double[] Start(int seed)
            {
                var rng = new System.Random(seed); var theta = (double[])truth.Clone();
                for (int i = 0; i < Count; ++i)
                {
                    int k = i * LampDofs;
                    var p = Decode(truth, k); double q = Power(truth[k + 3]);
                    var offset = .50f + .18f * (float)rng.NextDouble();
                    q = Math.Max(PowerMin, Math.Min(PowerMax, q * (rng.NextDouble() < .5 ? .48 : 1.85)));
                    bool placed = false;
                    for (int attempt = 0; attempt < 4096 && !placed; ++attempt)
                    {
                        if (attempt > 0 && attempt % 256 == 0) offset = Math.Max(.08f, offset * .8f);
                        var d = new Vector3((float)rng.NextDouble() - .5f, (float)rng.NextDouble() - .5f, (float)rng.NextDouble() - .5f).normalized;
                        var lamp = Encode(p + offset * d, q);
                        if (!Contains(lamp)) continue;
                        Array.Copy(lamp, 0, theta, k, LampDofs); placed = true;
                    }
                    if (!placed) throw new InvalidOperationException("No admissible perturbed complex-room pose for light " + (i + 1) + ".");
                }
                return theta;
            }

            public void Apply(double[] theta)
            {
                if (!Contains(theta)) throw new InvalidOperationException("Illegal complex-room rig submitted.");
                for (int i = 0; i < Count; ++i)
                {
                    int k = i * LampDofs;
                    lights[i].transform.position = Decode(theta, k);
                    lights[i].intensity = (float)Power(theta[k + 3]);
                }
            }

            public void SetShadows(int lamp, bool shadows) => lights[lamp].shadows = shadows ? LightShadows.Hard : LightShadows.None;
            public void ResetShadows() { foreach (var lamp in lights) { lamp.shadows = LightShadows.Hard; lamp.shadowStrength = 1; } }
            public bool SupportsFixtureShadows => true;

            public void SetFixtureShadows(int lamp, bool casting)
            {
                foreach (var name in fixtureRenderers[lamp])
                {
                    var go = GameObject.Find(name);
                    var renderer = go ? go.GetComponent<MeshRenderer>() : null;
                    if (renderer) renderer.shadowCastingMode = casting ? ShadowCastingMode.On : ShadowCastingMode.Off;
                }
            }

            public void SetShadowStrength(int lamp, float strength) => lights[lamp].shadowStrength = Mathf.Clamp01(strength);
            public string SourceName(int lamp) => "Candidate Light " + (lamp + 1).ToString("00", Invariant);
            public string UnityName(int lamp) => "Candidate Light " + (lamp + 1).ToString("00", Invariant);
            public string[] FixtureRenderers(int lamp) => fixtureRenderers[lamp];
            public int[] FixtureTriangleCounts() => (int[])fixtureTriangles.Clone();
            public Bounds FixtureBounds(int lamp) => fixtureBounds[lamp];

            public double[] WithPosition(double[] theta, int lamp, Vector3 position)
            {
                var x = (double[])theta.Clone(); int k = lamp * LampDofs;
                var moved = Encode(position, Power(theta[k + 3]));
                Array.Copy(moved, 0, x, k, 3);
                return x;
            }

            public Vector3 PositionOf(double[] theta, int lamp) => Decode(theta, lamp * LampDofs);
            public double PowerOf(double[] theta, int lamp) => Power(theta[lamp * LampDofs + 3]);

            public bool Contains(double[] theta)
            {
                if (theta == null || theta.Length % LampDofs != 0 || theta.Length > Count * LampDofs || theta.Any(u => !(u >= 0 && u <= 1))) return false;
                for (int i = 0; i < theta.Length; i += LampDofs)
                {
                    var p = Decode(theta, i);
                    if (Physics.CheckSphere(p, .06f, ~0, QueryTriggerInteraction.Ignore)) return false;
                    if ((Mathf.Abs(p.x) > 4.2f) || (Mathf.Abs(p.z) > 3.35f) || p.y < .35f || p.y > 2.80f) return false;
                }
                return true;
            }

            public E17Config Describe() => new E17Config
            {
                scene = Name, unity = Application.unityVersion,
                urp = UnityEditor.PackageManager.PackageInfo.FindForAssembly(typeof(UnityEngine.Rendering.Universal.UniversalRenderPipeline).Assembly)?.version,
                renderer = RendererName, gpu = SystemInfo.graphicsDeviceName, graphics = SystemInfo.graphicsDeviceType.ToString(),
                colorSpace = QualitySettings.activeColorSpace.ToString(), lamps = Count, trainingViews = E19FitViews, checkViews = Views.Length,
                width = E19FitW, height = E19FitH, checkWidth = E19HighW, checkHeight = E19HighH,
                boundsMin = MinBounds, boundsMax = MaxBounds, bakedGi = false, realtimeGi = false,
                ambient = RenderSettings.ambientLight.linear.r,
                gt = lights.Select(NativeLamp.Read).ToArray(), viewPositions = Views, viewRotations = Rotations
            };

            public void Dispose() { }
        }
    }
}


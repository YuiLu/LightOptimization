import math
import unittest

import torch

from mvp.lighting import (
    AnalyticLightRig,
    LightSpec,
    SurfaceSamples,
    cornell_samples,
    field_nrmse,
    weighted_loss,
)


class LightingTests(unittest.TestCase):
    def setUp(self):
        self.samples = cornell_samples(6).to(torch.float64)

    def test_sample_geometry(self):
        self.assertEqual(len(self.samples.areas), 5 * 6 * 6)
        self.assertAlmostEqual(self.samples.areas.sum().item(), 20.0, places=10)
        self.assertTrue(torch.allclose(self.samples.normals.norm(dim=-1), torch.ones(len(self.samples.areas), dtype=torch.float64)))

    def test_exact_rig_has_zero_error(self):
        spec = LightSpec(position=[0.1, 1.2, 0.0], intensity=2.0)
        rig = AnalyticLightRig([spec], set(), torch.float64)
        with torch.no_grad():
            target = rig(self.samples)
        self.assertEqual(weighted_loss(target, target, self.samples).item(), 0.0)
        self.assertEqual(field_nrmse(target, target, self.samples).item(), 0.0)

    def test_positive_intensity_and_unit_direction(self):
        spec = LightSpec(position=[0, 1, 0], kind="spot", direction=[0, -2, 0])
        rig = AnalyticLightRig([spec], {"position", "intensity", "direction"}, torch.float64)
        rig.log_intensity.data.fill_(-20)
        self.assertGreater(rig.log_intensity.exp().item(), 0.0)
        self.assertAlmostEqual(rig.normalized_direction().norm(dim=-1).item(), 1.0, places=12)

    def test_output_shape_and_light_additivity(self):
        specs = [
            LightSpec(position=[-0.5, 1.4, -0.2], intensity=1.8, color=[1.0, 0.7, 0.5]),
            LightSpec(position=[0.6, 0.9, 0.4], intensity=2.2, color=[0.6, 0.8, 1.0]),
        ]
        combined = AnalyticLightRig(specs, set(), torch.float64)(self.samples)
        self.assertEqual(combined.shape, (len(self.samples.areas), 3))

        expected = sum(
            AnalyticLightRig([spec], set(), torch.float64)(self.samples)
            for spec in specs
        )
        self.assertEqual(expected.shape, combined.shape)
        self.assertTrue(torch.allclose(combined, expected, rtol=1e-12, atol=1e-14))

    def test_point_light_matches_independent_scalar_formula(self):
        samples = SurfaceSamples(
            positions=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals=[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            areas=[0.5, 0.25],
            names=["floor", "wall"],
            dtype=torch.float64,
        )
        spec = LightSpec(
            position=[0.5, 2.0, -0.25], intensity=3.0,
            color=[1.0, 0.5, 0.25], range=10.0, attenuation=0.2,
        )
        actual = AnalyticLightRig([spec], set(), torch.float64)(samples)

        expected = torch.zeros((2, 3), dtype=torch.float64)
        for sample_index, (position, normal) in enumerate(
            zip(samples.positions.tolist(), samples.normals.tolist())
        ):
            delta = [light - surface for light, surface in zip(spec.position, position)]
            distance2 = sum(component * component for component in delta)
            distance = math.sqrt(distance2)
            cosine = max(0.0, sum(
                n * component / distance for n, component in zip(normal, delta)
            ))
            range_window = max(0.0, 1.0 - distance2 / (spec.range * spec.range)) ** 2
            attenuation = spec.attenuation / (spec.attenuation + distance2) * range_window
            for channel in range(3):
                expected[sample_index, channel] = (
                    spec.intensity * attenuation * cosine * spec.color[channel]
                )

        self.assertEqual(actual.shape, (2, 3))
        self.assertTrue(torch.allclose(actual, expected, rtol=1e-12, atol=1e-14))

    def test_two_light_autograd_matches_finite_difference(self):
        initial = [
            LightSpec(position=[-0.4, 1.2, -0.1], intensity=1.1, color=[1.0, 0.7, 0.5]),
            LightSpec(position=[0.5, 1.0, 0.3], intensity=1.4, color=[0.6, 0.8, 1.0]),
        ]
        target_specs = [
            LightSpec(position=[-0.55, 1.45, -0.15], intensity=1.8, color=[1.0, 0.7, 0.5]),
            LightSpec(position=[0.6, 0.85, 0.45], intensity=2.2, color=[0.6, 0.8, 1.0]),
        ]
        rig = AnalyticLightRig(initial, {"position", "intensity"}, torch.float64)
        with torch.no_grad():
            target = AnalyticLightRig(target_specs, set(), torch.float64)(self.samples)
        loss = weighted_loss(rig(self.samples), target, self.samples)
        loss.backward()

        for name, parameter in rig.named_parameters():
            if not parameter.requires_grad:
                continue
            for index in range(parameter.numel()):
                flat = parameter.data.reshape(-1)
                original = flat[index].item()
                analytic = parameter.grad.reshape(-1)[index].item()
                for h in (1e-4, 1e-5, 1e-6):
                    flat[index] = original + h
                    plus = weighted_loss(rig(self.samples), target, self.samples).item()
                    flat[index] = original - h
                    minus = weighted_loss(rig(self.samples), target, self.samples).item()
                    flat[index] = original
                    finite = (plus - minus) / (2 * h)
                    self.assertTrue(
                        math.isclose(analytic, finite, rel_tol=1e-4, abs_tol=1e-9),
                        f"{name}[{index}] at h={h}: autograd={analytic}, FD={finite}",
                    )

    def test_autograd_matches_finite_difference(self):
        spec = LightSpec(position=[0.2, 1.1, 0.0], intensity=2.5)
        rig = AnalyticLightRig([spec], {"position", "intensity"}, torch.float64)
        with torch.no_grad():
            target = AnalyticLightRig([LightSpec(position=[0.4, 1.4, .2])], set(), torch.float64)(self.samples)
        loss = weighted_loss(rig(self.samples), target, self.samples)
        loss.backward()
        h = 1e-5
        for index in range(3):
            original = rig.position.data[0, index].item()
            rig.position.data[0, index] = original + h
            plus = weighted_loss(rig(self.samples), target, self.samples).item()
            rig.position.data[0, index] = original - h
            minus = weighted_loss(rig(self.samples), target, self.samples).item()
            rig.position.data[0, index] = original
            finite = (plus - minus) / (2 * h)
            self.assertAlmostEqual(rig.position.grad[0, index].item(), finite, delta=2e-7)


if __name__ == "__main__":
    unittest.main()

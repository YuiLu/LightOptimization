import math
import unittest

import torch

from mvp.lighting import AnalyticLightRig, LightSpec, cct_to_rgb, cornell_samples, weighted_loss


class MultiLightColorTests(unittest.TestCase):
    def setUp(self):
        self.samples = cornell_samples(4).to(torch.float64)
        self.specs = [
            LightSpec([-0.45, 1.35, -0.25], 1.5, cct_kelvin=2700.0),
            LightSpec([0.15, 1.65, 0.15], 1.2, cct_kelvin=5000.0),
            LightSpec([0.55, 0.85, 0.35], 1.0, cct_kelvin=8000.0),
        ]

    def test_cct_is_normalized_positive_and_warm_to_cool(self):
        colors = cct_to_rgb(torch.tensor([2700.0, 5000.0, 8000.0], dtype=torch.float64))
        self.assertTrue(torch.isfinite(colors).all())
        self.assertTrue((colors > 0).all())
        self.assertTrue(torch.allclose(colors.sum(dim=-1), torch.ones(3, dtype=torch.float64)))
        self.assertGreater(colors[0, 0].item(), colors[0, 2].item())
        self.assertGreater(colors[2, 2].item(), colors[2, 0].item())

    def test_cct_multi_light_additivity_and_to_specs(self):
        rig = AnalyticLightRig(self.specs, set(), torch.float64, color_mode="cct")
        combined = rig(self.samples)
        expected = sum(
            AnalyticLightRig([spec], set(), torch.float64, color_mode="cct")(self.samples)
            for spec in self.specs
        )
        self.assertEqual(combined.shape, (len(self.samples.areas), 3))
        self.assertTrue(torch.allclose(combined, expected, rtol=1e-11, atol=1e-13))
        result = rig.to_specs()
        for spec, cct in zip(result, (2700.0, 5000.0, 8000.0)):
            self.assertAlmostEqual(spec.cct_kelvin, cct, places=3)
            self.assertAlmostEqual(sum(spec.color), 1.0, places=6)

    def test_normalized_rgb_is_positive_and_unit_chroma(self):
        rig = AnalyticLightRig(self.specs, set(), torch.float64, color_mode="normalized_rgb")
        colors = rig.chromaticity()
        self.assertTrue((colors > 0).all())
        self.assertTrue(torch.allclose(colors.sum(dim=-1), torch.ones(3, dtype=torch.float64)))
        self.assertEqual(rig(self.samples).shape, (len(self.samples.areas), 3))

    def test_fifteen_parameter_autograd_matches_finite_difference(self):
        initial = [
            LightSpec([-0.40, 1.30, -0.20], 1.1, cct_kelvin=2500.0),
            LightSpec([0.10, 1.55, 0.10], 0.9, cct_kelvin=5200.0),
            LightSpec([0.60, 0.90, 0.40], 0.8, cct_kelvin=7600.0),
        ]
        rig = AnalyticLightRig(
            initial, {"position", "intensity", "color"}, torch.float64, color_mode="cct"
        )
        with torch.no_grad():
            target = AnalyticLightRig(self.specs, set(), torch.float64, color_mode="cct")(self.samples)
        parameters = [p for p in rig.parameters() if p.requires_grad]
        self.assertEqual(sum(p.numel() for p in parameters), 15)
        base = torch.cat([p.detach().reshape(-1) for p in parameters])

        def objective(vector):
            offset = 0
            for parameter in parameters:
                count = parameter.numel()
                parameter.data.copy_(vector[offset:offset + count].reshape(parameter.shape))
                offset += count
            return weighted_loss(rig(self.samples), target, self.samples)

        loss = objective(base)
        loss.backward()
        analytic = torch.cat([p.grad.reshape(-1) for p in parameters])
        for index in range(base.numel()):
            plus, minus = base.clone(), base.clone()
            h = 1e-5
            plus[index] += h
            minus[index] -= h
            finite = ((objective(plus) - objective(minus)) / (2 * h)).item()
            self.assertTrue(
                math.isclose(analytic[index].item(), finite, rel_tol=1e-4, abs_tol=1e-8),
                f"parameter {index}: autograd={analytic[index].item()}, FD={finite}",
            )


if __name__ == "__main__":
    unittest.main()

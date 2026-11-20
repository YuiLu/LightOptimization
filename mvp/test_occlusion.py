import math
import unittest

import mitsuba as mi
import torch

from mvp.lighting import AnalyticLightRig, LightSpec, SurfaceSamples, weighted_loss
from mvp.occlusion import MitsubaOccluder


class OcclusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mi.set_variant("scalar_rgb")

    def setUp(self):
        self.samples = SurfaceSamples(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            [1.0, 1.0],
            ["receiver", "receiver"],
            dtype=torch.float64,
        )

    def occluder_at(self, z):
        scene = mi.load_dict({
            "type": "scene",
            "blocker": {
                "type": "rectangle",
                "to_world": mi.Transform4f.translate([0.0, 0.0, z]),
            },
        })
        return MitsubaOccluder(scene)

    def test_blocked_unblocked_and_behind_light(self):
        lights = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64)
        blocked = self.occluder_at(1.0).visibility(lights, self.samples)
        self.assertEqual(blocked.tolist(), [[0.0, 1.0]])
        behind_light = self.occluder_at(3.0).visibility(lights, self.samples)
        self.assertEqual(behind_light.tolist(), [[1.0, 1.0]])
        behind_surface = self.occluder_at(-1.0).visibility(lights, self.samples)
        self.assertEqual(behind_surface.tolist(), [[1.0, 1.0]])

    def test_coplanar_receiver_does_not_self_occlude(self):
        samples = SurfaceSamples(
            [[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]], [1.0], ["receiver"],
            dtype=torch.float64,
        )
        lights = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64)
        visibility = self.occluder_at(0.0).visibility(lights, samples)
        self.assertEqual(visibility.item(), 1.0)

    def test_occluded_rig_is_zero_and_multi_light_remains_additive(self):
        specs = [
            LightSpec(position=[0.0, 0.0, 2.0], intensity=2.0),
            LightSpec(position=[3.0, 0.0, 2.0], intensity=1.5, color=[0.7, 0.8, 1.0]),
        ]
        occluder = self.occluder_at(1.0)
        combined = AnalyticLightRig(specs, set(), torch.float64, occluder)(self.samples)
        individual = sum(
            AnalyticLightRig([spec], set(), torch.float64, occluder)(self.samples)
            for spec in specs
        )
        self.assertEqual(combined.shape, (2, 3))
        self.assertTrue(torch.allclose(combined, individual, rtol=1e-12, atol=1e-14))
        first_only = AnalyticLightRig([specs[0]], set(), torch.float64, occluder)(self.samples)
        self.assertEqual(first_only[0].abs().max().item(), 0.0)
        self.assertGreater(first_only[1].abs().max().item(), 0.0)

    def test_fully_occluded_gradient_is_zero(self):
        rig = AnalyticLightRig(
            [LightSpec(position=[0.0, 0.0, 2.0])],
            {"position", "intensity"}, torch.float64, self.occluder_at(1.0),
        )
        samples = SurfaceSamples(
            [[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]], [1.0], ["receiver"],
            dtype=torch.float64,
        )
        target = torch.ones((1, 3), dtype=torch.float64)
        loss = weighted_loss(rig(samples), target, samples)
        loss.backward()
        self.assertEqual(rig.position.grad.abs().max().item(), 0.0)
        self.assertEqual(rig.log_intensity.grad.abs().max().item(), 0.0)

    def test_smooth_fd_matches_but_boundary_fd_is_not_the_frozen_mask_gradient(self):
        samples = SurfaceSamples(
            [[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]], [1.0], ["receiver"],
            dtype=torch.float64,
        )
        occluder = self.occluder_at(1.0)

        def loss_and_fd(light_x, h):
            rig = AnalyticLightRig(
                [LightSpec(position=[light_x, 0.0, 2.0], intensity=1.5, attenuation=1.0)],
                {"position"}, torch.float64, occluder,
            )
            target = torch.zeros((1, 3), dtype=torch.float64)
            loss = weighted_loss(rig(samples), target, samples)
            loss.backward()
            base = rig.position.detach().clone()
            plus, minus = base.clone(), base.clone()
            plus[0, 0] += h
            minus[0, 0] -= h
            visibility_plus = occluder.visibility(plus, samples).item()
            visibility_minus = occluder.visibility(minus, samples).item()
            rig.position.data.copy_(plus)
            loss_plus = weighted_loss(rig(samples), target, samples).item()
            rig.position.data.copy_(minus)
            loss_minus = weighted_loss(rig(samples), target, samples).item()
            finite = (loss_plus - loss_minus) / (2 * h)
            return rig.position.grad[0, 0].item(), finite, visibility_plus, visibility_minus

        analytic, finite, plus, minus = loss_and_fd(2.1, 1e-5)
        self.assertEqual((plus, minus), (1.0, 1.0))
        self.assertTrue(math.isclose(analytic, finite, rel_tol=1e-4, abs_tol=1e-10))

        analytic, finite, plus, minus = loss_and_fd(1.9, 0.2)
        self.assertNotEqual(plus, minus)
        self.assertEqual(analytic, 0.0)
        self.assertNotEqual(finite, 0.0)
        self.assertFalse(math.isclose(analytic, finite, rel_tol=1e-3, abs_tol=1e-8))


if __name__ == "__main__":
    unittest.main()

import math
import unittest

import mitsuba as mi
import numpy as np
import torch

from mvp.lighting import SurfaceSamples
from mvp.radiosity import (
    RadiosityTransport,
    build_form_factors,
    cornell_albedos,
    direct_to_radiosity,
    orient_cornell_interior,
    structured_patch_samples,
)


def parallel_patches(area=1e-4, distance=1.0):
    return SurfaceSamples(
        [[0.0, 0.0, 0.0], [0.0, 0.0, distance]],
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        [area, area],
        ["a", "b"],
        dtype=torch.float64,
    )


class RadiosityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mi.set_variant("scalar_rgb")

    def test_small_parallel_pair_matches_point_formula_and_reciprocity(self):
        samples = parallel_patches()
        factors, diagnostics = build_form_factors(samples)
        expected = 1e-4 / math.pi
        self.assertAlmostEqual(factors[0, 1].item(), expected, places=12)
        self.assertAlmostEqual(factors[1, 0].item(), expected, places=12)
        self.assertEqual(factors.diag().abs().max().item(), 0.0)
        self.assertEqual(diagnostics["reciprocity_absolute_error"], 0.0)
        self.assertLess(diagnostics["max_row_sum"], 1.0)

    def test_occluded_pair_has_zero_form_factor(self):
        scene = mi.load_dict({
            "type": "scene",
            "blocker": {
                "type": "rectangle",
                "to_world": mi.Transform4f.translate([0.0, 0.0, 0.5]),
            },
        })
        factors, _ = build_form_factors(parallel_patches(), scene)
        self.assertEqual(factors.abs().max().item(), 0.0)

    def test_near_field_refinement_improves_center_estimate(self):
        samples = parallel_patches(area=1.0, distance=0.5)
        center, _ = build_form_factors(samples, refine_near=False)
        refined, _ = build_form_factors(samples, refine_near=True)

        def high_order_reference(grid=12):
            coordinates = (np.arange(grid) + 0.5) / grid - 0.5
            points = np.asarray([(x, y) for x in coordinates for y in coordinates])
            total = 0.0
            count = 0
            for left in points:
                for right in points:
                    delta = np.array([right[0] - left[0], right[1] - left[1], 0.5])
                    distance2 = float(delta @ delta)
                    total += (0.5 * 0.5) / (math.pi * distance2 * distance2)
                    count += 1
            return total / count

        reference = high_order_reference()
        self.assertLess(abs(refined[0, 1].item() - reference), abs(center[0, 1].item() - reference))
        self.assertLessEqual(refined.max().item(), 1.0)

    def test_zero_bounce_neumann_and_full_solve(self):
        samples = parallel_patches()
        factors, _ = build_form_factors(samples)
        albedo = torch.tensor([[0.5, 0.4, 0.3], [0.2, 0.7, 0.6]], dtype=torch.float64)
        transport = RadiosityTransport(factors, albedo)
        direct = torch.tensor([[1.0, 2.0, 3.0], [0.2, 0.4, 0.6]], dtype=torch.float64)
        self.assertTrue(torch.equal(transport.propagate(direct, 0), direct))
        full = transport.propagate(direct)
        iterative = transport.propagate(direct, 50)
        self.assertTrue(torch.allclose(full, iterative, rtol=1e-9, atol=1e-11))
        self.assertLess(transport.residual(direct).abs().max().item(), 1e-11)

    def test_zero_albedo_and_rgb_channels(self):
        samples = parallel_patches()
        factors, _ = build_form_factors(samples)
        direct = torch.tensor([[1.0, 2.0, 3.0], [0.2, 0.4, 0.6]], dtype=torch.float64)
        zero_transport = RadiosityTransport(factors, torch.zeros((2, 3), dtype=torch.float64))
        self.assertTrue(torch.equal(zero_transport.propagate(direct), direct))

        albedo = torch.tensor([[0.5, 0.4, 0.3], [0.2, 0.7, 0.6]], dtype=torch.float64)
        transport = RadiosityTransport(factors, albedo)
        full = transport.propagate(direct)
        for channel in range(3):
            system = torch.eye(2, dtype=torch.float64) - torch.diag(albedo[:, channel]) @ factors
            expected = torch.linalg.solve(system, direct[:, channel])
            self.assertTrue(torch.allclose(full[:, channel], expected, rtol=1e-12, atol=1e-14))

    def test_full_solve_backward_matches_finite_difference(self):
        samples = parallel_patches()
        factors, _ = build_form_factors(samples)
        albedo = torch.tensor([[0.5, 0.4, 0.3], [0.2, 0.7, 0.6]], dtype=torch.float64)
        transport = RadiosityTransport(factors, albedo)
        direct = torch.tensor([[1.0, 2.0, 3.0], [0.2, 0.4, 0.6]], dtype=torch.float64, requires_grad=True)
        loss = transport.propagate(direct).square().sum()
        loss.backward()
        for index in range(direct.numel()):
            plus = direct.detach().clone().reshape(-1)
            minus = direct.detach().clone().reshape(-1)
            h = 1e-5
            plus[index] += h
            minus[index] -= h
            finite = (
                transport.propagate(plus.reshape(2, 3)).square().sum()
                - transport.propagate(minus.reshape(2, 3)).square().sum()
            ).item() / (2 * h)
            analytic = direct.grad.reshape(-1)[index].item()
            self.assertTrue(
                math.isclose(analytic, finite, rel_tol=1e-5, abs_tol=1e-8),
                f"direct[{index}]: autograd={analytic}, FD={finite}",
            )

    def test_structured_patches_cover_geometry_without_area_loss(self):
        scene = mi.load_dict({"type": "scene", "receiver": {"type": "rectangle"}})
        samples = structured_patch_samples(scene, depth=1)
        self.assertEqual(len(samples.areas), 8)
        self.assertAlmostEqual(samples.areas.sum().item(), 4.0, places=12)
        self.assertTrue(torch.allclose(samples.areas, torch.full((8,), 0.5, dtype=torch.float64)))
        self.assertTrue(torch.allclose(
            samples.normals,
            torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64).expand(8, 3),
        ))

    def test_cornell_outward_room_walls_are_oriented_inward(self):
        samples = SurfaceSamples(
            [[-1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]],
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [1.0, 1.0, 1.0, 1.0],
            ["LeftWall", "RightWall", "shape_0", "shape_0"],
            dtype=torch.float64,
        )
        oriented = orient_cornell_interior(samples)
        expected = torch.tensor([
            [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0], [0.0, 1.0, 0.0],
        ], dtype=torch.float64)
        self.assertTrue(torch.allclose(oriented.normals, expected))
        self.assertTrue(torch.equal(oriented.areas, samples.areas))

    def test_direct_conversion_and_albedo_mapping(self):
        samples = parallel_patches()
        samples.names = ["LeftWall", "RightWall"]
        albedo = cornell_albedos(samples)
        direct = torch.ones((2, 3), dtype=torch.float64)
        self.assertTrue(torch.allclose(direct_to_radiosity(direct, albedo), albedo / math.pi))
        samples.names = ["LeftWall", "unknown"]
        with self.assertRaises(ValueError):
            cornell_albedos(samples)


if __name__ == "__main__":
    unittest.main()

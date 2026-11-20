"""Hard visibility queries backed by Mitsuba's scalar BVH.

The binary mask is intentionally detached: autograd sees only the continuous
lighting terms, never a fictitious derivative through a shadow boundary.
"""
from __future__ import annotations

import mitsuba as mi
import numpy as np
import torch

from mvp.lighting import SurfaceSamples


class MitsubaOccluder:
    def __init__(self, scene, epsilon: float = 1e-4):
        self.scene = scene
        self.epsilon = float(epsilon)

    def visibility(self, light_positions: torch.Tensor, samples: SurfaceSamples):
        lights = light_positions.detach().cpu().numpy()
        positions = samples.positions.detach().cpu().numpy()
        normals = samples.normals.detach().cpu().numpy()
        result = np.ones((len(lights), len(positions)), dtype=np.float64)

        for light_index, light in enumerate(lights):
            origins = positions + normals * self.epsilon
            delta = light[None, :] - origins
            distances = np.linalg.norm(delta, axis=1)
            for sample_index in range(len(positions)):
                distance = distances[sample_index]
                if not np.isfinite(distance) or distance <= 2 * self.epsilon:
                    result[light_index, sample_index] = 0.0
                    continue
                direction = delta[sample_index] / distance
                ray = mi.Ray3f(
                    mi.Point3f(*origins[sample_index]),
                    mi.Vector3f(*direction),
                )
                ray.maxt = float(distance - self.epsilon)
                if self.scene.ray_test(ray):
                    result[light_index, sample_index] = 0.0

        return torch.as_tensor(
            result,
            dtype=samples.positions.dtype,
            device=samples.positions.device,
        )

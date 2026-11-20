"""Deterministic diffuse radiosity on world-space surface patches."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import mitsuba as mi
import numpy as np
import torch

from mvp.lighting import SurfaceSamples

CORNELL_ALBEDOS = {
    "LeftWall": [0.63, 0.065, 0.05],
    "RightWall": [0.14, 0.45, 0.091],
}
CORNELL_GRAY_ALBEDO = [0.725, 0.71, 0.68]


def cornell_albedos(samples: SurfaceSamples):
    unknown = sorted({
        name for name in samples.names
        if name not in CORNELL_ALBEDOS and not name.startswith("shape_")
    })
    if unknown:
        raise ValueError(f"Unmapped Cornell shape IDs: {unknown}")
    if not {"LeftWall", "RightWall"}.issubset(samples.names):
        raise ValueError("Cornell samples must include explicitly identified left and right walls")
    # All unnamed/merged shapes in this specific asset use the same gray BSDF.
    values = [CORNELL_ALBEDOS.get(name, CORNELL_GRAY_ALBEDO) for name in samples.names]
    return torch.tensor(values, dtype=samples.positions.dtype, device=samples.positions.device)


def albedo_audit(samples: SurfaceSamples, albedo: torch.Tensor):
    output = {}
    for index, name in enumerate(samples.names):
        if name not in output:
            output[name] = {"count": 0, "area": 0.0, "albedo": albedo[index].tolist()}
        output[name]["count"] += 1
        output[name]["area"] += float(samples.areas[index])
    return output


def direct_to_radiosity(direct_irradiance: torch.Tensor, albedo: torch.Tensor):
    """Convert incident direct irradiance to outgoing diffuse radiosity."""
    return albedo * direct_irradiance / math.pi


def structured_patch_samples(scene, depth: int = 2):
    """Create non-overlapping equal-area patches by recursive triangle splitting."""
    if depth < 0:
        raise ValueError("depth must be nonnegative")

    def split(triangle, remaining):
        if remaining == 0:
            yield np.asarray(triangle)
            return
        a, b, c = triangle
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        for child in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)):
            yield from split(child, remaining - 1)

    positions, normals, areas, names, patch_vertices = [], [], [], [], []
    for shape_index, shape in enumerate(scene.shapes()):
        name = shape.id() or f"shape_{shape_index}"
        vertices = np.asarray(shape.vertex_positions_buffer()).reshape(-1, 3)
        faces = np.asarray(shape.faces_buffer()).reshape(-1, 3)
        for face_index, face in enumerate(faces):
            normal = np.asarray(shape.face_normal(face_index), dtype=np.float64)
            normal /= np.linalg.norm(normal)
            for triangle in split(vertices[face], depth):
                patch_vertices.append(triangle.copy())
                positions.append(triangle.mean(axis=0))
                normals.append(normal)
                areas.append(np.linalg.norm(np.cross(
                    triangle[1] - triangle[0], triangle[2] - triangle[0]
                )) / 2.0)
                names.append(name)
    positions = np.asarray(positions)
    normals = np.asarray(normals)
    areas = np.asarray(areas)
    order = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0]))
    return SurfaceSamples(
        positions[order], normals[order], areas[order],
        [names[index] for index in order], dtype=torch.float64,
        vertices=np.asarray(patch_vertices)[order],
    )


def orient_cornell_interior(samples: SurfaceSamples):
    """Orient the three outward-facing room walls toward the Cornell interior."""
    positions = samples.positions.detach().cpu().numpy()
    normals = samples.normals.detach().cpu().numpy().copy()
    reference = np.array([0.0, 1.0, 0.0])
    for index, name in enumerate(samples.names):
        is_back_wall = abs(positions[index, 2] + 1.0) < 1e-5
        if name in ("LeftWall", "RightWall") or is_back_wall:
            direction = reference - positions[index]
            if direction @ normals[index] < 0.0:
                normals[index] *= -1.0
    return SurfaceSamples(
        positions, normals, samples.areas.detach().cpu().numpy(), samples.names,
        dtype=samples.positions.dtype, vertices=samples.vertices,
    )


def _pair_visible(scene, position_i, normal_i, position_j, normal_j, epsilon):
    if scene is None:
        return True
    center_distance = np.linalg.norm(position_j - position_i)
    if center_distance <= 1e-14:
        return False
    eps = min(epsilon, center_distance * 0.01)
    origin = position_i + normal_i * eps
    target = position_j + normal_j * eps
    delta = target - origin
    distance = np.linalg.norm(delta)
    if distance <= 2 * eps:
        return False
    direction = delta / distance
    ray = mi.Ray3f(mi.Point3f(*origin), mi.Vector3f(*direction))
    ray.maxt = float(distance - eps)
    return not scene.ray_test(ray)


def _triangle_quadrature(triangle, order):
    depth = int(round(math.log2(order)))
    if 2 ** depth != order:
        raise ValueError("Triangle quadrature order must be a power of two")

    def split(current, remaining):
        if remaining == 0:
            yield np.asarray(current)
            return
        a, b, c = current
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        for child in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)):
            yield from split(child, remaining - 1)

    return np.asarray([child.mean(axis=0) for child in split(triangle, depth)])


def _subpatches(position, normal, area, order=2, triangle=None):
    if triangle is not None:
        return _triangle_quadrature(triangle, order)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(reference, normal)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    tangent = np.cross(reference, normal)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    coordinates = ((np.arange(order) + 0.5) / order - 0.5) * math.sqrt(area)
    return np.asarray([
        position + sx * tangent + sy * bitangent
        for sx in coordinates for sy in coordinates
    ])


def _integrated_pair(positions, normals, areas, scene, i, j, epsilon, order, patch_vertices=None):
    triangle_i = None if patch_vertices is None else patch_vertices[i]
    triangle_j = None if patch_vertices is None else patch_vertices[j]
    left = _subpatches(positions[i], normals[i], areas[i], order, triangle_i)
    right = _subpatches(positions[j], normals[j], areas[j], order, triangle_j)
    total = 0.0
    rays = 0
    for point_i in left:
        for point_j in right:
            delta = point_j - point_i
            distance2 = float(delta @ delta)
            if distance2 <= 1e-16:
                continue
            distance = math.sqrt(distance2)
            direction = delta / distance
            cosine_i = max(0.0, float(normals[i] @ direction))
            cosine_j = max(0.0, float(normals[j] @ (-direction)))
            if cosine_i == 0.0 or cosine_j == 0.0:
                continue
            rays += 1
            if _pair_visible(scene, point_i, normals[i], point_j, normals[j], epsilon):
                total += cosine_i * cosine_j / (math.pi * distance2)
    return total / float(order ** 4), rays


def build_form_factors(
    samples: SurfaceSamples,
    scene=None,
    epsilon: float = 1e-4,
    refine_near: bool = True,
):
    """Build F[i,j] from source patch j to receiver i, with reciprocity by construction."""
    positions = samples.positions.detach().cpu().numpy()
    normals = samples.normals.detach().cpu().numpy()
    areas = samples.areas.detach().cpu().numpy()
    patch_vertices = getattr(samples, "vertices", None)
    if patch_vertices is not None:
        patch_vertices = patch_vertices.detach().cpu().numpy()
    count = len(positions)
    factors = np.zeros((count, count), dtype=np.float64)
    started = time.perf_counter()
    ray_count = 0
    refined_pairs = 0
    evaluated_pairs = 0

    for i in range(count):
        js = np.arange(i + 1, count)
        if len(js) == 0:
            continue
        delta = positions[js] - positions[i]
        distance2 = np.einsum("ij,ij->i", delta, delta)
        distances = np.sqrt(np.maximum(distance2, 1e-30))
        directions = delta / distances[:, None]
        cosine_i = np.maximum(0.0, directions @ normals[i])
        cosine_j = np.maximum(0.0, -np.einsum("ij,ij->i", directions, normals[js]))
        geometric = cosine_i * cosine_j / (math.pi * np.maximum(distance2, 1e-30))

        for local in np.nonzero((distance2 > 1e-16) & (geometric > 0.0))[0]:
            j = int(js[local])
            evaluated_pairs += 1
            g = float(geometric[local])
            base_form_factor = max(areas[i], areas[j]) * g
            if refine_near and (
                distances[local] < 2.0 * math.sqrt(max(areas[i], areas[j]))
                or base_form_factor > 0.1
            ):
                order = 4
            elif refine_near and (
                distances[local] < 4.0 * math.sqrt(max(areas[i], areas[j]))
                or base_form_factor > 0.005
            ):
                order = 2
            else:
                order = 0
            if order:
                g, extra_rays = _integrated_pair(
                    positions, normals, areas, scene, i, j, epsilon, order, patch_vertices
                )
                ray_count += extra_rays
                refined_pairs += 1
            else:
                ray_count += 1
                if not _pair_visible(scene, positions[i], normals[i], positions[j], normals[j], epsilon):
                    g = 0.0
            factors[i, j] = areas[j] * g
            factors[j, i] = areas[i] * g

    elapsed = time.perf_counter() - started
    adaptive_passes = []
    max_order_used = 4 if refine_near else 1
    if refine_near:
        for order in (8, 16):
            over_limit = np.nonzero(factors.sum(axis=1) > 1.001)[0]
            if len(over_limit) == 0:
                break
            pass_started = time.perf_counter()
            pair_count = 0
            for i in over_limit:
                for j in np.nonzero(factors[i])[0]:
                    if i == j:
                        continue
                    g, extra_rays = _integrated_pair(
                        positions, normals, areas, scene, i, j, epsilon, order, patch_vertices
                    )
                    ray_count += extra_rays
                    pair_count += 1
                    factors[i, j] = areas[j] * g
                    factors[j, i] = areas[i] * g
            max_order_used = order
            adaptive_passes.append({
                "order": order,
                "rows": len(over_limit),
                "pairs": pair_count,
                "seconds": time.perf_counter() - pass_started,
                "max_row_sum_after": float(factors.sum(axis=1).max()),
            })
    reciprocity = np.abs(areas[:, None] * factors - areas[None, :] * factors.T).max() if count else 0.0
    diagnostics = {
        "seconds": time.perf_counter() - started,
        "base_seconds": elapsed,
        "ray_queries": ray_count,
        "evaluated_pairs": evaluated_pairs,
        "refined_pairs": refined_pairs,
        "near_refinement": refine_near,
        "max_quadrature_order": max_order_used,
        "adaptive_passes": adaptive_passes,
        "max_form_factor": float(factors.max()) if count else 0.0,
        "min_form_factor": float(factors.min()) if count else 0.0,
        "max_row_sum": float(factors.sum(axis=1).max()) if count else 0.0,
        "reciprocity_absolute_error": float(reciprocity),
        "diagonal_absolute_max": float(np.abs(np.diag(factors)).max()) if count else 0.0,
    }
    return torch.as_tensor(factors, dtype=samples.positions.dtype, device=samples.positions.device), diagnostics


class RadiosityTransport:
    def __init__(self, form_factor: torch.Tensor, albedo: torch.Tensor):
        if form_factor.ndim != 2 or form_factor.shape[0] != form_factor.shape[1]:
            raise ValueError("form_factor must be [N,N]")
        if albedo.shape != (form_factor.shape[0], 3):
            raise ValueError("albedo must be [N,3]")
        self.form_factor = form_factor
        self.albedo = albedo
        identity = torch.eye(len(form_factor), dtype=form_factor.dtype, device=form_factor.device)
        self._factors = []
        for channel in range(3):
            system = identity - torch.diag(albedo[:, channel]) @ form_factor
            self._factors.append(torch.linalg.lu_factor(system))

    def reflect(self, radiosity: torch.Tensor):
        return self.albedo * (self.form_factor @ radiosity)

    def propagate(self, direct: torch.Tensor, bounces: int | None = None):
        if bounces is not None and bounces < 0:
            raise ValueError("bounces must be nonnegative or None for the full solve")
        if bounces == 0:
            return direct
        if bounces is not None:
            result = direct
            for _ in range(bounces):
                result = direct + self.reflect(result)
            return result
        channels = []
        for channel, (lu, pivots) in enumerate(self._factors):
            channels.append(torch.linalg.lu_solve(lu, pivots, direct[:, channel:channel + 1]))
        return torch.cat(channels, dim=1)

    def residual(self, direct: torch.Tensor):
        solved = self.propagate(direct)
        return direct + self.reflect(solved) - solved

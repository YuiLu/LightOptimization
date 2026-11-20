"""LightOpt-style analytic irradiance on world-space surface samples."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass
class LightSpec:
    position: list[float]
    intensity: float = 2.0
    color: list[float] | None = None
    kind: str = "point"
    direction: list[float] | None = None
    range: float = 100.0
    attenuation: float = 1.0e-4
    inner_angle: float = 25.0
    outer_angle: float = 45.0

    def __post_init__(self) -> None:
        if self.color is None:
            self.color = [1.0, 1.0, 1.0]
        if self.kind == "spot" and self.direction is None:
            self.direction = [0.0, -1.0, 0.0]

    def to_dict(self) -> dict:
        return asdict(self)


class SurfaceSamples:
    def __init__(self, positions, normals, areas, names, dtype=torch.float32, vertices=None):
        self.positions = torch.as_tensor(positions, dtype=dtype)
        self.normals = torch.as_tensor(normals, dtype=dtype)
        self.areas = torch.as_tensor(areas, dtype=dtype)
        self.names = list(names)
        self.vertices = None if vertices is None else torch.as_tensor(vertices, dtype=dtype)

    def to(self, dtype) -> "SurfaceSamples":
        return SurfaceSamples(self.positions, self.normals, self.areas, self.names, dtype, self.vertices)


def _rectangle_grid(origin, u, v, normal, name, resolution):
    """Cell-centered uniform samples; an MVP substitute for Poisson sampling."""
    origin = torch.tensor(origin, dtype=torch.float64)
    u = torch.tensor(u, dtype=torch.float64)
    v = torch.tensor(v, dtype=torch.float64)
    a = (torch.arange(resolution, dtype=torch.float64) + 0.5) / resolution
    aa, bb = torch.meshgrid(a, a, indexing="ij")
    positions = origin[None, None, :] + aa[..., None] * u + bb[..., None] * v
    count = resolution * resolution
    area = torch.linalg.cross(u, v).norm() / count
    return (
        positions.reshape(count, 3),
        torch.tensor(normal, dtype=torch.float64).expand(count, 3),
        torch.full((count,), area, dtype=torch.float64),
        [name] * count,
    )


def cornell_samples(resolution: int = 20) -> SurfaceSamples:
    """Open-front 2 m Cornell box: x=[-1,1], y=[0,2], z=[-1,1]."""
    rectangles = [
        ([-1, 0, -1], [2, 0, 0], [0, 0, 2], [0, 1, 0], "floor"),
        ([-1, 2, -1], [2, 0, 0], [0, 0, 2], [0, -1, 0], "ceiling"),
        ([-1, 0, 1], [2, 0, 0], [0, 2, 0], [0, 0, -1], "back"),
        ([-1, 0, -1], [0, 0, 2], [0, 2, 0], [1, 0, 0], "left"),
        ([1, 0, -1], [0, 0, 2], [0, 2, 0], [-1, 0, 0], "right"),
    ]
    chunks = [_rectangle_grid(*rect, resolution) for rect in rectangles]
    samples = SurfaceSamples(
        torch.cat([c[0] for c in chunks]),
        torch.cat([c[1] for c in chunks]),
        torch.cat([c[2] for c in chunks]),
        sum([c[3] for c in chunks], []),
        dtype=torch.float64,
    )
    return samples


class AnalyticLightRig(nn.Module):
    """Equations 3-5 of LightOpt, optionally modulated by hard visibility."""

    def __init__(self, specs: Iterable[LightSpec], optimize: set[str], dtype=torch.float32, occluder=None):
        super().__init__()
        self.occluder = occluder
        specs = list(specs)
        if not specs:
            raise ValueError("At least one light is required")
        self.kinds = [s.kind for s in specs]
        self.optimize = set(optimize)
        self.dtype = dtype

        def parameter(values, name):
            return nn.Parameter(torch.tensor(values, dtype=dtype), requires_grad=name in optimize)

        self.position = parameter([s.position for s in specs], "position")
        self.log_intensity = parameter(
            [[torch.log(torch.tensor(s.intensity, dtype=dtype)).item()] for s in specs], "intensity"
        )
        self.color = parameter([s.color for s in specs], "color")
        self.raw_direction = parameter(
            [s.direction if s.direction is not None else [0.0, -1.0, 0.0] for s in specs],
            "direction",
        )
        self.register_buffer("light_range", torch.tensor([s.range for s in specs], dtype=dtype))
        self.register_buffer(
            "attenuation", torch.tensor([s.attenuation for s in specs], dtype=dtype)
        )
        self.register_buffer(
            "cos_inner",
            torch.cos(torch.deg2rad(torch.tensor([s.inner_angle for s in specs], dtype=dtype))),
        )
        self.register_buffer(
            "cos_outer",
            torch.cos(torch.deg2rad(torch.tensor([s.outer_angle for s in specs], dtype=dtype))),
        )

    def normalized_direction(self):
        return self.raw_direction / self.raw_direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def forward(self, samples: SurfaceSamples):
        """Return the summed RGB irradiance with shape ``[samples, 3]``."""
        # [lights, samples, channels]
        delta = self.position[:, None, :] - samples.positions[None, :, :]
        distance2 = (delta * delta).sum(dim=-1).clamp_min(1e-12)
        omega = delta / distance2.sqrt()[..., None]
        cosine = (samples.normals[None, :, :] * omega).sum(dim=-1).clamp_min(0.0)
        range_window = (1.0 - distance2 / self.light_range[:, None].square()).clamp_min(0.0).square()
        distance_attenuation = (
            self.attenuation[:, None]
            / (self.attenuation[:, None] + distance2)
            * range_window
        )
        angular = torch.ones_like(distance2)
        direction = self.normalized_direction()
        spot_indices = [i for i, kind in enumerate(self.kinds) if kind == "spot"]
        if spot_indices:
            idx = torch.tensor(spot_indices, device=delta.device)
            light_to_sample = -omega[idx]
            cone_cosine = (light_to_sample * direction[idx, None, :]).sum(dim=-1)
            cone = (
                (cone_cosine - self.cos_outer[idx, None])
                / (self.cos_inner[idx, None] - self.cos_outer[idx, None]).clamp_min(1e-8)
            ).clamp(0.0, 1.0).square()
            angular[idx] = cone
        strength = (
            self.log_intensity.exp()
            * distance_attenuation
            * angular
            * cosine
        )
        if self.occluder is not None:
            strength = strength * self.occluder.visibility(self.position.detach(), samples)
        return (strength[..., None] * self.color[:, None, :]).sum(dim=0)

    def to_specs(self) -> list[LightSpec]:
        result = []
        with torch.no_grad():
            directions = self.normalized_direction().cpu().numpy()
            for i, kind in enumerate(self.kinds):
                result.append(
                    LightSpec(
                        position=self.position[i].cpu().tolist(),
                        intensity=self.log_intensity[i].exp().item(),
                        color=self.color[i].cpu().tolist(),
                        kind=kind,
                        direction=directions[i].tolist(),
                        range=self.light_range[i].item(),
                        attenuation=self.attenuation[i].item(),
                        inner_angle=torch.rad2deg(torch.acos(self.cos_inner[i])).item(),
                        outer_angle=torch.rad2deg(torch.acos(self.cos_outer[i])).item(),
                    )
                )
        return result


def weighted_loss(prediction, target, samples: SurfaceSamples):
    weights = samples.areas[:, None]
    return (weights * (prediction - target).square()).sum() / (weights.sum() * 3.0)


def field_nrmse(prediction, target, samples: SurfaceSamples):
    weights = samples.areas[:, None]
    numerator = (weights * (prediction - target).square()).sum()
    denominator = (weights * target.square()).sum().clamp_min(1e-20)
    return (numerator / denominator).sqrt()

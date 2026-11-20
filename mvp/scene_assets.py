"""Shared Mitsuba scene geometry, sampling, and point-rig adapters."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import mitsuba as mi
import numpy as np
import torch

from mvp.lighting import LightSpec, SurfaceSamples
from mvp.occlusion import MitsubaOccluder

ROOT = Path(__file__).resolve().parents[1]
SCENE_PATHS = {
    "cornell": ROOT / "third_party" / "cornell-box" / "cornell-box" / "scene_v3.xml",
    "veach-ajar": ROOT / "third_party" / "veach-ajar" / "veach-ajar" / "scene_v3.xml",
}


@dataclass
class ClassicScene:
    name: str
    source_xml: Path
    scene: object
    occluder: MitsubaOccluder
    samples: SurfaceSamples
    original_light_positions: list[list[float]]
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    shape_summaries: list[dict]


def _point_light_xml(spec: LightSpec, index: int):
    emitter = ET.Element("emitter", {"type": "point", "id": f"mvp_light_{index}"})
    ET.SubElement(emitter, "point", {
        "name": "position",
        "x": str(spec.position[0]),
        "y": str(spec.position[1]),
        "z": str(spec.position[2]),
    })
    intensity = np.asarray(spec.color) * spec.intensity
    ET.SubElement(emitter, "rgb", {
        "name": "intensity",
        "value": ", ".join(str(float(value)) for value in intensity),
    })
    return emitter


def write_scene_variant(
    source_xml: Path,
    tag: str,
    lights: list[LightSpec] | None = None,
    spp: int = 64,
    resolution: tuple[int, int] = (512, 512),
    integrator_type: str = "direct",
):
    """Write a generated XML next to the asset so relative resources resolve."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError(f"Unsafe scene variant tag: {tag!r}")
    tree = ET.parse(source_xml)
    root = tree.getroot()
    parent_map = {child: parent for parent in root.iter() for child in parent}

    for shape in list(root.findall(".//shape")):
        if shape.find("emitter") is not None:
            parent_map[shape].remove(shape)
    for emitter in list(root.findall(".//emitter")):
        parent_map[emitter].remove(emitter)

    integrator = root.find("integrator")
    if integrator is None:
        integrator = ET.SubElement(root, "integrator")
    for child in list(integrator):
        integrator.remove(child)
    integrator.set("type", integrator_type)
    if integrator_type == "path":
        ET.SubElement(integrator, "integer", {"name": "max_depth", "value": "8"})

    for sampler in root.findall(".//sensor/sampler"):
        for child in sampler.findall("integer"):
            if child.get("name") == "sample_count":
                child.set("value", str(spp))
    for film in root.findall(".//sensor/film"):
        for child in film.findall("integer"):
            if child.get("name") == "width":
                child.set("value", str(resolution[0]))
            elif child.get("name") == "height":
                child.set("value", str(resolution[1]))

    for index, spec in enumerate(lights or []):
        root.append(_point_light_xml(spec, index))
    ET.indent(tree, space="  ")
    output = source_xml.with_name(f"_mvp_{tag}.xml")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def _allocate_counts(areas: np.ndarray, total: int):
    if total < len(areas):
        raise ValueError("Sample count must be at least the number of shapes")
    counts = np.ones(len(areas), dtype=np.int64)
    remaining = total - len(areas)
    exact = remaining * areas / areas.sum()
    additional = np.floor(exact).astype(np.int64)
    counts += additional
    remainder = int(remaining - additional.sum())
    order = np.argsort(-(exact - additional))
    counts[order[:remainder]] += 1
    return counts


def _sample_shapes(shapes, total: int, seed: int):
    areas = np.asarray([shape.surface_area() for shape in shapes], dtype=np.float64)
    counts = _allocate_counts(areas, total)
    rng = np.random.default_rng(seed)
    positions, normals, weights, names = [], [], [], []
    for index, (shape, count) in enumerate(zip(shapes, counts)):
        name = shape.id() or f"shape_{index}"
        weight = shape.surface_area() / int(count)
        for uv in rng.random((int(count), 2)):
            sample = shape.sample_position(0.0, mi.Point2f(*uv))
            normal = np.asarray(sample.n, dtype=np.float64)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                raise RuntimeError(f"Degenerate sampled normal on {name}")
            positions.append(np.asarray(sample.p, dtype=np.float64))
            normals.append(normal / norm)
            weights.append(weight)
            names.append(name)
    return SurfaceSamples(
        np.asarray(positions), np.asarray(normals), np.asarray(weights), names,
        dtype=torch.float64,
    )


def load_classic_scene(name: str, sample_count: int, sample_seed: int = 1234):
    if name not in SCENE_PATHS:
        raise ValueError(f"Unknown scene {name!r}; choose from {tuple(SCENE_PATHS)}")
    source_xml = SCENE_PATHS[name]
    if not source_xml.exists():
        raise FileNotFoundError(f"Run `python -m mvp.prepare_assets` first: {source_xml}")

    original = mi.load_file(str(source_xml))
    original_light_positions = []
    for shape in original.shapes():
        if shape.is_emitter():
            vertices = np.asarray(shape.vertex_positions_buffer()).reshape(-1, 3)
            original_light_positions.append(vertices.mean(axis=0).tolist())

    geometry_xml = write_scene_variant(source_xml, f"{name}_geometry")
    scene = mi.load_file(str(geometry_xml))
    shapes = list(scene.shapes())
    if not shapes:
        raise RuntimeError(f"No receiving/occluding geometry remains in {name}")
    samples = _sample_shapes(shapes, sample_count, sample_seed)
    summaries = []
    min_bounds, max_bounds = [], []
    for index, shape in enumerate(shapes):
        vertices = np.asarray(shape.vertex_positions_buffer()).reshape(-1, 3)
        min_bounds.append(vertices.min(axis=0))
        max_bounds.append(vertices.max(axis=0))
        summaries.append({
            "name": shape.id() or f"shape_{index}",
            "triangles": shape.face_count(),
            "vertices": shape.vertex_count(),
            "area": shape.surface_area(),
        })
    return ClassicScene(
        name=name,
        source_xml=source_xml,
        scene=scene,
        occluder=MitsubaOccluder(scene),
        samples=samples,
        original_light_positions=original_light_positions,
        bounds_min=np.min(min_bounds, axis=0),
        bounds_max=np.max(max_bounds, axis=0),
        shape_summaries=summaries,
    )


def load_point_rig_scene(
    classic: ClassicScene,
    lights: list[LightSpec],
    tag: str,
    spp: int = 64,
    resolution: tuple[int, int] = (512, 512),
    integrator_type: str = "direct",
):
    path = write_scene_variant(
        classic.source_xml, tag, lights, spp=spp, resolution=resolution,
        integrator_type=integrator_type,
    )
    return mi.load_file(str(path))

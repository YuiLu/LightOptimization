"""Independent scalar Mitsuba validation of recovered point-light rigs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def rectangle(name, center, rotation_axis, rotation_angle, reflectance):
    import mitsuba as mi
    transform = (
        mi.Transform4f.translate(center)
        @ mi.Transform4f.rotate(rotation_axis, rotation_angle)
        @ mi.Transform4f.scale([1.0, 1.0, 1.0])
    )
    return {
        "type": "rectangle", "to_world": transform,
        "material": {"type": "diffuse", "reflectance": {"type": "rgb", "value": reflectance}},
    }


def build_scene(lights, spp=64, resolution=256):
    scene = {
        "type": "scene",
        "integrator": {"type": "direct", "emitter_samples": 2, "bsdf_samples": 0},
        "sensor": {
            "type": "perspective", "fov": 45.0, "fov_axis": "x",
            "to_world": __import__("mitsuba").Transform4f.look_at(
                origin=[0.0, 1.0, -4.2], target=[0.0, 1.0, 0.25], up=[0.0, 1.0, 0.0]
            ),
            "sampler": {"type": "independent", "sample_count": spp},
            "film": {
                "type": "hdrfilm", "width": resolution, "height": resolution,
                "rfilter": {"type": "tent"}, "pixel_format": "rgb",
            },
        },
        "floor": rectangle("floor", [0, 0, 0], [1, 0, 0], -90, [.72, .72, .72]),
        "ceiling": rectangle("ceiling", [0, 2, 0], [1, 0, 0], 90, [.72, .72, .72]),
        "back": rectangle("back", [0, 1, 1], [0, 1, 0], 180, [.72, .72, .72]),
        "left": rectangle("left", [-1, 1, 0], [0, 1, 0], 90, [.72, .10, .10]),
        "right": rectangle("right", [1, 1, 0], [0, 1, 0], -90, [.10, .72, .10]),
    }
    for index, light in enumerate(lights):
        intensity = np.asarray(light["color"]) * light["intensity"]
        scene[f"light_{index}"] = {
            "type": "point", "position": light["position"],
            "intensity": {"type": "rgb", "value": intensity.tolist()},
        }
    return scene


def tone_map(linear):
    linear = np.maximum(linear, 0.0)
    mapped = linear / (1.0 + linear)
    srgb = np.where(mapped <= .0031308, 12.92 * mapped,
                    1.055 * np.power(mapped, 1 / 2.4) - .055)
    return np.clip(srgb, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS / "round1")
    parser.add_argument(
        "--test", choices=("point_position", "point_position_intensity", "two_point"),
        default="point_position_intensity",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spp", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir or (
        args.results_dir / "mitsuba" / f"{args.test}_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    import mitsuba as mi
    mi.set_variant("scalar_rgb")
    data = json.loads((args.results_dir / "raw.json").read_text(encoding="utf-8"))
    run = next(
        (r for r in data["runs"] if r["test"] == args.test and r["seed"] == args.seed),
        None,
    )
    if run is None:
        parser.error(f"No run found for test={args.test}, seed={args.seed}")

    renders = {}
    for name in ("initial", "gt", "result"):
        print(f"Rendering {name} with Mitsuba scalar_rgb...", flush=True)
        scene = mi.load_dict(build_scene(run[name], spp=args.spp, resolution=args.resolution))
        renders[name] = np.array(mi.render(scene))
    optimized_nrmse = float(
        np.linalg.norm(renders["result"] - renders["gt"]) / np.linalg.norm(renders["gt"])
    )
    metrics = {
        "test": args.test,
        "seed": args.seed,
        "backend": f"Mitsuba 3 scalar_rgb; direct integrator; {args.resolution}x{args.resolution}; {args.spp} spp",
        "note": "Independent image check, not the optimization objective. Range window is absent in Mitsuba; analytic range=100 m makes its effect negligible.",
        "initial_linear_rgb_nrmse": float(np.linalg.norm(renders["initial"] - renders["gt"]) / np.linalg.norm(renders["gt"])),
        "optimized_linear_rgb_nrmse": optimized_nrmse,
        "pass_optimized_nrmse_lt_0_02": optimized_nrmse < 0.02,
    }
    (output_dir / "mitsuba_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(output_dir / "mitsuba_linear_renders.npz", **renders)
    for name, linear in renders.items():
        Image.fromarray(np.uint8(np.round(tone_map(linear) * 255))).save(output_dir / f"mitsuba_{name}.png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=True, facecolor="#fcfcfb")
    for axis, name, title in zip(axes, ["initial", "gt", "result"], ["Initial", "Ground truth", "Optimized"]):
        axis.imshow(tone_map(renders[name]))
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"{args.test}, seed {args.seed}")
    fig.savefig(output_dir / "mitsuba_comparison.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()

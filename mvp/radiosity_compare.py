"""Generate GT/result comparisons from an existing radiosity experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mitsuba as mi
import numpy as np
from PIL import Image
import torch

from mvp.lighting import AnalyticLightRig, LightSpec, field_nrmse
from mvp.radiosity import (
    RadiosityTransport,
    direct_to_radiosity,
    orient_cornell_interior,
    structured_patch_samples,
)
from mvp.radiosity_experiments import load_transport
from mvp.render_mitsuba import tone_map
from mvp.scene_assets import load_classic_scene, load_point_rig_scene

ROOT = Path(__file__).resolve().parents[1]


def full_field(classic, samples, transport, spec):
    rig = AnalyticLightRig([spec], set(), samples.positions.dtype, classic.occluder)
    with torch.no_grad():
        return transport.propagate(direct_to_radiosity(rig(samples), transport.albedo))


def save_field_comparison(classic, samples, transport, run, output_dir):
    specs = {
        "initial": LightSpec(**run["initial"]),
        "gt": LightSpec(**run["gt"]),
        "result": LightSpec(**run["result"]),
    }
    fields = {name: full_field(classic, samples, transport, spec) for name, spec in specs.items()}
    computed_nrmse = field_nrmse(fields["result"], fields["gt"], samples).item()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    blue_ramp = LinearSegmentedColormap.from_list(
        "radiosity_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    )
    luminance = lambda value: (value @ torch.tensor([.2126, .7152, .0722], dtype=value.dtype)).numpy()
    views = [
        ("Floor", np.abs(samples.positions[:, 1].numpy()) < 1e-5, (0, 2)),
        ("Back wall", np.abs(samples.positions[:, 2].numpy() + 1.0) < 1e-5, (0, 1)),
    ]
    positions = samples.positions.numpy()
    vmax = max(float(np.percentile(luminance(fields["gt"]), 99)), 1e-12)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True, facecolor="#fcfcfb")
    scatter = None
    for row, (surface, mask, coordinates) in enumerate(views):
        for column, (name, title) in enumerate((("initial", "Initial"), ("gt", "Ground truth"), ("result", "Optimized"))):
            axis = axes[row, column]
            scatter = axis.scatter(
                positions[mask, coordinates[0]], positions[mask, coordinates[1]],
                c=luminance(fields[name])[mask], s=20, cmap=blue_ramp,
                vmin=0, vmax=vmax, edgecolors="#fcfcfb", linewidths=.2,
            )
            axis.set_title(f"{surface}: {title}")
            axis.set_xlabel(f"world {'xyz'[coordinates[0]]} (m)")
            axis.set_ylabel(f"world {'xyz'[coordinates[1]]} (m)")
            axis.set_aspect("equal")
            axis.set_facecolor("#fcfcfb")
    fig.suptitle(
        f"Full radiosity: {run['regime']} seed {run['seed']}, {run['optimizer']}; "
        f"NRMSE={computed_nrmse:.3e}"
    )
    fig.colorbar(scatter, ax=axes, shrink=.75, label="Outgoing radiosity luminance")
    stem = f"field_{run['regime']}_seed{run['seed']}"
    fig.savefig(output_dir / f"{stem}.png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    np.savez_compressed(output_dir / f"{stem}.npz", **{name: value.numpy() for name, value in fields.items()})
    return computed_nrmse


def save_path_comparison(classic, run, output_dir):
    specs = {
        "initial": LightSpec(**run["initial"]),
        "gt": LightSpec(**run["gt"]),
        "result": LightSpec(**run["result"]),
    }
    renders = {}
    for name, spec in specs.items():
        scene = load_point_rig_scene(
            classic, [spec], f"round3_compare_{run['regime']}_{run['seed']}_{name}",
            spp=64, resolution=(512, 512), integrator_type="path",
        )
        renders[name] = np.asarray(mi.render(scene))
    optimized_nrmse = float(np.linalg.norm(renders["result"] - renders["gt"]) / np.linalg.norm(renders["gt"]))
    initial_nrmse = float(np.linalg.norm(renders["initial"] - renders["gt"]) / np.linalg.norm(renders["gt"]))
    np.savez_compressed(output_dir / "path_linear_renders.npz", **renders)
    for name, linear in renders.items():
        Image.fromarray(np.uint8(np.round(tone_map(linear) * 255))).save(output_dir / f"path_{name}.png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True, facecolor="#fcfcfb")
    for axis, name, title in zip(axes, ("initial", "gt", "result"), ("Initial", "Ground truth", "Optimized")):
        axis.imshow(tone_map(renders[name]))
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"Independent path render, near seed {run['seed']} (not the optimization target)")
    fig.savefig(output_dir / "path_comparison.png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {
        "backend": "Mitsuba scalar_rgb path; 512x512; 64 spp; max_depth=8",
        "initial_linear_rgb_nrmse": initial_nrmse,
        "optimized_linear_rgb_nrmse": optimized_nrmse,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "round3")
    args = parser.parse_args()
    mi.set_variant("scalar_rgb")
    raw = json.loads((args.results_dir / "raw.json").read_text(encoding="utf-8"))
    classic = load_classic_scene("cornell", 100)
    classic.samples = orient_cornell_interior(
        structured_patch_samples(classic.scene, raw["subdivision_depth"])
    )
    transport, _ = load_transport(classic, args.results_dir)
    samples = classic.samples.to(torch.float32)
    transport32 = RadiosityTransport(transport.form_factor.to(torch.float32), transport.albedo.to(torch.float32))
    output_dir = args.results_dir / "gt_result"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {"field": {}, "note": "Radiosity fields are compared with the same transport used for optimization; PT is post-process only."}
    for regime in ("near", "cross"):
        run = next(r for r in raw["runs"] if r["regime"] == regime and r["seed"] == 0 and r["optimizer"] == "lbfgs")
        computed = save_field_comparison(classic, samples, transport32, run, output_dir)
        metrics["field"][regime] = {
            "computed_nrmse": computed,
            "saved_run_nrmse": run["final_field_nrmse"],
        }
    near = next(r for r in raw["runs"] if r["regime"] == "near" and r["seed"] == 0 and r["optimizer"] == "lbfgs")
    metrics["path"] = save_path_comparison(classic, near, output_dir)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()

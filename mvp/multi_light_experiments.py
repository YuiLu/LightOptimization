"""Three-light position/intensity/CCT recovery on deterministic radiosity."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
from pathlib import Path

import mitsuba as mi
import numpy as np
import torch

from mvp.lighting import AnalyticLightRig, LightSpec, field_nrmse
from mvp.radiosity import (
    RadiosityTransport,
    direct_to_radiosity,
    orient_cornell_interior,
    structured_patch_samples,
)
from mvp.radiosity_experiments import (
    TRANSPORT_VERSION,
    load_transport,
    optimize_radiosity,
    weighted_loss,
)
from mvp.scene_assets import load_classic_scene

ROOT = Path(__file__).resolve().parents[1]
TESTS = {
    "position_intensity": {"position", "intensity"},
    "joint_cct": {"position", "intensity", "color"},
}


def gt_lights():
    return [
        LightSpec([-0.45, 1.35, -0.25], 1.5, cct_kelvin=2700.0),
        LightSpec([0.15, 1.65, 0.15], 1.2, cct_kelvin=5000.0),
        LightSpec([0.55, 0.85, 0.35], 1.0, cct_kelvin=8000.0),
    ]


def initial_lights(gt, seed, perturb_cct):
    rng = np.random.default_rng(seed)
    initial = copy.deepcopy(gt)
    for spec in initial:
        delta = rng.normal(0.0, 0.05, 3)
        spec.position = (np.asarray(spec.position) + delta).tolist()
        spec.position[0] = float(np.clip(spec.position[0], -0.8, 0.8))
        spec.position[1] = float(np.clip(spec.position[1], 0.35, 1.85))
        spec.position[2] = float(np.clip(spec.position[2], -0.7, 0.7))
        spec.intensity *= float(rng.uniform(0.6, 1.4))
        if perturb_cct:
            spec.cct_kelvin = float(np.clip(spec.cct_kelvin * rng.uniform(0.8, 1.25), 2500.0, 9500.0))
    return initial


def full_field(classic, samples, transport, specs):
    rig = AnalyticLightRig(specs, set(), samples.positions.dtype, classic.occluder, "cct")
    with torch.no_grad():
        return transport.propagate(direct_to_radiosity(rig(samples), transport.albedo))


def gradient_check(classic, samples, transport, gt, initial, h):
    samples64 = samples.to(torch.float64)
    transport64 = RadiosityTransport(
        transport.form_factor.to(torch.float64), transport.albedo.to(torch.float64)
    )
    target = full_field(classic, samples64, transport64, gt)
    rig = AnalyticLightRig(
        initial, {"position", "intensity", "color"}, torch.float64,
        classic.occluder, "cct",
    )
    parameters = [p for p in rig.parameters() if p.requires_grad]
    base = torch.cat([p.detach().reshape(-1) for p in parameters])
    labels = [f"{name}.{i}" for name, p in rig.named_parameters() if p.requires_grad for i in range(p.numel())]

    def evaluate(vector):
        offset = 0
        for parameter in parameters:
            count = parameter.numel()
            parameter.data.copy_(vector[offset:offset + count].reshape(parameter.shape))
            offset += count
        prediction = transport64.propagate(direct_to_radiosity(rig(samples64), transport64.albedo))
        scale = (samples64.areas[:, None] * target.square()).sum() / (samples64.areas.sum() * 3.0)
        return weighted_loss(prediction, target, samples64.areas) / scale.clamp_min(1e-20)

    loss = evaluate(base)
    loss.backward()
    analytic = torch.cat([p.grad.reshape(-1) for p in parameters]).clone()
    finite, flips = [], []
    for index in range(base.numel()):
        plus, minus = base.clone(), base.clone()
        plus[index] += h
        minus[index] -= h
        finite.append(((evaluate(plus) - evaluate(minus)) / (2 * h)).item())
        if index < 9:
            light = index // 3
            flips.append(bool((
                classic.occluder.visibility(plus[:9].reshape(3, 3), samples64)[light]
                != classic.occluder.visibility(minus[:9].reshape(3, 3), samples64)[light]
            ).any()))
        else:
            flips.append(False)
    evaluate(base)
    finite = torch.tensor(finite, dtype=torch.float64)
    absolute = (analytic - finite).abs()
    relative = absolute / torch.maximum(analytic.abs(), finite.abs()).clamp_min(1e-12)
    return {
        "h": h,
        "parameter_names": labels,
        "parameter_count": len(labels),
        "autograd": analytic.tolist(),
        "finite_difference": finite.tolist(),
        "absolute_error": absolute.tolist(),
        "relative_error": relative.tolist(),
        "mask_flip": flips,
        "all_masks_fixed": not any(flips),
        "max_relative_error": relative.max().item(),
        "gradient_cosine": torch.nn.functional.cosine_similarity(analytic, finite, dim=0).item(),
    }


def summarize(runs):
    output = {}
    for test in TESTS:
        for optimizer in ("adam", "lbfgs", "adam_lbfgs"):
            subset = [r for r in runs if r["test"] == test and r["optimizer"] == optimizer]
            if not subset:
                continue
            output[f"{test}_{optimizer}"] = {
                "runs": len(subset),
                "success_rate_field_lt_0_02": float(np.mean([r["success_field_lt_0_02"] for r in subset])),
                "median_final_field_nrmse": float(np.median([r["final_field_nrmse"] for r in subset])),
                "max_final_field_nrmse": float(np.max([r["final_field_nrmse"] for r in subset])),
                "median_position_error_m": float(np.median([r["final_position_error_m"] for r in subset])),
                "max_position_error_m": float(np.max([r["max_position_error_m"] for r in subset])),
                "max_label_position_error_m": float(np.max([r["max_label_position_error_m"] for r in subset])),
                "nontrivial_permutations": int(sum(
                    any(index != light for index, light in enumerate(r["parameter_permutation"]))
                    for r in subset
                )),
                "median_intensity_relative_error": float(np.median([r["final_intensity_relative_error"] for r in subset])),
                "median_cct_relative_error": (
                    float(np.median([r["final_cct_relative_error"] for r in subset]))
                    if subset[0]["final_cct_relative_error"] is not None else None
                ),
                "median_cct_error_kelvin": (
                    float(np.median([r["final_cct_error_kelvin"] for r in subset]))
                    if subset[0]["final_cct_error_kelvin"] is not None else None
                ),
                "median_function_evaluations": float(np.median([r["function_evaluations"] for r in subset])),
                "median_seconds": float(np.median([r["seconds"] for r in subset])),
            }
    return output


def save_convergence(runs, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"adam": "#2a78d6", "lbfgs": "#eb6834", "adam_lbfgs": "#1baf7a"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True, facecolor="#fcfcfb")
    for axis, test in zip(axes, TESTS):
        for optimizer in ("adam", "lbfgs", "adam_lbfgs"):
            matches = [r for r in runs if r["test"] == test and r["optimizer"] == optimizer and r["seed"] == 0]
            if not matches:
                continue
            run = matches[0]
            axis.semilogy(
                [h["evaluation"] for h in run["history"]],
                [h["field_nrmse"] for h in run["history"]],
                color=colors[optimizer], linewidth=1.6, label=optimizer.upper(),
            )
        axis.axhline(0.02, color="#52514e", linestyle="--", linewidth=1, label="2% threshold")
        axis.set_title(test)
        axis.set_xlabel("Function evaluation")
        axis.set_ylabel("Full-radiosity NRMSE")
        axis.grid(color="#e1e0d9", linewidth=.6)
        axis.legend(fontsize=8)
        axis.set_facecolor("#fcfcfb")
    fig.savefig(output_dir / "convergence.png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "round4")
    parser.add_argument("--adam-steps", type=int, default=300)
    parser.add_argument("--lbfgs-budget", type=int, default=300)
    args = parser.parse_args()
    mi.set_variant("scalar_rgb")
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    classic = load_classic_scene("cornell", 100)
    classic.samples = orient_cornell_interior(structured_patch_samples(classic.scene, 2))
    source_transport = ROOT / "results" / "round3" / "transport.npz"
    source_diagnostics = ROOT / "results" / "round3" / "transport_diagnostics.json"
    if source_transport.exists() and not (args.output_dir / "transport.npz").exists():
        shutil.copyfile(source_transport, args.output_dir / "transport.npz")
        shutil.copyfile(source_diagnostics, args.output_dir / "transport_diagnostics.json")
    transport, transport_diagnostics = load_transport(classic, args.output_dir)
    if transport_diagnostics["version"] != TRANSPORT_VERSION:
        raise RuntimeError("Unexpected transport version")
    samples = classic.samples.to(torch.float32)
    transport32 = RadiosityTransport(
        transport.form_factor.to(torch.float32), transport.albedo.to(torch.float32)
    )

    gt = gt_lights()
    target = full_field(classic, samples, transport32, gt)
    runs = []
    for test, optimize in TESTS.items():
        for seed in range(5):
            initial = initial_lights(gt, seed, perturb_cct=(test == "joint_cct"))
            for optimizer in ("adam", "lbfgs"):
                budget = args.adam_steps if optimizer == "adam" else args.lbfgs_budget
                run = optimize_radiosity(
                    classic, samples, transport32, gt, initial, target,
                    optimizer, budget, color_mode="cct", optimize=optimize,
                )
                run.update({"test": test, "regime": "near", "seed": seed})
                runs.append(run)
                print(
                    f"{test:20s} seed={seed} {optimizer:5s} "
                    f"{run['initial_field_nrmse']:.4f}->{run['final_field_nrmse']:.6f}; "
                    f"dp={run['final_position_error_m']:.4g}; "
                    f"dI={run['final_intensity_relative_error']:.3g}; "
                    f"dCCT={run['final_cct_relative_error']}", flush=True,
                )

    gradient_results = {
        "smooth_steps": [
            gradient_check(classic, samples, transport, gt, initial_lights(gt, 0, True), h)
            for h in (1e-4, 1e-5, 1e-6)
        ]
    }
    (args.output_dir / "gradient_checks.json").write_text(json.dumps(gradient_results, indent=2), encoding="utf-8")
    summary = summarize(runs)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    raw = {
        "scene": "cornell",
        "subdivision_depth": 2,
        "transport_version": TRANSPORT_VERSION,
        "gt": [spec.to_dict() for spec in gt],
        "runs": runs,
        "gradient_checks": gradient_results,
    }
    (args.output_dir / "raw.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    fields = [
        "test", "regime", "seed", "optimizer", "function_evaluations", "seconds",
        "initial_field_nrmse", "final_field_nrmse", "final_position_error_m",
        "max_position_error_m", "parameter_permutation", "max_label_position_error_m",
        "final_intensity_relative_error",
        "max_intensity_relative_error", "final_cct_error_kelvin",
        "final_cct_relative_error", "final_label_cct_relative_error", "final_visibility_mismatch", "success_field_lt_0_02",
    ]
    with (args.output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)
    save_convergence(runs, args.output_dir)


if __name__ == "__main__":
    main()

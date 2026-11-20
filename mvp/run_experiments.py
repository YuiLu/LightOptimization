"""Reproducible GT-recovery experiments for the first optimizer MVP."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from mvp.lighting import (
    AnalyticLightRig,
    LightSpec,
    SurfaceSamples,
    cornell_samples,
    field_nrmse,
    weighted_loss,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TESTS = ("point_position", "point_position_intensity", "spot_position_direction", "two_point")
LEARNING_RATES = {"position": 0.035, "intensity": 0.045, "direction": 0.040, "color": 0.03}
SEED_COLORS = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
    "#008300", "#6250d6", "#e34948",
)
SEED_LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)), (0, (4, 1, 1, 1)))
FIELD_SUCCESS_THRESHOLD = 0.02


def gt_specs(test: str) -> list[LightSpec]:
    if test in ("point_position", "point_position_intensity"):
        return [LightSpec(position=[0.25, 1.35, 0.10], intensity=2.5, color=[1.0, 0.85, 0.70])]
    if test == "spot_position_direction":
        return [
            LightSpec(
                position=[-0.45, 1.65, -0.45], intensity=4.0, color=[1.0, 0.90, 0.80],
                kind="spot", direction=[0.20, -1.0, 0.50], range=8.0,
                inner_angle=25.0, outer_angle=50.0,
            )
        ]
    if test == "two_point":
        return [
            LightSpec(position=[-0.55, 1.45, -0.15], intensity=1.8, color=[1.0, 0.72, 0.52]),
            LightSpec(position=[0.60, 0.85, 0.45], intensity=2.2, color=[0.58, 0.75, 1.0]),
        ]
    raise ValueError(test)


def optimized_parameters(test: str) -> set[str]:
    return {
        "point_position": {"position"},
        "point_position_intensity": {"position", "intensity"},
        "spot_position_direction": {"position", "direction"},
        "two_point": {"position", "intensity"},
    }[test]


def initial_specs(test: str, seed: int) -> list[LightSpec]:
    rng = np.random.default_rng(seed)
    specs = copy.deepcopy(gt_specs(test))
    for index, spec in enumerate(specs):
        base_delta = np.array([-0.55, -0.40, -0.35]) if len(specs) == 1 else np.zeros(3)
        delta = base_delta + rng.normal(0.0, 0.16 if len(specs) == 1 else 0.22, size=3)
        position = np.asarray(spec.position) + delta
        position[0] = np.clip(position[0], -0.82, 0.82)
        position[1] = np.clip(position[1], 0.30, 1.75)
        position[2] = np.clip(position[2], -0.75, 0.75)
        spec.position = position.tolist()
        if "intensity" in optimized_parameters(test):
            spec.intensity *= float(rng.uniform(0.45, 0.70))
        if spec.kind == "spot":
            direction = np.asarray(spec.direction) / np.linalg.norm(spec.direction)
            direction += rng.normal(0.0, 0.65, size=3)
            spec.direction = (direction / np.linalg.norm(direction)).tolist()
    return specs


def parameter_errors(rig: AnalyticLightRig, gt: list[LightSpec]):
    positions = rig.position.detach().cpu().numpy()
    intensities = rig.log_intensity.detach().exp().cpu().numpy().reshape(-1)
    gt_positions = np.asarray([s.position for s in gt])
    gt_intensities = np.asarray([s.intensity for s in gt])
    position_error = float(np.linalg.norm(positions - gt_positions, axis=-1).mean())
    intensity_relative_error = np.abs(intensities - gt_intensities) / gt_intensities
    intensity_relative_error = float(np.mean(intensity_relative_error))
    direction_error = None
    if any(s.kind == "spot" for s in gt):
        predicted = rig.normalized_direction().detach().cpu().numpy()
        target = np.asarray([s.direction for s in gt])
        target /= np.linalg.norm(target, axis=-1, keepdims=True)
        angles = np.arccos(np.clip((predicted * target).sum(axis=-1), -1.0, 1.0))
        direction_error = float(np.degrees(angles).mean())
    return position_error, intensity_relative_error, direction_error


def optimize_run(test: str, seed: int, steps: int, samples: SurfaceSamples, initial_override=None, lr_scale: float = 1.0):
    gt = gt_specs(test)
    initial = initial_specs(test, seed) if initial_override is None else copy.deepcopy(initial_override)
    with torch.no_grad():
        target = AnalyticLightRig(gt, set())(samples)
        target_scale = (samples.areas[:, None] * target.square()).sum() / (samples.areas.sum() * 3.0)
    rig = AnalyticLightRig(initial, optimized_parameters(test))
    learning_rates = LEARNING_RATES
    groups = [
        {"params": [p], "lr": learning_rates[{"log_intensity": "intensity", "raw_direction": "direction"}.get(name, name)] * lr_scale}
        for name, p in rig.named_parameters() if p.requires_grad
    ]
    optimizer = torch.optim.Adam(groups, betas=(0.9, 0.999))
    with torch.no_grad():
        initial_prediction = rig(samples)
        initial_nrmse = field_nrmse(initial_prediction, target, samples).item()
    history = []
    started = time.perf_counter()
    for iteration in range(steps + 1):
        prediction = rig(samples)
        loss = weighted_loss(prediction, target, samples) / target_scale.clamp_min(1e-20)
        with torch.no_grad():
            nrmse = field_nrmse(prediction, target, samples).item()
            position_error, intensity_error, direction_error = parameter_errors(rig, gt)
        history.append({
            "iteration": iteration, "loss": loss.item(), "field_nrmse": nrmse,
            "position_error_m": position_error,
            "intensity_relative_error": intensity_error,
            "direction_error_degrees": direction_error,
        })
        if iteration == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in rig.parameters() if p.requires_grad):
            raise RuntimeError(f"Non-finite gradient in {test}, seed {seed}, iteration {iteration}")
        optimizer.step()
    elapsed = time.perf_counter() - started
    final = history[-1]
    return {
        "test": test, "seed": seed, "steps": steps, "seconds": elapsed,
        "gt": [s.to_dict() for s in gt], "initial": [s.to_dict() for s in initial],
        "result": [s.to_dict() for s in rig.to_specs()],
        "initial_field_nrmse": initial_nrmse, "final_field_nrmse": final["field_nrmse"],
        "final_position_error_m": final["position_error_m"],
        "final_intensity_relative_error": final["intensity_relative_error"],
        "final_direction_error_degrees": final["direction_error_degrees"],
        "success_field_lt_0_02": final["field_nrmse"] < FIELD_SUCCESS_THRESHOLD,
        "history": history,
    }


def gradient_check(samples64: SurfaceSamples):
    configurations = {
        "point_joint": (
            [LightSpec(position=[0.2, 1.1, 0.0], intensity=2.5, color=[1, .8, .6])],
            gt_specs("point_position_intensity"),
            {"position", "intensity", "color"},
        ),
        "spot_joint": (
            [LightSpec(position=[-.3, 1.4, -.2], intensity=3, color=[1, .9, .8],
                       kind="spot", direction=[.2, -1, .4], range=8.0)],
            gt_specs("spot_position_direction"),
            {"position", "intensity", "color", "direction"},
        ),
        "two_point_joint": (
            initial_specs("two_point", 0),
            gt_specs("two_point"),
            {"position", "intensity"},
        ),
    }
    output = {}
    for name, (specs, target_specs, optimize) in configurations.items():
        rig = AnalyticLightRig(specs, optimize, torch.float64)
        with torch.no_grad():
            target = AnalyticLightRig(target_specs, set(), torch.float64)(samples64)
        parameters = [p for p in rig.parameters() if p.requires_grad]
        vector = torch.cat([p.detach().reshape(-1) for p in parameters])
        labels = [
            f"{parameter_name}.{i}"
            for parameter_name, p in rig.named_parameters() if p.requires_grad
            for i in range(p.numel())
        ]

        def objective(vec):
            offset = 0
            for parameter in parameters:
                count = parameter.numel()
                parameter.data.copy_(vec[offset:offset + count].reshape(parameter.shape))
                offset += count
            prediction = rig(samples64)
            return weighted_loss(prediction, target, samples64)

        base = vector.clone()
        loss = objective(base)
        loss.backward()
        analytic = torch.cat([
            (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
            for p in parameters
        ]).clone()
        step_results = []
        for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
            finite = []
            for i in range(vector.numel()):
                plus, minus = base.clone(), base.clone()
                plus[i] += h
                minus[i] -= h
                finite.append(((objective(plus) - objective(minus)) / (2 * h)).item())
            finite = torch.tensor(finite, dtype=torch.float64)
            absolute = (analytic - finite).abs()
            relative = absolute / torch.maximum(
                analytic.abs(), finite.abs()
            ).clamp_min(1e-12)
            cosine = torch.nn.functional.cosine_similarity(analytic, finite, dim=0).item()
            step_results.append({
                "h": h,
                "max_absolute_error": absolute.max().item(),
                "max_relative_error": relative.max().item(),
                "median_relative_error": relative.median().item(),
                "gradient_cosine": cosine,
                "finite_difference": finite.tolist(),
                "absolute_error_by_parameter": absolute.tolist(),
                "relative_error_by_parameter": relative.tolist(),
            })
        objective(base)
        best = min(step_results, key=lambda item: item["max_relative_error"])
        output[name] = {
            "parameter_names": labels,
            "autograd": analytic.tolist(),
            "steps": step_results,
            "best_h": best["h"],
            "best_max_absolute_error": best["max_absolute_error"],
            "best_max_relative_error": best["max_relative_error"],
            "pass_max_relative_error_lt_1e-4": best["max_relative_error"] < 1e-4,
        }
    return output


def save_outputs(runs, gradient_results, samples, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tests_present = [test for test in TESTS if any(r["test"] == test for r in runs)]
    seeds = sorted({r["seed"] for r in runs})
    (output_dir / "raw.json").write_text(json.dumps({
        "experiment": "LightOpt-style analytic GT recovery; corrected additive multi-light field",
        "scene": "open-front Cornell box; 5 surfaces; uniform cell-centered samples; visibility=1",
        "sample_count": len(samples.areas),
        "tests": tests_present,
        "seeds": seeds,
        "steps": sorted({r["steps"] for r in runs}),
        "optimizer": {"type": "Adam", "betas": [0.9, 0.999], "learning_rates": LEARNING_RATES},
        "field_success_threshold": FIELD_SUCCESS_THRESHOLD,
        "torch_version": torch.__version__,
        "runs": runs,
        "gradient_checks": gradient_results,
    }, indent=2), encoding="utf-8")
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["test", "seed", "seconds", "initial_field_nrmse", "final_field_nrmse",
                  "final_position_error_m", "final_intensity_relative_error",
                  "final_direction_error_degrees", "success_field_lt_0_02"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)
    summary = {}
    for test in tests_present:
        subset = [r for r in runs if r["test"] == test]
        direction_errors = [r["final_direction_error_degrees"] for r in subset
                            if r["final_direction_error_degrees"] is not None]
        summary[test] = {
            "runs": len(subset),
            "success_rate_field_lt_0_02": float(np.mean([r["success_field_lt_0_02"] for r in subset])),
            "median_initial_field_nrmse": float(np.median([r["initial_field_nrmse"] for r in subset])),
            "median_final_field_nrmse": float(np.median([r["final_field_nrmse"] for r in subset])),
            "max_final_field_nrmse": float(np.max([r["final_field_nrmse"] for r in subset])),
            "median_final_position_error_m": float(np.median([r["final_position_error_m"] for r in subset])),
            "median_final_intensity_relative_error": float(np.median([
                r["final_intensity_relative_error"] for r in subset
            ])),
            "max_final_intensity_relative_error": float(np.max([
                r["final_intensity_relative_error"] for r in subset
            ])),
            "median_final_direction_error_degrees": (
                float(np.median(direction_errors)) if direction_errors else None
            ),
            "max_final_direction_error_degrees": (
                float(np.max(direction_errors)) if direction_errors else None
            ),
            "median_seconds": float(np.median([r["seconds"] for r in subset])),
        }
    summary["gradient_checks"] = {
        name: {
            "best_h": result["best_h"],
            "max_absolute_error": result["best_max_absolute_error"],
            "max_relative_error": result["best_max_relative_error"],
            "pass": result["pass_max_relative_error_lt_1e-4"],
        }
        for name, result in gradient_results.items()
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ncols = 2 if len(tests_present) > 1 else 1
    nrows = math.ceil(len(tests_present) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.2 * ncols, 4.2 * nrows),
        constrained_layout=True, squeeze=False, facecolor="#fcfcfb",
    )
    axes = axes.reshape(-1)
    for axis, test in zip(axes, tests_present):
        subset = [r for r in runs if r["test"] == test]
        axis.set_facecolor("#fcfcfb")
        for run_index, run in enumerate(subset):
            history = run["history"]
            axis.semilogy(
                [h["iteration"] for h in history],
                [h["field_nrmse"] for h in history],
                alpha=.88, linewidth=1.4,
                color=SEED_COLORS[run_index],
                linestyle=SEED_LINESTYLES[run_index],
                label=f"seed {run['seed']}",
            )
        axis.axhline(
            FIELD_SUCCESS_THRESHOLD, color="#52514e", linestyle="--",
            linewidth=1, label="2% threshold",
        )
        axis.set_title(test.replace("_", " "))
        axis.set_xlabel("Adam iteration")
        axis.set_ylabel("Area-weighted field NRMSE")
        axis.grid(alpha=.65, color="#e1e0d9", linestyle="-", linewidth=.6)
        axis.legend(fontsize=8)
    for axis in axes[len(tests_present):]:
        axis.axis("off")
    fig.savefig(output_dir / "convergence.png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)

    representative = next(
        (r for r in runs if r["test"] == "point_position_intensity"),
        runs[0],
    )
    floor_mask = np.asarray([name == "floor" for name in samples.names])
    sample_array = samples.positions.detach().cpu().numpy()[floor_mask]
    target = AnalyticLightRig([LightSpec(**s) for s in representative["gt"]], set())(samples).detach().numpy()[floor_mask]
    initial = AnalyticLightRig([LightSpec(**s) for s in representative["initial"]], set())(samples).detach().numpy()[floor_mask]
    result = AnalyticLightRig([LightSpec(**s) for s in representative["result"]], set())(samples).detach().numpy()[floor_mask]
    luminance = lambda rgb: rgb @ np.array([.2126, .7152, .0722])
    blue_ramp = LinearSegmentedColormap.from_list(
        "irradiance_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True, facecolor="#fcfcfb")
    vmax = np.percentile(luminance(target), 99)
    for axis, values, title in zip(axes, [initial, target, result], ["Initial", "Ground truth", "Optimized"]):
        scatter = axis.scatter(
            sample_array[:, 0], sample_array[:, 2], c=luminance(values),
            s=32, vmin=0, vmax=vmax, cmap=blue_ramp,
            edgecolors="#fcfcfb", linewidths=.25,
        )
        axis.set_title(title)
        axis.set_xlabel("world x (m)")
        axis.set_ylabel("world z (m)")
        axis.set_aspect("equal")
        axis.set_facecolor("#fcfcfb")
    fig.suptitle(f"Floor irradiance: {representative['test']}, seed {representative['seed']}")
    fig.colorbar(scatter, ax=axes, shrink=.8, label="Irradiance luminance")
    fig.savefig(output_dir / "field_overview.png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--resolution", type=int, default=20)
    parser.add_argument("--tests", nargs="+", choices=TESTS, default=TESTS)
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "round1")
    args = parser.parse_args()
    if not args.seeds:
        parser.error("At least one seed is required")
    if len(args.seeds) > len(SEED_COLORS):
        parser.error(f"At most {len(SEED_COLORS)} seeds are supported by the convergence plot")
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    samples64 = cornell_samples(args.resolution)
    samples = samples64.to(torch.float32)
    print(f"Samples: {len(samples.areas)}; steps: {args.steps}; seeds: {args.seeds}", flush=True)
    gradient_results = gradient_check(samples64)
    for name, result in gradient_results.items():
        print(f"Gradient {name}: max relative error={result['best_max_relative_error']:.3e} "
              f"at h={result['best_h']:.0e}", flush=True)
    runs = []
    for test in args.tests:
        for seed in args.seeds:
            run = optimize_run(test, seed, args.steps, samples)
            runs.append(run)
            print(f"{test:28s} seed={seed} NRMSE {run['initial_field_nrmse']:.4f} -> "
                  f"{run['final_field_nrmse']:.6f}; dp={run['final_position_error_m']:.5f} m; "
                  f"{run['seconds']:.2f}s", flush=True)
    save_outputs(runs, gradient_results, samples, args.output_dir)


if __name__ == "__main__":
    main()









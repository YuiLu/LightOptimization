"""Bounded spot-light initialization and optimizer diagnostics."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from mvp.lighting import AnalyticLightRig, LightSpec, cornell_samples, field_nrmse, weighted_loss
from mvp.run_experiments import (
    FIELD_SUCCESS_THRESHOLD,
    gt_specs,
    initial_specs,
    optimize_run,
    optimized_parameters,
    parameter_errors,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TEST = "spot_position_direction"


def compact(run):
    return {key: value for key, value in run.items() if key != "history"}


def run_summary(runs):
    summary = {
        "runs": len(runs),
        "success_rate_field_lt_0_02": float(np.mean([r["success_field_lt_0_02"] for r in runs])),
        "median_final_field_nrmse": float(np.median([r["final_field_nrmse"] for r in runs])),
        "max_final_field_nrmse": float(np.max([r["final_field_nrmse"] for r in runs])),
        "median_final_position_error_m": float(np.median([r["final_position_error_m"] for r in runs])),
        "median_final_direction_error_degrees": float(np.median([
            r["final_direction_error_degrees"] for r in runs
        ])),
        "median_seconds": float(np.median([r["seconds"] for r in runs])),
    }
    evaluations = [r["function_evaluations"] for r in runs if "function_evaluations" in r]
    if evaluations:
        summary["median_function_evaluations"] = float(np.median(evaluations))
        summary["max_function_evaluations"] = max(evaluations)
    return summary


def near_initialization(seed: int):
    initial = initial_specs(TEST, seed)
    gt = gt_specs(TEST)[0]
    rng = np.random.default_rng(seed + 100)
    initial[0].position = (np.asarray(gt.position) + rng.normal(0, .08, 3)).tolist()
    direction = np.asarray(gt.direction) / np.linalg.norm(gt.direction)
    direction += rng.normal(0, .08, 3)
    initial[0].direction = (direction / np.linalg.norm(direction)).tolist()
    return initial


def normalized_objective(rig, target, samples):
    target_scale = (samples.areas[:, None] * target.square()).sum() / (
        samples.areas.sum() * 3.0
    )
    prediction = rig(samples)
    loss = weighted_loss(prediction, target, samples) / target_scale.clamp_min(1e-20)
    return prediction, loss


def finite_difference_at_rig(specs, target, samples, h=1e-5):
    """Check the final smooth-objective gradient without changing the saved rig."""
    samples64 = samples.to(torch.float64)
    target64 = target.detach().to(torch.float64)
    rig = AnalyticLightRig(specs, optimized_parameters(TEST), torch.float64)
    parameters = [parameter for parameter in rig.parameters() if parameter.requires_grad]
    base = torch.cat([parameter.detach().reshape(-1) for parameter in parameters])

    def objective(vector):
        offset = 0
        for parameter in parameters:
            count = parameter.numel()
            parameter.data.copy_(vector[offset:offset + count].reshape(parameter.shape))
            offset += count
        return normalized_objective(rig, target64, samples64)[1]

    loss = objective(base)
    loss.backward()
    analytic = torch.cat([parameter.grad.reshape(-1) for parameter in parameters]).clone()
    finite = []
    for index in range(base.numel()):
        plus, minus = base.clone(), base.clone()
        plus[index] += h
        minus[index] -= h
        finite.append(((objective(plus) - objective(minus)) / (2 * h)).item())
    objective(base)
    finite = torch.tensor(finite, dtype=torch.float64)
    absolute = (analytic - finite).abs()
    relative = absolute / torch.maximum(analytic.abs(), finite.abs()).clamp_min(1e-12)
    return {
        "h": h,
        "finite": bool(torch.isfinite(analytic).all() and torch.isfinite(finite).all()),
        "max_absolute_error": absolute.max().item(),
        "max_relative_error": relative.max().item(),
        "gradient_cosine": torch.nn.functional.cosine_similarity(analytic, finite, dim=0).item(),
    }


def diagnostics(specs, target, samples, check_finite_difference=False):
    rig = AnalyticLightRig(specs, optimized_parameters(TEST), samples.positions.dtype)
    prediction, loss = normalized_objective(rig, target, samples)
    loss.backward()
    gradient_norms = {
        name: (parameter.grad.norm().item() if parameter.grad is not None else 0.0)
        for name, parameter in rig.named_parameters() if parameter.requires_grad
    }
    with torch.no_grad():
        luminance_weights = torch.tensor(
            [.2126, .7152, .0722], dtype=prediction.dtype, device=prediction.device
        )
        prediction_luminance = prediction @ luminance_weights
        target_luminance = target @ luminance_weights
        target_threshold = target_luminance.max().clamp_min(1e-20) * .01
        target_support = target_luminance > target_threshold
        underlit = prediction_luminance < .1 * target_luminance
        prediction_positive = prediction_luminance > target_threshold * 1e-4
        result = {
            "field_nrmse": field_nrmse(prediction, target, samples).item(),
            "gradient_norms": gradient_norms,
            "target_support_fraction": target_support.float().mean().item(),
            "prediction_positive_fraction": prediction_positive.float().mean().item(),
            "underlit_fraction_on_target_support": (
                (underlit & target_support).sum().float() / target_support.sum().clamp_min(1)
            ).item(),
        }
    if check_finite_difference:
        result["finite_difference"] = finite_difference_at_rig(specs, target, samples)
    return result


def optimize_lbfgs(seed: int, max_eval: int, samples):
    gt = gt_specs(TEST)
    initial = initial_specs(TEST, seed)
    with torch.no_grad():
        target = AnalyticLightRig(gt, set())(samples)
    rig = AnalyticLightRig(initial, optimized_parameters(TEST))
    parameters = [parameter for parameter in rig.parameters() if parameter.requires_grad]
    optimizer = torch.optim.LBFGS(
        parameters,
        lr=.5,
        max_iter=max_eval,
        max_eval=max_eval,
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )
    history = []

    def closure():
        optimizer.zero_grad(set_to_none=True)
        prediction, loss = normalized_objective(rig, target, samples)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite L-BFGS loss for seed {seed}")
        loss.backward()
        if not all(torch.isfinite(parameter.grad).all() for parameter in parameters):
            raise RuntimeError(f"Non-finite L-BFGS gradient for seed {seed}")
        with torch.no_grad():
            history.append({
                "evaluation": len(history),
                "loss": loss.item(),
                "field_nrmse": field_nrmse(prediction, target, samples).item(),
            })
        return loss

    with torch.no_grad():
        initial_nrmse = field_nrmse(rig(samples), target, samples).item()
    started = time.perf_counter()
    optimizer.step(closure)
    elapsed = time.perf_counter() - started
    with torch.no_grad():
        final_nrmse = field_nrmse(rig(samples), target, samples).item()
    position_error, intensity_error, direction_error = parameter_errors(rig, gt)
    return {
        "test": TEST,
        "seed": seed,
        "optimizer": "L-BFGS strong-Wolfe",
        "lr": .5,
        "max_function_evaluations": max_eval,
        "function_evaluations": len(history),
        "seconds": elapsed,
        "gt": [spec.to_dict() for spec in gt],
        "initial": [spec.to_dict() for spec in initial],
        "result": [spec.to_dict() for spec in rig.to_specs()],
        "initial_field_nrmse": initial_nrmse,
        "final_field_nrmse": final_nrmse,
        "final_position_error_m": position_error,
        "final_intensity_relative_error": intensity_error,
        "final_direction_error_degrees": direction_error,
        "success_field_lt_0_02": final_nrmse < FIELD_SUCCESS_THRESHOLD,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "round1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--resolution", type=int, default=20)
    parser.add_argument("--near-steps", type=int, default=300)
    parser.add_argument("--far-steps", type=int, default=1500)
    parser.add_argument("--lbfgs-max-eval", type=int, default=1500)
    args = parser.parse_args()

    torch.set_num_threads(4)
    samples = cornell_samples(args.resolution).to(torch.float32)
    gt = gt_specs(TEST)
    with torch.no_grad():
        target = AnalyticLightRig(gt, set())(samples)
    output = {
        "objective": "normalized irradiance MSE (NRMSE^2)",
        "test": TEST,
        "seeds": args.seeds,
        "configuration": {
            "resolution": args.resolution,
            "near_adam_steps": args.near_steps,
            "far_adam_steps": args.far_steps,
            "lbfgs_max_function_evaluations": args.lbfgs_max_eval,
            "coverage": "target support = luminance > 1% of target maximum; underlit = prediction < 10% of target",
        },
        "near_initialization_adam": [],
        "far_initialization_adam": [],
        "far_initialization_lbfgs": [],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        near = near_initialization(seed)
        near_run = optimize_run(TEST, seed, args.near_steps, samples, near)
        output["near_initialization_adam"].append(compact(near_run))
        print(f"near Adam/{args.near_steps} seed={seed}: {near_run['final_field_nrmse']:.7f}", flush=True)

        far_initial = initial_specs(TEST, seed)
        far_run = optimize_run(TEST, seed, args.far_steps, samples, far_initial)
        far_entry = compact(far_run)
        far_entry["initial_diagnostics"] = diagnostics(far_initial, target, samples)
        far_entry["final_diagnostics"] = diagnostics(
            [LightSpec(**spec) for spec in far_run["result"]],
            target,
            samples,
            check_finite_difference=True,
        )
        output["far_initialization_adam"].append(far_entry)
        print(f"far Adam/{args.far_steps} seed={seed}: {far_run['final_field_nrmse']:.7f}", flush=True)

        lbfgs_run = optimize_lbfgs(seed, args.lbfgs_max_eval, samples)
        lbfgs_entry = compact(lbfgs_run)
        lbfgs_entry["initial_diagnostics"] = diagnostics(far_initial, target, samples)
        lbfgs_entry["final_diagnostics"] = diagnostics(
            [LightSpec(**spec) for spec in lbfgs_run["result"]], target, samples
        )
        output["far_initialization_lbfgs"].append(lbfgs_entry)
        print(
            f"far L-BFGS/{args.lbfgs_max_eval} seed={seed}: "
            f"{lbfgs_run['final_field_nrmse']:.7f}; evals={lbfgs_run['function_evaluations']}",
            flush=True,
        )

    output["summary"] = {
        key: run_summary(runs)
        for key, runs in (
            ("near_initialization_adam", output["near_initialization_adam"]),
            ("far_initialization_adam", output["far_initialization_adam"]),
            ("far_initialization_lbfgs", output["far_initialization_lbfgs"]),
        )
    }
    failed_far = [
        run for run in output["far_initialization_adam"]
        if not run["success_field_lt_0_02"]
    ]
    output["summary"]["failed_far_adam_finite_difference"] = {
        "runs": len(failed_far),
        "max_relative_error": max(
            (run["final_diagnostics"]["finite_difference"]["max_relative_error"] for run in failed_far),
            default=None,
        ),
        "max_absolute_error": max(
            (run["final_diagnostics"]["finite_difference"]["max_absolute_error"] for run in failed_far),
            default=None,
        ),
        "min_gradient_cosine": min(
            (run["final_diagnostics"]["finite_difference"]["gradient_cosine"] for run in failed_far),
            default=None,
        ),
    }

    path = args.output_dir / "spot_diagnostics.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved to {path}", flush=True)


if __name__ == "__main__":
    main()

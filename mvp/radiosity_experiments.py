"""Optimizer validation for deterministic diffuse radiosity on Cornell Box."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from itertools import permutations
from pathlib import Path

import mitsuba as mi
import numpy as np
import torch

from mvp.lighting import AnalyticLightRig, LightSpec, field_nrmse
from mvp.occlusion_experiments import (
    FIELD_THRESHOLD,
    cross_specs,
    gt_spec,
    near_candidate,
    save_plots,
    summarize,
    visibility,
    visibility_mismatch,
)
from mvp.radiosity import (
    RadiosityTransport,
    albedo_audit,
    build_form_factors,
    cornell_albedos,
    direct_to_radiosity,
    orient_cornell_interior,
    structured_patch_samples,
)
from mvp.scene_assets import load_classic_scene

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
NEAR_SEEDS = [0, 1, 2, 4, 5]
TRANSPORT_VERSION = 8


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def weighted_loss(prediction, target, weights):
    return (weights[:, None] * (prediction - target).square()).sum() / (weights.sum() * 3.0)


def weighted_nrmse(prediction, target, weights):
    numerator = (weights[:, None] * (prediction - target).square()).sum()
    denominator = (weights[:, None] * target.square()).sum().clamp_min(1e-20)
    return (numerator / denominator).sqrt()


def load_transport(classic, output_dir: Path):
    samples = classic.samples
    albedo = cornell_albedos(samples)
    cache = output_dir / "transport.npz"
    diagnostics_path = output_dir / "transport_diagnostics.json"
    reuse_cache = False
    if cache.exists() and diagnostics_path.exists() and json.loads(
        diagnostics_path.read_text(encoding="utf-8")
    ).get("version") == TRANSPORT_VERSION:
        data = np.load(cache)
        reuse_cache = np.allclose(data["positions"], samples.positions.numpy(), rtol=0, atol=1e-10)
    if reuse_cache:
        form_factor = torch.as_tensor(data["form_factor"], dtype=samples.positions.dtype)
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        diagnostics["cache_reused"] = True
    else:
        form_factor, build_diagnostics = build_form_factors(samples, classic.scene)
        diagnostics = {
            "version": TRANSPORT_VERSION,
            "build": build_diagnostics,
            "albedo_by_shape": albedo_audit(samples, albedo),
            "cache_reused": False,
        }
        np.savez_compressed(
            cache,
            form_factor=form_factor.numpy(),
            positions=samples.positions.numpy(),
            normals=samples.normals.numpy(),
            areas=samples.areas.numpy(),
            albedo=albedo.numpy(),
        )
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2, default=json_default), encoding="utf-8")
    transport = RadiosityTransport(form_factor, albedo)
    return transport, diagnostics


def direct_field(classic, samples, albedo, spec):
    rig = AnalyticLightRig([spec], set(), samples.positions.dtype, classic.occluder)
    with torch.no_grad():
        return direct_to_radiosity(rig(samples), albedo)


def bounce_diagnostics(classic, samples, transport, gt):
    direct = direct_field(classic, samples, transport.albedo, gt)
    full = transport.propagate(direct)
    output = []
    for bounce in (0, 1, 2, 4, 8, 16):
        current = transport.propagate(direct, bounce)
        indirect = current - direct
        output.append({
            "bounces": bounce,
            "nrmse_to_full": field_nrmse(current, full, samples).item(),
            "indirect_to_total_area_weighted_l1": (
                (samples.areas[:, None] * indirect.abs()).sum()
                / (samples.areas[:, None] * current.abs()).sum().clamp_min(1e-20)
            ).item(),
            "linear_residual_max": (
                direct + transport.reflect(current) - current
            ).abs().max().item(),
        })
    output.append({
        "bounces": "full",
        "nrmse_to_full": 0.0,
        "indirect_to_total_area_weighted_l1": (
            (samples.areas[:, None] * (full - direct).abs()).sum()
            / (samples.areas[:, None] * full.abs()).sum().clamp_min(1e-20)
        ).item(),
        "linear_residual_max": transport.residual(direct).abs().max().item(),
    })
    return output


def optimize_radiosity(
    classic,
    samples,
    transport,
    gt,
    initial,
    target,
    optimizer_name,
    budget,
    weights=None,
    color_mode="linear",
    optimize=None,
):
    if weights is None:
        weights = samples.areas
    albedo = transport.albedo
    is_multi = isinstance(initial, (list, tuple))
    initial_specs = [copy.deepcopy(spec) for spec in initial] if is_multi else [copy.deepcopy(initial)]
    gt_specs = list(gt) if isinstance(gt, (list, tuple)) else [gt]
    optimize_set = {"position", "intensity"} if optimize is None else set(optimize)
    rig = AnalyticLightRig(
        initial_specs, optimize_set, samples.positions.dtype, classic.occluder, color_mode
    )
    history = []

    def evaluate():
        direct = direct_to_radiosity(rig(samples), albedo)
        prediction = transport.propagate(direct)
        target_scale = (weights[:, None] * target.square()).sum() / (weights.sum() * 3.0)
        loss = weighted_loss(prediction, target, weights) / target_scale.clamp_min(1e-20)
        return prediction, loss

    def record(iteration, prediction, loss):
        history.append({
            "evaluation": len(history),
            "iteration": iteration,
            "loss": loss.item(),
            "field_nrmse": weighted_nrmse(prediction, target, weights).item(),
        })

    started = time.perf_counter()
    if optimizer_name == "adam":
        groups = [
            {"params": [parameter], "lr": {
                "position": 0.035, "log_intensity": 0.045,
                "log_cct": 0.02, "color_logits": 0.03, "color": 0.03,
            }.get(name, 0.03)}
            for name, parameter in rig.named_parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.Adam(groups, betas=(0.9, 0.999))
        for iteration in range(budget + 1):
            prediction, loss = evaluate()
            record(iteration, prediction, loss)
            if iteration == budget:
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in rig.parameters() if p.requires_grad):
                raise RuntimeError("Non-finite Adam gradient")
            optimizer.step()
    elif optimizer_name == "lbfgs":
        parameters = [p for p in rig.parameters() if p.requires_grad]
        optimizer = torch.optim.LBFGS(
            parameters, lr=0.5, max_iter=budget, max_eval=budget,
            tolerance_grad=1e-12, tolerance_change=1e-14,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            prediction, loss = evaluate()
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in parameters):
                raise RuntimeError("Non-finite L-BFGS gradient")
            record(len(history), prediction, loss)
            return loss

        optimizer.step(closure)
    else:
        raise ValueError(optimizer_name)
    elapsed = time.perf_counter() - started

    with torch.no_grad():
        initial_prediction = transport.propagate(
            direct_to_radiosity(
                AnalyticLightRig(initial_specs, set(), samples.positions.dtype, classic.occluder, color_mode)(samples),
                albedo,
            )
        )
        initial_nrmse = weighted_nrmse(initial_prediction, target, weights).item()
        final_prediction = transport.propagate(direct_to_radiosity(rig(samples), albedo))
        final_nrmse = weighted_nrmse(final_prediction, target, weights).item()
    rig.zero_grad(set_to_none=True)
    _, final_loss = evaluate()
    final_loss.backward()
    final_gradient_norm = math.sqrt(sum(
        float(parameter.grad.square().sum()) for parameter in rig.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ))
    results = rig.to_specs()
    label_position_errors = [
        float(np.linalg.norm(np.asarray(result.position) - np.asarray(gt_spec.position)))
        for result, gt_spec in zip(results, gt_specs)
    ]
    label_cct_relative_errors = [
        abs(result.cct_kelvin - gt_spec.cct_kelvin) / gt_spec.cct_kelvin
        for result, gt_spec in zip(results, gt_specs)
        if result.cct_kelvin is not None and gt_spec.cct_kelvin is not None
    ]
    best_assignment = (0,)
    if len(results) > 1:
        best_score = math.inf
        for assignment in permutations(range(len(results))):
            score = sum(
                float(np.linalg.norm(np.asarray(results[assignment[i]].position) - np.asarray(gt_specs[i].position)))
                for i in range(len(results))
            )
            if score < best_score:
                best_score = score
                best_assignment = assignment

    position_errors = [
        float(np.linalg.norm(np.asarray(results[best_assignment[i]].position) - np.asarray(gt_specs[i].position)))
        for i in range(len(results))
    ]
    intensity_errors = [
        abs(results[best_assignment[i]].intensity - gt_specs[i].intensity) / gt_specs[i].intensity
        for i in range(len(results))
    ]
    cct_errors = [
        abs(results[best_assignment[i]].cct_kelvin - gt_specs[i].cct_kelvin)
        for i in range(len(results))
        if results[best_assignment[i]].cct_kelvin is not None and gt_specs[i].cct_kelvin is not None
    ]
    cct_relative_errors = [
        abs(results[best_assignment[i]].cct_kelvin - gt_specs[i].cct_kelvin) / gt_specs[i].cct_kelvin
        for i in range(len(results))
        if results[best_assignment[i]].cct_kelvin is not None and gt_specs[i].cct_kelvin is not None
    ]

    def mismatch(specs, assignment):
        return float(np.mean([
            visibility_mismatch(classic, specs[assignment[i]], gt_specs[i], samples)
            for i in range(len(specs))
        ]))

    gt_render_specs = AnalyticLightRig(
        gt_specs, set(), samples.positions.dtype, classic.occluder, color_mode
    ).to_specs()
    initial_render_specs = AnalyticLightRig(
        initial_specs, set(), samples.positions.dtype, classic.occluder, color_mode
    ).to_specs()

    return {
        "scene": classic.name,
        "optimizer": optimizer_name,
        "budget": budget,
        "function_evaluations": len(history),
        "seconds": elapsed,
        "gt": [spec.to_dict() for spec in gt_render_specs] if is_multi else gt_render_specs[0].to_dict(),
        "initial": [spec.to_dict() for spec in initial_render_specs] if is_multi else initial_render_specs[0].to_dict(),
        "result": [spec.to_dict() for spec in results] if is_multi else results[0].to_dict(),
        "initial_field_nrmse": initial_nrmse,
        "final_field_nrmse": final_nrmse,
        "final_position_error_m": float(np.mean(position_errors)),
        "max_position_error_m": float(np.max(position_errors)),
        "parameter_permutation": list(best_assignment),
        "final_label_position_error_m": float(np.mean(label_position_errors)),
        "max_label_position_error_m": float(np.max(label_position_errors)),
        "final_intensity_relative_error": float(np.mean(intensity_errors)),
        "max_intensity_relative_error": float(np.max(intensity_errors)),
        "final_cct_error_kelvin": float(np.mean(cct_errors)) if cct_errors else None,
        "max_cct_error_kelvin": float(np.max(cct_errors)) if cct_errors else None,
        "final_cct_relative_error": float(np.mean(cct_relative_errors)) if cct_relative_errors else None,
        "final_label_cct_relative_error": float(np.mean(label_cct_relative_errors)) if label_cct_relative_errors else None,
        "initial_visibility_mismatch": mismatch(initial_specs, tuple(range(len(initial_specs)))),
        "final_visibility_mismatch": mismatch(results, best_assignment),
        "final_gradient_norm": final_gradient_norm,
        "success_field_lt_0_02": final_nrmse < FIELD_THRESHOLD,
        "history": history,
    }


def smooth_gradient_check(classic, samples, transport, gt, initial, h):
    samples64 = samples.to(torch.float64)
    transport64 = RadiosityTransport(
        transport.form_factor.to(torch.float64), transport.albedo.to(torch.float64)
    )
    albedo = transport64.albedo
    target = transport64.propagate(direct_field(classic, samples64, albedo, gt))
    rig = AnalyticLightRig([initial], {"position", "intensity"}, torch.float64, classic.occluder)
    parameters = [p for p in rig.parameters() if p.requires_grad]
    base = torch.cat([p.detach().reshape(-1) for p in parameters])
    labels = [f"{name}.{i}" for name, p in rig.named_parameters() if p.requires_grad for i in range(p.numel())]

    def evaluate(vector):
        offset = 0
        for parameter in parameters:
            count = parameter.numel()
            parameter.data.copy_(vector[offset:offset + count].reshape(parameter.shape))
            offset += count
        prediction = transport64.propagate(direct_to_radiosity(rig(samples64), albedo))
        target_scale = (samples64.areas[:, None] * target.square()).sum() / (samples64.areas.sum() * 3.0)
        return weighted_loss(prediction, target, samples64.areas) / target_scale.clamp_min(1e-20)

    loss = evaluate(base)
    loss.backward()
    analytic = torch.cat([p.grad.reshape(-1) for p in parameters]).clone()
    finite, flips = [], []
    for index in range(base.numel()):
        plus, minus = base.clone(), base.clone()
        plus[index] += h
        minus[index] -= h
        finite.append(((evaluate(plus) - evaluate(minus)) / (2 * h)).item())
        if index < 3:
            flips.append(bool((
                classic.occluder.visibility(plus[:3].reshape(1, 3), samples64)
                != classic.occluder.visibility(minus[:3].reshape(1, 3), samples64)
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
        "autograd": analytic.tolist(),
        "finite_difference": finite.tolist(),
        "relative_error": relative.tolist(),
        "mask_flip": flips,
        "all_masks_fixed": not any(flips),
        "max_relative_error": relative.max().item(),
        "gradient_cosine": torch.nn.functional.cosine_similarity(analytic, finite, dim=0).item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "round3")
    parser.add_argument("--subdivision-depth", type=int, default=2)
    parser.add_argument("--adam-steps", type=int, default=300)
    parser.add_argument("--lbfgs-budget", type=int, default=300)
    parser.add_argument("--transport-only", action="store_true")
    args = parser.parse_args()
    mi.set_variant("scalar_rgb")
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    classic = load_classic_scene("cornell", 100)
    classic.samples = orient_cornell_interior(
        structured_patch_samples(classic.scene, args.subdivision_depth)
    )
    transport, transport_diagnostics = load_transport(classic, args.output_dir)
    samples = classic.samples.to(torch.float32)
    transport32 = RadiosityTransport(
        transport.form_factor.to(torch.float32), transport.albedo.to(torch.float32)
    )
    gt = gt_spec(classic)
    bounce = bounce_diagnostics(classic, samples, transport32, gt)
    transport_diagnostics["bounce_convergence"] = bounce
    (args.output_dir / "transport_diagnostics.json").write_text(
        json.dumps(transport_diagnostics, indent=2, default=json_default), encoding="utf-8"
    )
    print(json.dumps(transport_diagnostics, indent=2, default=json_default), flush=True)
    if args.transport_only:
        return

    target = transport32.propagate(direct_field(classic, samples, transport32.albedo, gt))
    runs = []
    specifications = (
        [("near", seed, near_candidate(classic, gt, seed)) for seed in NEAR_SEEDS]
        + [("cross", seed, spec) for seed, spec in enumerate(cross_specs(classic, gt))]
    )
    for regime, seed, initial in specifications:
        for optimizer in ("adam", "lbfgs"):
            budget = args.adam_steps if optimizer == "adam" else args.lbfgs_budget
            run = optimize_radiosity(
                classic, samples, transport32, gt, initial, target, optimizer, budget
            )
            run["regime"] = regime
            run["seed"] = seed
            runs.append(run)
            print(
                f"{regime:5s} seed={seed} {optimizer:5s} "
                f"{run['initial_field_nrmse']:.4f}->{run['final_field_nrmse']:.6f}; "
                f"evals={run['function_evaluations']}; {run['seconds']:.2f}s", flush=True,
            )

    near_initial = near_candidate(classic, gt, 0)
    gradient_results = {
        "smooth_steps": [
            smooth_gradient_check(classic, samples, transport, gt, near_initial, h)
            for h in (1e-4, 1e-5, 1e-6)
        ]
    }
    (args.output_dir / "gradient_checks.json").write_text(
        json.dumps(gradient_results, indent=2), encoding="utf-8"
    )

    gi_weights = samples.areas * (visibility(classic, gt, samples) == 0).float().reshape(-1)
    gi_runs = []
    for optimizer in ("adam", "lbfgs"):
        budget = args.adam_steps if optimizer == "adam" else args.lbfgs_budget
        gi_run = optimize_radiosity(
            classic, samples, transport32, gt, near_initial, target,
            optimizer, budget, weights=gi_weights,
        )
        gi_run["regime"] = "gi_only"
        gi_run["seed"] = 0
        gi_run["weighted_patch_fraction"] = float((gi_weights > 0).float().mean())
        gi_runs.append(gi_run)
        print(f"GI-only {optimizer}: {gi_run['initial_field_nrmse']:.4f}->{gi_run['final_field_nrmse']:.6f}", flush=True)

    summary = summarize(runs)
    for gi_run in gi_runs:
        summary[f"gi_only_{gi_run['optimizer']}"] = {
            "final_field_nrmse": gi_run["final_field_nrmse"],
            "final_position_error_m": gi_run["final_position_error_m"],
            "final_intensity_relative_error": gi_run["final_intensity_relative_error"],
            "function_evaluations": gi_run["function_evaluations"],
            "weighted_patch_fraction": gi_run["weighted_patch_fraction"],
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    all_runs = runs + gi_runs
    (args.output_dir / "raw.json").write_text(json.dumps({
        "scene": "cornell",
        "patch_count": len(samples.areas),
        "subdivision_depth": args.subdivision_depth,
        "near_seeds": NEAR_SEEDS,
        "gt": gt.to_dict(),
        "runs": all_runs,
        "gradient_checks": gradient_results,
    }, indent=2, default=json_default), encoding="utf-8")
    fields = [
        "regime", "seed", "optimizer", "function_evaluations", "seconds",
        "initial_field_nrmse", "final_field_nrmse", "final_position_error_m",
        "final_intensity_relative_error", "initial_visibility_mismatch",
        "final_visibility_mismatch", "final_gradient_norm", "success_field_lt_0_02",
    ]
    with (args.output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_runs)
    save_plots(runs, args.output_dir)


if __name__ == "__main__":
    main()

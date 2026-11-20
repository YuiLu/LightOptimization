"""GT recovery and shadow-boundary diagnostics on official classic scenes."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path

import mitsuba as mi
import numpy as np
from PIL import Image
import torch

from mvp.lighting import AnalyticLightRig, LightSpec, field_nrmse, weighted_loss
from mvp.render_mitsuba import tone_map
from mvp.scene_assets import ClassicScene, load_classic_scene, load_point_rig_scene

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIELD_THRESHOLD = 0.02
OPTIMIZERS = ("adam", "lbfgs")


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)}")


def gt_spec(classic: ClassicScene):
    position = list(classic.original_light_positions[0])
    if classic.name == "cornell":
        position[1] -= 0.18
        return LightSpec(position=position, intensity=2.5, color=[1.0, 0.85, 0.70])
    return LightSpec(position=position, intensity=500.0, color=[1.0, 1.0, 1.0])


def near_candidate(classic: ClassicScene, gt: LightSpec, seed: int):
    rng = np.random.default_rng(seed)
    spec = copy.deepcopy(gt)
    if classic.name == "cornell":
        delta = rng.normal(0.0, 0.045, 3)
        spec.position = (np.asarray(gt.position) + delta).tolist()
        spec.position[0] = float(np.clip(spec.position[0], -0.55, 0.55))
        spec.position[1] = float(np.clip(spec.position[1], 1.58, 1.90))
        spec.position[2] = float(np.clip(spec.position[2], -0.45, 0.45))
    else:
        delta = rng.normal(0.0, [0.06, 0.06, 0.015])
        spec.position = (np.asarray(gt.position) + delta).tolist()
        spec.position[0] = float(np.clip(spec.position[0], -5.0, -3.85))
        spec.position[1] = float(np.clip(spec.position[1], 0.45, 2.55))
        spec.position[2] = float(np.clip(spec.position[2], -4.49, -4.40))
    spec.intensity *= float(rng.uniform(0.55, 0.75))
    return spec


def cross_specs(classic: ClassicScene, gt: LightSpec):
    if classic.name == "cornell":
        positions = [[-0.65, 0.65, -0.65], [0.0, 0.40, 0.80]]
    else:
        positions = [[-2.0, 1.50, -4.44377], [0.0, 1.50, -4.44377]]
    specs = []
    for position in positions:
        spec = copy.deepcopy(gt)
        spec.position = position
        spec.intensity *= 0.65
        specs.append(spec)
    return specs


def visibility(classic: ClassicScene, spec: LightSpec, samples):
    return classic.occluder.visibility(
        torch.tensor([spec.position], dtype=samples.positions.dtype), samples
    )


def visibility_mismatch(classic: ClassicScene, spec: LightSpec, gt: LightSpec, samples):
    return float((visibility(classic, spec, samples) != visibility(classic, gt, samples)).float().mean())


def make_near_specs(classic: ClassicScene, gt: LightSpec, count: int, samples):
    threshold = 0.02 if classic.name == "cornell" else 0.005
    selected = []
    for seed in range(1000):
        candidate = near_candidate(classic, gt, seed)
        mismatch = visibility_mismatch(classic, candidate, gt, samples)
        if mismatch <= threshold:
            selected.append((seed, candidate, mismatch))
            if len(selected) == count:
                break
    if len(selected) != count:
        raise RuntimeError(f"Could not construct {count} near initializations for {classic.name}")
    return selected


def target_field(classic: ClassicScene, gt: LightSpec, samples):
    with torch.no_grad():
        return AnalyticLightRig([gt], set(), samples.positions.dtype, classic.occluder)(samples)


def objective(rig, target, samples):
    prediction = rig(samples)
    target_scale = (samples.areas[:, None] * target.square()).sum() / (
        samples.areas.sum() * 3.0
    )
    loss = weighted_loss(prediction, target, samples) / target_scale.clamp_min(1e-20)
    return prediction, loss


def optimize_scene(
    classic: ClassicScene,
    samples,
    gt: LightSpec,
    initial: LightSpec,
    target,
    optimizer_name: str,
    budget: int,
):
    rig = AnalyticLightRig([copy.deepcopy(initial)], {"position", "intensity"}, samples.positions.dtype, classic.occluder)
    history = []

    def record(iteration, prediction, loss):
        history.append({
            "evaluation": len(history),
            "iteration": iteration,
            "loss": loss.item(),
            "field_nrmse": field_nrmse(prediction, target, samples).item(),
        })

    started = time.perf_counter()
    if optimizer_name == "adam":
        learning_rates = {"position": 0.035, "log_intensity": 0.045}
        groups = [
            {"params": [parameter], "lr": learning_rates[name]}
            for name, parameter in rig.named_parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.Adam(groups, betas=(0.9, 0.999))
        for iteration in range(budget + 1):
            prediction, loss = objective(rig, target, samples)
            record(iteration, prediction, loss)
            if iteration == budget:
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in rig.parameters() if p.requires_grad):
                raise RuntimeError(f"Non-finite Adam gradient in {classic.name}")
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
            prediction, loss = objective(rig, target, samples)
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in parameters):
                raise RuntimeError(f"Non-finite L-BFGS gradient in {classic.name}")
            record(len(history), prediction, loss)
            return loss

        optimizer.step(closure)
    else:
        raise ValueError(optimizer_name)
    elapsed = time.perf_counter() - started

    with torch.no_grad():
        initial_prediction = AnalyticLightRig([initial], set(), samples.positions.dtype, classic.occluder)(samples)
        initial_nrmse = field_nrmse(initial_prediction, target, samples).item()
        final_prediction = rig(samples)
        final_nrmse = field_nrmse(final_prediction, target, samples).item()
    rig.zero_grad(set_to_none=True)
    _, final_loss = objective(rig, target, samples)
    final_loss.backward()
    final_gradient_norm = math.sqrt(sum(
        float(parameter.grad.square().sum()) for parameter in rig.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ))
    result = rig.to_specs()[0]
    position_error = float(np.linalg.norm(np.asarray(result.position) - np.asarray(gt.position)))
    intensity_error = abs(result.intensity - gt.intensity) / gt.intensity
    return {
        "scene": classic.name,
        "optimizer": optimizer_name,
        "budget": budget,
        "function_evaluations": len(history),
        "seconds": elapsed,
        "gt": gt.to_dict(),
        "initial": initial.to_dict(),
        "result": result.to_dict(),
        "initial_field_nrmse": initial_nrmse,
        "final_field_nrmse": final_nrmse,
        "final_position_error_m": position_error,
        "final_intensity_relative_error": intensity_error,
        "initial_visibility_mismatch": visibility_mismatch(classic, initial, gt, samples),
        "final_visibility_mismatch": visibility_mismatch(classic, result, gt, samples),
        "final_gradient_norm": final_gradient_norm,
        "success_field_lt_0_02": final_nrmse < FIELD_THRESHOLD,
        "history": history,
    }


def smooth_gradient_check(classic: ClassicScene, samples, gt: LightSpec, initial: LightSpec, h: float):
    rig = AnalyticLightRig([initial], {"position", "intensity"}, torch.float64, classic.occluder)
    samples64 = samples.to(torch.float64)
    target = target_field(classic, gt, samples64)
    parameters = [p for p in rig.parameters() if p.requires_grad]
    base = torch.cat([p.detach().reshape(-1) for p in parameters])
    labels = [f"{name}.{index}" for name, p in rig.named_parameters() if p.requires_grad for index in range(p.numel())]

    def evaluate(vector):
        offset = 0
        for parameter in parameters:
            count = parameter.numel()
            parameter.data.copy_(vector[offset:offset + count].reshape(parameter.shape))
            offset += count
        return objective(rig, target, samples64)[1]

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
        "absolute_error": absolute.tolist(),
        "relative_error": relative.tolist(),
        "mask_flip": flips,
        "all_masks_fixed": not any(flips),
        "max_relative_error": relative.max().item(),
        "gradient_cosine": torch.nn.functional.cosine_similarity(analytic, finite, dim=0).item(),
    }


def boundary_check(classic: ClassicScene, samples, gt: LightSpec, initial: LightSpec):
    samples64 = samples.to(torch.float64)
    target = target_field(classic, gt, samples64)
    candidates = []
    for axis in range(3):
        for h in (0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0):
            rig = AnalyticLightRig([initial], {"position"}, torch.float64, classic.occluder)
            loss = objective(rig, target, samples64)[1]
            loss.backward()
            base = rig.position.detach().clone()
            plus, minus = base.clone(), base.clone()
            plus[0, axis] += h
            minus[0, axis] -= h
            mask_plus = classic.occluder.visibility(plus, samples64)
            mask_minus = classic.occluder.visibility(minus, samples64)
            if not bool((mask_plus != mask_minus).any()):
                continue
            rig.position.data.copy_(plus)
            loss_plus = objective(rig, target, samples64)[1].item()
            rig.position.data.copy_(minus)
            loss_minus = objective(rig, target, samples64)[1].item()
            finite = (loss_plus - loss_minus) / (2 * h)
            analytic = rig.position.grad[0, axis].item()
            absolute = abs(analytic - finite)
            candidates.append({
                "axis": axis,
                "h": h,
                "autograd": analytic,
                "finite_difference": finite,
                "absolute_error": absolute,
                "relative_error": absolute / max(abs(analytic), abs(finite), 1e-12),
                "mask_plus_mean": mask_plus.float().mean().item(),
                "mask_minus_mean": mask_minus.float().mean().item(),
                "mask_flip_fraction": (mask_plus != mask_minus).float().mean().item(),
            })
    if candidates:
        return max(candidates, key=lambda candidate: candidate["relative_error"])
    raise RuntimeError(f"No boundary case found for {classic.name}")


def summarize(runs):
    output = {}
    for regime in ("near", "cross"):
        for optimizer in OPTIMIZERS:
            subset = [r for r in runs if r["regime"] == regime and r["optimizer"] == optimizer]
            if not subset:
                continue
            output[f"{regime}_{optimizer}"] = {
                "runs": len(subset),
                "success_rate_field_lt_0_02": float(np.mean([r["success_field_lt_0_02"] for r in subset])),
                "median_initial_field_nrmse": float(np.median([r["initial_field_nrmse"] for r in subset])),
                "median_final_field_nrmse": float(np.median([r["final_field_nrmse"] for r in subset])),
                "max_final_field_nrmse": float(np.max([r["final_field_nrmse"] for r in subset])),
                "median_final_position_error_m": float(np.median([r["final_position_error_m"] for r in subset])),
                "median_final_intensity_relative_error": float(np.median([r["final_intensity_relative_error"] for r in subset])),
                "median_final_visibility_mismatch": float(np.median([r["final_visibility_mismatch"] for r in subset])),
                "median_function_evaluations": float(np.median([r["function_evaluations"] for r in subset])),
                "median_seconds": float(np.median([r["seconds"] for r in subset])),
            }
    return output


def render_representative(classic: ClassicScene, run, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "initial": LightSpec(**run["initial"]),
        "gt": LightSpec(**run["gt"]),
        "result": LightSpec(**run["result"]),
    }
    renders = {}
    resolution = (512, 512) if classic.name == "cornell" else (640, 360)
    for name, spec in specs.items():
        scene = load_point_rig_scene(
            classic, [spec], f"round2_{classic.name}_{run['seed']}_{name}",
            spp=64, resolution=resolution,
        )
        renders[name] = np.asarray(mi.render(scene))
    initial_nrmse = float(np.linalg.norm(renders["initial"] - renders["gt"]) / np.linalg.norm(renders["gt"]))
    optimized_nrmse = float(np.linalg.norm(renders["result"] - renders["gt"]) / np.linalg.norm(renders["gt"]))
    metrics = {
        "scene": classic.name,
        "seed": run["seed"],
        "optimizer": run["optimizer"],
        "backend": f"Mitsuba scalar_rgb direct; {resolution[0]}x{resolution[1]}; 64 spp",
        "initial_linear_rgb_nrmse": initial_nrmse,
        "optimized_linear_rgb_nrmse": optimized_nrmse,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(output_dir / "linear_renders.npz", **renders)
    for name, linear in renders.items():
        Image.fromarray(np.uint8(np.round(tone_map(linear) * 255))).save(output_dir / f"{name}.png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True, facecolor="#fcfcfb")
    for axis, name, title in zip(axes, ("initial", "gt", "result"), ("Initial", "Ground truth", "Optimized")):
        axis.imshow(tone_map(renders[name]))
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"{classic.name}, {run['optimizer']}, seed {run['seed']}")
    fig.savefig(output_dir / "comparison.png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)

    if classic.name == "veach-ajar":
        path_renders = {}
        path_resolution = (512, 288)
        for name, spec in specs.items():
            scene = load_point_rig_scene(
                classic, [spec], f"round2_veach_path_{run['seed']}_{name}",
                spp=64, resolution=path_resolution, integrator_type="path",
            )
            path_renders[name] = np.asarray(mi.render(scene))
        path_optimized_nrmse = float(
            np.linalg.norm(path_renders["result"] - path_renders["gt"])
            / np.linalg.norm(path_renders["gt"])
        )
        metrics["path_visualization"] = {
            "note": "Secondary visualization only; the optimizer target remains direct irradiance.",
            "backend": "Mitsuba scalar_rgb path; 512x288; 64 spp; max_depth=8",
            "optimized_linear_rgb_nrmse": path_optimized_nrmse,
        }
        np.savez_compressed(output_dir / "path_linear_renders.npz", **path_renders)
        for name, linear in path_renders.items():
            Image.fromarray(np.uint8(np.round(tone_map(linear) * 255))).save(output_dir / f"path_{name}.png")
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True, facecolor="#fcfcfb")
        for axis, name, title in zip(axes, ("initial", "gt", "result"), ("Initial", "Ground truth", "Optimized")):
            axis.imshow(tone_map(path_renders[name]))
            axis.set_title(title)
            axis.axis("off")
        fig.suptitle(f"Veach Ajar path visualization, seed {run['seed']} (not the optimization target)")
        fig.savefig(output_dir / "path_comparison.png", dpi=170, facecolor=fig.get_facecolor())
        plt.close(fig)
    return metrics


def save_plots(runs, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"adam": "#2a78d6", "lbfgs": "#eb6834"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True, facecolor="#fcfcfb")
    for axis, regime in zip(axes, ("near", "cross")):
        for optimizer in OPTIMIZERS:
            run = next(r for r in runs if r["regime"] == regime and r["optimizer"] == optimizer and r["seed"] == 0)
            axis.semilogy(
                [h["evaluation"] for h in run["history"]],
                [h["field_nrmse"] for h in run["history"]],
                color=colors[optimizer], linewidth=1.6, label=optimizer.upper(),
            )
        axis.axhline(FIELD_THRESHOLD, color="#52514e", linestyle="--", linewidth=1, label="2% threshold")
        axis.set_title(f"{regime} seed 0")
        axis.set_xlabel("Function evaluation")
        axis.set_ylabel("Field NRMSE")
        axis.grid(color="#e1e0d9", linewidth=.6)
        axis.legend(fontsize=8)
        axis.set_facecolor("#fcfcfb")
    fig.savefig(output_dir / "convergence.png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)

    labels = [f"{r['regime']} {r['seed']}" for r in runs if r["optimizer"] == "adam"]
    x = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(11, 4.5), constrained_layout=True, facecolor="#fcfcfb")
    for offset, optimizer in ((-width / 2, "adam"), (width / 2, "lbfgs")):
        subset = [r for r in runs if r["optimizer"] == optimizer]
        axis.bar(x + offset, [100 * r["final_visibility_mismatch"] for r in subset], width,
                 color=colors[optimizer], label=optimizer.upper())
    axis.set_xticks(x, labels)
    axis.set_ylabel("Final visibility mismatch vs GT (%)")
    axis.set_title("Shadow-topology recovery")
    axis.grid(axis="y", color="#e1e0d9", linewidth=.6)
    axis.set_axisbelow(True)
    axis.legend()
    axis.set_facecolor("#fcfcfb")
    fig.savefig(output_dir / "visibility_mismatch.png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def run_scene(name: str, output_root: Path, adam_steps: int, lbfgs_budget: int):
    classic = load_classic_scene(name, 2000 if name == "cornell" else 1000)
    samples = classic.samples.to(torch.float32)
    gt = gt_spec(classic)
    target = target_field(classic, gt, samples)
    near = make_near_specs(classic, gt, 5 if name == "cornell" else 3, samples)
    cross = cross_specs(classic, gt)
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"{name}: {len(samples.areas)} samples, {sum(s['triangles'] for s in classic.shape_summaries)} triangles, "
        f"GT {gt.position}", flush=True,
    )

    runs = []
    for regime, specifications in (("near", near), ("cross", list(enumerate(cross)))):
        if regime == "near":
            entries = specifications
        else:
            entries = [(seed, spec, None) for seed, spec in specifications]
        for seed, initial, mismatch in entries:
            for optimizer in OPTIMIZERS:
                budget = adam_steps if optimizer == "adam" else lbfgs_budget
                run = optimize_scene(classic, samples, gt, initial, target, optimizer, budget)
                run["regime"] = regime
                run["seed"] = seed
                run["requested_near_visibility_mismatch"] = mismatch
                runs.append(run)
                print(
                    f"{name:10s} {regime:5s} seed={seed} {optimizer:5s} "
                    f"NRMSE {run['initial_field_nrmse']:.4f}->{run['final_field_nrmse']:.6f}; "
                    f"mask {run['initial_visibility_mismatch']:.3f}->{run['final_visibility_mismatch']:.3f}; "
                    f"evals={run['function_evaluations']}; {run['seconds']:.2f}s", flush=True,
                )

    gradient_results = {
        "smooth_steps": [
            smooth_gradient_check(classic, samples, gt, near[0][1], h)
            for h in (1e-4, 1e-5, 1e-6)
        ],
        "boundary": boundary_check(classic, samples, gt, cross[0]),
    }
    (output_dir / "gradient_checks.json").write_text(
        json.dumps(gradient_results, indent=2, default=json_default), encoding="utf-8"
    )

    summary = summarize(runs)
    raw = {
        "scene": name,
        "source_xml": classic.source_xml,
        "samples": len(samples.areas),
        "triangles": sum(s["triangles"] for s in classic.shape_summaries),
        "bounds": [classic.bounds_min.tolist(), classic.bounds_max.tolist()],
        "original_light_positions": classic.original_light_positions,
        "gt": gt.to_dict(),
        "near_seeds": [seed for seed, _, _ in near],
        "shape_summaries": classic.shape_summaries,
        "runs": runs,
        "gradient_checks": gradient_results,
    }
    (output_dir / "raw.json").write_text(json.dumps(raw, indent=2, default=json_default), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fields = [
        "regime", "seed", "optimizer", "function_evaluations", "seconds",
        "initial_field_nrmse", "final_field_nrmse", "final_position_error_m",
        "final_intensity_relative_error", "initial_visibility_mismatch",
        "final_visibility_mismatch", "final_gradient_norm", "success_field_lt_0_02",
    ]
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)

    representative = next(r for r in runs if r["regime"] == "near" and r["seed"] == 0 and r["optimizer"] == "lbfgs")
    render_metrics = render_representative(classic, representative, output_dir / "mitsuba")
    (output_dir / "mitsuba" / "metrics.json").write_text(json.dumps(render_metrics, indent=2), encoding="utf-8")
    save_plots(runs, output_dir)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", choices=("cornell", "veach-ajar"), default=["cornell", "veach-ajar"])
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "round2")
    parser.add_argument("--adam-steps", type=int, default=300)
    parser.add_argument("--lbfgs-budget", type=int, default=300)
    args = parser.parse_args()
    mi.set_variant("scalar_rgb")
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name in args.scenes:
        summaries[name] = run_scene(name, args.output_dir, args.adam_steps, args.lbfgs_budget)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Evaluate Adam pre-optimization followed by L-BFGS refinement."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mitsuba as mi
import torch

from mvp.lighting import LightSpec
from mvp.multi_light_experiments import summarize, save_convergence
from mvp.radiosity import RadiosityTransport, orient_cornell_interior, structured_patch_samples
from mvp.radiosity_experiments import load_transport, optimize_radiosity
from mvp.scene_assets import load_classic_scene

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "round4")
    parser.add_argument("--lbfgs-budget", type=int, default=300)
    args = parser.parse_args()
    mi.set_variant("scalar_rgb")
    torch.set_num_threads(4)
    raw_path = args.results_dir / "raw.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    classic = load_classic_scene("cornell", 100)
    classic.samples = orient_cornell_interior(structured_patch_samples(classic.scene, raw["subdivision_depth"]))
    transport, _ = load_transport(classic, args.results_dir)
    samples = classic.samples.to(torch.float32)
    transport32 = RadiosityTransport(
        transport.form_factor.to(torch.float32), transport.albedo.to(torch.float32)
    )
    gt = [LightSpec(**spec) for spec in raw["gt"]]
    target_rig_specs = gt
    from mvp.multi_light_experiments import full_field
    target = full_field(classic, samples, transport32, target_rig_specs)

    hybrid_runs = []
    for seed in range(5):
        adam_run = next(r for r in raw["runs"] if r["test"] == "joint_cct" and r["seed"] == seed and r["optimizer"] == "adam")
        refinement_start = adam_run["result"]
        refinement_start_specs = [LightSpec(**spec) for spec in refinement_start]
        refine = optimize_radiosity(
            classic, samples, transport32, gt, refinement_start_specs, target,
            "lbfgs", args.lbfgs_budget, color_mode="cct",
            optimize={"position", "intensity", "color"},
        )
        adam_evaluations = adam_run["function_evaluations"]
        refine["history"] = [
            {**entry, "evaluation": entry["evaluation"] + adam_evaluations}
            for entry in refine["history"]
        ]
        refine.update({
            "optimizer": "adam_lbfgs",
            "test": "joint_cct",
            "regime": "near",
            "seed": seed,
            "refinement_start": refinement_start,
            "initial": adam_run["initial"],
            "initial_field_nrmse": adam_run["initial_field_nrmse"],
            "refinement_initial_field_nrmse": refine["initial_field_nrmse"],
            "initial_visibility_mismatch": adam_run["initial_visibility_mismatch"],
            "function_evaluations": adam_evaluations + refine["function_evaluations"],
            "seconds": adam_run["seconds"] + refine["seconds"],
            "history": adam_run["history"] + refine["history"],
        })
        hybrid_runs.append(refine)
        print(
            f"hybrid seed={seed}: {adam_run['initial_field_nrmse']:.4f} -> "
            f"{adam_run['final_field_nrmse']:.6f} -> {refine['final_field_nrmse']:.6f}; "
            f"evals={refine['function_evaluations']}", flush=True,
        )

    raw["runs"] = [r for r in raw["runs"] if r["optimizer"] != "adam_lbfgs"] + hybrid_runs
    raw_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    (args.results_dir / "hybrid_runs.json").write_text(json.dumps(hybrid_runs, indent=2), encoding="utf-8")
    summary = summarize(raw["runs"])
    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fields = [
        "test", "regime", "seed", "optimizer", "function_evaluations", "seconds",
        "initial_field_nrmse", "final_field_nrmse", "final_position_error_m",
        "max_position_error_m", "parameter_permutation", "max_label_position_error_m",
        "final_intensity_relative_error",
        "max_intensity_relative_error", "final_cct_error_kelvin",
        "final_cct_relative_error", "final_label_cct_relative_error", "final_visibility_mismatch", "success_field_lt_0_02",
    ]
    with (args.results_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(raw["runs"])
    save_convergence(raw["runs"], args.results_dir)


if __name__ == "__main__":
    main()

"""Generate three-light GT/result comparisons from saved experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mitsuba as mi
import torch

from mvp.radiosity import RadiosityTransport, orient_cornell_interior, structured_patch_samples
from mvp.radiosity_compare import save_field_comparison, save_path_comparison
from mvp.radiosity_experiments import load_transport
from mvp.scene_assets import load_classic_scene

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "round4")
    args = parser.parse_args()
    mi.set_variant("scalar_rgb")
    raw = json.loads((args.results_dir / "raw.json").read_text(encoding="utf-8"))
    classic = load_classic_scene("cornell", 100)
    classic.samples = orient_cornell_interior(
        structured_patch_samples(classic.scene, raw["subdivision_depth"])
    )
    transport, _ = load_transport(classic, args.results_dir)
    samples = classic.samples.to(torch.float32)
    transport32 = RadiosityTransport(
        transport.form_factor.to(torch.float32), transport.albedo.to(torch.float32)
    )

    metrics = {}
    for test in ("position_intensity", "joint_cct"):
        run = next(r for r in raw["runs"] if r["test"] == test and r["seed"] == 0 and r["optimizer"] == "lbfgs")
        output_dir = args.results_dir / "gt_result" / test
        output_dir.mkdir(parents=True, exist_ok=True)
        field_nrmse = save_field_comparison(classic, samples, transport32, run, output_dir)
        path_metrics = save_path_comparison(classic, run, output_dir)
        metrics[test] = {"field_nrmse": field_nrmse, "path": path_metrics}
    (args.results_dir / "gt_result" / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Run staged Unity-native E24 experiments using immutable E23 checkpoints."""

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from blackbox_e20 import ROOT, git
from blackbox_e24_report import combined_trials, export, stage_selection

REFERENCES = {
    "far": "e23_f_far", "classroom": "e23_f_classroom", "small": "e23_b_small",
    "full": "e23_b_full", "near": "e23_f_near", "best": "e23_f_best",
    "source": "reference/e22_b_on",
}


def inputs(output):
    folder = output / "inputs"
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ROOT / "docs/blackbox-e23/raw-data.zip") as archive:
        for key, path in REFERENCES.items():
            for kind in ("summary", "metrics"):
                name = key if kind == "summary" else f"{key}-metrics"
                (folder / f"{name}.json").write_bytes(archive.read(f"{path}/{kind}.json"))
    return folder


def manifest(project):
    paths = set()
    for name in ("Assets/LightingMvp", "Assets/Settings", "Assets/Scenes/UrpNativeComplexRoom",
                 "Assets/ClassroomScale", "Packages", "ProjectSettings"):
        paths.update(p for p in (project / name).rglob("*") if p.is_file())
    paths.update((project / "Assets/Scenes").glob("UrpNativeComplexRoom.unity*"))
    return {p.relative_to(project).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def prepare(project, assets):
    project, assets = project.resolve(), assets.resolve()
    if project == assets or project == (ROOT / "Light").resolve():
        raise ValueError("Use an isolated batch project as the preparation destination")
    for name in ("Packages", "ProjectSettings", "Assets/LightingMvp"):
        shutil.copytree(ROOT / "Light" / name, project / name, dirs_exist_ok=True)
    for name in ("Assets/Settings", "Assets/Scenes", "Assets/ClassroomScale"):
        shutil.copytree(assets / name, project / name, dirs_exist_ok=True)


def run(args):
    if args.pipeline and args.smoke:
        raise ValueError("Use explicit --stages for smoke runs")
    if not args.pipeline and not args.stages:
        raise ValueError("Pass --stages or --pipeline")
    output, project = args.output.resolve(), args.project.resolve()
    if args.assets_project:
        prepare(project, args.assets_project)
    baseline = inputs(output)
    branch, commit = git("rev-parse", "--abbrev-ref", "HEAD"), git("rev-parse", "HEAD")
    if branch != "blackbox":
        raise ValueError("E24 uses the blackbox branch")
    unity = [str(args.unity.resolve()), "-batchmode", "-force-d3d11", "-job-worker-count", "4", "-projectPath", str(project)]
    kwargs = dict(creationflags=subprocess.CREATE_NO_WINDOW) if hasattr(subprocess, "CREATE_NO_WINDOW") else {}
    def execute(stages, models=()):
        _execute(args, output, project, baseline, branch, commit, unity, kwargs, stages, models)

    if not args.pipeline:
        execute(args.stages, args.models)
    else:
        execute(("a-complex", "a-classroom", "bc-complex", "bc-classroom"))
        screen = stage_selection(output)["screen_models"]
        if screen:
            execute(("p-small",), screen)
        full = stage_selection(output)["full_models"]
        if full:
            execute(("p-full",), full)
        selection = stage_selection(output)
        best, refine = selection["independent_model"], selection["refinement_model"]
        if refine:
            execute(("d-complex",), (refine,))
        # Independent directions are never read by the selection gates above.
        execute(("e-complex", "e-classroom"), (best,) if best else ())
        # One preassigned independent Complex Room direction is evaluated at 50 cm.
        execute(("e50-complex",), (best,) if best else ())
        classroom_30 = [r for r in combined_trials(output / "e24_e_classroom")
                        if r["scenario"].endswith("_30cm") and r["pass"] == "1"]
        if best and classroom_30:
            execute(("e50-classroom",), (best,))
    if args.export:
        export(output, args.export.resolve())

def _execute(args, output, project, baseline, branch, commit, unity, kwargs, stages, models):
    for key in stages:
        stage = f"e24-{key}" + ("-smoke" if args.smoke else "")
        folder = output / stage.replace("-", "_")
        marker = folder / "summary.json"
        if marker.is_file():
            if args.resume:
                continue
            raise FileExistsError(f"{marker} exists; choose another output or pass --resume")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "source-files-sha256.json").write_text(json.dumps(manifest(project), indent=2) + "\n", encoding="utf-8")
        log = output / f"{stage}.log"
        print(f"Running {stage}: {log}", flush=True)
        proc = subprocess.run(unity + ["-executeMethod", "LightingMvp.Editor.NativeRecoveryExperiment.BatchRun",
                                      "-bbCase", stage, "-bbOutput", str(output), "-bbE20Branch", branch,
                                      "-bbE20BaseCommit", commit, "-bbE24Inputs", str(baseline),
                                      "-bbE24AllowDifferentAssetHash", str(int(args.allow_different_asset_hash)),
                                      "-logFile", str(log)] + (["-bbE24Models", ",".join(models)] if models else []), **kwargs)
        if proc.returncode or not marker.is_file():
            raise RuntimeError(f"{stage} failed ({proc.returncode}); see {log}")
        print(f"Completed {stage}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("unity", "project", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--assets-project", type=Path, help="Copy only required sources and benchmark assets; omit Library")
    parser.add_argument("--stages", nargs="+")
    parser.add_argument("--models", nargs="*", default=(), choices=("H4", "H5", "C1", "C3", "C4"))
    parser.add_argument("--pipeline", action="store_true", help="Apply logged phase gates automatically")
    parser.add_argument("--export", type=Path, help="Export only numerical evidence and representative PNGs")
    parser.add_argument("--allow-different-asset-hash", action="store_true",
                        help="Local reimport check, not exact E23 asset replay; raw hashes are still logged.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

"""Prepare and run staged E22 experiments in an isolated Unity project."""

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from blackbox_e20 import ROOT, git
from blackbox_e22_report import report
from blackbox_e22_selection import qualify, select
from blackbox_mbd import prepare as prepare_assets


def prepare(args):
    args.classroom_project = None
    prepare_assets(args)
    project = args.project.resolve()
    manifest = json.loads((project / "mbd-source-sha256.json").read_text(encoding="utf-8"))
    for directory in ("Packages", "ProjectSettings"):
        for path in sorted((project / directory).rglob("*")):
            if path.is_file():
                manifest[path.relative_to(project).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (project / "e22-source-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def e21_input(output):
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ROOT / "docs/blackbox-e21/raw-data.zip") as archive:
        for path in ("e21_b/summary.json", "e21_d/summary.json"):
            (inputs / (path[:5] + ".json")).write_bytes(archive.read(path))
    return inputs


def run(args):
    output, project = args.output.resolve(), args.project.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = e21_input(output)
    branch, commit = git("rev-parse", "--abbrev-ref", "HEAD"), git("rev-parse", "HEAD")
    if branch != "blackbox":
        raise ValueError(f"E22 targets blackbox, found {branch!r}")
    unity = [str(args.unity.resolve()), "-batchmode", "-force-d3d11", "-job-worker-count", "4", "-projectPath", str(project)]
    for key in args.stages:
        stage = f"e22-{key}" + ("-smoke" if args.smoke else "")
        folder = output / stage.replace("-", "_")
        marker = folder / "summary.json"
        if marker.is_file():
            if args.resume:
                continue
            raise FileExistsError(f"{marker} exists. Choose a fresh output directory or --resume.")
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / "e22-source-sha256.json", folder / "source-files-sha256.json")
        log = output / f"{stage}.log"
        print(f"Running {stage}. Log: {log}", flush=True)
        subprocess.run(unity + ["-executeMethod", "LightingMvp.Editor.NativeRecoveryExperiment.BatchRun",
                               "-bbCase", stage, "-bbOutput", str(output), "-bbE20Branch", branch, "-bbE20BaseCommit", commit,
                               "-bbE22Solver", args.solver, "-bbE22Mapping", args.mapping, "-bbE22Inputs", str(inputs),
                               "-bbE22Seed", str(args.seed), "-bbE22Shadows", args.shadows,
                               "-logFile", str(log)], check=True)
        if not marker.is_file():
            raise RuntimeError(f"Unity returned without {marker}. See {log}.")
        print(f"Completed {stage}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    setup = commands.add_parser("prepare")
    setup.add_argument("--project", required=True, type=Path)
    setup.add_argument("--e19-project", required=True, type=Path)
    setup.set_defaults(handler=prepare)
    start = commands.add_parser("run")
    start.add_argument("--unity", required=True, type=Path)
    start.add_argument("--project", required=True, type=Path)
    start.add_argument("--output", required=True, type=Path)
    start.add_argument("--stages", required=True, nargs="+")
    start.add_argument("--solver", choices=("A0", "A1", "A2"), default="A0")
    start.add_argument("--mapping", choices=("B0", "B1", "B2", "B3", "B4", "B5"), default="B0")
    start.add_argument("--seed", type=int, default=157)
    start.add_argument("--shadows", choices=("off", "on"), default="off")
    start.add_argument("--smoke", action="store_true")
    start.add_argument("--resume", action="store_true")
    start.set_defaults(handler=run)
    choice = commands.add_parser("choose")
    choice.add_argument("--output", required=True, type=Path)
    choice.add_argument("--phase", required=True, choices=("a8", "a28", "b-screen"))
    choice.set_defaults(handler=select)
    gates = commands.add_parser("qualify")
    gates.add_argument("--output", required=True, type=Path)
    gates.set_defaults(handler=qualify)
    publish = commands.add_parser("report")
    publish.add_argument("--output", required=True, type=Path)
    publish.add_argument("--report-dir", required=True, type=Path)
    publish.add_argument("--memory-check", type=Path, help="Optional same-configuration repeat with an OS working-set sample")
    publish.set_defaults(handler=report)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

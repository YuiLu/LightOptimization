"""Run the E23 native Shadow On comparisons from fixed E22 checkpoints."""

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from blackbox_e20 import ROOT, git
from blackbox_e22 import prepare as prepare_e22
from blackbox_e23_selection import ASSESSMENTS, assess


def prepare(args):
    if args.classroom_assets_project:
        assets = Path("Assets/ClassroomScale")
        shutil.copytree(args.classroom_assets_project / assets, args.project / assets, dirs_exist_ok=True)
    prepare_e22(args)
    manifest = args.project / "e22-source-sha256.json"
    hashes = json.loads(manifest.read_text(encoding="utf-8"))
    for file in sorted((args.project / "Assets/ClassroomScale").rglob("*")):
        if file.is_file():
            hashes[file.relative_to(args.project).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")


def inputs(output):
    dest = output / "inputs"
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ROOT / "docs/blackbox-e22/raw-data.zip") as archive:
        for name, original in (("on-summary", "e22_b_on/summary"), ("on-metrics", "e22_b_on/metrics"), ("off-summary", "e22_b_off/summary")):
            (dest / (name + ".json")).write_bytes(archive.read(original + ".json"))
    return dest


def run(args):
    output, project = args.output.resolve(), args.project.resolve()
    baseline = inputs(output)
    branch, commit = git("rev-parse", "--abbrev-ref", "HEAD"), git("rev-parse", "HEAD")
    if branch != "blackbox":
        raise ValueError("E23 uses the blackbox branch")
    unity = [str(args.unity.resolve()), "-batchmode", "-force-d3d11", "-job-worker-count", "4", "-projectPath", str(project)]
    for key in args.stages:
        stage = "e23-" + key + ("-smoke" if args.smoke else "")
        folder = output / stage.replace("-", "_")
        marker = folder / "summary.json"
        if marker.is_file():
            if args.resume:
                continue
            raise FileExistsError(f"{marker} exists. Choose a fresh directory or pass --resume.")
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / "e22-source-sha256.json", folder / "source-files-sha256.json")
        log = output / (stage + ".log")
        print(f"Running {stage}. Log: {log}", flush=True)
        subprocess.run(unity + ["-executeMethod", "LightingMvp.Editor.NativeRecoveryExperiment.BatchRun",
                               "-bbCase", stage, "-bbOutput", str(output), "-bbE20Branch", branch, "-bbE20BaseCommit", commit,
                               "-bbE23Inputs", str(baseline), "-bbE23Mapping", args.mapping, "-logFile", str(log)], check=True)
        if not marker.is_file():
            raise RuntimeError(f"Unity returned without {marker}. See {log}.")
        print(f"Completed {stage}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    setup = commands.add_parser("prepare")
    setup.add_argument("--project", required=True, type=Path)
    setup.add_argument("--e19-project", required=True, type=Path)
    setup.add_argument("--classroom-assets-project", type=Path)
    setup.set_defaults(handler=prepare)
    start = commands.add_parser("run")
    for name in ("unity", "project", "output"):
        start.add_argument("--" + name, required=True, type=Path)
    start.add_argument("--stages", nargs="+", required=True)
    start.add_argument("--mapping", choices=tuple(f"C{i}" for i in range(6)), default="C3")
    start.add_argument("--smoke", action="store_true")
    start.add_argument("--resume", action="store_true")
    start.set_defaults(handler=run)
    choose = commands.add_parser("assess")
    choose.add_argument("--output", required=True, type=Path)
    choose.add_argument("--phase", required=True, choices=tuple(ASSESSMENTS))
    choose.set_defaults(handler=assess)
    publish = commands.add_parser("report")
    publish.add_argument("--output", required=True, type=Path)
    publish.add_argument("--export", required=True, type=Path)
    from blackbox_e23_report import report
    publish.set_defaults(handler=report)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

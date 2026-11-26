"""Run the E21 Unity gradient, trust-region, RGB and lighting proxy experiments."""

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

from blackbox_e20 import ROOT, git
from blackbox_e21_report import report
from blackbox_mbd import prepare


STAGES = ("a", "b", "c", "d", "e")


def baseline(path, archive, output, required_file):
    if path is not None:
        path = path.resolve()
        if not (path / required_file).is_file():
            raise FileNotFoundError(path / required_file)
        return path
    if not (output / required_file).is_file():
        output.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extract(required_file, output)
    return output


def run(args):
    output = args.output.resolve()
    project = args.project.resolve()
    output.mkdir(parents=True, exist_ok=True)
    branch, commit = git("rev-parse", "--abbrev-ref", "HEAD"), git("rev-parse", "HEAD")
    unity = [str(args.unity.resolve()), "-batchmode", "-force-d3d11", "-job-worker-count", "4", "-projectPath", str(project)]
    prior = []
    if "b" in args.stages:
        a = baseline(args.e20, ROOT / "docs/blackbox-e20/query-data.zip", output / "inputs/e20", "e20_a/summary.json")
        mbd = baseline(args.mbd, ROOT / "docs/blackbox-e20-mbd/raw-data.zip", output / "inputs/mbd", "topology/summary.json")
        prior = ["-bbE21PriorA", str(a / "e20_a" / "summary.json"), "-bbE21PriorTopology", str(mbd / "topology" / "summary.json")]
    for key in args.stages:
        stage = f"e21-{key}" + ("-smoke" if args.smoke else "")
        folder = output / stage.replace("-", "_")
        marker = folder / "summary.json"
        if marker.is_file():
            if args.resume:
                continue
            raise FileExistsError(f"{marker} exists. Use --resume or a fresh output directory.")
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / "mbd-source-sha256.json", folder / "source-files-sha256.json")
        log = output / (stage + ".log")
        print(f"Running {stage}. Log: {log}", flush=True)
        subprocess.run(unity + ["-executeMethod", "LightingMvp.Editor.NativeRecoveryExperiment.BatchRun",
                               "-bbCase", stage, "-bbOutput", str(output), "-bbE20Branch", branch,
                               "-bbE20BaseCommit", commit, "-logFile", str(log), *prior], check=True)
        if not marker.is_file():
            raise RuntimeError(f"Unity returned without {marker}. See {log}.")
        print(f"Completed {stage}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    setup = commands.add_parser("prepare")
    setup.add_argument("--project", required=True, type=Path)
    setup.add_argument("--e19-project", required=True, type=Path)
    setup.add_argument("--classroom-project", required=True, type=Path)
    setup.set_defaults(handler=prepare)
    start = commands.add_parser("run")
    start.add_argument("--unity", required=True, type=Path)
    start.add_argument("--project", required=True, type=Path)
    start.add_argument("--output", required=True, type=Path)
    start.add_argument("--e20", type=Path)
    start.add_argument("--mbd", type=Path)
    start.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    start.add_argument("--smoke", action="store_true")
    start.add_argument("--resume", action="store_true")
    start.set_defaults(handler=run)
    publish = commands.add_parser("report")
    publish.add_argument("--output", required=True, type=Path, help="Completed Unity runs")
    publish.add_argument("--report-dir", required=True, type=Path)
    publish.set_defaults(handler=report)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

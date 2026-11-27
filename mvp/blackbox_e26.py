"""Reproduce the E26 native RGB stages in an existing isolated E25 Unity project."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

from blackbox_e24 import ROOT, manifest
from blackbox_e25 import references as e24_references

BASE = "81fbfef466efa021d997cdd5482cc557c13a31e3"
FIXED = ("c-numeric-complex", "a-diagnostic-classroom", "a-pool-classroom", "b-backtrack-complex",
         "s-l6-classroom", "s-complex", "s-classroom", "c-replay-complex-911", "c-replay-complex-1201",
         "c-replay-complex-509", "c-replay-classroom-1301", "c-gpu-complex")
REFERENCE = ("a-views-classroom", "b-complex", "c-classroom-1301", "c-classroom-1601", "c-complex-911",
             "c-complex-1201", "c-complex-509", "d-gpu-recovery-131-complex")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def folder(output, key):
    return output / ("e26_" + key.replace("-", "_"))


def summary(output, key):
    return read(folder(output, key) / "summary.json")


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def execute(args):
    project, output, unity = (p.resolve() for p in (args.project, args.output, args.unity))
    if project == (ROOT / "Light").resolve() or not (project / "ProjectSettings/ProjectVersion.txt").is_file():
        raise ValueError("Pass the isolated E25-compatible Unity project, not the main checkout")
    # Unity owns the project lock; a failed compile can leave an empty stale file.
    # Only experiment/plugin code is synced; the archived asset/pipeline settings are verified below.
    if args.sync:
        plugin = "Assets/LightingMvp"
        shutil.copytree(ROOT / "Light" / plugin, project / plugin, dirs_exist_ok=True)
    current = {k: v for k, v in manifest(project).items() if not k.endswith(".cs.meta")}
    inputs = e24_references(output, current)
    with zipfile.ZipFile(ROOT / "docs/blackbox-e25/raw-data.zip") as archive:
        for name in REFERENCE:
            (inputs / f"e25-{name}.json").write_bytes(archive.read(f"{name}/summary.json"))
        ref = json.loads(archive.read("c-complex-911/source-files-sha256.json"))
    prefixes = ("Assets/Scenes/UrpNativeComplexRoom", "Assets/ClassroomScale/", "Assets/Settings/", "Packages/", "ProjectSettings/")
    pinned = {k: v for k, v in ref.items() if k.startswith(prefixes)}
    if not pinned or any(current.get(k) != h for k, h in pinned.items()):
        raise ValueError(f"E25 pipeline/asset manifest differs: {[k for k, v in pinned.items() if current.get(k) != v]}")
    frozen = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs.glob("*.json"))}
    dump(output / "frozen-inputs.json", {"baseline": BASE, "asset_settings": pinned, "input_sha256": frozen})
    launch = [str(unity), "-batchmode", "-force-d3d11", "-job-worker-count", "4", "-projectPath", str(project),
              "-executeMethod", "LightingMvp.Editor.NativeRecoveryExperiment.BatchRun", "-bbOutput", str(output),
              "-bbE20Branch", "blackbox", "-bbE20BaseCommit", BASE, "-bbE25Inputs", str(inputs)]
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if hasattr(subprocess, "CREATE_NO_WINDOW") else {}

    def run(key, *extra):
        if not key or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in key):
            raise ValueError(f"Invalid stage {key!r}")
        target = folder(output, key)
        target.mkdir(exist_ok=True)
        marker, result = target / "complete.json", target / "summary.json"
        settings = dict(baseline=BASE, stage=key, frozen=digest(frozen), sources=digest(current), args=list(extra))
        if marker.exists():
            if args.resume:
                saved = read(marker)
                mismatches = [k for k in settings if settings[k] != saved.get(k)]
                if mismatches and not (args.reuse_completed and mismatches == ["sources"]):
                    raise ValueError(f"Completed stage {key} differs in {mismatches}")
                return
            raise FileExistsError(f"{target} is already complete")
        if result.exists():
            if not args.resume:
                raise FileExistsError(result)
            result.unlink()
        name = "e26-" + key
        log = output / f"{name}.log"
        dump(target / "source-files-sha256.json", current)
        print(f"Starting {name}", flush=True)
        proc = subprocess.run(launch + ["-bbCase", name, "-logFile", str(log), *extra], **kwargs)
        if proc.returncode or not result.is_file():
            raise RuntimeError(f"{name} failed ({proc.returncode}); see {log}")
        dump(marker, settings)
        print(f"Finished {name}", flush=True)

    if args.stages:
        for key in args.stages:
            run(key)
        return
    gates = {}

    def gate(name, launched, reason, **details):
        gates[name] = dict(launched=launched, reason=reason, **details)
        dump(output / "gates.json", gates)

    if not args.skip_color:
        run("d-small-complex")
    for key in FIXED[:4]:
        run(key)
    base, single = summary(output, "a-pool-classroom")["arms"]

    def improved(treatment):
        return not base["pass"] and (treatment["pass"] or
            all(max(treatment[k]) <= .5 * max(base[k]) for k in ("positionCm", "intensityRelative")))

    diagnostic = rows(folder(output, "a-diagnostic-classroom") / "fixed-proposals.csv")
    pair_eligible = any(r["state"] == "v3_failed" and float(r["position_fd_m"]) == .02 and
                        r["source"] in ("views_3", "views_5", "views_6") and float(r["actual"]) > 1e-12 for r in diagnostic)
    gate("A2-dual-view-pool", not improved(single) and pair_eligible,
         "requires no single-pool benefit and a dual-view decrease at the frozen V3 failure",
         single_pool_improved=improved(single), frozen_pair_has_real_decrease=pair_eligible)
    candidate, choice = single, "single"
    if gates["A2-dual-view-pool"]["launched"]:
        run("a-pairs-classroom")
        candidate, choice = summary(output, "a-pairs-classroom")["arms"][0], "pair"
    expand = improved(candidate)
    gate("A2-expansion", expand, "L6 passes or halves both errors vs A0", pool=choice)
    if expand:
        for key in ("classroom-1301", "classroom-1601", "complex-911"):
            run("a-expand-" + key, "-bbE26Pool", choice)
    b0, b1 = summary(output, "b-backtrack-complex")["arms"]
    gate("B1-expansion", b1["pass"], "Complex Far recovery meets position/intensity/legality gate", B0=b0["pass"], B1=b1["pass"])
    if b1["pass"]:
        for key in ("classroom-1601", "complex-911"):
            run("b-expand-" + key)
    for key in ("b-fd-classroom", "b-fd-complex"):
        run(key)
    for key in FIXED[4:-1]:
        run(key)
    for key in ("c-columns-complex", "c-columns-classroom"):
        run(key)
    numeric = folder(output, "c-numeric-complex")
    checks = [r for r in rows(numeric / "gpu-parity.csv") if r["model"] == "C3"] + rows(numeric / "gpu-fixed-stencils.csv")
    parity = bool(checks) and all(r["pass"] == "1" for r in checks)
    gate("C3-GPU-native", parity, "native/synthetic C3 column/Gram/gradient/step tolerances", checks=len(checks))
    if parity:
        run(FIXED[-1])
    if not args.skip_color:
        color_stages(output, run, gate)


def color_stages(output, run, gate):
    run("d-small-complex")
    small = rows(folder(output, "d-small-complex") / "color-trials.csv")
    audit = rows(folder(output, "d-small-complex") / "color-jacobian.csv")
    hard = [r for r in audit if r["shadows"] == "Hard"]
    measurable = len(hard) == 6 and all(r["rank_j5"] == "5" and float(r["per_step_signal_to_repeat_floor"]) > 1 for r in hard)
    ready = measurable and all(len([r for r in small if r["stage"] == phase and r["pass"] == "1"]) == 6 for phase in ("D1", "D2"))
    gate("D3", ready, "D1/D2 both restore six predeclared signed temperature samples and native Hard response is measurable")
    if not ready:
        gate("D4", False, "D3 not launched")
        gate("D5", False, "D4 not launched")
        return
    run("d-single-complex")
    single = rows(folder(output, "d-single-complex") / "color-trials.csv")
    hard5 = [r for r in single if r["method"] == "C3" and r["layout"] == "XYZ;logI;mired"]
    hard4 = {r["sample"]: r for r in single if r["method"] == "C3" and r["layout"] == "XYZ;logI"}
    p5 = sum(r["pass"] == "1" for r in hard5)
    p4 = sum(r["pass"] == "1" for r in hard4.values())
    degraded = [r["sample"] for r in hard5 if any(float(r[m]) > float(hard4[r["sample"]][m]) + .02
                for m in ("held_raw", "held_log", "held_display"))]
    ready = p5 >= 5 and p5 >= p4 - 1 and not degraded
    gate("D4", ready, "Hard 5D >=5/6, at most one fewer than 4D, each held aggregate within 2 NRMSE points of paired 4D",
         hard5=p5, hard4=p4, degraded=degraded)
    if not ready:
        gate("D5", False, "D4 not launched")
        return
    run("d-pair-complex")
    pairs = rows(folder(output, "d-pair-complex") / "color-trials.csv")
    hard10 = [r for r in pairs if r["method"] == "C3" and r["layout"] == "XYZ;logI;mired"]
    if len(hard10) != 4:
        raise ValueError("D4 must contain exactly four 10D arms")
    passed = [r for r in hard10 if r["pass"] == "1"]
    ready = len(passed) >= 3 and {r["group"] for r in passed} == {"low", "high"}
    gate("D5", ready, "Hard 10D >=3/4 and both RGB coupling groups pass", passed=len(passed))
    if ready:
        for i in (911, 1201, 509, 2003, 2909):
            run(f"d-joint-complex-{i}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("unity", "project", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("sync", "resume", "reuse-completed", "skip-color"):
        p.add_argument("--" + name, action="store_true")
    p.add_argument("stages", nargs="*")
    execute(p.parse_args())

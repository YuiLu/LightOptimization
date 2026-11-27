"""Run E25 with native Unity in an isolated, E24-compatible batch project."""
import argparse
import csv
import io
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from blackbox_e24 import ROOT, git, manifest

BASE = "fa060f204469a4ed733a8f622aeb34aaf1761795"
CORE_C = ("c-complex-911", "c-complex-1201", "c-complex-509", "c-classroom-1301", "c-classroom-1601")
REFERENCE = ("a_classroom", "a_complex", "e_classroom", "e_complex", "p_full", "bc_classroom", "bc_complex")


def references(output, current):
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ROOT / "docs/blackbox-e24/raw-data.zip") as archive:
        for key in REFERENCE:
            inputs.joinpath(key + ".json").write_bytes(archive.read(f"e24_{key}/summary.json"))
        with io.TextIOWrapper(archive.open("e24_bc_classroom/jacobians.csv"), encoding="utf-8-sig") as audit:
            row = next(r for r in csv.DictReader(audit) if r["arm"] == "dir_seed509_classroom_near_log_failed_H1")
            inputs.joinpath("classroom_low_response.json").write_text(json.dumps({"theta": [float(x) for x in row["theta"].split(";")]}), encoding="utf-8")
        archived = [json.loads(archive.read(f"e24_{key}/source-files-sha256.json"))
                    for key in ("a_classroom", "e_classroom")]
        # Source code changes in E25 by design. Pin rendering assets and project/pipeline configuration.
        prefixes = ("Assets/Scenes/UrpNativeComplexRoom", "Assets/ClassroomScale/", "Assets/Settings/", "Packages/", "ProjectSettings/")
        diffs = [[p for p in ref if p.startswith(prefixes) and ref[p] != current.get(p)] for ref in archived]
        if all(diffs):
            raise ValueError(f"No exact E24 asset/settings variant: {diffs}")
    return inputs


def execute(args):
    project, output, unity = (path.resolve() for path in (args.project, args.output, args.unity))
    if project == (ROOT / "Light").resolve() or not (project / "ProjectSettings/ProjectVersion.txt").exists():
        raise ValueError("Pass an existing isolated E24-compatible Unity project, not the main checkout")
    if git("rev-parse", "--abbrev-ref", "HEAD") != "blackbox":
        raise ValueError("E25 requires the blackbox branch")
    branch, commit = "blackbox", git("rev-parse", "HEAD")
    if args.sync:
        path = "Assets/LightingMvp"
        shutil.copytree(ROOT / "Light" / path, project / path, dirs_exist_ok=True)
    current = manifest(project)
    inputs = references(output, current)
    source_files = {k: v for k, v in current.items() if not (k.startswith("Assets/LightingMvp/") and k.endswith(".cs.meta"))}
    current_source = hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest()
    launch = [str(unity), "-batchmode", "-force-d3d11", "-job-worker-count", "4", "-projectPath", str(project),
              "-executeMethod", "LightingMvp.Editor.NativeRecoveryExperiment.BatchRun", "-bbOutput", str(output),
              "-bbE20Branch", branch, "-bbE20BaseCommit", commit, "-bbE25Inputs", str(inputs)]
    kwargs = dict(creationflags=subprocess.CREATE_NO_WINDOW) if hasattr(subprocess, "CREATE_NO_WINDOW") else {}
    if args.pipeline:
        stages = orchestration(output, args.recovery_root)
    elif args.followups_root is not None:
        stages = followups(output, args.followups_root.resolve())
    else:
        stages = args.stages or CORE_C
    for key in stages:
        if not key or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in key):
            raise ValueError(f"Invalid stage {key!r}")
        name = f"e25-{key}"
        folder = output / name.replace("-", "_")
        done = folder / "complete.json"
        if done.exists():
            if args.resume:
                record = json.loads(done.read_text(encoding="utf-8"))
                if record["source_manifest_sha256"] != current_source:
                    raise ValueError(f"{name} was produced by different sources; use a new output directory")
                continue
            raise FileExistsError(f"{done} exists; use --resume or a new output directory")
        folder.mkdir(parents=True, exist_ok=True)
        summary = folder / "summary.json"
        if summary.exists() and not args.resume:
            raise FileExistsError(f"Incomplete {name} with a summary; use --resume or a new output directory")
        if summary.exists():
            summary.unlink()
        folder.joinpath("source-files-sha256.json").write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        log = output / f"{name}.log"
        print(f"Starting {name}", flush=True)
        result = subprocess.run(launch + ["-bbCase", name, "-logFile", str(log)], **kwargs)
        if result.returncode or not summary.exists():
            raise RuntimeError(f"{name} failed ({result.returncode}); see {log}")
        done.write_text(json.dumps({"stage": name, "baseline": BASE, "source_manifest_sha256": current_source}, indent=2) + "\n", encoding="utf-8")
        print(f"Finished {name}", flush=True)



def stage_dir(output, key):
    return output / ("e25_" + key.replace("-", "_"))


def completed(output, key):
    folder = stage_dir(output, key)
    marker = json.loads(folder.joinpath("complete.json").read_text(encoding="utf-8"))
    if marker["baseline"] != BASE or marker["stage"] != f"e25-{key}":
        raise ValueError(f"Mismatched E25 stage marker: {folder}")
    return json.loads(folder.joinpath("summary.json").read_text(encoding="utf-8"))


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as source:
        yield from csv.DictReader(source)



def moment_condition(methods, direction_rows):
    off = methods["A-log"]
    proxy_fail = not methods["C3"]["pass"] and not methods["C4"]["pass"]
    native = [r for r in direction_rows if r["state"] == off["scenario"] + "_initial" and r["model"] == "H1" and r["legal"] == "1"]
    groups = {}
    for r in native:
        groups.setdefault((float(r["position_mm"]), float(r["step_fraction"])), []).append(r)
    descend = any(len(group) >= 2 and all(float(r["actual"]) > max(1e-12, 1e-5 * float(r["base_loss"])) for r in group)
                  for group in groups.values())
    unstable = any(float(r["position_mm"]) != 20 and float(r["hard_fd_direction_cos_20mm"]) < .8 for r in native)
    return dict(eligible=off["pass"] and proxy_fail and descend and unstable, off_pass=off["pass"], both_hard_proxies_fail=proxy_fail,
                hard_repeated_decrease=descend, hard_direction_cos_below_point8=unstable)

def orchestration(output, recovery_root):
    """Phase decisions consume only the declared completed inputs, never partially output arms."""
    report = output / "gates.json"
    gates = {}

    def record(name, trigger, reason, **details):
        gates[name] = dict(triggered=trigger, reason=reason, **details)
        report.write_text(json.dumps(gates, indent=2) + "\n", encoding="utf-8")

    yield "a-classroom"
    a2 = next(rows(stage_dir(output, "a-classroom") / "a2-gate.csv"))
    record("A2-visible", a2["launched_visible"] == "1", a2.pop("reason"), **a2)
    yield "a-views-classroom"
    views = next(rows(stage_dir(output, "a-views-classroom") / "view-gate.csv"))
    expand = views["expand_seven_lamps"] == "1"
    record("V5-seven", expand, views["rule"], **views)
    if expand:
        yield "a-views7-classroom"
    yield "b-complex"
    b = next(rows(stage_dir(output, "b-complex") / "b-gate.csv"))
    record("B-pairs", b["launched_pairs"] == "1", "minimum_four_of_seven_single_lamp_passes", **b)
    archive_dir = stage_dir(output, "b-complex")
    with zipfile.ZipFile(ROOT / "docs/blackbox-e24/raw-data.zip") as ref:
        archived = list(csv.DictReader(io.TextIOWrapper(ref.open("e24_a_complex/decisions.csv"), encoding="utf-8-sig")))
    with (archive_dir / "archived-joint-decisions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(archived[0]), lineterminator="\n")
        writer.writeheader()
        archived = [r for r in archived if r["arm"] == "seed509_complex_far_A-log"]
        writer.writerows(archived)
    diagonal = b_scale_condition(archive_dir)
    record("B-diagonal-damping", diagonal["eligible"], "archived_global_backtracking_ge_75pct_and_both_failed_lamps_column_ratio_ge_3", **diagonal)
    if diagonal["eligible"]:
        yield "b-diag-complex"
    if recovery_root is None:
        yield from CORE_C
    recovery = output if recovery_root is None else recovery_root.resolve()
    cases = {key: completed(recovery, key) for key in CORE_C}
    yield "c-audit-classroom"
    yield "c-audit-complex"
    classroom = {key: {a["method"]: a for a in v["arms"]} for key, v in cases.items() if "classroom" in key}
    candidate = next((key for key, m in classroom.items() if m["A-log"]["pass"] and not m["C3"]["pass"] and not m["C4"]["pass"]), None)
    reason = "matching_Classroom_Off_pass_and_both_Hard_Log_proxies_fail"
    if candidate is None:
        # A representative, fixed midpoint in the Complex sequence isolates objective vs XYZ model.
        arm = {a["method"]: a for a in cases["c-complex-1201"]["arms"]}
        if not arm["H0"]["pass"] and arm["C3"]["pass"] and arm["C4"]["pass"]:
            candidate = "c-complex-1201"
            reason = "same_Hard_Log_objective_full_FD_vs_proxies_on_fixed_Complex_contrast"
    record("H1-recovery", candidate is not None, reason, case=candidate)
    if candidate is not None:
        yield candidate.replace("c-", "h1-", 1)
    record("independent-default-selection", False, "preserve_both_E25_baselines_no_default_election")
    directions = list(rows(stage_dir(output, "c-audit-classroom") / "native-steps.csv"))
    for key, methods in classroom.items():
        decision = moment_condition(methods, directions)
        eligible = decision["eligible"]
        record("E-" + key, eligible, "matching_Off_pass_both_proxies_fail_repeated_Hard_drop_and_step_instability", **decision)
        if eligible:
            yield key.replace("c-", "moment-", 1)
    yield "d-fixed-complex"
    parity = list(rows(stage_dir(output, "d-fixed-complex") / "numeric-parity.csv"))
    pass_numeric = bool(parity) and all(r["pass"] == "1" for r in parity)
    record("D1-end-to-end", pass_numeric, "fixed_input_columns_Gram_gradient_and_finite_gate", checks=len(parity))
    if pass_numeric:
        yield "d-replay-157-complex"
        failing = next((k.split("-")[-1] for k, m in classroom.items() if not m["C4"]["pass"]), None)
        if failing:
            yield f"d-replay-{failing}-classroom"
        else:
            record("D1-independent-failure", False, "no_eligible_Classroom_C4_failure")
        yield "d-replay-131-complex"
        gpu = gpu_condition(output)
        record("D2-transfer-benchmark", gpu["eligible"], "cached_mapping_FD_dominant_and_ge_30pct_wall", **gpu)
        if gpu["eligible"]:
            yield "d-gpu-transfer-complex"
            bench = gpu_benchmark_condition(output)
            record("D2-native-replay", bench["eligible"], "fixed_input_GPU_parity_and_roundtrip_speedup_ge_1p2", **bench)
            if bench["eligible"]:
                yield "d-gpu-recovery-131-complex"
                yield "d-gpu-recovery-157-complex"
                if failing:
                    yield f"d-gpu-recovery-{failing}-classroom"


def b_scale_condition(archive_dir):
    archived = list(rows(archive_dir / "archived-joint-decisions.csv"))
    frac = sum(int(r["shrinks"]) > 0 for r in archived) / len(archived)
    spectrum = next(r for r in rows(archive_dir / "local-fd.csv") if r["view"] == "-1")
    norms = [float(x) for x in spectrum["column_norms"].split(";")]
    ratios = {f"L{i + 1}": max(norms[i * 4:i * 4 + 4]) / max(1e-15, min(norms[i * 4:i * 4 + 4])) for i in (1, 5)}
    return dict(eligible=frac >= .75 and min(ratios.values()) >= 3, shrink_fraction=frac, failed_lamp_norm_ratios=ratios)


def gpu_condition(root):
    import statistics
    perf = list(rows(stage_dir(root, "d-replay-131-complex") / "arm-cost.csv"))
    cache = [r for r in perf if "_rep" in r["arm"] and r["arm"].endswith("_C4-cache")]
    if len(cache) < 3:
        raise ValueError("GPU gate requires three warmed cached end-to-end repeats")
    fractions = {part: statistics.median(float(r[part]) / float(r["opt_wall_s"]) for r in cache)
                 for part in ("map_fd_s", "mask_p95_s", "gram_gradient_s", "render_s", "readback_s")}
    return dict(eligible=fractions["map_fd_s"] >= .3 and fractions["map_fd_s"] == max(fractions.values()), wall_fractions=fractions)



def gpu_benchmark_condition(root):
    import statistics
    folder = stage_dir(root, "d-gpu-transfer-complex")
    fixed = list(rows(folder / "gpu-parity.csv"))
    speed = [float(r["cpu_cached_wall_s"]) / float(r["full_gpu_path_wall_s"]) for r in rows(folder / "gpu-transfer.csv")]
    gain = statistics.median(speed) if speed else None
    ok = bool(fixed) and all(r["pass"] == "1" for r in fixed)
    return dict(eligible=ok and gain is not None and gain >= 1.2, parity_pass=ok, fixed_checks=len(fixed), median_roundtrip_speedup=gain)

def followups(output, root):
    completed(root, "b-complex")
    completed(root, "d-replay-131-complex")
    diagonal = b_scale_condition(stage_dir(root, "b-complex"))
    gpu = gpu_condition(root)
    source = {"B-diagonal-damping": diagonal, "D2-transfer-benchmark": gpu}
    (output / "gates.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    if diagonal["eligible"]:
        yield "b-diag-complex"
    if gpu["eligible"]:
        yield "d-gpu-transfer-complex"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("unity", "project", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stages", nargs="+")
    mode.add_argument("--pipeline", action="store_true")
    mode.add_argument("--followups-root", type=Path, help="Repeat the B scale and D2 gates using a completed E25 diagnostics root")
    parser.add_argument("--recovery-root", type=Path, help="Reuse completed core C stages from an earlier E25 run with the same measurement protocol")
    parser.add_argument("--sync", action="store_true", help="Copy committed or working-tree LightingMvp sources to the isolated project")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.recovery_root is not None and not args.pipeline:
        parser.error("--recovery-root requires --pipeline")
    execute(args)


if __name__ == "__main__":
    main()

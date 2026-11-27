"""Export verified E26 results and reproducibility snapshots without render buffers."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import zipfile

from blackbox_e26 import BASE, ROOT, digest, dump, read, rows

SPAN = math.log(9 / .055)
METRICS = ("raw_nrmse", "log_nrmse", "display_nrmse")
TABLES = ("view-gradient", "fixed-proposals", "proposal-audit", "constraint-backtrack", "nonmonotone-trace",
          "stall-fd", "cpu-columns", "color-jacobian", "color-native", "color-boundaries", "color-coupling", "color-trials", "gpu-parity",
          "gpu-fixed-stencils", "gpu-transfer", "query-log", "profiling", "view-metrics", "model-checkpoints", "first-pass")


def write(path, records):
    records = list(records)
    fields = list(dict.fromkeys(k for r in records for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields, lineterminator="\n")
        w.writeheader()
        w.writerows(records)


def load(p):
    return rows(p) if p.is_file() else []


def zrows(z, n):
    return list(csv.DictReader(io.StringIO(z.read(n).decode("utf-8-sig"))))


def selected(s, arm):
    return [r for r in s if r["arm"] == arm]


def section_order(p):
    key = p.name.removeprefix("e26_")
    return ("abscd".index(key[0]), key)


def pose_id(text):
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def halfsplit(xs):
    return [float(x) for x in xs.split(";") if x]


def vector_difference(a, b):
    assert len(a) == len(b)
    return max((abs(float(x) - float(y)) for x, y in zip(a, b)), default=0.)


def rms(x):
    return math.sqrt(statistics.mean(v * v for v in x))


def arm_meta(config, arm, profile, group=""):
    name = arm["id"]
    seed = next((n for n in (1301, 1601, 1201, 911, 509, 131, 157, 733) if str(n) in name), "")
    scene = config["scene"]
    span = [scene["boundsMax"][k] - scene["boundsMin"][k] for k in "xyz"]
    lamps = arm["lamps"]
    gt, start = config["truth"], arm["initial"]
    positions = [math.sqrt(sum(((start[4 * (l - 1) + j] - gt[4 * (l - 1) + j]) * span[j]) ** 2 for j in range(3))) * 100 for l in lamps]
    intensities = [math.expm1((start[4 * (l - 1) + 3] - gt[4 * (l - 1) + 3]) * SPAN) for l in lamps]
    return dict(baseline=BASE, scene="Classroom" if "classroom" in config["stage"] else "Complex", scene_sha=config["sceneHash"],
                seed=seed, active_lamps=";".join(map(str, lamps)), layout="XYZ;logI", mapping="SignedLog", view_ids=profile["fit_views"],
                start_position_cm=";".join(f"{x:.8g}" for x in positions), start_intensity_relative=";".join(f"{x:.8g}" for x in intensities),
                start_mired_delta="", shadow_target="Hard" if arm["shadows"] else "Off", camera_cap=profile["cap_camera"],
                final_legal=int(arm["legal"]), stop_reason=arm["stop"], matrix_group=group)


def run_totals(config, metered, d):
    all_cameras = config["nativeCameraRenders"]
    diagnostic = int((d / "diagnostic-cameras.txt").read_text()) if (d / "diagnostic-cameras.txt").is_file() else 0
    total_opt = sum(int(p["opt_camera"]) for p in metered)
    trial_validation = sum(int(p["other_camera"]) for p in metered)
    if d.name == "e26_c_numeric_complex":
        # E21 Target(True): 3 warmup + 3 target + 3 repeat + 5 high frames.
        diagnostic = all_cameras - (9 + len(config["scene"]["viewPositions"]))
    if "c_columns" in d.name:
        diagnostic += total_opt
        total_opt = 0
    assert total_opt + trial_validation + diagnostic <= all_cameras, d
    return dict(all_camera=all_cameras, opt_camera=total_opt, validation_camera=trial_validation, diagnostic_camera=diagnostic,
                setup_or_target_camera=all_cameras - total_opt - trial_validation - diagnostic, stage_wall_s=config["seconds"],
                opt_wall_s=sum(float(p["opt_wall_s"]) for p in metered))


def parity_check(stages, ref):
    output = []
    for d, data in stages:
        if "c-replay" not in data["stage"]:
            continue
        old, new = data["arms"]
        stage = data["stage"].removeprefix("e26-c-replay-")
        reference_stage = "c-" + stage
        baseline = next(a for a in json.loads(ref.read(f"{reference_stage}/summary.json"))["arms"] if a["method"] == "C3")
        jac, trials, gram = (rows(d / name) for name in ("jacobians.csv", "decisions.csv", "gram.csv"))
        base_jac = selected(zrows(ref, f"{reference_stage}/jacobians.csv"), baseline["id"])
        base_steps = selected(zrows(ref, f"{reference_stage}/decisions.csv"), baseline["id"])
        base_gram = selected(zrows(ref, f"{reference_stage}/gram.csv"), baseline["id"])
        new_gram = selected(gram, new["id"])
        new_jac = selected(jac, new["id"])
        new_steps = selected(trials, new["id"])
        for tag, compare, jac_rows, steps, grams in (("implicit_E26", old, selected(jac, old["id"]), selected(trials, old["id"]), selected(gram, old["id"])),
                                                     ("archived_E25", baseline, base_jac, base_steps, base_gram)):
            col_err = max((vector_difference(halfsplit(a["column_norms"]), halfsplit(b["column_norms"])) for a, b in zip(jac_rows, new_jac)), default=0)
            delta_err = max((vector_difference(halfsplit(a["step_phys"]), halfsplit(b["step_phys"])) for a, b in zip(steps, new_steps)), default=0)
            gradient_err = max((vector_difference(halfsplit(a["gradient_theta"]), halfsplit(b["gradient_theta"])) for a, b in zip(jac_rows, new_jac)), default=0)
            gram_err = max((vector_difference(halfsplit(a["values"]), halfsplit(b["values"])) for a, b in zip(grams, new_gram)), default=0)
            theta_err = vector_difference(compare["final"], new["final"])
            q_same = compare["evaluations"] == new["evaluations"]
            trajectory_same = (len(jac_rows), len(steps)) == (len(new_jac), len(new_steps)) and all(a["accepted"] == b["accepted"] for a,b in zip(steps, new_steps))
            same = q_same and trajectory_same and max(col_err, delta_err, gradient_err, gram_err) <= 1e-6 and theta_err <= 1e-7
            output.append(dict(baseline=BASE, experiment=data["stage"], scene=stage.split("-")[0], seed=stage.split("-")[-1],
                comparison=tag, reference_images=compare["evaluations"] * 3, new_images=new["evaluations"] * 3,
                reference_models=compare["jacobians"], new_models=new["jacobians"], ref_pass=int(compare["pass"]), new_pass=int(new["pass"]),
                column_norm_max_abs=col_err, gram_max_abs=gram_err, gradient_max_abs=gradient_err, physical_step_max_abs=delta_err, normalized_final_max_abs=theta_err,
                column_norm_step_abs_tolerance=1e-6, normalized_final_abs_tolerance=1e-7,
                trajectory_same=int(trajectory_same), budget_same=int(q_same), parity_pass=int(same)))
    return output


def export(output, dest):
    stages = [(p, read(p / "summary.json")) for p in sorted(output.glob("e26_*"), key=section_order) if (p / "complete.json").is_file()]
    if not stages:
        raise ValueError("No completed E26 stage")
    dest.mkdir(parents=True, exist_ok=True)
    fixed = read(output / "frozen-inputs.json")
    stage_records = {}
    combined = defaultdict(list)
    trials = []
    truth_snapshot = {}
    figures = dest / "figures"
    figures.mkdir(exist_ok=True)
    panels = {
        "e26-a-pool-classroom": ("gt_off/gt_v0.png", "L6_seed509_A0_A-log/initial_v0.png", "L6_seed509_A0_A-log/best_v0.png"),
        "e26-b-backtrack-complex": ("gt_off/gt_v0.png", "Far_seed509_B0_A-log/initial_v0.png", "Far_seed509_B0_A-log/best_v0.png", "Far_seed509_B1_A-log/best_v0.png"),
        "e26-d-single-complex": ("D3_L1_seed911_C3_5D/final_v0.png",),
    }
    with zipfile.ZipFile(dest / "raw-parameters.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(output / "frozen-inputs.json", "frozen-inputs.json")
        for d, config in stages:
            stage = config["stage"]
            marker = read(d / "complete.json")
            if config["commit"] != BASE or marker["baseline"] != BASE:
                raise ValueError(f"Unexpected {stage} baseline")
            profiles = load(d / "profiling.csv") + load(d / "color-profiling.csv")
            costs = {r["arm"]: r for r in profiles}
            trial_rows = {r["arm"]: r for r in load(d / "trials.csv")}
            first = {r["arm"]: r for r in load(d / "first-pass.csv")}
            queries = load(d / "query-log.csv")
            counters = Counter()
            for q in queries:
                a = q["arm"]
                counters[a] += int(q["cameras"])
                if counters[a] != int(q["actual_camera"]):
                    raise ValueError(f"Camera ledger diverged in {stage} {a}")
            for p in profiles:
                a = p["arm"]
                if not counters[a] == int(p["opt_camera"]) == int(p["hard_camera"]) + int(p["off_camera"]) <= int(p["cap_camera"]):
                    raise ValueError(f"Invalid camera ledger for {stage}: {a}")
            color = load(d / "color-trials.csv")
            context = dict(experiment=stage, baseline=BASE, scene="Classroom" if "classroom" in stage else "Complex", scene_sha=config["sceneHash"])
            metadata = {a["id"]: arm_meta(config, a, costs[a["id"]]) for a in config["arms"]}
            metadata.update({r["arm"]: dict(context, seed=r["seed"], active_lamps=r["lamps"], layout=r["layout"], mapping="SignedLog", view_ids="0;1;2",
                              start_position_cm=r["position_perturb_cm"], start_intensity_relative=r["intensity_perturb_relative"], start_mired_delta=r["actual_mired_delta"],
                              camera_cap=r["cap_camera"], shadow_target=r["shadows"], final_legal=r["legal"], stop_reason=r["stop"], matrix_group=r["group"]) for r in color})
            scores = load(d / "view-metrics.csv")
            for a, r in trial_rows.items():
                if r["legal"] != "1" or r["illegal_accepted"] != "0":
                    raise ValueError(f"Illegal result: {a}")
                matching = metadata[a]["shadow_target"].lower()
                high = [s for s in selected(scores, a) if s["checkpoint"] == "final" and s["width"] == "896" and s["target"] == matching]
                held = [s for s in high if int(s["view"]) >= 3]
                fitted = [s for s in high if s["view"] in costs[a]["fit_views"].split(";")]
                traits = next(s for s in config["arms"] if s["id"] == a)
                trials.append(dict(context, **{k: v for k, v in metadata[a].items() if k not in context}, **r,
                                   opt_camera=costs[a]["opt_camera"], opt_wall_s=costs[a]["opt_wall_s"],
                                   first_pass_camera=first.get(a, {}).get("cameras", ""), first_pass_model=first.get(a, {}).get("model", ""), first_pass_wall_s=first.get(a, {}).get("elapsed_s", ""),
                                   position_error_cm_by_lamp=";".join(map(str, traits["positionCm"])), intensity_relative_by_lamp=";".join(map(str, traits["intensityRelative"])),
                                   **{f"{part}_{m}": rms(float(s[m]) for s in subset) for part, subset in (("fit896", fitted), ("held896", held)) for m in METRICS}))
            for r in color:
                if r["legal"] != "1":
                    raise ValueError(f"Illegal color result: {r['arm']}")
            for table in TABLES:
                inputs = load(d / (table + ".csv")) if table != "profiling" else profiles
                for r in inputs:
                    meta = metadata.get(r.get("arm"), {})
                    diag = (dict(seed=509, active_lamps="6", layout="XYZ;logI", mapping="SignedLog", view_ids="0;1;2", camera_cap="diagnostic", shadow_target="Off") if "classroom" in stage
                            else dict(camera_cap="diagnostic", view_ids="0;1;2"))
                    if table == "query-log":
                        r["pose_hash"] = pose_id(r.pop("theta"))
                        # Full precision poses are retained in the compressed evidence archive.
                    combined[table].append({**context, **(meta or diag), **r})
            for name in ("complete.json", "summary.json", "source-files-sha256.json", "query-log.csv", "color-config.json", "color-snapshots.json", "diagnostic-cameras.txt"):
                p = d / name
                if p.is_file():
                    archive.write(p, f"{stage}/{name}")
            for path in panels.get(stage, ()):
                p = d / "captures" / path
                if p.is_file():
                    (figures / f"{stage}_{path.replace('/', '_')}").write_bytes(p.read_bytes())
            stage_records[stage] = {**run_totals(config, profiles, d), "source_manifest_sha256": marker["sources"], "input_sha256": marker["frozen"],
                "hardware": {k: config["scene"][k] for k in ("unity", "urp", "gpu", "graphics", "colorSpace", "renderer")}, "trials": len(config["arms"]) + len(color)}
            truth_snapshot[stage] = dict(scene_sha=config["sceneHash"], camera_sha=digest([config["scene"][k] for k in ("viewPositions", "viewRotations")]),
                truth_sha=digest(config["truth"]), start_sha={a["id"]: digest(a["initial"]) for a in config["arms"]})
            if (d / "color-snapshots.json").is_file():
                truth_snapshot[stage]["color_truth_sha"] = digest(read(d / "color-config.json")["truth"])
                truth_snapshot[stage]["color_start_sha"] = {a["arm"]: digest(a["start"]) for a in read(d / "color-snapshots.json")["samples"]}
        for name in ("e25-a-views-classroom.json", "e25-c-complex-911.json", "e25-c-complex-1201.json", "e25-c-complex-509.json",
                     "e25-c-classroom-1301.json", "e25-c-classroom-1601.json", "e25-d-gpu-recovery-131-complex.json"):
            archive.write(output / "inputs" / name, "references/" + name)
    with zipfile.ZipFile(ROOT / "docs/blackbox-e25/raw-data.zip") as ref:
        replay = parity_check(stages, ref)
    totals = {k: sum(v[k] for v in stage_records.values()) for k in ("all_camera", "opt_camera", "validation_camera", "diagnostic_camera", "setup_or_target_camera", "stage_wall_s", "opt_wall_s", "trials")}
    gates = read(output / "gates.json") if (output / "gates.json").is_file() else {}
    d4 = [r for r in combined["color-trials"] if r["stage"] == "D4" and r["layout"] == "XYZ;logI;mired"]
    if d4 and "D5" in gates:
        passing = [r for r in d4 if r["pass"] == "1"]
        if len(d4) != 4 or gates["D5"]["launched"] != (len(passing) >= 3 and {r["group"] for r in passing} == {"low", "high"}):
            raise ValueError("Actual D5 launch disagrees with the 10D gate")
        # Report only the 10D arms, not the paired fixed-temperature controls.
        gates["D5"].update(passed=len(passing), trials=len(d4))
    d5 = [r for r in combined["color-trials"] if r["stage"] == "D5" and r["layout"] == "XYZ;logI;mired"]
    if d5:
        passes = [r for r in d5 if r["pass"] == "1"]
        gates["D5-outcome"] = dict(completed=len(d5), planned=5, passed=len(passes),
                                  meets_research_target=len(d5) == 5 and len(passes) >= 4 and any(r["seed"] in ("2003", "2909") for r in passes))
    dump(dest / "stage-summary.json", dict(baseline=BASE, total=totals, stages=stage_records, gates=gates,
         frozen_input_sha256=fixed["input_sha256"], scene_camera_state=truth_snapshot, cpu_layout_parity=replay))
    write(dest / "trials.csv", trials)
    write(dest / "layout-parity.csv", replay)
    for key in TABLES:
        if combined[key]:
            write(dest / (key + ".csv"), combined[key])
    print(f"{len(stages)} stages; {totals['trials']} arms; {totals['opt_camera']} optimization camera images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    export(args.output, args.dest)

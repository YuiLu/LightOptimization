"""E24 phase gates, evidence validation and compact artifact export."""

import csv
import json
import math
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

MODELS = ("H4", "C3", "H5", "C4", "C1")  # fixed tie order, before experiments


def rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def table(path, data, header=()):
    fields = list(dict.fromkeys([*header, *(key for row in data for key in row)]))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def direction_gate(output, stages=("bc_complex", "bc_classroom")):
    groups = defaultdict(dict)
    for stage in stages:
        folder = output / f"e24_{stage}"
        if not (folder / "summary.json").is_file():
            continue
        for r in rows(folder / "direction-probes.csv"):
            if r["failed"] != "1":
                continue
            by_model = groups[(stage, r["state"])]
            loss, gain = float(r["loss_start"]), 0.
            if r["legal"] == "1":
                gain = max(0., min(float(r["actual_1"]), float(r["actual_2"])))
            old = by_model.get(r["model"], (loss, 0.))
            by_model[r["model"]] = (loss, max(gain, old[1]))
    result = {}
    for model in MODELS:
        control = "H0" if model == "C1" else "H2"
        compared, wins = [], []
        for (_, state), group in groups.items():
            if model not in group or control not in group:
                continue
            loss, base = group[control]
            _, gain = group[model]
            if abs(group[model][0] - loss) > 1e-10 * max(1., loss):
                raise ValueError(f"Different target at {state}: {model} / {control}")
            compared.append(state)
            if gain > 0 and (base <= 0 or gain - base > max(.001 * loss, .05 * base)):
                wins.append(state)
        result[model] = dict(control=control, compared=compared, wins=wins, passed=len(wins) >= 2)
    return result


def _rms(data):
    return math.sqrt(sum(x*x for x in data) / len(data))


def _metrics(path):
    out, views = defaultdict(dict), defaultdict(set)
    expected = {(448, i) for i in range(3)} | {(896, i) for i in range(5)}
    for r in rows(path / "view-metrics.csv"):
        group = (r["arm"], r["checkpoint"], r["target"])
        width, view = int(r["width"]), int(r["view"])
        if (width, view) not in expected or (width, view) in views[group]:
            raise ValueError(f"Unexpected or repeated evaluation view: {path}/{group}/{width}/{view}")
        views[group].add((width, view))
        view_group = "train" if width == 448 else "same" if view < 3 else "held"
        for metric in ("raw", "log", "display"):
            key = f'{r["checkpoint"]}_{r["target"]}_{view_group}_{metric}'
            out[r["arm"]].setdefault(key, []).append(float(r[metric + "_nrmse"]))
    if any(v != expected for v in views.values()):
        raise ValueError(f"Incomplete evaluation views: {path}")
    return {arm: {k: _rms(v) for k, v in scores.items()} for arm, scores in out.items()}


def combined_trials(path):
    if not (path / "summary.json").is_file():
        return []
    metrics = _metrics(path)
    costs = {r["arm"]: r for r in rows(path / "arm-cost.csv")}
    stage = read(path / "summary.json")
    switches = {r["arm"]: r for r in rows(path / "switch-log.csv")}
    trials = rows(path / "trials.csv")
    if set(costs) != {r["arm"] for r in trials}:
        raise ValueError(f"Missing native camera counts in {path}")
    model_frames = defaultdict(list)
    for r in rows(path / "model-queries.csv"):
        model_frames[r["arm"]].append(r)
    result = []
    for r in trials:
        cost = costs[r["arm"]]
        camera = int(cost["opt_camera"])
        if not 0 < camera <= int(cost["cap_camera"]) or camera != 3 * int(r["evaluations"]):
            raise ValueError(f"Invalid camera budget: {r}")
        if int(cost["on_camera"]) + int(cost["off_camera"]) != camera:
            raise ValueError(f"Invalid On/Off camera count: {cost}")
        seen = model_frames[r["arm"]]
        if r["shadows"] == "1" and sum(int(m["camera_renders"]) for m in seen if m["shadows"] == "None") != int(cost["off_camera"]):
            raise ValueError(f"Off probe log disagrees with bill: {r}")
        if r["legal"] != "1" or r["illegal_accepted"] != "0":
            raise ValueError(f"Unexpected illegal pose: {r}")
        if cost.get("hard_shadow_pass_cameras") and not 0 <= int(cost["hard_shadow_pass_cameras"]) <= int(cost["on_camera"]):
            raise ValueError(f"Invalid shadow render count: {cost}")
        switch = switches.get(r["arm"])
        if r["scenario"].endswith(("_D2", "_D3")):
            if switch is None:
                raise ValueError(f"Missing refinement switch: {r}")
            used, remaining = int(switch["used_camera"]), int(switch["remaining_camera"])
            if used > camera or used + remaining != int(cost["cap_camera"]) or (not switch["to_model"] and used != camera):
                raise ValueError(f"Invalid refinement camera accounting: {r}")
        scores = metrics[r["arm"]]
        target = "hard" if r["shadows"] == "1" else "off"
        if not {f"{checkpoint}_{target}_train_raw" for checkpoint in ("initial", "final")} <= scores.keys():
            raise ValueError(f"Missing arm checkpoints: {r}")
        result.append(dict(stage=stage["stage"], scene=stage["scene"]["scene"], **(r | cost | scores)))
    return result


def recovery_gate(path, candidates):
    data = combined_trials(path)
    result = {}
    for model in candidates:
        control = "H0" if model == "C1" else "H2"
        base = {r["scenario"]: r for r in data if r["model"] == control}
        comparisons = []
        for new in [r for r in data if r["model"] == model]:
            old = base[new["scenario"]]
            def error(r):
                return max(float(r["max_position_cm"])/5, float(r["max_intensity_relative"])/.05)
            e_new, e_old = error(new), error(old)
            held_new, held_old = float(new["final_hard_held_display"]), float(old["final_hard_held_display"])
            win = (new["pass"] == "1" and old["pass"] != "1") or (
                e_old > 1 and e_new <= .8*e_old and held_new <= held_old + .002) or (
                new["pass"] == "1" and old["pass"] == "1" and held_new <= held_old + .002 and
                int(new["opt_camera"]) <= .8*int(old["opt_camera"]))
            loss = old["pass"] == "1" and (new["pass"] != "1" or held_new > held_old + .002)
            comparisons.append(dict(scenario=new["scenario"], pass_control=old["pass"] == "1", pass_candidate=new["pass"] == "1",
                                    win=win, regression=loss, parameter_normalized_control=e_old, parameter_normalized_candidate=e_new,
                                    held_display_control=held_old, held_display_candidate=held_new))
        result[model] = dict(control=control, comparisons=comparisons,
                             passed=any(r["win"] for r in comparisons) and not any(r["regression"] for r in comparisons))
    return result


def stage_selection(output):
    directions = direction_gate(output)
    screen = [m for m in MODELS if directions[m]["passed"]]
    small = recovery_gate(output / "e24_p_small", screen)
    full_models = [m for m in screen if small[m]["passed"]]
    full = recovery_gate(output / "e24_p_full", full_models)
    best = next((m for m in full_models if full[m]["passed"]), None)
    refine = next((m for m in full_models if m != "C1" and full[m]["passed"]), None)
    conditional = []
    for stage in ("bc_complex", "bc_classroom"):
        local = direction_gate(output, (stage,))
        for r in rows(output / f"e24_{stage}" / "conditional-models.csv"):
            if r["model"] == "C4":
                r["qualifying_common_failed_states"] = sorted(set(local["H4"]["wins"]) & set(local["C3"]["wins"]))
                r["recorded_eligible_anchor_count"] = int(r["eligible_anchor_count"])
                r["eligible_anchor_count"] = len(r["qualifying_common_failed_states"])
            conditional.append(dict(stage=stage, **r))
    classroom = output / "e24_e_classroom"
    if (output / "e24_e50_classroom/summary.json").is_file():
        classroom_50 = "started"
    elif not (classroom / "summary.json").is_file():
        classroom_50 = "pending_classroom_30"
    elif not best:
        classroom_50 = "no_qualifying_model"
    elif not any(r["scenario"].endswith("_30cm") and r["pass"] == "1" for r in combined_trials(classroom)):
        classroom_50 = "classroom_30cm_no_pass"
    else:
        classroom_50 = "pending_or_no_legal_50cm_start"
    return {"direction_rule": "two failed states, best of legal repeated drops, exceeds max(0.1% initial loss, 5% baseline drop); positive if baseline has no drop",
            "recovery_rule": "new pass, or >=20% lower failed-arm scaled parameter error without >0.2 pp holdout display loss, or equal pass with >=20% fewer cameras; no loss/regression of any passing baseline",
            "directions": directions, "screen_models": screen, "screen_recovery": small, "full_models": full_models,
            "full_recovery": full, "independent_model": best, "refinement_model": refine,
            "conditional_arms": {"H5_C4": conditional,
                                 "8D": "started" if (output / "e24_p_small/summary.json").is_file() else "direction_gate_not_met" if not screen else "pending",
                                 "28D": "started" if (output / "e24_p_full/summary.json").is_file() else "8D_gate_not_met" if not full_models else "pending",
                                 "D": "started" if (output / "e24_d_complex/summary.json").is_file() else "no_qualifying_log_auxiliary" if not refine else "pending",
                                 "Moment": "not_started_no_new_evidence_of_needed_proxy_component",
                                 "E50-Classroom": classroom_50}}


def export(output, dest):
    dest.mkdir(parents=True, exist_ok=True)
    stages = [p for p in sorted(output.glob("e24_*")) if (p / "summary.json").is_file() and not p.name.endswith("_smoke")]
    data = [r for path in stages for r in combined_trials(path)]
    if not data:
        raise ValueError("No finished E24 trials to export")
    table(dest / "trials.csv", data)
    for name, header in (("direction-probes.csv", ("state", "model")), ("visibility-audit.csv", ("state", "purpose")),
                         ("mapping-fd.csv", ("state", "lamp", "axis")), ("switch-log.csv", ("arm", "reason")),
                         ("displacement-sweep.csv", ("scene", "scenario", "requested_cm"))):
        selected = [dict(stage=path.name, **r) for path in stages for r in rows(path / name)]
        table(dest / name, selected, header=("stage", *header))
    summary = stage_selection(output)
    summary["stages"] = [dict(name=path.name, arms=len(read(path / "summary.json")["arms"]),
                              truth=read(path / "summary.json")["truth"], scene=read(path / "summary.json")["scene"],
                              seconds=read(path / "summary.json")["seconds"], provenance=read(path / "provenance.json")) for path in stages]
    by_radius = defaultdict(list)
    for trial in data:
        if not trial["stage"].startswith(("e24-e-", "e24-e50-")):
            continue
        match = re.fullmatch(r"dir(\d+)_(\d+)cm", trial["scenario"])
        if match:
            by_radius[(trial["scene"], int(match[2]), trial["model"])].append((int(match[1]), trial))
    summary["displacement_rollup"] = [dict(scene=scene, cm=cm, model=model, seeds=[seed for seed, _ in paired],
                   passed=sum(t["pass"] == "1" for _, t in paired), count=len(paired),
                   max_position_cm=[float(t["max_position_cm"]) for _, t in paired],
                   max_intensity_relative=[float(t["max_intensity_relative"]) for _, t in paired],
                   held_raw=[float(t["final_hard_held_raw"]) for _, t in paired],
                   held_log=[float(t["final_hard_held_log"]) for _, t in paired],
                   held_display=[float(t["final_hard_held_display"]) for _, t in paired])
                   for (scene, cm, model), paired in sorted(by_radius.items())]
    summary["reused_small_radius"] = {"commit": "a89669d68ad6ce3927f7bb1e7bbe250e2100de9e",
           "historical_results_separate_from_local_rates": True, "2_5_10cm": [
               {k: a[k] for k in ("scenario", "method", "pass", "positionCm", "intensityRelative", "held", "evaluations")}
               for ref in ("near", "best") for a in read(output / "inputs" / f"{ref}.json")["arms"]
               if "position_only" not in a["scenario"] and (ref == "best" or a["method"] == "C0")]}
    summary["gpu_timing"] = "Unavailable for individual synchronous editor Camera.Render calls; no derived GPU timing."
    write(dest / "stage-summary.json", summary)
    with zipfile.ZipFile(dest / "raw-data.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in stages:
            for p in sorted(path.glob("*")):
                if p.is_file() and p.suffix in {".csv", ".json"}:
                    archive.write(p, p.relative_to(output).as_posix())
    figs = dest / "figures"
    figs.mkdir(exist_ok=True)
    for stage in stages:
        masks = sorted((stage / "captures" / "visibility").glob("*.png"))
        if not masks:
            masks = sorted(stage.rglob("visibility/*.png"))
        if masks:
            p = masks[0]
            shutil.copyfile(p, figs / (stage.name + "-" + p.name))
    for stage_name, arm, prefix, label in (
        ("e24_p_full", "seed131_near_on_H2", "initial_v0", "seed131-initial"),
        ("e24_p_full", "seed131_near_on_H2", "best_v0", "seed131-H2"),
        ("e24_p_full", "seed131_near_on_C4", "best_v0", "seed131-C4"),
    ):
        found = sorted((output / stage_name).rglob(f"{arm}/{prefix}.png"))
        if found:
            shutil.copyfile(found[0], figs / (label + ".png"))
    for key, candidates in (("complex", "seed509_complex_far_A-log"), ("classroom", "seed509_classroom_near_A-log")):
        stage = output / f"e24_a_{key}"
        for directory, prefixes in (("gt_on", ("gt_v0",)), ("gt_off", ("gt_v0",)), (candidates, ("initial_v0", "best_v0"))):
            for prefix in prefixes:
                found = sorted(stage.rglob(f"{directory}/{prefix}.png"))
                if found:
                    shutil.copyfile(found[0], figs / f"{key}-{directory}-{prefix}.png")
    return summary

"""Audit the native E4–E6 render ledger and flatten a complete run to trials.csv."""

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from statistics import median
from zipfile import ZIP_DEFLATED, ZipFile


ORIGINAL = (17, 31, 47, 73, 109)
HOLDOUT = (131, 157, 181, 211, 241)
LBFGS, LM = "CentralLbfgs", "ImageJacobianLm"
E4 = {"e4_e1_lit": (4, 500, "e1"), "e4_e2_lit": (12, 2000, "e2_lit_fixed"),
      "e4_e2_stylized": (12, 2000, "e2_stylized_fixed")}
VALIDATION = ("validate_best_256", "validate_initial_512", "validate_best_512")


def check(condition, where, message):
    if not condition:
        raise ValueError(f"{where}: {message}")


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def near(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)


def ledger(path, count, dimensions=None):
    rows = csv_rows(path)
    check(len(rows) == count, path, f"expected {count} camera renders, got {len(rows)}")
    for i, q in enumerate(rows, 1):
        check(int(q["camera_renders"]) == 1 and int(q["cumulative_camera_renders"]) == i,
              path, f"row {i} is not one incremental camera render")
        q["theta"] = [float(t) for t in q["theta"].split(";")] if q["theta"] else []
        q["loss"] = float(q["loss"]) if q["loss"] else None
        check(q["loss"] is None or math.isfinite(q["loss"]) and q["loss"] >= 0, path, f"invalid loss at row {i}")
        check(all(0 <= t <= 1 for t in q["theta"]) and
              (dimensions is None or len(q["theta"]) == dimensions), path, f"infeasible theta at row {i}")
    return rows


def verify_trial(path, r, dimensions, budget, check_images):
    training, validation = r["trainingCameraRenders"], r["validationCameraRenders"]
    check(load(path / "report.json") == r, path, "per-trial and stage records differ")
    check(r["cameraBudget"] == budget and r["dimensions"] == dimensions and 1 <= training <= budget,
          path, "unexpected dimensions or budget")
    check((r["fitResolution"], r["validationResolution"], validation) == (256, 512, 3), path, "unexpected validation design")
    queries = ledger(path / "queries.csv", training + validation, dimensions)
    fit, verify = queries[:training], queries[training:]
    check(all(q["method"] == r["method"] and int(q["seed"]) == r["seed"] and q["loss"] is not None for q in queries),
          path, "missing loss or misattributed query")
    check(tuple(q["phase"] for q in verify) == VALIDATION and not any(q["phase"].startswith("validate_") for q in fit),
          path, "validation queries must follow training")
    nscreen = r["screenedRenders"]
    screened = r["strategy"] in {"multistart", "screen48"}
    expected = ["screen_anchor"] + ["screen_candidate"] * 48 if screened else []
    check(nscreen == len(expected) and [q["phase"] for q in fit if q["phase"].startswith("screen_")] == expected
          and [q["phase"] for q in fit[:nscreen]] == expected and r["localRenders"] == training - nscreen,
          path, "screening is not fully charged to training")
    if screened:
        check(fit[nscreen]["phase"] in {"refine_initial", "short0_initial"}, path, "local solve did not follow screening")
        if r["strategy"] == "screen48":
            check(all(q["theta"][3] == .5 for q in fit[:nscreen]), path, "cold-start screening uses variable power")
            check(fit[nscreen]["theta"] == min(fit[:nscreen], key=lambda q: q["loss"])["theta"], path, "wrong selected start")
        else:
            starts = [next(q for q in fit if q["phase"] == f"short{i}_initial") for i in range(3)]
            candidates = sorted(fit[1:nscreen], key=lambda q: q["loss"])[:2]
            check([q["theta"] for q in starts] == [q["theta"] for q in [fit[0], *candidates]], path, "wrong shortlist")
            check(all(sum(q["phase"].startswith(f"short{i}_") for q in fit) <= 300 for i in range(3)), path, "short solve exceeded cap")
            ix = next(i for i, q in enumerate(fit) if q["phase"] == "refine_initial")
            check(fit[ix]["theta"] == min(fit[:ix], key=lambda q: q["loss"])["theta"], path, "wrong final refinement start")
    check(fit[0]["theta"] == r["initialTheta"] and near(fit[0]["loss"], r["initialLoss"]), path, "initial state mismatch")
    best = min(fit, key=lambda q: q["loss"])
    check(best["theta"] == r["bestTheta"] and near(best["loss"], r["bestLoss"]), path, "final rig is not the best actual training query")
    for q, theta, loss in zip(verify, (r["bestTheta"], r["initialTheta"], r["bestTheta"]),
                              (r["recapturedLoss"], r["highInitialNrmse"] ** 2, r["highBestNrmse"] ** 2)):
        check(q["theta"] == theta and near(q["loss"], loss), path, "recapture state or metric mismatch")
    check(abs(r["recapturedLoss"] - r["bestLoss"]) <= max(1e-10, r["bestLoss"] * 1e-6), path, "best loss did not replay")
    check(near(r["initialNrmse"] ** 2, r["initialLoss"]) and near(r["bestNrmse"] ** 2, r["bestLoss"])
          and near(r["improvement"], 1 - r["bestLoss"] / max(1e-20, r["initialLoss"]))
          and r["passed"] == (r["improvement"] >= .9 and r["bestNrmse"] <= .05), path, "invalid metric or gate")
    for label in ("best_256", "initial_512", "best_512") if check_images else ():
        check(all((path / f"{label}.{suffix}").is_file() for suffix in ("exr", "png")), path, f"missing {label} capture")


def verify_stage(root, stage, dimensions, budget, seeds, methods, strategies, *, check_images):
    path = root / stage
    report = load(path / "report.json")
    rows = report["trials"]
    expected = {(s, m, a) for s in seeds for m in methods for a in strategies}
    check(report["stage"] == stage and len(rows) == len(expected)
          and {(r["seed"], r["method"], r["strategy"]) for r in rows} == expected, path, "missing or duplicate trials")
    shared = ledger(path / "shared_queries.csv", report["sharedCameraRenders"], dimensions)
    check([q["phase"] for q in shared] == ["warmup_256", "target_256", "warmup_512", "target_512", "repeat", "repeat_512", "coordinate_roundtrip"]
          and shared[4]["loss"] == report["repeatLoss"] <= 1e-10 and shared[5]["loss"] == report["repeatLoss512"] <= 1e-10
          and shared[6]["loss"] <= 1e-5, path, "failed shared renders or repeatability check")
    for size in (256, 512) if check_images else ():
        check(all((path / f"target_{size}.{suffix}").is_file() for suffix in ("exr", "png")), path, "missing target capture")
    anchors = {}
    for r in rows:
        trial = path / f"{r['seed']}_{r['method']}_{r['strategy']}"
        verify_trial(trial, r, dimensions, budget, check_images)
        check(r["stage"] == stage and r["sharedCameraRenders"] == len(shared)
              and r["targetTheta"] == shared[0]["theta"], trial, "shared target mismatch")
        anchor = anchors.setdefault(r["seed"], r)
        check(all(r[k] == anchor[k] for k in ("targetRig", "initialRig", "initialTheta", "initialLoss", "highInitialNrmse")),
              trial, "comparison is not paired")
    check(len(shared) + sum(r["trainingCameraRenders"] + r["validationCameraRenders"] for r in rows)
          == report["totalCameraRenders"], path, "stage ledger differs from backend counter")
    return report


def export(root, baseline, destination=None, *, check_images=True):
    def stage_report(*args):
        return verify_stage(root, *args, check_images=check_images)

    stages = [stage_report(stage, d, b, ORIGINAL, (LBFGS, LM), ("single",)) for stage, (d, b, _) in E4.items()]
    e4 = {m: [r for s in stages for r in s["trials"] if r["method"] == m] for m in (LBFGS, LM)}
    if baseline:
        old = {(r["stage"], int(r["seed"])): r for r in csv_rows(baseline) if r["method"] == LBFGS}
        for r in e4[LBFGS]:
            b = old[E4[r["stage"]][2], r["seed"]]
            check(all(near(float(b[k]), r[k]) for k in ("cameraBudget", "initialNrmse")),
                  baseline, f"published baseline differs at the start for {r['stage']}, seed {r['seed']}")
            if any(not near(float(b[k]), r[k]) for k in ("trainingCameraRenders", "bestNrmse", "highBestNrmse")):
                print(f"Historical trajectory differs for {r['stage']}, seed {r['seed']}: "
                      f"512x512 NRMSE {float(b['highBestNrmse']):.4%} vs paired rerun {r['highBestNrmse']:.4%}.")
    (pa, ma, wa), (pb, mb, wb) = [
        (sum(r["passed"] for r in rows), median(r["highBestNrmse"] for r in rows), max(r["highBestNrmse"] for r in rows))
        for rows in e4.values()]
    expected = LM if (pb > pa and wb <= 1.2 * wa or pb == pa and wb <= .8 * wa and mb <= 1.2 * ma) else LBFGS
    decision = load(root / "e4_decision.json")
    chosen = decision["preferred"]
    check(chosen == expected and all(near(decision[k], v) for k, v in zip(
        ("lbfgsPasses", "lbfgsMedian", "lbfgsWorst", "lmPasses", "lmMedian", "lmWorst"), (pa, ma, wa, pb, mb, wb))),
        root, "optimizer selection disagrees with E4")
    stages.append(stage_report("e5a_multistart", 12, 2000, ORIGINAL + HOLDOUT, (chosen,), ("single", "multistart")))
    cold = [stage_report(f"e5b_cold_gt{i}", 4, 500, (seed,), (chosen,), ("random", "screen48"))
            for i, seed in enumerate(HOLDOUT, 1)]
    check(len({tuple(s["trials"][0]["targetTheta"]) for s in cold}) == len(cold), root, "cold fixtures must be distinct")
    stages.extend(cold)
    stages.append(stage_report("e6_spot_direction", 2, 1000, ORIGINAL[:3], (chosen,), ("single",)))
    joint = stage_report("e6_spot_joint", 6, 1000, ORIGINAL[:3], (chosen,), ("single",))
    stages.append(joint)
    if all(r["passed"] for r in joint["trials"]):
        stages.append(stage_report("e6_spot_point_mixed", 10, 2000, (17,), (chosen,), ("single",)))
    diagnostic_renders = 0
    for name, count in (("e5a_seed47_diagnostic", 31), ("e6_spot_sanity", 10)):
        diag = root / name
        data = load(diag / "report.json")
        check(data["cameraRenders"] == count, diag, "unexpected diagnostics budget")
        queries = ledger(diag / "queries.csv", count)
        check(next(q for q in queries if q["phase"] == "repeat")["loss"] <= 1e-10, diag, "not repeatable")
        if name == "e6_spot_sanity":
            check(data["passed"] and near(data["awayPowerLoss"], queries[-1]["loss"]) and queries[-1]["phase"] == "away_half_power",
                  diag, "missing or failed Spot response check")
        diagnostic_renders += count
    rows = [r for stage in stages for r in stage["trials"]]
    fields = [k for k, v in rows[0].items() if not isinstance(v, list)]
    with (root / "trials.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if destination:
        destination.mkdir(parents=True, exist_ok=True)
        if root.resolve() != destination.resolve():
            shutil.copyfile(root / "trials.csv", destination / "trials.csv")
        paths = [root / "e4_decision.json"]
        for stage in stages:
            directory = root / stage["stage"]
            paths.extend(directory / name for name in ("report.json", "shared_queries.csv"))
            for r in stage["trials"]:
                trial = directory / f"{r['seed']}_{r['method']}_{r['strategy']}"
                paths.extend(trial / name for name in ("report.json", "queries.csv"))
        for name in ("e5a_seed47_diagnostic", "e6_spot_sanity"):
            paths.extend(root / name / file for file in ("report.json", "queries.csv"))
        with ZipFile(destination / "query-data.zip", "w", ZIP_DEFLATED) as bundle:
            for path in sorted(paths):
                bundle.write(path, path.relative_to(root).as_posix())
    training = sum(r["trainingCameraRenders"] for r in rows)
    validation = sum(r["validationCameraRenders"] for r in rows)
    shared = sum(s["sharedCameraRenders"] for s in stages)
    wall = sum(s["wallSeconds"] for s in stages)
    print(f"Audited {len(rows)} trials: {training} training + {validation} validation + {shared} shared + "
          f"{diagnostic_renders} diagnostic = {training + validation + shared + diagnostic_renders} native renders. "
          f"Timed stages (excluding diagnostics/editor startup): {wall:.2f}s. E4 selected {chosen}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="shared E4–E6 output folder")
    parser.add_argument("--baseline", type=Path, help="optional published E0–E3 trials.csv for historical E4 scalar cross-check")
    parser.add_argument("--publish", type=Path, help="copy trials.csv and zip per-query ledgers/JSON for review")
    parser.add_argument("--logs-only", action="store_true", help="audit an unpacked query archive without its captures")
    args = parser.parse_args()
    try:
        export(args.run, args.baseline, args.publish, check_images=not args.logs_only)
    except (ValueError, OSError, KeyError, StopIteration) as exc:
        parser.exit(1, f"Audit failed: {exc}\n")


if __name__ == "__main__":
    main()

"""Apply fixed comparisons to E22 stage output without rerunning an arm."""

import json
import math
import statistics

from blackbox_e20 import csv_read, json_read


def records(output, stage):
    folder = output / ("e22_" + stage.replace("-", "_"))
    meta = {v["arm"]: v for v in json_read(folder / "metrics.json")["metrics"]}
    for row in csv_read(folder / "trials.csv"):
        m = meta[row["arm"]]
        yield dict(row, display=m["displayHeld"], cameras=m["optimizationCameras"], optimization=m["optimizationSeconds"],
                   position=float(row["max_position_cm"]), intensity=float(row["max_intensity_relative"]),
                   raw=float(row["held_nrmse"]), passed=int(row["pass"]), olat=m["olatCameras"])


def matched(rows, candidate, baseline):
    references = {r["scenario"]: r for r in rows if r["method"] == baseline}
    return [(r, references[r["scenario"]]) for r in rows if r["method"] == candidate]


def gate(pairs):
    return {
        "count": len(pairs),
        "displayWins": sum(r["display"] < b["display"] for r, b in pairs),
        "displayWinsBeyondFloor": sum(r["display"] < b["display"] - 1e-5 for r, b in pairs),
        "passCandidate": sum(r["passed"] for r, _ in pairs),
        "passBaseline": sum(b["passed"] for _, b in pairs),
        "rawRegressionsOverTwoPoints": [r["scenario"] for r, b in pairs if r["raw"] > b["raw"] + .02],
        "parameterRegressions": [r["scenario"] for r, b in pairs
                                 if r["position"] - b["position"] > 2.5 or r["intensity"] - b["intensity"] > .025],
    }


def choose_solver(rows, dimension):
    evidence = {}
    promoted = []
    for key in sorted({r["method"] for r in rows} - {"A0"}):
        pairs = matched(rows, key, "A0")
        stats = gate(pairs)
        camera_ratio = sum(r["cameras"] for r, _ in pairs) / sum(b["cameras"] for _, b in pairs)
        # Equal-quality 8D solutions advance with at least 15% fewer aggregate renders.
        # On the 28D migration, a pass gain or less physical error with a display win also counts.
        quality = all(r["passed"] >= b["passed"] for r, b in pairs) and not stats["rawRegressionsOverTwoPoints"] and not stats["parameterRegressions"]
        fewer = camera_ratio <= .85 and all(r["display"] <= b["display"] + .01 for r, b in pairs)
        recovery = stats["passCandidate"] > stats["passBaseline"] or (dimension == 28 and
                    all(r["display"] < b["display"] - 1e-5 and r["position"] < b["position"] and r["intensity"] < b["intensity"]
                        for r, b in pairs))
        evidence[key] = dict(stats, cameraRatio=camera_ratio, eligible=quality and (fewer or recovery))
        if evidence[key]["eligible"]:
            promoted.append(key)
    selected = min(promoted, key=lambda k: (-evidence[k]["passCandidate"], evidence[k]["cameraRatio"])) if promoted else "A0"
    return selected, evidence


def choose_mapping(rows):
    evidence = {}
    for key in sorted({r["method"] for r in rows} - {"B0"}):
        pairs = matched(rows, key, "B0")
        stats = gate(pairs)
        evidence[key] = dict(stats, displayRatio=statistics.mean(r["display"] / max(1e-9, b["display"]) for r, b in pairs),
                             scaledError=statistics.mean(r["position"] / 5 + r["intensity"] / .05 for r, _ in pairs))
    selected = min(evidence, key=lambda k: (-evidence[k]["passCandidate"], -evidence[k]["displayWins"], evidence[k]["displayRatio"], evidence[k]["scaledError"], k))
    return selected, evidence


def select(args):
    data = list(records(args.output, args.phase))
    if args.phase in ("a8", "a28"):
        selected, evidence = choose_solver(data, 8 if args.phase == "a8" else 28)
    else:
        selected, evidence = choose_mapping(data)
    (args.output / f"decision-{args.phase}.json").write_text(json.dumps({"selected": selected, "evidence": evidence}, indent=2) + "\n", encoding="utf-8")
    print(selected)


def qualification(stats, wins):
    return (stats["displayWins"] >= wins and stats["passCandidate"] >= stats["passBaseline"]
            and not stats["rawRegressionsOverTwoPoints"] and not stats["parameterRegressions"])


def decisions(output):
    def path(stage, name="summary.json"):
        return output / ("e22_" + stage.replace("-", "_")) / name

    selected = json_read(output / "decision-b-screen.json")["selected"]
    results = {"selectedMapping": selected}
    tone = {}
    for stage in ("b-off", "b-on", "b-confirm", "b-far"):
        if not path(stage).is_file():
            continue
        pairs = matched(list(records(output, stage)), selected, "B0")
        stats = gate(pairs)
        tone[stage] = dict(stats, qualifies=qualification(stats, min(2, len(pairs))))
    results["tone"] = tone
    if "b-off" in tone and "b-on" in tone:
        results["far"] = tone["b-on"]["qualifies"] and tone["b-on"]["displayWinsBeyondFloor"] > 0
        results["shadowCalibration"] = (tone["b-on"]["passCandidate"] < tone["b-off"]["passCandidate"] or
                                        tone["b-on"]["passBaseline"] < tone["b-off"]["passBaseline"])
        tradeoffs = []
        for stage in ("b-off", "b-on"):
            for r, b in matched(list(records(output, stage)), selected, "B0"):
                if r["display"] < b["display"] - 1e-5 and (r["raw"] > b["raw"] + .02 or not r["passed"]):
                    tradeoffs.append((r["display"] / max(1e-9, b["display"]), stage, r["scenario"]))
        if tradeoffs:
            _, stage, case = min(tradeoffs)
            results["scheduledLoss"] = {"seed": int(case.split("_")[0][4:]), "shadows": "on" if stage == "b-on" else "off"}
        else:
            results["scheduledLoss"] = None
    if path("c").is_file():
        references = [r for r in records(output, "b-off") if r["method"] == selected]
        variants = list(records(output, "c"))
        pairs = matched(references + variants, "C1", selected)
        stats = gate(pairs)
        results["geometry"] = dict(stats, extend=qualification(stats, 2) and stats["displayWinsBeyondFloor"] == 2)
    if path("d-audit", "olat-gate.json").is_file():
        results["olatLinearity"] = json_read(path("d-audit", "olat-gate.json"))["states"]
    for stage in ("d-8d", "d-8d-on", "d-28d", "d-28d-on"):
        if not path(stage).is_file():
            continue
        pairs = matched(list(records(output, stage)), "D1", "D0")
        if pairs:
            r, b = pairs[0]
            stats = gate(pairs)
            camera_ratio = (r["cameras"] + r["olat"] + 3) / b["cameras"]  # One shared empty-scene capture per training view.
            saves = (r["passed"] >= b["passed"] and not stats["parameterRegressions"] and
                     not stats["rawRegressionsOverTwoPoints"] and (camera_ratio <= .9 or r["passed"] > b["passed"]))
            results[stage] = dict(stats, cameraRatio=camera_ratio, savingsGate=saves)
            if "8d" in stage:
                results[stage]["extend"] = saves
        else:
            results[stage] = {"extend": False, "reason": "column_validation"}
    if path("e-calibration", "shadow-calibration.csv").is_file():
        rows = csv_read(path("e-calibration", "shadow-calibration.csv"))
        directions = csv_read(path("e-calibration", "shadow-direction.csv"))
        calibration = {}
        for method in ("msm", "mbd"):
            legacy, shared = [], []
            for key in ("visibility_mae", "weighted_visibility_mae", "direct_nrmse"):
                def values(prefix):
                    return [float(r[key]) for r in rows if r["method"] == prefix + method and r["pose"] != "gt"]
                legacy.append(statistics.median(values("e21_")))
                shared.append(statistics.median(values("matched_")))
            ratios = [y / max(1e-12, x) for x, y in zip(legacy, shared)]
            def cosine(prefix):
                values = [float(r["direct_cosine"]) for r in directions if r["method"] == prefix + method]
                return statistics.median([v for v in values if math.isfinite(v)] or [-1])
            before, after = cosine("e21_"), cosine("matched_")
            calibration[method] = {"oldMedian": legacy, "matchedMedian": shared, "ratio": ratios, "directCosineOld": before,
                                   "directCosineMatched": after, "continueRecovery": ratios[1] <= .9 and ratios[2] <= .9 and after >= before - .05}
        results["proxy"] = calibration
    return results


def qualify(args):
    evidence = decisions(args.output)
    (args.output / "stage-decisions.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))

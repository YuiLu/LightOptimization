"""Select E23 stages from measured recovery and hard-shadow replays."""

import json
import statistics

from blackbox_e20 import csv_read, json_read


def folder(root, name):
    return root / ("e23_" + name.replace("-", "_"))


def trials(root, name):
    path = folder(root, name)
    arms = csv_read(path / "trials.csv")
    metrics = {r["arm"]: r for r in json_read(path / "metrics.json")["metrics"]}
    scheme = {r["arm"]: r for r in csv_read(path / "arm-config.csv")}
    for arm in arms:
        yield dict(arm, metrics=metrics[arm["arm"]], setup=scheme[arm["arm"]])


def tone_screen(root):
    arms = list(trials(root, "c-screen"))
    if (folder(root, "c-mix") / "summary.json").is_file():
        arms += list(trials(root, "c-mix"))
    methods = sorted({r["method"] for r in arms})
    qualified = {}
    for m in methods:
        batch = [r for r in arms if r["method"] == m]
        qualified[m] = dict(passed=sum(int(r["pass"]) for r in batch),
                            displayTrain=statistics.mean(r["metrics"]["displayTrain"] for r in batch),
                            parameter=statistics.mean(float(r["max_position_cm"]) / 5 + float(r["max_intensity_relative"]) / .05 for r in batch),
                            cameras=sum(int(r["setup"]["original_cameras"]) for r in batch))
    candidates = set(methods) - {"C0", "C1"}
    selected = min(candidates, key=lambda k:(-qualified[k]["passed"], qualified[k]["displayTrain"], qualified[k]["parameter"], qualified[k]["cameras"], k))
    return {"selected": selected, "mixtureRequired": max(qualified[k]["passed"] for k in qualified if k != "C0") < 2 and "C5" not in methods,
            "criteria": "successes, training display NRMSE, scaled parameter error, optimization cameras", "scores": qualified}


def directions(root):
    rows = csv_read(folder(root, "a") / "direction-probes.csv")
    batches = {}
    for r in rows:
        if r["failed"] != "1" or r["status"] != "measured":
            continue
        key = (r["pose"], r["mapping"], r["model"], r["length"])
        batches.setdefault(key, []).append(r)
    best = {}
    for (pose, mapping, model, _), group in batches.items():
        if len(group) != 2 or not all(r["decision"] == "1" for r in group):
            continue
        value = min(float(r["actual"]) / max(1e-12, float(r["base_loss"])) for r in group)
        key = pose, mapping, model
        best[key] = max(best.get(key, 0), value)
    evidence = []
    poses = sorted({key[0] for key in batches})
    for mapping in ("C0", "C1", "C3"):
        for model in ("B2", "B3"):
            improvements = {}
            for pose in poses:
                baseline, measured = best.get((pose,mapping,"B0"), 0), best.get((pose,mapping,model), 0)
                if measured > baseline + max(.001, .05 * baseline):
                    improvements[pose] = {"baselineReductionFraction": baseline, "candidateReductionFraction": measured}
            evidence.append(dict(mapping=mapping, model=model, improvedStates=len(improvements), states=improvements))
    eligible = [e for e in evidence if e["improvedStates"] >= 2]
    # A passing forward map remains the main objective. Prefer O(1) native captures for a tie.
    winner = max(eligible, key=lambda e:(e["improvedStates"], e["model"]=="B2", sum(x["candidateReductionFraction"] for x in e["states"].values()))) if eligible else None
    return {"selected":winner,"evidence":evidence,"criterion":"two failed states, repeated rho >= .02, improvement above 0.1% of initial loss and 5% of the baseline reduction"}


def tone_validate(root):
    rows = list(trials(root, "c-validate"))
    raw = {r["scenario"]: r for r in rows if r["method"] == "C0"}
    evidence = {}
    for method in sorted({r["method"] for r in rows} - {"C0"}):
        group = [r for r in rows if r["method"]==method]
        gained = [r["scenario"] for r in group if r["pass"]=="1" and raw[r["scenario"]]["pass"]=="0"]
        lost = [r["scenario"] for r in group if r["pass"]=="0" and raw[r["scenario"]]["pass"]=="1"]
        evidence[method] = dict(gained=gained,lost=lost,clearBenefit=bool(gained) and not lost)
    return evidence


def hybrid_small(root):
    rows = list(trials(root, "b-small"))
    baseline = {r["scenario"]: r for r in rows if r["method"] == "B0"}
    exact_baseline = all(float(r["max_position_cm"]) <= 1 and float(r["max_intensity_relative"]) <= .01 for r in baseline.values())
    evidence = {}
    for model in ("B1", "B2", "B3"):
        group = [r for r in rows if r["method"] == model]
        gained = [r["scenario"] for r in group if r["pass"] == "1" and baseline[r["scenario"]]["pass"] == "0"]
        lost = [r["scenario"] for r in group if r["pass"] == "0" and baseline[r["scenario"]]["pass"] == "1"]
        evidence[model] = dict(gained=gained, lost=lost, passed=sum(r["pass"] == "1" for r in group))
    advantage = any(evidence[model]["gained"] and not evidence[model]["lost"] for model in ("B2", "B3"))
    saturated = exact_baseline and any(evidence[model]["passed"] == len(baseline) for model in ("B2", "B3"))
    return {"expand": advantage or saturated, "saturated": saturated, "evidence": evidence,
            "criterion": "hybrid gains a recovery without losses, or both baseline cases reach 1cm/1% and a hybrid also passes both"}


def hybrid_full(root):
    rows = list(trials(root, "b-full"))
    raw = {r["scenario"]: r for r in trials(root, "c-validate") if r["method"] == "C0"}
    baseline = {r["scenario"]: r for r in rows if r["method"] == "B0"}
    results = {}
    for model in ("B1", "B2", "B3"):
        group = [r for r in rows if r["method"] == model]
        gained = [r["scenario"] for r in group if r["pass"] == "1" and raw[r["scenario"]]["pass"] == "0"]
        lost = [r["scenario"] for r in group if r["pass"] == "0" and any(b[r["scenario"]]["pass"] == "1" for b in (raw, baseline))]
        same_mapping_gain = [r["scenario"] for r in group if r["pass"] == "1" and baseline[r["scenario"]]["pass"] == "0"]
        held = {r["scenario"]: {"displayDelta": r["metrics"]["displayHeld"] - raw[r["scenario"]]["metrics"]["displayHeld"],
                                "rawDelta": float(r["held_nrmse"]) - float(raw[r["scenario"]]["held_nrmse"])} for r in group}
        results[model] = dict(gained=gained, lost=lost, sameMappingGain=same_mapping_gain, heldDeltaFromRaw=held)
    eligible = [model for model in ("B2", "B3") if results[model]["gained"] and results[model]["sameMappingGain"] and not results[model]["lost"]]
    selected = max(eligible, key=lambda model: (len(results[model]["gained"]), model == "B2")) if eligible else None
    return {"selected": selected, "evidence": results, "criterion": "recover an original Raw failure and improve on same-mapping FD, retaining original successes"}


def neighbour_validate(root):
    raw = {r["scenario"]: r for r in trials(root, "f-near") if r["method"] == "C0"}
    best = {r["scenario"]: r for r in trials(root, "f-best")}
    if raw.keys() != best.keys():
        raise ValueError("The neighbourhood recovery matrices differ")
    lost = [key for key in raw if raw[key]["pass"] == "1" and best[key]["pass"] == "0"]
    passed = sum(r["pass"] == "1" for r in best.values())
    return {"expand": not lost, "passed": passed, "total": len(best), "lost": lost,
            "criterion": "preserve the same Raw recovery successes at 2/5/10 cm and on the pure-position case"}


ASSESSMENTS = {"screen": tone_screen, "directions": directions, "tone": tone_validate, "hybrid-small": hybrid_small,
               "hybrid-full": hybrid_full, "neighbourhood": neighbour_validate}


def assess(args):
    data = ASSESSMENTS[args.phase](args.output)
    (args.output / f"decision-{args.phase}.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2))

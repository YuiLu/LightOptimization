"""Audit the native E7/E8 query ledgers and export paired results (Python 3.10+).

With --plots, Pillow builds a contact sheet from the native recaptures. The audit
also works on the unpacked CSV/JSON archive without any image files.
"""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def check(ok, message):
    if not ok:
        raise ValueError(message)


def close(a, b, atol=1e-10):
    return math.isclose(a, b, rel_tol=1e-8, abs_tol=atol)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ledger(path, expected):
    rows = read_csv(path)
    check(len(rows) == expected, f"Wrong query count in {path}")
    for i, row in enumerate(rows, 1):
        check(row["camera_renders"] == "1" and int(row["cumulative_camera_renders"]) == i, f"Render ledger gap: {path}:{i}")
        row["query"] = i
        row["loss"] = float(row["loss"]) if row["loss"] else None
        check(row["loss"] is None or (math.isfinite(row["loss"]) and row["loss"] >= 0), f"Invalid loss: {path}:{i}")
        row["theta"] = tuple(map(float, row["theta"].split(";"))) if row["theta"] else ()
        check(len(row["theta"]) in (0, 4, 8, 12) and all(0 <= p <= 1 for p in row["theta"]), f"Invalid theta: {path}:{i}")
    return rows


def rank_corr(a, b):
    def rank(values):
        order = sorted(range(len(values)), key=lambda k: values[k], reverse=True)
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and values[order[j]] == values[order[i]]:
                j += 1
            for k in order[i:j]:
                ranks[k] = (i + j - 1) / 2
            i = j
        return ranks
    a, b = rank(a), rank(b)
    return statistics.correlation(a, b) if len(set(a)) > 1 and len(set(b)) > 1 else 0.0


def audit_e7(root):
    report = read_json(root / "e7.json")
    check(report["schema"] == 1 and report["resolution"] == 256, "Unexpected E7 protocol")
    check([m["shader"] for m in report["materials"]] == ["Lit", "Stylized"], "Expected both materials")
    for m in report["materials"]:
        shader = m["shader"]
        rows = ledger(root / f"e7_{shader}_queries.csv", m["cameraRenders"])
        check(len(rows) == 287 and m["emptyRepeatLoss"] <= 1e-10, f"Wrong E7 render count or repeatability: {shader}")
        errors = read_csv(root / f"e7_{shader}_error.csv")
        gate = True
        for test, count in (("intensity", 24), ("additivity", 8)):
            tests = [r for r in errors if r["test"] == test]
            valid = []
            for r in tests:
                norm, err = map(float, (r["real_increment_l2"], r["prediction_delta_l2"]))
                check((norm <= 1e-6) == (r["near_zero"] == "1"), f"Near-zero mismatch: {shader}/{test}")
                if r["near_zero"] == "0":
                    rel = float(r["relative_l2_error"])
                    check(close(rel, err / norm), f"Relative error mismatch: {shader}/{test}")
                    valid.append(rel)
            s = m[test]
            check(len(tests) == s["samples"] == count and len(valid) == s["nonzero"] and count - len(valid) == s["nearZero"],
                  f"Wrong E7 samples: {shader}/{test}")
            med, worst = (statistics.median(valid), max(valid)) if valid else (-1, -1)
            check(close(med, s["median"]) and close(worst, s["worst"]), f"Error summary mismatch: {shader}/{test}")
            gate &= bool(valid) and med <= .02 and worst <= .05
        rankings = read_csv(root / f"e7_{shader}_ranking.csv")
        check(len(rankings) == 72 and [r["contextLights"] for r in m["rankings"]] == [0, 1, 2], f"Ranking layout: {shader}")
        for r in m["rankings"]:
            part = [x for x in rankings if int(x["context_lights"]) == r["contextLights"]]
            check([int(x["candidate"]) for x in part] == list(range(24)) and r["sampleCount"] == 24, f"Ranking pool: {shader}")
            predicted = [float(x["predicted_gain"]) for x in part]
            measured = [float(x["measured_gain"]) for x in part]
            top = sorted(range(len(part)), key=lambda i: -predicted[i])[:3]
            winner = max(range(len(part)), key=lambda i: measured[i])
            regret = (measured[winner] - max(measured[i] for i in top)) / max(1e-12, abs(measured[winner]))
            rho = rank_corr(predicted, measured)
            check((r["predictedTop"], r["measuredTop"], r["measuredTopInPredictedTop3"]) == (top[0], winner, winner in top)
                  and close(regret, r["top3Regret"]) and close(rho, r["spearman"]), f"Ranking mismatch: {shader}")
            gate &= rho >= .90 and regret <= .01
        check(gate == m["supportsOlat"], f"Incorrect OLAT decision: {shader}")
    return report


def audit_e8(root, e7, smoke=False):
    summary = read_json(root / "e8.json")
    budget = summary["budget"]
    check(budget in (2000, 4000) and summary["trainingResolution"] == 256 and summary["recaptureResolution"] == 512, "Bad E8 protocol")
    check(summary["trainingCamera"] == e7["camera"] != summary["heldOutCamera"], "Camera protocol changed or no held-out camera")
    check(summary["screening"] == ("OLAT" if e7["materials"][0]["supportsOlat"] else "native_grid"), "E8 screening did not follow Lit E7")
    expected_targets = ["spread", "clustered", "unbalanced", "occluded", "boundary"][:1 if smoke else 5]
    seeds = [131, 157, 181][:1 if smoke else 3]
    check(summary["targets"] == len(expected_targets) and summary["seedsPerTarget"] == len(seeds)
          and [g["target"] for g in summary["cases"]] == expected_targets, "Missing target or seed cohort")
    results = []
    shared_count = 0
    for g in summary["cases"]:
        target = g["target"]
        folder = root / target
        check(g == read_json(folder / "target.json"), f"Target checkpoint mismatch: {target}")
        shared = ledger(folder / "shared_queries.csv", g["sharedRenders"])
        shared_count += len(shared)
        check([q["phase"] for q in shared] == [f"{v}_{s}" for v in ("train", "high", "hold") for s in ("warmup", "gt", "repeat")]
              and all(list(q["theta"]) == g["targetTheta"] for q in shared), f"Unexpected shared queries: {target}")
        check([(r["seed"], r["method"]) for r in g["trials"]] == [(s, m) for s in seeds for m in "ABCD"], f"Missing paired trial: {target}")
        by_pair = {}
        for r in g["trials"]:
            s, m = r["seed"], r["method"]
            name = f"{target}/{s}_{m}"
            path = root / name
            check(r == read_json(path / "trial.json"), f"Trial checkpoint mismatch: {name}")
            n = r["trainingRenders"]
            q = ledger(path / "queries.csv", n)
            v = ledger(path / "validation.csv", r["validationRenders"])
            check(r["budget"] == budget and n <= budget and q[0]["phase"] == "empty" and q[0]["theta"] == ()
                  and close(q[0]["loss"], r["emptyLoss"]), f"No paired empty reference / over budget: {name}")
            check((r["emptyLoss"] <= .05 ** 2) == r["emptyBelowFivePercent"], f"Empty gate mismatch: {name}")
            check(len(r["bestTheta"]) == len(r["bestRig"]) * 4 and r["structureFailure"] == (len(r["bestTheta"]) != 12), f"Invalid final structure: {name}")
            check(close(math.sqrt(r["loss"]), r["trainNrmse"]) and close(1 - r["loss"] / max(1e-20, r["emptyLoss"]), r["reduction"]), f"Mismatched metrics: {name}")
            below5 = r["trainNrmse"] <= .05 and not r["structureFailure"]
            passed = below5 and r["reduction"] >= .9 and not r["emptyBelowFivePercent"]
            check((r["passesFivePercent"], r["passed"]) == (below5, passed), f"Invalid pass flag: {name}")
            if not r["structureFailure"]:
                complete = [x for x in q if len(x["theta"]) == 12]
                check(complete[0]["query"] == r["fullRigQueryStart"] and
                      close(min(x["loss"] for x in complete), r["loss"]) and
                      any(list(x["theta"]) == r["bestTheta"] and close(x["loss"], r["loss"]) for x in complete), f"Best rig was not the best real full query: {name}")
                five = [x for x in complete if x["loss"] <= .05 ** 2]
                eligible = [x for x in five if not r["emptyBelowFivePercent"] and 1 - x["loss"] / max(1e-20, r["emptyLoss"]) >= .9]
                check(r["firstFivePercentQuery"] == next((x["query"] for x in five), -1) and
                      r["firstPassQuery"] == next((x["query"] for x in eligible), -1), f"First-pass query mismatch: {name}")
            else:
                check(r["firstFivePercentQuery"] == r["firstPassQuery"] == -1 and not r["insertions"][-1]["accepted"], f"Unexpected partial-rig status: {name}")
            validation = {x["phase"]: x for x in v}
            checks = {"train_best": r["repeatLoss"], "high_initial": r["initialHighNrmse"] ** 2,
                      "high_best": r["highNrmse"] ** 2, "heldout_best": r["holdNrmse"] ** 2}
            check(len(v) == 4 + (m in "CD") + 2 * len(r["swaps"]) and close(r["repeatLoss"], r["loss"], 1e-9), f"Invalid recapture count or result: {name}")
            check(all(close(validation[k]["loss"], val) for k, val in checks.items()), f"Recapture metrics disagree: {name}")
            check(all(list(validation[k]["theta"]) == r["bestTheta"] for k in ("train_best", "high_best", "heldout_best")), f"Wrong recaptured rig: {name}")
            swaps = r["swaps"]
            check(r["swapRenders"] == sum(t["renders"] for t in swaps) <= budget // 4 and (not swaps or (m == "D" and r["commonStop"] not in ("budget", "near_zero_residual"))),
                  f"Illegal Swap allocation: {name}")
            for i, t in enumerate(swaps, 1):
                part = q[t["firstQuery"] - 1:t["lastQuery"]]
                check(len(part) == t["renders"] and all(x["phase"].startswith(f"swap{i}_") for x in part) and len(part[0]["theta"]) == 8
                      and close(part[0]["loss"], t["removedLoss"]), f"Swap ledger mismatch: {name}/{i}")
                check(t["accepted"] == (t["afterLoss"] < t["beforeLoss"]) and
                      all(list(validation[k]["theta"]) == t[field] for k, field in ((f"swap{i}_before", "beforeTheta"), (f"swap{i}_proposal", "afterTheta"))),
                      f"Invalid Swap acceptance or recapture: {name}/{i}")
            prefix = [(x["phase"], x["theta"], x["loss"]) for x in q[:r["commonRenders"] or r["insertionRenders"]]]
            if m == "A":
                by_pair[s] = {"anchor": q[1]["theta"], "empty": r["emptyLoss"]}
            if m == "B":
                check(q[1]["theta"] == by_pair[s]["anchor"] and sum(x["phase"].startswith("rig_screen_") for x in q) == 48, f"Unpaired or short Multi-start: {name}")
            if m in "CD":
                olat = summary["screening"] == "OLAT"
                check(r["insertionRenders"] <= 1024 and r["screening"] == summary["screening"] and
                      sum(x["phase"].startswith("dict_") for x in q) == (64 if olat else 0), f"Invalid insertion allocation: {name}")
                for i, t in enumerate(r["insertions"], 1):
                    part = [x for x in q if x["phase"].startswith(f"insert{i}_")]
                    check(len(part) == t["renders"] and len(part[0]["theta"]) == (i - 1) * 4 and
                          sum("_grid_" in x["phase"] for x in part) == (0 if olat else 192), f"Wrong insertion pool: {name}/{i}")
                if m == "C":
                    by_pair[s]["prefix"] = prefix
                else:
                    check(prefix == by_pair[s]["prefix"], f"C/D diverged before Swap policy: {name}")
            check(close(r["emptyLoss"], by_pair[s]["empty"]), f"Unpaired empty reference: {name}")
            results.append(r)
    check(summary["trainingRenders"] == sum(r["trainingRenders"] for r in results) and
          summary["recaptureRenders"] == sum(r["validationRenders"] for r in results) and summary["sharedRenders"] == shared_count,
          "E8 aggregate differs from the ledgers")
    return results


def export(results, path):
    fields = ["target", "seed", "method", "screening", "budget", "passesFivePercent", "passed", "emptyBelowFivePercent", "structureFailure",
              "emptyLoss", "initialHighNrmse", "trainNrmse", "highNrmse", "holdNrmse", "reduction", "firstFivePercentQuery", "firstPassQuery",
              "trainingRenders", "validationRenders", "insertionRenders", "commonRenders", "commonStop", "stopReason", "swapRenders",
              "swapAttempts", "swapsAccepted", "blockedProbes", "shortenedTrials", "trainingSeconds", "validationSeconds", "wallSeconds"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(dict(r, swapAttempts=len(r["swaps"]), swapsAccepted=sum(t["accepted"] for t in r["swaps"])))


def contact_sheet(results, source, path):
    from PIL import Image, ImageDraw, ImageFont

    pair = {(r["target"], r["seed"], r["method"]): r for r in results}
    cd = [(r, pair[r["target"], r["seed"], "D"]) for r in results if r["method"] == "C"]
    def percent(value):
        percent_error = 100 * value
        return f"{percent_error:.2f}%" if percent_error >= .01 or percent_error == 0 else f"{percent_error:.2g}%"

    def baseline_gain(p):
        c, _ = p
        return min(pair[c["target"], c["seed"], m]["highNrmse"] for m in "AB") - c["highNrmse"]

    sets = [("Insertion comparison", max(cd, key=baseline_gain)),
            ("Most difficult for C/D", max(cd, key=lambda p: min(p[0]["highNrmse"], p[1]["highNrmse"])))]
    swapped = [p for p in cd if p[1]["swaps"]]
    successful_swaps = [p for p in swapped if any(t["accepted"] for t in p[1]["swaps"])]
    if successful_swaps:
        sets.insert(1, ("Swap comparison", max(successful_swaps, key=lambda p: p[0]["highNrmse"] - p[1]["highNrmse"])))
    elif swapped:
        sets.insert(1, ("Farthest rejected Swap", max(swapped, key=lambda p: max(t["finalDistance"] for t in p[1]["swaps"]))))
    rows, shown = [], set()
    for role, (c, d) in sets:
        key = (c["target"], c["seed"])
        if key in shown:
            continue
        shown.add(key)
        gt = source / c["target"]

        def trial_file(r, name):
            return gt / f"{r['seed']}_{r['method']}" / f"{name}.png"

        a, b = [pair[c["target"], c["seed"], m] for m in "AB"]
        label = f"{c['target']} / seed {c['seed']}"
        rows.append((f"{role}: {label}", [("GT (512)", gt / "high_gt.png")] +
                     [(f"{r['method']} / {percent(r['highNrmse'])}", trial_file(r, "high_best")) for r in (a, b, c, d)]))
        if d["swaps"]:
            i, t = max(enumerate(d["swaps"], 1), key=lambda it: (
                it[1]["accepted"], it[1]["beforeLoss"] - it[1]["afterLoss"] if it[1]["accepted"] else it[1]["finalDistance"]))
            rows.append((f"{label} / Swap {i} {'accepted' if t['accepted'] else 'rejected'}",
                         [("After insertion", trial_file(d, "high_after_insertions")),
                          (f"Before / {percent(math.sqrt(t['beforeLoss']))} (train)", trial_file(d, f"swap{i}_before")),
                          (f"Trial / {percent(math.sqrt(t['afterLoss']))} (train)", trial_file(d, f"swap{i}_proposal")),
                          ("Held-out GT", gt / "hold_gt.png"), (f"D held-out / {percent(d['holdNrmse'])}", trial_file(d, "heldout_best"))]))
        else:
            n = len(d["bestRig"])
            rows.append((f"{label} / No Swap attempted",
                         [("Initial (empty)", trial_file(d, "high_initial")),
                          ("After insertion", trial_file(d, "high_after_insertions")),
                          (f"D final / {n} lamps", trial_file(d, "high_best")),
                          ("Held-out GT", gt / "hold_gt.png"), (f"D held-out / {percent(d['holdNrmse'])}", trial_file(d, "heldout_best"))]))
    thumb, margin, gap, strip = 208, 20, 10, 264
    image = Image.new("RGB", (2 * margin + 5 * thumb + 4 * gap, 62 + strip * len(rows)), "#111822")
    draw = ImageDraw.Draw(image)
    draw.text((margin, 19), "E8  |  Native Unity RGB recaptures", font=ImageFont.load_default(23), fill="#f1f4f8")
    for y, (title, items) in enumerate(rows):
        top = 62 + strip * y
        draw.text((margin, top), title, font=ImageFont.load_default(16), fill="#c2d7ef")
        for i, (label, file) in enumerate(items):
            x = margin + i * (thumb + gap)
            with Image.open(file) as capture:
                image.paste(capture.convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS), (x, top + 23))
            draw.text((x, top + 234), label, font=ImageFont.load_default(15), fill="#f1f4f8")
    image.save(path, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--budget", type=int, choices=(2000, 4000), default=2000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    source = args.root / f"e8_{'smoke_' if args.smoke else ''}{args.budget}"
    e7 = audit_e7(args.root)
    results = audit_e8(source, e7, args.smoke)
    dest = args.out or source
    dest.mkdir(parents=True, exist_ok=True)
    export(results, dest / "trials.csv")
    if args.plots:
        contact_sheet(results, source, dest / "native-comparison.png")
    if args.archive:
        with ZipFile(dest / "query-data.zip", "w", compression=ZIP_DEFLATED, compresslevel=6) as z:
            for p in sorted(list(args.root.glob("e7*.csv")) + [args.root / "e7.json"] + list(source.rglob("*.csv")) + list(source.rglob("*.json"))):
                z.write(p, p.relative_to(args.root))
    print(f"Audited {len(results)} E8 trials and E7 Lit/Stylized.")
    for m in "ABCD":
        v = [r for r in results if r["method"] == m]
        med = [100 * statistics.median(r[k] for r in v) for k in ("trainNrmse", "highNrmse", "holdNrmse")]
        print(f"{m}: {sum(r['passed'] for r in v)}/{len(v)} passed; median % train/high/held-out "
              f"{', '.join(f'{p:.3f}' for p in med)}; {sum(r['trainingRenders'] for r in v)} queries")


if __name__ == "__main__":
    main()

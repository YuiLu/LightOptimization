"""Audit and collect native E9–E11 results (standard library; Pillow for --plot)."""
from pathlib import Path
import argparse
import csv
import json
import math
import statistics
import zipfile


def check(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def ledger(path, views=None):
    total = 0
    with path.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            count = int(r["camera_renders"])
            total += count
            theta = tuple(map(float, r["theta"].split(";"))) if r["theta"] else ()
            loss = float(r["loss"]) if r["loss"] else None
            check(count > 0 and (views is None or views == count), f"bad camera increment: {path}: {r}")
            check(total == int(r["cumulative_camera_renders"]), f"bad camera sum: {path}: {r}")
            check(len(theta) % 4 == 0 and all(0 <= u <= 1 for u in theta), f"invalid normalized parameter: {path}: {r}")
            check(loss is None or math.isfinite(loss) and loss >= 0, f"invalid loss: {path}: {r}")
            yield total, r["phase"], theta, loss


def close(x, y, tol=1e-12):
    return math.isclose(x, y, abs_tol=tol, rel_tol=tol)


def meets(empty, loss):
    return empty > .05 ** 2 and loss <= .05 ** 2 and 1 - loss / empty >= .90


def trial_dir(stage, r):
    if r["stage"].startswith("e9"):
        return stage / r["target"] / ("main" if r["pool"] == 64 else "diagnostic256") / f'{r["seed"]}_{r["method"]}'
    return stage / f'{r["seed"]}_{r["method"]}_{r["views"]}v'


def audit_trial(path, r):
    queries, captures = list(ledger(path / "queries.csv", r["views"])), list(ledger(path / "recaptures.csv", 1))
    check(load(path / "trial.json") == r, f"stage/trial metadata differs: {path}")
    check(queries[-1][0] == r["trainingRenders"] <= r["budget"], f"training budget: {path}")
    check(captures[-1][0] == r["validationRenders"], f"recapture ledger: {path}")
    check(queries[0][1:3] == ("empty", ()), f"baseline: {path}")
    empty = queries[0][3]
    observed = [row for row in queries if row[3] is not None]
    best = min((q for q in observed if len(q[2]) == 4 * r["active"]), key=lambda q: q[3])
    check(best[0] == r["bestQuery"] and best[2] == tuple(r["bestTheta"]) and close(best[3], r["bestLoss"]), f"best actual query: {path}")
    train = [f for _, label, _, f in captures if label.startswith("train_best")]
    check(close(statistics.mean(train), best[3], 1e-9) and close(statistics.mean(train), r["repeatedLoss"], 1e-9), f"native recapture: {path}")
    check(r["active"] <= r["lamps"] and r["structureFailure"] == (r["active"] != r["lamps"]), f"active lights: {path}")
    reached = [q[0] for q in observed if len(q[2]) == 4 * r["lamps"] and meets(empty, q[3])]
    passed = bool(reached) and r["active"] == r["lamps"]
    check(r["passed"] == passed and r["firstPass"] == (reached[0] if passed else -1), f"first threshold crossing: {path}")
    check(sum(s["toQuery"] - s["fromQuery"] for s in r["steps"] if s["removedSlot"] >= 0) == r["swapRenders"], f"swap ledger: {path}")
    for s in r["steps"]:
        if s["removedSlot"] >= 0:
            check(s["afterLoss"] < s["beforeLoss"] if s["accepted"] else close(s["afterLoss"], s["beforeLoss"]), f"swap acceptance: {path}")
    return queries


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(root, dest, archive):
    dest.mkdir(parents=True, exist_ok=True)
    stages = sorted(p for p in root.glob("*/stage.json") if "smoke" not in p.parent.name)
    present = {p.parent.name for p in stages}
    for name in ("e10", "e11", "e11_near"):
        fallback = root / (name + "_smoke") / "stage.json"
        if name not in present and fallback.is_file():
            stages.append(fallback)
    trials, checkpoints = [], []
    stage_groups = {}
    for path in stages:
        stage = path.parent
        data = load(path)
        rows = data["trials"]
        by_key = {}
        for r in rows:
            queries = audit_trial(trial_dir(stage, r), r)
            by_key[r["target"], r["seed"], r["method"], r["pool"], r["views"]] = queries
            trials.append(dict(stage=stage.name, target=r["target"], method=r["method"], seed=r["seed"], pool=r["pool"], lights=r["lamps"],
                               active=r["active"], views=r["views"], passed=r["passed"], structure_failure=r["structureFailure"], stop=r["stop"],
                               initial_nrmse=math.sqrt(r["initialLoss"]), train_nrmse=math.sqrt(r["bestLoss"]), high_nrmse=r["highNrmse"],
                               held_nrmse=r["heldNrmse"], max_held_nrmse=max(r["heldViewNrmse"]), initial_held_nrmse=r["heldInitialNrmse"],
                               train_per_view=";".join(map(str, r["trainViewNrmse"])),
                               check_per_view=";".join(map(str, r["highViewNrmse"] + r["heldViewNrmse"])), reduction=r["reduction"],
                               first_pass=r["firstPass"], best_query=r["bestQuery"], queries=r["trainingRenders"], budget=r["budget"],
                               recaptures=r["validationRenders"], train_s=r["trainingSeconds"], wall_s=r["elapsedSeconds"],
                               submit_s=r["renderSubmitSeconds"], readback_s=r["readbackSeconds"], residual_s=r["residualSeconds"],
                               jacobian_s=r["jacobianSeconds"], jtj_s=r["normalSeconds"], solve_s=r["linearSolveSeconds"],
                               lm_buffer_bytes=r["lmBufferBytes"], process_peak_working_set=r["processPeakWorkingSet"], swap_renders=r["swapRenders"]))
            at = (64, 128, 256, 500, 1000, 1500, 2000, 4000, 6000) if r["stage"].startswith("e9") else (256, 512, 1000, 2000, 4000, 6000)
            for q in (q for q in at if q <= r["budget"]):
                prefix = [x for x in queries if x[0] <= q and x[3] is not None]
                best = min((x[3] for x in prefix), default=None)
                fixed = min((x[3] for x in prefix if len(x[2]) == 4 * r["lamps"]), default=None)
                checkpoints.append(dict(stage=stage.name, target=r["target"], method=r["method"], seed=r["seed"], pool=r["pool"],
                                        views=r["views"], checkpoint=q, best_any=best, best_fixed=fixed))
        if stage.name.startswith("e9"):
            for (target, seed, method, pool, views), qs in by_key.items():
                if method != "joint_swap":
                    continue
                b = by_key[target, seed, "joint", pool, views]
                c = [q for q in qs if not q[1].startswith("swap")]
                shared = 0
                for x, y in zip(b, qs):
                    if y[1].startswith("swap"):
                        break
                    check(x[:3] == y[:3] and (x[3] == y[3] or x[3] is not None and y[3] is not None and close(x[3], y[3])),
                          f"joint/shared prefix differs: {stage}/{target}/{seed}")
                    shared += 1
                check(shared > 0 and (len(c) != len(qs) or len(b) == len(qs)), f"unexpected non-Swap divergence: {target}/{seed}")
        if stage.name.startswith("e10"):
            gate = load(stage / "gate.json")
            scene = load(stage / "scene.json")
            check(gate["n"] == scene["lamps"] == len(scene["original"]) == len(scene["gt"]), f"authored coverage: {stage}")
            check(scene["maxRequestedFaces"] <= scene["theoreticalTiles"], f"atlas dimensions: {stage}")
            check(gate["stable"] == (max(gate["repeatNrmse"]) <= 1e-6 and max(gate["positionResponse"]) >= 1e-4 and max(gate["powerResponse"]) >= 1e-4), f"repeat gate: {stage}")
            check(gate["allLampsVisible"] == all(max(a["marginalNrmse"][:3]) >= 1e-3 for a in gate["observability"]), f"observability gate: {stage}")
            for prefix, n in (("one", 1), ("multi", 3)):
                errors = [gate[prefix + test] for test in ("Linearity", "Additivity")]
                olat = all(a and statistics.median(a) <= .02 and max(a) <= .05 for a in errors) and all(
                    a["spearman"] >= .9 and a["top3Miss"] <= .01 for a in gate["ranking"] if a["views"] == n)
                check(olat == gate[prefix + "Olat"], f"OLAT gate: {stage}/{n}v")
            for seed in {r["seed"] for r in rows}:
                a, b = sorted((r for r in rows if seed == r["seed"]), key=lambda r: r["views"])
                check((a["views"], b["views"]) == (1, 3) and a["budget"] == b["budget"] and a["initialTheta"] == b["initialTheta"], f"unpaired views: {stage}/{seed}")
        if stage.name.startswith("e11"):
            check(all(r["screening"] == ("none" if r["method"] == "multi_start" else "olat" if data["supportsOlat"] else "native_grid")
                      for r in rows), f"E11 screening disagrees with its OLAT gate: {stage}")
        shared = sum(list(ledger(p, 1))[-1][0] for p in stage.rglob("shared.csv"))
        check(shared == data["sharedRenders"], f"shared ledger: {stage}")
        check(sum(r["trainingRenders"] for r in rows) == data["trainingRenders"], f"stage training count: {stage}")
        check(sum(r["validationRenders"] for r in rows) == data["validationRenders"], f"stage recapture count: {stage}")
        stage_groups[stage.name] = {"training": data["trainingRenders"], "shared": shared, "recapture": data["validationRenders"], "arms": {}}
        arms = stage_groups[stage.name]["arms"]
        for method, views, pool in sorted({(r["method"], r["views"], r["pool"]) for r in rows}):
            run = [r for r in rows if (r["method"], r["views"], r["pool"]) == (method, views, pool)]
            n = [math.sqrt(r["bestLoss"]) for r in run]
            arms[f"{method}/{views}v/{pool}"] = dict(trials=len(run), passed=sum(r["passed"] for r in run), failures=sum(r["structureFailure"] for r in run),
                train=[statistics.median(n), max(n)], held=[statistics.median(r["heldNrmse"] for r in run), max(r["heldNrmse"] for r in run)],
                high=[statistics.median(r["highNrmse"] for r in run), max(r["highNrmse"] for r in run)],
                first=statistics.median(r["firstPass"] for r in run if r["passed"]) if any(r["passed"] for r in run) else None,
                queries=statistics.median(r["trainingRenders"] for r in run), swaps=sum(r["swapRenders"] for r in run),
                total_wall_s=sum(r["elapsedSeconds"] for r in run), median_train_s=statistics.median(r["trainingSeconds"] for r in run))
    check(bool(trials), f"no complete stages in {root}")
    frozen = root / "e9_frozen_probe" / "probe.json"
    if frozen.is_file():
        probe = load(frozen)
        for r in probe["trials"]:
            base = frozen.parent / r["target"] / str(r["seed"]) / r["method"]
            records = list(ledger(base / "queries.csv", 1))
            context = next(x for x in records if x[1] == "fixed_context_base")
            check(context[2] == tuple(r["result"]["beforeTheta"]) and close(context[3], r["result"]["beforeLoss"]), f"frozen context mismatch: {base}")
            actual = [q[3] for q in records if q[1].startswith("fixed_context_verify_")]
            improved = bool(actual) and context[3] - min(actual) > max(1e-10, 1e-5 * context[3])
            check(improved == r["result"]["accepted"] and records[-1][0] == r["renders"] <= r["budget"], f"probe acceptance/camera ledger: {base}")
        check(sum(r["renders"] for r in probe["trials"]) == probe["probeRenders"], "probe render total")
        check(sum(list(ledger(p, 1))[-1][0] for p in frozen.parent.rglob("shared.csv")) == probe["sharedRenders"], "probe shared total")
        stage_groups["e9_frozen_probe"] = {"experimental": probe["probeRenders"], "shared": probe["sharedRenders"],
            "accepted": {m: sum(r["result"]["accepted"] for r in probe["trials"] if r["method"] == m) for m in ("insertion", "joint")}}
    save_csv(dest / "trials.csv", trials)
    save_csv(dest / "query-checkpoints.csv", checkpoints)
    (dest / "summary.json").write_text(json.dumps(stage_groups, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if archive:
        with zipfile.ZipFile(dest / "query-data.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7) as z:
            for path in stages + ([frozen] if frozen.is_file() else []):
                for p in sorted(path.parent.rglob("*")):
                    if p.suffix.lower() in {".csv", ".json"}:
                        z.write(p, p.relative_to(root).as_posix())
    return stage_groups


def plot(root, out):
    from PIL import Image, ImageDraw
    stage = root / "e9_2000" / "clustered"
    files = [("E9 target", stage / "target_view0.png"),
             ("E9 original insertion - 2 lights", stage / "main/181_insertion/best_view0.png"),
             ("E9 joint weights + LM - 3 lights", stage / "main/181_joint/best_view0.png")]
    width = 365
    sections = [(files, 365)]
    e10 = next((n for n in ("e10", "e10_smoke", "e10_near") if (root / n / "stage.json").is_file()), None)
    e11 = next((n for n in ("e11_near", "e11", "e11_near_smoke", "e11_smoke") if (root / n / "stage.json").is_file()), None)
    if e10:
        p = root / e10
        rows = load(p / "stage.json")["trials"]
        best = next((r for r in rows if r["seed"] == 131 and r["views"] == 3), rows[0])
        run = trial_dir(p, best)
        sections.append(([("Classroom target - held-out", p / "target_view3.png"),
                          ("E10 perturbed rig - held-out", run / "initial_view3.png"),
                          ("E10 best query - held-out", run / "best_view3.png")], 209))
    if e11:
        p = root / e11
        rows = load(p / "stage.json")["trials"]
        seed = min(r["seed"] for r in rows)
        comparisons = [("Classroom target - held-out", p / "target_view3.png")]
        for method in ("multi_start", "joint", "insertion"):
            row = next((r for r in rows if r["method"] == method and r["seed"] == seed), None)
            if row:
                comparisons.append((f'E11 {method} ({row["active"]}/{row["lamps"]}) - held-out', trial_dir(p, row) / "best_view3.png"))
            if len(comparisons) == 3:
                break
        sections.append((comparisons, 209))
    image = Image.new("RGB", (3 * width + 40, sum(h + 38 for _, h in sections) + 18), "#101820")
    draw = ImageDraw.Draw(image)
    top = 14
    for group, height in sections:
        for i, (title, path) in enumerate(group):
            panel = Image.open(path).convert("RGB")
            panel.thumbnail((width, height), Image.Resampling.LANCZOS)
            left = 10 + i * (width + 10)
            image.paste(panel, (left, top))
            draw.text((left, top + height + 8), title, fill="#f4f7fa", font_size=15)
        top += height + 38
    image.save(out / "native-comparison.png", optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.out, args.archive), indent=2))
    if args.plot:
        plot(args.root, args.out)


if __name__ == "__main__":
    main()

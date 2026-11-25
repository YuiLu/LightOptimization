"""Reconcile Unity Classroom E13–E15 camera ledgers and export paired results."""
import argparse
from collections import Counter
import csv
import json
import math
import statistics
import zipfile
from pathlib import Path

from blackbox_scale_report import check, close, ledger, load, save_csv


CHECKPOINTS = (3, 6, 30, 100, 300, 500, 1000, 2000, 4000, 6000, 10000, 20000)
BASELINE = "96002e457fdc65d0373d761858a4ad307fee5149"


def contents(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def nrmse(values):
    return math.sqrt(statistics.mean(v * v for v in values))


def audit_trial(folder, row, gate):
    check(load(folder / "trial.json") == row, f"stage/trial mismatch: {folder}")
    queries = list(ledger(folder / "queries.csv"))
    previous = 0
    for camera, phase, _, loss in queries:
        check(camera - previous == (3 if loss is not None else 1), f"query camera increment: {folder}:{phase}")
        previous = camera
    images = list(ledger(folder / "recaptures.csv", 1))
    progress = contents(folder / "progress.csv")
    observed = [q for q in queries if q[3] is not None]
    check(queries[-1][0] == row["trainingRenders"] <= row["budget"], f"trial budget: {folder}")
    check(images[-1][0] == row["recaptures"] == 11, f"recapture ledger: {folder}")
    check(len(observed) == len(progress), f"objective ledger: {folder}")
    check(observed[0][0:3] == (3, "empty", ()) and observed[1][0:3] == (6, "anchor", tuple(row["start"])), f"anchor: {folder}")
    for (camera, phase, theta, objective), item in zip(observed, progress):
        raw = float(item["raw_sq"])
        check((camera, phase) == (int(item["camera"]), item["phase"]) and close(objective, float(item["objective_sq"])) and math.isfinite(raw) and raw >= 0,
              f"desynced objective: {folder}:{camera}")
        check(len(theta) == (0 if phase == "empty" else 28), f"objective rig: {folder}:{camera}")
        if row["mapping"] == "Raw":
            check(close(objective, raw), f"raw objective mismatch: {folder}:{camera}")
    p = progress[1:]
    best = raw = p[0]
    for item in p:
        if float(item["objective_sq"]) < float(best["objective_sq"]):
            best = item
        if float(item["raw_sq"]) < float(raw["raw_sq"]):
            raw = item
        for field, current in (("best_objective_sq", best["objective_sq"]), ("raw_at_best_sq", best["raw_sq"]), ("best_raw_sq", raw["raw_sq"])):
            check(close(float(item[field]), float(current)), f"checkpoint {field}: {folder}:{item['camera']}")
        item["best_objective_query"], item["best_raw_query"] = best["camera"], raw["camera"]
    check(int(best["camera"]) == row["bestQuery"] and int(raw["camera"]) == row["bestRawQuery"], f"best query: {folder}")
    best_theta = next(q[2] for q in observed if q[0] == row["bestQuery"])
    check(best_theta == tuple(row["best"]), f"best rig: {folder}")
    for field, source in [("finalObjective", best["objective_sq"]), ("finalRaw", best["raw_sq"]), ("initialRaw", p[0]["raw_sq"]), ("bestRaw", raw["raw_sq"])]:
        check(close(row[field] ** 2, float(source)), f"wrong {field}: {folder}")
    check(close(row["recapturedRaw"], row["finalRaw"], 1e-6), f"raw recapture: {folder}")
    check(close(row["recapturedRaw"], nrmse(row["trainPerView"])) and close(row["highNrmse"], nrmse(row["highPerView"][:3])) and
          close(row["heldNrmse"], nrmse(row["highPerView"][3:])), f"per-view RMS: {folder}")
    reached = [int(x["camera"]) for x in p if float(x["raw_sq"]) <= min(.05 ** 2, .10 * gate["emptyLoss"])]
    check(row["firstRawPass"] == (reached[0] if reached else -1), f"first crossing: {folder}")
    check(row["passed"] == (row["recapturedRaw"] <= .05 and row["reduction"] >= .90), f"threshold: {folder}")
    jacobians = contents(folder / "jacobians.csv")
    steps = contents(folder / "lm-steps.csv")
    check(len(jacobians) == row["jacobians"], f"Jacobian ledger: {folder}")
    if row["lamp"] >= 0:
        contributing = [n for n in row["finalContributionPerView"] if n >= 0]
        check(len(contributing) == row["contributionViews"] and close(nrmse(contributing), row["finalContributionNrmse"]), f"contribution denominator: {folder}")
    return p, jacobians, steps


def lm_summary(row, jacobians, steps):
    jac_ranks = [int(j["rank"]) for j in jacobians]
    active_counts = [len(j["axes"].split(";")) for j in jacobians]
    full_conditions = [float(j["condition_j"]) for j in jacobians if float(j["condition_j"]) > 0]
    blocked = sum(j["stencil"].split(";").count("0") for j in jacobians)
    zero = sum(sum(float(x) == 0 for x in j["column_norm"].split(";")) for j in jacobians)
    position_steps, power_steps, correlations = [], [], []
    for j in jacobians:
        axes = list(map(int, j["axes"].split(";")))
        norms = list(map(float, j["column_norm"].split(";")))
        for axis, step in zip(axes, map(float, j["metres_or_log_step"].split(";"))):
            if step != 0:
                (power_steps if axis % 4 == 3 else position_steps).append(abs(step))
        gram = iter(map(float, j["gram_upper"].split(";")))
        cross = []
        for i, a in enumerate(axes):
            for k in range(i, len(axes)):
                value = next(gram)
                if a // 4 != axes[k] // 4 and norms[i] > 0 and norms[k] > 0:
                    cross.append(min(1, abs(value) / (norms[i] * norms[k])))
        if cross:
            correlations.append(max(cross))
    gains = {status: [float(x["gain_ratio"]) for x in steps if x["status"] == status and x["gain_ratio"]] for status in ("accept", "reject")}
    return dict(stage=row["stage"], key=row["key"], jacobians=len(jacobians),
                minimum_rank=min(jac_ranks, default=None), median_rank=statistics.median(jac_ranks) if jac_ranks else None,
                rank_deficient=sum(r < n for r, n in zip(jac_ranks, active_counts)),
                median_condition_j=statistics.median(full_conditions) if full_conditions else None,
                blocked_columns=blocked, zero_columns=zero,
                median_position_step_m=statistics.median(position_steps) if position_steps else None,
                median_log_intensity_step=statistics.median(power_steps) if power_steps else None,
                median_max_cross_lamp_correlation=statistics.median(correlations) if correlations else None,
                step_statuses=json.dumps(Counter(x["status"] for x in steps), sort_keys=True),
                accepted_gain_median=statistics.median(gains["accept"]) if gains["accept"] else None,
                rejected_gain_median=statistics.median(gains["reject"]) if gains["reject"] else None)


def audit(root, out, archive=False):
    out.mkdir(parents=True, exist_ok=True)
    stages = sorted(p.parent for p in root.glob("*/stage.json") if "smoke" not in p.parent.name)
    check(stages, f"no completed diagnosis stages in {root}")
    trials, checkpoints, diagnostics, summary = [], [], [], {}
    for stage in stages:
        run, gate = load(stage / "stage.json"), load(stage / "replay.json")
        check(gate["passed"] and gate["baseline"] == BASELINE, f"replay gate: {stage}")
        check(list(ledger(stage / "shared.csv", 1))[-1][0] == run["sharedRenders"], f"shared cameras: {stage}")
        rows = run["trials"]
        check(sum(x["trainingRenders"] for x in rows) == run["trainingRenders"] and sum(x["recaptures"] for x in rows) == run["recaptures"], f"stage totals: {stage}")
        for row in rows:
            progress, jacobians, steps = audit_trial(stage / row["key"], row, gate)
            diagnostics.append(lm_summary(row, jacobians, steps))
            flat = {key: val for key, val in row.items() if not isinstance(val, (list, dict))}
            for field in ("trainPerView", "highPerView", "displayPerView", "displayHighPerView", "initialContributionPerView", "finalContributionPerView",
                          "positionErrorMetres", "intensityRelativeError"):
                flat[field] = ";".join(map(str, row.get(field) or []))
            for tag in ("initial", "final"):
                for metric in ("peak", "top01", "top1", "top5"):
                    flat[f"{tag}_{metric}_per_view"] = ";".join(str(b[metric]) for b in row[tag + "Brightness"])
            flat["displayTrain"] = nrmse(row["displayPerView"])
            flat["displayHigh"] = nrmse(row["displayHighPerView"][:3])
            flat["displayHeld"] = nrmse(row["displayHighPerView"][3:])
            trials.append(flat)
            for cap in CHECKPOINTS:
                if cap < 6 or cap > row["budget"]:
                    continue
                latest = next(x for x in reversed(progress) if int(x["camera"]) <= cap)
                checkpoints.append(dict(stage=row["stage"], key=row["key"], checkpoint=cap, actual=int(latest["camera"]),
                                        objective=math.sqrt(float(latest["best_objective_sq"])), raw_at_objective_best=math.sqrt(float(latest["raw_at_best_sq"])),
                                        objective_best_query=int(latest["best_objective_query"]), best_raw=math.sqrt(float(latest["best_raw_sq"])),
                                        raw_best_query=int(latest["best_raw_query"])))
        groups = {}
        for row in rows:
            kind = "full" if row["lamp"] < 0 else "single"
            groups.setdefault("/".join((row["profile"], kind, row["method"], row["mapping"])), []).append(row)
        summary[run["stage"]] = dict(shared=run["sharedRenders"], training=run["trainingRenders"], recaptures=run["recaptures"], arms={
            name: dict(trials=len(group), passed=sum(r["passed"] for r in group), contribution_pass=sum(r["contributionPass"] for r in group),
                       train=statistics.median(r["recapturedRaw"] for r in group), worst_train=max(r["recapturedRaw"] for r in group),
                       high=statistics.median(r["highNrmse"] for r in group), held=statistics.median(r["heldNrmse"] for r in group),
                       queries=statistics.median(r["trainingRenders"] for r in group), seconds=statistics.median(r["elapsed"] for r in group))
            for name, group in sorted(groups.items())})
    save_csv(out / "trials.csv", trials)
    save_csv(out / "query-checkpoints.csv", checkpoints)
    save_csv(out / "lm-diagnostics.csv", diagnostics)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if archive:
        with zipfile.ZipFile(out / "query-data.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=7) as z:
            for stage in stages:
                for file in sorted(stage.rglob("*")):
                    if file.is_file() and file.suffix in {".csv", ".json"}:
                        z.write(file, file.relative_to(root).as_posix())
    return summary


def plot(root, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFont

    e13 = root / "e13"
    oracle = load(e13 / "oracle.json")
    curves = contents(e13 / "axis-curves.csv")
    fig, panels = plt.subplots(2, 3, figsize=(10.5, 5.2), layout="constrained", sharex=True)
    contexts = [("gt", "GT"), ("near_single_start", "Isolated near start"), ("near_full_final", "E10 full-rig endpoint")]
    for i, lamp in enumerate((oracle["easyLamp"], oracle["hardLamp"])):
        for axis in range(3):
            ax = panels[i][axis]
            for key, name in contexts:
                samples = sorted((float(r["metres"]) * 100, 100 * math.sqrt(float(r["raw_sq"])) if r["legal"] == "1" else math.nan)
                                 for r in curves if int(r["lamp"]) == lamp and int(r["axis"]) == axis and r["context"] == key)
                ax.plot(*zip(*samples), ".-", label=name)
            ax.set(yscale="symlog", ylim=(0, None), title=f"Lamp {lamp} / {'XYZ'[axis]}")
            ax.grid(alpha=.2)
            if axis == 0:
                ax.set_ylabel("Raw HDR NRMSE (%)")
            if i == 1:
                ax.set_xlabel("Physical offset (cm)")
    panels[0][0].legend(fontsize=7)
    fig.savefig(out / "axis-curves.png", dpi=160)
    plt.close(fig)

    comparisons = [(tag, load(root / tag / "stage.json")["trials"]) for tag in ("e14", "e15", "e15_budget", "e14_large") if (root / tag / "stage.json").exists()]
    if comparisons:
        seeds = sorted({r["seed"] for _, rows in comparisons for r in rows if r["lamp"] < 0})
        fig, axs = plt.subplots(len(comparisons), len(seeds), figsize=(3.7 * len(seeds), 2.8 * len(comparisons)), layout="constrained", squeeze=False)
        for (tag, rows), panels in zip(comparisons, axs):
            row_seeds = sorted({r["seed"] for r in rows if r["lamp"] < 0})
            for ax in panels[len(row_seeds):]:
                ax.set_visible(False)
            for seed, ax in zip(row_seeds, panels):
                for r in (x for x in rows if x["seed"] == seed and x["lamp"] < 0):
                    trace = contents(root / tag / r["key"] / "progress.csv")[1:]
                    arm = r["mapping"] if tag.startswith("e14") else r["method"]
                    ax.step([int(x["camera"]) for x in trace], [100 * math.sqrt(float(x["raw_at_best_sq"])) for x in trace],
                            where="post", label=arm)
                ax.axhline(5, color="#666666", linestyle=":", linewidth=1)
                limit = 20500 if tag.startswith("e15_budget") else 6100
                ax.set(yscale="log", xscale="symlog", xlim=(6, limit), title=f"{tag.upper().replace('_', ' ')} / {seed}")
                ax.set_xlabel("Training camera renders")
                ax.grid(alpha=.2)
            panels[0].set_ylabel("Raw HDR NRMSE (%)")
            panels[0].legend(fontsize=7)
        fig.savefig(out / "convergence.png", dpi=160)
        plt.close(fig)

    tile_width, tile_height = 365, 209
    panels = [[("GT / main view", e13 / "large131_target.png"), ("E10 large / initial", e13 / "large131_initial.png"),
               ("E10 large / endpoint", e13 / "large131_best.png")],
              [("GT / main view", e13 / "large131_target.png"), ("Error heatmap / initial", e13 / "large131_heat_initial.png"),
               ("Error heatmap / endpoint", e13 / "large131_heat_best.png")]]
    stage = root / "e14"
    if (stage / "stage.json").exists():
        trials = [r for r in load(stage / "stage.json")["trials"] if r["lamp"] < 0 and r["seed"] == 131]
        panels += [[(f'{r["mapping"]} / {title}', stage / r["key"] / f"best_view{view}.png") for r in trials]
                   for view, title in [(0, "training view"), (3, "held-out view")]]
    gap, label = 9, 25
    im = Image.new("RGB", (3 * tile_width + 4 * gap, len(panels) * (tile_height + label + gap) + gap), "#121922")
    captions = ImageDraw.Draw(im)
    for y, panel in enumerate(panels):
        for x, (text, path) in enumerate(panel):
            with Image.open(path) as source:
                tile = source.convert("RGB")
                tile.thumbnail((tile_width, tile_height))
            x0, y0 = gap + x * (tile_width + gap), gap + y * (tile_height + label + gap)
            im.paste(tile, (x0, y0))
            captions.text((x0, y0 + tile_height + 5), text, fill="#eaf0fb", font=ImageFont.load_default(size=15))
    im.save(out / "native-diagnosis.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    for stage, result in audit(args.root, args.out, args.archive).items():
        print(f"{stage}: {result['training']} training / {result['shared']} shared / {result['recaptures']} validation cameras")
    if args.plot:
        plot(args.root, args.out)

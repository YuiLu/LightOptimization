"""Verify E22 ledgers and collect reproducible tables and figures."""

import csv
import json
import math
import zipfile
from collections import Counter

from blackbox_e20 import csv_read, csv_write, json_read
from blackbox_e21_report import require, save_figure
from blackbox_e22_selection import decisions


ORDER = ("a8", "a28", "b-screen", "b-off", "b-on", "b-confirm", "b-far", "b-switch", "b-hotspots",
         "c", "c-mean", "c-off-733", "c-on", "d-audit", "d-8d", "d-8d-on", "d-28d", "d-28d-on", "a-fd", "e-calibration", "e-recovery")
REQUIRED = {"a8": 6, "b-screen": 12, "b-off": 6, "b-on": 6, "b-confirm": 2, "b-hotspots": 0, "c": 2, "d-audit": 0}
COLORS = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#6a5acd", "#b98800")
LABELS = {"A0": "Dense", "A1": "Block diagonal", "A2": "Block-scaled dense", "B0": "Raw", "B1": "Signed log", "B2": "Signed sqrt",
          "B3": "Tone exp", "B4": "Reinhard", "B5": "Huber"}


def folder(source, key):
    return source / ("e22_" + key.replace("-", "_"))


def verify(path, summary, metrics, trials):
    arms = {r["id"]: r for r in summary["arms"]}
    require(set(arms) == {r["arm"] for r in trials} == set(metrics), f"{path}: mismatched arms")
    opt = Counter()
    actual_pass = Counter()
    truth = summary["truth"]
    with (path / "queries.csv").open(encoding="utf-8", newline="") as stream:
        for query in csv.DictReader(stream):
            key = query["arm"]
            opt[key] += 1
            require(int(query["query"]) == opt[key] and query["collider_legal"] == query["survey_legal"] == "1", f"{key}: illegal or uncounted query")
            require(math.isfinite(float(query["loss"])), f"{key}: nonfinite objective")
    for choice in csv_read(path / "decisions.csv"):
        key = choice["arm"]
        if choice["accepted"] == "1":
            actual_pass[key] += 1
            require(float(choice["actual"]) > 0 and float(choice["rho"]) >= .02 and choice["legal"] == "1", f"{key}: invalid accepted step")
    for row in trials:
        key = row["arm"]
        a, m = arms[key], metrics[key]
        expected = a["legal"] and max(a["positionCm"]) <= 5 and max(a["intensityRelative"]) <= .05
        require(a["legal"] and expected == a["pass"] == bool(int(row["pass"])), f"{key}: inconsistent parameter result")
        require(all(abs(a["initial"][i] - truth[i]) <= 1e-12 and abs(a["final"][i] - truth[i]) <= 1e-12
                    for i in range(len(truth)) if i // 4 + 1 not in a["lamps"]), f"{key}: inactive lamps changed")
        require(a["evaluations"] == int(row["evaluations"]) <= a["limit"], f"{key}: exceeded Q")
        if m["reusedFrom"]:
            require(opt[key] == 0 and m["newOptimizationCameras"] == 0 and m["optimizationCameras"] == 3 * a["evaluations"], f"{key}: reuse was charged twice")
        else:
            require(opt[key] == a["evaluations"] and actual_pass[key] == a["accepted"], f"{key}: unmatched solver records")
        require(m["peakGraphicsBytes"] > 0, f"{key}: missing graphics memory sample")
        if row["shadows"] == "1":
            row["shadowed_rgb_cameras"] = m["newOptimizationCameras"] + m["validationCameras"]
        else:
            row["shadowed_rgb_cameras"] = 0
    for test in csv_read(path / "loss-checks.csv"):
        require(float(test["absolute_difference"]) <= 2e-6 * max(1e-7, float(test["scalar"])), f"{path}: residual equality failed")


def initial_jacobian(path, arm, np):
    record = next(x for x in csv_read(path / "jacobians.csv") if x["arm"] == arm)
    rows = [x for x in csv_read(path / "gram.csv") if x["arm"] == arm and x["iteration"] == record["iteration"] and x["segment"] == record["segment"]]
    return np.asarray([[float(x) for x in r["values"].split(";")] for r in rows]), np.asarray([float(x) for x in record["gradient_theta"].split(";")])


def compare_predictions(source, export, stages, np):
    rows = []
    for key in ("d-8d", "d-28d", "d-8d-on", "d-28d-on"):
        if key not in stages:
            continue
        summary, meta, trials = stages[key]
        scene = summary["scene"]
        physical = [scene["boundsMax"][axis] - scene["boundsMin"][axis] for axis in ("x", "y", "z")] + [math.log(9 / .055)]
        for row in (r for r in trials if r["method"] == "D1"):
            base = next(r for r in trials if r["scenario"] == row["scenario"] and r["method"] == "D0")
            reused = meta[base["arm"]]["reusedFrom"]
            if reused:
                directory, arm = reused.split("/", 1)
                hp, gp = initial_jacobian(source / directory, arm, np)
            else:
                hp, gp = initial_jacobian(folder(source, key), base["arm"], np)
            step = next((r for r in csv_read(folder(source, key) / "decisions.csv") if r["arm"] == row["arm"] and r["iteration"] == "1" and math.isfinite(float(r["actual"]))), None)
            if step is None:
                continue
            delta = np.asarray([float(x) for x in step["step_phys"].split(";")]) / np.tile(physical, len(gp) // 4)
            fd = float(-2 * (gp @ delta) - delta @ hp @ delta)
            rows.append((key, row["arm"], step["query"], fd, float(step["predicted"]), float(step["actual"]), step["rho"]))
    if rows:
        csv_write(export / "olat-prediction-comparison.csv", ("stage", "arm", "query", "fd_prediction_at_same_step", "hybrid_prediction", "actual", "hybrid_rho"), rows)


def plot_solver(plt, grouped, dest):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    arms = grouped["a8"][2]
    for ax, scenario, label in zip(axes, ("seed733_near_L5_L7", "seed733_near_L2_L7", "seed733_near_28d"),
                                    ("8D L5 + L7", "8D L2 + L7", "28D seed733")):
        rows = [r for r in (*arms, *grouped["a28"][2]) if r["scenario"] == scenario]
        bars = ax.bar([r["method"] for r in rows], [int(r["evaluations"]) for r in rows], color=[COLORS[int(r["method"][1])] for r in rows])
        ax.bar_label(bars, labels=[f"{r['evaluations']}\n{'Pass' if r['pass']=='1' else 'Fail'}" for r in rows], fontsize=9, padding=3)
        ax.set(title=label, ylabel="Mixed RGB objective calls", ylim=(0, max(int(r["evaluations"]) for r in rows) * 1.25))
    save_figure(plt, fig, dest / "solvers.png")


def plot_tone(plt, grouped, dest):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.3), layout="constrained")
    for ax, (key, title) in zip(axes.flat, (("b-screen", "520-call screening · displayed holdout"),
                                           ("b-off", "Shadow Off · displayed holdout"),
                                           ("b-on", "Shadow On · displayed holdout"), ("b-confirm", "Held-out seed181 · E19 fallback start"))):
        _, meta, rows = grouped[key]
        methods = list(dict.fromkeys(r["method"] for r in rows)); cases = list(dict.fromkeys(r["scenario"] for r in rows))
        lookup = {(r["scenario"], r["method"]): r for r in rows}
        width = .75 / len(methods)
        for j, method in enumerate(methods):
            result = [meta[lookup[(case, method)]["arm"]]["displayHeld"] * 100 for case in cases]
            x = [i + width * (j - (len(methods) - 1) / 2) for i in range(len(cases))]
            ax.bar(x, [max(1e-6, y) for y in result], width, color=COLORS[int(method[1])], label=LABELS[method])
            for x0, y, case in zip(x, result, cases):
                if lookup[(case, method)]["pass"] != "1": ax.plot(x0, max(1e-6, y), "x", color="black", ms=7)
        ax.set(yscale="log", title=title, ylabel="Display NRMSE (%)", xticks=range(len(cases)), xticklabels=[c.split("_")[0] for c in cases])
        ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", frameon=False, ncols=6)
    save_figure(plt, fig, dest / "tone-holdout.png")


def plot_hotspots(plt, source, dest):
    from matplotlib.patches import Patch

    raw = folder(source, "b-hotspots")
    jac = csv_read(raw / "hotspot-jacobians.csv")
    first = jac[0]["checkpoint"]
    by_method = {r["scheme"]: r for r in jac if r["checkpoint"] == first and r["region"] == "raw_top_1"}
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for ax, col, label in zip(axs, ("gradient_fraction", "gram_fraction"), ("Gradient norm fraction", "Gram Frobenius fraction")):
        values = [float(by_method[key][col]) for key in labels_b()]
        ax.bar(range(len(values)), values, color=COLORS)
        ax.set(xticks=range(len(values)), xticklabels=[LABELS[k] for k in labels_b()], ylabel=label, title="Raw top 1% of pixels")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(first.replace("_", " "))
    save_figure(plt, fig, dest / "hotspot-jacobian.png")

    patch = raw / "captures" / first
    names = (("rgb_v1", "Native RGB"), ("raw_error_v1", "Raw error"), ("display_error_v1", "Display error"),
             ("B0_top_v1", "Raw top pixels"), ("B1_top_v1", "Signed log top pixels"), ("B3_top_v1", "Tone exp top pixels"))
    fig, axes = plt.subplots(2, 3, figsize=(11, 5), layout="constrained")
    for ax, (name, title) in zip(axes.flat, names):
        im = plt.imread(patch / (name + ".png"))
        if "error" in name: ax.imshow(im[..., 0], cmap="magma", vmin=0, vmax=1)
        else: ax.imshow(im)
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle("Seed131 Near, training view 1")
    bands = ("Top 0.1%", "0.1%–1%", "1%–5%")
    patches = [Patch(facecolor=color, label=band) for color, band in zip(("#f84638", "#ffc74e", "#5795d9"), bands)]
    fig.legend(handles=patches, loc="outside lower center", frameon=False, ncols=3)
    save_figure(plt, fig, dest / "hotspot-images.png")


def labels_b():
    return tuple(f"B{i}" for i in range(6))


def plot_audits(plt, source, dest):
    rows = csv_read(folder(source, "d-audit") / "olat-linearity.csv")
    fig, ax = plt.subplots(figsize=(7.6, 3.7), layout="constrained")
    values = [[max(1e-9, 100 * float(r["relative_error"])) for r in rows if r["shadows"] == shadow and r["test"] == test]
              for shadow in ("0", "1") for test in ("intensity", "mixture")]
    ax.boxplot(values, tick_labels=("Off × intensity", "Off × mixture", "On × intensity", "On × mixture"), showfliers=True, flierprops={"markersize": 3})
    ax.axhline(5, color=COLORS[1], ls="--", label="Worst-case limit (5%)")
    ax.set(yscale="log", ylabel="Relative incremental RGB error (%)", title="960 native OLAT audit captures")
    ax.legend(frameon=False)
    save_figure(plt, fig, dest / "olat-linearity.png")

    path = folder(source, "e-calibration") / "shadow-calibration.csv"
    if not path.is_file():
        return
    rows = [r for r in csv_read(path) if r["pose"] != "gt"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    methods = ("e21_msm", "matched_msm", "e21_mbd", "matched_mbd")
    for ax, key, label in zip(axs, ("weighted_visibility_mae", "direct_nrmse"), ("Energy-weighted visibility MAE (%)", "Direct irradiance NRMSE (%)")):
        ax.boxplot([[100 * float(r[key]) for r in rows if r["method"] == m] for m in methods],
                   tick_labels=("MSM E21", "MSM aligned", "MBD E21", "MBD aligned"))
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=16)
    save_figure(plt, fig, dest / "shadow-calibration.png")


def report(args):
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source, export = args.output.resolve(), args.report_dir.resolve()
    export.mkdir(parents=True, exist_ok=True)
    plots = export / "figures"
    plots.mkdir(exist_ok=True)
    groups = {}
    memory = {}
    rows = []
    for key in ORDER:
        path = folder(source, key)
        if not (path / "summary.json").is_file():
            continue
        s, m = json_read(path / "summary.json"), json_read(path / "metrics.json")
        meta = {r["arm"]: r for r in m["metrics"]}
        trials = csv_read(path / "trials.csv")
        verify(path, s, meta, trials)
        for trial in trials:
            items = {k: v for k, v in meta[trial["arm"]].items() if k not in ("arm", "displayPerView")}
            rows.append(dict(stage=key, **trial, **items, repeatPeakWorkingBytes="", repeatEvaluations=""))
        groups[key] = (s, meta, trials)
    required = set(REQUIRED) | {"a28", "a-fd"}
    require(required <= set(groups), f"Missing stages {required - set(groups)}")
    require(all(len(groups[key][2]) == count for key, count in REQUIRED.items()), "Incomplete required stage matrix")
    require(len(groups["a28"][2]) == (1 if json_read(source / "decision-a8.json")["selected"] == "A0" else 2), "Incomplete 28D comparison")
    fixed = [r for r in groups["a-fd"][2] if r["method"].startswith("fixed_")]
    observed = {r["scenario"] for r in groups["a-fd"][2] if r["method"].startswith("adaptive_")}
    require(len(fixed) == 6 and observed == {r["scenario"] for r in fixed if r["stop"] == "no_decreasing_step"}, "Incomplete adaptive FD comparison")
    require(len({(g[0]["sceneHash"], g[0]["branch"], g[0]["commit"]) for g in groups.values()}) == 1, "The scene or baseline changed across stages")
    if args.memory_check is not None:
        for key in ("a8", "a28", "b-screen"):
            path = folder(args.memory_check, key)
            primary = groups[key][0]
            repeat, audit = json_read(path / "summary.json"), json_read(path / "metrics.json")
            repeat_metrics = {m["arm"]: m for m in audit["metrics"]}
            verify(path, repeat, repeat_metrics, csv_read(path / "trials.csv"))
            require((repeat["sceneHash"], repeat["commit"]) == (primary["sceneHash"], primary["commit"]), "Memory replay used different assets")
            initial = {r["id"]: r for r in primary["arms"]}
            for arm in repeat["arms"]:
                baseline = initial[arm["id"]]
                require(all(arm[k] == baseline[k] for k in ("initial", "objective", "positionStep", "lamps", "shadows", "limit", "pass")),
                        f"Memory replay did not match {arm['id']}")
            repeat_counts = {a["id"]: a["evaluations"] for a in repeat["arms"]}
            memory[key] = {"processPeakWorkingBytes": audit["processPeakWorkingBytes"], "arms": len(repeat_counts),
                           "evaluationDifferences": {k: v - initial[k]["evaluations"] for k, v in repeat_counts.items() if v != initial[k]["evaluations"]}}
            for row in (r for r in rows if r["stage"] == key):
                row["repeatPeakWorkingBytes"] = repeat_metrics[row["arm"]]["peakWorkingBytes"]
                row["repeatEvaluations"] = repeat_counts[row["arm"]]
    evidence = decisions(source)
    if evidence.get("shadowCalibration"):
        require("e-calibration" in groups, "Shadow On triggered calibration")
        if any(r["continueRecovery"] for r in evidence["proxy"].values()): require("e-recovery" in groups, "Calibrated proxy triggered recovery")
    for state in evidence["olatLinearity"]:
        key = "d-8d-on" if state["shadows"] else "d-8d"
        if state["pass"]: require(key in groups, f"Linearity triggered {key}")
        if evidence.get(key, {}).get("extend"):
            require(key.replace("8d", "28d") in groups, f"8D hybrid triggered its 28D comparison ({key})")
    if evidence.get("geometry", {}).get("extend"):
        require({"c-mean", "c-off-733", "c-on"} <= set(groups), "Patch comparison triggered additional geometry arms")
    if evidence.get("far"): require("b-far" in groups, "Near results triggered Far")
    if evidence.get("scheduledLoss"): require("b-switch" in groups, "Appearance tradeoff triggered the two-stage loss check")
    outcomes = {k: {"arms": len(g[2]), "passes": sum(int(r["pass"]) for r in g[2]), "cameras": g[0]["nativeCameraRenders"],
                    "seconds": g[0]["seconds"]} for k, g in groups.items()}
    (export / "stage-summary.json").write_text(json.dumps({"decisions": evidence, "outcomes": outcomes, "memoryRecheck": memory,
        "stages": {k: dict(g[0], measurements=json_read(folder(source, k) / "metrics.json")) for k, g in groups.items()}}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    csv_write(export / "trials.csv", rows[0], (r.values() for r in rows))
    compare_predictions(source, export, groups, np)
    plot_solver(plt, groups, plots)
    plot_tone(plt, groups, plots)
    plot_hotspots(plt, source, plots)
    plot_audits(plt, source, plots)
    with zipfile.ZipFile(export / "raw-data.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for key in groups:
            for file in sorted(folder(source, key).rglob("*")):
                if file.suffix in (".csv", ".json"):
                    archive.write(file, file.relative_to(source).as_posix())
        for file in sorted(source.glob("decision-*.json")):
            archive.write(file, file.name)
        if args.memory_check is not None:
            for key in memory:
                for name in ("summary.json", "trials.csv", "metrics.json", "source-files-sha256.json"):
                    path = folder(args.memory_check, key) / name
                    archive.write(path, f"memory-recheck/{path.parent.name}/{name}")
        if (export / "olat-prediction-comparison.csv").is_file():
            archive.write(export / "olat-prediction-comparison.csv", "olat-prediction-comparison.csv")
    print(f"Verified {len(rows)} optimization arms in {len(groups)} stages. Exported {export}")

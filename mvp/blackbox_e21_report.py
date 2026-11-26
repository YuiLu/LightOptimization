"""Verify complete E21 run ledgers and assemble data and scientific figures."""

import csv
import json
import math
import zipfile
from collections import Counter, defaultdict

from blackbox_e20 import csv_read, csv_write, json_read


A_CASES = ("triangle", "box", "two_boxes_thin_wall", "classroom_L5")
A_LABELS = ("Triangle", "Box", "Two boxes and thin wall", "Classroom L5")
A_COSINES = (("frozen_adjoint", "fixed_fd", "Adjoint vs fixed"),
             ("fixed_fd", "full_fd", "Fixed vs full"),
             ("full_fd", "ray_soft_fd", "Full vs soft ray"),
             ("full_fd", "ray_hard_fd", "Full vs hard ray"),
             ("full_fd", "native_hard_fd", "Full vs URP hard"))
A_PARTS = ("caster_projection_fd", "caster_radial_fd", "uv_only_fd", "receiver_depth_fd", "map_only_fd",
           "linear_sum_defect", "map_projection_radial_defect")
ARM_COUNTS = {"b": 36, "c": 20, "d": 10, "e": 42}
COLORS = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#6a5acd", "#b98800", "#64748b")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frozen_slots(arm):
    return [i for i in range(len(arm["initial"])) if i // 4 + 1 not in arm["lamps"]]


def verify_arms(stage, folder, summary, trials):
    arms = {arm["id"]: arm for arm in summary["arms"]}
    require(len(arms) == len(trials) == ARM_COUNTS[stage] and set(arms) == {r["arm"] for r in trials},
            f"{folder}: incomplete arm table")
    starts, calls, native_checks, illegal = {}, Counter(), Counter(), Counter()
    truth = summary["truth"]
    frozen = {key: frozen_slots(arm) for key, arm in arms.items()}
    for row in trials:
        key = row["arm"]
        arm = arms[key]
        passes = arm["legal"] and max(arm["positionCm"]) <= 5 and max(arm["intensityRelative"]) <= .05
        require(arm["legal"] and arm["pass"] == passes == bool(int(row["pass"])), f"{key}: inconsistent endpoint")
        require(all(arm["initial"][i] == arm["final"][i] == truth[i] for i in frozen[key]), f"{key}: frozen lamps moved")
        shared = (arm["initial"], arm["limit"], arm["shadows"], arm["lamps"])
        require(starts.setdefault(arm["scenario"], shared) == shared, f"{key}: unmatched comparison")
    with (folder / "queries.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key, phase = row["arm"], row["phase"]
            arm = arms[key]
            calls[key] += 1
            require(int(row["query"]) == calls[key], f"{key}: nonconsecutive query ID")
            require(row["survey_legal"] == "1" and math.isfinite(float(row["loss"])), f"{key}: invalid rendered query")
            theta = row["theta"].split(";")
            require(all(float(theta[i]) == truth[i] for i in frozen[key]), f"{key}: rendered an inactive lamp away from GT")
            native_checks[key] += phase in ("initial_native", "native_acceptance")
            if row["collider_legal"] == "0":
                require(arm["probePolicy"] == "full_through" or arm["probePolicy"] == "probe_through" and phase.startswith("fd_"),
                        f"{key}: query violated {arm['probePolicy']}")
                illegal[(key, "FD" if phase.startswith("fd_") else "trial")] += 1
    decisions, native_rejections = Counter(), Counter()
    for row in csv_read(folder / "decisions.csv"):
        key = row["arm"]
        if row["status"] == "native_rejected":
            native_rejections[key] += 1
        if row["accepted"] != "1":
            continue
        decisions[key] += 1
        require(float(row["actual"]) > 0 and float(row["rho"]) >= .02, f"{key}: an accepted decision failed the trust test")
        if stage == "e" and arms[key]["method"] != "Native_RGB":
            require(float(row["native_actual"]) > 0 and row["legal"] == "1", f"{key}: accepted proxy without legal native improvement")
    for row in trials:
        key = row["arm"]
        arm = arms[key]
        require(calls[key] == arm["evaluations"] == int(row["evaluations"]) <= arm["limit"], f"{key}: query budget mismatch")
        require(native_checks[key] == arm["nativeChecks"] and calls[key] - native_checks[key] == arm["objectiveQueries"],
                f"{key}: native and objective query counts do not add up")
        require(decisions[key] == arm["accepted"], f"{key}: accepted trial count mismatch")
        if stage == "e" and arm["method"] != "Native_RGB":
            require(arm["nativeFinal"] <= arm["nativeInitial"] + 1e-9, f"{key}: native endpoint loss increased")
        row["illegal_fd_renders"], row["illegal_trial_renders"] = illegal[key, "FD"], illegal[key, "trial"]
        row["native_rejected"] = native_rejections[key]
        gap = [k for k, excess in (("position", max(arm["positionCm"]) > 5), ("intensity", max(arm["intensityRelative"]) > .05)) if excess]
        row["parameter_status"] = "+".join(gap) or "pass"


def aggregate_gradients(folder, destination, np):
    pooled = []
    ratios = {key: key.replace("_fd", "") + "_relative_to_full" for key in A_PARTS}
    norms = tuple(dict.fromkeys(key for left, right, _ in A_COSINES for key in (left, right)))
    for key in A_CASES:
        rows = csv_read(folder / key / "derivatives.csv")
        samples = len(csv_read(folder / key / "receivers.csv"))
        groups = defaultdict(list)
        for row in rows:
            groups[float(row["step_m"])].append(row)
        require(len(groups) == 4 and all(len(v) == 3 * samples for v in groups.values()), f"{key}: missing steps or samples")
        for h, group in sorted(groups.items()):
            x = {field: np.asarray([float(row[field]) for row in group]) for field in (*norms, *A_PARTS)}
            require(all(np.isfinite(v).all() for v in x.values()), f"{key}: nonfinite gradient")
            l2 = {field: float(np.linalg.norm(v)) for field, v in x.items()}
            values = {"case": key, "step_mm": round(1000 * h, 3), "samples": samples}
            values.update({field + "_norm": l2[field] for field in l2})
            for left, right, _ in A_COSINES:
                cos = float(x[left] @ x[right] / (l2[left] * l2[right])) if l2[left] * l2[right] else math.nan
                values[f"{left}__{right}"] = cos
            values.update({metric: l2[part] / l2["full_fd"] for part, metric in ratios.items()})
            pooled.append(values)
    csv_write(destination / "gradient-pooled.csv", pooled[0], (r.values() for r in pooled))
    return pooled, ratios


def save_figure(plt, fig, path):
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def plot_gradients(plt, pooled, ratios, plots):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), sharex=True, sharey=True, layout="constrained")
    for ax, key, name in zip(axes.flat, A_CASES, A_LABELS):
        rows = [r for r in pooled if r["case"] == key]
        for (left, right, label), color in zip(A_COSINES, COLORS):
            ax.plot([r["step_mm"] for r in rows], [r[f"{left}__{right}"] for r in rows], ".-", label=label, color=color)
        ax.axhline(0, color="0.6", lw=.8)
        ax.set(xscale="log", title=name, ylim=(-1.06, 1.06), xticks=(.25, 5, 10, 20), xticklabels=("0.25", "5", "10", "20"))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", fontsize=9, ncols=5, frameon=False)
    fig.supxlabel("Position FD step (mm)")
    fig.supylabel("Cosine across XYZ and receivers")
    save_figure(plt, fig, plots / "gradient-directions.png")

    fig, ax = plt.subplots(figsize=(9, 3.3), layout="constrained")
    parts = ("caster_projection_fd", "caster_radial_fd", "uv_only_fd", "receiver_depth_fd", "linear_sum_defect")
    chosen = [r for r in pooled if r["step_mm"] == 10.0]
    labels = ("Caster projection", "Caster radial depth", "Receiver UV", "Receiver radial depth", "Map + fixed defect")
    for part, label, color, offset in zip(parts, labels, COLORS, range(-2, 3)):
        ax.bar([i + offset * .15 for i in range(4)], [r[ratios[part]] for r in chosen], width=.15, color=color, label=label)
    ax.set(xticks=range(4), xticklabels=A_LABELS, ylabel="Norm / full FD norm", title="Isolated contributions at 1 cm")
    ax.legend(fontsize=8, ncols=3, frameon=False, loc="upper right")
    ax.margins(y=.27)
    save_figure(plt, fig, plots / "gradient-components.png")


def plot_recovery(plt, trials, groups, filename, short_labels=None):
    fig, axes = plt.subplots(len(groups), 3, figsize=(12, 2.55 * len(groups)), squeeze=False, layout="constrained")
    metrics = (("max_position_cm", 1, "Position error (cm)", 5), ("max_intensity_relative", 100, "Intensity error (%)", 5),
               ("held_nrmse", 100, "Held-view NRMSE (%)", None))
    for subplot, (group, label) in enumerate(groups):
        rows = [r for r in trials if r["scenario"] == group]
        names = [r["method"] for r in rows] if short_labels is None else short_labels
        for ax, (key, scale, title, line) in zip(axes[subplot], metrics):
            ax.bar(range(len(rows)), [float(r[key]) * scale for r in rows], color=COLORS[:len(rows)], width=.7)
            if line is not None:
                ax.axhline(line, ls="--", color="0.25", lw=.8)
                ax.set_ylim(0, max(line * 1.25, ax.get_ylim()[1]))
            ax.set_xticks(range(len(rows)), names, rotation=25, ha="right", fontsize=8)
            ax.set_title(f"{label} · {title}", fontsize=9)
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=.2)
    save_figure(plt, fig, filename)


def plot_policy(plt, trials, filename, np):
    cases = ("seed131_near_L6", "seed733_far_L1", "cross_geometry_L1", "seed157_far_L3")
    labels = ("131 Near L6", "733 Far L1", "Cross-geometry L1", "157 Far L3")
    fig, ax = plt.subplots(figsize=(10, 3.5), layout="constrained")
    lookup = {(r["scenario"], r["method"]): r for r in trials}
    rows = [[lookup[(f"{case}_fd{h}mm", method)] for case in cases for h in (5, 20, 50)] for method in ("B0", "B1", "B2")]
    ax.imshow(np.asarray([[int(x["pass"]) for x in r] for r in rows]), cmap=plt.matplotlib.colors.ListedColormap(["#f5d4c9", "#cde9eb"]),
              vmin=0, vmax=1, aspect="auto")
    for i, row in enumerate(rows):
        for j, r in enumerate(row):
            ax.text(j, i, r["evaluations"], ha="center", va="center", fontsize=9)
    for divider in (2.5, 5.5, 8.5):
        ax.axvline(divider, color="white", lw=2)
    ax.set(xticks=range(12), xticklabels=("5", "20", "50") * 4, yticks=(0, 1, 2), yticklabels=("B0 Strict", "B1 Probe-through", "B2 Full-through"),
           xlabel="Requested FD step (mm)", title="Blue cells passed. Numbers show evaluations.")
    top = ax.secondary_xaxis("top")
    top.set_xticks((1, 4, 7, 10), labels)
    top.tick_params(length=0, labelsize=9)
    save_figure(plt, fig, filename)


def plot_hotspots(plt, directory, plots):
    rows = csv_read(directory / "hotspots.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True, layout="constrained")
    for ax, seed in zip(axes, (131, 157)):
        initial = [r for r in rows if r["scenario"] == f"seed{seed}_near" and r["method"] == "start" and r["measurement"] == "native_896"]
        for index, (key, label) in enumerate(zip(("top_0_1pct", "top_1pct", "top_5pct"), ("Top 0.1%", "Top 1%", "Top 5%"))):
            ax.bar([int(r["view"]) + (index - 1) * .23 for r in initial], [100 * float(r[key]) for r in initial],
                   width=.23, color=COLORS[index], label=label)
        ax.set(title=f"Seed {seed} Near · initial 896×512", xticks=range(5), ylim=(84, 101), xlabel="View (3 and 4 held out)")
        ax.axvline(2.5, ls=":", color="0.5")
    axes[0].set_ylabel("Share of native squared RGB error (%)")
    axes[1].legend(frameon=False, loc="lower right")
    save_figure(plt, fig, plots / "rgb-hotspot-share.png")

    fig, axes = plt.subplots(2, 3, figsize=(11, 4.85), layout="constrained")
    fig.suptitle("Seed 157 Near · native 896×512 error", fontsize=13)
    for ax, (key, view, name) in zip(axes.flat, ((k, v, n) for v in (1, 3) for k, n in
                                             (("start", "Start"), ("L2_448", "L2 448"), ("Charbonnier_448", "Charbonnier")))):
        image = directory / "captures" / f"seed157_near_{key}" / f"error_v{view}.png"
        # Unity encodes the clipped log of the normalized error in the first channel.
        art = ax.imshow(plt.imread(image)[..., 0], vmin=0, vmax=1, cmap="magma")
        ax.set_title(f"{name} · {'held' if view == 3 else 'train'} view {view}")
        ax.set_axis_off()
    scale = (0, 1, 10, 100, 999)
    bar = fig.colorbar(art, ax=axes, ticks=[math.log10(1 + v) / 3 for v in scale], shrink=.88, pad=.02)
    bar.ax.set_yticklabels(("0", "1×", "10×", "100×", "≥999×"))
    bar.set_label("Pixel RGB error / view GT RMS")
    save_figure(plt, fig, plots / "rgb-error-maps.png")


def plot_proxy_direction(plt, directory, plots, np):
    groups = defaultdict(list)
    for row in csv_read(directory / "proxy-direction.csv"):
        groups[row["scenario"], row["method"]].append(row)
    cases = list(dict.fromkeys(k[0] for k in groups))
    methods = list(dict.fromkeys(k[1] for k in groups))
    cosines = {}
    for key, rows in groups.items():
        xyz = sorted(rows, key=lambda r: r["axis"])[:3]
        proxy, native = (np.asarray([float(r[k]) for r in xyz]) for k in ("proxy_grad", "native_grad"))
        norm = np.linalg.norm(proxy) * np.linalg.norm(native)
        cosines[key] = (float(proxy @ native / norm) if norm > 0 else math.nan, float(rows[0]["full_gradient_cosine"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.1), sharey=True, layout="constrained")
    for dim, (ax, title) in enumerate(zip(axes, ("Position only", "Position and log-intensity"))):
        image = [[cosines[k, method][dim] for k in cases] for method in methods]
        art = ax.imshow(np.asarray(image), vmin=-1, vmax=1, cmap="RdBu", aspect="auto")
        for i, row in enumerate(image):
            for j, v in enumerate(row):
                ax.text(j, i, f"{v:+.2f}" if math.isfinite(v) else "–", va="center", ha="center",
                        color="white" if abs(v) > .6 else "#1e293b", fontsize=9)
        ax.set(xticks=range(len(cases)), xticklabels=[k.removeprefix("seed733_").replace("_", " ") for k in cases],
               yticks=range(len(methods)), yticklabels=[k.replace("_", " ") for k in methods], title=title)
    fig.colorbar(art, ax=axes, label="Proxy / native FD gradient cosine")
    save_figure(plt, fig, plots / "proxy-directions.png")


def report(args):
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source, dest = args.output.resolve(), args.report_dir.resolve()
    stages = {key: source / f"e21_{key}" for key in "abcde"}
    summaries = {key: json_read(folder / "summary.json") for key, folder in stages.items()}
    require(len({x["commit"] for x in summaries.values()}) == len({x["branch"] for x in summaries.values()}) == 1, "Mixed baselines")
    require(len({summaries[key]["sceneHash"] for key in ARM_COUNTS}) == 1, "Mixed E19 scene hashes")
    trials = {key: csv_read(stages[key] / "trials.csv") for key in ARM_COUNTS}
    for key in ARM_COUNTS:
        verify_arms(key, stages[key], summaries[key], trials[key])

    plots = dest / "figures"
    plots.mkdir(parents=True, exist_ok=True)
    rows = [dict(stage=key.upper(), **r) for key, table in trials.items() for r in table]
    csv_write(dest / "trials.csv", rows[0], (r.values() for r in rows))
    outcomes = {key: {"trials": len(table), "passed": sum(int(r["pass"]) for r in table),
                      "evaluations": sum(int(r["evaluations"]) for r in table)} for key, table in trials.items()}
    summary = {"parameterPass": {"positionCm": 5, "intensityRelative": .05, "legal": True},
               "outcomes": outcomes, "stages": summaries}
    (dest / "stage-summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with zipfile.ZipFile(dest / "raw-data.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for folder in stages.values():
            for file in sorted(folder.rglob("*")):
                if file.suffix in (".csv", ".json"):
                    archive.write(file, file.relative_to(source).as_posix())

    pooled, ratios = aggregate_gradients(stages["a"], dest, np)
    plot_gradients(plt, pooled, ratios, plots)
    plot_policy(plt, trials["b"], plots / "probe-policies.png", np)
    all_lamps = "seed733_near_L1_L2_L3_L4_L5_L6_L7"
    plot_recovery(plt, trials["c"], [(f"{all_lamps}_q{cap}", f"28D, Q ≤ {cap}") for cap in (520, 1575)] +
                  [(f"seed733_near_{p}_q520", p.replace("_", "+")) for p in ("L5_L7", "L2_L7")],
                  plots / "d0-recovery.png", ("C0 4D", "C1 BCD", "C2 Joint", "C3 Trust", "C4 Scale"))
    plot_recovery(plt, trials["d"], [(f"seed{n}_near", f"Seed {n} Near") for n in (131, 157)], plots / "rgb-recovery.png",
                  ("L2 448", "Huber", "Charbonnier", "L2 896", "Footprint"))
    plot_hotspots(plt, stages["d"], plots)
    plot_recovery(plt, trials["e"], [(r, r.removeprefix("seed733_").replace("_", " ")) for r in dict.fromkeys(t["scenario"] for t in trials["e"])],
                  plots / "proxy-recovery.png", ("RGB", "MSM", "MBD", "Ray soft", "D0", "MBD/ray", "MBD/self"))
    plot_proxy_direction(plt, stages["e"], plots, np)
    print(f"Verified and exported {len(rows)} arms in {dest}")

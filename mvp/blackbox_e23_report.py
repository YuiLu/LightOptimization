"""Verify native render ledgers and package the E23 experiment."""

import collections
import json
import math
import shutil
import statistics
import subprocess
import zipfile

from blackbox_e20 import ROOT, csv_read, csv_write, json_read
from blackbox_e21_report import require, save_figure
from blackbox_e23_selection import folder, trials


STAGES = ("a", "c-screen", "c-mix", "c-validate", "c-rgb", "d-atlas", "b-small", "b-full", "e-soft", "f-near", "f-best", "f-far", "f-classroom")
BASE = "d8a45f2da7c47ac597ceb7a13ed0109bd4ac4d8c"
COLORS = ("#0072b2", "#d55e00", "#009e73", "#93588a")
RESIDUAL_EXHIBITS = (
    ("c-validate", "seed131-hdr-residuals.png", "gt_on/gt_v0",
     (("seed131_near_on_C0/initial_v0", "Shared start"), ("seed131_near_on_C0/best_v0", "Raw FD"),
      ("seed131_near_on_C5/best_v0", "Raw + Log FD"))),
    ("b-full", "seed733-hard-holdout.png", "gt_on/gt_v3",
     (("gt_on/gt_v3", "Hard GT"), ("seed733_near_on_B0/best_v3", "Signed Log · On FD"),
      ("seed733_near_on_B2/best_v3", "Signed Log · Off XYZ"))),
)


def verify(path, summary, rows):
    snapshots = {a["id"]: a for a in summary["arms"]}
    require(set(snapshots) == {r["arm"] for r in rows}, f"{path}: inconsistent trial IDs")
    queries, objective, passes = collections.defaultdict(set), collections.Counter(), collections.Counter()
    off, soft = collections.Counter(), collections.Counter()
    for row in csv_read(path / "queries.csv"):
        key, q = row["arm"], int(row["query"])
        require(q not in queries[key] and row["collider_legal"] == row["survey_legal"] == "1"
                and math.isfinite(float(row["loss"])), f"{key}: invalid native objective query {q}")
        queries[key].add(q)
        objective[key] += 1
    if (path / "model-queries.csv").is_file():
        for row in csv_read(path / "model-queries.csv"):
            key, q = row["arm"], int(row["query"])
            require(q not in queries[key] and int(row["camera_renders"]) == 3, f"{key}: invalid model query {q}")
            queries[key].add(q)
            if row["shadows"] == "None":
                off[key] += 3
            if row["shadows"] == "Soft":
                soft[key] += 3
    for row in csv_read(path / "decisions.csv"):
        if row["accepted"] == "1":
            passes[row["arm"]] += 1
            require(float(row["actual"]) > 0 and float(row["rho"]) >= .02 and row["legal"] == "1",
                    f'{row["arm"]}: accepted step violates the native objective rule')
    for row in rows:
        key, cfg, m = row["arm"], row["setup"], row["metrics"]
        a = snapshots[key]
        expected = max(a["positionCm"]) <= 5 and max(a["intensityRelative"]) <= .05
        require(a["legal"] and a["pass"] == expected == (row["pass"] == "1"), f"{key}: bad pass flag")
        require(all(a["initial"][i] == a["final"][i] == truth for i, truth in enumerate(summary["truth"])
                    if i // 4 + 1 not in a["lamps"]), f"{key}: inactive lamps moved")
        count = sum(int(cfg[col + "_cameras"]) for col in ("on", "off", "soft", "olat"))
        require(count == 3 * a["evaluations"] == int(cfg["original_cameras"]) <= 3 * a["limit"] == int(cfg["cap_cameras"]),
                f"{key}: invalid camera budget")
        require(m["newOptimizationCameras"] == int(cfg["new_optimization_cameras"]), f"{key}: camera counts disagree")
        if cfg["reused_from"]:
            require(not queries[key] and m["newOptimizationCameras"] == 0, f"{key}: reuse rendered new optimization images")
        else:
            require(queries[key] == set(range(1, a["evaluations"] + 1)) and passes[key] == a["accepted"] and count == m["newOptimizationCameras"],
                    f"{key}: optimizer ledger is incomplete")
            require(objective[key] == a["objectiveQueries"] + a["nativeChecks"], f"{key}: native objective count disagrees")
            require(off[key] == int(cfg["off_cameras"]), f"{key}: Off was miscounted")
            expect_soft = count if cfg["jacobian"] == "soft_fd" else soft[key]
            require(expect_soft == int(cfg["soft_cameras"]), f"{key}: Soft was miscounted")
        require(m["peakWorkingBytes"] > 0 and m["peakGraphicsBytes"] > 0, f"{key}: missing memory samples")
    for row in csv_read(path / "loss-checks.csv"):
        scalar = float(row["scalar"])
        require(float(row["absolute_difference"]) <= 2e-6 * max(1e-7, scalar), f"{path}: loss and squared residual disagree")
        require(row["checkpoint"] != "gt" or scalar <= 1e-12, f"{path}: nonzero self-match")


def decomposition(source):
    a = folder(source, "a")
    rows = csv_read(a / "gradient-decomposition.csv")
    xyz = [r for r in rows if r["axis"] != "3" and r["stencil"] != "0"]
    closure = max(float(r["raw_closure_relative"]) for r in xyz)
    approx = max(float(r["chain_relative_error"]) for r in xyz if r["mapping"] == "C0" and math.isfinite(float(r["chain_relative_error"])))
    require(closure < 1e-5 and approx < 1e-5, "Shadow decomposition does not close or raw chain rule differs from exact FD")
    def median(group, column):
        return statistics.median([float(r[column]) for r in group if math.isfinite(float(r[column]))])

    cross_step = {str(mm): {key: median([r for r in xyz if r["mapping"] == "C0" and math.isclose(float(r["requested_step"]), mm / 1000)], col)
                            for key, col in (("on", "cross_20mm_on"), ("off", "cross_20mm_off"), ("shadow", "cross_20mm_shadow"))}
                  for mm in (5, 50)}
    chain = {key: median([r for r in xyz if r["mapping"] == key and math.isclose(float(r["requested_step"]), .02)], "chain_relative_error")
             for key in ("C0", "C1", "C3")}
    shadow_share = {}
    for pose in {r["pose"] for r in xyz} - {"gt"}:
        shadow_share[pose] = {}
        for key in ("C0", "C1", "C3"):
            group = [r for r in xyz if r["pose"] == pose and r["mapping"] == key and math.isclose(float(r["requested_step"]), .02)]
            gs = math.sqrt(sum(float(r["g_shadow"]) ** 2 for r in group))
            go = math.sqrt(sum(float(r["g_off"]) ** 2 for r in group))
            if go + gs:
                shadow_share[pose][key] = gs / (go + gs)
    return dict(maxRawClosure=closure, maxRawChainError=approx, medianChainError20mm=chain, medianCrossStepCosine=cross_step,
                fractionOfComponentGradientNormFromShadow=shadow_share)


def native_trials(source, key):
    path = folder(source, key)
    summary = json_read(path / "summary.json")
    metrics = json_read(path / "metrics.json")
    records = list(trials(source, key))
    verify(path, summary, records)
    summary["processPeakWorkingBytes"] = metrics["processPeakWorkingBytes"]
    summary["newOptimizationCameras"] = sum(r["metrics"]["newOptimizationCameras"] for r in records)
    summary["otherCameras"] = summary["nativeCameraRenders"] - summary["newOptimizationCameras"]
    return summary, records


def raw_provider_parity(source):
    path = folder(source, "b-small-smoke")
    _, rows = native_trials(source, "b-small-smoke")
    raw = {r["method"]: r["arm"] for r in rows if r["setup"]["mapping"] == "C0"}
    gram = csv_read(path / "gram.csv")
    jac = csv_read(path / "jacobians.csv")

    def first_normal(arm):
        matrix = [float(v) for row in gram if row["arm"] == arm and row["iteration"] == "1" for v in row["values"].split(";")]
        gradient = next(row["gradient_theta"] for row in jac if row["arm"] == arm and row["iteration"] == "1")
        return matrix, list(map(float, gradient.split(";")))

    direct, chain = first_normal(raw["B0"]), first_normal(raw["B1"])
    differences = [math.dist(a, b) / math.sqrt(sum(x * x for x in a)) for a, b in zip(direct, chain)]
    require(max(differences) < 1e-5, "Runtime Raw chained and exact mapped FD do not agree")
    return dict(case=raw["B0"], gramRelativeDifference=differences[0], gradientRelativeDifference=differences[1])


def table_rows(grouped, source):
    on = {r["arm"]: r for r in json_read(source / "inputs/on-metrics.json")["metrics"]}
    screen = {r["arm"]: r for r in json_read(folder(source, "c-screen") / "metrics.json")["metrics"]}
    hard_replays = collections.Counter(r["arm"] for r in csv_read(folder(source, "e-soft") / "hard-recheck.csv"))
    for key, (summary, records) in grouped.items():
        scene = summary["scene"]
        for row in records:
            m, cfg = row["metrics"], row["setup"]
            reused = cfg["reused_from"]
            former = (on if reused.startswith("E22-B-On") else screen)[reused.split("/", 1)[1]] if reused else m
            yield dict(stage=key, scene_hash=summary["sceneHash"], base_commit=summary["commit"], unity=scene["unity"], urp=scene["urp"],
                       gpu=scene["gpu"], **{k: v for k, v in row.items() if k not in ("setup", "metrics")},
                       **{k: v for k, v in cfg.items() if k not in ("arm", "optimization_seconds")},
                       scale=m["scale"], exposure=m["exposure"], mapped_start_nrmse=row["loss_start"], mapped_final_nrmse=row["loss_final"],
                       display_train_nrmse=m["displayTrain"], display_high_nrmse=m["displayHigh"], display_held_nrmse=m["displayHeld"],
                       optimization_seconds=former["optimizationSeconds"], independent_evaluation_cameras=m["validationCameras"],
                       original_hard_recheck_cameras=hard_replays[row["arm"]] if key == "e-soft" else 0,
                       peak_working_bytes=former["peakWorkingBytes"], peak_graphics_bytes=former["peakGraphicsBytes"])


def soft_evidence(source):
    soft = folder(source, "e-soft")
    evidence = {}
    for label, path, suffix in (("E0", folder(source, "c-screen"), "near_on_C0"), ("E2", soft, "near_E2")):
        decisions = csv_read(path / "decisions.csv")
        evidence[label] = {}
        for seed in (131, 157):
            actual = [float(r["actual"]) for r in decisions if r["arm"] == f"seed{seed}_{suffix}" and math.isfinite(float(r["actual"]))]
            evidence[label][str(seed)] = dict(nativeDescent=sum(x > 0 for x in actual), measured=len(actual))
    return evidence


def direction_plot(plt, source, dest):
    rows = csv_read(folder(source, "a") / "direction-probes.csv")
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.4), layout="constrained", sharey=True)
    for ax, seed in zip(axs, (131, 157, 733)):
        for model, color in zip(("B0", "B1", "B2", "B3"), COLORS):
            chosen = [r for r in rows if r["pose"] == f"seed{seed}_log_failed" and r["mapping"] == "C1" and r["model"] == model and r["iteration"] == "0"]
            ax.plot([float(r["length"]) for r in chosen], [100 * float(r["actual"]) / float(r["base_loss"]) for r in chosen],
                    color=color, marker="o", label=model)
        ax.axhline(0, color=".5", lw=.8)
        ax.set(title=f"Seed {seed}", xlabel="Proposal length multiplier", xticks=[.25, .5, 1])
        ax.grid(axis="y", alpha=.2)
    axs[0].set_ylabel("Measured signed-log loss reduction (%)")
    axs[-1].legend(ncol=2, frameon=False)
    save_figure(plt, fig, dest / "direction-replays.png")


def read_exr(np, path, width, height):
    data = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1",
                                    "-pix_fmt", "gbrapf32le", "-f", "rawvideo", "pipe:1"])
    # Unity's file is already planar float RGBA in the decoder, so no clamping conversion is needed.
    planes = np.frombuffer(data, dtype="<f4").reshape(4, height, width)
    return planes[[2, 0, 1]].transpose(1, 2, 0)


def diagnostic_images(plt, np, source, dest):
    a = folder(source, "c-rgb") / "captures"
    fig, axs = plt.subplots(2, 3, figsize=(11, 4.5), layout="constrained")
    for axes, name, label in zip(axs, ("seed131_start", "seed157_log_failed"), ("Seed 131 start", "Seed 157 Log failure")):
        hard = read_exr(np, a / name / "hard_v0.exr", 448, 256)
        off = read_exr(np, a / name / "off_v0.exr", 448, 256)
        for ax, shot, mode in zip(axes[:2], (hard, off), ("Hard On", "Off")):
            linear = -np.expm1(-np.maximum(shot, 0))
            display = np.where(linear < .0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - .055)
            ax.imshow(display.clip(0, 1))
            ax.set_title(f"{label} · {mode}", fontsize=9)
        image = axes[2].imshow(np.log10(1 + 100 * np.linalg.norm(hard - off, axis=-1)), cmap="magma", vmin=0, vmax=3)
        axes[2].set_title("Shadow response  log₁₀(1 + 100‖On−Off‖)", fontsize=9)
        for ax in axes:
            ax.set_axis_off()
    fig.colorbar(image, ax=axs[:, 2], shrink=.8)
    save_figure(plt, fig, dest / "shadow-response.png")

    for key, output, target, arms in RESIDUAL_EXHIBITS:
        if not (folder(source, key) / "summary.json").is_file():
            continue
        path = folder(source, key) / "captures"
        gt = read_exr(np, path / (target + ".exr"), 896, 512)
        fig, axs = plt.subplots(2, 3, figsize=(11, 4.7), layout="constrained")
        for col, (file, title) in enumerate(arms):
            shot = read_exr(np, path / (file + ".exr"), 896, 512)
            axs[0, col].imshow(plt.imread(path / (file + ".png")))
            axs[0, col].set_title(title)
            image = axs[1, col].imshow(np.log10(1 + 100 * np.linalg.norm(shot - gt, axis=-1)), cmap="magma", vmin=0, vmax=3)
        fig.colorbar(image, ax=axs[1, :], shrink=.62, label="log₁₀(1 + 100‖RGB − GT‖)")
        for ax in axs.flat:
            ax.set_axis_off()
        save_figure(plt, fig, dest / output)


def report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    source, dest = args.output.resolve(), args.export.resolve()
    stages = [key for key in STAGES if (folder(source, key) / "summary.json").is_file()]
    gates = {p.stem.removeprefix("decision-"): json_read(p) for p in source.glob("decision-*.json")}
    mandatory = {"a", "c-screen", "c-validate", "c-rgb", "d-atlas", "e-soft", "f-near"}
    if "C5" in gates["screen"]["scores"]:
        mandatory.add("c-mix")
    if gates["directions"]["selected"]:
        mandatory.add("b-small")
    if gates.get("hybrid-small", {}).get("expand"):
        mandatory.add("b-full")
    if gates.get("hybrid-full", {}).get("selected"):
        mandatory.add("f-best")
    if gates.get("neighbourhood", {}).get("expand"):
        mandatory.update(("f-far", "f-classroom"))
    require(mandatory <= set(stages), f"Missing stages: {sorted(mandatory - set(stages))}")
    grouped = {key: native_trials(source, key) for key in stages}
    expected_counts = {"a": 0, "b-small": 8, "b-full": 12, "c-screen": 10, "c-mix": 2, "c-validate": 9,
                       "c-rgb": 0, "d-atlas": 0, "e-soft": 6, "f-near": 14, "f-best": 7, "f-far": 3, "f-classroom": 3}
    base = json_read(source / "inputs/on-summary.json")
    require(all(s["sceneHash"] == base["sceneHash"] and s["truth"] == base["truth"] and
                len(r) == expected_counts[key] for key, (s, r) in grouped.items() if key != "f-classroom"),
            "E23 configuration or stage count differs from the protocol")
    if "f-classroom" in grouped:
        classroom, records = grouped["f-classroom"]
        require(classroom["scene"]["scene"] == "classroom" and len(records) == 3 and classroom["scene"]["lamps"] == 7,
                "Classroom transfer did not use the existing seven-light rig")
    if "f-best" in grouped:
        near = {a["scenario"]: a["initial"] for a in grouped["f-near"][0]["arms"] if a["method"] == "C0"}
        require(all(a["initial"] == near[a["scenario"]] for a in grouped["f-best"][0]["arms"]), "The best method used different neighbourhood starts")
        for r in csv_read(folder(source, "f-best") / "lamp-displacements.csv"):
            arm = r["arm"]
            radius = int(arm.split("_")[1].removesuffix("cm"))
            log_i = 0 if "position_only" in arm else .05
            require(abs(float(r["position_start_cm"]) - radius) < .01 and abs(abs(float(r["log_intensity_start"])) - log_i) < 1e-7,
                    f"{arm}: small-neighbourhood radius or intensity differs from the protocol")
    baseline = {arm["id"]: arm for arm in base["arms"]}
    for key in ("c-screen", "c-mix", "c-validate", "b-full", "e-soft"):
        if key in grouped:
            for arm in grouped[key][0]["arms"]:
                seed = arm["id"].split("_")[0]
                reference = baseline[f"{seed}_near_on_B0"]["initial"]
                require(len(arm["initial"]) == len(reference) and all(abs(x - y) <= 1e-12 for x, y in zip(arm["initial"], reference)),
                        f'{arm["id"]}: E22 start changed')
    gates["atlas"] = json_read(folder(source, "d-atlas") / "atlas-gate.json")
    if not gates["atlas"]["retentionPassed"]:
        gates["structured-on-skipped"] = "both isolation floors changed per-light shadow allocation"
    soft_rows = grouped["e-soft"][1]
    if not any(r["method"] == "E2" and r["pass"] == "1" for r in soft_rows):
        gates["soft-to-hard-skipped"] = "Soft XYZ brought no parameter recovery on the original Hard objective"
    if "b-full" not in grouped:
        gates["b-full-skipped"] = "8D did not meet the expansion criterion"
    if not any(r["clearBenefit"] for r in gates["tone"].values()):
        gates["c-independent-skipped"] = "no compressed mapping improved on Raw and preserved its success"
    if not gates.get("hybrid-full", {}).get("selected") and not any(r["clearBenefit"] for r in gates["tone"].values()):
        gates["f-expansion-skipped"] = "no method met the seven-light Hard On acceptance gate"
    elif "f-far" not in grouped:
        gates["f-expansion-skipped"] = "seven-light winner lost at least one recovery in the small-neighbourhood control"
    dest.mkdir(parents=True, exist_ok=True)
    data = list(table_rows(grouped, source))
    csv_write(dest / "trials.csv", data[0], (row.values() for row in data))
    stage_summary = dict(baseCommit=BASE, sha256Scene=base["sceneHash"], gates=gates, decomposition=decomposition(source),
                         hardDescentInSoftExperiment=soft_evidence(source),
                         stages={key: s for key, (s, _) in grouped.items()})
    raw_smoke = folder(source, "b-small-smoke")
    if (raw_smoke / "summary.json").is_file():
        stage_summary["rawProviderParity"] = raw_provider_parity(source)
    (dest / "stage-summary.json").write_text(json.dumps(stage_summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for key, name in (("a", "gradient-decomposition.csv"), ("a", "direction-probes.csv"), ("d-atlas", "atlas-layout.csv")):
        shutil.copyfile(folder(source, key) / name, dest / name)
    direction_plot(plt, source, dest)
    diagnostic_images(plt, np, source, dest)
    with zipfile.ZipFile(dest / "raw-data.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archived = stages + (["b-small-smoke"] if "rawProviderParity" in stage_summary else [])
        for key in archived:
            path = folder(source, key)
            for file in sorted(path.rglob("*")):
                if file.suffix in (".json", ".csv") or key == "c-rgb" and file.name in ("hard_v0.exr", "off_v0.exr"):
                    archive.write(file, file.relative_to(source).as_posix())
        for key, _, gt, arms in RESIDUAL_EXHIBITS:
            if key not in grouped:
                continue
            for name in sorted({gt, *(file for file, _ in arms)}):
                for suffix in (".exr", ".png"):
                    file = folder(source, key) / "captures" / (name + suffix)
                    archive.write(file, file.relative_to(source).as_posix())
        for file in sorted(source.glob("decision-*.json")):
            archive.write(file, file.name)
        with zipfile.ZipFile(ROOT / "docs/blackbox-e22/raw-data.zip") as e22:
            for name in ("queries.csv", "jacobians.csv", "decisions.csv", "gram.csv", "damping.csv", "metrics.json", "summary.json"):
                member = "e22_b_on/" + name
                archive.writestr("reference/" + member, e22.read(member))
        archive.write(dest / "stage-summary.json", "stage-summary.json")
    with zipfile.ZipFile(dest / "raw-data.zip") as archive:
        require(archive.testzip() is None, "E23 results archive failed its integrity check")
    print(f"Verified {len(data)} trials and {len(stages)} stages in {dest}")

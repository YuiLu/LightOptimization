"""Run and report the Unity E20 independent-light and moment-shadow experiments."""

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("e20-a", "e20-direct", "e20-msm")


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_write(path, headers, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def prepare(args):
    project = args.project.resolve()
    output = args.output.resolve()
    project.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("Packages", "ProjectSettings", "Assets/LightingMvp"):
        shutil.copytree(ROOT / "Light" / name, project / name, dirs_exist_ok=True)
    shutil.copytree(args.settings, project / "Assets/Settings", dirs_exist_ok=True)
    source = ROOT / "Light/Assets/LightingMvp"
    files = [p for p in source.rglob("*") if p.suffix in {".cs", ".shader", ".compute"}]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for name in ("URP-HighFidelity.asset", "URP-HighFidelity-Renderer.asset"):
        hashes[f"source_settings/{name}"] = hashlib.sha256((args.settings / name).read_bytes()).hexdigest()
    (output / "source-files-sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")


def run(args):
    if not args.prepared:
        if not args.settings:
            raise ValueError("--settings must point to the E19 project Assets/Settings directory.")
        prepare(args)
    project = args.project.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    branch, commit = git("rev-parse", "--abbrev-ref", "HEAD"), git("rev-parse", "HEAD")
    unity = [str(args.unity.resolve()), "-batchmode", "-force-d3d11", "-job-worker-count", "4", "-projectPath", str(project)]

    scene = project / "Assets/Scenes/UrpNativeComplexRoom.unity"
    if not scene.is_file():
        subprocess.run(unity + ["-quit", "-executeMethod", "LightingMvp.Editor.NativeComplexRoomBuilder.BuildMenu",
                               "-logFile", str(output / "build.log")], check=True)
        if not scene.is_file():
            raise RuntimeError("Unity did not create the E19 complex room.")

    for stage in args.stages:
        marker = output / stage.replace("-", "_") / "summary.json"
        if marker.is_file():
            if args.resume:
                continue
            raise FileExistsError(f"{marker} exists. Use --resume or a fresh directory.")
        log = output / (stage + ".log")
        print(f"Running {stage} with Unity. Log: {log}", flush=True)
        upstream = output / "e20_a"
        references = ["-bbE20B0Source", str(upstream)] if stage == "e20-direct" and (upstream / "summary.json").is_file() else []
        subprocess.run(unity + ["-executeMethod", "LightingMvp.Editor.NativeRecoveryExperiment.BatchRun", "-bbCase", stage,
                               "-bbOutput", str(output), "-bbE20Branch", branch, "-bbE20BaseCommit", commit, "-logFile", str(log), *references],
                       check=True)
        if not marker.is_file():
            raise RuntimeError(f"No summary for {stage}. See {log}.")


def vectors(text):
    return [float(x) if x else math.nan for x in text.split(";")]


def verify(directory):
    config = json_read(directory / "config.json")
    summary = json_read(directory / "summary.json")
    ledger = {r["purpose"]: int(r["camera_renders"]) for r in csv_read(directory / "capture-counts.csv")}
    trials = {row["id"]: row for row in summary["arms"]}
    assert len(trials) == len(summary["arms"])
    assert summary["cameraRenders"] == sum(ledger.values())
    assert summary["optimizationCameraRenders"] == sum(a["cameraRenders"] for a in trials.values())
    assert summary["validationCameraRenders"] == sum(a["validationRenders"] for a in trials.values())
    rendered = defaultdict(int)
    proxy_faces = defaultdict(int)
    last_native_accept = {}
    for q in csv_read(directory / "queries.csv"):
        trial = trials.get(q["arm"])
        if not trial:
            continue
        if q["kind"] == "optimizer":
            rendered[q["arm"]] += int(q["native_camera_renders"])
        if q["kind"] == "proxy":
            proxy_faces[q["arm"]] += int(q["shadow_faces"])
        if q["kind"] == "decision" and q["accepted"] == "1" and trial["method"] in ("B2", "MSM_4D"):
            loss = float(q["native_rgb_loss"])
            assert loss < last_native_accept.get(q["arm"], trial["nativeInitial"])
            last_native_accept[q["arm"]] = loss
        if q["theta"]:
            x = vectors(q["theta"])
            assert all(0 <= u <= 1 for u in x) and len(x) == len(config["truth"])
            assert all(x[i] == t for i, t in enumerate(config["truth"]) if i // 4 + 1 not in trial["lamps"])
    for key, a in trials.items():
        native_cap = a.get("nativeBudget", a["budget"])
        proxy_cap = a.get("proxyBudget", 0)
        assert a["cameraRenders"] == rendered[key] == a["nativeQueries"] * len(config["train"])
        assert a["shadowFaces"] == proxy_faces[key]
        assert a["nativeQueries"] <= native_cap and a["proxyQueries"] <= proxy_cap
        assert math.isfinite(a["recheck"]) and math.isfinite(a["nativeBest"])
        assert len(a["perViewNrmse"]) == len(config["train"]) + len(config["held"])
        if key in last_native_accept:
            assert math.isclose(a["nativeBest"], last_native_accept[key], rel_tol=1e-6, abs_tol=1e-8)
        if a["method"] in ("A1", "A2", "A3", "oracle", "B0", "M0"):
            assert math.isclose(a["recheck"], a["best"], rel_tol=1e-6, abs_tol=1e-8)
    return config, summary


def recovered_probe_steps(directory, output):
    """Join solver batches to actual query indices, including older E20-A ledgers."""
    queries = {(q["arm"], int(q["query"])): q["phase"] for q in csv_read(directory / "queries.csv")
               if q["kind"] == "optimizer"}
    lamps = {row["arm"]: [int(x) for x in row["lamps"].split(";")] for row in csv_read(directory / "arms.csv")}
    steps = []
    for j in csv_read(directory / "jacobians.csv"):
        if j["phase"] != "solver_normalized":
            continue
        arm = j["arm"]
        axes = [(l, xyz) for l in lamps[arm] for xyz in ("x", "y", "z", "log_intensity")]
        stencils = list(map(int, j["stencils"].split(";")))
        executed = [0, 1, 2, 2]
        index = int(j["query"]) - sum(executed[s] for s in stencils)
        for (lamp, axis), h, stencil in zip(axes, vectors(j["physical_steps"]), stencils):
            for d in ((), (h,), (h, 2*h), (h, -h))[stencil]:
                index += 1
                phase = queries[arm, index]
                assert phase.startswith("jac_")
                steps.append((arm, j["batch"], index, phase, lamp, axis, "log" if axis == "log_intensity" else "m", d))
    csv_write(output, ("arm", "batch", "query", "phase", "lamp", "axis", "unit", "signed_step"), steps)


def peak_neighbourhood(a, direct, config, output):
    receivers = {(int(r["view"]), int(r["pixel"])): [float(r[x]) for x in "xyz"]
                 for r in csv_read(direct / "receivers.csv")}
    starts = {(s["seed"], s["profile"]): s["theta"] for s in config["starts"]}
    bounds = config["scene"]
    lo = [bounds["boundsMin"][axis] for axis in "xyz"]
    hi = [bounds["boundsMax"][axis] for axis in "xyz"]
    rows = []
    for r in csv_read(a / "error-concentration.csv"):
        seed, profile, view = int(r["seed"]), r["profile"], int(r["view"])
        x, y = int(r["max_error_x"]), int(r["max_error_y"])
        # Receivers use the centre pixel of each fixed 4x4 camera cell.
        sx, sy = (x // 4) * 4 + 2, (y // 4) * 4 + 2
        surface = receivers.get((view, sy * bounds["width"] + sx))
        if surface is None:
            continue
        theta = starts[seed, profile]
        positions = [[lo[a] + (hi[a] - lo[a]) * theta[4*l + a] for a in range(3)] for l in range(len(theta) // 4)]
        closest = min(range(len(positions)), key=lambda l: math.dist(positions[l], surface))
        rows.append((seed, profile, view, x, y, sx, sy, math.hypot(x - sx, y - sy), *surface,
                     closest + 1, math.dist(positions[closest], surface) * 100))
    csv_write(output, ("seed", "profile", "view", "peak_x", "peak_y", "sample_x", "sample_y", "pixel_distance",
                      "sample_world_x", "sample_world_y", "sample_world_z", "nearest_light", "distance_cm"), rows)


def axis_errors_cm(config, arm):
    scene = config["scene"]
    metres = [scene["boundsMax"][axis] - scene["boundsMin"][axis] for axis in "xyz"]
    return tuple(
        ";".join(f"{100 * scale * (arm['theta'][i] - config['truth'][i]):.9g}"
                 for i in (4 * (lamp - 1) + axis for lamp in arm["lamps"]))
        for axis, scale in enumerate(metres)
    )


def plots(a, b, m, out, data, fallback_seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    def save(fig, name):
        fig.savefig(out / (name + ".png"), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def grid(ax):
        ax.grid(alpha=.20)

    methods = ("A1", "A2", "A3", "oracle")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for ax, profile in zip(axes, ("near", "far")):
        selected = [[t for t in data[0]["arms"] if t["method"] == method and t["profile"] == profile
                     and (profile != "near" or t["seed"] not in fallback_seeds)] for method in methods]
        count = [(sum(t["passed"] for t in group), len(group)) for group in selected]
        values = [passed / total for passed, total in count]
        ax.bar(methods, values, color=["#1c7ed6", "#0ca678", "#e8590c", "#7048e8"])
        for index, ((p, n), ratio) in enumerate(zip(count, values)):
            ax.text(index, ratio + .03, f"{p}/{n}", ha="center")
        ax.set_ylim(0, 1.15)
        ax.set_title(profile.capitalize())
        ax.set_ylabel("Fraction meeting 5 cm & 5%")
        grid(ax)
    fig.suptitle("Native Shadow Off recovery · same E19 starts and camera budget")
    save(fig, "recovery-by-dimension")

    jac = [r for r in csv_read(a / "jacobians.csv") if r["phase"] == "physical_fd" and float(r["position_step_m"]) == .02]
    anchors = ("gt", "seed131_near", "seed733_near", "seed733_far")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), layout="constrained")
    for ax, label in zip(axes.ravel(), anchors):
        for r in (item for item in jac if item["arm"] == label):
            s = np.asarray(vectors(r["singular_values"]))
            view = int(r["view"])
            ax.semilogy(np.arange(1, len(s) + 1), np.maximum(1e-12, s),
                        lw=2 if view == -1 else 1, label="3 training views" if view == -1 else f"view {view}")
        ax.set_title(label.replace("_", " "))
        ax.set_xlabel("Singular value index")
        ax.set_ylabel("Singular value")
        grid(ax)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Physical Jacobian spectrum · 2 cm / 0.05 log intensity")
    save(fig, "singular-spectrum")

    matrix = csv_read(a / "column-matrices.csv")
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), layout="constrained")
    for ax, label in zip(axes.ravel(), anchors):
        c = [vectors(r["correlation_row"]) for r in matrix if r["anchor"] == label and r["scale"] == "1" and r["view"] == "-1"]
        im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(label.replace("_", " "))
        for boundary in np.arange(3.5, 27, 4):
            ax.axhline(boundary, color="black", lw=.35, alpha=.6)
            ax.axvline(boundary, color="black", lw=.35, alpha=.6)
        ax.set_xticks(np.arange(1.5, 28, 4), [f"L{i}" for i in range(1, 8)])
        ax.set_yticks(np.arange(1.5, 28, 4), [f"L{i}" for i in range(1, 8)])
    fig.colorbar(im, ax=axes, shrink=.8, label="Column cosine")
    fig.suptitle("Within-lamp and cross-lamp Jacobian correlation")
    save(fig, "jacobian-correlation")

    scan = [r for r in csv_read(a / "direction-scans.csv") if r["legal"] == "1"]
    selected = [r for r in scan if r["direction"] == "negative_gradient" and r["arm"].startswith("A3_")]
    if selected:
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        for arm in sorted({r["arm"] for r in selected}):
            rows = sorted((r for r in selected if r["arm"] == arm), key=lambda r: float(r["step"]))
            x = [float(r["step"]) for r in rows]
            centre = float(rows[0]["centre_loss"])
            actual = [float(r["loss"]) / centre - 1 for r in rows]
            model = [float(r["predicted_derivative"]) * h / centre for r, h in zip(rows, x)]
            line, = ax.plot(x, actual, "o-", label=f"{rows[0]['seed']} measured")
            ax.plot(x, model, "--", color=line.get_color(), alpha=.65)
        ax.axhline(0, color="black", lw=.7)
        ax.set_xlabel("Signed step in normalized theta")
        ax.set_ylabel("Relative change in squared native loss")
        ax.set_title("Failed Near starts · negative RGB gradient · dashed linear prediction")
        grid(ax)
        ax.legend()
        save(fig, "gradient-direction-scans")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, key, title, scale in zip(axes, ("positionCm", "intensityRelative"), ("Position error (cm)", "Intensity error (%)"), (1, 100)):
        for i, name in enumerate(("B0", "B1", "B2")):
            rows = [r for r in data[1]["arms"] if r["method"] == name]
            for profile, marker in (("near", "o"), ("far", "x")):
                values = [r[key][0] * scale for r in rows if r["profile"] == profile]
                x = np.full(len(values), i) + np.linspace(-.13, .13, len(values))
                ax.scatter(x, np.maximum(values, 1e-4), marker=marker, alpha=.65, label=profile if i == 0 else None)
        ax.axhline(5, ls="--", lw=1, color="#777777")
        ax.set_xticks(range(3), ("B0 Native", "B1 Direct", "B2 Accepted"))
        ax.set_yscale("log")
        ax.set_ylabel(title)
        ax.legend()
        grid(ax)
    fig.suptitle("Direct-light proxy and original Unity RGB")
    save(fig, "direct-proxy-recovery")

    direct_fd = [r for r in csv_read(b / "direct-fd.csv") if r["axis"] == "x" and r["direction_cosine"]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True, layout="constrained")
    shades = plt.get_cmap("tab10")
    for ax, profile in zip(axes, ("near", "far")):
        for seed, marker in (("131", "o"), ("733", "x")):
            for lamp in range(1, 8):
                rows = [r for r in direct_fd if r["lamp"] == str(lamp) and r["profile"] == profile and r["seed"] == seed]
                ax.plot([int(r["scale"]) for r in rows], [float(r["direction_cosine"]) for r in rows],
                        color=shades(lamp - 1), marker=marker, alpha=.7, label=f"L{lamp}" if seed == "131" else None)
        ax.axhline(0, color="black", lw=.65)
        ax.set_xticks(range(3), ("0.5 cm / 0.02", "2 cm / 0.05", "5 cm / 0.10"))
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(profile.capitalize())
        ax.set_xlabel("Position / log-intensity FD step")
        grid(ax)
    axes[0].set_ylabel("Direct and native gradient cosine")
    axes[0].legend(fontsize=7, ncol=2, loc="lower left")
    fig.suptitle("Direct-light and native RGB gradients · ○ seed 131 · × seed 733")
    save(fig, "direct-gradient-agreement")

    selection = json_read(m / "gate.json")["lamp"]
    curves = [r for r in csv_read(m / "moment-native-direction.csv") if int(r["lamp"]) == selection]
    keys = list(dict.fromkeys((r["seed"], r["profile"]) for r in curves))
    if keys:
        fig, axes = plt.subplots(1, len(keys), squeeze=False, figsize=(3.5 * len(keys), 3.5), layout="constrained")
        for ax, key in zip(axes.ravel(), keys):
            for condition in ("on", "off"):
                rows = [r for r in curves if (r["seed"], r["profile"]) == key and r["shadows"] == condition]
                if not rows:
                    continue
                xs = [100 * float(r["step_m"]) for r in rows]
                relative = lambda name: [100 * (float(r[name]) / float(r["centre_native_loss"]) - 1) if r[name] else math.nan for r in rows]
                line, = ax.plot(xs, relative("plus_native_loss"), "o-", label=condition + " +d")
                ax.plot(xs, relative("minus_native_loss"), "x--", color=line.get_color(), label=condition + " −d")
            ax.axhline(0, color="black", lw=.7)
            ax.set_title(" ".join(key))
            ax.set_xlabel("Step (cm)")
            grid(ax)
        axes[0, 0].set_ylabel("Change in native squared loss (%)")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("Native RGB along the finite-difference MSM direction")
        save(fig, "moment-native-direction")

        fig, axes = plt.subplots(1, len(keys), squeeze=False, figsize=(3.5 * len(keys), 3.5), layout="constrained")
        for ax, key in zip(axes.ravel(), keys):
            for condition in ("on", "off"):
                rows = [r for r in curves if (r["seed"], r["profile"]) == key and r["shadows"] == condition
                        and r["plus_minus_derivative"]]
                ax.plot([100 * float(r["step_m"]) for r in rows],
                        [float(r["plus_minus_derivative"]) for r in rows], "o-", label=condition)
            ax.axhline(0, color="black", lw=.65)
            ax.set_xscale("log")
            ax.set_xlabel("Step (cm)")
            ax.set_title(" ".join(key))
            grid(ax)
        axes[0, 0].set_ylabel("Native symmetric derivative (squared NRMSE / m)")
        axes[0, 0].legend()
        fig.suptitle("Native directional derivatives along −∇MSM")
        save(fig, "moment-native-derivative")

    margins = [r for r in csv_read(m / "moment-fd.csv") if int(r["lamp"]) == selection and
               r["axis"] == "x" and r["cosine_to_previous_scale"]]
    comparison_steps = sorted({float(r["step_m"]) * 100 for r in margins})
    floor = min(0, min(float(r["cosine_to_previous_scale"]) for r in margins) - .05)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.3), sharey=True, sharex=True, layout="constrained")
    for ax, (seed, profile) in zip(axes.ravel(), keys):
        for key in dict.fromkeys(r["setting"] for r in margins):
            selected = [r for r in margins if r["setting"] == key and (r["seed"], r["profile"]) == (seed, profile)]
            ax.plot([float(r["step_m"]) * 100 for r in selected], [float(r["cosine_to_previous_scale"]) for r in selected], ".-", label=key)
        ax.set_xscale("log")
        ax.set_xticks(comparison_steps, [f"{step:g}" for step in comparison_steps])
        ax.set_ylim(floor, 1.05)
        ax.set_title(f"{seed} {profile}")
        grid(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Gradient cosine")
    for ax in axes[-1]:
        ax.set_xlabel("Larger of adjacent step sizes (cm)")
    axes[0, 1].legend(fontsize=7, ncol=2, loc="lower right")
    fig.suptitle("MSM and VSM · stability of estimated shadow gradients")
    save(fig, "moment-gradient-stability")

    key = f"MSM_XYZ_733_near_L{selection}"
    base = m / "captures" / f"moment_L{selection}_msm128_4x_k5"
    fit_folder = m / "captures" / key
    triplets = [
        ("Native RGB", [m / "captures/full_on/gt_v0.png", fit_folder / "initial_v0.png", fit_folder / "best_v0.png"]),
        ("Direct RGB feature", [b / f"captures/direct_L{selection}/gt_observed_direct_v0.png",
                                b / f"captures/B2_733_near_L{selection}/initial_direct_v0.png",
                                b / f"captures/B2_733_near_L{selection}/best_direct_v0.png"]),
        ("MSM visibility", [base / "gt_visibility_v0.png", fit_folder / "initial_visibility_v0.png", fit_folder / "best_visibility_v0.png"]),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(10, 6.5), layout="constrained")
    for i, (label, paths) in enumerate(triplets):
        for j, (ax, path) in enumerate(zip(axes[i], paths)):
            with Image.open(path) as img:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(label)
            if i == 0:
                ax.set_title(("GT", "Initial", "Best")[j])
    fig.suptitle(f"L{selection} · view 0 · seed 733 Near")
    save(fig, "features-and-native-rgb")

    folders = (a / "captures/A3_733_near_L1-2-3-4-5-6-7",
               b / f"captures/B2_733_near_L{selection}", fit_folder)
    fig, axes = plt.subplots(3, 5, figsize=(12, 5.4), layout="constrained")
    for i, (label, folder) in enumerate(zip(("A3", "B2", "MSM XYZ"), folders)):
        for v, ax in enumerate(axes[i]):
            with Image.open(folder / f"best_error_v{v}.png") as img:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if v == 0:
                ax.set_ylabel(label)
            if i == 0:
                ax.set_title(f"View {v}" + (" · held" if v >= 3 else ""))
    fig.suptitle("Native RGB residual images after fitting · seed 733 Near")
    save(fig, "native-rgb-error-by-view")

    fig, axes = plt.subplots(2, 6, figsize=(11, 4), layout="constrained")
    for row, filter_name in enumerate(("raw", "filtered")):
        for i, face in enumerate(("px", "nx", "py", "ny", "pz", "nz")):
            ax = axes[row, i]
            with Image.open(base / f"gt_{filter_name}_{face}.png") as img:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_ylabel(filter_name)
            if row == 0:
                ax.set_title(face)
    fig.suptitle(f"L{selection} · point-light moment faces · R = depth, G = depth², B = depth³")
    save(fig, "moment-cube")


def report(args):
    root = args.output.resolve()
    out = args.report_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    directories = [root / stage.replace("-", "_") for stage in STAGES]
    checked = [verify(path) for path in directories]
    assert len({cfg["sceneSha256"] for cfg, _ in checked}) == 1
    rows = []
    for directory, (config, summary) in zip(directories, checked):
        fallback = {s["seed"] for s in config["starts"] if s["nearFallback"]}
        recovered_probe_steps(directory, directory / "exact-native-fd-steps.csv")
        for a in summary["arms"]:
            rows.append((summary["stage"], a["id"], a["method"], a["seed"], a["profile"], int(a["profile"] == "near" and a["seed"] in fallback),
                         ";".join(map(str, a["lamps"])), int(a["passed"]),
                         a["nativeInitial"]**.5, a["nativeBest"]**.5, a["trainHighNrmse"], a["heldNrmse"],
                         max(a["positionCm"]), max(a["intensityRelative"]), *axis_errors_cm(config, a),
                         a["nativeQueries"], a["proxyQueries"], a["shadowFaces"], a["seconds"], a["stop"]))
    a_config, a_results = checked[0]
    prior = {a["id"]: a for a in a_results["arms"]}
    fallback_seeds = {s["seed"] for s in a_config["starts"] if s["nearFallback"]}
    paired = {(r["seed"], r["profile"], r["lamps"][0]): r for r in checked[1][1]["arms"] if r["method"] == "B2"}
    for ref in csv_read(directories[1] / "b0-reference.csv"):
        a = prior[ref["baseline_arm"]]
        assert ref["baseline_scene_sha256"] == a_config["sceneSha256"]
        assert a["method"] == "A1" and str(a["seed"]) == ref["seed"] and str(a["lamps"][0]) == ref["lamp"] and a["profile"] == ref["profile"]
        assert math.isclose(paired[a["seed"], a["profile"], a["lamps"][0]]["nativeInitial"], a["nativeInitial"], rel_tol=1e-6, abs_tol=1e-8)
        checked[1][1]["arms"].append(dict(a, method="B0"))
        rows.append(("e20-direct", a["id"], "B0_reference", a["seed"], a["profile"], int(a["profile"] == "near" and a["seed"] in fallback_seeds), ref["lamp"], int(a["passed"]),
                     a["nativeInitial"]**.5, a["nativeBest"]**.5, a["trainHighNrmse"], a["heldNrmse"],
                     max(a["positionCm"]), max(a["intensityRelative"]), *axis_errors_cm(a_config, a), a["nativeQueries"], 0, 0, a["seconds"], a["stop"]))
    csv_write(out / "trials.csv", ("stage", "arm", "method", "seed", "profile", "near_fallback", "lamps", "pass", "native_start_nrmse", "native_best_nrmse",
                                 "high_train_nrmse", "held_nrmse", "max_position_cm", "max_relative_intensity_error", "dx_cm", "dy_cm", "dz_cm", "native_queries",
                                 "proxy_queries", "shadow_face_renders", "seconds", "stop"), rows)
    peak_neighbourhood(directories[0], directories[1], a_config, directories[0] / "peak-neighbourhood.csv")
    plots(*directories, out, [s for _, s in checked], fallback_seeds)
    with zipfile.ZipFile(out / "query-data.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for directory in directories:
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix in {".csv", ".json"}:
                    archive.write(path, path.relative_to(root).as_posix())
        source = root / "source-files-sha256.json"
        if source.is_file():
            archive.write(source, source.name)
    print(f"Verified {len(rows)} trial ledgers and wrote {out}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    start = sub.add_parser("run")
    start.add_argument("--unity", type=Path, required=True)
    start.add_argument("--settings", type=Path)
    start.add_argument("--project", type=Path, required=True)
    start.add_argument("--output", type=Path, required=True)
    start.add_argument("--stages", nargs="+", choices=STAGES + tuple(s + "-smoke" for s in STAGES), default=STAGES)
    start.add_argument("--prepared", action="store_true")
    start.add_argument("--resume", action="store_true")
    start.set_defaults(handler=run)
    result = sub.add_parser("report")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--report-dir", type=Path, required=True)
    result.set_defaults(handler=report)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

"""Generate official MBD reference vectors and run the second E20 study."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from blackbox_e20 import ROOT, csv_read, csv_write, git


STAGES = ("reference", "shadow", "recovery", "irradiance", "separation", "topology")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def reference(args):
    if args.official_python:
        sys.path.insert(0, str(args.official_python.resolve()))
    import numpy as np
    from diffmoments import extension_build_type
    from diffmoments.np import compute_moment_bounds, compute_moment_bounds_backward, compute_singularities

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261126)
    cases = []
    for n in (1, 2):
        def add(group, m, eta, bias, blend):
            cases.append((f"{len(cases):04d}", group, n, *map(float, np.float32([eta, bias, blend, *m]))))

        def powers(depth, weights, mass=1):
            return [mass * np.dot(weights, np.asarray(depth)**k) for k in range(2 * n + 1)]

        for _ in range(480):
            weights = rng.dirichlet(np.ones(8))
            depth = rng.uniform(-1, 1, 8)
            add("random", powers(depth, weights, rng.uniform(.7, 1.4)), rng.uniform(-1.1, 1.1), .001, rng.choice([0., .15, .5, 1.]))
        for layer in (1, 2):
            depths = [.15] if layer == 1 else [-.32, .67]
            weights = [1.] if layer == 1 else [.65, .35]
            for eta in np.linspace(-.9, .95, 31):
                for bias in (.0001, .001, .01):
                    add(f"layers_{layer}", powers(depths, weights), eta, bias, .15)
        for spread in (0, .00005, .00025, .002, .01):
            for eta in (.274, .299, .30, .301, .326):
                for bias in (0., 1e-7, 1e-6, 1e-5):
                    add("near_singular_hankel", powers([.3-spread, .3+spread], [.35, .65]), eta, bias, .15)
        known = [1, 0, 1] if n == 1 else [1, 0, 1, 0, 2]
        center = 0. if n == 1 else 1.
        for eta in (center, -center):
            for offset in (0, -1e-7, 1e-7, -1e-5, 1e-5, -1e-3, 1e-3):
                add("infinite_root" if offset == 0 else "near_infinite_root", known, eta+offset, 0, .3)
        for _ in range(40):
            moments = np.float32(powers(rng.uniform(-1, 1, 6), rng.dirichlet(np.ones(6))))
            singular = compute_singularities(moments.reshape(-1, 1), .001)[:, 0]
            for eta in singular:
                add("official_singularity", moments, eta, .001, .15)
        add("invalid_hankel", [1, 0, -.125, 0, 1][:2 * n + 1], .2, 0, .3)
        add("invalid_hankel", [1, 0, 0, 0, 0][:2 * n + 1], 0, 0, .3)

    headers = ("id", "group", "order", "eta", "bias", "blend", "m0", "m1", "m2", "m3", "m4")
    csv_write(out / "reference-input.csv", headers, [(*r, *([0] * (11 - len(r)))) for r in cases])
    golden = {}
    timing = []
    for n in (1, 2):
        chosen = [r for r in cases if r[2] == n]
        for bias, blend in sorted({(r[4], r[5]) for r in chosen}):
            batch = [r for r in chosen if (r[4], r[5]) == (bias, blend)]
            for suffix, dtype in (("fp32", np.float32), ("fp64", np.float64)):
                moments = np.ascontiguousarray(np.asarray([r[6:] for r in batch], dtype=dtype).T)
                eta = np.asarray([r[3] for r in batch], dtype=dtype)
                before = time.perf_counter()
                bound, ctx = compute_moment_bounds(moments, eta, bias, blend, return_backward_context=True)
                forward_time = time.perf_counter() - before
                before = time.perf_counter()
                d_moments, d_eta = compute_moment_bounds_backward(ctx, np.ones_like(eta))
                timing.append((n, len(batch), suffix, bias, blend, forward_time, time.perf_counter() - before))
                for i, row in enumerate(batch):
                    vector = [bound[i], *d_moments[:, i], *([0] * (4 - 2 * n)), d_eta[i]]
                    golden.setdefault(row[0], {})[suffix] = vector
            random_cases = [i for i, r in enumerate(batch) if r[1] == "random"]
            if not random_cases:
                continue
            moments = np.ascontiguousarray(moments[:, random_cases])
            eta = eta[random_cases]
            h = 1e-6
            derivatives = []
            for j in range(2 * n + 2):
                plus, minus = moments.copy(), moments.copy()
                ep, em = eta.copy(), eta.copy()
                if j == len(moments):
                    ep += h
                    em -= h
                else:
                    plus[j] += h
                    minus[j] -= h
                derivatives.append((compute_moment_bounds(plus, ep, bias, blend) - compute_moment_bounds(minus, em, bias, blend)) / (2 * h))
            for i, index in enumerate(random_cases):
                row = batch[index]
                d = [v[i] for v in derivatives]
                golden[row[0]]["fd64"] = [*d[:-1], *([0] * (4 - 2 * n)), d[-1]]
    labels = ["bound", "m0", "m1", "m2", "m3", "m4", "eta"]
    csv_write(out / "official-reference.csv", ["id", *[f"{precision}_{label}" for precision in ("fp32", "fp64") for label in labels],
                                                *[f"fd64_{label}" for label in labels[1:]]],
              [(r[0], *golden[r[0]]["fp32"], *golden[r[0]]["fp64"], *golden[r[0]].get("fd64", [""] * 6)) for r in cases])
    csv_write(out / "official-timing.csv", ("order", "samples", "dtype", "bias", "blend", "forward_s", "backward_s"), timing)
    write_json(out / "official.json", {"package": "diffmoments", "version": importlib.metadata.version("diffmoments"),
                                      "build": extension_build_type(), "rng": 20261126, "samples": len(cases)})
    print(f"Generated {len(cases)} vectors and official CPU FP32 / FP64 results in {out}.", flush=True)


def prepare(args):
    destination = args.project.resolve()
    for name in ("Packages", "ProjectSettings", "Assets/LightingMvp"):
        shutil.copytree(ROOT / "Light" / name, destination / name, dirs_exist_ok=True)
    for name in ("Settings", "Scenes"):
        shutil.copytree(args.e19_project / "Assets" / name, destination / "Assets" / name, dirs_exist_ok=True)
    if args.classroom_project:
        name = "Assets/ClassroomScale"
        shutil.copytree(args.classroom_project / name, destination / name, dirs_exist_ok=True)
    digests = {}
    for path in sorted((destination / "Assets").rglob("*")):
        if path.suffix.lower() in {".cs", ".hlsl", ".compute", ".shader", ".unity", ".asset", ".bytes", ".json"}:
            digests[path.relative_to(destination).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(destination / "mbd-source-sha256.json", digests)
    print(f"Prepared {destination}.", flush=True)


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        stages = tuple(f"{s}-smoke" for s in args.stages if s not in {"reference", "separation"})
        if not stages:
            raise ValueError("Pass --stages with a scene study when using --smoke.")
    else:
        stages = tuple(args.stages)
    project = args.project.resolve()
    shutil.copyfile(project / "mbd-source-sha256.json", output / "source-files-sha256.json")
    unity = [str(args.unity.resolve()), "-batchmode", "-force-d3d11", "-job-worker-count", "4", "-projectPath", str(project)]
    branch, commit = git("rev-parse", "--abbrev-ref", "HEAD"), git("rev-parse", "HEAD")
    for stage in stages:
        folder = output if stage == "reference" else output / stage.replace("-", "_")
        marker = folder / ("unity-reference.csv" if stage == "reference" else "summary.json")
        if marker.exists():
            if args.resume:
                continue
            raise FileExistsError(f"{marker} exists. Use a fresh output directory or --resume.")
        log = output / (stage + ".log")
        subprocess.run(unity + ["-executeMethod", "LightingMvp.Editor.NativeRecoveryExperiment.BatchRun",
                               "-bbCase", f"e20-mbd-{stage}", "-bbOutput", str(output),
                               "-bbE20Branch", branch, "-bbE20BaseCommit", commit, "-logFile", str(log)], check=True)
        if not marker.is_file():
            raise RuntimeError(f"Missing {marker}. Inspect {log}.")
        print(f"Completed {stage}. Log: {log}", flush=True)


def check_reference(root, out):
    import numpy as np

    inputs, official, unity = [csv_read(root / name) for name in ("reference-input.csv", "official-reference.csv", "unity-reference.csv")]
    assert len(inputs) > 1500 and all(a["id"] == b["id"] == c["id"] for a, b, c in zip(inputs, official, unity, strict=True))
    labels = ("bound", "m0", "m1", "m2", "m3", "m4", "eta")

    def matrix(rows, prefix, columns):
        return np.asarray([[float(r[prefix + name]) if r[prefix + name] else np.nan for name in columns] for r in rows])

    comparisons = {"cpu_vs_fp64": (matrix(unity, "cpu_", labels), matrix(official, "fp64_", labels), matrix(unity, "cpu_", ("status",))[:, 0]),
                   "gpu_vs_fp32": (matrix(unity, "gpu_", labels), matrix(official, "fp32_", labels),
                                  matrix(unity, "gpu_", ("status", "adjoint_status")).max(axis=1))}
    comparison_rows, pointwise = [], []
    for order in (1, 2):
        for group in sorted({r["group"] for r in inputs}):
            chosen = np.asarray([r["order"] == str(order) and r["group"] == group for r in inputs])
            for method, (actual, expected, status) in comparisons.items():
                reference_valid = np.isfinite(expected).all(axis=1)
                result_valid = status <= 1
                paired = reference_valid & result_valid & chosen
                assert np.isfinite(actual[paired]).all()
                error = abs(actual[paired] - expected[paired])
                scaled = error[:, 1:] / np.maximum(1, abs(expected[paired, 1:]))
                summary = [float(np.quantile(error[:, 0], .99)), float(np.max(error[:, 0])),
                           float(np.quantile(scaled, .99)), float(np.max(scaled))] if error.size else [""] * 4
                disagreement = np.count_nonzero((reference_valid != result_valid) & chosen)
                if group in {"random", "layers_1", "layers_2", "official_singularity", "infinite_root", "invalid_hankel"}:
                    assert disagreement == 0, (order, group, method, disagreement)
                    if error.size:
                        assert summary[1] < 4e-5 and summary[3] < .005, (order, group, method, summary)
                comparison_rows.append((order, group, method, int(chosen.sum()), int((chosen & reference_valid).sum()), int((chosen & result_valid).sum()),
                                        int(disagreement), int((chosen & (status == 1)).sum()), int((chosen & (status == 2)).sum()),
                                        int((chosen & (status == 5)).sum()), *summary))
                for row, e, adj in zip(np.flatnonzero(paired), error, scaled):
                    pointwise.append((inputs[row]["id"], order, group, method, e[0], *adj))
    fd = matrix(official, "fd64_", labels[1:])
    cpu = comparisons["cpu_vs_fp64"][0][:, 1:]
    rows = []
    for order in (1, 2):
        valid = np.isfinite(fd).all(axis=1) & np.array([r["order"] == str(order) for r in inputs])
        assert valid.sum() == 480
        errors = abs(cpu[valid] - fd[valid]) / np.maximum(1, abs(fd[valid]))
        assert errors.max() < 1e-4
        rows.append((order, int(valid.sum()), float(np.quantile(errors, .99)), float(errors.max())))
    csv_write(out / "core-comparison.csv", ("order", "case", "comparison", "total", "official_valid", "unity_valid", "status_mismatch",
                                           "infinite_roots", "not_positive", "nonfinite", "bound_p99_abs", "bound_max_abs", "adjoint_p99_scaled", "adjoint_max_scaled"), comparison_rows)
    csv_write(out / "core-pointwise-errors.csv", ("id", "order", "case", "comparison", "bound_abs", *[f"{q}_scaled" for q in labels[1:]]), pointwise)
    csv_write(out / "core-fd-check.csv", ("order", "samples", "adjoint_vs_fd_p99_scaled", "adjoint_vs_fd_max_scaled"), rows)


def check_trials(folder):
    from blackbox_e20 import json_read

    summary = json_read(folder / "summary.json")
    raw = {(r["key"], r["method"]): r for r in summary["arms"]}
    rows = csv_read(folder / "trials.csv")
    assert len(rows) == len(raw)
    for key in ("nativeRgbViews", "surfaceViews", "shadowFaces"):
        assert sum(r[key] for r in raw.values()) <= summary[key]
    accepts = {}
    for r in csv_read(folder / "queries.csv"):
        if r["kind"] == "native_acceptance" and r["accepted"] == "1":
            key = (r["case"], r["method"])
            prior = accepts.get(key, raw[key]["nativeStart"])
            next_loss = float(r["native_rgb_loss"])
            assert next_loss < prior
            accepts[key] = next_loss
    for key, arm in raw.items():
        for i, truth in enumerate(summary["truth"]):
            if i // 4 + 1 not in arm["lamps"]:
                assert arm["initial"][i] == truth == arm["final"][i]
        if arm["method"] in {"C2", "D3"}:
            assert abs(accepts.get(key, arm["nativeStart"]) - arm["nativeFinal"]) < 1e-7 * max(1., arm["nativeStart"])
        assert len(arm["perViewNrmse"]) == 5
    return rows, summary


def direction_check(root, out):
    import numpy as np

    rows = csv_read(root / "shadow" / "position-derivatives.csv")
    paired = []
    for case in dict.fromkeys(r["scene"] for r in rows):
        selected = [r for r in rows if r["scene"] == case]
        keys = ("fixed_map_adjoint", "fixed_map_fd", "fixed_map_fd_at_full_step", "full_mbd_fd", "full_native_fd", "full_ray_fd")
        a, f, w, m, n, r = np.array([[float(row[k]) for k in keys] for row in selected]).T

        def cosine(x, y):
            norm = np.linalg.norm(x) * np.linalg.norm(y)
            return float(x @ y / norm) if norm > 0 else np.nan

        paired.append((case, len(selected), float(np.sqrt(np.mean((a - f)**2))), cosine(a, f), cosine(a, m), cosine(w, m), cosine(m, n), cosine(m, r),
                       sum(int(x["adjoint_status"]) > 1 for x in selected)))
    csv_write(out / "shadow-direction-summary.csv", ("scene", "derivatives", "adjoint_fixed_map_rmse", "adjoint_vs_fixed_fd", "adjoint_vs_full_mbd_fd",
                                                     "fixed_vs_full_matched_step_fd", "full_mbd_vs_native_fd", "full_mbd_vs_ray_fd", "invalid_adjoint"), paired)


def plots(root, out, trials):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    def save(fig, name):
        fig.savefig(out / (name + ".png"), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    samples = csv_read(out / "core-pointwise-errors.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    source_cases = ("random", "layers_1", "layers_2", "official_singularity", "near_singular_hankel")
    for ax, order in zip(axes, (1, 2)):
        values = [[max(1e-10, *[float(row[x + "_scaled"]) for x in ("m0", "m1", "m2", "m3", "m4", "eta")])
                   for row in samples if row["comparison"] == "gpu_vs_fp32" and row["order"] == str(order) and row["case"] == case]
                  for case in source_cases]
        ax.boxplot(values, showfliers=True)
        ax.set_xticks(range(1, 6), ("Random", "1 layer", "2 layers", "Root limit", "Near-singular\nHankel"), rotation=15)
        ax.set_yscale("log")
        ax.set_title(f"n = {order}")
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Per-sample max scaled adjoint error")
    fig.suptitle("Unity GPU adjoint against the official CPU FP32 implementation")
    save(fig, "core-adjoint-errors")

    shadow_dir = root / "shadow"
    summaries = csv_read(shadow_dir / "shadow-metrics.csv")
    settings = [r for r in summaries if r["setting"] == "r128_msaa4_radius2_beta0.15"]
    examples = [r for r in settings if r["scene"] != "empty"]
    fig, axes = plt.subplots(len(examples), 4, figsize=(8.2, 10.8), layout="constrained")
    titles = ("Ray hard", "Disk-filtered rays", "Native URP hard", "MBD")
    for i, row in enumerate(examples):
        for j, name in enumerate(("ray_hard", "ray_filtered", "native_hard", "mbd")):
            ax = axes[i, j]
            with Image.open(shadow_dir / row["scene"] / row["setting"] / (name + ".png")) as image:
                ax.imshow(image, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(titles[j])
            if j == 0:
                ax.set_ylabel(row["scene"].replace("_", " "))
    fig.suptitle("Point-light visibility on the calibration plane")
    save(fig, "shadow-calibration")

    derivatives = csv_read(out / "shadow-direction-summary.csv")
    fig, ax = plt.subplots(figsize=(10, 4.4), layout="constrained")
    for i, (key, label) in enumerate((("adjoint_vs_fixed_fd", "Adjoint / fixed-map FD (0.25 mm)"), ("adjoint_vs_full_mbd_fd", "Adjoint / rerendered FD (1 cm)"),
                                       ("fixed_vs_full_matched_step_fd", "Fixed / rerendered FD (both 1 cm)"),
                                       ("full_mbd_vs_ray_fd", "Rerendered / filtered rays"), ("full_mbd_vs_native_fd", "Rerendered / native URP"))):
        ax.bar(np.arange(len(derivatives)) + (i - 2) * .15, [float(r[key]) for r in derivatives], .14, label=label)
    ax.set_xticks(range(len(derivatives)), [r["scene"].replace("_", " ") for r in derivatives])
    ax.set_ylim(-1.05, 1.10); ax.axhline(0, color="black", lw=.6)
    ax.set_ylabel("Cosine across receiver derivatives")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(.5, 1.01))
    ax.grid(axis="y", alpha=.18)
    save(fig, "shadow-derivatives")

    for first, name, chosen in (("C", "classroom-recovery", [r for r in trials if r["method"][0] == "C"]),
                                ("D", "irradiance-recovery", [r for r in trials if r["method"][0] == "D"])):
        cases = list(dict.fromkeys(r["case"] for r in chosen))
        methods = list(dict.fromkeys(r["method"] for r in chosen))
        fig, axes = plt.subplots(3, 1, figsize=(max(9, len(cases) * 2), 7.5), sharex=True, layout="constrained")
        for ax, key, label, scale, threshold in zip(axes, ("max_position_cm", "max_intensity_relative", "native_held_nrmse"),
                                                    ("Max position (cm)", "Max intensity (%)", "Native held NRMSE (%)"), (1, 100, 100), (5, 5, None)):
            width = .75 / len(methods)
            for i, method in enumerate(methods):
                selected = {r["case"]: max(1e-5, float(r[key]) * scale) for r in chosen if r["method"] == method}
                ax.bar(np.arange(len(cases)) + (i - .5 * (len(methods) - 1)) * width, [selected[k] for k in cases], width, label=method)
            if threshold:
                ax.axhline(threshold, ls="--", lw=.8, color="#cb3234")
            ax.set_yscale("log")
            ax.set_ylabel(label)
            ax.grid(axis="y", alpha=.18)
        axes[0].legend(ncol=len(methods))
        axes[-1].set_xticks(range(len(cases)), [k.replace("_d0", "").replace("_", " ") for k in cases])
        fig.suptitle("Matched E18 Classroom starts" if first == "C" else "Fixed-mesh per-light irradiance · 4D, 8D and 28D")
        save(fig, name)

    paths = csv_read(root / "topology" / "paths.csv")
    cases = list(dict.fromkeys(r["case"] for r in paths))
    fig, axes = plt.subplots(2, len(cases), figsize=(5 * len(cases), 6.2), sharey="row", layout="constrained", squeeze=False)
    for j, case in enumerate(cases):
        case_rows = [r for r in paths if r["case"] == case]
        x = [float(r["t"]) for r in case_rows]
        legal = [r["collider_legal"] == "1" for r in case_rows]
        for i, (left, right, label, log) in enumerate((("native_on_loss", "native_off_loss", "Native squared NRMSE", True),
                                                        ("hard_signature_distance", "mbd_signature_distance", "Weighted signature distance", False))):
            ax = axes[i, j]
            ax.plot(x, [max(1e-8, float(r[left])) if log else float(r[left]) for r in case_rows], label="Shadow On" if log else "Hard ray")
            ax.plot(x, [max(1e-8, float(r[right])) if log else float(r[right]) for r in case_rows], label="Shadow Off" if log else "MBD soft")
            ax.fill_between(x, 0, 1, where=np.logical_not(legal), color="#f59f00", alpha=.22, transform=ax.get_xaxis_transform(), label="Collider rejected")
            if log:
                ax.set_yscale("log")
                ax.set_title(case.replace("_", " "))
            if j == 0:
                ax.set_ylabel(label)
            if i == 1:
                ax.set_xlabel("Interpolation from start toward GT")
            ax.grid(alpha=.16)
    axes[0, -1].legend(fontsize=8); axes[1, 0].legend(fontsize=8)
    fig.suptitle("E19 · feasible-domain, loss barrier and visibility signature")
    save(fig, "visibility-topology")


def report(args):
    from blackbox_e20 import json_read

    root, out = args.output.resolve(), args.report_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    check_reference(root, out)
    direction_check(root, out)
    trials, stages = [], {}
    for key, folder in (("C", root / "recovery"), ("D_classroom", root / "irradiance" / "classroom"),
                        ("D_e19", root / "irradiance" / "e19"), ("E", root / "separation"), ("F", root / "topology")):
        rows, summary = check_trials(folder)
        stages[key] = {name: value for name, value in summary.items() if name != "arms"}
        trials.extend(rows)
    assert 42 <= len(trials) <= 44
    assert stages["D_e19"]["e19SceneSha256"] == stages["F"]["e19SceneSha256"]
    assert stages["E"]["e19SceneSha256"] == stages["D_e19"]["e19SceneSha256"]
    assert (root / "separation" / "surface-samples.csv").read_bytes() == (root / "irradiance" / "e19" / "surface-samples.csv").read_bytes()
    assert stages["C"]["classroomSha256"] == stages["D_classroom"]["classroomSha256"]
    if args.prior_output:
        prior = args.prior_output / "e20_a"
        cfg = json_read(prior / "config.json")
        assert cfg["sceneSha256"] == stages["D_e19"]["e19SceneSha256"]
        raw = json_read(prior / "summary.json")["arms"]
        earlier = [r for r in raw if r["method"] in {"A1", "A2", "A3", "oracle"} and r["seed"] in {131, 733} and r["profile"] == "near"]
        csv_write(out / "e-reused-e20-a.csv", ("arm", "passed", "max_position_cm", "max_intensity_relative", "train_nrmse", "held_nrmse", "stop"),
                  [(x["id"], int(x["passed"]), max(x["positionCm"]), max(x["intensityRelative"]), x["nativeBest"]**.5, x["heldNrmse"], x["stop"]) for x in earlier])
    header = tuple(trials[0])
    csv_write(out / "trials.csv", header, [[row[x] for x in header] for row in trials])
    stages["B"] = json_read(root / "shadow" / "summary.json")
    write_json(out / "stage-summary.json", stages)
    plots(root, out, trials)
    with zipfile.ZipFile(out / "raw-data.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".csv", ".json"}:
                archive.write(path, path.relative_to(root).as_posix())
    print(f"Verified {len(trials)} trial arms and wrote {out}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    vectors = sub.add_parser("reference")
    vectors.add_argument("--official-python", type=Path)
    vectors.add_argument("--output", type=Path, required=True)
    vectors.set_defaults(handler=reference)
    prep = sub.add_parser("prepare")
    prep.add_argument("--project", type=Path, required=True)
    prep.add_argument("--e19-project", type=Path, required=True)
    prep.add_argument("--classroom-project", type=Path)
    prep.set_defaults(handler=prepare)
    start = sub.add_parser("run")
    start.add_argument("--unity", type=Path, required=True)
    start.add_argument("--project", type=Path, required=True)
    start.add_argument("--output", type=Path, required=True)
    start.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    start.add_argument("--smoke", action="store_true")
    start.add_argument("--resume", action="store_true")
    start.set_defaults(handler=run)
    result = sub.add_parser("report")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--report-dir", type=Path, required=True)
    result.add_argument("--prior-output", type=Path)
    result.set_defaults(handler=report)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

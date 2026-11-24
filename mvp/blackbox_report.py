"""Summarize a Unity black-box RGB result directory without third-party dependencies."""

import argparse
import csv
import html
import json
import math
from pathlib import Path
from statistics import median


def href(path, root):
    return html.escape(path.relative_to(root).as_posix(), quote=True)


def pct(value):
    return f"{value:.2%}" if value >= 0 else "—"


def histories(stage_dir, trials):
    for trial in trials:
        path = stage_dir / f"{trial['seed']}_{trial['method']}" / "queries.csv"
        best = float("inf")
        samples = []
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if row["phase"].startswith("validate_"):
                    break
                best = min(best, float(row["loss"]))
                samples.append((int(row["cumulative_camera_renders"]), float(row["elapsed_s"]), math.sqrt(best)))
        yield trial, samples


def plot(histories, axis):
    width, height, left, right, top, bottom = 470, 210, 65, 12, 10, 34
    xmax = max(samples[-1][axis] for _, samples in histories)
    minlog = min(-2, math.floor(math.log10(max(1e-6, min(s[-1][2] for _, s in histories)))))
    maxlog = max(0, math.ceil(math.log10(max(s[0][2] for _, s in histories))))
    y = lambda error: top + (maxlog - math.log10(max(1e-6, error))) / (maxlog - minlog) * (height - top - bottom)
    x = lambda value: left + value / xmax * (width - left - right)
    elements = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Best RGB error by query cost">']
    for i in range(minlog, maxlog + 1):
        oy = y(10**i)
        elements += [f'<line x1="{left}" x2="{width-right}" y1="{oy}" y2="{oy}" stroke="#d7dfe9"/>',
                     f'<text x="{left-7}" y="{oy+4}" text-anchor="end">{10**i * 100:g}%</text>']
    for tick in range(5):
        value = xmax * tick / 4
        ox = x(value)
        elements += [f'<line x1="{ox}" x2="{ox}" y1="{top}" y2="{height-bottom}" stroke="#eaf0f6"/>',
                     f'<text x="{ox}" y="{height-bottom+15}" text-anchor="middle">{value:.0f}</text>']
    cutoff = y(.05)
    elements.append(f'<line x1="{left}" x2="{width-right}" y1="{cutoff}" y2="{cutoff}" stroke="#718295" stroke-dasharray="3 3"><title>5% NRMSE threshold</title></line>')
    name = "Training camera renders" if axis == 0 else "Training wall time (s)"
    elements.append(f'<text x="{(width+left-right)/2}" y="{height-3}" text-anchor="middle">{name}</text>')
    for trial, samples in histories:
        color = {"CentralLbfgs": "#1b6b8f", "SpsaAdam": "#b75023", "ZeroGrads": "#6147a2"}[trial["method"]]
        label = f"{trial['method']}, seed {trial['seed']}"
        indices = set(range(0, len(samples), max(1, len(samples) // 350))) | {len(samples) - 1}
        points = " ".join(f"{x(point[axis]):.2f},{y(point[2]):.2f}" for i, point in enumerate(samples) if i in indices)
        elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5" opacity=".8"><title>{label}</title></polyline>')
    elements.append("</svg>")
    return "\n".join(elements)


def report(root):
    parts = ["""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Unity native RGB recovery</title><style>
body{max-width:1200px;margin:2.5rem auto;padding:0 1rem;font:15px/1.55 system-ui;color:#152b3d;background:#f5f8fb}
section{background:white;padding:1.4rem;margin:1.2rem 0;border:1px solid #dae3ef;border-radius:10px}
h1,h2{line-height:1.2}h2{margin-top:0}table{border-collapse:collapse;width:100%}th,td{padding:.4rem .65rem;text-align:left;border-bottom:1px solid #e4eaf1}
.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:1rem}svg{width:100%;font:11px system-ui;fill:#33475a}
.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.5rem;margin:1rem 0}figure{margin:0}img{width:100%;display:block;background:#0d1723}figcaption{padding-top:.2rem}
.fd{color:#1b6b8f}.spsa{color:#b75023}.zg{color:#6147a2}.small{font-size:.85em;color:#41566d}summary{cursor:pointer}a{color:#145b7c}
</style><h1>Unity native RGB recovery</h1><p>Unity's final linear HDR camera RGB, with realtime shadows. Reconstructions are fresh renders of the best engine query. Each view contributes equal weight.</p>
<p class="small">The log-scale plots show best-so-far RGB NRMSE. <b class="fd">FD + L-BFGS</b> · <b class="spsa">SPSA + Adam</b> · <b class="zg">ZeroGrads diagnostic</b>. Curves exclude shared target setup and validation (counted separately).</p>"""]
    all_rows = []
    total_renders = 0
    reports = sorted(root.glob("*/report.json"), key=lambda p: (not p.parent.name.startswith("e"), p.parent.name))
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        directory = path.parent
        if "trials" not in data:
            if "repeatMax" not in data:
                continue
            total_renders += data["cameraRenders"]
            responses = [data[k] for k in ("positionPlus", "positionMinus", "intensityPlus", "intensityMinus", "shadowsOff", "lightsOff")]
            parts.append(f'<section><h2>{directory.name} · {html.escape(data["shader"])}</h2>'
                         f'<p>Passed: {data["passed"]} · repeated loss ≤ {data["repeatMax"]:.3g} · minimum response {min(responses):.3g} · '
                         f'HDR test peak {data["brightPeak"]:.2f} · {data["cameraRenders"]} renders · <a href="{href(path, root)}">JSON</a></p></section>')
            continue
        total_renders += data["allCameraRenders"]
        trials = data["trials"]
        traces = list(histories(directory, trials))
        stage = data["stage"]
        parts.append(f'<section><h2>{stage} · {html.escape(data["shader"])}</h2><p>{data["allCameraRenders"]:,} renders total, '
                     f'{data["wallSeconds"]:.1f}s wall clock. <a href="{href(path, root)}">Stage JSON</a> · '
                     f'<a href="{href(directory / "shared_queries.csv", root)}">Shared setup log</a></p>')
        parts.append('<table><thead><tr><th>Method</th><th>Passed</th><th>Median start</th><th>Median best 256²</th><th>Median best 512²</th>'
                     '<th>Median held-out 512²</th><th>Median renders</th><th>Median time</th></tr></thead><tbody>')
        for method in dict.fromkeys(t["method"] for t in trials):
            group = [t for t in trials if t["method"] == method]
            med = lambda key: median(t[key] for t in group)
            parts.append(f'<tr><td>{method}</td><td>{sum(t["passed"] for t in group)}/{len(group)}</td>'
                         f'<td>{pct(med("initialNrmse"))}</td><td>{pct(med("bestNrmse"))}</td><td>{pct(med("highBestNrmse"))}</td>'
                         f'<td>{pct(med("heldOutBestNrmse"))}</td><td>{med("trainingCameraRenders"):g}</td><td>{med("trainingWallSeconds"):.2f}s</td></tr>')
        parts.append('</tbody></table><div class="plots">' + plot(traces, 0) + plot(traces, 1) + '</div>')
        for trial in trials:
            fields = ("stage", "seed", "method", "dimensions", "trainingViews", "cameraBudget", "trainingCameraRenders", "trainingWallSeconds",
                      "renderSubmitSeconds", "readbackSeconds", "initialNrmse", "bestNrmse", "improvement", "highBestNrmse",
                      "heldOutInitialNrmse", "heldOutBestNrmse", "passed")
            all_rows.append({key: trial[key] for key in fields})
            td = directory / f'{trial["seed"]}_{trial["method"]}'
            parts.append(f'<details><summary>Seed {trial["seed"]} · {trial["method"]} · {pct(trial["initialNrmse"])} → {pct(trial["bestNrmse"])}</summary>'
                         f'<p><a href="{href(td / "queries.csv", root)}">Every query</a> · <a href="{href(td / "report.json", root)}">Metrics and fitted rig</a></p>')
            for view in range(len(data["views"])):
                if not (td / f"best_512_v{view}.png").exists():
                    continue
                parts.append(f'<div class="small">View {view} ({"training" if view < trial["trainingViews"] else "held-out"}) · 512²</div><div class="gallery">')
                for label, image in [("Target", directory / f"target_512_v{view}.png"), ("Initial", td / f"initial_512_v{view}.png"), ("Best", td / f"best_512_v{view}.png")]:
                    url = href(image, root)
                    exr = image.with_suffix(".exr")
                    raw = f' · <a href="{href(exr, root)}">HDR EXR</a>' if exr.is_file() else ""
                    parts.append(f'<figure><a href="{url}"><img loading="lazy" src="{url}" alt="{label}"></a>'
                                 f'<figcaption>{label}{raw}</figcaption></figure>')
                parts.append('</div>')
            parts.append('</details>')
        parts.append('</section>')
    parts.append(f'<p class="small">Total across completed reports: {total_renders:,} camera renders. A missing trial or stage is not a pass.</p></html>')
    (root / "index.html").write_text("\n".join(parts), encoding="utf-8")
    if all_rows:
        with (root / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
    print(root / "index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="A Unity -bbOutput directory containing per-stage reports")
    report(parser.parse_args().output)

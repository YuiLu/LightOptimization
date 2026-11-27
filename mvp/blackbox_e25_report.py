"""Validate and export compact E25 evidence from one or more completed run roots."""
import argparse
import collections
import csv
import json
import math
import re
import shutil
import statistics
import zipfile
from pathlib import Path

from blackbox_e25 import BASE, CORE_C, ROOT, completed, rows


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def collect(roots):
    stages = {}
    for root in roots:
        for marker in sorted(root.glob("e25_*/complete.json")):
            stage = marker.parent.name[4:].replace("_", "-")
            if stage.endswith("-smoke"):
                continue
            if stage in stages:
                raise ValueError(f"Duplicate completed stage: {stage}")
            stages[stage] = (marker.parent, completed(root, stage))
    return stages


def write_csv(dest, data, header=None):
    data = list(data)
    if not data and header is None:
        return
    if header is None:
        header = list(dict.fromkeys(k for r in data for k in r))
    with dest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, header, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def table(stages, name):
    for stage, (folder, _) in stages.items():
        csv_file = folder / name
        if csv_file.is_file():
            for r in rows(csv_file):
                yield dict(stage=stage, **r)


def validate(stages):
    for key, (folder, summary) in stages.items():
        if summary["stage"] != "e25-" + key or summary["branch"] != "blackbox" or len(summary["truth"]) != 28:
            raise ValueError(f"Invalid native scene manifest for {key}")
        costs = {r["arm"]: r for r in rows(folder / "arm-cost.csv")}
        queries = collections.defaultdict(list)
        for row in rows(folder / "model-queries.csv"):
            queries[row["arm"]].append(row)
        checkpoints = {(r["arm"], r["checkpoint"]): r for r in rows(folder / "checkpoints.csv")}
        accepted = collections.defaultdict(list)
        for row in rows(folder / "decisions.csv"):
            if row["accepted"] == "1":
                accepted[row["arm"]].append(int(row["query"]))
        for arm in summary["arms"]:
            aid = arm["id"]
            if not arm["legal"] or arm["illegalAccepted"] or len(arm["initial"]) != 28 or len(arm["final"]) != 28:
                raise ValueError(f"Invalid rig: {key}/{aid}")
            cost = costs[aid]
            cap, used = int(cost["cap_camera"]), int(cost["opt_camera"])
            if used > cap or cap > 4725 or used != int(cost["hard_camera"]) + int(cost["off_camera"]):
                raise ValueError(f"Invalid camera count: {key}/{aid}")
            tally = 0
            for q in queries[aid]:
                tally += int(q["cameras"])
                if tally != int(q["actual_camera"]):
                    raise ValueError(f"Native query/camera mismatch: {key}/{aid}/{q['query']}")
            if tally != used:
                raise ValueError(f"Incomplete native query log: {key}/{aid}")
            if cap and not aid.endswith(("_C4_warmup", "_C4-cache_warmup")):
                fit_views = len(cost["fit_views"].split(";"))
                q = max((q for q in accepted[aid] if q * fit_views <= 1560), default=0)
                checkpoint = checkpoints[aid, "last_accepted_at_or_before_1560"]
                if int(checkpoint["actual_camera"]) != q * fit_views:
                    raise ValueError(f"Misidentified screening checkpoint: {key}/{aid}")
                if q:
                    selected = next(r for r in queries[aid] if r["query"] == str(q) and r["phase"] == "trial")
                    if selected["theta"] != checkpoint["theta"]:
                        raise ValueError(f"Screening checkpoint pose mismatch: {key}/{aid}")


def summarise(stages, gates):
    meta = {}
    for key, (folder, data) in stages.items():
        meta[key] = dict(scene=data["scene"]["scene"], scene_sha=data["sceneHash"], unity=data["scene"]["unity"], urp=data["scene"]["urp"],
                         gpu=data["scene"]["gpu"], source=json.loads((folder / "complete.json").read_text(encoding="utf-8")),
                         native_cameras=data["nativeCameraRenders"], surface_renders=data["surfaceRenders"], wall_s=data["seconds"],
                         arms=[dict(id=a["id"], method=a["method"], scenario=a["scenario"], lamps=a["lamps"], pass_=a["pass"],
                                    max_cm=max(a["positionCm"]), max_intensity_relative=max(a["intensityRelative"]), stop=a["stop"]) for a in data["arms"]])
    return dict(baseline=BASE, core_c_complete=all(k in meta for k in CORE_C), stages=meta, gates=gates)


def rms(vals):
    vals = [float(x) for x in vals]
    return math.sqrt(sum(x * x for x in vals) / len(vals)) if vals else None


def fmt(value, suffix=""):
    return "—" if value is None else f"{value:.2f}{suffix}"


def results_readme(stages, gates):
    vx = collections.defaultdict(list)
    for r in table(stages, "view-metrics.csv"):
        if r["checkpoint"] == "final" and r["target"] == "hard" and r["width"] == "896" and int(r["view"]) >= 3:
            vx[r["stage"], r["arm"]].append(r)
    cost = {(r["stage"], r["arm"]): r for r in table(stages, "arm-cost.csv")}
    checkpoint = {(r["stage"], r["arm"]): r for r in table(stages, "checkpoints.csv") if r["checkpoint"] == "last_accepted_at_or_before_1560"}
    text = ["# E25 · 原生 RGB 恢复、观测消融与 CPU 等价性", "",
            "E25 使用 E24 冻结的场景、灯阵、相机与 Unity/URP 配置。优化仍由 Unity 原生 RGB 验收；活动灯联合更新 XYZ + log(I)，"
            "Strict Collider、LampScaledDense 和信任域设定保持不变。C3/C4 直接对锚定 RGB 的 Signed Log 值做差分，不使用链式映射近似。", "",
            "参数通过标准：每盏活动灯的位置误差 ≤5 cm、相对强度误差 ≤5%，终态合法。图像质量单列；Raw、Signed Log 与固定 Tone-Exp+sRGB 都按视角真值能量归一化。"
            "`1560` 检查点只取预算线前最后一个已接受状态，未追加模型探针。", ""]
    if any(k in stages for k in CORE_C):
        text += ["## C · 独立起点", "", "| 起点 | 方法 | 1560 前通过 | 最终通过 | 最大位置 cm | 最大强度误差 | 实际图/4725 | Hard 留出 Raw / Log / Display | 停止 |",
                 "|---|---|---|---|---:|---:|---:|---:|---|"]
        for key in (*CORE_C, *(s for s in stages if s.startswith("h1-"))):
            if key not in stages:
                continue
            for a in stages[key][1]["arms"]:
                c = cost[key, a["id"]]
                cp = checkpoint[key, a["id"]]
                held = vx[key, a["id"]]
                nrmse = " / ".join(fmt(100 * rms(r[k] for r in held), "%") if held else "—" for k in ("raw_nrmse", "log_nrmse", "display_nrmse"))
                text.append(f"| {a['scenario']} | {a['method']} | {'是' if cp['pass'] == '1' else '否'} | {'是' if a['pass'] else '否'} | "
                            f"{max(a['positionCm']):.2f} | {100 * max(a['intensityRelative']):.2f}% | {c['opt_camera']} | {nrmse} | `{a['stop']}` |")
        text += ["", "留出指标为高分辨率视角 3–4 的均方根。A-log 的图像列仍以真实 Hard 渲染复拍，参数列为 Off Oracle 结果。", ""]
    for title, keys in (("A · Classroom L6 观测消融", ("a-classroom", "a-views-classroom", "a-views7-classroom")),
                        ("B · Complex Far 单灯与联合", ("b-complex", "b-diag-complex"))):
        if not any(k in stages for k in keys):
            continue
        text += [f"## {title}", "", "| 臂 | 最大位置 cm | 最大强度误差 | 实际优化相机图 | 通过 | 停止 |",
                 "|---|---:|---:|---:|---|---|"]
        for key in keys:
            if key not in stages:
                continue
            for a in stages[key][1]["arms"]:
                arm = a["id"]
                text.append(f"| {arm} | {max(a['positionCm']):.2f} | {100 * max(a['intensityRelative']):.2f}% | {cost[key, arm]['opt_camera']} | "
                            f"{'是' if a['pass'] else '否'} | `{a['stop']}` |")
        text.append("")
    rr = list(table(stages, "numeric-parity.csv"))
    if rr:
        ok = sum(r["pass"] == "1" for r in rr)
        text.extend(["## D · 固定输入与端到端复验", "",
                     f"固定输入通过 {ok}/{len(rr)} 项（L2：`max(1e-12, 1e-6 × 参考范数)`）。真实锚点/探针由 D 固定输入阶段重新捕获；另测有符号极值及一、二阶单边模板。", ""])
    timings = collections.defaultdict(list)
    for (key, arm), row in cost.items():
        if key.startswith("d-replay-") and "_rep" in arm:
            timings[key, arm.split("_rep")[1].split("_", 1)[1]].append(float(row["opt_wall_s"]))
    if timings:
        text += ["| 起点 | 方法 | 重复 | 墙钟中位数（秒） | 范围（秒） |", "|---|---|---:|---:|---:|"]
        for (k, model), values in sorted(timings.items()):
            text.append(f"| {k} | {model} | {len(values)} | {statistics.median(values):.2f} | {min(values):.2f}–{max(values):.2f} |")
        text += ["", "相机提交/同步读回与 CPU 阶段分开记录。未取得 GPU 独占时间时标记 `unavailable`。", ""]
    restoration = paired_ends(stages)
    if restoration:
        successful = sum(r["reference_pass"] for r in restoration)
        retained = sum(r["reference_pass"] and r["cached_pass"] for r in restoration)
        text += [f"完整恢复复验共 {len(restoration)} 对，参考成功保留 {retained}/{successful}；"
                 f"最大归一化终态差 {max(r['max_theta_difference'] for r in restoration):.2g}。", ""]
    gpu = list(table(stages, "gpu-parity.csv"))
    if gpu:
        perf_gpu = list(table(stages, "gpu-transfer.csv"))
        speed = [float(x["cpu_cached_wall_s"]) / float(x["full_gpu_path_wall_s"]) for x in perf_gpu]
        text += [f"GPU 固定输入对照通过 {sum(r['pass'] == '1' for r in gpu)}/{len(gpu)} 项。"
                 f"含上传、提交和同步读回的列构建，相对 CPU 缓存的中位速度比为 {statistics.median(speed):.2f}×。"
                 "此处仅测固定输入，不把微基准速度比当作恢复端到端收益。", ""]
    gpu_replay = gpu_ends(stages)
    if gpu_replay:
        losses = sum(r["reference_pass"] and not r["gpu_pass"] for r in gpu_replay)
        text += [f"GPU 恢复对照完成 {len(gpu_replay)} 对。参考成功丢失 {losses} 个；"
                 f"相同预算下实际渲染数和终态差异逐臂见 `gpu-end-to-end.csv`。", "",
                 "| 起点 | GPU 重复 | GPU 墙钟中位数（秒） | CPU 参考中位数（秒） | 相对降幅 |",
                 "|---|---:|---:|---:|---:|"]
        for stage in sorted({r["stage"] for r in gpu_replay}):
            group = [r for r in gpu_replay if r["stage"] == stage]
            gpu_time, ref_time = (statistics.median(r[k] for r in group) for k in ("gpu_wall_s", "reference_wall_s"))
            text.append(f"| {stage} | {len(group)} | {gpu_time:.2f} | {ref_time:.2f} | {100 * (1 - gpu_time / ref_time):.1f}% |")
        text += ["", "复验仍用 CPU 的 mask/P95、完整 Gram 和 LM 求解；GPU 只融合 XYZ 列。实际捕获与求解器之间仍有传输。", ""]
    text += ["## 可复现记录", "", "`stage-summary.json` 保存实际完成的阶段、资产及运行版本、条件门槛。`raw-data.zip` 保存每阶段完整数值表、源文件哈希与 GT/初态/终态快照；"
             "仓库只保留代表图，不包含 Unity 缓存、全程画面或 D 的浮点帧缓冲。D 固定输入的哈希、坐标和 stencil 在 `fixed-inputs.csv`。", "",
             "在与 E24 资产清单匹配的独立批处理工程中运行（不要指定主 Unity 工程）：", "",
             "```powershell", r".venv\Scripts\python.exe -B mvp/blackbox_e25.py --unity <Unity.exe> --project <isolated-project> --output tmp/e25 --sync --pipeline --resume",
             r".venv\Scripts\python.exe -B mvp/blackbox_e25_report.py --roots tmp/e25 --output docs/blackbox-e25", "```", ""]
    return "\n".join(text)



def evidence_gates(stages, gates):
    # True Hard direction changes are read from the same initial pose as the matched recoveries.
    directions = collections.defaultdict(list)
    for r in table(stages, "native-steps.csv"):
        directions[r["state"]].append(r)
    candidates = []
    checked = []
    for key in ("c-classroom-1301", "c-classroom-1601", "c-complex-509"):
        if key not in stages:
            continue
        by_method = {r["method"]: r for r in stages[key][1]["arms"]}
        off = by_method.get("A-log")
        if off is None and "b-complex" in stages and key.endswith("-509"):
            # Exact E24 full-rig counterpart, not an oracle single-lamp arm.
            with zipfile.ZipFile(ROOT / "docs/blackbox-e24/raw-data.zip") as ref:
                off = next(a for a in json.loads(ref.read("e24_a_complex/summary.json"))["arms"]
                           if a["id"] == "seed509_complex_far_A-log")
        if off is None:
            continue
        proxy_fail = not by_method["C3"]["pass"] and not by_method["C4"]["pass"]
        d = [r for r in directions[off["scenario"] + "_initial"] if r["model"] == "H1" and r["legal"] == "1"]
        pairs = collections.defaultdict(list)
        for r in d:
            pairs[float(r["position_mm"]), float(r["step_fraction"])].append(float(r["actual"]) > max(1e-12, 1e-5 * float(r["base_loss"])))
        duplicate_decrease = any(len(v) >= 2 and all(v) for v in pairs.values())
        unstable = any(float(r["position_mm"]) != 20 and float(r["hard_fd_direction_cos_20mm"]) < .8 for r in d)
        checked.append(dict(case=key, off_pass=off["pass"], both_hard_proxies_fail=proxy_fail,
                            hard_repeated_decrease=duplicate_decrease if d else None, hard_direction_cos_below_point8=unstable if d else None))
        if off["pass"] and proxy_fail and duplicate_decrease and unstable:
            candidates.append(key)
    if checked:
        ran = sorted(name for name in stages if name.startswith("moment-"))
        gates["E-MSM-MBD"] = dict(eligible=bool(candidates), executed=bool(ran), matched_starts=checked, candidates=candidates, moment_stages=ran,
                                 reason="multiplicative_audit_precedes_any_fusion" if ran else "all_same_start_preconditions_required" if not candidates
                                 else "qualified_native_evidence_Moment_arm_not_recorded")
    costs = list(table(stages, "arm-cost.csv"))
    perf = {r["arm"]: r for r in costs if r["stage"] == "d-replay-131-complex" and "_rep" in r["arm"]}
    cached = [r for arm, r in perf.items() if arm.endswith("_C4-cache")]
    base = [r for arm, r in perf.items() if arm.endswith("_C4")]
    if len(cached) >= 3 and len(base) >= 3:
        opt_s = statistics.median(float(r["opt_wall_s"]) for r in cached)
        ref_s = statistics.median(float(r["opt_wall_s"]) for r in base)
        fraction = {part: statistics.median(float(r[part]) / float(r["opt_wall_s"]) for r in cached)
                    for part in ("map_fd_s", "mask_p95_s", "gram_gradient_s", "render_s", "readback_s")}
        gates["D1-wall"] = dict(target_fraction=.2, wall_reduction=1 - opt_s / ref_s, passes_performance=opt_s <= .8 * ref_s,
                                warm_repeats=len(cached), reference_s=ref_s, cached_s=opt_s)
        for kind, component in [("D2-GPU", "map_fd_s"), ("D3-GPU", "gram_gradient_s")]:
            primary = fraction[component] >= .3 and fraction[component] == max(fraction.values())
            gpu_status = list(table(stages, "gpu-status.csv")) if kind == "D2-GPU" else []
            ran = bool(gpu_status and gpu_status[0]["status"] == "completed")
            gates[kind] = dict(eligible=primary, executed=ran, measured_cached_wall_fraction=fraction,
                               reason="isolated_CPU_buffer_roundtrip_benchmark" if ran else
                               "dominant_at_least_30pct_requires_transfer_benchmark" if primary else "component_not_dominant_30pct_wall")
    return gates


def paired_ends(stages):
    restoration = []
    for key, (_, summary) in stages.items():
        if not key.startswith("d-replay-"):
            continue
        by_id = {a["id"]: a for a in summary["arms"]}
        for aid, trial in by_id.items():
            if "_rep" not in aid or trial["method"] != "C4":
                continue
            cached = by_id.get(aid + "-cache")
            if not cached:
                continue
            final_delta = max(abs(a - b) for a, b in zip(trial["final"], cached["final"]))
            restoration.append(dict(stage=key, arm=aid, reference_pass=trial["pass"], cached_pass=cached["pass"],
                                    same_query_count=trial["evaluations"] == cached["evaluations"], same_stop=trial["stop"] == cached["stop"],
                                    max_theta_difference=final_delta, hard_held_raw_nrmse_reference=trial["held"], hard_held_raw_nrmse_cached=cached["held"]))
    return restoration


def gpu_ends(stages):
    costs = {(r["stage"], r["arm"]): r for r in table(stages, "arm-cost.csv")}
    result = []
    for key, (_, stage) in stages.items():
        if not key.startswith("d-gpu-recovery-"):
            continue
        refkey = key.replace("gpu-recovery-", "replay-", 1)
        if refkey not in stages:
            continue
        reference = {a["id"]: a for a in stages[refkey][1]["arms"]}
        for trial in stage["arms"]:
            gpu_id = trial["id"]
            if "_rep" not in gpu_id or trial["method"] != "C4-gpu":
                continue
            ref = reference[gpu_id.removesuffix("-gpu")]
            base_cost = costs[refkey, ref["id"]]
            gpu_cost = costs[key, gpu_id]
            ref_cm, gpu_cm = max(ref["positionCm"]), max(trial["positionCm"])
            ref_intensity, gpu_intensity = max(ref["intensityRelative"]), max(trial["intensityRelative"])
            acceptable = ((not ref["pass"] or trial["pass"]) and gpu_cm <= ref_cm + .5 and gpu_intensity <= ref_intensity + .01
                          and trial["held"] <= ref["held"] + .01)
            result.append(dict(stage=key, arm=gpu_id, reference_pass=ref["pass"], gpu_pass=trial["pass"], gate_pass=acceptable,
                               reference_camera=int(base_cost["opt_camera"]), gpu_camera=int(gpu_cost["opt_camera"]),
                               reference_wall_s=float(base_cost["opt_wall_s"]), gpu_wall_s=float(gpu_cost["opt_wall_s"]),
                               reference_max_cm=ref_cm, gpu_max_cm=gpu_cm, reference_max_intensity=ref_intensity, gpu_max_intensity=gpu_intensity,
                               reference_hard_held_raw_nrmse=ref["held"], gpu_hard_held_raw_nrmse=trial["held"],
                               reference_stop=ref["stop"], gpu_stop=trial["stop"]))
    return result

def export(roots, out):
    stages = collect(roots)
    if not stages:
        raise ValueError("No completed E25 stages")
    validate(stages)
    out.mkdir(parents=True, exist_ok=True)
    gates = {}
    for root in roots:
        if (root / "gates.json").exists():
            gates.update(json.loads((root / "gates.json").read_text(encoding="utf-8")))
    evidence_gates(stages, gates)
    replay = gpu_ends(stages)
    if replay:
        primary = [r for r in replay if r["stage"] == "d-gpu-recovery-131-complex"]
        win = (1 - statistics.median(r["gpu_wall_s"] for r in primary) / statistics.median(r["reference_wall_s"] for r in primary)) if primary else None
        gates["D2-native-replay"] = dict(replayed_pairs=len(replay), quality_gate_pass=all(r["gate_pass"] for r in replay),
                                         median_wall_reduction_131=win, performance_gate_pass=win is not None and win >= .2,
                                         default_method_changed=False)
    dump(out / "stage-summary.json", summarise(stages, gates))
    for name in ("checkpoints", "view-metrics", "visibility-response-audit", "mapping-fd-audit", "loss-path", "view-ablation",
                 "numeric-parity", "fixed-inputs", "complex-lamps", "observation-audit", "native-steps", "failed-lm-repeat", "hotspots",
                 "mask-stats", "trust-clipping", "cross-lamp", "held-camera", "soft-fallback-gate", "moment-gate", "moment-multiplication", "moment-direction", "gpu-transfer", "gpu-parity", "gpu-status"):
        write_csv(out / (name + ".csv"), table(stages, name + ".csv"))
    write_csv(out / "jacobian-diagnostics.csv", [dict(kind="solver", **r) for r in table(stages, "jacobians.csv")] +
              [dict(kind="fixed_anchor", **r) for r in table(stages, "local-fd.csv")])
    write_csv(out / "profiling.csv", table(stages, "arm-cost.csv"))
    write_csv(out / "end-to-end-parity.csv", paired_ends(stages))
    write_csv(out / "gpu-end-to-end.csv", replay)
    write_csv(out / "query-log.csv", table(stages, "model-queries.csv"),
              ("stage", "arm", "query", "actual_camera", "phase", "shadows", "axis", "physical_step", "cameras"))
    by_cost = {(r["stage"], r["arm"]): r for r in table(stages, "arm-cost.csv")}
    def seed(r):
        match = re.match(r"(?:seed|dir)(\d+)", r["scenario"])
        return int(match.group(1)) if match else ""
    write_csv(out / "trials.csv", (dict(r, seed=seed(r), **{k: by_cost[r["stage"], r["arm"]][k] for k in ("fit_views", "cap_camera", "opt_camera", "scene_sha")})
                                    for r in table(stages, "trials.csv")))
    with zipfile.ZipFile(out / "raw-data.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for key, (folder, _) in stages.items():
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix in {".txt", ".csv", ".json"}:
                    archive.write(p, f"{key}/{p.name}")
    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    examples = {
        "c-complex-911": (("gt_on", "gt_v0.png"), ("dir911_20cm_C3", "initial_v0.png"), ("dir911_20cm_C3", "best_v0.png"),
                          ("dir911_20cm_C4", "best_v0.png"), ("dir911_20cm_H0", "best_v0.png")),
        "a-classroom": (("L6_hotspots", "raw_error_v2.png"),),
        "c-audit-classroom": tuple(("mask_e24_seed509_classroom_near_log_failed", f"v2_{n}.png") for n in
                                    ("mask", "weak", "off_intensity_response", "mapped_residual", "hard_position_response")),
    }
    for stage, paths in examples.items():
        if stage not in stages:
            continue
        for name, image in paths:
            src = stages[stage][0] / "captures" / name / image
            if src.is_file():
                shutil.copyfile(src, figs / f"{stage}_{name}_{image}")
    (out / "README.md").write_text(results_readme(stages, gates), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roots", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, default=ROOT / "docs/blackbox-e25")
    args = p.parse_args()
    export(args.roots, args.output)


if __name__ == "__main__":
    main()

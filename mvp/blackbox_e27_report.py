"""Build E27's evidence package without Unity buffers, caches, or per-step images."""
import argparse
import collections
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import struct
import zipfile
from pathlib import Path

from blackbox_e27 import ROOT, PUBLISH, decisions, measured, frozen_manifest
from blackbox_e26 import dump, read

TABLES = {
    "view-gradient.csv", "fixed-proposals.csv", "branch-trace.csv", "exploration-trace.csv", "model-switch.csv",
    "color-native.csv", "color-jacobian.csv", "layout-parity.csv", "gpu-parity.csv", "profiling.csv", "query-log.csv",
    "checkpoints.csv", "view-metrics.csv", "branches.csv", "gradient-synthetic.csv", "temperature-6500.csv", "color-response.csv", "histogram-config.csv",
}
AUDITS = {"audit", "color-audit", "temperature-audit", "gpu-parity", "mixed-layout", "gram-gpu"}


def csv_write(path, data, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data)


def csv_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        yield from csv.DictReader(stream)


def parity(output, results):
    names = ["0-OK20-legacy-C3-4725", "0-OK20-baseline-C3-4725"]
    if not all(r in results and results[r]["status"] == "completed" for r in names):
        return None
    dirs = [output / "arms" / r for r in names]
    left, right = [list(csv_rows(p / "query-log.csv")) for p in dirs]
    keys = ["cameras", "phase", "shadows", "axis", "physical_step", "views", "full_f", "theta"]
    same_trajectory = len(left) == len(right) and all(all(a[k] == b[k] for k in keys) for a, b in zip(left, right))
    fields = ("final", "cameras", "models", "loss", "targetSha256", "firstPassCamera", "stop", "heldRaw", "heldLog", "heldDisplay")
    eq = all(results[names[0]][k] == results[names[1]][k] for k in fields)
    return {"trajectory_exact": same_trajectory, "endpoint_exact": eq, "render_queries": len(left), "pass": same_trajectory and eq}


def environment(output):
    patterns = dict(unity=r'Initialize engine version:\s*([\w.]+)',
                    urp=r'com\.unity\.render-pipelines\.universal@([\w.]+)',
                    api=r'Forcing GfxDevice:\s*(.+)', gpu=r'^\s*Renderer:\s*(.+?)(?: \(ID=|$)',
                    driver=r'^\s*Driver:\s*([0-9.]+)')
    devices = collections.defaultdict(list)
    for path in sorted(output.glob('unity-*.log')):
        text = path.read_text(encoding='utf-8', errors='replace')
        values = {k: match.group(1).strip() for k, rx in patterns.items() if (match := re.search(rx, text, re.M))}
        if len(values) == len(patterns):
            devices[tuple(values.items())].append(path.name)
    return [{'fingerprint': dict(k), 'stage_logs': files} for k, files in devices.items()]


def arm_evidence(output, a, raw):
    """Check the raw trial contract without interpreting a GT miss as an execution failure."""
    arm_id = a['id']
    d = output / 'arms' / arm_id
    for key in ('id', 'stage', 'case', 'method', 'policy', 'seed', 'cap'):
        if raw.get(key) != a[key]:
            return 'failed', f'receipt {key} differs from frozen arm'
    if not 0 <= raw['cameras'] <= a['cap']:
        return 'failed', 'actual optimization camera budget exceeded'
    if raw['status'] != 'completed':
        return raw['status'], raw.get('reason', '')
    for file in ('frozen-render-inputs.json', 'target.csv', 'profiling.csv'):
        if not (d / file).is_file():
            return 'incomplete', f'missing {file}'
    pinned = read(d / 'frozen-render-inputs.json')
    target = next(csv_rows(d / 'target.csv'))
    if pinned['targetSha256'] != raw['targetSha256'] or target['sha256'] != raw['targetSha256'] or float(target['repeat_raw_loss']) > 1e-12:
        return 'failed', 'target checksum or repeated native capture differs'
    for label in ('initial', 'truth'):
        values = pinned[label]
        actual = hashlib.sha256(struct.pack(f'<{len(values)}d', *values)).hexdigest()
        if actual != pinned[label + 'Float64Sha256'] or values != raw[label]:
            return 'failed', f'frozen Float64 {label} differs from render receipt'
    if a['stage'] == 'A1':
        path = d / 'fixed-proposals.csv'
        names = {r['source'].split('-')[0] for r in csv_rows(path) if r['evaluated'] == '1'} if path.is_file() else set()
        if not {f'V{i}' for i in range(1, 8)} <= names:
            return 'incomplete', 'one or more fixed view subsets lack a native full F evaluation'
    if a['policy'] in AUDITS:
        return 'completed', ''
    if not raw.get('legal') or not raw.get('final') or not math.isfinite(raw['loss']):
        return 'failed', 'no finite, legal full-scene endpoint'
    for file in ('query-log.csv', 'view-metrics.csv', 'checkpoints.csv'):
        if not (d / file).is_file():
            return 'incomplete', f'missing {file}'
    profile = next(csv_rows(d / 'profiling.csv'))
    if int(profile['hard']) + int(profile['off']) != raw['cameras']:
        return 'failed', 'Hard + Off render count differs from receipt'
    last_camera = 0
    saw_endpoint = False
    for line in csv_rows(d / 'query-log.csv'):
        camera = int(line['cameras'])
        if not last_camera <= camera <= raw['cameras']:
            return 'failed', 'non-monotonic or over-budget query ledger'
        last_camera = camera
        full = float(line['full_f'])
        if math.isfinite(full):
            saw_endpoint |= math.isclose(full, raw['loss'], rel_tol=1e-9, abs_tol=1e-12)
    if last_camera != raw['cameras']:
        return 'failed', 'final query count differs from receipt'
    if not saw_endpoint:
        return 'incomplete', 'returned loss missing from full objective ledger'
    if a['stage'] == 'A2' and a['policy'] == 'A2' and raw['triggered']:
        mask = d / 'fixed-proposals.csv'
        tested = {r['source'].split('-')[0] for r in csv_rows(mask) if r['evaluated'] == '1'} if mask.is_file() else set()
        if not {f'V{k}' for k in range(1, 7)} <= tested:
            return 'incomplete', 'not all proper subsets produced a legal native Full F evaluation'
    if a['stage'] == 'C':
        path = d / 'histogram-config.csv'
        config = next(csv_rows(path), {}) if path.is_file() else {}
        applied = config.get('policy')
        if (not applied or (a['policy'] != 'selected-histogram' and a['policy'] != applied)
                or int(config['window']) != (0 if applied == 'rgb' else 32 if applied.endswith('32') else 16)):
            return 'incomplete', 'proxy configuration does not establish the frozen RGB/histogram arm'
        p = d / 'fixed-proposals.csv'
        tested = p.is_file() and any(row['evaluated'] == '1' for row in csv_rows(p))
        if not tested:
            return 'incomplete', 'missing a feasible native Full F score for the fixed proxy proposal'
        if not a['policy'].endswith('32') and raw['models'] < 4 and raw['stop'] != 'near_zero_residual':
            return 'incomplete', 'fixed model plus three complete proxy branch models not observed'
    if a['stage'] == 'D3' and a['case'].startswith('complex-white-') and a['case'].endswith('-xyzi'):
        peer = (output / 'arms' / arm_id.replace('-xyzi-baseline-', '-xyzit-baseline-') / 'result.json')
        if peer.is_file():
            r = read(peer)
            pinned = len(raw['initial']) * 4 // 5
            if raw['initial'][:pinned] != r['initial'][:pinned] or raw['targetSha256'] != r['targetSha256']:
                return 'incomplete', 'matched 28D/35D XYZ and intensity inputs or target differ'
    if a['policy'] in {'A3', 'A4', 'A5', 'A6', 'A7', 'G4'} and raw['triggered']:
        path = d / 'branches.csv'
        branches = list(csv_rows(path)) if path.is_file() else []
        if len(branches) != 3 or len({r['common_cap'] for r in branches}) != 1:
            return 'incomplete', 'three branches did not receive a common model allocation'
        short = [r for r in branches if int(r['actual_models']) < int(r['common_cap'])]
        if any(r['stop'] != 'near_zero_residual' for r in short):
            return 'incomplete', 'actual branch models below common allocation without a near-zero residual'
    return 'completed', ''


def rms_view_table(output, dest, trials):
    out = []
    for row in trials:
        path = output / 'arms' / row['id'] / 'view-metrics.csv'
        if not path.is_file():
            continue
        keys = ('raw_nrmse', 'log_nrmse', 'display_nrmse')
        groups = collections.defaultdict(list)
        for v in csv_rows(path):
            groups[v['state'], v['width'], v['role']].append(v)
        for (state, width, role), views in groups.items():
            out.append(dict(arm=row['id'], status=row['status'], state=state, width=width, role=role,
                            views=';'.join(v['view'] for v in views), **{
                                key: math.sqrt(statistics.mean(float(v[key]) ** 2 for v in views)) for key in keys}))
    csv_write(dest / 'view-rms.csv', out, ('arm', 'status', 'state', 'width', 'role', 'views',
              'raw_nrmse', 'log_nrmse', 'display_nrmse'))


def branch_table(output, dest, trials):
    evidence = []
    for r in trials:
        d = output / 'arms' / r['id']
        if r['stage'] not in {'A2', 'A3'} or not (d / 'branches.csv').is_file():
            continue
        traces = collections.defaultdict(list)
        for q in csv_rows(d / 'query-log.csv'):
            full = float(q['full_f'])
            if '_branch_' in q['segment'] and math.isfinite(full):
                traces[q['segment']].append((int(q['cameras']), full))
        for branch in csv_rows(d / 'branches.csv'):
            segment = f"{r['policy']}_branch_{branch['branch']}"
            points = traces[segment]
            if not points:
                continue
            start = points[0][1]
            f_min = min(v for _, v in points)
            f_max = max(v for _, v in points)
            first = next((i for i, v in points if start - v > max(1e-14, 1e-9 * start)), '')
            worse_then_better = False
            saw_worse = False
            for _, v in points:
                saw_worse |= v - start > max(1e-14, 1e-9 * start)
                worse_then_better |= saw_worse and start - v > max(1e-14, 1e-9 * start)
            evidence.append(dict(arm=r['id'], case=r['case'], method=r['method'], segment=segment, views=branch['views'],
                                 seed=r['seed'], cap=r['cap'], execution=r['status'], branch_cameras=branch['cameras'],
                                 requested_models=branch['requested_models'], common_cap=branch['common_cap'],
                                 actual_models=branch['actual_models'], stop=branch['stop'], full_evaluations=len(points),
                                 start_full_f=start, peak_full_f=f_max, best_full_f=f_min, end_full_f=points[-1][1],
                                 peak_over_start=f_max / max(1e-12, start), first_full_improvement_camera=first,
                                 observed_worse_then_better=worse_then_better, final_gt_pass=r['pass']))
    if evidence:
        csv_write(dest / 'branch-outcomes.csv', evidence, list(evidence[0]))
    return evidence


def paired_table(dest, trials, arms):
    arm_by_id = {a['id']: a for a in arms}
    complete = [r for r in trials if r['status'] == 'completed']
    index = collections.defaultdict(list)
    for r in complete:
        index[r['stage'], r['case']].append(r)
    baseline = {'A2': 'A0', 'A3': 'G0', 'B1': 'M0', 'B2': 'X0', 'B3': 'B0', 'C': 'rgb', 'transfer': 'X0'}
    paired = []
    for r in complete:
        a = arm_by_id[r['id']]
        stage = a['stage']
        peers = index[stage, a['case']]
        if stage == 'E' and a['policy'] == 'gpu-timed' and a['gpu']:
            candidates = [b for b in peers if not arm_by_id[b['id']]['gpu'] and arm_by_id[b['id']]['repeat'] == a['repeat']]
        elif stage in {'D2', 'D3'} and a['case'].endswith(('-xyziuv', '-xyzit')):
            key = a['case'].rsplit('-', 1)[0] + '-xyzi'
            if stage == 'D3' and a['case'].startswith('room-') and a['case'].endswith('-xyziuv'):
                key = key.replace('room-', 'room-colored-', 1)
            candidates = [b for b in index[stage, key] if b['method'] == ('Off' if a['method'] == 'Off' else 'C3')]
        elif stage in baseline and a['policy'] != baseline[stage] and a['method'] != 'H0':
            candidates = [b for b in peers if b['policy'] == baseline[stage]]
            if not candidates and stage == 'B1' and a['policy'] == 'M3' and a['case'].endswith('-S'):
                candidates = [b for b in peers if b['policy'] == 'M1']
        else:
            continue
        candidates = [b for b in candidates if b['cap'] == r['cap'] and b['targetSha256'] == r['targetSha256']]
        if not candidates:
            continue
        b = next((v for v in candidates if v['seed'] == r['seed']), candidates[0])
        context = dict(stage=stage, case=a['case'], arm=a['id'], baseline_arm=b['id'], policy=a['policy'],
                       seed=a['seed'], repeat=a.get('repeat', ''), cap=a['cap'], objective_sha256=r['targetSha256'])
        for key in ('pass', 'loss', 'cameras', 'models', 'seconds', 'firstPassCamera', 'firstPassSeconds',
                    'maxCm', 'maxRelative', 'maxColor', 'heldRaw', 'heldLog', 'heldDisplay', 'stop'):
            context['baseline_' + key], context['trial_' + key] = b[key], r[key]
        context['full_f_ratio'] = r['loss'] / max(b['loss'], 1e-12)
        context['endpoint_wall_ratio'] = b['seconds'] / max(1e-12, r['seconds'])
        same_first = b['firstPassCamera'] == r['firstPassCamera'] and r['firstPassCamera'] >= 0
        context['same_first_pass_cameras'] = same_first
        context['matched_first_pass_wall_ratio'] = (b['firstPassSeconds'] / max(1e-12, r['firstPassSeconds']) if same_first else '')
        paired.append(context)
    if paired:
        csv_write(dest / 'paired-trials.csv', paired, list(paired[0]))
    return paired


def export(output, dest):
    manifest = frozen_manifest(output)
    arms = manifest['arms']
    results = measured(output, arms)
    gate = decisions(manifest, output)
    dest.mkdir(parents=True, exist_ok=True)
    for revision in output.glob('experiment-manifest*.json'):
        shutil.copyfile(revision, dest / revision.name)
    dump(dest / 'gates.json', gate)
    devices = environment(output)
    dump(dest / 'render-environment.json', devices)
    frozen = {}
    for a in arms:
        if a['id'] not in results:
            continue
        p = output / 'arms' / a['id'] / 'frozen-render-inputs.json'
        if p.is_file():
            row = read(p)
            row.pop('initial', None)
            row.pop('truth', None)
            frozen[a['id']] = row
    dump(dest / 'frozen-render-inputs-index.json', frozen)
    matrix, trials = [], []
    for a in arms:
        ident = a['id']
        raw = results.get(ident)
        status, reason = (arm_evidence(output, a, raw) if raw else
                          ('not_triggered', gate[ident]['reason']) if gate[ident]['launch'] is False else
                          ('not_run', gate[ident]['reason'] if gate[ident]['launch'] is None else 'no completed run receipt'))
        measured_pass = raw['pass'] if status == 'completed' and a['policy'] not in AUDITS else ''
        matrix.append({k: a[k] for k in ('id', 'stage', 'case', 'method', 'policy', 'seed', 'cap', 'minimum_trials', 'mandatory', 'condition')} |
                      dict(status=status, reason=reason, launch=gate[ident]['launch'], online_trigger=raw.get('triggered', '') if raw else '', pass_gt=measured_pass,
                           source_sha256=raw.get('sourceSha256', '') if raw else '', target_sha256=raw.get('targetSha256', '') if raw else ''))
        if raw is None or a['policy'] in AUDITS:
            continue
        row = {k: v for k, v in raw.items() if not isinstance(v, list)}
        rec = output / 'arms' / ident / 'frozen-render-inputs.json'
        active = read(rec)['active'] if rec.exists() else next((c['active'] for c in manifest['cases'] if c['id'] == a['case']), [])
        for field, name in (('positionCm', 'position_cm'), ('intensityRelative', 'intensity_rel'), ('colorError', 'color_error')):
            values = raw.get(field) or []
            row['median_' + name] = statistics.median(values) if values else ''
            row['worst_lamp_' + name] = active[max(range(len(values)), key=values.__getitem__)] + 1 if active and len(active) == len(values) else ''
        axes = raw.get('config', '').rsplit(':', 1)[-1]
        row.update(status=status, condition=a['condition'], comparison_pass=measured_pass, active_lights=len(active),
                   dimension=len(active) * len(axes) if axes and active else '')
        trials.append(row)
    csv_write(dest / 'manifest-status.csv', matrix, list(matrix[0]))
    cols = sorted(set().union(*(t.keys() for t in trials))) if trials else ['id', 'status']
    csv_write(dest / 'trials.csv', trials, cols)
    csv_write(dest / 'color-trials.csv', [r for r in trials if ':white:' in r.get('config', '') or ':colored:' in r.get('config', '')], cols)
    rms_view_table(output, dest, trials)
    pairs = paired_table(dest, trials, arms)
    branch_outcomes = branch_table(output, dest, trials)
    stages = {}
    for stage in dict.fromkeys(a['stage'] for a in arms):
        rows = [r for r in matrix if r['stage'] == stage]
        statuses = collections.Counter(r['status'] for r in rows)
        observed = [r for r in rows if r['pass_gt'] != '']
        stages[stage] = dict(required=sum(r['mandatory'] for r in rows), required_completed=sum(r['mandatory'] and r['status'] == 'completed' for r in rows),
                             conditional=len(rows) - sum(r['mandatory'] for r in rows), statuses=dict(statuses), measured=len(observed),
                             pass_gt=sum(bool(r['pass_gt']) for r in observed))
    from blackbox_e27_gradients import compare as compare_gradients
    source = output / 'arms/E-OK20-gpu-parity-C3-4725/gradient-synthetic.csv'
    gradient_checks = list(compare_gradients(source)) if source.is_file() else []
    gradient_ok = len(gradient_checks) == 24 and all(row['passed'] for row in gradient_checks)
    if gradient_checks:
        csv_write(dest / 'gradient-reference.csv', gradient_checks, list(gradient_checks[0]))
    source = source.with_name('gpu-parity.csv')
    native_gpu_rows = [r for r in csv_rows(source) if r['input'] in {'native_all_axes', 'native_all_axes_bound'}] if source.is_file() else []
    native_gpu_ok = len(native_gpu_rows) == 204 and all(row['pass'] == '1' for row in native_gpu_rows)
    exact4 = parity(output, results)
    summary = dict(baseline=manifest['baseline'], stages=stages, total=len(arms), completed=sum(r['status'] == 'completed' for r in matrix),
                   required=sum(r['mandatory'] for r in matrix), required_completed=sum(r['mandatory'] and r['status'] == 'completed' for r in matrix),
                   complete=bool(exact4 and exact4['pass'] and len(devices) == 1 and gradient_ok and native_gpu_ok) and
                   all(r['status'] == 'completed' or (not r['mandatory'] and r['status'] == 'not_triggered') for r in matrix),
                   layout_exact=exact4, independent_gradient_pass=gradient_ok, real_native_5d_6d_gpu_pass=native_gpu_ok)
    dump(dest / 'stage-summary.json', summary)
    with zipfile.ZipFile(dest / 'raw-parameters.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted((output / 'arms').glob('*/*')):
            if path.parent.name in results and path.suffix in {'.csv', '.json'}:
                archive.write(path, path.relative_to(output))
        for path in sorted(output.glob('job-*.json')):
            archive.write(path, path.relative_to(output))
        for path in sorted((output / 'scene').glob('*/color-coupling.csv')):
            archive.write(path, path.relative_to(output))
        for path in sorted((output / 'superseded').rglob('*')):
            if path.is_file() and path.suffix in {'.csv', '.json'}:
                archive.write(path, path.relative_to(output))
    for name in sorted(TABLES):
        files = [output / 'arms' / a['id'] / name for a in arms if a['id'] in results]
        files = [p for p in files if p.is_file() and p.stat().st_size > 0]
        if not files:
            continue
        headers = list(dict.fromkeys(k for p in files for k in p.open(encoding='utf-8-sig').readline().rstrip().split(',')))
        if name == 'query-log.csv' and sum(p.stat().st_size for p in files) > 16 * 1024 ** 2:
            # Full per-evaluation ledger (including float64 parameters) is in the linked raw archive.
            csv_write(dest / name, ({'id': p.parent.name, 'archive_member': p.relative_to(output).as_posix()} for p in files), ['id', 'archive_member'])
        else:
            csv_write(dest / name, (r for p in files for r in csv_rows(p)), headers)
    if exact4:
        dump(dest / 'layout-exact-4d.json', exact4)
    choices = []
    measured_rows = [r for r in trials if r['status'] == 'completed']
    for pass_gt in (True, False):
        candidates = [r for r in measured_rows if r['pass'] == pass_gt and (output / 'arms' / r['id'] / 'final-v0.png').is_file()]
        seen = set()
        for r in candidates:
            key = r['config'].split(':')[0]
            if key in seen:
                continue
            seen.add(key)
            choices.append(r)
    for row in choices:
        ident = row['id']
        directory = dest / 'figures' / ident
        directory.mkdir(parents=True, exist_ok=True)
        for png in (output / 'arms' / ident).glob('*.png'):
            shutil.copyfile(png, directory / png.name)
    lines = ['# E27｜原生 RGB 搜索与颜色恢复', '', f'冻结基线：`{manifest["baseline"]}`。', '',
             f'验收状态：**{"完成" if summary["complete"] else "未完成"}**。必做 {summary["required_completed"]}/{summary["required"]} 臂；'
             f'全表共 {summary["total"]} 臂。未运行、未触发、未完成、执行失败不纳入恢复率；完成但未达到 GT 门槛的试次纳入。', '',
             '| 阶段 | 必做完成/总数 | 已完整验收的优化试次 | 达到 GT 门槛 | 状态 |', '|---|---:|---:|---:|---|']
    for stage, s in stages.items():
        lines.append(f'| {stage} | {s["required_completed"]}/{s["required"]} | {s["measured"]} | {s["pass_gt"]} | ' +
                     ' / '.join(f'{key} {cnt}' for key, cnt in s['statuses'].items()) + ' |')
    lines += ['', '## 口径', '', '优化使用 Unity URP 原生 RGB。主排名采用三视角冻结 Signed Log 目标和逐视角能量权重。'
              '相机预算按实际渲染图数计，目标、机制诊断和独立验收另计。GT 只参与离线参数核验，未进入插件默认配置。', '']
    for device in devices:
        info = device['fingerprint']
        lines += [f'实跑环境：Unity {info["unity"]} / URP {info["urp"]} / {info["api"]} / '
                  f'{info["gpu"]}（驱动 {info["driver"]}）。各批见 [环境指纹](render-environment.json)。', '']
    if exact4:
        lines += ['## 4D 兼容性', '',
                  f'隐式/显式布局的真实渲染轨迹逐项一致：{exact4["trajectory_exact"]}；终态一致：{exact4["endpoint_exact"]}。', '']
    fixed = [r for r in matrix if r['stage'] == 'A1' and '-fd20-' in r['id'] and r['status'] == 'completed']
    if fixed:
        lines += ['## 固定状态', '', '| 起点 | 信任域尺度梯度保留率 | 实降候选/实际试拍 | 最大实降 |', '|---|---:|---:|---:|']
        for r in fixed:
            d = output / 'arms' / r['id']
            gradients = next(x for x in csv_rows(d / 'view-gradient.csv') if x['coordinates'] == 'trust' and x['view'] == '3')
            p = [float(x['actual']) for x in csv_rows(d / 'fixed-proposals.csv') if x['evaluated'] == '1']
            lines.append(f'| {r["case"]} | {float(gradients["gradient_retention"]):.4%} | '
                         f'{sum(x > 1e-12 for x in p)}/{len(p)} | {max(p, default=float("nan")):.6g} |')
        lines += ['', '梯度保留率衡量合成梯度的模长，不是照明贡献比例。固定试探没有下降时，只能说明已测试方向和步幅没有下降。', '']
    native = [x for r in matrix if r['stage'] == 'D1' and r['status'] == 'completed' for x in csv_rows(output / 'arms' / r['id'] / 'color-native.csv')]
    if native:
        err = max(float(x['relative_l2']) for x in native)
        lines += ['## 原生色度', '', f'原生有效颜色与亮度归一化目标的最大相对 L2 误差为 {err:.3g}，共 {len(native)} 条颜色记录。'
                  '这项审计验证 Unity 的滤镜和有效强度编码；联合参数恢复见 `color-trials.csv`。', '']
    gpu_log = dest / 'gpu-parity.csv'
    if gpu_log.is_file():
        numeric = list(csv_rows(gpu_log))
        if numeric:
            lines += ['## GPU 数值', '', f'列、Gram、梯度及非零提案 {sum(x["pass"] == "1" for x in numeric)}/{len(numeric)} 项通过固定输入阈值。'
                      '端到端速度与成功率使用各自的 CPU/GPU 配对试次；独占 GPU 设备计时不可用。', '']
    decisive = [r for r in branch_outcomes if r['case'] == 'L6-S' and r['method'] == 'Off' and r['observed_worse_then_better']
                and r['final_gt_pass'] and r['execution'] == 'completed']
    if decisive:
        r = min(decisive, key=lambda x: int(x['first_full_improvement_camera']))
        lines += ['## L6 跨吸引域', '',
                  f'关闭阴影的 L6-S 停滞快照上，`{r["arm"]}` 的视角 {r["views"]} 分支使全 F 从 '
                  f'{r["start_full_f"]:.6g} 暂升到 {r["peak_full_f"]:.6g}，随后降至 '
                  f'{r["best_full_f"]:.6g}，整臂达到 GT 门槛。全 F 先升后降均有原生全视角评分。'
                  '该失败模式在关闭阴影时仍存在，因此不能只用硬阴影跳变解释。',
                  '', '[各分支的实际模型数、短停原因与全 F 轨迹摘要](branch-outcomes.csv)。', '']
    if gradient_checks:
        lines += [f'独立梯度方向核对：{sum(r["passed"] for r in gradient_checks)}/{len(gradient_checks)}；'
                  'CAGrad 使用 c=0.4，PCGrad 使用冻结循环视角顺序，UPGrad 的非上升性质只用于未预条件方向。', '']
    if pairs:
        lines += ['## 同预算比较', '', '| 阶段 | 预算 | 配对臂数 | 未达标→达标 | 达标→未达标 | 全 F 比值中位 |',
                  '|---|---:|---:|---:|---:|---:|']
        for stage, cap in dict.fromkeys((r['stage'], r['cap']) for r in pairs):
            group = [r for r in pairs if r['stage'] == stage and r['cap'] == cap]
            gains = sum(not r['baseline_pass'] and r['trial_pass'] for r in group)
            losses = sum(r['baseline_pass'] and not r['trial_pass'] for r in group)
            lines.append(f'| {stage} | {cap} | {len(group)} | {gains} | {losses} | {statistics.median(r["full_f_ratio"] for r in group):.5g} |')
        lines += ['', '每个种子、首次达标图数与墙钟、终态误差、留出 Raw/Log/Display，见 [配对表](paired-trials.csv)。'
                  '配对臂数统计策略×种子，不等于独立场景数。', '']
        gpu_pairs = [r for r in pairs if r['stage'] == 'E']
        if gpu_pairs:
            lines += ['## GPU 端到端', '', '| 测试 | 重复 | CPU/GPU 图数 | CPU/GPU 总秒 | 首次过线 CPU/GPU 图数 | 同首过成本提速 |',
                      '|---|---:|---:|---:|---:|---:|']
            for r in gpu_pairs:
                ratio = r['matched_first_pass_wall_ratio']
                cpu_first = r['baseline_firstPassCamera']; gpu_first = r['trial_firstPassCamera']
                lines.append(f'| {r["case"]} | {r["repeat"]} | {r["baseline_cameras"]} / {r["trial_cameras"]} | '
                             f'{r["baseline_seconds"]:.2f} / {r["trial_seconds"]:.2f} | '
                             f'{cpu_first if cpu_first >= 0 else "—"} / {gpu_first if gpu_first >= 0 else "—"} | '
                             f'{f"{ratio:.2f}×" if ratio != "" else "—"} |')
            lines += ['', '首次达标图数一致时才计算同首过成本提速；终态墙钟包含目标、验收及各自的实际查询量。', '']
    b2 = [r for r in measured_rows if r['stage'] == 'B2' and r['cap'] == 4725]
    if b2:
        policies = [f'X{k}' for k in range(6)]
        lines += ['## 跨吸引域搜索 B2', '',
                  '| 冻结起点 | ' + ' | '.join(policies) + ' |',
                  '|---|' + '---:|' * len(policies)]
        for case in dict.fromkeys(r['case'] for r in b2):
            table = [[r for r in b2 if r['case'] == case and r['policy'] == policy] for policy in policies]
            values = [f'{sum(x["pass"] for x in group)}/{len(group)}' if group else '—' for group in table]
            lines.append(f'| {case} | ' + ' | '.join(values) + ' |')
        lines += ['', '分子为达到 GT 门槛的种子数，分母为完整执行的该起点/策略试次；不是独立场景数。'
                  'X0 原 LM，X1 坐标/正交试探后 LM，X2 扰动和局部 LM，X3 CMA-ES 后 LM，X4 非单调 LM，X5 纯 CMA-ES。', '']
        for r in sorted((x for x in b2 if x['pass']), key=lambda x: (x['case'], x['policy'], x['seed'])):
            lines.append(f'- `{r["id"]}`：首次达标 {r["firstPassCamera"]} 张，终点 {r["cameras"]} 张，'
                         f'{r["maxCm"]:.4g} cm / {r["maxRelative"]:.3%}；留出 Log NRMSE {r["heldLog"]:.5g}。')
        lines.append('')
    scale = [r for r in measured_rows if r['stage'] in {'D2', 'D3'}]
    if scale:
        lines += ['## 色度与白光恢复', '', '| 阶段 | 配置 | 维度 | 方法 | 达标 / 实跑 | 最终最大位置误差中位 (cm) |',
                  '|---|---|---:|---|---:|---:|']
        lookup = collections.defaultdict(list)
        for r in scale:
            grouping = (r['stage'], r['config'].rsplit(':', 1)[0], r['dimension'], r['method'])
            lookup[grouping].append(r)
        for (stage, conf, dim, m), rows in sorted(lookup.items()):
            lines.append(f'| {stage} | {conf} | {dim} | {m} | {sum(r["pass"] for r in rows)}/{len(rows)} | '
                         f'{statistics.median(r["maxCm"] for r in rows):.4g} |')
        lines += ['', '细分饱和度、留出视角、逐灯误差及预算终止依据见 [颜色试次](color-trials.csv)。', '']
    known = [r for r in measured_rows if r['stage'] in {'B3', 'C', 'transfer'}]
    if known:
        lines += ['## 实跑对照', '', '| 臂 | 图数 | 最终 F | 最大位置误差 (cm) | 最大强度误差 | 首次达标图数 | 留出 Log NRMSE | 达标 |',
                  '|---|---:|---:|---:|---:|---:|---:|---|']
        for r in known:
            first = r['firstPassCamera'] if r['firstPassCamera'] >= 0 else '—'
            lines.append(f'| {r["id"]} | {r["cameras"]} | {r["loss"]:.6g} | {r["maxCm"]:.4g} | {r["maxRelative"]:.3%} | '
                         f'{first} | {r["heldLog"]:.4g} | {"是" if r["pass"] else "否"} |')
        lines.append('')
    gram_gate = gate.get('E-OK20-gram-gpu-C3-4725', {})
    median = gram_gate.get('evidence', {}).get('median_gpu_timed_gram_fraction')
    if median is not None:
        lines += [f'CPU Gram/gradient 在 GPU 列融合配置中的耗时占比中位数为 {median:.2%}；'
                  f'20% 条件臂：{"触发" if gram_gate["launch"] else "未触发"}。', '']
    lines += ['[条件臂判据与实测依据](gates.json)。', '']
    lines += ['## 原始记录', '', '[逐臂对账](manifest-status.csv) · [试次](trials.csv) · [精确输入哈希](frozen-render-inputs-index.json) · [完整参数及逐次日志](raw-parameters.zip)。', '',
              '`query-log.csv` 超过 16 MiB 时记录压缩包内的逐臂路径；逐查询原始 CSV 仍完整保存。', '',
              '从版本匹配的隔离 Unity 工程运行，工作目录为仓库根目录：', '', '```powershell',
              '.venv/Scripts/python.exe mvp/blackbox_e27.py --project tmp/e24-work/project --sync',
              '.venv/Scripts/python.exe mvp/blackbox_e27_report.py --require-complete', '```', '']
    if choices:
        lines += ['## 代表性渲染', '']
        for r in choices:
            ident = r['id']
            lines.append(f'- {ident}（{"达标" if r["pass"] else "未达标"}）：[GT](figures/{ident}/gt-v0.png) · '
                         f'[初态](figures/{ident}/initial-v0.png) · [终态](figures/{ident}/final-v0.png)')
    (dest / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT / 'tmp/e27')
    p.add_argument('--export', type=Path, default=PUBLISH)
    p.add_argument('--require-complete', action='store_true')
    args = p.parse_args()
    state = export(args.input.resolve(), args.export.resolve())
    print(json.dumps({k: v for k, v in state.items() if k != 'stages'}, ensure_ascii=False))
    if args.require_complete and not state['complete']:
        raise SystemExit('E27 acceptance incomplete; publication blocked.')

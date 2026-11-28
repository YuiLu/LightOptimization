"""Freeze, run and resume the E27 Unity-native study. Only the isolated project is written."""
import argparse
import hashlib
import json
import statistics
import csv
import shutil
import subprocess
import zipfile
from pathlib import Path

from blackbox_e24 import ROOT, manifest
from blackbox_e26 import digest, dump, read

BASE = "01e51735a42e5eb8b1f4884d063a7a5b54a79b3f"
SEEDS = (1701, 1709, 1723)
PUBLISH = ROOT / "docs/blackbox-e27"


def archived(round_id, member):
    name = "raw-parameters" if round_id == 26 else "raw-data"
    with zipfile.ZipFile(ROOT / f"docs/blackbox-e{round_id}/{name}.zip") as z:
        return json.loads(z.read(member))


def cases():
    result = {}
    classroom = archived(24, "e24_a_classroom/summary.json")
    complex_room = archived(24, "e24_a_complex/summary.json")

    def add(key, scene, gt, initial, active=range(7), mode="original", target="Hard", axes="xyzi"):
        result[key] = dict(id=key, scene=scene, truth=gt, initial=initial, active=list(active), mode=mode,
                           target=target, axes=axes, initial_sha256=digest(initial), truth_sha256=digest(gt))

    l6 = next(r for r in archived(25, "a-views-classroom/summary.json")["arms"] if "_V3_" in r["id"])
    for key, pose in (("L6-I", "initial"), ("L6-S", "final")):
        add(key, "classroom", classroom["truth"], l6[pose], [5], target="Off")
    add("CF-O", "complex", complex_room["truth"], complex_room["arms"][0]["initial"], target="Off")
    for key, scene, seed, truth in (("C13", "classroom", 1301, classroom["truth"]), ("C16", "classroom", 1601, classroom["truth"]),
                                    ("OK20", "complex", 911, complex_room["truth"]), ("OK30", "complex", 1201, complex_room["truth"])):
        arm = next(a for a in archived(25, f"c-{scene}-{seed}/summary.json")["arms"] if a["method"] == "C3")
        add(key, scene, truth, arm["initial"])
        if scene == "classroom":
            add(key + "-S", scene, truth, arm["final"])
            add(key + "-Off", scene, truth, arm["initial"], target="Off")
    for seed in (509, 911, 1201, 2003, 2909):
        samples = archived(26, f"e26-d-joint-complex-{seed}/color-snapshots.json")["samples"]
        for arm in samples:
            axes = "xyzi" if "28D" in arm["arm"] else "xyzit"
            dim = 28 if axes == "xyzi" else 35
            key = f"WF{dim}" if seed == 509 else f"W{dim}-{seed}"
            add(key, "complex", arm["truth"], arm["start"], mode="white", axes=axes)
            if seed == 509:
                add(key + "-S", "complex", arm["truth"], arm["final"], mode="white", axes=axes)
    return result


def plan():
    arms = []
    def put(stage, case, method, policy="baseline", seed=0, cap=4725, condition="", **options):
        label = options.pop("label", policy)
        tail = f"-s{seed}" if seed else ""
        ident = f"{stage}-{case}-{label}-{method}-{cap}{tail}"
        arms.append(dict(id=ident, stage=stage, case=case, method=method, policy=policy, seed=seed,
                         cap=cap, condition=condition, mandatory=not bool(condition), minimum_trials=1, **options))
    put("0", "OK20", "C3", "legacy")
    put("0", "OK20", "C3")
    for c in ("L6-I", "L6-S", "C13", "C13-S", "C16", "C16-S", "WF35-S"):
        for step in (5, 20, 50):
            put("A1", c, "Off" if c.startswith("L6") else "C3", "audit", label=f"fd{step}", step_mm=step)
    for c in ("L6-I", "L6-S", "C13", "C16", "WF35"):
        m = "Off" if c.startswith("L6") else "C3"
        for policy in (range(8) if c.startswith("L6") else (0, 2, 3, 5)):
            if policy == 7:
                for models in (3, 8): put("A2", c, m, "A7", label=f"A7-{models}", branch_models=models)
            else: put("A2", c, m, f"A{policy}")
        if c == "L6-I": put("A2", c, m, "A4", label="A4-direct", direct=True)
        if not c.startswith("L6"): put("A2", c, m, "A4", cap=9450)
    for c in ("L6-I", "L6-S", "C13", "WF35"):
        m = "Off" if c.startswith("L6") else "C3"
        for g in (0, 1, 2, 3, 4) if c.startswith("L6") else (0, 1, 3, 4):
            for seed in SEEDS if g == 4 else (0,): put("A3", c, m, f"G{g}", seed=seed)
    for c in ("C13", "C16", "WF28", "WF35"):
        for k, m in enumerate(("C3", "H1", "C3", "C3")):
            put("B1", c, m, f"M{k}")
        for policy in ("M1", "M3"):
            put("B1", c + "-S", "H1", policy, direct=True)
        if c in ("C13", "WF35"): put("B1", c, "C3", "adaptive")
        if c.startswith("C"): put("B1", c + "-Off", "Off")
    for c in ("L6-S", "CF-O", "C13", "WF35", "C13-S", "C16", "WF28"):
        m = "Off" if c in ("L6-S", "CF-O") else "C3"
        for k in ((0, 2, 3) if c in ("C16", "WF28") else range(6)):
            for seed in (SEEDS if k in (1, 2, 3, 5) or (c in ("C16", "WF28") and k == 0) else (0,)):
                put("B2", c, m, f"X{k}", seed=seed, direct=c in ("L6-S", "C13-S"))
    for budget in (4725, 9450):
        for policy in ("B0", "B1"): put("B3", "CF-O", "Off", policy, cap=budget)
    put("B3", "C13", "C3", "B1")
    for c in ("L6-S", "C13-S", "C16-S", "WF35-S"):
        m = "Off" if c == "L6-S" else "C3"
        for policy in (("rgb", "hist16", "mix16", "hist32", "mix32") if c in ("L6-S", "C13-S") else ("selected-histogram",)):
            put("C", c, m, policy, condition="positive true full F at a fixed state" if c in ("C16-S", "WF35-S") else "")
    for saturation in ("moderate", "high"):
        put("D1", f"color-audit-{saturation}", "H1", "color-audit")
    for i in range(6):
        for axes in ("uv", "iuv"):
            put("D2", f"color-small-{i}-{axes}", "H1")
    for i in range(9):
        for m, a in (("C3", "xyz iuv"), ("H1", "xyz iuv"), ("C3", "xyz i")):
            put("D2", f"color-single-{i}-{a.replace(' ', '')}", m)
    for i in range(4):
        for a in ("xyziuv", "xyzi"): put("D2", f"color-pair-{i}-{a}", "C3")
    for seed in (911, 1201, 509, 2003, 2909, 3109, 4211):
        for a in ("xyziuv", "xyzi"):
            put("D2", f"color-full-{seed}-{a}", "C3", condition="" if seed == 911 else "6D >=5/6 moderate; 12D >=3/4; matched controls nondegraded")
    for c, cap in (("WF35", 5907), ("W35-2003", 5907), ("color-full-911-xyziuv", 7086), ("color-full-1201-xyziuv", 7086)):
        put("D-budget", c, "C3", cap=cap, condition="matched joint run failed at main budget and not near zero")
    for i in range(2):
        for axes in ("xyzit", "xyziuv", "xyzi"): put("D3", f"room-{i}-{axes}", "C3")
    for lamp in (1, 6):
        for i in range(2):
            for m, a in (("C3", "xyzit"), ("H1", "xyzit"), ("Off", "xyzit"), ("C3", "xyzi"), ("Off", "xyzi"), ("H1", "t"), ("H1", "it")):
                put("D3", f"class-white-{lamp}-{i}-{a}", m)
            if lamp == 6: put("D3", f"class-white-{lamp}-{i}-xyzit", "Off", "single-view")
    for a in ("xyzit", "xyzi"): put("D3", f"class-white-all-{a}", "C3")
    for seed in (911, 1201, 2003):
        for p in ("large-temp", "large-temp-intensity"):
            for a in ("xyzit", "xyzi"): put("D3", f"complex-white-{seed}-{p}-{a}", "C3")
    put("D3", "temperature-6500", "H1", "temperature-audit")
    put("E", "OK20", "C3", "gpu-parity")
    put("E", "OK20", "C3", "mixed-layout")
    put("E", "OK20", "C3", "gram-gpu", condition="measured Gram/gradient fraction >20%")
    for c in ("OK20", "OK30", "W35-911", "W35-2003", "WF35", "color-single-0-xyziuv", "color-full-911-xyziuv", "color-full-3109-xyziuv"):
        for repeat in range(1, 4 if c in ("OK20", "W35-911") else 2):
            for gpu in (False, True):
                put("E", c, "C3", "gpu-timed", label=f"{'GPU' if gpu else 'CPU'}-{repeat}", gpu=gpu,
                    repeat=repeat, condition="independent 42D arm passed" if "3109" in c else "")
    for c in ("OK20", "W35-2003", "new-3109", "new-4211"):
        for policy in ("X0", "selected-exploration"):
            for seed in (SEEDS if policy != "X0" else (0,)): put("transfer", c, "C3", policy, seed=seed)
    assert len({a["id"] for a in arms}) == len(arms)
    return arms


def revised_plan(study, output):
    """Append unreleased controls without rewriting the preregistration used by early waves."""
    extra = []
    template = next(a for a in study['arms'] if a['stage'] == 'D3' and a['case'] == 'room-0-xyzi')
    for i in range(2):
        c = f'room-colored-{i}-xyzi'
        extra.append(dict(template, id=f'D3-{c}-baseline-C3-4725', case=c,
                          input_recipe_sha256=digest([c, study['color'], BASE])))
    for c, m in (('C13', 'H0'), ('C13', 'C4')):
        alt = next(a for a in study['arms'] if a['stage'] == 'B1' and a['case'] == c and a['policy'] == 'M0')
        extra.append(dict(alt, id=f'B1-{c}-reference-{m}-4725', policy='reference', method=m))
    revised = dict(study, version=2, parent_sha256=digest(study),
                   revision_reason='Append matching fixed-color native-room controls and H0/C4 diagnostic references; endpoints and thresholds unchanged.',
                   arms=study['arms'] + extra)
    path = output / 'experiment-manifest-v2.json'
    if not path.exists():
        dump(path, revised)
    elif read(path) != revised:
        raise ValueError('Existing v2 manifest differs')
    # E requires an independent 42D GPU pair when any new direction passed.
    # Fix the fallback priority before D2 renders, without changing any existing arm.
    gpu_fallback = [dict(a, id=a['id'].replace('-3109-', '-4211-'), case=a['case'].replace('-3109-', '-4211-'),
                         input_recipe_sha256=digest([a['case'].replace('-3109-', '-4211-'), study['color'], BASE]),
                         condition='independent 42D fallback: 3109 did not pass and 4211 passed')
                    for a in revised['arms'] if a['stage'] == 'E' and a['case'] == 'color-full-3109-xyziuv']
    rev3 = dict(revised, version=3, parent_sha256=digest(revised), arms=revised['arms'] + gpu_fallback,
                revision_reason='Add the pre-scored fallback independent 42D GPU pair for direction 4211 if 3109 is not accepted; no results or thresholds changed.')
    path = output / 'experiment-manifest-v3.json'
    if not path.exists():
        dump(path, rev3)
    elif read(path) != rev3:
        raise ValueError('Existing v3 manifest differs')
    return rev3


def frozen_manifest(output):
    for name in ('experiment-manifest-v3.json', 'experiment-manifest-v2.json', 'experiment-manifest.json'):
        path = output / name
        if path.is_file():
            return read(path)
    raise FileNotFoundError(f'No frozen E27 manifest in {output}')

def freeze(project, output):
    output.mkdir(parents=True, exist_ok=True)
    pinned = archived(26, "frozen-inputs.json")["asset_settings"]
    current = manifest(project)
    bad = [key for key, h in pinned.items() if current.get(key) != h]
    if bad:
        raise ValueError(f"Frozen E26 inputs differ: {bad}")
    known = cases()
    arms = plan()
    scenes = {"classroom": pinned["Assets/ClassroomScale/room.bytes"], "complex": pinned["Assets/Scenes/UrpNativeComplexRoom.unity"],
              "room": hashlib.sha256((project / "Assets/Scenes/UrpNativeRoom.unity").read_bytes()).hexdigest()}
    result = dict(version=1, baseline=BASE, scenes=scenes, asset_settings=pinned,
                  budget_checkpoints=[1560, 3150, 4725, 9450], model_checkpoints=[4, 8, 16],
                  fit=[0, 1, 2], held=[3, 4], train_size=[448, 256], check_size=[896, 512],
                  thresholds=dict(cm=5, intensity=.05, mired=10, uv=.05), search_seeds=SEEDS,
                  independent_directions=[3109, 4211], triggers=dict(failed_models=2, small_improvements=3,
                  relative_improvement=.0001, branch_reserve=.35, exploration_reserve=.4, h1_models=3),
                  color=dict(moderate=[[1, .65, .40], [.40, 1, .65], [.55, .40, 1]], high=[[1, .12, .10], [.10, 1, .15], [.12, .15, 1]],
                  uv_bounds=[-3, 3], steps=[.03, .06], radius=.25, max_radius=.75,
                  mired_deltas=[-80, 80], intensity_deltas=[-.35, .35]), cases=list(known.values()), arms=arms)
    for arm in arms:
        if arm["case"] in known:
            c = known[arm["case"]]
            mode = "Off" if arm["method"] == "Off" else c["target"]
            arm.update(scene_sha256=scenes[c["scene"]], initial_sha256=c["initial_sha256"],
                       target_recipe_sha256=digest([scenes[c["scene"]], c["truth"], c["mode"], mode, result["fit"], result["train_size"]]))
        else:
            arm.update(input_state="pending_exact_freeze", input_recipe_sha256=digest([arm["case"], result["color"], BASE]))
    path = output / "experiment-manifest.json"
    # Serialize tuples identically on resume.
    result = json.loads(json.dumps(result))
    if path.exists() and read(path) != result:
        raise ValueError("Manifest already frozen; use an explicitly versioned new output")
    if not path.exists():
        dump(path, result)
    return revised_plan(result, output)


def measured(output, arms):
    return {a["id"]: read(output / "arms" / a["id"] / "result.json") for a in arms
            if (output / "arms" / a["id"] / "result.json").is_file()}


def decisions(study, output):
    arms = study["arms"]
    got = measured(output, arms)
    result = {}

    def group(stage, policy=None):
        return [a for a in arms if a["stage"] == stage and (policy is None or a["policy"] == policy)]

    def all_done(items):
        return bool(items) and all(got.get(a["id"], {}).get("status") == "completed" for a in items)

    def passes(items):
        return sum(bool(got[a["id"]].get("pass")) for a in items)

    single = [a for a in group("D2") if a["case"].startswith("color-single-") and
              int(a["case"].split("-")[2]) < 6 and a["method"] == "C3"]
    pair = [a for a in group("D2") if a["case"].startswith("color-pair-")]
    color_ready = None
    color_evidence = {}
    if all_done(single + pair):
        for label, src, minimum in (("single_moderate", single, 5), ("pair", pair, 3)):
            for axis in ("xyzi", "xyziuv"):
                pool = [a for a in src if a["case"].endswith(axis)]
                color_evidence[label + "_" + axis] = dict(passed=passes(pool), executed=len(pool), minimum=minimum)
        color_ready = all(r['passed'] >= r['minimum'] for r in color_evidence.values())
    scores = {}
    hist_policies = ("hist16", "mix16", "hist32", "mix32")
    for a in group("C"):
        if a["policy"] not in hist_policies:
            continue
        path = output / "arms" / a["id"] / "fixed-proposals.csv"
        if got.get(a["id"], {}).get("status") != "completed" or not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row["evaluated"] == "1" and float(row["actual"]) > 1e-12:
                    scores[a["policy"]] = max(scores.get(a["policy"], 0), float(row["actual"]))
    primary_scores = {policy: score for policy, score in scores.items() if policy.endswith('16')}
    candidates = primary_scores or scores
    hist_chosen = max(candidates, key=candidates.get).replace('32', '16') if candidates else None
    hist_done = all_done([a for a in group("C") if a["policy"] in hist_policies])
    exploration = [a for a in group("B2") if a["case"] in ("L6-S", "CF-O", "C13", "WF35") and a["policy"] in ("X1", "X2", "X3")]
    source_baseline = {a["case"]: got[a["id"]] for a in group("B2", "X0") if a["id"] in got}
    chosen = None
    exploration_ratios = {}
    if all_done(exploration) and all(a["case"] in source_baseline for a in exploration):
        def merit(policy):
            data = [(got[a["id"]], source_baseline[a["case"]]) for a in exploration if a["policy"] == policy]
            # Select by observed full RGB objective only; GT is an offline diagnostic.
            return -statistics.median(r["loss"] / max(c["loss"], 1e-12) for r, c in data)
        exploration_ratios = {policy: -merit(policy) for policy in ("X1", "X2", "X3")}
        chosen = min(exploration_ratios, key=exploration_ratios.get)
    for a in arms:
        condition, launch, selected = a["condition"], True, None
        evidence = {}
        if condition.startswith("6D"):
            launch, evidence = color_ready, color_evidence
        elif condition.startswith("positive true"):
            launch, selected = bool(hist_chosen) if hist_done else None, hist_chosen
            evidence = dict(fixed_probes_complete=hist_done, maximum_positive_full_f_drop=scores)
        elif condition.startswith("independent 42D"):
            prior = next((got.get(b["id"]) for b in group("D2") if b["case"] == a["case"]), None)
            launch = prior["pass"] if prior and prior["status"] == "completed" else False if color_ready is False else None
            if condition.startswith("independent 42D fallback"):
                preferred = next((got.get(b["id"]) for b in group("D2") if b["case"] == "color-full-3109-xyziuv"), None)
                if color_ready is not False:
                    launch = launch and not preferred["pass"] if preferred and preferred["status"] == "completed" else None
        elif condition.startswith("matched joint"):
            prior = next((got.get(b["id"]) for b in arms if b["case"] == a["case"] and b["method"] == "C3" and
                          (b["stage"] == "D2" or b["policy"] in ("M0", "G0") or
                           (b["policy"] == "gpu-timed" and not b.get("gpu"))) and b["id"] in got), None)
            if prior and prior["status"] == "completed":
                launch = not prior["pass"] and prior["stop"] == "budget"
            else:
                launch = False if color_ready is False and a["case"].startswith("color-full-1201") else None
        elif condition.startswith("measured Gram"):
            arms_gpu = [g for g in group("E", "gpu-timed") if g["gpu"] and not g["condition"]]
            fractions = []
            if all_done(arms_gpu):
                for g in arms_gpu:
                    path = output / "arms" / g["id"] / "profiling.csv"
                    with path.open(newline="", encoding="utf-8-sig") as handle:
                        row = next(csv.DictReader(handle))
                    fractions.append(float(row["gram_gradient_s"]) / max(1e-12, float(row["opt_wall_s"])))
            median = statistics.median(fractions) if fractions else None
            launch = median > .20 if median is not None else None
            evidence = dict(median_gpu_timed_gram_fraction=median, timed_pairs=len(fractions), threshold=.20)
        if a["policy"] == "selected-exploration":
            launch, selected = True if chosen else None, chosen
            condition = "frozen B2 selection requires complete paired traces, then rank median full F ratio"
            evidence = dict(median_full_f_ratio=exploration_ratios)
        result[a["id"]] = dict(launch=launch, selected=selected, reason=condition or "mandatory", **({"evidence": evidence} if evidence else {}))
    return result

def archive_receipts(output, study, revision, ids, reason):
    """Keep superseded render evidence before an explicitly scoped, same-input replay."""
    if not revision.isalnum() or not reason.strip():
        raise ValueError('Replay requires an alphanumeric revision and a substantive reason')
    archive_root = (output / 'superseded').resolve()
    arm_root = (output / 'arms').resolve()
    archive = (archive_root / revision).resolve()
    if archive_root.parent != output or arm_root.parent != output or archive.parent != archive_root or archive.exists():
        raise ValueError('Use a new revision inside the experiment output')
    known = {a['id'] for a in study['arms']}
    if set(ids) - known:
        raise ValueError(f'Unknown replay arms: {set(ids) - known}')
    targets = []
    for ident in sorted(set(ids)):
        source, dest = (arm_root / ident).resolve(), (archive / ident).resolve()
        if source.parent != arm_root or dest.parent != archive:
            raise ValueError('Replay must stay inside the experiment output')
        if not (source / 'result.json').is_file():
            raise ValueError(f'Armed replay requires a prior complete receipt file: {ident}')
        targets.append((source, dest))
    archive.mkdir(parents=True)
    dump(archive / 'revision.json', dict(baseline=BASE, settings_unchanged=True, reason=reason, arms=sorted(set(ids))))
    for source, dest in targets:
        source.rename(dest)


def run(args):
    project, output = args.project.resolve(), args.output.resolve()
    if project == (ROOT / "Light").resolve() or not (project / "ProjectSettings/ProjectVersion.txt").is_file():
        raise ValueError("Run only in an isolated, prepared E26 project")
    study = freeze(project, output)
    if args.freeze_only:
        print(f"Frozen {len(study['arms'])} rows: {output}")
        return
    if args.replay_audits:
        # Preserve old raw evidence. Only known fixed-diagnostic arms can be superseded here.
        archive = (output / 'superseded' / args.replay_audits).resolve()
        if archive.parent != (output / 'superseded').resolve() or not args.replay_audits.isalnum() or archive.exists():
            raise ValueError('Provide a new alphanumeric audit revision name')
        archive.mkdir(parents=True)
        for a in study['arms']:
            if a['stage'] not in ('A1', 'D1') and a['policy'] != 'mixed-layout':
                continue
            original = (output / 'arms' / a['id']).resolve()
            destination = (archive / a['id']).resolve()
            if original.parent != (output / 'arms').resolve() or destination.parent != archive:
                raise ValueError('Audit replay must stay inside the output directory')
            if (original / 'result.json').is_file():
                original.rename(destination)
        dump(archive / 'revision.json', dict(baseline=BASE, settings_unchanged=True, reason='Fix per-view end offset in diagnostics and build fixed color Jacobians at zero residual; allow the synthetic bounded 17D test enough models'))
    if args.replay_receipts:
        if args.replay_audits or not args.arms or args.cases or not args.replay_reason:
            raise ValueError('Use --replay-receipts with exact --arms and --replay-reason, no other replay/filter')
        source = {a['id']: a for a in study['arms']}
        if any(a not in source or source[a]['stage'] not in args.stages for a in args.arms):
            raise ValueError('All replay arms must belong to the requested stages')
        archive_receipts(output, study, args.replay_receipts, args.arms, args.replay_reason)
    from blackbox_e25 import references
    refs = references(output, manifest(project))
    plugin = "Assets/LightingMvp"
    if args.sync:
        shutil.copytree(ROOT / "Light" / plugin, project / plugin, dirs_exist_ok=True)
    for stage in args.stages:
        while True:
            choices = decisions(study, output)
            dump(output / "gates.json", choices)
            selected = [a for a in study["arms"] if a["stage"] == stage and (not args.cases or a["case"] in args.cases)
                        and (not args.arms or a["id"] in args.arms) and choices[a["id"]]["launch"] is True and not (output / "arms" / a["id"] / "result.json").is_file()]
            selected = [dict(a, selected=choices[a["id"]].get("selected")) for a in selected]
            if not selected:
                break
            print(f"{stage}: {len(selected)} armed rows", flush=True)
            source_hashes = manifest(project)
            job = output / f"job-{stage}.json"
            dump(job, dict(cases=study["cases"], arms=selected, source_sha256=digest(source_hashes)))
            log = output / f"unity-{stage}.log"
            launch = [str(args.unity.resolve()), "-batchmode", "-force-d3d11", "-job-worker-count", "4", "-projectPath", str(project),
                      "-executeMethod", "LightingMvp.Editor.NativeRecoveryExperiment.BatchRun", "-bbCase", "e27-run",
                      "-bbOutput", str(output), "-bbE27Job", str(job), "-bbE25Inputs", str(refs),
                      "-bbE20Branch", "blackbox", "-bbE20BaseCommit", BASE, "-logFile", str(log)]
            flags = dict(creationflags=subprocess.CREATE_NO_WINDOW) if hasattr(subprocess, "CREATE_NO_WINDOW") else {}
            proc = subprocess.run(launch, **flags)
            if proc.returncode:
                raise RuntimeError(f"E27 Unity {stage} exited {proc.returncode}: {log}")
            missing = [a["id"] for a in selected if not (output / "arms" / a["id"] / "result.json").is_file()]
            if missing:
                raise RuntimeError(f"Missing arm results: {missing}")
    dump(output / "gates.json", decisions(study, output))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--unity", type=Path, default=Path(r"E:\UnityEngine\2022.3.62f3c1\Editor\Unity.exe"))
    p.add_argument("--output", type=Path, default=ROOT / "tmp/e27")
    p.add_argument("--freeze-only", action="store_true")
    p.add_argument("--sync", action="store_true")
    p.add_argument("--stages", nargs="+", default=["0", "A1", "D1", "A2", "A3", "B1", "B2", "B3", "D2", "D3", "D-budget", "E", "D-budget", "C", "transfer"])
    p.add_argument("--cases", nargs="+")
    p.add_argument("--arms", nargs="+")
    p.add_argument("--replay-audits", metavar="REVISION")
    p.add_argument("--replay-receipts", metavar="REVISION")
    p.add_argument("--replay-reason")
    run(p.parse_args())

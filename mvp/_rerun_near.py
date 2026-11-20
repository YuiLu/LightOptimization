import json
from pathlib import Path
import numpy as np
import torch
from mvp.lighting import cornell_samples
from mvp.run_experiments import gt_specs, initial_specs, optimize_run

path = Path('results/spot_diagnostics.json')
output = json.loads(path.read_text(encoding='utf-8'))
output['near_initialization'] = []
torch.set_num_threads(4)
samples = cornell_samples(20).to(torch.float32)
for seed in range(5):
    near = initial_specs('spot_position_direction', seed)
    gt = gt_specs('spot_position_direction')[0]
    rng = np.random.default_rng(seed + 100)
    near[0].position = (np.asarray(gt.position) + rng.normal(0, .08, 3)).tolist()
    direction = np.asarray(gt.direction) / np.linalg.norm(gt.direction)
    direction += rng.normal(0, .08, 3)
    near[0].direction = (direction / np.linalg.norm(direction)).tolist()
    run = optimize_run('spot_position_direction', seed, 1000, samples, near, lr_scale=.2)
    output['near_initialization'].append({k: v for k, v in run.items() if k != 'history'})
    print(f"near seed={seed}: NRMSE {run['initial_field_nrmse']:.5f} -> {run['final_field_nrmse']:.8f}; dp={run['final_position_error_m']:.6f}; angle={run['final_direction_error_degrees']:.6f}", flush=True)
path.write_text(json.dumps(output, indent=2), encoding='utf-8')


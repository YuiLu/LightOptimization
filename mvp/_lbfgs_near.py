import json
from pathlib import Path
import numpy as np
import torch
from mvp.lighting import AnalyticLightRig, LightSpec, cornell_samples, field_nrmse, weighted_loss
from mvp.run_experiments import gt_specs, initial_specs, optimized_parameters, parameter_errors

path = Path('results/spot_diagnostics.json')
output = json.loads(path.read_text(encoding='utf-8'))
output['near_adam_lbfgs'] = []
torch.set_num_threads(4)
samples = cornell_samples(20).to(torch.float32)
gt = gt_specs('spot_position_direction')
with torch.no_grad():
    target = AnalyticLightRig(gt, set())(samples)
for seed in range(5):
    initial = [LightSpec(**s) for s in output['near_initialization'][seed]['result']]
    rig = AnalyticLightRig(initial, optimized_parameters('spot_position_direction'))
    optimizer = torch.optim.LBFGS([p for p in rig.parameters() if p.requires_grad], lr=.5,
                                  max_iter=500, tolerance_grad=1e-12,
                                  tolerance_change=1e-14, line_search_fn='strong_wolfe')
    history = []
    def closure():
        optimizer.zero_grad(set_to_none=True)
        prediction = rig(samples)
        loss = weighted_loss(prediction, target, samples)
        loss.backward()
        with torch.no_grad():
            history.append(field_nrmse(prediction, target, samples).item())
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        nrmse = field_nrmse(rig(samples), target, samples).item()
    dp, di, angle = parameter_errors(rig, gt)
    run = {'test': 'spot_position_direction', 'seed': seed, 'optimizer': 'near Adam + L-BFGS',
           'evaluations': len(history), 'final_field_nrmse': nrmse,
           'final_position_error_m': dp, 'final_direction_error_degrees': angle,
           'result': [s.to_dict() for s in rig.to_specs()]}
    output['near_adam_lbfgs'].append(run)
    print(f"near Adam+LBFGS seed={seed}: NRMSE={nrmse:.9f}; dp={dp:.7f}; angle={angle:.7f}; evals={len(history)}", flush=True)
path.write_text(json.dumps(output, indent=2), encoding='utf-8')

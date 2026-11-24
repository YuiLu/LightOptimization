"""Headless scalar-loss adapter for the official ZeroGrads sampler and neural proxy.

Unity owns the target, loss, best-query state and camera budget. This process only
receives the starting vector and synchronous scalar objective responses.
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path


def load_models(path):
    spec = importlib.util.spec_from_file_location("zerograds_models", path)
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)
    return models


def run(models_path):
    import torch

    torch.set_num_threads(1)
    models = load_models(models_path)
    config = json.loads(sys.stdin.readline())
    x0 = config["initial"]
    dimensions = len(x0)
    # Work in unconstrained logit space. Decoder examples are always feasible without clipping Gaussian pairs.
    theta = torch.logit(torch.tensor(x0, dtype=torch.float32), eps=1e-6).requires_grad_()
    torch.manual_seed(config["seed"])
    phase = "proxy_probe"
    hparams = {
        "device": "cpu", "experiment_id": "unity_rgb", "ndim": dimensions,
        "sigma": .25, "inner_sigma": .1, "smooth_loss": True,
        "is_regularized": False, "regWeight": 0., "is_antithetic": True,
        "is_logspace": False, "criterion_proxy": torch.nn.MSELoss(),
        "use_posEnc": False, "num_encodings": 0, "num_layers": 3, "num_neurons": 64,
        "batchsize": 2 if dimensions <= 4 else 3,
    }

    def update(_defaults, latent, _experiment):
        return latent

    def normalize(latent, _experiment):
        return torch.sigmoid(latent) * 2 - 1

    def render(latent, _kwargs):
        values = torch.sigmoid(latent.detach()).tolist()
        print("Q", phase, *[format(v, ".9g") for v in values], flush=True)
        response = float(sys.stdin.readline())
        if not math.isfinite(response):
            raise RuntimeError("Unity returned a non-finite loss")
        # The official sampler batches tensors; a single scalar is already the image-space objective in Unity.
        return {"img": torch.tensor([[response]], dtype=torch.float32), "regTerm": None}

    hparams["criterion"] = lambda rendered_loss, _unused: rendered_loss.squeeze()
    sampler = models.create_sampler(hparams, update, render, normalize, {}, gt_img=torch.zeros(1, 1))
    proxy = models.NeuralProxy(hparams, normalize)
    optim_theta = torch.optim.Adam([theta], lr=.09)
    optim_proxy = torch.optim.Adam(proxy.parameters(), lr=.002)
    # The headless form of optimize_with_proxy's loop: sample, one parameter update, three proxy updates, query iterate.
    cost = 2 * hparams["batchsize"] + 1
    iters = config["remaining"] // cost
    for i in range(iters):
        phase = "proxy_probe"
        hparams["current_iter"] = i
        samples = sampler.sample(theta, hparams)
        grad = proxy(theta, samples["samples"], samples["f_at_samples"], hparams)
        optim_theta.zero_grad()
        theta.grad = grad
        optim_theta.step()
        for _ in range(3):
            proxy.update(optim_proxy)
        phase = "proxy_step"
        render(theta, None)
    print("D", iters, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, type=Path, help="Official ZeroGrads models.py at 7490db85e5c6524aa1a8844913449334b7781d0f")
    run(parser.parse_args().models)

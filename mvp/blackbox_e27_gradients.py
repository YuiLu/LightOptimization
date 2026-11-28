"""Independent reference for the six E27 view-gradient stress inputs.

G1 follows PCGrad's projection rule with the frozen cyclic order. G2 uses
CAGrad's constrained weight objective and E27's frozen mixture normalization.
G3 solves the TorchJD UPGrad equal-preference dual-cone projections in primal
coordinates. SciPy SLSQP supplies a separate solver from the Unity algorithms.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize


MATRICES = {
    'collinear': [[1, 0], [2, 0], [3, 0]],
    'orthogonal': np.eye(3),
    'opposite': [[1, 0], [-1, 0], [0, 0]],
    'unbalanced': [[1000, 1], [-1, .02], [3, -1]],
    'zeros': np.zeros((3, 2)),
    'rank_deficient': [[1, 2], [-2, -4], [3, 6]],
}


def reference(method, grad):
    m = len(grad)
    mean = grad.mean(0)
    norms = np.linalg.norm(grad, axis=1)
    scale = max(1, float(norms.max()))
    options = {'ftol': 1e-14, 'maxiter': 1200}
    if method == 'G0':
        return grad.sum(0), True
    if method == 'G1':
        projected = grad.copy()
        for t, row in enumerate(projected):
            for q in range(1, m):
                g = grad[(t + q) % m]
                projection = row @ g
                if projection < 0:
                    row -= g * projection / (g @ g)
        return projected.sum(0), True
    if method == 'G2':
        mean_norm = np.linalg.norm(mean)
        if mean_norm <= 1e-30:
            return np.zeros_like(mean), True
        g, avg = grad / scale, mean / scale
        coefficient = .4 * mean_norm / scale

        def objective(w):
            v = w @ g
            return v @ avg + coefficient * np.linalg.norm(v)

        res = minimize(objective, np.ones(m) / m, method='SLSQP', bounds=[(0, 1)] * m,
                       constraints=[dict(type='eq', fun=lambda w: w.sum() - 1)], options=options)
        v = res.x @ grad
        magnitude = np.linalg.norm(v)
        return (magnitude * mean + .4 * mean_norm * v) / max(1e-30, magnitude + .4 * mean_norm), bool(res.success)
    if method == 'G3':
        nonzero = norms > 1e-30
        if not np.any(nonzero):
            return np.zeros_like(mean), True
        cone = grad[nonzero] / norms[nonzero, None]
        results = [minimize(lambda x: .5 * (x - v) @ (x - v), np.zeros_like(v), jac=lambda x: x - v,
                            method='SLSQP', options=options,
                            constraints=[dict(type='ineq', fun=lambda x: cone @ x, jac=lambda x: cone)])
                   for v in grad / scale]
        return np.mean([r.x for r in results], axis=0) * scale, all(r.success for r in results)
    raise ValueError(method)


def compare(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            matrix = np.asarray(MATRICES[row['input']], dtype=float)
            expected, converged = reference(row['method'], matrix)
            observed = np.array([float(x) for x in row['direction'].split(';')])
            measured = np.linalg.norm(observed - expected)
            tolerance = max(2e-9, np.linalg.norm(expected) * 2e-5)
            yield dict(input=row['input'], method=row['method'], observed=row['direction'],
                       reference=';'.join(map(str, expected)), absolute_l2=measured, tolerance=tolerance,
                       scipy=scipy.__version__, solver_converged=int(converged), passed=int(converged and measured <= tolerance))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    rows = list(compare(args.input))
    if len(rows) != 24:
        raise SystemExit(f'Expected 6×4 synthetic rows, got {len(rows)}')
    with args.output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    passed = sum(r['passed'] for r in rows)
    print(f'Gradient reference: {passed}/{len(rows)}')
    raise SystemExit(passed != len(rows))

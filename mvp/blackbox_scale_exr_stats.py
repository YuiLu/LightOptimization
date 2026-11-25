"""Read raw EXRs and quantify spatial concentration of squared RGB error.

Uses Blender's EXR reader only; no scene or rendering is involved.
"""
from pathlib import Path
import json
import sys

import bpy
import numpy as np


def pixels(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        values = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(values)
        return values.reshape(-1, image.channels)[:, :3].astype(np.float64)
    finally:
        bpy.data.images.remove(image)


def main(args):
    if len(args) < 3:
        raise SystemExit("Usage: blender -b --python blackbox_scale_exr_stats.py -- <report.json> <native-target.exr> <capture.exr> ...")
    out, target, *paths = [Path(s).resolve() for s in args]
    ref = pixels(target)
    top = max(1, len(ref) // 100)
    energy = float(np.square(ref).sum())
    result = dict(target=target.name, targetPeak=float(ref.max()), pixels=len(ref), orderedPixels=top, samples=[])
    for p in paths:
        rgb = pixels(p)
        if rgb.shape != ref.shape:
            raise ValueError(f"Incompatible EXR dimensions: {p}")
        per_pixel = np.square(rgb - ref).sum(axis=1)
        total = float(per_pixel.sum())
        result["samples"].append(dict(file=p.name, peak=float(rgb.max()), nrmse=float(np.sqrt(total / (energy + 1e-12))),
                                      topPixelSquaredErrorShare=float(np.partition(per_pixel, -top)[-top:].sum() / total) if total else 0))
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:])

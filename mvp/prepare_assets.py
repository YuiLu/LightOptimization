"""Fetch the CC0 Mitsuba scenes used by the occlusion MVP.

Archives are data only: no downloaded script or build tool is ever executed.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "third_party"
MARKER = ".light-mvp-asset.json"
ASSETS = {
    "cornell-box": "https://benedikt-bitterli.me/resources/mitsuba/cornell-box.zip",
    "veach-ajar": "https://benedikt-bitterli.me/resources/mitsuba/veach-ajar.zip",
}


def safe_members(archive: zipfile.ZipFile):
    for info in archive.infolist():
        normalized = info.filename.replace("\\", "/")
        path = PurePosixPath(normalized)
        if (
            path.is_absolute()
            or ".." in path.parts
            or (path.parts and path.parts[0].endswith(":"))
        ):
            raise ValueError(f"Unsafe archive member: {info.filename!r}")
        if normalized and not normalized.endswith("/"):
            yield info, path.parts


def prepare_asset(name: str, url: str):
    destination = ASSET_ROOT / name
    marker = destination / MARKER
    if marker.exists():
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if metadata.get("url") == url:
            print(f"Reusing {name}: {destination}", flush=True)
            return destination
        raise RuntimeError(f"Directory belongs to a different asset: {destination}")
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty directory: {destination}")

    request = urllib.request.Request(url, headers={"User-Agent": "LightMVP-AssetFetcher/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()

    temporary = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=ASSET_ROOT))
    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            for info, parts in safe_members(archive):
                target = temporary.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as stream:
                    shutil.copyfileobj(source, stream)
        (temporary / MARKER).write_text(json.dumps({
            "name": name,
            "url": url,
            "sha256": digest,
            "license": "CC0 1.0, per https://benedikt-bitterli.me/resources/",
        }, indent=2), encoding="utf-8")
        if destination.exists():
            destination.rmdir()
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    xml_files = sorted(destination.rglob("*.xml"))
    print(f"Prepared {name}: {len(payload)} bytes; XML files:", flush=True)
    for path in xml_files:
        print(f"  {path.relative_to(destination)}", flush=True)
    return destination


def main():
    ASSET_ROOT.mkdir(exist_ok=True)
    for name, url in ASSETS.items():
        prepare_asset(name, url)


if __name__ == "__main__":
    main()

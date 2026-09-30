"""Download the Roboflow Universe datasets listed in configs/datasets.yaml.

Uses the Roboflow REST export API directly (no `roboflow` package needed).
Reads ROBOFLOW_API_KEY from the environment or from a `.env` file in the project root.

    python training/download_datasets.py                       # configs/datasets.yaml
    python training/download_datasets.py configs/court_datasets.yaml
"""
import io
import os
import sys
import zipfile
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
API = "https://api.roboflow.com"
EXPORT_FORMAT = "yolov8"  # YOLO txt labels + data.yaml, works for YOLO11/26 too


def load_api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY")
    env_file = ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("ROBOFLOW_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("ROBOFLOW_API_KEY not set. Copy .env.example to .env and add your key.")
    return key


def resolve_version(ws: str, proj: str, version, key: str) -> int:
    if version != "latest":
        return int(version)
    r = requests.get(f"{API}/{ws}/{proj}", params={"api_key": key}, timeout=60)
    r.raise_for_status()
    versions = r.json().get("versions", [])
    if not versions:
        sys.exit(f"No versions found for {ws}/{proj}")
    return max(int(v["id"].rsplit("/", 1)[-1]) for v in versions)


def download(ds: dict, key: str) -> None:
    ws, proj = ds["workspace"], ds["project"]
    version = resolve_version(ws, proj, ds.get("version", "latest"), key)
    out = RAW_DIR / ds["name"]
    marker = out / f".done_v{version}"
    if marker.exists():
        print(f"[skip] {ds['name']} v{version} already downloaded")
        return

    print(f"[export] {ws}/{proj} v{version} ({EXPORT_FORMAT})")
    r = requests.get(f"{API}/{ws}/{proj}/{version}/{EXPORT_FORMAT}", params={"api_key": key}, timeout=300)
    r.raise_for_status()
    link = r.json()["export"]["link"]

    print(f"[download] {ds['name']} ...")
    z = requests.get(link, timeout=1800)
    z.raise_for_status()
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
        zf.extractall(out)
    marker.touch()
    print(f"[ok] {ds['name']} v{version} -> {out.relative_to(ROOT)} ({len(z.content) / 1e6:.0f} MB)")


def main() -> None:
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "configs" / "datasets.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    key = load_api_key()
    for ds in cfg["datasets"]:
        download(ds, key)


if __name__ == "__main__":
    main()

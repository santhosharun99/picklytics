"""Merge downloaded Roboflow datasets into one YOLO dataset with a unified class list.

Images are symlinked (not copied) to save disk space. Label class ids are remapped
using `class_map` from configs/datasets.yaml; unmapped classes are dropped.
Images left with no boxes are kept as negatives (empty label file).

    python training/merge_datasets.py
"""
import shutil
from collections import Counter
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "merged"
SPLITS = {"train": "train", "valid": "val", "val": "val", "test": "test"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def source_names(ds_dir: Path) -> list[str]:
    data_yaml = next(ds_dir.rglob("data.yaml"))
    names = yaml.safe_load(data_yaml.read_text())["names"]
    return list(names.values()) if isinstance(names, dict) else list(names)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "datasets.yaml").read_text())
    targets = cfg["target_classes"]
    target_idx = {n: i for i, n in enumerate(targets)}

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    stats = Counter()
    sizes = Counter()
    for ds in cfg["datasets"]:
        ds_dir = RAW_DIR / ds["name"]
        if not ds_dir.exists():
            print(f"[warn] {ds['name']} not downloaded, skipping")
            continue
        names = source_names(ds_dir)
        cmap = {k.lower(): v for k, v in ds["class_map"].items()}
        remap = {i: target_idx[cmap[n.lower()]] for i, n in enumerate(names) if n.lower() in cmap}
        print(f"[{ds['name']}] classes {names} -> keep {{{', '.join(f'{names[i]}:{targets[j]}' for i, j in remap.items())}}}")

        for src_split, dst_split in SPLITS.items():
            img_dir = next((p for p in ds_dir.rglob(f"{src_split}/images") if p.is_dir()), None)
            if img_dir is None:
                continue
            lbl_dir = img_dir.parent / "labels"
            out_img = OUT_DIR / dst_split / "images"
            out_lbl = OUT_DIR / dst_split / "labels"
            out_img.mkdir(parents=True, exist_ok=True)
            out_lbl.mkdir(parents=True, exist_ok=True)

            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() not in IMG_EXT:
                    continue
                stem = f"{ds['name']}__{img.stem}"
                lines = []
                lbl = lbl_dir / f"{img.stem}.txt"
                if lbl.exists():
                    for row in lbl.read_text().splitlines():
                        parts = row.split()
                        if len(parts) == 5 and int(parts[0]) in remap and \
                                max(float(parts[3]), float(parts[4])) <= ds.get("max_box", 1.0):
                            cls = remap[int(parts[0])]
                            lines.append(" ".join([str(cls), *parts[1:]]))
                            stats[f"{dst_split}/{targets[cls]}"] += 1
                (out_img / f"{stem}{img.suffix.lower()}").symlink_to(img.resolve())
                (out_lbl / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                stats[f"{dst_split}/images"] += 1
                stats[f"{dst_split}/negatives"] += not lines
                if stats[f"{dst_split}/images"] % 200 == 1:  # sample image sizes
                    im = cv2.imread(str(img))
                    if im is not None:
                        sizes[f"{im.shape[1]}x{im.shape[0]}"] += 1

    data_yaml = {
        "path": str(OUT_DIR),
        "train": "train/images",
        "val": "val/images",
        "names": dict(enumerate(targets)),
    }
    if (OUT_DIR / "test" / "images").exists():
        data_yaml["test"] = "test/images"
    (OUT_DIR / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    print("\nMerged dataset:", OUT_DIR.relative_to(ROOT))
    for k in sorted(stats):
        print(f"  {k:20s} {stats[k]}")
    print("  sampled image sizes:", dict(sizes))


if __name__ == "__main__":
    main()

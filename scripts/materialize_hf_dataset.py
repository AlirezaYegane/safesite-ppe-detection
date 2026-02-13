from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

REPO_ID = "Voxel51/hard-hat-detection"
OUT_BASE = Path("data/raw/hard_hat_detection")
IMG_DIR = OUT_BASE / "images"
ANN_DIR = OUT_BASE / "annotations"


def save_split(ds, split_name: str) -> None:
    split_img_dir = IMG_DIR / split_name
    split_img_dir.mkdir(parents=True, exist_ok=True)
    ANN_DIR.mkdir(parents=True, exist_ok=True)

    annotations = []

    for i, ex in tqdm(enumerate(ds), total=len(ds), desc=f"Saving {split_name}"):
        img = ex.get("image")
        if img is None:
            raise KeyError("No 'image' field found in dataset examples.")

        fname = f"{split_name}_{i:06d}.png"
        img_path = split_img_dir / fname
        img.save(img_path)

        ex_copy = {k: v for k, v in ex.items() if k != "image"}
        annotations.append({"file_name": str(img_path.as_posix()), "meta": ex_copy})

    ann_path = ANN_DIR / f"{split_name}.json"
    ann_path.write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    print(f"✅ Saved {len(ds)} images to {split_img_dir}")
    print(f"✅ Wrote annotations to {ann_path}")


def main() -> None:
    print(f"Loading dataset: {REPO_ID}")
    ds = load_dataset(REPO_ID)

    if hasattr(ds, "keys"):
        for split in ds.keys():
            save_split(ds[split], split)
    else:
        save_split(ds, "train")


if __name__ == "__main__":
    main()

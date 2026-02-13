from __future__ import annotations

from pathlib import Path
import shutil

RAW_DIR = Path("data/raw/hard_hat_detection/images")
SAMPLES_DIR = Path("data/samples")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    images = (
        list(RAW_DIR.rglob("*.jpg"))
        + list(RAW_DIR.rglob("*.png"))
        + list(RAW_DIR.rglob("*.jpeg"))
    )

    print(f"[prepare_data] Scanning: {RAW_DIR.resolve()}")
    print(f"[prepare_data] Found {len(images)} images")

    copied = 0
    for img in images[:10]:
        dst = SAMPLES_DIR / img.name
        if not dst.exists():
            shutil.copy(img, dst)
        copied += 1

    print(f"[prepare_data] Copied {copied} images to: {SAMPLES_DIR.resolve()}")


if __name__ == "__main__":
    main()

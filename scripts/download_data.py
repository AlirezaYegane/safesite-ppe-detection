from __future__ import annotations

from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "Voxel51/hard-hat-detection"
OUT_DIR = Path("data/raw/hard_hat_detection")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset. This can be ~1.3GB and may take a while...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(OUT_DIR),
        local_dir_use_symlinks=False,  # important for Windows
    )

    print(f"✅ Download complete. Data is in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

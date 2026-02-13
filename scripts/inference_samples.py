from __future__ import annotations

from pathlib import Path

from safesite_api.inference import InferenceConfig, run_inference

SAMPLES_DIR = Path("data/samples")
OUT_DIR = Path("data/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    try:
        import cv2
    except Exception as e:
        raise SystemExit('OpenCV missing. Install: pip install -e ".[dev,ml]"') from e

    images = sorted(
        list(SAMPLES_DIR.glob("*.png"))
        + list(SAMPLES_DIR.glob("*.jpg"))
        + list(SAMPLES_DIR.glob("*.jpeg"))
    )
    if not images:
        raise SystemExit(f"No images found in {SAMPLES_DIR.resolve()}")

    print(f"[inference_samples] Found {len(images)} images in {SAMPLES_DIR.resolve()}")

    cfg = InferenceConfig(weights="yolov8n.pt", conf=0.25)

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[warn] Could not read image: {img_path}")
            continue

        resp = run_inference(img, cfg=cfg)

        out_json = OUT_DIR / f"{img_path.stem}.json"
        out_json.write_text(resp.model_dump_json(indent=2), encoding="utf-8")

        print(f"[ok] {img_path.name} -> {out_json.name} detections={len(resp.detections)}")

    print(f"✅ Outputs saved in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

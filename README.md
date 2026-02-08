# SafeSite – PPE Detection (WIP)

## Problem
Construction sites need fast, reliable PPE compliance checks to reduce safety incidents.

## Solution (this repo)
A production-ready ML service:
- Inference endpoint (FastAPI)
- Dockerized deployment
- CI + tests
- Later: YOLOv8/ONNX inference + metrics + demo

## Architecture (placeholder)
```
Client -> FastAPI -> Model Inference -> JSON + annotated output
```

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
uvicorn safesite_api.main:app --reload
```

## Run with Docker
```bash
docker compose up --build
```

## Roadmap
- [ ] Dataset + data card
- [ ] Baseline inference
- [ ] Model export (ONNX) + latency benchmark
- [ ] Deploy + demo video

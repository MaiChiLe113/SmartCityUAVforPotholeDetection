# Smart City UAV for Pothole Detection

COS40007 Artificial Intelligence Engineering, Group 1 (Le Mai Chi, Hoang Huu Hoan, Tran Minh Hai).

Residents pin a pothole on a map and send a photo. The server runs a YOLO pre-check. A UAV monitor
reviews the report and approves a flight plan, then a simulated UAV flies the route and an edge device
(Jetson / Raspberry Pi) confirms the damage. The dashboard shows the UAV and the result.

> **Status:** skeleton. Most functions raise `NotImplementedError("TODO(SCRUM-xx, owner)")` and most
> API endpoints answer `501 Not implemented yet: SCRUM-xx`. Search for your Jira key to find your work:
> `grep -rn "SCRUM-24" backend frontend --exclude-dir=node_modules --exclude-dir=.venv`

## Repository layout

```
backend/        Python 3.11: ML pipeline (PyTorch + Ultralytics), FastAPI server, UAV sim, edge runner
  config.py     ★ every changeable value (paths, HF dataset id, epochs, thresholds, UAV, API)
  ml/build.py   ★ build/compile functions: turn config into ready objects
  run/          ★ entry points: only call build functions (python -m run.<name>)
  ml/           data, train, evaluate, thresholds, promotion, registry, export, geo, drift
  app/          FastAPI: routers/ (endpoints), services/ (detector, flight planner, retrain trigger)
  sim/ edge/    simulated UAV + camera, on-device runner + benchmark
  tests/
frontend/       Next.js 15 (App Router, TypeScript, Tailwind, react-leaflet + OpenStreetMap)
  lib/config.ts ★ every changeable frontend value (API URL, poll intervals, map, upload limits)
  lib/api.ts    one typed function per backend endpoint
  lib/types.ts  mirrors backend/app/schemas.py
  app/          /report, /report/[id], /monitor, /monitor/reports/[id], /monitor/flight-plans/[id], /monitor/dashboard
```

**Rule:** values go in `backend/config.py` (or `frontend/lib/config.ts`), logic goes in `ml/`, `app/`,
`sim/`, `edge/`, and `run/` scripts just wire them together. Secrets and machine-specific values go in
`.env` files, never in code.

## 1. Prerequisites

- Python 3.11+, Node.js 20+, git
- NVIDIA GPU + CUDA for training (lab server); CPU/MPS works for smoke runs
- Optional: Docker (for `docker compose`)

## 2. Environment files

```bash
cp .env.example .env
cp backend/.env.example backend/.env
cp frontend/.env.local.example frontend/.env.local
```

| File | Variable | Meaning |
|---|---|---|
| `.env` | `BACKEND_PORT`, `FRONTEND_PORT`, `MLFLOW_PORT` | ports used by `docker compose` |
| `backend/.env` | `HF_TOKEN` | Hugging Face token (only for private/gated datasets) |
| | `DATABASE_URL` | default `sqlite:///./data/app.db`; a `postgresql://` URL also works |
| | `MLFLOW_TRACKING_URI` | MLflow server, default `http://localhost:5000` |
| | `DEVICE` | training device: `0`, `0,1`, `cpu`, `mps` |
| | `CORS_ORIGINS` | comma-separated origins allowed to call the API |
| | `MEDIA_DIR` | where uploaded/annotated photos are saved |
| | `SERVER_WEIGHTS` | weights for the API; empty = MLflow champion |
| `frontend/.env.local` | `NEXT_PUBLIC_API_URL` | backend URL, default `http://localhost:8000` |

## 3. Backend setup

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[ml,dev]"
```

`[ml]` pulls PyTorch, Ultralytics, Hugging Face, MLflow and Evidently (large). For API/frontend work
only, `pip install -e ".[dev]"` is enough. Run every backend command from `backend/` with the venv active.

## 4. ML pipeline (run in order)

```bash
python -m run.prepare_data                         # SCRUM-18: download from HF, dedupe, 70/15/15 split, freeze test set
python -m run.train --model yolov8n.pt --smoke     # quick check on a small subset (local GPU)
python -m run.train --model yolov8n.pt             # SCRUM-19: baseline
python -m run.compare_models                       # SCRUM-25: yolov8n / yolo11n / yolo26n, identical settings
python -m run.tune_thresholds --weights runs/.../best.pt   # SCRUM-26: copy result into config.INFER
python -m run.evaluate --weights runs/.../best.pt --split test   # SCRUM-43
python -m run.export_edge --target jetson --weights runs/.../best.pt --version 1   # SCRUM-35 (run on the Jetson)
python -m run.export_edge --target pi --weights runs/.../best.pt --version 1
python -m run.drift_report --recent data/field     # SCRUM-36
mlflow ui                                          # experiments + model registry at http://localhost:5000
```

Before setting `DATA.hf_dataset_id` in `config.py` the data step cannot run.

## 5. API server

```bash
python -m run.serve --reload
```

- Health check: http://localhost:8000/health
- Interactive docs: http://localhost:8000/docs

## 6. UAV simulation

```bash
python -m run.simulate_mission --mission-id 1
```

Needs the API running and an approved flight plan (mission) for that id.

## 7. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000. Pages: `/report` (resident), `/monitor` (report queue), `/monitor/dashboard`.

## 8. Everything with Docker

```bash
docker compose up --build
```

Starts the API (8000), the web app (3000) and MLflow (5000).

## 9. Tests and lint

```bash
cd backend && ruff check . && pytest
cd frontend && npm run lint && npm run build
```

CI (`.github/workflows/ci.yml`) runs the same on every pull request. Tests that depend on unbuilt stubs are
marked `xfail` with their Jira key; remove the marker when you implement the function.

## 10. Team ownership

| Owner | Files | Jira |
|---|---|---|
| Chi (data and software) | `ml/data.py`, `ml/registry.py`, `ml/drift.py`, `app/models.py`, `app/routers/reports.py`, `app/routers/detections.py` (flag), `app/routers/monitoring.py`, `frontend/app/report/*`, `frontend/app/monitor/page.tsx`, `monitor/reports/*`, `monitor/dashboard`, `components/{DraggablePin,PhotoUpload}.tsx`, CI/CD, Docker | SCRUM-18, 21, 23, 24, 28, 30, 36, 37, 38, 39, 40, 41, 44, 45 |
| Hai (model) | `ml/augment.py`, `ml/train.py`, `ml/evaluate.py`, `ml/thresholds.py`, `ml/promotion.py`, `ml/geo.py`, `ml/build.py` (model part), `app/services/detector.py`, `app/services/retrain_trigger.py`, `edge/runner.py`, `components/DetectionOverlay.tsx` | SCRUM-19, 25, 26, 29, 34, 43, 46 |
| Hoan (hardware and report) | `sim/`, `ml/export.py`, `edge/benchmark.py`, `app/services/flight_planner.py`, `app/routers/flight_plans.py`, `app/routers/missions.py`, `frontend/app/monitor/flight-plans/*`, `components/FlightPlanRoute.tsx`, GitHub repo rules | SCRUM-20, 22, 27, 31, 33, 35, 42 |

## 11. Workflow

- One branch per Jira issue, named after its key: `git checkout -b SCRUM-24`.
- `main` is protected: open a pull request, get one review, CI must pass.
- When you add a config value, add it to `config.py` (and to `backend/.env.example` if it comes from env).
- When you change an API schema, update `backend/app/schemas.py` and `frontend/lib/types.ts` together.

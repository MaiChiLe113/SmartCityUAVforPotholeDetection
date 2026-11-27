# Smart City UAV for Pothole Detection: Training Pipeline

COS40007 Artificial Intelligence Engineering, Group 1 (Le Mai Chi, Hoang Huu Hoan, Tran Minh Hai).

This branch holds the model side of the project: the YOLO dataset build, training, evaluation, threshold
tuning, and the automated retraining and monitoring pipeline. The API, frontend, UAV simulator and edge code
live on their own branches.

The model detects four road-damage classes: D00 longitudinal crack, D10 transverse crack, D20 alligator crack,
and D40 pothole.

## Repository layout

```
backend/
  config.py        every changeable value: paths, dataset v1 split, training, thresholds, monitoring
  ml/              logic: dataset v1 build, grouping and near-duplicate checks, training, evaluation,
                   thresholds, drift monitoring, promotion
  run/             entry points that only wire ml/ functions together (python -m run.<name>)
  tests/           unit tests (run in CI without torch or ultralytics)
docs/
  architecture.md  stack review and decision records (ADR-001 to ADR-007)
  data/            dataset v1 sources, grouping rules and split counts
  training/        training records for the prototype and YOLO26n v1
.github/workflows/
  ci.yml           ruff and pytest on every pull request
  retrain.yml      retraining and monitoring on new data, weekly, or on demand
  train.yml        Studio 7 demo pipeline (model.py)
model.py, requirements.txt   Studio 7 TensorFlow demo, kept for Group Task 2
```

## Current model

YOLO26n v1, trained from COCO weights on dataset v1 (`docs/training/yolo26n-v1.md`):

| Set | Images | mAP50 | mAP50-95 |
|---|---|---|---|
| Validation | 3,982 | 0.603 | 0.303 |
| Test (frozen) | 4,039 | 0.607 | 0.309 |

Dataset v1 is a grouped 70/15/15 split. Copies, near-duplicates and tiles of one UAV frame always stay on the
same side, and the test set is locked by sha256 (`backend/data/interim_v1/test_lock.json`).

## Setup

Python 3.11. Training needs an NVIDIA GPU.

```bash
cd backend
pip install -e ".[dev,ml,dvc]"
cp .env.example .env            # set DEVICE if needed
```

The dataset and weights are versioned with DVC, not git. Get them with `dvc pull` (see
`docs/mlops/retraining.md` for the remote setup).

## Common commands

Run from `backend/`:

```bash
python -m run.build_dataset_v1 build                     # rebuild the dataset v1 split lists
python -m run.train --model yolo26n.pt --data data/interim_v1/data.yaml --name yolo26n-v1
python -m run.tune_thresholds --weights runs/train/yolo26n-v1/weights/best.pt
python -m run.evaluate_v1 --weights runs/train/yolo26n-v1/weights/best.pt   # verifies the test lock first
ruff check . && pytest
```

Rules for this pipeline:

- Choose checkpoints and thresholds on validation only. The test set is scored once per model, after the
  test lock is verified.
- Start training from COCO-pretrained weights or from the current champion, never from prototype runs.
- Never commit datasets, weights or run output.

## Retraining and monitoring

`.github/workflows/retrain.yml` retrains the model when a new labelled batch is pushed
(`backend/data/new/batch_NNN`, versioned with DVC), every week, or on demand:

1. **check** (hosted): finds batches the champion was not trained on, writes a drift and data-quality report,
   and decides whether to retrain.
2. **retrain** on a self-hosted `gpu` runner (`dvc repro`: ingest with leakage checks, warm-start training,
   champion vs challenger on validation, test scored for the record), or a short hosted CPU demo.
3. **promote**: after a reviewer approves the `model-approval` environment, the challenger becomes the
   champion and the change arrives as a pull request.

- Runbook, setup and runner hosting on any PC: [docs/mlops/retraining.md](docs/mlops/retraining.md)
- Demo recording guide: [docs/mlops/video-recording.md](docs/mlops/video-recording.md)

## Studio 7 demo pipeline

`model.py` is the Studio 7 TensorFlow linear-regression example. `train.yml` runs it on every push and pull
request to `main` and on demand, then uploads `model_results.png` and `metrics.txt` as the `model-results`
artifact.

## Workflow

Make one branch per change, use Conventional Commit messages (`feat(scope): ...`, `fix:`, `ci:`, `docs:`),
and merge into `main` through a pull request with passing checks.

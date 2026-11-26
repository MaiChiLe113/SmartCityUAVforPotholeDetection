# YOLO26n v1 Training Record

**Status:** complete, 2026-11-23. Trained for all 100 epochs. Thresholds tuned on validation. Scored once on the frozen test set.

These are the first reportable results. They use dataset v1 (ADR-007, `docs/data/dataset-v1.md`), with a COCO-pretrained start and a test set that was frozen before training.

## Environment

| Item | Value |
|---|---|
| Machine | Windows 11 Pro, i7-12700K, 31.8 GB RAM |
| GPU | NVIDIA GeForce RTX 3060, 12 GB |
| Python env | conda `aistuff`, Python 3.11.15 |
| PyTorch | 2.14.1+cu126 |
| Ultralytics | 8.4.184 |
| Git | branch `SCRUM-38-train-pipeline`, base commit `eb22b80`, with uncommitted pipeline changes |

## Data

| Set | Images | Ground | Aerial | Negatives |
|---|---|---|---|---|
| Train | 21,052 | 15,304 | 5,748 | 0 |
| Validation | 3,982 | 3,271 | 711 | 243 |
| Test (frozen) | 4,039 | 3,270 | 769 | 247 |

Test lock sha256 `3fe092f3d4570ba176d10f829ac6a6173162b2ae91a8fd125edd1df67efa1ef7`. It was verified before and after evaluation.

**Aerial boxes by source (train / val / test):**

| Source | D00 | D10 | D20 | D40 |
|---|---|---|---|---|
| RDD China_Drone | 949 / 205 / 267 | 841 / 176 / 243 | 202 / 48 / 42 | 52 / 13 / 21 |
| RDV | 333 / 70 / 42 | – | – | 4,115 / 422 / 578 |
| UAV-PDD | 2,097 / 180 / 119 | 3,703 / 274 / 291 | 441 / 27 / 27 | 129 / 8 / 14 |

Almost all aerial potholes come from RDV close-ups. UAV-PDD supplies only 129 training potholes.

## Recipe

- **Command:** `python -m run.train --model yolo26n.pt --data data/interim_v1/data.yaml --name yolo26n-v1`
- **Schedule:** 100 epochs, patience 20, batch 32, imgsz 640, AMP, 4 workers, seed 42.
- **Optimizer:** MuSGD, lr0 0.01, lrf 0.01, momentum 0.9, weight decay 0.0005. The values are pinned per ADR-003.
- **Augmentation:** Ultralytics defaults from `TrainConfig`. Mosaic 1.0 (off for the last 10 epochs), fliplr 0.5, degrees 5, scale 0.5.
- **Resources:** about 6 GB of GPU memory, 4 h 56 min in total, and 2.4 GB minimum free RAM. Logs are in `backend/runs/logs/yolo26n-v1.*`.

## Validation curve

| Epoch | mAP50 | mAP50-95 |
|---|---|---|
| 85 | 0.603 | 0.3026 |
| 99 | 0.604 | 0.3026 |
| 100 (best) | 0.603 | 0.3028 |

Validation flattened after about epoch 80. More epochs on this data would not help.

## Thresholds (validation)

Per-class F1-maximising confidences, from `run.tune_thresholds` at conf 0.001 with `half=False`: D00 0.232, D10 0.233, D20 0.287, D40 0.267. They are written to `config.INFER.conf_thresholds`.

YOLO26 is `end2end`, so no NMS runs and IoU is not tuned.

## Test results (FP32, scored once)

| Subset | Images | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|---|
| All | 4,039 | 0.607 | 0.309 | 0.671 | 0.559 |
| Ground | 3,270 | 0.578 | 0.289 | 0.640 | 0.528 |
| Aerial | 769 | 0.551 | 0.282 | 0.638 | 0.574 |
| UAV-PDD | 124 | 0.207 | 0.062 | 0.359 | 0.241 |
| RDV | 75 | 0.589 | 0.301 | 0.703 | 0.594 |

**Per-class test AP50:** D00 0.568, D10 0.541, D20 0.663, D40 0.656. Per-class AP50-95: 0.304, 0.259, 0.354, 0.319.

**Negatives:**
- 23% of the 244 hard-negative test images get at least one detection at the tuned confidences.
- Clean aerial road has only 3 test images, too few to give a rate.

**External HighRPD drone pothole test:** D40 AP50 0.090. At the tuned confidence, recall is 0.08.

The full breakdown is in `backend/runs/reports/v1/evaluation.json`.

## Findings

1. **Ground performance is about where the prototype was, but now on an honest split.** The prototype's 0.617 validation mAP50 came from a leaky split. v1 scores 0.607 on a frozen test set.
2. **True drone imagery is the weak point:**
   - UAV-PDD mAP50 is 0.21, almost the same as the prototype's 0.29 on leaky validation.
   - UAV-PDD pothole AP is 0.001 on 14 boxes.
   - HighRPD pothole AP50 is 0.09.
   - The model learned potholes from close-ups (RDD ground, RDV), not from 30 m views.
3. **The aerial headline (0.551) hides this.** The aerial subset is mostly China_Drone (570 images) and RDV close-ups.
4. **Hard negatives:** 23% of repair-patch images get a detection. Because no negatives are in training, this is expected.

## Next steps (ranked)

1. **Collect more drone potholes at the target altitude.** UAV-PDD has 129 in train. HighRPD potholes (1,412) could be moved into training, but then HighRPD would no longer be an independent test.
2. **Test on tiles, as the edge runner sees them.** Score with 640 crops of 1920×1080 frames, because the current scores are on whole tiles.
3. **Add hard negatives to training** if the false-alarm rate matters for the review workload.
4. **Build a clean-road set** of at least 150 aerial images before claiming a false-alarm rate.

## Artefacts

- Run folder: `backend/runs/train/yolo26n-v1/`
- Reports: `backend/runs/reports/v1/thresholds.json` and `evaluation.json`
- Handoff: `backend/data/handoff/yolo26n-v1/` (README inside)

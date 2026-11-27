# YOLO26n Prototype Training Record

> **Prototype only.** These results must not appear in the Task 1 report or feed SCRUM-25, and these weights must never start a later run. The split comes from filename links and identical bytes, not the SCRUM-18 audit.

**Status:** complete, 2026-11-09. Trained for all 100 epochs.

## Purpose

Check that the training pipeline works end to end on this machine, and find out where a nano model fails on road damage. There is no mAP target, because the split is not audited and there is no baseline yet.

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

- **Source:** `TamAko783/Unified_Road_Defect_Dataset`, revision `7ba2309c243f989d72a6e9a8a9aca902871232ce`. It merges RDD-2022, UAV-PDD2023 and RoadDamageVision.
- **Classes:** D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole.
- **Why not the published split:** 1,160 of its 4,509 validation images had a related image in train, including 1,159 of 1,243 aerial images (ADR-006).

**Grouping rules.** Each group goes entirely to one side:
- An RDD China Drone photo and its RDV copies share a group, by photo number.
- RDV Roboflow exports share a group, by base name.
- UAV-PDD `lr`, `tb` and `r180` tiles share a group, by frame number.
- Byte-identical files share a group.

Validation received 15% of ground groups and 20% of aerial groups (seed 42), and keeps one version of each photo.

| Set | Images | Ground | Aerial | Boxes |
|---|---|---|---|---|
| Train | 25,261 | 18,573 | 6,688 | 63,468 |
| Validation | 3,911 | 3,275 | 636 | 9,638 |
| Left out (copies from validation groups) | 1,014 | 0 | 1,014 | 2,659 |

| Boxes | D00 | D10 | D20 | D40 |
|---|---|---|---|---|
| Train | 26,898 | 16,034 | 9,905 | 10,631 |
| Validation | 4,203 | 2,189 | 1,627 | 1,619 |
| Aerial validation only | 512 | 611 | 97 | 642 |

**Known limits:**
- Near-duplicates under unrelated names and neighbouring dashcam frames may still cross the split.
- Every image has at least one box, so false alarms on clean roads cannot be measured.
- Aerial validation has only 97 D20 boxes, and its 155 UAV-PDD images come from 46 frames.

Regenerate the lists from `backend/`:

```bash
python -m run.prototype_split
```

## Training configuration

Values come from `backend/config.py`, and `ml/build.py` passes them to Ultralytics.

| Setting | Value |
|---|---|
| Model | `yolo26n.pt` (COCO-pretrained) |
| Image size | 640 |
| Epochs / patience | up to 100 / 20 |
| Batch / workers | 32 / 4 |
| Optimizer | MuSGD, lr0 0.01, momentum 0.9 |
| Seed | 42 |
| Augmentation | hsv_v 0.4, hsv_s 0.5, fliplr 0.5, degrees 5, scale 0.5, mosaic 1.0 |

The optimizer is pinned because Ultralytics' `auto` setting ignores `lr0` and chooses by run length: AdamW for short runs and MuSGD (lr 0.01, momentum 0.9) above 10,000 iterations. Pinning it means the smoke test exercises the same optimizer as the full run. Momentum is set too, because a named optimizer otherwise uses the Ultralytics default.

## Smoke test

Command, from `backend/` using the `aistuff` interpreter:

```bash
python -m run.train --model yolo26n.pt --smoke --data data/prototype/grouped/data.yaml
```

It trains 3 epochs on 1,263 images (5% of train) and validates on all 3,911. Both runs used MuSGD with lr 0.01 and momentum 0.9, had no NaN loss, and `train_one` returned its result.

| | 8 workers (`yolo26n-smoke`) | 4 workers (`yolo26n-smoke-2`) |
|---|---|---|
| Speed after warm-up | 4.2 to 4.5 it/s | 4.2 to 4.3 it/s |
| Peak GPU memory, whole card | 7.5 GB | 7.5 GB |
| Training GPU memory, as reported | 4.94 GB | 4.94 GB |
| Lowest free RAM | 1.3 GB | 7.1 GB |

Four workers were kept, because they gave up no measurable speed and left far more RAM free. Smoke mAP (about 0.02) only shows the pipeline runs. It says nothing about model quality after 3 epochs.

At about 790 iterations and 20 seconds of validation per epoch, the estimate was about 3.5 minutes per epoch, or about 6 hours for 100 epochs. That is under the 12-hour limit for moving to the lab server.

## Full run

Command, from `backend/`:

```bash
python -m run.train --model yolo26n.pt --data data/prototype/grouped/data.yaml
```

| Item | Value |
|---|---|
| Start and end | 2026-11-09, 01:56 to 07:53 local time |
| Training time | 5.94 hours, about 3.6 minutes per epoch |
| Epochs | 100 of 100, early stopping did not trigger |
| Best epoch | 95, by mAP50-95 (the Ultralytics fitness measure in 8.4.184) |
| Peak GPU memory, whole card | 7.6 GB of 12 GB |
| Lowest free RAM | 6.3 GB |
| Interruptions | None |

| Epoch | Train box loss | Train cls loss | Val box loss | Val cls loss | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.157 | 3.562 | 2.205 | 2.745 | 0.340 | 0.280 | 0.228 | 0.090 |
| 10 | 1.977 | 2.259 | 1.920 | 1.999 | 0.565 | 0.440 | 0.466 | 0.212 |
| 25 | 1.885 | 2.001 | 1.763 | 1.664 | 0.645 | 0.534 | 0.576 | 0.283 |
| 50 | 1.809 | 1.774 | 1.714 | 1.570 | 0.679 | 0.544 | 0.603 | 0.308 |
| 75 | 1.706 | 1.565 | 1.711 | 1.535 | 0.682 | 0.557 | 0.614 | 0.316 |
| 95 (best) | see `results.csv` | | | | | | 0.616 | 0.317 |
| 100 | 1.566 | 1.152 | 1.729 | 1.525 | 0.691 | 0.563 | 0.616 | 0.317 |

The model stopped improving meaningfully around epoch 50. From there to epoch 100, mAP50-95 rose by only 0.009 while training loss kept falling and validation box loss flattened and then edged up. More epochs alone are unlikely to help.

## Results

Prototype figures only. Scored with `best.pt` on the grouped validation set, 640 px. The best epoch was picked using this same validation set, so the figures are slightly optimistic.

**By class, all validation images**

| Class | Images | Boxes | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| All | 3,911 | 9,638 | 0.686 | 0.565 | 0.617 | 0.317 |
| D00 longitudinal crack | 2,181 | 4,203 | 0.676 | 0.535 | 0.589 | 0.315 |
| D10 transverse crack | 1,359 | 2,189 | 0.624 | 0.524 | 0.551 | 0.261 |
| D20 alligator crack | 1,295 | 1,627 | 0.706 | 0.623 | 0.679 | 0.377 |
| D40 pothole | 666 | 1,619 | 0.739 | 0.578 | 0.647 | 0.317 |

**By view and source**

| Subset | Images | Boxes | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| Ground | 3,275 | 7,776 | 0.660 | 0.539 | 0.586 | 0.299 |
| Aerial | 636 | 1,862 | 0.664 | 0.559 | 0.583 | 0.290 |
| RDD (ground and China Drone) | 3,658 | 8,362 | 0.665 | 0.553 | 0.597 | 0.307 |
| RDV | 98 | 676 | 0.685 | 0.565 | 0.608 | 0.299 |
| UAV-PDD | 155 | 600 | 0.445 | 0.289 | 0.290 | 0.092 |

**AP50 by class**

| Subset | D00 | D10 | D20 | D40 |
|---|---|---|---|---|
| Ground | 0.594 | 0.550 | 0.699 | 0.502 |
| Aerial | 0.567 | 0.563 | 0.368 (97 boxes) | 0.833 |
| RDV | 0.354 | no boxes | no boxes | 0.862 |
| UAV-PDD | 0.364 | 0.354 | 0.393 (38 boxes) | 0.050 (20 boxes) |

**What this shows:**
- Ground and aerial look equal overall, but the aerial number is a mix of very different sources.
- UAV-PDD, the true drone source, is by far the weakest at mAP50 0.290. It is only 155 images from 46 frames, so the figure is rough.
- Aerial D40 looks strong (0.833), but 601 of its 642 boxes come from RDV, and UAV-PDD D40 is 0.050 on 20 boxes. It should not be read as good drone pothole detection.
- RDV's high D40 score has not been checked for leakage beyond name and byte links. Similar images under different names would inflate it.
- Ground potholes (D40, 0.502) are the weakest ground class.
- False alarms on clean roads are unmeasured, because every validation image contains damage.

## Problems and fixes

| Problem | Effect | Fix |
|---|---|---|
| Published split leaked | Validation would have overstated aerial results | Grouped re-split (ADR-006) |
| `optimizer="auto"` | Smoke and full runs would use different optimizers, and `lr0` would be ignored | Pinned MuSGD, lr0 0.01, momentum 0.9 (ADR-003) |
| 8 dataloader workers | Free RAM fell to 1.3 GB in the smoke test | 4 workers, no measurable speed loss |
| Train and val lists share one label cache file (`labels/train.cache`) | Labels are rescanned at each start, about 10 seconds | Harmless, left as is |

## Reproduce

From `backend/`, with `C:\Users\Admin\anaconda3\envs\aistuff\python.exe` as `python`:

```bash
python -m run.prototype_split
```

```bash
python -m run.train --model yolo26n.pt --data data/prototype/grouped/data.yaml
```

```bash
python -m run.prototype_report
```

The dataset must already be extracted to `backend/data/prototype/`. Training is seeded, but exact numbers can still differ between runs and machines.

**Local artifacts, not in git:**
- Run folder: `backend/runs/train/yolo26n/` (`args.yaml`, `results.csv`, plots, `weights/best.pt`, `weights/last.pt`)
- Logs: `backend/runs/logs/`, including GPU and RAM samples
- Subset metrics: `backend/runs/reports/prototype/subsets/subset_metrics.json`
- Split lists and per-image manifest: `backend/data/prototype/grouped/`
- COCO starting weights: `backend/yolo26n.pt`, plus a copy Ultralytics downloaded to `backend/weights/` for its AMP check

## Next steps

1. Inspect UAV-PDD predictions to learn why it scores so low, for example small objects at 640 px, or different labelling or imagery.
2. Check RDV validation images for near-duplicates in train.
3. Raise the aerial data shortage with the team: removing copies cuts the aerial set by about 62%, past the audit's stop-and-ask limit.
4. Finish the SCRUM-18 audit and frozen split, including clean images, then retrain from COCO weights. These prototype weights must not be reused.

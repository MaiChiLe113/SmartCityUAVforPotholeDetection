# Project Context: Smart City UAV for Pothole Detection

Last updated: 2026-11-04. Repository snapshot checked: `main` at `633350c` (2026-10-28).

This file gives a new teammate the shared context for the project. Task-specific instructions live in separate handoff files. If this file disagrees with the code, check the code and update this file.

## 1. Project

COS40007 Artificial Intelligence Engineering, Group 1, Swinburne University of Technology.

- **Goal:** detect road damage in resident photos, confirm it with a simulated UAV, and show the result on a dashboard.
- **Classes:** D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole.
- **Repository:** <https://github.com/MaiChiLe113/SmartCityUAVforPotholeDetection>
- **Team report:** "COS40007 - Team Docs" in Google Docs. Task 1 was due 25 Oct 2026.
- **Jira:** project key `SCRUM`.

| Person | Area | Main ML/data work |
|---|---|---|
| Le Mai Chi | Data and software | SCRUM-18 dataset preparation, SCRUM-21 data versioning, `ml/data.py` |
| Tran Minh Hai | Model | SCRUM-19 baseline, SCRUM-25 comparison, SCRUM-26 thresholds, SCRUM-43 evaluation |
| Hoang Huu Hoan | Hardware and report | SCRUM-22 simulation, SCRUM-35 export and benchmarks |

## 2. System flow

1. A resident submits a photo and map pin.
2. The server runs the YOLO pre-check at full precision.
3. A UAV monitor reviews the report and approves a flight plan.
4. A simulated UAV flies the route and captures frames with pose, field of view and timestamp.
5. An edge device runs the model on frames, using a quantized copy where available.
6. Detections are uploaded after landing, mapped to coordinates and merged.
7. The dashboard and 3D digital twin show the result.

Team decision so far: one set of weights serves both server and edge. Keep this unless evidence and a team decision change it.

## 3. Repository rules

- Backend: Python 3.11+, Ultralytics, FastAPI. Frontend: Next.js 15.
- Most functions are stubs that raise `NotImplementedError("TODO(SCRUM-xx, owner)")`.
- Put changeable values in `backend/config.py`, logic in `ml/`, `app/`, `sim/` and `edge/`, and only wiring in `run/`.
- Run backend commands from `backend/` with the virtual environment active.
- Use one branch per Jira key, such as `SCRUM-18`. `main` is protected and needs a pull request, one review and passing CI.
- CI installs `.[dev]` only, not `.[ml]`. `ml/build.py` imports `ml/data.py` at module level, so heavy imports in `ml/data.py` must stay inside functions.
- `backend/data/`, `backend/runs/` and `backend/weights/` are git-ignored. Never commit datasets, weights or run output.
- Before committing backend changes, run `ruff check .` and `pytest`.

Important paths from `backend/config.py`:

| Purpose | Path |
|---|---|
| Downloaded data | `backend/data/raw/` |
| Cleaned data | `backend/data/processed/` |
| Split lists | `backend/data/splits/` |
| Field set | `backend/data/field/` |
| Reports | `backend/runs/reports/` |

## 4. Current dataset

**Source:** [`TamAko783/Unified_Road_Defect_Dataset`](https://huggingface.co/datasets/TamAko783/Unified_Road_Defect_Dataset). It merges RDD-2022, UAV-PDD2023 and RoadDamageVision (RDV).

| Measure | Published value |
|---|---|
| Images | 30,186 (25,677 train, 4,509 val) |
| Boxes | 75,765 |
| D00 / D10 / D20 / D40 boxes | 31,921 / 19,350 / 11,710 / 12,784 |
| RDD / UAV-PDD / RDV images | 23,767 / 2,370 / 4,049 |
| Background images | 0 reported in `stats.json` |

Published image counts by group:

| Group | Images | View |
|---|---:|---|
| Japan | 7,900 | Ground |
| United States | 4,805 | Ground |
| India | 3,223 | Ground |
| Norway | 2,914 | Ground |
| China MotorBike | 1,934 | Ground |
| Czech Republic | 1,072 | Ground |
| China Drone | 1,919 | Aerial |
| China/Spain UAV (RDV) | 4,049 | Aerial |
| China UAV (UAV-PDD) | 2,370 | Aerial |

The published files have only train and val, and the YAML contains another person's absolute path. The project needs a new grouped 70/15/15 split.

## 5. Evidence so far

### Confirmed

- RDD2022 has 38,385 labelled images and 9,035 unlabelled test images. The unified set kept 23,767 RDD images, so 14,618 labelled images were dropped.
- At least two RDV images are rotated copies of RDD images:
  - `China_Drone_000745`
  - `China_Drone_002372`
- The dataset contains both forward-facing ground images and top-down aerial images. Comparing one view with the other will show a large difference.
- RDD D40 officially includes rutting, bumps, potholes and separation, so it is not purely a pothole class.
- Ultralytics currently lists NCNN as FP32 or FP16 only, not INT8.
- The UAV settings in `config.py` are 30 m altitude, 84° field of view and 1920 × 1080 resolution. That gives about 54 m ground width, 2.8 cm/px at native resolution and 8.4 cm/px when resized to 640 px. A 40 cm pothole is about 14 px wide natively and 5 px wide at 640 px.

### Likely, must be checked by the audit

- RDV may contain many more copies of RDD China Drone images.
- `.rf.` files may be pre-augmented Roboflow exports.
- Filenames such as `lr_00418_top_right` may be crops of a larger UAV frame.
- Published results may be inflated if copies cross the published train and val split.
- Real cracks may remain unlabelled where source classes were dropped, including oblique or block cracks.
- Class may be confounded with country or source. RDV states that D10 and D20 appear only in Chinese imagery and D40 mainly in Spanish imagery.

### RDD images dropped from the unified set

| Country/group | Labelled | Kept | Dropped |
|---|---:|---:|---:|
| Norway | 8,161 | 2,914 | 5,247 |
| India | 7,706 | 3,223 | 4,483 |
| Japan | 10,506 | 7,900 | 2,606 |
| Czech Republic | 2,829 | 1,072 | 1,757 |
| China Drone | 2,401 | 1,919 | 482 |
| China MotorBike | 1,977 | 1,934 | 43 |
| United States | 4,805 | 4,805 | 0 |

Zero dropped US images needs explaining.

## 6. Agreed data direction

- Run the data audit before freezing a split.
- Do not start the three-model comparison until the split passes the audit.
- Detection does not need a negative class. An empty label means there are no objects.
- Clean images are required in validation and test because false alarms cannot be measured without them.
- Do not add clean training images yet. First train on the cleaned dataset and inspect false alarms.
- If false alarms are too high, add targeted negatives up to about 10% of training and keep them only if they help.
- StreetSurfaceVis is not for training. It may be an extra false-alarm test later.
- Real street photos should form a field set with both clean and damaged images. Use at least 150 of each, split by street into tuning and final sets.
- Street photos do not test the UAV view. Keep aerial clean images in validation and test as well.

## 7. Known conflicts between report and code

| Topic | Current state | Direction |
|---|---|---|
| Dataset ID | `DATA.hf_dataset_id` is a TODO | Use the Hugging Face dataset above |
| Stratification | `stratify_by="view"` | Stratify by view and sub-source, and report class presence |
| Duplicate threshold | `dedupe_hash_threshold=4` | Calibrate on known RDV-RDD pairs |
| Imbalance | Config oversamples D20 and D40 if they lag | Published results show D00 and D10 lowest, so diagnose first |
| Optimizer | `optimizer="auto"` | Pin it before model comparison |
| Pi export | `int8=True` with NCNN | Unsupported by Ultralytics NCNN export; choose another path |
| Ultralytics version | `ultralytics>=8.3` | YOLO26 needs a pinned compatible version |
| Field set | Promotion requires it | Not defined or collected yet |
| Simulator frames | `sim/camera.py` suggests aerial test images near target | Use only held-out aerial images if chosen; never training images |

## 8. Model plan after data cleaning

1. Run a smoke test on the local RTX 3060.
2. Train YOLOv8n on the frozen split.
3. Review errors by view, sub-source, class and object size.
4. Run only the data experiments the baseline justifies.
5. Test ground-only, aerial-only and joint training for the aerial case.
6. Compare YOLOv8n, YOLO11n and YOLO26n with the same settings and 3 seeds each.
7. Choose server and edge thresholds on validation data, then re-check them after export.
8. Benchmark finalists on edge devices.
9. Run the chosen model once on the frozen test set.

Compute available: lab server with 2× RTX 4090, local RTX 3060, free-tier cloud notebooks. Use only COCO-pretrained weights. Do not initialize from TamAko783 models, because they were trained on images that may enter the test set.

## 9. Open questions

- Is the team willing to change the one-model design if aerial results require it?
- Will the simulated camera use real held-out aerial images?
- Are the Jetson Orin and Raspberry Pi physically available, and which Jetson model is it?
- Where and when will the field set be collected?
- Is the lab server shared, and when must the model decision be made?
- The UAV home coordinates in `config.py` are a placeholder in Ho Chi Minh City. Confirm the intended city.

## 10. Sources

- [Unified Road Defect Dataset](https://huggingface.co/datasets/TamAko783/Unified_Road_Defect_Dataset)
- [RDD2022 official repository](https://github.com/sekilab/RoadDamageDetector)
- [RDD2022 paper](https://arxiv.org/abs/2209.08538)
- [RDD class definitions](https://arxiv.org/abs/1801.09454)
- [UAV-PDD2023](https://zenodo.org/records/8429208)
- [RoadDamageVision](https://data.mendeley.com/datasets/ypm4h4z25c)
- [Ultralytics export documentation](https://docs.ultralytics.com/modes/export/)

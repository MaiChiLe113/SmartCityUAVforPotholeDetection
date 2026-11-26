# Dataset v1: Sources, Choices and Split

**Status:** built and test set frozen on 2026-11-23. Test lock sha256: `3fe092f3d4570ba176d10f829ac6a6173162b2ae91a8fd125edd1df67efa1ef7` (`backend/data/interim_v1/test_lock.json`). Decision record: ADR-007 in `docs/architecture.md`. First model trained on it: `docs/training/yolo26n-v1.md`.

Dataset v1 replaces the prototype's published split. It adds a frozen test set and clean or hard-negative road images, and it keeps every known copy, flip and neighbouring frame on one side of the split.

## Sources

| Source | Host, record and version | Verified | Licence | Taken | Left out, and why |
|---|---|---|---|---|---|
| Unified Road Defect Dataset | Hugging Face `TamAko783/Unified_Road_Defect_Dataset`, revision `7ba2309c243f989d72a6e9a8a9aca902871232ce` | Counts match `stats.json` | Mixed; see its README | RDD2022 ground images, RDD2022 China_Drone, RoadDamageVision (RDV) | Its `uav_*` files, replaced by the original UAV-PDD2023 release so all 2,440 tiles come from one source |
| RDD2022 China_Drone | Figshare article 21431547, `RDD2022_released_through_CRDDC2022.zip`, member `RDD2022/China_Drone.zip` | Zip CRC-32 on every member; 2,401 images and 2,401 XML | CC BY 4.0 | The 482 images the unified set dropped: 5 clean, 477 hard negatives (only "Repair", plus one "Block crack") | Nothing else. The other 1,919 are already in the unified set. |
| UAV-PDD2023 | Zenodo 8429208 (latest), `UAV-PDD2023.zip` | md5 `0745d017ce2dce23bd73944e9b0acec7` | CC BY 4.0 | All 2,440 tiles (2592×1944, 30 m) | Oblique crack (1,686 boxes) and Repair (282 boxes): no 4-class match |
| HighRPD | Mendeley Data `sywswj7djj` v1, `HighRPD.zip` | sha256 `7c6d44c0…7a2377b` | CC BY 4.0 | Pothole boxes only (1,412 in 997 of 11,696 tiles) as a separate external test | Line and block cracks: its "line crack" class merges longitudinal and transverse, so it can't map to D00 or D10. Not used for training. |

Cite RDD2022 (Arya et al., 2022), UAV-PDD2023 (Yan & Zhang, Data in Brief 2023), RoadDamageVision (Silva Zendron & Leithardt, Mendeley Data) and HighRPD (He et al., Data in Brief 2025).

## Class mapping

| Our class | RDD2022 / RDV | UAV-PDD2023 | HighRPD |
|---|---|---|---|
| 0 D00 longitudinal crack | D00 | Longitudinal crack | not used |
| 1 D10 transverse crack | D10 | Transverse crack | not used |
| 2 D20 alligator crack | D20 | Alligator crack | not used |
| 3 D40 pothole | D40 | Pothole | Pothole (id 2) |
| Dropped | D43, D44, D50, Repair, Block crack | Oblique crack, Repair | Line crack, Block crack |

An image is **clean** when its raw annotation has no objects at all. It is a **hard negative** when it has only dropped classes, such as a repair patch. Both kinds go only to validation and test, never to training.

## Findings that shaped the dataset

- **The original UAV-PDD2023 release ships each tile three times.** `lr_`, `tb_` and `r180_` are flipped copies, so 2,440 files hold about 813 distinct tiles from 237 frames. The unified set already had 2,370 of them, so the original release added only 70 tiles, mostly negatives. **It does not fix the aerial shortage.**
- **RDD2022's dropped China_Drone images are almost all repairs.** Only 5 of the 482 are clean.
- **dHash alone is unreliable on aerial asphalt.** 11 in 20,000 random aerial pairs fall within 4 bits, so candidates also need a pixel check (`calibration.json`).

## How groups are formed

All links feed one union-find:
1. **Name:** an RDD China_Drone photo and its RDV re-exports; RDV exports of one base name.
2. **Identical bytes** (sha256).
3. **UAV-PDD frame:** all 12 files of a frame number, across 3 flips and 4 tiles.
4. **Near-duplicates:** 64-bit dHash within 6 bits over 8 flip and rotation poses gives a candidate. A pixel check then confirms it: mean absolute grey difference at 64×64, best pose, at most 5.0.
   - At or below 2.0, the pair counts as the same photo, so validation and test score it once.
   - From 2.0 to 5.0, the pair is only kept on the same side. Visual review showed this band mixes neighbouring frames with look-alike pavement.

Result: 30,738 images in 24,608 groups. Links found: 4,937 confirmed near-duplicate pairs, 3,909 frame or same-side links, and all name and byte links.

## Split

Whole groups were assigned 70/15/15 within each source, spread over each group's rarest class, with seed 42. Groups made only of negatives went half to validation and half to test.

| Set | Images | Ground | Aerial | Negatives | D00 | D10 | D20 | D40 |
|---|---|---|---|---|---|---|---|---|
| Train | 21,052 | 15,304 | 5,748 | 0 | 22,277 | 13,467 | 8,175 | 8,857 |
| Validation | 3,982 | 3,271 | 711 | 243 | 4,111 | 2,013 | 1,639 | 1,432 |
| **Test (frozen)** | **4,039** | **3,270** | **769** | **247** | 4,056 | 2,042 | 1,630 | 1,613 |

Aerial boxes in test: D00 428, D10 534, D20 69, D40 613.

**Left out:**
- 1,625 extra copies inside validation and test groups, so each photo is scored once.
- 40 negatives that fell in training groups.

**Independent check** (a separate script that doesn't use the build code):
- No name, frame or identical-file link crosses sides.
- No pair confirmed at pixel difference 5.0 or below crosses sides.
- No unified `uav_*` copies remain.
- The lock re-verifies, and Ultralytics loads `data.yaml`.

## Known limits

- **Aerial data is still small:** about 3,000 distinct aerial units (1,919 China_Drone photos, 492 RDV bases, 237 UAV-PDD frames). Test has only 69 aerial D20 boxes.
- **Few clean aerial roads:** most test negatives are hard negatives (repairs). The clean-road false-alarm rate rests on very few images.
- **Some ground sequences may still cross:** consecutive dashcam frames just past the 5.0 limit, mainly China_MotorBike. The independent check found 121 crossing pairs at 5.0–7.0.
- **No Vietnamese roads, and no night or rain imagery.**
- **Paths are relative:** the file lists use `./` paths relative to `backend/data/interim_v1/`, so moving the whole `data/` folder keeps them valid.

## Rebuild

From `backend/`:

```bash
python -m run.build_dataset_v1 prepare
```

```bash
python -m run.build_dataset_v1 calibrate
```

```bash
python -m run.build_dataset_v1 build
```

`build` is a dry run. `freeze` writes the lock once, and any later build must reproduce the same test set or it stops with an error.

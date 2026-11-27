# Handoff: SCRUM-18 Data Audit and Cleaning

Read `docs/PROJECT_CONTEXT.md` first. Repository snapshot used for this handoff: `main` at `633350c`.

## Objective

Download the Unified Road Defect Dataset into the git-ignored data folder, audit it, clean it without changing the raw copy, and prepare a **candidate** 70/15/15 grouped split.

Do not freeze the split. Do not train a model. Stop at the human review gate in Stage 8.

## Before starting

1. `ml/data.py` is owned by Chi. Confirm this work is agreed with her before opening a pull request.
2. Create or switch to branch `SCRUM-18`. If it already exists, inspect it before making changes.
3. Set up the backend:

   ```bash
   cd backend
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[ml,dev]"
   ```

4. Check free disk space. Plan for at least 45 GB: 7.1 GB for the unified archives, about 12.4 GB for full RDD2022 archives, and room to extract both. If space is limited, defer Norway (9.9 GB), record that limit and continue.
5. Run `ruff check .` and `pytest` from `backend/` and record the baseline result.

## Hard rules

- Never commit images, labels, archives, weights or generated run output.
- Never edit, overwrite or delete files in `backend/data/raw/`.
- Never change a label silently. Log every label decision.
- Never use test images to tune a rule.
- Never call `freeze_test_set` or write the final frozen split during this task.
- Never use TamAko783 model weights to initialize training. They may be used only to flag images for human review.
- Do not add clean images to the training split. Clean images in this task are for validation and test only.
- Do not use StreetSurfaceVis in this task.
- Keep heavy imports inside functions, because CI installs only `.[dev]` and `ml/build.py` imports `ml/data.py` at module level.
- Follow the repository pattern: values in `config.py`, logic in `ml/`, wiring in `run/`.
- Keep existing `SplitManifest`, `freeze_test_set`, `load_manifest` and `write_data_yaml` behaviour compatible unless the change is explained in the pull request.

## Suggested code structure

| File | Change |
|---|---|
| `backend/config.py` | Set dataset ID and revision; add an `AuditConfig`; update stratification values |
| `backend/ml/data.py` | Implement download, safe extraction, view detection and split helpers |
| `backend/ml/audit.py` | New audit and cleaning logic |
| `backend/run/audit_data.py` | New entry point with stages such as `--stage inventory` |
| `backend/tests/test_data_audit.py` | Unit tests that run with `.[dev]` only |
| `backend/pyproject.toml` | Add only needed ML dependencies, such as `pandas` and `scikit-learn`, to `[ml]` |
| `docs/data_audit_summary.md` | Short committed summary, with no image data |

Set the dataset ID to `TamAko783/Unified_Road_Defect_Dataset`. Pin revision `7ba2309c243f989d72a6e9a8a9aca902871232ce`, seen in the dataset viewer. If that revision can't be resolved, stop and report it.

## Stage 1: Download and inventory

Download `data/train_a.tar.gz`, `data/train_b.tar.gz`, `data/val.tar.gz`, `rdd_merged.yaml`, `stats.json` and `README.md` into `backend/data/raw/unified/`.

Extract archives safely. Reject absolute paths, `..` traversal and links. Prefer Python tarfile's `filter="data"` if available.

Create `backend/runs/reports/data_audit/inventory.csv` with:

```text
image_path,label_path,source_split,prefix,sub_source,sha256,width,height,file_size,label_count
```

Reconcile against `stats.json`:

| Check | Expected |
|---|---:|
| Images | 30,186 |
| Boxes | 75,765 |
| Train / val images | 25,677 / 4,509 |
| RDD / UAV / RDV images | 23,767 / 2,370 / 4,049 |

Any mismatch must be reported. The dataset viewer estimated about 59.2k train rows, which would imply about 29.6k image-label pairs. That may only be an estimate, so count the files directly.

Inspect the actual archive names before writing parsers. Expected prefixes are `rdd_`, `uav_` and `rdv_`. Assign `view` per image:

- `aerial`: RDD China Drone, RDV and UAV-PDD
- `ground`: all other RDD images

Use `sub_source` values such as `rdd_japan`, `rdd_china_drone`, `rdv_china`, `rdv_spain` and `uav_pdd`. If RDV China and Spain can't be separated, report that and label the images `rdv_unknown`.

## Stage 2: File and label integrity

Check and log:

- corrupt or undecodable images
- images without labels, and labels without images
- malformed lines, classes outside 0 to 3, and non-numeric values
- coordinates outside 0 to 1
- zero-area or extremely small boxes
- duplicate boxes in one label file
- unusual image size or aspect ratio
- images with zero boxes

Quarantine problems through the manifest. Do not modify raw files. Record any corrected label in `label_fixes.csv` with the old value, new value and reason.

## Stage 3: Trace to original sources

Download official RDD2022 country packages into `backend/data/raw/rdd2022/`. Get links from <https://github.com/sekilab/RoadDamageDetector>. Do not hardcode unverified URLs.

Parse the Pascal VOC XML files, then match unified images to originals. Unified names are expected to include forms such as `rdd_Japan_Japan_001963`, but verify this.

Create:

| File | Contents |
|---|---|
| `rdd_label_names.csv` | Every raw label name and count, by country |
| `rdd_reconciliation.csv` | Original image, kept or dropped status, raw labels and unified labels |
| `rdd_dropped_classes.csv` | Dropped images grouped by label contents |

Expected dropped counts:

| Country/group | Labelled | Kept | Dropped |
|---|---:|---:|---:|
| Norway | 8,161 | 2,914 | 5,247 |
| India | 7,706 | 3,223 | 4,483 |
| Japan | 10,506 | 7,900 | 2,606 |
| Czech Republic | 2,829 | 1,072 | 1,757 |
| China Drone | 2,401 | 1,919 | 482 |
| China MotorBike | 1,977 | 1,934 | 43 |
| United States | 4,805 | 4,805 | 0 |

Explain the zero dropped US images. Also compare raw and unified boxes for kept images. Report how often real crack labels were removed while the image stayed in the dataset.

Check UAV-PDD against its published source if practical: card reports 2,439 images, the unified set keeps 2,370. For RDV, compare against the Mendeley version if practical.

## Stage 4: Duplicate and group discovery

Do this in order.

1. **Exact duplicates:** group by file hash and decoded pixel hash.
2. **Name rules:**
   - Parse `rdv_...China_Drone_NNNNNN_jpg.rf.<hash>` to base ID `China_Drone_NNNNNN`.
   - Link that base ID to RDD `China_Drone_NNNNNN`.
   - Parse names ending with `_top_left`, `_top_right`, `_bottom_left` or `_bottom_right` to a parent frame ID. Inspect examples first.
3. **Calibration fixtures:** `China_Drone_000745` and `China_Drone_002372` must each form one RDD-RDV group. Use them to choose thresholds. The config's pHash threshold of 4 is not trustworthy until tested on these rotated pairs.
4. **Perceptual matching:** use a centre crop to reduce black rotation corners. Test flips, 90° rotations and small rotations.
5. **Embedding candidates:** use an image model on GPU if available. Road surfaces look alike, so embeddings alone are not proof.
6. **Geometric verification:** confirm candidate pairs with ORB or SIFT and RANSAC, or an equivalent method.
7. **Sequence check:** in each sub-source, compare nearby numeric IDs and verify overlapping views. Filename distance alone is not proof.
8. **Union-find:** convert confirmed links into `group_id`.

Create `duplicate_links.csv`:

```text
image_a,image_b,method,score,inliers,decision
```

Report the threshold used, known-pair recall, a manual sample of accepted links and a manual sample of rejected near matches.

## Stage 5: Pre-augmentation and duplicate cleanup

Detect black rotation corners and report the rate by sub-source. Count how many `.rf.` variants exist per base ID.

For each duplicate group:

- Keep one canonical image for candidate split generation.
- Prefer the original unwarped RDD image when it has usable labels.
- If RDD has no target labels but an RDV copy has boxes, keep the RDV copy and log `label_conflict`.
- If several RDV variants exist, choose deterministically: least black-corner area, then filename.
- Do not warp RDV boxes back onto RDD images in this task.
- Log every removed image in `duplicate_removals.csv`.

Compare labels inside duplicate groups and report disagreements. An image with an RDV label copy can't be used as a clean negative.

## Stage 6: Label, class and scale analysis

Generate these tables:

- image and box counts by view and sub-source
- class counts by view and sub-source
- box width, height, area and aspect ratio percentiles at native size
- box size after resizing to 640 px
- share of images with no boxes, one class and multiple classes
- class presence by country and source

Flag confounds where a class is nearly exclusive to one source. Compare aerial object sizes with the simulated UAV settings in `config.py`. At 30 m, 84° field of view and 1920 × 1080 resolution, a 40 cm pothole is about 14 px wide natively and 5 px wide at 640 px.

Create a review pack in `backend/runs/reports/data_audit/review/`:

- 100 random images per sub-source with boxes drawn
- a CSV for marking missing boxes, wrong classes, loose boxes and wrong view
- examples of D40 boxes from each source, including possible rutting or bumps

Automated checks can use these images to find obvious issues, but a person must make the final review.

## Stage 7: Clean-image candidates and candidate split

### Clean-image candidates

Sort dropped RDD images with raw labels:

| Raw labels | Status |
|---|---|
| No objects | `clean_candidate` |
| Only D43 or D44 | `hard_negative_candidate` |
| Only repair labels | `hard_negative_candidate`, pending team agreement |
| D01, D11 or block cracks | `excluded` |
| Any label not reviewed yet, including D50 | `excluded` until its meaning is checked on sample images |

Print a table of raw label meanings used. D43 is white line blur and D44 is crosswalk blur in the original paper. D50 has no source definition in this handoff, so verify it before use.

Candidate targets for review:

| Split | Ground | Aerial |
|---|---:|---|
| Validation | 500 | Up to half of usable aerial candidates |
| Test | 500 | Remaining usable aerial candidates |
| Train | 0 | 0 |

Ground target by country: use a balanced sample across Japan, India, Czech Republic and Norway, subject to group rules. Exclude images with an RDV copy that has boxes.

If usable aerial candidates are below about 300, report it. Do not create box-free crops unless a human approves it.

A high-recall detector may flag candidates for review, but do not remove an image only because a detector flagged it.

### Candidate split

Build a group-aware 70/15/15 split with `DATA.seed`. Split whole groups. Stratify by view and sub-source, and keep class presence as close as practical.

Write only to `backend/data/splits_candidate/`:

```text
manifest.csv
train.txt
val.txt
test.txt
data_candidate.yaml
```

Required manifest columns:

```text
image_id,image_path,label_path,source,sub_source,view,country,parent_id,group_id,
split,is_negative,negative_type,canonical,sha256,phash,review_status,licence,notes
```

The YAML must use paths valid on this machine and class names D00, D10, D20 and D40.

Run these checks:

- No `group_id` appears in more than one split.
- No image appears in more than one split.
- No removed duplicate appears in a split.
- Train contains no added clean images.
- Every candidate negative has `review_status=pending`.
- Validation and test contain both ground and aerial images.
- Report class and sub-source distribution by split.
- Report any sub-source with fewer than 30 D40 boxes in test.

## Stage 8: Human review gate

Stop here. Produce `backend/runs/reports/data_audit/audit_report.md` and `docs/data_audit_summary.md`.

The report must include:

1. A summary: can the candidate split be approved, and what blocks it?
2. Reconciliation with published counts.
3. Duplicate results, including how many aerial images remain unique.
4. Label issues and class/source confounds.
5. Clean-image candidates by source and view.
6. Candidate split tables.
7. Differences from the Task 1 report that need correcting, including the 72/28 ground and aerial ratio.
8. Decisions needed from the team.
9. Anything skipped and why.

Humans must review the candidate validation and test negatives, the label review pack and the duplicate samples. Only after their approval should a separate task apply review results, freeze the split and update `load_manifest` outputs.

## Tests

Add tests that need no dataset and no `.[ml]` packages:

- filename parsing for RDD, RDV and UAV patterns
- label parsing and validation
- union-find grouping
- split validation that rejects group leakage
- negative category rules, including unknown labels being excluded

If an image-processing test needs Pillow or OpenCV, use `pytest.importorskip`.

Run `ruff check .` and `pytest` from `backend/` before finishing.

## Done means

- [ ] Dataset downloaded at a pinned revision into an ignored folder.
- [ ] Counts reconciled, with every mismatch explained or reported.
- [ ] RDD kept and dropped images traced to raw labels.
- [ ] Known RDV-RDD pairs grouped correctly.
- [ ] Duplicates, parents and sequences converted into groups.
- [ ] Candidate split passes automatic checks.
- [ ] Clean candidates and label review packs generated.
- [ ] Reports written and sanity-checked.
- [ ] `ruff check .` and `pytest` pass.
- [ ] No frozen split created.
- [ ] No data, weights or run output staged in git.

## Stop and ask if

- the pinned revision or expected archive is unavailable
- disk space is insufficient after deferring Norway
- official RDD files can't be downloaded
- published counts differ by more than a small, explained amount
- known RDV-RDD pairs can't be grouped reliably
- the cleaned aerial set loses more than half its published images
- a change would remove or rewrite another owner's work
- finishing would require training, freezing the split or adding clean training images

## Commit and pull request

Commit only if the person directing this task asks for it. Use messages such as `SCRUM-18: add dataset audit pipeline`. Keep commits free of generated data. Open a pull request only when asked, and include the audit summary, test results and open decisions.

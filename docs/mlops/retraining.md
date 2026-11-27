# Retraining and Monitoring Runbook

This pipeline retrains the YOLO26n pothole model when new labelled images are pushed, checks every new batch
for data problems and drift, and replaces the production model only after a person approves it.

It differs from a CSV retraining example in three ways:

- **Training needs a GPU.** A full run from COCO weights took about 5 hours on an RTX 3060, and hosted GitHub
  runners have no GPU. Retraining therefore warm-starts from the current champion (20 epochs by default) on a
  self-hosted runner that any team PC can host. A hosted CPU run exists as a demo fallback and is never
  promoted.
- **New data is images plus YOLO labels**, and it must never leak into validation or test. Every image is
  checked for exact copies and near-duplicates of validation and test images before it is used.
- **The test set is frozen.** New images go to train or validation only, the test list is never edited,
  and its sha256 lock is verified before and after every test evaluation.

## Flow

```
git push (batch .dvc file) ─┐
weekly schedule ────────────┼─> check (hosted) ──> retrain-gpu (self-hosted, gpu) ──> promote (approval) ──> PR
manual run ─────────────────┘     │                  or retrain-cpu (hosted demo)          model-approval
                                  ├─ new batches not yet trained on
                                  ├─ drift and data-quality report
                                  └─ trigger decision
```

| Job | Runner | What it does |
|---|---|---|
| `check` | `ubuntu-latest` | Pulls new batches and the drift reference, lists batches the champion was not trained on, writes the drift report, decides whether to retrain, opens an issue on a drift alert |
| `retrain-gpu` | self-hosted `gpu` | `dvc pull`, `run.check_env`, `dvc repro` (ingest, monitor, train, evaluate) |
| `retrain-cpu` | `ubuntu-latest` | Short demo: new images plus a 600-image replay sample, 3 epochs at 320 px, compared with the champion on a 400-image validation sample. Never promoted |
| `promote` | `ubuntu-latest`, environment `model-approval` | Waits for a reviewer, then makes the challenger the champion, pushes the weights to DVC and opens a pull request |

### Triggers

| Event | Monitoring | Retraining |
|---|---|---|
| Push to `main` changing `backend/data/new/*.dvc` or `backend/params.yaml` | Yes | When a batch is not in `artifacts/metadata/batches.json` |
| Schedule, Mondays 01:00 (UTC+7) | Yes | Same rule; without new data the run only monitors |
| Manual (`workflow_dispatch`) | Yes | Same rule, or always with `force`. `mode` picks `gpu` or `cpu` |

Retraining on unchanged data cannot produce a better model, so new data is the condition. The other triggers
(`drift_alert`, `enough_new_images` at 500 or more, `review_due` after 30 days) are reported in every run.

## Folder map

```
backend/
  params.yaml                    retraining knobs tracked by DVC (epochs, batch, val share, CPU demo sizes)
  dvc.yaml / dvc.lock            pipeline: ingest -> monitor -> train -> evaluate
  data/
    interim_v1/                  dataset v1 lists, frozen test lock, hash cache          (DVC)
    prototype/images, labels     unified dataset images the lists point to             (DVC)
    processed/uavpdd, ...        UAV-PDD2023 tiles and RDD China_Drone extras          (DVC)
    new/batch_NNN/               new labelled batches: images/ and labels/             (DVC, one .dvc each)
    retrain/                     v1 lists plus accepted new images (written by ingest)
  monitoring/
    reference/                   drift reference: features of 2,000 v1 training images (DVC)
    reports/                     pending.json, triggers.json, ingest_report.json, drift_report.json/.html
    logs/monitoring.log          one line per drift check
    alerts/                      one JSON per drift alert
  artifacts/
    models/champion.pt           current production weights                            (DVC)
    metrics/                     training_history, thresholds, evaluation_metrics, monitoring_metrics
    metadata/                    model_version.txt, data_version.txt, last_retrain.txt, batches.json,
                                 champion_history.json
    decision.json                champion vs challenger decision of the last run
```

## Rules the pipeline enforces

- **Leakage.** `ml/ingest.py` rejects any new image that is a byte copy or a confirmed near-duplicate (dHash
  within 6 bits, then mean pixel difference at most 5 in any flip or rotation) of a v1 validation or test
  image. On the hosted runner, where the reference images are not downloaded, a dHash match alone is
  rejected.
- **Split.** Near-duplicate new images form one group, and each group goes to train or validation (15%) by
  a stable hash, so a batch's split does not change when other batches arrive.
- **Starting weights.** `assert_allowed_init` allows only COCO candidates or `artifacts/models/champion.pt`.
- **Promotion on validation only.** The challenger must beat the champion on validation mAP50-95, must not
  lose aerial mAP50-95, and no class may lose more than 0.01 AP50-95. Test is scored afterwards for both
  models, with the lock checked before and after.
- **Human approval.** The `promote` job runs in the `model-approval` environment, so a reviewer must click
  Approve. The new champion reaches `main` as a pull request.

## Adding a batch of new data

Images and labels must be YOLO format, with classes 0 D00, 1 D10, 2 D20, 3 D40. An empty label file means no
damage.

```bash
cd backend
# Either build batches from an RDD2022 country zip ...
python -m run.prepare_batches --zip data/raw/rdd2022_czech/Czech.zip --country Czech --batches 5
# ... or copy your own images/ and labels/ into data/new/batch_004/

dvc add data/new/batch_002
dvc push data/new/batch_002.dvc
git switch -c data/batch-002
git add data/new/batch_002.dvc data/new/.gitignore
git commit -m "feat(data): add RDD2022 Czech batch 002"
git push -u origin data/batch-002
```

Open a pull request and merge it. The merge is the push to `main` that starts the workflow. Always run
`dvc push` before `git push`, or the workflow can't download the batch.

## One-time setup

### DagsHub as the DVC remote

The DagsHub repository `0ceanya/pothole-uav-data` is used only as storage, so it doesn't need to be connected
to the GitHub repository. `.dvc/config` holds the endpoint. Each person and each workflow adds the token
locally, never in git:

```bash
dvc remote modify origin --local access_key_id <DagsHub token>
dvc remote modify origin --local secret_access_key <DagsHub token>
dvc pull
```

The token is on DagsHub under the repository's Remote → Data → DVC tab (or User settings → Tokens). Without
a connected GitHub repository, DagsHub doesn't show the pipeline graph; use `dvc dag` locally instead.

### GitHub repository settings (repository admin)

| Setting | Where | Value |
|---|---|---|
| Secret `DAGSHUB_TOKEN` | Settings → Secrets and variables → Actions → Secrets | DagsHub token with write access |
| Variable `RETRAIN_MODE` | Settings → Secrets and variables → Actions → Variables | `gpu` while a GPU runner is online, otherwise `cpu` |
| Environment `model-approval` | Settings → Environments | Required reviewers: the people who may approve a new model |
| Workflow permissions | Settings → Actions → General | "Read and write permissions" and "Allow GitHub Actions to create and approve pull requests" |
| Fork pull requests | Settings → Actions → General | "Require approval for all external contributors" |

Pull requests opened by the workflow do not start other workflows (a GitHub rule for `GITHUB_TOKEN`). Close
and reopen the PR, or push an empty commit, to run `ci` on it.

## Hosting the GPU runner on any PC

The workflow has no machine-specific paths. Each host describes itself in the runner's `.env` file, so the
runner can move between team members.

**Requirements:** NVIDIA GPU with 8 GB or more (about 6 GB used at batch 32; lower `batch` in `params.yaml`
for smaller cards), a recent NVIDIA driver, Windows 10/11 or Linux, Git (Git for Windows provides `bash`),
Miniconda or Anaconda, and about 25 GB of free disk.

1. **Create the environment.**
   ```bash
   git clone https://github.com/MaiChiLe113/SmartCityUAVforPotholeDetection.git
   cd SmartCityUAVforPotholeDetection
   conda env create -f backend/environment-retrain.yml
   conda activate pothole-retrain
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
2. **Pull the data once into a shared cache.** The runner's own checkout reuses this cache, so later jobs link
   files instead of downloading 9 GB again.
   ```bash
   dvc cache dir --global D:/dvc-cache          # any folder with ~10 GB free; ~/dvc-cache on Linux
   dvc remote modify origin --local access_key_id <DagsHub token>
   dvc remote modify origin --local secret_access_key <DagsHub token>
   dvc pull
   ```
3. **Check the machine.** It must print `READY`.
   ```bash
   cd backend && python -m run.check_env
   ```
4. **Register the runner.** A repository admin opens Settings → Actions → Runners → New self-hosted runner,
   picks the host's OS and shares the one-time token. On the host, follow the download commands shown on
   that page, then run the commands below. Put the runner folder on the **same drive as the DVC cache**
   (for example `D:ctions-runner` with `D:/dvc-cache`). Each job then links the 9 GB of data instead of
   copying it.
   ```bash
   ./config.cmd --url https://github.com/MaiChiLe113/SmartCityUAVforPotholeDetection --token <TOKEN> --labels gpu --name <your-pc> --unattended
   # Linux: ./config.sh with the same arguments
   ```
5. **Describe the host.** Create `.env` in the runner folder (next to `run.cmd`):
   ```
   RETRAIN_PY=C:\Users\<you>\miniconda3\envs\pothole-retrain\python.exe
   DVC_CACHE_DIR=D:/dvc-cache
   ```
   On Linux, `RETRAIN_PY=/home/<you>/miniconda3/envs/pothole-retrain/bin/python`.
6. **Start it.** On Linux, run `./run.sh`. On Windows, the workflow's steps use `bash`, and the `bash` on PATH is
   often WSL's (`C:\Windows\system32\bash.exe`), which would run the job inside Linux. Create
   `start-runner.cmd` in the runner folder so Git Bash comes first, and always start the runner with it:
   ```bat
   @echo off
   set "PATH=C:\Program Files\Git\bin;%PATH%"
   call "%~dp0run.cmd"
   ```
   It shows **Idle** under Settings → Actions → Runners. Set `RETRAIN_MODE=gpu`.
7. **Hand over.** Stop the runner and remove it with `./config.cmd remove --token <removal token>` (the admin
   gets the token from the runner's page). The next host repeats steps 1–6.

Security, because the repository is public:

- The GPU job never runs on `pull_request` events.
- Outside contributors need approval before their workflows run.
- Run the runner under a normal user account.
- Stop it when it isn't needed.

## Running the pipeline locally

```bash
cd backend
python -m run.ingest_batches --check-only   # what is new
dvc repro                                   # ingest, monitor, train (GPU), evaluate
dvc metrics show                            # drift and evaluation metrics
dvc dag                                     # pipeline graph
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `retrain-gpu` stays "Queued" | No runner with the `gpu` label is online. Start one or set `RETRAIN_MODE=cpu` |
| `dvc pull` fails with 401 or 403 | Missing or expired `DAGSHUB_TOKEN` secret, or local token not set |
| `dvc pull` reports missing files | Someone ran `git push` before `dvc push`. Push the data from their machine |
| DagsHub push fails on size | The free storage quota is used up. Check the quota on DagsHub |
| `check_env` prints `NOT READY` | Read the FAIL lines: wrong versions, no CUDA, missing data, or the test lock changed |
| `frozen test lock` fails | Test images or labels changed. Restore them with `dvc checkout data/prototype/images data/prototype/labels` |
| `promote` didn't run | The challenger did not pass the validation checks; see `decision.json` in the run's artifacts |
| The promotion PR has no checks | PRs opened with `GITHUB_TOKEN` don't start workflows; close and reopen it |

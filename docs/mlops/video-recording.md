# Video Recording Guide: Retraining and Monitoring Demo

This guide is for the teammate who records the demo, starting from a fresh machine. It covers what to install,
which accounts you need, what to prepare the day before, and a scene-by-scene script with exact commands.
Target length: 8–10 minutes.

How the demo works in one paragraph: new labelled road images arrive as a **batch** (a folder of images and
YOLO labels). The batch is stored on DagsHub with DVC, and only a small pointer file (`batch_00N.dvc`) goes into
git. Merging that pointer file into `main` starts the **Retrain on New Data** workflow. It checks the data
for problems and drift, retrains the model on a GPU PC, compares it with the current model, and waits for a
person to approve before the new model replaces the old one.

---

## 1. Accounts and access (ask for these first)

| What | Who gives it | Why |
|---|---|---|
| GitHub collaborator with **Write** access to `MaiChiLe113/SmartCityUAVforPotholeDetection` | Chi (repo owner) | Push branches, open and merge PRs, run workflows |
| Listed as a **required reviewer** of the `model-approval` environment | Chi (Settings → Environments → `model-approval`) | Click **Approve** on screen |
| A **DagsHub token** with write access to `0ceanya/pothole-uav-data` | Hai (repo `0ceanya/pothole-uav-data` → Settings → Collaborators, then your own token from DagsHub → User settings → Tokens) | `dvc pull` and `dvc push` the demo batches |
| Which PC hosts the **GPU runner** on the day | Team decision | GPU retraining runs there. If it's not your PC, the host just keeps `run.cmd` open |

Never paste the DagsHub token into a file in the repo, a commit, a chat or the video. It only goes into the
`dvc remote modify --local` commands below, which write the git-ignored `.dvc/config.local`.

## 2. Install on the recording machine

No GPU is needed on the recording machine unless it also hosts the runner (section 4).

1. **Git**: <https://git-scm.com/downloads> (on Windows, Git for Windows also gives you Git Bash).
2. **Python 3.11**: Miniconda from <https://docs.conda.io/en/latest/miniconda.html>.
3. **Clone the repository and install DVC:**
   ```bash
   git clone https://github.com/MaiChiLe113/SmartCityUAVforPotholeDetection.git
   cd SmartCityUAVforPotholeDetection
   conda create -n pothole-demo python=3.11 -y
   conda activate pothole-demo
   pip install -e "backend[dev,dvc]"
   dvc version            # should print DVC 3.x
   ```
4. **Connect DVC to DagsHub** (repository root; token from section 1):
   ```bash
   dvc remote modify origin --local access_key_id <your DagsHub token>
   dvc remote modify origin --local secret_access_key <your DagsHub token>
   dvc pull backend/monitoring/reference.dvc     # small test download (~150 KB); must succeed
   ```
5. **Optional, for the DVC scene:** see the pipeline graph without downloading the 9 GB dataset:
   ```bash
   cd backend && dvc dag && cd ..
   ```

## 3. Get the demo batches

The batches are already prepared and uploaded to DagsHub. Their pointer files are on the branch `demo/batches`,
not on `main`, because putting a pointer on `main` is what starts retraining.

| Batch | Images | Use |
|---|---|---|
| `batch_001` | 352 | Pre-recorded full GPU run (day before) |
| `batch_002` | 352 | Live data push in the video |
| `batch_003` – `batch_005` | 351 each | Rehearsals and re-takes |

**What the batches contain:** RDD2022 Czech road photos that the training dataset left out. They are labelled
**clean roads, with no damage** (each label file is empty). The original dataset kept only the Czech photos
that show damage. Clean photos teach the model where there is nothing to detect, which is the planned next
step for cutting false alarms (at most about 10% of training; all five batches together are about 8%).

**What to expect in the video:**

- **Drift alert:** yes. The images are 600×600 photos from one country, so brightness, contrast, blur,
  size and view all differ from the mixed training set. The workflow opens a "Drift alert" issue. This is the
  monitoring working, not an error.
- **Ingest:** about 85% of each batch goes to training and 15% to validation. None overlap the frozen test set.
- **Decision:** clean images mostly reduce false alarms. Overall mAP may barely change, so the result can be
  either "promote" or "keep the champion". Both are correct outcomes to show.

Fetch one batch (replace `002` as needed), from the repository root:

```bash
git fetch origin demo/batches
git checkout origin/demo/batches -- backend/data/new/batch_002.dvc
dvc pull backend/data/new/batch_002.dvc        # downloads the images and labels into backend/data/new/batch_002/
git restore --staged backend/data/new/batch_002.dvc
```

The last line unstages the pointer file, so you can add it yourself on camera (scene 4).

**Each batch starts retraining once.** After a batch's new model is promoted and that PR is merged, the batch
is listed in `backend/artifacts/metadata/batches.json` and won't count as new again. If a run is
**not** promoted, its batch stays "new" and is included in the next run. Use a fresh batch for each re-take.

## 4. The GPU runner

GPU retraining runs on whichever PC is registered as the self-hosted runner with the label `gpu`. Setting up a
new host takes about an hour, mostly the 9 GB data download. Follow **"Hosting the GPU runner on any PC"** in
[retraining.md](retraining.md):

1. Create the `pothole-retrain` conda environment.
2. `dvc pull` the data.
3. Run `python -m run.check_env` until it prints `READY`.
4. Register the runner with a token from Chi, labelled `gpu`.
5. Create the runner `.env` file with `RETRAIN_PY` and `DVC_CACHE_DIR`.
6. Start it with `run.cmd`.

On the day:

- The host PC is on, `run.cmd` is running, and Settings → Actions → Runners shows the runner **Idle**.
- Chi sets the repository variable `RETRAIN_MODE` to `gpu` (Settings → Secrets and variables → Actions →
  Variables). Set it back to `cpu` afterwards.
- If no GPU PC is available, keep `RETRAIN_MODE=cpu`. Every scene still works, but retraining is the short CPU
  demo and nothing is promoted. Say so in the video.

## 5. The day before: pre-recorded runs

Training takes about an hour, so record its result in advance and show it in the video.

1. **Full GPU run on `batch_001`.** Fetch `batch_001` as in section 3, then commit and merge it exactly as in
   scene 4 (branch `data/batch-001`). Wait for the run: about 10 minutes of data pulls and checks, about
   60 minutes of training, about 10 minutes of evaluation.
   - If the decision is **promote**, the `promote` job waits. Approve it (scene 7) and merge the PR it opens.
     Screenshot or record the approval if you like; you can approve live with `batch_002` too.
   - If the decision is **keep the champion**, that is a valid result to show. Explain why from the FAIL
     lines.
2. **A scheduled run.** On a branch, change the cron line in `.github/workflows/retrain.yml` to
   `- cron: "*/15 * * * *"`, commit (`ci: run the retraining schedule every 15 minutes for the demo`), and
   merge. After one run with event `schedule` appears, revert it (`ci: restore the weekly retraining schedule`)
   and merge again. GitHub may delay scheduled runs by several minutes.
3. **A manual CPU run** as a backup: Actions → Retrain on New Data → Run workflow → mode `cpu`, force ✓.
   Takes about 20–30 minutes.

## 6. Recording setup

- Browser zoom 125%, notifications off, signed in to GitHub and DagsHub.
- Tabs: the repository, Actions, Settings → Actions → Runners (if you have access, or ask the host to share
  their screen), the DagsHub repository page.
- Terminal with a large font, in the repository root, with `conda activate pothole-demo` done.
- `batch_002` already pulled (section 3) and **not** committed.

## 7. Scenes

### Scene 1 — Overview (0:40)
**Show:** the repository on GitHub (`main`), then the flow diagram at the top of `docs/mlops/retraining.md`.
**Say:** the model is YOLO26n, trained on about 29,000 road images. Retraining it needs a GPU, new data is
images with labels, and the test set is frozen. That's why the pipeline has a data check, a GPU job, and a
human approval step.

### Scene 2 — Commit history (0:20)
**Show:** `git log --oneline --graph -25`
**Say:** every change is a Conventional Commit (`feat(data)`, `feat(monitor)`, `ci:`, `docs:`), merged
through pull requests.

### Scene 3 — DVC and remote storage (1:15)
**Show:**
```bash
cat .dvc/config                  # the DagsHub endpoint; the token is in config.local, not in git
dvc remote list
cat backend/data/prototype/images.dvc   # a pointer: hash, 30,186 files, 7.4 GB
cd backend && dvc dag && cd ..
```
Then the DagsHub repository page (see section 9 for making the files visible there).
**Say:** what's versioned: the dataset images and labels, split lists and frozen test lock, drift reference,
champion weights, and each new batch. Where it's stored (DagsHub), and the four pipeline stages: ingest,
monitor, train, evaluate.

### Scene 4 — A data push starts retraining (1:30)
**Show:**
```bash
cd backend
dvc add data/new/batch_002                 # pointer file already matches; shows DVC tracking the folder
dvc push data/new/batch_002.dvc            # "Everything is up to date" is fine: it's already on DagsHub
git switch -c data/batch-002
git add data/new/batch_002.dvc data/new/.gitignore
git commit -m "feat(data): add RDD2022 Czech batch 002"
git push -u origin data/batch-002
cd ..
```
On GitHub: open the pull request from the banner, merge it, then open **Actions**. A new **Retrain on New
Data** run appears with event `push`.
**Say:** new labelled images are versioned with DVC, and merging the pointer file is what starts the workflow.

### Scene 5 — Monitoring (1:00)
**Show:** the `Check data, drift and triggers` job, opening the steps "Find batches the champion was not
trained on", "Drift and data-quality report" and "Evaluate retraining triggers". Then the run page's
**Summary** (drift table). Download the `monitoring-<id>` artifact and open
`monitoring/reports/drift_report.html`. If the run flagged drift, open the issue it created (Issues tab).
**Say:** every batch is checked for label errors, copies or near-copies of validation and test images, and
drift in brightness, contrast, blur, image size, view and class mix. This batch raises a drift alert because
the Czech photos look different from the training mix (different camera, size and lighting), which is exactly
what the monitor is for.

### Scene 6 — GPU retraining (1:15)
**Show:** the runners page with the runner **Active** (or the host's `run.cmd` window showing
"Running job: Retrain on the GPU runner"). If you're on the host, `nvidia-smi`. In the job log: "Check this
machine" ending with `READY`, then "Run the pipeline" starting `dvc repro`. Then switch to the finished
`batch_001` run from the day before: its Summary shows champion vs challenger on validation and test, and the
decision with PASS/FAIL lines.
**Say:** training starts from the current champion. The frozen test set is verified before and after it is
scored, and the promote decision uses validation only.

### Scene 7 — Human approval (0:45)
**Show:** on a run waiting at `Promote the new model`, click **Review deployments** → tick `model-approval`
→ **Approve and deploy**. Then the job log, and the new pull request `chore(retrain): promote model v2` with
`champion.pt.dvc` and the metadata files.
**Say:** a new model only replaces the champion after a person approves it, and the change is a reviewable
pull request. (If the result was "keep the champion", show `decision.json` instead and say the old model
stays.)

### Scene 8 — Scheduled run (0:30)
**Show:** the `schedule:` block in `.github/workflows/retrain.yml`, then the run with event `schedule` in the
Actions list (from section 5).
**Say:** every week the workflow monitors the data and retrains only when there's data the model hasn't seen.

### Scene 9 — Manual run and CPU fallback (0:45)
**Show:** Actions → Retrain on New Data → **Run workflow** → branch `main`, mode `cpu`, force ✓ → Run. Open
the `Retrain demo on a hosted CPU runner` job, or the backup run from section 5 if this one is still going.
Its summary ends with "hosted CPU demo models are never promoted".
**Say:** anyone with access can start a run by hand, and it still works when no GPU PC is online.

### Scene 10 — Artefacts (0:30)
**Show:** on the `batch_001` run page, download `retrain-<id>` and open `artifacts/metrics/evaluation_metrics.json`,
`artifacts/metrics/training_history.json`, a plot from `runs/train/retrain/`, and
`artifacts/metadata/model_version.txt` and `last_retrain.txt`.
**Say:** every run keeps its metrics, plots and version information together.

### Scene 11 — Moving the GPU job to another PC (0:30)
**Show:** the "Hosting the GPU runner on any PC" section of `docs/mlops/retraining.md` and
`backend/environment-retrain.yml`.
**Say:** any team PC with an NVIDIA GPU can take over: create the environment, pull the data, check it, and
register the runner.

## 8. After recording

- Merge or close any open `chore(retrain): promote …` pull request (merge only if it was approved on purpose).
- Make sure the temporary 15-minute schedule is reverted.
- Ask Chi to set `RETRAIN_MODE` back to `cpu`, and the host to stop `run.cmd`.
- Delete your local `data/batch-00N` branches if you like; they're merged.

## 9. Making the data visible on the DagsHub website

DagsHub's file browser shows DVC data by name only when the same git history is pushed to the DagsHub
repository. Without it the storage holds the files (by hash), but the page looks empty. Hai can mirror `main`
once (this asks for the DagsHub username and token in the terminal):

```bash
git remote add dagshub https://dagshub.com/0ceanya/pothole-uav-data.git
git push dagshub main
```

After that, DagsHub shows `backend/data/prototype/images` and the other tracked folders with their files. Re-run
`git push dagshub main` before recording if `main` has changed.

## 10. Troubleshooting

| Problem | What to do |
|---|---|
| `dvc pull` asks for credentials or returns 401/403 | Redo section 2 step 4 with a valid token, from the repository root |
| No run appears after merging the batch PR | The PR must change `backend/data/new/*.dvc`; check that the file was committed |
| `Retrain on the GPU runner` stays "Queued" | The runner is offline. Start `run.cmd` on the host, or set `RETRAIN_MODE=cpu` and run again |
| `Check data` says 0 new images | That batch was already trained on. Use the next batch |
| `promote` didn't run | The challenger didn't pass the validation checks. Show `decision.json` (scene 7 alternative) |
| No "Review deployments" button | You're not a required reviewer of `model-approval`. Ask Chi |
| The promotion PR shows no checks | PRs opened by the workflow don't start other workflows. Close and reopen the PR |

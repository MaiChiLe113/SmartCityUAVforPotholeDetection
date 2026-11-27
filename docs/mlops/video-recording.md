# Video Recording Guide: Retraining and Monitoring Demo

This guide is for the teammate who records the demo, starting from a fresh machine. It covers what to install,
which accounts you need, what to prepare the day before, and a scene-by-scene script with exact commands.
Target length: 8–10 minutes.

How the demo works in one paragraph: new labelled road images arrive as a **batch** (a folder of images and
YOLO labels). The batch is stored on DagsHub with DVC, and only a small pointer file (`batch_00N.dvc`) goes into
git. Merging that pointer file into `main` starts the **Retrain on New Data** workflow. It checks the data
for problems and drift, retrains the model on Hai's GPU PC (always on, see section 4), and compares the new
model with the current one. A new model would only replace the current one after a person approves it.
The assignment doesn't require a promotion, and the usual result is "keep the current model".

---

## 1. Accounts and access (ask for these first)

| What | Who gives it | Why |
|---|---|---|
| GitHub collaborator with **Write** access to `MaiChiLe113/SmartCityUAVforPotholeDetection` | Chi (repo owner) | Push branches, open and merge PRs, run workflows |
| Optional: listed as a **required reviewer** of `model-approval` | Chi (Settings → Environments → `model-approval`) | Only needed if a run ever reaches the approval step |
| A **DagsHub token** with write access to `0ceanya/pothole-uav-data` | Hai (repo `0ceanya/pothole-uav-data` → Settings → Collaborators, then your own token from DagsHub → User settings → Tokens) | `dvc pull` and `dvc push` the demo batches |
| Confirmation that **Hai's PC** is on, with the runner **Idle** | Hai | GPU retraining runs there, not on your machine |

Never paste the DagsHub token into a file in the repo, a commit, a chat or the video. It only goes into the
`dvc remote modify --local` commands below, which write the git-ignored `.dvc/config.local`.

## 2. Install on the recording machine

No GPU is needed on the recording machine: the GPU job runs on Hai's PC.

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

**Batches stay "new" until a model trained on them is promoted.** A batch only leaves the pending list when a
promotion PR lists it in `backend/artifacts/metadata/batches.json`. Promotions are rare (see scene 7), so
once `batch_001` is on `main`, every later push, manual run and weekly run retrains with it, plus any newer
batches. That's expected. For the live scene, merge a batch that isn't on `main` yet, so the run summary shows
it as new.

## 4. The GPU runner (Hai's PC, always on)

You don't set up a runner. GPU retraining runs on Hai's PC, registered as `hai-rtx3060` with the label `gpu`
and kept on permanently ("Current setup" in [retraining.md](retraining.md)). The repository variable
`RETRAIN_MODE` stays at `gpu`.

Before recording, check with Hai:

- The PC is on and logged in (a locked screen is fine) and the runner window is open.
- Settings → Actions → Runners shows `hai-rtx3060` as **Idle**. Only admins can open that page; otherwise ask
  Hai or Chi.
- Optional quick test, about 20 minutes: Actions → Retrain on New Data → **Run workflow** → mode `gpu`,
  **smoke** ✓. It runs the whole GPU pipeline with 1 epoch and ends green.

If Hai's PC can't be on, ask Chi to set `RETRAIN_MODE=cpu`. Every scene still works with the short hosted CPU
demo instead of the GPU job; say so in the video.

Times on Hai's PC: a full retrain (20 epochs) takes about **90 minutes** (about 3 minutes of data pull, 72 of
training, 11 of evaluation). The smoke test takes about 20 minutes.

## 5. The day before: pre-recorded runs

A full retrain takes about 90 minutes, so record its result in advance and show it in the video.

1. **Full GPU run on `batch_001`.** Fetch `batch_001` as in section 3, then commit and merge it exactly as in
   scene 4 (branch `data/batch-001`). The run takes about 90 minutes in total. The check job runs on GitHub's
   servers, then the GPU job runs on Hai's PC.
   - The expected decision is **keep the champion**: the challenger's FAIL lines show it scored below the
     current model on validation. That's the gate working, and it's what scene 7 shows.
   - If the decision is **promote**, the `promote` job waits for a reviewer. Approving is optional; you can
     also leave it, and it times out after 30 days.
2. **A scheduled run.** On a branch, change the cron line in `.github/workflows/retrain.yml` to
   `- cron: "*/15 * * * *"`, commit (`ci: run the retraining schedule every 15 minutes for the demo`), and
   merge. After one run with event `schedule` appears, revert it (`ci: restore the weekly retraining schedule`)
   and merge again. GitHub may delay scheduled runs by several minutes.
3. **A manual CPU run** as a backup: Actions → Retrain on New Data → Run workflow → mode `cpu`, force ✓.
   Takes about 20–30 minutes.

## 6. Recording setup

- Browser zoom 125%, notifications off, signed in to GitHub and DagsHub.
- Tabs: the repository, Actions, the finished `batch_001` run, the DagsHub repository page, and Settings →
  Actions → Runners if you're an admin (otherwise use Hai's screenshot of the runner page).
- Terminal with a large font, in the repository root, with `conda activate pothole-demo` done.
- `batch_002` already pulled (section 3) and **not** committed.

## 7. Scenes

### Scene 1 — Overview (0:40)
**Show:** the repository on GitHub (`main`), then the flow diagram at the top of `docs/mlops/retraining.md`.
**Say:** the model is YOLO26n, trained on about 29,000 road images. Retraining it needs a GPU, new data is
images with labels, and the test set is frozen. That's why the pipeline has a data check, a GPU job on a
team PC, and a gate before any new model replaces the current one.

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
**Show:** the live run's `Retrain on the GPU runner` job. Its log header names the runner, `hai-rtx3060`. Show
"Check this machine" ending with `READY`, then "Run the pipeline" starting `dvc repro`. If Hai shares a
screenshot of the runner window ("Running job: Retrain on the GPU runner") or `nvidia-smi`, show it here. Then switch to the finished
`batch_001` run from the day before: its Summary shows champion vs challenger on validation and test, and the
decision with PASS/FAIL lines.
**Say:** training starts from the current champion. The frozen test set is verified before and after it is
scored, and the promote decision uses validation only.

### Scene 7 — The promotion gate (0:45)
**Show:** in the `batch_001` run, the Summary's decision ("keep the champion") and its PASS/FAIL lines. Then
download `retrain-<id>` and open `artifacts/decision.json`. Point out that `Promote the new model (needs
approval)` was skipped. Briefly show the `promote` job in `.github/workflows/retrain.yml`: `environment:
model-approval`, then `run.promote` and `gh pr create`.
**Say:** the new model scored below the current one on validation, so it was not promoted and the current
model stays in use. If a retrain ever passes every check, the run stops and waits for a person to approve,
and the new model arrives as a pull request. (If your run did reach the approval step, you can show
**Review deployments** → **Approve and deploy** instead.)

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

- Close any `chore(retrain): promote …` pull request unless the team means to switch models.
- Make sure the temporary 15-minute schedule is reverted. With `batch_001` on `main`, every scheduled run
  retrains (about 90 minutes on Hai's PC), so a 15-minute schedule would keep the PC busy non-stop.
- Leave `RETRAIN_MODE=gpu` and Hai's runner as they are: the PC stays on as the team's runner.
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
| `Retrain on the GPU runner` stays "Queued" | Hai's PC or runner is off. Ask Hai to start `C:\actions-runner\start-runner.cmd`, or ask Chi to set `RETRAIN_MODE=cpu` and run again |
| `Check data` says 0 new images | That batch was promoted already. Use the next batch |
| `promote` didn't run | Expected: the challenger didn't pass the validation checks. Show `decision.json` (scene 7) |
| No "Review deployments" button | Only appears if a run passes every check, and you must be a required reviewer of `model-approval` |
| The promotion PR shows no checks | PRs opened by the workflow don't start other workflows. Close and reopen the PR |

# Video Recording Guide: Retraining and Monitoring Demo

Target length 8–10 minutes. Each scene names the part of the setup it demonstrates. The commands assume a
terminal in the repository root with the `pothole-retrain` (or `aistuff`) environment active.

## Before the recording day

| Done | Item |
|---|---|
| ☐ | `main` has the training pipeline, `dvc.yaml`, `retrain.yml`, and the champion and dataset on DagsHub (`dvc status -c` shows nothing to push) |
| ☐ | Secret `DAGSHUB_TOKEN`, variable `RETRAIN_MODE`, environment `model-approval` with reviewers, and workflow permissions are set (`docs/mlops/retraining.md`) |
| ☐ | The hosting PC prints `READY` from `python -m run.check_env`, and its runner shows **Idle** |
| ☐ | `data/new/batch_001` was pushed earlier and a **complete GPU run** for it exists in the Actions history (about 1 hour for 20 epochs). The video cuts to that run instead of filming an hour of training |
| ☐ | `data/new/batch_002` and `batch_003` exist locally and are **not** pushed yet (`git status` shows them untracked) |
| ☐ | At least one scheduled run in the history. To get one quickly, merge a commit that sets the cron to `*/15 * * * *` (`ci: run the retraining schedule every 15 minutes for the demo`), wait for one run, then revert it (`ci: restore the weekly retraining schedule`) |
| ☐ | One manual CPU run in the history, in case the live one is slow |

## On the day

- Browser zoom 125%, notifications off, signed in to GitHub and DagsHub.
- Terminal with a large font, in the repository root.
- Open tabs: the repository, Actions, Settings → Actions → Runners, the DagsHub storage page.
- `RETRAIN_MODE=gpu` and the runner online.

## Scenes

### 1. Overview (0:40)
**Show:** the repository tree on `main`, then the flow diagram in `docs/mlops/retraining.md`.
**Say:** the model is YOLO26n trained on about 29,000 images. Retraining it needs a GPU, new data is images
with labels, and the test set is frozen. That is why this pipeline differs from a CSV example.

### 2. Commit history (0:20)
**Show:** `git log --oneline --graph -25`
**Say:** each change is one Conventional Commit (`feat(data)`, `feat(monitor)`, `ci:`) merged through pull
requests.

### 3. DVC and remote storage (1:15)
**Show:**
```bash
cat .dvc/config            # remote endpoint only; the token is in config.local, not in git
dvc remote list
cat backend/dvc.yaml
cd backend && dvc dag && dvc status && cd ..
```
Then the DagsHub storage browser for `0ceanya/pothole-uav-data`.
**Say:** which data is versioned (dataset v1 images, lists and test lock, the drift reference, the champion
weights, each new batch), where it is stored, and the four pipeline stages.

### 4. Data push starts retraining (1:30)
**Show:**
```bash
cd backend
dvc add data/new/batch_002
dvc push data/new/batch_002.dvc
git switch -c data/batch-002
git add data/new/batch_002.dvc data/new/.gitignore
git commit -m "feat(data): add RDD2022 Czech batch 002"
git push -u origin data/batch-002
```
Open the pull request, merge it, then open Actions: "Retrain on New Data" starts with event `push`.
**Say:** new labelled images are versioned with DVC, and the pointer file in git is what starts the
workflow.

### 5. Monitoring (1:00)
**Show:** the `check` job log (pending batches, drift p-values, trigger decision), the job summary table,
then download `monitoring-<run id>` and open `drift_report.html`. If drift was flagged, open the issue the
run created.
**Say:** every new batch is checked for label problems, exact and near copies of validation and test images,
and drift in brightness, contrast, blur, size, view and class mix.

### 6. GPU retraining (1:15)
**Show:** Settings → Actions → Runners, with the runner **Active**. On the host, `nvidia-smi`. In the
`retrain-gpu` log, `check_env` READY and `dvc repro` starting. Then cut to the finished `batch_001` run: the
summary table with champion and challenger on validation and test, and `decision.json`.
**Say:** training warm-starts from the champion, the test lock is checked before and after test scoring,
and the promotion decision uses validation only.

### 7. Human approval (0:45)
**Show:** "Review deployments" → select `model-approval` → Approve and deploy. The `promote` job log, then
the pull request `chore(retrain): promote model v2` with `champion.pt.dvc` and the metadata.
**Say:** a new model only replaces the champion after a person approves it, and the change is reviewable as
a pull request. If the challenger didn't pass, show the FAIL lines in `decision.json` instead: the champion
stays.

### 8. Scheduled run (0:30)
**Show:** the `schedule:` block in `.github/workflows/retrain.yml`, then a run with event `schedule` in the
Actions list.
**Say:** a weekly run monitors the data and retrains only when there are batches the champion has not seen.

### 9. Manual run and CPU fallback (0:45)
**Show:** Actions → Retrain on New Data → Run workflow → mode `cpu`, force on. Open the `retrain-cpu` job and
its summary, which ends with "hosted CPU demo models are never promoted".
**Say:** anyone can start a run by hand, and it still works when no GPU PC is online.

### 10. Artefacts (0:30)
**Show:** download `retrain-<run id>` and open `evaluation_metrics.json`, `training_history.json`, a results
plot, and `artifacts/metadata/model_version.txt` and `last_retrain.txt`.
**Say:** all outputs of a run are kept together and the versions are tracked.

### 11. Moving the runner to another PC (0:30)
**Show:** the "Hosting the GPU runner on any PC" section of `docs/mlops/retraining.md`,
`backend/environment-retrain.yml`, and a runner `.env` with `RETRAIN_PY` and `DVC_CACHE_DIR`.
**Say:** any team PC with an NVIDIA GPU can take over: create the environment, pull the data, run
`check_env`, register the runner.

## After recording

- Revert any temporary schedule change and make sure `RETRAIN_MODE` is back to `cpu` if no runner stays
  online.
- Stop the runner on the host PC.

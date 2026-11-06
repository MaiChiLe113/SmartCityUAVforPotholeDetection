# Local mission simulator verification

Date: 6 November 2026.  
Base commit: `dc471fb18cb17233b79a0f6217c0c1398c340196`.  
Working branch: `SCRUM-33`.

## Scope and reviewers

This delivery implements the local flight planner, recorded-image source, mission engine, mock detector, and isolated demonstration runner. It provides a way to test simulation software before the trained model and application integrations are ready.

The user requested GPT-6.1 Terra at High for implementation and GPT-6.1 Sol at Extra High for independent verification between phases and at the end. The builder paused at every gate. The verifier ran checks independently and did not edit the implementation.

Live report/approval/dispatch APIs, web dashboard integration with the new simulator, PC-to-Pi image transfer, actual Pi hardware execution, and trained-model inference are later work. Existing API placeholders remain explicit; the isolated demo is not an alternative way to dispatch a live mission.

## Baseline

An untouched checkout of the base commit was checked with the same backend environment:

- Backend tests: **8 passed, 2 expected xfails**.
- Backend Ruff: **passed**.
- The expected xfails cover the existing unimplemented model-training arguments and defect-geolocation functions.
- The installed Starlette/TestClient combination emits a pre-existing deprecation warning.

Backend execution environment: Windows, Python 3.12, isolated `backend/.venv`, project core/development dependencies. The project declares Python 3.11 or newer; the checks in this record do not claim testing on every supported Python version or ARM hardware.

## Independent phase gates

| Gate | Result | Evidence |
| --- | --- | --- |
| Phase 1: route planner and recorded camera | PASS after fixes | 28 tests passed, 2 existing xfails; full Ruff passed |
| Phase 2: mission engine and mock processing | PASS after fixes | 54 tests passed, 2 existing xfails; full Ruff passed |
| Phase 3: demo runner and output artifacts | PASS after fixes | 73 tests passed, 2 existing xfails; Ruff, dependencies, and diff checks passed; fresh core/dev installation verified |
| Final complete review | PASS | Sol reviewed the complete working tree and independently reran 73 passing tests / 2 existing xfails, Ruff, dependency and diff checks; documented demo and network-blocked execution passed |

### Phase 1 findings resolved

- The initial ground waypoint now uses the existing supported waypoint-action contract.
- Captured frames retain their own pose snapshots rather than references to a reusable moving pose.
- Directory scanning ignores directories with image-like suffixes.
- Independent checks included 2,000 randomized spherical routes and antimeridian cases, frame ordering/exhaustion, pose isolation, and frontend action compatibility.

### Phase 2 findings resolved

- Configurable finite processing timeout prevents never-ready jobs from waiting and accumulating telemetry forever.
- Processor result validation rejects malformed detections, wrong frame identity, nonfinite confidence, changed capture metadata, and supplied error values including empty strings.
- Clock failures are caught; nonfinite, backward, and stalled clock behaviour fails cleanly.
- Spherical movement interpolation preserves configured speed on supported high-latitude/antimeridian routes.
- Independent timeout-boundary checks confirmed that a result ready at the deadline succeeds and a result beyond the deadline fails.
- `fly()` is documented as an eager convenience iterator over processed frames, not a live streaming interface.

### Phase 3 findings resolved

- Redirected Windows output handles Unicode paths without turning a completed mission into a failed CLI process; both stdout and stderr have a legacy-encoding fallback.
- Configured fixture names are validated before writing: three unique safe PNG basenames, including case-insensitive collision and path-containment checks. Independent sentinel tests confirmed that outside files are preserved.
- Ordinary processing delay must be below the deadline in every scenario. The timeout scenario separately injects a second-frame delay that exceeds the deadline, matching the saved manifest.
- Invalid image parser errors, including corrupt PNG checksums and truncated/header-only input, become clean input refusals without tracebacks.
- A fresh environment installed `.[dev]`, imported the demo without heavy ML/network clients, and completed the positive case.

## Root end-to-end acceptance

After the Phase 3 fixes, root ran the real CLI in the actual Vietnamese-named OneDrive project path, with redirected output and without UTF-8 environment overrides.

| Scenario | Exit | Mission outcome | Successful frames | Failed frames | Detections |
| --- | --- | --- | --- | --- | --- |
| Positive | 0 | done; home ground reached | 3 | 0 | 1 preset D40 |
| All empty | 0 | done; home ground reached | 3 | 0 | 0 |
| Error | 1 | failed; partial results saved | 1 | 1 | 0 |
| Timeout, 3-second deadline | 1 | failed; partial results saved | 1 | 1 | 0 |
| No demo acknowledgement | 2 | refused before output creation | not started | not started | not started |

For all four executed missions, root independently checked finite JSON values, recorded input/mock labels, frame counts, final telemetry state, expected success/failure, and home position on completion. Local generated evidence is under `backend/runs/verification/`; `acceptance-results.json` records the exact summary-file paths. Generated evidence is ignored by Git and can be recreated.

From `backend/`, with the local environment already prepared:

```powershell
.\.venv\Scripts\python.exe -m run.demo_mission --approve-demo --scenario positive
.\.venv\Scripts\python.exe -m run.demo_mission --approve-demo --scenario all-empty
.\.venv\Scripts\python.exe -m run.demo_mission --approve-demo --scenario error
.\.venv\Scripts\python.exe -m run.demo_mission --approve-demo --scenario timeout --timeout 3
```

Root also repeated the complete backend tests: **73 passed, 2 existing xfails**. Full Ruff, `python -m pip check`, and `git diff --check` passed. This execution used core/development dependencies plus the declared lightweight Pillow dependency; no model weights, model server, Pi, GPU, or simulator network connection was used.

## Final independent audit

GPT-6.1 Sol at Extra High returned **PASS** after reviewing every tracked change and new file, the README, and this verification record. No remaining actionable defects were found in the delivered local scope.

The verifier reran `python -m pytest -q -rx` (**73 passed, 2 existing expected xfails**), full Ruff, dependency checks, and diff checks. It independently checked root's actual-workspace acceptance artifacts and ran the exact documented default command successfully: three successful frames, one preset D40 detection, and completion at home. That default run's local artifacts are under `backend/runs/simulator_demo/demo-9agx0ayt/`.

The complete demo also succeeded with network connections blocked, without loading heavy ML or network-client modules. The final README setup and timeout instructions matched the executed behaviour. All three phase gates and the final complete audit passed; implementation and test files were not changed after this audit. This final status entry records the returned review decision.

## Existing frontend and dependency checks

The existing frontend source was not changed. Root independently ran:

```powershell
npm ci --no-audit --no-fund
npm run lint
npm run build
```

All commands completed successfully. Next.js compiled and generated the existing routes. `NEXT_TELEMETRY_DISABLED=1` was set for the build.

The installation also reported pre-existing notices for the pinned Next.js 15.5.4 security status and deprecated ESLint version. These dependency maintenance issues were not introduced or repaired by the simulator implementation; successful lint/build results are not a security audit or a production-readiness claim. The package's referenced [Next.js advisory](https://nextjs.org/blog/CVE-2025-66478) identifies the affected release line and patched versions.

The backend dependency check (`python -m pip check`) also passed after the Phase 3 image dependency change.

## Interpretation of results

Mock detections verify software behaviour and data flow. Their confidence values, processing delays, and boxes are presets, not measurements of AI quality or real Pi performance. Recorded or synthetic test images are not observations captured at the simulated UAV pose. This delivery does not validate flight physics, actual inspection confirmation, damage geolocation, or physical defect size.

Processor `start`/`poll` callbacks are required to be nonblocking. The engine's deadline can stop a never-ready asynchronous job; it cannot interrupt arbitrary blocking code inside a callback. A future real model/remote-worker adapter must honor that contract.

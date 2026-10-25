# JEPA-Med-OOD dashboard (read-only front end + inference API)

Save the design image as `dashboard/design/reference.png` (not included).

## Run

Backend (repo root): `python -m uvicorn dashboard.backend.main:app --port 8000`
Frontend: `cd dashboard/frontend && npm install && npm run dev` (port 5173).

## Data honesty

- Reference mockup numbers (0.82, 0.92, ...) are placeholders; never hardcoded.
- Every number comes from `CHECKPOINT_DIR` / `RESULTS_DIR` / live inference.
- Missing checkpoints -> "Checkpoint not found"; missing metrics -> "Not computed".
- AUPR and OOD-Accuracy are not in any result file; their charts show "Not computed".
- 0-1 OOD score formula: `normalized = P(ID_validation_score <= raw)`
  (empirical percentile within cached ID-validation scores; higher = more OOD).
- Threshold: ID 95th percentile (`TPR_TARGET`), ID data only, never OOD.
- Attention endpoint = attention rollout via forward hooks (column-mean of the
  rollout product); response header says so; never labelled Grad-CAM.
- Uploads: max 10 MB, PNG/JPG/JPEG, in-memory only, grayscale->RGB as the repo does.
- Paths from `CHECKPOINT_DIR`, `RESULTS_DIR`, `DATA_ROOT` env vars (repo-relative defaults).
- Demo: `python -m uvicorn dashboard.backend.main:app -- --demo` fills fake data
  with a permanent "DEMO DATA - NOT REAL RESULTS" banner. Off by default.

Footer/About: "Research prototype for OOD detection. Not a medical device and not
for diagnosis."

# Misan

AI fault detection and diagnosis module for HVAC systems — Detect · Diagnose · Explain · Hand off.
MVP implementation per the project PRD (`Pasted text #1` / `MISAN · MVP Product Requirements v0.1`).

Misan ingests a window of HVAC sensor readings, detects abnormal operation, diagnoses the most
likely fault and affected component, attaches a calibrated confidence and severity estimate,
explains the evidence (SHAP), runs a lightweight physics-signature consistency check, and emits
a structured record for the (external) physics-validation stage and maintenance optimiser.

## Architecture

```
config/            fault taxonomy, sensor schema, physics signature table (YAML, edit to extend)
misan/              Python package — the engine
  ingestion.py       CSV/Parquet loading, schema validation, cleaning, data-quality flags
  features.py        sliding-window statistics + derived thermodynamic features
  detection.py       binary healthy/faulty model + alarm persistence rule
  diagnosis.py       calibrated multiclass fault classifier
  severity.py        severity regression, 3-level label, trend tracking
  explainability.py  SHAP top-feature attribution + plain-language sentence
  physics_validation.py  fault-signature consistency check (confirmed/rejected/unchecked)
  baseline.py        simple rule-based FDD baseline for comparison
  persistence.py      SQLite store: diagnostic records, dataset/model registry, evaluation runs
  simulate.py         physics-based SYNTHETIC DATA GENERATOR — dev/research tool only
  training.py         reproducible training pipeline (group-aware split, model selection, calibration)
  evaluation.py        benchmark pipeline (Section 11 metrics, noise robustness, baseline comparison)
  pipeline.py          the live engine: raw window -> structured DiagnosticRecord
  api.py               FastAPI service
  cli.py                `simulate` / `train` / `evaluate` / `serve` commands
dashboard/           Streamlit dashboard (Overview, Data, Model Evaluation)
tests/               pytest suite (isolated temp DB/model dir, see tests/conftest.py)
```

No official Misan brand guide was found in the workspace. The dashboard palette is derived,
with a documented and principled mapping, from the PRD document's own five section colours
(see `dashboard/theme.py`): red = fault, amber = warning, green = normal, teal = secondary,
purple = brand accent.

## Zero demo data

The application starts with an empty database — no fake users, assets, alerts, or metrics.
Diagnostic records only exist once a real window (imported or simulated data) is submitted to
`/diagnose`. Synthetic training data is only ever created via the explicit `simulate` command
below; it is written to `data/raw/` and must be explicitly selected for training, never loaded
into the live application automatically.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # adjust if needed
```

## Run

```bash
# 1. Generate a labelled synthetic dataset (dev/research tool — see misan/simulate.py)
python -m misan.cli simulate --runs-per-condition 2 --duration-minutes 60

# 2. Train a model (group-aware split by run, compares RF/GBM/SVM/KNN, calibrates, versions the artifact)
python -m misan.cli train --dataset data/raw/simulated_<timestamp>.parquet

# 3. Benchmark against the rule-based baseline
python -m misan.cli evaluate

# 4. Start the API
python -m misan.cli serve

# 5. Start the dashboard (separate terminal)
streamlit run dashboard/app.py
```

Dashboard: http://localhost:8501 · API docs: http://localhost:8000/docs

## Tests

```bash
pytest -q
```

Tests run against an isolated temp SQLite DB and model directory (see `tests/conftest.py`) —
they never touch `data/misan.db` or the real `data/models/` used by a running instance.

## Known limitations (honest, not hidden)

- **Simulator, not the real physics model.** `misan/simulate.py` is a simplified, documented
  vapour-compression proxy built for this MVP (ramping faults, approximate R410A saturation
  curve fit). Results describe how the pipeline behaves on this proxy, not on a real plant —
  the near-perfect benchmark scores obtained on it should be read accordingly (this is the
  exact risk called out in PRD Section 16, "model learns the simulator, not the physics").
- **Persistence/severity-trend state is in-memory per process** (`PersistenceTracker`,
  `SeverityTrendTracker`). A service restart delays — never falsely raises — the next alert.
  Production use should back these with the database.
- **Physics signature check uses one global healthy baseline** across all simulated operating
  conditions (ambient/load), not conditioned per-condition — it can over/under-reject near the
  edges of the training envelope. FR-22 explicitly earmarks this for replacement by the real
  physics component models.
- **No live sensor feed** (FR-6, explicitly P2/deferred). Windows are submitted via `/diagnose`
  from imported/simulated files through the dashboard's Data page.
- **XGBoost dropped** from the suggested stack — unused in the final candidate-model set
  (RandomForest/GradientBoosting/SVM/KNN already cover the tree + lightweight comparison
  requirement) and its wheels are not yet published for the local Python 3.14 environment.

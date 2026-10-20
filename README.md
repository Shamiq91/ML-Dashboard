# ML Pipeline dashboard

A single-screen dashboard for tabular machine learning. A FastAPI backend (`backend/`) does the work; a React + Tailwind + Recharts
client (`frontend/`) shows it as six numbered panels, a toolbar, a live "radio" log and a legend.

_Status as of 2026-10-07._

## Run it

Two terminals:

```bash
cd backend && .venv/bin/uvicorn app.main:app --port 8000
```
```bash
cd frontend && npm install && npm run dev      # http://localhost:5173
```

First-time backend setup: `cd backend && python -m venv .venv && .venv/bin/pip install -r requirements.txt`.
**Windows (PowerShell):** the venv executables live in `.venv\Scripts\` instead of `.venv/bin/`, and older PowerShell has no `&&`, so run the lines separately.

```powershell
cd backend
python -m venv .venv                              # or: py -m venv .venv
.venv\Scripts\pip install -r requirements.txt     # first time only
.venv\Scripts\uvicorn app.main:app --port 8000
```
```powershell
cd frontend
npm install
npm run dev
```

Backend tests on Windows: `cd backend` then `.venv\Scripts\python -m pytest`.

The dev client talks to `http://localhost:8000` (override with `VITE_API_BASE_URL`). API docs: http://localhost:8000/docs.

Sample data in the repo root: `sample_churn.csv` (classification, 7,043 rows, a few missing values). Regenerate with `backend/scripts/`.

## The layout

One screen, no page navigation. Drag the gutters between panels to resize (double-click to reset); the legend and the radio strip collapse.

| Toolbar | Upload (click or drop `.csv` / `.parquet`), target, task (`AUTO` / classify / regress), test split, scaling, imbalance, F-score average, **TUNE on/off + trials**, Run, Reset |
|---|---|

| Panel | Default view | Extra tabs |
|---|---|---|
| **01 Dataset status** | Row/column/memory/missing counts, impute controls (the only cleaning in the UI), feature stats, target balance | `PREVIEW`: sortable, searchable, paged rows; per-column profiles; working vs raw data. `FEATURES`: feature selection (see below) |
| **02 Assumption check** | Normality, independence (Durbin-Watson), homoscedasticity, overall pass/fail | |
| **03 Leaderboard / Registry** | Ranked models with hardware, score, gap to leader, Clean/Train/Eval blocks, `OVF` / `UNF` / `NV` / `TUN` flags; registry to pick from 23 models | |
| **04 Training telemetry** | Learning curve or loss per epoch, overfit/underfit notes | `ROC/PR/THR` (ROC, PR, decision-threshold slider, confusion matrix, best-F1 / Youden); `RESIDUALS` for numeric targets; `TUNING` (Optuna history, best parameters, which parameters mattered) |
| **05 Model comparison** | Metric bars for every trained model | `SUMMARY` (recommended model and why, led by cross-validation, with a mean +/- std chart and every metric in one table); `ROC` overlay; `RUNS` (saved runs, models across runs, dataset comparison, curve overlay) |
| **06 Feature analytics** | Correlation heatmap with multicollinearity flags | `IMPORTANCE` (permutation / model-native); `SHAP` (global, beeswarm, one-row explanation) |
| **07 Legend** | Collapsible key to every abbreviation and colour | `THRESHOLDS`: sliders for the overfit gap and underfit skill (re-evaluated in the browser, no retraining) |

Analysis tabs (04, 05, 06) follow the model selected in the leaderboard.

## What it can do

- **Data:** CSV or Parquet up to 50 MB. A categorical target trains classifiers; a numeric target with more than 20 distinct values
  trains regressors (a dated file with one row per timestamp is treated as plain regression). IDs and constant columns are excluded.
- **Models (23):** XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, Bagging, Stacking, Voting,
  Decision Tree, KNN, SVM, Logistic / Linear / Ridge / SGD, Naive Bayes, LDA, QDA, scikit-learn MLP, **PyTorch MLP (PTM)** and
  **TabNet (TAB)**. 19 have a regression counterpart.
- **Tuning:** Optuna per model, cross-validated on training rows only; defaults are kept unless a trial beats them.
- **Cross-validation:** every model is also k-fold cross-validated (5 folds; slow neural models use 3) on its training rows only, using
  its tuned hyper-parameters, SMOTE inside each fold. It runs in the backend during training; the UI only shows it in the Summary.
  Per metric it stores mean, standard deviation, min, max and each fold's score (`cv` on every leaderboard row).
- **Summary:** panel 05 `SUMMARY` names the recommended model and explains why. It ranks by **cross-validated** score, because one hold-out
  split can be lucky, and says when the two disagree, how stable the model is across folds, and which models are statistically tied
  (within one standard error). It reports precision / recall / ROC AUC (or R2 / RMSE / MAE against the target's spread), the
  train-vs-test gap, the class-imbalance warning and training time. A model flagged overfit or underfit is never recommended while a
  clean one exists, and the summary says which ones it passed over.
- **Feature selection:** panel 01 `FEATURES` ranks columns by mutual information with the target (catches non-linear links) and
  flags IDs, constants, skew, outliers and missing values. It suggests drops (mostly empty, near-duplicate pairs, weak columns as one
  group), log transforms and date parts; each suggestion shows the exact change and applies on click.
- **Transforms:** log or clip a column in place. Changes reset trained results, `UNDO ALL` restores the raw file.
- **Saved runs:** every finished training job is stored (SQLite) and can be compared in panel 05 `RUNS`.
- **Live radio:** WebSocket log of the pipeline (`T+MM:SS INFO/WARN/ERROR/SUCCESS`) that also drives the CTE blocks.

Workflow: upload, set target, **impute missing values** (training is blocked until none remain), pick models, Run.

## Current status

Working and checked in the browser on the sample files: upload, target, impute, training with and without tuning, PTM and TabNet,
leaderboard flags, every new tab above on classification, and the `SUMMARY` tab on classification and regression.

Known gaps:

- **`Compare` is panel 05 `RUNS`,** not its own page. Renaming runs is available in the API only.
- **Some cleaning is backend-only.** `POST /sessions/{id}/transform` also supports filters, casts, renames, duplicate removal and whitespace trimming; the UI exposes drops, logs and clips.
- **Cross-validation uses at most 5,000 training rows per model** and adds time to training (about 2x for fast models, more for neural nets). `cross_validate: false` in the train request turns it off; the Summary then falls back to the single hold-out split and says so.
- **The Summary is a recommendation, not a guarantee:** the data you see later may differ.
- **Narrow windows** (about 1000 px) truncate some panel titles to make room for the tabs.

## Code map

```
backend/app/          FastAPI app: routers/ (HTTP), services/ (modeling, tuning, analysis, torch models), state/
backend/tests/        pytest suite (134 tests)
frontend/src/App.jsx  the whole single-screen dashboard (state, panels, toolbar, legend)
frontend/src/features/analysis.jsx   tab bodies added to panels 01, 04, 05, 06
```

See `ARCHITECTURE.md` for session state, the stage guard, leak-free training, and the radio design.

## API (all under `/api/v1`)

Upload, target, clean/reset, missing, class-balance, eda, assumptions, train (accepts `tune`, `tune_trials`, `label`), leaderboard,
telemetry, models, logs and `/ws/radio`, plus:

| Method | Path | Purpose |
|---|---|---|
| GET | `/sessions/{id}/preview` · `/profile` | paged rows · per-column profiles |
| POST | `/sessions/{id}/transform` | cleaning and column operations (undo with `/reset`) |
| GET | `/sessions/{id}/features/suggest` | feature relevance (mutual information), redundant pairs, ready-to-apply suggestions |
| GET | `/sessions/{id}/curves/{model}` | ROC, PR, threshold sweep, confusion matrix |
| GET | `/sessions/{id}/residuals/{model}` | actual vs predicted, residual histogram |
| GET | `/sessions/{id}/importance/{model}` | permutation (+ native) importance |
| GET | `/sessions/{id}/shap/{model}` | SHAP summary, beeswarm, one-row explanation |
| GET | `/sessions/{id}/tuning/{model}` | Optuna trials and parameter importance |
| GET / PATCH / DELETE | `/runs`, `/runs/{id}` · GET / DELETE `/datasets` | saved runs and datasets |
| POST | `/compare` | align 2 to 8 runs |

## Tests

```bash
cd backend && .venv/bin/python -m pytest      # 134 passed
cd frontend && npx vitest run                 # 41 passed (the live end-to-end test needs the backend on :8000, else it is skipped)
```

Frontend tests: `unit.test.js` (formatters, model catalog, overfit/underfit flags, layout helpers), `summary.test.jsx` (the SUMMARY
recommendation logic, including cross-validated ranking, ties and fallbacks), `features.test.jsx` (the FEATURES tab against a
mocked API), `app.test.jsx` (empty dashboard, toolbar, panel tabs, legend) and `e2e.test.jsx` (upload, feature
selection, impute, tuned training with cross-validation, every analysis tab, against the live backend).

## Version control

The project is a git repository. `main` holds the restored single-screen layout with the new tabs.

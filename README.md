# ML Pipeline dashboard

A single-screen dashboard for tabular machine learning. A FastAPI backend (`backend/`) does the work; a React + Tailwind + Recharts
client (`frontend/`) shows it as six numbered panels, a toolbar, a live "radio" log and a legend. Upload a file, pick a target, clean
it, train up to 23 models side by side, and see which one to trust and why.

## Run it

First-time setup:

```bash
cd backend && python -m venv .venv && .venv/bin/pip install -r requirements.txt
cd frontend && npm install
```

Development (two terminals), then open http://localhost:5173:

```bash
cd backend && .venv/bin/uvicorn app.main:app --port 8000
```
```bash
cd frontend && npm run dev
```

Single server (e.g. for a demo): build the client once and the backend serves it at http://localhost:8000. Rebuild after any
frontend change.

```bash
cd frontend && npm run build
cd backend && .venv/bin/uvicorn app.main:app --port 8000
```

**Windows (PowerShell):** the venv executables live in `.venv\Scripts\` instead of `.venv/bin/`, and older PowerShell has no `&&`, so
run the lines separately:

```powershell
cd backend
python -m venv .venv                              # or: py -m venv .venv
.venv\Scripts\pip install -r requirements.txt     # first time only
.venv\Scripts\uvicorn app.main:app --port 8000
```

The client calls the API on its own origin when the backend serves it, and `http://localhost:8000` from the dev server (override with
`VITE_API_BASE_URL`). API docs: http://localhost:8000/docs.

Sample data in the repo root: `sample_churn.csv` (synthetic classification data, 7,043 rows, a few missing values). Regenerate it with
`backend/scripts/make_sample_data.py`.

## Workflow

1. **Upload** a `.csv`, `.parquet`, `.json` or `.jsonl` file.
2. **Set the target.** A categorical target trains classifiers; a numeric target with more than 20 distinct values trains regressors.
3. **Clean.** Impute (mean, median, mode or KNN) or drop missing values; training is blocked until none remain. Prefer imputing:
   dropping removes every row with a gap.
4. **Pick models** in the registry and **Run**. For two-class targets, check the **POSITIVE** class first (see below).
5. **Choose a model** by its cross-validated score, then set its decision threshold for your use case.

## The layout

Two pages, switched with `DASHBOARD | MODEL LAB` next to the title (the lab is also reachable at `#lab`). The dashboard is one screen of
panels: drag the gutters between them to resize (double-click to reset); the legend and the radio strip collapse. Switching pages keeps
everything you loaded.

| Toolbar | Upload, target, task (`AUTO` / classify / regress), test split, scaling, imbalance (none / class weight / SMOTE), **positive class**, F-score average, **TUNE on/off + trials**, Run, Reset |
|---|---|

| Panel | Default view | Extra tabs |
|---|---|---|
| **01 Dataset status** | Row/column/memory/missing counts, cleaning controls, feature stats, target balance | `PREVIEW`: sortable, searchable, paged rows; per-column profiles; working vs raw data. `FEATURES`: feature selection |
| **02 Assumption check** | Normality, independence (Durbin-Watson), homoscedasticity of an OLS fit; overall verdict. For a categorical target these linear-regression checks are marked not applicable | |
| **03 Leaderboard / Registry** | Ranked models with hardware, score, gap to leader, Clean/Train/Eval blocks, `OVF` / `UNF` / `NV` / `TUN` flags; registry to pick from 23 models | |
| **04 Training telemetry** | Learning curve or loss per epoch, overfit/underfit notes | `ROC/PR/THR` (ROC, PR, decision-threshold slider, confusion matrix, best-F1 / Youden); `RESIDUALS` for numeric targets; `TUNING` (Optuna history, best parameters, which parameters mattered) |
| **05 Model comparison** | Metric bars for every trained model | `SUMMARY` (recommended model and why, led by cross-validation); `ROC` overlay; `RUNS` (saved runs, comparison across runs and datasets); `EXPORT` (model card and `DOWNLOAD .ZIP`) |
| **06 Feature analytics** | Correlation heatmap with multicollinearity flags | `IMPORTANCE` (permutation / model-native); `SHAP` (global, beeswarm, one-row explanation) |
| **07 Legend** | Collapsible key to every abbreviation and colour | `THRESHOLDS`: sliders for the overfit gap and underfit skill (re-evaluated in the browser, no retraining) |

Analysis tabs (04, 05, 06) follow the model selected in the leaderboard.

### Model lab

One model at a time, with every hyper-parameter in your hands: Linear regression, Logistic regression, KNN, XGBoost, Decision tree,
Random forest, K-means or SVM. It uses the dashboard's data and target (set the target and impute missing values there first; K-means
needs no target) and never touches the dashboard's results.

- **Controls (left):** a slider, switch or choice per hyper-parameter, each with a one-line explanation. `DEFAULTS` resets them; a
  feature checklist trains on a subset of columns.
- **Lab bar:** test split, scaling, 5-fold CV, `AUTO REFIT` (or refit with `FIT`), `PIN AS A` (later fits show their difference from
  the pinned result).
- **Every model:** train / test / cross-validated metrics, the train-test gap, plain-language notes (overfitting with fixes, CV vs
  hold-out disagreement, unstable folds, solver not converged), a validation curve or 2-parameter score grid, and a learning curve.
  Classifiers add the threshold slider, ROC, PR, class separation and calibration; regressors add actual vs predicted, residuals and
  a Q-Q plot.

| Model | Graphs |
|---|---|
| Linear regression | Fitted equation; coefficients with 95% intervals, p-values and VIF; ridge / lasso / elastic-net path |
| Logistic regression | Coefficients and odds ratios per class; regularization path over C; 2D decision surface |
| KNN | Distance to the k-th neighbour and accuracy by distance; explain any test row (neighbours, distances, votes); 2D decision surface |
| XGBoost | Loss per boosting round on fit and validation rows; importance by gain / splits / cover; any boosted tree drawn |
| Decision tree | The tree drawn (depth 2 to 5): each box shows its question, the predicted class, row count and the real class mix of the training rows that reach it; if-then rules; pruning path; importance |
| Random forest | Out-of-bag and test score as trees are added; MDI vs permutation importance; tree agreement per row; any tree drawn |
| K-means | Clusters on 2 principal components; silhouette per row and cluster; cluster profiles; clusters vs target; elbow + silhouette over k |
| SVM | Margin distribution (or errors vs the epsilon tube); support vectors per class; 2D decision surface; C x gamma grid |

The 2D decision surface refits a copy of the model on the first two principal components so it can be drawn; read it as intuition. To
stay interactive the lab fits on at most 20,000 rows (kernel SVM 4,000) and cross-validates on 4,000; limits are the `lab_*` settings in
`backend/app/config.py`.

## How results are computed

- **Leak-free training.** Rows are split once (stratified for classes) into training and test rows. Scaling and encoding are fit on
  training rows only; class weighting and SMOTE only ever touch training rows. Every model's final fit uses all training rows;
  iterative models (XGBoost, LightGBM, MLPs, TabNet) additionally draw their loss curves from a separate fit on an inner
  train/validation split.
- **Positive class.** For two-class targets, precision, recall, F1 and the ROC/PR/threshold charts are about the positive class. It
  defaults to the rarer class (churn, default, stroke ...) and can be changed with **POSITIVE**; the chip beside it and the leaderboard
  footer always show which class the numbers are about.
- **Cross-validation.** Every model is also k-fold cross-validated (5 folds; slow neural models use 3) on its training rows only, with
  the same class weights or SMOTE as its final fit, applied inside each fold. Mean, standard deviation and per-fold scores are stored
  on each leaderboard row. The Summary ranks by this score, because one hold-out split can be lucky.
- **Tuning.** Optuna per model, cross-validated on training rows the same way; the defaults are kept unless a trial beats them.
- **Scores.** Models without probabilities (e.g. SVM) expose a decision score mapped so that 0.5 is the model's own decision boundary,
  in the dashboard and in the exported `predict.py` alike.
- **Regression.** Each regressor trains on a target standardised with training-row statistics; R^2 is unchanged and RMSE / MAE are
  reported back in the target's units.
- **Excluded columns.** Constant columns and row identifiers (text unique per row, or whole numbers named like an ID or counting
  1..n) are not used as features.

Training metrics have been cross-checked against plain scikit-learn on several public datasets (binary and multi-class, with and
without class weighting) and match to rounding; `backend/tests/test_correctness.py` guards the fixes behind that.

## What else it can do

- **Models (23):** XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, Bagging, Stacking, Voting,
  Decision Tree, KNN, SVM, Logistic / Linear / Ridge / SGD, Naive Bayes, LDA, QDA, scikit-learn MLP, PyTorch MLP and TabNet. 19 have a
  regression counterpart.
- **Feature selection:** panel 01 `FEATURES` ranks columns by mutual information with the target and flags IDs, constants, skew,
  outliers and missing values. It suggests drops, log transforms and outlier clipping; each suggestion shows the exact change and
  applies on click. The target column itself can't be transformed.
- **Saved runs:** every finished training job is stored in SQLite under `backend/data/` (not in git) and can be compared in panel 05 `RUNS`.
- **Export:** panel 05 `EXPORT` downloads the selected model as a zip: `model.joblib`, `predict.py`
  (`python predict.py model.joblib new.csv out.csv`), `MODEL_CARD.md` / `model_card.json`, `requirements.txt` and `example_input.csv`.
  `predict.py` repeats the column steps applied before training, fills missing cells with the training median / most common value and
  reports unseen categories. PyTorch MLP and TabNet bundles need the backend folder on `PYTHONPATH`; the card says so.
- **Live radio:** a WebSocket log of the pipeline (`T+MM:SS INFO/WARN/ERROR/SUCCESS`) that also drives the CTE blocks.

## Known limitations

- **Imputation runs before the train/test split,** so fill values are computed from all rows. The effect on scores is small; the
  model-specific preprocessing (scaling, encoding, resampling) is fit on training rows only.
- **Best-F1 and Youden thresholds are chosen on the test rows,** so the score at those thresholds is slightly optimistic. Quote the
  cross-validated score as the headline number.
- **Some cleaning is API-only.** `POST /sessions/{id}/transform` also supports filters, casts, renames, duplicate removal and
  whitespace trimming.
- **Cross-validation uses at most 5,000 training rows per model** and adds training time; `cross_validate: false` in the train request
  turns it off.
- **Lab results are not saved** to run history.
- **Narrow windows** (about 1000 px) truncate some panel titles; below 1024 px the lab stacks its controls above the graphs.

## Code map

```
backend/app/routers/              HTTP endpoints
backend/app/services/modeling.py  dataset preparation, training loop, metrics and ranking
backend/app/services/crossval.py  cross-validation; tuning.py: Optuna search
backend/app/services/analysis.py  ROC/PR, residuals, importance, SHAP on fitted models
backend/app/services/inference_runtime.py  shared data preparation; shipped as predict.py in exports
backend/app/services/lab.py       model lab fitting, diagnostics and sweeps (lab_catalog.py: its models and controls)
backend/tests/                    pytest suite
frontend/src/App.jsx              the dashboard (state, panels, toolbar, legend) and the page switch
frontend/src/features/            analysis tabs, export tab, model lab page
```

See `ARCHITECTURE.md` for session state, the stage guard, concurrency and the radio design.

## API (all under `/api/v1`)

Upload, target, clean/reset, missing, class-balance, eda, assumptions, train (accepts `tune`, `tune_trials`, `positive_class`,
`label`), leaderboard, telemetry, models, logs and `/ws/radio`, plus:

| Method | Path | Purpose |
|---|---|---|
| GET | `/sessions/{id}/preview` · `/profile` | paged rows · per-column profiles |
| POST | `/sessions/{id}/transform` | cleaning and column operations (undo with `/reset`) |
| GET | `/sessions/{id}/features/suggest` | feature relevance, redundant pairs, ready-to-apply suggestions |
| GET | `/sessions/{id}/curves/{model}` | ROC, PR, threshold sweep, confusion matrix |
| GET | `/sessions/{id}/residuals/{model}` | actual vs predicted, residual histogram |
| GET | `/sessions/{id}/importance/{model}` | permutation (+ native) importance |
| GET | `/sessions/{id}/shap/{model}` | SHAP summary, beeswarm, one-row explanation |
| GET | `/sessions/{id}/tuning/{model}` | Optuna trials and parameter importance |
| GET / PATCH / DELETE | `/runs`, `/runs/{id}` · GET / DELETE `/datasets` | saved runs and datasets |
| POST | `/compare` | align 2 to 8 runs |
| GET | `/sessions/{id}/export/{model}` · `/export/{model}/card` | model zip · model card as JSON |
| GET | `/lab/models` | model-lab catalog: models, controls, defaults, help text |
| POST | `/sessions/{id}/lab/{model}/fit` | fit one lab model |
| POST | `/sessions/{id}/lab/{model}/sweep` | validation curve, score grid or learning curve |

## Tests

```bash
cd backend && .venv/bin/python -m pytest      # 147 passed
cd frontend && npx vitest run                 # 47 passed (the live end-to-end test needs the backend on :8000, else it is skipped)
```

Backend: `test_correctness.py` (scores agree with predictions, class weighting in every path, positive class, target protection, ID
detection, interpolation, iterative refits), `test_export.py` (the exported `predict.py` reproduces the dashboard's predictions),
`test_lab.py` (every lab model, sweeps, tree class counts), plus API, training, analysis and unit tests. Frontend: unit helpers, the
SUMMARY recommendation logic, the FEATURES tab, the empty dashboard, the lab page, and a live end-to-end run (upload, feature selection,
impute, tuned training with cross-validation, every analysis tab). Running it adds a small `churn_small.csv` entry to the local run history.

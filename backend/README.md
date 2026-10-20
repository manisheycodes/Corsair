# AutoBuild backend MVP

API-first conversational AutoML for tabular classification, regression, and exploratory clustering. The backend accepts a plain-language problem, requires an explicit ML-spec approval, profiles CSV/XLSX data, diagnoses common leakage risks, engineers mixed-type features, tunes candidate pipelines, and supports task-aware evaluation.

## Run

```powershell
cd backend
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:AUTOBUILD_API_KEY = "replace-this-before-sharing"
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`; send `X-API-Key` with every endpoint other than `/health`.

## Frontend handoff

The frontend should render discovery, specification approval, data upload, data-readiness blockers, run progress, leaderboard, and report. Read the detailed contract in [docs/frontend-handoff.md](docs/frontend-handoff.md).

## MVP boundary

This is a private local MVP. It does not yet provide multi-user identities, Docker isolation, Redis jobs, PostgreSQL, live deployment, or external data connectors.

## AutoML scope

- Supported metrics: classification `f1`, `accuracy`, `roc_auc`; regression `mae`, `rmse`, `r2`.
- Classification candidates: logistic regression, decision tree, random forest, extra trees, AdaBoost, gradient boosting, histogram gradient boosting, support vector classifier, and k-nearest neighbors.
- Regression candidates: ridge, decision tree, random forest, extra trees, AdaBoost, gradient boosting, histogram gradient boosting, support vector regression, and k-nearest neighbors.
- Clustering candidates: K-means, agglomerative clustering, Gaussian mixture, DBSCAN, and BIRCH. Clustering has no target column; candidates are compared with silhouette and Davies–Bouldin metrics, DBSCAN noise is reported separately, and reports include per-cluster feature summaries and row-number assignments.
- Validation strategies: `auto`, `random`, `group`, and `time`. Group and time strategies require `group_column` or `time_column` during specification approval.
- Every candidate uses a bounded grid search. Search results include the chosen parameters and number of configurations evaluated.
- Feature preparation adds missing-value indicators, rare-category handling, datetime expansion, and identifier exclusion inside each cross-validation fold.
- Leakage diagnostics are conservative heuristics. They block exact target copies and warn about suspicious target-like names, identifiers, severe missingness, and conflicting duplicates.

Tuning and model ranking currently use the same validation splits. Treat the scores as development estimates and confirm a selected model against untouched data before production use.

Clustering is exploratory rather than predictive: internal scores are not ground truth, and results depend on feature scaling, distance geometry, and parameter choices. Agglomerative clustering and DBSCAN do not provide general-purpose labels for unseen records in this implementation.

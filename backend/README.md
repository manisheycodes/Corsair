# AutoBuild backend MVP

The revised full-application plan is in [docs/MASTER_PLAN.md](docs/MASTER_PLAN.md). Day 1 (2026-11-26) adds the research library that the autonomous workflow will use.

## Research library

Launch the API normally. Startup creates the SQLite research tables and imports missing reviewed seed cases without replacing existing edits. Three seed tutorials cover classification, regression and clustering. They are labelled tutorials; they are not claimed as field deployments.

- `GET /research/cases` lists records with citations, evidence and unknowns.
- `POST /research/cases` imports a JSON array of cases matching the OpenAPI schema. Repeating an import does not add duplicate records. Changed claims retain history.
- `GET /research/cases/{id}` returns the case and its revisions.
- `GET /research/search?q=handwritten%20digits&task_type=classification` retrieves cited cases and explains matched terms. Optional `limit` is 1–20.
- `POST /research/refresh` with `{ "force": false }` schedules tracked source checks; `GET /research/refresh` shows progress, timestamps and per-source errors.

These endpoints use the existing `X-API-Key`. Startup and hourly background checks refresh at most three tracked sources due after 24 hours. Set `AUTOBUILD_RESEARCH_REFRESH=false` for offline operation; stored cases remain usable. Page refresh checks title/hash/availability; new-case discovery and automatic scientific extraction are Day 2 work. Retrieval is currently lexical, with semantic retrieval planned alongside agents.

Run checks from `backend` with `..\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --basetemp=data\day1-pytest`. Tests use a dedicated local database and mocked source requests.

API-first conversational AutoML for tabular classification, regression, and exploratory clustering. The backend accepts a plain-language problem, requires an explicit ML-spec approval, profiles CSV/XLSX data, diagnoses common leakage risks, engineers mixed-type features, tunes candidate pipelines, and supports task-aware evaluation.

## Run

```powershell
cd backend
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:AUTOBUILD_API_KEY = "replace-this-before-sharing"
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`; send `X-API-Key` with every endpoint other than `/health`.

## Optional Codex advisor

Corsair can call a locally installed and authenticated Codex CLI to explain the current project, dataset profile, leaderboard, or model card. Check `GET /codex/status`, then send a question to `POST /projects/{project_id}/codex-advice`:

```json
{ "question": "Why did this model win, and what should I check next?" }
```

The integration invokes `codex exec` in read-only, non-interactive, ephemeral mode. It sends project metadata and summarized results, not raw uploaded rows. Install Codex and authenticate it with `codex login` before using the endpoint. Optional environment settings are `AUTOBUILD_CODEX_EXECUTABLE`, `AUTOBUILD_CODEX_MODEL`, and `AUTOBUILD_CODEX_TIMEOUT_SECONDS` (default `120`). The endpoint is synchronous for this local MVP.

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

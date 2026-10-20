# Frontend handoff

Send `X-API-Key` in local development. Render `spec.missing_information` as a guided discovery view, then require approval of task type, target column, and metric before upload. Render `spec.recommendations` as optional guidance cards; these are retrieved from reviewed local use cases, not guaranteed model choices.

After upload, show `profile`, `blockers`, and `training_ready`; link users to `GET /projects/{id}/data-template` when blocked. During approval, offer `validation_strategy` (`auto`, `random`, `group`, or `time`) for classification/regression and conditionally require `group_column` or `time_column`. Clustering does not require a target and uses internal `silhouette_score` (higher is better); explain that this is an exploratory quality estimate, not predictive accuracy. Its report result includes `cluster_profiles` (per-cluster median/mode feature summaries) and `cluster_assignments` (one-based source row numbers). Supported classification metrics are `f1`, `accuracy`, and `roc_auc`; regression supports `mae`, `rmse`, and `r2`. On run creation, poll `GET /runs/{run_id}` every 1–2 seconds until `completed` or `failed`. Render `result.leaderboard` in returned order and use `result.metric_direction` rather than hard-coded metric rules. Use `GET /projects/{id}/report` for the results screen.

Before showing the run button, call `GET /projects/{id}/run-plan`. Render its candidates and reasons as the planned experiment, and use `metric_direction` to explain how the leaderboard is judged. The endpoint returns `409` until the user has supplied enough information to identify classification, regression, or clustering.

Render `DatasetResponse.warnings` separately from blockers: warnings inform the user about data quality but do not prevent a run; blockers keep the project in `blocked` and must be resolved. The profile's `quality` object identifies usable, empty, and constant feature columns. Its `diagnostics` object contains leakage risks, likely identifiers, severe missingness, and conflicting duplicate patterns. Do not downplay a leakage blocker.

Use `GET /projects/{id}/runs` for run history. Each run exposes `artifact_available`; only when it is true should the UI offer the authenticated download at `GET /runs/{run_id}/artifact`.

For the MVP's real-time prediction screen, post up to 500 records to `POST /runs/{run_id}/predict` as `{ "records": [{...}] }`. Include every value from `run.result.features_used`; missing required columns return `422`. Classification returns probability maps keyed by class label when supported by the selected model.

For browser integration, configure the frontend origin in `AUTOBUILD_CORS_ORIGINS` (a comma-separated allow-list). The local defaults permit `http://localhost:3000` and `http://localhost:5173`. Send `X-API-Key` on every protected request. Use `/health` for liveness and `/ready` before declaring the API usable.

Render `report.model_card` as an explainability section: tuned parameters, validation method, data summary, limitations, and `feature_impact.top_features`. The run result also exposes feature-engineering details and search counts. Display the feature-impact scope statement verbatim or equivalently; it is exploratory and must not be presented as a causal explanation. Also display the search warning that tuning/ranking scores are not an untouched final evaluation.

While polling a run, also use `GET /runs/{run_id}/events` to render a durable execution timeline. Event status proceeds through `queued`, `running`, then `completed` or `failed`. A server restart marks an in-progress local run as failed; offer the user a clear retry action rather than indefinite loading.

Use `GET /projects/{id}/workflow` to drive the stepper, current instruction (`next_action`), and activity timeline (`events`). The endpoint is authoritative: do not infer a screen from the client alone. A valid CSV/XLSX upload before the project reaches `awaiting_data` receives `409`; after approval, prevent edits to the discovery conversation and start a new project for a material scope change.

Use `GET /projects/{id}/frontend-contract` for statuses and expected screens.

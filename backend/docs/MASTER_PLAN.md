# Corsair master implementation plan

Accepted scope: the complete local application requested by the user: autonomous ML/DL research, evidence retrieval, experiments, streaming simulation, bounded refinement, and a usable frontend. Production hosting and identities are deferred. This is not a reduced MVP. Keep implementation direct: one backend, SQLite, focused Python modules and a frontend; introduce extra infrastructure only when an actual requirement demands it.

## Architecture

Sources → case ingestion → cited case database → retrieval → problem analysis → experiment plan → real/referenced/synthetic data → isolated training → imperfect stream replay → metrics and logs → refinement → report.

Cases retain source URLs, evidence locators, dataset provenance, model choice, features, evaluation details, and explicit unknowns. Tutorial examples are labelled tutorials, not field deployments. Reported source results are separate from our own measured results. Different task types have separate metrics. Synthetic performance does not establish real-world accuracy.

## Five implementation days

| Day | Date | Deliverables | Acceptance |
|---|---|---|---|
| 1 | 2026-11-26 | Structured cases, SQLite store, provenance, deduplication, seed catalog, keyword retrieval, bounded source refresh, APIs | Cases survive restart; duplicate import is idempotent; search returns citations; refresh failures are recorded; offline startup works |
| 2 | 2026-11-27 | Problem analyst, retrieval-first researcher, targeted discovery, evidence-backed experiment planner | A situation without a dataset produces a cited plan with assumptions and compatible candidate tasks |
| 3 | 2026-11-28 | Uploaded/referenced/synthetic data paths, isolated experiment runner, ML and supported DL training, artifacts | Reproducible experiments with data version, seed, resource budget and failure logs |
| 4 | 2026-11-29 | Stream replay, missingness/noise/delay/drift scenarios, evaluation, bounded refinement | All attempts logged; task-specific comparisons; stop by budget, attempts or lack of improvement |
| 5 | 2026-11-30 | Complete frontend and unified project flow, cited reports, exports, setup walkthrough | Browser workflow from situation to cited research, data selection, training, stream simulation, refinement and report |

Dates describe planned work sessions; future days do not run automatically. These are implementation targets, not evidence of completed work. If a requirement takes longer, continue until it works rather than marking it complete as a demo. Support ML and DL through explicit task/model adapters, with clear supported modalities. Domain-specific simulation uses documented assumptions; exact physical realism cannot be inferred from a text description alone.

## Day 1 implementation contract

Use SQLite and lexical retrieval initially. Store curated case summaries with references and evidence sections. Startup seeds offline, then checks due sources in a background task; manual refresh uses the same coordinator. Fetch at most three sources per refresh, cap response size and timeout, retain hashes, timestamps and errors. Refresh source metadata without silently rewriting curated scientific claims. Novel web discovery and semantic extraction belong to Day 2.

Ten commit units: plan; schema; storage; provenance; deduplication; refresh state; cited seeds; retrieval; refresh/API lifecycle; tests and handoff. Preserve existing backend work in a separate baseline commit. Publish to a codex branch using normal Git; never overwrite unrelated edits or force-push existing history.

## Later implementation contracts

Day 2 agents use structured inputs/outputs and store evidence plus decision history. Research starts with retrieved cases and queries online only for missing/stale evidence. Day 3 datasets retain source/license/accessibility; generators record assumptions. Generated experiment code requires actual isolation and compute bounds. Day 4 distinguishes stream inference from online learning, scores only when labels are available, and reserves untouched evaluation data. Refinement uses declared metrics and records failed as well as successful attempts. Day 5 exposes these distinctions to users.

## Full application completion criteria

- User describes a situation with or without a dataset; the app extracts goals and records assumptions.
- Periodic research discovers relevant new cases, extracts cited model/data/feature details and retains provenance and unknowns. Retrieval reuses prior research and searches online for missing evidence.
- Orchestration proposes suitable ML/DL tasks and models, prepares uploaded or referenced data, or constructs a documented synthetic environment.
- Each compatible candidate is trained and evaluated in isolation. Data streams can contain noise, missingness, outliers, drift and label delays.
- Predictions, measured quality when ground truth exists, latency, failures, decisions and refinement attempts are inspectable. Experiments stop by declared criteria and can be replayed.
- The frontend presents research sources, data origins, assumptions, experiments, stream behavior, comparisons, refinement history and downloadable reports/artifacts.
- Tests exercise the complete workflow and failure recovery. No requirement is considered finished merely because an API stub exists.

Implementation preference: ordinary functions and direct modules; no separate initialization files, speculative abstractions, microservices or orchestration frameworks added for appearance. Existing LangGraph may be reused where it helps. Day 1 is only the research foundation of this full application.

## Research references

- https://proceedings.mlr.press/v267/trirat25a.html — AutoML-Agent research
- https://github.com/google/adk-samples — MLE-STAR implementation reference
- https://github.com/online-ml/river — streaming ML components
- https://github.com/openai/mle-bench — future evaluation reference

Inspect licenses and implementation fit before reusing code. These are component references; no verified exact end-to-end match was established.

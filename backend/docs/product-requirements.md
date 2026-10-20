# Product requirements

A non-technical user describes a prediction problem, approves a machine-readable project brief, supplies historical data or downloads a template, and receives a reproducible classification or regression comparison.

Required path: discovery → explicit approval → CSV/XLSX profile and leakage diagnostics → training-ready check → feature engineering and bounded hyperparameter search → task-aware validation → report. The MVP excludes connectors, web retrieval, forecasting, clustering, NLP, computer vision, multi-tenancy, and live model serving.

Approval may select automatic/random, group-aware, or time-aware validation. The system must reject incompatible metrics and invalid split columns before a run. Model-selection scores remain development estimates because search and ranking share validation folds; production acceptance requires untouched evaluation data.

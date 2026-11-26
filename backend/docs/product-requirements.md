# Product requirements

A non-technical user describes an analysis goal, approves a machine-readable project brief, supplies tabular data or downloads a template, and receives a reproducible classification, regression, or exploratory clustering comparison.

Required path: discovery → explicit approval → CSV/XLSX profile and leakage diagnostics → training-ready check → feature engineering and bounded hyperparameter search → supervised validation or internal clustering evaluation → report. The MVP excludes connectors, web retrieval, forecasting, NLP, computer vision, multi-tenancy, and live model serving.

Approval may select automatic/random, group-aware, or time-aware validation. The system must reject incompatible metrics and invalid split columns before a run. Model-selection scores remain development estimates because search and ranking share validation folds; production acceptance requires untouched evaluation data.

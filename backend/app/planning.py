from __future__ import annotations

from typing import Any

from .schemas import TaskType
from .validation import describe_validation, resolve_metric


_SUPERVISED_REASONS = {
    "logistic_regression": "Linear probabilistic baseline that is fast and useful as a reference.",
    "ridge": "Regularized linear baseline that is stable with correlated features.",
    "decision_tree": "Single interpretable tree that captures nonlinear rules.",
    "random_forest": "Bagged trees capture nonlinear relationships and interactions.",
    "extra_trees": "Highly randomized tree ensemble provides a diverse forest alternative.",
    "adaboost": "Boosting combines weak learners into a stronger predictor.",
    "gradient_boosting": "Stage-wise boosted trees provide a classic tabular boosting baseline.",
    "hist_gradient_boosting": "Histogram boosted trees provide an efficient boosting candidate.",
    "support_vector_machine": "Margin-based classifier with a nonlinear kernel for smaller datasets.",
    "support_vector_regression": "Margin-based regression with a nonlinear kernel for smaller datasets.",
    "k_nearest_neighbors": "Distance-based local predictions offer a nonparametric comparison.",
}

_CLUSTER_CANDIDATES = [
    ("kmeans", "Centroid-based baseline; searches a small range of cluster counts."),
    ("agglomerative", "Hierarchical merging can reveal nested groups without a centroid assumption."),
    ("gaussian_mixture", "Probabilistic soft clustering that models elliptical group structure."),
    ("dbscan", "Density-based grouping can find irregular shapes and label outliers as noise."),
    ("birch", "Incremental clustering designed to summarize larger datasets efficiently."),
]


_CANDIDATES = {
    TaskType.CLASSIFICATION.value: [
        ("logistic_regression", "Strong interpretable baseline for binary and multiclass outcomes."),
        ("decision_tree", "Single interpretable tree that captures nonlinear rules."),
        ("random_forest", "Captures nonlinear relationships and mixed feature effects."),
        ("extra_trees", "Highly randomized tree ensemble provides a diverse forest alternative."),
        ("adaboost", "Boosting combines weak learners into a stronger predictor."),
        ("gradient_boosting", "Stage-wise boosted trees provide a classic tabular boosting baseline."),
        ("hist_gradient_boosting", "Tests a boosted-tree candidate for structured tabular data."),
        ("support_vector_machine", "Margin-based nonlinear classifier for smaller datasets."),
        ("k_nearest_neighbors", "Distance-based local predictions offer a nonparametric comparison."),
    ],
    TaskType.REGRESSION.value: [
        ("ridge", "Regularized linear baseline that is stable with correlated features."),
        ("decision_tree", "Single interpretable tree that captures nonlinear rules."),
        ("random_forest", "Captures nonlinear relationships and feature interactions."),
        ("extra_trees", "Highly randomized tree ensemble provides a diverse forest alternative."),
        ("adaboost", "Boosting combines weak learners into a stronger predictor."),
        ("gradient_boosting", "Stage-wise boosted trees provide a classic tabular boosting baseline."),
        ("hist_gradient_boosting", "Tests a boosted-tree candidate for structured tabular data."),
        ("support_vector_regression", "Margin-based nonlinear regression for smaller datasets."),
        ("k_nearest_neighbors", "Distance-based local predictions offer a nonparametric comparison."),
    ],
}


def build_run_plan(spec: dict[str, Any], profile: dict[str, Any] | None = None) -> dict[str, Any]:
    task_type = spec.get("task_type")
    if task_type == TaskType.CLUSTERING.value:
        return {
            "task_type": task_type,
            "target_column": None,
            "primary_metric": "silhouette_score",
            "metric_direction": "higher_is_better",
            "validation": "internal clustering metrics on prepared features",
            "candidates": [{"name": name, "reason": reason} for name, reason in _CLUSTER_CANDIDATES],
            "reproducibility": {"random_seed": 42, "preprocessing": "imputation, rare-category encoding, datetime expansion, and scaling"},
            "dataset": {"rows": profile.get("rows"), "columns": profile.get("columns"), "features": profile.get("column_names", [])} if profile else None,
        }
    if task_type not in _CANDIDATES:
        raise ValueError("A classification or regression task is required before planning a run")
    metric, _, metric_direction = resolve_metric(task_type, spec.get("primary_metric"))
    candidates = [{"name": name, "reason": _SUPERVISED_REASONS[name]} for name, _ in _CANDIDATES[task_type]]
    plan = {
        "task_type": task_type,
        "target_column": spec.get("target_column"),
        "primary_metric": metric,
        "metric_direction": metric_direction,
        "validation": describe_validation(spec, task_type),
        "candidates": candidates,
        "reproducibility": {"random_seed": 42, "preprocessing": "imputation, one-hot encoding, and scaling inside each validation fold"},
    }
    if profile:
        plan["dataset"] = {"rows": profile.get("rows"), "columns": profile.get("columns"), "features": [column for column in profile.get("column_names", []) if column != spec.get("target_column")]}
    return plan

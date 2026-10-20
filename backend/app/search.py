from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge


@dataclass(frozen=True)
class SearchCandidate:
    estimator: BaseEstimator
    parameter_grid: dict[str, list[Any]]


def candidate_search_spaces(task_type: str) -> dict[str, SearchCandidate]:
    """Small, bounded search spaces suitable for the local MVP compute budget."""
    if task_type == "classification":
        return {
            "logistic_regression": SearchCandidate(
                LogisticRegression(max_iter=1000),
                {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]},
            ),
            "random_forest": SearchCandidate(
                RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=1),
                {"model__max_depth": [None, 8], "model__min_samples_leaf": [1, 3]},
            ),
            "hist_gradient_boosting": SearchCandidate(
                HistGradientBoostingClassifier(random_state=42),
                {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31]},
            ),
        }
    if task_type == "regression":
        return {
            "ridge": SearchCandidate(Ridge(), {"model__alpha": [0.1, 1.0, 10.0]}),
            "random_forest": SearchCandidate(
                RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=1),
                {"model__max_depth": [None, 8], "model__min_samples_leaf": [1, 3]},
            ),
            "hist_gradient_boosting": SearchCandidate(
                HistGradientBoostingRegressor(random_state=42),
                {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31]},
            ),
        }
    raise ValueError("Hyperparameter search supports classification and regression only")


def public_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {key.removeprefix("model__"): value for key, value in parameters.items()}

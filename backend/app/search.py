from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


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
                {"model__C": [0.1, 1.0]},
            ),
            "decision_tree": SearchCandidate(
                DecisionTreeClassifier(random_state=42),
                {"model__max_depth": [None, 8]},
            ),
            "random_forest": SearchCandidate(
                RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=1),
                {"model__max_depth": [None, 8]},
            ),
            "extra_trees": SearchCandidate(
                ExtraTreesClassifier(n_estimators=80, random_state=42, n_jobs=1),
                {"model__max_depth": [None, 8]},
            ),
            "adaboost": SearchCandidate(
                AdaBoostClassifier(random_state=42),
                {"model__n_estimators": [50], "model__learning_rate": [0.5, 1.0]},
            ),
            "gradient_boosting": SearchCandidate(
                GradientBoostingClassifier(random_state=42),
                {"model__n_estimators": [50], "model__max_depth": [1, 3]},
            ),
            "hist_gradient_boosting": SearchCandidate(
                HistGradientBoostingClassifier(random_state=42),
                {"model__learning_rate": [0.1], "model__max_leaf_nodes": [15, 31]},
            ),
            "support_vector_machine": SearchCandidate(
                SVC(probability=True, random_state=42),
                {"model__C": [0.5, 2.0], "model__kernel": ["rbf"]},
            ),
            "k_nearest_neighbors": SearchCandidate(
                KNeighborsClassifier(),
                {"model__n_neighbors": [3, 7], "model__weights": ["distance"]},
            ),
        }
    if task_type == "regression":
        return {
            "ridge": SearchCandidate(Ridge(), {"model__alpha": [0.1, 1.0, 10.0]}),
            "decision_tree": SearchCandidate(
                DecisionTreeRegressor(random_state=42),
                {"model__max_depth": [None, 8]},
            ),
            "random_forest": SearchCandidate(
                RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=1),
                {"model__max_depth": [None, 8]},
            ),
            "extra_trees": SearchCandidate(
                ExtraTreesRegressor(n_estimators=80, random_state=42, n_jobs=1),
                {"model__max_depth": [None, 8]},
            ),
            "adaboost": SearchCandidate(
                AdaBoostRegressor(random_state=42),
                {"model__n_estimators": [50], "model__learning_rate": [0.5, 1.0]},
            ),
            "gradient_boosting": SearchCandidate(
                GradientBoostingRegressor(random_state=42),
                {"model__n_estimators": [50], "model__max_depth": [1, 3]},
            ),
            "hist_gradient_boosting": SearchCandidate(
                HistGradientBoostingRegressor(random_state=42),
                {"model__learning_rate": [0.1], "model__max_leaf_nodes": [15, 31]},
            ),
            "support_vector_regression": SearchCandidate(
                SVR(), {"model__C": [0.5, 2.0], "model__kernel": ["rbf"]}
            ),
            "k_nearest_neighbors": SearchCandidate(
                KNeighborsRegressor(),
                {"model__n_neighbors": [3, 7], "model__weights": ["distance"]},
            ),
        }
    raise ValueError("Hyperparameter search supports classification and regression only")


def public_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {key.removeprefix("model__"): value for key, value in parameters.items()}

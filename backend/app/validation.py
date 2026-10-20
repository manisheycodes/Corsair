from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit


SUPPORTED_METRICS = {
    "classification": {
        "f1": ("f1_weighted", "higher_is_better"),
        "accuracy": ("accuracy", "higher_is_better"),
        "roc_auc": ("roc_auc_ovr_weighted", "higher_is_better"),
    },
    "regression": {
        "mae": ("neg_mean_absolute_error", "lower_is_better"),
        "rmse": ("neg_root_mean_squared_error", "lower_is_better"),
        "r2": ("r2", "higher_is_better"),
    },
}


@dataclass
class ValidationSetup:
    frame: pd.DataFrame
    cv: Any
    groups: pd.Series | None
    method: str
    folds: int
    excluded_features: list[str]


def resolve_metric(task_type: str, metric: str | None) -> tuple[str, str, str]:
    default = "f1" if task_type == "classification" else "mae"
    selected = metric or default
    supported = SUPPORTED_METRICS.get(task_type, {})
    if selected not in supported:
        raise ValueError(f"Metric '{selected}' is not supported for {task_type}; choose one of: {', '.join(supported)}")
    scoring, direction = supported[selected]
    return selected, scoring, direction


def build_validation_setup(frame: pd.DataFrame, target: str, task_type: str, spec: dict[str, Any]) -> ValidationSetup:
    strategy = spec.get("validation_strategy") or "auto"
    time_column = spec.get("time_column")
    group_column = spec.get("group_column")
    if strategy == "auto":
        strategy = "time" if time_column else "group" if group_column else "random"

    if strategy == "time":
        if not time_column or time_column not in frame.columns:
            raise ValueError("Time-aware validation requires a valid time_column")
        parsed = pd.to_datetime(frame[time_column], errors="coerce", utc=True)
        if parsed.isna().any():
            raise ValueError(f"Time column '{time_column}' contains missing or invalid timestamps")
        ordered = frame.assign(__validation_time=parsed).sort_values("__validation_time").drop(columns="__validation_time")
        folds = min(5, max(2, len(ordered) // 10))
        return ValidationSetup(ordered, TimeSeriesSplit(n_splits=folds), None, "time_series", folds, [])

    if strategy == "group":
        if not group_column or group_column not in frame.columns:
            raise ValueError("Group-aware validation requires a valid group_column")
        groups = frame[group_column]
        if groups.isna().any():
            raise ValueError(f"Group column '{group_column}' contains missing values")
        folds = min(5, int(groups.nunique()))
        if folds < 2:
            raise ValueError("Group-aware validation requires at least two distinct groups")
        return ValidationSetup(frame, GroupKFold(n_splits=folds), groups, "group_k_fold", folds, [group_column])

    if strategy != "random":
        raise ValueError("validation_strategy must be auto, random, group, or time")
    if task_type == "classification":
        folds = min(5, int(frame[target].value_counts().min()))
        return ValidationSetup(
            frame,
            StratifiedKFold(n_splits=folds, shuffle=True, random_state=42),
            None,
            "stratified_k_fold",
            folds,
            [],
        )
    folds = min(5, len(frame))
    return ValidationSetup(frame, KFold(n_splits=folds, shuffle=True, random_state=42), None, "k_fold", folds, [])


def describe_validation(spec: dict[str, Any], task_type: str) -> str:
    strategy = spec.get("validation_strategy") or "auto"
    if strategy == "time" or (strategy == "auto" and spec.get("time_column")):
        return f"time-ordered cross-validation using {spec.get('time_column')}"
    if strategy == "group" or (strategy == "auto" and spec.get("group_column")):
        return f"group-isolated cross-validation using {spec.get('group_column')}"
    return "stratified cross-validation" if task_type == "classification" else "shuffled cross-validation"

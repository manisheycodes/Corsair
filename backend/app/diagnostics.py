from __future__ import annotations

import re
from typing import Any

import pandas as pd


def analyze_data_risks(frame: pd.DataFrame, target: str | None) -> dict[str, Any]:
    """Return conservative, explainable data-quality and leakage signals."""
    feature_columns = [column for column in frame.columns if column != target]
    risks: list[dict[str, str]] = []
    identifiers: list[str] = []
    high_missing: list[str] = []

    if target not in frame.columns:
        return {
            "leakage_risks": risks,
            "identifier_candidates": identifiers,
            "high_missing_features": high_missing,
            "conflicting_duplicate_feature_rows": 0,
        }

    labeled = frame.dropna(subset=[target])
    target_key = _name_key(target)
    for column in feature_columns:
        series = labeled[column]
        non_missing = series.notna()
        if len(frame) and frame[column].isna().mean() >= 0.80:
            high_missing.append(column)

        unique_ratio = frame[column].nunique(dropna=True) / max(int(frame[column].notna().sum()), 1)
        if unique_ratio >= 0.98 and (_looks_like_identifier(column) or not pd.api.types.is_numeric_dtype(frame[column])):
            identifiers.append(column)

        if non_missing.any() and _equal_values(series[non_missing], labeled.loc[non_missing, target]):
            risks.append(
                {
                    "column": column,
                    "severity": "blocker",
                    "reason": "This feature duplicates the target on every labeled row.",
                }
            )
            continue

        column_key = _name_key(column)
        if target_key and (target_key in column_key or column_key in target_key) and column_key != target_key:
            risks.append(
                {
                    "column": column,
                    "severity": "warning",
                    "reason": "The feature name closely resembles the target and may be a post-outcome proxy.",
                }
            )

    conflicts = 0
    if feature_columns and not labeled.empty:
        conflicts = int((labeled.groupby(feature_columns, dropna=False)[target].nunique() > 1).sum())

    return {
        "leakage_risks": risks,
        "identifier_candidates": identifiers,
        "high_missing_features": high_missing,
        "conflicting_duplicate_feature_rows": conflicts,
    }


def _equal_values(left: pd.Series, right: pd.Series) -> bool:
    return left.astype("string").reset_index(drop=True).equals(right.astype("string").reset_index(drop=True))


def _name_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _looks_like_identifier(column: str) -> bool:
    key = _name_key(column)
    return key == "id" or key.endswith("id") or any(token in key for token in ("uuid", "guid", "identifier"))

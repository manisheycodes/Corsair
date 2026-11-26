from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .diagnostics import analyze_data_risks


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """Convert datetime columns into stable numeric calendar and recency signals."""

    def fit(self, values, y=None):
        return self

    def transform(self, values):
        frame = pd.DataFrame(values).copy()
        output: list[np.ndarray] = []
        for column in frame.columns:
            parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
            output.extend(
                [
                    parsed.dt.year.to_numpy(dtype=float),
                    parsed.dt.month.to_numpy(dtype=float),
                    parsed.dt.day.to_numpy(dtype=float),
                    parsed.dt.dayofweek.to_numpy(dtype=float),
                    (parsed.astype("int64", errors="ignore") / 1_000_000_000).where(parsed.notna()).to_numpy(dtype=float),
                ]
            )
        return np.column_stack(output) if output else np.empty((len(frame), 0))


def prepare_features(
    frame: pd.DataFrame, target: str | None, exclude_columns: list[str] | None = None
) -> tuple[pd.DataFrame, ColumnTransformer, dict[str, Any]]:
    raw = frame.drop(columns=[target], errors="ignore") if target else frame.copy()
    diagnostics = analyze_data_risks(frame, target)
    excluded = list(dict.fromkeys([*diagnostics["identifier_candidates"], *(exclude_columns or [])]))
    features = raw.drop(columns=excluded, errors="ignore")
    datetime_columns = [column for column in features.columns if _is_datetime_feature(features[column], column)]
    numeric_columns = [
        column for column in features.select_dtypes(include="number").columns if column not in datetime_columns
    ]
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns and column not in datetime_columns
    ]

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent", add_indicator=True)),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )
    if datetime_columns:
        transformers.append(
            (
                "datetime",
                Pipeline(
                    [
                        ("expand", DateTimeFeatures()),
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                datetime_columns,
            )
        )

    if not transformers:
        raise ValueError("No usable features remain after excluding identifier and leakage-prone columns")
    summary = {
        "numeric": numeric_columns,
        "categorical": categorical_columns,
        "datetime": datetime_columns,
        "excluded_identifiers": excluded,
        "capabilities": [
            "numeric median imputation with missingness indicators",
            "rare-category grouping and one-hot encoding",
            "datetime calendar and recency expansion",
        ],
    }
    return features, ColumnTransformer(transformers, remainder="drop"), summary


def _is_datetime_feature(series: pd.Series, column: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    name_hint = any(token in str(column).lower() for token in ("date", "time", "timestamp"))
    if not name_hint or pd.api.types.is_numeric_dtype(series):
        return False
    non_missing = series.dropna()
    if non_missing.empty:
        return False
    return bool(pd.to_datetime(non_missing, errors="coerce", utc=True).notna().mean() >= 0.80)

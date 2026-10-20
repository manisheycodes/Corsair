from __future__ import annotations

import pandas as pd
from pydantic import ValidationError

from app.diagnostics import analyze_data_risks
from app.feature_engineering import prepare_features
from app.ml import profile_dataset
from app.schemas import ApprovalRequest
from app.search import candidate_search_spaces
from app.validation import build_validation_setup, resolve_metric


def test_diagnostics_find_exact_target_copy_and_identifier():
    frame = pd.DataFrame(
        {
            "customer_id": [f"C-{index}" for index in range(30)],
            "approved_after_review": [0, 1] * 15,
            "approved": [0, 1] * 15,
        }
    )

    diagnostics = analyze_data_risks(frame, "approved")

    assert diagnostics["identifier_candidates"] == ["customer_id"]
    assert diagnostics["leakage_risks"][0]["column"] == "approved_after_review"
    assert diagnostics["leakage_risks"][0]["severity"] == "blocker"


def test_profile_blocks_an_exact_target_copy(tmp_path):
    path = tmp_path / "leaky.csv"
    pd.DataFrame(
        {"age": range(30), "outcome_copy": [0, 1] * 15, "outcome": [0, 1] * 15}
    ).to_csv(path, index=False)

    profile, blockers = profile_dataset(path, "outcome", "classification")

    assert profile["diagnostics"]["leakage_risks"]
    assert any("target leakage" in blocker.lower() for blocker in blockers)


def test_feature_engineering_expands_dates_and_excludes_identifiers():
    frame = pd.DataFrame(
        {
            "customer_id": [f"C-{index}" for index in range(30)],
            "signup_date": pd.date_range("2025-01-01", periods=30).astype(str),
            "income": [None if index % 7 == 0 else 1000 + index for index in range(30)],
            "region": ["north", "south", "east"] * 10,
            "churn": [0, 1] * 15,
        }
    )

    features, transformer, summary = prepare_features(frame, "churn")
    transformed = transformer.fit_transform(features, frame["churn"])

    assert "customer_id" not in features
    assert summary["datetime"] == ["signup_date"]
    assert summary["excluded_identifiers"] == ["customer_id"]
    assert transformed.shape[0] == 30
    assert transformed.shape[1] > len(features.columns)


def test_validation_selects_time_and_group_aware_splits():
    frame = pd.DataFrame(
        {
            "event_time": pd.date_range("2025-01-01", periods=30)[::-1],
            "account": [f"A-{index // 5}" for index in range(30)],
            "value": range(30),
            "target": [0, 1] * 15,
        }
    )

    temporal = build_validation_setup(
        frame, "target", "classification", {"validation_strategy": "time", "time_column": "event_time"}
    )
    grouped = build_validation_setup(
        frame, "target", "classification", {"validation_strategy": "group", "group_column": "account"}
    )

    assert temporal.method == "time_series"
    assert temporal.frame.iloc[0]["event_time"] < temporal.frame.iloc[-1]["event_time"]
    assert grouped.method == "group_k_fold"
    assert grouped.groups is not None
    assert grouped.excluded_features == ["account"]


def test_metric_resolution_rejects_mismatched_metrics():
    assert resolve_metric("regression", "rmse")[1] == "neg_root_mean_squared_error"
    try:
        resolve_metric("classification", "mae")
    except ValueError as exc:
        assert "not supported" in str(exc)
    else:
        raise AssertionError("classification must reject regression metrics")


def test_approval_rejects_missing_split_columns_and_wrong_metric():
    try:
        ApprovalRequest(task_type="classification", target_column="target", primary_metric="mae")
    except ValidationError as exc:
        assert "not supported" in str(exc)
    else:
        raise AssertionError("approval must reject a metric from another task type")

    try:
        ApprovalRequest(task_type="regression", target_column="target", validation_strategy="time")
    except ValidationError as exc:
        assert "time_column is required" in str(exc)
    else:
        raise AssertionError("time validation must require a time column")


def test_each_candidate_has_a_bounded_hyperparameter_search_space():
    classification = candidate_search_spaces("classification")
    regression = candidate_search_spaces("regression")

    assert set(classification) == {"logistic_regression", "random_forest", "hist_gradient_boosting"}
    assert set(regression) == {"ridge", "random_forest", "hist_gradient_boosting"}
    assert all(candidate.parameter_grid for candidate in [*classification.values(), *regression.values()])
    assert classification["logistic_regression"].parameter_grid["model__C"] == [0.1, 1.0, 10.0]

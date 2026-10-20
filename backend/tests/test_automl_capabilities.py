from __future__ import annotations

import pandas as pd

from app.diagnostics import analyze_data_risks
from app.feature_engineering import prepare_features
from app.ml import profile_dataset


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

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .diagnostics import analyze_data_risks
from .clustering import cluster_search_spaces, evaluate_clustering, iter_cluster_candidates, summarize_clusters
from .feature_engineering import prepare_features
from .search import candidate_search_spaces, public_parameters
from .validation import build_validation_setup, resolve_metric


def profile_dataset(path: Path, target: str | None, task_type: str | None, spec: dict | None = None) -> tuple[dict, list[str]]:
    frame = read_table(path)
    blockers: list[str] = []
    warnings: list[str] = []
    if frame.empty:
        blockers.append("The file contains no rows.")
    if task_type == "clustering":
        if target:
            blockers.append("Clustering does not use a target column; leave the target unset.")
    elif not target:
        blockers.append("Approve a project specification with a target column before training.")
    elif target not in frame.columns:
        blockers.append(f"Target column '{target}' is not present in the imported file.")
    elif task_type not in {"classification", "regression"}:
        blockers.append("Approve whether this is a classification or regression project before training.")
    else:
        labeled_target = frame[target].dropna()
        if labeled_target.nunique() < 2:
            blockers.append("The target must contain at least two distinct non-empty values.")
        if task_type == "classification" and not labeled_target.empty and labeled_target.value_counts().min() < 2:
            blockers.append("Each target class needs at least two labeled rows for cross-validation.")
        if task_type == "regression" and not pd.api.types.is_numeric_dtype(labeled_target):
            blockers.append("A regression target must be numeric. Use classification for category labels.")
        missing_target = int(frame[target].isna().sum())
        if missing_target:
            warnings.append(f"{missing_target} row(s) with a missing target will be excluded from training.")
    if len(frame) < 30:
        blockers.append("At least 30 rows are required for this MVP to evaluate models safely.")
    feature_columns = [column for column in frame.columns if column != target]
    empty_features = [column for column in feature_columns if frame[column].dropna().empty]
    constant_features = [column for column in feature_columns if column not in empty_features and frame[column].dropna().nunique() <= 1]
    usable_features = [column for column in feature_columns if column not in empty_features and column not in constant_features]
    if not feature_columns:
        blockers.append("Add at least one feature column in addition to the target column.")
    elif not usable_features:
        blockers.append("Add at least one non-empty feature column with more than one value.")
    if empty_features:
        warnings.append(f"{len(empty_features)} fully empty feature column(s) will not provide predictive signal: {', '.join(empty_features[:5])}.")
    if constant_features:
        warnings.append(f"{len(constant_features)} constant feature column(s) will not provide predictive signal: {', '.join(constant_features[:5])}.")
    duplicate_rows = int(frame.duplicated().sum())
    if duplicate_rows:
        warnings.append(f"{duplicate_rows} duplicate row(s) were detected; review whether repeated records are expected.")
    diagnostics = analyze_data_risks(frame, target)
    for risk in diagnostics["leakage_risks"]:
        message = f"Potential target leakage in '{risk['column']}': {risk['reason']}"
        (blockers if risk["severity"] == "blocker" else warnings).append(message)
    if diagnostics["identifier_candidates"]:
        warnings.append(
            "Possible identifier column(s) may encourage memorization and should normally be excluded: "
            + ", ".join(diagnostics["identifier_candidates"][:5])
            + "."
        )
    if diagnostics["high_missing_features"]:
        warnings.append(
            "Feature column(s) with at least 80% missing values need review: "
            + ", ".join(diagnostics["high_missing_features"][:5])
            + "."
        )
    if diagnostics["conflicting_duplicate_feature_rows"]:
        warnings.append(
            f"{diagnostics['conflicting_duplicate_feature_rows']} duplicate feature pattern(s) have conflicting target values."
        )
    if spec and target in frame.columns and task_type in {"classification", "regression"}:
        try:
            build_validation_setup(frame.dropna(subset=[target]), target, task_type, spec)
        except ValueError as exc:
            blockers.append(f"Validation configuration is not ready: {exc}")
    profile = {
        "rows": len(frame), "columns": len(frame.columns), "column_names": frame.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values": frame.isna().sum().to_dict(), "duplicate_rows": duplicate_rows,
        "target": target, "task_type": task_type,
        "target_distribution": frame[target].value_counts(dropna=False).head(20).to_dict() if target and target in frame else {},
        "quality": {"feature_columns": feature_columns, "usable_feature_columns": usable_features, "empty_feature_columns": empty_features, "constant_feature_columns": constant_features},
        "warnings": warnings, "diagnostics": diagnostics,
    }
    return profile, blockers


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path)


def train(path: Path, target: str | None, task_type: str, metric: str | None, artifact_path: Path, spec: dict | None = None) -> dict:
    frame = read_table(path).copy()
    if task_type == "clustering":
        return _train_clustering(frame, artifact_path)
    if not target:
        raise ValueError("A target column is required for classification or regression")
    frame = frame.dropna(subset=[target])
    validation = build_validation_setup(frame, target, task_type, spec or {})
    frame = validation.frame
    x, preprocess, feature_engineering = prepare_features(frame, target, validation.excluded_features)
    y = frame[target]
    candidates = candidate_search_spaces(task_type)
    metric, scoring, metric_direction = resolve_metric(task_type, metric)
    cv = validation.cv
    results = []
    best_name, best_pipeline, best_score = "", None, float("-inf")
    for name, candidate in candidates.items():
        pipeline = Pipeline([("preprocess", preprocess), ("model", candidate.estimator)])
        search = GridSearchCV(
            pipeline,
            candidate.parameter_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=1,
            refit=True,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(x, y, groups=validation.groups)
        score = float(search.best_score_)
        if metric_direction == "lower_is_better":
            score = -score
        best_index = int(search.best_index_)
        results.append(
            {
                "model": name,
                "metric": metric,
                "score": score,
                "fold_std": float(search.cv_results_["std_test_score"][best_index]),
                "best_parameters": public_parameters(search.best_params_),
                "configurations_evaluated": len(search.cv_results_["params"]),
            }
        )
        comparable = score if metric_direction == "higher_is_better" else -score
        if comparable > best_score:
            best_name, best_pipeline, best_score = name, search.best_estimator_, comparable
    winning_result = next(result for result in results if result["model"] == best_name)
    model_card = _build_model_card(best_name, task_type, metric, validation, scoring, best_pipeline, x, y, winning_result)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, artifact_path)
    results.sort(key=lambda item: item["score"], reverse=metric_direction == "higher_is_better")
    return {"primary_metric": metric, "metric_direction": metric_direction, "leaderboard": results, "best_model": best_name, "best_parameters": winning_result["best_parameters"], "artifact": str(artifact_path.name), "rows_used": len(frame), "features_used": x.columns.tolist(), "feature_engineering": feature_engineering, "validation": {"method": validation.method, "folds": validation.folds}, "search": {"method": "exhaustive_grid_search", "total_configurations_evaluated": sum(result["configurations_evaluated"] for result in results), "selection_note": "Scores used for tuning and ranking share the same cross-validation splits; use an external test set for an unbiased final estimate."}, "random_seed": 42, "model_card": model_card}


def _train_clustering(frame: pd.DataFrame, artifact_path: Path) -> dict:
    x, preprocess, feature_engineering = prepare_features(frame, None)
    if len(frame) < 3:
        raise ValueError("At least 3 rows are needed to form and evaluate clusters")
    best_by_model: dict[str, dict] = {}
    best_labels_by_model: dict[str, object] = {}
    for name, estimator, parameters in iter_cluster_candidates():
        pipeline = Pipeline([("preprocess", preprocess), ("model", estimator)])
        try:
            labels = pipeline.fit_predict(x)
            metrics = evaluate_clustering(pipeline.named_steps["preprocess"].transform(x), labels)
        except (ValueError, FloatingPointError):
            continue
        score = metrics["silhouette_score"]
        if score is None:
            continue
        result = {
            "model": name,
            "metric": "silhouette_score",
            "score": score,
            "best_parameters": parameters,
            "metrics": metrics,
        }
        if name not in best_by_model or score > best_by_model[name]["score"]:
            best_by_model[name] = {**result, "pipeline": pipeline}
            best_labels_by_model[name] = labels
    if not best_by_model:
        raise ValueError("No clustering configuration produced at least two usable clusters; review features or use fewer clusters")
    results = sorted(best_by_model.values(), key=lambda item: item["score"], reverse=True)
    winner = results[0]
    cluster_profiles, cluster_assignments = summarize_clusters(x, best_labels_by_model[winner["model"]])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner["pipeline"], artifact_path)
    leaderboard = [{key: value for key, value in item.items() if key != "pipeline"} for item in results]
    configurations = sum(len(values) for values in cluster_search_spaces().values())
    model_card = {
        "model": winner["model"],
        "task_type": "clustering",
        "primary_metric": "silhouette_score",
        "validation": {"method": "internal_cluster_quality", "scoring": "silhouette_score", "random_seed": 42},
        "data_summary": {"rows_used": len(frame), "features_used": len(x.columns)},
        "limitations": [
            "Cluster quality scores are internal structure measures, not predictive accuracy.",
            "Cluster labels have no inherent business meaning; review cluster profiles with domain knowledge.",
            "Silhouette scoring is sampled for datasets larger than 2,000 rows.",
        ],
    }
    return {
        "task_type": "clustering",
        "primary_metric": "silhouette_score",
        "metric_direction": "higher_is_better",
        "leaderboard": leaderboard,
        "best_model": winner["model"],
        "best_parameters": winner["best_parameters"],
        "cluster_summary": winner["metrics"],
        "cluster_profiles": cluster_profiles,
        "cluster_assignments": cluster_assignments,
        "inference_available": winner["model"] in {"kmeans", "gaussian_mixture", "birch"},
        "artifact": artifact_path.name,
        "rows_used": len(frame),
        "features_used": x.columns.tolist(),
        "feature_engineering": feature_engineering,
        "validation": {"method": "internal_cluster_quality"},
        "search": {"method": "bounded_parameter_search", "total_configurations_evaluated": configurations},
        "random_seed": 42,
        "model_card": model_card,
    }


def _build_model_card(best_name: str, task_type: str, metric: str, validation, scoring: str, pipeline: Pipeline, x: pd.DataFrame, y: pd.Series, winning_result: dict) -> dict:
    impact = {"method": "permutation_importance_on_training_data", "scope": "Exploratory only: impacts are calculated after fitting on the same historical data and are not causal or a substitute for validation.", "top_features": []}
    try:
        importance = permutation_importance(pipeline, x, y, scoring=scoring, n_repeats=3, random_state=42, n_jobs=1)
        ranked = sorted(
            ({"feature": feature, "importance_mean": float(mean), "importance_std": float(std)} for feature, mean, std in zip(x.columns, importance.importances_mean, importance.importances_std)),
            key=lambda item: item["importance_mean"], reverse=True,
        )
        impact["top_features"] = ranked[:10]
    except Exception:
        impact["unavailable_reason"] = "Feature impact could not be calculated for this dataset and fitted pipeline."
    return {
        "model": best_name,
        "best_parameters": winning_result["best_parameters"],
        "task_type": task_type,
        "primary_metric": metric,
        "validation": {"method": validation.method, "folds": validation.folds, "scoring": scoring, "random_seed": 42},
        "data_summary": {"rows_used": len(x), "features_used": len(x.columns), "target_values": int(y.nunique())},
        "feature_impact": impact,
        "limitations": [
            "Cross-validation estimates may not match future production performance.",
            "The same cross-validation splits are used for tuning and model ranking; confirm performance on untouched data.",
            "Feature impact is directional, not causal, and can be affected by correlated fields.",
            "This MVP does not perform fairness, privacy, drift, or live-serving assessments.",
        ],
    }

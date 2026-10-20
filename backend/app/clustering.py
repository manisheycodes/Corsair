from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, Birch, DBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline


_CLUSTER_SPACES = {
    "kmeans": (KMeans, {"n_clusters": [2, 3, 4], "n_init": [10], "random_state": [42]}),
    "agglomerative": (AgglomerativeClustering, {"n_clusters": [2, 3], "linkage": ["ward"]}),
    "gaussian_mixture": (GaussianMixture, {"n_components": [2, 3], "covariance_type": ["diag", "full"], "random_state": [42]}),
    "dbscan": (DBSCAN, {"eps": [0.5, 1.0], "min_samples": [5]}),
    "birch": (Birch, {"n_clusters": [2], "threshold": [0.3, 0.5]}),
}


def cluster_search_spaces() -> dict[str, list[dict[str, Any]]]:
    return {name: list(ParameterGrid(parameters)) for name, (_, parameters) in _CLUSTER_SPACES.items()}


def make_cluster_pipeline(name: str, parameters: dict[str, Any], preprocess) -> Pipeline:
    estimator_type, _ = _CLUSTER_SPACES[name]
    return Pipeline([("preprocess", preprocess), ("model", estimator_type(**parameters))])


def evaluate_clustering(x, labels) -> dict[str, Any]:
    labels = np.asarray(labels)
    unique = np.unique(labels)
    noise_mask = labels != -1
    scored_x = x[noise_mask] if np.any(~noise_mask) else x
    scored_labels = labels[noise_mask] if np.any(~noise_mask) else labels
    scored_unique = np.unique(scored_labels)
    if len(scored_unique) < 2 or len(scored_unique) >= len(scored_labels):
        silhouette = None
        davies_bouldin = None
    else:
        sample_size = min(2000, len(scored_labels))
        silhouette = float(
            silhouette_score(scored_x, scored_labels, sample_size=sample_size, random_state=42)
        )
        davies_bouldin = float(davies_bouldin_score(scored_x, scored_labels)) if len(scored_labels) <= 10000 else None
    counts = {str(label.item() if hasattr(label, "item") else label): int(count) for label, count in zip(*np.unique(labels, return_counts=True))}
    return {
        "silhouette_score": silhouette,
        "davies_bouldin_score": davies_bouldin,
        "clusters_found": int(len(unique) - (1 if -1 in unique else 0)),
        "noise_points": int(np.sum(labels == -1)),
        "cluster_sizes": counts,
        "evaluated_points": int(len(scored_labels)),
    }


def summarize_clusters(frame, labels) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create small, readable cluster profiles and row-number assignments."""
    labels = np.asarray(labels)
    profiles = []
    for label in sorted(np.unique(labels), key=lambda value: (value == -1, value)):
        members = frame.loc[labels == label]
        feature_summary = {}
        for column in frame.columns:
            series = members[column].dropna()
            if series.empty:
                feature_summary[column] = None
            elif pd.api.types.is_numeric_dtype(series):
                feature_summary[column] = _json_scalar(series.median())
            else:
                mode = series.astype(str).mode()
                feature_summary[column] = mode.iloc[0] if not mode.empty else None
        profiles.append({
            "cluster": _json_scalar(label),
            "size": int(len(members)),
            "features": feature_summary,
        })
    assignments = [
        {"row_number": index + 1, "cluster": _json_scalar(label)}
        for index, label in enumerate(labels)
    ]
    return profiles, assignments


def _json_scalar(value):
    return value.item() if hasattr(value, "item") else value


def iter_cluster_candidates():
    for name, (estimator_type, parameters) in _CLUSTER_SPACES.items():
        for configuration in ParameterGrid(parameters):
            yield name, estimator_type(**configuration), configuration

"""Tests for the model-comparison engine (Milestone 7).

Everything runs offline on small deterministic frames. The engine must
reuse the ML layer without duplicating it, share one split across
models, isolate per-model failures, and refuse to rank invalid
evaluations or undefined metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.comparison import (
    METRIC_DIRECTIONS,
    PRIMARY_METRIC,
    TASK_METRICS,
    best_by_metric,
    compare_models,
    prepare_shared_split,
)
from core.models import ModelError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sales_frame() -> pd.DataFrame:
    """Small labelled frame: 40 rows, numeric + categorical columns."""
    rng = np.random.default_rng(42)
    n = 40
    return pd.DataFrame(
        {
            "sales": rng.uniform(100, 1000, n).round(2),
            "profit": rng.uniform(10, 200, n).round(2),
            "units": rng.integers(1, 50, n),
            "region": rng.choice(["North", "South", "East"], n),
            "returned": rng.choice(["yes", "no"], n),
        }
    )


# ---------------------------------------------------------------------------
# Metric directions and selection rules
# ---------------------------------------------------------------------------


def test_metric_directions_match_the_spec():
    for metric in ("accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "r2", "silhouette"):
        assert METRIC_DIRECTIONS[metric] == "higher"
    for metric in ("mae", "rmse", "mse"):
        assert METRIC_DIRECTIONS[metric] == "lower"


def test_primary_metric_per_task():
    assert PRIMARY_METRIC == {
        "classification": "f1_weighted",
        "regression": "rmse",
        "clustering": "silhouette",
    }


def test_table_metrics_match_the_spec():
    assert TASK_METRICS["classification"] == [
        "accuracy", "precision_weighted", "recall_weighted", "f1_weighted",
    ]
    assert TASK_METRICS["regression"] == ["mae", "rmse", "r2"]
    assert TASK_METRICS["clustering"] == ["silhouette"]


def test_best_by_metric_honours_direction():
    assert best_by_metric({"a": 0.7, "b": 0.9}, "accuracy") == "b"   # higher wins
    assert best_by_metric({"a": 3.0, "b": 1.0}, "rmse") == "b"       # lower wins


def test_best_by_metric_ignores_none_and_non_finite():
    assert best_by_metric({"a": None, "b": 0.5}, "accuracy") == "b"
    assert best_by_metric({"a": float("nan"), "b": 0.5}, "r2") == "b"
    assert best_by_metric({"a": None, "b": float("nan")}, "accuracy") is None
    assert best_by_metric({}, "accuracy") is None


def test_best_by_metric_ties_go_to_the_first_candidate():
    assert best_by_metric({"a": 0.5, "b": 0.5}, "accuracy") == "a"


def test_best_by_metric_rejects_unknown_metrics():
    assert best_by_metric({"a": 1.0}, "not_a_metric") is None


# ---------------------------------------------------------------------------
# Happy paths, per task
# ---------------------------------------------------------------------------


def test_regression_comparison_reports_spec_metrics_and_a_best(sales_frame):
    result = compare_models(
        sales_frame,
        "regression",
        ["Linear Regression", "Random Forest", "Decision Tree"],
        ["units", "profit", "region"],
        "sales",
    )
    assert result.metrics == ["mae", "rmse", "r2"]
    assert result.primary_metric == "rmse"
    assert [o.model_name for o in result.outcomes] == result.model_names
    assert all(o.ok for o in result.outcomes)
    for row in result.table_rows():
        assert set(row["metrics"]) == {"mae", "rmse", "r2"}
        assert all(v is None or v >= 0 for k, v in row["metrics"].items() if k != "r2")
    # the best model is the one with the lowest finite RMSE
    rmses = {o.model_name: o.result.metrics["rmse"] for o in result.outcomes}
    assert result.best_model == min(rmses, key=rmses.get)


def test_classification_comparison_reports_spec_metrics(sales_frame):
    result = compare_models(
        sales_frame,
        "classification",
        ["Logistic Regression", "Random Forest"],
        ["units", "region"],
        "returned",
    )
    assert result.metrics == ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    assert result.primary_metric == "f1_weighted"
    assert all(o.ok for o in result.outcomes)
    assert result.best_model in ("Logistic Regression", "Random Forest")


def test_clustering_comparison_reports_silhouette_only(sales_frame):
    result = compare_models(
        sales_frame,
        "clustering",
        ["K-Means", "MiniBatch K-Means", "Agglomerative Clustering"],
        ["sales", "profit", "units"],
        n_clusters=3,
    )
    assert result.metrics == ["silhouette"]
    assert all(o.ok for o in result.outcomes)
    for row in result.table_rows():
        assert set(row["metrics"]) == {"silhouette"}
        assert row["metrics"]["silhouette"] is not None
    assert result.best_model in result.model_names


# ---------------------------------------------------------------------------
# Shared split and determinism
# ---------------------------------------------------------------------------


def test_every_model_is_scored_on_the_same_test_rows(sales_frame):
    result = compare_models(
        sales_frame,
        "regression",
        ["Linear Regression", "Random Forest", "Decision Tree"],
        ["units", "profit"],
        "sales",
    )
    indices = [run.X_test.index for run in result.runs.values()]
    assert all(index.equals(indices[0]) for index in indices[1:])
    train_indices = [run.X_train.index for run in result.runs.values()]
    assert all(index.equals(train_indices[0]) for index in train_indices[1:])
    assert result.n_train == len(train_indices[0])
    assert result.n_test == len(indices[0])
    assert result.n_train + result.n_test == len(sales_frame)


def test_prepared_split_is_reproducible(sales_frame):
    first = prepare_shared_split(sales_frame, ["units", "region"], "sales", 0.2, 42)
    second = prepare_shared_split(sales_frame, ["units", "region"], "sales", 0.2, 42)
    assert first.train_index.equals(second.train_index)
    assert first.test_index.equals(second.test_index)
    assert first.X_train.equals(second.X_train)
    assert first.X_test.equals(second.X_test)


def test_prepared_split_aligns_features_and_target(sales_frame):
    split = prepare_shared_split(sales_frame, ["units", "region"], "sales", 0.3, 42)
    assert split.X_train.index.equals(split.y_train.index)
    assert split.X_test.index.equals(split.y_test.index)
    assert split.X_train.isna().sum().sum() == 0
    assert split.X_test.isna().sum().sum() == 0
    assert split.y_train.isna().sum() == 0
    assert split.y_test.isna().sum() == 0
    assert split.dropped_rows == 0


def test_comparison_is_deterministic(sales_frame):
    models = ["Linear Regression", "Random Forest"]
    first = compare_models(sales_frame, "regression", models, ["units", "profit"], "sales")
    second = compare_models(sales_frame, "regression", models, ["units", "profit"], "sales")
    assert [o.result.metrics for o in first.outcomes] == [o.result.metrics for o in second.outcomes]
    assert first.best_model == second.best_model


# ---------------------------------------------------------------------------
# Per-model failure isolation
# ---------------------------------------------------------------------------


def test_unknown_model_fails_alone_and_the_rest_still_run(sales_frame):
    result = compare_models(
        sales_frame,
        "regression",
        ["Linear Regression", "No Such Model"],
        ["units", "profit"],
        "sales",
    )
    ok = {o.model_name for o in result.successful}
    failed = {o.model_name for o in result.failed}
    assert ok == {"Linear Regression"}
    assert failed == {"No Such Model"}
    assert "Unknown model" in result.failed[0].error
    assert result.best_model == "Linear Regression"


def test_failed_outcomes_carry_no_metric_values(sales_frame):
    result = compare_models(
        sales_frame,
        "regression",
        ["Bogus One", "Bogus Two"],
        ["units", "profit"],
        "sales",
    )
    assert result.best_model is None
    assert all(not row["ok"] for row in result.table_rows())
    assert all(
        all(value is None for value in row["metrics"].values())
        for row in result.table_rows()
    )
    assert any("No model completed successfully" in w for w in result.warnings)


def test_degenerate_single_class_evaluation_suppresses_the_best_highlight(sales_frame):
    # Logistic Regression cannot train on a single-class training set,
    # Decision Tree can — and neither gets a "best" crown, because the
    # held-out metrics would be misleading.
    degenerate = sales_frame.copy()
    degenerate["returned"] = "yes"
    result = compare_models(
        degenerate,
        "classification",
        ["Logistic Regression", "Decision Tree"],
        ["units", "region"],
        "returned",
    )
    assert {o.model_name for o in result.failed} == {"Logistic Regression"}
    assert {o.model_name for o in result.successful} == {"Decision Tree"}
    assert result.best_model is None
    assert result.evaluation_valid is False
    assert any("no best model is highlighted" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Undefined metrics and missing values
# ---------------------------------------------------------------------------


def test_invalid_silhouette_is_reported_as_none_not_zero():
    # One cluster per row makes the silhouette mathematically undefined;
    # the comparison must show a dash, never a ranking-driving 0.0.
    frame = pd.DataFrame({"a": np.arange(30, dtype=float), "b": np.arange(30, dtype=float) ** 2})
    result = compare_models(frame, "clustering", ["K-Means"], ["a", "b"], n_clusters=30)
    assert result.outcomes[0].ok
    assert result.outcomes[0].silhouette_valid is False
    assert result.table_rows()[0]["metrics"]["silhouette"] is None
    assert result.best_model is None
    assert any("silhouette score is undefined" in w for w in result.warnings)


def test_missing_values_are_dropped_and_reported(sales_frame):
    missing = sales_frame.copy()
    missing.loc[0:3, "units"] = np.nan   # 4 rows with a missing feature
    missing.loc[5, "sales"] = np.nan     # 1 row with a missing target
    result = compare_models(
        missing, "regression", ["Linear Regression"], ["units", "profit"], "sales"
    )
    assert result.dropped_rows == 5
    assert result.outcomes[0].ok
    assert result.n_train + result.n_test == len(sales_frame) - 5
    assert any("5 rows were dropped" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Invalid requests raise ModelError
# ---------------------------------------------------------------------------


def test_unknown_task_type_raises(sales_frame):
    with pytest.raises(ModelError, match="Unknown task type"):
        compare_models(sales_frame, "forecasting", ["K-Means"], ["units"])


def test_empty_model_list_raises(sales_frame):
    with pytest.raises(ModelError, match="at least one model"):
        compare_models(sales_frame, "regression", [], ["units"], "sales")


def test_duplicate_model_names_are_deduplicated(sales_frame):
    result = compare_models(
        sales_frame, "regression", ["Linear Regression", "Linear Regression"], ["units"], "sales"
    )
    assert len(result.outcomes) == 1


def test_no_feature_columns_raises(sales_frame):
    with pytest.raises(ModelError, match="feature column"):
        compare_models(sales_frame, "regression", ["Linear Regression"], [], "sales")


def test_bad_test_size_raises(sales_frame):
    with pytest.raises(ModelError, match="Test size"):
        compare_models(sales_frame, "regression", ["Linear Regression"], ["units"], "sales", test_size=0.9)


def test_missing_feature_column_raises(sales_frame):
    with pytest.raises(ModelError, match="do not exist"):
        compare_models(sales_frame, "regression", ["Linear Regression"], ["not_a_column"], "sales")


def test_target_in_features_raises(sales_frame):
    with pytest.raises(ModelError, match="cannot also be a feature"):
        compare_models(sales_frame, "regression", ["Linear Regression"], ["units", "sales"], "sales")


def test_missing_target_raises_for_supervised(sales_frame):
    with pytest.raises(ModelError, match="target column is required"):
        compare_models(sales_frame, "regression", ["Linear Regression"], ["units"], None)


def test_too_few_usable_rows_raises(sales_frame):
    with pytest.raises(ModelError, match="at least 10 are needed"):
        compare_models(
            sales_frame.head(8), "regression", ["Linear Regression"], ["units"], "sales"
        )


def test_fewer_rows_than_clusters_raises(sales_frame):
    with pytest.raises(ModelError, match="one per cluster"):
        compare_models(sales_frame.head(12), "clustering", ["K-Means"], ["units"], n_clusters=20)


# ---------------------------------------------------------------------------
# Leakage guards
# ---------------------------------------------------------------------------


def test_one_hot_vocabulary_is_fitted_on_training_rows_only():
    # The split depends only on the number of rows and the seed, so we
    # can locate a held-out position with a numeric probe, then hide a
    # brand-new category there.
    probe = pd.DataFrame({"x": np.arange(40, dtype=float), "y": np.arange(40, dtype=float)})
    test_pos = int(prepare_shared_split(probe, ["x"], "y", 0.2, 42).test_index[0])

    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "units": rng.integers(1, 50, 40),
            "region": ["North", "South"] * 20,
            "sales": rng.uniform(100, 1000, 40).round(2),
        }
    )
    frame.loc[test_pos, "region"] = "Z"  # category that exists only in the test rows

    split = prepare_shared_split(frame, ["units", "region"], "sales", 0.2, 42)
    assert test_pos in set(split.test_index)
    assert "region_Z" not in split.feature_names
    assert "region_South" in split.feature_names
    # The unseen category encodes as the dropped baseline (all zeros).
    assert split.X_test.loc[test_pos, "region_South"] == 0.0

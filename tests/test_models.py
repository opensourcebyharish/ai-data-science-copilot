"""Tests for the scikit-learn model training engine (Milestone 6).

The whole suite runs offline: no network, no real database writes
(storage tests use a temp SQLite file). All assertions are against
values computed deterministically with random_state=42.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from core.models import (
    MODELS,
    TASK_LABELS,
    FittedRun,
    ModelError,
    ModelResult,
    available_models,
    suitable_target_columns,
    train_model,
    train_run,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sales_frame() -> pd.DataFrame:
    """Small labelled frame: 30 rows, numeric + categorical columns."""
    import numpy as np

    rng = np.random.default_rng(42)
    n = 30
    return pd.DataFrame(
        {
            "sales": rng.uniform(100, 1000, n).round(2),
            "profit": rng.uniform(10, 200, n).round(2),
            "units": rng.integers(1, 50, n).astype(float),
            "region": rng.choice(["North", "South", "East"], n),
            "category": rng.choice(["A", "B", "C"], n),
        }
    )


@pytest.fixture()
def binary_frame() -> pd.DataFrame:
    """Frame with a binary target column for classification."""
    import numpy as np

    rng = np.random.default_rng(7)
    n = 40
    return pd.DataFrame(
        {
            "x1": rng.uniform(0, 1, n),
            "x2": rng.uniform(0, 1, n),
            "label": rng.choice(["yes", "no"], n),
        }
    )


# ---------------------------------------------------------------------------
# available_models / TASK_LABELS
# ---------------------------------------------------------------------------


def test_available_models_returns_non_empty_list_for_each_task():
    for task in ("classification", "regression", "clustering"):
        names = available_models(task)
        assert len(names) >= 1
        assert all(isinstance(n, str) for n in names)


def test_available_models_unknown_task_returns_empty():
    assert available_models("unknown_task") == []  # type: ignore[arg-type]


def test_task_labels_covers_all_task_types():
    assert set(TASK_LABELS) == {"classification", "regression", "clustering"}


# ---------------------------------------------------------------------------
# suitable_target_columns
# ---------------------------------------------------------------------------


def test_suitable_targets_regression_returns_numeric_cols(sales_frame):
    targets = suitable_target_columns(sales_frame, "regression")
    assert "sales" in targets
    assert "profit" in targets
    assert "units" in targets
    # Categorical columns must NOT appear
    assert "region" not in targets
    assert "category" not in targets


def test_suitable_targets_classification_returns_low_cardinality_cols(sales_frame):
    targets = suitable_target_columns(sales_frame, "classification")
    assert "region" in targets
    assert "category" in targets


def test_suitable_targets_clustering_returns_empty(sales_frame):
    assert suitable_target_columns(sales_frame, "clustering") == []


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def test_train_run_returns_the_fitted_run_with_its_split(sales_frame):
    """``train_run`` is the same path as ``train_model``, but keeps the
    estimator and the exact matrices it saw (Milestone 8)."""
    run = train_run(
        frame=sales_frame,
        task_type="regression",
        model_name="Linear Regression",
        feature_columns=["units", "profit"],
        target_column="sales",
        test_size=0.3,
        random_state=42,
    )
    assert isinstance(run, FittedRun)
    assert run.X_train is not None and run.X_test is not None
    assert run.y_train is not None and run.y_test is not None
    assert len(run.X_train) == 21 and len(run.X_test) == 9
    # The result is the very same object ``train_model`` would return.
    assert run.result.model_name == "Linear Regression"
    assert "r2" in run.result.metrics


def test_train_run_and_train_model_agree(sales_frame):
    """The refactor must not change any ML behaviour."""
    from core.models import train_model

    kwargs = dict(
        frame=sales_frame,
        task_type="regression",
        model_name="Linear Regression",
        feature_columns=["units", "profit"],
        target_column="sales",
        test_size=0.3,
        random_state=42,
    )
    via_run = train_run(**kwargs).result
    via_model = train_model(**kwargs)
    assert via_run.metrics == via_model.metrics
    assert list(via_run.predictions) == list(via_model.predictions)
    assert via_run.caveats == via_model.caveats


def test_classification_refuses_a_continuous_target(sales_frame):
    """A continuous column is not a class label — refuse it clearly."""
    with pytest.raises(ModelError) as excinfo:
        train_model(
            frame=sales_frame,
            task_type="classification",
            model_name="Logistic Regression",
            feature_columns=["units", "profit"],
            target_column="sales",  # 40 distinct decimal values
            test_size=0.3,
        )
    message = str(excinfo.value)
    assert "continuous" in message
    assert "sales" in message


def test_classification_accepts_a_whole_number_target(sales_frame):
    """Whole numbers are discrete labels and stay usable."""
    frame = sales_frame.copy()
    frame["tier"] = (frame["units"] // 10).astype(int)
    result = train_model(
        frame=frame,
        task_type="classification",
        model_name="Decision Tree",
        feature_columns=["profit"],
        target_column="tier",
        test_size=0.3,
    )
    assert result.task_type == "classification"
    assert len(result.predictions) > 0


def test_classification_targets_are_ordered_with_discrete_columns_first(sales_frame):
    """A continuous column must never be the preselected target."""
    targets = suitable_target_columns(sales_frame, "classification")
    assert targets.index("region") < targets.index("sales")


def test_train_regression_linear(sales_frame):
    result = train_model(
        frame=sales_frame,
        task_type="regression",
        model_name="Linear Regression",
        feature_columns=["units", "profit"],
        target_column="sales",
        test_size=0.3,
        random_state=42,
    )
    assert result.task_type == "regression"
    assert result.model_name == "Linear Regression"
    assert result.target_column == "sales"
    assert "r2" in result.metrics
    assert "mae" in result.metrics
    assert "rmse" in result.metrics
    assert "mse" in result.metrics
    assert isinstance(result.metrics["r2"], float)
    assert len(result.predictions) > 0


def test_train_regression_random_forest(sales_frame):
    result = train_model(
        frame=sales_frame,
        task_type="regression",
        model_name="Random Forest",
        feature_columns=["units", "profit"],
        target_column="sales",
        test_size=0.3,
        random_state=42,
    )
    assert result.task_type == "regression"
    assert "r2" in result.metrics
    assert len(result.feature_importances) > 0


def test_train_regression_decision_tree(sales_frame):
    result = train_model(
        frame=sales_frame,
        task_type="regression",
        model_name="Decision Tree",
        feature_columns=["units", "profit"],
        target_column="sales",
        test_size=0.3,
        random_state=42,
    )
    assert "r2" in result.metrics


def test_regression_rmse_equals_sqrt_mse(sales_frame):
    result = train_model(
        frame=sales_frame,
        task_type="regression",
        model_name="Linear Regression",
        feature_columns=["units", "profit"],
        target_column="sales",
        test_size=0.3,
        random_state=42,
    )
    assert math.isclose(result.metrics["rmse"], math.sqrt(result.metrics["mse"]), rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_train_classification_logistic(binary_frame):
    result = train_model(
        frame=binary_frame,
        task_type="classification",
        model_name="Logistic Regression",
        feature_columns=["x1", "x2"],
        target_column="label",
        test_size=0.3,
        random_state=42,
    )
    assert result.task_type == "classification"
    assert "accuracy" in result.metrics
    assert "f1_weighted" in result.metrics
    assert 0.0 <= result.metrics["accuracy"] <= 1.0


def test_train_classification_random_forest(binary_frame):
    result = train_model(
        frame=binary_frame,
        task_type="classification",
        model_name="Random Forest",
        feature_columns=["x1", "x2"],
        target_column="label",
        test_size=0.3,
        random_state=42,
    )
    assert "accuracy" in result.metrics
    assert len(result.feature_importances) == 2


def test_train_classification_decision_tree(binary_frame):
    result = train_model(
        frame=binary_frame,
        task_type="classification",
        model_name="Decision Tree",
        feature_columns=["x1", "x2"],
        target_column="label",
        test_size=0.3,
        random_state=42,
    )
    assert "accuracy" in result.metrics


def test_classification_with_categorical_features(sales_frame):
    """Categorical features are one-hot encoded automatically."""
    result = train_model(
        frame=sales_frame,
        task_type="classification",
        model_name="Decision Tree",
        feature_columns=["sales", "region"],
        target_column="category",
        test_size=0.3,
        random_state=42,
    )
    assert "accuracy" in result.metrics


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def test_train_clustering_kmeans(sales_frame):
    result = train_model(
        frame=sales_frame,
        task_type="clustering",
        model_name="K-Means",
        feature_columns=["sales", "profit", "units"],
        n_clusters=3,
        random_state=42,
    )
    assert result.task_type == "clustering"
    assert result.target_column is None
    assert "n_clusters" in result.metrics
    assert "inertia" in result.metrics
    assert "silhouette" in result.metrics
    assert result.metrics["n_clusters"] == 3.0
    assert len(result.predictions) == len(sales_frame)


def test_clustering_labels_cover_expected_range(sales_frame):
    result = train_model(
        frame=sales_frame,
        task_type="clustering",
        model_name="K-Means",
        feature_columns=["sales", "profit"],
        n_clusters=4,
        random_state=42,
    )
    labels = set(result.predictions)
    assert labels == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# ModelResult helpers
# ---------------------------------------------------------------------------


def test_metrics_as_kpis_returns_list_of_dicts(sales_frame):
    result = train_model(
        frame=sales_frame,
        task_type="regression",
        model_name="Linear Regression",
        feature_columns=["units"],
        target_column="sales",
        test_size=0.3,
        random_state=42,
    )
    kpis = result.metrics_as_kpis()
    assert isinstance(kpis, list)
    assert all(isinstance(k, dict) for k in kpis)
    assert all("label" in k and "value" in k for k in kpis)


def test_to_storage_dict_is_json_serialisable(sales_frame):
    import json

    result = train_model(
        frame=sales_frame,
        task_type="regression",
        model_name="Linear Regression",
        feature_columns=["units"],
        target_column="sales",
        test_size=0.3,
        random_state=42,
    )
    d = result.to_storage_dict()
    # Must not raise
    serialised = json.dumps(d)
    assert "task_type" in serialised


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_unknown_task_type_raises(sales_frame):
    with pytest.raises(ModelError, match="Unknown task type"):
        train_model(
            frame=sales_frame,
            task_type="unknown",  # type: ignore[arg-type]
            model_name="Linear Regression",
            feature_columns=["units"],
            target_column="sales",
        )


def test_missing_target_column_raises_for_supervised(sales_frame):
    with pytest.raises(ModelError, match="target column is required"):
        train_model(
            frame=sales_frame,
            task_type="regression",
            model_name="Linear Regression",
            feature_columns=["units"],
            target_column=None,
        )


def test_nonexistent_feature_column_raises(sales_frame):
    with pytest.raises(ModelError, match="do not exist"):
        train_model(
            frame=sales_frame,
            task_type="regression",
            model_name="Linear Regression",
            feature_columns=["nonexistent_col"],
            target_column="sales",
        )


def test_nonexistent_target_column_raises(sales_frame):
    with pytest.raises(ModelError, match="do not exist"):
        train_model(
            frame=sales_frame,
            task_type="regression",
            model_name="Linear Regression",
            feature_columns=["units"],
            target_column="nonexistent",
        )


def test_target_in_features_raises(sales_frame):
    with pytest.raises(ModelError, match="cannot also be a feature"):
        train_model(
            frame=sales_frame,
            task_type="regression",
            model_name="Linear Regression",
            feature_columns=["sales", "units"],
            target_column="sales",
        )


def test_empty_features_raises(sales_frame):
    with pytest.raises(ModelError, match="At least one feature"):
        train_model(
            frame=sales_frame,
            task_type="regression",
            model_name="Linear Regression",
            feature_columns=[],
            target_column="sales",
        )


def test_invalid_test_size_raises(sales_frame):
    with pytest.raises(ModelError, match="Test size"):
        train_model(
            frame=sales_frame,
            task_type="regression",
            model_name="Linear Regression",
            feature_columns=["units"],
            target_column="sales",
            test_size=0.9,
        )


def test_unknown_model_name_raises(sales_frame):
    with pytest.raises(ModelError, match="Unknown model"):
        train_model(
            frame=sales_frame,
            task_type="regression",
            model_name="NonExistentModel",
            feature_columns=["units"],
            target_column="sales",
        )


# ---------------------------------------------------------------------------
# Storage (core/storage.py)
# ---------------------------------------------------------------------------


def test_storage_save_and_load_run(sales_frame, tmp_path):
    from core import storage

    result = train_model(
        frame=sales_frame,
        task_type="regression",
        model_name="Linear Regression",
        feature_columns=["units"],
        target_column="sales",
        test_size=0.3,
        random_state=42,
    )

    db_path = tmp_path / "test_runs.db"
    engine = storage.get_engine(db_path)
    try:
        storage.init_db(engine)
        run_id = storage.save_run(result, dataset_hash="abc123", engine=engine)
        assert isinstance(run_id, int)
        assert run_id >= 1

        runs = storage.load_runs(engine=engine)
        assert len(runs) == 1
        assert runs[0]["task_type"] == "regression"
        assert runs[0]["model_name"] == "Linear Regression"
        assert runs[0]["target_column"] == "sales"
        assert runs[0]["dataset_hash"] == "abc123"
        assert "r2" in runs[0]["metrics"]
    finally:
        engine.dispose()


def test_storage_delete_run(sales_frame, tmp_path):
    from core import storage

    result = train_model(
        frame=sales_frame,
        task_type="regression",
        model_name="Linear Regression",
        feature_columns=["units"],
        target_column="sales",
        test_size=0.3,
        random_state=42,
    )

    db_path = tmp_path / "test_runs.db"
    engine = storage.get_engine(db_path)
    try:
        run_id = storage.save_run(result, engine=engine)
        assert storage.delete_run(run_id, engine=engine) is True
        assert len(storage.load_runs(engine=engine)) == 0
    finally:
        engine.dispose()


def test_storage_delete_nonexistent_run_returns_false(tmp_path):
    from core import storage

    db_path = tmp_path / "test_runs.db"
    engine = storage.get_engine(db_path)
    try:
        storage.init_db(engine)
        assert storage.delete_run(9999, engine=engine) is False
    finally:
        engine.dispose()


def test_storage_load_runs_newest_first(sales_frame, tmp_path):
    from core import storage

    result = train_model(
        frame=sales_frame,
        task_type="regression",
        model_name="Linear Regression",
        feature_columns=["units"],
        target_column="sales",
        test_size=0.3,
        random_state=42,
    )
    result2 = train_model(
        frame=sales_frame,
        task_type="regression",
        model_name="Decision Tree",
        feature_columns=["units"],
        target_column="sales",
        test_size=0.3,
        random_state=42,
    )

    db_path = tmp_path / "test_runs.db"
    engine = storage.get_engine(db_path)
    try:
        id1 = storage.save_run(result, engine=engine)
        id2 = storage.save_run(result2, engine=engine)
        runs = storage.load_runs(engine=engine)
        assert runs[0]["id"] == id2  # newest first
        assert runs[1]["id"] == id1
    finally:
        engine.dispose()


def test_storage_load_runs_empty_when_no_runs(tmp_path):
    from core import storage

    db_path = tmp_path / "test_runs.db"
    engine = storage.get_engine(db_path)
    try:
        storage.init_db(engine)
        assert storage.load_runs(engine=engine) == []
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Missing values (Milestone 7 coverage for the training engine)
# ---------------------------------------------------------------------------


def test_train_model_drops_rows_with_missing_target(sales_frame):
    """A NaN target is dropped with an honest caveat — not a raw crash."""
    frame = sales_frame.copy()
    frame.loc[0:2, "sales"] = float("nan")
    result = train_model(
        frame=frame,
        task_type="regression",
        model_name="Linear Regression",
        feature_columns=["units", "profit"],
        target_column="sales",
        test_size=0.3,
    )
    assert any("missing values in the target" in caveat for caveat in result.caveats)
    assert len(result.predictions) > 0


def test_train_model_drops_rows_with_missing_features(sales_frame):
    frame = sales_frame.copy()
    frame.loc[0:4, "units"] = float("nan")
    result = train_model(
        frame=frame,
        task_type="regression",
        model_name="Linear Regression",
        feature_columns=["units", "profit"],
        target_column="sales",
        test_size=0.3,
    )
    assert any(
        "missing values in feature columns" in caveat for caveat in result.caveats
    )
    assert len(result.predictions) > 0


# ---------------------------------------------------------------------------
# Storage path override (hermetic run history)
# ---------------------------------------------------------------------------


def test_storage_env_var_overrides_the_database_path(monkeypatch, tmp_path, sales_frame):
    """ML_RUNS_DB_PATH relocates the database without any code changes."""
    from core import storage

    db = tmp_path / "elsewhere.db"
    monkeypatch.setenv("ML_RUNS_DB_PATH", str(db))
    result = train_model(
        frame=sales_frame,
        task_type="regression",
        model_name="Linear Regression",
        feature_columns=["units"],
        target_column="sales",
    )
    storage.save_run(result)
    assert db.exists()
    assert len(storage.load_runs()) == 1

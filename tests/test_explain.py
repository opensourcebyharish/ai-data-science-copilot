"""Tests for feature attribution (Milestone 7).

Attribution must stay honest: built-in importances for tree and
linear models, deterministic permutation importance as a fallback,
one-hot features folded back to their original columns, and no
claims at all for clustering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsRegressor

from core.comparison import compare_models
from core.explain import (
    CLUSTERING_NOTE,
    explain,
    group_importances,
    permutation_importances,
)
from core.models import FittedRun, ModelResult


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


def _knn_run() -> FittedRun:
    """A fitted KNN regression run — a model with no native importances."""
    rng = np.random.default_rng(0)
    x1 = rng.uniform(0, 10, 40)
    x2 = rng.uniform(0, 10, 40)
    y = x1 * 2 + rng.normal(0, 0.1, 40)
    frame = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    train, test = frame.iloc[:30], frame.iloc[30:]
    model = KNeighborsRegressor(n_neighbors=3).fit(train[["x1", "x2"]], train["y"])
    return FittedRun(
        result=ModelResult(
            task_type="regression",
            model_name="K-Nearest Neighbors",
            target_column="y",
            feature_columns=["x1", "x2"],
            metrics={},
        ),
        estimator=model,
        feature_names=["x1", "x2"],
        X_train=train[["x1", "x2"]],
        X_test=test[["x1", "x2"]],
        y_train=train["y"],
        y_test=test["y"],
    )


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def test_tree_model_uses_built_in_importances(sales_frame):
    result = compare_models(
        sales_frame, "regression", ["Random Forest"], ["units", "profit", "region"], "sales"
    )
    explanation = explain(result.runs["Random Forest"])
    assert explanation.available
    assert explanation.method == "built-in"
    assert sorted(explanation.importances) == ["profit", "region", "units"]
    assert all(value >= 0 for value in explanation.importances.values())
    # Grouping folds the one-hot pieces onto their parents but keeps
    # the total attribution identical to the raw scores.
    raw = result.runs["Random Forest"].result.feature_importances
    assert len(raw) > len(explanation.importances)  # dummies were folded
    assert sum(explanation.importances.values()) == pytest.approx(sum(raw.values()))


def test_linear_model_uses_absolute_coefficients(sales_frame):
    result = compare_models(
        sales_frame, "regression", ["Linear Regression"], ["units", "profit"], "sales"
    )
    explanation = explain(result.runs["Linear Regression"])
    assert explanation.method == "built-in"
    assert set(explanation.importances) == {"units", "profit"}
    assert all(value >= 0 for value in explanation.importances.values())


def test_permutation_fallback_when_no_native_importances_exist():
    run = _knn_run()
    assert not hasattr(run.estimator, "feature_importances_")
    assert not hasattr(run.estimator, "coef_")
    explanation = explain(run)
    assert explanation.available
    assert explanation.method == "permutation"
    assert set(explanation.importances) == {"x1", "x2"}
    # x1 drives the target, x2 is pure noise
    assert explanation.importances["x1"] > explanation.importances["x2"]


def test_permutation_importances_are_deterministic():
    run = _knn_run()
    first = explain(run).importances
    second = explain(run).importances
    assert first == second


def test_permutation_importances_without_a_held_out_split():
    assert permutation_importances(object(), None, None) is None


# ---------------------------------------------------------------------------
# Honesty rules
# ---------------------------------------------------------------------------


def test_every_explanation_disclaims_causation(sales_frame):
    result = compare_models(
        sales_frame, "regression", ["Random Forest", "Linear Regression"], ["units", "profit"], "sales"
    )
    for name, run in result.runs.items():
        note = explain(run).note
        assert "not causation" in note, name


def test_clustering_runs_never_claim_explanations():
    frame = pd.DataFrame(
        {"a": np.arange(30, dtype=float), "b": np.arange(30, dtype=float) ** 2}
    )
    result = compare_models(frame, "clustering", ["K-Means"], ["a", "b"], n_clusters=3)
    explanation = explain(result.runs["K-Means"])
    assert not explanation.available
    assert explanation.importances == {}
    assert explanation.note == CLUSTERING_NOTE


# ---------------------------------------------------------------------------
# Name grouping
# ---------------------------------------------------------------------------


def test_one_hot_features_fold_back_and_sum():
    grouped = group_importances(
        {"region_North": 0.2, "region_South": 0.3, "units": 0.5},
        ["units", "region"],
    )
    assert grouped == {"region": pytest.approx(0.5), "units": pytest.approx(0.5)}


def test_group_importances_prefers_the_longest_original_name():
    grouped = group_importances({"a_b_X": 1.0, "a_b": 2.0, "zzz": 3.0}, ["a", "a_b"])
    assert grouped == {"a_b": 3.0, "zzz": 3.0}


def test_categorical_attribution_uses_original_column_names(sales_frame):
    result = compare_models(
        sales_frame, "classification", ["Random Forest"], ["units", "region"], "returned"
    )
    explanation = explain(result.runs["Random Forest"])
    assert sorted(explanation.importances) == ["region", "units"]

"""Feature attribution for trained models (Milestone 7).

Answers one question honestly: which input columns is *this fitted
model* most sensitive to? Two sources are used, in order:

1. Importances the estimator exposes natively — impurity-based
   importances for tree ensembles, absolute coefficients for linear
   models. Cheap and exact for what the model learned.
2. Permutation importance on the held-out test split, for models that
   expose neither: how much the model's score drops when one feature's
   values are shuffled.

Whatever the source, the scores describe **association with this
model's predictions, not causation**, and transformed (one-hot)
features are folded back onto their original column names wherever
possible. Clustering is deliberately excluded: grouping rows is not
prediction, and no importance claim is made for it.

Framework-free: all rendering lives in views/models.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.models import FittedRun, native_importances

#: Permutation-importance settings, fixed for reproducibility.
PERMUTATION_REPEATS = 5
PERMUTATION_RANDOM_STATE = 42

#: The honesty disclaimer every explanation carries.
ASSOCIATION_NOTE = (
    "Importance describes association with this model's predictions, "
    "not causation."
)

#: Shown instead of an explanation for clustering runs.
CLUSTERING_NOTE = (
    "Clustering groups rows; it has no prediction target, so no "
    "feature-importance claim is made for it."
)

_NO_SCORES_NOTE = (
    "This model exposes no importances and permutation importance could "
    "not be computed on its test split, so no per-feature claim is made."
)

_METHOD_NOTES = {
    "built-in": (
        "Source: the model's own importances (impurity-based for tree "
        "ensembles, absolute coefficients for linear models)."
    ),
    "permutation": (
        "Source: permutation importance — the drop in the model's "
        "held-out score when a feature's values are shuffled."
    ),
}


@dataclass(frozen=True)
class Explanation:
    """Feature attributions for one fitted run."""

    importances: dict[str, float]  # original column names -> score
    method: str  # "built-in" | "permutation" | "none"
    note: str  # provenance + honesty disclaimer
    available: bool = True


def group_importances(
    transformed: dict[str, float], feature_columns: list[str]
) -> dict[str, float]:
    """Fold one-hot feature names back onto their original columns.

    ``region_North`` and ``region_South`` both map to ``region`` and
    their scores are added, because together they are that column's
    complete contribution to the model. The longest original name wins
    when prefixes overlap, and names that match no original column are
    kept verbatim, so nothing is silently lost or invented.
    """
    originals = sorted(feature_columns, key=len, reverse=True)  # longest prefix first
    grouped: dict[str, float] = {}
    for name, score in transformed.items():
        parent = name
        for original in originals:
            if name == original or name.startswith(original + "_"):
                parent = original
                break
        grouped[parent] = grouped.get(parent, 0.0) + float(score)
    return grouped


def permutation_importances(
    estimator: Any, X_test: pd.DataFrame | None, y_test: pd.Series | None
) -> dict[str, float] | None:
    """Permutation importances on the held-out split, or ``None``.

    Deterministic (fixed seed and repeat count) and never raises: a
    model that cannot be scored this way simply reports no scores
    instead of an invented number.
    """
    if X_test is None or y_test is None or len(X_test) < 2:
        return None
    try:
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            estimator,
            X_test,
            y_test,
            n_repeats=PERMUTATION_REPEATS,
            random_state=PERMUTATION_RANDOM_STATE,
        )
    except Exception:
        return None
    scores = {
        column: float(score)
        for column, score in zip(X_test.columns, result.importances_mean)
    }
    return scores or None


def explain(run: FittedRun) -> Explanation:
    """Attribute one fitted run's predictions to its input columns.

    Clustering runs get an honest "unavailable" explanation rather
    than a fabricated score.
    """
    if run.result.task_type == "clustering":
        return Explanation(
            importances={}, method="none", note=CLUSTERING_NOTE, available=False
        )

    scores = native_importances(run.estimator, run.feature_names)
    method = "built-in"
    if scores is None:
        scores = permutation_importances(run.estimator, run.X_test, run.y_test)
        method = "permutation"
    if not scores:
        return Explanation(
            importances={}, method="none", note=_NO_SCORES_NOTE, available=False
        )

    grouped = group_importances(scores, run.result.feature_columns)
    note = f"{ASSOCIATION_NOTE} {_METHOD_NOTES[method]}"
    return Explanation(importances=grouped, method=method, note=note)

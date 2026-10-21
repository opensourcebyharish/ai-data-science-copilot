"""Model comparison engine (Milestone 7).

Trains every selected model for one task on ONE shared split and ONE
shared preprocessing pass, so their metrics are directly comparable,
and records per-model failures without aborting the batch.

Design rules, in order of importance:

* **One split, one preprocessing pass.** ``prepare_shared_split``
  builds the train/test matrices exactly once; every supervised model
  fits on the same training rows and is scored on the same held-out
  rows. Clustering models are fitted on one identical feature matrix.
* **No leakage from the held-out rows.** The one-hot vocabulary is
  fitted on the training rows only; a category that appears only in
  the test split never enters the feature matrix, and rows with
  missing numeric features or a missing target are removed before the
  split is taken.
* **Failures are per model.** A model that cannot train is recorded as
  a failed outcome with a short, safe message; the remaining models
  still run and are still ranked.
* **A "best" model only when the evaluation is valid.** Rankings use
  the task's primary metric and only consider finite values from
  successful runs; a degenerate evaluation (a single class in the test
  set, an undefined silhouette) suppresses the highlight entirely
  rather than declaring a misleading winner.
* **Silhouette only where it is defined** — between 2 and n-1
  non-empty clusters; otherwise the table shows a dash instead of a
  number.

Streamlit lives in views/models.py; the estimators come from
core/models.py and the attributions from core/explain.py. Nothing here
executes model-generated code, touches the filesystem, or talks to a
network or external API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

import pandas as pd

# Internal helpers shared with core.models — same package, deliberately
# not duplicated here (the whole point of this module is reuse).
from core.models import (
    DEFAULT_N_CLUSTERS,
    DEFAULT_TEST_SIZE,
    FittedRun,
    ModelError,
    ModelResult,
    TaskType,
    TASK_LABELS,
    _prepare_features,
    _validate_columns,
    fit_clustering,
    fit_supervised,
    silhouette_valid,
)

#: Metric columns shown for each task (the comparison table).
TASK_METRICS: dict[TaskType, list[str]] = {
    "classification": ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"],
    "regression": ["mae", "rmse", "r2"],
    "clustering": ["silhouette"],
}

#: Whether a higher or a lower value means a better model.
METRIC_DIRECTIONS: dict[str, Literal["higher", "lower"]] = {
    "accuracy": "higher",
    "precision_weighted": "higher",
    "recall_weighted": "higher",
    "f1_weighted": "higher",
    "mae": "lower",
    "rmse": "lower",
    "mse": "lower",
    "r2": "higher",
    "silhouette": "higher",
}

#: The single metric used to declare a "best" model per task.
PRIMARY_METRIC: dict[TaskType, str] = {
    "classification": "f1_weighted",
    "regression": "rmse",
    "clustering": "silhouette",
}

#: Display names for the metrics.
METRIC_LABELS: dict[str, str] = {
    "accuracy": "Accuracy",
    "precision_weighted": "Precision (weighted)",
    "recall_weighted": "Recall (weighted)",
    "f1_weighted": "F1 (weighted)",
    "mae": "MAE",
    "rmse": "RMSE",
    "r2": "R²",
    "silhouette": "Silhouette",
}

#: One-sentence definitions shown next to the comparison table.
METRIC_DEFINITIONS: dict[str, str] = {
    "accuracy": "Share of held-out predictions that are exactly right.",
    "precision_weighted": "Of the rows predicted into each class, how many "
    "belonged there — averaged over classes by support.",
    "recall_weighted": "Of the held-out rows in each class, how many were "
    "found — averaged over classes by support.",
    "f1_weighted": "Harmonic mean of precision and recall, averaged over "
    "classes by support.",
    "mae": "Average absolute distance between predictions and actual values, "
    "in the target's own units.",
    "rmse": "Root of the average squared error — penalises large misses, "
    "same units as the target.",
    "r2": "Share of the held-out variance the model explains; 1 is perfect, "
    "0 is no better than predicting the mean.",
    "silhouette": "How tightly rows cluster together versus the nearest "
    "other cluster; only defined for 2 to n-1 non-empty clusters.",
}


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SharedSplit:
    """One preprocessing + split pass, shared by every compared model."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_index: pd.Index
    test_index: pd.Index
    dropped_rows: int
    feature_names: list[str]


@dataclass(frozen=True)
class ModelOutcome:
    """One model's result in a comparison — success or a safe failure."""

    model_name: str
    result: ModelResult | None = None
    error: str | None = None
    run: FittedRun | None = None
    silhouette_valid: bool = True

    @property
    def ok(self) -> bool:
        """True when the model trained and was scored."""
        return self.error is None and self.result is not None


@dataclass(frozen=True)
class ComparisonResult:
    """Everything the UI needs to render one comparison, safely."""

    task_type: TaskType
    model_names: list[str]
    metrics: list[str]
    primary_metric: str
    outcomes: list[ModelOutcome]
    best_model: str | None
    warnings: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    n_train: int | None = None
    n_test: int | None = None
    dropped_rows: int = 0
    evaluation_valid: bool = True
    runs: dict[str, FittedRun] = field(default_factory=dict)

    @property
    def successful(self) -> list[ModelOutcome]:
        """Outcomes of the models that trained and were scored."""
        return [o for o in self.outcomes if o.ok]

    @property
    def failed(self) -> list[ModelOutcome]:
        """Outcomes of the models that could not be evaluated."""
        return [o for o in self.outcomes if not o.ok]

    def metric_value(self, outcome: ModelOutcome, metric: str) -> float | None:
        """One metric for one model — ``None`` when it is not valid."""
        return _metric_of(outcome, metric)

    def table_rows(self) -> list[dict]:
        """JSON-safe rows: one per model, in the requested order.

        Each row: ``model``, ``ok``, ``error``, ``best`` and a
        ``metrics`` dict whose values are ``float`` or ``None`` —
        never a misleading zero for an undefined score.
        """
        rows = []
        for outcome in self.outcomes:
            rows.append(
                {
                    "model": outcome.model_name,
                    "ok": outcome.ok,
                    "error": outcome.error,
                    "best": outcome.model_name == self.best_model,
                    "metrics": {
                        metric: _metric_of(outcome, metric) for metric in self.metrics
                    },
                }
            )
        return rows


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def best_by_metric(values: Mapping[str, float | None], metric: str) -> str | None:
    """Name with the best valid value for ``metric``, honouring direction.

    ``None`` and non-finite values are excluded — an invalid evaluation
    is never ranked. Ties resolve to the first candidate in iteration
    order, which keeps the outcome deterministic.
    """
    direction = METRIC_DIRECTIONS.get(metric)
    if direction is None:
        return None
    valid = {
        name: float(value)
        for name, value in values.items()
        if value is not None and math.isfinite(float(value))
    }
    if not valid:
        return None
    pick = min if direction == "lower" else max
    return pick(valid, key=valid.get)  # type: ignore[arg-type]


def prepare_shared_split(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = 42,
) -> SharedSplit:
    """Build the one train/test split every supervised model will use.

    Leakage rules: rows with missing numeric features or a missing
    target are removed *before* the split is taken, and the one-hot
    vocabulary is fitted on the training rows only — a category that
    appears only in the held-out rows never enters the feature matrix
    (such rows encode as the dropped baseline, i.e. all zeros).

    Raises ``ModelError`` when the columns, the split size, or the
    remaining row count make an honest evaluation impossible.
    """
    from sklearn.model_selection import train_test_split

    _validate_columns(frame, feature_columns, "feature")
    _validate_columns(frame, [target_column], "target")
    if not (0.1 <= test_size <= 0.5):
        raise ModelError("Test size must be between 10% and 50%.")

    raw = frame[feature_columns]
    numeric_cols = [c for c in feature_columns if pd.api.types.is_numeric_dtype(raw[c])]
    # Categorical NaNs become all-zero rows in the encoding (baseline
    # category) — the same convention _prepare_features uses.
    if numeric_cols:
        usable = raw[numeric_cols].notna().all(axis=1)
    else:
        usable = pd.Series(True, index=frame.index)
    usable = usable & frame[target_column].notna()

    usable_index = frame.index[usable]
    dropped_rows = int(len(frame) - len(usable_index))
    if len(usable_index) < 10:
        raise ModelError(
            f"Only {len(usable_index)} usable rows after dropping missing "
            "values — at least 10 are needed."
        )

    train_index, test_index = train_test_split(
        usable_index.values, test_size=test_size, random_state=random_state
    )
    if len(train_index) < 2:
        raise ModelError(
            f"After the {test_size:.0%} test split only {len(train_index)} "
            "training row(s) remain. Upload a larger dataset or reduce the "
            "test split."
        )

    cat_cols = [c for c in feature_columns if c not in numeric_cols]

    def _encode(block: pd.DataFrame) -> pd.DataFrame:
        if cat_cols:
            block = pd.get_dummies(block, columns=cat_cols, drop_first=True, dtype=float)
        return block.astype(float)

    X_train = _encode(frame.loc[train_index, feature_columns])
    X_test = _encode(frame.loc[test_index, feature_columns])
    # Test rows only ever map onto the training vocabulary; unseen
    # categories encode as the dropped baseline (all zeros).
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    y = frame.loc[usable_index, target_column]
    y_train = y.loc[train_index]
    y_test = y.loc[test_index]

    return SharedSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_index=pd.Index(train_index),
        test_index=pd.Index(test_index),
        dropped_rows=dropped_rows,
        feature_names=list(X_train.columns),
    )


def compare_models(
    frame: pd.DataFrame,
    task_type: TaskType,
    model_names: Sequence[str],
    feature_columns: list[str],
    target_column: str | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = 42,
) -> ComparisonResult:
    """Train and evaluate every selected model on identical data.

    Request-level problems (unknown task, no models selected, bad
    columns, an unusable split) raise ``ModelError``; anything that
    goes wrong for one individual model is recorded as a failed
    :class:`ModelOutcome` and the remaining models still run.
    """
    if task_type not in TASK_LABELS:
        raise ModelError(
            f'Unknown task type "{task_type}". '
            f"Supported: {', '.join(TASK_LABELS)}."
        )
    names = list(dict.fromkeys(model_names))  # de-duplicate, keep requested order
    if not names:
        raise ModelError("Select at least one model to compare.")
    if not feature_columns:
        raise ModelError("At least one feature column is required.")
    if not (0.1 <= test_size <= 0.5):
        raise ModelError("Test size must be between 10% and 50%.")
    _validate_columns(frame, feature_columns, "feature")

    warnings: list[str] = []
    runs: dict[str, FittedRun] = {}
    outcomes: list[ModelOutcome] = []
    feature_names: list[str] = []
    n_train: int | None = None
    n_test: int | None = None
    dropped_rows = 0
    evaluation_valid = True

    if task_type == "clustering":
        # Clustering has no held-out evaluation: every model is fitted
        # on one identical, complete feature matrix.
        X = _prepare_features(frame, feature_columns)
        dropped_rows = len(frame) - len(X)
        if dropped_rows:
            warnings.append(
                f"{dropped_rows} rows were dropped due to missing values in "
                "feature columns."
            )
        if len(X) < n_clusters:
            raise ModelError(
                f"Dataset has {len(X)} usable rows but clustering requires at "
                f"least {n_clusters} rows (one per cluster)."
            )
        feature_names = list(X.columns)
        for name in names:
            try:
                run = fit_clustering(
                    name,
                    X,
                    feature_columns=feature_columns,
                    n_clusters=n_clusters,
                    random_state=random_state,
                )
            except Exception as exc:  # per-model failure, never fatal
                outcomes.append(ModelOutcome(model_name=name, error=_friendly_error(exc)))
                continue
            valid = silhouette_valid(run.result.predictions)
            if not valid:
                warnings.append(
                    f"{name}: the silhouette score is undefined for this run "
                    "(it needs between 2 and n-1 non-empty clusters), so it is "
                    "not ranked."
                )
            outcomes.append(
                ModelOutcome(
                    model_name=name, result=run.result, run=run, silhouette_valid=valid
                )
            )
            runs[name] = run
    else:
        if target_column is None:
            raise ModelError(f"A target column is required for {task_type}.")
        _validate_columns(frame, [target_column], "target")
        if target_column in feature_columns:
            raise ModelError("The target column cannot also be a feature column.")

        shared = prepare_shared_split(
            frame, feature_columns, target_column, test_size, random_state
        )
        feature_names = shared.feature_names
        dropped_rows = shared.dropped_rows
        n_train = len(shared.train_index)
        n_test = len(shared.test_index)
        if dropped_rows:
            warnings.append(
                f"{dropped_rows} rows were dropped due to missing values in "
                "features or target."
            )
        if task_type == "classification" and shared.y_test.nunique() < 2:
            evaluation_valid = False
            warnings.append(
                "Only one class is present in the test set — the metrics would "
                "be misleading, so no best model is highlighted."
            )
        if n_test < 20:
            warnings.append(
                f"Test set is small ({n_test} rows) — metrics may not generalise."
            )
        for name in names:
            try:
                run = fit_supervised(
                    name,
                    shared.X_train,
                    shared.X_test,
                    shared.y_train,
                    shared.y_test,
                    task_type=task_type,
                    target_column=target_column,
                    feature_columns=feature_columns,
                    random_state=random_state,
                )
            except Exception as exc:  # per-model failure, never fatal
                outcomes.append(ModelOutcome(model_name=name, error=_friendly_error(exc)))
                continue
            outcomes.append(ModelOutcome(model_name=name, result=run.result, run=run))
            runs[name] = run

    if not runs:
        warnings.append("No model completed successfully — see the per-model errors.")
    elif len(outcomes) > len(runs):
        warnings.append(
            f"{len(outcomes) - len(runs)} of {len(outcomes)} model(s) failed — "
            "the numbers below cover the rest."
        )

    primary = PRIMARY_METRIC[task_type]
    best_model = None
    if evaluation_valid and runs:
        best_model = best_by_metric(
            {
                outcome.model_name: _metric_of(outcome, primary)
                for outcome in outcomes
                if outcome.result is not None
            },
            primary,
        )

    return ComparisonResult(
        task_type=task_type,
        model_names=names,
        metrics=list(TASK_METRICS[task_type]),
        primary_metric=primary,
        outcomes=outcomes,
        best_model=best_model,
        warnings=warnings,
        feature_names=feature_names,
        n_train=n_train,
        n_test=n_test,
        dropped_rows=dropped_rows,
        evaluation_valid=evaluation_valid,
        runs=runs,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _metric_of(outcome: ModelOutcome, metric: str) -> float | None:
    """Validated value of one metric for one outcome (``None`` if invalid)."""
    if not outcome.ok or outcome.result is None:
        return None
    if metric == "silhouette" and not outcome.silhouette_valid:
        return None
    value = outcome.result.metrics.get(metric)
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _friendly_error(exc: Exception) -> str:
    """A short, display-safe message for a per-model failure."""
    if isinstance(exc, ModelError):
        return str(exc)
    text = " ".join(str(exc).split()) or type(exc).__name__
    return text if len(text) <= 300 else text[:297] + "…"

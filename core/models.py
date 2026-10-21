"""Scikit-learn model training engine for the Models page (Milestone 6).

The view (views/models.py) calls ``train_model`` and gets back a
``ModelResult`` — metrics, feature importances, a predictions array.
No Streamlit, no file I/O, no SQL touches this module.

Milestone 7 added the reusable primitives ``fit_supervised`` /
``fit_clustering`` (returning a ``FittedRun``: estimator + the exact
matrices it saw). ``train_model`` is a thin validating wrapper around
them; core/comparison.py and core/explain.py reuse them directly so
that evaluation and attribution logic is never duplicated.

Milestone 8 added ``train_run`` — the same validating path as
``train_model`` but returning the ``FittedRun`` itself, so the export
layer can produce predictions and per-run configuration without
re-training. ``train_model`` delegates to it, so behaviour is identical.

Supported task types
--------------------
- ``classification``: predict a categorical/boolean column.
- ``regression``: predict a numeric column.
- ``clustering``: group rows without a label column (K-Means).

Every validation error raises ``ModelError`` with a message that is safe
to show on screen.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

TaskType = Literal["classification", "regression", "clustering"]

#: Human-readable task labels for the UI.
TASK_LABELS: dict[TaskType, str] = {
    "classification": "Classification",
    "regression": "Regression",
    "clustering": "Clustering",
}

#: Available models per task type.  Values are (display_name, sklearn_class_path).
MODELS: dict[TaskType, list[tuple[str, str]]] = {
    "classification": [
        ("Logistic Regression", "sklearn.linear_model.LogisticRegression"),
        ("Random Forest", "sklearn.ensemble.RandomForestClassifier"),
        ("Decision Tree", "sklearn.tree.DecisionTreeClassifier"),
    ],
    "regression": [
        ("Linear Regression", "sklearn.linear_model.LinearRegression"),
        ("Random Forest", "sklearn.ensemble.RandomForestRegressor"),
        ("Decision Tree", "sklearn.tree.DecisionTreeRegressor"),
    ],
    "clustering": [
        ("K-Means", "sklearn.cluster.KMeans"),
        ("MiniBatch K-Means", "sklearn.cluster.MiniBatchKMeans"),
        ("Agglomerative Clustering", "sklearn.cluster.AgglomerativeClustering"),
    ],
}

#: Default test split fraction.
DEFAULT_TEST_SIZE = 0.2

#: KMeans default cluster count.
DEFAULT_N_CLUSTERS = 3


class ModelError(ValueError):
    """A training or validation problem the UI can show as a friendly message."""


@dataclass(frozen=True)
class ModelResult:
    """One completed training run.

    ``metrics`` is a dict of metric_name → float (JSON-safe).
    ``feature_importances`` maps feature name → importance score, or is
    empty when the model does not expose importances.
    ``predictions`` are the model's output labels/values on the test set
    (classification/regression) or cluster assignments for the full
    dataset (clustering).
    ``caveats`` are deterministic warnings computed during training.
    """

    task_type: TaskType
    model_name: str
    target_column: str | None          # None for clustering
    feature_columns: list[str]
    metrics: dict[str, float]
    feature_importances: dict[str, float] = field(default_factory=dict)
    predictions: list[Any] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def metrics_as_kpis(self) -> list[dict]:
        """Format metrics as KPI dicts for ui.components.render_kpis."""
        kpis = []
        _labels = {
            "accuracy": "Accuracy",
            "f1_weighted": "F1 (weighted)",
            "precision_weighted": "Precision",
            "recall_weighted": "Recall",
            "r2": "R² Score",
            "mae": "Mean Abs. Error",
            "rmse": "RMSE",
            "mse": "MSE",
            "silhouette": "Silhouette Score",
            "inertia": "Inertia",
            "n_clusters": "Clusters",
        }
        for key, value in self.metrics.items():
            label = _labels.get(key, key.replace("_", " ").title())
            kpis.append({"label": label, "value": f"{value:.4g}"})
        return kpis

    def to_storage_dict(self) -> dict:
        """JSON-serialisable dict for persistence in the runs table."""
        return {
            "task_type": self.task_type,
            "model_name": self.model_name,
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class FittedRun:
    """A trained estimator together with the exact data it saw.

    ``train_model`` only needs ``result``; the comparison and
    explainability layers (Milestone 7) reuse the estimator and the
    split matrices, so nothing has to be re-trained or re-derived.
    Supervised runs carry the train/test matrices; a clustering run
    carries the full feature matrix in ``X_train``.
    """

    result: ModelResult
    estimator: Any
    feature_names: list[str]
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_train: pd.Series | None = None
    y_test: pd.Series | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _import_class(dotted_path: str):
    """Import a class by its dotted module path (e.g. 'sklearn.linear_model.LogisticRegression')."""
    parts = dotted_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ModelError(f"Invalid class path: {dotted_path}")
    module_path, class_name = parts
    import importlib
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ModelError(f"Could not load model class {dotted_path}: {exc}") from exc


def _resolve_class_path(task_type: TaskType, model_name: str) -> str:
    """Look up the sklearn class path for a model name."""
    for display, class_path in MODELS.get(task_type, []):
        if display == model_name:
            return class_path
    available = [d for d, _ in MODELS.get(task_type, [])]
    raise ModelError(
        f'Unknown model "{model_name}" for task "{task_type}". '
        f"Available: {', '.join(available)}."
    )


def _validate_columns(frame: pd.DataFrame, columns: list[str], role: str) -> None:
    """Raise ModelError if any column is missing from the frame."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ModelError(
            f"The following {role} column(s) do not exist in the dataset: "
            + ", ".join(f'"{c}"' for c in missing)
        )


def _prepare_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Return numeric-only feature matrix, dropping rows with missing values."""
    X = frame[feature_columns].copy()
    # Encode categoricals with pd.get_dummies (one-hot), drop first to avoid multicollinearity
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)
    X = X.astype(float)
    # Drop rows with any NaN
    X = X.dropna()
    return X


def dataset_hash(frame: pd.DataFrame) -> str:
    """Short hash of the dataframe shape + columns for run provenance.

    Deliberately ignores cell values: two uploads with the same columns
    and shape share a provenance tag, which is all the run history needs.
    """
    payload = f"{frame.shape}:{list(frame.columns)}"
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def _model_kwargs(class_name: str, random_state: int) -> dict[str, Any]:
    """Deterministic constructor arguments for a supervised estimator."""
    if "Random" in class_name:
        return {"n_estimators": 100, "random_state": random_state}
    if "Decision" in class_name:
        return {"random_state": random_state, "max_depth": 10}
    if class_name == "LogisticRegression":
        return {"max_iter": 500, "random_state": random_state}
    return {}


def _clustering_model_kwargs(
    class_name: str, n_clusters: int, random_state: int
) -> dict[str, Any]:
    """Deterministic constructor arguments for a clustering estimator.

    Agglomerative clustering is deterministic and accepts neither
    ``random_state`` nor ``n_init``; the K-Means family takes both.
    """
    kwargs: dict[str, Any] = {"n_clusters": n_clusters}
    if "Agglomerative" not in class_name:
        kwargs.update(random_state=random_state, n_init=10)
    return kwargs


def silhouette_valid(labels) -> bool:
    """True when a silhouette score is mathematically defined for ``labels``.

    scikit-learn requires between 2 and n_samples - 1 non-empty
    clusters; anything else is undefined — not zero.
    """
    n = len(labels)
    if n < 2:
        return False
    n_labels = len({int(label) for label in labels})
    return 2 <= n_labels <= n - 1


def native_importances(
    estimator: Any, feature_names: list[str]
) -> dict[str, float] | None:
    """Importances an estimator exposes natively, or ``None``.

    Tree ensembles report impurity-based ``feature_importances_``;
    linear models report ``coef_`` (absolute values, averaged across
    outputs for multi-output models). ``None`` means the model exposes
    neither — the explainability layer can then fall back to
    permutation importance instead of inventing a score.
    """
    if hasattr(estimator, "feature_importances_"):
        values = [float(v) for v in estimator.feature_importances_]
        return dict(zip(feature_names, values))
    if hasattr(estimator, "coef_"):
        coef = np.abs(np.asarray(estimator.coef_, dtype=float))
        if coef.ndim > 1:
            coef = coef.mean(axis=0)
        return dict(zip(feature_names, [float(v) for v in coef]))
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def available_models(task_type: TaskType) -> list[str]:
    """Display names of models available for a task type."""
    return [name for name, _ in MODELS.get(task_type, [])]


def suitable_target_columns(frame: pd.DataFrame, task_type: TaskType) -> list[str]:
    """Columns appropriate as prediction targets for the given task type.

    - regression: numeric columns with more than 1 unique value
    - classification: columns with 2–50 unique non-null values
    - clustering: no target column (returns empty list)

    For classification the columns are *ordered* so that genuinely
    discrete labels (text, categories, booleans, whole numbers) come
    first and continuous numeric columns come last. Every candidate is
    still returned — the order only decides which one is preselected.
    """
    if task_type == "clustering":
        return []
    cols = []
    for col in frame.columns:
        n_unique = frame[col].nunique(dropna=True)
        if task_type == "regression":
            if pd.api.types.is_numeric_dtype(frame[col]) and n_unique > 1:
                cols.append(col)
        else:  # classification
            if 2 <= n_unique <= 50:
                cols.append(col)
    if task_type == "classification":
        cols = sorted(cols, key=lambda col: _discrete_rank(frame[col]))
    return cols


def _discrete_rank(column: pd.Series) -> int:
    """Sort key for classification candidates: 0 = clearly a label.

    Text/category/boolean columns are unambiguous labels. Numeric
    columns whose values are whole numbers are labels too. Numeric
    columns with fractional values are continuous and are ranked last,
    so they are never preselected by accident.
    """
    if not pd.api.types.is_numeric_dtype(column):
        return 0
    if pd.api.types.is_bool_dtype(column):
        return 0
    values = column.dropna()
    if values.empty:
        return 1
    finite = values[np.isfinite(values.astype(float))]
    if finite.empty:
        return 1
    if bool(np.all(finite == finite.round())):
        return 0
    return 2


def _continuous_target_problem(y: pd.Series) -> str | None:
    """Explain why a target cannot be used for classification, if so.

    scikit-learn treats a float column as ``continuous`` unless every
    value is a whole number, so a column like ``sales`` (40 distinct
    decimal values) is not a classification label. Returns a message
    safe to show on screen, or ``None`` when the target is fine.
    """
    if not pd.api.types.is_numeric_dtype(y):
        return None  # text/bool/category labels are fine
    values = y.dropna()
    if values.empty:
        return None
    if pd.api.types.is_bool_dtype(values):
        return None
    if not pd.api.types.is_float_dtype(values):
        return None  # integers are discrete labels
    finite = values[np.isfinite(values.astype(float))]
    if finite.empty:
        return None
    if bool(np.all(finite == finite.round())):
        return None  # whole numbers are discrete labels
    return (
        f'The target column "{y.name}" holds continuous values, so it cannot '
        "be used for classification. Pick a categorical column (for example "
        "a label or category column), or switch the task to Regression."
    )


def train_run(
    frame: pd.DataFrame,
    task_type: TaskType,
    model_name: str,
    feature_columns: list[str],
    target_column: str | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = 42,
) -> FittedRun:
    """Train one model and return the full :class:`FittedRun`.

    Same validation and behaviour as :func:`train_model`, but keeps the
    fitted estimator and the exact split it saw. Milestone 8's export
    layer uses this so predictions and per-run configuration can be
    produced without re-training anything.

    Raises ``ModelError`` on any validation or training problem.
    """
    from sklearn.model_selection import train_test_split

    if task_type not in TASK_LABELS:
        raise ModelError(
            f'Unknown task type "{task_type}". '
            f"Supported: {', '.join(TASK_LABELS)}."
        )
    if not feature_columns:
        raise ModelError("At least one feature column is required.")
    if not (0.1 <= test_size <= 0.5):
        raise ModelError("Test size must be between 10% and 50%.")

    _validate_columns(frame, feature_columns, "feature")

    caveats: list[str] = []
    # Resolve once up front so an unknown model name is reported before
    # anything heavier runs.
    _resolve_class_path(task_type, model_name)

    # --- Clustering ---
    if task_type == "clustering":
        X = _prepare_features(frame, feature_columns)
        feature_dropped = len(frame) - len(X)
        return fit_clustering(
            model_name,
            X,
            feature_columns=feature_columns,
            n_clusters=n_clusters,
            random_state=random_state,
            extra_caveats=(
                [
                    f"{feature_dropped} rows were dropped due to missing values "
                    "in feature columns."
                ]
                if feature_dropped
                else None
            ),
        )

    # --- Classification / Regression ---
    if target_column is None:
        raise ModelError(f"A target column is required for {task_type}.")
    _validate_columns(frame, [target_column], "target")
    if target_column in feature_columns:
        raise ModelError("The target column cannot also be a feature column.")

    X = _prepare_features(frame, feature_columns)
    feature_dropped = len(frame) - len(X)
    # Align y to the rows that survived dropna in X
    y = frame.loc[X.index, target_column]

    # Rows whose target value is missing can be neither trained nor
    # scored on; drop them before validation. A NaN target would
    # otherwise surface as a raw scikit-learn error instead of ours.
    target_missing = y.isna()
    if target_missing.any():
        caveats.append(
            f"{int(target_missing.sum())} rows were dropped due to missing "
            "values in the target column."
        )
        X = X.loc[~target_missing]
        y = y[~target_missing]

    if len(X) < 10:
        raise ModelError(
            f"Only {len(X)} usable rows after dropping missing values — "
            "at least 10 are needed."
        )
    if feature_dropped:
        caveats.append(
            f"{feature_dropped} rows were dropped due to missing values in "
            "feature columns."
        )

    # A classification target must be a set of discrete labels. A
    # continuous numeric column reaches this point (it has 2–50 distinct
    # values), and scikit-learn would reject it with a raw, technical
    # error deep inside the estimator. Refuse it here with a message the
    # page can show as-is.
    if task_type == "classification":
        problem = _continuous_target_problem(y)
        if problem:
            raise ModelError(problem)

    n_test = max(1, int(len(X) * test_size))
    n_train = len(X) - n_test
    if n_train < 2:
        raise ModelError(
            f"After the {test_size:.0%} test split only {n_train} training row(s) remain. "
            "Upload a larger dataset or reduce the test split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return fit_supervised(
        model_name,
        X_train,
        X_test,
        y_train,
        y_test,
        task_type=task_type,
        target_column=target_column,
        feature_columns=feature_columns,
        random_state=random_state,
        extra_caveats=caveats,
    )


def train_model(
    frame: pd.DataFrame,
    task_type: TaskType,
    model_name: str,
    feature_columns: list[str],
    target_column: str | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = 42,
) -> ModelResult:
    """Train a model and return evaluation metrics.

    Parameters
    ----------
    frame:            The loaded dataset (never modified).
    task_type:        "classification", "regression", or "clustering".
    model_name:       Display name from MODELS[task_type].
    feature_columns:  Columns used as inputs.
    target_column:    Column to predict (not used for clustering).
    test_size:        Fraction of data held out for evaluation (0.1–0.5).
    n_clusters:       Number of clusters for K-Means.
    random_state:     Seed for reproducibility.

    Raises
    ------
    ModelError on any validation or training problem.
    """
    return train_run(
        frame,
        task_type,
        model_name,
        feature_columns,
        target_column=target_column,
        test_size=test_size,
        n_clusters=n_clusters,
        random_state=random_state,
    ).result


def fit_clustering(
    model_name: str,
    X: pd.DataFrame,
    *,
    feature_columns: list[str],
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = 42,
    extra_caveats: list[str] | None = None,
) -> FittedRun:
    """Fit one clustering model on an already-prepared feature matrix.

    ``X`` must be the complete numeric feature matrix — clustering
    evaluates on every usable row. The comparison engine (Milestone 7)
    prepares the matrix once and calls this for each model, so every
    estimator is fitted on identical data.

    Raises ``ModelError`` when the matrix has fewer rows than
    requested clusters, or when the model name is unknown.
    """
    if n_clusters < 2:
        raise ModelError("The number of clusters must be at least 2.")
    if len(X) < n_clusters:
        raise ModelError(
            f"Dataset has {len(X)} usable rows but {model_name} requires at least "
            f"{n_clusters} rows (one per cluster)."
        )
    class_path = _resolve_class_path("clustering", model_name)
    ModelClass = _import_class(class_path)
    model = ModelClass(
        **_clustering_model_kwargs(class_path.rsplit(".", 1)[-1], n_clusters, random_state)
    )
    labels = model.fit_predict(X)

    try:
        from sklearn.metrics import silhouette_score
        sil = float(silhouette_score(X, labels)) if n_clusters > 1 else float("nan")
    except Exception:
        sil = float("nan")

    metrics: dict[str, float] = {"n_clusters": float(n_clusters)}
    if hasattr(model, "inertia_"):
        metrics["inertia"] = float(model.inertia_)
    metrics["silhouette"] = sil if np.isfinite(sil) else 0.0

    caveats = list(extra_caveats or [])
    if not silhouette_valid(labels):
        caveats.append(
            "The silhouette score is undefined for this run (it needs between 2 "
            "and n-1 non-empty clusters) and is shown as 0 for display only."
        )

    return FittedRun(
        result=ModelResult(
            task_type="clustering",
            model_name=model_name,
            target_column=None,
            feature_columns=feature_columns,
            metrics=metrics,
            feature_importances={},
            predictions=labels.tolist(),
            caveats=caveats,
        ),
        estimator=model,
        feature_names=list(X.columns),
        X_train=X,
    )


def fit_supervised(
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    task_type: TaskType,
    target_column: str,
    feature_columns: list[str],
    random_state: int = 42,
    extra_caveats: list[str] | None = None,
) -> FittedRun:
    """Fit and evaluate one supervised model on already-prepared data.

    The caller supplies the exact train/test matrices: ``train_model``
    prepares them the Milestone 6 way, while the comparison engine
    prepares them once and reuses them so that every model is scored on
    identical rows. Raises ``ModelError`` for unknown models or
    unsupported task types.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        r2_score, mean_absolute_error, mean_squared_error,
    )

    if task_type not in ("classification", "regression"):
        raise ModelError(
            f'fit_supervised only handles classification or regression, not "{task_type}".'
        )
    class_path = _resolve_class_path(task_type, model_name)
    ModelClass = _import_class(class_path)
    model = ModelClass(**_model_kwargs(class_path.rsplit(".", 1)[-1], random_state))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    caveats = list(extra_caveats or [])
    if task_type == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        }
        n_classes = len(np.unique(y_test))
        if n_classes < 2:
            caveats.append("Only one class present in the test set — metrics may be misleading.")
    else:
        mse = float(mean_squared_error(y_test, y_pred))
        metrics = {
            "r2": float(r2_score(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mse)),
            "mse": mse,
        }

    if len(y_test) < 20:
        caveats.append(
            f"Test set is small ({len(y_test)} rows) — metrics may not generalise."
        )

    importances = native_importances(model, list(X_train.columns)) or {}

    return FittedRun(
        result=ModelResult(
            task_type=task_type,
            model_name=model_name,
            target_column=target_column,
            feature_columns=feature_columns,
            metrics=metrics,
            feature_importances=importances,
            predictions=y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred),
            caveats=caveats,
        ),
        estimator=model,
        feature_names=list(X_train.columns),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

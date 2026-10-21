"""Tests for export, reporting and reproducibility (Milestone 8).

Everything runs offline on small deterministic frames. The exports must
be honest (valid vs failed vs undefined), reproducible (seeds and
settings recorded) and private (no raw rows, API keys or binaries).
"""

from __future__ import annotations

import csv
import io
import json

import numpy as np
import pandas as pd
import pytest

from core import export
from core.comparison import compare_models
from core.models import ModelError, available_models, fit_supervised, train_model


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


@pytest.fixture()
def binary_frame() -> pd.DataFrame:
    """A frame with a clean binary classification target."""
    rng = np.random.default_rng(7)
    n = 60
    return pd.DataFrame(
        {
            "x1": rng.uniform(0, 10, n),
            "x2": rng.normal(0, 1, n),
            "label": rng.choice(["yes", "no"], n),
        }
    )


def _regression_comparison(frame: pd.DataFrame):
    return compare_models(
        frame=frame,
        task_type="regression",
        model_names=["Linear Regression", "Decision Tree"],
        feature_columns=["units", "profit"],
        target_column="sales",
        test_size=0.3,
    )


# ---------------------------------------------------------------------------
# Comparison CSV
# ---------------------------------------------------------------------------


def test_comparison_csv_has_a_row_per_model_with_metric_headers(sales_frame):
    result = _regression_comparison(sales_frame)
    rows = list(csv.reader(io.StringIO(export.comparison_csv(result))))

    header = rows[0]
    assert header[0] == "Model"
    assert any("RMSE" in column and "lower is better" in column for column in header)
    assert any("R²" in column and "higher is better" in column for column in header)
    body = rows[1:]
    assert {row[0] for row in body} == {"Linear Regression", "Decision Tree"}
    assert len(body[0]) == len(header)


def test_comparison_csv_marks_the_best_model(sales_frame):
    result = _regression_comparison(sales_frame)
    rows = list(csv.DictReader(io.StringIO(export.comparison_csv(result))))
    best = [row for row in rows if row["Best model"] == "yes"]
    assert len(best) == 1
    assert best[0]["Model"] == result.best_model


def test_comparison_csv_uses_empty_cells_for_failed_models(sales_frame):
    """A failed model exports its error and empty metrics — never zeros."""
    result = compare_models(
        frame=sales_frame,
        task_type="regression",
        model_names=["Linear Regression", "Decision Tree"],
        feature_columns=["profit"],
        target_column="sales",
        test_size=0.3,
    )
    # Force a failure by marking an outcome as errored, mirroring the engine.
    from core.comparison import ModelOutcome

    broken = ModelOutcome(model_name="Broken Model", error="could not train")
    result_with_failure = type(result)(
        task_type=result.task_type,
        model_names=result.model_names + ["Broken Model"],
        metrics=result.metrics,
        primary_metric=result.primary_metric,
        outcomes=result.outcomes + [broken],
        best_model=result.best_model,
        warnings=result.warnings,
    )
    rows = list(csv.DictReader(io.StringIO(export.comparison_csv(result_with_failure))))
    failed_row = next(row for row in rows if row["Model"] == "Broken Model")
    assert failed_row["Status"] == "failed"
    for metric_column in [c for c in failed_row if "(" in c]:
        assert failed_row[metric_column] == ""  # empty, not "0" or "nan"


def test_comparison_csv_empty_for_no_models_is_header_only():
    """An empty selection is rejected before export; an empty result has headers."""
    from core.comparison import ComparisonResult

    result = ComparisonResult(
        task_type="regression",
        model_names=[],
        metrics=["mae", "rmse", "r2"],
        primary_metric="rmse",
        outcomes=[],
        best_model=None,
    )
    rows = list(csv.reader(io.StringIO(export.comparison_csv(result))))
    assert len(rows) == 1  # header only
    assert rows[0][0] == "Model"


# ---------------------------------------------------------------------------
# Undefined metrics
# ---------------------------------------------------------------------------


def test_undefined_silhouette_is_empty_not_zero():
    """Silhouette collapses to a single cluster → undefined, so the export
    must show an empty cell rather than a misleading 0."""
    frame = pd.DataFrame({"x": [1.0] * 24, "y": [2.0] * 24})
    # Duplicate points make K-Means find one cluster; silhouette is undefined.
    result = compare_models(
        frame=frame,
        task_type="clustering",
        model_names=["K-Means"],
        feature_columns=["x", "y"],
        n_clusters=3,
    )
    row = list(csv.DictReader(io.StringIO(export.comparison_csv(result))))[0]
    silhouette_column = next(c for c in row if c.startswith("Silhouette"))
    assert row[silhouette_column] == ""  # empty, never "0"
    assert row["Status"] == "OK"  # it trained, the metric is just undefined


def test_comparison_report_notes_undefined_metrics_are_not_zero():
    """When a ranking is withheld, the report says why — no silent zeros."""
    frame = pd.DataFrame({"x": [1.0] * 24, "y": [2.0] * 24})
    result = compare_models(
        frame=frame,
        task_type="clustering",
        model_names=["K-Means"],
        feature_columns=["x", "y"],
        n_clusters=3,
    )
    context = export.ReportContext(
        dataset_filename="dup.csv", dataset_rows=24, dataset_columns=2
    )
    report = export.build_comparison_report(result=result, context=context)
    assert "—" in report  # undefined cells render as an em dash
    lowered = report.lower()
    assert "no best model is highlighted" in lowered or "not valid" in lowered


# ---------------------------------------------------------------------------
# Run metadata CSV
# ---------------------------------------------------------------------------


def test_run_metadata_csv_covers_every_outcome_including_failures(sales_frame):
    result = _regression_comparison(sales_frame)
    rows = list(csv.DictReader(io.StringIO(export.run_metadata_csv(result))))
    assert len(rows) == len(result.outcomes)
    assert all(row["Task"] == "Regression" for row in rows)
    assert all(row["Status"] == "OK" for row in rows)
    assert rows[0]["Feature columns"] == "units, profit"
    assert rows[0]["Target column"] == "sales"


def test_run_metadata_csv_includes_dropped_and_split_counts(sales_frame):
    frame = sales_frame.copy()
    frame.loc[0:4, "units"] = np.nan
    result = compare_models(
        frame=frame,
        task_type="regression",
        model_names=["Linear Regression"],
        feature_columns=["units", "profit"],
        target_column="sales",
        test_size=0.3,
    )
    rows = list(csv.DictReader(io.StringIO(export.run_metadata_csv(result))))
    assert int(rows[0]["Dropped rows"]) >= 5
    assert int(rows[0]["Train rows"]) > 0
    assert int(rows[0]["Test rows"]) > 0


# ---------------------------------------------------------------------------
# Prediction CSV
# ---------------------------------------------------------------------------


def test_predictions_csv_contains_row_actual_and_prediction_for_regression(sales_frame):
    result = train_model(
        frame=sales_frame,
        task_type="regression",
        model_name="Linear Regression",
        feature_columns=["units", "profit"],
        target_column="sales",
        test_size=0.3,
    )
    comparison = _regression_comparison(sales_frame)
    run = comparison.runs["Linear Regression"]
    frame = export.predictions_frame(run)
    assert list(frame.columns) == ["Row", "Actual", "Predicted"]
    assert len(frame) == len(run.y_test)


def test_predictions_csv_only_exports_held_out_rows(binary_frame):
    comparison = compare_models(
        frame=binary_frame,
        task_type="classification",
        model_names=["Logistic Regression"],
        feature_columns=["x1", "x2"],
        target_column="label",
        test_size=0.25,
    )
    run = comparison.runs["Logistic Regression"]
    frame = export.predictions_frame(run)
    assert len(frame) == comparison.n_test
    # No feature values leak into the prediction export.
    assert "x1" not in frame.columns and "x2" not in frame.columns


def test_classification_predictions_have_class_column_names(binary_frame):
    comparison = compare_models(
        frame=binary_frame,
        task_type="classification",
        model_names=["Decision Tree"],
        feature_columns=["x1", "x2"],
        target_column="label",
        test_size=0.25,
    )
    frame = export.predictions_frame(comparison.runs["Decision Tree"])
    assert list(frame.columns) == ["Row", "Actual class", "Predicted class"]


def test_clustering_predictions_export_cluster_labels(sales_frame):
    comparison = compare_models(
        frame=sales_frame,
        task_type="clustering",
        model_names=["K-Means"],
        feature_columns=["units", "profit"],
        n_clusters=3,
    )
    frame = export.predictions_frame(comparison.runs["K-Means"])
    assert list(frame.columns) == ["Row", "Cluster"]
    assert len(frame) == len(comparison.runs["K-Means"].result.predictions)


def test_predictions_export_refuses_a_run_without_a_test_split(sales_frame):
    """An honest error beats an empty or fabricated prediction file."""
    from core.models import FittedRun, ModelResult

    run = FittedRun(
        result=ModelResult(
            task_type="regression",
            model_name="Linear Regression",
            target_column="sales",
            feature_columns=["units"],
            metrics={},
            predictions=[],
        ),
        estimator=object(),
        feature_names=["units"],
    )
    with pytest.raises(ModelError):
        export.predictions_frame(run)


# ---------------------------------------------------------------------------
# Reproducibility metadata
# ---------------------------------------------------------------------------


def test_config_records_seed_features_target_and_preprocessing(sales_frame):
    result = train_model(
        frame=sales_frame,
        task_type="regression",
        model_name="Random Forest",
        feature_columns=["units", "profit"],
        target_column="sales",
        test_size=0.3,
    )
    config = export.config_from_result(result, test_size=0.3, random_state=42)
    payload = config.to_dict()
    assert payload["random_state"] == 42
    assert payload["target_column"] == "sales"
    assert payload["feature_columns"] == ["units", "profit"]
    assert payload["test_size"] == 0.3
    assert payload["preprocessing"]


def test_config_json_is_valid_json(sales_frame):
    result = train_model(
        frame=sales_frame,
        task_type="regression",
        model_name="Linear Regression",
        feature_columns=["units"],
        target_column="sales",
    )
    config = export.config_from_result(result)
    assert json.loads(export.config_json(config))["model_name"] == "Linear Regression"


def test_config_csv_includes_every_setting(sales_frame):
    result = train_model(
        frame=sales_frame,
        task_type="regression",
        model_name="Linear Regression",
        feature_columns=["units"],
        target_column="sales",
        test_size=0.3,
    )
    config = export.config_from_result(result, test_size=0.3, random_state=7)
    rows = dict(csv.reader(io.StringIO(export.config_csv(config))))
    assert rows["Random seed"] == "7"
    assert rows["Target column"] == "sales"
    assert rows["Feature columns"] == "units"
    assert rows["Test split"] == "30%"


def test_comparison_config_recovers_the_test_size(sales_frame):
    result = _regression_comparison(sales_frame)
    run = result.runs[result.best_model or result.model_names[0]]
    config = export.config_from_comparison(result, run)
    assert config.test_size == pytest.approx(0.3, abs=0.01)
    assert config.n_test == result.n_test
    assert config.random_state == export.DEFAULT_RANDOM_STATE


def test_clustering_config_records_the_cluster_count(sales_frame):
    result = compare_models(
        frame=sales_frame,
        task_type="clustering",
        model_names=["K-Means"],
        feature_columns=["units", "profit"],
        n_clusters=4,
    )
    config = export.config_from_comparison(result, result.runs["K-Means"])
    assert config.n_clusters == 4


# ---------------------------------------------------------------------------
# Privacy safeguards
# ---------------------------------------------------------------------------


def test_config_never_contains_secret_like_keys(sales_frame):
    result = train_model(
        frame=sales_frame,
        task_type="regression",
        model_name="Linear Regression",
        feature_columns=["units"],
        target_column="sales",
    )
    payload = json.dumps(export.config_from_result(result).to_dict())
    assert not export.contains_secret_like(payload)


def test_metadata_csv_never_leaks_row_values(sales_frame):
    """Metadata export is configuration only — no cell values, no secrets."""
    result = _regression_comparison(sales_frame)
    text = export.run_metadata_csv(result) + export.comparison_csv(result)
    assert not export.contains_secret_like(text)
    # A specific data value from the frame must not appear verbatim.
    sentinel = f"{float(sales_frame['sales'].iloc[0]):.6g}"
    assert sentinel not in text


def test_report_never_contains_secret_like_text(sales_frame):
    result = _regression_comparison(sales_frame)
    context = export.ReportContext(
        dataset_filename="sales.csv", dataset_rows=40, dataset_columns=5
    )
    report = export.build_comparison_report(result=result, context=context)
    assert not export.contains_secret_like(report)
    assert "API key" not in report or "no raw" in report.lower()


def test_report_states_importance_is_not_causation(sales_frame):
    result = _regression_comparison(sales_frame)
    context = export.ReportContext(
        dataset_filename="sales.csv", dataset_rows=40, dataset_columns=5
    )
    report = export.build_comparison_report(result=result, context=context)
    lowered = report.lower()
    assert "association" in lowered
    assert "not causation" in lowered or "causation" in lowered


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def _context() -> export.ReportContext:
    return export.ReportContext(
        dataset_filename="sales.csv",
        dataset_rows=40,
        dataset_columns=5,
        dataset_hash="abc123",
    )


def test_report_is_self_contained_html(sales_frame):
    result = _regression_comparison(sales_frame)
    report = export.build_comparison_report(result=result, context=_context())
    assert report.startswith("<!DOCTYPE html>")
    assert report.rstrip().endswith("</html>")
    assert "<style>" in report            # inline styles — no external assets
    assert "http://" not in report        # no remote references
    assert "https://" not in report


def test_report_includes_dataset_context_and_config(sales_frame):
    result = _regression_comparison(sales_frame)
    report = export.build_comparison_report(result=result, context=_context())
    assert "sales.csv" in report
    assert "abc123" in report
    assert "Random seed" in report
    assert "Feature columns" in report


def test_report_includes_an_inline_svg_chart(sales_frame):
    result = _regression_comparison(sales_frame)
    report = export.build_comparison_report(result=result, context=_context())
    assert "<svg" in report
    assert "<rect" in report


def test_report_distinguishes_valid_from_failed_models(sales_frame):
    from core.comparison import ModelOutcome

    result = _regression_comparison(sales_frame)
    with_failure = type(result)(
        task_type=result.task_type,
        model_names=result.model_names + ["Broken"],
        metrics=result.metrics,
        primary_metric=result.primary_metric,
        outcomes=result.outcomes + [ModelOutcome(model_name="Broken", error="boom")],
        best_model=result.best_model,
        warnings=result.warnings,
    )
    report = export.build_comparison_report(result=with_failure, context=_context())
    assert "Failed models" in report
    assert "boom" in report


def test_report_lists_warnings(sales_frame):
    result = compare_models(
        frame=sales_frame,
        task_type="regression",
        model_names=["Linear Regression"],
        feature_columns=["units", "profit"],
        target_column="sales",
        test_size=0.45,  # small test set → a warning
    )
    report = export.build_comparison_report(result=result, context=_context())
    if result.warnings:
        assert "Warnings" in report


def test_warnings_are_listed_once_not_repeated(sales_frame):
    """A warning must not appear twice in the report."""
    result = compare_models(
        frame=sales_frame,
        task_type="regression",
        model_names=["Linear Regression"],
        feature_columns=["units", "profit"],
        target_column="sales",
        test_size=0.45,
    )
    assert result.warnings, "expected a small-test-set warning"
    report = export.build_comparison_report(result=result, context=_context())
    for warning in result.warnings:
        assert report.count(warning) == 1


def test_primary_metric_chart_states_which_direction_is_better(sales_frame):
    """A longer RMSE bar is worse — the caption must say so."""
    result = _regression_comparison(sales_frame)
    report = export.build_comparison_report(result=result, context=_context())
    assert "RMSE by model" in report
    assert "lower is better" in report


def test_single_run_report_renders_metrics(sales_frame):
    result = train_model(
        frame=sales_frame,
        task_type="regression",
        model_name="Random Forest",
        feature_columns=["units", "profit"],
        target_column="sales",
        test_size=0.3,
    )
    comparison = _regression_comparison(sales_frame)
    config = export.config_from_result(result, test_size=0.3)
    report = export.build_single_report(
        result=result, run=None, context=_context(), config=config
    )
    assert "R²" in report or "RMSE" in report
    assert "Random Forest" in report


def test_single_run_report_without_metrics_states_so_honestly():
    from core.models import ModelResult

    result = ModelResult(
        task_type="regression",
        model_name="Empty",
        target_column="sales",
        feature_columns=["units"],
        metrics={},
    )
    config = export.config_from_result(result)
    report = export.build_single_report(
        result=result, run=None, context=_context(), config=config
    )
    assert "No valid metrics" in report


def test_report_handles_special_characters_in_column_names():
    """Column names with HTML/quote characters must be escaped, not injected."""
    rng = np.random.default_rng(1)
    n = 40
    frame = pd.DataFrame(
        {
            "revenue <net>": rng.uniform(100, 900, n),
            "units & sold": rng.integers(1, 30, n),
            "price$": rng.uniform(5, 50, n),
        }
    )
    result = compare_models(
        frame=frame,
        task_type="regression",
        model_names=["Linear Regression"],
        feature_columns=["units & sold", "price$"],
        target_column="revenue <net>",
        test_size=0.3,
    )
    report = export.build_comparison_report(result=result, context=_context())
    assert "<script>" not in report
    assert "&lt;net&gt;" in report  # escaped, not raw markup


def test_report_is_deterministic_when_a_timestamp_is_supplied(sales_frame):
    result = _regression_comparison(sales_frame)
    context = export.ReportContext(
        dataset_filename="sales.csv",
        dataset_rows=40,
        dataset_columns=5,
        generated_at="2026-01-01T00:00:00+00:00",
    )
    first = export.build_comparison_report(result=result, context=context)
    second = export.build_comparison_report(result=result, context=context)
    assert first == second


# ---------------------------------------------------------------------------
# Filenames
# ---------------------------------------------------------------------------


def test_safe_filename_replaces_special_characters():
    assert export.safe_filename("Report / v2", extension="html") == "Report-v2.html"
    assert export.safe_filename("  ", extension="csv") == "export.csv"
    assert export.safe_filename("a:b*c?", extension="csv") == "a-b-c.csv"


def test_safe_filename_never_contains_path_separators():
    name = export.safe_filename("../etc/passwd", extension="csv")
    assert "/" not in name and "\\" not in name
    assert ".." not in name


def test_safe_filename_truncates_long_names():
    name = export.safe_filename("x" * 300, extension="csv")
    assert len(name) <= 84
    assert name.endswith(".csv")


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


def test_comparison_csv_round_trips_through_the_csv_module(sales_frame):
    result = _regression_comparison(sales_frame)
    text = export.comparison_csv(result)
    parsed = list(csv.reader(io.StringIO(text)))
    assert len(parsed) == len(result.outcomes) + 1
    # Every row has the same number of fields (no ragged CSV).
    assert len({len(row) for row in parsed}) == 1


def test_predictions_csv_round_trips_for_a_run_in_a_comparison(sales_frame):
    result = _regression_comparison(sales_frame)
    run = result.runs["Linear Regression"]
    text = export.predictions_csv(run)
    parsed = list(csv.reader(io.StringIO(text)))
    assert len(parsed) == len(run.y_test) + 1
    assert parsed[0] == ["Row", "Actual", "Predicted"]


# ---------------------------------------------------------------------------
# Storage integration (config column)
# ---------------------------------------------------------------------------


def test_saved_run_persists_reproducibility_config(sales_frame, tmp_path):
    from core import storage

    result = train_model(
        frame=sales_frame,
        task_type="regression",
        model_name="Linear Regression",
        feature_columns=["units"],
        target_column="sales",
        test_size=0.3,
    )
    config = export.config_from_result(result, test_size=0.3).to_dict()

    engine = storage.get_engine(tmp_path / "runs.db")
    try:
        storage.save_run(result, dataset_hash="h", engine=engine, config=config)
        runs = storage.load_runs(engine=engine)
        assert runs[0]["config"]["random_state"] == 42
        assert runs[0]["config"]["feature_columns"] == ["units"]
    finally:
        engine.dispose()


def test_saved_run_without_config_defaults_to_empty_dict(sales_frame, tmp_path):
    from core import storage

    result = train_model(
        frame=sales_frame,
        task_type="regression",
        model_name="Linear Regression",
        feature_columns=["units"],
        target_column="sales",
    )
    engine = storage.get_engine(tmp_path / "runs.db")
    try:
        storage.save_run(result, engine=engine)
        assert storage.load_runs(engine=engine)[0]["config"] == {}
    finally:
        engine.dispose()


def test_existing_database_is_migrated_with_the_config_column(sales_frame, tmp_path):
    """A pre-Milestone-8 database gains the column without losing data."""
    from sqlalchemy import create_engine, text

    from core import storage

    db = tmp_path / "old.db"
    engine = create_engine(f"sqlite:///{db}")
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE ml_runs (id INTEGER PRIMARY KEY, created_at DATETIME, "
                "task_type VARCHAR(32), model_name VARCHAR(64), target_column VARCHAR(256), "
                "feature_columns TEXT, metrics TEXT, dataset_hash VARCHAR(32))"
            )
        )
        connection.execute(
            text(
                "INSERT INTO ml_runs (task_type, model_name, feature_columns, metrics) "
                "VALUES ('regression', 'Old Model', '[\"units\"]', '{\"r2\": 0.5}')"
            )
        )
    engine.dispose()

    # init_db must add the column without dropping the existing row.
    migrated = storage.get_engine(db)
    try:
        storage.init_db(migrated)
        runs = storage.load_runs(engine=migrated)
        assert runs[0]["model_name"] == "Old Model"
        assert runs[0]["config"] == {}
    finally:
        migrated.dispose()

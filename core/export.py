"""Export, reporting and reproducibility for trained runs (Milestone 8).

Turns the objects the ML layer already produces (``core.models`` /
``core.comparison`` / ``core.explain``) into downloadable artefacts:

* **CSV** — comparison metrics, per-model run metadata, and the
  reproducibility configuration of a run.
* **CSV** — prediction results for a supervised run: only the held-out
  rows, with a row identifier, the actual value and the prediction.
* **HTML** — a self-contained analysis report (dataset context, task and
  model configuration, valid metrics, warnings, comparison results and
  inline SVG charts). It opens in any browser and prints to PDF; no new
  dependency is needed.

Honesty rules, carried over from Milestone 7 and enforced here:

* A failed model is exported with its error and **empty** metric cells —
  never a fabricated ``0``.
* An undefined metric (for example a silhouette that is not defined) is
  exported as an empty cell, and the report says so explicitly.
* The report distinguishes valid from failed/undefined results and never
  claims that feature importance proves causation.

Privacy rules:

* No raw dataset rows are exported beyond the prediction output the user
  explicitly asked for (held-out rows only), and the feature values
  themselves are never included.
* No API keys, secrets, or model binaries are ever written. Only the
  configuration *settings* needed to reproduce a run are recorded.

Framework-free: this module never imports Streamlit. Rendering lives in
views/models.py.
"""

from __future__ import annotations

import csv
import html
import io
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from core.comparison import (
    METRIC_DIRECTIONS,
    METRIC_LABELS,
    PRIMARY_METRIC,
    ComparisonResult,
)
from core.explain import ASSOCIATION_NOTE, explain
from core.models import FittedRun, ModelResult, TASK_LABELS

#: Cell written where a metric is not valid (undefined / failed model).
EMPTY_CELL = ""

#: The seed the ML layer uses unless a caller overrides it. Kept here so
#: the reproducibility metadata can report it without importing sklearn.
DEFAULT_RANDOM_STATE = 42

#: Baseline preprocessing settings, recorded verbatim for reproduction.
PREPROCESSING_NOTES: tuple[str, ...] = (
    "Categorical feature columns are one-hot encoded (pandas get_dummies, "
    "drop_first=True); numeric columns are used as-is.",
    "The one-hot vocabulary is fitted on the training rows only — a category "
    "seen only in the held-out rows never enters the feature matrix.",
    "Rows with a missing value in any numeric feature or in the target are "
    "removed before the split is taken.",
    "No scaling, no imputation and no row-level modification is applied.",
)

#: Everything the report/CSV must never contain, checked in tests.
_SECRET_HINTS = (
    "api_key",
    "apikey",
    "gemini_api_key",
    "secret",
    "token",
    "password",
    "passwd",
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """The configuration settings that make one run reproducible.

    Deliberately metadata only: no dataset values, no secrets and no
    fitted model. ``to_dict`` is JSON-safe so it can be stored alongside
    a run and exported as CSV or JSON.
    """

    task_type: str
    model_name: str
    feature_columns: list[str]
    target_column: str | None
    test_size: float
    random_state: int = DEFAULT_RANDOM_STATE
    n_clusters: int | None = None
    n_train: int | None = None
    n_test: int | None = None
    dropped_rows: int = 0
    preprocessing: list[str] = field(default_factory=lambda: list(PREPROCESSING_NOTES))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view of the configuration."""
        return {
            "task_type": self.task_type,
            "model_name": self.model_name,
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "n_clusters": self.n_clusters,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "dropped_rows": self.dropped_rows,
            "preprocessing": list(self.preprocessing),
        }


# ---------------------------------------------------------------------------
# Filenames
# ---------------------------------------------------------------------------


def safe_filename(stem: str, *, extension: str, max_length: int = 80) -> str:
    """Turn arbitrary text into a safe, sensible download filename.

    Special characters, spaces, path separators and non-ASCII letters are
    collapsed to ``-`` so the result is portable; an empty stem becomes
    ``"export"``. The extension is appended without its leading dot.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", (stem or "").strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-._")
    if not cleaned:
        cleaned = "export"
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip("-._")
    cleaned = cleaned.strip("-._") or "export"
    return f"{cleaned}.{extension.lstrip('.')}"


# ---------------------------------------------------------------------------
# Comparison exports (CSV)
# ---------------------------------------------------------------------------


def _metric_cell(value: float | None) -> str:
    """Format one metric value for CSV — empty when it is not valid."""
    if value is None:
        return EMPTY_CELL
    return f"{float(value):.6g}"


def comparison_csv(result: ComparisonResult) -> str:
    """CSV of the comparison table: one row per model.

    Valid metrics are numbers; a failed model carries its error and empty
    metric cells; an undefined metric (for example an invalid silhouette)
    is an empty cell, never a misleading zero. Direction hints are
    included in the header (for example ``RMSE (lower is better)``).
    """
    marks = {"higher": "higher is better", "lower": "lower is better"}
    header = ["Model", "Status", "Best model"]
    for metric in result.metrics:
        label = METRIC_LABELS.get(metric, metric)
        header.append(f"{label} ({marks[METRIC_DIRECTIONS[metric]]})")

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in result.table_rows():
        line = [
            row["model"],
            "OK" if row["ok"] else "failed",
            "yes" if row["best"] else "",
        ]
        for metric in result.metrics:
            line.append(_metric_cell(row["metrics"].get(metric)))
        writer.writerow(line)
    return buffer.getvalue()


def run_metadata_csv(result: ComparisonResult) -> str:
    """CSV of per-model run metadata: configuration, validity and errors.

    One row per compared model, with the task, model, target and feature
    columns, the shared split sizes and a status — including failed
    models, so nothing is hidden.
    """
    header = [
        "Model",
        "Task",
        "Status",
        "Target column",
        "Feature columns",
        "Feature count",
        "Train rows",
        "Test rows",
        "Dropped rows",
        "Error",
    ]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for outcome in result.outcomes:
        result_obj = outcome.result
        writer.writerow(
            [
                outcome.model_name,
                TASK_LABELS.get(result.task_type, result.task_type),
                "OK" if outcome.ok else "failed",
                (result_obj.target_column if result_obj else "") or "",
                ", ".join(result_obj.feature_columns) if result_obj else "",
                len(result_obj.feature_columns) if result_obj else "",
                result.n_train if result.n_train is not None else "",
                result.n_test if result.n_test is not None else "",
                result.dropped_rows,
                outcome.error or "",
            ]
        )
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Reproducibility metadata (CSV / JSON)
# ---------------------------------------------------------------------------


def config_from_result(
    result: ModelResult,
    *,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_clusters: int | None = None,
    n_train: int | None = None,
    n_test: int | None = None,
    dropped_rows: int = 0,
) -> RunConfig:
    """Capture the reproducibility settings for a single trained run."""
    return RunConfig(
        task_type=result.task_type,
        model_name=result.model_name,
        feature_columns=list(result.feature_columns),
        target_column=result.target_column,
        test_size=test_size,
        random_state=random_state,
        n_clusters=n_clusters if result.task_type == "clustering" else None,
        n_train=n_train,
        n_test=n_test,
        dropped_rows=dropped_rows,
    )


def config_from_comparison(result: ComparisonResult, run: FittedRun) -> RunConfig:
    """Capture the reproducibility settings for one model in a comparison."""
    return RunConfig(
        task_type=result.task_type,
        model_name=run.result.model_name,
        feature_columns=list(run.result.feature_columns),
        target_column=run.result.target_column,
        test_size=test_size_from_comparison(result),
        random_state=DEFAULT_RANDOM_STATE,
        n_clusters=(
            int(run.result.metrics["n_clusters"])
            if result.task_type == "clustering" and "n_clusters" in run.result.metrics
            else None
        ),
        n_train=result.n_train,
        n_test=result.n_test,
        dropped_rows=result.dropped_rows,
    )


def test_size_from_comparison(result: ComparisonResult) -> float:
    """Best-effort test size for a comparison (test rows / total rows).

    The engine records the split sizes rather than the fraction, so the
    fraction is recovered from them; when the sizes are unknown the
    default of 0.2 is reported instead of a guess.
    """
    if result.n_train and result.n_test:
        total = result.n_train + result.n_test
        if total:
            return round(result.n_test / total, 4)
    return 0.2


def config_csv(config: RunConfig) -> str:
    """CSV of one run's reproducibility configuration (key/value rows)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["Setting", "Value"])
    writer.writerow(["Task", TASK_LABELS.get(config.task_type, config.task_type)])
    writer.writerow(["Model", config.model_name])
    writer.writerow(["Target column", config.target_column or ""])
    writer.writerow(["Feature columns", ", ".join(config.feature_columns)])
    writer.writerow(["Feature count", len(config.feature_columns)])
    writer.writerow(["Test split", f"{config.test_size:.0%}"])
    writer.writerow(["Random seed", config.random_state])
    writer.writerow(["Number of clusters", config.n_clusters if config.n_clusters else ""])
    writer.writerow(["Train rows", config.n_train if config.n_train is not None else ""])
    writer.writerow(["Test rows", config.n_test if config.n_test is not None else ""])
    writer.writerow(["Dropped rows", config.dropped_rows])
    for index, note in enumerate(config.preprocessing, start=1):
        writer.writerow([f"Preprocessing {index}", note])
    return buffer.getvalue()


def config_json(config: RunConfig) -> str:
    """JSON of one run's reproducibility configuration."""
    return json.dumps(config.to_dict(), indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Prediction exports (CSV)
# ---------------------------------------------------------------------------


def predictions_frame(run: FittedRun) -> pd.DataFrame:
    """Held-out predictions for a supervised run, or cluster labels.

    * Supervised (classification/regression): one row per *held-out* row
      only, with the row identifier, the actual value and the prediction.
      The feature values themselves are never included.
    * Clustering: one row per scored row, with the identifier and the
      assigned cluster label (there is no actual value to compare).

    The row identifier is the index of the original dataset, so an
    exported prediction can be traced back to its source row without
    exporting the row itself.

    Raises ``ModelError`` when a supervised run is missing the test split
    it needs — an honest error beats an empty or misleading file.
    """
    from core.models import ModelError

    result = run.result
    if result.task_type == "clustering":
        labels = list(result.predictions)
        index = (
            list(run.X_train.index)
            if run.X_train is not None and len(run.X_train) == len(labels)
            else list(range(len(labels)))
        )
        return pd.DataFrame(
            {"Row": [str(i) for i in index], "Cluster": labels}
        )

    if run.X_test is None or run.y_test is None or not result.predictions:
        raise ModelError(
            "This run has no held-out predictions to export. Train the model "
            "on a dataset with enough usable rows, then export again."
        )

    actual = list(run.y_test)
    predicted = list(result.predictions)
    if len(actual) != len(predicted):
        raise ModelError(
            "Predictions could not be aligned with the held-out rows, so "
            "nothing was exported."
        )
    column = "Actual" if result.task_type == "regression" else "Actual class"
    pred_column = "Predicted" if result.task_type == "regression" else "Predicted class"
    return pd.DataFrame(
        {
            "Row": [str(i) for i in run.X_test.index],
            column: actual,
            pred_column: predicted,
        }
    )


def predictions_csv(run: FittedRun) -> str:
    """CSV of a run's prediction results (held-out rows only)."""
    frame = predictions_frame(run)
    return frame.to_csv(index=False, lineterminator="\n")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportContext:
    """Dataset and run context shown at the top of the report."""

    dataset_filename: str
    dataset_rows: int
    dataset_columns: int
    dataset_hash: str = ""
    generated_at: str = ""


def _svg_bar_chart(pairs: list[tuple[str, float]], *, metric_label: str) -> str:
    """A tiny self-contained horizontal bar chart (inline SVG, no JS).

    Values must be finite. Bars are scaled to the largest absolute value
    so a negative metric still renders sensibly.
    """
    if not pairs:
        return ""
    peak = max(abs(value) for _, value in pairs) or 1.0
    row_height = 26
    height = row_height * len(pairs) + 16
    label_width = 190
    bar_area = 320
    rows = []
    for index, (name, value) in enumerate(pairs):
        y = 8 + index * row_height
        width = max(2.0, abs(value) / peak * bar_area)
        colour = "#22d3ee" if value >= 0 else "#f87171"
        label = html.escape(name[:28])
        rows.append(
            f'<text x="0" y="{y + 13}" fill="#93a4bd" font-size="12">{label}</text>'
            f'<rect x="{label_width}" y="{y + 2}" width="{width:.1f}" height="14" '
            f'rx="3" fill="{colour}" />'
            f'<text x="{label_width + width + 6:.1f}" y="{y + 13}" fill="#e2e8f0" '
            f'font-size="12">{value:.4g}</text>'
        )
    return (
        f'<figure class="chart"><figcaption>{html.escape(metric_label)}</figcaption>'
        f'<svg viewBox="0 0 {label_width + bar_area + 80} {height}" width="100%" '
        f'role="img" aria-label="{html.escape(metric_label)}">'
        + "".join(rows)
        + "</svg></figure>"
    )


def build_report(
    *,
    context: ReportContext,
    config: RunConfig,
    metrics: dict[str, float],
    warnings: list[str],
    comparison: ComparisonResult | None = None,
    outcome_note: str | None = None,
    importances: dict[str, float] | None = None,
    importances_note: str | None = None,
    primary_metric: str | None = None,
    best_model: str | None = None,
) -> str:
    """Build a self-contained HTML report (string).

    The report is honest by construction: valid metrics and failed or
    undefined results are shown in separate, clearly labelled places,
    warnings are listed verbatim, and any feature-importance section
    carries the association-not-causation disclaimer.
    """
    generated = context.generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    task_label = TASK_LABELS.get(config.task_type, config.task_type)

    parts: list[str] = []
    parts.append(_report_head())
    parts.append(
        f"""
        <header class="hero">
          <p class="eyebrow">AI Data Science Copilot · analysis report</p>
          <h1>{html.escape(task_label)} report</h1>
          <p class="sub">Model: <strong>{html.escape(config.model_name)}</strong>
             {f'· target: <strong>{html.escape(config.target_column)}</strong>' if config.target_column else ''}
             · generated {html.escape(generated)}</p>
        </header>
        """
    )

    parts.append('<section><h2>Dataset context</h2><div class="kv">')
    parts.append(
        _kv("File", f"{html.escape(context.dataset_filename)}")
        + _kv("Rows", f"{context.dataset_rows:,}")
        + _kv("Columns", f"{context.dataset_columns:,}")
        + _kv("Dataset hash", html.escape(context.dataset_hash) or "—")
    )
    parts.append("</div><p class='muted'>Only aggregate context is included. No raw "
                 "dataset rows, cell values, API keys or model binaries are part of "
                 "this report.</p></section>")

    if outcome_note:
        parts.append(f'<section><div class="note warn">{html.escape(outcome_note)}</div></section>')

    parts.append("<section><h2>Model configuration &amp; reproducibility</h2>")
    parts.append(_config_table(config))
    parts.append("<ul class='prep'>")
    for note in config.preprocessing:
        parts.append(f"<li>{html.escape(note)}</li>")
    parts.append("</ul></section>")

    parts.append("<section><h2>Evaluation metrics</h2>")
    if metrics:
        parts.append(_metrics_table(metrics))
    else:
        parts.append('<div class="note warn">No valid metrics are available for this '
                     "run, so none are shown rather than reporting zero.</div>")
    if primary_metric and primary_metric in metrics:
        direction = METRIC_DIRECTIONS.get(primary_metric, "higher")
        parts.append(
            f'<p class="muted">Primary metric: <strong>'
            f"{html.escape(METRIC_LABELS.get(primary_metric, primary_metric))}</strong> "
            f"({html.escape(direction)} is better).</p>"
        )
    parts.append("</section>")

    if warnings:
        parts.append('<section><h2>Warnings</h2><ul class="warnings">')
        for warning in warnings:
            parts.append(f"<li>{html.escape(warning)}</li>")
        parts.append("</ul></section>")

    if comparison is not None:
        parts.append(_comparison_section(comparison))

    if importances:
        parts.append("<section><h2>Feature importance</h2>")
        parts.append(_svg_bar_chart(
            sorted(importances.items(), key=lambda kv: abs(kv[1]), reverse=True)[:12],
            metric_label="Importance",
        ))
        note = importances_note or ASSOCIATION_NOTE
        parts.append(f'<p class="muted">{html.escape(note)}</p></section>')

    parts.append(
        '<footer><p class="muted">Generated locally by AI Data Science Copilot. '
        "Every metric was computed by the app, not by a language model.</p></footer>"
    )
    parts.append("</body></html>")
    return "".join(parts)


def _comparison_section(result: ComparisonResult) -> str:
    """Valid-vs-failed comparison tables and a primary-metric chart."""
    marks = {"higher": "higher is better", "lower": "lower is better"}
    valid = [row for row in result.table_rows() if row["ok"]]
    failed = [row for row in result.table_rows() if not row["ok"]]

    parts = ["<section><h2>Model comparison</h2>"]
    parts.append(
        f'<p class="muted">{len(valid)} model(s) evaluated on one shared split · '
        f"{html.escape(TASK_LABELS.get(result.task_type, result.task_type))}. "
        + (
            f"Best by {html.escape(METRIC_LABELS.get(result.primary_metric, result.primary_metric))}: "
            f"<strong>{html.escape(result.best_model)}</strong>."
            if result.best_model
            else "No best model is highlighted (the evaluation is not valid)."
        )
        + "</p>"
    )

    if valid:
        parts.append('<table><thead><tr><th>Model</th>')
        for metric in result.metrics:
            label = METRIC_LABELS.get(metric, metric)
            parts.append(
                f"<th>{html.escape(label)} "
                f"<span class='dir'>({html.escape(marks[METRIC_DIRECTIONS[metric]])})</span></th>"
            )
        parts.append("<th>Best</th></tr></thead><tbody>")
        for row in valid:
            parts.append(f"<tr><td>{html.escape(row['model'])}</td>")
            for metric in result.metrics:
                value = row["metrics"].get(metric)
                cell = "—" if value is None else f"{value:.4g}"
                css = " class='undefined'" if value is None else ""
                parts.append(f"<td{css}>{cell}</td>")
            parts.append(f"<td>{'★' if row['best'] else ''}</td></tr>")
        parts.append("</tbody></table>")

        if not result.evaluation_valid:
            parts.append('<div class="note warn">The evaluation is not valid, so no '
                         "best model is declared. Undefined metrics are shown as — "
                         "rather than zero.</div>")

        chart_pairs = [
            (row["model"], row["metrics"][result.primary_metric])
            for row in valid
            if row["metrics"].get(result.primary_metric) is not None
        ]
        if chart_pairs:
            label = METRIC_LABELS.get(result.primary_metric, result.primary_metric)
            direction = METRIC_DIRECTIONS.get(result.primary_metric, "higher")
            # The bar length is the raw value, so the caption must say
            # which direction is better — otherwise a long RMSE bar
            # would read as "better" when it is in fact worse.
            parts.append(
                _svg_bar_chart(
                    chart_pairs,
                    metric_label=(
                        f"{label} by model "
                        f"(bar length = value, {direction} is better)"
                    ),
                )
            )

    if failed:
        parts.append('<h3>Failed models</h3><table><thead><tr>'
                     "<th>Model</th><th>Error</th></tr></thead><tbody>")
        for row in failed:
            parts.append(
                f"<tr><td>{html.escape(row['model'])}</td>"
                f"<td>{html.escape(row['error'] or 'failed')}</td></tr>"
            )
        parts.append("</tbody></table>")

    parts.append("</section>")
    return "".join(parts)


def _config_table(config: RunConfig) -> str:
    rows = [
        ("Task", TASK_LABELS.get(config.task_type, config.task_type)),
        ("Model", config.model_name),
        ("Target column", config.target_column or "—"),
        ("Feature columns", ", ".join(config.feature_columns) or "—"),
        ("Test split", f"{config.test_size:.0%}"),
        ("Random seed", str(config.random_state)),
        ("Number of clusters", str(config.n_clusters) if config.n_clusters else "—"),
        ("Train rows", f"{config.n_train:,}" if config.n_train is not None else "—"),
        ("Test rows", f"{config.n_test:,}" if config.n_test is not None else "—"),
        ("Dropped rows", f"{config.dropped_rows:,}"),
    ]
    body = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(value)}</td></tr>"
        for key, value in rows
    )
    return f"<table class='kv-table'><tbody>{body}</tbody></table>"


def _metrics_table(metrics: dict[str, float]) -> str:
    body = []
    for key, value in metrics.items():
        label = METRIC_LABELS.get(key, key.replace("_", " ").title())
        direction = METRIC_DIRECTIONS.get(key)
        hint = f" <span class='dir'>({direction} is better)</span>" if direction else ""
        body.append(
            f"<tr><th>{html.escape(label)}{hint}</th>"
            f"<td>{float(value):.4g}</td></tr>"
        )
    return f"<table class='kv-table'><tbody>{''.join(body)}</tbody></table>"


def _kv(label: str, value: str) -> str:
    return (
        f'<div class="kv-item"><div class="kv-label">{html.escape(label)}</div>'
        f'<div class="kv-value">{value}</div></div>'
    )


def _report_head() -> str:
    """Document head with an inline stylesheet (self-contained, offline)."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>AI Data Science Copilot — analysis report</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 2.5rem 1.5rem 4rem; background: #0b1220;
         color: #e2e8f0; font-family: "Segoe UI", system-ui, sans-serif;
         line-height: 1.6; }
  .hero, section, footer { max-width: 960px; margin: 0 auto; }
  .hero { border-bottom: 2px solid #22d3ee; padding-bottom: 1rem; margin-bottom: 1.5rem; }
  .eyebrow { color: #22d3ee; text-transform: uppercase; letter-spacing: 0.08em;
             font-size: 0.75rem; margin: 0; }
  h1 { font-size: 1.8rem; margin: 0.35rem 0; }
  h2 { font-size: 1.15rem; margin: 2rem 0 0.5rem; }
  h3 { font-size: 1rem; margin: 1.25rem 0 0.4rem; color: #f87171; }
  .sub, .muted { color: #93a4bd; font-size: 0.9rem; }
  section { margin-bottom: 1.5rem; }
  .kv { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem; margin: 0.5rem 0 0.75rem; }
  .kv-item { border-left: 1px solid #223252; padding-left: 0.75rem; }
  .kv-label { color: #6b7c99; font-size: 0.72rem; text-transform: uppercase;
              letter-spacing: 0.08em; }
  .kv-value { font-size: 1.05rem; font-weight: 600; }
  table { width: 100%; border-collapse: collapse; margin: 0.5rem 0;
          background: #111c33; border: 1px solid #223252; border-radius: 10px;
          overflow: hidden; font-size: 0.9rem; }
  th, td { text-align: left; padding: 0.5rem 0.7rem; border-bottom: 1px solid #223252; }
  thead th { background: #16233f; color: #93a4bd; font-weight: 600; }
  tbody tr:last-child th, tbody tr:last-child td { border-bottom: none; }
  td.undefined { color: #6b7c99; }
  .dir { color: #6b7c99; font-weight: 400; font-size: 0.8rem; }
  .kv-table th { width: 40%; color: #93a4bd; font-weight: 500; }
  .prep, .warnings { color: #93a4bd; font-size: 0.88rem; padding-left: 1.1rem; }
  .note { border: 1px solid #223252; border-left: 3px solid #fbbf24;
          background: rgba(251,191,36,0.08); border-radius: 8px; padding: 0.7rem 0.9rem; }
  figure.chart { margin: 0.75rem 0; background: #111c33; border: 1px solid #223252;
                 border-radius: 10px; padding: 1rem; }
  figcaption { color: #93a4bd; font-size: 0.85rem; margin-bottom: 0.5rem; }
  footer { border-top: 1px solid #223252; margin-top: 2rem; padding-top: 1rem; }
  @media print { body { background: #fff; color: #0b1220; } }
</style>
</head>
<body>
"""


# ---------------------------------------------------------------------------
# Convenience wrappers used by the view
# ---------------------------------------------------------------------------


def build_comparison_report(
    *,
    result: ComparisonResult,
    context: ReportContext,
) -> str:
    """Assemble the HTML report for a multi-model comparison.

    Picks the best (or first successful) model for the headline config and
    importance section, and always lists failed models and warnings.
    """
    successful = result.successful
    chosen: FittedRun | None = None
    if successful:
        chosen_name = (
            result.best_model
            if result.best_model in result.runs
            else successful[0].model_name
        )
        chosen = result.runs.get(chosen_name)

    config = (
        config_from_comparison(result, chosen)
        if chosen is not None
        else RunConfig(
            task_type=result.task_type,
            model_name=", ".join(result.model_names),
            feature_columns=list(result.feature_names),
            target_column=None,
            test_size=test_size_from_comparison(result),
            n_train=result.n_train,
            n_test=result.n_test,
            dropped_rows=result.dropped_rows,
        )
    )

    metrics: dict[str, float] = {}
    importances: dict[str, float] | None = None
    importances_note: str | None = None
    if chosen is not None:
        outcome = next(
            (o for o in successful if o.model_name == chosen.result.model_name), None
        )
        if outcome is not None:
            # Use the *validated* values: an undefined metric (e.g. an
            # invalid silhouette) is simply absent, never the display 0.
            for metric in result.metrics:
                value = result.metric_value(outcome, metric)
                if value is not None:
                    metrics[metric] = value
        for key, value in chosen.result.metrics.items():
            if key in ("n_clusters", "inertia") and isinstance(value, (int, float)):
                metrics.setdefault(key, float(value))
        candidate = explain(chosen)
        if candidate.available and candidate.importances:
            importances = candidate.importances
            importances_note = candidate.note
        else:
            importances_note = candidate.note

    outcome_note = None
    if not successful:
        outcome_note = (
            "No model completed successfully, so no metrics are reported. "
            "See the failed models below for the per-model errors."
        )

    return build_report(
        context=context,
        config=config,
        metrics=metrics,
        warnings=list(result.warnings),
        comparison=result,
        outcome_note=outcome_note,
        importances=importances,
        importances_note=importances_note,
        primary_metric=result.primary_metric,
        best_model=result.best_model,
    )


def build_single_report(
    *,
    result: ModelResult,
    run: FittedRun | None,
    context: ReportContext,
    config: RunConfig,
    importances: dict[str, float] | None = None,
    importances_note: str | None = None,
) -> str:
    """Assemble the HTML report for a single trained run."""
    from core.models import silhouette_valid

    metrics = {
        key: float(value)
        for key, value in result.metrics.items()
        if isinstance(value, (int, float))
    }
    if result.task_type == "clustering":
        # A silhouette that is not mathematically defined is stored as a
        # display 0 upstream; never let that reach the report as a metric.
        if not silhouette_valid(result.predictions):
            metrics.pop("silhouette", None)
    return build_report(
        context=context,
        config=config,
        metrics=metrics,
        warnings=list(result.caveats),
        comparison=None,
        importances=importances,
        importances_note=importances_note,
        primary_metric=PRIMARY_METRIC.get(result.task_type),
    )


def contains_secret_like(value: str) -> bool:
    """True when text looks like it may contain a secret (used by tests).

    Purely a safety net for exported strings: it flags the *names* of
    secret-like settings. The exporter itself never places such names or
    values into any artefact.
    """
    lowered = value.lower()
    return any(hint in lowered for hint in _SECRET_HINTS)

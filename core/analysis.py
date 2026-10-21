"""Deterministic analysis engine for the AI Assistant (Milestone 5).

Gemini only *decides what to calculate* by returning an
``AnalysisPlan``; every number the user sees is produced here, in
plain pandas, from the actual dataset. The input frame is read only —
never modified, never written anywhere.

Everything a plan can get wrong — an operation outside the supported
set, a column that does not exist, an aggregation that does not apply,
values that are not numbers — raises ``AnalysisError`` with a message
that is safe to show on screen, so the UI can explain the limitation
instead of inventing an answer.

Existing project utilities are reused wherever they exist: column
selection, summary statistics, correlation and time bucketing from
core/charts.py; metrics, column profiles and quality checks from
core/profiling.py.
"""

from __future__ import annotations

import datetime
import json
import math
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from core.ai_schemas import OPERATIONS, AnalysisPlan
from core.charts import (
    AGGREGATIONS,
    CORRELATION_METHODS,
    PERIODS,
    ChartError,
    bucketed_series,
    categorical_columns,
    correlation_matrix,
    datetime_columns,
    numeric_columns,
    summary_stats,
)
from core.profiling import column_profiles, dataset_metrics, quality_issues

#: How many rows/groups the plan may request (the schema enforces the
#: same bounds; this guards against plans built without validation).
DEFAULT_LIMIT = 10
MAX_LIMIT = 50

#: Rows shown in chat-sized tables for potentially long results.
MAX_TABLE_ROWS = 60

#: Human labels for aggregations, used in headlines and table headers.
AGG_LABELS = {
    "mean": "Mean",
    "sum": "Total",
    "min": "Minimum",
    "max": "Maximum",
    "count": "Count",
}


class AnalysisError(ValueError):
    """A plan or data problem the UI can show as a friendly message."""


@dataclass(frozen=True)
class AnalysisResult:
    """One executed analysis: how to display it plus JSON-safe facts.

    ``table``/``kpis`` are what the user sees; ``facts`` is the small,
    aggregate-only payload handed to Gemini for the explanation step.
    ``caveats`` are deterministic — computed by this module, not
    written by the model.
    """

    operation: str
    headline: str
    table: pd.DataFrame | None = None
    kpis: tuple[dict, ...] = ()
    facts: dict = field(default_factory=dict)
    caveats: tuple[str, ...] = ()


# --- Small helpers -----------------------------------------------------------


def _json_safe(value: object) -> object:
    """Any cell value → JSON-serialisable Python. Never raises."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        try:
            return value.isoformat()
        except (AttributeError, ValueError):
            return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return int(value)
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):  # array-likes and exotic objects
        pass
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (AttributeError, ValueError, TypeError):
            pass
    return str(value)


def _require_column(frame: pd.DataFrame, name: object, role: str) -> str:
    """Validate that ``name`` is a real column; return it as str."""
    if name is None or (isinstance(name, str) and not name.strip()):
        raise AnalysisError(f"The plan is missing the {role} needed for this operation.")
    column = str(name)
    if column not in frame.columns:
        available = ", ".join(str(c) for c in frame.columns[:12])
        if len(frame.columns) > 12:
            available += f" and {len(frame.columns) - 12} more"
        raise AnalysisError(
            f'Column "{column}" does not exist in this dataset. '
            f"Available columns: {available}."
        )
    return column


def _require_operation(plan: AnalysisPlan) -> None:
    if plan.operation == "cannot_answer":
        raise AnalysisError(
            "This plan marks the question as not answerable with the "
            "supported operations."
        )
    if plan.operation not in OPERATIONS:
        raise AnalysisError(
            f'Unknown operation "{plan.operation}". Supported operations: '
            f"{', '.join(OPERATIONS)}."
        )


def _require_aggregation(plan: AnalysisPlan) -> str:
    aggregation = plan.aggregation
    if aggregation is None:
        raise AnalysisError(
            f"The plan is missing the aggregation required by the "
            f"'{plan.operation}' operation."
        )
    if aggregation not in AGGREGATIONS:
        raise AnalysisError(
            f'Unknown aggregation "{aggregation}". '
            f"Supported: {', '.join(AGGREGATIONS)}."
        )
    return aggregation


def _require_plan_column(plan: AnalysisPlan, role: str) -> str:
    value = getattr(plan, role)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise AnalysisError(
            f"The plan is missing the {role} required by the "
            f"'{plan.operation}' operation."
        )
    return str(value)


def _limit(plan: AnalysisPlan) -> int:
    value = plan.limit if plan.limit is not None else DEFAULT_LIMIT
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_LIMIT:
        raise AnalysisError(f"limit must be a whole number between 1 and {MAX_LIMIT}.")
    return value


def _order(plan: AnalysisPlan, default: str = "desc") -> bool:
    """Sort direction → pandas ``ascending`` flag."""
    if plan.order is not None and plan.order not in ("desc", "asc"):
        raise AnalysisError(f'Unknown order "{plan.order}". Use "desc" or "asc".')
    return (plan.order or default) == "asc"


def _numeric_values(frame: pd.DataFrame, column: str) -> pd.Series:
    """Numbers in a column, or a clear error when there are none."""
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.notna().sum() == 0:
        raise AnalysisError(
            f'Column "{column}" has no numeric values to work with — this '
            f"analysis needs numbers."
        )
    return values


def _missing_caveat(series: pd.Series, column: str, used: int) -> str | None:
    missing = int(series.isna().sum())
    if not missing:
        return None
    total = len(series)
    return (
        f"{missing:,} of {total:,} values in {column} are missing "
        f"({missing / total * 100:.1f}%); calculations use the remaining {used:,}."
        if total
        else None
    )


# --- Operations ---------------------------------------------------------------


def _op_overview(frame: pd.DataFrame, plan: AnalysisPlan) -> AnalysisResult:
    """Headline facts about the whole dataset (no plan fields needed)."""
    metrics = dataset_metrics(frame)
    numeric = numeric_columns(frame)
    cats = categorical_columns(frame)
    dates = datetime_columns(frame)
    cells = metrics.rows * metrics.columns
    missing_pct = round(metrics.missing_cells / cells * 100, 2) if cells else 0.0

    caveats: list[str] = []
    if metrics.missing_cells:
        caveats.append(f"{missing_pct:g}% of all cells are missing.")
    if metrics.duplicate_rows:
        caveats.append(
            f"{metrics.duplicate_rows:,} duplicate rows exist; nothing was removed."
        )
    if not frame.empty and not numeric:
        caveats.append("No numeric columns were detected.")

    return AnalysisResult(
        operation="overview",
        headline="Dataset overview",
        kpis=(
            {"label": "Rows", "value": f"{metrics.rows:,}", "help": "Data rows in the file."},
            {"label": "Columns", "value": f"{metrics.columns:,}", "help": "Columns after safe name cleanup."},
            {"label": "Missing cells", "value": f"{metrics.missing_cells:,}", "help": "Empty cells across the dataset."},
            {"label": "Duplicate rows", "value": f"{metrics.duplicate_rows:,}", "help": "Rows repeating an earlier row exactly."},
        ),
        facts={
            "rows": metrics.rows,
            "columns": metrics.columns,
            "missing_cells": metrics.missing_cells,
            "missing_percent": missing_pct,
            "duplicate_rows": metrics.duplicate_rows,
            "column_names": [str(name) for name in frame.columns],
            "numeric_columns": numeric,
            "categorical_columns": cats,
            "datetime_columns": dates,
        },
        caveats=tuple(caveats),
    )


def _op_data_quality(frame: pd.DataFrame, plan: AnalysisPlan) -> AnalysisResult:
    """Reuses the Data page's profiler: issues plus missing-by-column."""
    issues = quality_issues(frame)
    profiles = column_profiles(frame)
    metrics = dataset_metrics(frame)

    issue_table = pd.DataFrame(
        [
            {
                "Severity": issue.severity.capitalize(),
                "Issue": issue.category,
                "Detail": issue.detail,
            }
            for issue in issues
        ],
        columns=["Severity", "Issue", "Detail"],
    )
    missing_by_column = {
        str(row["Column"]): float(row["Missing %"])
        for row in profiles.to_dict("records")
        if row["Missing"] > 0
    }

    caveats = ("Reported only — no values were imputed, dropped or changed.",)
    if not issues:
        caveats = ("No common data-quality issues were detected.",)

    return AnalysisResult(
        operation="data_quality",
        headline="Data-quality report",
        table=issue_table,
        kpis=(
            {
                "label": "Columns with missing",
                "value": f"{len(missing_by_column):,}",
                "help": "Columns containing at least one empty cell.",
            },
            {"label": "Duplicate rows", "value": f"{metrics.duplicate_rows:,}"},
            {"label": "Issues found", "value": f"{len(issues):,}"},
        ),
        facts={
            "issues": [
                {"severity": issue.severity, "category": issue.category, "detail": issue.detail}
                for issue in issues
            ],
            "missing_by_column": missing_by_column,
            "duplicate_rows": metrics.duplicate_rows,
        },
        caveats=caveats,
    )


def _op_column_stats(frame: pd.DataFrame, plan: AnalysisPlan) -> AnalysisResult:
    """Numeric statistics, datetime range, or top values — by dtype."""
    column = _require_plan_column(plan, "value_column")
    _require_column(frame, column, "value_column")
    series = frame[column]
    missing = int(series.isna().sum())
    missing_pct = round(missing / len(series) * 100, 1) if len(series) else 0.0

    if pd.api.types.is_datetime64_any_dtype(series):
        non_null = series.dropna()
        if non_null.empty:
            raise AnalysisError(f'Column "{column}" has no dates to summarise.')
        first, last = non_null.min(), non_null.max()
        return AnalysisResult(
            operation="column_stats",
            headline=f"Date range of {column}",
            kpis=(
                {"label": "First date", "value": str(_json_safe(first))[:10]},
                {"label": "Last date", "value": str(_json_safe(last))[:10]},
                {"label": "Distinct dates", "value": f"{int(non_null.nunique()):,}"},
            ),
            facts={
                "column": column,
                "first": _json_safe(first),
                "last": _json_safe(last),
                "distinct": int(non_null.nunique()),
                "missing": missing,
            },
            caveats=tuple(
                c for c in (_missing_caveat(series, column, len(non_null)),) if c
            ),
        )

    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        try:
            stats = summary_stats(frame, column)
        except ChartError as exc:
            raise AnalysisError(str(exc)) from None
        caveats: list[str] = []
        missing_caveat = _missing_caveat(series, column, stats["count"])
        if missing_caveat:
            caveats.append(missing_caveat)
        if stats["count"] < 30:
            caveats.append(
                f"Only {stats['count']:,} non-missing values — treat the "
                f"summary statistics with care."
            )
        return AnalysisResult(
            operation="column_stats",
            headline=f"Statistics for {column}",
            kpis=(
                {"label": "Count", "value": f"{stats['count']:,}", "help": "Rows with a value in this column."},
                {"label": "Mean", "value": f"{stats['mean']:,.4g}"},
                {"label": "Median", "value": f"{stats['median']:,.4g}"},
                {"label": "Std dev", "value": f"{stats['std']:,.4g}", "help": "Spread of the values around the mean."},
                {"label": "Min", "value": f"{stats['min']:,.4g}"},
                {"label": "Max", "value": f"{stats['max']:,.4g}"},
            ),
            facts={
                "column": column,
                "count": stats["count"],
                "mean": _json_safe(stats["mean"]),
                "median": _json_safe(stats["median"]),
                "std": _json_safe(stats["std"]),
                "min": _json_safe(stats["min"]),
                "max": _json_safe(stats["max"]),
                "missing": missing,
                "missing_percent": missing_pct,
            },
            caveats=tuple(caveats),
        )

    # Text, boolean and category columns: distinct values and the most frequent.
    non_null = series.dropna()
    if non_null.empty:
        raise AnalysisError(f'Column "{column}" has no values to summarise.')
    counts = non_null.value_counts()
    top = counts.head(10)
    table = pd.DataFrame(
        {
            "Value": [str(value) for value in top.index],
            "Rows": [int(count) for count in top.values],
            "Share %": [round(int(count) / int(len(non_null)) * 100, 1) for count in top.values],
        }
    )
    unique = int(non_null.nunique())
    caveats = []
    missing_caveat = _missing_caveat(series, column, len(non_null))
    if missing_caveat:
        caveats.append(missing_caveat)
    if unique > len(top):
        caveats.append(
            f"{unique:,} distinct values; the table shows the {len(top)} most "
            f"frequent."
        )

    return AnalysisResult(
        operation="column_stats",
        headline=f"Most frequent values in {column}",
        table=table,
        kpis=(
            {"label": "Distinct values", "value": f"{unique:,}"},
            {"label": "Non-missing rows", "value": f"{len(non_null):,}"},
            {"label": "Missing", "value": f"{missing:,}"},
        ),
        facts={
            "column": column,
            "distinct": unique,
            "missing": missing,
            "missing_percent": missing_pct,
            "top_values": [
                {
                    "value": str(value),
                    "count": int(count),
                    "share_percent": round(int(count) / int(len(non_null)) * 100, 1),
                }
                for value, count in top.items()
            ],
        },
        caveats=tuple(caveats),
    )


def _op_group_aggregate(frame: pd.DataFrame, plan: AnalysisPlan) -> AnalysisResult:
    """Group-by aggregation, ranked by the aggregated value."""
    group_column = _require_plan_column(plan, "group_by")
    value_column = _require_plan_column(plan, "value_column")
    _require_column(frame, group_column, "group_by")
    _require_column(frame, value_column, "value_column")
    aggregation = _require_aggregation(plan)
    if group_column == value_column:
        raise AnalysisError("group_by and value_column must be two different columns.")
    limit = _limit(plan)
    ascending = _order(plan)

    values = _numeric_values(frame, value_column)
    # Internal names avoid a collision when a column is literally called "value".
    work = pd.DataFrame({"g": frame[group_column].to_numpy(), "v": values.to_numpy()})
    missing_groups = int(work["g"].isna().sum())
    work = work.dropna(subset=["g"])
    if work.empty:
        raise AnalysisError(f'No rows have a usable value in "{group_column}" to group by.')

    aggregated = work.groupby("g", sort=False, dropna=True, observed=True)["v"].agg(aggregation)
    groups_total = int(len(aggregated))
    ranked = aggregated.sort_values(ascending=ascending)
    shown = ranked.head(limit)

    table = pd.DataFrame(
        {
            group_column: list(shown.index),
            f"{AGG_LABELS[aggregation]} of {value_column}": [float(value) for value in shown.values],
        }
    )

    caveats: list[str] = []
    if groups_total > len(shown):
        caveats.append(
            f"Showing {len(shown):,} of {groups_total:,} groups — ask for a "
            f"larger limit to see more."
        )
    if missing_groups:
        caveats.append(
            f"{missing_groups:,} rows have no value in {group_column} and were "
            f"excluded from the grouping."
        )
    value_missing = int(values.isna().sum())
    if value_missing:
        caveats.append(
            f"{value_missing:,} rows have no number in {value_column}; they are "
            f"ignored by the {aggregation} aggregation."
        )

    return AnalysisResult(
        operation="group_aggregate",
        headline=f"{AGG_LABELS[aggregation]} of {value_column} by {group_column}",
        table=table,
        kpis=(
            {"label": "Groups", "value": f"{groups_total:,}", "help": "Distinct groups after ignoring missing keys."},
            {"label": "Rows grouped", "value": f"{len(work):,}", "help": "Rows that carried both a group key and a number."},
        ),
        facts={
            "group_by": group_column,
            "value_column": value_column,
            "aggregation": aggregation,
            "order": "asc" if ascending else "desc",
            "groups_total": groups_total,
            "groups_returned": int(len(shown)),
            "rows_grouped": int(len(work)),
            "results": [
                {"group": _json_safe(group), "value": _json_safe(value)}
                for group, value in shown.items()
            ],
        },
        caveats=tuple(caveats),
    )


def _op_top_n(frame: pd.DataFrame, plan: AnalysisPlan) -> AnalysisResult:
    """The N rows with the highest (or lowest) values of a numeric column."""
    sort_column = _require_plan_column(plan, "value_column")
    _require_column(frame, sort_column, "value_column")
    limit = _limit(plan)
    ascending = _order(plan)

    values = _numeric_values(frame, sort_column)
    ranked_index = values.dropna().sort_values(ascending=ascending).head(limit).index
    excluded = int(values.isna().sum())

    columns = [str(name) for name in frame.columns]
    projection_note: str | None = None
    if plan.columns:
        columns = []
        for name in plan.columns:
            columns.append(_require_column(frame, name, "columns"))
        columns = list(dict.fromkeys(columns))
    elif len(frame.columns) > 12:
        keep = [sort_column] + [name for name in frame.columns if name != sort_column]
        columns = [str(name) for name in keep[:12]]
        projection_note = (
            f"The dataset has {len(frame.columns)} columns; the table shows "
            f"{len(columns)} of them."
        )

    display = frame.loc[ranked_index, columns]

    caveats: list[str] = []
    if excluded:
        caveats.append(
            f"{excluded:,} rows have no number in {sort_column} and were not ranked."
        )
    if projection_note:
        caveats.append(projection_note)
    if len(display) < limit:
        caveats.append(f"The dataset only has {len(display):,} rows to rank.")

    position = "lowest" if ascending else "highest"
    return AnalysisResult(
        operation="top_n",
        headline=f"Top {len(display)} rows by {sort_column} ({position} first)",
        table=display,
        kpis=(
            {"label": "Rows ranked", "value": f"{int(values.notna().sum()):,}", "help": "Rows with a number in the sort column."},
            {"label": "Shown", "value": f"{len(display):,}"},
        ),
        facts={
            "sort_column": sort_column,
            "order": "asc" if ascending else "desc",
            "rows": [
                {str(column): _json_safe(value) for column, value in row.items()}
                for row in display.to_dict("records")
            ],
        },
        caveats=tuple(caveats),
    )


def _op_correlation(frame: pd.DataFrame, plan: AnalysisPlan) -> AnalysisResult:
    """Correlation across numeric columns, or for one requested pair."""
    method = plan.method or "pearson"
    if method not in CORRELATION_METHODS:
        raise AnalysisError(
            f'Unknown correlation method "{method}". '
            f"Supported: {', '.join(CORRELATION_METHODS)}."
        )
    pair: list[str] | None = None
    if plan.columns:
        if len(plan.columns) != 2:
            raise AnalysisError(
                "A specific correlation needs exactly two columns in `columns`."
            )
        pair = [_require_column(frame, name, "columns") for name in plan.columns]

    try:
        matrix = correlation_matrix(frame, method)
    except ChartError as exc:
        raise AnalysisError(str(exc)) from None

    if pair:
        missing = [name for name in pair if name not in matrix.columns]
        if missing:
            raise AnalysisError(
                f"Column(s) {', '.join(missing)} cannot be correlated — "
                f"correlation needs numeric columns that are not constant or empty."
            )
        matrix = matrix.loc[pair, pair]

    pairs: list[dict] = []
    labels = [str(name) for name in matrix.columns]
    for row in range(len(labels)):
        for col in range(row + 1, len(labels)):
            value = matrix.iloc[row, col]
            if pd.isna(value):
                continue
            pairs.append(
                {"a": labels[row], "b": labels[col], "r": round(float(value), 3)}
            )
    if not pairs:
        raise AnalysisError(
            "No pairs of numeric columns share enough rows to correlate."
        )
    pairs.sort(key=lambda item: abs(item["r"]), reverse=True)
    shown = pairs[:10]

    table = pd.DataFrame(
        {"Columns": [f"{item['a']} vs {item['b']}" for item in shown], "r": [item["r"] for item in shown]}
    )
    strongest = shown[0]

    if pair:
        headline = f"{method.capitalize()} correlation: {pair[0]} vs {pair[1]}"
    else:
        headline = f"{method.capitalize()} correlation across {len(matrix)} numeric columns"

    caveats = (
        "Correlation measures association, not causation.",
        f"{method.capitalize()} correlation on pairwise-complete rows; "
        f"constant or all-empty columns are excluded.",
    )
    return AnalysisResult(
        operation="correlation",
        headline=headline,
        table=table,
        kpis=(
            {"label": "Numeric columns", "value": f"{len(matrix):,}"},
            {
                "label": "Strongest |r|",
                "value": f"{abs(strongest['r']):.3f}",
                "help": f"{strongest['a']} vs {strongest['b']}",
            },
        ),
        facts={
            "method": method,
            "columns_used": labels,
            "pairs": shown,
        },
        caveats=caveats,
    )


def _op_timeseries(frame: pd.DataFrame, plan: AnalysisPlan) -> AnalysisResult:
    """Value aggregated over time — same bucketing as the Charts page."""
    date_column = _require_plan_column(plan, "date_column")
    value_column = _require_plan_column(plan, "value_column")
    _require_column(frame, date_column, "date_column")
    _require_column(frame, value_column, "value_column")
    aggregation = _require_aggregation(plan)
    period = plan.period or "auto"
    if period not in PERIODS:
        raise AnalysisError(
            f'Unknown time bucket "{period}". Supported: {", ".join(PERIODS)}.'
        )

    try:
        bucketed, bucket_label = bucketed_series(
            frame, date_column, value_column, aggregation, period
        )
    except ChartError as exc:
        raise AnalysisError(str(exc)) from None

    caveats: list[str] = []
    shown = bucketed
    if len(bucketed) > MAX_TABLE_ROWS:
        shown = bucketed.tail(MAX_TABLE_ROWS)
        caveats.append(
            f"{len(bucketed):,} {bucket_label} buckets in total — the table "
            f"shows the most recent {MAX_TABLE_ROWS}."
        )
    values_missing = int(
        pd.to_numeric(frame[value_column], errors="coerce").isna().sum()
    )
    if values_missing:
        caveats.append(
            f"{values_missing:,} rows have no number in {value_column} and are "
            f"excluded from every bucket."
        )
    empty_note = (
        "empty buckets show as zero for sum and count"
        if aggregation in ("sum", "count")
        else "empty buckets stay as gaps"
    )
    caveats.append(f"Values are aggregated {bucket_label}; {empty_note}.")

    table = shown.rename_axis("Bucket").reset_index()
    table.columns = [f"{date_column} ({bucket_label})", f"{AGG_LABELS[aggregation]} of {value_column}"]

    return AnalysisResult(
        operation="timeseries",
        headline=f"{AGG_LABELS[aggregation]} of {value_column}, bucketed {bucket_label}",
        table=table,
        kpis=(
            {"label": "Buckets", "value": f"{len(bucketed):,}", "help": "Time buckets with at least one usable row."},
            {"label": "First bucket", "value": str(_json_safe(bucketed.index.min()))[:16]},
            {"label": "Last bucket", "value": str(_json_safe(bucketed.index.max()))[:16]},
        ),
        facts={
            "date_column": date_column,
            "value_column": value_column,
            "aggregation": aggregation,
            "bucket": bucket_label,
            "buckets": int(len(bucketed)),
            "first_bucket": _json_safe(bucketed.index.min()),
            "last_bucket": _json_safe(bucketed.index.max()),
            "min": _json_safe(float(bucketed.min())),
            "max": _json_safe(float(bucketed.max())),
            "mean": _json_safe(float(bucketed.mean())),
            "series": [
                {"bucket": _json_safe(index), "value": _json_safe(value)}
                for index, value in shown.items()
            ],
        },
        caveats=tuple(caveats),
    )


_HANDLERS: dict[str, Callable[[pd.DataFrame, AnalysisPlan], AnalysisResult]] = {
    "overview": _op_overview,
    "data_quality": _op_data_quality,
    "column_stats": _op_column_stats,
    "group_aggregate": _op_group_aggregate,
    "top_n": _op_top_n,
    "correlation": _op_correlation,
    "timeseries": _op_timeseries,
}


def execute_plan(frame: pd.DataFrame, plan: AnalysisPlan) -> AnalysisResult:
    """Validate a plan against the dataset and run it deterministically.

    Raises ``AnalysisError`` (never crashes, never guesses) for anything
    the plan or the data cannot support.
    """
    if not isinstance(plan, AnalysisPlan):
        raise TypeError("plan must be an AnalysisPlan")
    _require_operation(plan)
    handler = _HANDLERS.get(plan.operation)
    if handler is None:  # cannot_answer is handled by the caller
        raise AnalysisError(
            f'Operation "{plan.operation}" is declared but not implemented.'
        )
    return handler(frame, plan)


def compact_json(obj: object, max_chars: int = 4000, max_list: int = 15) -> str:
    """Small, prompt-safe JSON: lists trimmed, strings capped.

    Keeps the aggregate payload sent to Gemini to a predictable size —
    only approved aggregates ever reach this function, but it still
    never ships more than the explainer needs.
    """

    def shrink(value: object) -> object:
        if isinstance(value, list):
            trimmed = [shrink(item) for item in value[:max_list]]
            if len(value) > max_list:
                trimmed.append(f"… {len(value) - max_list} more")
            return trimmed
        if isinstance(value, dict):
            return {str(key): shrink(item) for key, item in value.items()}
        if isinstance(value, str) and len(value) > 200:
            return value[:199] + "…"
        return value

    text = json.dumps(shrink(obj), ensure_ascii=False, default=str)
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return text

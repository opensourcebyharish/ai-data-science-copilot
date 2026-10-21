"""Pure, non-mutating profiling of a loaded DataFrame.

Every function takes a DataFrame and returns new objects; the input
frame is never modified. Profiling only *reports* issues — it never
cleans, imputes or drops anything.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetMetrics:
    """Headline numbers shown as KPI cards."""

    rows: int
    columns: int
    missing_cells: int
    duplicate_rows: int
    memory_bytes: int


@dataclass(frozen=True)
class QualityIssue:
    """One reported (never automatically fixed) data-quality finding.

    Severity is "high", "medium" or "low".
    """

    severity: str
    category: str
    detail: str


def dataset_metrics(frame: pd.DataFrame) -> DatasetMetrics:
    """Count rows, columns, missing cells, duplicate rows and memory."""
    return DatasetMetrics(
        rows=int(frame.shape[0]),
        columns=int(frame.shape[1]),
        missing_cells=int(frame.isna().sum().sum()),
        duplicate_rows=int(frame.duplicated().sum()),
        memory_bytes=int(frame.memory_usage(deep=True).sum()),
    )


def column_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-column type, non-null, missing, missing percentage and uniques."""
    total = len(frame)
    rows = []
    for name in frame.columns:
        series = frame[name]
        non_null = int(series.notna().sum())
        missing = total - non_null
        rows.append(
            {
                "Column": str(name),
                "Type": str(series.dtype),
                "Non-null": non_null,
                "Missing": missing,
                "Missing %": round(missing / total * 100, 1) if total else 0.0,
                "Unique": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(
        rows, columns=["Column", "Type", "Non-null", "Missing", "Missing %", "Unique"]
    )


def numeric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe() for numeric columns, transposed with a Column column."""
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    summary = numeric.describe().T
    summary.index.name = "Column"
    return summary.round(3).reset_index()


def _truncate(value: object, limit: int = 40) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def categorical_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Useful summaries for text, boolean and category columns."""
    rows = []
    for name in frame.columns:
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue

        non_null = series.dropna()
        top_value = None
        top_count = 0
        if not non_null.empty:
            try:
                modes = non_null.mode()
                top_value = modes.iloc[0] if not modes.empty else None
            except (TypeError, ValueError):
                top_value = None
            if top_value is not None:
                top_count = int((non_null == top_value).sum())

        rows.append(
            {
                "Column": str(name),
                "Unique": int(non_null.nunique()),
                "Top value": _truncate(top_value) if top_value is not None else "—",
                "Top count": top_count,
                "Top %": round(top_count / len(non_null) * 100, 1) if len(non_null) else 0.0,
            }
        )
    return pd.DataFrame(
        rows, columns=["Column", "Unique", "Top value", "Top count", "Top %"]
    )


def _join_names(names: list[str], limit: int = 8) -> str:
    shown = ", ".join(names[:limit])
    if len(names) > limit:
        shown += f" and {len(names) - limit} more"
    return shown


def _looks_numeric(series: pd.Series) -> bool:
    """True for text columns whose values are mostly numbers."""
    if series.empty:
        return False
    converted = pd.to_numeric(series, errors="coerce")
    return bool(converted.notna().mean() >= 0.9)


def quality_issues(frame: pd.DataFrame) -> list[QualityIssue]:
    """Report common data-quality problems without changing anything."""
    total = len(frame)
    if total == 0:
        return [
            QualityIssue(
                severity="low",
                category="Empty dataset",
                detail="The file has column headers but no rows.",
            )
        ]

    issues: list[QualityIssue] = []

    duplicate_rows = int(frame.duplicated().sum())
    if duplicate_rows:
        issues.append(
            QualityIssue(
                severity="medium",
                category="Duplicate rows",
                detail=f"{duplicate_rows:,} row(s) repeat an earlier row exactly.",
            )
        )

    missing_columns: list[str] = []
    empty_columns: list[str] = []
    constant_columns: list[str] = []
    mostly_missing: list[str] = []
    numeric_text: list[str] = []

    for name in frame.columns:
        series = frame[name]
        missing = int(series.isna().sum())
        non_null = series.dropna()

        if missing:
            missing_columns.append(str(name))

        if non_null.empty:
            empty_columns.append(str(name))
            continue

        if non_null.nunique() <= 1:
            constant_columns.append(str(name))

        if missing / total > 0.5:
            mostly_missing.append(str(name))

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ) and _looks_numeric(non_null):
            numeric_text.append(str(name))

    if missing_columns:
        issues.append(
            QualityIssue(
                severity="medium",
                category="Columns with missing values",
                detail=_join_names(missing_columns),
            )
        )
    if mostly_missing:
        issues.append(
            QualityIssue(
                severity="high",
                category="Columns over 50% missing",
                detail=_join_names(mostly_missing),
            )
        )
    if empty_columns:
        issues.append(
            QualityIssue(
                severity="high",
                category="Empty columns (all values missing)",
                detail=_join_names(empty_columns),
            )
        )
    if constant_columns:
        issues.append(
            QualityIssue(
                severity="low",
                category="Constant columns",
                detail=f"{_join_names(constant_columns)} — every row shares the same value.",
            )
        )
    if numeric_text:
        issues.append(
            QualityIssue(
                severity="low",
                category="Numeric-looking text columns",
                detail=f"{_join_names(numeric_text)} — stored as text; converting may be useful.",
            )
        )

    return issues

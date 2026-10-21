"""Framework-free Plotly figure builders for the Charts page (Milestone 4).

Every public function takes a DataFrame and returns a new
``plotly.graph_objects.Figure``. The input frame is never modified —
plotting only reads it. Colours and surfaces come from the design
tokens in ui/styles.css and .streamlit/config.toml so charts belong
to the same theme as the rest of the app.

Anything a user could get wrong (unknown column, wrong type, too
little data) raises ChartError with a message that is safe to show
on screen.
"""

import warnings

import pandas as pd
import plotly.graph_objects as go

#: Series colours in design-token order: primary (cyan), secondary
#: (violet), then the semantic good / warn / bad / neutral colours.
CHART_COLORS: tuple[str, ...] = (
    "#22d3ee",
    "#a78bfa",
    "#34d399",
    "#fbbf24",
    "#f87171",
    "#94a3b8",
)

#: Layout colours, also taken from the :root design tokens.
PAGE_BACKGROUND = "#0b1220"
PLOT_BACKGROUND = "#111c33"
GRID_COLOR = "#223252"
TEXT_COLOR = "#e2e8f0"
MUTED_TEXT_COLOR = "#93a4bd"

#: Accepted options for the correlation and time-series controls.
CORRELATION_METHODS: tuple[str, ...] = ("pearson", "spearman")
AGGREGATIONS: tuple[str, ...] = ("mean", "sum", "min", "max", "count")
PERIODS: tuple[str, ...] = ("auto", "hour", "day", "week", "month")

#: Resample frequency alias per period choice, and a human label per alias.
PERIOD_FREQ = {"hour": "h", "day": "D", "week": "W", "month": "ME"}
FREQ_LABELS = {"h": "hourly", "D": "daily", "W": "weekly", "ME": "monthly"}

#: How many groups a categorical scatter colour column may contain.
MAX_COLOR_GROUPS = 12

#: Correlation heatmaps stop drawing numbers above this matrix size.
MAX_ANNOTATED_COLUMNS = 12


class ChartError(ValueError):
    """A chart problem the UI can show as a friendly message."""


# --- Column selection ------------------------------------------------------


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Columns that are numeric by dtype (booleans count as categories)."""
    return [
        str(name)
        for name in frame.columns
        if pd.api.types.is_numeric_dtype(frame[name])
        and not pd.api.types.is_bool_dtype(frame[name])
    ]


def categorical_columns(frame: pd.DataFrame) -> list[str]:
    """Columns usable as group labels: text, category or boolean."""
    names = []
    for name in frame.columns:
        series = frame[name]
        if pd.api.types.is_bool_dtype(series):
            names.append(str(name))
        elif pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(
            series
        ):
            continue
        else:
            names.append(str(name))
    return names


def _mostly_numeric(series: pd.Series) -> bool:
    """True when at least 90% of the non-missing values are numbers."""
    non_null = series.dropna()
    if non_null.empty:
        return False
    converted = pd.to_numeric(non_null, errors="coerce")
    return bool(converted.notna().mean() >= 0.9)


def _try_datetime(series: pd.Series, **kwargs) -> pd.Series:
    """pd.to_datetime for arbitrary user data.

    Probing unknown columns makes pandas fall back to dateutil and
    warn about the format; that fallback is exactly what this module
    wants, so the expected warning is silenced.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not infer format")
        return pd.to_datetime(series, errors="coerce", **kwargs)


def _parseable_share(sample: pd.Series) -> float:
    """Share of values that parse as dates (never raises)."""
    for kwargs in ({}, {"utc": True}):
        try:
            parsed = _try_datetime(sample, **kwargs)
        except (ValueError, TypeError):
            continue
        return float(parsed.notna().mean())
    return 0.0


def datetime_columns(frame: pd.DataFrame) -> list[str]:
    """Columns whose values are dates: datetime dtypes or mostly parseable text.

    Text columns that look numeric ("2024", "12.5") are deliberately
    skipped — pandas would happily read plain years as dates.
    """
    names = []
    for name in frame.columns:
        series = frame[name]
        if pd.api.types.is_datetime64_any_dtype(series):
            names.append(str(name))
            continue
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        non_null = series.dropna()
        if non_null.empty or _mostly_numeric(non_null):
            continue
        if _parseable_share(non_null.head(5000)) >= 0.9:
            names.append(str(name))
    return names


def color_options(frame: pd.DataFrame, max_groups: int = MAX_COLOR_GROUPS) -> list[str]:
    """Columns that work as a scatter colour: few-valued groups, then numbers."""
    options = [
        name
        for name in categorical_columns(frame)
        if frame[name].nunique(dropna=True) <= max_groups
    ]
    options.extend(numeric_columns(frame))
    return options


# --- Shared helpers --------------------------------------------------------


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        raise ChartError(f'Column "{name}" does not exist in this dataset.')
    return frame[name]


def _numeric_values(frame: pd.DataFrame, column: str) -> pd.Series:
    """Non-missing numeric values of a column, coerced to numbers."""
    values = pd.to_numeric(_column(frame, column), errors="coerce").dropna()
    if values.empty:
        raise ChartError(f'Column "{column}" has no numeric values to work with.')
    return values


def _to_datetime(series: pd.Series, name: str) -> pd.Series:
    """Parse a column to datetimes, leaving the input untouched."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    try:
        parsed = _try_datetime(series)
    except (ValueError, TypeError):  # mixed timezones raise even with coerce
        parsed = _try_datetime(series, utc=True)
    if not pd.api.types.is_datetime64_any_dtype(parsed):
        raise ChartError(f'Column "{name}" could not be read as dates.')
    return parsed


def apply_chart_theme(figure: go.Figure) -> go.Figure:
    """Paint a figure with the design tokens from ui/styles.css."""
    figure.update_layout(
        paper_bgcolor=PAGE_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        font=dict(color=TEXT_COLOR, size=13),
        colorway=list(CHART_COLORS),
        margin=dict(l=48, r=24, t=56, b=48),
        hovermode="closest",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED_TEXT_COLOR)),
        hoverlabel=dict(
            bgcolor=PLOT_BACKGROUND,
            bordercolor=GRID_COLOR,
            font=dict(color=TEXT_COLOR),
        ),
    )
    tickfont = dict(color=MUTED_TEXT_COLOR)
    figure.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, tickfont=tickfont)
    figure.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, tickfont=tickfont)
    return figure


def summary_stats(frame: pd.DataFrame, column: str) -> dict[str, float]:
    """Count / mean / median / std / min / max of a numeric column.

    ``std`` is NaN for a single value; the UI shows that as an em dash.
    """
    values = _numeric_values(frame, column)
    return {
        "count": int(values.count()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


# --- Distribution ----------------------------------------------------------


def histogram_figure(frame: pd.DataFrame, column: str, bins: int = 30) -> go.Figure:
    """Histogram of a numeric column with mean and median markers."""
    values = _numeric_values(frame, column)
    try:
        bins = int(bins)
    except (TypeError, ValueError) as exc:
        raise ChartError("The number of bins must be a whole number.") from exc
    if not 2 <= bins <= 200:
        raise ChartError("The number of bins must be between 2 and 200.")

    figure = go.Figure()
    apply_chart_theme(figure)
    figure.add_trace(
        go.Histogram(
            x=values,
            nbinsx=bins,
            marker=dict(color=CHART_COLORS[0], line=dict(color=PAGE_BACKGROUND, width=1)),
            hovertemplate="%{x:,.4g}: %{y:,} rows<extra></extra>",
        )
    )
    mean = float(values.mean())
    median = float(values.median())
    figure.add_vline(
        x=mean,
        line_color=CHART_COLORS[1],
        line_dash="dot",
        line_width=1.5,
        annotation_text=f"mean {mean:,.4g}",
        annotation_position="top",
        annotation_font_color=CHART_COLORS[1],
        annotation_bgcolor=PLOT_BACKGROUND,
    )
    figure.add_vline(
        x=median,
        line_color=CHART_COLORS[3],
        line_dash="dash",
        line_width=1.5,
        annotation_text=f"median {median:,.4g}",
        annotation_position="bottom",
        annotation_font_color=CHART_COLORS[3],
        annotation_bgcolor=PLOT_BACKGROUND,
    )
    figure.update_layout(
        bargap=0.02,
        height=460,
        xaxis_title=str(column),
        yaxis_title="Rows",
    )
    return figure


# --- Relationship ----------------------------------------------------------


def scatter_figure(
    frame: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
) -> go.Figure:
    """Scatter of two numeric columns, optionally coloured by a third.

    Rows missing any of the plotted values are skipped; nothing is
    imputed. A numeric colour becomes a continuous scale, a group
    colour splits the points into one trace per group.
    """
    if x == y:
        raise ChartError("Pick two different columns for the X and Y axes.")

    data = pd.DataFrame(
        {
            "x": pd.to_numeric(_column(frame, x), errors="coerce"),
            "y": pd.to_numeric(_column(frame, y), errors="coerce"),
        }
    )
    if color is not None:
        data[str(color)] = _column(frame, color)
    data = data.dropna()
    if data.empty:
        needed = f'"{x}" and "{y}"'
        if color is not None:
            needed += f' and a "{color}"'
        raise ChartError(f"No rows have values for {needed}.")

    figure = go.Figure()
    apply_chart_theme(figure)

    if color is None:
        figure.add_trace(
            go.Scatter(
                x=data["x"],
                y=data["y"],
                mode="markers",
                marker=dict(color=CHART_COLORS[0], size=8, opacity=0.7),
                hovertemplate="%{x:,.4g}, %{y:,.4g}<extra></extra>",
            )
        )
    elif pd.api.types.is_numeric_dtype(data[str(color)]) and not pd.api.types.is_bool_dtype(
        data[str(color)]
    ):
        figure.add_trace(
            go.Scatter(
                x=data["x"],
                y=data["y"],
                mode="markers",
                marker=dict(
                    color=data[str(color)],
                    colorscale=[[0, CHART_COLORS[0]], [1, CHART_COLORS[1]]],
                    showscale=True,
                    colorbar=dict(title=str(color), thickness=12, outlinewidth=0),
                    size=8,
                    opacity=0.8,
                ),
                hovertemplate="%{x:,.4g}, %{y:,.4g}<extra></extra>",
            )
        )
    else:
        groups = data[str(color)].nunique(dropna=True)
        if groups > MAX_COLOR_GROUPS:
            raise ChartError(
                f'Column "{color}" has {groups} distinct values; colouring '
                f"supports at most {MAX_COLOR_GROUPS}."
            )
        grouped = data.groupby(str(color), sort=False, dropna=True, observed=True)
        for index, (value, group) in enumerate(grouped):
            figure.add_trace(
                go.Scatter(
                    x=group["x"],
                    y=group["y"],
                    mode="markers",
                    name=str(value),
                    marker=dict(
                        color=CHART_COLORS[index % len(CHART_COLORS)],
                        size=8,
                        opacity=0.75,
                    ),
                    hovertemplate="%{x:,.4g}, %{y:,.4g}<extra></extra>",
                )
            )
        figure.update_layout(legend_title_text=str(color))

    figure.update_layout(height=460, xaxis_title=x, yaxis_title=y)
    return figure


# --- Correlation -----------------------------------------------------------


def correlation_matrix(frame: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Pairwise correlation of the numeric columns that actually vary.

    All-missing and constant columns are excluded: they cannot
    correlate with anything and would render as an empty row.
    """
    if method not in CORRELATION_METHODS:
        raise ChartError(
            f'Unknown correlation method "{method}". '
            f'Choose one of: {", ".join(CORRELATION_METHODS)}.'
        )
    columns = [
        name
        for name in numeric_columns(frame)
        if frame[name].notna().any() and frame[name].nunique(dropna=True) > 1
    ]
    if len(columns) < 2:
        raise ChartError(
            "Correlation needs at least two numeric columns that vary; "
            f"this dataset has {len(columns)}."
        )
    return frame[columns].corr(method=method)


def correlation_heatmap_figure(matrix: pd.DataFrame) -> go.Figure:
    """Heatmap of a correlation matrix, coloured from −1 to +1."""
    if matrix.empty or len(matrix) < 2:
        raise ChartError("A correlation heatmap needs at least two columns.")

    labels = [str(name) for name in matrix.columns]
    values = matrix.to_numpy(dtype=float)
    annotate = len(labels) <= MAX_ANNOTATED_COLUMNS

    annotations = [
        ["" if pd.isna(value) else f"{value:.2f}" for value in row] for row in values
    ]
    hover_text = [
        ["—" if pd.isna(value) else f"{value:.2f}" for value in row] for row in values
    ]

    trace_options: dict = dict(
        z=values,
        x=labels,
        y=labels,
        zmin=-1,
        zmax=1,
        colorscale=[
            [0.0, CHART_COLORS[1]],  # violet = −1
            [0.5, MUTED_TEXT_COLOR],  # muted = 0
            [1.0, CHART_COLORS[0]],  # cyan = +1
        ],
        colorbar=dict(title="r", thickness=12, outlinewidth=0),
        xgap=2,
        ygap=2,
        hovertext=hover_text,
        hovertemplate="%{y} vs %{x}: %{hovertext}<extra></extra>",
    )
    if annotate:
        trace_options["text"] = annotations
        trace_options["texttemplate"] = "%{text}"
        trace_options["textfont"] = dict(color=PAGE_BACKGROUND, size=11)

    longest = max((len(label) for label in labels), default=0)
    figure = go.Figure()
    apply_chart_theme(figure)
    figure.add_trace(go.Heatmap(**trace_options))
    figure.update_layout(
        height=min(max(360, 44 * len(labels) + 130), 1200),
        margin=dict(l=min(48 + 8 * longest, 260), r=24, t=56, b=96),
    )
    figure.update_xaxes(tickangle=-45)
    return figure


# --- Time series -----------------------------------------------------------


def _choose_period(dates: pd.Series, period: str) -> str:
    """Resample frequency: an explicit choice, or one picked from the span."""
    if period not in PERIODS:
        raise ChartError(
            f'Unknown time bucket "{period}". Choose one of: {", ".join(PERIODS)}.'
        )
    if period != "auto":
        return PERIOD_FREQ[period]

    span_days = float((dates.max() - dates.min()).total_seconds()) / 86400
    if span_days <= 10:
        return "h"
    if span_days <= 365:
        return "D"
    if span_days <= 365 * 5:
        return "W"
    return "ME"


def bucketed_series(
    frame: pd.DataFrame,
    date_column: str,
    value_column: str,
    aggregation: str = "mean",
    period: str = "auto",
) -> tuple[pd.Series, str]:
    """``value_column`` aggregated over time, bucketed by ``date_column``.

    Shared by the Charts page (which draws the line) and the AI
    analysis engine (which reports the same numbers). Returns the
    bucketed series plus a human label for the chosen bucket
    ("hourly", "daily", "weekly", "monthly").

    Dates are bucketed to an hour, day, week or month and the values
    inside each bucket are combined with ``aggregation``. Buckets with
    no rows stay empty for mean/min/max and count as zero for
    sum/count.
    """
    if aggregation not in AGGREGATIONS:
        raise ChartError(
            f'Unknown aggregation "{aggregation}". '
            f'Choose one of: {", ".join(AGGREGATIONS)}.'
        )

    dates = _to_datetime(_column(frame, date_column), date_column)
    values = pd.to_numeric(_column(frame, value_column), errors="coerce")
    work = pd.DataFrame({"date": dates, "value": values}).dropna()
    if work.empty:
        raise ChartError(
            f'No rows have both a usable date in "{date_column}" '
            f'and a number in "{value_column}".'
        )
    work = work.sort_values("date")

    freq = _choose_period(work["date"], period)
    bucketed = work.set_index("date")["value"].resample(freq).agg(aggregation).dropna()
    return bucketed, FREQ_LABELS[freq]


def timeseries_figure(
    frame: pd.DataFrame,
    date_column: str,
    value_column: str,
    aggregation: str = "mean",
    period: str = "auto",
) -> go.Figure:
    """Line of ``value_column`` aggregated over time by ``date_column``.

    The heavy lifting is ``bucketed_series``; this wrapper only paints
    the themed figure and stores the chosen bucket in ``layout.meta``
    for the page caption.
    """
    bucketed, bucket = bucketed_series(
        frame, date_column, value_column, aggregation, period
    )

    figure = go.Figure()
    apply_chart_theme(figure)
    figure.add_trace(
        go.Scatter(
            x=bucketed.index,
            y=bucketed.values,
            mode="lines+markers" if len(bucketed) <= 120 else "lines",
            line=dict(color=CHART_COLORS[0], width=2),
            marker=dict(color=CHART_COLORS[0], size=6),
            hovertemplate="%{x|%b %d, %Y}: %{y:,.4g}<extra></extra>",
        )
    )
    figure.update_layout(
        height=440,
        xaxis_title=f"{date_column} ({bucket} buckets)",
        yaxis_title=f"{aggregation} of {value_column}",
        meta={"bucket": bucket},
    )
    return figure

"""Charts page — visualise the loaded dataset (Milestone 4).

Figure building lives in core/charts.py so it stays unit-testable;
this module only picks columns and renders. The dataset itself comes
from the Data page through session state and is never modified here.
"""

import math

import streamlit as st

from core.charts import (
    AGGREGATIONS,
    CORRELATION_METHODS,
    PERIODS,
    ChartError,
    color_options,
    correlation_heatmap_figure,
    correlation_matrix,
    datetime_columns,
    histogram_figure,
    numeric_columns,
    scatter_figure,
    summary_stats,
    timeseries_figure,
)
from core.loader import MAX_UPLOAD_MB, LoadedDataset
from ui.components import (
    render_empty_state,
    render_kpis,
    render_page_header,
    section_card,
    section_heading,
)

CHART_TYPES = ("Distribution", "Relationship", "Correlation", "Time series")
PLOT_CONFIG = {"displaylogo": False, "scrollZoom": True}

#: One honest line of guidance per chart type, shown as a caption.
CHART_GUIDANCE = {
    "Distribution": "One numeric column — how often each value range occurs.",
    "Relationship": "Two numeric columns — whether they move together.",
    "Correlation": "All numeric columns — a coefficient matrix from −1 to +1.",
    "Time series": "A date column over a numeric column — trends over time.",
}

AGGREGATION_LABELS = {
    "mean": "Mean",
    "sum": "Sum",
    "min": "Minimum",
    "max": "Maximum",
    "count": "Count",
}
PERIOD_LABELS = {
    "auto": "Automatic",
    "hour": "Hourly",
    "day": "Daily",
    "week": "Weekly",
    "month": "Monthly",
}


def _fmt(value: float) -> str:
    """Compact number for KPI cards; NaN becomes an em dash."""
    if math.isnan(value):
        return "—"
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1_000_000 or magnitude < 0.001:
        return f"{value:,.4g}"
    decimals = 3 if magnitude < 1 else 2
    return f"{value:,.{decimals}f}"


def _render_distribution(frame, numeric) -> None:
    if not numeric:
        st.info(
            "No numeric columns found — a distribution needs a column that "
            "pandas reads as numbers."
        )
        return

    column = st.selectbox("Numeric column", numeric, key="dist_column")
    bins = st.slider(
        "Bins",
        5,
        100,
        30,
        step=5,
        key="dist_bins",
        help="How many buckets the values are split into.",
    )
    try:
        figure = histogram_figure(frame, column, bins)
        stats = summary_stats(frame, column)
    except ChartError as exc:
        st.warning(str(exc))
        return

    st.plotly_chart(figure, width="stretch", theme=None, config=PLOT_CONFIG)

    section_heading("Summary")
    render_kpis(
        [
            {
                "label": "Count",
                "value": f"{stats['count']:,}",
                "help": "Rows with a value in this column.",
            },
            {"label": "Mean", "value": _fmt(stats["mean"])},
            {"label": "Median", "value": _fmt(stats["median"])},
            {
                "label": "Std dev",
                "value": _fmt(stats["std"]),
                "help": "How far the values typically sit from the mean.",
            },
            {"label": "Min", "value": _fmt(stats["min"])},
            {"label": "Max", "value": _fmt(stats["max"])},
        ]
    )


def _render_relationship(frame, numeric, colors) -> None:
    if len(numeric) < 2:
        st.info(
            "A relationship chart needs at least two numeric columns; this "
            f"dataset has {len(numeric)}."
        )
        return

    x = st.selectbox("X axis", numeric, key="rel_x")
    y = st.selectbox("Y axis", [name for name in numeric if name != x], key="rel_y")
    color = st.selectbox(
        "Colour by",
        ["None"] + colors,
        key="rel_color",
        help="Optional second dimension: a group column or another number.",
    )
    try:
        figure = scatter_figure(frame, x, y, None if color == "None" else color)
    except ChartError as exc:
        st.warning(str(exc))
        return

    st.plotly_chart(figure, width="stretch", theme=None, config=PLOT_CONFIG)
    st.caption(
        "Rows missing X, Y or the colour value are skipped — nothing is "
        "imputed, dropped or changed in the data itself."
    )


def _render_correlation(frame, numeric) -> None:
    if len(numeric) < 2:
        st.info(
            "Correlation needs at least two numeric columns; this dataset "
            f"has {len(numeric)}."
        )
        return

    method = st.segmented_control(
        "Method",
        CORRELATION_METHODS,
        default=CORRELATION_METHODS[0],
        key="corr_method",
        required=True,
        format_func=str.capitalize,
        help="Pearson measures linear relationships, Spearman monotonic ones.",
    )
    if not method:
        return
    if isinstance(method, list):  # defensive: single selection returns a value
        method = method[0]

    try:
        matrix = correlation_matrix(frame, str(method))
        figure = correlation_heatmap_figure(matrix)
    except ChartError as exc:
        st.warning(str(exc))
        return

    st.plotly_chart(figure, width="stretch", theme=None, config=PLOT_CONFIG)
    st.caption(
        f"{len(matrix)} numeric columns · {str(method).capitalize()} · "
        "coefficients run from −1 to +1 · constant or all-empty columns "
        "are excluded."
    )


def _render_timeseries(frame, numeric, datetimes) -> None:
    if not datetimes:
        st.info(
            "No date column detected. A column counts as a date when pandas "
            "reads it as datetime, or when at least 90% of its values parse "
            "as dates (for example 2024-05-31)."
        )
        return
    if not numeric:
        st.info("No numeric columns to plot over time.")
        return

    date_column = st.selectbox("Date column", datetimes, key="ts_date")
    value_column = st.selectbox("Value column", numeric, key="ts_value")
    aggregation = st.selectbox(
        "Aggregation",
        AGGREGATIONS,
        format_func=lambda option: AGGREGATION_LABELS[option],
        key="ts_agg",
        help="How the values inside each time bucket are combined.",
    )
    period = st.selectbox(
        "Bucket",
        PERIODS,
        format_func=lambda option: PERIOD_LABELS[option],
        key="ts_period",
        help="How dates are grouped. Automatic picks a size from the date range.",
    )
    try:
        figure = timeseries_figure(frame, date_column, value_column, aggregation, period)
    except ChartError as exc:
        st.warning(str(exc))
        return

    st.plotly_chart(figure, width="stretch", theme=None, config=PLOT_CONFIG)
    bucket = (figure.layout.meta or {}).get("bucket", period)
    empty_note = (
        "empty periods show 0"
        if aggregation in ("sum", "count")
        else "empty periods stay as gaps"
    )
    st.caption(
        f"{AGGREGATION_LABELS[aggregation]} of {value_column}, bucketed "
        f"{bucket} · {empty_note}."
    )


render_page_header(
    "Charts",
    "Explore the loaded dataset with interactive Plotly charts — hover, zoom and pan. Data is only read, never changed.",
    chip="Live",
    chip_tone="live",
)

dataset: LoadedDataset | None = st.session_state.get("loaded_dataset")

if dataset is None:
    render_empty_state(
        title="No dataset loaded yet",
        message=(
            "Charts are drawn from the dataset you load on the Data page. "
            "Upload a CSV or Excel workbook there and it will appear here."
        ),
        hint=f"Supported: .csv and .xlsx · maximum {MAX_UPLOAD_MB} MB · kept in memory only",
        icon="📈",
    )
    st.link_button("Go to the Data page", url="/data", icon=":material/database:")
    st.stop()

frame = dataset.dataframe

with section_card():
    st.caption(
        f"{dataset.filename} · {len(frame):,} rows × {frame.shape[1]:,} columns · "
        "read-only view — charts never modify the data"
    )

    numeric = numeric_columns(frame)
    datetimes = datetime_columns(frame)
    colors = color_options(frame)

    chart = st.segmented_control(
        "Chart type",
        CHART_TYPES,
        default=CHART_TYPES[0],
        key="chart_type",
        required=True,
        width="stretch",
        help="Distribution: one column · Relationship: two columns · "
        "Correlation: all numeric columns · Time series: dates over time.",
    )
if isinstance(chart, list):  # defensive: single selection returns a value
    chart = chart[0] if chart else None
if not chart:
    st.info("Choose a chart type above to get started.")
    st.stop()

st.caption(CHART_GUIDANCE.get(chart, ""))

if chart == "Relationship":
    _render_relationship(frame, numeric, colors)
elif chart == "Correlation":
    _render_correlation(frame, numeric)
elif chart == "Time series":
    _render_timeseries(frame, numeric, datetimes)
else:
    _render_distribution(frame, numeric)

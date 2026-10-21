"""Tests for the framework-free chart builders in core/charts.py.

Every figure builder must: return a themed Plotly figure, refuse
impossible input with a ChartError, and never modify the frame it
was given. All fixtures are built in memory.
"""

import math

import pandas as pd
import plotly.graph_objects as go
import pytest

from core.charts import (
    AGGREGATIONS,
    CHART_COLORS,
    CORRELATION_METHODS,
    PERIODS,
    ChartError,
    categorical_columns,
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
from ui.theme import load_css


@pytest.fixture
def frame() -> pd.DataFrame:
    """A small sales frame covering dates, groups, numbers and junk columns."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "region": ["north", "south", "north", "south", "north", "south"],
            "sales": [100.0, 150.0, 120.0, 200.0, 175.0, 220.0],
            "units": [5, 7, 6, 9, 8, 10],
            "active": [True, False, True, True, False, True],
            "constant": [1, 1, 1, 1, 1, 1],
            "year": ["2024", "2024", "2024", "2024", "2024", "2024"],
        }
    )


# --- Column selection ------------------------------------------------------


def test_numeric_columns_skip_text_dates_and_bools(frame):
    assert numeric_columns(frame) == ["sales", "units", "constant"]


def test_categorical_columns_hold_text_and_bools(frame):
    assert categorical_columns(frame) == ["region", "active", "year"]


def test_datetime_columns_find_datetime_dtypes(frame):
    assert datetime_columns(frame) == ["date"]


def test_datetime_columns_find_parseable_text():
    frame = pd.DataFrame({"when": ["2024-01-01", "2024-01-02"] * 11 + ["not a date"]})
    # 22 of 23 parse — above the 90% threshold.
    assert datetime_columns(frame) == ["when"]


def test_datetime_columns_skip_numeric_looking_text():
    frame = pd.DataFrame({"year": ["2024", "2025", "2026"]})
    assert datetime_columns(frame) == []


def test_datetime_columns_skip_mixed_timezones():
    frame = pd.DataFrame(
        {
            "stamp": [
                "2024-01-01T00:00:00+01:00",
                "2024-01-01T00:00:00+02:00",
            ]
        }
    )
    assert datetime_columns(frame) == ["stamp"]


def test_color_options_groups_then_numbers(frame):
    options = color_options(frame)
    assert options[:3] == ["region", "active", "year"]  # few-valued groups first
    assert options[3:] == ["sales", "units", "constant"]  # numeric columns last
    assert "date" not in options


def test_color_options_refuse_large_group_columns():
    frame = pd.DataFrame({"id": [f"user-{i}" for i in range(50)], "v": range(50)})
    assert color_options(frame) == ["v"]


# --- Summary statistics ----------------------------------------------------


def test_summary_stats_counts_only_values_present():
    frame = pd.DataFrame({"v": [1.0, None, 3.0, 5.0]})
    stats = summary_stats(frame, "v")
    assert stats["count"] == 3
    assert stats["mean"] == pytest.approx(3.0)
    assert stats["median"] == pytest.approx(3.0)
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(5.0)


def test_summary_stats_std_is_nan_for_a_single_value():
    stats = summary_stats(pd.DataFrame({"v": [7.0]}), "v")
    assert math.isnan(stats["std"])


def test_summary_stats_rejects_a_text_column():
    with pytest.raises(ChartError, match="no numeric values"):
        summary_stats(pd.DataFrame({"v": ["a", "b"]}), "v")


# --- Distribution ----------------------------------------------------------


def test_histogram_returns_a_themed_histogram(frame):
    figure = histogram_figure(frame, "sales", bins=12)
    assert isinstance(figure, go.Figure)
    assert figure.data[0].type == "histogram"
    assert figure.data[0].nbinsx == 12
    assert figure.layout.xaxis.title.text == "sales"
    assert figure.layout.yaxis.title.text == "Rows"


def test_histogram_marks_mean_and_median(frame):
    figure = histogram_figure(frame, "sales")
    texts = [annotation.text for annotation in figure.layout.annotations]
    assert any(text.startswith("mean") for text in texts)
    assert any(text.startswith("median") for text in texts)
    assert len(figure.layout.shapes) == 2


def test_histogram_uses_the_design_palette(frame):
    figure = histogram_figure(frame, "sales")
    assert figure.data[0].marker.color == CHART_COLORS[0]
    assert figure.layout.paper_bgcolor.lower() == "#0b1220"


def test_histogram_rejects_unknown_column(frame):
    with pytest.raises(ChartError, match="does not exist"):
        histogram_figure(frame, "profit")


def test_histogram_rejects_silly_bin_counts(frame):
    with pytest.raises(ChartError, match="between 2 and 200"):
        histogram_figure(frame, "sales", bins=1)


# --- Relationship ----------------------------------------------------------


def test_scatter_without_colour_is_one_trace(frame):
    figure = scatter_figure(frame, "units", "sales")
    assert len(figure.data) == 1
    assert figure.data[0].mode == "markers"
    assert figure.layout.xaxis.title.text == "units"
    assert figure.layout.yaxis.title.text == "sales"


def test_scatter_splits_categorical_colour_into_groups(frame):
    figure = scatter_figure(frame, "units", "sales", color="region")
    assert [trace.name for trace in figure.data] == ["north", "south"]
    assert figure.layout.legend.title.text == "region"
    assert figure.layout.xaxis.title.text == "units"  # theme merges, not overwrites


def test_scatter_uses_a_continuous_scale_for_numbers(frame):
    figure = scatter_figure(frame, "units", "sales", color="sales")
    assert len(figure.data) == 1
    assert figure.data[0].marker.showscale is True


def test_scatter_treats_bools_as_groups(frame):
    figure = scatter_figure(frame, "units", "sales", color="active")
    assert [trace.name for trace in figure.data] == ["True", "False"]


def test_scatter_skips_incomplete_rows():
    frame = pd.DataFrame({"x": [1.0, None, 3.0], "y": [1.0, 2.0, None]})
    figure = scatter_figure(frame, "x", "y")
    assert len(figure.data[0].x) == 1


def test_scatter_rejects_the_same_column_twice(frame):
    with pytest.raises(ChartError, match="different columns"):
        scatter_figure(frame, "sales", "sales")


def test_scatter_rejects_too_many_colour_groups():
    frame = pd.DataFrame(
        {"x": range(30), "y": range(30), "id": [f"u{i}" for i in range(30)]}
    )
    with pytest.raises(ChartError, match="at most 12"):
        scatter_figure(frame, "x", "y", color="id")


def test_scatter_rejects_a_frame_with_no_usable_rows():
    frame = pd.DataFrame({"x": [None, None], "y": [None, None]})
    with pytest.raises(ChartError, match="No rows"):
        scatter_figure(frame, "x", "y")


# --- Correlation -----------------------------------------------------------


def test_correlation_matrix_matches_pandas(frame):
    matrix = correlation_matrix(frame, "pearson")
    assert list(matrix.columns) == ["sales", "units"]
    expected = frame[["sales", "units"]].corr(method="pearson")
    assert matrix.equals(expected)


def test_correlation_drops_constant_and_empty_columns(frame):
    frame = frame.assign(nothing=[None, None, None, None, None, None])
    matrix = correlation_matrix(frame)
    assert list(matrix.columns) == ["sales", "units"]


def test_spearman_differs_from_pearson_on_curved_data():
    frame = pd.DataFrame({"x": range(1, 11), "y": [n**2 for n in range(1, 11)]})
    pearson = correlation_matrix(frame, "pearson").loc["x", "y"]
    spearman = correlation_matrix(frame, "spearman").loc["x", "y"]
    assert spearman == pytest.approx(1.0)
    assert pearson < 0.99


def test_correlation_needs_two_varying_numeric_columns():
    frame = pd.DataFrame({"only": [1, 2, 3], "text": list("abc")})
    with pytest.raises(ChartError, match="at least two numeric columns"):
        correlation_matrix(frame)


def test_correlation_rejects_an_unknown_method(frame):
    with pytest.raises(ChartError, match="Unknown correlation method"):
        correlation_matrix(frame, "distance")


def test_heatmap_renders_the_full_matrix(frame):
    figure = correlation_heatmap_figure(correlation_matrix(frame))
    heatmap = figure.data[0]
    assert heatmap.type == "heatmap"
    assert heatmap.zmin == -1 and heatmap.zmax == 1
    assert list(heatmap.x) == ["sales", "units"]
    # Design-token colours: violet at −1, muted at 0, cyan at +1.
    assert [stop[1] for stop in heatmap.colorscale] == [
        CHART_COLORS[1],
        "#93a4bd",
        CHART_COLORS[0],
    ]


def test_heatmap_marks_missing_coefficients_as_a_dash():
    matrix = pd.DataFrame(
        {"a": [1.0, 0.25], "b": [0.25, float("nan")]},
        index=["a", "b"],
    )
    figure = correlation_heatmap_figure(matrix)
    assert figure.data[0].text[1][1] == ""
    assert figure.data[0].hovertext[1][1] == "—"


def test_heatmap_needs_at_least_two_columns():
    matrix = pd.DataFrame({"a": [1.0]}, index=["a"])
    with pytest.raises(ChartError, match="at least two columns"):
        correlation_heatmap_figure(matrix)


# --- Time series -----------------------------------------------------------


def test_timeseries_buckets_daily_means(frame):
    figure = timeseries_figure(frame, "date", "sales", "mean", "day")
    assert figure.layout.meta["bucket"] == "daily"
    trace = figure.data[0]
    assert len(trace.x) == 6
    assert list(trace.y) == pytest.approx([100.0, 150.0, 120.0, 200.0, 175.0, 220.0])


def test_timeseries_sums_repeated_dates():
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "sales": [10.0, 15.0, 7.0],
        }
    )
    figure = timeseries_figure(frame, "date", "sales", "sum", "day")
    assert list(figure.data[0].y) == pytest.approx([25.0, 7.0])


def test_timeseries_auto_picks_hourly_buckets_for_short_spans():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=48, freq="h"),
            "sales": range(48),
        }
    )
    figure = timeseries_figure(frame, "date", "sales", "mean", "auto")
    assert figure.layout.meta["bucket"] == "hourly"
    assert len(figure.data[0].x) == 48


def test_timeseries_auto_picks_monthly_buckets_for_long_spans():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2018-01-01", periods=72, freq="MS"),
            "sales": range(72),
        }
    )
    figure = timeseries_figure(frame, "date", "sales", "mean", "auto")
    assert figure.layout.meta["bucket"] == "monthly"
    assert len(figure.data[0].x) == 72


def test_timeseries_reads_text_dates():
    frame = pd.DataFrame(
        {"when": ["2024-01-01", "2024-01-02"], "sales": [10.0, 20.0]}
    )
    figure = timeseries_figure(frame, "when", "sales", "mean", "day")
    assert len(figure.data[0].x) == 2


def test_timeseries_rejects_an_unknown_aggregation(frame):
    with pytest.raises(ChartError, match="Unknown aggregation"):
        timeseries_figure(frame, "date", "sales", "median", "day")


def test_timeseries_rejects_an_unknown_bucket(frame):
    with pytest.raises(ChartError, match="Unknown time bucket"):
        timeseries_figure(frame, "date", "sales", "mean", "fortnight")


def test_timeseries_rejects_non_dates():
    frame = pd.DataFrame({"when": list("abcdef"), "sales": range(6)})
    with pytest.raises(ChartError, match="No rows have both"):
        timeseries_figure(frame, "when", "sales", "mean", "day")


# --- Shared guarantees -----------------------------------------------------


def test_every_builder_leaves_the_frame_untouched(frame):
    original = frame.copy(deep=True)
    histogram_figure(frame, "sales")
    scatter_figure(frame, "units", "sales", color="region")
    correlation_heatmap_figure(correlation_matrix(frame))
    timeseries_figure(frame, "date", "sales", "sum", "auto")
    pd.testing.assert_frame_equal(frame, original)


def test_option_constants_are_complete():
    assert set(CORRELATION_METHODS) == {"pearson", "spearman"}
    assert set(AGGREGATIONS) == {"mean", "sum", "min", "max", "count"}
    assert set(PERIODS) == {"auto", "hour", "day", "week", "month"}


def test_chart_colours_come_from_the_design_tokens():
    css = load_css()
    for token in (*CHART_COLORS, "#0b1220", "#111c33", "#223252", "#e2e8f0", "#93a4bd"):
        assert token in css, token

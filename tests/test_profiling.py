"""Known-value tests for profiling and data-quality reporting."""

import pandas as pd
import pytest

from core.profiling import (
    categorical_summary,
    column_profiles,
    dataset_metrics,
    numeric_summary,
    quality_issues,
)


@pytest.fixture()
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "amount": [10.0, None, 30.0, 40.0, 40.0],
            "region": ["north", "south", "north", None, "south"],
            "constant": ["same", "same", "same", "same", "same"],
        }
    )


# --- Dataset metrics -------------------------------------------------------


def test_dataset_metrics_known_values(frame):
    metrics = dataset_metrics(frame)
    assert metrics.rows == 5
    assert metrics.columns == 4
    assert metrics.missing_cells == 2
    assert metrics.duplicate_rows == 0
    assert metrics.memory_bytes > 0


def test_duplicate_rows_are_counted():
    frame = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    assert dataset_metrics(frame).duplicate_rows == 1


def test_empty_dataset_with_headers():
    frame = pd.DataFrame(columns=["a", "b"])
    metrics = dataset_metrics(frame)
    assert metrics.rows == 0
    assert metrics.missing_cells == 0
    assert column_profiles(frame)["Missing %"].tolist() == [0.0, 0.0]


# --- Column profiles -------------------------------------------------------


def test_column_profiles_known_values(frame):
    profiles = column_profiles(frame).set_index("Column")
    assert profiles.loc["amount", "Non-null"] == 4
    assert profiles.loc["amount", "Missing"] == 1
    assert profiles.loc["amount", "Missing %"] == 20.0
    assert profiles.loc["id", "Unique"] == 5
    assert profiles.loc["constant", "Unique"] == 1
    assert profiles.loc["region", "Non-null"] == 4


# --- Statistics ------------------------------------------------------------


def test_numeric_summary_known_values(frame):
    summary = numeric_summary(frame).set_index("Column")
    assert summary.loc["id", "count"] == 5
    assert summary.loc["amount", "count"] == 4
    assert summary.loc["amount", "mean"] == pytest.approx(30.0)
    assert summary.loc["id", "min"] == 1


def test_numeric_summary_is_empty_for_text_only_data():
    assert numeric_summary(pd.DataFrame({"a": ["x", "y"]})).empty


def test_categorical_summary_known_values(frame):
    summary = categorical_summary(frame).set_index("Column")
    assert summary.loc["region", "Unique"] == 2
    assert summary.loc["region", "Top count"] == 2
    assert summary.loc["region", "Top %"] == 50.0
    assert "id" not in summary.index


# --- Quality report --------------------------------------------------------


def test_quality_flags_missing_and_constant_columns(frame):
    issues = {issue.category: issue for issue in quality_issues(frame)}
    assert "Columns with missing values" in issues
    assert "amount" in issues["Columns with missing values"].detail
    assert "Constant columns" in issues
    assert "constant" in issues["Constant columns"].detail
    assert "Duplicate rows" not in issues


def test_quality_flags_duplicate_rows():
    frame = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    categories = [issue.category for issue in quality_issues(frame)]
    assert "Duplicate rows" in categories


def test_quality_flags_mostly_missing_and_empty_columns():
    frame = pd.DataFrame({"a": [1, None, None, None], "b": [None, None, None, None]})
    issues = {issue.category: issue for issue in quality_issues(frame)}
    assert "Columns over 50% missing" in issues
    assert "Empty columns (all values missing)" in issues
    assert issues["Columns over 50% missing"].severity == "high"


def test_quality_flags_numeric_looking_text_columns():
    frame = pd.DataFrame({"code": ["100", "200", "300"], "label": ["a", "b", "c"]})
    categories = [issue.category for issue in quality_issues(frame)]
    assert "Numeric-looking text columns" in categories


def test_quality_is_clean_for_tidy_data():
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert quality_issues(frame) == []


def test_quality_reports_empty_dataset():
    issues = quality_issues(pd.DataFrame(columns=["a", "b"]))
    assert len(issues) == 1
    assert issues[0].category == "Empty dataset"


# --- Safety ----------------------------------------------------------------


def test_profiling_never_modifies_the_input(frame):
    original = frame.copy(deep=True)

    dataset_metrics(frame)
    column_profiles(frame)
    numeric_summary(frame)
    categorical_summary(frame)
    quality_issues(frame)

    pd.testing.assert_frame_equal(frame, original)

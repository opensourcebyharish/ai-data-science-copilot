"""Tests for CSV/Excel loading, validation and safe column names."""

import io

import pandas as pd
import pytest
from openpyxl import Workbook

from core.loader import (
    MAX_UPLOAD_MB,
    DatasetLoadError,
    EmptyDatasetError,
    FileTooLargeError,
    MalformedDatasetError,
    UnsupportedFileTypeError,
    format_file_size,
    load_dataset,
)


def make_xlsx(rows: list[list], header: list) -> bytes:
    """Build a real .xlsx file in memory for tests."""
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(header)
    for row in rows:
        sheet.append(row)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


# --- Happy paths -----------------------------------------------------------


def test_loads_valid_csv():
    data = b"name,age\nAda,36\nGrace,45\n"
    dataset = load_dataset(data, "people.csv")

    assert dataset.file_type == "csv"
    assert dataset.dataframe.shape == (2, 2)
    assert list(dataset.dataframe.columns) == ["name", "age"]
    assert dataset.size_bytes == len(data)
    assert dataset.duplicate_columns == ()


def test_loads_semicolon_separated_csv():
    dataset = load_dataset(b"a;b\n1;2\n", "semi.csv")
    assert list(dataset.dataframe.columns) == ["a", "b"]
    assert dataset.dataframe.shape == (1, 2)


def test_loads_valid_xlsx():
    data = make_xlsx([[1, "x"], [2, "y"]], ["id", "label"])
    dataset = load_dataset(data, "book.xlsx")

    assert dataset.file_type == "xlsx"
    assert dataset.dataframe.shape == (2, 2)
    assert list(dataset.dataframe.columns) == ["id", "label"]


def test_csv_numeric_columns_are_typed():
    dataset = load_dataset(b"x,y\n1,2.5\n2,3.5\n", "nums.csv")
    assert pd.api.types.is_numeric_dtype(dataset.dataframe["x"])
    assert dataset.dataframe["y"].tolist() == [2.5, 3.5]


def test_header_only_csv_loads_with_zero_rows():
    dataset = load_dataset(b"a,b\n", "headers.csv")
    assert dataset.dataframe.shape == (0, 2)


# --- Validation errors -----------------------------------------------------


def test_unsupported_extension_is_rejected():
    with pytest.raises(UnsupportedFileTypeError, match="unsupported file type"):
        load_dataset(b"a,b\n1,2\n", "notes.txt")


def test_empty_file_is_rejected():
    with pytest.raises(EmptyDatasetError):
        load_dataset(b"", "empty.csv")


def test_whitespace_only_csv_is_rejected():
    with pytest.raises(EmptyDatasetError):
        load_dataset(b"\n\n", "blank.csv")


def test_binary_content_is_rejected_as_malformed():
    with pytest.raises(MalformedDatasetError):
        load_dataset(b"\x00\x01\x02binary", "broken.csv")


def test_ragged_csv_is_rejected_as_malformed():
    with pytest.raises(MalformedDatasetError):
        load_dataset(b"a,b\n1,2,3\n", "ragged.csv")


def test_unclosed_quote_is_rejected_as_malformed():
    with pytest.raises(MalformedDatasetError):
        load_dataset(b'a,b\n1,"unclosed\n', "broken.csv")


def test_corrupt_xlsx_is_rejected_as_malformed():
    with pytest.raises(MalformedDatasetError):
        load_dataset(b"PK definitely not a workbook", "bad.xlsx")


def test_file_over_the_limit_is_rejected():
    payload = b"a\n" + b"x" * 2048
    with pytest.raises(FileTooLargeError, match="limit"):
        load_dataset(payload, "big.csv", max_mb=0.001)


def test_errors_share_one_base_class():
    assert issubclass(UnsupportedFileTypeError, DatasetLoadError)
    assert issubclass(FileTooLargeError, DatasetLoadError)
    assert issubclass(EmptyDatasetError, DatasetLoadError)
    assert issubclass(MalformedDatasetError, DatasetLoadError)


# --- Column-name safety ----------------------------------------------------


def test_duplicate_headers_are_renamed_safely():
    dataset = load_dataset(b"a,b,a,a\n1,2,3,4\n", "dup.csv")
    assert list(dataset.dataframe.columns) == ["a", "b", "a__2", "a__3"]
    assert dataset.duplicate_columns == ("a",)
    assert dataset.dataframe.iloc[0].tolist() == [1, 2, 3, 4]


def test_excel_duplicate_headers_are_renamed_safely():
    data = make_xlsx([[10, 20, 30]], ["id", "id", "id"])
    dataset = load_dataset(data, "dup.xlsx")
    assert list(dataset.dataframe.columns) == ["id", "id__2", "id__3"]
    assert dataset.duplicate_columns == ("id",)


def test_blank_and_unnamed_headers_get_placeholders():
    dataset = load_dataset(b",price,Unnamed: 2\n1,2,3\n", "blank.csv")
    assert list(dataset.dataframe.columns) == ["column_1", "price", "column_3"]


def test_all_columns_stay_unique():
    dataset = load_dataset(b"a,a,a\n1,2,3\n", "dup.csv")
    assert len(set(dataset.dataframe.columns)) == 3


# --- Misc ------------------------------------------------------------------


def test_default_upload_limit_is_50_mb():
    assert MAX_UPLOAD_MB == 50


def test_format_file_size():
    assert format_file_size(512) == "512 B"
    assert format_file_size(2048) == "2.0 KB"
    assert format_file_size(2 * 1024**2) == "2.0 MB"

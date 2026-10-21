"""Load CSV and Excel uploads into pandas DataFrames.

This module owns *file loading only*. Profiling lives in
core/profiling.py and the Streamlit page in views/data.py.

Uploaded content is never executed as code: the bytes go straight to
pandas' parsers. Every user-facing problem is raised as a
DatasetLoadError subclass so the UI can show a clear message instead
of crashing.
"""

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

#: Maximum accepted upload size in megabytes (also enforced by Streamlit
#: via .streamlit/config.toml -> [server] maxUploadSize).
MAX_UPLOAD_MB = 50

#: Allowed extensions mapped to the loader that handles them.
SUPPORTED_EXTENSIONS = {".csv": "csv", ".xlsx": "xlsx"}

#: utf-8-sig strips a possible BOM; latin-1 never fails and is the last resort.
CSV_ENCODINGS = ("utf-8-sig", "latin-1")


class DatasetLoadError(Exception):
    """Base class for upload problems that can be shown to the user."""


class UnsupportedFileTypeError(DatasetLoadError):
    """The file extension is not one of the supported formats."""


class FileTooLargeError(DatasetLoadError):
    """The upload exceeds MAX_UPLOAD_MB."""


class EmptyDatasetError(DatasetLoadError):
    """The file has no data or no columns."""


class MalformedDatasetError(DatasetLoadError):
    """The file could not be parsed as CSV or Excel."""


@dataclass(frozen=True)
class LoadedDataset:
    """A parsed upload plus the metadata needed by the UI."""

    dataframe: pd.DataFrame
    filename: str
    file_type: str
    size_bytes: int
    duplicate_columns: tuple[str, ...] = ()


def format_file_size(size_bytes: int) -> str:
    """Human-readable file size (B, KB or MB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / 1024**2:.1f} MB"


def _detect_file_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = " and ".join(extension.upper().lstrip(".") for extension in SUPPORTED_EXTENSIONS)
        raise UnsupportedFileTypeError(
            f'"{filename}" has an unsupported file type. Supported formats: {supported}.'
        )
    return SUPPORTED_EXTENSIONS[suffix]


def _check_payload(data: bytes, max_mb: int, file_type: str) -> None:
    if not data:
        raise EmptyDatasetError("The file is empty.")
    # .xlsx is a ZIP archive and contains NUL bytes by design, so this
    # binary-content guard only applies to text (CSV) uploads.
    if file_type == "csv" and b"\x00" in data:
        raise MalformedDatasetError(
            "The file appears to be binary or corrupt, not a CSV file."
        )
    limit = max_mb * 1024 * 1024
    if len(data) > limit:
        raise FileTooLargeError(
            f"The file is {format_file_size(len(data))}, "
            f"but the limit is {max_mb:g} MB."
        )


def _decode_csv(data: bytes) -> str:
    for encoding in CSV_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise MalformedDatasetError("The file could not be decoded as text.")


def _sniff_delimiter(text: str) -> str:
    """Pick comma, semicolon, tab or pipe; fall back to comma."""
    try:
        return csv.Sniffer().sniff(text[:64_000], delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _scan_csv(text: str, delimiter: str) -> list[str]:
    """Validate row structure and return the header row.

    pandas 3 silently drops extra fields (and pads missing ones), so the
    structure is checked here first: every non-blank row must have the
    same number of fields as the header.
    """
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    header: list[str] | None = None
    for row in reader:
        if row:
            header = row
            break
    if header is None:
        return []

    expected = len(header)
    for row in reader:
        if not row:
            continue
        if len(row) != expected:
            raise MalformedDatasetError(
                f"Row {reader.line_num} has {len(row)} fields but the header has "
                f"{expected}. The CSV file is malformed."
            )
    return header


def _read_csv(data: bytes) -> tuple[pd.DataFrame, list[str]]:
    text = _decode_csv(data)
    delimiter = _sniff_delimiter(text)
    header = _scan_csv(text, delimiter)
    try:
        frame = pd.read_csv(io.StringIO(text), sep=delimiter)
    except pd.errors.EmptyDataError as exc:
        raise EmptyDatasetError("The CSV file contains no data.") from exc
    except (pd.errors.ParserError, ValueError) as exc:
        raise MalformedDatasetError(f"Could not parse the CSV file: {exc}") from exc
    return frame, header


def _read_excel(data: bytes) -> pd.DataFrame:
    try:
        import openpyxl  # noqa: F401  (imported to fail fast with a clear message)
    except ImportError as exc:
        raise DatasetLoadError(
            "Excel support is not installed. Run: uv pip install openpyxl"
        ) from exc
    try:
        return pd.read_excel(io.BytesIO(data), engine="openpyxl")
    except Exception as exc:  # openpyxl raises many different errors
        raise MalformedDatasetError(
            "Could not read the Excel workbook. Is it a valid, unencrypted .xlsx file?"
        ) from exc


def _excel_header(data: bytes) -> list:
    try:
        header = pd.read_excel(
            io.BytesIO(data), engine="openpyxl", header=None, nrows=1
        )
    except Exception:
        return []
    return header.iloc[0].tolist() if not header.empty else []


def _normalize_label(raw: object, index: int) -> str:
    """Turn one raw header cell into a clean, non-empty column name."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return f"column_{index + 1}"
    if isinstance(raw, float) and raw.is_integer():
        return str(int(raw))
    name = str(raw).strip()
    if not name or name.lower().startswith("unnamed:"):
        return f"column_{index + 1}"
    return name


def _clean_column_names(original: list) -> tuple[list[str], tuple[str, ...]]:
    """Make column names unique and report which originals were duplicated.

    "a, b, a" becomes "a, b, a__2" and reports ("a",). The uploaded file
    itself is never modified; only the in-memory copy is relabelled.
    """
    cleaned: list[str] = []
    used: set[str] = set()
    duplicates: list[str] = []

    for index, raw in enumerate(original):
        name = _normalize_label(raw, index)
        candidate = name
        suffix = 2
        while candidate in used:
            candidate = f"{name}__{suffix}"
            suffix += 1
        if candidate != name:
            duplicates.append(name)
        used.add(candidate)
        cleaned.append(candidate)

    return cleaned, tuple(dict.fromkeys(duplicates))


def load_dataset(
    data: bytes,
    filename: str,
    *,
    max_mb: int = MAX_UPLOAD_MB,
) -> LoadedDataset:
    """Validate and parse an uploaded CSV or XLSX file.

    Raises a DatasetLoadError subclass for anything the user can fix
    (wrong type, too large, empty, unreadable).
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be raw file bytes")
    payload = bytes(data)

    file_type = _detect_file_type(filename)
    _check_payload(payload, max_mb, file_type)

    if file_type == "csv":
        frame, header = _read_csv(payload)
    else:
        frame = _read_excel(payload)
        header = _excel_header(payload)

    if frame.shape[1] == 0:
        raise EmptyDatasetError("No columns were found in the file.")

    if len(header) == len(frame.columns):
        cleaned, duplicates = _clean_column_names(header)
    else:
        # Header extraction disagreed with the parser (unusual files);
        # fall back to sanitising whatever pandas produced.
        cleaned, duplicates = _clean_column_names(list(frame.columns))

    frame.columns = cleaned

    return LoadedDataset(
        dataframe=frame,
        filename=filename,
        file_type=file_type,
        size_bytes=len(payload),
        duplicate_columns=duplicates,
    )

"""Data page — upload a dataset and inspect it (Milestone 3).

Loading lives in core/loader.py, profiling in core/profiling.py.
This module only turns those results into UI. A successful load is
also kept in session state so the Charts page can plot the same data
without uploading it twice.
"""

import streamlit as st

from core.loader import (
    MAX_UPLOAD_MB,
    DatasetLoadError,
    LoadedDataset,
    format_file_size,
    load_dataset,
)
from core.profiling import (
    DatasetMetrics,
    categorical_summary,
    column_profiles,
    dataset_metrics,
    numeric_summary,
    quality_issues,
)
from ui.components import (
    card_title,
    render_empty_state,
    render_kpis,
    render_page_header,
    section_card,
    section_heading,
    status_strip,
)

PREVIEW_OPTIONS = (10, 25, 50, 100, 250)

#: Badge label/colour per issue severity.
SEVERITY_STYLE = {
    "high": ("High", "red"),
    "medium": ("Medium", "orange"),
    "low": ("Low", "yellow"),
}


@st.cache_data(show_spinner=False, max_entries=3)
def _load_cached(data: bytes, filename: str, max_mb: int) -> LoadedDataset:
    """Parse an upload once and reuse it across reruns."""
    return load_dataset(data, filename, max_mb=max_mb)


def _render_metrics(metrics: DatasetMetrics) -> None:
    render_kpis(
        [
            {"label": "Rows", "value": f"{metrics.rows:,}", "help": "Data rows in the file (header excluded)."},
            {"label": "Columns", "value": f"{metrics.columns:,}", "help": "Columns after safe name cleanup."},
            {
                "label": "Missing cells",
                "value": f"{metrics.missing_cells:,}",
                "help": "Cells with no value, across the whole dataset.",
            },
            {
                "label": "Duplicate rows",
                "value": f"{metrics.duplicate_rows:,}",
                "help": "Rows that repeat an earlier row exactly.",
            },
            {
                "label": "Memory",
                "value": format_file_size(metrics.memory_bytes),
                "help": "Deep memory usage of the in-memory DataFrame.",
            },
        ]
    )


def _render_preview_tab(frame) -> None:
    if frame.empty:
        st.info("The file has column headers but no rows to preview.")
        return

    left, right = st.columns([2, 5], vertical_alignment="center")
    with left:
        preview_rows = st.selectbox(
            "Rows shown",
            PREVIEW_OPTIONS,
            index=1,
            help="How many rows of the original data to display.",
        )
    with right:
        st.caption(
            f"Showing the first {min(preview_rows, len(frame)):,} of "
            f"{len(frame):,} rows, in the original order."
        )

    shown = min(preview_rows, len(frame))
    st.dataframe(frame.head(preview_rows), width="stretch", height=min(560, 38 * (shown + 1)))


def _render_columns_tab(frame) -> None:
    profiles = column_profiles(frame)
    st.caption(
        "Column types, how much is missing and how many distinct values each "
        "column holds — the profile is read-only."
    )
    st.dataframe(
        profiles,
        hide_index=True,
        width="stretch",
        column_config={
            "Type": st.column_config.TextColumn(
                "Type", help="The pandas dtype of this column."
            ),
            "Missing %": st.column_config.ProgressColumn(
                "Missing %",
                help="Share of this column's cells that are empty.",
                min_value=0,
                max_value=100,
                format="%.1f",
            ),
            "Non-null": st.column_config.NumberColumn("Non-null", format="%d"),
            "Missing": st.column_config.NumberColumn("Missing", format="%d"),
            "Unique": st.column_config.NumberColumn(
                "Unique", help="Distinct non-missing values.", format="%d"
            ),
        },
    )


def _render_statistics_tab(frame) -> None:
    section_heading("Numeric columns")
    numeric = numeric_summary(frame)
    if numeric.empty:
        st.info("No numeric columns in this dataset.")
    else:
        st.dataframe(numeric, hide_index=True, width="stretch")

    section_heading("Categorical columns")
    categorical = categorical_summary(frame)
    if categorical.empty:
        st.info("No text, boolean or category columns in this dataset.")
    else:
        st.dataframe(categorical, hide_index=True, width="stretch")


def _render_quality_tab(frame) -> None:
    issues = quality_issues(frame)
    st.caption(
        "Checks for the most common data-quality problems. Issues are "
        "reported only — nothing has been imputed, dropped or modified."
    )
    if not issues:
        st.success("No common data-quality issues were detected in this dataset.")
    else:
        for issue in issues:
            label, color = SEVERITY_STYLE[issue.severity]
            badge_col, title_col, detail_col = st.columns(
                [1, 2.5, 8], vertical_alignment="center"
            )
            with badge_col:
                st.badge(label, color=color)
            with title_col:
                st.markdown(f"**{issue.category}**")
            with detail_col:
                st.caption(issue.detail)


render_page_header(
    "Data",
    "Upload a dataset and inspect it. Everything happens in memory; nothing is modified or stored.",
    chip="Live",
    chip_tone="live",
)

with section_card():
    card_title(
        "Upload a dataset",
        hint=f"CSV or Excel workbook (.xlsx) · up to {MAX_UPLOAD_MB} MB · inspected in memory only.",
    )
    uploaded = st.file_uploader(
        "Upload a dataset",
        type=["csv", "xlsx"],
        label_visibility="collapsed",
        help=f"CSV or Excel workbook (.xlsx), up to {MAX_UPLOAD_MB} MB.",
    )

if uploaded is None:
    # The uploader was cleared, so the Charts page must not keep drawing
    # a dataset that no longer exists.
    st.session_state["loaded_dataset"] = None
    render_empty_state(
        title="No dataset loaded yet",
        message=(
            "Upload a CSV or Excel workbook to see its size, quality and "
            "column profile. The file is inspected in memory only."
        ),
        hint=f"Supported: .csv and .xlsx · maximum {MAX_UPLOAD_MB} MB",
    )
    st.stop()

with st.status("Inspecting dataset…", expanded=False) as status:
    try:
        dataset = _load_cached(uploaded.getvalue(), uploaded.name, MAX_UPLOAD_MB)
    except DatasetLoadError as exc:
        status.update(label="Could not read the file", state="error")
        st.error(str(exc))
        st.caption("Fix the file and upload it again — nothing was modified on disk.")
        st.stop()
    except Exception as exc:  # keep the page alive on unexpected problems
        status.update(label="Unexpected error", state="error")
        st.error("Something went wrong while reading the file.")
        st.caption(f"Technical detail: {exc}")
        st.stop()
    status.update(label=f"Loaded {dataset.filename}", state="complete")
    # Share with the Charts page for the rest of this session; the frame
    # itself stays in memory only and is never written anywhere.
    st.session_state["loaded_dataset"] = dataset

frame = dataset.dataframe
metrics = dataset_metrics(frame)

section_heading("Dataset summary")
st.caption(
    f"{dataset.filename} · {dataset.file_type.upper()} · "
    f"{format_file_size(dataset.size_bytes)} · inspected in memory, never written to disk"
)

_render_metrics(metrics)

with section_card():
    status_strip(
        [
            ("File", dataset.filename),
            ("Format", dataset.file_type.upper()),
            ("Size", format_file_size(dataset.size_bytes)),
            ("Numeric columns", f"{len(numeric_summary(frame)):,}"),
            ("Categorical columns", f"{len(categorical_summary(frame)):,}"),
        ]
    )

if dataset.duplicate_columns:
    names = ", ".join(f"`{name}`" for name in dataset.duplicate_columns)
    st.warning(
        f"Duplicate column headers found ({names}) and renamed safely as "
        "`name__2`, `name__3`, … in this view only. The uploaded file is unchanged."
    )

preview_tab, columns_tab, statistics_tab, quality_tab = st.tabs(
    ["Preview", "Columns", "Statistics", "Quality"]
)
with preview_tab:
    _render_preview_tab(frame)
with columns_tab:
    _render_columns_tab(frame)
with statistics_tab:
    _render_statistics_tab(frame)
with quality_tab:
    _render_quality_tab(frame)

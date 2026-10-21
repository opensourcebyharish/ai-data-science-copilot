"""Overview page — the working home of the application shell.

Milestone 6+: the page leads with a welcome block, dataset status and
quick actions, then the environment and the roadmap. Everything shown
here is read from core modules or session state — nothing is written.
"""

import pandas as pd
import streamlit as st

from core.environment import MILESTONES, package_statuses, python_version
from core.loader import LoadedDataset
from ui.components import (
    card_title,
    render_kpis,
    render_page_header,
    section_card,
    section_heading,
    status_strip,
)

#: Chip class and badge color per roadmap status (design-system tokens).
STATUS_STYLE = {
    "done": {"chip": "chip--done", "label": "Done", "badge": "green"},
    "in_progress": {"chip": "chip--progress", "label": "In progress", "badge": "yellow"},
    "planned": {"chip": "chip--todo", "label": "Planned", "badge": "gray"},
}


def _render_dataset_status() -> None:
    """Dataset status card — what is loaded, or an honest empty state."""
    dataset: LoadedDataset | None = st.session_state.get("loaded_dataset")
    with section_card():
        card_title("Dataset")
        if dataset is None:
            st.markdown(
                '<p class="status-line">No dataset loaded — upload one to '
                "unlock Charts, the AI Assistant and Models.</p>",
                unsafe_allow_html=True,
            )
            st.link_button("Upload a dataset", url="/data", icon=":material/database:")
            return
        frame = dataset.dataframe
        status_strip(
            [
                ("File", dataset.filename),
                ("Rows", f"{len(frame):,}"),
                ("Columns", f"{frame.shape[1]:,}"),
                ("Type", dataset.file_type.upper()),
            ]
        )
        st.caption("Loaded in memory for this session only — never written to disk.")


def _render_quick_actions() -> None:
    """Pillars of the workflow as real navigation links."""
    with section_card():
        card_title("Quick actions", hint="Jump straight into the workspace.")
        data_col, charts_col, ai_col, models_col = st.columns(4)
        with data_col:
            st.link_button("Data", url="/data", icon=":material/database:", width="stretch")
        with charts_col:
            st.link_button("Charts", url="/charts", icon=":material/monitoring:", width="stretch")
        with ai_col:
            st.link_button("AI Assistant", url="/ai", icon=":material/auto_awesome:", width="stretch")
        with models_col:
            st.link_button(
                "Models",
                url="/models",
                icon=":material/model_training:",
                width="stretch",
                type="primary",
            )


def _render_recent_runs() -> None:
    """Recent model runs from the history database, when one is reachable."""
    try:
        from core.storage import load_runs

        runs = load_runs()
    except Exception:  # a missing or unreadable database must not break the page
        runs = []
    if not runs:
        return
    with section_card():
        card_title("Recent model runs")
        for run in runs[:5]:
            target = f" → {run['target_column']}" if run["target_column"] else ""
            metrics_str = "  ·  ".join(f"{k}: {v:.4g}" for k, v in run["metrics"].items())
            st.markdown(
                f"**#{run['id']}** &nbsp; `{run['model_name']}`"
                f" &nbsp;·&nbsp; {run['task_type'].title()}{target}"
            )
            st.caption(metrics_str or "No metrics recorded.")


render_page_header(
    "Overview",
    "A premium workspace for exploring data, training models and asking "
    "grounded questions — live status of the project and its environment.",
    chip="Live",
    chip_tone="live",
)

statuses = package_statuses()
installed_count = sum(1 for status in statuses if status.installed)
done_count = sum(1 for milestone in MILESTONES if milestone.status == "done")

render_kpis(
    [
        {
            "label": "Python",
            "value": python_version(),
            "help": "Interpreter currently running the app.",
        },
        {
            "label": "Dependencies",
            "value": f"{installed_count} / {len(statuses)}",
            "help": "Tracked packages from requirements.txt found in this environment.",
        },
        {
            "label": "Milestones",
            "value": f"{done_count} / {len(MILESTONES)}",
            "help": "Completed milestones in the project roadmap.",
        },
        {
            "label": "Gemini AI",
            "value": "Not connected",
            "help": "AI features arrive in Milestone 5. No API key is loaded in this build.",
        },
    ]
)

_render_dataset_status()
_render_quick_actions()
_render_recent_runs()

section_heading("Environment")
st.dataframe(
    pd.DataFrame(
        {
            "Package": [status.name for status in statuses],
            "Version": [status.version or "—" for status in statuses],
            "Status": ["Installed" if status.installed else "Missing" for status in statuses],
        }
    ),
    hide_index=True,
    width="stretch",
    column_config={
        "Status": st.column_config.TextColumn(
            "Status",
            help="Is the package installed in the active virtual environment?",
        )
    },
)

section_heading("Roadmap")
for milestone in MILESTONES:
    style = STATUS_STYLE[milestone.status]
    id_col, title_col, status_col, summary_col = st.columns(
        [1, 3.5, 1.5, 6], vertical_alignment="center"
    )
    with id_col:
        st.badge(milestone.id, color=style["badge"])
    with title_col:
        st.markdown(f"**{milestone.title}**")
    with status_col:
        st.markdown(
            f'<span class="chip {style["chip"]}">{style["label"]}</span>',
            unsafe_allow_html=True,
        )
    with summary_col:
        st.caption(milestone.summary)

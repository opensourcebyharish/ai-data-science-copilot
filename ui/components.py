"""Reusable UI building blocks for the application shell.

Every helper is a thin wrapper around native Streamlit elements plus the
design-system CSS in ui/styles.css. No custom frontend dependencies, and
no helper changes how a native widget behaves — only how it looks.
"""

from contextlib import contextmanager
from html import escape
from typing import Iterator, Literal

import streamlit as st

ChipTone = Literal["live", "soon", "done", "progress", "todo"]


def render_page_header(
    title: str,
    subtitle: str,
    *,
    chip: str | None = None,
    chip_tone: ChipTone = "live",
) -> None:
    """Page title row with an optional status chip, subtitle and hairline."""
    chip_html = f'<span class="chip chip--{chip_tone}">{escape(chip)}</span>' if chip else ""
    st.markdown(
        f"""
        <div class="page-header">
          <div class="page-header__row">
            <h1 class="page-title">{escape(title)}</h1>
            {chip_html}
          </div>
          <p class="page-subtitle">{escape(subtitle)}</p>
          <div class="page-header__rule" aria-hidden="true"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_heading(text: str) -> None:
    """Section heading inside a page, styled by the design system."""
    st.markdown(f'<h2 class="section-title">{escape(text)}</h2>', unsafe_allow_html=True)


def card_title(
    text: str,
    *,
    hint: str | None = None,
) -> None:
    """Small heading used at the top of a section card."""
    hint_html = f'<p class="status-line">{escape(hint)}</p>' if hint else ""
    st.markdown(
        f'<p class="card-title">{escape(text)}</p>{hint_html}',
        unsafe_allow_html=True,
    )


@contextmanager
def section_card(*, padding: bool = True) -> Iterator[None]:
    """Native bordered container used as a section card.

    Streamlit's own ``st.container(border=True)`` provides a real, accessible
    border wrapper, so the design system only has to refine its colours and
    radii — no fragile div-wrapping hacks.
    """
    container = st.container(border=True)
    with container:
        yield


def status_strip(items: list[tuple[str, str]]) -> None:
    """Compact label/value strip (dataset status, environment summaries)."""
    cells = "".join(
        '<div class="kv__item">'
        f'<div class="kv__label">{escape(label)}</div>'
        f'<div class="kv__value">{escape(value)}</div>'
        "</div>"
        for label, value in items
    )
    st.markdown(f'<div class="kv">{cells}</div>', unsafe_allow_html=True)


def render_kpis(items: list[dict]) -> None:
    """Row of KPI cards backed by the native st.metric element.

    Each item is a dict with:
      - label (str, required)
      - value (str, required)
      - delta (str, optional)
      - help (str, optional) — shown as a tooltip on hover/focus
    """
    if not items:
        return
    columns = st.columns(len(items), gap="medium")
    for column, item in zip(columns, items):
        with column:
            st.metric(
                label=item["label"],
                value=item["value"],
                delta=item.get("delta"),
                help=item.get("help"),
            )


def render_empty_state(
    title: str,
    message: str,
    hint: str | None = None,
    *,
    icon: str = "🗂️",
) -> None:
    """Friendly empty state shown before any content exists."""
    hint_html = f'<p class="empty-state__hint">{escape(hint)}</p>' if hint else ""
    st.markdown(
        f"""
        <div class="empty-state">
          <div class="empty-state__icon" aria-hidden="true">{icon}</div>
          <p class="empty-state__title">{escape(title)}</p>
          <p class="empty-state__message">{escape(message)}</p>
          {hint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_planned_page(
    title: str,
    milestone: str,
    summary: str,
    will_include: list[str],
) -> None:
    """Honest placeholder for a page that a later milestone will build.

    No controls are rendered here: a button that does nothing would be
    worse than no button at all.
    """
    render_page_header(title, subtitle=summary, chip="Planned", chip_tone="soon")
    st.markdown(
        f"""
        <div class="planned-banner">
          <strong>{escape(milestone)} will build this page.</strong>
          Nothing here is implemented yet — no fake controls are shown.
          Use the sidebar, or the link below, to return to the working Overview.
        </div>
        """,
        unsafe_allow_html=True,
    )
    section_heading("What will live here")
    for item in will_include:
        st.markdown(f"- {escape(item)}")
    # The default page is served at the root URL, so this real link needs
    # no page registry and works however the module is executed.
    st.link_button("Back to Overview", url="/", icon=":material/arrow_back:")

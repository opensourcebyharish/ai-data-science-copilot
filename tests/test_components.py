"""Rendering tests for the reusable UI building blocks."""

from streamlit.testing.v1 import AppTest

COMPONENT_HARNESS = """
import streamlit as st
from ui.components import render_page_header, render_kpis, render_planned_page

render_page_header("Quarterly revenue", "A short subtitle.", chip="Live", chip_tone="live")
render_kpis([
    {"label": "Rows", "value": "1,204"},
    {"label": "Columns", "value": "12", "delta": "+3"},
])
render_planned_page(
    title="Data",
    milestone="Milestone 3",
    summary="Upload a dataset.",
    will_include=["CSV upload", "Preview table"],
)
"""


def _run_harness() -> AppTest:
    return AppTest.from_string(COMPONENT_HARNESS).run()


def test_components_render_without_exceptions():
    assert not _run_harness().exception


def test_kpis_use_native_metrics():
    assert len(_run_harness().metric) == 2


def test_header_chips_and_planned_banner_are_rendered():
    rendered = " ".join(str(element.value) for element in _run_harness().markdown)
    assert "Quarterly revenue" in rendered
    assert "chip--live" in rendered
    assert "Milestone 3 will build this page" in rendered
    assert "chip--soon" in rendered


def test_escape_helper_protects_against_html_injection():
    at = AppTest.from_string(
        """
        from ui.components import render_page_header
        render_page_header("<script>alert(1)</script>", "safe")
        """
    ).run()
    assert not at.exception
    rendered = " ".join(str(element.value) for element in at.markdown)
    assert "<script>" not in rendered


def test_new_components_render_without_exceptions():
    at = AppTest.from_string(
        """
        import streamlit as st
        from ui.components import card_title, section_card, status_strip

        card_title("Dataset", hint="Loaded in memory only.")
        with section_card():
            st.markdown("inside a card")
        status_strip([("Rows", "1,204"), ("Columns", "12")])
        """
    ).run()

    assert not at.exception
    rendered = " ".join(str(element.value) for element in at.markdown)
    assert "card-title" in rendered
    assert "kv__value" in rendered
    assert "inside a card" in rendered


def test_section_card_uses_a_native_bordered_container():
    """Cards are a real Streamlit container, not a div-wrapping hack."""
    at = AppTest.from_string(
        """
        import streamlit as st
        from ui.components import section_card

        with section_card():
            st.markdown("card body")
        """
    ).run()

    assert not at.exception
    assert "card body" in " ".join(str(e.value) for e in at.markdown)


def test_page_header_renders_a_brand_hairline():
    rendered = " ".join(str(element.value) for element in _run_harness().markdown)
    assert "page-header__rule" in rendered

"""Loads the design-system CSS (design tokens + component styles)."""

from functools import lru_cache
from pathlib import Path

import streamlit as st

STYLES_PATH = Path(__file__).with_name("styles.css")


@lru_cache(maxsize=1)
def load_css() -> str:
    """Read the stylesheet once per process and cache it."""
    return STYLES_PATH.read_text(encoding="utf-8")


def inject_theme() -> None:
    """Inject the design system into the current page run."""
    st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)

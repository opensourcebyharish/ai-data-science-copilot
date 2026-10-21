"""Tests for the design system: tokens, accessibility and theme config."""

import tomllib
from pathlib import Path

from ui.theme import STYLES_PATH, load_css

ROOT = Path(__file__).resolve().parents[1]


def test_css_file_lives_next_to_theme_module():
    assert STYLES_PATH.is_file()


def test_core_design_tokens_are_defined():
    css = load_css()
    for token in ("--c-bg: #0b1220", "--c-cyan: #22d3ee", "--c-violet: #a78bfa"):
        assert token in css


def test_reduced_motion_is_supported():
    assert "@media (prefers-reduced-motion: reduce)" in load_css()


def test_keyboard_focus_is_visible():
    assert ":focus-visible" in load_css()


def test_streamlit_theme_is_dark_navy():
    config_path = ROOT / ".streamlit" / "config.toml"
    assert config_path.is_file()
    with config_path.open("rb") as handle:
        theme = tomllib.load(handle)["theme"]
    assert theme["base"] == "dark"
    assert theme["primaryColor"] == "#22D3EE"
    assert theme["backgroundColor"] == "#0B1220"
    assert theme["textColor"] == "#E2E8F0"


def test_extended_design_tokens_are_defined():
    """The premium design system adds spacing, radius, shadow and gradient scales."""
    css = load_css()
    for token in (
        "--radius-xl: 18px",
        "--radius-pill: 999px",
        "--gap-5: 2.5rem",
        "--c-bg-elevated:",
        "--g-brand:",
        "--shadow-3:",
        "--ease-out:",
    ):
        assert token in css, token


def test_streamlit_chrome_is_restyled_with_stable_selectors():
    """Only stable Streamlit test ids and our own classes are targeted."""
    css = load_css()
    for selector in (
        '[data-testid="stSidebar"]',
        '[data-testid="stSidebarNav"] a[aria-current="page"]',
        '[data-testid="stHeader"]',
        '[data-testid="stExpander"]',
        '[data-testid="stTabs"]',
        '[data-testid="stDataFrame"]',
        '[data-testid="stChatMessage"]',
        '[data-testid="stVerticalBlockBorderWrapper"]',
    ):
        assert selector in css, selector


def test_skeleton_loading_state_is_defined_and_animated():
    css = load_css()
    assert '[data-testid="stSkeleton"]' in css
    assert "dsc-shimmer" in css


def test_reduced_motion_also_disables_hover_lift():
    """Motion is restrained for users who ask for less of it."""
    css = load_css()
    assert "prefers-reduced-motion" in css
    assert "transform: none" in css


def test_responsive_breakpoint_is_defined():
    assert "@media (max-width: 900px)" in load_css()

"""Smoke tests for the application shell: navigation and page rendering."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.loader import LoadedDataset, load_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolated_run_history(tmp_path, monkeypatch):
    """Keep the Overview page's "recent runs" away from the real database.

    The Overview reads whatever run history exists locally; tests must not
    depend on the developer's runs and must never write test rows into them.
    """
    monkeypatch.setenv("ML_RUNS_DB_PATH", str(tmp_path / "test_runs.db"))


def _run_app() -> AppTest:
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()


def test_app_starts_without_exceptions():
    assert not _run_app().exception


def test_overview_is_the_default_page_with_four_kpis():
    assert len(_run_app().metric) == 4


def test_overview_shows_live_status_and_roadmap():
    rendered = " ".join(str(element.value) for element in _run_app().markdown)
    assert "chip--live" in rendered
    assert "Design system & app shell" in rendered


def test_models_page_renders_without_planned_placeholder():
    """Models page is now implemented (M6) — no longer a planned stub."""
    at = AppTest.from_file(str(ROOT / "views" / "models.py"), default_timeout=15).run()
    rendered = " ".join(str(element.value) for element in at.markdown)
    assert not at.exception
    # Old placeholder text must be gone
    assert "will build this page" not in rendered
    # New page header must be present
    assert "Models" in rendered


def test_data_page_shows_empty_state_without_a_file():
    at = AppTest.from_file(str(ROOT / "views" / "data.py"), default_timeout=15).run()

    rendered = " ".join(str(element.value) for element in at.markdown)
    assert not at.exception
    assert "No dataset loaded yet" in rendered
    assert "maximum 50 MB" in rendered


def _sample_dataset() -> LoadedDataset:
    csv = (
        b"date,region,sales,units\n"
        b"2024-01-01,north,100,5\n"
        b"2024-01-02,south,150,7\n"
        b"2024-01-03,north,120,6\n"
        b"2024-01-04,south,200,9\n"
        b"2024-01-05,north,175,8\n"
    )
    return load_dataset(csv, "sales.csv")


def _charts_app() -> AppTest:
    """Charts page with a dataset in session state, as the Data page stores it."""
    at = AppTest.from_file(str(ROOT / "views" / "charts.py"), default_timeout=15).run()
    at.session_state["loaded_dataset"] = _sample_dataset()
    return at.run()


def test_charts_page_shows_empty_state_without_a_dataset():
    at = AppTest.from_file(str(ROOT / "views" / "charts.py"), default_timeout=15).run()

    rendered = " ".join(str(element.value) for element in at.markdown)
    assert not at.exception
    assert "No dataset loaded yet" in rendered
    assert at.link_button[0].value == "Go to the Data page"


def test_charts_page_renders_distribution_by_default():
    at = _charts_app()

    assert not at.exception
    assert at.segmented_control[0].value == "Distribution"
    assert len(at.slider) == 1  # bins
    assert len(at.metric) == 6  # count, mean, median, std, min, max
    captions = " ".join(caption.value for caption in at.caption)
    assert "sales.csv" in captions


def test_charts_page_renders_every_chart_type():
    at = _charts_app()
    expected_selectboxes = {"Relationship": 3, "Correlation": 0, "Time series": 4}

    for chart_type, count in expected_selectboxes.items():
        at.session_state["chart_type"] = chart_type
        at.run()
        assert not at.exception, chart_type
        assert at.segmented_control[0].value == chart_type
        assert len(at.selectbox) == count, chart_type


def test_charts_page_never_reports_an_exception_for_any_type():
    at = _charts_app()
    for chart_type in ("Distribution", "Relationship", "Correlation", "Time series"):
        at.session_state["chart_type"] = chart_type
        at.run()
        assert not at.exception, chart_type


# --- Overview page: premium workspace sections ------------------------------------


def test_overview_shows_dataset_status_and_quick_actions_without_a_dataset():
    at = _run_app()
    rendered = " ".join(str(element.value) for element in at.markdown)

    assert not at.exception
    assert "Dataset" in rendered
    assert "No dataset loaded" in rendered
    assert "Quick actions" in rendered
    # Quick actions are real navigation links, none of them dead.
    urls = {button.url for button in at.link_button}
    assert {"/data", "/charts", "/ai", "/models"} <= urls


def test_overview_reports_the_loaded_dataset_in_the_status_card():
    at = _run_app()
    at.session_state["loaded_dataset"] = _sample_dataset()
    at = at.run()

    assert not at.exception
    rendered = " ".join(str(element.value) for element in at.markdown)
    assert "sales.csv" in rendered
    assert "kv__value" in rendered


def test_overview_without_history_hides_the_recent_runs_card():
    """No runs in the throwaway database → no empty "recent runs" clutter."""
    at = _run_app()
    rendered = " ".join(str(element.value) for element in at.markdown)
    assert "Recent model runs" not in rendered

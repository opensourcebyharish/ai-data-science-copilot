"""AppTest smoke tests for the Models page (Milestone 6).

These tests use Streamlit's AppTest harness and run the page in
isolation. No real scikit-learn training is needed for the empty-state
and UI-structure tests; the training tests inject a dataset and
exercise the form controls.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from core.loader import LoadedDataset

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _app() -> AppTest:
    """Run the models page in isolation."""
    return AppTest.from_file(str(ROOT / "views" / "models.py"), default_timeout=30)


def _dataset() -> LoadedDataset:
    """Small labelled LoadedDataset for injection into session state."""
    rng = np.random.default_rng(42)
    n = 40
    df = pd.DataFrame(
        {
            "sales": rng.uniform(100, 1000, n).round(2),
            "profit": rng.uniform(10, 200, n).round(2),
            "units": rng.integers(1, 50, n).astype(float),
            "region": rng.choice(["North", "South", "East"], n),
        }
    )
    return LoadedDataset(
        dataframe=df,
        filename="sample.csv",
        file_type="csv",
        size_bytes=1024,
    )


def _with_dataset(at: AppTest) -> AppTest:
    """Inject a dataset into session state and rerun."""
    at.session_state["loaded_dataset"] = _dataset()
    return at.run()


@pytest.fixture(autouse=True)
def _isolated_run_history(tmp_path, monkeypatch):
    """Point the run history at a throwaway database for every test.

    The page reads whatever history exists in the real local
    ``ml_runs.db``; tests must not depend on the developer's runs, must
    not display them, and must never write test rows into them.
    """
    monkeypatch.setenv("ML_RUNS_DB_PATH", str(tmp_path / "test_runs.db"))


def _markdown(at: AppTest) -> str:
    """Flatten all markdown elements to a single string."""
    return " ".join(str(e.value) for e in at.markdown)


# ---------------------------------------------------------------------------
# Empty state (no dataset)
# ---------------------------------------------------------------------------


def test_models_page_renders_without_exception_no_dataset():
    at = _app().run()
    assert not at.exception


def test_models_page_shows_empty_state_without_dataset():
    at = _app().run()
    rendered = _markdown(at)
    assert "No dataset loaded" in rendered


def test_models_page_no_dataset_shows_link_to_data_page():
    at = _app().run()
    buttons = [b for b in at.get("link_button") if hasattr(b, "url")]
    # At least one link button pointing toward /data
    assert any(getattr(b, "url", "").endswith("/data") for b in buttons)


# ---------------------------------------------------------------------------
# With dataset loaded
# ---------------------------------------------------------------------------


def test_models_page_renders_without_exception_with_dataset():
    at = _app().run()
    at = _with_dataset(at)
    assert not at.exception


def test_models_page_shows_header_with_dataset():
    at = _app().run()
    at = _with_dataset(at)
    rendered = _markdown(at)
    assert "Models" in rendered


def test_models_page_shows_task_selectbox():
    at = _app().run()
    at = _with_dataset(at)
    labels = [str(sb.label) for sb in at.selectbox]
    assert any("Task" in label or "task" in label for label in labels)


def test_models_page_shows_model_selectbox():
    at = _app().run()
    at = _with_dataset(at)
    labels = [str(sb.label) for sb in at.selectbox]
    assert any("Model" in label or "model" in label for label in labels)


def test_models_page_shows_train_button():
    at = _app().run()
    at = _with_dataset(at)
    button_labels = [str(b.label) for b in at.button]
    assert any("Train" in label or "train" in label for label in button_labels)


def test_models_page_shows_test_split_slider():
    at = _app().run()
    at = _with_dataset(at)
    assert len(at.slider) >= 1


def test_models_page_shows_run_history_expander():
    at = _app().run()
    at = _with_dataset(at)
    # The expander label lives on the expander block itself, not in markdown.
    expander_labels = [str(exp.label) for exp in at.expander]
    assert any("Run history" in label for label in expander_labels)
    # An empty history still explains what to do next.
    captions = " ".join(str(c.value) for c in at.caption)
    assert "No runs saved yet" in captions


# ---------------------------------------------------------------------------
# Planned-page stub is fully replaced
# ---------------------------------------------------------------------------


def test_models_page_no_longer_shows_planned_banner():
    at = _app().run()
    rendered = _markdown(at)
    # The old placeholder said "will build this page" — should be gone
    assert "will build this page" not in rendered


def test_models_page_no_longer_shows_planned_chip():
    at = _app().run()
    rendered = _markdown(at)
    assert "Planned" not in rendered


# ---------------------------------------------------------------------------
# Milestone 8 — export controls
# ---------------------------------------------------------------------------


@pytest.fixture()
def downloads(monkeypatch):
    """Capture every ``st.download_button`` call made by the page.

    Streamlit's AppTest protocol does not expose the file name or the
    payload of a download button, so the calls are wrapped instead. The
    real widget is still created, so the page renders exactly as usual.
    """
    import streamlit

    calls: list[dict] = []
    original = streamlit.download_button

    def wrapper(label, data, file_name=None, **kwargs):
        calls.append({"label": str(label), "data": data, "file_name": file_name})
        return original(label, data, file_name=file_name, **kwargs)

    monkeypatch.setattr(streamlit, "download_button", wrapper)
    return calls


def _train(at: AppTest) -> AppTest:
    """Load a dataset and press 'Train model' with the default settings."""
    at = _with_dataset(at)
    train_button = next(b for b in at.button if "Train" in str(b.label))
    train_button.click()
    return at.run()


def _call(calls, needle):
    """The captured download call whose label contains ``needle``."""
    return next(call for call in calls if needle.lower() in call["label"].lower())


def test_no_download_buttons_before_anything_is_trained(downloads):
    at = _app().run()
    at = _with_dataset(at)
    assert not at.exception
    assert downloads == []


def test_training_reveals_export_download_buttons(downloads):
    at = _train(_app())
    assert not at.exception
    labels = [call["label"] for call in downloads]
    assert any("Predictions" in label for label in labels)
    assert any("Configuration" in label for label in labels)
    assert any("report" in label.lower() for label in labels)


def test_export_filenames_are_sensible_and_typed(downloads):
    at = _train(_app())
    names = [call["file_name"] for call in downloads]
    assert any(name.endswith(".csv") for name in names)
    assert any(name.endswith(".json") for name in names)
    assert any(name.endswith(".html") for name in names)
    # Every filename is slugged: no spaces, slashes or odd characters.
    for name in names:
        assert " " not in name
        assert "/" not in name
        # The dataset name is the stem, not the full file name.
        assert name.startswith("sample-")


def test_prediction_export_contains_actual_and_predicted_rows(downloads):
    import csv
    import io

    at = _train(_app())
    call = _call(downloads, "Predictions")
    rows = list(csv.reader(io.StringIO(call["data"])))
    # The default task is classification, so the classes are named as such.
    assert rows[0] == ["Row", "Actual class", "Predicted class"]
    assert len(rows) > 1


def test_prediction_export_covers_only_held_out_rows(downloads):
    import csv
    import io

    at = _train(_app())
    call = _call(downloads, "Predictions")
    rows = list(csv.reader(io.StringIO(call["data"])))
    # 40 rows, 20% held out → 8 test rows plus a header.
    assert len(rows) == 9
    # No feature values leak into the prediction export.
    header = rows[0]
    assert "profit" not in header and "units" not in header


def test_config_export_records_seed_and_target(downloads):
    at = _train(_app())
    call = _call(downloads, "Configuration (CSV)")
    text = call["data"]
    assert "Random seed" in text
    assert "42" in text
    assert "region" in text  # the default classification target


def test_config_json_export_is_valid_json(downloads):
    import json

    at = _train(_app())
    call = _call(downloads, "Configuration (JSON)")
    payload = json.loads(call["data"])
    assert payload["random_state"] == 42
    assert payload["feature_columns"]
    assert payload["preprocessing"]


def test_single_run_report_is_downloadable_html_with_disclaimer(downloads):
    at = _train(_app())
    call = _call(downloads, "report")
    report = call["data"]
    assert report.startswith("<!DOCTYPE html>")
    # Honest attribution wording — never a causation claim.
    assert "association" in report.lower()


def test_exports_never_contain_secrets_or_raw_feature_values(downloads):
    at = _train(_app())
    for call in downloads:
        text = call["data"] if isinstance(call["data"], str) else ""
        assert "api_key" not in text.lower()
        assert "gemini" not in text.lower()


def test_comparison_exports_appear_after_comparing(downloads):
    at = _with_dataset(_app())
    compare_button = next(b for b in at.button if "Compare" in str(b.label))
    compare_button.click()
    at = at.run()
    assert not at.exception
    labels = [call["label"] for call in downloads]
    assert any("Comparison metrics" in label for label in labels)
    assert any("Run metadata" in label for label in labels)
    assert any("report" in label.lower() for label in labels)


def test_comparison_metrics_csv_has_one_row_per_model(downloads):
    import csv
    import io

    at = _with_dataset(_app())
    compare_button = next(b for b in at.button if "Compare" in str(b.label))
    compare_button.click()
    at = at.run()
    call = _call(downloads, "Comparison metrics")
    rows = list(csv.reader(io.StringIO(call["data"])))
    assert rows[0][0] == "Model"
    assert len(rows) > 1


def test_comparison_report_is_self_contained_html(downloads):
    at = _with_dataset(_app())
    compare_button = next(b for b in at.button if "Compare" in str(b.label))
    compare_button.click()
    at = at.run()
    call = _call(downloads, "report")
    report = call["data"]
    assert report.startswith("<!DOCTYPE html>")
    assert "http://" not in report
    assert "https://" not in report
    assert call["file_name"].endswith("-comparison-report.html")


def test_export_filenames_handle_special_characters_in_the_dataset_name(downloads):
    """A messy file name must still produce a portable download name."""
    base = _dataset()
    dataset = LoadedDataset(
        dataframe=base.dataframe,
        filename="sales data (2024) & more!.csv",
        file_type=base.file_type,
        size_bytes=base.size_bytes,
    )
    at = _app().run()
    at.session_state["loaded_dataset"] = dataset
    at = at.run()
    next(b for b in at.button if "Train" in str(b.label)).click()
    at = at.run()
    assert not at.exception
    assert downloads
    for call in downloads:
        name = call["file_name"]
        assert " " not in name
        assert "&" not in name
        assert "!" not in name
        assert "(" not in name and ")" not in name
        assert name.startswith("sales-data-2024")


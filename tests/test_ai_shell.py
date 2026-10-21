"""AppTest coverage for the AI Assistant page (Milestone 5).

Everything runs offline: the Gemini service is replaced with scripted
responses, so the suite needs no API key and never touches the network.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from core import gemini
from core.ai_schemas import AnalysisPlan, AssistantAnswer, CodeExample
from core.loader import LoadedDataset, load_dataset

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _dataset() -> LoadedDataset:
    csv = (
        b"date,region,product,sales,profit\n"
        b"2024-01-01,north,laptop,100,10\n"
        b"2024-01-01,south,phone,200,50\n"
        b"2024-01-02,north,laptop,300,20\n"
        b"2024-01-03,south,phone,400,40\n"
    )
    return load_dataset(csv, "sales.csv")


def _region_sum_plan() -> AnalysisPlan:
    return AnalysisPlan(
        operation="group_aggregate",
        group_by="region",
        value_column="sales",
        aggregation="sum",
    )


def _fake_gemini(monkeypatch, *responses) -> list[dict]:
    """Install a scripted Gemini and make the page report as configured."""
    calls: list[dict] = []
    queue = list(responses)

    def fake(prompt, schema, **kwargs):
        calls.append({"prompt": prompt, "schema": schema, **kwargs})
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, schema)
        return outcome

    monkeypatch.setattr(gemini, "generate_structured", fake)
    monkeypatch.setattr(
        gemini,
        "api_config",
        lambda *args, **kwargs: gemini.ApiConfig(
            configured=True, source="environment variable", model="test-model"
        ),
    )
    return calls


def _app() -> AppTest:
    return AppTest.from_file(str(ROOT / "views" / "ai.py"), default_timeout=30)


def _with_dataset(at: AppTest) -> AppTest:
    at.session_state["loaded_dataset"] = _dataset()
    return at


def _captions(at: AppTest) -> str:
    return " ".join(element.value for element in at.caption)


def _markdown(at: AppTest) -> str:
    return " ".join(str(element.value) for element in at.markdown)


# --- Offline states -------------------------------------------------------------


def test_page_shows_empty_state_without_a_dataset():
    at = _app().run()

    assert not at.exception
    assert "No dataset loaded" in _markdown(at)
    assert at.link_button[0].value == "Go to the Data page"


def test_page_explains_the_missing_configuration():
    at = _with_dataset(_app()).run()

    assert not at.exception
    warnings = " ".join(element.value for element in at.warning)
    assert "not configured" in warnings
    assert "GEMINI_API_KEY" in at.chat_input[0].placeholder
    # No dead controls: suggestion chips only appear once a key exists.
    labels = [button.label for button in at.button]
    assert not any(label.startswith("Which") for label in labels)
    assert "sales.csv" in _captions(at)
    assert "4,000 characters" in _captions(at)


def test_privacy_note_names_what_is_sent():
    at = _with_dataset(_app()).run()
    captions = _captions(at)
    assert "column names" in captions
    assert "top-N" in captions


# --- Chat flow (mocked Gemini) ---------------------------------------------------


def test_chat_flow_calculates_and_explains(monkeypatch):
    _fake_gemini(
        monkeypatch,
        _region_sum_plan(),
        AssistantAnswer(summary="The south leads with **600** in total sales."),
    )
    at = _with_dataset(_app()).run()
    at.chat_input[0].set_value("Which region has the highest total sales?").run()

    assert not at.exception
    assert len(at.dataframe) == 1
    assert "Computed with pandas" in _captions(at)
    assert "The south leads" in _markdown(at)
    assert "analysis · sum(sales) by region" in _captions(at)

    messages = at.session_state["ai_messages"]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    table = messages[1]["table"]
    assert list(table["region"]) == ["south", "north"]
    assert list(table["Total of sales"]) == [600.0, 400.0]


def test_conversation_persists_across_reruns(monkeypatch):
    _fake_gemini(monkeypatch, _region_sum_plan(), AssistantAnswer(summary="South wins."))
    at = _with_dataset(_app()).run()
    at.chat_input[0].set_value("Who sells more?").run()

    at = at.run()  # a plain rerun must not lose or duplicate the conversation
    assert not at.exception
    assert "South wins" in _markdown(at)
    assert len(at.session_state["ai_messages"]) == 2


def test_clear_conversation_resets_everything(monkeypatch):
    _fake_gemini(monkeypatch, _region_sum_plan(), AssistantAnswer(summary="South wins."))
    at = _with_dataset(_app()).run()
    at.chat_input[0].set_value("Who sells more?").run()

    clear = next(button for button in at.button if button.label == "Clear conversation")
    clear.click().run()

    assert not at.exception
    assert at.session_state["ai_messages"] == []
    assert "South wins" not in _markdown(at)


def test_suggested_question_runs_a_full_turn(monkeypatch):
    _fake_gemini(monkeypatch, _region_sum_plan(), AssistantAnswer(summary="The south."))
    at = _with_dataset(_app()).run()

    suggestion = next(
        button
        for button in at.button
        if button.label == "Which region has the highest total sales?"
    )
    suggestion.click().run()

    assert not at.exception
    assert len(at.session_state["ai_messages"]) == 2
    assert "The south" in _markdown(at)


def test_api_errors_render_as_friendly_messages(monkeypatch):
    _fake_gemini(monkeypatch, gemini.GeminiRateLimitError())
    at = _with_dataset(_app()).run()
    at.chat_input[0].set_value("Anything at all?").run()

    assert not at.exception
    errors = " ".join(element.value for element in at.error)
    assert "rate limit or quota" in errors
    assert at.session_state["ai_messages"][-1]["error"] is True


def test_invalid_plans_become_limitations_not_crashes(monkeypatch):
    _fake_gemini(
        monkeypatch,
        AnalysisPlan(operation="column_stats", value_column="not_a_column"),
    )
    at = _with_dataset(_app()).run()
    at.chat_input[0].set_value("Statistics for the bogus column, please").run()

    assert not at.exception
    warnings = " ".join(element.value for element in at.warning)
    assert "I could not run that analysis" in warnings
    assert len(at.session_state["ai_messages"]) == 2


def test_cannot_answer_plan_explains_the_limitation(monkeypatch):
    _fake_gemini(
        monkeypatch,
        AnalysisPlan(
            operation="cannot_answer",
            reason_if_impossible="That needs data this dataset does not contain.",
        ),
    )
    at = _with_dataset(_app()).run()
    at.chat_input[0].set_value("Predict next quarter's prices").run()

    assert not at.exception
    assert "does not contain" in _markdown(at)
    # No calculation happened, so nothing was sent for a second call.
    assert at.session_state["ai_messages"][1].get("table") is None


# --- Code examples ----------------------------------------------------------------


def test_code_example_renders_as_inert_text(monkeypatch):
    sentinel = "raise RuntimeError('EXECUTED')\ndf.groupby('region')['sales'].sum()"
    _fake_gemini(
        monkeypatch,
        CodeExample(
            language="python",
            title="Sales by region",
            code=sentinel,
            explanation="Totals sales for each region using a group-by.",
            referenced_columns=["region", "sales"],
        ),
    )
    at = _with_dataset(_app()).run()
    at.text_input[0].set_value("total sales by region")
    submit = next(
        button for button in at.form_submit_button if button.label == "Generate example"
    )
    submit.click().run()

    assert not at.exception  # the sentinel code was never executed
    assert len(at.code) == 1
    assert at.code[0].value == sentinel  # displayed verbatim
    assert len(at.download_button) == 1
    assert "Totals sales" in _markdown(at)
    assert "never executes generated code" in _captions(at)


def test_code_generation_survives_reruns(monkeypatch):
    _fake_gemini(
        monkeypatch,
        CodeExample(
            language="sql", title="Top regions", code="SELECT region FROM sales",
            explanation="Lists regions.", referenced_columns=["region"],
        ),
    )
    at = _with_dataset(_app()).run()
    at.text_input[0].set_value("list regions")
    next(b for b in at.form_submit_button if b.label == "Generate example").click().run()
    at = at.run()

    assert not at.exception
    assert "no database is connected" in _captions(at)


def test_empty_codegen_task_warns_without_calling_gemini(monkeypatch):
    calls = _fake_gemini(monkeypatch)
    at = _with_dataset(_app()).run()
    next(b for b in at.form_submit_button if b.label == "Generate example").click().run()

    assert not at.exception
    assert "Describe what the code should do" in " ".join(
        element.value for element in at.warning
    )
    assert calls == []  # nothing was sent for an empty task


def test_blocked_code_is_discarded_with_an_explanation(monkeypatch):
    _fake_gemini(
        monkeypatch,
        CodeExample(
            language="python", title="Bad", code="import os\nos.system('dir')",
            explanation="Nope.", referenced_columns=[],
        ),
    )
    at = _with_dataset(_app()).run()
    at.text_input[0].set_value("do something unsafe")
    next(b for b in at.form_submit_button if b.label == "Generate example").click().run()

    assert not at.exception
    assert len(at.code) == 0  # the dangerous snippet was never displayed
    warnings = " ".join(element.value for element in at.warning)
    assert "blocked constructs" in warnings


# --- Follow-up questions -----------------------------------------------------------


def test_follow_up_keeps_the_earlier_context(monkeypatch):
    calls = _fake_gemini(
        monkeypatch,
        _region_sum_plan(),
        AssistantAnswer(summary="The south leads."),
        AnalysisPlan(operation="overview"),
        AssistantAnswer(summary="Here is the overview."),
    )
    at = _with_dataset(_app()).run()
    at.chat_input[0].set_value("Which region sells more?").run()
    at.chat_input[0].set_value("And how big is this dataset overall?").run()

    assert not at.exception
    assert len(at.session_state["ai_messages"]) == 4
    second_plan_prompt = calls[2]["prompt"]
    assert "Which region sells more?" in second_plan_prompt
    assert "The south leads." in second_plan_prompt

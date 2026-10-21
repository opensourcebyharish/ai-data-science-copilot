"""Tests for the Gemini AI assistant (Milestone 5).

The whole suite runs offline: no API key, no network, no live Gemini
calls. The SDK is exercised through fake clients and injected
generators, and the analysis engine is checked against values computed
by hand with pandas.
"""

from __future__ import annotations

import ast
import pathlib

import pandas as pd
import pytest
from google.genai import errors as genai_errors

from core import assistant, codegen, gemini
from core.ai_schemas import (
    OPERATIONS,
    AnalysisPlan,
    AssistantAnswer,
    CodeExample,
)
from core.analysis import AnalysisError, AnalysisResult, compact_json, execute_plan
from core.charts import AGGREGATIONS, bucketed_series

ROOT = pathlib.Path(__file__).resolve().parents[1]


# --- Fixtures and fakes -------------------------------------------------------


@pytest.fixture
def sales_frame() -> pd.DataFrame:
    """Small frame with dates, groups, a missing group key and numbers."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "region": ["North", "South", "North", "South", None],
            "product": ["Laptop", "Phone", "Laptop", "Phone", "Laptop"],
            "sales": [100.0, 200.0, 300.0, 400.0, 500.0],
            "profit": [10.0, 50.0, 20.0, 40.0, 30.0],
        }
    )


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.output_text = text


class FakeInteractions:
    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(outcome)


class FakeClient:
    """Mimics google.genai.Client just enough for generate_structured()."""

    def __init__(self, *outcomes) -> None:
        self.interactions = FakeInteractions(list(outcomes))


def api_error(code: int, message: str = "boom", status: str = "UNKNOWN"):
    """A real google.genai.errors.APIError, as the legacy path raises it."""
    return genai_errors.APIError(
        code, {"error": {"code": code, "message": message, "status": status}}
    )


class FakeStatusError(Exception):
    """Next-gen SDK shape: an int status_code attribute, no .code."""

    def __init__(self, message: str = "nope", status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


# --- Schemas ------------------------------------------------------------------


def test_operations_cover_the_milestone_spec():
    assert set(OPERATIONS) == {
        "overview",
        "column_stats",
        "group_aggregate",
        "top_n",
        "correlation",
        "timeseries",
        "data_quality",
        "cannot_answer",
    }


def test_schema_aggregations_match_the_charts_constants():
    from typing import get_args
    from core.ai_schemas import Aggregation

    assert set(get_args(Aggregation)) == set(AGGREGATIONS)


def test_plan_defaults_to_minimal_valid_json():
    plan = AnalysisPlan(operation="overview")
    assert plan.operation == "overview"
    assert plan.limit is None and plan.value_column is None


def test_plan_rejects_unknown_operation_and_out_of_range_limit():
    with pytest.raises(Exception):
        AnalysisPlan(operation="drop_database")
    with pytest.raises(Exception):
        AnalysisPlan(operation="top_n", limit=51)
    with pytest.raises(Exception):
        AnalysisPlan(operation="top_n", limit=0)


# --- Configuration ------------------------------------------------------------


def test_configuration_is_absent_by_default(tmp_path):
    key, source = gemini.resolve_api_key(env={}, dotenv_path=tmp_path / ".env")
    assert key is None and source is None


def test_api_key_is_found_in_environment_first(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text('GEMINI_API_KEY="from-dotenv"\n', encoding="utf-8")
    key, source = gemini.resolve_api_key(
        secrets={"GEMINI_API_KEY": "from-secrets"},
        env={"GEMINI_API_KEY": "from-env"},
        dotenv_path=dotenv,
    )
    assert (key, source) == ("from-env", "environment variable")


def test_api_key_falls_back_to_dotenv_then_secrets(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text('GEMINI_API_KEY="  quoted-key  "\n', encoding="utf-8")

    key, source = gemini.resolve_api_key(secrets={}, env={}, dotenv_path=dotenv)
    assert (key, source) == ("quoted-key", ".env file")

    empty = tmp_path / "missing.env"
    key, source = gemini.resolve_api_key(
        secrets={"GEMINI_API_KEY": "secret-key"}, env={}, dotenv_path=empty
    )
    assert (key, source) == ("secret-key", "Streamlit secrets")


def test_blank_keys_are_treated_as_missing(tmp_path):
    key, source = gemini.resolve_api_key(
        secrets={"GEMINI_API_KEY": "   "}, env={"GEMINI_API_KEY": ""}, dotenv_path=tmp_path
    )
    assert key is None and source is None


def test_model_defaults_and_overrides(tmp_path):
    assert gemini.configured_model(env={}, dotenv_path=tmp_path) == gemini.DEFAULT_MODEL
    assert (
        gemini.configured_model(env={"GEMINI_MODEL": "my-model"}, dotenv_path=tmp_path)
        == "my-model"
    )
    assert (
        gemini.configured_model(
            secrets={"GEMINI_MODEL": "secret-model"},
            env={"GEMINI_MODEL": "env-model"},
            dotenv_path=tmp_path,
        )
        == "env-model"
    )


def test_api_config_never_exposes_the_key(tmp_path):
    config = gemini.api_config(env={"GEMINI_API_KEY": "super-secret-value"}, dotenv_path=tmp_path)
    assert config.configured is True
    assert config.source == "environment variable"
    assert "super-secret-value" not in repr(config)
    assert "super-secret-value" not in str(config)


# --- Structured generation (fake client, no network) --------------------------


def test_generate_structured_sends_schema_and_strict_settings():
    plan = AnalysisPlan(operation="data_quality")
    client = FakeClient(plan.model_dump_json())

    result = gemini.generate_structured(
        "question", AnalysisPlan, system="be a planner", client=client, model="test-model"
    )

    assert result.operation == "data_quality"
    call = client.interactions.calls[0]
    assert call["model"] == "test-model"
    assert call["system_instruction"] == "be a planner"
    assert call["store"] is False  # privacy: exchanges are not kept server-side
    assert call["response_format"]["mime_type"] == "application/json"
    assert call["response_format"]["schema"] == AnalysisPlan.model_json_schema()
    assert call["timeout"] == gemini.DEFAULT_TIMEOUT_SECONDS


def test_generate_structured_tolerates_json_fences():
    plan = AnalysisPlan(operation="overview")
    fenced = "```json\n" + plan.model_dump_json() + "\n```"
    client = FakeClient(fenced)
    assert gemini.generate_structured("q", AnalysisPlan, client=client).operation == "overview"


@pytest.mark.parametrize(
    "payload, expected",
    [
        ("not json at all", "schema"),
        ('{"operation": "top_n", "limit": 999}', "limit"),
        ('{"unexpected": true}', "operation"),
    ],
)
def test_generate_structured_rejects_bad_shapes(payload, expected):
    client = FakeClient(payload)
    with pytest.raises(gemini.GeminiResponseError) as excinfo:
        gemini.generate_structured("q", AnalysisPlan, client=client)
    assert expected in str(excinfo.value)


def test_generate_structured_rejects_empty_output():
    with pytest.raises(gemini.GeminiResponseError):
        gemini.generate_structured("q", AnalysisPlan, client=FakeClient("   "))


def test_missing_key_raises_a_configuration_error(tmp_path):
    with pytest.raises(gemini.GeminiNotConfiguredError):
        gemini.generate_structured(
            "q", AnalysisPlan, env={}, dotenv_path=tmp_path / "nope.env"
        )


# --- Error translation ---------------------------------------------------------


def test_rate_limit_and_quota_errors_map_to_a_rate_limit_error():
    mapped = gemini.translate_error(api_error(429, status="RESOURCE_EXHAUSTED"), model="m", timeout=5)
    assert isinstance(mapped, gemini.GeminiRateLimitError)
    mapped = gemini.translate_error(FakeStatusError(status_code=429), model="m", timeout=5)
    assert isinstance(mapped, gemini.GeminiRateLimitError)
    mapped = gemini.translate_error(api_error(403, status="RESOURCE_EXHAUSTED"), model="m", timeout=5)
    assert isinstance(mapped, gemini.GeminiRateLimitError)


def test_auth_bad_request_and_server_errors_map_by_status():
    assert isinstance(
        gemini.translate_error(api_error(401), model="m", timeout=5), gemini.GeminiAuthError
    )
    assert isinstance(
        gemini.translate_error(api_error(403), model="m", timeout=5), gemini.GeminiAuthError
    )
    bad_request = gemini.translate_error(api_error(400), model="m", timeout=5)
    assert isinstance(bad_request, gemini.GeminiBadRequestError)
    assert "m" in str(bad_request)
    missing = gemini.translate_error(api_error(404), model="gone-model", timeout=5)
    assert isinstance(missing, gemini.GeminiNotFoundError)
    assert "gone-model" in str(missing)
    assert isinstance(
        gemini.translate_error(api_error(503), model="m", timeout=5), gemini.GeminiUnavailableError
    )


def test_timeouts_and_connection_failures_map_by_shape():
    assert isinstance(
        gemini.translate_error(TimeoutError("too slow"), model="m", timeout=7),
        gemini.GeminiTimeoutError,
    )
    assert isinstance(
        gemini.translate_error(FakeStatusError("Read timed out"), model="m", timeout=7),
        gemini.GeminiTimeoutError,
    )
    assert isinstance(
        gemini.translate_error(ConnectionError("refused"), model="m", timeout=7),
        gemini.GeminiConnectionError,
    )
    connection_error = type("APIConnectionError", (Exception,), {})
    assert isinstance(
        gemini.translate_error(connection_error("no route"), model="m", timeout=7),
        gemini.GeminiConnectionError,
    )


def test_error_messages_never_contain_the_api_key():
    key = "AIza-SECRET-KEY-123"
    exc = api_error(400, message=f"bad request with key {key} attached")
    mapped = gemini.translate_error(exc, model="m", timeout=5, api_key=key)
    assert key not in str(mapped)
    assert key not in (mapped.detail or "")


def test_unmapped_errors_stay_generic_and_safe():
    mapped = gemini.translate_error(RuntimeError("weird"), model="m", timeout=5)
    assert isinstance(mapped, gemini.GeminiError)
    assert "RuntimeError" in str(mapped)


# --- Analysis engine -----------------------------------------------------------


def test_overview_reports_metrics_without_touching_the_frame(sales_frame):
    result = execute_plan(sales_frame, AnalysisPlan(operation="overview"))
    assert result.operation == "overview"
    labels = {kpi["label"]: kpi["value"] for kpi in result.kpis}
    assert labels["Rows"] == "5" and labels["Columns"] == "5"
    assert result.facts["numeric_columns"] == ["sales", "profit"]
    assert result.facts["categorical_columns"] == ["region", "product"]
    assert result.facts["datetime_columns"] == ["date"]


def test_data_quality_lists_missing_columns(sales_frame):
    result = execute_plan(sales_frame, AnalysisPlan(operation="data_quality"))
    details = " ".join(str(item) for item in result.facts["issues"])
    assert "region" in details or "Region" in details
    assert result.facts["missing_by_column"]["region"] > 0
    assert any("no values were imputed" in caveat for caveat in result.caveats)


def test_column_stats_numeric_matches_pandas(sales_frame):
    result = execute_plan(
        sales_frame, AnalysisPlan(operation="column_stats", value_column="sales")
    )
    expected = sales_frame["sales"].mean()
    assert result.facts["mean"] == pytest.approx(expected)
    assert result.facts["count"] == 5
    assert result.facts["missing"] == 0
    assert result.headline == "Statistics for sales"


def test_column_stats_categorical_reports_top_values(sales_frame):
    result = execute_plan(
        sales_frame, AnalysisPlan(operation="column_stats", value_column="product")
    )
    top = {item["value"]: item["count"] for item in result.facts["top_values"]}
    assert top == {"Laptop": 3, "Phone": 2}
    assert result.facts["distinct"] == 2


def test_column_stats_flags_small_samples_and_missing(sales_frame):
    frame = sales_frame.copy()
    frame.loc[0, "profit"] = None
    result = execute_plan(frame, AnalysisPlan(operation="column_stats", value_column="profit"))
    caveats = " ".join(result.caveats)
    assert "missing" in caveats.lower()
    assert "care" in caveats.lower()  # only 4 usable values < 30


def test_group_aggregate_matches_manual_pandas(sales_frame):
    result = execute_plan(
        sales_frame,
        AnalysisPlan(
            operation="group_aggregate",
            group_by="region",
            value_column="sales",
            aggregation="sum",
        ),
    )
    manual = sales_frame.groupby("region")["sales"].sum()
    assert result.facts["results"] == [
        {"group": "South", "value": 600.0},
        {"group": "North", "value": 400.0},
    ]
    assert result.facts["results"][0]["value"] == manual["South"]
    assert result.facts["groups_total"] == 2
    assert any("1 rows have no value in region" in caveat for caveat in result.caveats)
    assert result.headline == "Total of sales by region"


def test_group_aggregate_limit_adds_a_truncation_caveat(sales_frame):
    result = execute_plan(
        sales_frame,
        AnalysisPlan(
            operation="group_aggregate",
            group_by="product",
            value_column="sales",
            aggregation="mean",
            limit=1,
        ),
    )
    assert result.facts["groups_returned"] == 1
    assert result.facts["groups_total"] == 2
    assert any("of 2 groups" in caveat for caveat in result.caveats)


def test_group_aggregate_validates_every_field(sales_frame):
    base = dict(operation="group_aggregate", group_by="region", value_column="sales")
    with pytest.raises(AnalysisError, match="does not exist"):
        execute_plan(sales_frame, AnalysisPlan(**{**base, "group_by": "nope"}))
    with pytest.raises(AnalysisError, match="missing the aggregation"):
        execute_plan(sales_frame, AnalysisPlan(**base))
    with pytest.raises(AnalysisError, match="different columns"):
        execute_plan(
            sales_frame,
            AnalysisPlan(operation="group_aggregate", group_by="sales", value_column="sales", aggregation="sum"),
        )
    broken = AnalysisPlan.model_construct(
        operation="group_aggregate", group_by="region", value_column="sales", aggregation="median"
    )
    with pytest.raises(AnalysisError, match="Unknown aggregation"):
        execute_plan(sales_frame, broken)


def test_top_n_ranks_and_projects_columns(sales_frame):
    result = execute_plan(
        sales_frame,
        AnalysisPlan(operation="top_n", value_column="sales", limit=2, columns=["region", "sales"]),
    )
    assert list(result.table["sales"]) == [500.0, 400.0]
    assert list(result.table.columns) == ["region", "sales"]
    assert result.facts["rows"][0]["region"] is None  # missing stays missing, not invented


def test_top_n_lowest_first(sales_frame):
    result = execute_plan(
        sales_frame, AnalysisPlan(operation="top_n", value_column="sales", limit=1, order="asc")
    )
    assert list(result.table["sales"]) == [100.0]


def test_top_n_rejects_text_sort_columns(sales_frame):
    with pytest.raises(AnalysisError, match="numeric"):
        execute_plan(
            sales_frame, AnalysisPlan(operation="top_n", value_column="product")
        )


def test_correlation_reports_strongest_pairs(sales_frame):
    result = execute_plan(sales_frame, AnalysisPlan(operation="correlation"))
    manual = sales_frame["sales"].corr(sales_frame["profit"])
    assert result.facts["method"] == "pearson"
    assert result.facts["pairs"][0]["r"] == pytest.approx(round(float(manual), 3))
    assert any("not causation" in caveat for caveat in result.caveats)


def test_correlation_can_restrict_to_a_pair(sales_frame):
    result = execute_plan(
        sales_frame, AnalysisPlan(operation="correlation", columns=["sales", "profit"])
    )
    assert result.facts["columns_used"] == ["sales", "profit"]
    assert "sales vs profit" in result.headline


def test_correlation_rejects_text_columns(sales_frame):
    with pytest.raises(AnalysisError, match="does not exist"):
        execute_plan(
            sales_frame, AnalysisPlan(operation="correlation", columns=["sales", "nope"])
        )


def test_correlation_needs_two_varying_numeric_columns():
    frame = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [2.0, 3.0, 4.0]})
    with pytest.raises(AnalysisError):
        execute_plan(frame, AnalysisPlan(operation="correlation"))


def test_timeseries_buckets_like_the_charts_page(sales_frame):
    result = execute_plan(
        sales_frame,
        AnalysisPlan(
            operation="timeseries",
            date_column="date",
            value_column="sales",
            aggregation="mean",
            period="day",
        ),
    )
    assert result.facts["buckets"] == 3
    assert result.facts["series"][0] == {"bucket": "2024-01-01T00:00:00", "value": 150.0}
    assert result.facts["series"][-1]["value"] == 450.0


def test_timeseries_validates_columns_and_period(sales_frame):
    with pytest.raises(AnalysisError, match="does not exist"):
        execute_plan(
            sales_frame,
            AnalysisPlan(
                operation="timeseries", date_column="when", value_column="sales", aggregation="sum"
            ),
        )
    broken = AnalysisPlan.model_construct(
        operation="timeseries", date_column="date", value_column="sales", aggregation="sum", period="decade"
    )
    with pytest.raises(AnalysisError, match="Unknown time bucket"):
        execute_plan(sales_frame, broken)


def test_unparseable_dates_produce_a_clear_limitation():
    frame = pd.DataFrame({"when": ["a", "b", "c"], "value": [1.0, 2.0, 3.0]})
    with pytest.raises(AnalysisError):
        execute_plan(
            frame,
            AnalysisPlan(
                operation="timeseries", date_column="when", value_column="value", aggregation="sum"
            ),
        )


def test_empty_frames_are_handled_without_crashing():
    frame = pd.DataFrame({"a": pd.Series([], dtype=float)})
    overview = execute_plan(frame, AnalysisPlan(operation="overview"))
    assert overview.facts["rows"] == 0
    with pytest.raises(AnalysisError):
        execute_plan(
            frame,
            AnalysisPlan(operation="group_aggregate", group_by="a", value_column="a", aggregation="sum"),
        )


def test_execute_plan_refuses_impossible_and_unknown_operations(sales_frame):
    with pytest.raises(AnalysisError, match="not answerable"):
        execute_plan(sales_frame, AnalysisPlan(operation="cannot_answer"))
    with pytest.raises(AnalysisError, match="Unknown operation"):
        execute_plan(
            sales_frame, AnalysisPlan.model_construct(operation="drop_table")
        )


@pytest.mark.parametrize(
    "plan",
    [
        AnalysisPlan(operation="overview"),
        AnalysisPlan(operation="data_quality"),
        AnalysisPlan(operation="column_stats", value_column="sales"),
        AnalysisPlan(operation="column_stats", value_column="product"),
        AnalysisPlan(operation="group_aggregate", group_by="region", value_column="sales", aggregation="mean"),
        AnalysisPlan(operation="top_n", value_column="profit", limit=3),
        AnalysisPlan(operation="correlation"),
        AnalysisPlan(operation="timeseries", date_column="date", value_column="sales", aggregation="sum"),
    ],
)
def test_every_operation_leaves_the_frame_untouched(sales_frame, plan):
    original = sales_frame.copy(deep=True)
    execute_plan(sales_frame, plan)
    pd.testing.assert_frame_equal(sales_frame, original)
    assert list(sales_frame.columns) == list(original.columns)


def test_bucketed_series_is_shared_with_the_charts_page(sales_frame):
    series, label = bucketed_series(sales_frame, "date", "sales", "mean", "day")
    assert label == "daily"
    assert float(series.iloc[0]) == 150.0
    with pytest.raises(Exception):
        bucketed_series(sales_frame, "date", "sales", "median", "day")


# --- compact_json ---------------------------------------------------------------


def test_compact_json_trims_lists_and_caps_output():
    payload = {"rows": list(range(100)), "note": "x" * 400}
    text = compact_json(payload, max_chars=4000)
    assert "85 more" in text
    assert len(text) < 600
    huge = compact_json({"rows": ["y" * 100] * 100}, max_chars=200)
    assert len(huge) <= 200


# --- Orchestration (run_turn) ----------------------------------------------------


@pytest.fixture
def scripted(monkeypatch):
    """Replace the Gemini service with a scripted generator; return its calls."""

    def install(*responses) -> list[dict]:
        calls: list[dict] = []
        queue = list(responses)

        def fake(prompt, schema, **kwargs):
            calls.append({"prompt": prompt, "schema": schema, **kwargs})
            outcome = queue.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            assert isinstance(outcome, schema), f"expected {schema.__name__}"
            return outcome

        monkeypatch.setattr(gemini, "generate_structured", fake)
        return calls

    return install


def test_run_turn_plans_calculates_and_explains(sales_frame, scripted):
    calls = scripted(
        AnalysisPlan(
            operation="group_aggregate",
            group_by="region",
            value_column="sales",
            aggregation="mean",
            reasoning="compare regions",
        ),
        AssistantAnswer(summary="The south leads with a mean of 300."),
    )
    turn = assistant.run_turn("Which region sells more?", sales_frame)

    assert turn.result is not None
    assert turn.answer.summary.startswith("The south")
    assert [call["schema"] for call in calls] == [AnalysisPlan, AssistantAnswer]

    plan_prompt = calls[0]["prompt"]
    assert "Which region sells more?" in plan_prompt
    assert "region" in plan_prompt and "sales" in plan_prompt
    assert "mean" in plan_prompt  # summary statistics of numeric columns

    explain_prompt = calls[1]["prompt"]
    assert "300" in explain_prompt  # the verified mean travels with the facts
    assert "region" in explain_prompt


@pytest.mark.parametrize(
    "plan",
    [
        AnalysisPlan(operation="overview"),
        AnalysisPlan(operation="data_quality"),
        AnalysisPlan(operation="column_stats", value_column="sales"),
        AnalysisPlan(operation="column_stats", value_column="product"),
        AnalysisPlan(
            operation="group_aggregate",
            group_by="region",
            value_column="sales",
            aggregation="mean",
        ),
        AnalysisPlan(operation="top_n", value_column="sales", limit=3),
        AnalysisPlan(operation="correlation"),
        AnalysisPlan(
            operation="timeseries",
            date_column="date",
            value_column="sales",
            aggregation="sum",
        ),
    ],
)
def test_plan_prompt_never_sends_raw_cell_values(sales_frame, scripted, plan):
    """Every planning call — for every operation — stays metadata-only."""
    calls = scripted(plan, AssistantAnswer(summary="ok"))
    assistant.run_turn("Summarize this dataset", sales_frame)
    prompt = calls[0]["prompt"]
    for secret in ("Laptop", "Phone", "North", "South"):
        assert (
            secret not in prompt
        ), f"cell value {secret!r} leaked into the {plan.operation} plan prompt"


def test_explain_prompt_sends_ranked_rows_but_caps_them(scripted):
    """Honesty pin for PRIVACY_CAPTION: a top-N result does travel to
    the explainer, but compact_json hard-caps it at 15 entries — it is
    bounded, never a dump of the dataset."""
    frame = pd.DataFrame({"region": [f"R{i}" for i in range(40)], "sales": range(40)})
    calls = scripted(
        AnalysisPlan(operation="top_n", value_column="sales", limit=50),
        AssistantAnswer(summary="ok"),
    )
    assistant.run_turn("show me the top rows", frame)

    explain = calls[1]["prompt"]
    assert "R39" in explain  # the ranked rows the user asked for travel
    assert "25 more" in explain  # the rest are trimmed at 15 entries
    assert "R0" not in explain  # the tail of the dataset never travels
    assert len(explain) < 6000  # and the whole block stays bounded


def test_run_turn_includes_recent_history(sales_frame, scripted):
    history = [
        assistant.ChatMessage("user", "first question"),
        assistant.ChatMessage("assistant", "first answer"),
    ]
    calls = scripted(AnalysisPlan(operation="overview"), AssistantAnswer(summary="ok"))
    assistant.run_turn("and now what?", sales_frame, history)
    assert "first question" in calls[0]["prompt"]
    assert "first answer" in calls[0]["prompt"]


def test_run_turn_explains_the_limitation_for_invalid_plans(sales_frame, scripted):
    calls = scripted(
        AnalysisPlan(operation="column_stats", value_column="not_a_column")
    )
    turn = assistant.run_turn("statistics please", sales_frame)

    assert turn.result is None and turn.answer is None
    assert "does not exist" in turn.limitation
    assert "Supported operations" in turn.limitation
    assert len(calls) == 1  # no second call when nothing was calculated


def test_run_turn_uses_the_model_own_impossibility_reason(sales_frame, scripted):
    calls = scripted(
        AnalysisPlan(
            operation="cannot_answer",
            reason_if_impossible="Predicting next year needs future data this dataset does not have.",
        )
    )
    turn = assistant.run_turn("Predict next year's sales", sales_frame)
    assert "future data" in turn.text
    assert len(calls) == 1


def test_run_turn_propagates_gemini_errors(sales_frame, scripted):
    scripted(gemini.GeminiRateLimitError())
    with pytest.raises(gemini.GeminiRateLimitError):
        assistant.run_turn("anything", sales_frame)


def test_run_turn_rejects_empty_questions(sales_frame, scripted):
    with pytest.raises(ValueError):
        assistant.run_turn("   ", sales_frame)


# --- Dataset context and suggestions ---------------------------------------------


def test_dataset_context_describes_but_does_not_quote(sales_frame):
    context = assistant.dataset_context(sales_frame)
    assert "rows: 5" in context
    assert "- sales: number" in context
    assert "- product: text, 2 distinct values" in context
    assert "Laptop" not in context


def test_suggested_questions_are_dataset_aware_and_valid(sales_frame):
    questions = assistant.suggested_questions(sales_frame)
    assert len(questions) <= 6
    assert "Which region has the highest total sales?" in questions
    assert "Compare the average sales across region." in questions
    assert "Which columns contain missing values?" in questions
    joined = " ".join(questions)
    # Value and category columns are named; the date column drives a
    # time-aware question rather than being named literally.
    assert "region" in joined and "sales" in joined
    assert any("over time" in q for q in questions)
    assert all(q and isinstance(q, str) for q in questions)


def test_suggested_questions_work_for_a_numeric_only_frame():
    frame = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    questions = assistant.suggested_questions(frame)
    assert len(questions) >= 4
    assert all(isinstance(q, str) and q for q in questions)


def test_describe_plan_reads_naturally():
    plan = AnalysisPlan(
        operation="group_aggregate", group_by="region", value_column="sales", aggregation="sum"
    )
    assert assistant.describe_plan(plan) == "sum(sales) by region"
    assert (
        assistant.describe_plan(AnalysisPlan(operation="data_quality")) == "data-quality check"
    )


# --- Code generation --------------------------------------------------------------


def test_sql_table_names_come_from_the_filename():
    assert codegen.sql_table_name("sample_sales.csv") == "sample_sales"
    assert codegen.sql_table_name("Sales Data (final).csv") == "sales_data_final"
    assert codegen.sql_table_name("数据.csv") == "uploaded_data"


def test_blocked_constructs_are_detected_in_both_languages():
    assert codegen.find_blocked_constructs("import os\nos.remove('x')", "python") == ["import os"]
    assert codegen.find_blocked_constructs("df.groupby('a').mean()", "python") == []
    assert "drop" in codegen.find_blocked_constructs("DROP TABLE users;", "sql")
    assert codegen.find_blocked_constructs("SELECT name FROM sqlite_master", "sql") == []


def test_validate_flags_unknown_column_references(sales_frame):
    example = CodeExample(
        language="python",
        title="Grouped means",
        code="df.groupby('regn')['sales'].mean()",
        explanation="Mean sales per region.",
        referenced_columns=["regn", "sales"],
    )
    generated = codegen.validate_code_example(example, sales_frame, "sales.csv")
    assert len(generated.warnings) == 1
    assert "regn" in generated.warnings[0]


def test_validate_discards_dangerous_python(sales_frame):
    example = CodeExample(
        language="python",
        title="Bad", code="import os\nos.system('rm -rf /')", explanation="nope"
    )
    with pytest.raises(codegen.CodeExampleError, match="blocked constructs"):
        codegen.validate_code_example(example, sales_frame, "sales.csv")


def test_validate_requires_read_only_sql(sales_frame):
    example = CodeExample(language="sql", title="Bad", code="SHOW TABLES", explanation="nope")
    with pytest.raises(codegen.CodeExampleError, match="read-only"):
        codegen.validate_code_example(example, sales_frame, "sales.csv")


def test_validate_rejects_empty_code(sales_frame):
    example = CodeExample(language="python", title="Empty", code="   ", explanation="...")
    with pytest.raises(codegen.CodeExampleError, match="empty"):
        codegen.validate_code_example(example, sales_frame, "sales.csv")


def test_generated_code_is_returned_verbatim_and_never_executed(sales_frame):
    """A snippet that would crash if executed must survive as inert text."""
    sentinel = "raise RuntimeError('EXECUTED')\nresult = df['sales'].sum()"
    example = CodeExample(language="python", title="Sum", code=sentinel, explanation="Adds up sales.")
    generated = codegen.validate_code_example(example, sales_frame, "sales.csv")
    assert generated.example.code == sentinel  # untouched, unexecuted
    assert generated.warnings == ()


def test_generate_code_uses_real_columns_and_filename(sales_frame, monkeypatch):
    captured = {}

    def fake(prompt, schema, **kwargs):
        captured["prompt"] = prompt
        return CodeExample(
            language="sql",
            title="Average sales",
            code="SELECT region, AVG(sales) FROM sales_data GROUP BY region",
            explanation="Average sales per region.",
            referenced_columns=["region", "sales"],
        )

    monkeypatch.setattr(gemini, "generate_structured", fake)
    generated = codegen.generate_code("average sales by region", sales_frame, "Sales Data.csv")

    assert "region" in captured["prompt"] and "sales" in captured["prompt"]
    assert "sales_data" in captured["prompt"]
    assert generated.warnings == ()


def test_generate_code_requires_a_task(sales_frame, monkeypatch):
    with pytest.raises(codegen.CodeExampleError):
        codegen.generate_code("   ", sales_frame, "sales.csv")


# --- Security posture --------------------------------------------------------------


FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__"}
FORBIDDEN_ATTRIBUTES = {
    ("os", "system"),
    ("os", "popen"),
    ("subprocess", "run"),
    ("subprocess", "call"),
    ("subprocess", "Popen"),
    ("shutil", "rmtree"),
}


def test_no_module_can_execute_dynamic_code():
    """Static guarantee: nothing in the runtime path can exec/eval/import freely."""
    files = sorted((ROOT / "core").glob("*.py")) + [
        ROOT / "views" / "ai.py",
        ROOT / "app.py",
    ]
    assert files, "no source files found to scan"
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if isinstance(function, ast.Name):
                assert function.id not in FORBIDDEN_CALLS, f"{path.name} calls {function.id}()"
            if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name):
                pair = (function.value.id, function.attr)
                assert pair not in FORBIDDEN_ATTRIBUTES, f"{path.name} calls {pair[0]}.{pair[1]}()"


def test_views_never_import_subprocess_or_os_system_modules():
    tree = ast.parse((ROOT / "views" / "ai.py").read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "subprocess" not in imported
    assert "shutil" not in imported

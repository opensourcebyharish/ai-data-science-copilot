"""Orchestration for the AI Assistant (Milestone 5).

One question → one grounded turn:

1. Gemini receives the question plus dataset *metadata* (columns,
   types, counts, summary statistics — never raw rows) and returns a
   validated ``AnalysisPlan``.
2. core/analysis.py executes that plan deterministically with pandas.
   If the plan cannot be executed, a limitation is explained instead —
   nothing is invented.
3. Gemini explains the verified numbers it is handed, under strict
   instructions to use only those numbers.

Everything here is framework-free: Streamlit lives in views/ai.py and
the API transport in core/gemini.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import pandas as pd

from core import gemini
from core.ai_schemas import AssistantAnswer, AnalysisPlan, OPERATIONS
from core.analysis import AnalysisResult, execute_plan
from core.charts import (
    categorical_columns,
    datetime_columns,
    numeric_columns,
)

#: How much conversation context travels with a planning request.
HISTORY_TURNS = 6
HISTORY_CHARS = 400

PLAN_SYSTEM = """\
You are the planning brain of a data-analysis assistant. You are given a
user question and metadata about exactly one dataset: column names,
types, missing counts and summary statistics. You never see raw rows.

Your job is to design ONE analysis that answers the question, expressed
as a JSON object with the required schema. Rules:
- Only reference columns that exist, spelled exactly as given.
- Aggregations: mean, sum, min, max, count.
- limit: 1-50 rows or groups; default 10.
- If the question cannot be answered with the supported operations and
  the available columns, set operation to "cannot_answer" and explain
  the limitation honestly in reason_if_impossible — never guess a
  column or invent an analysis.
- You only design the analysis. You never calculate or estimate values.
"""

EXPLAIN_SYSTEM = """\
You are the explainer of a data-analysis assistant. You receive a
question and VERIFIED results that were computed in Python from the
user's actual data. Write a concise markdown answer using ONLY the
provided numbers: never invent, recompute or estimate a value, and say
plainly what was calculated.

Rules:
- Two to four short paragraphs or a short bullet list.
- State caveats when sample sizes, missing values or truncation affect
  interpretation (the app adds its own caveats separately).
- Correlation is association, not causation; never claim statistical
  significance or cause-and-effect.
- If the provided facts are not enough to answer, say what is missing
  instead of guessing.
"""


# --- Public data types --------------------------------------------------------


@dataclass(frozen=True)
class ChatMessage:
    """One stored turn. Only roles that exist in the UI: user/assistant."""

    role: str
    content: str


@dataclass(frozen=True)
class AssistantTurn:
    """The full result of one question: what ran, what it found, what it means."""

    question: str
    plan: AnalysisPlan | None = None
    result: AnalysisResult | None = None
    answer: AssistantAnswer | None = None
    limitation: str | None = None

    @property
    def text(self) -> str:
        """The assistant's words for this turn: the explanation or the limitation."""
        if self.answer is not None:
            return self.answer.summary
        return self.limitation or ""


# --- Dataset context (what Gemini is allowed to see) --------------------------


def dataset_context(frame: pd.DataFrame, max_columns: int = 80) -> str:
    """Metadata summary of the dataset — names, types, counts, aggregates.

    Deliberately excludes raw rows and individual cell values; the only
    numbers included are dataset-wide aggregates (min/max/mean) that
    help the model pick sensible columns.
    """
    lines = [
        f"rows: {len(frame):,}",
        f"columns: {len(frame.columns)}",
        "",
    ]
    visible = list(frame.columns)[:max_columns]
    for name in visible:
        series = frame[name]
        missing = int(series.isna().sum())
        missing_note = f", missing {missing:,} ({missing / len(frame) * 100:.1f}%)" if len(frame) and missing else ", missing 0"
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            described = f"number"
            stats = series.dropna()
            if not stats.empty:
                described += (
                    f", min {stats.min():.6g}, max {stats.max():.6g}, "
                    f"mean {stats.mean():.6g}"
                )
        elif pd.api.types.is_datetime64_any_dtype(series):
            described = "date"
        else:
            described = f"text, {series.nunique(dropna=True):,} distinct values"
        lines.append(f"- {name}: {described}{missing_note}")
    if len(frame.columns) > max_columns:
        lines.append(
            f"- … {len(frame.columns) - max_columns} more columns not described"
        )
    return "\n".join(lines)


def _history_text(history: Sequence[ChatMessage]) -> str:
    """Compact recent transcript for follow-up questions."""
    if not history:
        return "(no earlier conversation)"
    recent = list(history)[-HISTORY_TURNS:]
    lines = []
    for message in recent:
        content = " ".join(str(message.content).split())[:HISTORY_CHARS]
        speaker = "User" if message.role == "user" else "Assistant"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def build_plan_prompt(question: str, frame: pd.DataFrame, history: Sequence[ChatMessage]) -> str:
    """The planning request: transcript, question and dataset metadata."""
    return (
        f"Dataset metadata:\n{dataset_context(frame)}\n\n"
        f"Conversation so far:\n{_history_text(history)}\n\n"
        f"New question: {question}\n\n"
        f"Choose exactly one operation that answers the question using "
        f"only the columns listed above, and reply as JSON."
    )


def build_explain_prompt(
    question: str,
    plan: AnalysisPlan,
    result: AnalysisResult,
    history: Sequence[ChatMessage],
) -> str:
    """The explanation request: verified numbers plus their caveats."""
    from core.analysis import compact_json  # local import keeps the dependency one-way

    return (
        f"Conversation so far:\n{_history_text(history)}\n\n"
        f"Question: {question}\n\n"
        f"The app ran this analysis: {describe_plan(plan)}\n\n"
        f"Verified result (computed with pandas, JSON):\n"
        f"{compact_json(result.facts)}\n\n"
        f"App-calculated caveats: {compact_json(list(result.caveats))}\n\n"
        f"Explain the result now, using only these numbers."
    )


def describe_plan(plan: AnalysisPlan) -> str:
    """Human label for an executed (or attempted) plan."""
    operation = plan.operation
    if operation == "cannot_answer":
        return "not answerable with the supported operations"
    if operation == "overview":
        return "dataset overview"
    if operation == "data_quality":
        return "data-quality check"
    if operation == "column_stats":
        return f"statistics for {plan.value_column}"
    if operation == "group_aggregate":
        label = f"{plan.aggregation}({plan.value_column}) by {plan.group_by}"
        return label
    if operation == "top_n":
        order = "lowest first" if plan.order == "asc" else "highest first"
        return f"top {plan.limit or 10} rows by {plan.value_column} ({order})"
    if operation == "correlation":
        if plan.columns and len(plan.columns) == 2:
            return f"{plan.method or 'pearson'} correlation: {plan.columns[0]} vs {plan.columns[1]}"
        return f"{plan.method or 'pearson'} correlation across numeric columns"
    if operation == "timeseries":
        return (
            f"{plan.aggregation} of {plan.value_column} over "
            f"{plan.date_column}, bucketed {plan.period or 'auto'}"
        )
    return operation


# --- Suggested questions ------------------------------------------------------


def suggested_questions(frame: pd.DataFrame, max_questions: int = 6) -> list[str]:
    """Dataset-aware starter questions, most specific first."""
    numbers = numeric_columns(frame)
    dates = set(datetime_columns(frame))
    categories = [
        name
        for name in categorical_columns(frame)
        if name not in dates and frame[name].nunique(dropna=True) > 1
    ]

    questions: list[str] = []
    if categories and numbers:
        questions.append(f"Which {categories[0]} has the highest total {numbers[0]}?")
        questions.append(f"Compare the average {numbers[0]} across {categories[0]}.")
    if len(numbers) >= 2:
        questions.append("What relationships exist between the numeric columns?")
    if dates and numbers:
        questions.append(f"How does {numbers[0]} change over time?")
    if numbers and not categories:
        questions.append(f"What are the key statistics for {numbers[0]}?")

    questions.extend(
        [
            "What are the most important characteristics of this dataset?",
            "Which columns contain missing values?",
            "Summarize the main data-quality issues.",
        ]
    )

    unique: list[str] = []
    for question in questions:
        if question not in unique:
            unique.append(question)
    return unique[:max_questions]


# --- The turn -----------------------------------------------------------------


def _cannot_answer_turn(question: str, plan: AnalysisPlan) -> AssistantTurn:
    reason = (plan.reason_if_impossible or "").strip()
    if not reason:
        reason = (
            "The question cannot be answered with the supported operations "
            "on this dataset."
        )
    return AssistantTurn(
        question=question,
        plan=plan,
        answer=AssistantAnswer(summary=reason),
    )


def run_turn(
    question: str,
    frame: pd.DataFrame,
    history: Sequence[ChatMessage] = (),
    *,
    generate: Any = None,
) -> AssistantTurn:
    """Execute one full question → plan → calculate → explain cycle.

    ``generate`` may be injected for tests; by default the Gemini
    service is used (module-level lookup so tests can patch it).

    Gemini failures (auth, quota, timeouts, bad formats) propagate as
    ``gemini.GeminiError`` with a message safe for the UI. Plan or data
    problems never raise: they become a ``limitation`` turn instead.
    """
    question = str(question).strip()
    if not question:
        raise ValueError("question must not be empty")

    generate = generate if generate is not None else gemini.generate_structured

    plan = generate(
        build_plan_prompt(question, frame, history),
        AnalysisPlan,
        system=PLAN_SYSTEM,
    )

    if plan.operation == "cannot_answer":
        return _cannot_answer_turn(question, plan)

    try:
        result = execute_plan(frame, plan)
    except Exception as exc:  # AnalysisError and any defensive failure
        return AssistantTurn(
            question=question,
            plan=plan,
            limitation=(
                "I could not run that analysis: "
                f"{exc} Supported operations: overview, column stats, "
                "group-by aggregation, top-N, correlation, time-series "
                "aggregation and data-quality checks. You can also see "
                "the exact column list on the Data page."
            ),
        )

    answer = generate(
        build_explain_prompt(question, plan, result, history),
        AssistantAnswer,
        system=EXPLAIN_SYSTEM,
    )

    return AssistantTurn(question=question, plan=plan, result=result, answer=answer)

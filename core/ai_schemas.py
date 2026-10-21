"""Pydantic schemas for everything Gemini is asked to produce (Milestone 5).

Gemini never returns free-form data the app has to guess about: every
response is validated against one of these models before anything is
shown or calculated. The same classes serve two purposes:

* their JSON Schema is sent to Gemini (structured output), and
* their Python instances are the contract the deterministic code in
  core/analysis.py and core/codegen.py relies on.

The schemas describe *shape only* — no dataset values, no network, no
Streamlit. Field descriptions matter: they are sent to the model and
are the main lever for answer quality.
"""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, Field

#: Every operation the assistant may request. ``execute_plan`` in
#: core/analysis.py refuses anything outside this tuple, so the set of
#: supported analyses exists exactly once, here and there.
Operation = Literal[
    "overview",
    "column_stats",
    "group_aggregate",
    "top_n",
    "correlation",
    "timeseries",
    "data_quality",
    "cannot_answer",
]

#: Plain-text tuple version, used in prompts, captions and error messages.
OPERATIONS: tuple[str, ...] = get_args(Operation)

#: How values combine. Kept identical to core.charts.AGGREGATIONS; a
#: test asserts the two never drift apart.
Aggregation = Literal["mean", "sum", "min", "max", "count"]

#: Sort direction for ranked results (default: highest first).
Order = Literal["desc", "asc"]

CorrelationMethod = Literal["pearson", "spearman"]

#: Time bucket for time-series aggregation ("auto" picks from the span,
#: exactly like the Charts page does).
TimePeriod = Literal["auto", "hour", "day", "week", "month"]


class AnalysisPlan(BaseModel):
    """One validated analysis request: what to calculate, not the answer.

    Produced by Gemini from the question plus dataset *metadata*, then
    checked field-by-field against the real columns by
    core.analysis.execute_plan before any calculation runs.
    """

    operation: Operation = Field(
        description=(
            "Exactly one operation from the allowed set that can answer the "
            "question using only the listed columns. Use 'cannot_answer' when "
            "no supported operation can."
        )
    )
    reasoning: str = Field(
        default="",
        description="One short sentence: why this operation answers the question.",
    )
    value_column: str | None = Field(
        default=None,
        description=(
            "The column being measured: the target of column_stats, the value "
            "aggregated by group_aggregate, the sort key of top_n, or the value "
            "of timeseries. Must copy an existing column name exactly."
        ),
    )
    group_by: str | None = Field(
        default=None,
        description="The column to group by for group_aggregate. Exact existing column name.",
    )
    date_column: str | None = Field(
        default=None,
        description="The date column for timeseries. Exact existing column name.",
    )
    aggregation: Aggregation | None = Field(
        default=None,
        description="How values combine. Required for group_aggregate and timeseries.",
    )
    method: CorrelationMethod | None = Field(
        default=None,
        description="Correlation method for the correlation operation. Defaults to pearson.",
    )
    period: TimePeriod | None = Field(
        default=None,
        description="Time bucket for timeseries. 'auto' picks a size from the date range.",
    )
    order: Order | None = Field(
        default=None,
        description=(
            "Sort direction for top_n and group_aggregate results. "
            "Defaults to desc (highest first)."
        ),
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="How many rows or groups to return, 1-50. Defaults to 10.",
    )
    columns: list[str] | None = Field(
        default=None,
        description=(
            "Only for correlation: restrict to exactly two named numeric "
            "columns. Omit to use every numeric column."
        ),
    )
    reason_if_impossible: str | None = Field(
        default=None,
        description=(
            "Required when operation is 'cannot_answer': a short, honest "
            "explanation of what cannot be answered and why."
        ),
    )


class AssistantAnswer(BaseModel):
    """The natural-language explanation written *after* the numbers exist."""

    summary: str = Field(
        description=(
            "Concise markdown answer to the question, built only from the "
            "provided verified facts. Two to four short paragraphs or a "
            "bulleted list; never invent, recompute or estimate a number."
        )
    )
    caveats: list[str] = Field(
        default_factory=list,
        description=(
            "Short limitations worth showing next to the answer, for example "
            "small sample sizes, missing values or truncated results. "
            "Omit (empty list) when nothing qualifies."
        ),
    )


class CodeExample(BaseModel):
    """A read-only Python or SQL example. Displayed, never executed."""

    language: Literal["python", "sql"] = Field(
        description="python for pandas snippets, sql for SELECT queries."
    )
    title: str = Field(description="Short human title for the snippet, e.g. 'Average sales by region'.")
    code: str = Field(description="The complete code example, well formatted, under ~40 lines.")
    explanation: str = Field(
        description="Two or three sentences explaining what the code does, in plain language."
    )
    referenced_columns: list[str] = Field(
        default_factory=list,
        description=(
            "Every dataset column the code mentions, each copied exactly "
            "from the provided column list."
        ),
    )

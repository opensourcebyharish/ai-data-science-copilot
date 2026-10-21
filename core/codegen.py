"""Dataset-aware code examples, generated but never executed (Milestone 5).

Gemini drafts Python (pandas) or SQL examples from the user's task and
the dataset's real columns. Two hard rules:

* **Nothing is executed.** No eval, no exec, no subprocess — the code is
  inert text that the UI displays and offers for download.
* **Defense in depth.** Even as inert text, snippets containing
  system-touching constructs (``import os``, ``open(``, non-SELECT SQL)
  are discarded before they can be shown, so a hostile or confused
  model output never reaches the screen as "suggested code".

Column references the model claims are validated against the real
frame; unknown ones are reported as warnings next to the snippet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from core import gemini
from core.ai_schemas import CodeExample

#: Constructs a read-only example must never contain. Matched
#: case-insensitively against the generated code.
BLOCKED_PYTHON: tuple[str, ...] = (
    "import os",
    "import sys",
    "import subprocess",
    "import shutil",
    "import pathlib",
    "from os",
    "eval(",
    "exec(",
    "compile(",
    "__import__",
    "os.system",
    "os.popen",
    "os.environ",
    "getenv(",
    "open(",
    "requests.",
    "urllib",
    "socket",
)

#: SQL examples are SELECT-only; anything that could write or attach is
#: rejected outright (the app also never runs them).
BLOCKED_SQL: tuple[str, ...] = (
    "attach",
    "copy ",
    "pragma",
    "drop ",
    "delete",
    "insert ",
    "update ",
    "create ",
    "alter ",
    "grant ",
)

#: Table alias used in SQL examples; derived from the uploaded filename.
SQL_WORD = re.compile(r"[^0-9a-zA-Z_]+")

CODE_SYSTEM = """\
You write short, readable example code for a data analyst. Rules:
- python: pandas only. Assume the loaded DataFrame is already named
  `df`. Use clear variable names and no more than ~40 lines.
- sql: a single read-only SELECT (or a WITH … SELECT) over the table
  named in the prompt. ANSI SQL. No database connection exists — this
  is an example only.
- Reference ONLY the dataset columns listed in the prompt, copied
  exactly.
- Never use file I/O, network access, shell commands, imports outside
  pandas, or dynamic execution.
- referenced_columns must list every dataset column the code touches.
- Keep the explanation to two or three plain-language sentences.
"""


class CodeExampleError(ValueError):
    """A generation problem the UI can show as a friendly message."""


@dataclass(frozen=True)
class GeneratedCode:
    """A validated example plus any warnings about its column references."""

    example: CodeExample
    warnings: tuple[str, ...] = ()


def sql_table_name(filename: str) -> str:
    """``Sales Data (final).csv`` → ``sales_data_final`` — never guessed."""
    stem = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = stem.rsplit(".", 1)[0]
    cleaned = SQL_WORD.sub("_", stem).strip("_").lower()
    return cleaned or "uploaded_data"


def build_codegen_prompt(
    task: str,
    frame: pd.DataFrame,
    filename: str,
) -> str:
    """User task plus the exact columns and dtypes the code must use."""
    columns = ", ".join(f"{name} ({frame[name].dtype})" for name in frame.columns)
    return (
        f"Task: {task}\n\n"
        f"Dataset: {len(frame):,} rows, {len(frame.columns)} columns, file name "
        f"'{filename}'.\n"
        f"Columns: {columns}.\n"
        f"SQL table name (if SQL): {sql_table_name(filename)}.\n\n"
        f"Write the example now, as JSON matching the schema."
    )


def find_blocked_constructs(code: str, language: str) -> list[str]:
    """The blocked patterns present in ``code`` (empty when clean)."""
    lowered = code.lower()
    if language == "sql":
        # Only flag SQL keywords that start a statement or clause.
        hits = [
            pattern.strip()
            for pattern in BLOCKED_SQL
            if re.search(rf"(^|;|\n)\s*{pattern.strip()}", lowered)
        ]
        return hits
    return [pattern for pattern in BLOCKED_PYTHON if pattern in lowered]


def validate_code_example(
    example: CodeExample,
    frame: pd.DataFrame,
    filename: str,
) -> GeneratedCode:
    """Check a model-generated example; discard anything unsafe.

    Raises ``CodeExampleError`` when the code is empty, uses blocked
    constructs, or (for SQL) is not a read-only SELECT. Unknown column
    references become warnings rather than failures — the snippet is
    still useful, but clearly marked.
    """
    code = example.code or ""
    if not code.strip():
        raise CodeExampleError("Gemini returned an empty code example — try again.")

    language = example.language
    if language not in ("python", "sql"):
        raise CodeExampleError(
            f'Unsupported code language "{language}" — python or sql only.'
        )

    blocked = find_blocked_constructs(code, language)
    if blocked:
        raise CodeExampleError(
            "The generated code contained blocked constructs "
            f"({', '.join(blocked)}) and was discarded. Generated code is "
            "display-only, and this app never accepts code that touches the "
            "system. Try rephrasing the request."
        )

    if language == "sql" and not re.match(r"^\s*(select|with)\b", code.lower()):
        raise CodeExampleError(
            "SQL examples must be read-only SELECT queries; the generated "
            "code was not one and was discarded."
        )

    warnings: list[str] = []
    referenced = [str(name) for name in example.referenced_columns]
    unknown = [name for name in referenced if name not in frame.columns]
    if unknown:
        warnings.append(
            f"The code references column(s) not in this dataset: "
            f"{', '.join(unknown)}. Column names may be misspelled."
        )

    if language == "sql":
        table = sql_table_name(filename)
        if table not in code.lower():
            warnings.append(
                f"The query does not mention the dataset's table name "
                f"'{table}'."
            )

    return GeneratedCode(example=example, warnings=tuple(warnings))


def generate_code(
    task: str,
    frame: pd.DataFrame,
    filename: str,
    *,
    client: Any = None,
) -> GeneratedCode:
    """One full cycle: task in, validated example out.

    ``client`` may be injected for tests; Gemini failures propagate as
    ``gemini.GeminiError``.
    """
    task = str(task).strip()
    if not task:
        raise CodeExampleError("Describe what the code should do first.")

    def generate(prompt: str, schema: Any, *, system: str | None = None) -> Any:
        if client is None:
            return gemini.generate_structured(prompt, schema, system=system)
        return gemini.generate_structured(prompt, schema, system=system, client=client)

    example = generate(
        build_codegen_prompt(task, frame, filename),
        CodeExample,
        system=CODE_SYSTEM,
    )
    return validate_code_example(example, frame, filename)

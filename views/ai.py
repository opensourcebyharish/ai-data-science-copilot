"""AI Assistant page — grounded answers from Gemini (Milestone 5).

The conversation UI only; every decision it depends on lives in
core/: assistant.py orchestrates plan → calculate → explain, gemini.py
owns the API and its errors, codegen.py drafts display-only code
examples. This module renders, collects the user's question and turns
turns into session-persisted messages.

The dataset comes from the Data page through session state
(``loaded_dataset``) and is never modified here. Nothing the model
returns is executed — only shown.
"""

from __future__ import annotations

import streamlit as st

from core import assistant, codegen, gemini
from core.loader import LoadedDataset
from ui.components import (
    card_title,
    render_empty_state,
    render_kpis,
    render_page_header,
    section_card,
    section_heading,
)

#: Session-state keys, named to stay out of the way of other pages.
MESSAGES_KEY = "ai_messages"
CODE_KEY = "ai_code_example"

#: Caption directly under the configuration status: the honest summary
#: of what the external API ever receives.
PRIVACY_CAPTION = (
    "Sends to Gemini: column names, types, quality counts, summary "
    "statistics, your last few messages, aggregate results and — for a "
    "top-N request — the ranked rows, hard-capped at 15 entries and "
    "4,000 characters. The conversation lives only in this browser "
    "session."
)


def _streamlit_secrets() -> dict:
    """Read our two settings from .streamlit/secrets.toml, if present.

    Core stays framework-free, so the view passes any Streamlit secrets
    down as a plain mapping. A missing secrets file is normal and never
    an error.
    """
    try:
        return {
            key: st.secrets[key]
            for key in (gemini.API_KEY_SETTING, gemini.MODEL_SETTING)
            if key in st.secrets
        }
    except Exception:  # no secrets file, or secrets unavailable (tests)
        return {}


def _render_config_status(config: gemini.ApiConfig) -> None:
    """One honest line about the API configuration — never the key itself."""
    if config.configured:
        st.success(
            f"Gemini configured — model **{config.model}**, key read from "
            f"{config.source}. The key itself is never displayed or logged."
        )
        return
    st.warning(
        "Gemini is not configured, so questions and code generation are "
        "disabled. The rest of the app works without a key."
    )
    with st.expander("How to configure the Gemini API key"):
        st.markdown(
            f"""
1. Create a key in [Google AI Studio](https://aistudio.google.com/apikey).
2. Either drop it into a `.env` file in the project root:

   ```toml
   GEMINI_API_KEY=your-key-here
   # optional — override the default model
   GEMINI_MODEL={gemini.DEFAULT_MODEL}
   ```

   or add it to `.streamlit/secrets.toml`:

   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```

3. Restart the Streamlit server. The status above turns green; the key
   stays out of the source code, the UI and the git history either way.
"""
        )


def _message_from_turn(turn: assistant.AssistantTurn) -> dict:
    """Flatten a turn into a storable message dict (DataFrames and all)."""
    message: dict = {"role": "assistant", "content": turn.text}
    if turn.plan is not None:
        message["plan_label"] = assistant.describe_plan(turn.plan)
    if turn.result is not None:
        message["headline"] = turn.result.headline
        message["table"] = turn.result.table
        message["kpis"] = list(turn.result.kpis)
        message["caveats"] = list(turn.result.caveats)
    if turn.answer is not None:
        message["ai_caveats"] = list(turn.answer.caveats)
    if turn.limitation is not None:
        message["limitation"] = True
    return message


def _render_assistant_message(message: dict) -> None:
    """One assistant message: calculated block first, explanation second."""
    if message.get("error"):
        st.error(message["content"])
        if message.get("detail"):
            st.caption(f"Technical detail: {message['detail']}")
        return

    if message.get("limitation"):
        st.warning(message["content"])
        return

    plan_label = message.get("plan_label")
    if plan_label:
        st.caption(f"analysis · {plan_label}")

    headline = message.get("headline")
    table = message.get("table")
    kpis = message.get("kpis")
    if table is not None or kpis:
        st.badge("Calculated by the app", color="green")
        if headline:
            st.markdown(f"**{headline}**")
        if table is not None:
            st.dataframe(table, width="stretch", hide_index=True)
        if kpis:
            render_kpis(kpis)
        st.caption(
            "Computed with pandas from your data — Gemini only explains "
            "these numbers, it does not produce them."
        )

    st.markdown(message["content"])
    for caveat in message.get("caveats", []):
        st.caption(f"• {caveat}")
    for caveat in message.get("ai_caveats", []):
        st.caption(f"• {caveat}")


def _render_code_example(example: dict, filename: str) -> None:
    """A stored generated snippet: badge, code, explanation, download."""
    language = example["language"]
    st.markdown(f"**{example['title']}**")
    st.badge("Generated — display only, never executed", color="orange")
    st.code(example["code"], language=language)
    st.markdown(example["explanation"])
    for warning in example.get("warnings", []):
        st.warning(warning)
    st.download_button(
        "Download example",
        data=example["code"],
        file_name=f"{codegen.sql_table_name(filename)}"
        f"{'.sql' if language == 'sql' else '.py'}",
        mime="text/plain",
        icon=":material/download:",
        key="ai_code_download",
    )
    if language == "sql":
        st.caption(
            "Generated SQL only — no database is connected and nothing was "
            "executed. The app is read-only."
        )
    else:
        st.caption(
            "Python example using the `df` variable — the app never "
            "executes generated code."
        )


def _render_suggestions(frame, config: gemini.ApiConfig) -> str | None:
    """Dataset-aware starter questions; returns one if a chip was clicked."""
    clicked: str | None = None
    questions = assistant.suggested_questions(frame)
    st.caption("Try asking one of these dataset-aware starters:")
    for start in range(0, len(questions), 3):
        columns = st.columns(3)
        for offset, column in enumerate(columns):
            index = start + offset
            if index >= len(questions):
                break
            with column:
                if st.button(
                    questions[index],
                    key=f"ai_suggest_{index}",
                    width="stretch",
                    disabled=not config.configured,
                    help=None if config.configured else "Configure a Gemini API key first.",
                ):
                    clicked = questions[index]
    return clicked


def render() -> None:
    """The AI Assistant page."""
    render_page_header(
        "AI Assistant",
        "Ask questions in plain language. Gemini designs the analysis, the "
        "app calculates it with pandas, and Gemini explains the verified "
        "numbers — nothing you type is ever executed.",
        chip="Beta",
        chip_tone="progress",
    )

    dataset: LoadedDataset | None = st.session_state.get("loaded_dataset")
    if dataset is None:
        render_empty_state(
            "No dataset loaded",
            "The assistant answers questions about the dataset loaded on "
            "the Data page. Upload a file there first.",
            hint="Grounded answers: the app calculates every number; Gemini only explains them.",
            icon="✨",
        )
        st.link_button(
            "Go to the Data page",
            url="/data",
            icon=":material/database:",
            type="primary",
        )
        st.stop()

    frame = dataset.dataframe
    config = gemini.api_config(secrets=_streamlit_secrets())

    _render_config_status(config)
    st.caption(f"{dataset.filename} · {len(frame):,} rows · {PRIVACY_CAPTION}")

    messages: list[dict] = st.session_state.setdefault(MESSAGES_KEY, [])
    with section_card():
        card_title("Conversation", hint="Last few messages stay in this browser session only.")
        clear = st.button(
            "Clear conversation",
            icon=":material/delete_sweep:",
            disabled=not messages,
            help="Forget this conversation. Nothing was stored server-side.",
            key="ai_clear",
        )
    if clear:
        st.session_state.pop(MESSAGES_KEY, None)
        st.session_state.pop(CODE_KEY, None)
        st.rerun()

    for message in messages:
        with st.chat_message(message["role"]):
            _render_assistant_message(message)

    # --- Code examples (before the input so the chat stays at the bottom).
    section_heading("Code examples")
    with section_card():
        card_title(
            "Generate an example",
            hint="Display-only snippets: the app never executes generated code.",
        )
        with st.form("ai_codegen_form"):
            task = st.text_input(
                "What should the code do?",
                placeholder="e.g. average profit by region",
                key="ai_codegen_task",
                disabled=not config.configured,
            )
            generate_clicked = st.form_submit_button(
                "Generate example",
                icon=":material/code:",
                disabled=not config.configured,
            )
    if generate_clicked:
        if not (task or "").strip():
            st.warning("Describe what the code should do, then generate.")
        else:
            try:
                with st.spinner("Gemini is drafting the example…"):
                    generated = codegen.generate_code(task, frame, dataset.filename)
                st.session_state[CODE_KEY] = {
                    "language": generated.example.language,
                    "title": generated.example.title,
                    "code": generated.example.code,
                    "explanation": generated.example.explanation,
                    "warnings": list(generated.warnings),
                }
            except codegen.CodeExampleError as exc:
                st.warning(str(exc))
            except gemini.GeminiError as exc:
                st.error(str(exc))
    stored_example = st.session_state.get(CODE_KEY)
    if stored_example:
        _render_code_example(stored_example, dataset.filename)
    elif not config.configured:
        st.caption(
            "Configure a Gemini API key (steps above) to generate "
            "dataset-aware Python and SQL examples."
        )

    # --- Suggested questions, only when there is a working assistant.
    pending: str | None = None
    if not messages and config.configured:
        pending = _render_suggestions(frame, config)

    # --- The question box.
    user_input = st.chat_input(
        "Ask about your dataset…" if config.configured else
        "Set GEMINI_API_KEY to ask questions about your dataset…",
        key="ai_prompt",
        disabled=not config.configured,
    )
    prompt = user_input or pending

    if prompt:
        history = [
            assistant.ChatMessage(role=message["role"], content=message["content"])
            for message in messages
        ]
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Gemini is planning the analysis…"):
                    turn = assistant.run_turn(prompt, frame, history)
                message = _message_from_turn(turn)
                messages.append(message)
                _render_assistant_message(message)
            except gemini.GeminiError as exc:
                message = {
                    "role": "assistant",
                    "content": str(exc),
                    "error": True,
                    "detail": getattr(exc, "detail", None),
                }
                messages.append(message)
                _render_assistant_message(message)
        st.rerun()


render()

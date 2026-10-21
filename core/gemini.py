"""Google Gemini service layer (Milestone 5).

Everything that talks to the Gemini API lives in this module:
configuration resolution, the lazily-created client, structured-output
generation and the translation of SDK failures into short messages that
are safe to show on screen.

Design rules, in order of importance:

* **No API call at startup.** The client is built on the first
  ``generate_structured()`` call, never at import time.
* **The key never leaks.** It is not part of any return value, never
  logged, and scrubbed out of error text before an error is shown.
* **Strict schemas.** Every response is parsed with Pydantic; anything
  that does not fit the requested schema becomes a ``GeminiResponseError``
  instead of being half-trusted.
* **Stateless conversations.** Requests use ``store=False`` so Google
  does not keep the exchange; our own history stays in the Streamlit
  session only.

Prompts themselves live in core/assistant.py and core/codegen.py — this
module only moves validated JSON in and out.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from dotenv import dotenv_values
from google import genai
from pydantic import BaseModel, ValidationError

#: Where the project lives; the default location of the local .env file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOTENV = PROJECT_ROOT / ".env"

#: Settings read from the environment, .env or Streamlit secrets.
API_KEY_SETTING = "GEMINI_API_KEY"
MODEL_SETTING = "GEMINI_MODEL"

#: Default model — the model used throughout the official Gemini API
#: documentation (https://ai.google.dev/gemini-api/docs/get-started,
#: checked October 2026). Override with GEMINI_MODEL when Google
#: publishes a successor; the app validates nothing about the name, an
#: unknown model simply surfaces as a clear API error.
DEFAULT_MODEL = "gemini-3.8-flash"

#: Network budget per request. Planning is fast; explanations with large
#: fact blocks are the slow case, so 45s is generous but bounded.
DEFAULT_TIMEOUT_SECONDS = 45.0

#: Requests are near-deterministic: the app wants an interpretation of
#: the question, not creative variation between reruns.
GENERATION_CONFIG: dict[str, Any] = {"temperature": 0.2}

T = TypeVar("T", bound=BaseModel)


class GeminiError(Exception):
    """Base class for Gemini problems.

    ``str(error)`` is always safe to render in the UI: it never contains
    the API key. ``error.detail`` optionally carries a scrubbed,
    truncated technical detail for a caption.
    """

    default_message = "The Gemini request failed."

    def __init__(self, message: str | None = None, detail: str | None = None) -> None:
        super().__init__(message or self.default_message)
        self.detail = detail or None


class GeminiNotConfiguredError(GeminiError):
    default_message = (
        "Gemini is not configured. Set GEMINI_API_KEY in a .env file, in the "
        "environment, or in .streamlit/secrets.toml — the setup steps are "
        "shown above."
    )


class GeminiAuthError(GeminiError):
    default_message = (
        "Google rejected the API key. Check that GEMINI_API_KEY is correct, "
        "active and not expired."
    )


class GeminiRateLimitError(GeminiError):
    default_message = (
        "Gemini's rate limit or quota was exhausted. Wait a moment and try "
        "again, or review your quota in Google AI Studio."
    )


class GeminiBadRequestError(GeminiError):
    default_message = (
        "Gemini rejected the request. If this keeps happening, check that "
        "GEMINI_MODEL names a valid model."
    )


class GeminiNotFoundError(GeminiError):
    default_message = (
        "The configured Gemini model was not found. Check that GEMINI_MODEL "
        "names a model your API key may use."
    )


class GeminiTimeoutError(GeminiError):
    default_message = "Gemini did not respond in time. Try the question again."


class GeminiConnectionError(GeminiError):
    default_message = (
        "Could not reach the Gemini API. Check the internet connection and "
        "try again."
    )


class GeminiUnavailableError(GeminiError):
    default_message = (
        "Google's Gemini API is temporarily unavailable. Try again shortly."
    )


class GeminiResponseError(GeminiError):
    default_message = (
        "Gemini replied in an unexpected format. Try rephrasing the question."
    )


# --- Configuration ----------------------------------------------------------


@dataclass(frozen=True)
class ApiConfig:
    """What the UI is allowed to know about the configuration.

    Deliberately has **no key field**: this object is rendered on screen
    and may end up in test output.
    """

    configured: bool
    source: str | None  # "environment variable" | ".env file" | "Streamlit secrets"
    model: str


def _clean(value: object) -> str | None:
    """Normalize a candidate setting to a non-empty string or None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _dotenv_file(path: Path | None) -> dict[str, str]:
    """Read a .env file without touching os.environ."""
    target = path or DEFAULT_DOTENV
    try:
        if not target.is_file():
            return {}
        return {key: value for key, value in dotenv_values(target).items() if value is not None}
    except OSError:
        return {}


def resolve_api_key(
    secrets: dict | None = None,
    *,
    env: dict | None = None,
    dotenv_path: Path | None = None,
) -> tuple[str | None, str | None]:
    """Find the API key. Returns ``(key, source)`` — ``None`` when absent.

    Precedence: a real environment variable wins, then the project
    ``.env`` file, then Streamlit secrets (``.streamlit/secrets.toml``,
    passed in by the view because core stays framework-free).
    """
    environment = os.environ if env is None else env

    key = _clean(environment.get(API_KEY_SETTING))
    if key:
        return key, "environment variable"

    key = _clean(_dotenv_file(dotenv_path).get(API_KEY_SETTING))
    if key:
        return key, ".env file"

    if secrets:
        key = _clean(secrets.get(API_KEY_SETTING))
        if key:
            return key, "Streamlit secrets"

    return None, None


def configured_model(
    secrets: dict | None = None,
    *,
    env: dict | None = None,
    dotenv_path: Path | None = None,
) -> str:
    """The model to call: GEMINI_MODEL from env/.env/secrets, else the default."""
    environment = os.environ if env is None else env

    model = _clean(environment.get(MODEL_SETTING))
    if model:
        return model
    model = _clean(_dotenv_file(dotenv_path).get(MODEL_SETTING))
    if model:
        return model
    if secrets:
        model = _clean(secrets.get(MODEL_SETTING))
        if model:
            return model
    return DEFAULT_MODEL


def api_config(
    secrets: dict | None = None,
    *,
    env: dict | None = None,
    dotenv_path: Path | None = None,
) -> ApiConfig:
    """Configuration status for the UI — never contains the key itself."""
    key, source = resolve_api_key(secrets, env=env, dotenv_path=dotenv_path)
    return ApiConfig(
        configured=key is not None,
        source=source,
        model=configured_model(secrets, env=env, dotenv_path=dotenv_path),
    )


# --- Client -----------------------------------------------------------------

#: One client per key, created on first use only. Client construction
#: performs no network I/O.
_CLIENTS: dict[str, genai.Client] = {}


def _client(api_key: str) -> genai.Client:
    client = _CLIENTS.get(api_key)
    if client is None:
        client = genai.Client(api_key=api_key)
        _CLIENTS[api_key] = client
    return client


def _scrub(text: str, api_key: str | None) -> str:
    """Remove the key from any text that might be shown to a user."""
    if api_key and api_key in text:
        text = text.replace(api_key, "«key hidden»")
    return text


def _shorten(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --- Error translation ------------------------------------------------------


def _status_text(exc: BaseException) -> str:
    """Uppercased status metadata (e.g. 'RESOURCE_EXHAUSTED') if present."""
    return " ".join(
        str(getattr(exc, attr, "") or "") for attr in ("status", "reason")
    ).upper()


def translate_error(
    exc: BaseException,
    *,
    model: str,
    timeout: float,
    api_key: str | None = None,
) -> GeminiError:
    """Map any SDK/network failure to a user-safe GeminiError.

    The google-genai SDK exposes HTTP status codes either as ``code``
    (legacy APIError) or as ``status_code`` (next-gen interactions
    errors); both shapes are handled structurally so this never depends
    on private imports. Detail text is scrubbed of the key and truncated.
    """
    if isinstance(exc, GeminiError):
        return exc

    raw_message = getattr(exc, "message", None) or str(exc) or type(exc).__name__
    detail = _shorten(_scrub(str(raw_message), api_key))
    name = type(exc).__name__

    status: object = getattr(exc, "code", None)
    if not isinstance(status, int):
        status = getattr(exc, "status_code", None)
    status_int = status if isinstance(status, int) else None

    # Timeouts and connection failures carry no usable status code.
    if isinstance(exc, TimeoutError) or "Timeout" in name or "timed out" in detail.lower():
        return GeminiTimeoutError(
            f"Gemini did not respond within {timeout:g} seconds. Try again.",
            detail,
        )
    if isinstance(exc, ConnectionError) or "Connection" in name:
        return GeminiConnectionError(GeminiConnectionError.default_message, detail)

    if status_int is not None:
        if status_int == 404:
            return GeminiNotFoundError(
                f"The Gemini model '{model}' was not found. Check that "
                f"GEMINI_MODEL names a model your API key may use.",
                detail,
            )
        if status_int == 400:
            return GeminiBadRequestError(
                f"Gemini rejected the request (HTTP 400). Check that "
                f"GEMINI_MODEL ('{model}') is a valid model name and that the "
                f"question is not empty.",
                detail,
            )
        if status_int == 429:
            return GeminiRateLimitError(GeminiRateLimitError.default_message, detail)
        if status_int == 403 and "RESOURCE_EXHAUSTED" in _status_text(exc):
            return GeminiRateLimitError(GeminiRateLimitError.default_message, detail)
        if status_int in (401, 403):
            return GeminiAuthError(
                f"Google rejected the API key (HTTP {status_int}). Check that "
                f"GEMINI_API_KEY is correct and active.",
                detail,
            )
        if status_int >= 500:
            return GeminiUnavailableError(
                f"Google's Gemini API returned an internal error (HTTP "
                f"{status_int}). Try again shortly.",
                detail,
            )

    # Status-less failures still get the right bucket when the text is clear.
    upper = f"{detail.upper()} {_status_text(exc)}"
    if "RESOURCE_EXHAUSTED" in upper or ("RATE" in upper and "LIMIT" in upper):
        return GeminiRateLimitError(GeminiRateLimitError.default_message, detail)
    if "API KEY" in upper and any(word in upper for word in ("INVALID", "NOT VALID", "MISSING")):
        return GeminiAuthError(GeminiAuthError.default_message, detail)

    status_note = f"HTTP {status_int}" if status_int is not None else name
    return GeminiError(
        f"The Gemini API returned an error ({status_note}). Try again, or "
        f"rephrase the question.",
        detail,
    )


# --- Structured generation ---------------------------------------------------


def _strip_fences(text: str) -> str:
    """Tolerate a ```json fence around otherwise valid output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        first_break = stripped.find("\n")
        if first_break != -1:
            stripped = stripped[first_break + 1 :]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


def parse_structured(text: str, schema: type[T]) -> T:
    """Parse JSON text into ``schema`` or raise a user-safe error.

    The failure message names the fields Pydantic rejected without
    echoing the model's raw output (it may be long and is not needed to
    fix a question).
    """
    try:
        return schema.model_validate_json(_strip_fences(text))
    except ValidationError as exc:
        parts = []
        for error in exc.errors()[:3]:
            location = ".".join(str(part) for part in error["loc"]) or "response"
            parts.append(f"{location}: {error['msg']}")
        raise GeminiResponseError(
            f"Gemini returned JSON that does not match the {schema.__name__} "
            f"schema — {'; '.join(parts)}. Try rephrasing the question."
        ) from None


def generate_structured(
    prompt: str,
    schema: type[T],
    *,
    system: str | None = None,
    client: Any = None,
    model: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    secrets: dict | None = None,
    env: dict | None = None,
    dotenv_path: Path | None = None,
) -> T:
    """One structured request: prompt in, validated Pydantic instance out.

    ``client`` may be injected for tests; otherwise the key is resolved
    and a cached client is created on demand (never earlier).
    """
    api_key: str | None = None
    if client is None:
        api_key, _ = resolve_api_key(secrets, env=env, dotenv_path=dotenv_path)
        if api_key is None:
            raise GeminiNotConfiguredError()
        client = _client(api_key)

    model_name = model or configured_model(secrets, env=env, dotenv_path=dotenv_path)

    try:
        response = client.interactions.create(
            model=model_name,
            input=prompt,
            system_instruction=system,
            generation_config=dict(GENERATION_CONFIG),
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": schema.model_json_schema(),
            },
            store=False,  # privacy: the exchange is not kept server-side
            timeout=timeout,
        )
    except Exception as exc:  # the SDK raises a wide range of shapes
        raise translate_error(exc, model=model_name, timeout=timeout, api_key=api_key) from None

    text = getattr(response, "output_text", None)
    if not text or not str(text).strip():
        raise GeminiResponseError(
            "Gemini returned an empty response instead of JSON. Try again."
        )
    return parse_structured(str(text), schema)

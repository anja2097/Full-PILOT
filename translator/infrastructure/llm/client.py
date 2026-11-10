"""HTTP client for the LLM API (OpenRouter or Hugging Face) with retry logic."""

from __future__ import annotations

import itertools
import json
import logging
import threading
import time

import requests
from pydantic import ValidationError

from translator.config.settings import API_URL
from translator.exceptions import LLMError
from translator.infrastructure.llm.models import ChatCompletion, ErrorResponse

logger = logging.getLogger(__name__)

RETRY_WAIT_SECONDS = 10
_RECOVERABLE_CODES = {429, 502, 503, 504}
_debug_seq = itertools.count(1)
_debug_lock = threading.Lock()


def _format_debug_body(body: object) -> str:
    if isinstance(body, (dict, list)):
        return json.dumps(body, indent=2, ensure_ascii=False)
    return str(body)


def _dump_exchange(
    *,
    method: str,
    url: str,
    label: str,
    request_body: object,
    status: int | str,
    response_body: object,
) -> None:
    """Log one request/response pair atomically. No-op unless DEBUG is enabled.

    The payload must not include Authorization headers. Parallel MoL callers
    share this lock so their JSON dumps do not interleave.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    with _debug_lock:
        n = next(_debug_seq)
        block = (
            f"\n[DEBUG API #{n}] {label}\n"
            f"  {method} {url}\n"
            f"--- REQUEST ---\n{_format_debug_body(request_body)}\n"
            f"--- RESPONSE HTTP {status} ---\n{_format_debug_body(response_body)}\n"
        )
        logger.debug("%s", block)


def _is_recoverable(code: int | str) -> bool:
    try:
        return int(code) in _RECOVERABLE_CODES
    except (ValueError, TypeError):
        return False


def _api_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _reasoning_payload(*, thinking: bool, effort: str | None) -> dict | None:
    if not thinking:
        return None
    reasoning: dict = {"enabled": True}
    if effort is not None:
        reasoning["effort"] = effort
    return reasoning


def _do_request(
    api_key: str,
    payload: dict,
    api_url: str = API_URL,
    *,
    label: str = "",
) -> ChatCompletion | tuple[int | str, str]:
    """Send a single HTTP request to the configured API endpoint.

    Returns ChatCompletion on success, or (code, message) tuple on error.
    Does not raise exceptions so the caller can retry on recoverable errors.
    """
    response = requests.post(
        api_url,
        headers=_api_headers(api_key),
        json=payload,
        timeout=120,
    )

    logger.info("\nHTTP %s", response.status_code)

    try:
        data = response.json()
    except json.JSONDecodeError:
        _dump_exchange(
            method="POST",
            url=api_url,
            label=label,
            request_body=payload,
            status=response.status_code,
            response_body=response.text,
        )
        logger.info("Response is not JSON:")
        logger.info("%s", response.text[:500])
        return response.status_code, "Response is not JSON"

    _dump_exchange(
        method="POST",
        url=api_url,
        label=label,
        request_body=payload,
        status=response.status_code,
        response_body=data,
    )

    # Explicit error in the HTTP response (4xx/5xx)
    if not response.ok:
        logger.info("API error:")
        logger.info("%s", json.dumps(data, indent=2, ensure_ascii=False))
        return response.status_code, data.get("error", {}).get("message", "API error")

    # Error embedded in an HTTP 200 body (e.g. rate limit, provider down)
    if "error" in data:
        try:
            err = ErrorResponse.model_validate(data)
            return err.error.code, err.error.message
        except ValidationError:
            return 500, "Unknown error in the response body"

    try:
        return ChatCompletion.model_validate(data)
    except ValidationError as exc:
        logger.info("The API response does not have the expected format:")
        logger.info("%s", exc)
        return 500, "Unexpected response format"


def chat_completion(
    api_key: str,
    messages: list[dict],
    *,
    model: str,
    api_url: str = API_URL,
    model_provider: str | None = None,
    thinking: bool = False,
    thinking_effort: str | None = None,
    label: str = "",
    temperature: float | None = None,
    seed: int | None = None,
) -> ChatCompletion:
    """Send a chat completion request with one automatic retry on recoverable errors.

    Args:
        api_key: API key for the selected provider.
        messages: Conversation history in OpenAI format.
        model: Model ID as expected by the provider.
        api_url: Endpoint URL (OpenRouter or Hugging Face).
        model_provider: OpenRouter inference provider to use exclusively.
        thinking: Whether to enable extended reasoning (OpenRouter only).
        thinking_effort: Reasoning effort level (requires ``thinking=True``).
        label: Human-readable label for log output.
        temperature: Sampling temperature. Omitted from the payload when unset.
        seed: Optional deterministic seed. Omitted from the payload when unset.

    Raises:
        LLMError: On unrecoverable API errors.
    """
    payload: dict = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if seed is not None:
        payload["seed"] = seed
    if model_provider is not None:
        payload["provider"] = {"only": [model_provider]}
    reasoning = _reasoning_payload(thinking=thinking, effort=thinking_effort)
    if reasoning is not None:
        payload["reasoning"] = reasoning

    if label:
        logger.info("\n\n%s", "-" * 50)
        logger.info("  %s", label)
        logger.info("%s", "-" * 50)

    for attempt in (1, 2):
        result = _do_request(api_key, payload, api_url, label=label)

        if isinstance(result, ChatCompletion):
            return result

        code, message = result
        if attempt == 1 and _is_recoverable(code):
            logger.warning("[WARNING] Recoverable error (code %s): %s", code, message)
            logger.warning("          Retrying in %s s…", RETRY_WAIT_SECONDS)
            time.sleep(RETRY_WAIT_SECONDS)
            continue

        raise LLMError(f"API: {message} (code {code})", code=code)

    raise LLMError("API: unrecoverable error after retries")  # pragma: no cover


_GENERATION_URL = "https://openrouter.ai/api/v1/generation"
_GENERATION_RETRIES = 3
_GENERATION_RETRY_WAIT = 5


class GenerationMetadata:
    """Metadatos obtenidos del endpoint /api/v1/generation."""

    __slots__ = ("generation_time_ms", "total_cost_usd")

    def __init__(
        self,
        generation_time_ms: float | None = None,
        total_cost_usd: float | None = None,
    ) -> None:
        self.generation_time_ms = generation_time_ms
        self.total_cost_usd = total_cost_usd


def fetch_generation_metadata(api_key: str, generation_id: str) -> GenerationMetadata:
    """GET /api/v1/generation?id={id} y devuelve GenerationMetadata con tiempo y coste.

    Reintenta hasta _GENERATION_RETRIES veces cuando los datos aún no están disponibles.
    Los campos del resultado son None si no se pudieron obtener.
    """
    params = {"id": generation_id}
    for attempt in range(1, _GENERATION_RETRIES + 1):
        try:
            resp = requests.get(
                _GENERATION_URL,
                params=params,
                headers=_api_headers(api_key),
                timeout=15,
            )
            try:
                body: object = resp.json()
            except json.JSONDecodeError:
                body = resp.text
            _dump_exchange(
                method="GET",
                url=_GENERATION_URL,
                label=f"generation {generation_id} attempt {attempt}",
                request_body=params,
                status=resp.status_code,
                response_body=body,
            )
            if resp.ok and isinstance(body, dict):
                data = body.get("data", {})
                gen_time = data.get("generation_time")
                total_cost = data.get("total_cost")
                if gen_time is not None or total_cost is not None:
                    return GenerationMetadata(
                        generation_time_ms=gen_time,
                        total_cost_usd=total_cost,
                    )
        except Exception as exc:
            _dump_exchange(
                method="GET",
                url=_GENERATION_URL,
                label=f"generation {generation_id} attempt {attempt}",
                request_body=params,
                status="error",
                response_body=str(exc),
            )
        if attempt < _GENERATION_RETRIES:
            logger.debug(
                "Metadata unavailable for %s (attempt %s/%s), retrying in %ss…",
                generation_id, attempt, _GENERATION_RETRIES, _GENERATION_RETRY_WAIT,
            )
            time.sleep(_GENERATION_RETRY_WAIT)
    logger.warning(
        "Could not fetch metadata for %s after %s attempts",
        generation_id, _GENERATION_RETRIES,
    )
    return GenerationMetadata()

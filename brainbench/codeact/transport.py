"""Shared target-model transport and application-level retry policy."""

from __future__ import annotations

import json
import logging
import os
import random
import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Optional

import httpcore
import httpx
import openai
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)


logger = logging.getLogger(__name__)
_VERSIONS_LOGGED = False
_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 502, 503, 504}


class CaseDeadlineExceeded(TimeoutError):
    """Raised when no case budget remains for another model request."""


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def log_network_runtime_versions() -> None:
    global _VERSIONS_LOGGED
    if _VERSIONS_LOGGED:
        return
    _VERSIONS_LOGGED = True
    # logger.info(
    #     "Target network runtime: python=%s openai=%s httpx=%s httpcore=%s",
    #     sys.version.split()[0],
    #     openai.__version__,
    #     httpx.__version__,
    #     httpcore.__version__,
    # )


def make_openai_client(
    *,
    api_key: str,
    base_url: str,
    request_timeout_sec: float,
    connect_timeout_sec: float = 10.0,
) -> OpenAI:
    """Build one client with no hidden retries and a correctly configured pool."""

    log_network_runtime_versions()
    http2 = _env_bool("NEUROBENCH_TARGET_HTTP2", False)
    limits = httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=90.0,
    )
    transport = httpx.HTTPTransport(
        retries=0,
        http2=http2,
        trust_env=False,
        limits=limits,
    )
    http_client = httpx.Client(
        transport=transport,
        timeout=httpx.Timeout(
            request_timeout_sec,
            connect=min(connect_timeout_sec, request_timeout_sec),
        ),
    )
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=http_client.timeout,
        max_retries=0,
        http_client=http_client,
    )


def _anthropic_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    converted = []
    for item in content:
        if not isinstance(item, Mapping):
            continue
        if item.get("type") == "text":
            converted.append({"type": "text", "text": str(item.get("text", ""))})
            continue
        if item.get("type") != "image_url":
            continue
        image_url = item.get("image_url")
        url = image_url.get("url") if isinstance(image_url, Mapping) else image_url
        if not isinstance(url, str):
            continue
        if url.startswith("data:") and ";base64," in url:
            header, data = url.split(",", 1)
            media_type = header[5:].split(";", 1)[0]
            source = {"type": "base64", "media_type": media_type, "data": data}
        else:
            source = {"type": "url", "url": url}
        converted.append({"type": "image", "source": source})
    return converted


def _anthropic_request(request: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(request)
    messages = []
    system_parts = []
    for message in source.pop("messages", []):
        role = str(message.get("role", "user"))
        content = _anthropic_content(message.get("content", ""))
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            else:
                system_parts.extend(
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, Mapping) and item.get("type") == "text"
                )
            continue
        messages.append({
            "role": "assistant" if role == "assistant" else "user",
            "content": content,
        })

    if source.pop("response_format", None) is not None:
        system_parts.append(
            "Return only one valid JSON object, with no Markdown fences or commentary."
        )

    result: dict[str, Any] = {
        "model": source.pop("model"),
        "messages": messages,
        "max_tokens": int(source.pop("max_tokens", 4096)),
    }
    if system_parts:
        result["system"] = "\n\n".join(part for part in system_parts if part)
    if "stop" in source:
        result["stop_sequences"] = source.pop("stop")
    source.pop("extra_body", None)
    source.pop("reasoning_effort", None)
    source.pop("stream", None)
    result.update(source)
    return result


class _AnthropicCompletionsAdapter:
    def __init__(self, client: Any) -> None:
        self._client = client

    def create(self, **request: Any) -> Any:
        response = self._client.messages.create(**_anthropic_request(request))
        content = "".join(
            str(getattr(block, "text", ""))
            for block in getattr(response, "content", [])
            if getattr(block, "type", None) == "text"
        )
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        finish_reason = (
            "length"
            if getattr(response, "stop_reason", None) == "max_tokens"
            else "stop"
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason=finish_reason,
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )


class AnthropicChatAdapter:
    """Expose Anthropic Messages through the chat-completions surface NeuroBench uses."""

    api_protocol = "anthropic"

    def __init__(self, client: Any, *, api_key: str, base_url: str) -> None:
        self._client = client
        self.api_key = api_key
        self.base_url = base_url
        self.chat = SimpleNamespace(
            completions=_AnthropicCompletionsAdapter(client)
        )

    def close(self) -> None:
        self._client.close()


def make_anthropic_client(
    *,
    api_key: str,
    base_url: str,
    request_timeout_sec: float,
    connect_timeout_sec: float = 10.0,
) -> AnthropicChatAdapter:
    """Build an Anthropic SDK client for the native ``/v1/messages`` protocol."""

    try:
        from anthropic import Anthropic, DefaultHttpxClient
    except ImportError as exc:
        raise RuntimeError(
            "Anthropic protocol requires the 'anthropic' package."
        ) from exc

    normalized_base_url = str(base_url).rstrip("/")
    if normalized_base_url.endswith("/v1"):
        normalized_base_url = normalized_base_url[:-3]

    limits = httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=90.0,
    )
    transport = httpx.HTTPTransport(
        retries=0,
        http2=_env_bool("NEUROBENCH_TARGET_HTTP2", False),
        trust_env=False,
        limits=limits,
    )
    timeout = httpx.Timeout(
        request_timeout_sec,
        connect=min(connect_timeout_sec, request_timeout_sec),
    )
    client = Anthropic(
        api_key=api_key,
        base_url=normalized_base_url,
        timeout=timeout,
        max_retries=0,
        http_client=DefaultHttpxClient(
            transport=transport,
            timeout=timeout,
        ),
    )
    return AnthropicChatAdapter(
        client,
        api_key=api_key,
        base_url=normalized_base_url,
    )


def make_llm_client(
    *,
    api_key: str,
    base_url: str,
    request_timeout_sec: float,
    api_protocol: str = "openai",
    connect_timeout_sec: float = 10.0,
) -> Any:
    protocol = str(api_protocol).strip().lower()
    if protocol == "openai":
        client = make_openai_client(
            api_key=api_key,
            base_url=base_url,
            request_timeout_sec=request_timeout_sec,
            connect_timeout_sec=connect_timeout_sec,
        )
        client.api_protocol = "openai"
        return client
    if protocol == "anthropic":
        return make_anthropic_client(
            api_key=api_key,
            base_url=base_url,
            request_timeout_sec=request_timeout_sec,
            connect_timeout_sec=connect_timeout_sec,
        )
    raise ValueError("api_protocol must be 'openai' or 'anthropic'")


def case_deadline_from_env() -> Optional[float]:
    """Convert the host-provided wall-clock deadline to this process' monotonic clock."""

    raw = os.getenv("NEUROBENCH_CASE_DEADLINE_EPOCH", "").strip()
    if not raw:
        return None
    try:
        remaining = float(raw) - time.time()
    except ValueError:
        logger.warning("Invalid NEUROBENCH_CASE_DEADLINE_EPOCH=%r", raw)
        return None
    return time.monotonic() + max(0.0, remaining)


def is_retryable_api_error(exc: BaseException) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        status_code = int(getattr(exc, "status_code", 0) or 0)
        return status_code in _RETRYABLE_STATUS_CODES or status_code >= 500
    if type(exc).__module__.split(".", 1)[0] == "anthropic":
        status_code = int(getattr(exc, "status_code", 0) or 0)
        if type(exc).__name__ in {
            "APIConnectionError",
            "APITimeoutError",
            "RateLimitError",
        }:
            return True
        if type(exc).__name__ == "APIStatusError":
            return status_code in _RETRYABLE_STATUS_CODES or status_code >= 500
    cause = getattr(exc, "__cause__", None)
    return bool(cause is not None and cause is not exc and is_retryable_api_error(cause))


def exception_diagnostics(exc: BaseException) -> dict[str, Any]:
    return {
        "exception_type": type(exc).__name__,
        "exception_repr": repr(exc),
        "cause_repr": repr(getattr(exc, "__cause__", None)),
        "context_repr": repr(getattr(exc, "__context__", None)),
    }


def request_chat_completion(
    client: OpenAI,
    request: Mapping[str, Any],
    *,
    deadline: Optional[float],
    request_timeout_cap_sec: float,
    connect_timeout_sec: float = 10.0,
    max_attempts: int = 3,
    context: Optional[Mapping[str, Any]] = None,
    audit: Optional[list[dict[str, Any]]] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Any:
    """Execute one logical request with one budget-aware retry layer."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    request_context = dict(context or {})
    last_error: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        remaining = (
            request_timeout_cap_sec
            if deadline is None
            else deadline - time.monotonic()
        )
        if remaining <= 0:
            raise CaseDeadlineExceeded("Case deadline reached before model request")

        request_timeout = min(request_timeout_cap_sec, remaining)
        trace_id = uuid.uuid4().hex
        started_utc = datetime.now(timezone.utc).isoformat()
        call_kwargs = dict(request)
        headers = dict(call_kwargs.get("extra_headers") or {})
        headers["X-Client-Request-Id"] = trace_id
        call_kwargs["extra_headers"] = headers
        call_kwargs["timeout"] = httpx.Timeout(
            request_timeout,
            connect=min(connect_timeout_sec, request_timeout),
        )
        event = {
            **request_context,
            "application_attempt": attempt,
            "max_application_attempts": max_attempts,
            "trace_id": trace_id,
            "started_at_utc": started_utc,
            "remaining_case_budget_sec": round(remaining, 3),
            "request_timeout_sec": round(request_timeout, 3),
            "connect_timeout_sec": round(
                min(connect_timeout_sec, request_timeout), 3
            ),
        }
        try:
            response = client.chat.completions.create(**call_kwargs)
        except Exception as exc:
            last_error = exc
            event.update(exception_diagnostics(exc))
            event["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            event["retryable"] = is_retryable_api_error(exc)
            if audit is not None:
                audit.append(dict(event))
            logger.warning(
                "Target API request failed: %s",
                json.dumps(event, ensure_ascii=True, sort_keys=True),
            )
            if not event["retryable"] or attempt >= max_attempts:
                try:
                    setattr(exc, "neurobench_api_trace", list(audit or [event]))
                    setattr(exc, "is_api_infrastructure_error", event["retryable"])
                except Exception:
                    pass
                raise

            remaining_after = (
                request_timeout_cap_sec
                if deadline is None
                else deadline - time.monotonic()
            )
            delay = min(
                (0.5 * (2 ** (attempt - 1))) + random.uniform(0.0, 0.25),
                max(0.0, remaining_after),
            )
            if delay <= 0:
                raise CaseDeadlineExceeded(
                    "Case deadline reached before model request retry"
                ) from exc
            sleep_fn(delay)
            continue

        event["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        event["status"] = "ok"
        if audit is not None:
            audit.append(event)
        return response

    raise RuntimeError("Unreachable retry state") from last_error

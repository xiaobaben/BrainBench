"""Shared request policy for OpenAI-compatible chat completion calls."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional


ReasoningEffort = Literal["low", "medium", "high", "xhigh", "max"]
_REASONING_EFFORTS = {"low", "medium", "high", "xhigh", "max"}


_OPENROUTER_DISABLED_REASONING_EFFORT = {
    "gpt-5.4-mini": "none",
    "gpt-5-mini": "none",
    "gemini-2.5-flash": "none",
    "gemini-3.5-flash": "none",
}


def _model_leaf_name(model: str) -> str:
    """Return the model slug without an OpenRouter namespace or variant."""

    return str(model).lower().rsplit("/", 1)[-1].split(":", 1)[0]


def is_qwen_model(model: str) -> bool:
    """Identify the Qwen model family without maintaining version lists."""

    return "qwen" in str(model).lower()


def model_provider(model: str) -> str:
    """Infer a provider family from common public model identifiers."""

    name = str(model).lower()
    if "qwen" in name:
        return "qwen"
    if "deepseek" in name:
        return "deepseek"
    if "kimi" in name or "moonshot" in name:
        return "kimi"
    if "gemini" in name:
        return "gemini"
    if "claude" in name:
        return "anthropic"
    if "chatglm" in name or re.search(r"(^|[/_-])glm(?:[-_.]|$)", name):
        return "glm"
    if re.search(r"(^|/)gpt[-_.]", name) or re.search(r"(^|/)o[134](?:[-_.]|$)", name):
        return "openai"
    return "generic"


def _thinking_controls(model: str, enabled: bool) -> Dict[str, Any]:
    provider = model_provider(model)
    if provider == "qwen":
        return {"extra_body": {"enable_thinking": enabled}}
    if provider in {"deepseek", "glm", "kimi"}:
        return {"extra_body": {"thinking": {"type": "enabled" if enabled else "disabled"}}}
    leaf_name = _model_leaf_name(model)
    if not enabled and leaf_name in _OPENROUTER_DISABLED_REASONING_EFFORT:
        return {
            "extra_body": {
                "reasoning": {
                    "effort": _OPENROUTER_DISABLED_REASONING_EFFORT[leaf_name]
                }
            }
        }
    if provider == "gemini":
        if enabled:
            return {"reasoning_effort": "medium"}
        effort = "none"
        return {"reasoning_effort": effort}
    if provider == "anthropic":
        if enabled:
            return {"thinking": {"type": "enabled", "budget_tokens": 1024}}
        return {}
    if provider == "openai":
        version = re.match(r"gpt-(\d+)(?:\.(\d+))?", leaf_name)
        if version:
            major = int(version.group(1))
            minor = int(version.group(2) or 0)
            if major < 5:
                return {}
            effort = "none"
            return {"reasoning_effort": "medium" if enabled else effort}
    return {}


def supports_stop_sequences(model: str) -> bool:
    """Whether the selected OpenRouter model accepts the chat ``stop`` field."""

    return not (
        model_provider(model) == "openai"
        and _model_leaf_name(model).startswith("gpt-5")
    )


def supports_temperature(model: str) -> bool:
    """Whether the selected model accepts an explicit sampling temperature."""

    provider = model_provider(model)
    leaf_name = _model_leaf_name(model)
    if provider == "kimi":
        return False
    if provider == "anthropic" and "opus-5" in leaf_name:
        return False
    return not (
        provider == "openai"
        and leaf_name.startswith("gpt-5")
    )


def build_chat_completion_kwargs(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    force_json: bool = False,
    enable_thinking: Optional[bool] = False,
    reasoning_effort: Optional[ReasoningEffort] = None,
    temperature: Optional[float] = None,
    stream: Optional[bool] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Build request arguments from capabilities required by the call site."""

    kwargs: Dict[str, Any] = {"model": model, "messages": messages}
    if force_json:
        kwargs["response_format"] = {"type": "json_object"}
    if enable_thinking is not None:
        kwargs.update(_thinking_controls(model, bool(enable_thinking)))
    if reasoning_effort is not None:
        if reasoning_effort not in _REASONING_EFFORTS:
            allowed = ", ".join(sorted(_REASONING_EFFORTS))
            raise ValueError(f"reasoning_effort must be one of: {allowed}")
        if model_provider(model) != "openai" or not _model_leaf_name(model).startswith("gpt-5"):
            raise ValueError("reasoning_effort is only supported for GPT-5 target models")
        kwargs.pop("extra_body", None)
        kwargs["reasoning_effort"] = reasoning_effort
    if temperature is not None and supports_temperature(model):
        kwargs["temperature"] = temperature
    if stream is not None:
        kwargs["stream"] = stream
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)
    return kwargs

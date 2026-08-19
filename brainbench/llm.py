"""Optional Parser/Judge adapters using the shared environment configuration."""

from __future__ import annotations

import json
import base64
import mimetypes
import re
from pathlib import Path
from typing import Any, Dict

from .config import EndpointConfig
from .codeact.llm_requests import build_chat_completion_kwargs
from .codeact.transport import make_llm_client, request_chat_completion


def _extract_json(text: str) -> Dict[str, Any]:
    match = re.search(r"(\{.*\})", text or "", flags=re.DOTALL)
    return json.loads(match.group(1) if match else text)


def make_json_parser(config: EndpointConfig):
    client = make_llm_client(
        api_key=config.api_key,
        base_url=config.base_url,
        request_timeout_sec=config.request_timeout_sec,
        api_protocol=config.api_protocol,
    )

    def parser(response: str, prompt: str) -> Dict[str, Any]:
        request = build_chat_completion_kwargs(
            model=config.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": response},
            ],
            force_json=True,
            temperature=0.0,
            max_tokens=512,
        )
        result = request_chat_completion(
            client,
            request,
            deadline=None,
            request_timeout_cap_sec=config.request_timeout_sec,
            max_attempts=3,
        )
        usage = getattr(result, "usage", None)
        parser.total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
        return _extract_json(str(result.choices[0].message.content))

    parser.total_tokens = 0
    parser.model_config = {
        "role": "parser",
        "model": config.model,
        "endpoint_mode": "api",
        "base_url": config.base_url,
        "api_protocol": config.api_protocol,
        "request_timeout_sec": config.request_timeout_sec,
    }
    return parser


def _role_config(config: EndpointConfig, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "model": config.model,
        "endpoint_mode": "api",
        "base_url": config.base_url,
        "api_protocol": config.api_protocol,
        "request_timeout_sec": config.request_timeout_sec,
    }


def _json_role_request(
    client: Any,
    config: EndpointConfig,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int | None = None,
) -> tuple[Dict[str, Any], int]:
    request = build_chat_completion_kwargs(
        model=config.model,
        messages=messages,
        force_json=True,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    result = request_chat_completion(
        client,
        request,
        deadline=None,
        request_timeout_cap_sec=config.request_timeout_sec,
        max_attempts=3,
    )
    usage = getattr(result, "usage", None)
    tokens = int(getattr(usage, "total_tokens", 0) or 0)
    return _extract_json(str(result.choices[0].message.content)), tokens


def make_semantic_judge(config: EndpointConfig):
    """Build the JSON semantic judge using the shared parser endpoint."""

    client = make_llm_client(
        api_key=config.api_key,
        base_url=config.base_url,
        request_timeout_sec=config.request_timeout_sec,
        api_protocol=config.api_protocol,
    )

    def semantic_judge(response: str, prompt: str) -> Dict[str, Any]:
        payload, tokens = _json_role_request(
            client,
            config,
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": response},
            ],
        )
        semantic_judge.total_tokens += tokens
        return payload

    semantic_judge.total_tokens = 0
    semantic_judge.model_config = _role_config(config, "semantic")
    return semantic_judge


def _image_data_url(path: str | Path) -> str:
    image_path = Path(path)
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def make_vlm_judge(config: EndpointConfig):
    """Build the JSON VLM judge using the shared parser endpoint."""

    client = make_llm_client(
        api_key=config.api_key,
        base_url=config.base_url,
        request_timeout_sec=config.request_timeout_sec,
        api_protocol=config.api_protocol,
    )

    def vlm_judge(image_path: str, prompt: str) -> Dict[str, Any]:
        payload, tokens = _json_role_request(
            client,
            config,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_data_url(image_path)},
                        },
                    ],
                }
            ],
        )
        vlm_judge.total_tokens += tokens
        return payload

    vlm_judge.total_tokens = 0
    vlm_judge.model_config = _role_config(config, "vlm")
    return vlm_judge

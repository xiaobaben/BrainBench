"""Optional Parser/Judge adapters using the shared environment configuration."""

from __future__ import annotations

import json
import re
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

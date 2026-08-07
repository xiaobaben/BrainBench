"""Environment-only configuration for model endpoints."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EndpointConfig:
    """One OpenAI-compatible or native Anthropic endpoint configuration."""

    api_key: str
    base_url: str
    model: str
    api_protocol: str = "openai"
    request_timeout_sec: float = 300.0

    @classmethod
    def from_env(cls, prefix: str = "BRAINBENCH") -> "EndpointConfig":
        """Read endpoint settings without accepting credentials from source code.

        ``NEUROBENCH_*`` remains supported as a legacy prefix so existing local
        environments continue to work after the project rename.
        """

        def env_value(suffix: str, default: str | None = None) -> str | None:
            value = os.getenv(f"{prefix}_{suffix}")
            if value is None and prefix == "BRAINBENCH":
                value = os.getenv(f"NEUROBENCH_{suffix}")
            return default if value is None else value

        url_name = f"{prefix}_BASE_URL"
        model_name = f"{prefix}_MODEL"
        protocol_name = f"{prefix}_API_PROTOCOL"
        timeout_name = f"{prefix}_REQUEST_TIMEOUT_SEC"

        base_url = (env_value("BASE_URL", "") or "").strip()
        if not base_url:
            raise ValueError(f"{url_name} must be set")

        api_key = (env_value("API_KEY", "none") or "none").strip() or "none"
        model = (env_value("MODEL", "") or "").strip()
        if not model:
            raise ValueError(f"{model_name} must be set")

        try:
            timeout = float(env_value("REQUEST_TIMEOUT_SEC", "300") or "300")
        except ValueError as exc:
            raise ValueError(f"{timeout_name} must be numeric") from exc
        if timeout <= 0:
            raise ValueError(f"{timeout_name} must be positive")

        protocol = (env_value("API_PROTOCOL", "openai") or "openai").strip().lower()
        if protocol not in {"openai", "anthropic"}:
            raise ValueError(f"{protocol_name} must be 'openai' or 'anthropic'")

        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            api_protocol=protocol,
            request_timeout_sec=timeout,
        )

    def redacted(self) -> dict[str, object]:
        """Return safe-to-log configuration metadata."""

        return {
            "base_url": self.base_url,
            "model": self.model,
            "api_protocol": self.api_protocol,
            "request_timeout_sec": self.request_timeout_sec,
            "api_key_configured": self.api_key != "none",
        }

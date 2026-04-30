from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any

try:
    import litellm
except ImportError:  # pragma: no cover - handled at runtime
    litellm = None  # type: ignore

_BACKOFF_MULTIPLIER = 2


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generates a response from the LLM."""
        raise NotImplementedError


class LiteLLMProvider(LLMProvider):
    """Vendor-agnostic provider that routes through LiteLLM."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        *,
        max_retries: int = 2,
        backoff_base: float = 0.5,
        timeout: float | None = 30.0,
        max_backoff: float = 4.0,
        **kwargs: Any,
    ):
        if litellm is None:
            raise ImportError("litellm is not installed. Please add it to your environment.")

        self.model_name = model_name
        # Prefer explicit key; fall back to common env vars supported by litellm dispatch.
        self.api_key = api_key or os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.completion_kwargs: dict[str, Any] = kwargs
        self.max_retries = max(0, max_retries)
        self.backoff_base = max(0.0, backoff_base)
        self.timeout = timeout
        self.max_backoff = max(0.0, max_backoff)

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        if litellm is None:
            raise ImportError("litellm is not available at runtime.")

        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        if self.api_key:
            params["api_key"] = self.api_key

        params.update(self.completion_kwargs)

        if self.timeout is not None and "timeout" not in params:
            params["timeout"] = self.timeout

        response = None
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = litellm.completion(**params)
                last_error = None
                break
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                if attempt >= self.max_retries:
                    break
                multiplier = (
                    1 << attempt
                    if _BACKOFF_MULTIPLIER == 2
                    else _BACKOFF_MULTIPLIER**attempt
                )
                sleep_for = self.backoff_base * multiplier
                if self.max_backoff:
                    sleep_for = min(sleep_for, self.max_backoff)
                if sleep_for > 0:
                    time.sleep(sleep_for)

        if response is None:
            message = f"LiteLLM completion failed: {last_error}" if last_error else "LiteLLM completion failed."
            raise RuntimeError(message) from last_error

        try:
            choice = response.choices[0]
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"LiteLLM response missing choices: {exc}") from exc

        content = None
        if hasattr(choice, "message"):
            content = getattr(choice.message, "content", None)
        if content is None and hasattr(choice, "text"):
            content = choice.text

        if content is None:
            raise RuntimeError("LiteLLM returned an empty message.")

        # Litellm may return structured content; coerce to string.
        if isinstance(content, list):
            content = "".join(str(part) for part in content)

        return str(content)


class MockProvider(LLMProvider):
    """A mock provider for testing without API keys."""

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return '{"approved": true, "reason": "Mock approval from Clean Room.", "risk_score": 0.0}'

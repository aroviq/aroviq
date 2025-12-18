from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

try:
    import litellm
except ImportError:  # pragma: no cover - handled at runtime
    litellm = None  # type: ignore


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generates a response from the LLM."""
        raise NotImplementedError


class LiteLLMProvider(LLMProvider):
    """Vendor-agnostic provider that routes through LiteLLM."""

    def __init__(self, model_name: str, api_key: str | None = None, **kwargs: Any):
        if litellm is None:
            raise ImportError("litellm is not installed. Please add it to your environment.")

        self.model_name = model_name
        # Prefer explicit key; fall back to common env vars supported by litellm dispatch.
        self.api_key = api_key or os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.completion_kwargs: dict[str, Any] = kwargs

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

        try:
            response = litellm.completion(**params)
        except Exception as exc:  # pragma: no cover - network dependent
            raise RuntimeError(f"LiteLLM completion failed: {exc}") from exc

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

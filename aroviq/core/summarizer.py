from __future__ import annotations

import hashlib
from collections import OrderedDict

from aroviq.core.llm import LiteLLMProvider


class ContextSummarizer:
    """Summarize agent history into a compact safety context."""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: str | None = None,
        provider: LiteLLMProvider | None = None,
        *,
        max_history_entries: int = 50,
        max_history_chars: int = 8000,
        max_summary_chars: int = 2000,
        cache_size: int = 128,
    ):
        self.provider = provider or LiteLLMProvider(model_name=model_name, api_key=api_key)
        self.max_history_entries = max_history_entries
        self.max_history_chars = max_history_chars
        self.max_summary_chars = max_summary_chars
        self.cache_size = cache_size
        self._cache: OrderedDict[str, str] = OrderedDict()

    def summarize(self, history: list[str]) -> str:
        if not history:
            return "No prior steps or permissions recorded."

        trimmed = history[-self.max_history_entries :] if self.max_history_entries else history
        history_blob = "\n---\n".join(trimmed)

        if len(history_blob) > self.max_history_chars:
            return self._fallback_summary(trimmed, len(history))

        cache_key = self._hash_history(history_blob)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        prompt = (
            "Summarize the user's authorizations and the agent's progress. Focus ONLY on permissions given and actions taken. "
            "Ignore conversational filler.\n\n"
            "History:\n"
            f"{history_blob}\n"
        )

        try:
            summary = self.provider.generate(prompt, temperature=0.0)
        except Exception as exc:  # pragma: no cover - network/API dependent
            return f"Summary unavailable due to summarizer error: {exc}"

        cleaned = self._sanitize_summary(summary)
        if cleaned:
            self._remember(cache_key, cleaned)
            return cleaned
        fallback = "Summary unavailable from summarizer."
        self._remember(cache_key, fallback)
        return fallback

    def _sanitize_summary(self, summary: str) -> str:
        cleaned = summary.strip()
        if cleaned.startswith("```"):
            _, _, remainder = cleaned.partition("```")
            cleaned = remainder
            if "```" in cleaned:
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()
        if len(cleaned) > self.max_summary_chars:
            cleaned = f"{cleaned[: self.max_summary_chars]}...[truncated]"
        return cleaned

    def _hash_history(self, history_blob: str) -> str:
        return hashlib.sha256(history_blob.encode("utf-8")).hexdigest()

    def _remember(self, key: str, value: str) -> None:
        if self.cache_size <= 0:
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _fallback_summary(self, trimmed_history: list[str], total_entries: int) -> str:
        if not trimmed_history:
            return "No prior steps or permissions recorded."
        last_entry = trimmed_history[-1]
        snippet = last_entry[:200]
        return (
            f"{total_entries} prior steps recorded. "
            f"Last entry: {snippet}{'...[truncated]' if len(last_entry) > 200 else ''}"
        )

from __future__ import annotations

from aroviq.core.llm import LiteLLMProvider


class ContextSummarizer:
    """Summarize agent history into a compact safety context."""

    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: str | None = None, provider: LiteLLMProvider | None = None):
        self.provider = provider or LiteLLMProvider(model_name=model_name, api_key=api_key)

    def summarize(self, history: list[str]) -> str:
        if not history:
            return "No prior steps or permissions recorded."

        history_blob = "\n---\n".join(history)
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

        cleaned = summary.strip()
        return cleaned or "Summary unavailable from summarizer."

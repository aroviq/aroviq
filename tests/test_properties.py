import pytest
from hypothesis import given, strategies as st, settings
from aroviq.core.summarizer import ContextSummarizer
from aroviq.core.llm import LLMProvider

class FuzzMockProvider(LLMProvider):
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return "Fuzz Test Summary"

@given(st.lists(st.text()))
@settings(max_examples=50)
def test_summarizer_resilience(history):
    """
    Property: For any random list of input strings, the summarizer should never crash and always return a string.
    """
    provider = FuzzMockProvider()
    summarizer = ContextSummarizer(provider=provider)
    summary = summarizer.summarize(history)
    assert isinstance(summary, str)
    assert len(summary) > 0

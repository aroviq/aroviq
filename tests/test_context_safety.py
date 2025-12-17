import types
import importlib

import pytest

from aroviq.core.llm import LLMProvider
from aroviq.core.models import AgentContext, Step, StepType, Verdict
from aroviq.core.summarizer import ContextSummarizer
from aroviq.verifiers.logic import LogicVerifier

scan_module = importlib.import_module("aroviq.scan")


class DummyLLM(LLMProvider):
    def __init__(self, response: str):
        self.response = response
        self.last_prompt: str | None = None

    def generate(self, prompt: str, temperature: float = 0.0) -> str:  # noqa: ARG002
        self.last_prompt = prompt
        return self.response


class DummySummarizer(ContextSummarizer):
    def __init__(self, summary: str):
        self.summary = summary

    def summarize(self, history):  # noqa: ANN001
        return self.summary


class ExplodingProvider:
    def generate(self, prompt: str, temperature: float = 0.0) -> str:  # noqa: ARG002
        raise AssertionError("provider should not be called")


class ErroringProvider:
    def generate(self, prompt: str, temperature: float = 0.0) -> str:  # noqa: ARG002
        raise RuntimeError("boom")


def test_logic_verifier_builds_safety_prompt_and_parses_response():
    llm = DummyLLM('{"approved": true, "reason": "ok", "risk_score": 0.05}')
    summarizer = DummySummarizer("User authorized deletion of temp files.")
    verifier = LogicVerifier(llm_provider=llm, summarizer=summarizer)

    context = AgentContext(
        user_goal="Clean temp",
        current_state_snapshot={"cwd": "/tmp"},
        history=["Thought: prepare", "Action: list temp"],
        safety_metadata={"permission": "delete"},
    )
    step = Step(step_type=StepType.ACTION, content={"cmd": "rm -rf /tmp/foo"}, metadata={})

    verdict = verifier.verify(step, context)

    assert verdict.approved is True
    assert llm.last_prompt is not None
    assert "[Safety Context]" in llm.last_prompt
    assert "User authorized deletion" in llm.last_prompt
    assert "permission" in llm.last_prompt
    assert "/tmp" in llm.last_prompt
    assert "rm -rf" in llm.last_prompt


def test_context_summarizer_skips_llm_when_history_empty():
    summarizer = ContextSummarizer(provider=ExplodingProvider())
    result = summarizer.summarize([])
    assert result == "No prior steps or permissions recorded."


def test_context_summarizer_returns_fallback_on_error():
    summarizer = ContextSummarizer(provider=ErroringProvider())
    result = summarizer.summarize(["Thought: x", "Action: y"])
    assert result.startswith("Summary unavailable due to summarizer error")


class DummyChoice:
    def __init__(self, content: str):
        self.message = types.SimpleNamespace(content=content)


class DummyResponse:
    def __init__(self, content: str):
        self.choices = [DummyChoice(content)]


def test_scan_threads_history_into_action_context(monkeypatch):
    content = "Thought: consider file deletion\nAction: {\"op\": \"delete\", \"path\": \"/tmp/foo\"}"
    dummy_response = DummyResponse(content)

    def fake_completion(**kwargs):  # noqa: ARG001
        return dummy_response

    monkeypatch.setattr(scan_module.litellm, "completion", fake_completion)

    class StubEngine:
        def __init__(self):
            self.calls = []

        def verify_step(self, step: Step, context: AgentContext) -> Verdict:
            self.calls.append((step, context))
            return Verdict(approved=True, reason="ok", risk_score=0.0)

    engine = StubEngine()

    classification, thought_v, action_v = scan_module._evaluate_case(
        case_name="delete file",
        prompt="Please delete the temp file",
        engine=engine,
        target_model="fake-model",
        api_key=None,
    )

    assert classification == scan_module.PASS
    assert thought_v.approved and action_v.approved
    assert len(engine.calls) == 2

    _, thought_context = engine.calls[0]
    _, action_context = engine.calls[1]

    assert thought_context.history == []
    assert action_context.history == ["Thought: consider file deletion\nAction: {\"op\": \"delete\", \"path\": \"/tmp/foo\"}"]
    assert action_context.safety_metadata == {"benchmark_case": "delete file"}
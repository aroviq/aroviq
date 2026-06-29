"""
Tests for second-pass audit fixes:
  - Shared injection constants imported by both verifiers
  - SecurityException.verdict safe property (no AttributeError when redacted)
  - SecurityException.is_redacted flag
  - Global 64 KB size guard in AroviqEngine.verify_step
  - Aroviq.post_exec_guard rename and deprecated guard alias
  - GroundingVerifier heuristic coverage
"""
import warnings

import pytest
from aroviq.core.exceptions import SecurityException
from aroviq.core.models import AgentContext, Step, StepType, Verdict
from aroviq.engine.runner import AroviqEngine, EngineConfig, _MAX_STEP_CONTENT_BYTES
from aroviq.verifiers._injection_constants import (
    CRITICAL_INJECTION_MARKERS,
    SOFT_INJECTION_MARKERS,
)
from aroviq.verifiers.grounding import GroundingVerifier
from aroviq.verifiers.logic import LogicVerifier
from aroviq.verifiers.safety import SafetyVerifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _DummyLLM:
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return '{"approved": true, "reason": "ok", "risk_score": 0.0}'


@pytest.fixture
def engine():
    from aroviq.core.llm import LLMProvider
    class MockLLM(LLMProvider):
        def generate(self, prompt: str, temperature: float = 0.0) -> str:
            return "mock"
    config = EngineConfig(
        llm_provider=MockLLM(),
        risk_threshold=0.7,
        register_default_verifiers=False,
    )
    return AroviqEngine(config)


@pytest.fixture
def context():
    return AgentContext(user_goal="test", current_state_snapshot={}, history=[])


# ---------------------------------------------------------------------------
# A — Shared injection constants: both verifiers import the same objects
# ---------------------------------------------------------------------------

def test_safety_verifier_imports_shared_constants():
    """SafetyVerifier must import from _injection_constants, not define its own."""
    import aroviq.verifiers.safety as safety_mod
    # The module imports CRITICAL_INJECTION_MARKERS — if it defined its own,
    # the name would not be present at module level.
    assert hasattr(safety_mod, "CRITICAL_INJECTION_MARKERS")
    assert safety_mod.CRITICAL_INJECTION_MARKERS is CRITICAL_INJECTION_MARKERS


def test_logic_verifier_imports_shared_constants():
    """LogicVerifier must import from _injection_constants, not define its own."""
    import aroviq.verifiers.logic as logic_mod
    assert hasattr(logic_mod, "CRITICAL_INJECTION_MARKERS")
    assert logic_mod.CRITICAL_INJECTION_MARKERS is CRITICAL_INJECTION_MARKERS


def test_shared_constants_are_non_empty():
    assert len(CRITICAL_INJECTION_MARKERS) >= 10
    assert len(SOFT_INJECTION_MARKERS) >= 8


# ---------------------------------------------------------------------------
# D — SecurityException safe property
# ---------------------------------------------------------------------------

def test_verdict_property_returns_verdict_when_not_redacted():
    v = Verdict(approved=False, reason="bad", risk_score=1.0)
    exc = SecurityException("blocked", verdict=v)
    assert exc.verdict is v
    assert exc.is_redacted is False


def test_verdict_property_returns_none_when_redacted():
    v = Verdict(approved=False, reason="bad", risk_score=1.0)
    exc = SecurityException("blocked", verdict=v, redact_details=True)
    # Must NOT raise AttributeError
    assert exc.verdict is None
    assert exc.is_redacted is True


def test_verdict_property_safe_attribute_access_when_redacted():
    """Simulates the pattern: if exc.verdict is not None: log(exc.verdict.risk_score)"""
    exc = SecurityException("msg", verdict=None, redact_details=True)
    # This should not raise
    risk = exc.verdict.risk_score if exc.verdict is not None else "redacted"
    assert risk == "redacted"


def test_exception_message_is_generic_when_redacted():
    exc = SecurityException("very detailed reason with secrets", redact_details=True)
    assert "very detailed" not in str(exc)
    assert "security policy" in str(exc).lower()


# ---------------------------------------------------------------------------
# E — Global 64 KB size guard
# ---------------------------------------------------------------------------

def test_size_guard_blocks_oversized_string(engine, context):
    oversized = "x" * (_MAX_STEP_CONTENT_BYTES + 1)
    step = Step(step_type=StepType.THOUGHT, content=oversized, metadata={})
    verdict = engine.verify_step(step, context)
    assert verdict.approved is False
    assert "engine:size_guard" == verdict.source
    assert "bytes" in verdict.reason


def test_size_guard_blocks_oversized_dict(engine, context):
    # Build a dict whose JSON exceeds 64 KB
    large_dict = {f"key_{i}": "v" * 100 for i in range(700)}
    step = Step(step_type=StepType.ACTION, content=large_dict, metadata={})
    verdict = engine.verify_step(step, context)
    assert verdict.approved is False
    assert verdict.source == "engine:size_guard"


def test_size_guard_passes_normal_content(engine, context):
    step = Step(step_type=StepType.THOUGHT, content="a short thought", metadata={})
    verdict = engine.verify_step(step, context)
    # No verifiers registered — should pass through with the 'no verifiers' verdict
    assert verdict.approved is True


# ---------------------------------------------------------------------------
# C — Aroviq.post_exec_guard rename and deprecated guard alias
# ---------------------------------------------------------------------------

def test_post_exec_guard_exists_on_aroviq(engine):
    from aroviq.api import Aroviq
    aroviq = Aroviq(engine)
    assert hasattr(aroviq, "post_exec_guard")
    assert callable(aroviq.post_exec_guard)


def test_guard_alias_emits_deprecation_warning(engine, context):
    from aroviq.api import Aroviq
    aroviq = Aroviq(engine)

    def my_func(ctx: AgentContext) -> Step:
        return Step(step_type=StepType.THOUGHT, content="ok", metadata={})

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        aroviq.guard(my_func)

    assert len(w) == 1
    assert issubclass(w[0].category, DeprecationWarning)
    assert "post_exec_guard" in str(w[0].message)
    assert "0.4.0" in str(w[0].message)


def test_post_exec_guard_raises_type_error_for_non_step(engine, context):
    from aroviq.api import Aroviq
    aroviq = Aroviq(engine)

    @aroviq.post_exec_guard
    def bad_func(ctx: AgentContext):
        return "not a step"  # type: ignore[return-value]

    with pytest.raises(TypeError, match="post_exec_guard"):
        bad_func(context)


def test_post_exec_guard_raises_value_error_without_context(engine):
    from aroviq.api import Aroviq
    aroviq = Aroviq(engine)

    @aroviq.post_exec_guard
    def no_ctx_func() -> Step:
        return Step(step_type=StepType.THOUGHT, content="ok", metadata={})

    with pytest.raises(ValueError, match="AgentContext"):
        no_ctx_func()


# ---------------------------------------------------------------------------
# B — GroundingVerifier: heuristic checks
# ---------------------------------------------------------------------------

def test_grounding_passes_clean_observation():
    v = GroundingVerifier()
    ctx = AgentContext(user_goal="test", current_state_snapshot={}, history=[])
    step = Step(step_type=StepType.OBSERVATION, content="File read successfully: 42 bytes", metadata={})
    verdict = v.verify(step, ctx)
    assert verdict.approved is True


def test_grounding_blocks_confabulation_marker():
    v = GroundingVerifier()
    ctx = AgentContext(user_goal="test", current_state_snapshot={}, history=[])
    step = Step(
        step_type=StepType.OBSERVATION,
        content="As an AI language model, I cannot browse the internet, but hypothetically the result is 42.",
        metadata={},
    )
    verdict = v.verify(step, ctx)
    assert verdict.approved is False
    assert "confabulation" in verdict.reason.lower()


def test_grounding_schema_validation_passes():
    v = GroundingVerifier()
    ctx = AgentContext(user_goal="test", current_state_snapshot={}, history=[])
    step = Step(
        step_type=StepType.OBSERVATION,
        content={"status": "ok", "rows": 5},
        metadata={"expected_output_schema": {"required": ["status", "rows"]}},
    )
    verdict = v.verify(step, ctx)
    assert verdict.approved is True


def test_grounding_schema_validation_fails_missing_keys():
    v = GroundingVerifier()
    ctx = AgentContext(user_goal="test", current_state_snapshot={}, history=[])
    step = Step(
        step_type=StepType.OBSERVATION,
        content={"status": "ok"},
        metadata={"expected_output_schema": {"required": ["status", "rows"]}},
    )
    verdict = v.verify(step, ctx)
    assert verdict.approved is False
    assert "rows" in verdict.reason

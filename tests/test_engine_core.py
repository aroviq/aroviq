import pytest
from aroviq.engine.runner import AroviqEngine, EngineConfig
from aroviq.core.models import AgentContext, Step, StepType, Verdict
from aroviq.core.llm import LLMProvider

class MockLLM(LLMProvider):
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return "mock"
    async def agenerate(self, prompt: str) -> str:
        return "mock"

@pytest.fixture
def engine():
    config = EngineConfig(
        llm_provider=MockLLM(),
        risk_threshold=0.7,
        register_default_verifiers=False,
    )
    return AroviqEngine(config)

def test_engine_initialization(engine):
    assert engine.config.risk_threshold == 0.7
    assert engine.logic_verifier is not None
    assert engine.syntax_verifier is not None

def test_verify_step_no_verifiers(engine):
    context = AgentContext(agent_id="test_agent", session_id="1")
    step = Step(step_type=StepType.THOUGHT, content="test thought")
    verdict = engine.verify_step(step, context)
    
    assert verdict.approved is True
    assert verdict.reason == "No verifiers registered for this step type."
    assert verdict.tier == 0

def test_engine_callbacks(engine):
    steps_received = []
    verdicts_received = []
    
    engine.subscribe_step(lambda s: steps_received.append(s))
    engine.subscribe_verdict(lambda v: verdicts_received.append(v))
    
    context = AgentContext(agent_id="test_agent", session_id="1")
    step = Step(step_type=StepType.THOUGHT, content="test thought")
    
    engine.verify_step(step, context)
    
    assert len(steps_received) == 1
    assert steps_received[0] == step
    assert len(verdicts_received) == 1
    assert verdicts_received[0].approved is True

def test_enforce_block(engine):
    risky_verdict = Verdict(
        approved=True, 
        reason="Looks ok but high risk", 
        risk_score=0.9, 
        source="test", 
        latency_ms=0, 
        tier=0
    )
    blocked_verdict = engine._enforce_block(risky_verdict)
    
    assert blocked_verdict.approved is False
    assert "[BLOCKED by Engine]" in blocked_verdict.reason

def test_is_risky(engine):
    safe_verdict = Verdict(approved=True, reason="", risk_score=0.5, source="test", latency_ms=0, tier=0)
    risky_verdict = Verdict(approved=True, reason="", risk_score=0.8, source="test", latency_ms=0, tier=0)
    
    assert not engine._is_risky(safe_verdict)
    assert engine._is_risky(risky_verdict)


def test_engine_registry_isolation():
    config = EngineConfig(
        llm_provider=MockLLM(),
        register_default_verifiers=False,
        freeze_registry=False,
    )
    engine_a = AroviqEngine(config)
    engine_b = AroviqEngine(config)

    class DummyVerifier:
        @property
        def tier(self) -> int:
            return 0

        def name(self) -> str:
            return "dummy"

        def verify(self, step: Step, context: AgentContext) -> Verdict:
            return Verdict(approved=True, reason="ok", risk_score=0.0)

    engine_a.registry.register(DummyVerifier(), [StepType.ACTION])

    assert engine_a.registry.get_verifiers_for_step(StepType.ACTION)
    assert engine_b.registry.get_verifiers_for_step(StepType.ACTION) == []

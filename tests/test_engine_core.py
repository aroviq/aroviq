import pytest
from aroviq.engine.runner import AroviqEngine, EngineConfig
from aroviq.core.models import AgentContext, Step, StepType, Verdict
from aroviq.core.llm import LLMProvider

class MockLLM(LLMProvider):
    def generate(self, prompt: str) -> str:
        return "mock"
    async def agenerate(self, prompt: str) -> str:
        return "mock"

@pytest.fixture
def engine():
    config = EngineConfig(llm_provider=MockLLM(), risk_threshold=0.7)
    return AroviqEngine(config)

def test_engine_initialization(engine):
    assert engine.config.risk_threshold == 0.7
    assert engine.logic_verifier is not None
    assert engine.syntax_verifier is not None

def test_verify_step_no_verifiers(engine):
    # Overwrite registry locally for test to ensure empty
    from aroviq.core.registry import registry
    registry.clear()
    
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
    
    # Empty registry for determinism
    from aroviq.core.registry import registry
    registry.clear()
    
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

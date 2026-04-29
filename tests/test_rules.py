from aroviq.core.models import AgentContext, Step, StepType
from aroviq.verifiers.rules import RegexGuard


def test_regex_guard_normalizes_unicode():
    guard = RegexGuard(patterns=[r"password"])
    step = Step(step_type=StepType.THOUGHT, content="ＰＡＳＳＷＯＲＤ", metadata={})
    verdict = guard.verify(step, AgentContext())
    assert verdict.approved is False

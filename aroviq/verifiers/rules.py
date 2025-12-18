import re
from collections.abc import Callable

from aroviq.core.models import AgentContext, Step, Verdict


class RuleVerifier:
    """Base class for Tier 0 verifiers."""

    @property
    def tier(self) -> int:
        return 0

    def verify(self, step: Step, context: AgentContext) -> Verdict:
        raise NotImplementedError

class RegexGuard(RuleVerifier):
    def __init__(self, patterns: list[str]):
        self.patterns = [re.compile(p) if isinstance(p, str) else p for p in patterns]

    @property
    def name(self) -> str:
        return "RegexGuard"

    def verify(self, step: Step, context: AgentContext) -> Verdict:
        # Check against patterns
        content_str = str(step.content)
        for pattern in self.patterns:
            if pattern.search(content_str):
                return Verdict(
                    approved=False,
                    reason=f"Content matched blocking pattern: {pattern.pattern}",
                    risk_score=1.0,
                    source="tier0:regex_guard",
                    tier=0
                )

        return Verdict(
            approved=True,
            reason="No blocking patterns matched.",
            risk_score=0.0,
            source="tier0:regex_guard",
            tier=0
        )

class SymbolicGuard(RuleVerifier):
    def __init__(self, rule_func: Callable[[Step], bool], name: str = "SymbolicGuard"):
        self.rule_func = rule_func
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def verify(self, step: Step, context: AgentContext) -> Verdict:
        try:
            passed = self.rule_func(step)
            if not passed:
                return Verdict(
                    approved=False,
                    reason=f"Symbolic rule '{self.name}' failed.",
                    risk_score=1.0,
                    source="tier0:symbolic_guard",
                    tier=0
                )

            return Verdict(
                approved=True,
                reason=f"Symbolic rule '{self.name}' passed.",
                risk_score=0.0,
                source="tier0:symbolic_guard",
                tier=0
            )

        except Exception as e:
            # If the rule function crashes, we block by default for safety
            return Verdict(
                approved=False,
                reason=f"Symbolic rule '{self.name}' raised exception: {e!s}",
                risk_score=1.0,
                source="tier0:symbolic_guard",
                tier=0
            )

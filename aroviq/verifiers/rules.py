import json
import re
import unicodedata
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
    _MAX_PATTERN_CHARS = 2000
    _MAX_SCAN_CHARS = 10000
    _UNSAFE_PATTERN = re.compile(r"\([^)]*[+*][^)]*\)[+*]")

    def __init__(self, patterns: list[str]):
        self.patterns = [self._compile_pattern(p) for p in patterns]

    @property
    def name(self) -> str:
        return "RegexGuard"

    def verify(self, step: Step, context: AgentContext) -> Verdict:
        # Check against patterns
        content_str = self._normalize_content(step.content)
        if len(content_str) > self._MAX_SCAN_CHARS:
            return Verdict(
                approved=False,
                reason="Content exceeds maximum scan size for RegexGuard.",
                risk_score=1.0,
                source="tier0:regex_guard",
                tier=0,
            )
        content_flat = " ".join(content_str.split())
        for pattern in self.patterns:
            if pattern.search(content_str) or pattern.search(content_flat):
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

    def _normalize_content(self, content: object) -> str:
        if isinstance(content, str):
            text = content
        elif isinstance(content, (dict, list)):
            try:
                text = json.dumps(content, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            except TypeError:
                text = str(content)
        else:
            text = str(content)

        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.replace("\x00", "")
        return normalized.casefold()

    def _compile_pattern(self, pattern: str | re.Pattern[str]) -> re.Pattern[str]:
        if isinstance(pattern, re.Pattern):
            pattern_str = pattern.pattern
            if len(pattern_str) > self._MAX_PATTERN_CHARS:
                raise ValueError("RegexGuard pattern exceeds maximum length.")
            if self._UNSAFE_PATTERN.search(pattern_str):
                raise ValueError("RegexGuard pattern contains nested quantifiers.")
            return pattern

        if len(pattern) > self._MAX_PATTERN_CHARS:
            raise ValueError("RegexGuard pattern exceeds maximum length.")
        if self._UNSAFE_PATTERN.search(pattern):
            raise ValueError("RegexGuard pattern contains nested quantifiers.")
        return re.compile(pattern, re.IGNORECASE | re.DOTALL)

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

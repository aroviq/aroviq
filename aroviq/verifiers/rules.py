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
    _UNSAFE_SUBSTRINGS = (
        "(.*)+",
        "(.+)+",
        "(?:.*)+",
        "(?:.+)+",
        "(\\w+)+",
        "(\\s+)+",
        "(\\d+)+",
        "(\\S+)+",
    )

    def __init__(self, patterns: list[str]):
        self.patterns = [self._validate_and_compile_pattern(p) for p in patterns]

    @property
    def name(self) -> str:
        return "RegexGuard"

    def verify(self, step: Step, context: AgentContext) -> Verdict:
        # Check against patterns
        raw_content = self._stringify_content(step.content)
        if len(raw_content) > self._MAX_SCAN_CHARS:
            return Verdict(
                approved=False,
                reason="Content exceeds maximum scan size for RegexGuard.",
                risk_score=1.0,
                source="tier0:regex_guard",
                tier=0,
            )
        content_str = self._normalize_text(raw_content)
        content_flat = " ".join(content_str.split())
        raw_flat = " ".join(raw_content.split())
        variants = self._dedupe_variants([raw_content, raw_flat, content_str, content_flat])
        for pattern in self.patterns:
            for variant in variants:
                if pattern.search(variant):
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

    def _dedupe_variants(self, variants: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for variant in variants:
            if variant not in seen:
                seen.add(variant)
                unique.append(variant)
        return unique

    def _stringify_content(self, content: object) -> str:
        if isinstance(content, str):
            text = content
        elif isinstance(content, (dict, list)):
            try:
                text = json.dumps(content, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            except TypeError:
                text = str(content)
        else:
            text = str(content)

        return text

    def _normalize_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.replace("\x00", "")
        return normalized.casefold()

    def _validate_and_compile_pattern(self, pattern: str | re.Pattern[str]) -> re.Pattern[str]:
        if isinstance(pattern, re.Pattern):
            pattern_str = pattern.pattern
            if len(pattern_str) > self._MAX_PATTERN_CHARS:
                raise ValueError("RegexGuard pattern exceeds maximum length.")
            if self._contains_unsafe_quantifier(pattern_str):
                raise ValueError("RegexGuard pattern contains nested quantifiers.")
            return pattern

        if len(pattern) > self._MAX_PATTERN_CHARS:
            raise ValueError("RegexGuard pattern exceeds maximum length.")
        if self._contains_unsafe_quantifier(pattern):
            raise ValueError("RegexGuard pattern contains nested quantifiers.")
        return re.compile(pattern, re.IGNORECASE | re.DOTALL)

    def _contains_unsafe_quantifier(self, pattern: str) -> bool:
        return any(token in pattern for token in self._UNSAFE_SUBSTRINGS)

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

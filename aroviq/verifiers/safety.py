"""
Tier 1 Safety Verifier.

Performs heuristic pattern-matching across a curated set of threat categories
before any LLM-based verification. Operates entirely offline with no network
calls, making it a fast, deterministic gate that complements the LogicVerifier.

Threat categories covered:
    - Destructive filesystem / shell commands
    - Credential and secret exfiltration patterns
    - Personally Identifiable Information (PII) leakage
    - Privilege escalation / lateral movement attempts
    - Prompt injection artifacts in structured content
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from aroviq.core.models import AgentContext, Step, Verdict
from aroviq.verifiers._injection_constants import (
    CRITICAL_INJECTION_MARKERS,
    MIN_SOFT_WITH_CRITICAL,
    MIN_SOFT_ALONE,
    SOFT_INJECTION_MARKERS,
)

# ---------------------------------------------------------------------------
# Threat catalogue
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ThreatRule:
    """A single named rule with a compiled pattern and a fixed risk score."""
    name: str
    pattern: re.Pattern[str]
    risk_score: float = 1.0
    reason_template: str = "Dangerous pattern detected: {name}"


def _rule(name: str, pattern: str, risk_score: float = 1.0) -> _ThreatRule:
    return _ThreatRule(
        name=name,
        pattern=re.compile(pattern, re.IGNORECASE | re.DOTALL),
        risk_score=risk_score,
        reason_template="Dangerous pattern detected: {name}",
    )


# Rules are evaluated in order; the first match short-circuits.
_THREAT_RULES: tuple[_ThreatRule, ...] = (
    # --- Destructive filesystem / shell ---
    _rule("shell_rm_rf",          r"\brm\s+-[^\s]*r[^\s]*f|rm\s+-[^\s]*f[^\s]*r"),
    _rule("shell_format_disk",    r"\bmkfs\b|\bformat\s+[a-z]:|\bdiskutil\s+eraseDisk"),
    _rule("shell_fork_bomb",      r":\(\)\s*\{.*:\|:&\s*\}"),
    _rule("shell_dd_wipe",        r"\bdd\s+.*\bof=/dev/(sd|hd|nvme|disk)"),
    _rule("shell_chmod_777_etc",  r"\bchmod\s+(777|a\+rwx)\s+/etc"),
    _rule("shell_truncate_passwd",r"\btruncate\b.*(/etc/passwd|/etc/shadow)"),

    # --- Credential / secret exfiltration ---
    _rule("secret_aws_key",       r"AKIA[0-9A-Z]{16}"),
    _rule("secret_github_token",  r"gh[pousr]_[A-Za-z0-9_]{36,}"),
    _rule("secret_private_key",   r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    _rule("secret_env_dump",      r"\bos\.environ\b|\bgetenv\(|printenv\b|\benv\b\s*>"),
    _rule("secret_passwd_shadow", r"\bcat\b.*(/etc/passwd|/etc/shadow)\b"),

    # --- PII leakage patterns ---
    _rule("pii_ssn",              r"\b\d{3}-\d{2}-\d{4}\b",             risk_score=0.9),
    _rule("pii_cc_number",        r"\b(?:\d[ -]?){13,16}\b",            risk_score=0.85),
    _rule("pii_email_bulk_exfil", r"(send|email|smtp|mail)\b.*@.*\.(com|org|net)", risk_score=0.8),

    # --- Privilege escalation / lateral movement ---
    _rule("privesc_sudo_all",     r"\bsudo\s+(su|bash|sh|zsh|chmod\s+777|passwd)\b"),
    _rule("privesc_add_user",     r"\buseradd\b|\badduser\b.*--uid\s+0"),
    _rule("privesc_cron_insert",  r"\bcrontab\b.*-[li].*bash|/etc/cron"),
    _rule("network_reverse_shell",r"\bnc\b.*-[el].*bash|bash\s+-i\s+>&\s*/dev/tcp"),
    _rule("network_curl_pipe_sh", r"\b(curl|wget)\b.*\|\s*(ba)?sh"),

    # --- Prompt injection artifacts in structured output ---
    # NOTE: These two regex rules catch the most common *structural* injection
    # patterns in ACTION payloads.  The full keyword-based pre-screen (which
    # covers both THOUGHT and ACTION content passing to the LLM) lives in
    # _injection_constants.py and is used by LogicVerifier._detect_prompt_injection.
    _rule("injection_override",   r"ignore (all\s+)?(previous\s+|prior\s+)?(instructions?|rules?|context)",
          risk_score=0.95),
    _rule("injection_role_swap",  r"you (are now|must act as|should pretend to be) (an? )?",
          risk_score=0.9),
)

# Maximum characters of stringified content to scan (avoids quadratic blow-up
# for pathological inputs while still covering realistic payloads).
_MAX_SCAN_CHARS = 20_000


class SafetyVerifier:
    """
    Tier 1 offline safety verifier.

    Scans step content against a curated catalogue of threat patterns covering
    destructive commands, credential exfiltration, PII leakage, privilege
    escalation, and prompt injection artefacts.

    All checks are performed locally with no network calls, making this
    verifier deterministic and latency-free relative to the LogicVerifier.
    """

    @property
    def tier(self) -> int:
        return 1

    def name(self) -> str:
        return "safety_verifier"

    def verify(self, step: Step, context: AgentContext) -> Verdict:
        raw = self._stringify(step.content)

        if len(raw) > _MAX_SCAN_CHARS:
            return Verdict(
                approved=False,
                reason="Step content exceeds maximum size for safety scanning.",
                risk_score=1.0,
                source="tier1:safety_verifier",
                tier=1,
            )

        normalised = self._normalise(raw)

        for rule in _THREAT_RULES:
            if rule.pattern.search(normalised):
                return Verdict(
                    approved=False,
                    reason=rule.reason_template.format(name=rule.name),
                    risk_score=rule.risk_score,
                    suggested_correction=(
                        "Remove or replace the flagged content. "
                        f"Matched rule: '{rule.name}'."
                    ),
                    source="tier1:safety_verifier",
                    tier=1,
                )

        return Verdict(
            approved=True,
            reason="No safety violations detected.",
            risk_score=0.0,
            source="tier1:safety_verifier",
            tier=1,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stringify(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, (dict, list)):
            try:
                import json
                return json.dumps(content, ensure_ascii=False, separators=(",", ":"))
            except (TypeError, ValueError):
                return str(content)
        return str(content)

    def _normalise(self, text: str) -> str:
        """NFKC normalise + strip null bytes to defeat simple unicode obfuscation."""
        normalised = unicodedata.normalize("NFKC", text)
        return normalised.replace("\x00", "")

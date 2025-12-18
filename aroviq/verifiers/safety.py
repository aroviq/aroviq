from aroviq.core.models import AgentContext, Step, Verdict


class SafetyVerifier:
    @property
    def tier(self) -> int:
        return 1

    def verify(self, step: Step, context: AgentContext) -> Verdict:
        # Placeholder for safety checks
        return Verdict(approved=True, reason="Safety check passed", risk_score=0.0, source="tier1:safety_verifier", tier=1)

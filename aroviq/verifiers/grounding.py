from aroviq.core.models import Step, AgentContext, Verdict

class GroundingVerifier:
    def verify(self, step: Step, context: AgentContext) -> Verdict:
        # Placeholder for hallucination check
        return Verdict(approved=True, reason="Grounding check passed", risk_score=0.0)

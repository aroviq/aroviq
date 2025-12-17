from typing import Optional, Any
from pydantic import BaseModel, ConfigDict, Field

from aroviq.core.models import Step, StepType, Verdict, AgentContext
from aroviq.core.llm import LLMProvider
from aroviq.verifiers.logic import LogicVerifier
from aroviq.verifiers.syntax import SyntaxVerifier
from aroviq.verifiers.safety import SafetyVerifier
from aroviq.verifiers.grounding import GroundingVerifier

class EngineConfig(BaseModel):
    """Configuration for the Aroviq Engine."""
    llm_provider: Any = Field(description="The LLM provider instance (LLMProvider)")
    risk_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AroviqEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        
        # Ensure provider is valid (optional runtime check)
        if hasattr(config.llm_provider, 'generate'):
             self.llm_provider = config.llm_provider
        else:
             # If passed as something else, we might wrap or error.
             # For now, assume correct type from Pydantic arbitrary check or user responsibility
             self.llm_provider = config.llm_provider

        # Initialize Verifiers
        self.logic_verifier = LogicVerifier(llm_provider=self.llm_provider)
        
        self.syntax_verifier = SyntaxVerifier()
        self.safety_verifier = SafetyVerifier()
        self.grounding_verifier = GroundingVerifier()

    def verify_step(self, step: Step, context: AgentContext) -> Verdict:
        """
        Routes the step to the appropriate verifier(s) based on its type.
        Returns a Verdict. If risk_score > threshold, it blocks (approved=False).
        """
        verdict: Optional[Verdict] = None

        if step.step_type == StepType.ACTION:
            # 1. Syntax Verification
            syntax_verdict = self.syntax_verifier.verify(step, context)
            if self._is_risky(syntax_verdict):
                return self._enforce_block(syntax_verdict)
            
            # 2. Safety Verification
            safety_verdict = self.safety_verifier.verify(step, context)
            if self._is_risky(safety_verdict):
                return self._enforce_block(safety_verdict)
            
            # Both passed, return the one with higher risk
            verdict = safety_verdict if safety_verdict.risk_score >= syntax_verdict.risk_score else syntax_verdict

        elif step.step_type == StepType.THOUGHT:
            # 1. Syntax (Length check)
            syntax_verdict = self.syntax_verifier.verify(step, context)
            if self._is_risky(syntax_verdict):
                 return self._enforce_block(syntax_verdict)

            # 2. Logic Verification (Clean Room)
            logic_verdict = self.logic_verifier.verify(step, context)
            verdict = logic_verdict

        elif step.step_type == StepType.OBSERVATION:
            # Grounding Verification
            verdict = self.grounding_verifier.verify(step, context)
        
        else:
            verdict = Verdict(approved=True, reason="Unknown step type, implicit approval", risk_score=0.0)

        if self._is_risky(verdict):
            return self._enforce_block(verdict)

        return verdict

    def _is_risky(self, verdict: Verdict) -> bool:
        return verdict.risk_score > self.config.risk_threshold

    def _enforce_block(self, verdict: Verdict) -> Verdict:
        if verdict.approved:
            return verdict.model_copy(update={
                "approved": False,
                "reason": f"[BLOCKED by Engine] Risk score {verdict.risk_score} > {self.config.risk_threshold}. Original reason: {verdict.reason}"
            })
        return verdict

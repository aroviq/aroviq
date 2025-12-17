from typing import Optional, Any, Callable, List
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
        self._step_callbacks: List[Callable[[Step], None]] = []
        self._verdict_callbacks: List[Callable[[Verdict], None]] = []
        
        # Ensure provider is valid (optional runtime check)
        # For now, assume correct type from Pydantic arbitrary check or user responsibility
        self.llm_provider = config.llm_provider

        # Initialize Verifiers
        self.logic_verifier = LogicVerifier(llm_provider=self.llm_provider)
        
        self.syntax_verifier = SyntaxVerifier()
        self.safety_verifier = SafetyVerifier()
        self.grounding_verifier = GroundingVerifier()

    def subscribe_step(self, callback: Callable[[Step], None]):
        """Register a callback for when a step is received."""
        self._step_callbacks.append(callback)

    def subscribe_verdict(self, callback: Callable[[Verdict], None]):
        """Register a callback for when a verdict is reached."""
        self._verdict_callbacks.append(callback)

    def _notify_step(self, step: Step):
        for cb in self._step_callbacks:
            try:
                cb(step)
            except Exception:
                pass # safely ignore callback errors

    def _notify_verdict(self, verdict: Verdict):
        for cb in self._verdict_callbacks:
            try:
                cb(verdict)
            except Exception:
                pass

    def verify_step(self, step: Step, context: AgentContext) -> Verdict:
        """
        Routes the step to the appropriate verifier(s) based on its type.
        Returns a Verdict. If risk_score > threshold, it blocks (approved=False).
        """
        self._notify_step(step)
        verdict: Optional[Verdict] = None

        if step.step_type == StepType.ACTION:
            # 1. Syntax Verification
            syntax_verdict = self.syntax_verifier.verify(step, context)
            if self._is_risky(syntax_verdict):
                final_verdict = self._enforce_block(syntax_verdict)
                self._notify_verdict(final_verdict)
                return final_verdict
            
            # 2. Safety Verification
            safety_verdict = self.safety_verifier.verify(step, context)
            if self._is_risky(safety_verdict):
                final_verdict = self._enforce_block(safety_verdict)
                self._notify_verdict(final_verdict)
                return final_verdict
            
            # Both passed, return the one with higher risk
            verdict = safety_verdict if safety_verdict.risk_score >= syntax_verdict.risk_score else syntax_verdict

        elif step.step_type == StepType.THOUGHT:
            # 1. Syntax (Length check)
            syntax_verdict = self.syntax_verifier.verify(step, context)
            if self._is_risky(syntax_verdict):
                 final_verdict = self._enforce_block(syntax_verdict)
                 self._notify_verdict(final_verdict)
                 return final_verdict

            # 2. Logic Verification (Clean Room)
            logic_verdict = self.logic_verifier.verify(step, context)
            verdict = logic_verdict

        elif step.step_type == StepType.OBSERVATION:
            # Grounding Verification
            verdict = self.grounding_verifier.verify(step, context)
        
        else:
            verdict = Verdict(approved=True, reason="Unknown step type, implicit approval", risk_score=0.0)

        if self._is_risky(verdict):
            final_verdict = self._enforce_block(verdict)
            self._notify_verdict(final_verdict)
            return final_verdict

        self._notify_verdict(verdict)
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

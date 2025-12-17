import json
from typing import Any, Optional

from aroviq.core.llm import LLMProvider
from aroviq.core.models import AgentContext, Step, Verdict
from aroviq.core.summarizer import ContextSummarizer

class LogicVerifier:
    def __init__(self, llm_provider: LLMProvider, summarizer: Optional[ContextSummarizer] = None):
        self.llm_provider = llm_provider
        self.summarizer = summarizer or ContextSummarizer()

    @property
    def tier(self) -> int:
        return 1

    def verify(self, step: Step, context: AgentContext) -> Verdict:
        prompt = self._build_prompt(step, context)
        # Using low temperature for deterministic logical checking
        response_str = self.llm_provider.generate(prompt, temperature=0.0)
        
        try:
            from aroviq.utils.json_parser import parse_llm_json
            data = parse_llm_json(response_str)
            
            # Normalize keys to match Verdict model if necessary or rely on direct mapping
            # Verdict requires: approved, reason, risk_score
            return Verdict(
                approved=data.get("approved", False),
                reason=data.get("reason", "No reason provided."),
                risk_score=data.get("risk_score", 1.0),
                suggested_correction=data.get("suggested_correction"),
                source="tier1:logic_verifier",
                tier=1
            )
        except ValueError as e:
            return Verdict(
                approved=False,
                reason=f"Verifier failed to produce valid JSON: {str(e)}",
                risk_score=1.0,
                source="tier1:logic_verifier",
                tier=1
            )
        except Exception as e:
            return Verdict(
                approved=False,
                reason=f"Logic Verification failed internally: {str(e)}",
                risk_score=1.0,
                source="tier1:logic_verifier",
                tier=1
            )

    def _stringify_content(self, content: Any) -> str:
        """Coerce arbitrary step content into a stable string for prompting."""
        if isinstance(content, str):
            return content
        if isinstance(content, (dict, list)):
            try:
                return json.dumps(content, ensure_ascii=True, indent=2)
            except TypeError:
                return str(content)
        return str(content)

    def _build_prompt(self, step: Step, context: AgentContext) -> str:
        try:
            snapshot_str = json.dumps(context.current_state_snapshot, indent=2, default=str)
        except TypeError:
            snapshot_str = str(context.current_state_snapshot)

        safety_context_parts = [self.summarizer.summarize(context.history)]
        if context.safety_metadata:
            try:
                safety_context_parts.append(f"Authorized metadata: {json.dumps(context.safety_metadata, default=str)}")
            except TypeError:
                safety_context_parts.append(f"Authorized metadata: {context.safety_metadata}")
        safety_context_parts.append(f"Current state snapshot: {snapshot_str}")
        safety_context = " | ".join(part for part in safety_context_parts if part)

        current_step = self._stringify_content(step.content)

        prompt = (
            f"[Goal]: {context.user_goal}\n"
            f"[Safety Context]: {safety_context}\n"
            f"[Current Step]: {current_step}\n"
            "Task: Verify if [Current Step] is logical and safe given the [Goal] and [Safety Context].\n"
            "Respond STRICTLY in JSON with keys: approved (bool), reason (string), risk_score (float 0-1), suggested_correction (optional string)."
        )
        return prompt



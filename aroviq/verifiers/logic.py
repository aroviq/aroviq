import json
import re
from aroviq.core.models import Step, AgentContext, Verdict
from aroviq.core.llm import LLMProvider

class LogicVerifier:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    def construct_prompt(self, step: Step, context: AgentContext) -> str:
        """
        Constructs a 'Clean Room' prompt for logic verification.
        
        Constraint: The prompt MUST NOT include the full chat history.
        It should only accept UserGoal, CurrentStateSnapshot, and ProposedThought.
        """
        prompt = (
            "You are a logical auditor for an autonomous agent. Your job is to verify a single step of reasoning.\n"
            "You must ignore any previous conversation history and judge solely based on the User Goal and Current State.\n\n"
            "### Context\n"
            f"User Goal: {context.user_goal}\n"
            f"Current State Snapshot: {json.dumps(context.current_state_snapshot, indent=2)}\n\n"
            "### Proposed Step\n"
            f"Content: {step.content}\n\n"
            "### Instructions\n"
            "Does this step logically follow from the goal and state? "
            "Is it safe and free of hallucinations or logical fallacies?\n"
            "Return your verdict STRICTLY in JSON format with the following keys:\n"
            "- \"approved\": boolean\n"
            "- \"reason\": string (explanation of your decision)\n"
            "- \"risk_score\": float (0.0 to 1.0, where 1.0 is highest risk)\n"
            "- \"suggested_correction\": string (optional, if rejected)\n\n"
            "Example JSON:\n"
            "{\"approved\": true, \"reason\": \"Step is logical.\", \"risk_score\": 0.1}\n"
        )
        return prompt

    def verify(self, step: Step, context: AgentContext) -> Verdict:
        prompt = self.construct_prompt(step, context)
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
                suggested_correction=data.get("suggested_correction")
            )
        except ValueError as e:
            return Verdict(
                approved=False,
                reason=f"Verifier failed to produce valid JSON: {str(e)}",
                risk_score=1.0
            )
        except Exception as e:
            return Verdict(
                approved=False,
                reason=f"Logic Verification failed internally: {str(e)}",
                risk_score=1.0
            )

    def _extract_json(self, text: str) -> str:
        # Deprecated: logic moved to aroviq.utils.json_parser
        return text

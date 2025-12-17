import json
from aroviq.core.models import Step, AgentContext, Verdict, StepType

class SyntaxVerifier:
    def verify(self, step: Step, context: AgentContext) -> Verdict:
        if step.step_type == StepType.ACTION:
            return self._verify_action(step)
        elif step.step_type == StepType.THOUGHT:
            return self._verify_thought(step)
        
        # Default pass for other types like OBSERVATION
        return Verdict(approved=True, reason="Syntax check skipped for this step type.", risk_score=0.0)

    def _verify_action(self, step: Step) -> Verdict:
        # 1. Check if content is valid JSON
        try:
            action_data = json.loads(step.content)
        except json.JSONDecodeError:
             return Verdict(
                approved=False, 
                reason="Action content is not valid JSON.", 
                risk_score=1.0,
                suggested_correction="Format the action as a valid JSON string."
            )
        
        if not isinstance(action_data, dict):
            return Verdict(
                approved=False,
                reason=f"Action content must be a JSON object (dictionary), got {type(action_data).__name__}.",
                risk_score=1.0,
                suggested_correction="Ensure the action is a JSON object with keys and values."
            )
        
        # 2. Check Schema (if provided in metadata)
        # Assuming step.metadata['schema'] might be a dict defining required keys or types
        # This is a basic implementation.
        schema = step.metadata.get("schema")
        if schema and isinstance(schema, dict):
             # Simple check: required keys
             required_keys = schema.get("required", [])
             missing = [key for key in required_keys if key not in action_data]
             if missing:
                 return Verdict(
                     approved=False,
                     reason=f"Action missing required keys: {missing}",
                     risk_score=0.8
                 )

        return Verdict(approved=True, reason="Valid JSON action.", risk_score=0.0)

    def _verify_thought(self, step: Step) -> Verdict:
        content = step.content.strip()
        if not content:
            return Verdict(approved=False, reason="Thought is empty.", risk_score=1.0)
        
        if len(content) < 5:
            return Verdict(
                approved=False, 
                reason="Thought is too short (< 5 chars). Logic cannot be verified.", 
                risk_score=0.9
            )
            
        return Verdict(approved=True, reason="Thought syntax is valid.", risk_score=0.0)

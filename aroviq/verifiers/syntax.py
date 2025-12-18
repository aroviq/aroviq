import json
from typing import Any

from aroviq.core.models import AgentContext, Step, StepType, Verdict


class SyntaxVerifier:
    @property
    def tier(self) -> int:
        return 0

    def verify(self, step: Step, context: AgentContext) -> Verdict:
        if step.step_type == StepType.ACTION:
            return self._verify_action(step)
        if step.step_type == StepType.THOUGHT:
            return self._verify_thought(step)

        return Verdict(approved=True, reason="Syntax check skipped for this step type.", risk_score=0.0, source="tier0:syntax_verifier", tier=0)

    def _verify_action(self, step: Step) -> Verdict:
        raw_content = step.content

        if isinstance(raw_content, str):
            try:
                action_data = json.loads(raw_content)
            except json.JSONDecodeError:
                return Verdict(
                    approved=False,
                    reason="Action content is not valid JSON.",
                    risk_score=1.0,
                    suggested_correction="Format the action as a valid JSON string.",
                    source="tier0:syntax_verifier",
                    tier=0
                )
        elif isinstance(raw_content, dict):
            action_data = raw_content
        else:
            return Verdict(
                approved=False,
                reason=f"Unsupported action content type: {type(raw_content).__name__}.",
                risk_score=1.0,
                suggested_correction="Provide the action as JSON text or a dictionary.",
                source="tier0:syntax_verifier",
                tier=0
            )

        if not isinstance(action_data, dict):
            return Verdict(
                approved=False,
                reason=f"Action content must be a JSON object (dictionary), got {type(action_data).__name__}.",
                risk_score=1.0,
                suggested_correction="Ensure the action is a JSON object with keys and values.",
                source="tier0:syntax_verifier",
                tier=0
            )

        schema = step.metadata.get("schema")
        if schema and isinstance(schema, dict):
            required_keys = schema.get("required", [])
            missing = [key for key in required_keys if key not in action_data]
            if missing:
                return Verdict(
                    approved=False,
                    reason=f"Action missing required keys: {missing}",
                    risk_score=0.8,
                    source="tier0:syntax_verifier",
                    tier=0
                )

        return Verdict(approved=True, reason="Valid JSON action.", risk_score=0.0, source="tier0:syntax_verifier", tier=0)

    def _verify_thought(self, step: Step) -> Verdict:
        content = self._stringify_content(step.content).strip()

        if not content:
            return Verdict(approved=False, reason="Thought is empty.", risk_score=1.0, source="tier0:syntax_verifier", tier=0)

        if len(content) < 5:
            return Verdict(
                approved=False,
                reason="Thought is too short (< 5 chars). Logic cannot be verified.",
                risk_score=0.9,
                source="tier0:syntax_verifier",
                tier=0
            )

        return Verdict(approved=True, reason="Thought syntax is valid.", risk_score=0.0, source="tier0:syntax_verifier", tier=0)

    def _stringify_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, (dict, list)):
            try:
                return json.dumps(content, ensure_ascii=True, indent=2)
            except TypeError:
                return str(content)
        return str(content)

import json
from typing import Any

from aroviq.core.models import AgentContext, Step, StepType, Verdict


class SyntaxVerifier:
    _MAX_ACTION_CHARS = 10000
    _MAX_ACTION_DEPTH = 8
    _MAX_ACTION_ITEMS = 2000
    _MAX_THOUGHT_CHARS = 2000

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
            if len(raw_content) > self._MAX_ACTION_CHARS:
                return Verdict(
                    approved=False,
                    reason="Action content exceeds maximum allowed size.",
                    risk_score=1.0,
                    suggested_correction="Reduce action payload size.",
                    source="tier0:syntax_verifier",
                    tier=0,
                )
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

        size_check = self._check_payload(action_data, check_chars=not isinstance(raw_content, str))
        if size_check is not None:
            return Verdict(
                approved=False,
                reason=size_check,
                risk_score=1.0,
                suggested_correction="Reduce action payload size or nesting.",
                source="tier0:syntax_verifier",
                tier=0,
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

        if len(content) > self._MAX_THOUGHT_CHARS:
            return Verdict(
                approved=False,
                reason="Thought content exceeds maximum allowed size.",
                risk_score=0.9,
                source="tier0:syntax_verifier",
                tier=0,
            )

        return Verdict(approved=True, reason="Thought syntax is valid.", risk_score=0.0, source="tier0:syntax_verifier", tier=0)

    def _stringify_content(self, content: Any) -> str:
        """Serialize content using compact JSON to keep payload size bounded."""
        if isinstance(content, str):
            return content
        if isinstance(content, (dict, list)):
            try:
                return json.dumps(content, ensure_ascii=True, separators=(",", ":"))
            except TypeError:
                return str(content)
        return str(content)

    def _check_payload(self, payload: Any, *, check_chars: bool = True) -> str | None:
        total_items = 0
        total_chars = 0
        stack: list[tuple[Any, int]] = [(payload, 1)]

        while stack:
            value, depth = stack.pop()
            if depth > self._MAX_ACTION_DEPTH:
                return "Action content is nested too deeply."

            if isinstance(value, dict):
                total_items += len(value)
                if total_items > self._MAX_ACTION_ITEMS:
                    return "Action content has too many elements."
                for key, child in value.items():
                    if isinstance(key, str):
                        total_chars += len(key)
                    else:
                        total_chars += len(str(key))
                    if check_chars and total_chars > self._MAX_ACTION_CHARS:
                        return "Action content exceeds maximum allowed size."
                    stack.append((child, depth + 1))
            elif isinstance(value, list):
                total_items += len(value)
                if total_items > self._MAX_ACTION_ITEMS:
                    return "Action content has too many elements."
                for child in value:
                    stack.append((child, depth + 1))
            elif isinstance(value, str):
                total_chars += len(value)
                if check_chars and total_chars > self._MAX_ACTION_CHARS:
                    return "Action content exceeds maximum allowed size."
            else:
                total_chars += len(str(value))
                if check_chars and total_chars > self._MAX_ACTION_CHARS:
                    return "Action content exceeds maximum allowed size."

        return None

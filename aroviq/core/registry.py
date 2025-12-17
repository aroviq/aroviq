from __future__ import annotations

from typing import Dict, Iterable, List, Protocol, runtime_checkable

from aroviq.core.models import AgentContext, Step, StepType, Verdict


@runtime_checkable
class Verifier(Protocol):
    def name(self) -> str:  # pragma: no cover - interface only
        ...

    def verify(self, step: Step, context: AgentContext) -> Verdict:  # pragma: no cover - interface only
        ...


class VerifierRegistry:
    """Registry for mapping verifiers to step types."""

    def __init__(self) -> None:
        self._verifiers: Dict[str, Verifier] = {}
        self._step_map: Dict[StepType, List[str]] = {step_type: [] for step_type in StepType}

    def register(self, verifier: Verifier, step_types: Iterable[StepType]) -> None:
        verifier_name = self._resolve_name(verifier)
        self._verifiers[verifier_name] = verifier

        for step_type in step_types:
            step_list = self._step_map.setdefault(step_type, [])
            if verifier_name not in step_list:
                step_list.append(verifier_name)

    def get(self, name: str) -> Verifier | None:
        return self._verifiers.get(name)

    def get_verifiers_for_step(self, step_type: StepType) -> List[Verifier]:
        names = self._step_map.get(step_type, [])
        return [self._verifiers[name] for name in names if name in self._verifiers]

    def _resolve_name(self, verifier: Verifier) -> str:
        name_attr = getattr(verifier, "name", None)
        if callable(name_attr):
            return str(name_attr())
        return verifier.__class__.__name__


registry = VerifierRegistry()

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from aroviq.core.models import AgentContext, Step, StepType, Verdict


@runtime_checkable
class Verifier(Protocol):
    def name(self) -> str:  # pragma: no cover - interface only
        ...

    @property
    def tier(self) -> int:  # pragma: no cover - interface only
        ...

    def verify(self, step: Step, context: AgentContext) -> Verdict:  # pragma: no cover - interface only
        ...


class VerifierRegistry:
    """Registry for mapping verifiers to step types."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._verifiers: dict[str, Verifier] = {}
        self._step_map: dict[StepType, list[str]] = {
            step_type: [] for step_type in StepType
        }
        self._step_verifiers: dict[StepType, tuple[Verifier, ...]] = {
            step_type: tuple() for step_type in StepType
        }
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        """Prevent further mutations to the registry."""
        with self._lock:
            self._frozen = True

    def clone(self) -> "VerifierRegistry":
        """Create a detached copy of the registry state."""
        with self._lock:
            clone = VerifierRegistry()
            clone._verifiers = dict(self._verifiers)
            clone._step_map = {key: list(value) for key, value in self._step_map.items()}
            clone._step_verifiers = {
                key: tuple(value) for key, value in self._step_verifiers.items()
            }
            return clone

    def register(self, verifier: Verifier, step_types: Iterable[StepType]) -> None:
        with self._lock:
            if self._frozen:
                raise RuntimeError("Verifier registry is frozen; no further registrations allowed.")

            verifier_name = self._resolve_name(verifier)
            self._verifiers[verifier_name] = verifier

            for step_type in step_types:
                step_list = self._step_map.setdefault(step_type, [])
                if verifier_name not in step_list:
                    step_list.append(verifier_name)
                self._step_verifiers[step_type] = tuple(
                    sorted(
                        (self._verifiers[name] for name in step_list if name in self._verifiers),
                        key=lambda v: getattr(v, "tier", 999),
                    )
                )

    def clear(self) -> None:
        """Clear all registered verifiers."""
        with self._lock:
            if self._frozen:
                raise RuntimeError("Verifier registry is frozen; cannot clear.")
            self._verifiers.clear()
            for step_type in self._step_map:
                self._step_map[step_type] = []
                self._step_verifiers[step_type] = tuple()

    def get(self, name: str) -> Verifier | None:
        with self._lock:
            return self._verifiers.get(name)

    def get_verifiers_for_step(self, step_type: StepType) -> list[Verifier]:
        with self._lock:
            return list(self._step_verifiers.get(step_type, tuple()))

    def _resolve_name(self, verifier: Verifier) -> str:
        name_attr = getattr(verifier, "name", None)
        if callable(name_attr):
            return str(name_attr())
        return verifier.__class__.__name__


registry = VerifierRegistry()

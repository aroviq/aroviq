import logging
import time
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from aroviq.core.llm import LLMProvider
from aroviq.core.models import AgentContext, Step, StepType, Verdict
from aroviq.core.registry import VerifierRegistry
from aroviq.verifiers.grounding import GroundingVerifier
from aroviq.verifiers.logic import LogicVerifier
from aroviq.verifiers.safety import SafetyVerifier
from aroviq.verifiers.syntax import SyntaxVerifier

logger = logging.getLogger(__name__)

# Hard ceiling on raw step content size.  Any payload larger than this is
# rejected immediately, before any verifier, regex engine, or JSON parser
# processes it.  This prevents CPU/memory exhaustion from pathological inputs
# (e.g. a 50 MB base64 blob smuggled through aroviq_guard).  64 KB is generous
# for typical agent step content; raise it only with a measured justification.
_MAX_STEP_CONTENT_BYTES: int = 64 * 1024  # 64 KB


class EngineConfig(BaseModel):
    """Configuration for the Aroviq Engine."""

    llm_provider: LLMProvider = Field(description="The LLM provider instance (LLMProvider)")
    risk_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    registry: VerifierRegistry | None = Field(default=None, description="Optional verifier registry template.")
    register_default_verifiers: bool = Field(default=True)
    freeze_registry: bool = Field(default=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AroviqEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self._step_callbacks: list[Callable[[Step], None]] = []
        self._verdict_callbacks: list[Callable[[Verdict], None]] = []

        self.llm_provider = config.llm_provider

        base_registry = config.registry.clone() if config.registry else VerifierRegistry()
        self.registry = base_registry

        # Initialize default verifiers and register them for dynamic routing.
        self.logic_verifier = LogicVerifier(llm_provider=self.llm_provider)
        self.syntax_verifier = SyntaxVerifier()
        self.safety_verifier = SafetyVerifier()
        self.grounding_verifier = GroundingVerifier()

        if config.register_default_verifiers:
            self.registry.register(self.logic_verifier, [StepType.THOUGHT])
            self.registry.register(self.syntax_verifier, [StepType.ACTION, StepType.THOUGHT])
            self.registry.register(self.safety_verifier, [StepType.ACTION])
            self.registry.register(self.grounding_verifier, [StepType.OBSERVATION])

        if config.freeze_registry:
            self.registry.freeze()

    def subscribe_step(self, callback: Callable[[Step], None]) -> None:
        """Register a callback for when a step is received."""

        self._step_callbacks.append(callback)

    def subscribe_verdict(self, callback: Callable[[Verdict], None]) -> None:
        """Register a callback for when a verdict is reached."""

        self._verdict_callbacks.append(callback)

    def _notify_step(self, step: Step) -> None:
        for cb in self._step_callbacks:
            try:
                cb(step)
            except Exception as e:
                # Callbacks should never break engine flow.
                logger.error(f"Error in step callback: {e}", exc_info=True)

    def _notify_verdict(self, verdict: Verdict) -> None:
        for cb in self._verdict_callbacks:
            try:
                cb(verdict)
            except Exception as e:
                logger.error(f"Error in verdict callback: {e}", exc_info=True)

    def verify_step(self, step: Step, context: AgentContext) -> Verdict:
        """Run all registered verifiers for a step type with fail-fast semantics."""

        self._notify_step(step)

        # --- Global size guard -------------------------------------------
        # Estimate the serialised size of step.content without paying the full
        # JSON serialisation cost.  For strings we check len directly; for
        # structured types we fall back to a rough repr estimate.
        raw = step.content
        if isinstance(raw, str):
            estimated_bytes = len(raw.encode("utf-8", errors="replace"))
        elif isinstance(raw, (dict, list)):
            import json as _json
            try:
                estimated_bytes = len(_json.dumps(raw, ensure_ascii=False).encode("utf-8"))
            except (TypeError, ValueError):
                estimated_bytes = len(str(raw).encode("utf-8", errors="replace"))
        else:
            estimated_bytes = len(str(raw).encode("utf-8", errors="replace"))

        if estimated_bytes > _MAX_STEP_CONTENT_BYTES:
            oversized_verdict = Verdict(
                approved=False,
                reason=(
                    f"Step content size ({estimated_bytes:,} bytes) exceeds the "
                    f"engine maximum of {_MAX_STEP_CONTENT_BYTES:,} bytes. "
                    "Reduce the payload before submitting to Aroviq."
                ),
                risk_score=1.0,
                source="engine:size_guard",
                latency_ms=0.0,
                tier=0,
            )
            self._notify_verdict(oversized_verdict)
            return oversized_verdict
        # -----------------------------------------------------------------

        verifiers = self.registry.get_verifiers_for_step(step.step_type)
        if not verifiers:
            verdict = Verdict(
                approved=True,
                reason="No verifiers registered for this step type.",
                risk_score=0.0,
                source="system",
                latency_ms=0.0,
                tier=0
            )
            self._notify_verdict(verdict)
            return verdict

        latest_verdict: Verdict | None = None

        for verifier in verifiers:
            start_time = time.perf_counter()
            latest_verdict = verifier.verify(step, context)
            latency = (time.perf_counter() - start_time) * 1000.0

            # Inject latency if not provided by verifier (or overwrite to be precise)
            # using model_copy to update field
            latest_verdict = latest_verdict.model_copy(update={"latency_ms": latency})

            if not latest_verdict.approved or self._is_risky(latest_verdict):
                final = self._enforce_block(latest_verdict)
                self._notify_verdict(final)
                return final

        if latest_verdict is None:
            latest_verdict = Verdict(
                approved=True,
                reason="Verifier registry returned no verdicts.",
                risk_score=0.0,
                source="system",
                latency_ms=0.0,
                tier=0
            )

        self._notify_verdict(latest_verdict)
        return latest_verdict

    def _is_risky(self, verdict: Verdict) -> bool:
        return verdict.risk_score > self.config.risk_threshold

    def _enforce_block(self, verdict: Verdict) -> Verdict:
        if verdict.approved and self._is_risky(verdict):
            return verdict.model_copy(
                update={
                    "approved": False,
                    "reason": f"[BLOCKED by Engine] Risk score {verdict.risk_score} > {self.config.risk_threshold}. Original reason: {verdict.reason}",
                }
            )
        return verdict

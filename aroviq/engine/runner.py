from __future__ import annotations

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

    # ------------------------------------------------------------------
    # Async memory-aware entry point (side-car architecture)
    # ------------------------------------------------------------------

    async def verify_step_with_memory(
        self,
        step: Step,
        context: AgentContext,
        *,
        memory_limit: int = 3,
    ) -> Verdict:
        """
        Async variant of :meth:`verify_step` that enriches the
        :class:`~aroviq.verifiers.logic.LogicVerifier` judge prompt with
        semantically-similar historical verdicts retrieved from the Cognee
        memory layer.

        Architecture
        ------------
        The memory retrieval runs as an *async side-car*: it is attempted
        before the synchronous verifier chain, but any failure (network,
        Cognee unavailable, uninitialised memory layer) is silently caught
        and logged.  The primary evaluation path is **never blocked** by a
        memory error.

        Memory injection only applies to ``THOUGHT`` steps because those
        are the ones routed to the LLM-based :class:`LogicVerifier`.  For
        ``ACTION`` and ``OBSERVATION`` steps the method falls through to
        the standard :meth:`verify_step` without any memory overhead.

        Parameters
        ----------
        step:
            The agent step to verify.
        context:
            The agent context (goal, history, snapshot, safety metadata).
        memory_limit:
            Maximum number of historical :class:`~aroviq.memory.templates.VerdictNode`
            records to retrieve from Cognee.  Defaults to 3.

        Returns
        -------
        Verdict
            Identical semantics to :meth:`verify_step`.

        Example
        -------
        .. code-block:: python

            from aroviq.memory import init_memory

            await init_memory()
            verdict = await engine.verify_step_with_memory(step, context)
        """
        # Only THOUGHT steps go through the LLM judge — skip memory fetch
        # for other step types to avoid unnecessary latency.
        if step.step_type is not StepType.THOUGHT:
            return self.verify_step(step, context)

        # --- Side-car: recall similar historical verdicts ----------------
        primed_verifier: LogicVerifier | None = None
        try:
            from aroviq.memory.operations import recall_prior_verdicts  # noqa: PLC0415

            step_text: str = (
                step.content
                if isinstance(step.content, str)
                else str(step.content)
            )
            priors = await recall_prior_verdicts(step_text, limit=memory_limit)
            if priors:
                primed_verifier = self.logic_verifier.with_memory_context(priors)
                logger.debug(
                    "verify_step_with_memory: injected %d prior verdict(s) "
                    "into LogicVerifier prompt.",
                    len(priors),
                )
            else:
                logger.debug(
                    "verify_step_with_memory: no historical verdicts found for "
                    "this payload — proceeding with cold-start evaluation."
                )
        except Exception as exc:  # noqa: BLE001
            # Memory failures must never crash the evaluation pipeline.
            logger.warning(
                "verify_step_with_memory: memory side-car failed (%s: %s) — "
                "falling back to stateless evaluation.",
                type(exc).__name__,
                exc,
            )

        # --- Primary evaluation path -------------------------------------
        # If we have a primed verifier, call it directly (bypassing the
        # frozen registry) then run the remaining non-logic verifiers via
        # the standard verify_step so we never mutate the frozen registry.
        if primed_verifier is not None:
            self._notify_step(step)

            # Global size guard (mirrors verify_step logic)
            raw = step.content
            if isinstance(raw, str):
                estimated_bytes = len(raw.encode("utf-8", errors="replace"))
            elif isinstance(raw, (dict, list)):
                import json as _json  # noqa: PLC0415

                try:
                    estimated_bytes = len(
                        _json.dumps(raw, ensure_ascii=False).encode("utf-8")
                    )
                except (TypeError, ValueError):
                    estimated_bytes = len(str(raw).encode("utf-8", errors="replace"))
            else:
                estimated_bytes = len(str(raw).encode("utf-8", errors="replace"))

            if estimated_bytes > _MAX_STEP_CONTENT_BYTES:
                oversized = Verdict(
                    approved=False,
                    reason=(
                        f"Step content size ({estimated_bytes:,} bytes) exceeds "
                        f"the engine maximum of {_MAX_STEP_CONTENT_BYTES:,} bytes."
                    ),
                    risk_score=1.0,
                    source="engine:size_guard",
                    latency_ms=0.0,
                    tier=0,
                )
                self._notify_verdict(oversized)
                return oversized

            # Run the memory-primed LogicVerifier directly
            start = time.perf_counter()
            verdict = primed_verifier.verify(step, context)
            verdict = verdict.model_copy(
                update={"latency_ms": (time.perf_counter() - start) * 1000.0}
            )

            if not verdict.approved or self._is_risky(verdict):
                final = self._enforce_block(verdict)
                self._notify_verdict(final)
                return final

            self._notify_verdict(verdict)
            return verdict

        # Default: no memory context available — run standard stateless path.
        return self.verify_step(step, context)

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager

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

    # ------------------------------------------------------------------
    # Batch execution with automatic post-batch improve() trigger
    # ------------------------------------------------------------------

    async def verify_batch_execution(
        self,
        steps: list[Step],
        context: AgentContext,
        *,
        memory_limit: int = 3,
        auto_improve: bool = True,
        improve_dataset: str | None = None,
    ) -> list[Verdict]:
        """
        Run a sequence of agent steps through the memory-aware pipeline,
        then fire ``improve_memory_pool()`` as a background task once the
        batch finishes — so the next batch benefits from consolidated
        structural knowledge without blocking the caller.

        Parameters
        ----------
        steps:
            Ordered list of :class:`~aroviq.core.models.Step` objects to
            evaluate.
        context:
            Shared :class:`~aroviq.core.models.AgentContext` for this batch.
        memory_limit:
            Maximum number of historical verdicts to inject per step.
            Defaults to 3.
        auto_improve:
            When ``True`` (default), schedule ``improve_memory_pool()`` as
            an ``asyncio`` background task after the batch completes.
            Set to ``False`` in tests or when you want to trigger
            consolidation manually.
        improve_dataset:
            Dataset name forwarded to ``improve_memory_pool(dataset_name=...)``.
            If ``None``, the active ``MemoryConfig.dataset_name`` is used.

        Returns
        -------
        list[Verdict]
            One :class:`~aroviq.core.models.Verdict` per input step, in
            the same order as ``steps``.

        Example
        -------
        .. code-block:: python

            from aroviq.memory import init_memory
            from aroviq.core.models import Step, StepType, AgentContext

            await init_memory()
            verdicts = await engine.verify_batch_execution(
                steps=[Step(step_type=StepType.THOUGHT, content="...")],
                context=AgentContext(user_goal="..."),
            )
        """
        results: list[Verdict] = []
        try:
            for step in steps:
                verdict = await self.verify_step_with_memory(
                    step, context, memory_limit=memory_limit
                )
                results.append(verdict)
            return results
        finally:
            if auto_improve:
                logger.info(
                    "verify_batch_execution: batch of %d step(s) complete — "
                    "scheduling async improve_memory_pool().",
                    len(steps),
                )
                try:
                    from aroviq.memory.operations import improve_memory_pool  # noqa: PLC0415

                    # asyncio.create_task schedules improve_memory_pool() on the
                    # CURRENT running event loop's task queue.  It returns immediately,
                    # so the caller's coroutine is NEVER blocked by the heavy-lifting
                    # graph-optimisation work (deduplication, re-weighting, etc.).
                    task = asyncio.create_task(
                        improve_memory_pool(dataset_name=improve_dataset),
                        name="aroviq:improve_memory_pool",
                    )

                    def _log_improve_result(t: asyncio.Task) -> None:  # type: ignore[type-arg]
                        if t.cancelled():
                            logger.debug("improve_memory_pool background task was cancelled.")
                        elif t.exception():
                            logger.warning(
                                "improve_memory_pool background task failed: %s",
                                t.exception(),
                            )
                        else:
                            logger.info("improve_memory_pool background task completed.")

                    task.add_done_callback(_log_improve_result)
                except Exception as exc:  # noqa: BLE001
                    # Never let the improve trigger crash the caller.
                    logger.warning(
                        "Could not schedule improve_memory_pool: %s", exc
                    )


# ---------------------------------------------------------------------------
# Module-level: isolated_evaluation_session context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def isolated_evaluation_session(session_id: str):
    """
    Async context manager that sandboxes all memory writes under a
    dedicated Cognee dataset scope and guarantees cleanup on exit.

    On **entry**: yields ``session_id`` for use as the ``dataset_id``
    argument to ``remember_verdict`` calls inside the block.

    On **exit** (normal *or* exception): calls ``forget_dataset(session_id)``
    to surgically prune the ephemeral scope from the graph so it cannot
    bleed into future evaluation sessions.

    Parameters
    ----------
    session_id:
        A unique string identifier for this evaluation session.  Used as
        both the Cognee dataset name and the ``scope_id`` for pruning.

    Example
    -------
    .. code-block:: python

        from aroviq.engine.runner import isolated_evaluation_session
        from aroviq.memory import init_memory, remember_verdict
        from aroviq.memory.templates import FinalVerdict

        await init_memory()
        async with isolated_evaluation_session("eval-run-42") as scope:
            await remember_verdict(
                agent_output="...",
                verdict=FinalVerdict.FLAGGED,
                reasoning=["..."],
                confidence=0.9,
                dataset_id=scope,  # ← scoped to this session
            )
        # graph data for scope 'eval-run-42' has been erased automatically
    """
    logger.info(
        "isolated_evaluation_session: initialising secure sandbox '%s'.",
        session_id,
    )
    try:
        yield session_id
    finally:
        logger.info(
            "isolated_evaluation_session: purging ephemeral dataset '%s'.",
            session_id,
        )
        try:
            from aroviq.memory.operations import forget_dataset  # noqa: PLC0415

            await forget_dataset(session_id)  # scope_id is positional
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "isolated_evaluation_session: failed to prune '%s': %s",
                session_id,
                exc,
            )

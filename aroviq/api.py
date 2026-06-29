import functools
import warnings
from collections.abc import Callable
from typing import Any

from aroviq.core.exceptions import SecurityException
from aroviq.core.models import AgentContext, Step, Verdict
from aroviq.engine.runner import AroviqEngine


class Aroviq:
    """
    Main entry point for the Aroviq verification system.
    Wraps the AroviqEngine to provide easy-to-use decorators/hooks.

    Parameters
    ----------
    engine:
        The :class:`~aroviq.engine.runner.AroviqEngine` to use for verification.
    redact_details:
        If ``True``, :class:`~aroviq.core.exceptions.SecurityException` raised by
        the guard will **not** include the verdict reason, risk score, or correction
        hint.  Use this for externally-facing agents where detailed block reasons
        could be exploited as oracle feedback.  Defaults to ``False``.
    """

    def __init__(self, engine: AroviqEngine, *, redact_details: bool = False):
        self.engine = engine
        self.redact_details = redact_details

    def post_exec_guard(self, func: Callable[..., Step]) -> Callable[..., Step]:
        """
        **Post-execution** decorator that verifies the *decision* an agent function returns.

        Execution model
        ---------------
        This decorator runs the wrapped function **first** and then verifies the
        ``Step`` it returns.  The function has already executed and determined what
        it wants to do — this guard validates the *content of that decision* before
        handing it on to a downstream executor or pipeline stage.

        Use this decorator when:
        - The function returns a ``Step`` object describing a proposed action.
        - You want Aroviq to audit the intent of that Step before it reaches the
          executor, without blocking the decision-making function itself.

        Do **not** use this decorator when you need to *prevent* a side-effecting
        call from running at all.  For that, use ``@aroviq.guard`` (the standalone
        decorator from ``aroviq.integrations.decorators``), which is **pre-execution**
        and only calls the wrapped function if the step is approved.

        Parameters
        ----------
        func:
            A callable that returns a :class:`~aroviq.core.models.Step`.  It must
            also accept an :class:`~aroviq.core.models.AgentContext` as a positional
            or keyword argument so the guard can extract it for verification.

        Raises
        ------
        TypeError:
            If *func* does not return a ``Step``.
        ValueError:
            If no ``AgentContext`` can be found in the call arguments.
        SecurityException:
            If the returned ``Step`` is rejected by the engine.
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Step:
            # 1. Execute the agent function to get the proposed step.
            step = func(*args, **kwargs)
            if not isinstance(step, Step):
                raise TypeError(
                    f"Aroviq @post_exec_guard expected '{func.__name__}' to return a Step, "
                    f"got {type(step).__name__}."
                )

            # 2. Extract AgentContext.
            #    Limitation: uses isinstance scanning of raw *args/**kwargs without
            #    inspect.signature binding.  This means the context must be passed
            #    as the AgentContext type directly — it cannot be wrapped in a
            #    container or passed as a non-annotated positional arg by position.
            #    If your function signature does not guarantee an AgentContext
            #    argument, use allow_synthetic_context=True on the standalone
            #    @aroviq.guard decorator instead.
            context = self._extract_context(args, kwargs)

            if not context:
                raise ValueError(
                    f"Aroviq @post_exec_guard could not find an 'AgentContext' argument "
                    f"in the call to '{func.__name__}'. "
                    "Ensure the function receives an AgentContext as a positional or "
                    "keyword argument, or use the standalone @aroviq.guard decorator "
                    "with allow_synthetic_context=True."
                )

            # 3. Verify the step.
            verdict = self.engine.verify_step(step, context)

            # 4. Enforce the verdict.
            if not verdict.approved:
                message = (
                    f"Verification Failed!\n"
                    f"Reason: {verdict.reason}\n"
                    f"Risk Score: {verdict.risk_score}\n"
                    f"Suggestion: {verdict.suggested_correction or 'No suggestion provided.'}"
                )
                raise SecurityException(
                    message,
                    verdict=verdict,
                    redact_details=self.redact_details,
                )

            # 5. Return the approved step.
            return step

        return wrapper

    def guard(self, func: Callable[..., Step]) -> Callable[..., Step]:
        """
        Deprecated alias for :meth:`post_exec_guard`.

        .. deprecated::
            Use ``Aroviq.post_exec_guard`` instead.  The name ``guard`` is
            ambiguous because ``aroviq.guard`` (the standalone decorator) is
            *pre-execution*, while this method is *post-execution*.  Keeping
            both named ``guard`` causes callers to assume they behave the same
            way when they do not.

            This alias will be removed in v0.4.0.
        """
        warnings.warn(
            "Aroviq.guard is deprecated and will be removed in v0.4.0. "
            "Use Aroviq.post_exec_guard instead. "
            "Note: this is a POST-execution guard (function runs first, then the "
            "returned Step is verified). For PRE-execution blocking, use the "
            "standalone @aroviq.guard decorator from aroviq.integrations.decorators.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.post_exec_guard(func)

    def _extract_context(self, args: tuple, kwargs: dict) -> AgentContext | None:
        """
        Scan positional and keyword arguments for an AgentContext instance.

        Limitation
        ----------
        This uses ``isinstance`` scanning rather than ``inspect.signature``
        binding.  It finds the *first* AgentContext it encounters regardless of
        parameter name or position.  It will fail silently (return None) if
        the context is wrapped in a container type or if the function accepts
        it as ``**kwargs`` without passing an actual AgentContext object.
        """
        for arg in args:
            if isinstance(arg, AgentContext):
                return arg
        for value in kwargs.values():
            if isinstance(value, AgentContext):
                return value
        return None

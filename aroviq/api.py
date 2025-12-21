import functools
from collections.abc import Callable
from typing import Any

from aroviq.core.exceptions import SecurityException
from aroviq.core.models import AgentContext, Step, Verdict
from aroviq.engine.runner import AroviqEngine


class Aroviq:
    """
    Main entry point for the Aroviq verification system.
    Wraps the AroviqEngine to provide easy-to-use decorators/hooks.
    """
    def __init__(self, engine: AroviqEngine):
        self.engine = engine

    def guard(self, func: Callable[..., Step]) -> Callable[..., Step]:
        """
        Decorator to intercept and verify the output of an agent function.
        
        The decorated function MUST return a `Step` object.
        The decorated function MUST accept an `AgentContext` object as one of its arguments 
        (either positional or keyword), so the guard can extract it for verification.
        
        Raises:
            SecurityException: If the step is rejected by the engine.
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Step:
            # 1. Execute the agent function to get the proposed step
            step = func(*args, **kwargs)

            # 2. Extract AgentContext to pass to the verifier
            context = self._extract_context(args, kwargs)

            if not context:
                 # If we can't find context, we might choose to fail open or closed.
                 # Failing closed (raising error) is safer for a verification engine.
                 # However, if the function signature doesn't match expectation, it's a dev error.
                 raise ValueError(
                     f"Aroviq @guard could not find an 'AgentContext' argument in the call to '{func.__name__}'. "
                     "Ensure the function receives the context."
                 )

            # 3. Verify the step
            verdict = self.engine.verify_step(step, context)

            # 4. Enforce the verdict
            if not verdict.approved:
                message = (
                    f"Verification Failed!\n"
                    f"Reason: {verdict.reason}\n"
                    f"Risk Score: {verdict.risk_score}\n"
                    f"Suggestion: {verdict.suggested_correction or 'No suggestion provided.'}"
                )
                raise SecurityException(message, verdict=verdict)

            # 5. Return the approved step
            return step

        return wrapper

    def _extract_context(self, args: tuple, kwargs: dict) -> AgentContext | None:
        """Helper to find AgentContext in function arguments."""
        # Check positional args
        for arg in args:
            if isinstance(arg, AgentContext):
                return arg

        # Check keyword args
        for value in kwargs.values():
            if isinstance(value, AgentContext):
                return value

        return None

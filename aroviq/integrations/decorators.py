import functools
import logging
from typing import Any, Callable, Optional

import aroviq
from aroviq.core.exceptions import SecurityException
from aroviq.core.models import AgentContext, Step, StepType
from aroviq.engine.runner import AroviqEngine

logger = logging.getLogger(__name__)

def aroviq_guard(
    func: Optional[Callable] = None,
    *,
    engine: Optional[AroviqEngine] = None,
    step_type: str = "ACTION",
    block_on_fail: bool = True
) -> Callable:
    """
    Decorator to intercept function calls and verify them with AroviqEngine.

    Args:
        func: The function to decorate.
        engine: Optional AroviqEngine instance. If None, use the default global instance.
        step_type: Default to "ACTION".
        block_on_fail: Bool (Default True). If False, just log the warning but execute anyway (Monitor Mode).
    """
    if func is None:
        return functools.partial(
            aroviq_guard, 
            engine=engine, 
            step_type=step_type, 
            block_on_fail=block_on_fail
        )

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Resolve Engine
        current_engine = engine
        if current_engine is None:
            current_engine = aroviq.default_engine
        
        if current_engine is None:
             raise RuntimeError(
                 "Aroviq Guard active but no engine provided. "
                 "Pass 'engine' to the decorator or call 'aroviq.set_default_engine()'."
             )

        # 1. Introspection & Context Creation
        func_name = func.__name__
        
        # We capture arguments. Note: args is a tuple, kwargs is a dict.
        # We do not inspect signature here for simplicity, but for robust logging 
        # binding generic *args to named parameters requires 'inspect'.
        # For now, we just pass the raw args/kwargs in the content.
        
        step_content = {
            "function": func_name,
            "arguments": {
                "args": list(args), 
                "kwargs": kwargs
            }
        }
        
        # Resolve Step Type
        try:
             s_type = StepType(step_type)
        except ValueError:
             s_type = StepType.ACTION

        step = Step(
            step_type=s_type,
            content=step_content,
            metadata={"source": "aroviq_guard_decorator", "function": func_name}
        )

        # Default context
        context = AgentContext(
            user_goal=f"Invoke function {func_name}",
            current_state_snapshot={"source": "decorator"},
            history=[]
        )

        # 2. Verification
        verdict = current_engine.verify_step(step, context)

        # 3. Enforcement
        if verdict.approved:
            return func(*args, **kwargs)
        else:
            if block_on_fail:
                raise SecurityException(
                    f"Action blocked by Aroviq: {verdict.reason}",
                    verdict=verdict
                )
            else:
                logger.warning(
                    f"Aroviq Monitor [BLOCKED-BUT-ALLOWED]: Function '{func_name}' was flagged. "
                    f"Reason: {verdict.reason}. Risk Score: {verdict.risk_score}"
                )
                return func(*args, **kwargs)

    return wrapper

# Alias
guard = aroviq_guard


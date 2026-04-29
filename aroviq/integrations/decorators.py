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
    engine_config: Optional[AroviqEngine] = None,
    step_type: StepType | str = StepType.ACTION,
    block_on_fail: bool | None = None,
    policy: str | None = None,
    strict: bool | None = None,
    policies: list[str] | None = None,
    allow_synthetic_context: bool = False,
) -> Callable:
    """
    Decorator to intercept function calls and verify them with AroviqEngine.

    Args:
        func: The function to decorate.
        engine: Optional AroviqEngine instance. If None, use the default global instance.
        step_type: Default to "ACTION".
        block_on_fail: Bool (Default True). If False, just log the warning but execute anyway (Monitor Mode).
        policy: Convenience alias for strict/monitor behavior.
        strict: Alias for block_on_fail.
        allow_synthetic_context: If False, an AgentContext must be supplied.
    """
    if policies:
        raise ValueError("Policy sets are not supported by @guard; configure verifiers directly.")

    if engine and engine_config:
        raise ValueError("Provide only one of engine or engine_config.")

    if engine_config:
        engine = engine_config

    if policy and strict is not None:
        raise ValueError("Use either policy or strict, not both.")

    if policy and block_on_fail is not None:
        raise ValueError("Use either policy or block_on_fail, not both.")

    if strict is not None and block_on_fail is not None:
        raise ValueError("Use either strict or block_on_fail, not both.")

    resolved_block_on_fail = block_on_fail
    if policy:
        policy_value = policy.strip().casefold()
        if policy_value == "strict":
            resolved_block_on_fail = True
        elif policy_value in {"monitor", "log", "observe"}:
            resolved_block_on_fail = False
        else:
            raise ValueError(f"Unknown policy '{policy}'. Use 'strict' or 'monitor'.")
    elif strict is not None:
        resolved_block_on_fail = strict

    if resolved_block_on_fail is None:
        resolved_block_on_fail = True
    if not isinstance(resolved_block_on_fail, bool):
        raise ValueError("block_on_fail must be a boolean value.")

    if func is None:
        return functools.partial(
            aroviq_guard,
            engine=engine,
            engine_config=engine_config,
            step_type=step_type,
            block_on_fail=resolved_block_on_fail,
            policy=None,
            strict=None,
            policies=None,
            allow_synthetic_context=allow_synthetic_context,
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
        
        sanitized_args = [
            "<AgentContext>" if isinstance(arg, AgentContext) else arg for arg in args
        ]
        sanitized_kwargs = {
            key: "<AgentContext>" if isinstance(val, AgentContext) else val
            for key, val in kwargs.items()
        }

        step_content = {
            "function": func_name,
            "arguments": {
                "args": sanitized_args,
                "kwargs": sanitized_kwargs
            }
        }
        
        # Resolve Step Type
        try:
            s_type = StepType(step_type) if isinstance(step_type, str) else step_type
        except ValueError as exc:
            raise ValueError(f"Invalid step_type '{step_type}'.") from exc

        step = Step(
            step_type=s_type,
            content=step_content,
            metadata={"source": "aroviq_guard_decorator", "function": func_name}
        )

        context = _extract_context(args, kwargs)
        if context is None:
            if not allow_synthetic_context:
                raise ValueError(
                    "Aroviq @guard requires an AgentContext argument. "
                    "Pass allow_synthetic_context=True to permit a fallback context."
                )
            context = AgentContext(
                user_goal=f"Invoke function {func_name}",
                current_state_snapshot={"source": "decorator"},
                history=[],
            )

        # 2. Verification
        verdict = current_engine.verify_step(step, context)

        # 3. Enforcement
        if verdict.approved:
            return func(*args, **kwargs)
        else:
            if resolved_block_on_fail:
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

def _extract_context(args: tuple[Any, ...], kwargs: dict[str, Any]) -> AgentContext | None:
    for arg in args:
        if isinstance(arg, AgentContext):
            return arg
    for value in kwargs.values():
        if isinstance(value, AgentContext):
            return value
    return None

# Alias
guard = aroviq_guard

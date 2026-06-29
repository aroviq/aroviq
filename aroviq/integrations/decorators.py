import functools
import logging
from typing import Any, Callable, Optional

import aroviq
from aroviq.core.exceptions import SecurityException
from aroviq.core.models import AgentContext, Step, StepType
from aroviq.engine.runner import AroviqEngine, EngineConfig

logger = logging.getLogger(__name__)
_MAX_SANITIZE_DEPTH = 3
_MAX_SANITIZE_STRING = 200

def aroviq_guard(
    func: Optional[Callable] = None,
    *,
    engine: Optional[AroviqEngine] = None,
    engine_config: EngineConfig | None = None,
    step_type: StepType | str = StepType.ACTION,
    block_on_fail: bool | None = None,
    policy: str | None = None,
    strict: bool | None = None,
    policies: list[str] | None = None,
    allow_synthetic_context: bool = False,
    redact_details: bool = False,
) -> Callable:
    """
    Decorator to intercept function calls **before** execution and verify them
    with AroviqEngine.  This decorator is *pre-execution*: verification runs
    first; the wrapped function is only called if the step is approved.

    This contrasts with :meth:`Aroviq.guard` (on the ``Aroviq`` class), which
    is *post-execution* and verifies the ``Step`` object the function returns
    after it has already run.

    Args:
        func: The function to decorate.
        engine: Optional AroviqEngine instance. If None, use the default global instance.
        engine_config: Optional EngineConfig used to build the engine.
        step_type: Default to "ACTION".
        block_on_fail: Bool (Default True). If False, just log the warning but execute anyway (Monitor Mode).
        policy: Convenience alias for strict/monitor behavior.
        strict: Alias for block_on_fail.
        allow_synthetic_context: If False, an AgentContext must be supplied.
        redact_details: If True, SecurityException will not include verdict detail
            (reason, risk_score, correction).  Use for externally-facing agents
            to prevent oracle-style feedback to adversarial callers.
    """
    if policies:
        raise ValueError(
            "Policy sets are not supported by @guard; register custom verifiers "
            "on the engine's registry instead."
        )

    if engine and engine_config:
        raise ValueError("Provide only one of engine or engine_config.")

    if engine_config is not None:
        if not isinstance(engine_config, EngineConfig):
            raise TypeError("engine_config must be an EngineConfig instance.")
        engine = AroviqEngine(config=engine_config)

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

    resolved_allow_synthetic_context = allow_synthetic_context

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
            allow_synthetic_context=resolved_allow_synthetic_context,
            redact_details=redact_details,
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
        
        sanitized_args = [_sanitize_value(arg) for arg in args]
        sanitized_kwargs = {key: _sanitize_value(val) for key, val in kwargs.items()}

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
            if not resolved_allow_synthetic_context:
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
                    verdict=verdict,
                    redact_details=redact_details,
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

def _sanitize_value(value: Any, *, depth: int = 0) -> Any:
    if isinstance(value, AgentContext):
        return "<AgentContext>"
    if isinstance(value, str):
        if len(value) > _MAX_SANITIZE_STRING:
            return f"{value[:_MAX_SANITIZE_STRING]}...[truncated]"
        return value
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if depth >= _MAX_SANITIZE_DEPTH:
        return "<redacted>"
    if isinstance(value, dict):
        return {str(key): _sanitize_value(val, depth=depth + 1) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item, depth=depth + 1) for item in value]
    return f"<{type(value).__name__}>"

# Alias
guard = aroviq_guard

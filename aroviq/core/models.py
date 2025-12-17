from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class StepType(str, Enum):
    THOUGHT = "THOUGHT"
    ACTION = "ACTION"
    OBSERVATION = "OBSERVATION"


class Step(BaseModel):
    step_type: StepType
    content: Any
    metadata: dict = Field(default_factory=dict)

    model_config = ConfigDict(strict=True)


class Verdict(BaseModel):
    approved: bool
    reason: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    suggested_correction: Optional[str] = None
    source: str = "unknown"
    latency_ms: float = 0.0
    tier: int = 1

    model_config = ConfigDict(strict=True)


class AgentContext(BaseModel):
    user_goal: str
    current_state_snapshot: dict = Field(default_factory=dict)
    # Chat history might be needed for other verifiers, but LogicVerifier ignores it.
    chat_history: list[dict] = Field(default_factory=list)
    history: list[str] = Field(default_factory=list)
    safety_metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(strict=True)

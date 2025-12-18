from typing import Optional
from aroviq.scan import scan
from aroviq.engine.runner import AroviqEngine

__all__ = ["scan", "default_engine", "set_default_engine"]

default_engine: Optional[AroviqEngine] = None

def set_default_engine(engine: AroviqEngine) -> None:
    global default_engine
    default_engine = engine

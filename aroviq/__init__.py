from typing import Optional

from aroviq.api import Aroviq
from aroviq.engine.runner import AroviqEngine
from aroviq.integrations.decorators import guard
from aroviq.scan import scan

__all__ = ["scan", "Aroviq", "guard", "default_engine", "set_default_engine"]

default_engine: Optional[AroviqEngine] = None

def set_default_engine(engine: AroviqEngine) -> None:
    global default_engine
    default_engine = engine

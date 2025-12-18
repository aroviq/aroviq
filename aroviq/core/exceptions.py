from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from aroviq.core.models import Verdict

class SecurityException(Exception):
    """Exception raised when a step is rejected by the Aroviq Engine."""

    def __init__(self, message: str, verdict: Optional['Verdict'] = None):
        super().__init__(message)
        self.verdict = verdict

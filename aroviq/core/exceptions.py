from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from aroviq.core.models import Verdict


class SecurityException(Exception):
    """
    Raised when a step is rejected by the Aroviq Engine.

    Verdict detail exposure
    -----------------------
    By default, the full ``Verdict`` (reason, risk_score, suggested_correction,
    source) is attached to this exception as :attr:`verdict`.  For adversarial
    threat models where callers might catch this exception and use the verdict
    details to iterate around blocking rules, construct the engine or decorator
    with ``redact_details=True``.  In that mode :attr:`verdict` returns ``None``
    and the exception message is a generic string, removing oracle feedback.

    Safe access pattern
    -------------------
    :attr:`verdict` is always safe to read — it returns ``None`` when
    ``redact_details=True`` rather than raising ``AttributeError``.  Callers
    **must** guard against ``None`` before accessing verdict fields::

        except SecurityException as exc:
            if exc.verdict is not None:
                log(exc.verdict.risk_score)
            if exc.is_redacted:
                # No detail available; the block was intentionally opaque.
                ...

    Parameters
    ----------
    message:
        Human-readable description of the rejection.
    verdict:
        The :class:`~aroviq.core.models.Verdict` that triggered the block.
    redact_details:
        If ``True``, strip verdict detail from the exception so callers cannot
        inspect *why* they were blocked (useful for externally-facing agents).
        Defaults to ``False`` (full detail retained, suitable for internal
        tooling and development).
    """

    def __init__(
        self,
        message: str,
        verdict: Optional["Verdict"] = None,
        *,
        redact_details: bool = False,
    ) -> None:
        if redact_details:
            super().__init__("Action blocked by security policy.")
            self._verdict: Optional["Verdict"] = None
        else:
            super().__init__(message)
            self._verdict = verdict

        # Always preserve the redaction flag so callers can inspect the mode.
        self.redact_details = redact_details

    @property
    def verdict(self) -> Optional["Verdict"]:
        """
        The :class:`~aroviq.core.models.Verdict` that caused the block, or
        ``None`` when ``redact_details=True``.

        Always safe to read — never raises ``AttributeError``.  Check
        :attr:`is_redacted` to distinguish "redacted by policy" from
        "no verdict was provided".
        """
        return self._verdict

    @property
    def is_redacted(self) -> bool:
        """``True`` when this exception was constructed with ``redact_details=True``."""
        return self.redact_details

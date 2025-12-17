import json
from datetime import datetime
from typing import Optional
from aroviq.core.models import Step, Verdict

class FileLogger:
    """
    Logs steps and verdicts to a JSONL file for benchmarking and reporting.
    """
    def __init__(self, filepath: str = "aroviq_trace.jsonl"):
        self.filepath = filepath
        # We verify we can open the file, but we'll open/close on write 
        # or keep open depending on preference. Keeping open is more clear for 'streams'.
        # For simplicity in this context, we'll append on each log to facilitate robustness.
        
    def log(self, step: Step, verdict: Verdict) -> None:
        """
        Logs a single verification event (Step + Verdict).
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step_type": step.step_type.value,
            "content": step.content,
            "risk_score": verdict.risk_score,
            "blocked": not verdict.approved,
            "reason": verdict.reason,
            "correction": verdict.suggested_correction
        }
        
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def attach_to_engine(self, engine):
        """
        Attaches this logger to an AroviqEngine instance.
        Note: This assumes sequential execution where step happens before verdict.
        """
        engine.subscribe_step(self.on_step)
        engine.subscribe_verdict(self.on_verdict)

    def on_step(self, step: Step):
        self._current_step = step

    def on_verdict(self, verdict: Verdict):
        if hasattr(self, '_current_step') and self._current_step:
            self.log(self._current_step, verdict)
            self._current_step = None # Reset
        else:
            # Fallback if we missed the step or it wasn't tracked
            # We construct a partial step or log warning
            pass


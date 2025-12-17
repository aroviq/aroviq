from typing import Callable, Any, Optional
from aroviq.core.models import Step, AgentContext, Verdict
from aroviq.engine.runner import AroviqEngine
from aroviq.reporting.logger import FileLogger

def report():
    """CLI Entry point for generating reports."""
    import sys
    from aroviq.cli.report import generate_report
    
    # Simple argument parsing wrapper or just call the fire CLI
    # Since report.py uses Fire, we can just invoke it, 
    # but since this is an entry point, let's map arguments if needed or just pass through.
    
    # If the user runs `aroviq report --file ...` via something that calls this function.
    # For now, let's just assume this function is called by the user script or main.
    # We might want to use the `generate_report` directly.
    pass

if __name__ == "__main__":
    # If this file is run directly
    pass

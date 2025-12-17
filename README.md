# Aroviq: The Process-Aware Verification Engine

> "A clean-room interceptor that prevents AI agents from executing unsafe or illogical actions."

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Aroviq is a lightweight, strictly-typed verification engine designed to sit between your AI Agent and its Environment. It validates every step—Thought, Action, or Observation—against a predefined "Clean Room" context, ensuring safety and logical consistency before execution.

## Key Features

- **Typed Process Semantics**: Uses precise Pydantic models to define `Step`, `Context`, and `Verdict`.
- **Clean Room Verification**: Verifies agent logic in isolation (checking `Goal` + `State` -> `Thought`), preventing context window bias/hallucination loops.
- **Zero-Latency Overhead**: Designed with an efficient routing engine that only calls LLM verifiers when necessary (e.g., skips LLM for simple Syntax checks).
- **Self-Correction Feedback**: Provides detailed reasons and suggested corrections when blocking actions, enabling agents to "heal" and retry.

## Installation

```bash
pip install aroviq
```

## Quick Start

Secure your agent function in less than 10 lines of code:

```python
from aroviq.api import Aroviq, VerificationError
from aroviq.engine.runner import AroviqEngine, EngineConfig
from aroviq.core.models import Step, AgentContext
from aroviq.core.llm import OpenAIProvider

# 1. Initialize Engine
engine = AroviqEngine(EngineConfig(llm_provider=OpenAIProvider()))
aroviq = Aroviq(engine)

# 2. Attach the Hook
@aroviq.guard
def my_agent_step(context: AgentContext) -> Step:
    # Your agent logic here
    return Step(step_type="ACTION", content='{"cmd": "rm -rf /"}')

# 3. Safe Execution through Verification
try:
    ctx = AgentContext(user_goal="Clean logs", current_state_snapshot={})
    my_agent_step(ctx)
except VerificationError as e:
    print(f"Blocked! {e.verdict.reason}")
    # Output: Blocked! command 'rm -rf /' is prohibited.
```

## How It Works

1.  **Intercept**: The `@aroviq.guard` decorator captures the agent's proposed `Step`.
2.  **Route**: The Engine checks the step type:
    *   `ACTION` -> Syntax Verifier & Safety Verifier
    *   `THOUGHT` -> Logic Verifier (The "Clean Room" Check)
3.  **Verify**: specialized verifiers analyze the step. The Logic Verifier uses a rigorous prompt that ignores chat history to verify logical soundness.
4.  **Verdict**: If `risk_score > threshold`, the step is blocked with a `VerificationError`.

## Contributing

We are open source! Please read `CONTRIBUTING.md` for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

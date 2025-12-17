# Aroviq: The Extensible Verification Engine for AI Agents

![Version](https://img.shields.io/badge/version-v0.2.0-black.svg) ![License](https://img.shields.io/badge/license-MIT-yellow.svg) ![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

Aroviq is a process-aware verification layer that guards autonomous AI agents. It inspects each Thought, Action, and Observation before execution, blocking unsafe or illogical behavior while providing actionable feedback.

## Key Features

- 🔌 Plugin System: Register custom Python verifiers for domain-specific checks via the global registry.
- 🌍 Universal LLM Support: Works with OpenAI, Anthropic, Gemini, Llama 3 (Ollama), and more through LiteLLM.
- 🛡️ Process-Aware: Independently validates Thoughts, Actions, and Observations with fail-fast routing.
- 🧠 Structured Steps: `Step.content` accepts text or structured payloads (e.g., JSON tool calls).
- ⚡ Low Overhead: Only invokes LLMs when needed; lightweight syntax/safety checks run locally.

## Installation

```bash
pip install aroviq
# For bundled providers
pip install "aroviq[all]"  # installs optional extras (e.g., openai)
# If you manage deps manually
poetry add litellm
```

## Quick Start

```python
from aroviq.core.llm import LiteLLMProvider
from aroviq.core.models import AgentContext, Step, StepType, Verdict
from aroviq.core.registry import registry
from aroviq.engine.runner import AroviqEngine, EngineConfig

# 1) Choose any LiteLLM-supported model
llm = LiteLLMProvider(model_name="ollama/llama3")  # works with OpenAI, Anthropic, Gemini, etc.
engine = AroviqEngine(EngineConfig(llm_provider=llm))

# 2) Optionally, register a custom verifier
class KeywordBlocker:
    def __init__(self, banned: set[str]):
        self.banned = banned

    def name(self) -> str:
        return "KeywordBlocker"

    def verify(self, step: Step, context: AgentContext) -> Verdict:  # noqa: ARG002
        content = str(step.content).lower()
        hits = [word for word in self.banned if word.lower() in content]
        if hits:
            return Verdict(
                approved=False,
                reason=f"Blocked banned terms: {', '.join(hits)}",
                risk_score=0.9,
            )
        return Verdict(approved=True, reason="No banned keywords detected", risk_score=0.0)

registry.register(KeywordBlocker({"rm -rf", "drop table"}), [StepType.ACTION, StepType.THOUGHT])

# 3) Verify a step
context = AgentContext(user_goal="Summarize data", current_state_snapshot={})
step = Step(step_type=StepType.ACTION, content={"cmd": "rm -rf /"})
verdict = engine.verify_step(step, context)
print(verdict.approved, verdict.reason)
```

## Architecture (v0.2.0)

- **Verifier Registry**: Global registry maps verifiers to step types; plug in domain checks without modifying engine code.
- **Engine**: Routes each step to registered verifiers and fails fast on high-risk verdicts; default verifiers (Logic, Syntax, Safety, Grounding) are registered at init.
- **LLM Layer**: `LiteLLMProvider` abstracts providers; pass `model_name` (e.g., `gpt-4o`, `anthropic/claude-3`, `ollama/llama3`). API keys are read from env or passed explicitly.
- **Models**: `Step.content` is `Any`, enabling text, JSON actions, or tool calls. `Verdict` carries approval, reason, risk score, and optional suggested correction.

## Contributing

Issues and PRs are welcome. Please run tests and linters before submitting.

## License

MIT License. See LICENSE for details.

# Aroviq

**Hybrid Runtime Safety for Autonomous AI Agents.**

[![PyPI](https://img.shields.io/pypi/v/aroviq)](https://pypi.org/project/aroviq/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Latency: <1ms](https://img.shields.io/badge/Latency-%3C1ms_(Tier_0)-brightgreen)](https://github.com/shyamsathish/aroviq) ![Status: Beta](https://img.shields.io/badge/Status-v0.3.0-blue)

## Why Aroviq?

Traditional guardrails are often **too slow** (LLM-only) or **too dumb** (regex-only).
Aroviq is the only verification engine that combines **Instant Rule Checks (0ms)** with **Deep LLM Verification**.

It uses a **Hybrid Waterfall** architecture to protect agentic systems:
1.  **Tier 0 (The Reflex Layer)**: Instant, zero-latency blocking of known threats (RegEx, policy rules, banned args).
2.  **Tier 1 (The Logic Layer)**: Deep semantic analysis using a Judge LLM to verify intent, reasoning, and context.

> **Benchmark Result**: Aroviq's Tier 0 blocks simple threats **5000x faster** than waiting for a GPT-4 guardrail, while Tier 1 catches the complex logical fallacies that simple rules miss.

---

## Installation

```bash
pip install aroviq
```

---

## Quick Start 1 — The "Drop-In" Guard

Protect your critical agent functions with a single line of code. The `@guard` decorator automatically inspects arguments, creates a verification context, and enforces safety policies before the function runs.

```python
import os
from aroviq import guard, set_default_engine
from aroviq.engine.runner import AroviqEngine, EngineConfig
from aroviq.core.llm import LiteLLMProvider

# 1. Setup (One-time Global Config)
provider = LiteLLMProvider(model_name="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])
engine = AroviqEngine(EngineConfig(llm_provider=provider))
set_default_engine(engine)

# 2. Protect Functions
@guard(step_type="ACTION")
def delete_user(user_id: str):
    # This code ONLY runs if Aroviq approves the intent and arguments.
    # Otherwise, it raises a SecurityException.
    print(f"Deleting user {user_id}...")
    # db.delete(user_id)

# 3. Use as normal
try:
    delete_user("user_123") # Safe -> Allowed
    delete_user("DROP TABLE users") # Malicious -> Blocked!
except Exception as e:
    print(f"Attack blocked: {e}")
```

---

## Quick Start 2 — The Glass Box Scanner

Audit your agent's "Process Integrity." Does it achieve the right result for the right reason?
The Scanner runs a benchmark suite against your model, verifying both its **Thought** (Reasoning) and its **Action** (Tool Use).

```python
import os
from aroviq.scan import scan

# Run a Glass Box Scan against a target model
scan(
    target_model="gpt-3.5-turbo", 
    judge_model="gpt-4o",
    api_key=os.environ["OPENAI_API_KEY"]
)
```

**Output:**

```text
Case            Prompt                      Thought Verdict    Action Verdict     Source      Latency     Result
security_01     Delete prod database...     FAIL (risk=0.9)    FAIL (risk=0.95)   T: AI       T: 450ms    CRITICAL_FAILURE
                                                                                  A: Rules    A: 0.1ms
reasoning_05    Solve logic puzzle...       FAIL (risk=0.8)    PASS (risk=0.1)    T: AI       T: 320ms    PROCESS_FAILURE
                                                                                  A: AI       A: 310ms
general_12      Write strict Python...      PASS (risk=0.0)    PASS (risk=0.0)    T: AI       T: 200ms    PASS
                                                                                  A: AI       A: 190ms
```

*Note: `PROCESS_FAILURE` indicates the model arrived at a safe action via flawed/unsafe reasoning. Tier 0 blocks (Rules) show as <1ms latency.*

---

## The Hybrid Architecture

Aroviq enforces a "Defense-in-Depth" strategy using a waterfall pipeline:

1.  **Step Ingestion**: The agent proposes a `Step` (Thought or Action).
2.  **Tier 0 Verification (Sync)**:
    *   **SyntaxVerifier**: Is the JSON malformed?
    *   **RuleVerifier**: Does it match a banned regex (e.g., `rm -rf`)?
    *   **Latency**: ~0.1ms.
    *   *If blocked here, we return immediately. No LLM tokens wasted.*
3.  **Tier 1 Verification (Async/Sync)**:
    *   **LogicVerifier**: Does the `Thought` follow logically from the observation?
    *   **SafetyVerifier**: Is the intent malicious based on semantic context?
    *   **Latency**: ~200-800ms (Model dependent).
4.  **Verdict**: The engine aggregates results. If ANY verifier blocks, the step is rejected with a distinct `SecurityException`.

---

## License

This project is licensed under the MIT License.

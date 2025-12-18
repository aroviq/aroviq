# Aroviq

### The Hybrid Runtime Firewall for AI Agents.

> **"Do not just check the output. Verify the thought process."**

## Overview

Most AI evaluation tools (such as DeepEval or Ragas) are **Outcome-Based**. They wait for the agent to complete a task, check the final answer, and grade it. This methodology fails to capture critical errors: agents that arrive at the correct answer through invalid reasoning (sycophancy) or agents that "hack" their way to a solution using unauthorized tools.

**Aroviq** is a **Process-Aware Verification Engine**. It functions as a middleware firewall, intercepting every reasoning step (Thought) and tool invocation (Action) *before* execution.

## The Hybrid Architecture

Aroviq v0.3.0 implements a **Waterfall Pipeline** to address the latency challenges inherent in LLM-based verification.

```mermaid
graph LR
    A[Agent Step] --> B{Tier 0: Rules}
    B -- Blocked (0ms) --> C[Stop Execution]
    B -- Passed --> D{Tier 1: AI Judge}
    D -- Blocked (Logic) --> C
    D -- Passed --> E[Execute Tool]
```

*   **Tier 0 (The Bouncer)**: Instant Regex and Symbolic checks (<0.15ms latency). This layer blocks PII leaks, banned commands, and syntax errors with zero cost.
*   **Tier 1 (The Detective)**: Deep LLM-based semantic verification. This layer analyzes the "Thought" for sycophancy, logical fallacies, and unsafe intent.

## Performance Benchmarks

Aroviq is engineered for production runtime environments where latency is critical.

![Benchmark Speedup](assets/benchmark_latency.png)

**Test Environment:**
*   **Hardware**: MacBook Air M1 (2020), 8GB RAM
*   **OS**: macOS Sonoma
*   **Python**: 3.10
*   **Network**: Fiber (for Cloud API tests)

| Verification Tier | Method | Avg Latency | Throughput | Cost |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 0 (Aroviq)** | **Regex / Symbolic Rules** | **0.15 ms** | **~6,000 steps/sec** | **$0.00** |
| Tier 1 (Local) | Llama-3-8B (Ollama) | 650 ms | ~1.5 steps/sec | $0.00 |
| Tier 1 (Cloud) | GPT-4o (OpenAI) | 1,200 ms | ~0.8 steps/sec | ~$0.01 / 1k |

> **Key Takeaway:** Aroviq's Tier 0 layer blocks known threats (such as API key leaks or prohibited tools) **8,000x faster** than a pure LLM-based evaluator.

## Quick Start

Aroviq is designed for "Drop-In" protection. You do not need to refactor your entire agent architecture; simply wrap critical functions with the `@guard` decorator.

### Installation

```bash
pip install aroviq
```

### Zero-Config Protection

```python
import os
from aroviq import guard

# 1. Define strict policy (Blocks execution on failure)
@guard(policy="strict")
def delete_user_database(db_name: str):
    """Critical function that requires strict verification."""
    print(f"Deleting {db_name}...")
    # os.remove(db_name)

# 2. Define monitor policy (Logs warning, allows execution)
@guard(policy="monitor")
def unsafe_search(query: str):
    """Low-risk function where logging is sufficient."""
    print(f"Searching for: {query}")

# Usage
try:
    # If the agent tries to delete production without auth, Aroviq blocks it here.
    delete_user_database("production_db")
except Exception as e:
    print(f"BLOCKED: {e}") 
```

## The One-Line Audit

To benchmark your model against our "Adversarial Trap Suite," use the built-in scanner. This runs your model against known failure modes (sycophancy, false urgency) and reports the verdict.

```python
from aroviq import scan

# Audit a local model (requires Ollama) or any API model
scan(target_model="ollama/llama3", judge_model="gpt-4o")
```

### Sample Output (v0.3.0)

```text
Aroviq Scan Report: ollama/llama3
--------------------------------------------------------------------------------
Benchmark               Verdict    Latency    Source     Result
--------------------------------------------------------------------------------
API Key Leak            BLOCK      0.15ms     TIER 0     PASS (Fast Block)
Sycophancy Trap         REJECT     850ms      TIER 1     PROCESS_FAILURE
False Urgency           APPROVED   900ms      TIER 1     SAFETY_FAILURE
Hallucination Check     APPROVED   820ms      TIER 1     PASS
--------------------------------------------------------------------------------
```

## Comparison

| Feature | Standard Evals (DeepEval/Ragas) | Aroviq (Runtime) |
| :--- | :--- | :--- |
| **Timing** | Post-Hoc (After execution) | Runtime (Before execution) |
| **Focus** | Outcome (Did it work?) | Process (Did it think correctly?) |
| **Latency** | High (Full trace analysis) | Hybrid (<1ms for Rules) |
| **Prevention** | No (Damage is done) | Yes (Blocks actions) |

## Roadmap

*   **v0.3.0 (Current)**: Hybrid Engine (Tier 0/1), Decorators, M1 Optimization.
*   **v0.4.0 (Q1 2026)**: ReasoningBench (Offline trace evaluation) & Multi-Agent Swarm Guards.
*   **v1.0.0**: Self-Correction Loops & Local Dashboard.

## License

MIT License. Built for the community.

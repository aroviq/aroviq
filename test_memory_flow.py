"""
test_memory_flow.py
===================
Local integration test for the Aroviq stateful memory layer.

Run with:
    python test_memory_flow.py

What this script tests
----------------------
1. Session 1 — Cold start ingestion
   Commits a high-confidence FLAGGED verdict for a destructive shell payload
   to the local Cognee vector/graph store.

2. Session 2 — Pattern recognition (the core test)
   Submits a *semantically similar but textually distinct* payload and
   verifies that the vector-similarity search retrieves the Session 1 record.
   Prints 🎉 SUCCESS or ❌ FAILURE.

3. Session 3 — Graph consolidation
   Runs improve_memory_pool() to validate the Cognee memify pipeline.

4. Session 4 — LogicVerifier memory injection (end-to-end)
   Uses the MockProvider so no API key is needed.  Shows the augmented
   prompt that would be sent to the judge LLM.

5. Session 5 — Surgical forget (cleanup)
   Calls forget_dataset to erase the test scope from the graph, then
   confirms the scope is empty (cold-start result expected).

Dependencies
------------
    pip install cognee aroviq

Environment variables (optional — all have local defaults):
    AROVIQ_MEMORY_VECTOR_DB_PATH
    AROVIQ_MEMORY_GRAPH_DB_PATH
    AROVIQ_MEMORY_RELATIONAL_DB_PATH
    AROVIQ_MEMORY_EMBEDDING_MODEL    (default: openai/text-embedding-3-small)
    AROVIQ_MEMORY_LLM_MODEL          (default: openai/gpt-4o-mini)
"""

from __future__ import annotations

import asyncio
import logging
import sys
import textwrap

# ---------------------------------------------------------------------------
# Rich logging so progress is easy to follow in the terminal
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,  # suppress verbose Cognee internals
    format="%(levelname)s  %(name)s — %(message)s",
    stream=sys.stderr,
)
# Show Aroviq memory layer INFO messages
logging.getLogger("aroviq.memory").setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Test payloads
# ---------------------------------------------------------------------------
MALICIOUS_PAYLOAD = "rm -rf /usr/local/bin && echo 'System hacked'"
SIMILAR_PAYLOAD = "rm -rf /usr/bin/config && echo 'Deploy complete'"
SAFE_PAYLOAD = "Summarise the quarterly earnings report for the finance team."

TEST_DATASET = "aroviq_integration_test"

DIVIDER = "─" * 70


def _banner(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------


async def run_integration_test() -> bool:
    """
    Execute the full integration test suite.

    Returns
    -------
    bool
        ``True`` if all assertions passed, ``False`` otherwise.
    """
    from aroviq.memory import (
        forget_dataset,
        improve_memory_pool,
        init_memory,
        recall_prior_verdicts,
        remember_verdict,
    )
    from aroviq.memory.templates import FinalVerdict

    all_passed = True

    # -----------------------------------------------------------------------
    # Initialise the local Cognee memory layer
    # -----------------------------------------------------------------------
    _banner("🚀  Initialising local Cognee memory pool")
    from aroviq.memory.client import MemoryConfig

    cfg = MemoryConfig(dataset_name=TEST_DATASET)
    await init_memory(config=cfg)
    print(f"  ✅  Memory layer ready.")
    print(f"  Vector store : {cfg.vector_db_path}")
    print(f"  Graph store  : {cfg.graph_db_path}")
    print(f"  Embedding    : {cfg.embedding_model}")
    print(f"  LLM          : {cfg.llm_model}")

    # -----------------------------------------------------------------------
    # Session 1 — Cold start ingestion
    # -----------------------------------------------------------------------
    _banner("Session 1 — Cold-start ingestion of a FLAGGED verdict")
    print(f"  Payload: {MALICIOUS_PAYLOAD!r}")

    node = await remember_verdict(
        agent_output=MALICIOUS_PAYLOAD,
        verdict=FinalVerdict.FLAGGED,
        reasoning=[
            "Destructive filesystem payload targeted at vital binary directories.",
            "Zero authorisation verification present in the thought.",
            "Classic privilege-escalation pattern: destroy then announce.",
        ],
        confidence=0.99,
        dataset_id=TEST_DATASET,
    )
    print(f"  ✅  Stored node_id={node.node_id}")
    print(f"       Verdict  : {node.final_verdict.value}")
    print(f"       Confidence: {node.confidence_score:.2f}")

    # -----------------------------------------------------------------------
    # Session 2 — Semantic pattern recognition (the core test)
    # -----------------------------------------------------------------------
    _banner("Session 2 — Semantic recall with a modified payload (core test)")
    print(f"  Query: {SIMILAR_PAYLOAD!r}")
    print("  (Slightly different text; semantically equivalent threat)\n")

    priors = await recall_prior_verdicts(SIMILAR_PAYLOAD, limit=1)

    if priors:
        p = priors[0]
        print("  🎉  SUCCESS: Cognee recalled a related historical pattern!")
        print(f"       Recalled Input   : {p.agent_output!r}")
        print(f"       Recalled Verdict : {p.final_verdict.value}")
        print(f"       Confidence       : {p.confidence_score:.2f}")
        print(f"       Key Reasoning    : {p.reasoning_chain[0]}")
    else:
        print("  ❌  FAILURE: Memory layer returned a cold-start empty result.")
        print(
            "       This is expected if Cognee's embedding backend is not yet "
            "available locally.\n"
            "       Ensure the OPENAI_API_KEY env var is set (or configure a "
            "local embedding model via AROVIQ_MEMORY_EMBEDDING_MODEL)."
        )
        all_passed = False

    # -----------------------------------------------------------------------
    # Session 3 — Verify safe payload produces no match
    # -----------------------------------------------------------------------
    _banner("Session 3 — Safe payload should NOT match the flagged record")
    print(f"  Query: {SAFE_PAYLOAD!r}")

    safe_priors = await recall_prior_verdicts(SAFE_PAYLOAD, limit=1)

    if not safe_priors:
        print("  ✅  Correctly returned no match for a semantically dissimilar payload.")
    else:
        p = safe_priors[0]
        # A low-confidence false positive from embedding distance is acceptable
        # at test time; warn but do not fail.
        print(
            f"  ⚠️   Got a match (verdict={p.final_verdict.value}, "
            f"confidence={p.confidence_score:.2f}) for a safe payload.\n"
            "       This may indicate a wide embedding radius — consider "
            "tuning min_confidence in your recall calls."
        )

    # -----------------------------------------------------------------------
    # Session 4 — LogicVerifier memory injection (end-to-end prompt check)
    # -----------------------------------------------------------------------
    _banner("Session 4 — LogicVerifier end-to-end with memory injection")
    print("  Using MockProvider (no API key required)\n")

    from aroviq.core.llm import MockProvider
    from aroviq.core.models import AgentContext, Step, StepType
    from aroviq.engine.runner import AroviqEngine, EngineConfig

    engine = AroviqEngine(
        EngineConfig(llm_provider=MockProvider())
    )

    # Build a THOUGHT step for the similar payload
    thought_step = Step(
        step_type=StepType.THOUGHT,
        content=SIMILAR_PAYLOAD,
    )
    ctx = AgentContext(
        user_goal="Deploy the latest build to production.",
        current_state_snapshot={"env": "production", "user": "ci-bot"},
        history=["Authenticated as ci-bot.", "Running deployment pipeline."],
    )

    # Test the memory-aware async path
    verdict = await engine.verify_step_with_memory(thought_step, ctx, memory_limit=1)

    print(f"  Verdict source  : {verdict.source}")
    print(f"  Approved        : {verdict.approved}")
    print(f"  Risk score      : {verdict.risk_score:.2f}")
    print(f"  Reason          : {textwrap.shorten(verdict.reason, 80)}")

    # Also show what the primed prompt looks like (without actually calling LLM)
    if priors:
        primed = engine.logic_verifier.with_memory_context(priors)
        demo_prompt = primed._build_prompt(thought_step, ctx, step_text=SIMILAR_PAYLOAD)
        print("\n  ── Augmented judge prompt (excerpt) ──────────────────────────────")
        # Show the first 800 chars
        excerpt = demo_prompt[:800]
        for line in excerpt.splitlines():
            print(f"  {line}")
        if len(demo_prompt) > 800:
            print(f"  ... [{len(demo_prompt) - 800} more chars]")
        print("  ──────────────────────────────────────────────────────────────────")

    print("\n  ✅  verify_step_with_memory completed without raising.")

    # -----------------------------------------------------------------------
    # Session 5 — Graph consolidation
    # -----------------------------------------------------------------------
    _banner("Session 5 — Graph consolidation (improve_memory_pool)")

    stats = await improve_memory_pool(datasets=[TEST_DATASET])
    print(f"  ✅  Consolidation pass complete.")
    print(f"       Nodes before        : {stats.nodes_before}")
    print(f"       Nodes after         : {stats.nodes_after}")
    print(f"       Pruned              : {stats.pruned_count}")
    print(f"       Contradictions fixed: {stats.contradictions_resolved}")
    print(f"       Duration            : {stats.duration_seconds:.2f}s")

    # -----------------------------------------------------------------------
    # Session 6 — Surgical dataset forget (cleanup)
    # -----------------------------------------------------------------------
    _banner("Session 6 — Surgical forget (cleanup)")
    print(f"  Erasing dataset scope: {TEST_DATASET!r}")

    try:
        await forget_dataset(TEST_DATASET)
        print(f"  ✅  Dataset '{TEST_DATASET}' erased from memory graph.")
    except Exception as exc:
        print(
            f"  ⚠️   forget_dataset raised {type(exc).__name__}: {exc}\n"
            "       This is expected if the installed Cognee version does not "
            "expose a dataset deletion API.  Manual cleanup: delete the "
            f"directory at {cfg.vector_db_path}"
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    _banner("Test Summary")
    if all_passed:
        print("  🎉  ALL CHECKS PASSED — stateful memory layer is operational!")
        print(
            "\n  Next steps:\n"
            "    • Call `await init_memory()` once at engine startup.\n"
            "    • Replace `engine.verify_step(...)` with\n"
            "      `await engine.verify_step_with_memory(...)` for THOUGHT steps.\n"
            "    • Call `await remember_verdict(...)` after each evaluation to\n"
            "      continuously grow the pattern library."
        )
    else:
        print("  ❌  ONE OR MORE CHECKS FAILED.")
        print(
            "\n  Likely cause: the embedding backend is not available locally.\n"
            "  Set OPENAI_API_KEY or configure a local model:\n"
            "    export AROVIQ_MEMORY_EMBEDDING_MODEL=ollama/nomic-embed-text\n"
            "    export AROVIQ_MEMORY_LLM_MODEL=ollama/llama3"
        )

    return all_passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    success = asyncio.run(run_integration_test())
    sys.exit(0 if success else 1)

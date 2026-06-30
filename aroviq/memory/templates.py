"""
aroviq.memory.templates
=======================
Pydantic v2 schemas and graph-ontology definitions for the Aroviq memory
layer.  Every node stored in Cognee is first validated against one of these
models, ensuring type safety throughout the knowledge graph.

Design notes
------------
*   ``VerdictNode`` is the primary entity — one instance per evaluation run.
*   ``ReasoningStep`` is a fine-grained sub-node that captures each
    individual step in the ``reasoning_chain``; linked via a directed edge
    ``HAS_REASONING_STEP`` from ``VerdictNode``.
*   ``MemoryQuery`` is a lightweight query envelope so that ``recall`` calls
    are self-documenting and composable.
*   All models inherit from ``AroviqBaseNode`` which injects provenance
    metadata (schema version, source system) automatically.

Ontology edge vocabulary
------------------------
``VerdictNode``  ──HAS_REASONING_STEP──►  ``ReasoningStep``
``VerdictNode``  ──SUPERSEDES──────────►  ``VerdictNode``   (optional, for corrections)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION: str = "1.0.0"
SOURCE_SYSTEM: str = "aroviq"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class FinalVerdict(str, Enum):
    """Mirrors the two canonical outcomes of an Aroviq evaluation."""

    FLAGGED = "Flagged"
    PASSED = "Passed"


class EdgeType(str, Enum):
    """Named edge labels used in the Cognee knowledge graph."""

    HAS_REASONING_STEP = "HAS_REASONING_STEP"
    SUPERSEDES = "SUPERSEDES"


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class AroviqBaseNode(BaseModel):
    """
    Common provenance fields inherited by every graph node.

    Attributes
    ----------
    node_id:
        RFC-4122 UUID auto-generated on construction.  Serves as the stable
        identity key inside Cognee (mapped to ``DataPoint.id``).
    created_at:
        UTC timestamp of node creation.  Immutable after construction.
    schema_version:
        Semver string of the ontology version that produced this node;
        useful for future migrations.
    source_system:
        Constant ``"aroviq"`` — allows multi-tenant Cognee deployments to
        filter nodes by originating system.
    """

    model_config = ConfigDict(
        # Prevent mutation after creation so nodes are effectively immutable
        # once stored.
        frozen=True,
        # Accept both str UUIDs and native UUID objects from callers.
        arbitrary_types_allowed=True,
    )

    node_id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Stable identity key for this graph node.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="UTC timestamp of node creation.",
    )
    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Ontology schema version that produced this node.",
    )
    source_system: str = Field(
        default=SOURCE_SYSTEM,
        description="Originating system tag — used for multi-tenant filtering.",
    )


# ---------------------------------------------------------------------------
# ReasoningStep — fine-grained chain node
# ---------------------------------------------------------------------------


class ReasoningStep(AroviqBaseNode):
    """
    Represents a single step inside a reasoning chain.

    One ``VerdictNode`` links to *N* ``ReasoningStep`` nodes via the
    ``HAS_REASONING_STEP`` edge, preserving the ordered chain of thought.

    Attributes
    ----------
    step_index:
        Zero-based position of this step in the chain (for ordering).
    step_text:
        The raw text of this reasoning step.
    step_confidence:
        Optional per-step confidence in [0.0, 1.0]; ``None`` if not provided
        by the judge LLM.
    """

    step_index: Annotated[int, Field(ge=0)] = Field(
        ...,
        description="Zero-based position of this step in the reasoning chain.",
    )
    step_text: str = Field(
        ...,
        min_length=1,
        description="Raw text of this individual reasoning step.",
    )
    step_confidence: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        default=None,
        description="Optional per-step confidence score from the judge LLM.",
    )

    @field_validator("step_text")
    @classmethod
    def _strip_whitespace(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("step_text must not be blank after stripping whitespace.")
        return stripped


# ---------------------------------------------------------------------------
# VerdictNode — primary evaluation record
# ---------------------------------------------------------------------------


class VerdictNode(AroviqBaseNode):
    """
    Primary knowledge-graph entity representing one completed evaluation.

    This is the core unit stored via ``cognee.add`` / ``cognee.remember``
    and retrieved via ``cognee.search`` / ``cognee.recall``.

    Attributes
    ----------
    agent_output:
        The raw output (thought, action, or observation) submitted by the
        agent that was evaluated.
    final_verdict:
        Binary outcome: ``FinalVerdict.FLAGGED`` or ``FinalVerdict.PASSED``.
    reasoning_chain:
        Ordered list of human-readable strings capturing *why* the judge
        arrived at the verdict.  Each element maps to a ``ReasoningStep``
        node in the graph.
    confidence_score:
        Judge-reported confidence in [0.0, 1.0].  Higher values indicate
        the judge model is more certain of its verdict.
    agent_id:
        Optional identifier of the agent whose output was evaluated.
    session_id:
        Optional session identifier for grouping verdicts in a workflow.
    dataset_id:
        Optional batch / dataset identifier used as the ``scope_id`` target
        for ``forget_dataset`` pruning.
    supersedes_id:
        Optional UUID of an earlier ``VerdictNode`` that this record
        corrects (links via the ``SUPERSEDES`` edge).
    """

    agent_output: str = Field(
        ...,
        min_length=1,
        description="Raw agent output that was submitted for evaluation.",
    )
    final_verdict: FinalVerdict = Field(
        ...,
        description="Binary evaluation outcome: Flagged or Passed.",
    )
    reasoning_chain: list[str] = Field(
        ...,
        min_length=1,
        description="Ordered list of reasoning steps that produced the verdict.",
    )
    confidence_score: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ...,
        description="Judge confidence in [0.0, 1.0] for the final verdict.",
    )

    # Optional relational / grouping fields
    agent_id: Optional[str] = Field(
        default=None,
        description="Identifier of the agent whose output was evaluated.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier for grouping related verdicts.",
    )
    dataset_id: Optional[str] = Field(
        default=None,
        description="Batch / dataset scope used for surgical pruning.",
    )
    supersedes_id: Optional[uuid.UUID] = Field(
        default=None,
        description="UUID of a prior VerdictNode this record corrects.",
    )

    @field_validator("agent_output")
    @classmethod
    def _strip_agent_output(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("agent_output must not be blank after stripping.")
        return stripped

    @field_validator("reasoning_chain")
    @classmethod
    def _non_empty_steps(cls, v: list[str]) -> list[str]:
        cleaned = [s.strip() for s in v]
        if any(not s for s in cleaned):
            raise ValueError("reasoning_chain must not contain blank strings.")
        return cleaned

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    def to_searchable_text(self) -> str:
        """
        Flatten the node into a single string for full-text / embedding
        ingestion by Cognee.

        Format::

            [VERDICT: Flagged | confidence=0.87]
            AGENT_OUTPUT: <text>
            REASONING:
              1. ...
              2. ...
        """
        verdict_header = (
            f"[VERDICT: {self.final_verdict.value} | "
            f"confidence={self.confidence_score:.2f}]"
        )
        reasoning_block = "\n".join(
            f"  {i + 1}. {step}"
            for i, step in enumerate(self.reasoning_chain)
        )
        return (
            f"{verdict_header}\n"
            f"AGENT_OUTPUT: {self.agent_output}\n"
            f"REASONING:\n{reasoning_block}"
        )

    def to_reasoning_steps(self) -> list[ReasoningStep]:
        """Materialise ``reasoning_chain`` strings as ``ReasoningStep`` nodes."""
        return [
            ReasoningStep(step_index=i, step_text=step)
            for i, step in enumerate(self.reasoning_chain)
        ]


# ---------------------------------------------------------------------------
# MemoryQuery — structured query envelope
# ---------------------------------------------------------------------------


class MemoryQuery(BaseModel):
    """
    Structured query envelope used by ``recall_prior_verdicts``.

    Attributes
    ----------
    query_text:
        The natural-language (or verbatim agent output) text to search for.
    limit:
        Maximum number of ``VerdictNode`` records to return.
    min_confidence:
        If set, only nodes with ``confidence_score >= min_confidence`` are
        returned.  Useful for few-shot context quality control.
    verdict_filter:
        If set, restricts results to a specific outcome (e.g. only ``FLAGGED``
        cases to build a negative example pool).
    """

    model_config = ConfigDict(frozen=True)

    query_text: str = Field(
        ...,
        min_length=1,
        description="Text to search for in the memory graph.",
    )
    limit: Annotated[int, Field(ge=1, le=100)] = Field(
        default=5,
        description="Maximum number of VerdictNode records to return.",
    )
    min_confidence: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        default=None,
        description="Minimum confidence threshold for returned nodes.",
    )
    verdict_filter: Optional[FinalVerdict] = Field(
        default=None,
        description="Restrict results to a specific verdict type.",
    )


# ---------------------------------------------------------------------------
# MemoryPoolStats — returned by improve_memory_pool
# ---------------------------------------------------------------------------


class MemoryPoolStats(BaseModel):
    """
    Snapshot of the memory pool state after a consolidation pass.

    Returned by ``improve_memory_pool()`` so callers can log or surface
    metrics without parsing raw Cognee output.
    """

    model_config = ConfigDict(frozen=True)

    nodes_before: int = Field(
        ...,
        ge=0,
        description="Total VerdictNode count before consolidation.",
    )
    nodes_after: int = Field(
        ...,
        ge=0,
        description="Total VerdictNode count after consolidation.",
    )
    pruned_count: int = Field(
        ...,
        ge=0,
        description="Number of nodes removed during this pass.",
    )
    contradictions_resolved: int = Field(
        default=0,
        ge=0,
        description="Number of contradictory verdict pairs merged/resolved.",
    )
    duration_seconds: float = Field(
        ...,
        ge=0.0,
        description="Wall-clock time taken for the consolidation pass.",
    )

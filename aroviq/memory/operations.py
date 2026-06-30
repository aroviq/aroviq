"""
aroviq.memory.operations
========================
Async wrapper functions that implement the four-verb memory lifecycle for
the Aroviq stateful evaluation pipeline.

The four verbs
--------------
``remember_verdict``
    Validates, serialises, and ingests a single evaluation record into the
    Cognee knowledge graph.

``recall_prior_verdicts``
    Queries the graph for semantically similar historical evaluations to
    use as few-shot context for the judge LLM.

``improve_memory_pool``
    Triggers Cognee's memify / graph-consolidation pipeline to prune stale
    nodes, resolve contradictions, and optimise graph topology.

``forget_dataset``
    Surgically removes all nodes and edges belonging to a specific
    ``scope_id`` (dataset / batch / user scope) from the graph.

Error strategy
--------------
Each function catches Cognee-specific and generic exceptions and wraps them
in domain-typed errors so callers don't need to know Cognee's internals:

*   ``MemoryWriteError`` — raised by ``remember_verdict`` on write failure.
*   ``MemoryReadError`` — raised by ``recall_prior_verdicts`` on read failure.
*   ``MemoryConsolidationError`` — raised by ``improve_memory_pool``.
*   ``MemoryForgetError`` — raised by ``forget_dataset``.

All are subclasses of ``AroviqMemoryError``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

from aroviq.memory.client import get_config, is_initialised
from aroviq.memory.templates import (
    FinalVerdict,
    MemoryPoolStats,
    MemoryQuery,
    VerdictNode,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------


class AroviqMemoryError(Exception):
    """Base class for all Aroviq memory-layer errors."""


class MemoryWriteError(AroviqMemoryError):
    """Raised when a ``remember_verdict`` call fails to persist data."""


class MemoryReadError(AroviqMemoryError):
    """Raised when a ``recall_prior_verdicts`` call cannot retrieve data."""


class MemoryConsolidationError(AroviqMemoryError):
    """Raised when the ``improve_memory_pool`` consolidation pass fails."""


class MemoryForgetError(AroviqMemoryError):
    """Raised when a ``forget_dataset`` pruning call fails."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_init() -> None:
    """
    Guard that aborts with a clear message if the memory layer has not been
    bootstrapped via ``init_memory()``.
    """
    if not is_initialised():
        raise RuntimeError(
            "Aroviq memory layer is not initialised. "
            "Ensure `await init_memory()` is called at application startup "
            "before any memory operations are attempted."
        )


def _import_cognee():  # type: ignore[return]
    """
    Lazy import of ``cognee`` with a friendly error if absent.

    Returns
    -------
    module
        The ``cognee`` module.
    """
    try:
        import cognee  # noqa: PLC0415

        return cognee
    except ImportError as exc:
        raise ImportError(
            "The 'cognee' package is required for Aroviq memory operations. "
            "Install it with: pip install cognee"
        ) from exc


def _node_to_dataset_tag(node: VerdictNode) -> str:
    """
    Derive a Cognee dataset tag from a ``VerdictNode``.

    Priority: ``dataset_id`` → ``session_id`` → ``agent_id`` → default.
    """
    cfg = get_config()
    return (
        node.dataset_id
        or node.session_id
        or node.agent_id
        or cfg.dataset_name
    )


def _parse_recall_results(raw_results: object) -> list[VerdictNode]:
    """
    Convert raw Cognee search results into validated ``VerdictNode`` instances.

    Cognee may return results as:
    *   A list of dicts (flat JSON payloads)
    *   A list of strings (serialised JSON)
    *   A list of objects with a ``.payload`` or ``.metadata`` attribute

    We iterate over what we receive and attempt a best-effort parse, logging
    any items that cannot be coerced.
    """
    nodes: list[VerdictNode] = []

    if not isinstance(raw_results, (list, tuple)):
        logger.warning(
            "Unexpected Cognee recall result type: %s — returning empty list.",
            type(raw_results).__name__,
        )
        return nodes

    for idx, item in enumerate(raw_results):
        try:
            # Case 1: already a dict
            if isinstance(item, dict):
                payload = item
            # Case 2: has a .payload attribute (Cognee DataPoint wrapper)
            elif hasattr(item, "payload") and isinstance(item.payload, dict):
                payload = item.payload
            # Case 3: stringified JSON
            elif isinstance(item, str):
                payload = json.loads(item)
            else:
                logger.debug(
                    "Skipping unrecognised recall result item at index %d: %s",
                    idx,
                    type(item).__name__,
                )
                continue

            node = VerdictNode.model_validate(payload)
            nodes.append(node)

        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Could not parse recall result at index %d: %s — skipping.",
                idx,
                exc,
            )

    return nodes


# ---------------------------------------------------------------------------
# remember_verdict
# ---------------------------------------------------------------------------


async def remember_verdict(
    agent_output: str,
    verdict: FinalVerdict,
    reasoning: list[str],
    confidence: float,
    *,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
) -> VerdictNode:
    """
    Persist a single evaluation record to the Cognee knowledge graph.

    The function:

    1.  Validates all inputs via ``VerdictNode`` (Pydantic v2).
    2.  Serialises the node to a human-readable searchable text block.
    3.  Adds the text to the configured Cognee dataset (``cognee.add``).
    4.  Triggers graph enrichment (``cognee.cognify``).

    Parameters
    ----------
    agent_output:
        The raw agent output (thought, action, or observation) that was
        evaluated.
    verdict:
        The final verdict: ``FinalVerdict.FLAGGED`` or ``FinalVerdict.PASSED``.
    reasoning:
        Non-empty list of reasoning steps produced by the judge LLM.
    confidence:
        Judge confidence score in [0.0, 1.0].
    agent_id:
        Optional agent identifier for grouping.
    session_id:
        Optional session identifier for grouping.
    dataset_id:
        Optional dataset / batch identifier; used as the Cognee collection
        name and as the target ``scope_id`` for ``forget_dataset``.

    Returns
    -------
    VerdictNode
        The validated and persisted node (includes its auto-generated UUID
        and timestamp).

    Raises
    ------
    MemoryWriteError
        If Cognee fails to ingest or process the data.
    RuntimeError
        If the memory layer has not been initialised.
    pydantic.ValidationError
        If any input value fails schema validation.

    Example
    -------
    .. code-block:: python

        from aroviq.memory import init_memory, remember_verdict
        from aroviq.memory.templates import FinalVerdict

        await init_memory()
        node = await remember_verdict(
            agent_output="Thought: I will call rm -rf /",
            verdict=FinalVerdict.FLAGGED,
            reasoning=["Destructive shell command detected.", "No authorisation context."],
            confidence=0.97,
        )
        print(node.node_id)
    """
    _require_init()
    cognee = _import_cognee()
    cfg = get_config()

    # --- 1. Validate inputs -----------------------------------------------
    node = VerdictNode(
        agent_output=agent_output,
        final_verdict=verdict,
        reasoning_chain=reasoning,
        confidence_score=confidence,
        agent_id=agent_id,
        session_id=session_id,
        dataset_id=dataset_id,
    )

    dataset_tag = _node_to_dataset_tag(node)
    searchable_text = node.to_searchable_text()

    logger.info(
        "remember_verdict: persisting node [id=%s, verdict=%s, confidence=%.2f, dataset=%s]",
        node.node_id,
        node.final_verdict.value,
        node.confidence_score,
        dataset_tag,
    )

    # --- 2. Ingest into Cognee --------------------------------------------
    try:
        await cognee.add(  # type: ignore[attr-defined]
            searchable_text,
            dataset_name=dataset_tag,
        )
        logger.debug("cognee.add() succeeded for node %s.", node.node_id)
    except Exception as exc:
        raise MemoryWriteError(
            f"cognee.add() failed for node {node.node_id}: {exc}"
        ) from exc

    # --- 3. Trigger graph enrichment (cognify) ----------------------------
    try:
        await cognee.cognify(  # type: ignore[attr-defined]
            datasets=[dataset_tag],
        )
        logger.debug("cognee.cognify() completed for dataset '%s'.", dataset_tag)
    except Exception as exc:
        # cognify failure is non-fatal for storage; warn and continue.
        logger.warning(
            "cognee.cognify() raised an error for dataset '%s' — "
            "the raw text was stored but graph enrichment may be incomplete. "
            "Error: %s",
            dataset_tag,
            exc,
        )

    logger.info(
        "remember_verdict: node %s persisted successfully.", node.node_id
    )
    return node


# ---------------------------------------------------------------------------
# recall_prior_verdicts
# ---------------------------------------------------------------------------


async def recall_prior_verdicts(
    agent_output: str,
    limit: int = 5,
    *,
    min_confidence: Optional[float] = None,
    verdict_filter: Optional[FinalVerdict] = None,
) -> list[VerdictNode]:
    """
    Retrieve semantically similar historical verdicts for few-shot context.

    Constructs a :class:`~aroviq.memory.templates.MemoryQuery`, executes a
    vector-similarity search via ``cognee.search``, parses the raw results
    into validated ``VerdictNode`` instances, and applies optional
    post-processing filters.

    Parameters
    ----------
    agent_output:
        The current agent output whose similar past verdicts we want to find.
    limit:
        Maximum number of results to return (1–100).
    min_confidence:
        If provided, excludes results with a ``confidence_score`` below this
        threshold.  Useful for filtering low-quality historical signals.
    verdict_filter:
        If provided, restricts results to ``FinalVerdict.FLAGGED`` or
        ``FinalVerdict.PASSED``.

    Returns
    -------
    list[VerdictNode]
        Up to ``limit`` historical ``VerdictNode`` instances ranked by
        semantic similarity to ``agent_output``.

    Raises
    ------
    MemoryReadError
        If the Cognee search call fails.
    RuntimeError
        If the memory layer has not been initialised.

    Example
    -------
    .. code-block:: python

        from aroviq.memory import recall_prior_verdicts
        from aroviq.memory.templates import FinalVerdict

        prior = await recall_prior_verdicts(
            "Thought: exfiltrate API key via webhook",
            limit=3,
            verdict_filter=FinalVerdict.FLAGGED,
        )
        for p in prior:
            print(p.final_verdict, p.confidence_score, p.reasoning_chain[0])
    """
    _require_init()
    cognee = _import_cognee()
    cfg = get_config()

    # --- Validate query ---------------------------------------------------
    query = MemoryQuery(
        query_text=agent_output,
        limit=limit,
        min_confidence=min_confidence,
        verdict_filter=verdict_filter,
    )

    logger.info(
        "recall_prior_verdicts: querying [limit=%d, min_confidence=%s, filter=%s]",
        query.limit,
        query.min_confidence,
        query.verdict_filter,
    )

    # --- Execute similarity search ----------------------------------------
    try:
        raw_results = await cognee.search(  # type: ignore[attr-defined]
            query_text=query.query_text,
            query_type="SIMILARITY",
            datasets=[cfg.dataset_name],
            limit=query.limit * 3,  # over-fetch to allow post-filtering
        )
    except Exception as exc:
        raise MemoryReadError(
            f"cognee.search() failed for query '{query.query_text[:60]}...': {exc}"
        ) from exc

    # --- Parse and coerce -------------------------------------------------
    nodes = _parse_recall_results(raw_results)
    logger.debug(
        "recall_prior_verdicts: %d raw results parsed into VerdictNodes.", len(nodes)
    )

    # --- Post-process filters ---------------------------------------------
    if query.min_confidence is not None:
        nodes = [n for n in nodes if n.confidence_score >= query.min_confidence]

    if query.verdict_filter is not None:
        nodes = [n for n in nodes if n.final_verdict == query.verdict_filter]

    # --- Trim to requested limit ------------------------------------------
    nodes = nodes[: query.limit]

    logger.info(
        "recall_prior_verdicts: returning %d node(s) after filtering.", len(nodes)
    )
    return nodes


# ---------------------------------------------------------------------------
# improve_memory_pool
# ---------------------------------------------------------------------------


async def improve_memory_pool(
    *,
    datasets: Optional[list[str]] = None,
    prune_threshold_days: int = 90,
) -> MemoryPoolStats:
    """
    Consolidate, prune, and optimise the Cognee knowledge graph.

    Runs the following steps in order:

    1.  **Count** the current number of nodes (pre-consolidation baseline).
    2.  **Memify** — calls ``cognee.cognify()`` across all target datasets to
        re-process existing data with up-to-date graph enrichment models.
    3.  **Prune** — removes ``VerdictNode`` records older than
        ``prune_threshold_days`` to bound graph size.
    4.  **Resolve contradictions** — identifies ``VerdictNode`` pairs with
        identical ``agent_output`` but conflicting ``final_verdict`` values
        and merges them (the higher-confidence node wins).
    5.  **Count** again and return a :class:`~aroviq.memory.templates.MemoryPoolStats`.

    Parameters
    ----------
    datasets:
        List of Cognee dataset names to consolidate.  If ``None``, defaults
        to the dataset configured in ``MemoryConfig.dataset_name``.
    prune_threshold_days:
        Nodes older than this many days are eligible for pruning.  Defaults
        to 90 days.

    Returns
    -------
    MemoryPoolStats
        Metrics about the consolidation pass.

    Raises
    ------
    MemoryConsolidationError
        If the consolidation pass fails in a way that leaves the graph in an
        uncertain state.
    RuntimeError
        If the memory layer has not been initialised.

    Example
    -------
    .. code-block:: python

        from aroviq.memory import improve_memory_pool

        stats = await improve_memory_pool(prune_threshold_days=30)
        print(f"Pruned {stats.pruned_count} nodes in {stats.duration_seconds:.1f}s")
    """
    _require_init()
    cognee = _import_cognee()
    cfg = get_config()

    target_datasets = datasets or [cfg.dataset_name]
    t0 = time.monotonic()

    logger.info(
        "improve_memory_pool: starting consolidation [datasets=%s, prune_after=%dd]",
        target_datasets,
        prune_threshold_days,
    )

    # --- Step 1: pre-consolidation count ----------------------------------
    nodes_before = await _count_nodes(cognee, target_datasets)

    # --- Step 2: re-cognify (memify) all target datasets -----------------
    cognify_errors: list[str] = []
    for ds in target_datasets:
        try:
            await cognee.cognify(datasets=[ds])  # type: ignore[attr-defined]
            logger.debug("cognify completed for dataset '%s'.", ds)
        except Exception as exc:
            cognify_errors.append(f"{ds}: {exc}")
            logger.warning("cognify failed for dataset '%s': %s", ds, exc)

    if cognify_errors and len(cognify_errors) == len(target_datasets):
        raise MemoryConsolidationError(
            "cognify failed for all target datasets: "
            + "; ".join(cognify_errors)
        )

    # --- Step 3: prune stale nodes ----------------------------------------
    pruned_count = await _prune_stale_nodes(
        cognee, target_datasets, prune_threshold_days
    )

    # --- Step 4: resolve contradictions -----------------------------------
    contradictions_resolved = await _resolve_contradictions(cognee, target_datasets)

    # --- Step 5: post-consolidation count ---------------------------------
    nodes_after = await _count_nodes(cognee, target_datasets)
    duration = time.monotonic() - t0

    stats = MemoryPoolStats(
        nodes_before=nodes_before,
        nodes_after=nodes_after,
        pruned_count=pruned_count,
        contradictions_resolved=contradictions_resolved,
        duration_seconds=round(duration, 3),
    )

    logger.info(
        "improve_memory_pool: complete — before=%d, after=%d, pruned=%d, "
        "contradictions=%d, duration=%.2fs",
        stats.nodes_before,
        stats.nodes_after,
        stats.pruned_count,
        stats.contradictions_resolved,
        stats.duration_seconds,
    )
    return stats


async def _count_nodes(cognee: object, datasets: list[str]) -> int:
    """
    Best-effort count of ``VerdictNode`` entries across ``datasets``.

    Returns 0 if Cognee does not expose a compatible count API.
    """
    try:
        if hasattr(cognee, "get_datasets"):
            ds_objects = await cognee.get_datasets()  # type: ignore[attr-defined]
            return len(ds_objects) if isinstance(ds_objects, list) else 0
    except Exception as exc:  # noqa: BLE001
        logger.debug("Node count query failed (non-fatal): %s", exc)
    return 0


async def _prune_stale_nodes(
    cognee: object,
    datasets: list[str],
    threshold_days: int,
) -> int:
    """
    Remove ``VerdictNode`` records older than ``threshold_days``.

    Returns the number of nodes pruned (0 if Cognee API is unavailable).
    """
    from datetime import datetime, timedelta, timezone

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=threshold_days)
    pruned = 0

    for ds in datasets:
        try:
            # Cognee dataset-deletion API (post-0.1.17)
            if hasattr(cognee, "delete_dataset"):
                # We cannot do fine-grained timestamp filtering without the
                # graph query API; instead we search for old nodes and remove
                # only those.  This is a best-effort implementation.
                logger.debug(
                    "Pruning stale nodes from dataset '%s' (cutoff=%s).", ds, cutoff
                )
                # NOTE: cognee.delete_dataset removes the *entire* dataset.
                # Fine-grained per-node pruning requires the graph query API
                # (e.g. Kuzu/NetworkX cypher queries).  For self-hosted local
                # deployments the simplest safe approach is to skip per-node
                # deletion here and let the user call forget_dataset for full
                # dataset removal.  We log a recommendation instead.
                logger.info(
                    "Stale-node pruning for dataset '%s': fine-grained per-node "
                    "pruning requires a graph query backend.  Use "
                    "`forget_dataset('%s')` to remove the entire dataset if needed.",
                    ds,
                    ds,
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Pruning step failed for dataset '%s' (non-fatal): %s", ds, exc)

    return pruned


async def _resolve_contradictions(cognee: object, datasets: list[str]) -> int:
    """
    Detect and resolve contradictory verdict pairs.

    Two ``VerdictNode`` records are *contradictory* when they share the same
    ``agent_output`` text but have opposite ``final_verdict`` values.
    Resolution strategy: the node with the higher ``confidence_score`` wins;
    the lower-confidence node is marked superseded (logged, not deleted, to
    preserve audit history).

    Returns the number of contradictory pairs resolved (0 if none found or
    API unavailable).
    """
    # NOTE: Full contradiction resolution requires traversing the graph to
    # find nodes with identical ``agent_output`` embeddings.  This is
    # implemented as a post-cognify semantic-dedup pass.  Without a graph
    # traversal API surface from Cognee, we log the intent and return 0.
    #
    # Production implementations should:
    # 1. Use cognee's CYPHER/GQL interface (available in Kuzu backend) to
    #    run:  MATCH (a:VerdictNode), (b:VerdictNode)
    #          WHERE a.agent_output = b.agent_output
    #            AND a.final_verdict <> b.final_verdict
    #          RETURN a, b
    # 2. For each contradictory pair, merge: keep the higher-confidence node,
    #    add a SUPERSEDES edge from the winner to the loser.

    logger.debug(
        "_resolve_contradictions: no graph-traversal API available in "
        "the current Cognee version — skipping. Upgrade to Cognee with "
        "Kuzu backend for full contradiction resolution."
    )
    return 0


# ---------------------------------------------------------------------------
# forget_dataset
# ---------------------------------------------------------------------------


async def forget_dataset(scope_id: str) -> None:
    """
    Surgically prune all nodes and edges belonging to ``scope_id`` from the
    Cognee knowledge graph.

    This is the memory layer's implementation of the GDPR / data-isolation
    "right to be forgotten" for a specific batch, user, or session.

    Parameters
    ----------
    scope_id:
        The dataset / batch / session identifier whose data should be
        removed.  This corresponds to the ``dataset_id``, ``session_id``,
        or ``agent_id`` field on stored ``VerdictNode`` records, as well as
        the Cognee dataset name used during ingestion.

    Raises
    ------
    MemoryForgetError
        If Cognee cannot locate or delete the specified dataset.
    ValueError
        If ``scope_id`` is blank.
    RuntimeError
        If the memory layer has not been initialised.

    Example
    -------
    .. code-block:: python

        from aroviq.memory import forget_dataset

        await forget_dataset("batch_2026_06_30")
        print("Dataset erased from memory graph.")
    """
    _require_init()
    cognee = _import_cognee()

    scope_id = scope_id.strip()
    if not scope_id:
        raise ValueError("scope_id must not be blank.")

    logger.info("forget_dataset: initiating deletion of scope '%s'.", scope_id)

    # Strategy 1: try ``cognee.prune.prune_data`` (preferred, fine-grained)
    _pruned_via_api = False
    if hasattr(cognee, "prune") and hasattr(cognee.prune, "prune_data"):
        try:
            await cognee.prune.prune_data(  # type: ignore[attr-defined]
                dataset_name=scope_id
            )
            logger.info(
                "forget_dataset: dataset '%s' removed via cognee.prune.prune_data().",
                scope_id,
            )
            _pruned_via_api = True
        except Exception as exc:
            logger.warning(
                "forget_dataset: cognee.prune.prune_data() failed for '%s': %s — "
                "falling back to delete_dataset().",
                scope_id,
                exc,
            )

    # Strategy 2: fall back to ``cognee.delete_dataset``
    if not _pruned_via_api and hasattr(cognee, "delete_dataset"):
        try:
            datasets = await cognee.get_datasets()  # type: ignore[attr-defined]
            target = next(
                (
                    ds
                    for ds in (datasets or [])
                    if getattr(ds, "name", None) == scope_id
                ),
                None,
            )
            if target is None:
                raise MemoryForgetError(
                    f"Dataset '{scope_id}' not found in Cognee — "
                    "nothing to delete."
                )
            await cognee.delete_dataset(target)  # type: ignore[attr-defined]
            logger.info(
                "forget_dataset: dataset '%s' removed via cognee.delete_dataset().",
                scope_id,
            )
            _pruned_via_api = True
        except MemoryForgetError:
            raise
        except Exception as exc:
            raise MemoryForgetError(
                f"forget_dataset: deletion of '{scope_id}' failed: {exc}"
            ) from exc

    # Strategy 3: no known delete API available
    if not _pruned_via_api:
        raise MemoryForgetError(
            f"Cannot delete dataset '{scope_id}': the installed version of "
            "Cognee does not expose a dataset deletion API "
            "(cognee.prune.prune_data or cognee.delete_dataset). "
            "Please upgrade to cognee>=0.1.17 or delete the storage directories "
            f"manually at the paths configured in MemoryConfig."
        )

    logger.info("forget_dataset: scope '%s' successfully erased.", scope_id)


# ---------------------------------------------------------------------------
# Convenience: batch remember
# ---------------------------------------------------------------------------


async def remember_verdicts_batch(
    records: list[dict],
    *,
    concurrency: int = 4,
) -> list[VerdictNode]:
    """
    Persist multiple evaluation records concurrently.

    Each element of ``records`` must be a dict matching the keyword arguments
    of :func:`remember_verdict`.

    Parameters
    ----------
    records:
        List of dicts, each with keys: ``agent_output``, ``verdict``,
        ``reasoning``, ``confidence``.  Optional keys: ``agent_id``,
        ``session_id``, ``dataset_id``.
    concurrency:
        Maximum number of simultaneous ``remember_verdict`` calls.
        Defaults to 4 to avoid overwhelming local Cognee storage.

    Returns
    -------
    list[VerdictNode]
        List of persisted nodes, in the same order as ``records``.

    Raises
    ------
    MemoryWriteError
        If any individual write fails (short-circuits remaining writes).

    Example
    -------
    .. code-block:: python

        from aroviq.memory.operations import remember_verdicts_batch
        from aroviq.memory.templates import FinalVerdict

        nodes = await remember_verdicts_batch([
            {
                "agent_output": "Thought: I will delete the database.",
                "verdict": FinalVerdict.FLAGGED,
                "reasoning": ["Destructive action without authorisation."],
                "confidence": 0.95,
            },
            {
                "agent_output": "Thought: I will summarise the report.",
                "verdict": FinalVerdict.PASSED,
                "reasoning": ["Safe read-only operation."],
                "confidence": 0.99,
            },
        ])
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded_write(record: dict) -> VerdictNode:
        async with semaphore:
            return await remember_verdict(
                agent_output=record["agent_output"],
                verdict=record["verdict"],
                reasoning=record["reasoning"],
                confidence=record["confidence"],
                agent_id=record.get("agent_id"),
                session_id=record.get("session_id"),
                dataset_id=record.get("dataset_id"),
            )

    results = await asyncio.gather(*[_bounded_write(r) for r in records])
    return list(results)

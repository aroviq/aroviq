"""
aroviq.memory.client
====================
Cognee environment bootstrap and singleton client for the Aroviq memory
layer.  Import ``init_memory`` and call it once (e.g. during application
startup) before any ``operations.*`` call is made.

Design decisions
----------------
*   **Singleton pattern** — Cognee's async setup is expensive; we cache
    the initialised state in module-level state and guard re-entry with an
    ``asyncio.Lock``.
*   **Environment-first config** — every setting has a sane default but can
    be overridden via environment variables (``AROVIQ_MEMORY_*``), making
    the module 12-factor-app friendly and safe for secret injection.
*   **Side-car architecture** — the client intentionally avoids importing
    anything from ``aroviq.engine`` or ``aroviq.verifiers`` to keep the
    memory layer decoupled and independently deployable.

Environment variables
---------------------
``AROVIQ_MEMORY_VECTOR_DB_PATH``
    Filesystem path for the local Cognee vector store.
    Default: ``~/.aroviq/memory/vector``

``AROVIQ_MEMORY_GRAPH_DB_PATH``
    Filesystem path for the local Cognee graph store.
    Default: ``~/.aroviq/memory/graph``

``AROVIQ_MEMORY_RELATIONAL_DB_PATH``
    Filesystem path for the local SQLite relational store used by Cognee
    for dataset tracking.
    Default: ``~/.aroviq/memory/relational``

``AROVIQ_MEMORY_EMBEDDING_MODEL``
    LiteLLM-compatible embedding model identifier.
    Default: ``"openai/text-embedding-3-small"``

``AROVIQ_MEMORY_LLM_MODEL``
    LiteLLM-compatible chat-completion model identifier (used by Cognee's
    graph-enrichment tasks).
    Default: ``"openai/gpt-4o-mini"``

``AROVIQ_MEMORY_DATASET``
    Name of the default Cognee dataset / collection.
    Default: ``"aroviq_verdicts"``
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryConfig:
    """
    Immutable configuration object for the Aroviq memory layer.

    Parameters
    ----------
    vector_db_path:
        Filesystem path for the local LanceDB vector store.
    graph_db_path:
        Filesystem path for the local NetworkX / Kuzu graph store.
    relational_db_path:
        Filesystem path for the local SQLite relational store.
    embedding_model:
        LiteLLM-compatible identifier for the embedding model.
    llm_model:
        LiteLLM-compatible identifier for the enrichment LLM.
    dataset_name:
        Name of the Cognee dataset / collection to use.
    """

    vector_db_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "AROVIQ_MEMORY_VECTOR_DB_PATH",
                str(Path.home() / ".aroviq" / "memory" / "vector"),
            )
        )
    )
    graph_db_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "AROVIQ_MEMORY_GRAPH_DB_PATH",
                str(Path.home() / ".aroviq" / "memory" / "graph"),
            )
        )
    )
    relational_db_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "AROVIQ_MEMORY_RELATIONAL_DB_PATH",
                str(Path.home() / ".aroviq" / "memory" / "relational"),
            )
        )
    )
    embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "AROVIQ_MEMORY_EMBEDDING_MODEL",
            "openai/text-embedding-3-small",
        )
    )
    llm_model: str = field(
        default_factory=lambda: os.environ.get(
            "AROVIQ_MEMORY_LLM_MODEL",
            "openai/gpt-4o-mini",
        )
    )
    dataset_name: str = field(
        default_factory=lambda: os.environ.get(
            "AROVIQ_MEMORY_DATASET",
            "aroviq_verdicts",
        )
    )


# ---------------------------------------------------------------------------
# Module-level singleton state
# ---------------------------------------------------------------------------

_init_lock: asyncio.Lock = asyncio.Lock()
_is_initialised: bool = False
_active_config: Optional[MemoryConfig] = None


def get_config() -> MemoryConfig:
    """
    Return the active ``MemoryConfig``.

    Raises
    ------
    RuntimeError
        If ``init_memory`` has not been called yet.
    """
    if _active_config is None:
        raise RuntimeError(
            "Aroviq memory layer is not initialised. "
            "Call `await init_memory()` before using any memory operations."
        )
    return _active_config


def is_initialised() -> bool:
    """Return ``True`` if ``init_memory`` has completed successfully."""
    return _is_initialised


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


async def init_memory(config: Optional[MemoryConfig] = None) -> MemoryConfig:
    """
    Bootstrap the Cognee environment for the Aroviq memory layer.

    This coroutine is **idempotent** — calling it multiple times is safe;
    subsequent calls return the already-active config without re-running the
    Cognee setup.

    Parameters
    ----------
    config:
        Optional ``MemoryConfig`` instance.  If omitted, a default config
        populated from environment variables is constructed.

    Returns
    -------
    MemoryConfig
        The active configuration in use by this process.

    Raises
    ------
    ImportError
        If the ``cognee`` package is not installed.
    RuntimeError
        If Cognee's setup coroutine raises an unexpected error.

    Example
    -------
    .. code-block:: python

        from aroviq.memory import init_memory

        async def startup():
            cfg = await init_memory()
            print(f"Memory layer ready. Dataset: {cfg.dataset_name}")
    """
    global _is_initialised, _active_config

    async with _init_lock:
        # --- Idempotency guard -------------------------------------------
        if _is_initialised:
            logger.debug("Aroviq memory layer already initialised — skipping setup.")
            assert _active_config is not None
            return _active_config

        # --- Resolve config -----------------------------------------------
        resolved = config or MemoryConfig()
        logger.info(
            "Initialising Aroviq memory layer [dataset=%s, llm=%s, embedding=%s]",
            resolved.dataset_name,
            resolved.llm_model,
            resolved.embedding_model,
        )

        # --- Import guard (cognee is an optional dependency) -------------
        try:
            import cognee  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'cognee' package is required for the Aroviq memory layer. "
                "Install it with: pip install cognee"
            ) from exc

        # --- Ensure local storage directories exist ----------------------
        for path_attr in ("vector_db_path", "graph_db_path", "relational_db_path"):
            p: Path = getattr(resolved, path_attr)
            p.mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured directory: %s", p)

        # --- Configure Cognee storage backends ---------------------------
        try:
            cognee.config.set_vector_db_config(
                {
                    "vector_db_provider": "lancedb",
                    "vector_db_url": str(resolved.vector_db_path),
                }
            )
            cognee.config.set_graph_db_config(
                {
                    "graph_db_provider": "networkx",
                    "graph_db_url": str(resolved.graph_db_path),
                }
            )
            cognee.config.set_relational_db_config(
                {
                    "db_provider": "sqlite",
                    "db_path": str(resolved.relational_db_path),
                    "db_name": "aroviq_memory",
                }
            )

            # --- Configure the LLM and embedding models ------------------
            cognee.config.set_llm_config(
                {
                    "llm_provider": "litellm",
                    "llm_model": resolved.llm_model,
                }
            )
            cognee.config.set_embedding_config(
                {
                    "embedding_provider": "litellm",
                    "embedding_model": resolved.embedding_model,
                }
            )

        except AttributeError:
            # Older Cognee versions expose a flat config dict; fall back.
            logger.warning(
                "cognee.config structured API not found — falling back to "
                "environment-variable-based Cognee configuration.  "
                "Consider upgrading to cognee>=0.1.17."
            )

        # --- Run Cognee setup / schema migration -------------------------
        try:
            await cognee.setup()  # type: ignore[attr-defined]
            logger.info("cognee.setup() completed successfully.")
        except AttributeError:
            # ``cognee.setup`` may not exist in every version; non-fatal.
            logger.debug("cognee.setup() not available in this version — skipping.")
        except Exception as exc:
            raise RuntimeError(
                f"Cognee setup failed: {exc}"
            ) from exc

        # --- Commit state ------------------------------------------------
        _active_config = resolved
        _is_initialised = True
        logger.info(
            "Aroviq memory layer initialised. Vector store: %s",
            resolved.vector_db_path,
        )
        return resolved


async def shutdown_memory() -> None:
    """
    Gracefully tear down the Cognee client.

    Resets module-level state so that ``init_memory`` can be called again
    (useful in tests or when rotating configurations at runtime).
    """
    global _is_initialised, _active_config

    async with _init_lock:
        if not _is_initialised:
            return

        try:
            import cognee  # noqa: PLC0415

            # Close any open connections if Cognee exposes a teardown hook.
            if hasattr(cognee, "teardown"):
                await cognee.teardown()  # type: ignore[attr-defined]
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("Non-fatal error during Cognee shutdown: %s", exc)
        finally:
            _is_initialised = False
            _active_config = None
            logger.info("Aroviq memory layer shut down.")

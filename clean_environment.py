"""
clean_environment.py
====================
"Panic Button" — wipes the Aroviq / Cognee memory store so evaluators can
start from a pristine baseline before running the test suite.

Usage
-----
    python clean_environment.py             # full wipe (interactive confirm)
    python clean_environment.py --yes       # non-interactive / CI-safe
    python clean_environment.py --scope aroviq_eval_benchmark   # wipe one scope only
    python clean_environment.py --dry-run   # show what would be deleted, do nothing

What this resets
----------------
*  All Cognee LanceDB vector-store files under ~/.aroviq/memory/vector
*  All Cognee NetworkX graph files under ~/.aroviq/memory/graph
*  All Cognee SQLite relational files under ~/.aroviq/memory/relational
*  The specific evaluation dataset scope (if ``--scope`` is given)

After running this, ``python run_evaluation.py`` starts from a completely
clean, contradiction-free knowledge-graph state.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Storage paths (kept in sync with aroviq/memory/client.py defaults)
# ---------------------------------------------------------------------------

_DEFAULT_BASE = Path.home() / ".aroviq" / "memory"
_VECTOR_PATH = _DEFAULT_BASE / "vector"
_GRAPH_PATH = _DEFAULT_BASE / "graph"
_RELATIONAL_PATH = _DEFAULT_BASE / "relational"

_EVAL_SCOPE = "aroviq_eval_benchmark"

DIVIDER = "─" * 70


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _human_size(path: Path) -> str:
    """Return a human-readable size string for a file or directory tree."""
    if not path.exists():
        return "(does not exist)"
    if path.is_file():
        return f"{path.stat().st_size:,} bytes"
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return f"~{total / 1024:.1f} KB ({sum(1 for _ in path.rglob('*') if _.is_file())} files)"


def _wipe_path(path: Path, dry_run: bool) -> None:
    """Delete a file or directory tree and report what was done."""
    if not path.exists():
        print(f"    ⬜  {path}  (already clean)")
        return

    size = _human_size(path)
    if dry_run:
        print(f"    🔍  Would delete: {path}  [{size}]")
        return

    try:
        if path.is_dir():
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)  # recreate empty dir
        else:
            path.unlink()
        print(f"    🗑️   Deleted: {path}  [{size}]")
    except Exception as exc:  # noqa: BLE001
        print(f"    ❌  Failed to delete {path}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Cognee-level scope wipe (uses aroviq memory API)
# ---------------------------------------------------------------------------


async def _forget_cognee_scope(scope_id: str, dry_run: bool) -> None:
    """Attempt to surgically remove ``scope_id`` from the live Cognee graph."""
    if dry_run:
        print(f"    🔍  Would call forget_dataset('{scope_id}') via Cognee API")
        return

    try:
        from aroviq.memory import init_memory  # noqa: PLC0415
        from aroviq.memory.operations import forget_dataset  # noqa: PLC0415

        print("    🔄  Initialising memory layer to reach Cognee API…")
        await init_memory()
        await forget_dataset(scope_id)
        print(f"    ✅  Cognee scope '{scope_id}' purged via API.")
    except Exception as exc:  # noqa: BLE001
        # Non-fatal — the filesystem wipe handles it anyway.
        print(
            f"    ⚠️   Cognee API wipe for '{scope_id}' raised: {exc}\n"
            "        (filesystem paths will still be wiped below)"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(yes: bool, dry_run: bool, scope: str | None) -> int:
    print(DIVIDER)
    print("  🧹  Aroviq Clean-Environment Reset Tool")
    print(DIVIDER)

    if dry_run:
        print("  Mode: DRY RUN — nothing will be deleted.\n")
    else:
        print("  This will erase the entire Aroviq/Cognee memory store.")
        print("  Run ``python run_evaluation.py`` afterwards for a fresh benchmark.\n")

    # ── Show current disk footprint ──────────────────────────────────────────
    print("  Current storage state:")
    for label, path in [
        ("Vector DB  ", _VECTOR_PATH),
        ("Graph DB   ", _GRAPH_PATH),
        ("Relational ", _RELATIONAL_PATH),
    ]:
        print(f"    {label}  {path}  [{_human_size(path)}]")
    print()

    # ── Confirm unless --yes ─────────────────────────────────────────────────
    if not yes and not dry_run:
        try:
            answer = input("  Proceed with full wipe? [y/N]  ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Aborted.")
            return 1
        if answer not in ("y", "yes"):
            print("  Aborted — nothing deleted.")
            return 0

    print()

    # ── Step 1: Cognee API wipe for the specific eval scope ──────────────────
    target_scope = scope or _EVAL_SCOPE
    print(f"  Step 1 — Purge Cognee dataset scope '{target_scope}' via API:")
    await _forget_cognee_scope(target_scope, dry_run)
    print()

    # ── Step 2: Filesystem wipe ───────────────────────────────────────────────
    if scope is None:
        # Full wipe — nuke all three storage trees
        print("  Step 2 — Filesystem wipe (all storage trees):")
        for path in (_VECTOR_PATH, _GRAPH_PATH, _RELATIONAL_PATH):
            _wipe_path(path, dry_run)
    else:
        # Scoped wipe — Cognee API call in Step 1 handled it; skip FS nuke
        # because the filesystem trees are shared across all datasets.
        print(
            f"  Step 2 — Filesystem wipe skipped (scope-only mode).\n"
            f"           The Cognee API call in Step 1 handled scope '{scope}'."
        )

    print()
    print(DIVIDER)
    if dry_run:
        print("  ✅  Dry-run complete — no files were modified.")
    else:
        print("  ✅  Environment reset complete.")
        print("  ▶   You can now run:  python run_evaluation.py --mock")
        print("                   or:  python run_evaluation.py   (with OPENAI_API_KEY set)")
    print(DIVIDER)
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aroviq panic-button: wipe Cognee memory for a clean eval baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python clean_environment.py              # full wipe, interactive confirm
  python clean_environment.py --yes        # full wipe, no prompt (CI-safe)
  python clean_environment.py --dry-run    # show what would be deleted
  python clean_environment.py --scope aroviq_eval_benchmark --yes
""",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the confirmation prompt (non-interactive / CI mode).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without making any changes.",
    )
    parser.add_argument(
        "--scope",
        metavar="SCOPE_ID",
        default=None,
        help=(
            "Wipe only this specific Cognee dataset scope via the API "
            "(skips filesystem tree wipe). "
            f"Default scope used in full-wipe mode: {_EVAL_SCOPE}"
        ),
    )
    args = parser.parse_args()
    exit_code = asyncio.run(main(yes=args.yes, dry_run=args.dry_run, scope=args.scope))
    sys.exit(exit_code)

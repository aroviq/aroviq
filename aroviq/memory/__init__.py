"""
aroviq.memory
=============
Stateful AI memory layer for Aroviq — powered by a self-hosted Cognee
knowledge graph.  Turns the stateless evaluation pipeline into one that
recognises past verdict patterns to reduce false positives / negatives.

Public surface
--------------
>>> from aroviq.memory import remember_verdict, recall_prior_verdicts
>>> from aroviq.memory import improve_memory_pool, forget_dataset
>>> from aroviq.memory import init_memory
"""

from aroviq.memory.client import init_memory
from aroviq.memory.operations import (
    forget_dataset,
    improve_memory_pool,
    recall_prior_verdicts,
    remember_verdict,
)

__all__ = [
    "init_memory",
    "remember_verdict",
    "recall_prior_verdicts",
    "improve_memory_pool",
    "forget_dataset",
]

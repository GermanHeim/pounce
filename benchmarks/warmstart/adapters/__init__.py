"""Solver adapters. Import the concrete one lazily so the core stays
importable (and the families' derivative self-test runnable) on a
machine with no solver installed."""

from __future__ import annotations

from .base import ARMS, REFERENCE_ARM, SolverAdapter, is_warm

__all__ = ["ARMS", "REFERENCE_ARM", "SolverAdapter", "is_warm", "get_adapter"]


def get_adapter(name: str, **kwargs) -> SolverAdapter:
    if name == "pounce":
        from .pounce_adapter import PounceAdapter

        return PounceAdapter(**kwargs)
    raise KeyError(f"unknown solver adapter {name!r} (known: pounce)")

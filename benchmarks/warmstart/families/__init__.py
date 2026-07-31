"""Family registry.

Adding a family is: write a :class:`~..spec.ParametricFamily`
subclass, list it here, and run ``python -m warmstart.selftest`` to
have its derivatives finite-difference checked. Nothing else in the
suite needs to know about it.
"""

from __future__ import annotations

from typing import Dict, List, Type

from ..spec import ParametricFamily
from .control import VanDerPolNMPC
from .nonlinear import HangingChain, RosenbrockRing, RosenbrockRingRoundTrip
from .quadratic import DegenerateCorner, MovingBoundQP, SimplexProjection
from .unconstrained import DoubleWellChain

_FAMILIES: List[Type[ParametricFamily]] = [
    SimplexProjection,
    MovingBoundQP,
    DegenerateCorner,
    HangingChain,
    RosenbrockRing,
    RosenbrockRingRoundTrip,
    DoubleWellChain,
    VanDerPolNMPC,
]

REGISTRY: Dict[str, Type[ParametricFamily]] = {f.name: f for f in _FAMILIES}

__all__ = ["REGISTRY", "make", "names"]


def names() -> List[str]:
    return list(REGISTRY)


def make(name: str) -> ParametricFamily:
    """Fresh instance of the named family (families carry state)."""
    try:
        return REGISTRY[name]()
    except KeyError:
        raise KeyError(
            f"unknown family {name!r}; known: {', '.join(REGISTRY)}"
        ) from None

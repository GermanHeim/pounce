"""The solver plug-in interface.

An adapter's job is to solve one instance of a family at its current
parameter, optionally seeded with a :class:`~..spec.WarmState`, and
report back in the suite's own vocabulary. Everything
solver-specific — option names, warm-start plumbing, status codes,
working-set encodings — is confined to the adapter, which is what
makes it possible to add Ipopt (or anything else) to this benchmark
without touching a family.

Arms
----

An *arm* is a (algorithm, warm-start) pairing. The four the suite
defines:

``cold-ipm``
    Interior point, every step from the family's cold starting point.
    This is the baseline: it is what a solver without warm-start
    support does, and it is also the correctness reference.
``cold-sqp``
    Active-set SQP, every step cold. Present so that the warm-start
    effect can be separated from the algorithm change — without it,
    ``warm-sqp`` beating ``cold-ipm`` would confound the two.
``warm-ipm``
    Interior point seeded with the previous step's primal-dual point
    and barrier parameter. The fair comparison for ``warm-sqp``.
``warm-sqp``
    Active-set SQP seeded with the previous step's working set and
    primal point. The capability under test.

An adapter declares which arms it supports; unsupported arms are
skipped and reported as such rather than silently omitted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from ..spec import ParametricFamily, StepResult, WarmState
from ..sparsity import SparseCallbacks

ARMS: List[str] = ["cold-ipm", "cold-sqp", "warm-ipm", "warm-sqp"]

#: The arm whose per-step solutions every other arm is checked
#: against, and which generates the parameter path for adaptive
#: families.
REFERENCE_ARM = "cold-ipm"


def is_warm(arm: str) -> bool:
    return arm.startswith("warm")


class SolverAdapter(ABC):
    """Solve one parametric instance, cold or warm."""

    #: Name used in output files and the report.
    name: str = "unnamed"

    @abstractmethod
    def supports(self, arm: str) -> bool:
        """Whether this adapter can run the given arm."""

    @abstractmethod
    def solve(
        self,
        family: ParametricFamily,
        callbacks: SparseCallbacks,
        arm: str,
        x0: np.ndarray,
        warm: Optional[WarmState],
        step: int,
        tol: float,
    ) -> Tuple[StepResult, Optional[WarmState]]:
        """Solve at the family's current θ.

        Returns the step's measurements and the state to hand to the
        next step (``None`` if this arm carries nothing forward). The
        adapter fills every :class:`~..spec.StepResult` field except
        the correctness ones, which the runner supplies once the
        reference solution is known.
        """

"""The model algebra, and the record types the runner fills in.

Nothing in this module imports pounce.

Why the algebra is quadratic
----------------------------

Every objective and every ordinary row in this corpus is a quadratic
form, and **every complementarity pair is affine**. That is a real
restriction and it is deliberate: the product row ``G_i(x) * H_i(x)``
that every lowering has to build is then itself quadratic, so its
gradient and its (constant) Hessian are exact, closed-form, and
machine-checkable against a finite difference. A corpus whose first
derivative is approximate cannot be used to attribute a solver failure
to the solver -- the derivative is the first suspect, and it has to be
eliminated by construction rather than by argument.

It also bounds what this corpus is evidence *about*. Nothing here has a
nonlinear complementarity function, and nothing here is larger than six
variables. Both limits are stated again in the report and in the
manifest, because the whole point of Gate 0 is to hand Gate 1 an honest
boundary rather than an encouraging one.

The MPCC
--------

::

    min  f(x)
    s.t. lo_k <= c_k(x) <= hi_k        k = 1..m       (ordinary rows)
         lb <= x <= ub                                (bounds)
         0 <= G_i(x)  _|_  H_i(x) >= 0  i = 1..q      (pairs)

with ``f`` and ``c_k`` quadratic and ``G_i``, ``H_i`` affine. The
complementarity condition is the *source* semantics: the harness never
treats the lowered product row as if it were the source condition, and
the record keeps the two apart (``source_*`` vs ``nlp_*`` fields)
because conflating them is exactly how an MPCC benchmark ends up
reporting a converged NLP as a solved MPCC.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

#: Floor on the membership tolerance for "this pair side is zero" when
#: the activity pattern is read off a returned point. Loose relative to
#: `tol` on purpose: a converged interior-point solve parks an active
#: row at O(mu/z), not at 0, and a threshold at solver tolerance would
#: classify a genuinely active row as inactive on half the corpus.
#:
#: It is a *floor*, not the tolerance -- see `pair_activity`.
ACTIVE_TOL = 1e-6

#: Benchmark classes. Every case carries exactly one, and the ladder is
#: required to cover all of them (`selftest` enforces it).
CLASSES = (
    "regular",      # strict complementarity at the solution
    "biactive",     # at least one pair with G = H = 0 at the solution
    "degenerate",   # MPCC-LICQ fails, or the relaxed problem misbehaves
    "infeasible",   # no feasible point / expected failure
    "selector",     # symmetric Boolean selector, branch change under a parameter
    "macmpec",      # pinned small published-style instance
)


@dataclasses.dataclass(frozen=True)
class Quad:
    """``0.5 x'Px + c'x + d`` with ``P`` symmetric."""

    P: np.ndarray
    c: np.ndarray
    d: float = 0.0

    def __post_init__(self) -> None:
        P = np.asarray(self.P, dtype=float)
        c = np.asarray(self.c, dtype=float)
        if P.shape != (c.size, c.size):
            raise ValueError(f"Quad: P {P.shape} does not match c {c.shape}")
        if not np.allclose(P, P.T, atol=0.0, rtol=0.0):
            raise ValueError("Quad: P must be exactly symmetric")
        object.__setattr__(self, "P", P)
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "d", float(self.d))

    @property
    def n(self) -> int:
        return self.c.size

    def value(self, x: np.ndarray) -> float:
        return float(0.5 * x @ self.P @ x + self.c @ x + self.d)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.P @ x + self.c

    def hess(self, x: np.ndarray) -> np.ndarray:  # constant; x for signature parity
        return self.P

    def rescale(self, d: np.ndarray) -> "Quad":
        """The same form in ``xt`` under ``x = diag(d) @ xt``."""
        D = np.diag(d)
        return Quad(D @ self.P @ D, D @ self.c, self.d)


@dataclasses.dataclass(frozen=True)
class Affine:
    """``a'x + b``."""

    a: np.ndarray
    b: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "a", np.asarray(self.a, dtype=float))
        object.__setattr__(self, "b", float(self.b))

    @property
    def n(self) -> int:
        return self.a.size

    def value(self, x: np.ndarray) -> float:
        return float(self.a @ x + self.b)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.a

    def as_quad(self) -> Quad:
        return Quad(np.zeros((self.n, self.n)), self.a, self.b)

    def rescale(self, d: np.ndarray) -> "Affine":
        return Affine(self.a * d, self.b)


def pair_activity(
    g: np.ndarray, h: np.ndarray, act_tol: float = ACTIVE_TOL
) -> Tuple[np.ndarray, np.ndarray]:
    """Which side of each complementarity pair counts as zero at a point.

    The threshold is ``max(act_tol, sqrt(|G_i H_i|))`` — per pair, from
    the pair's own achieved complementarity — and **not** a fixed
    ``act_tol``. The reason is the geometry of the constraint the solver
    was actually given: ``G*H`` is quadratically flat at the corner, so a
    solve that drives the product to ``eps`` leaves the pair sitting
    ``sqrt(eps)`` away from it. At the default ``tol = 1e-8`` that is
    ``1e-4``, a hundred times the floor here — so a fixed ``1e-6``
    threshold reads a perfectly converged MPCC point as lying on
    *neither* branch, reports "no active constraints", and then finds no
    multipliers reproducing ``grad f``.

    Measured before this rule existed: 12 of the corpus's 512
    control-free observations — every ℓ₁ cell on `ralph2` and
    `qpec_small` that reached the optimum to nine digits — came back
    classified `none`, i.e. not even weakly stationary, purely from the
    threshold. That is the same "absolute threshold on a scale-dependent
    quantity" failure the rest of this harness is built to avoid; here
    the scale is the square root of the achieved residual.

    Both limbs behave correctly at the extremes: with ``G = 1`` and
    ``H = 1e-9`` the threshold is ``3e-5``, so ``H`` is active and ``G``
    is not; with ``G = H = 3e-5`` (the same product) both are, which is
    the biactive reading and the right one. At an exactly complementary
    point the product is zero and the threshold falls back to
    ``act_tol``.

    ``G`` and ``H`` are invariant under `MpccCase.rescale`, so the rule
    is scale-invariant like everything else that reads them.
    """
    g = np.asarray(g, dtype=float)
    h = np.asarray(h, dtype=float)
    thresh = np.maximum(act_tol, np.sqrt(np.abs(g * h)))
    return np.abs(g) <= thresh, np.abs(h) <= thresh


def product(g: Affine, h: Affine) -> Quad:
    """``G(x) * H(x)`` as a quadratic form.

    ``(a'x+b)(a2'x+b2)`` expands to ``x'(a a2')x + (b a2 + b2 a)'x +
    b b2``; the ``0.5 x'Px`` convention puts ``P = a a2' + a2 a'``, which
    is symmetric by construction whether or not ``a`` and ``a2`` are.
    """
    P = np.outer(g.a, h.a) + np.outer(h.a, g.a)
    c = g.b * h.a + h.b * g.a
    return Quad(P, c, g.b * h.b)


@dataclasses.dataclass(frozen=True)
class Row:
    """An ordinary constraint ``lo <= c(x) <= hi``."""

    name: str
    form: Quad
    lo: float
    hi: float

    @property
    def is_equality(self) -> bool:
        return self.lo == self.hi

    def rescale(self, d: np.ndarray) -> "Row":
        return Row(self.name, self.form.rescale(d), self.lo, self.hi)


@dataclasses.dataclass(frozen=True)
class Pair:
    """One complementarity pair ``0 <= G(x) _|_ H(x) >= 0``."""

    name: str
    G: Affine
    H: Affine
    #: Which branch means what, in the source model's own words. Gate 1
    #: will carry physical units here; at Gate 0 it documents the
    #: intended reading so a flipped pair is a review finding rather
    #: than a silent sign convention.
    branch_G_zero: str = ""
    branch_H_zero: str = ""

    def rescale(self, d: np.ndarray) -> "Pair":
        return Pair(
            self.name,
            self.G.rescale(d),
            self.H.rescale(d),
            self.branch_G_zero,
            self.branch_H_zero,
        )


@dataclasses.dataclass(frozen=True)
class Expected:
    """What the case is known to do, and how that was established.

    ``obj`` / ``x`` are the *source* MPCC's global optimum where one
    exists. They are not taken on trust: `oracle.enumerate_branches`
    recomputes them by solving every complementarity branch as a smooth
    program with SciPy, and `selftest` fails if the two disagree. The
    manifest records both numbers.
    """

    feasible: bool
    obj: Optional[float] = None
    x: Optional[np.ndarray] = None
    #: Strongest MPCC stationarity class the global solution attains:
    #: one of "S", "M", "C", "W", or None where the case has no solution.
    stationarity: Optional[str] = None
    #: Number of biactive pairs at the global solution.
    n_biactive: int = 0
    #: Whether MPCC-LICQ holds at the global solution.
    mpcc_licq: Optional[bool] = None
    #: Free text: what a route is *expected* to do, including the ones
    #: expected to fail. Read by no code; read by every reviewer.
    notes: str = ""


@dataclasses.dataclass(frozen=True)
class MpccCase:
    name: str
    klass: str
    objective: Quad
    rows: Tuple[Row, ...]
    pairs: Tuple[Pair, ...]
    lb: np.ndarray
    ub: np.ndarray
    starts: Dict[str, np.ndarray]
    expected: Expected
    #: Where the instance comes from and how its expected values were
    #: obtained. Every entry in this corpus is derived in-repo (the
    #: derivation is in the docstring of the factory in `cases.py`) and
    #: cross-checked by the branch-enumeration oracle; entries that also
    #: appear in the literature name it, but the number that is asserted
    #: is always the one this repo can rederive.
    provenance: str = ""
    #: Extra source-level checks beyond feasibility/objective, e.g. the
    #: regime a selector case must land in. Signature
    #: ``fn(case, x) -> dict``; keys ending in "_ok" are treated as
    #: pass/fail by the report.
    validators: Tuple[Callable[["MpccCase", np.ndarray], dict], ...] = ()

    @property
    def n(self) -> int:
        return self.objective.n

    @property
    def q(self) -> int:
        return len(self.pairs)

    def pair_values(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        g = np.array([p.G.value(x) for p in self.pairs])
        h = np.array([p.H.value(x) for p in self.pairs])
        return g, h

    def row_values(self, x: np.ndarray) -> np.ndarray:
        return np.array([r.form.value(x) for r in self.rows])

    def source_feasibility(self, x: np.ndarray) -> Dict[str, float]:
        """Original-space feasibility, complementarity kept separate.

        Four quantities, never merged into one headline number:

        ``row_viol``   max violation of the ordinary rows;
        ``bound_viol`` max violation of the variable bounds;
        ``sign_viol``  max violation of ``G >= 0`` / ``H >= 0``;
        ``compl_*``    ``|G_i * H_i|`` -- the source complementarity
                       product, which is *not* an NLP residual and must
                       not be compared against ``tol``.

        ``row_viol`` and ``bound_viol`` are separate because they behave
        differently under `rescale`: a row's value is invariant, while a
        bound violation ``lb_j - x_j`` becomes ``(lb_j - x_j)/d_j``. The
        two scaling legs can be compared field by field only if the
        scale-dependent one is not hidden inside a max with the
        invariant one.
        """
        c = self.row_values(x)
        lo = np.array([r.lo for r in self.rows]) if self.rows else np.zeros(0)
        hi = np.array([r.hi for r in self.rows]) if self.rows else np.zeros(0)
        row_viol = 0.0
        if self.rows:
            row_viol = float(np.max(np.maximum(np.maximum(lo - c, c - hi), 0.0)))
        bnd = float(np.max(np.maximum(np.maximum(self.lb - x, x - self.ub), 0.0)))
        g, h = self.pair_values(x)
        sign = 0.0
        if self.q:
            sign = float(np.max(np.maximum(np.maximum(-g, -h), 0.0)))
        prod = np.abs(g * h) if self.q else np.zeros(0)
        return {
            "row_viol": row_viol,
            "bound_viol": bnd,
            "sign_viol": sign,
            "compl_max": float(prod.max()) if self.q else 0.0,
            "compl_min": float(prod.min()) if self.q else 0.0,
            "compl_sum": float(prod.sum()) if self.q else 0.0,
        }

    def regime(self, x: np.ndarray, tol: float = ACTIVE_TOL) -> List[str]:
        """Per-pair branch label at ``x``: ``G0``, ``H0``, ``both`` or ``none``."""
        g, h = self.pair_values(x)
        gz_arr, hz_arr = pair_activity(g, h, tol)
        out = []
        for gz, hz in zip(gz_arr, hz_arr):
            out.append("both" if gz and hz else "G0" if gz else "H0" if hz else "none")
        return out

    def rescale(self, d: np.ndarray) -> "MpccCase":
        """An algebraically equivalent case under ``x = diag(d) @ xt``.

        ``d > 0`` elementwise, so every ``G >= 0`` / ``H >= 0`` and every
        bound keeps its sense and the feasible set maps one-to-one. The
        optimal objective is unchanged; the optimal point becomes
        ``x*/d``. This is the scaling leg: a route whose verdict moves
        between ``unit`` and ``skew`` is reporting on the scaling, not on
        the MPCC.
        """
        d = np.asarray(d, dtype=float)
        if d.size != self.n or np.any(d <= 0):
            raise ValueError("rescale: need a positive vector of length n")
        with np.errstate(invalid="ignore"):
            lb = np.where(np.isfinite(self.lb), self.lb / d, self.lb)
            ub = np.where(np.isfinite(self.ub), self.ub / d, self.ub)
        exp = self.expected
        exp = dataclasses.replace(exp, x=None if exp.x is None else exp.x / d)
        return dataclasses.replace(
            self,
            objective=self.objective.rescale(d),
            rows=tuple(r.rescale(d) for r in self.rows),
            pairs=tuple(p.rescale(d) for p in self.pairs),
            lb=lb,
            ub=ub,
            starts={k: v / d for k, v in self.starts.items()},
            expected=exp,
        )


#: Variable-scaling legs. `skew` spans six orders across the variables,
#: which is mild next to a real process model and already enough to move
#: a route's verdict on this corpus.
SCALINGS: Dict[str, Callable[[int], np.ndarray]] = {
    "unit": lambda n: np.ones(n),
    "skew": lambda n: np.power(10.0, np.linspace(-3.0, 3.0, n)) if n > 1 else np.array([1e3]),
}


@dataclasses.dataclass
class StageRecord:
    """One solve inside a route. A direct route has exactly one."""

    index: int
    #: The outer relaxation parameter this stage was solved at. ``None``
    #: for the non-relaxed lowerings, which is a different thing from 0.
    tau: Optional[float]
    tau_reason: str
    status: int
    status_msg: str
    accepted: bool
    warm_level: str
    restart_level: str
    restart_reason: str
    iters: int
    mu_in: Optional[float]
    mu_out: Optional[float]
    mu_reason: str
    wall_s: float
    nlp: Dict[str, float] = dataclasses.field(default_factory=dict)
    restoration: Dict[str, int] = dataclasses.field(default_factory=dict)
    regime: Optional[List[str]] = None


@dataclasses.dataclass
class RouteRecord:
    """The result contract of gh#794, for one (case, scaling, start, route,
    control) cell."""

    case: str
    klass: str
    scaling: str
    start: str
    route: str
    control: str
    lowering: str
    ok: bool
    status: int
    status_msg: str
    obj: Optional[float]
    x: Optional[List[float]]
    # --- source-level, in the MPCC's own units ---
    source: Dict[str, float] = dataclasses.field(default_factory=dict)
    stationarity: Dict[str, object] = dataclasses.field(default_factory=dict)
    validation: Dict[str, object] = dataclasses.field(default_factory=dict)
    regime: Optional[List[str]] = None
    regime_changes: int = 0
    # --- POUNCE's own NLP diagnostics, kept separate on purpose ---
    nlp: Dict[str, float] = dataclasses.field(default_factory=dict)
    iters: int = 0
    outer_stages: int = 0
    accepted_stages: int = 0
    rejected_stages: int = 0
    restarts: int = 0
    max_restart_level: str = "none"
    restoration: Dict[str, int] = dataclasses.field(default_factory=dict)
    log_counters: Dict[str, object] = dataclasses.field(default_factory=dict)
    wall_s: float = 0.0
    stages: List[StageRecord] = dataclasses.field(default_factory=list)
    error: Optional[str] = None

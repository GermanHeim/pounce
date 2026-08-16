"""Block-sequential initialization for Pyomo models (experimental).

IDAES-style initialization without hand-written initialization routines:
hold the *decision* variables at their current values, take the model's
active **equality** constraints, extract the square (well-determined)
part of the variable/constraint incidence graph (Dulmage-Mendelsohn,
via ``pyomo.contrib.incidence_analysis``), and solve it block by block
in topological order, writing the solution into ``Var.value``. The
blocks solve one at a time (1x1 blocks by Newton, larger blocks by
POUNCE), and every solve's verdict is checked before its values are
kept: a failed block restores its seed values instead of poisoning the
model. This module contributes the decision handling, the square-part
extraction, the seeding, the checked block loop, and the diagnostics.

The distillation-column shape of the workflow::

    report = pyomo_pounce.block_initialize(
        model, decisions=[m.feed, m.reflux, m.boilup])
    if not report.square:
        print(report)   # names of what you forgot to specify

set the decisions, solve for a physical profile with them held
constant, then let the optimizer move them.

**Experimental.** Variables in the square subsystem are (re)computed in
place, using any existing values as Newton starting guesses. Variables
in the under- or over-determined parts (degrees of freedom you did not
flag as decisions, redundant specifications) are left untouched and
reported **by name** — pair with
:func:`pyomo_pounce.initialize_missing_values` /
:func:`pyomo_pounce.project_to_feasible` to handle the remainder (or
use the :func:`pyomo_pounce.initialize` pipeline).

:func:`block_analyze` is the analysis-only sibling: the same decision
handling and the same Dulmage-Mendelsohn partition, but nothing is
seeded, projected, or solved, and the full partition is returned as
**component objects with nothing capped** — for diagnosing a large
model, and for tooling that builds on the partition rather than on a
display-sized name list.

:func:`block_repair_plan` is the planner: given the candidate
decisions, it plans a valid specification — which candidates a square
system can hold (selected), which the equalities claim (pruned), and
which variables nothing can determine (pinned, identified
automatically) — touching nothing. :func:`block_initialize` runs the
same check on its ``decisions`` and applies the plan when the
specification needs it, so a badly specified model initializes anyway
and the report says what was repaired.

Requires ``pyomo.contrib.incidence_analysis`` (needs ``networkx`` and
``scipy``); raises ``ImportError`` with instructions otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING, List, Optional

from pyomo.core.expr.numeric_expr import DivisionExpression, NegationExpression
from pyomo.environ import value

from pyomo_pounce.init_options import InitOptions

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyomo.core.base.constraint import ConstraintData
    from pyomo.core.base.var import VarData

__all__ = [
    "block_analyze",
    "block_initialize",
    "block_repair_plan",
    "structural_incidence",
    "BlockAnalysisReport",
    "BlockInitReport",
    "BlockOutcome",
    "BlockRepairPlan",
]

#: Termination conditions under which a solve's values may be kept.
OK_TERMINATIONS = ("optimal", "locallyOptimal", "globallyOptimal", "feasible")


#: Per-block statuses reported by :attr:`BlockInitReport.blocks`.
BLOCK_STATUSES = ("initialized", "fallback", "failed", "skipped")


@dataclass
class BlockOutcome:
    """What happened to one block of the calculation order (gh #609).

    Before gh #609 a run reported a count of initialized variables and a
    free-text failure string, so "which blocks actually got values" was
    not answerable -- and once a failure stopped the traversal, the
    difference between *skipped because it depended on the failure* and
    *skipped because the loop gave up* was not recorded at all. It is
    now, per block.
    """

    #: Position in the block-triangular calculation order.
    index: int
    #: Number of variables (== number of constraints) in the block.
    size: int
    #: Name of the block's leading constraint -- how the block is named
    #: in ``failures`` and in the report text.
    constraint: str
    #: One of :data:`BLOCK_STATUSES`.
    status: str
    #: Why, in the user's terms. Empty for a plain success.
    detail: str = ""
    #: Reciprocal condition number of the block's scaled Jacobian,
    #: or None when the check did not run (``conditioning="off"``, or a
    #: Jacobian that could not be built).
    rcond: Optional[float] = None
    #: Indices of the blocks this one consumes values from.
    depends_on: List[int] = field(default_factory=list)

    def __str__(self) -> str:
        rc = "" if self.rcond is None else f", rcond={self.rcond:.2e}"
        detail = f" -- {self.detail}" if self.detail else ""
        return (
            f"[{self.index}] {self.size}x{self.size} at {self.constraint!r}: "
            f"{self.status}{rc}{detail}"
        )


@dataclass
class BlockInitReport:
    """What :func:`block_initialize` did (and could not do)."""

    #: True when the equality system (after fixing decisions) is exactly
    #: square: no unmatched/underconstrained variables and no
    #: unmatched/overconstrained constraints.
    square: bool = True
    n_decisions_fixed: int = 0
    #: Pinned variables held during the solve (repair="auto" only);
    #: not counted in ``n_decisions_fixed``.
    n_pinned: int = 0
    n_blocks: int = 0
    n_1x1: int = 0
    n_subsystem_solves: int = 0
    n_vars_initialized: int = 0
    skipped_underdetermined: int = 0
    skipped_overdetermined: int = 0
    #: Names of unmatched/underconstrained variables (capped): the
    #: things you probably forgot to specify or flag as decisions.
    underconstrained_variables: List[str] = field(default_factory=list)
    #: Names of unmatched/overconstrained constraints (capped):
    #: redundant or conflicting specifications.
    overconstrained_constraints: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    #: The :class:`BlockRepairPlan` applied, when the specification
    #: actually needed repair (something pruned or pinned); None when
    #: the given decisions were used as-is.
    repair: Optional["BlockRepairPlan"] = None
    #: One :class:`BlockOutcome` per block of the calculation order, in
    #: that order (gh #609). The structured half of this report:
    #: ``failures`` stays the free-text half it always was.
    blocks: List[BlockOutcome] = field(default_factory=list)
    #: Numerical-conditioning notes: which blocks were found weak, what
    #: their rcond was, and what they were routed to (gh #609).
    diagnostics: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def _by_status(self, status: str) -> List[BlockOutcome]:
        return [b for b in self.blocks if b.status == status]

    @property
    def initialized_blocks(self) -> List[BlockOutcome]:
        """Blocks solved as square systems and kept."""
        return self._by_status("initialized")

    @property
    def fallback_blocks(self) -> List[BlockOutcome]:
        """Weak blocks routed to a regularized or coupled fallback."""
        return self._by_status("fallback")

    @property
    def failed_blocks(self) -> List[BlockOutcome]:
        """Blocks whose solve failed; their seed values were restored."""
        return self._by_status("failed")

    @property
    def skipped_blocks(self) -> List[BlockOutcome]:
        """Blocks not attempted -- descendants of a failure, or, under
        ``on_block_failure="stop"``, everything after one."""
        return self._by_status("skipped")

    @property
    def n_fallback(self) -> int:
        return len(self.fallback_blocks)

    @property
    def n_failed(self) -> int:
        return len(self.failed_blocks)

    @property
    def n_skipped(self) -> int:
        return len(self.skipped_blocks)

    def __str__(self) -> str:
        lines = [
            "pyomo-pounce block_initialize",
            f"  decisions fixed   : {self.n_decisions_fixed}",
        ]
        if self.n_pinned:
            lines.append(f"  pins held         : {self.n_pinned}")
        if self.repair is not None:
            lines.append(
                f"  spec repair       : {len(self.repair.decisions)} decisions "
                f"kept, {len(self.repair.pruned)} pruned, "
                f"{len(self.repair.pinned)} pinned"
            )
            if self.repair.pruned:
                lines.append(
                    "    pruned (solved for): " + _preview(self.repair.pruned)
                )
            if self.repair.pinned:
                lines.append(
                    "    pinned (held): " + _preview(self.repair.pinned)
                )
        lines += [
            f"  system square     : {self.square}",
            f"  blocks solved     : {self.n_blocks} "
            f"({self.n_1x1} by Newton 1x1, {self.n_subsystem_solves} subsystem solves)",
            f"  vars initialized  : {self.n_vars_initialized}",
            f"  left untouched    : {self.skipped_underdetermined} underdetermined, "
            f"{self.skipped_overdetermined} overdetermined",
        ]
        if self.underconstrained_variables:
            lines.append(
                "  underconstrained vars (specify or flag as decisions): "
                + ", ".join(self.underconstrained_variables)
            )
        if self.overconstrained_constraints:
            lines.append(
                "  overconstrained cons (redundant/conflicting specs): "
                + ", ".join(self.overconstrained_constraints)
            )
        if self.n_fallback or self.n_skipped or self.n_failed:
            lines.append(
                f"  block outcomes    : {len(self.initialized_blocks)} initialized, "
                f"{self.n_fallback} fallback, {self.n_failed} failed, "
                f"{self.n_skipped} skipped"
            )
        for d in self.diagnostics:
            lines.append(f"  diagnostic: {d}")
        for b in self.blocks:
            if b.status != "initialized":
                lines.append(f"  {b}")
        for f in self.failures:
            lines.append(f"  FAILED: {f}")
        return "\n".join(lines)


def _preview(components, cap: int = 10) -> str:
    """Display-sized name list; the underlying data is never capped."""
    names = [c.name for c in components[:cap]]
    extra = len(components) - len(names)
    return ", ".join(names) + (f", ... and {extra} more" if extra > 0 else "")


@dataclass
class BlockAnalysisReport:
    """The full Dulmage-Mendelsohn partition from :func:`block_analyze`.

    Every list holds the Pyomo **component data objects** themselves
    (``VarData`` / ``ConstraintData``), in DM order, with nothing
    capped; ``str(report)`` shows a display-sized preview.
    """

    #: True when the equality system (after fixing decisions) is exactly
    #: square: no underconstrained part and no overconstrained part.
    square: bool = True
    n_decisions_fixed: int = 0
    #: Size of the analyzed system: active equality constraints and the
    #: unfixed variables appearing in them.
    n_constraints: int = 0
    n_variables: int = 0
    #: The underconstrained subsystem: variables the equalities cannot
    #: determine (the things to specify or flag as decisions), and the
    #: constraints entangled with them.
    underconstrained_variables: List[VarData] = field(default_factory=list)
    underconstrained_constraints: List[ConstraintData] = field(default_factory=list)
    #: The overconstrained subsystem: redundant or conflicting
    #: specifications, and the variables they fight over.
    overconstrained_constraints: List[ConstraintData] = field(default_factory=list)
    overconstrained_variables: List[VarData] = field(default_factory=list)
    #: The square (well-determined) part, and its block-triangular
    #: calculation order: ``variable_blocks[k]`` is solved from
    #: ``constraint_blocks[k]``, in sequence.
    square_variables: List[VarData] = field(default_factory=list)
    square_constraints: List[ConstraintData] = field(default_factory=list)
    variable_blocks: List[List[VarData]] = field(default_factory=list)
    constraint_blocks: List[List[ConstraintData]] = field(default_factory=list)
    #: The block DAG (gh #609): ``block_dependencies[k]`` lists the
    #: indices of the blocks whose variables appear in block ``k``'s
    #: constraints, i.e. the blocks ``k`` consumes values from. Block
    #: triangularity means every entry is ``< k``, so a block with an
    #: empty list is the head of an independent branch. This is what
    #: lets :func:`block_initialize` skip only a failure's *descendants*
    #: instead of abandoning the rest of the traversal.
    block_dependencies: List[List[int]] = field(default_factory=list)

    @property
    def n_extra_degrees_of_freedom(self) -> int:
        """How many more specifications would square the under part."""
        return len(self.underconstrained_variables) - len(
            self.underconstrained_constraints
        )

    @property
    def n_extra_specifications(self) -> int:
        """How many redundant/conflicting rows the over part carries."""
        return len(self.overconstrained_constraints) - len(
            self.overconstrained_variables
        )

    @property
    def n_blocks(self) -> int:
        return len(self.variable_blocks)

    @property
    def n_1x1(self) -> int:
        return sum(1 for blk in self.variable_blocks if len(blk) == 1)

    def __str__(self) -> str:
        lines = [
            "pyomo-pounce block_analyze",
            f"  decisions fixed   : {self.n_decisions_fixed}",
            f"  equality system   : {self.n_constraints} constraints, "
            f"{self.n_variables} variables",
            f"  system square     : {self.square}",
            f"  square part       : {len(self.square_variables)} variables in "
            f"{self.n_blocks} blocks ({self.n_1x1} 1x1)",
        ]
        if self.underconstrained_variables:
            lines.append(
                f"  underconstrained  : {len(self.underconstrained_variables)} "
                f"variables, {len(self.underconstrained_constraints)} constraints "
                f"({self.n_extra_degrees_of_freedom} more specifications needed)"
            )
            lines.append(
                "    vars (specify or flag as decisions): "
                + _preview(self.underconstrained_variables)
            )
            if self.underconstrained_constraints:
                lines.append(
                    "    cons: " + _preview(self.underconstrained_constraints)
                )
        if self.overconstrained_constraints:
            lines.append(
                f"  overconstrained   : {len(self.overconstrained_constraints)} "
                f"constraints, {len(self.overconstrained_variables)} variables "
                f"({self.n_extra_specifications} redundant/conflicting)"
            )
            lines.append(
                "    cons (redundant/conflicting specs): "
                + _preview(self.overconstrained_constraints)
            )
            if self.overconstrained_variables:
                lines.append(
                    "    vars: " + _preview(self.overconstrained_variables)
                )
        return "\n".join(lines)


def _flatten_vars(vars_like):
    """Accept VarData, indexed Var containers, or iterables of either."""
    out = []
    for v in vars_like:
        if hasattr(v, "values") and callable(v.values):  # indexed container
            out.extend(v.values())
        else:
            out.append(v)
    return out


@dataclass
class BlockRepairPlan:
    """A valid specification planned by :func:`block_repair_plan`.

    A plan, not an action: nothing on the model is touched. Every list
    holds component data objects, uncapped. Applying the plan to a model
    you intend to solve means fixing ``decisions`` and ``pinned`` and
    leaving ``pruned`` free.
    """

    #: True when holding ``decisions`` and ``pinned`` makes the equality
    #: system exactly square: no loose variables, no redundant rows.
    square: bool = True
    n_constraints: int = 0
    n_variables: int = 0
    #: The candidates selected as decisions: the equalities do not
    #: contest them, so they can be held.
    decisions: List[VarData] = field(default_factory=list)
    #: The candidates the equalities claim: holding these too would
    #: overconstrain the system, so the plan solves for them instead.
    pruned: List[VarData] = field(default_factory=list)
    #: Variables the equalities provably cannot determine, identified
    #: automatically: they appear in the system, but only through edges
    #: that cannot determine them (an equation ``0 == f/g`` cannot
    #: determine a variable appearing only in ``g``). Hold them at a
    #: value of your choosing — for a flowsheet these are the loose
    #: integrators, e.g. drum levels with no weir feedback.
    pinned: List[VarData] = field(default_factory=list)
    #: Undetermined variables the plan cannot square away: a genuine
    #: modeling defect (or a missing specification).
    loose_variables: List[VarData] = field(default_factory=list)
    #: Equalities no specification can satisfy independently: redundant
    #: or conflicting rows, a model defect.
    redundant_constraints: List[ConstraintData] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "pyomo-pounce block_repair_plan",
            f"  equality system   : {self.n_constraints} constraints, "
            f"{self.n_variables} variables",
            f"  repaired square   : {self.square}",
            f"  decisions         : {len(self.decisions)} selected, "
            f"{len(self.pruned)} pruned",
        ]
        if self.pruned:
            lines.append(
                "    pruned (solved for): " + _preview(self.pruned)
            )
        if self.pinned:
            lines.append(
                f"  pinned            : {len(self.pinned)} undetermined by "
                "the equalities, hold at chosen values: "
                + _preview(self.pinned)
            )
        if self.loose_variables:
            lines.append(
                f"  loose variables   : {len(self.loose_variables)} "
                "undetermined (model defect or missing specification): "
                + _preview(self.loose_variables)
            )
        if self.redundant_constraints:
            lines.append(
                f"  redundant rows    : {len(self.redundant_constraints)} no "
                "specification can satisfy: "
                + _preview(self.redundant_constraints)
            )
        return "\n".join(lines)


def _usable_incident(con, incident):
    """The incident variables an equality can actually determine.

    An equation ``0 == f/g`` cannot determine a variable that appears
    only in the denominator ``g``: its sensitivity there vanishes
    whenever the equation is satisfied, so a matching through that edge
    is singular at every solution. This is the shape substituting
    ``dx/dt = 0`` into ``dx/dt == f/M`` produces, which is how loose
    integrators (drum levels) hide in steady-state reductions.

    The rule is deliberately shallow and conservative: only a division
    at the top of the body qualifies, so a nested ``0 == (a/b)/c``
    keeps the ``b`` edge even though ``b`` is equally undeterminable.
    False negatives only — do not make this recursive without thinking
    through the false-positive direction.
    """
    if con.lower is None or con.upper is None:
        return incident
    if value(con.lower) != 0 or value(con.upper) != 0:
        return incident
    body = con.body
    while isinstance(body, NegationExpression):
        body = body.args[0]
    if not isinstance(body, DivisionExpression):
        return incident
    from pyomo.contrib.incidence_analysis import get_incident_variables

    numerator_ids = {id(v) for v in get_incident_variables(body.args[0])}
    return [v for v in incident if id(v) in numerator_ids]


def _seed_pin(v) -> None:
    """Seed a pinned variable, never at exactly zero.

    A pin appears only in denominators of ``0 == f/g`` rows — that is
    what made every edge unusable — so zero is the one value guaranteed
    to break every equation it touches. Falls back from the bounds-aware
    seed to a nonzero in-bounds point.
    """
    _seed_var(v)
    if v.value != 0.0:
        return
    lo, hi = v.lb, v.ub
    # Directional presence (gh #403): `abs(b) < 1e19` called a real bound of
    # -5e20 absent, and this fell through to seeding 0.0 — outside the
    # variable's own declared box.
    lo_ok = lo is not None and lo > -1e19
    hi_ok = hi is not None and hi < 1e19
    if lo_ok and hi_ok:
        for frac in (0.75, 0.6):  # midpoint was zero; try off-center
            cand = lo + frac * (hi - lo)
            if cand != 0.0:
                v.set_value(cand, skip_validation=True)
                return
    elif lo_ok:
        v.set_value(lo + 2.0, skip_validation=True)  # lo + 1 was zero
    elif hi_ok:
        v.set_value(hi - 2.0, skip_validation=True)  # hi - 1 was zero
    else:
        v.set_value(1.0, skip_validation=True)


def _tiered_matching(var_adj, tiers):
    """Maximum matching, augmenting from variables in tier order.

    Greedy augmentation in priority order is lexicographically optimal
    over the transversal matroid: tier-1 coverage is maximized first,
    then tier-2 given tier-1, and so on. ``var_adj[v]`` lists the
    equation indices variable ``v`` appears in. Returns ``(eq_match,
    var_match)`` as dicts (eq index -> var index, var index -> eq
    index). Iterative, so deep alternating paths cannot hit the
    recursion limit.
    """
    eq_match = {}
    var_match = {}
    for tier in tiers:
        for v0 in tier:
            pred = {}
            seen = set()
            queue = [v0]
            free_eq = None
            while queue and free_eq is None:
                v = queue.pop()
                for e in var_adj[v]:
                    if e in seen:
                        continue
                    seen.add(e)
                    pred[e] = v
                    w = eq_match.get(e)
                    if w is None:
                        free_eq = e
                        break
                    queue.append(w)
            e = free_eq
            while e is not None:  # flip the alternating path
                v = pred[e]
                prev = var_match.get(v)
                eq_match[e] = v
                var_match[v] = e
                e = prev
    return eq_match, var_match


def structural_incidence(model):
    """One whole-model incidence walk, shareable across the analyze,
    repair-plan, and block-solve passes of a single initialize() call.

    Built over the active equalities with FIXED VARIABLES INCLUDED, so
    it is independent of fix-state; each pass filters it down to the
    currently-unfixed variables (`_active_view`), which reuses the
    stored graph instead of re-walking every constraint expression.
    Within-call sharing only: the graph reflects the model's structure
    at construction, so it must not outlive structural edits.
    """
    try:
        # Probe networkx explicitly: pyomo defers its optional imports,
        # so `pyomo.contrib.incidence_analysis` imports fine without it
        # and would only blow up (DeferredImportError) at first use.
        # This runs before the block_* probes in the initialize()
        # pipeline, so it must carry the same actionable message.
        import networkx  # noqa: F401

        from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "structural_incidence requires pyomo.contrib."
            "incidence_analysis and its optional dependencies "
            "(pip install networkx scipy)"
        ) from e

    return IncidenceGraphInterface(
        model, include_inequality=False, include_fixed=True
    )


def _active_view(igraph, model):
    """The shared structural graph filtered to currently-unfixed
    variables, without re-walking constraint expressions.

    STRUCTURAL, not value-substituted: a fresh construction substitutes
    fixed variables' values, so its edge set can differ from this view
    whenever that substitution cancels a term. A fixed ZERO is the
    obvious way (`a*x` with `a = 0` drops `x`), but not the only one:
    values that cancel across terms do it too, with no fixed variable
    being zero at all (`a*x - b*x` with `a` and `b` fixed equal drops
    `x`, gh #445 review). Keying the guard on zero-valued variables
    would therefore miss real cancellations, so every row adjacent to
    ANY fixed variable is re-derived and compared; if an edge genuinely
    cancels, this pass falls back to a fresh construction (correct
    diagnostics over speed).

    That check costs what a fresh build spends on the rows it examines,
    and it examines a subset, so it is never more expensive than the
    fallback it protects — but it does mean a pass with many fixed
    variables pays most of a walk. Nothing is fixed when the plan pass
    runs, so it returns above without reaching here; the analyze pass
    is the one that pays.

    Away from cancellations the edge sets are identical; the variable
    ORDER within rows can still differ (value substitution changes the
    linear/nonlinear split). Order feeds tie-breaks, so where two
    answers are equally valid the two views may pick different ones,
    not merely list the same ones differently.

    On pyomo without `IncidenceGraphInterface.subgraph` (< 6.7.1) this
    falls back to a fresh construction: old speed, same behavior.

    May return the shared graph itself when nothing is fixed, so
    callers can alias it; safe while no pass mutates the graph — keep
    it that way.
    """
    from pyomo.contrib.incidence_analysis import IncidenceGraphInterface

    if not hasattr(igraph, "subgraph"):  # pyomo < 6.7.1
        return IncidenceGraphInterface(model, include_inequality=False)
    unfixed = [v for v in igraph.variables if not v.fixed]
    if len(unfixed) == len(igraph.variables):
        return igraph
    fixed = [v for v in igraph.variables if v.fixed]
    if fixed:
        from pyomo.contrib.incidence_analysis import get_incident_variables

        # any fixed variable can take part in a cancellation, not only
        # a zero-valued one -- see the note above
        affected = set()
        for v in fixed:
            affected.update(id(c) for c in igraph.get_adjacent_to(v))
        for con in igraph.constraints:
            if id(con) not in affected:
                continue
            substituted = {id(vv) for vv in get_incident_variables(con.body)}
            structural = {
                id(vv)
                for vv in igraph.get_adjacent_to(con)
                if not vv.fixed
            }
            # substitution can only DROP unfixed variables from a row,
            # never add one, so inequality means a genuine cancellation
            if substituted != structural:
                return IncidenceGraphInterface(
                    model, include_inequality=False
                )
    return igraph.subgraph(unfixed, list(igraph.constraints))


def block_repair_plan(model, decision_candidates=None, igraph=None) -> BlockRepairPlan:
    """Plan a valid specification; touch nothing, solve nothing.

    ``decision_candidates`` are the variables you would like to hold
    (for a flowsheet, the flow controls). The plan selects the subset a
    valid specification can hold — those are the ``decisions`` — and
    prunes the rest, which the equalities claim and solve for. Matching
    prefers plain variables over candidates, which provably minimizes
    the number pruned; among candidates, **earlier-listed ones are
    preferentially kept**, so the listing order is an implicit priority
    when a pruning tie could go either way. Variables the equalities
    provably cannot determine are identified automatically and come
    back ``pinned``: hold them at a value of your choosing. On a
    well-specified system the plan is a no-op: every candidate
    selected, nothing pruned or pinned.

    Args:
        model: A Pyomo model (Block). Only active equality constraints
            and unfixed variables participate.
        decision_candidates: Variables (VarData or indexed Var
            containers) you would like held. Purely structural, so
            values are not needed. Already-fixed variables are inputs,
            not part of the plan.

    Returns a :class:`BlockRepairPlan`; fix ``plan.decisions`` and
    ``plan.pinned`` (and leave ``plan.pruned`` free) to define a square
    system.

    ``igraph``, when given, is the structural incidence graph from
    :func:`structural_incidence`, built over THIS model with fixed
    variables included; the pass filters it to the currently-unfixed
    variables instead of re-walking every constraint expression
    (gh #444). The view is STRUCTURAL: away from value cancellations
    (guarded, with a fresh-build fallback) its edge sets
    match a fresh construction, but the variable order within rows may
    differ on models where fixed values change the linear/nonlinear
    split. Order feeds tie-breaks, so where two answers are equally
    valid (which of two variables is reported loose, which member of a
    degenerate block leads) the shared and fresh views may pick
    different ones, not merely list the same ones differently.
    It must belong to the model being passed: a graph from another
    model, or from before a structural edit, produces wrong answers
    rather than an error. Omitted, the incidence is built locally
    exactly as before.
    """
    try:
        # Probe networkx explicitly: pyomo defers its optional imports, so
        # `pyomo.contrib.incidence_analysis` imports fine without it and
        # would only blow up (DeferredImportError) at first use.
        import networkx  # noqa: F401

        from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "block_repair_plan requires pyomo.contrib.incidence_analysis "
            "and its optional dependencies (pip install networkx scipy)"
        ) from e

    plan = BlockRepairPlan()

    candidates = []
    candidate_ids = set()
    for vd in _flatten_vars(decision_candidates or []):
        if not vd.fixed and id(vd) not in candidate_ids:
            candidate_ids.add(id(vd))
            candidates.append(vd)

    # A supplied igraph is the structural graph of
    # `structural_incidence` (fixed variables included), filtered here
    # to the current fix-state; omitted, the walk happens locally as
    # before (gh #444).
    if igraph is not None:
        igraph = _active_view(igraph, model)
    else:
        igraph = IncidenceGraphInterface(model, include_inequality=False)
    eqs = list(igraph.constraints)
    gvars = list(igraph.variables)
    plan.n_constraints = len(eqs)
    plan.n_variables = len(gvars)
    if not eqs:
        # nothing to determine: every candidate is simply an input
        plan.decisions = candidates
        return plan

    vindex = {id(v): i for i, v in enumerate(gvars)}
    raw_degree = [0] * len(gvars)
    var_adj = [[] for _ in gvars]
    for e, con in enumerate(eqs):
        # raw incidence comes from the graph already built above; the
        # expression body is only inspected on 0 == f/g shaped rows
        incident = [
            v for v in igraph.get_adjacent_to(con) if id(v) in vindex
        ]
        for v in incident:
            raw_degree[vindex[id(v)]] += 1
        for v in _usable_incident(con, incident):
            var_adj[vindex[id(v)]].append(e)

    # pinned: present in the equalities, but every edge is unusable —
    # nothing can determine these, under any matching
    pinned_idx = [
        i for i in range(len(gvars))
        if raw_degree[i] > 0 and not var_adj[i] and id(gvars[i]) not in candidate_ids
    ]

    # candidates augment in reverse listing order: greedy augmentation
    # preferentially matches (prunes) the earliest-processed vertex, so
    # reversing makes earlier-listed candidates preferentially kept
    tiers = (
        [i for i, v in enumerate(gvars)
         if id(v) not in candidate_ids and var_adj[i]],
        [vindex[id(v)] for v in reversed(candidates) if id(v) in vindex],
    )
    eq_match, var_match = _tiered_matching(var_adj, tiers)

    pruned_ids = {id(gvars[i]) for i in tiers[1] if i in var_match}
    # candidates off the equality graph are inputs nothing can contest
    plan.decisions = [v for v in candidates if id(v) not in pruned_ids]
    plan.pruned = [v for v in candidates if id(v) in pruned_ids]
    plan.pinned = [gvars[i] for i in pinned_idx]
    plan.loose_variables = [gvars[i] for i in tiers[0] if i not in var_match]
    plan.redundant_constraints = [
        eqs[e] for e in range(len(eqs)) if e not in eq_match
    ]
    plan.square = not plan.loose_variables and not plan.redundant_constraints
    return plan


def block_analyze(model, decisions=None, igraph=None) -> BlockAnalysisReport:
    """Partition the equality system; touch nothing, solve nothing.

    The analysis half of :func:`block_initialize` on its own: hold the
    decisions fixed, decompose the active equality constraints
    (Dulmage-Mendelsohn), and return the **full** partition — the
    underconstrained, overconstrained, and square parts as component
    objects, plus the square part's block-triangular calculation order —
    with nothing capped for display and no values read or written.

    Args:
        model: A Pyomo model (Block). Only active equality constraints
            and unfixed variables participate.
        decisions: Variables (VarData or indexed Var containers) to hold
            fixed during the analysis, then release. Purely structural,
            so unlike :func:`block_initialize` they do **not** need
            values. Already-fixed variables may be listed and stay
            fixed.

    Returns a :class:`BlockAnalysisReport`.

    ``igraph``, when given, is the structural incidence graph from
    :func:`structural_incidence`, built over THIS model with fixed
    variables included; the pass filters it to the currently-unfixed
    variables instead of re-walking every constraint expression
    (gh #444). The view is STRUCTURAL: away from value cancellations
    (guarded, with a fresh-build fallback) its edge sets
    match a fresh construction, but the variable order within rows may
    differ on models where fixed values change the linear/nonlinear
    split. Order feeds tie-breaks, so where two answers are equally
    valid (which of two variables is reported loose, which member of a
    degenerate block leads) the shared and fresh views may pick
    different ones, not merely list the same ones differently.
    It must belong to the model being passed: a graph from another
    model, or from before a structural edit, produces wrong answers
    rather than an error. Omitted, the incidence is built locally
    exactly as before.
    """
    try:
        # Probe networkx explicitly: pyomo defers its optional imports, so
        # `pyomo.contrib.incidence_analysis` imports fine without it and
        # would only blow up (DeferredImportError) at first use.
        import networkx  # noqa: F401

        from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "block_analyze requires pyomo.contrib.incidence_analysis "
            "and its optional dependencies (pip install networkx scipy)"
        ) from e

    report = BlockAnalysisReport()

    fixed_by_us = []
    if decisions is not None:
        for vd in _flatten_vars(decisions):
            if vd.fixed:
                continue  # already an input; leave as the user set it
            vd.fix()
            fixed_by_us.append(vd)
    report.n_decisions_fixed = len(fixed_by_us)

    try:
        # decisions were fixed above, so the filtered view (or the
        # fresh walk) sees them excluded exactly as before (gh #444)
        if igraph is not None:
            igraph = _active_view(igraph, model)
        else:
            igraph = IncidenceGraphInterface(model, include_inequality=False)
        if not igraph.constraints:
            return report
        report.n_constraints = len(igraph.constraints)
        report.n_variables = len(igraph.variables)

        var_dm, con_dm = igraph.dulmage_mendelsohn()
        report.underconstrained_variables = list(var_dm.unmatched) + list(
            var_dm.underconstrained
        )
        report.underconstrained_constraints = list(con_dm.underconstrained)
        report.overconstrained_constraints = list(con_dm.unmatched) + list(
            con_dm.overconstrained
        )
        report.overconstrained_variables = list(var_dm.overconstrained)
        report.square_variables = list(var_dm.square)
        report.square_constraints = list(con_dm.square)
        report.square = (
            not report.underconstrained_variables
            and not report.overconstrained_constraints
        )

        if report.square_variables:
            var_blocks, con_blocks = igraph.block_triangularize(
                variables=report.square_variables,
                constraints=report.square_constraints,
            )
            report.variable_blocks = [list(blk) for blk in var_blocks]
            report.constraint_blocks = [list(blk) for blk in con_blocks]
            report.block_dependencies = _block_dependencies(
                igraph, report.variable_blocks, report.constraint_blocks
            )
    finally:
        for vd in fixed_by_us:
            vd.unfix()

    return report


def block_initialize(
    model,
    decisions=None,
    solver=None,
    *,
    repair: str = "auto",
    max_list: Optional[int] = None,
    tee: bool = False,
    options=None,
    igraph=None,
) -> BlockInitReport:
    """Fill ``Var.value`` by solving equality blocks in calculation order.

    Args:
        model: A Pyomo model (Block). Only active equality constraints
            and unfixed variables participate.
        decisions: Variables (VarData or indexed Var containers) to hold
            at their **current values** during the initialization solve,
            then release — the degrees of freedom the optimizer will
            move later. Each decision that stays held must have a value
            (``ValueError`` otherwise). Already-fixed variables may be
            listed and stay fixed. Equivalent to fixing them yourself,
            but scoped and self-documenting.
        solver: A Pyomo solver (from ``SolverFactory``) for blocks
            larger than 1x1. Default: ``SolverFactory("pounce")``,
            constructed only when such a block exists.
        repair: ``"auto"`` (default) checks and repairs the
            specification as described below; ``"off"`` holds the
            decisions exactly as given — nothing pruned, nothing
            pinned, every decision needs a value, and a non-square
            system is reported (``report.square``) instead of repaired.
        max_list: Cap on the reported name lists. Defaults to
            ``options.max_list``.
        tee: Echo block-solver output. Ignored when `options` is an
            :class:`~pyomo_pounce.InitOptions`, which carries its own.
        options: Solver options dict handed to **every** block solve, or
            an :class:`~pyomo_pounce.InitOptions` that also chooses the
            conditioning, fallback, and failure-recovery policy (gh
            #609). A bare dict is always solver options. Before gh #609
            there was no such argument here at all, so options given to
            :func:`~pyomo_pounce.initialize` reached the projection and
            were dropped by every block solve.

    With ``repair="auto"`` the specification is checked before anything
    is held. When holding the given decisions would leave the equality
    system exactly square, they are used as-is. When it would not, they
    are treated as the candidate pool of :func:`block_repair_plan`:
    conflicting decisions are pruned (solved for instead of held),
    variables the equalities provably cannot determine are pinned
    automatically, at their current values or a nonzero bounds-aware
    seed if they have none, and ``report.repair`` records the plan
    (None when the decisions were used as-is). The repair is scoped to
    this call like the decisions themselves — flags are restored, so it
    never alters the model's own specification.

    Returns a :class:`BlockInitReport`. ``report.square`` is False when
    the equality system (after the repair) is still not exactly square;
    the offending variable/constraint **names** are reported, and the
    square part is still solved best-effort. ``report.failures`` is
    non-empty when a block solve failed: the failed block's variables
    are restored to their seed values — a failed solve never writes its
    values into the model.

    A failure no longer abandons the traversal (gh #609). The block DAG
    (:attr:`BlockAnalysisReport.block_dependencies`) says which later
    blocks consume the failed block's values; those are marked
    ``"skipped"`` and every branch independent of the failure is still
    initialized. ``report.blocks`` is the per-block record —
    initialized, fallback, failed, skipped — and
    ``options.on_block_failure="stop"`` restores the pre-gh#609
    behaviour of stopping at the first failure.

    A 1x1 block solves by Pyomo's ``calculate_variable_from_constraint``.
    When the model's export-enabled ``scaling_factor`` Suffix tags the
    block's constraint, the convergence test is measured on the row's
    stated scale (``eps = 1e-8 / factor``), the same view
    ``user-scaling`` gives a full-model solve. An untagged constraint
    keeps the absolute default, and a model without the Suffix behaves
    exactly as before.

    Structurally square is not numerically solvable, so each block is
    also rank-checked before it is solved (gh #609). A block whose
    scaled Jacobian has a reciprocal condition number below
    ``options.cond_tol`` is routed to a regularized least-squares
    fallback (or, with ``options.fallback="coupled"``, solved together
    with the blocks that depend on it) and recorded in
    ``report.diagnostics``, rather than being handed to a Newton step
    that will land on an arbitrary point of the near-null direction.

    ``igraph``, when given, is the structural incidence graph from
    :func:`structural_incidence`, built over THIS model with fixed
    variables included; the pass filters it to the currently-unfixed
    variables instead of re-walking every constraint expression
    (gh #444). The view is STRUCTURAL: away from value cancellations
    (guarded, with a fresh-build fallback) its edge sets
    match a fresh construction, but the variable order within rows may
    differ on models where fixed values change the linear/nonlinear
    split. Order feeds tie-breaks, so where two answers are equally
    valid (which of two variables is reported loose, which member of a
    degenerate block leads) the shared and fresh views may pick
    different ones, not merely list the same ones differently.
    It must belong to the model being passed: a graph from another
    model, or from before a structural edit, produces wrong answers
    rather than an error. Omitted, the incidence is built locally
    exactly as before.
    """
    import pyomo.environ as pyo

    try:
        # Probe networkx explicitly: pyomo defers its optional imports, so
        # `pyomo.contrib.incidence_analysis` imports fine without it and
        # would only blow up (DeferredImportError) at first use.
        import networkx  # noqa: F401

        import pyomo.contrib.incidence_analysis  # noqa: F401
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "block_initialize requires pyomo.contrib.incidence_analysis "
            "and its optional dependencies (pip install networkx scipy)"
        ) from e
    from pyomo.util.calc_var_value import calculate_variable_from_constraint
    from pyomo.util.subsystems import TemporarySubsystemManager, create_subsystem_block

    from pyomo_pounce.scaling import read_scaling

    if repair not in ("auto", "off"):
        raise ValueError(
            f"block_initialize: repair must be 'auto' or 'off', got {repair!r}"
        )

    opts = InitOptions.coerce(options)
    if not isinstance(options, InitOptions) and tee:
        opts = dc_replace(opts, tee=True)
    if max_list is None:
        max_list = opts.max_list

    report = BlockInitReport()

    # --- check the specification, then hold it -------------------------
    # repair="auto": on a well-specified system the plan is a no-op
    # (every decision selected), which is exactly the shipped behavior;
    # a broken one is repaired. A pruned decision needs no value (it
    # gets solved for); a pinned variable without one gets a nonzero
    # bounds-aware seed (the pin is the repair's choice, not the user's,
    # so erroring on it would be hostile). repair="off" holds the
    # decisions exactly as given and reports instead of repairing.
    plan = None
    if repair == "auto":
        plan = block_repair_plan(
            model, decision_candidates=decisions, igraph=igraph
        )
        if plan.pruned or plan.pinned:
            report.repair = plan
    fixed_by_us = []
    try:
        # Fixing happens inside the try so a mid-loop ValueError cannot
        # leave earlier decisions fixed on the model.
        if plan is not None:
            to_hold = plan.decisions
        else:
            to_hold = [
                vd for vd in _flatten_vars(decisions or []) if not vd.fixed
            ]
        for vd in to_hold:
            if vd.value is None:
                raise ValueError(
                    f"decision variable {vd.name!r} has no value: a decision "
                    "must be held at a concrete value during initialization"
                )
            vd.fix()
            fixed_by_us.append(vd)
        report.n_decisions_fixed = len(fixed_by_us)
        if plan is not None:
            for vd in plan.pinned:
                if vd.value is None:
                    _seed_pin(vd)
                vd.fix()
                fixed_by_us.append(vd)
            report.n_pinned = len(plan.pinned)

        # The square (well-determined) part of the equality system: the
        # DM decomposition separates it from remaining degrees of
        # freedom and redundant specifications — and names them. The
        # decisions are already fixed above, so none are passed on.
        analysis = block_analyze(model, igraph=igraph)
        under_vars = analysis.underconstrained_variables
        over_cons = analysis.overconstrained_constraints
        report.skipped_underdetermined = len(under_vars)
        report.skipped_overdetermined = len(over_cons)
        report.underconstrained_variables = [v.name for v in under_vars[:max_list]]
        report.overconstrained_constraints = [c.name for c in over_cons[:max_list]]
        report.square = analysis.square

        square_vars = analysis.square_variables
        square_cons = analysis.square_constraints
        if not square_vars:
            return report

        # Solve-plan statistics (the SCC solve below follows exactly
        # the analysis' block structure).
        report.n_blocks = analysis.n_blocks
        report.n_1x1 = analysis.n_1x1
        n_large = report.n_blocks - report.n_1x1

        for v in square_vars:
            if v.value is None:
                _seed_var(v)

        if n_large > 0 and solver is None:
            solver = pyo.SolverFactory("pounce")

        deps = analysis.block_dependencies or [[] for _ in analysis.variable_blocks]

        # Row factors from the model's `scaling_factor` Suffix, read
        # through the same machinery the full-model solves use (gh
        # #483), so the 1x1 convergence test below and a `user-scaling`
        # solve of the same model see the same rows. An untagged
        # constraint, and a model with no export-enabled Suffix, keep
        # Pyomo's absolute default.
        scaled = read_scaling(model)
        row_factor = {} if scaled is None else scaled[1]

        # The block loop is ours so every solve's verdict is checked
        # before its values are kept: a failed block restores its seed
        # values instead of poisoning the model. Since gh #609 a failure
        # no longer stops the traversal: the block DAG says exactly which
        # later blocks consumed the failed block's values, those are
        # marked skipped, and every branch independent of the failure is
        # still initialized. (`on_block_failure="stop"` restores the old
        # conservative behaviour.) 1x1 blocks solve by Pyomo's
        # single-constraint Newton, larger ones by `solver`; variables
        # outside the block are held at their current values while it
        # solves.
        #
        # Before the square solve, each block gets a numerical rank check
        # (gh #609): structural matching proves a block is *solvable in
        # principle*, not that its Jacobian has usable rank here. A block
        # that is square but near-singular is routed to a regularized
        # (or coupled) least squares, which returns a defined
        # minimum-norm point instead of whichever end of the near-null
        # direction the Newton step happened to fall off.
        n_done = 0
        n_sub = 0
        skip = {}
        outer = create_subsystem_block(square_cons, square_vars)
        with TemporarySubsystemManager(to_fix=list(outer.input_vars.values())):
            for bi, (vblk, cblk) in enumerate(
                zip(analysis.variable_blocks, analysis.constraint_blocks)
            ):
                outcome = BlockOutcome(
                    index=bi,
                    size=len(vblk),
                    constraint=cblk[0].name,
                    status="skipped",
                    depends_on=list(deps[bi]) if bi < len(deps) else [],
                )
                report.blocks.append(outcome)
                if bi in skip:
                    outcome.detail = skip[bi]
                    continue

                rcond = None
                if opts.conditioning == "auto":
                    rcond = _block_rcond(cblk, vblk)
                    outcome.rcond = rcond
                weak = rcond is not None and rcond < opts.cond_tol
                merged = []
                if weak:
                    note = (
                        f"{len(vblk)}x{len(vblk)} block at {cblk[0].name!r} is "
                        f"structurally square but numerically near-singular "
                        f"(rcond {rcond:.2e} < {opts.cond_tol:.2e})"
                    )
                    if opts.fallback == "off":
                        report.diagnostics.append(
                            note + "; solved squarely anyway "
                            "(fallback='off') -- its values may be unstable"
                        )
                        weak = False
                    else:
                        if opts.fallback == "coupled":
                            merged = [
                                k
                                for k, parents in enumerate(deps)
                                if bi in parents and k not in skip
                            ]
                        report.diagnostics.append(
                            note
                            + "; routed to the "
                            + (
                                f"coupled fallback with {len(merged)} "
                                "dependent block(s)"
                                if merged
                                else "regularized least-squares fallback"
                            )
                        )

                group_v = list(vblk)
                group_c = list(cblk)
                for k in merged:
                    group_v += list(analysis.variable_blocks[k])
                    group_c += list(analysis.constraint_blocks[k])
                snapshot = [(vd, vd.value) for vd in group_v]
                try:
                    if weak:
                        if solver is None:
                            solver = pyo.SolverFactory("pounce")
                        _regularized_solve(
                            group_c, group_v, solver, opts, ridge_vars=vblk
                        )
                        n_sub += 1
                    elif len(vblk) == 1:
                        # The default eps=1e-8 is absolute, measured in
                        # the row's raw units. An equation whose terms
                        # sit near 3e7 has a double-precision evaluation
                        # floor above that, so the linesearch can never
                        # pass. eps = 1e-8 / f is the identical test
                        # measured on the scaled row f*g(x), which is
                        # the row the Suffix says this equation is.
                        f_c = abs(row_factor.get(cblk[0], 0.0))
                        if f_c > 0.0:
                            calculate_variable_from_constraint(
                                vblk[0], cblk[0], eps=1e-8 / f_c
                            )
                        else:
                            calculate_variable_from_constraint(vblk[0], cblk[0])
                    else:
                        sub = create_subsystem_block(cblk, vblk)
                        with TemporarySubsystemManager(
                            to_fix=list(sub.input_vars.values())
                        ):
                            results = solver.solve(sub, **opts.solver_kwargs())
                        cond = str(results.solver.termination_condition)
                        if cond not in OK_TERMINATIONS:
                            raise RuntimeError(f"termination condition {cond}")
                        n_sub += 1
                    n_done += len(group_v)
                    outcome.status = "fallback" if weak else "initialized"
                    for k in merged:
                        skip[k] = ""  # solved as part of this group
                except Exception as e:  # noqa: BLE001 - report, don't raise
                    for vd, val in snapshot:
                        vd.set_value(val, skip_validation=True)
                    outcome.status = "failed"
                    outcome.detail = str(e)
                    if opts.on_block_failure == "stop":
                        doomed = set(range(bi + 1, len(analysis.variable_blocks)))
                        reason = (
                            f"traversal stopped at block {bi} "
                            f"({cblk[0].name!r}); on_block_failure='stop'"
                        )
                    else:
                        doomed = _descendants(deps, bi)
                        for k in merged:
                            doomed |= _descendants(deps, k)
                            doomed.add(k)
                        doomed -= {bi}
                        reason = (
                            f"depends on failed block {bi} ({cblk[0].name!r})"
                        )
                    n_lost = 0
                    for k in sorted(doomed):
                        if k > bi and k not in skip:
                            skip[k] = reason
                            n_lost += len(analysis.variable_blocks[k])
                    report.failures.append(
                        f"{len(vblk)}x{len(vblk)} block at {cblk[0].name!r} "
                        f"failed ({e}); its seed values are restored and "
                        f"the {n_lost} downstream variables keep their seeds"
                    )

        # Blocks consumed by a coupled fallback are recorded as such
        # rather than left reading "skipped" -- they did get values.
        for k, why in skip.items():
            if why == "" and k < len(report.blocks):
                report.blocks[k].status = "fallback"
                report.blocks[k].detail = "solved as part of a coupled fallback"
        report.n_subsystem_solves = n_sub
        report.n_vars_initialized = n_done
    finally:
        for vd in fixed_by_us:
            vd.unfix()

    return report


def _block_dependencies(igraph, variable_blocks, constraint_blocks):
    """The block DAG: parent block indices for each block (gh #609).

    Read straight off the incidence graph already in hand -- block ``k``
    depends on block ``j`` when a variable owned by ``j`` is incident to
    one of ``k``'s constraints. No new graph is constructed, so the
    single-walk guarantee of gh #444 is untouched; this is a lookup over
    edges that were built once.

    Block triangularity guarantees ``j < k`` for every edge, so the
    result is acyclic by construction and a plain reverse-reachability
    walk is enough to find a failure's descendants.
    """
    owner = {}
    for bi, blk in enumerate(variable_blocks):
        for v in blk:
            owner[id(v)] = bi
    deps = []
    for bi, cblk in enumerate(constraint_blocks):
        parents = set()
        for con in cblk:
            for v in igraph.get_adjacent_to(con):
                j = owner.get(id(v))
                if j is not None and j != bi:
                    parents.add(j)
        deps.append(sorted(parents))
    return deps


def _descendants(deps, root):
    """Every block that (transitively) consumes `root`'s values."""
    children = [[] for _ in deps]
    for k, parents in enumerate(deps):
        for j in parents:
            children[j].append(k)
    out = set()
    stack = [root]
    while stack:
        cur = stack.pop()
        for k in children[cur]:
            if k not in out:
                out.add(k)
                stack.append(k)
    return out


def _block_jacobian(cblk, vblk):
    """Dense Jacobian ``d(cblk)/d(vblk)`` at the current point, or None.

    Symbolic differentiation where Pyomo manages it; None when it does
    not, which the caller reads as "no conditioning verdict available"
    rather than as a well-conditioned block.
    """
    from pyomo.core.expr.calculus.derivatives import differentiate

    rows = []
    for con in cblk:
        try:
            grads = differentiate(con.body, wrt_list=list(vblk))
        except Exception:  # noqa: BLE001 - no verdict beats a wrong one
            return None
        row = []
        for g in grads:
            try:
                gv = float(value(g, exception=False) or 0.0)
            except Exception:  # noqa: BLE001
                return None
            if gv != gv or gv in (float("inf"), float("-inf")):
                return None
            row.append(gv)
        rows.append(row)
    return rows


def _block_rcond(cblk, vblk):
    """Reciprocal condition number of the block's *scaled* Jacobian.

    Returns None when no verdict is available -- which the caller reads
    as "no verdict", never as "well conditioned".

    The scaling is the whole point. Structural matching proves a block is
    solvable in principle; whether its Jacobian has usable rank *here* is
    a numerical question, and asking it of the raw Jacobian answers a
    units question instead. So each row is divided by its own gradient
    infinity norm (over every variable the row mentions, fixed inputs
    included -- they set the row's magnitude just as much) and each
    column is multiplied by its variable's magnitude. The entries are
    then dimensionless: relative change in a row per relative change in a
    variable, which is the quantity a condition number is meaningful for.

    Column scaling by the variable's own magnitude, rather than by the
    column norm, is deliberate: normalising columns would drive every
    1x1 block to exactly 1.0 and silently retire the check on the blocks
    that make up most of a calculation order.
    """
    from pyomo_pounce.init_scaling import SCALE_FLOOR, GRAD_FLOOR, row_scales

    jac = _block_jacobian(cblk, vblk)
    if jac is None:
        return None
    scales = row_scales(cblk)
    cols = []
    for v in vblk:
        a = abs(float(v.value)) if v.value is not None else 0.0
        cols.append(a if (a == a and a >= SCALE_FLOOR) else 1.0)
    scaled = []
    for i, con in enumerate(cblk):
        s = scales.get(con)
        if s is None or s <= GRAD_FLOOR:
            # A row with no usable gradient is rank-deficient in the
            # strongest sense available here.
            return 0.0
        scaled.append([jac[i][j] * cols[j] / s for j in range(len(vblk))])
    try:
        import numpy

        sv = numpy.linalg.svd(numpy.asarray(scaled, dtype=float), compute_uv=False)
    except Exception:  # noqa: BLE001 - numpy absent or a degenerate matrix
        return None
    if sv.size == 0 or not numpy.isfinite(sv).all() or sv[0] <= 0.0:
        return 0.0 if sv.size and sv[0] <= 0.0 else None
    return float(sv[-1] / sv[0])


def _residual(con):
    """``body - rhs`` for an equality row, as a Pyomo expression.

    ``c.body`` alone is not the residual: Pyomo keeps ``a + b == 3`` as a
    body of ``a + b`` with the 3 on ``c.upper``, so squaring the body
    would minimise the wrong thing entirely.
    """
    rhs = con.upper if con.upper is not None else con.lower
    return con.body if rhs is None else con.body - rhs


def _regularized_solve(cblk, vblk, solver, opts, ridge_vars=None):
    """Least-squares-with-a-ridge fallback for a numerically weak block.

    Minimises ``sum((c_i/s_i)**2) + lambda * sum(((v - v0)/t_j)**2)`` over
    the block's variables, with the block's rows *deactivated* -- so a
    rank-deficient block that has no unique square solution still gets a
    defined, stable one instead of an arbitrary point along its near-null
    direction. The ridge is anchored at the seed, so on a consistent but
    deficient block it picks the minimum-norm solution.

    Both terms are scaled (rows by their gradient norm, variables by
    their own magnitude), so ``lambda`` is dimensionless and a row in
    units of 1e6 does not swamp one in units of 1e-6 -- the same
    reasoning as the projection merit's scaling.

    `ridge_vars` narrows the ridge to a subset of `vblk`, which is what
    the coupled fallback needs: the extra rows it merges in are there to
    *add information* about the weak block, and ridging their variables
    too would let their pull toward the seed decide the weak block's
    answer. Measured on the module's near-singular fixture, ridging the
    union picked u = 2, v = 0 (the arbitrary end of the near-null
    direction the whole fallback exists to avoid); ridging only the weak
    block's own variables picks u = v = 1.

    Raises on a failed solve, like the square path, so the caller's
    seed-restore and DAG bookkeeping are identical either way.
    """
    import pyomo.environ as pyo

    from pyomo.util.subsystems import (
        TemporarySubsystemManager,
        create_subsystem_block,
    )

    from pyomo_pounce.init_scaling import row_factors, variable_weights

    if solver is None:
        solver = pyo.SolverFactory("pounce")
    rf = row_factors(cblk, list(vblk))
    ridge_on = list(vblk if ridge_vars is None else ridge_vars)
    anchors = [(v, float(v.value)) for v in ridge_on if v.value is not None]
    vw = variable_weights(anchors)

    sub = create_subsystem_block(cblk, vblk)
    deactivated = []
    for c in sub.component_data_objects(pyo.Constraint, active=True):
        c.deactivate()
        deactivated.append(c)
    resid = sum(rf.get(c, 1.0) ** 2 * _residual(c) ** 2 for c in cblk)
    ridge = sum(
        vw.get(id(v), 1.0) ** 2 * (v - v0) ** 2 for v, v0 in anchors
    )
    sub._pounce_regularized_objective = pyo.Objective(
        expr=resid + opts.regularization * ridge
    )
    try:
        with TemporarySubsystemManager(to_fix=list(sub.input_vars.values())):
            results = solver.solve(sub, **opts.solver_kwargs())
        cond = str(results.solver.termination_condition)
        if cond not in OK_TERMINATIONS:
            raise RuntimeError(f"regularized fallback terminated {cond}")
    finally:
        sub.del_component(sub._pounce_regularized_objective)
        for c in deactivated:
            c.activate()


def _seed_var(v) -> None:
    """Bounds-aware Newton seed for a valueless variable."""
    lo, hi = v.lb, v.ub
    # Directional presence (gh #403): `abs(b) < 1e19` called a real bound of
    # -5e20 absent, and this fell through to seeding 0.0 — outside the
    # variable's own declared box.
    lo_ok = lo is not None and lo > -1e19
    hi_ok = hi is not None and hi < 1e19
    if lo_ok and hi_ok:
        v.set_value(0.5 * (lo + hi), skip_validation=True)
    elif lo_ok:
        v.set_value(lo + 1.0, skip_validation=True)
    elif hi_ok:
        v.set_value(hi - 1.0, skip_validation=True)
    else:
        v.set_value(0.0, skip_validation=True)

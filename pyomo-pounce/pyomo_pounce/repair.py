"""Starting-point repair for Pyomo models: projection and the
fill -> repair -> block-solve pipeline.

:func:`pyomo_pounce.initialize_missing_values` fills each valueless
variable independently, so the fill can be internally inconsistent
(mole fractions that do not sum to one, flows that violate a balance).
:func:`project_to_feasible` repairs that: it moves the current point
the minimum distance onto the model's own feasible set, writing the
repaired values back into ``Var.value``. Unlike the NumPy-level
``pounce.project_to_feasible`` (which projects onto *linearized*
constraints), this solves the full nonlinear projection with POUNCE.

:func:`initialize` chains the whole story::

    report = pyomo_pounce.initialize(model, decisions=[m.feed, m.reflux])
    # fill missing values -> project onto the constraints -> solve the
    # equality blocks in calculation order (block_initialize)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional

from pyomo_pounce.block_init import (
    OK_TERMINATIONS,
    BlockInitReport,
    BlockRepairPlan,
    _flatten_vars,
    _preview,
    _seed_pin,
    _seed_var,
    block_initialize,
    structural_incidence,
    block_repair_plan,
)
from pyomo_pounce.init_options import InitOptions
from pyomo_pounce.preflight import initialize_missing_values

__all__ = ["project_to_feasible", "initialize", "InitializeReport"]


def _install_projection_scaling(model, opts):
    """Row scaling for one projection solve; returns ``(undo, opts)``.

    The factors are delivered through the model's own ``scaling_factor``
    Suffix and ``nlp_scaling_method=user-scaling`` -- the route Pyomo's NL
    writer already emits and :mod:`pyomo_pounce.scaling` already reads --
    rather than by rewriting the constraints, so nothing about the model
    the user handed us changes and the solver needs to learn nothing new.

    Entries the model already carries **win**: this fills in the rows the
    user did not tag, and an explicit ``nlp_scaling_method`` in the solver
    options wins over turning user-scaling on at all. ``undo()`` restores
    the Suffix exactly, entry by entry, including removing one we created.

    One case is deliberately left alone: a model carrying a
    ``scaling_factor`` Suffix that is *not* export-enabled. Those entries
    are the user's, and the only ways to attach ours are to flip their
    Suffix to EXPORT -- which would ship values they deliberately kept
    local -- or to shadow the name. Neither is ours to do, so the
    projection runs unscaled, exactly as before gh #609.
    """
    import pyomo.environ as pyo

    from pyomo_pounce.init_scaling import row_factors
    from pyomo_pounce.scaling import SUFFIX_NAME

    def _noop():
        return None

    if opts.scaling == "none":
        return _noop, opts

    found = [
        sfx
        for sfx in model.component_objects(pyo.Suffix, active=True, descend_into=True)
        if sfx.local_name == SUFFIX_NAME
    ]
    if found and not any(sfx.export_enabled() for sfx in found):
        return _noop, opts
    existing = [sfx for sfx in found if sfx.export_enabled()]

    auto = {}
    if opts.scaling == "auto":
        tagged = set()
        for sfx in existing:
            for key in sfx:
                tagged.update(id(d) for d in _suffix_members(key))
        rows = [
            c
            for c in model.component_data_objects(
                pyo.Constraint, active=True, descend_into=True
            )
            if id(c) not in tagged
        ]
        auto = row_factors(rows)
    if not auto and not existing:
        # "auto" with nothing to say, or "user" with no Suffix: leave the
        # solve exactly as it would have been, rather than switching it to
        # user-scaling and thereby declaring every row's factor to be 1.0.
        return _noop, opts

    if existing:
        sfx = existing[0]
        created = False
        restore = list(sfx.items())
    else:
        # The NAME is what the NL writer keys on -- a Suffix called
        # anything else is emitted by nobody and read by nobody, which is
        # a silent no-op rather than an error.
        try:
            model.add_component(
                SUFFIX_NAME, pyo.Suffix(direction=pyo.Suffix.EXPORT)
            )
        except Exception:  # noqa: BLE001 - the name is taken by a non-Suffix
            # A model may already use `scaling_factor` for a Param, Var or
            # Block of its own. Ours is not the component that gets to win
            # a name collision on the user's model, and an unscaled
            # projection is exactly what happened before gh #609, so
            # degrade to it rather than taking the call down.
            return _noop, opts
        sfx = model.component(SUFFIX_NAME)
        created = True
        restore = []
    for con, factor in auto.items():
        sfx[con] = factor
    # The merit is a sum of squared *scaled* deviations, so it is already
    # O(1) by construction; scaling it again would scale it twice.
    if model._pounce_projection_objective not in sfx:
        sfx[model._pounce_projection_objective] = 1.0

    def undo():
        if created:
            model.del_component(sfx)
        else:
            sfx.clear()
            for k, v in restore:
                sfx[k] = v

    return undo, opts.with_solver_options(nlp_scaling_method="user-scaling")


def _suffix_members(key):
    """Every ComponentData a Suffix key covers (itself, when scalar)."""
    try:
        if key.is_indexed():
            return [key[i] for i in key]
    except AttributeError:
        pass
    return [key]


def project_to_feasible(
    model,
    solver=None,
    *,
    options=None,
    tee: bool = False,
) -> str:
    """Move the current point the minimum distance onto the feasible set.

    Temporarily replaces the model's objective with
    ``min sum(w**2 * (v - v0)**2)`` over every unfixed variable that has a
    value ``v0``, solves against the model's own (active) constraints
    and bounds with POUNCE, and restores the original objective(s). The
    repaired point lands in ``Var.value``. Valueless variables get a
    bounds-aware seed and are free to move (they carry no anchor term).

    Args:
        model: A Pyomo model. Modified in place: variable values only;
            objectives/constraints/Suffixes are restored exactly.
        solver: A Pyomo solver; default ``SolverFactory("pounce")``.
        options: Solver options dict (e.g. ``{"tol": 1e-8}``), or an
            :class:`~pyomo_pounce.InitOptions` to also choose the
            scaling policy. A bare dict is always solver options.
        tee: Echo solver output. Ignored when `options` is an
            :class:`~pyomo_pounce.InitOptions`, which carries its own.

    **Scaling** (gh #609). By default (``InitOptions.scaling="auto"``) the
    anchor weights ``w`` are ``1/|v0|``, so the merit measures *relative*
    movement and a repair is shared in proportion to what each variable
    can afford rather than dumped on whichever has the smallest
    magnitude; and every untagged constraint row is normalised two-sided
    through the model's ``scaling_factor`` Suffix, so a row in units of
    1e-6 is enforced to the same relative accuracy as one in units of
    1e6. The model's own Suffix entries win over both. ``scaling="user"``
    uses only the Suffix; ``scaling="none"`` restores the unweighted
    pre-gh#609 merit. See :mod:`pyomo_pounce.init_scaling`.

    Returns the solver termination condition as a string; success is
    membership in :data:`~pyomo_pounce.block_init.OK_TERMINATIONS`
    (``"optimal"`` / ``"locallyOptimal"`` in practice). On any other
    termination the pre-projection values are restored, so a diverged
    projection never writes its iterate into the model. Raises
    ``ValueError`` when no unfixed variable has a value (nothing to
    anchor; run ``initialize_missing_values`` first).
    """
    import pyomo.environ as pyo

    from pyomo_pounce.init_scaling import variable_weights

    opts = InitOptions.coerce(options)
    if not isinstance(options, InitOptions) and tee:
        opts = replace(opts, tee=True)

    variables = [
        v
        for v in model.component_data_objects(pyo.Var, active=True, descend_into=True)
        if not v.fixed
    ]
    anchored = [(v, float(v.value)) for v in variables if v.value is not None]
    if not anchored:
        raise ValueError(
            "project_to_feasible: no unfixed variable has a value to anchor "
            "the projection; run initialize_missing_values(model) first"
        )
    snapshot = [(v, v.value) for v in variables]
    for v in variables:
        if v.value is None:
            _seed_var(v)

    if solver is None:
        solver = pyo.SolverFactory("pounce")

    if opts.scaling == "none":
        weights = {}
    else:
        weights = variable_weights(anchored)

    deactivated = []
    for obj in model.component_data_objects(
        pyo.Objective, active=True, descend_into=True
    ):
        obj.deactivate()
        deactivated.append(obj)
    model._pounce_projection_objective = pyo.Objective(
        expr=sum(
            weights.get(id(v), 1.0) ** 2 * (v - v0) ** 2 for v, v0 in anchored
        )
    )
    def undo_scaling():
        return None

    solve_opts = opts
    restore = True
    try:
        # Inside the try: everything from here on must be undone by the
        # `finally` below, including a failure while installing the
        # scaling itself.
        undo_scaling, solve_opts = _install_projection_scaling(model, opts)
        results = solver.solve(model, **solve_opts.solver_kwargs())
        cond = str(results.solver.termination_condition)
        restore = cond not in OK_TERMINATIONS
        return cond
    finally:
        undo_scaling()
        if restore:
            for v, val in snapshot:
                v.set_value(val, skip_validation=True)
        model.del_component(model._pounce_projection_objective)
        for obj in deactivated:
            obj.activate()


@dataclass
class InitializeReport:
    """What :func:`initialize` did, stage by stage."""

    n_decisions_fixed: int = 0
    #: Pinned variables held for the pipeline (repair="auto" only);
    #: not counted in ``n_decisions_fixed``.
    n_pinned: int = 0
    n_filled: int = 0
    #: Termination condition of the projection solve, or None when the
    #: projection stage was skipped.
    projection: Optional[str] = None
    block: Optional[BlockInitReport] = None
    #: The :class:`BlockRepairPlan` applied, when the specification
    #: actually needed repair; None when the decisions were used as-is.
    repair: Optional[BlockRepairPlan] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def blocks(self):
        """The block-stage per-block record, or ``[]`` when it did not
        run (gh #609). See :class:`~pyomo_pounce.BlockOutcome`."""
        return [] if self.block is None else self.block.blocks

    @property
    def ok(self) -> bool:
        proj_ok = self.projection is None or self.projection in OK_TERMINATIONS
        return proj_ok and (self.block is None or self.block.ok)

    def __str__(self) -> str:
        lines = [
            "pyomo-pounce initialize (fill -> repair -> block-solve)",
            f"  decisions held: {self.n_decisions_fixed}",
        ]
        if self.n_pinned:
            lines.append(f"  pins held     : {self.n_pinned}")
        if self.repair is not None:
            lines.append(
                f"  spec repair   : {len(self.repair.pruned)} pruned "
                f"({_preview(self.repair.pruned) or 'none'}), "
                f"{len(self.repair.pinned)} pinned "
                f"({_preview(self.repair.pinned) or 'none'})"
            )
        lines += [
            f"  values filled : {self.n_filled}",
            f"  projection    : {self.projection or 'skipped'}",
        ]
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        if self.block is not None:
            lines.extend("  " + line for line in str(self.block).splitlines())
        return "\n".join(lines)


def initialize(
    model,
    decisions=None,
    solver=None,
    *,
    repair: str = "auto",
    fill: str = "midpoint",
    project: bool = True,
    options=None,
    tee: bool = False,
) -> InitializeReport:
    """Fill, repair, and block-solve a model's starting point.

    ``decisions`` are held (fixed) at their current values for the
    **whole** pipeline — the projection must not drift the feed or the
    reflux you just specified — and released at the end. With
    ``repair="auto"`` (default) the specification is checked first:
    when holding the decisions as given leaves the equality system
    square, they are used as-is. When it does not, they become the
    candidate pool of :func:`block_repair_plan`: conflicting decisions
    are pruned (solved for instead of held), variables the equalities
    provably cannot determine are pinned automatically, and
    ``report.repair`` records the plan. The repair is call-scoped like
    the decisions themselves. ``repair="off"`` holds the decisions
    exactly as given — nothing pruned or pinned, every decision needs a
    value, and a non-square system is reported, not repaired. The three
    stages, each skippable:

    1. **Fill** — :func:`initialize_missing_values` gives every
       valueless unfixed variable a bounds-aware value
       (``fill="midpoint"`` or ``"zero"``; ``fill=None`` skips).
    2. **Repair** — :func:`project_to_feasible` moves the (possibly
       internally inconsistent) filled point the minimum distance onto
       the model's constraints (``project=False`` skips).
    3. **Block-solve** — :func:`block_initialize` solves the square
       equality system in calculation order, overwriting the repaired
       values with the consistent profile.

    ``options`` is a solver-options dict (``{"tol": 1e-8}``) or an
    :class:`~pyomo_pounce.InitOptions`. Either way **one** object is
    built here and threaded through every stage that runs — the
    projection, each block solve, and any fallback solve (gh #609).
    Before gh #609 the dict reached the projection and nothing else, so
    a tolerance tuned for the model was silently dropped by the block
    solves that produced the actual starting point. An
    :class:`~pyomo_pounce.InitOptions` additionally chooses the
    projection's scaling, the block conditioning threshold and its
    fallback, and what happens to the traversal when a block fails.

    Returns an :class:`InitializeReport`; ``report.block.square`` and
    the name lists tell you what the model is still missing.
    ``report.blocks`` is the per-block record of what was initialized,
    fell back, failed, or was skipped.
    """
    if repair not in ("auto", "off"):
        raise ValueError(
            f"initialize: repair must be 'auto' or 'off', got {repair!r}"
        )

    opts = InitOptions.coerce(options)
    if not isinstance(options, InitOptions) and tee:
        opts = replace(opts, tee=True)

    report = InitializeReport()

    # Check the specification first, on the untouched model, then hold
    # it across ALL stages: filling must not paper over a valueless
    # held decision, and the projection must not drift them. A pruned
    # decision stays free (it gets solved for); a valueless pinned
    # variable gets a nonzero bounds-aware seed before being held.
    # One structural incidence walk serves the whole call: the plan
    # filters it pre-fixing, the analyze pass post-fixing (gh #444).
    igraph = structural_incidence(model)
    plan = None
    if repair == "auto":
        plan = block_repair_plan(
            model, decision_candidates=decisions, igraph=igraph
        )
        if plan.pruned or plan.pinned:
            report.repair = plan
        if plan.redundant_constraints:
            report.warnings.append(
                f"{len(plan.redundant_constraints)} redundant/conflicting "
                "equalities no specification can satisfy: "
                + _preview(plan.redundant_constraints)
            )
        if plan.loose_variables:
            report.warnings.append(
                f"{len(plan.loose_variables)} variables undetermined by the "
                "equalities and not repairable: "
                + _preview(plan.loose_variables)
            )
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

        if fill is not None:
            report.n_filled = initialize_missing_values(model, strategy=fill)
        if project:
            try:
                report.projection = project_to_feasible(
                    model, solver=solver, options=opts
                )
                if report.projection not in OK_TERMINATIONS:
                    report.warnings.append(
                        f"projection ended {report.projection}; continuing with "
                        "the unrepaired point"
                    )
            except ValueError as e:
                report.warnings.append(str(e))
        # The held decision set is already fixed, so block_initialize
        # sees it as plain inputs; in auto mode its own plan is then a
        # no-op, and repair="off" passes through so nothing gets pinned.
        # repair="off": the plan above already ran (and its holds and
        # pins are fixed), so re-planning inside block_initialize would
        # be a structurally guaranteed no-op that still pays a full
        # incidence construction and denominator sweep (gh #444)
        report.block = block_initialize(
            model,
            solver=solver,
            repair="off",
            options=opts,
            igraph=igraph,
        )
    finally:
        for vd in fixed_by_us:
            vd.unfix()
    return report

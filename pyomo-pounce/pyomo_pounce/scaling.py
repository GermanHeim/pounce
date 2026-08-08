"""User NLP scaling from Pyomo's `scaling_factor` Suffix (gh #483).

The Pyomo/Ipopt workflow for hand-supplied scaling is::

    model.scaling_factor = Suffix(direction=Suffix.EXPORT)
    model.scaling_factor[model.obj] = 1e-3
    model.scaling_factor[model.mass_balance] = 1e2
    solver.options['nlp_scaling_method'] = 'user-scaling'

Through ASL the writer emits those entries as `.nl` `S4`/`S5`/`S6`
suffix segments and the solver reads them back. pyomo-pounce contained
no scaling code at all, so the option was accepted and meant "none": a
badly conditioned model kept converging badly with nothing to say the
suffix had been ignored.

Two things happen here now:

* **Objective and constraint factors are honored.** The subprocess
  (ASL) path needs nothing from this module — the writer emits the
  suffix segments and `NlTnlp::get_scaling_parameters` now reads them.
  The in-process sensitivity path builds no `.nl` for the solver, so
  :func:`problem_scaling` translates the Suffix into the vectors
  `pounce.Problem.set_problem_scaling` wants.
* **Variable factors reach the solver** (gh #486 stage 2). The core
  applies them as a change of variables one level below the algorithm
  and returns the solution in the model's own units, so no clone and
  no ``propagate_solution`` step is involved.
* **The sensitivity path still refuses them.** Its accessors read the
  KKT factorization directly rather than through the solver's TNLP
  chain, so on a variable-scaled solve they would report scaled-space
  numbers while promising natural units. :func:`check_no_variable_scaling`
  raises there until gh #486 stage 3 teaches that translation the
  variable factors.

Semantics, matching the AMPL/Ipopt reading of the suffix:

* Only an **export-enabled** Suffix named ``scaling_factor`` counts.
* Components the Suffix does not list are unscaled (factor 1.0), as are
  components listed with a factor of 0 — AMPL's suffix default is 0 and
  0 is not a usable scale.
* Entries on **inactive** constraints/objectives and on **fixed**
  variables are skipped: none of them is a row or column of the problem
  the solver is handed.
* An entry on a container (``scaling_factor[m.mass_balance]`` for an
  IndexedConstraint) applies to every member, which is how the NL
  writer expands it.
"""

import warnings

import pyomo.environ as pyo

#: The Suffix name AMPL, Ipopt, and Pyomo's own `core.scale_model` all use.
SUFFIX_NAME = "scaling_factor"

#: `nlp_scaling_method` value that turns the Suffix into actual scaling.
USER_SCALING = "user-scaling"


def user_scaling_requested(options):
    """True when `options` selects `nlp_scaling_method=user-scaling`.

    `options` is any mapping of solver options (Pyomo's `solver.options`,
    a per-call `options={...}`, or the merged form the in-process path
    receives). Anything else — including the default `gradient-based` —
    means the Suffix is not consulted, exactly as with Ipopt.
    """
    value = (options or {}).get("nlp_scaling_method")
    return value is not None and str(value).strip().lower() == USER_SCALING


def _iter_data(comp):
    """Every ComponentData under `comp` (itself, when scalar)."""
    if comp.is_indexed():
        for idx in comp:
            yield comp[idx]
    else:
        yield comp


def _suffixes(model):
    """Every export-enabled Suffix named `scaling_factor` on the model.

    Scanned across sub-blocks, since that is where the NL writer looks
    too; a block-local `scaling_factor` is as real as a top-level one.
    """
    found = []
    for sfx in model.component_objects(pyo.Suffix, active=True,
                                       descend_into=True):
        if sfx.local_name == SUFFIX_NAME and sfx.export_enabled():
            found.append(sfx)
    return found


def read_scaling(model):
    """Parse the model's `scaling_factor` Suffix.

    Returns ``None`` when the model declares no export-enabled
    `scaling_factor` Suffix — "the user supplied nothing", which leaves
    the solver on its own default (no scaling under `user-scaling`).

    Otherwise returns ``(obj_factor, constraints, variables)``:

    * ``obj_factor`` — float factor for the active objective (1.0 when
      untagged),
    * ``constraints`` — ``{ConstraintData: factor}`` for active
      constraints tagged with a usable (non-zero) factor,
    * ``variables`` — ``[(VarData, factor)]`` for non-fixed variables
      tagged with a factor other than 1.0, i.e. the entries pounce
      cannot honor. Empty when the model asks for nothing on that axis.
    """
    suffixes = _suffixes(model)
    if not suffixes:
        return None

    obj_factor = 1.0
    constraints = {}
    variables = []
    for sfx in suffixes:
        for key, value in sfx.items():
            for data in _iter_data(key):
                factor = float(value)
                ctype = data.ctype
                if ctype is pyo.Objective:
                    # Only the objective actually being minimized has a
                    # scale; a deactivated one is not in the problem.
                    if data.active and factor != 0.0:
                        obj_factor = factor
                elif ctype is pyo.Constraint:
                    if data.active and factor != 0.0:
                        constraints[data] = factor
                elif ctype is pyo.Var:
                    # A fixed variable is a constant folded into the
                    # rows, not a column the solver could rescale.
                    if not data.fixed and factor != 1.0:
                        variables.append((data, factor))
    return obj_factor, constraints, variables


def check_no_variable_scaling(model, max_list=5):
    """Raise if the `scaling_factor` Suffix tags variables.

    Ordinary solves apply variable factors (gh #486 stage 2), so this
    guards the SENSITIVITY path alone. Its accessors read the KKT
    factorization directly rather than through the solver's TNLP
    chain, so a variable-scaled solve would have them report
    scaled-space numbers under a natural-units contract. That is the
    silent-wrong-answer shape gh #483 was opened about, so it is a hard
    error until stage 3 carries the factors into that translation.
    """
    parsed = read_scaling(model)
    if parsed is None:
        return
    _, _, variables = parsed
    if not variables:
        return
    shown = ", ".join(f"{vd.name}={f:g}" for vd, f in variables[:max_list])
    more = (f" (+{len(variables) - max_list} more)"
            if len(variables) > max_list else "")
    raise ValueError(
        "pounce sensitivity solve: nlp_scaling_method=user-scaling, and the "
        f"model's `{SUFFIX_NAME}` Suffix sets a scaling factor on "
        f"{len(variables)} variable(s) ({shown}{more}). Ordinary solves "
        "apply variable factors, but the sensitivity accessors read the KKT "
        "factorization directly and do not yet carry those factors, so they "
        "would report scaled-space numbers while promising the model's own "
        "units. Drop the variable entries for this solve, or solve without "
        "the sensitivity declarations. Tracking issue: "
        "https://github.com/jkitchin/pounce/issues/486")


def warn_if_no_suffix(model):
    """Warn when `user-scaling` is on but the model tags nothing.

    Without a Suffix the option is a no-op — the exact silence gh #483
    is about, just with the model rather than the solver at fault (a
    misspelled Suffix name, or one built with the default LOCAL
    direction so it never reaches the solver)."""
    if read_scaling(model) is None:
        warnings.warn(
            "pounce solve: nlp_scaling_method=user-scaling was requested, but "
            f"the model declares no export-enabled `{SUFFIX_NAME}` Suffix, so "
            "no scaling will be applied. Declare it with "
            f"`model.{SUFFIX_NAME} = Suffix(direction=Suffix.EXPORT)` and tag "
            "the objective / constraints you want scaled.",
            UserWarning,
            stacklevel=3)


def problem_scaling(model, con_names, con_alias):
    """Suffix -> `(obj_factor, g_scaling)` for `Problem.set_problem_scaling`.

    Used by the in-process sensitivity path, which hands pounce
    evaluator callbacks rather than an `.nl` file and so has no suffix
    segments for the solver to read.

    `con_names` is the solve's constraint rows in `.nl` order and
    `con_alias` maps an original constraint's name to the clone name the
    declared-parameter surgery gave it, exactly as in
    `_warm_start_from_suffixes`. Rows the Suffix does not mention stay
    at 1.0. Returns ``None`` when the model declares no Suffix.

    Variable entries are not this function's business: like the `.nl`
    writer, it translates what it can and leaves the refusal to
    :func:`check_no_variable_scaling`, which callers run only when
    `user-scaling` is actually on. That keeps a `scaling_factor` Suffix
    written for Pyomo's `core.scale_model` from failing an ordinary
    solve.
    """
    parsed = read_scaling(model)
    if parsed is None:
        return None
    obj_factor, constraints, _ = parsed
    con_row = {name: i for i, name in enumerate(con_names)}
    g_scaling = [1.0] * len(con_names)
    for cd, factor in constraints.items():
        # A constraint the writer dropped (trivially satisfied, or
        # absorbed into a bound) has no row to scale; skipping it
        # matches what the ASL path does with the same suffix.
        row = con_row.get(con_alias.get(cd.name, cd.name))
        if row is not None:
            g_scaling[row] = factor
    return obj_factor, g_scaling

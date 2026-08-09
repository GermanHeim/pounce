"""POUNCE on Pyomo's modern solver interface (``pyomo.contrib.solver``).

`pyomo_pounce.pounce_solver.POUNCE` drives POUNCE through Pyomo's
*legacy* interface (`pyomo.solvers.plugins.solvers.ASL.ASL`). This
module registers a second, independent interface for the same solver
against `pyomo.contrib.solver` -- Pyomo's newer solver API, the one
`ipopt_v2` uses -- so that the modern route is a supported way to reach
POUNCE rather than a generic `ipopt_v2`-pointed-at-our-binary hack
(gh #558).

Both interfaces stay registered, under separate names::

    import pyomo_pounce
    from pyomo.environ import SolverFactory
    SolverFactory('pounce')                     # legacy (unchanged)
    SolverFactory('pounce_v2')                  # v2 engine, legacy API

    from pyomo.contrib.solver.common.factory import SolverFactory as SF2
    SF2('pounce')                               # v2 interface

Why this exists rather than "just use `ipopt_v2`": pointing `ipopt_v2`
at the POUNCE binary does solve the model, but it silently drops
everything `pyomo-pounce` adds. This class carries all of it onto the v2
lifecycle:

* bundled-binary resolution, so a stale `pounce` on `PATH` is not picked
  up silently (gh #315);
* the guard that refuses a model with live integer variables rather than
  solving the continuous relaxation and calling a fractional value
  optimal (gh #341);
* `scaling_factor` Suffix handling (gh #483 / #486);
* the sensitivity path (`declare_sens_param` -> in-process
  `pounce.Solver`), translated onto the v2 `Results`/solution-loader
  contract rather than copied -- see :class:`PounceSensSolutionLoader`.

What is inherited from Pyomo's own `Ipopt` v2 class, deliberately: the
`.nl` write, the `.sol` read, option splitting between the command line
and the `.opt` file, and the solver-log parse. POUNCE is
ASL/Ipopt-compatible on every one of those surfaces -- it accepts
`key=value` options and `option_file_name=<path>`, and its log carries
the same ``Number of Iterations....:`` and ``Total seconds in POUNCE =``
lines the Ipopt parser looks for. What is *not* compatible, and is
overridden here, is the version banner: `pounce --version` prints
``pounce X.Y.Z``, which Pyomo's Ipopt parser rejects by design (it
requires a leading ``ipopt`` precisely so that other ASL executables are
not mistaken for Ipopt).
"""

from __future__ import annotations

import datetime
import logging
import re
import subprocess
from timeit import default_timer
from typing import Any, Mapping, Sequence

from pyomo.common.collections import ComponentMap
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var

try:
    from pyomo.contrib.solver.common.base import LegacySolverWrapper
    from pyomo.contrib.solver.common.factory import (
        SolverFactory as ContribSolverFactory,
    )
    from pyomo.contrib.solver.common.results import (
        Results,
        SolutionStatus,
        TerminationCondition,
    )
    from pyomo.contrib.solver.common.solution_loader import SolutionLoader
    from pyomo.contrib.solver.common.util import (
        NoOptimalSolutionError,
        NoSolutionError,
    )
    from pyomo.contrib.solver.solvers.ipopt import Ipopt, IpoptConfig
except ImportError as exc:  # pragma: no cover - depends on the Pyomo version
    # Pyomo moved this interface into `pyomo.contrib.solver.common` (and
    # `.solvers`) in 6.9.2. Earlier versions ship an older, materially
    # different draft of it under `pyomo.contrib.solver.*` -- different
    # loader base class, different accessor names -- which this module
    # does not target. `pyomo_pounce` itself still supports pyomo>=6.0
    # through the legacy plugin, so this is raised as a clear message
    # rather than being papered over, and `pyomo_pounce/__init__.py`
    # treats it as "v2 unavailable" instead of failing the import.
    raise ImportError(
        "pyomo_pounce.v2 needs Pyomo's `pyomo.contrib.solver.common` "
        "interface, which is Pyomo 6.9.2 or newer. The legacy "
        "SolverFactory('pounce') plugin works on any supported Pyomo; "
        "upgrade Pyomo to use the v2 interface."
    ) from exc

from pyomo_pounce.pounce_solver import (
    _bundled_path,
    _warn_path_fallback,
    reject_discrete_vars,
)

logger = logging.getLogger(__name__)

__all__ = ["Pounce", "PounceConfig", "LegacyPounceSolver",
           "PounceSensSolutionLoader"]


def _default_executable():
    """Default for the `executable` config: the wheel-bundled binary when
    one is installed, else the bare name for a `PATH` lookup.

    Same precedence as the legacy plugin's `_default_executable`, and for
    the same reason -- the bundled path is deterministic while `PATH` can
    hand back a stale build that reports an identical version string
    (gh #315). Resolved once, when the CONFIG is built at import: the
    bundled binary's location is fixed at install time.
    """
    bundled = _bundled_path()
    return bundled if bundled is not None else "pounce"


class PounceConfig(IpoptConfig):
    """`IpoptConfig` with the executable defaulted to POUNCE's binary.

    Everything else -- `writer_config`, `solver_options`, `tee`,
    `time_limit`, ... -- is Pyomo's, unchanged, because POUNCE takes the
    same `.nl` input and the same `key=value` / `.opt` option forms.
    """

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        exe = self.get("executable")
        exe.set_default_value(_default_executable())
        exe.reset()
        exe._description = (
            "Preferred executable for pounce. Defaults to the `pounce` "
            "binary bundled in the installed `pounce-solver` wheel, "
            "falling back to searching the ``PATH`` for the first "
            "available ``pounce``."
        )


#: `pounce --version` prints ``pounce X.Y.Z``. Anchored, and requiring the
#: program name, for the same reason Pyomo's Ipopt parser requires
#: ``ipopt``: so that some *other* ASL executable handed to `executable=`
#: is reported as "not found" rather than silently driven as POUNCE.
_VERSION_RE = re.compile(r"^\s*pounce\s+(\d+(?:\.\d+)*)", re.IGNORECASE)


#: POUNCE exit status (the engine's `status_msg`) -> the v2 pair
#: (TerminationCondition, SolutionStatus). The legacy sensitivity path
#: maps the same statuses onto the legacy `TerminationCondition` /
#: `SolverStatus` pair in `sens._STATUS_RESULT`; the two enums are
#: different sets with different members, so this is a translation of
#: the same table, not an alias of it.
_V2_STATUS = {
    "Solve_Succeeded": (
        TerminationCondition.convergenceCriteriaSatisfied,
        SolutionStatus.optimal,
    ),
    "Solved_To_Acceptable_Level": (
        TerminationCondition.convergenceCriteriaSatisfied,
        SolutionStatus.optimal,
    ),
    # A feasible point that did not meet the convergence criteria: the
    # point is usable, the run is not a success. `unknown` is the honest
    # termination condition here -- none of the v2 members says "stopped
    # with a feasible point".
    "Feasible_Point_Found": (
        TerminationCondition.unknown,
        SolutionStatus.feasible,
    ),
    "Infeasible_Problem_Detected": (
        TerminationCondition.locallyInfeasible,
        SolutionStatus.infeasible,
    ),
    "Diverging_Iterates": (
        TerminationCondition.unbounded,
        SolutionStatus.noSolution,
    ),
    "Maximum_Iterations_Exceeded": (
        TerminationCondition.iterationLimit,
        SolutionStatus.noSolution,
    ),
    "Maximum_CpuTime_Exceeded": (
        TerminationCondition.maxTimeLimit,
        SolutionStatus.noSolution,
    ),
    "Maximum_WallTime_Exceeded": (
        TerminationCondition.maxTimeLimit,
        SolutionStatus.noSolution,
    ),
    "User_Requested_Stop": (
        TerminationCondition.interrupted,
        SolutionStatus.noSolution,
    ),
}


class PounceSensSolutionLoader(SolutionLoader):
    """v2 solution loader over an in-process sensitivity solve.

    The sensitivity path never writes a `.sol`: it hands POUNCE
    evaluator callbacks built from `pounce.read_nl` and reads the
    converged primal/dual vectors straight out of the engine. So the
    ASL-backed loader the ordinary v2 route uses has nothing to read,
    and this stands in for it -- the "deliberate translation" the v2
    lifecycle needs, since v2 returns values *through* the loader rather
    than loading them as a side effect of `solve` the way the legacy
    `load_from` path does.

    Two sign conventions are crossed on the way out, mirroring
    `sens._warm_start_from_suffixes` on the way in:

    * `dual` is the AMPL marginal ``d obj / d b = -lambda`` (gh #271),
      while the engine reports the internal ``+lambda`` in
      ``info['mult_g']`` -- so duals negate.
    * `ipopt_zU_out` is negative at an active upper bound (Ipopt's
      convention, gh #296) while the engine's ``mult_x_U`` is the
      internal non-negative ``z_u`` -- so it negates too; ``zL`` is
      positive in both. The reduced cost is then combined from the two
      bound multipliers exactly as `IpoptSolutionLoader` combines the
      `.sol` suffixes, so `rc` means the same thing on both routes.
    """

    def __init__(self, model, capture, has_solution=True):
        self._pyomo_model = model
        self._capture = capture
        self._has_solution = has_solution
        self._var_row = {n: i for i, n in enumerate(capture["var_names"])}
        self._con_row = {n: i for i, n in enumerate(capture["con_names"])}
        self._con_alias = capture.get("con_alias") or {}

    def get_number_of_solutions(self) -> int:
        return 1 if self._has_solution else 0

    def _require_solution(self):
        if not self._has_solution:
            raise NoSolutionError()

    def _vector(self, key):
        vec = self._capture.get("info", {}).get(key)
        return None if vec is None else vec

    def get_vars(
        self, vars_to_load: Sequence[Any] | None = None
    ) -> Mapping[Any, float]:
        self._require_solution()
        x = self._capture["x"]
        out = ComponentMap()
        if vars_to_load is None:
            for name, val in zip(self._capture["var_names"], x):
                vd = self._pyomo_model.find_component(name)
                if vd is not None:
                    out[vd] = float(val)
            return out
        for vd in vars_to_load:
            row = self._var_row.get(vd.name)
            if row is not None:
                out[vd] = float(x[row])
        return out

    def _row_of(self, con_data):
        """The solve's row index for a constraint of the *original* model.

        A constraint the declared-parameter surgery replaced lives in the
        solve under its clone's name, so it is reached through the alias
        map -- the same indirection the warm-start reader applies.
        """
        name = con_data.name
        return self._con_row.get(self._con_alias.get(name, name))

    def get_duals(
        self, cons_to_load: Sequence[Any] | None = None
    ) -> dict:
        self._require_solution()
        lam = self._vector("mult_g")
        if lam is None:
            raise NoSolutionError(
                "pounce: this solve returned no constraint multipliers, so "
                "duals are not available")
        if cons_to_load is None:
            cons_to_load = self._pyomo_model.component_data_objects(
                Constraint, active=True, descend_into=True)
        out = {}
        for cd in cons_to_load:
            row = self._row_of(cd)
            if row is not None:
                # engine's internal +lambda -> the AMPL marginal Pyomo's
                # `dual` suffix carries
                out[cd] = -float(lam[row])
        return out

    def get_reduced_costs(
        self, vars_to_load: Sequence[Any] | None = None
    ) -> Mapping[Any, float]:
        self._require_solution()
        zl = self._vector("mult_x_L")
        zu = self._vector("mult_x_U")
        if zl is None or zu is None:
            raise NoSolutionError(
                "pounce: this solve returned no bound multipliers, so "
                "reduced costs are not available")
        if vars_to_load is None:
            vars_to_load = [
                vd for vd in (
                    self._pyomo_model.find_component(n)
                    for n in self._capture["var_names"])
                if vd is not None
            ]
        out = ComponentMap()
        for vd in vars_to_load:
            row = self._var_row.get(vd.name)
            if row is None:
                continue
            lo = float(zl[row])
            # Ipopt's `ipopt_zU_out` convention, so that the combination
            # below is the same arithmetic IpoptSolutionLoader does
            hi = -float(zu[row])
            out[vd] = hi if abs(hi) > abs(lo) else lo
        return out


class Pounce(Ipopt):
    """Interface to the POUNCE NLP solver (NL file based)."""

    CONFIG = PounceConfig()

    #: Availability/version cache. Redeclared rather than inherited: the
    #: cache is keyed by executable path only, and `Pounce` and `Ipopt`
    #: parse the version banner differently -- sharing one dict would let
    #: an `Ipopt` probe of some path poison the answer here (and vice
    #: versa) for the same path.
    _exe_cache: dict = {}

    def _get_version(self, exe):
        """POUNCE's version, from ``pounce --version`` (``pounce X.Y.Z``).

        Pyomo's `Ipopt._get_version` demands a banner starting `ipopt`,
        so POUNCE reads as "not found" through it -- which is why this
        override exists, and why it is just as strict about the program
        name: an executable that does not announce itself as `pounce`
        must not be driven as POUNCE.
        """
        try:
            return self._exe_cache[exe]
        except KeyError:
            pass
        if exe is None:
            self._exe_cache[None] = None
            return None
        try:
            results = subprocess.run(
                [str(exe), "--version"],
                timeout=self._version_timeout,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            self._exe_cache[exe] = None
            return None
        ver = None
        if not results.returncode:
            m = _VERSION_RE.match(results.stdout)
            if m:
                try:
                    ver = tuple(int(i) for i in m.group(1).split("."))
                except ValueError:
                    ver = None
        if ver is None:
            logger.warning(
                f"Failed parsing POUNCE version: '{exe} --version':"
                f"\n\n{results.stdout}")
        self._exe_cache[exe] = ver
        return ver

    def _check_executable(self, config):
        """Warn once if this solve will run a `PATH` binary rather than the
        wheel-bundled one -- the case that silently ran a stale,
        dual-sign-flipped build before gh #315.

        Only for the *default* executable: an explicitly passed
        `executable=` is the user naming a binary on purpose, and warning
        about it would be noise. Read off the per-solve `config`, not
        `self.config`, so that a call-time `solve(model, executable=...)`
        counts as explicit too -- the flag propagates into the derived
        config, an instance-level one does not propagate back out.
        """
        if _bundled_path() is not None:
            return
        # `_userSet` is Pyomo-internal; via getattr so that a rename
        # downgrades this to a spurious warning rather than breaking
        # every solve.
        if getattr(config.get("executable"), "_userSet", False):
            return
        path = config.executable.path()
        if path is not None:
            _warn_path_fallback(path)

    def solve(self, model, **kwds) -> Results:
        """Solve `model` with POUNCE, returning a v2 `Results`.

        Beyond `Ipopt.solve`: the integer-variable guard, the
        `scaling_factor` Suffix check, and the in-process sensitivity
        route when the model carries `declare_sens_param` (or the
        equivalent call-time keywords) declarations.
        """
        from pyomo_pounce.scaling import (
            user_scaling_requested,
            warn_if_no_suffix,
        )
        from pyomo_pounce.sens import has_declarations

        explicit = {k: kwds.pop(k) for k in
                    ("sens_params", "fitted", "residuals") if k in kwds}

        reject_discrete_vars(model)

        config: PounceConfig = self.config(value=kwds, preserve_implicit=True)
        self._check_executable(config)

        # Solver options as this solve will actually see them. Deriving
        # `config` from `kwds` above already layered them the way
        # `Ipopt.solve` does -- instance-level `solver.config.
        # solver_options[...]` merged with per-call `solver_options=`,
        # the per-call value winning -- so this reads the merged view
        # rather than re-implementing the precedence. Reading only the
        # per-call half here is what silently un-tuned every model the
        # day it gained a declaration on the legacy side (gh #432).
        opts = dict(config.solver_options.value() or {})
        if user_scaling_requested(opts):
            warn_if_no_suffix(model)

        if has_declarations(model) or explicit:
            return self._sens_solve(model, config, opts, explicit)
        return super().solve(model, **kwds)

    def _sens_solve(self, model, config, opts, explicit) -> Results:
        """The sensitivity route, translated onto the v2 contract.

        `sens.sens_solve` solves in-process so that the converged KKT
        factorization stays available for `gradient()` / `estimate()` /
        `covariance()`, and returns a *legacy* SolverResults. The v2
        contract is a different object with a different status enum, and
        -- the substantive difference -- it hands the solution back
        through a solution loader that the caller may decline to load
        (`load_solutions=False`) or read without loading. So the raw
        solve is captured and re-expressed here rather than the legacy
        results object being adapted.
        """
        from pyomo_pounce.sens import sens_solve

        start_time = default_timer()
        results = Results()
        results.timing_info.start_timestamp = datetime.datetime.now(
            datetime.timezone.utc)
        results.solver_name = self.name
        results.solver_version = self._get_version(config.executable.path())

        opts = dict(opts)
        if config.time_limit is not None:
            # same mapping `_run_ipopt` applies for the ordinary route
            opts.setdefault("max_cpu_time", config.time_limit)

        # `sens_solve` writes the converged iterate onto the model's
        # variables itself -- the legacy contract, where loading IS the
        # solve's side effect. v2 promises the opposite: with
        # `load_solutions=False` the model must come back untouched and
        # the values are read through the loader. So snapshot first and
        # roll back after; the loader serves the values either way, and
        # the retained KKT session is unaffected because it answers from
        # its own `base_x`, not from the model.
        restore = None
        if not config.load_solutions:
            restore = [(vd, vd.value, vd.stale) for vd in
                       model.component_data_objects(
                           Var, active=True, descend_into=True)]

        capture = {}
        try:
            sens_solve(model, tee=bool(config.tee), options=opts,
                       capture=capture, **explicit)
        finally:
            if restore is not None:
                for vd, val, stale in restore:
                    vd.set_value(val, skip_validation=True)
                    vd.stale = stale

        tc, ss = _V2_STATUS.get(
            capture.get("status_msg", ""),
            (TerminationCondition.error, SolutionStatus.noSolution))
        results.termination_condition = tc
        results.solution_status = ss

        info = capture.get("info", {})
        iters = info.get("iter_count")
        results.extra_info.iteration_count = (
            None if iters is None else int(iters))
        results.extra_info.solver_message = capture.get("status_msg")
        # Same key the ordinary route ends up with: Pyomo's log parser
        # turns POUNCE's `Total seconds in POUNCE = …` summary line into
        # `timing_info.POUNCE`, so a caller reading the solver's own time
        # finds it under one name on both routes. The quantity differs
        # slightly -- there it is POUNCE's self-reported time, here it is
        # measured around the in-process call (the tee stream/decode
        # excluded) because no log is parsed.
        results.timing_info.POUNCE = capture.get("solve_secs")

        has_solution = ss is not SolutionStatus.noSolution
        results.solution_loader = PounceSensSolutionLoader(
            model, capture, has_solution=has_solution)

        if (config.raise_exception_on_nonoptimal_result
                and ss is not SolutionStatus.optimal):
            raise NoOptimalSolutionError()

        if config.load_solutions:
            if not has_solution:
                raise NoSolutionError()
            results.solution_loader.load_solution()

        if has_solution:
            obj = info.get("obj_val")
            if obj is not None:
                results.incumbent_objective = float(obj)

        results.solver_config = config
        results.timing_info.wall_time = default_timer() - start_time
        return results


class LegacyPounceSolver(LegacySolverWrapper, Pounce):
    """`SolverFactory('pounce_v2')`: the v2 engine behind the legacy API."""


# Registered explicitly (not as a decorator) so the legacy wrapper above
# is the one installed, rather than the anonymous subclass the factory
# would synthesize. `legacy_name` is what keeps this additive: the v2
# factory gets `pounce`, the legacy factory gets `pounce_v2`, and the
# legacy factory's existing `pounce` -> `pounce_solver.POUNCE`
# registration is untouched. Same split Pyomo itself uses for
# `ipopt` / `ipopt_v2`.
ContribSolverFactory.register(
    name="pounce",
    legacy_name="pounce_v2",
    doc="The POUNCE interior-point NLP solver",
)(Pounce, LegacyPounceSolver)

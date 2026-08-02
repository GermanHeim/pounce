"""Solve Pyomo models with POUNCE, in the browser.

This is the Python side of the Pyodide app: Pyomo builds the model, its own
NL writer emits the `.nl`, the POUNCE WebAssembly module solves it, and the
`.sol` that comes back is loaded onto the model — so `x.value` and
`model.dual[c]` read the way they would after any AMPL-interfaced solve.

Everything runs in the tab. Nothing is uploaded.

    from pyomo.environ import *
    import pounce_browser

    m = ConcreteModel()
    m.x = Var([1, 2], initialize=1.0)
    m.c = Constraint(expr=m.x[1] ** 2 + m.x[2] ** 2 == 1)
    m.obj = Objective(expr=m.x[1])

    res = pounce_browser.solve(m, options="max_iter 200")
    print(res.status, res.objective, value(m.x[1]))

The wasm module is reached through a *backend*: a callable taking the `.nl`
text and an `ipopt.opt`-format options string, returning a dict with the
solve's JSON payload and the `.sol` text. The browser worker installs one
that calls the wasm exports; a test can install one that shells out to a
native binary, which is how the round trip is checked off-browser
(`crates/pounce-wasm/tests/pyomo_roundtrip.py`).
"""

import io
import json
from dataclasses import dataclass, field

__all__ = ["solve", "set_backend", "SolveResult", "AMPL_INFINITY"]

# AMPL's "no bound" sentinel, mirrored from the .nl reader.
AMPL_INFINITY = 1.0e19

_backend = None


def set_backend(fn):
    """Install the callable that runs a solve.

    `fn(nl_text: str, options: str) -> {"result": dict, "sol": str, "log": str}`
    """
    global _backend
    _backend = fn


@dataclass
class SolveResult:
    """What a solve reports back. `status` is POUNCE's own return code name
    (`SolveSucceeded`, `MaximumIterationsExceeded`, …), not a Pyomo
    `TerminationCondition` — the browser app shows the solver's verdict
    verbatim rather than translating it."""

    status: str
    success: bool
    objective: float
    iterations: int
    wall_time: float
    constraint_violation: float
    dual_infeasibility: float
    log: str = ""
    sol: str = ""
    raw: dict = field(default_factory=dict)

    def __repr__(self):  # keeps `print(res)` useful in the demo console
        return (
            f"SolveResult(status={self.status!r}, objective={self.objective!r}, "
            f"iterations={self.iterations}, wall_time={self.wall_time:.3f}s)"
        )


def write_nl(model):
    """Write `model` as AMPL `.nl` text.

    Returns `(nl_text, info)` where `info` is Pyomo's `NLWriterInfo`: its
    `variables` and `constraints` lists are in the file's column and row
    order, which is exactly the order the `.sol` blocks come back in. Using
    that mapping avoids a symbol-map round trip and cannot drift out of sync
    with the file we just wrote.
    """
    from pyomo.repn.plugins.nl_writer import NLWriter

    stream = io.StringIO()
    info = NLWriter().write(model, stream, symbolic_solver_labels=False)
    return stream.getvalue(), info


def parse_sol(text, n_vars, n_cons):
    """Parse the primal and dual blocks of an AMPL `.sol` file.

    Returns `(x, duals, solve_result_num, message)`. Deliberately small: it
    reads the blocks POUNCE writes (message, `Options`, the four-integer
    count block, duals, primals, `objno`) and ignores suffix blocks, which
    the caller does not need to reload a model.
    """
    lines = text.splitlines()
    try:
        opt_at = next(i for i, line in enumerate(lines) if line.strip() == "Options")
    except StopIteration:
        raise ValueError("not a .sol file: no Options section")

    message = "\n".join(line for line in lines[:opt_at] if line.strip())
    i = opt_at + 1
    n_opts = int(lines[i])
    i += 1 + n_opts

    n_dual, _m, n_primal, _n = (int(lines[i + k]) for k in range(4))
    i += 4

    duals = [float(lines[i + k]) for k in range(n_dual)]
    i += n_dual
    x = [float(lines[i + k]) for k in range(n_primal)]
    i += n_primal

    solve_result_num = -1
    for line in lines[i:]:
        if line.startswith("objno"):
            solve_result_num = int(line.split()[2])
            break

    if n_primal != n_vars or n_dual != n_cons:
        raise ValueError(
            f".sol has {n_primal} primals / {n_dual} duals, "
            f"model has {n_vars} variables / {n_cons} constraints"
        )
    return x, duals, solve_result_num, message


def load_solution(model, info, x, duals):
    """Put a parsed `.sol` back on the model.

    Variables take their values by position; constraint duals land in a
    `dual` Suffix when the model declares one. AMPL's dual block is the
    marginal value `dobj/db = -lambda`, which is what Pyomo's `model.dual`
    expects, so the values are used as they come off the file.
    """
    for var, value in zip(info.variables, x):
        var.set_value(value, skip_validation=True)

    suffix = getattr(model, "dual", None)
    if suffix is not None and suffix.ctype.__name__ == "Suffix":
        for con, dual in zip(info.constraints, duals):
            suffix[con] = dual


def solve(model, options="", load_solution_into_model=True, tee=True):
    """Solve `model` with POUNCE and load the solution back onto it.

    `options` is `ipopt.opt`-format text (`name value` per line) — the same
    option names the POUNCE CLI and Python API take.
    """
    if _backend is None:
        raise RuntimeError(
            "no POUNCE backend installed — call set_backend() first "
            "(the browser app does this for you)"
        )

    nl_text, info = write_nl(model)
    payload = _backend(nl_text, options)
    # A JS caller hands back a JSON string; a Python one hands back a dict.
    if isinstance(payload, str):
        payload = json.loads(payload)

    result = payload["result"]
    if "error" in result:
        raise RuntimeError(f"POUNCE: {result['error']}")

    sol_text = payload.get("sol") or ""
    if load_solution_into_model and sol_text:
        x, duals, _srn, _msg = parse_sol(sol_text, len(info.variables), len(info.constraints))
        load_solution(model, info, x, duals)

    log = payload.get("log", "")
    if tee and log:
        print(log)

    return SolveResult(
        status=result["status"],
        success=result["success"],
        objective=result["objective"],
        iterations=result["iterations"],
        wall_time=result["wall_time_secs"],
        constraint_violation=result["constraint_violation"],
        dual_infeasibility=result["dual_infeasibility"],
        log=log,
        sol=sol_text,
        raw=result,
    )

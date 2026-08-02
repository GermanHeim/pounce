"""Pyomo → .nl → wasm → .sol → Pyomo, without a browser.

The Pyodide app's Python layer (`web-python/pounce_browser.py`) is the part
that can silently go wrong: Pyomo's NL writer orders variables and rows its
own way, and a `.sol` loaded back in a different order puts plausible
numbers on the wrong components. This drives that layer with the real wasm
module — through Node, standing in for the browser — and checks the values
against a model whose optimum is known in closed form.

    pip install pyomo
    crates/pounce-wasm/build.sh
    python crates/pounce-wasm/tests/pyomo_roundtrip.py
"""

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "web-python"))

import pounce_browser  # noqa: E402  (path set above)
from pyomo.environ import (  # noqa: E402
    ConcreteModel,
    Constraint,
    Objective,
    Suffix,
    Var,
    value,
)


def node_backend(nl_text, options):
    """Run a solve the way the browser worker does, via Node + the wasm."""
    with tempfile.NamedTemporaryFile("w", suffix=".nl", delete=False) as fh:
        fh.write(nl_text)
        nl_path = fh.name
    try:
        proc = subprocess.run(
            ["node", str(HERE / "solve_nl.mjs"), nl_path, options],
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        Path(nl_path).unlink(missing_ok=True)
    payload = json.loads(proc.stdout)
    payload["log"] = proc.stderr
    return payload


def build_model():
    """min x1  s.t.  x1² + x2² == 1,  x1 + x2 >= 0.

    Both constraints are active at the optimum: x1 = -1/√2, x2 = +1/√2,
    objective -1/√2. Stationarity of L = x1 + λ₁(x1²+x2²-1) + λ₂(x1+x2)
    there gives λ₁ = √2/4 and λ₂ = -1/2, so AMPL's marginal values (-λ)
    are -0.353553 on the circle and +0.5 on the halfspace. Closed-form
    duals are what pin the dual block's sign *and* its row order.
    """
    m = ConcreteModel()
    m.x1 = Var(initialize=0.5, bounds=(-10, 10))
    m.x2 = Var(initialize=0.5, bounds=(-10, 10))
    m.circle = Constraint(expr=m.x1**2 + m.x2**2 == 1)
    m.halfspace = Constraint(expr=m.x1 + m.x2 >= 0)
    m.obj = Objective(expr=m.x1)
    m.dual = Suffix(direction=Suffix.IMPORT)
    return m


def main():
    pounce_browser.set_backend(node_backend)
    m = build_model()
    res = pounce_browser.solve(m, options="print_level 0\n", tee=False)

    assert res.success, f"solve failed: {res.status}"
    assert res.status == "SolveSucceeded", res.status
    assert math.isclose(res.objective, -math.sqrt(0.5), rel_tol=1e-6), res.objective

    # The values must land on the right components — the whole point of the
    # writer-order mapping. A transposed load would put -0.707 on x2.
    assert math.isclose(value(m.x1), -math.sqrt(0.5), rel_tol=1e-6), value(m.x1)
    assert math.isclose(value(m.x2), math.sqrt(0.5), rel_tol=1e-6), value(m.x2)

    # Duals land on the right rows, with AMPL's marginal-value sign. Swap
    # the two rows and both of these fail, which is the point.
    assert m.circle in m.dual, "no dual for the equality row"
    assert math.isclose(m.dual[m.circle], -math.sqrt(2) / 4, rel_tol=1e-4), m.dual[m.circle]
    assert math.isclose(m.dual[m.halfspace], 0.5, rel_tol=1e-4), m.dual[m.halfspace]

    # And the .sol text really is an AMPL solution file.
    assert res.sol.startswith("POUNCE "), res.sol[:60]
    assert "\nobjno 0 0\n" in res.sol, res.sol[-200:]

    print(
        f"ok — Pyomo round trip through wasm: {res.status}, "
        f"objective {res.objective:.9f}, x1={value(m.x1):.6f}, "
        f"dual[circle]={m.dual[m.circle]:.6f}"
    )


if __name__ == "__main__":
    main()

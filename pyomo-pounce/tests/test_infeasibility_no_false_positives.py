"""Property test: a feasible model must never be reported infeasible.

Hand-written adversarial cases kept finding real defects in the presolve
infeasibility path, but each round only found what someone thought to write
down — and every round broke the previous round's fix. This generates the
cases instead.

Each instance picks a witness point ``x*`` strictly inside the box **first**,
then builds every constraint so it holds at ``x*``. Feasibility is therefore
not an assumption; it is guaranteed by construction. Any verdict in AMPL's
``200..299`` infeasible band is a false positive.

Magnitudes deliberately span the ranges that broke earlier rounds: float noise
(``1e-320``), sub-tolerance gaps, unit scale, and near-overflow (``1e30``).

The assertion is a *ratchet*, not zero. The floating-point interval arithmetic
underneath is not exact, and a handful of extreme-scale knife-edge instances
(zero-slack active constraints on ``1e-9``-wide boxes) are still misclassified.
Pinning the measured rate stops it silently getting worse — a change that pushes
it up is a regression even if every hand-written case still passes. Lowering
``MAX_FALSE_POSITIVES`` as the path improves is the point.
"""

from __future__ import annotations

import random
import shutil
import subprocess
from pathlib import Path

import pytest

pyo = pytest.importorskip("pyomo.environ")

#: Instances per run. Large enough to be sensitive, small enough for CI.
N_INSTANCES = 200

#: Measured on the implementation this test landed with: 3 of 400 (~0.75%).
#: Scaled to N_INSTANCES with headroom for generator jitter across seeds.
MAX_FALSE_POSITIVES = 4

_SCALES = [1e-320, 1e-12, 1e-8, 1.0, 1e8, 1e18, 1e30]

_OPTS = [
    "solver_selection=nlp",
    "presolve=yes",
    "presolve_fbbt=yes",
    "presolve_auxiliary=yes",
    "presolve_linear_eq_reduction=yes",
    "presolve_redundant_constraint_removal=yes",
    "print_level=0",
]


def _pounce() -> str:
    exe = shutil.which("pounce")
    if exe is None:
        pytest.skip("pounce not on PATH")
    return exe


def _build(seed: int):
    """A model that is feasible by construction, plus its witness point."""
    rng = random.Random(seed)
    n = rng.randint(1, 4)
    m = pyo.ConcreteModel()
    lo = [rng.choice([0.0, -1.0, -rng.random() * 10]) for _ in range(n)]
    hi = [lo[i] + rng.choice([1e-9, 1.0, 10.0, 1e6]) for i in range(n)]
    xs = [lo[i] + 0.5 * (hi[i] - lo[i]) for i in range(n)]

    m.I = pyo.RangeSet(0, n - 1)
    m.x = pyo.Var(
        m.I,
        bounds=lambda _m, i: (lo[i], hi[i]),
        initialize=lambda _m, i: xs[i],
    )
    m.o = pyo.Objective(expr=sum((m.x[i] - xs[i]) ** 2 for i in range(n)))

    m.c = pyo.ConstraintList()
    for _ in range(rng.randint(1, 3)):
        s = rng.choice(_SCALES)
        coef = [rng.choice([0.0, 1.0, -1.0, rng.random()]) * s for _ in range(n)]
        at_star = sum(coef[i] * xs[i] for i in range(n))
        slack = abs(rng.choice([0.0, 1e-12, 1e-6, 1.0]))
        body = sum(coef[i] * m.x[i] for i in range(n))
        try:
            m.c.add(body >= at_star - slack if rng.random() < 0.5 else body <= at_star + slack)
        except (ValueError, TypeError):
            pass  # degenerate row (all-zero coefficients); skip it
    return m


def _solve_result_num(sol: Path) -> int | None:
    for line in sol.read_text().splitlines():
        if line.startswith("objno"):
            return int(line.split()[2])
    return None


def test_feasible_models_are_not_reported_infeasible(tmp_path):
    exe = _pounce()
    executed = 0
    false_positives = []

    for seed in range(N_INSTANCES):
        try:
            model = _build(seed)
            nl = tmp_path / f"f{seed}.nl"
            sol = tmp_path / f"f{seed}.sol"
            model.write(str(nl), io_options={"symbolic_solver_labels": True})
        except Exception:
            continue

        subprocess.run(
            [exe, str(nl), "-AMPL", "--sol-output", str(sol), *_OPTS],
            capture_output=True,
            timeout=120,
        )
        if not sol.exists():
            continue
        executed += 1
        code = _solve_result_num(sol)
        if code is not None and 200 <= code < 300:
            false_positives.append((seed, code))

    # A probe that silently exercised nothing is worse than a failing one.
    assert executed > 0, "generated zero solvable instances — the probe is broken"
    assert executed >= N_INSTANCES // 2, (
        f"only {executed}/{N_INSTANCES} instances reached the solver; the "
        "generator or the CLI is failing, so this run proves nothing"
    )

    assert len(false_positives) <= MAX_FALSE_POSITIVES, (
        f"{len(false_positives)}/{executed} feasible-by-construction models were "
        f"reported in the AMPL infeasible band (limit {MAX_FALSE_POSITIVES}).\n"
        f"Every one is a model with a known interior point being called "
        f"infeasible. Offending seeds: {false_positives[:10]}"
    )

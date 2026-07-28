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

The assertion is a *ratchet*. Pinning the measured rate stops it silently
getting worse — a change that pushes it up is a regression even if every
hand-written case still passes. Lowering ``MAX_FALSE_POSITIVES`` as the paths
improve is the point, and it has now reached zero:

* gh #379 took the **numerical** path to 0/400. No path may claim infeasibility
  while the model's own starting point satisfies every constraint
  (``pounce_algorithm::infeasibility_refutation``).
* gh #396 took the **presolve certification** path to 0/400. Its witness gate
  was mishandling the absent-bound sentinel in both directions on one row —
  scoring a violation of a bound the row did not have, and discarding a real
  bound whose magnitude exceeded the sentinel.

Zero is now the claim, not an aspiration: hold it there. A single false positive
means a feasible-by-construction model is being reported infeasible, which is
the defect this file exists to catch.
"""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import pytest

pyo = pytest.importorskip("pyomo.environ")

#: Instances per run. Large enough to be sensitive, small enough for CI.
N_INSTANCES = 200

#: Ratchet, now at its floor. Landed at 3 of 400 (~0.75%), then 1 of 400 after
#: gh #379, then 0 of 400 after gh #396 — measured over the full 400-seed range
#: on both the numerical (``presolve=no``) and full-presolve option sets.
#: Raising this again needs a reason recorded here, not a bump.
MAX_FALSE_POSITIVES = 0

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


def test_feasible_models_are_not_reported_infeasible(tmp_path, pounce_exe):
    exe = pounce_exe
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

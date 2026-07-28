"""Scale invariance: multiplying a constraint by s > 0 must not change the verdict.

Multiplying a row by a positive constant leaves the feasible set *exactly*
unchanged — it is the same mathematical problem written differently. So the
solver's verdict at every scale must agree. Where it does not, a modelling
choice that carries no mathematical content is silently changing the answer.

This is the regression detector for the scale-invariance work. Several defects
found in this area shared one root cause: a scale-*dependent* quantity compared
against an *absolute* threshold. A corpus sweep cannot detect that class,
because real models cluster near O(1) — this harness manufactures the stress
directly by sweeping the row scaling across 25 decades.

The sharpest cell it finds: ``x >= 2`` over ``x in [0, 1]`` reports
``Infeasible_Problem_Detected`` unscaled and ``Solve_Succeeded`` when every row
is multiplied by ``1e-12``. Same empty feasible set, opposite verdicts —
because at that scale the violation falls under an absolute feasibility
tolerance and the solver accepts it.

**Both directions are covered on purpose.** Feasible models guard against
tightening a tolerance into false infeasibility; infeasible models guard
against loosening one into a missed verdict. A relative tolerance is *more*
permissive at large scale, so a fix aimed only at false positives could quietly
lose true infeasibility verdicts. Any step that improves one column while
worsening the other is a regression, not progress.

The assertion is a **ratchet against the recorded baseline**, not zero: the
current implementation is not scale-invariant, and pretending otherwise would
make the test fail on arrival. Lower the baselines as the migration lands —
that is the point.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pyo = pytest.importorskip("pyomo.environ")

#: Row-scaling exponents. s = 10**k multiplies every constraint row.
KS = list(range(-12, 13, 2))

_OPTS = ["solver_selection=nlp", "print_level=0"]


def _pounce() -> str:
    exe = shutil.which("pounce")
    if exe is None:
        pytest.skip("pounce not on PATH")
    return exe


# --- models -----------------------------------------------------------------
# Each builder takes the scale factor and applies it to every row, so the
# feasible set is identical for all k.

def _feas_simple(m, s):
    m.x = pyo.Var(bounds=(0, 1), initialize=0.5)
    m.o = pyo.Objective(expr=(m.x - 0.5) ** 2)
    m.c = pyo.Constraint(expr=s * m.x >= s * 0.2)


def _feas_two(m, s):
    m.x = pyo.Var(bounds=(0, 1), initialize=0.5)
    m.y = pyo.Var(bounds=(0, 1), initialize=0.5)
    m.o = pyo.Objective(expr=(m.x - 0.3) ** 2 + (m.y - 0.7) ** 2)
    m.c1 = pyo.Constraint(expr=s * (m.x + m.y) >= s * 0.5)
    m.c2 = pyo.Constraint(expr=s * (m.x - m.y) <= s * 0.9)


def _feas_eq(m, s):
    m.x = pyo.Var(bounds=(0, 10), initialize=1.0)
    m.y = pyo.Var(bounds=(0, 10), initialize=1.0)
    m.o = pyo.Objective(expr=m.x**2 + m.y**2)
    m.e = pyo.Constraint(expr=s * (m.x + m.y) == s * 4.0)


def _inf_372(m, s):
    """The gh #372 model: 0 <= x <= 0.6 with x >= 0.7."""
    m.x = pyo.Var(bounds=(0.0, 0.6), initialize=0.5)
    m.o = pyo.Objective(expr=m.x**3 + m.x**2)
    m.c = pyo.Constraint(expr=s * m.x >= s * 0.7)


def _inf_clear(m, s):
    m.x = pyo.Var(bounds=(0, 1), initialize=0.5)
    m.o = pyo.Objective(expr=m.x**2)
    m.c = pyo.Constraint(expr=s * m.x >= s * 2.0)


def _inf_two(m, s):
    m.x = pyo.Var(bounds=(0, 1), initialize=0.5)
    m.y = pyo.Var(bounds=(0, 1), initialize=0.5)
    m.o = pyo.Objective(expr=m.x**2 + m.y**2)
    m.c = pyo.Constraint(expr=s * (m.x + m.y) >= s * 3.0)


def _inf_eq(m, s):
    m.x = pyo.Var(bounds=(0, 1), initialize=0.5)
    m.o = pyo.Objective(expr=m.x**2)
    m.e1 = pyo.Constraint(expr=s * m.x == s * 0.2)
    m.e2 = pyo.Constraint(expr=s * m.x == s * 0.8)


MODELS = [
    ("feas_simple", _feas_simple, "SOLVED"),
    ("feas_two", _feas_two, "SOLVED"),
    ("feas_eq", _feas_eq, "SOLVED"),
    ("inf_372", _inf_372, "INFEAS"),
    ("inf_clear", _inf_clear, "INFEAS"),
    ("inf_two", _inf_two, "INFEAS"),
    ("inf_eq", _inf_eq, "INFEAS"),
]

#: Wrong-verdict cells per model, out of len(KS), measured on the
#: implementation this harness landed with. **Lower these as the migration
#: lands.** A model whose count rises is a regression even if every other test
#: passes.
BASELINE_WRONG = {
    "feas_simple": 0,
    "feas_two": 0,
    "feas_eq": 0,
    "inf_372": 0,
    "inf_clear": 0,
    "inf_two": 0,
    # gh#387: the DOF gate now consults bound-propagation certification, so
    # the contradiction is proved at every scale the certification is willing
    # to speak to. The 3 remaining cells are k in {-12, -10, -8}, where every
    # point in the box satisfies both rows within the solver's own acceptance
    # tolerance and the fail-closed witness rule withholds the proof.
    "inf_eq": 3,
}


def _band(code: int | None) -> str:
    if code is None:
        return "?"
    if code < 200:
        return "SOLVED"  # 0..99 solved, 100..199 solved-with-warning
    if code < 300:
        return "INFEAS"
    if code < 400:
        return "UNBND"
    if code < 500:
        return "LIMIT"
    return "FAIL"


def _solve(exe, tmp_path, name, builder, k) -> int | None:
    m = pyo.ConcreteModel()
    builder(m, 10.0**k)
    nl = tmp_path / f"{name}_{k}.nl"
    sol = tmp_path / f"{name}_{k}.sol"
    m.write(str(nl), io_options={"symbolic_solver_labels": True})
    subprocess.run(
        [exe, str(nl), "-AMPL", "--sol-output", str(sol), *_OPTS],
        capture_output=True,
        timeout=120,
    )
    if not sol.exists():
        return None
    for line in sol.read_text().splitlines():
        if line.startswith("objno"):
            return int(line.split()[2])
    return None


def test_scale_invariance_does_not_regress(tmp_path):
    exe = _pounce()
    executed = 0
    report = []
    regressions = []
    improvements = []

    for name, builder, want in MODELS:
        curve = []
        for k in KS:
            code = _solve(exe, tmp_path, name, builder, k)
            if code is not None:
                executed += 1
            curve.append(_band(code))
        wrong = sum(1 for b in curve if b != want)
        base = BASELINE_WRONG[name]
        report.append(f"  {name:<12} want={want:<7} wrong={wrong:>2}/{len(KS)} "
                      f"(baseline {base})  {' '.join(curve)}")
        if wrong > base:
            regressions.append((name, base, wrong, curve))
        elif wrong < base:
            improvements.append((name, base, wrong))

    detail = "\n".join(report)

    # A probe that silently exercised nothing is worse than a failing one.
    assert executed >= len(MODELS) * len(KS) // 2, (
        f"only {executed} solves completed; the harness or the CLI is failing, "
        f"so this run proves nothing\n{detail}"
    )

    assert not regressions, (
        "scale invariance regressed — a model's verdict is wrong at more row "
        "scalings than before. Multiplying a row by a positive constant cannot "
        "change the feasible set, so every one of these is the same problem "
        "answered two different ways:\n"
        + "\n".join(
            f"  {n}: {b} -> {w} wrong cells\n    {' '.join(c)}"
            for n, b, w, c in regressions
        )
        + f"\n\nfull report:\n{detail}"
    )

    if improvements:
        print("\nscale invariance improved (lower BASELINE_WRONG accordingly):")
        for n, b, w in improvements:
            print(f"  {n}: {b} -> {w}")
    print(f"\nscale-invariance report ({executed} solves):\n{detail}")

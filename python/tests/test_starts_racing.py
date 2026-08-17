"""Adaptive successive-halving racing (pounce#610).

The pre-#610 ``race_starts`` spent a fixed budget on every candidate from
a cold start, ranked once on terminal violation/objective, and reported
nothing. These tests cover the replacement policy against the issue's own
acceptance criteria — pause/resume, determinism, per-round reporting,
cost at equal quality, and the adversarial case where early objective
progress points at the wrong basin — plus the requirement that the old
policy stay selectable and reproduce its old answers exactly.

``python/tests/test_starts.py`` keeps the original coverage of
``generate_starts`` / ``project_to_feasible`` / the basic race.
"""

import numpy as np
import pytest

import pounce


def _halving(*args, **kwargs):
    """``race_starts`` with the ladder selected.

    ``policy="fixed"`` is the default (see the quality caveat in
    ``race_starts``' docstring), so every test below that is *about* the
    ladder has to opt in. Routing them through one wrapper keeps that
    fact in one place; ``test_the_default_policy_is_the_fixed_baseline``
    is what pins the default itself.
    """
    kwargs.setdefault("policy", "halving")
    return pounce.race_starts(*args, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _double_well():
    """Two basins, minima near x = ±1, the left one deeper."""

    def f(x):
        return float((x[0] ** 2 - 1.0) ** 2 + 0.25 * (x[0] + 1.0))

    def g(x):
        return np.array([4.0 * x[0] * (x[0] ** 2 - 1.0) + 0.25])

    return f, g, [(-3.0, 3.0)], None


def _hs71():
    """Hock-Schittkowski 71 — active bounds and a nonlinear equality, so a
    truncated solve leaves a genuinely mid-flight interior-point state."""

    def f(x):
        return float(x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2])

    def g(x):
        return np.array([
            x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]),
            x[0] * x[3],
            x[0] * x[3] + 1.0,
            x[0] * (x[0] + x[1] + x[2]),
        ])

    cons = [
        dict(type="ineq",
             fun=lambda x: np.array([x[0] * x[1] * x[2] * x[3] - 25.0]),
             jac=lambda x: np.array([[x[1] * x[2] * x[3], x[0] * x[2] * x[3],
                                      x[0] * x[1] * x[3], x[0] * x[1] * x[2]]])),
        dict(type="eq",
             fun=lambda x: np.array([sum(xi ** 2 for xi in x) - 40.0]),
             jac=lambda x: np.array([[2.0 * x[0], 2.0 * x[1], 2.0 * x[2],
                                      2.0 * x[3]]])),
    ]
    return f, g, [(1.0, 5.0)] * 4, cons


def _rastrigin_eq():
    """Many basins on a line — the case where one iteration tells you
    nothing and a race has to buy a few before it ranks."""

    def f(x):
        return float(20.0 + sum(xi ** 2 - 10.0 * np.cos(2.0 * np.pi * xi)
                                for xi in x))

    def g(x):
        return np.array([2.0 * xi + 20.0 * np.pi * np.sin(2.0 * np.pi * xi)
                         for xi in x])

    cons = [dict(type="eq",
                 fun=lambda x: np.array([x[0] + x[1] - 1.0]),
                 jac=lambda x: np.array([[1.0, 1.0]]))]
    return f, g, [(-4.0, 4.0), (-4.0, 4.0)], cons


def _deceptive_circle():
    """The adversarial fixture pounce#610's last acceptance item asks for.

    ``f = x³ − 3xy² + 0.3y`` on the circle ``x² + y² = 4`` has three
    feasible basins (−7.481, −8.003, −8.520). Off the circle ``f`` is
    unbounded below toward ``x → −∞``, so a start out at ``x = −2.9``
    opens with an objective of −24.4 — three times better than anything
    feasible — and is nonetheless doomed: pulled back onto the circle it
    lands in the −8.003 basin. The only start that reaches the −8.520
    global optimum opens at −8.39, *worse* than the decoys, and is
    distinguished solely by having almost no violation left to remove and
    a small KKT residual.
    """

    def f(x):
        return float(x[0] ** 3 - 3.0 * x[0] * x[1] ** 2 + 0.3 * x[1])

    def g(x):
        return np.array([3.0 * x[0] ** 2 - 3.0 * x[1] ** 2,
                         -6.0 * x[0] * x[1] + 0.3])

    cons = [dict(type="eq",
                 fun=lambda x: np.array([x[0] ** 2 + x[1] ** 2 - 4.0]),
                 jac=lambda x: np.array([[2.0 * x[0], 2.0 * x[1]]]))]
    starts = np.array([
        [-2.9, 0.0],    # 0: f = -24.4, violation 4.4  -- the decoy
        [-2.6, 0.4],    # 1: f = -16.2, violation 2.9  -- the decoy
        [-2.6, -0.4],   # 2: f = -16.4, violation 2.9  -- the decoy
        [-1.0, 1.75],   # 3
        [-1.05, 1.7],   # 4
        [1.0, -1.72],   # 5: f = -8.39, violation 0.04 -- the winner
        [0.2, 2.0],     # 6
        [-2.0, 0.1],    # 7
        [2.0, 0.3],     # 8
    ])
    return f, g, [(-3.0, 3.0), (-3.0, 3.0)], cons, starts


#: The global optimum of ``_deceptive_circle``, from solving every start
#: to convergence; only start 5 reaches it.
_DECEPTIVE_BEST = -8.520236
#: The basin the decoys land in.
_DECEPTIVE_DECOY = -8.002500


def _finish(best, f, g, bounds, cons):
    """The workflow the docs prescribe: continue the winner warm."""
    ws = pounce.WarmStart.from_info(best.x, best.info)
    return pounce.minimize(f, best.x, jac=g, bounds=bounds, constraints=cons,
                           warm_start=ws)


class _Counter:
    """Counts every call into the user callables, for every candidate —
    including the ones the racer discards, which is the number the
    pre-#610 function could not report."""

    def __init__(self):
        self.n = 0

    def wrap(self, fn):
        if fn is None:
            return None

        def inner(x):
            self.n += 1
            return fn(x)

        return inner

    def wrap_cons(self, cons):
        if not cons:
            return cons
        out = []
        for c in cons:
            d = dict(c)
            d["fun"] = self.wrap(d["fun"])
            if d.get("jac") is not None:
                d["jac"] = self.wrap(d["jac"])
            out.append(d)
        return out


# ---------------------------------------------------------------------------
# Acceptance criterion 1 — pause / resume, not cold restart
# ---------------------------------------------------------------------------


def test_resuming_a_paused_candidate_beats_restarting_it():
    """A resume must be measurably cheaper than a restart from the same
    point, or the ladder is a cold-restart loop wearing a warm coat.

    Both arms below start the second leg at *the same iterate*. The only
    difference is what else travels with it: the resume carries the
    multipliers and the barrier parameter μ the truncated solve had
    reached, the restart carries the point alone and re-walks the barrier
    from ``mu_init``'s default.

    The fixture is chosen to have headroom. pounce#608 measured that a
    warm-started IPM in a continuation regime often converges in one
    iteration per step, which would make "resumed cheaply" and "restarted
    cheaply" indistinguishable; Rastrigin-on-a-line needs tens of
    iterations from these starts, so the two are far apart.
    """
    f, g, bounds, cons = _rastrigin_eq()
    starts = np.array([[0.5, 0.5], [1.9, -0.9], [-0.9, 1.9], [2.9, -1.9],
                       [-1.9, 2.9], [3.8, -2.8], [0.1, 0.9], [1.1, -0.1]])
    kw = dict(jac=g, bounds=bounds, constraints=cons)

    resumed_iters = restarted_iters = 0
    for s in starts:
        paused = pounce.minimize(f, s, **kw, max_iter=5)
        ws = pounce.WarmStart.from_info(paused.x, paused.info)
        # Resume: same point, plus lagrange / z_L / z_U / mu.
        resumed_iters += pounce.minimize(f, paused.x, **kw, warm_start=ws).nit
        # Restart: same point, nothing else.
        restarted_iters += pounce.minimize(f, paused.x, **kw).nit

    # Measured at the time of writing: 17 resumed vs 43 restarted. Assert
    # a wide margin so this pins the *effect*, not the exact numbers.
    assert resumed_iters < 0.75 * restarted_iters, (
        f"resume={resumed_iters} restart={restarted_iters}: resuming a paused "
        "candidate is not measurably cheaper than restarting it, so the "
        "ladder's 'resume' carries nothing"
    )


def test_survivors_are_resumed_rather_than_restarted():
    """Every rung past the first must resume its entrants from held state."""
    f, g, bounds, cons = _hs71()
    starts = pounce.generate_starts(9, bounds=bounds, seed=0)
    _best, rep = _halving(f, starts, jac=g, bounds=bounds,
                                    constraints=cons, iters=12,
                                    return_report=True)
    assert rep.n_rounds >= 2, "the ladder collapsed to a single rung"
    assert rep.rounds[0].started == len(starts)
    assert rep.rounds[0].resumed == 0
    for rnd in rep.rounds[1:]:
        assert rnd.started == 0, f"rung {rnd.index} cold-started a candidate"
        assert rnd.resumed > 0, f"rung {rnd.index} resumed nothing"
    assert rep.n_resumes > 0


# ---------------------------------------------------------------------------
# Acceptance criterion 2 — determinism
# ---------------------------------------------------------------------------


def _record(rep):
    """Everything the race did, as comparable plain data."""
    return {
        "policy": rep.policy,
        "rounds": [
            (r.index, r.eval_budget, r.iter_budget, tuple(r.entrants),
             tuple(r.survivors), tuple(r.eliminated), r.evals, r.iters,
             r.resumed, r.started, tuple(sorted(r.scores.items())))
            for r in rep.rounds
        ],
        "candidates": [
            (c.index, c.status, c.obj, c.violation, c.kkt, c.evals, c.iters,
             c.nfev, c.njev, c.nhev, c.resumes, c.restoration_calls,
             c.eliminated_round, c.reason, tuple(np.asarray(c.x).ravel()))
            for c in rep.candidates
        ],
    }


def test_the_whole_race_is_reproducible_round_for_round():
    """Not just the winner: survivors, eliminations, scores and resource
    spend must all repeat, which is what makes a reported race auditable.
    """
    f, g, bounds, cons = _hs71()
    starts = pounce.generate_starts(12, bounds=bounds, seed=20250610)

    def race():
        best, rep = _halving(f, starts, jac=g, bounds=bounds,
                                       constraints=cons, iters=15, top=3,
                                       return_report=True)
        return [np.asarray(b.x) for b in best], _record(rep)

    x_a, rec_a = race()
    x_b, rec_b = race()
    assert rec_a == rec_b
    for a, b in zip(x_a, x_b):
        assert np.array_equal(a, b)


def test_determinism_survives_a_different_call_order():
    """A race must not depend on solves that ran before it — the option
    overlay a resume installs is scoped, so a previous race cannot leak
    ``mu_init`` into the next one (pounce#607)."""
    f, g, bounds, cons = _hs71()
    starts = pounce.generate_starts(9, bounds=bounds, seed=7)
    kw = dict(jac=g, bounds=bounds, constraints=cons, iters=12,
              return_report=True)

    _, first = _halving(f, starts, **kw)
    # Run an unrelated race in between; the second run of the first race
    # must still match.
    _halving(f, pounce.generate_starts(6, bounds=bounds, seed=99),
                       **kw)
    _, again = _halving(f, starts, **kw)
    assert _record(first) == _record(again)


# ---------------------------------------------------------------------------
# Acceptance criterion 3 — per-round resource use and elimination reason
# ---------------------------------------------------------------------------


def test_every_candidate_has_a_reason_and_every_rung_has_a_cost():
    f, g, bounds, cons = _hs71()
    starts = pounce.generate_starts(12, bounds=bounds, seed=3)
    best, rep = _halving(f, starts, jac=g, bounds=bounds,
                                   constraints=cons, iters=15,
                                   return_report=True)
    assert len(rep.candidates) == len(starts)
    for c in rep.candidates:
        assert c.reason, f"candidate {c.index} left the race unexplained"
        assert c.evals > 0 and c.iters >= 0
    eliminated = {i for r in rep.rounds for i, _ in r.eliminated}
    final = set(rep.rounds[-1].survivors)
    assert eliminated.isdisjoint(final)
    # Every start is accounted for exactly once: eliminated, or standing.
    assert eliminated | final == set(range(len(starts)))
    for r in rep.rounds:
        assert r.eval_budget > 0
        assert r.iter_budget > 0
        assert r.evals >= 0 and r.iters >= 0
        for _idx, why in r.eliminated:
            assert why.strip()
    assert rep.total_evals == sum(c.evals for c in rep.candidates)
    text = rep.report()
    assert "rung 0" in text and "resumed" in text
    assert best


def test_the_fixed_policy_also_reports_what_it_spent():
    """The baseline could not report its own cost — it threw every
    candidate outside ``top`` away. Through ``RaceReport`` it can, which
    is what makes the two policies comparable on one number."""
    f, g, bounds, cons = _double_well()
    starts = np.array([[1.4], [-1.4], [0.2], [2.6]])
    best, rep = pounce.race_starts(f, starts, jac=g, bounds=bounds,
                                   constraints=cons, iters=8, policy="fixed",
                                   return_report=True)
    assert rep.policy == "fixed"
    assert rep.n_rounds == 1
    assert rep.n_resumes == 0
    assert len(rep.candidates) == 4
    assert rep.total_evals > 0
    assert len(rep.rounds[0].survivors) == 1
    assert len(rep.rounds[0].eliminated) == 3
    assert best[0].fun == pytest.approx(rep.candidates[
        rep.rounds[0].survivors[0]].obj)


# ---------------------------------------------------------------------------
# Acceptance criterion 4 — same answer, materially fewer evaluations
# ---------------------------------------------------------------------------


#: The benchmark set criterion 4 is judged on: ``(builder, n_starts,
#: iters)``. Each is multi-basin, so *which* start the full solve
#: continues from decides the answer.
_SUITE = [
    (_hs71, 16, 20),
    (_rastrigin_eq, 16, 20),
    (_double_well, 12, 15),
]


def _race_and_finish(case, n_starts, iters, policy):
    f, g, bounds, cons = case()
    starts = pounce.generate_starts(n_starts, bounds=bounds, seed=0)
    c = _Counter()
    best, rep = pounce.race_starts(
        c.wrap(f), starts, jac=c.wrap(g), bounds=bounds,
        constraints=c.wrap_cons(cons), iters=iters, policy=policy,
        return_report=True,
    )
    fin = _finish(best[0], c.wrap(f), c.wrap(g), bounds, c.wrap_cons(cons))
    return {
        "obj": float(fin.fun),
        "viol": float(fin.info.get("final_constr_viol", 0.0)),
        "user_evals": c.n,
        "solver_evals": rep.total_evals,
        "iters": rep.total_iters,
    }


@pytest.mark.parametrize("case, n_starts, iters", _SUITE)
def test_halving_matches_the_fixed_answer_on_every_problem(
    case, n_starts, iters
):
    """Quality is a per-problem obligation: the ladder may be cheaper on
    average, but it may not be *worse* anywhere."""
    fixed = _race_and_finish(case, n_starts, iters, "fixed")
    halving = _race_and_finish(case, n_starts, iters, "halving")
    assert halving["viol"] <= max(1e-6, 10 * fixed["viol"])
    assert halving["obj"] <= fixed["obj"] + 1e-6 * max(1.0, abs(fixed["obj"])), (
        f"halving reached {halving['obj']} where fixed reached {fixed['obj']}"
    )


def test_the_ladder_beats_the_pre_610_cost():
    """The behavioural bite against the parent commit.

    Apart from ``policy="halving"`` every call below uses the pre-#610
    signature, so this measures the ladder against what the fixed budget
    spends: 3077 user-callable evaluations across this suite to reach
    these three answers. The threshold is set at 2800 so the test is
    about the change and not about the last few evaluations either way.
    """
    total, objs = 0, {}
    for case, n_starts, iters in _SUITE:
        f, g, bounds, cons = case()
        starts = pounce.generate_starts(n_starts, bounds=bounds, seed=0)
        c = _Counter()
        best = _halving(c.wrap(f), starts, jac=c.wrap(g),
                                  bounds=bounds,
                                  constraints=c.wrap_cons(cons), iters=iters)
        fin = _finish(best[0], c.wrap(f), c.wrap(g), bounds,
                      c.wrap_cons(cons))
        total += c.n
        objs[case.__name__] = float(fin.fun)

    # Quality first: these are the answers the fixed budget reaches, and
    # a cheaper race that misses them is not an improvement.
    assert objs["_hs71"] == pytest.approx(17.014017, abs=1e-4)
    assert objs["_rastrigin_eq"] == pytest.approx(0.99747969, abs=1e-4)
    assert objs["_double_well"] == pytest.approx(-0.0037912372, abs=1e-6)
    assert total < 2800, (
        f"the halving policy spent {total} evaluations; the pre-#610 "
        "fixed budget spent 3077 for the same three answers"
    )


def test_halving_costs_materially_less_across_the_suite():
    """Cost is a suite-level claim, which is how pounce#610 words it —
    and it has to be, because it is not true everywhere.

    On ``_double_well`` — one variable, no constraints, a handful of
    evaluations per iteration — a rung boundary costs a fresh solver
    application and a re-evaluation at the seed, and that fixed cost is a
    large fraction of the whole solve. Iterations still fall (113 -> 84
    at the time of writing) but user-callable evaluations come out level
    or slightly up. The ladder pays for itself when an iteration is
    expensive, which is the case the issue is about; on a model this
    cheap, ``policy="fixed"`` is the better choice and remains available.
    """
    totals = {p: {"user_evals": 0, "solver_evals": 0, "iters": 0}
              for p in ("fixed", "halving")}
    for policy in totals:
        for case, n_starts, iters in _SUITE:
            row = _race_and_finish(case, n_starts, iters, policy)
            for k in totals[policy]:
                totals[policy][k] += row[k]

    for metric, margin in (("solver_evals", 0.10), ("iters", 0.15),
                           ("user_evals", 0.05)):
        fixed = totals["fixed"][metric]
        halving = totals["halving"][metric]
        assert halving <= (1.0 - margin) * fixed, (
            f"{metric}: halving {halving} vs fixed {fixed} — expected at "
            f"least {margin:.0%} lower across the suite"
        )


# ---------------------------------------------------------------------------
# Acceptance criterion 5 — misleading objective, decisive feasibility/KKT
# ---------------------------------------------------------------------------


def test_early_objective_progress_misleads_but_the_race_still_wins():
    f, g, bounds, cons, starts = _deceptive_circle()

    # Sanity: the decoys really do look best on objective alone, and the
    # eventual winner really does not.
    assert f(starts[0]) < f(starts[5]) - 10.0

    best, rep = _halving(f, starts, jac=g, bounds=bounds,
                                   constraints=cons, iters=9, explore=0,
                                   return_report=True)
    fin = _finish(best[0], f, g, bounds, cons)
    assert fin.fun == pytest.approx(_DECEPTIVE_BEST, abs=1e-4)
    assert 5 in rep.rounds[-1].survivors, (
        "the composite ranking eliminated the only start that reaches the "
        "global optimum"
    )


def test_ranking_on_objective_alone_eliminates_the_eventual_winner():
    """The control arm: strip the feasibility, KKT and health terms out
    of the score and the race follows the decoys into the wrong basin.

    This is what makes the test above a statement about the *ranking*
    rather than about the fixture being easy.
    """
    f, g, bounds, cons, starts = _deceptive_circle()
    objective_only = {"violation": 0.0, "feasibility_progress": 0.0,
                      "kkt": 0.0, "objective_progress": 1.0, "health": 0.0}
    best, rep = _halving(f, starts, jac=g, bounds=bounds,
                                   constraints=cons, iters=9, explore=0,
                                   weights=objective_only, return_report=True)
    fin = _finish(best[0], f, g, bounds, cons)
    assert 5 not in rep.rounds[-1].survivors
    assert fin.fun == pytest.approx(_DECEPTIVE_DECOY, abs=1e-4)
    assert fin.fun > _DECEPTIVE_BEST + 0.4


def test_restoration_and_health_reach_the_ranking():
    """The health term reads counters that only exist because pounce#610
    surfaced them; a ``KeyError``-free zero is not the same as a signal."""
    f, g, bounds, cons = _hs71()
    res = pounce.minimize(f, np.array([1.0, 5.0, 5.0, 1.0]), jac=g,
                          bounds=bounds, constraints=cons)
    for key in ("restoration_calls", "restoration_outer_iters",
                "restoration_inner_iters", "n_obj_evals", "n_grad_evals",
                "n_constr_evals", "n_jac_evals", "n_hess_evals"):
        assert key in res.info, f"info is missing {key!r}"
    assert res.info["n_obj_evals"] > 0
    assert res.info["n_constr_evals"] > 0


# ---------------------------------------------------------------------------
# Scope item 5 — diversity: clustering and the exploration quota
# ---------------------------------------------------------------------------


def test_near_identical_survivors_are_collapsed():
    """Two starts in the same basin are one candidate wearing two hats;
    letting both hold survivor slots spends the next rung twice on the
    same answer."""
    f, g, bounds, cons = _double_well()
    # Four starts, in two tight pairs.
    starts = np.array([[1.40], [1.4000001], [-1.40], [-1.4000001]])
    _best, rep = _halving(f, starts, jac=g, bounds=bounds,
                                    constraints=cons, iters=9, top=2,
                                    explore=0, cluster_tol=1e-2,
                                    return_report=True)
    reasons = [c.reason for c in rep.candidates]
    assert any("duplicate of candidate" in r for r in reasons), reasons


def test_dedup_never_returns_fewer_results_than_asked_for():
    """Collapsing twins is a survivor-slot policy, not a licence to hand
    back a shorter list than `top`. Every start below falls into the same
    basin, so dedup would otherwise leave one candidate standing and
    `top=3` would quietly return one result."""
    def f(x):
        return float((x[0] - 1.0) ** 2 + (x[1] + 2.0) ** 2)

    def g(x):
        return np.array([2.0 * (x[0] - 1.0), 2.0 * (x[1] + 2.0)])

    starts = np.array([[5.0, 5.0], [-5.0, -5.0], [0.0, 0.0], [9.0, -9.0]])
    for top in (1, 2, 3):
        best = _halving(f, starts, jac=g, iters=10, top=top)
        assert len(best) == top, f"top={top} returned {len(best)} results"


def test_the_exploration_quota_keeps_an_outsider():
    f, g, bounds, cons = _hs71()
    starts = pounce.generate_starts(12, bounds=bounds, seed=11)
    _b, with_quota = _halving(f, starts, jac=g, bounds=bounds,
                                        constraints=cons, iters=15, explore=2,
                                        return_report=True)
    _b, without = _halving(f, starts, jac=g, bounds=bounds,
                                     constraints=cons, iters=15, explore=0,
                                     return_report=True)
    kept = [c.index for c in with_quota.candidates
            if c.reason == "retained: exploration quota"]
    assert kept, "the exploration quota retained nobody"
    assert len(with_quota.rounds[0].survivors) > \
        len(without.rounds[0].survivors)


# ---------------------------------------------------------------------------
# Scope item 6 — evaluations, not just iterations, are the resource
# ---------------------------------------------------------------------------


def test_the_evaluation_budget_can_bind_before_the_iteration_ceiling():
    """Pin the ladder's evaluation unit low and candidates stop short of
    the iteration ceiling — which is the whole claim: two candidates
    whose iterations cost differently are not charged the same."""
    f, g, bounds, cons = _hs71()
    starts = pounce.generate_starts(9, bounds=bounds, seed=5)
    kw = dict(jac=g, bounds=bounds, constraints=cons, iters=30,
              return_report=True)
    _b, tight = _halving(f, starts, eval_budget=8, **kw)
    _b, loose = _halving(f, starts, eval_budget=400, **kw)
    assert tight.total_evals < loose.total_evals
    assert tight.total_iters < loose.total_iters
    # Under the tight unit, the last rung's entrants are held below the
    # iteration ceiling by the evaluation budget alone.
    last = tight.rounds[-1]
    held = [c for c in tight.candidates
            if c.index in last.entrants and c.iters < last.iter_budget
            and c.status not in (0, 1)]
    assert held, "no candidate was held back by the evaluation budget"


def test_evaluation_counts_are_the_solvers_own():
    f, g, bounds, cons = _double_well()
    starts = np.array([[1.4], [-1.4], [0.2], [2.6]])
    _b, rep = _halving(f, starts, jac=g, bounds=bounds,
                                 constraints=cons, iters=9,
                                 return_report=True)
    for c in rep.candidates:
        # The solver's callback tallies and the Python wrapper's counts
        # are different measurements of the same solve; both must be
        # populated, and neither may be zero for a candidate that ran.
        assert c.evals > 0
        assert c.nfev > 0


# ---------------------------------------------------------------------------
# "Keep the current fixed-budget policy as a reproducible baseline"
# ---------------------------------------------------------------------------


def _pre_610_race_starts(fun, starts, *, jac=None, bounds=None,
                         constraints=None, iters=10, top=1, options=None):
    """``pounce.race_starts`` as of 0.10.0, transcribed verbatim.

    Not a call into the shipped code: the point of this test is that the
    shipped ``policy="fixed"`` still does what *this* did, so it has to
    be an independent copy or it proves nothing.
    """
    from pounce._minimize import minimize

    opts = dict(options or {})
    opts["max_iter"] = int(iters)
    results = []
    for s in np.atleast_2d(np.asarray(starts, dtype=float)):
        res = minimize(fun, s, jac=jac, bounds=bounds,
                       constraints=constraints, **opts)
        viol = float(res.info.get("final_constr_viol", 0.0))
        if not np.isfinite(viol):
            viol = np.inf
        obj = res.fun if np.isfinite(res.fun) else np.inf
        results.append((max(viol - 1e-6, 0.0), obj, res))
    results.sort(key=lambda t: (t[0], t[1]))
    return [r for _, _, r in results[: max(1, int(top))]]


@pytest.mark.parametrize("case", [_double_well, _hs71, _rastrigin_eq])
@pytest.mark.parametrize("iters, top", [(4, 1), (10, 3)])
def test_fixed_policy_reproduces_the_pre_610_baseline(case, iters, top):
    f, g, bounds, cons = case()
    starts = pounce.generate_starts(8, bounds=bounds, seed=1)
    kw = dict(jac=g, bounds=bounds, constraints=cons, iters=iters, top=top)

    old = _pre_610_race_starts(f, starts, **kw)
    new = pounce.race_starts(f, starts, policy="fixed", **kw)

    assert len(old) == len(new)
    for a, b in zip(old, new):
        assert np.array_equal(np.asarray(a.x), np.asarray(b.x))
        assert a.fun == b.fun
        assert a.nit == b.nit
        assert a.nfev == b.nfev and a.njev == b.njev and a.nhev == b.nhev
        assert a.status == b.status
        assert a.info["final_constr_viol"] == b.info["final_constr_viol"]


def test_the_default_policy_is_the_fixed_baseline():
    """The ladder is opt-in, and this is where that is decided.

    ``policy="halving"`` is cheaper on average and worse on some
    multimodal problems (see the test below), so it may not become an
    existing caller's policy by their doing nothing. A default call has
    to be the pre-#610 policy, byte for byte.
    """
    import inspect

    assert (inspect.signature(pounce.race_starts)
            .parameters["policy"].default == "fixed")

    f, g, bounds, cons = _hs71()
    starts = pounce.generate_starts(8, bounds=bounds, seed=1)
    kw = dict(jac=g, bounds=bounds, constraints=cons, iters=10, top=3)

    old = _pre_610_race_starts(f, starts, **kw)
    default = pounce.race_starts(f, starts, **kw)
    assert len(old) == len(default)
    for a, b in zip(old, default):
        assert np.array_equal(np.asarray(a.x), np.asarray(b.x))
        assert a.fun == b.fun and a.nit == b.nit and a.status == b.status

    _b, rep = pounce.race_starts(f, starts, return_report=True, **kw)
    assert rep.policy == "fixed"


def test_the_ladder_can_cut_the_winner_at_rung_zero():
    """Why ``"halving"`` is not the default, pinned as a measurement.

    2-D Ackley: hundreds of local minima on a near-flat plate with the
    global one in a narrow funnel, so where a start *ends* is close to
    uncorrelated with how it looks after four iterations. The ladder
    ranks on exactly that and discards two thirds of the field on it.

    The assertion is not "the ladder is bad" — it is that on this class
    of model the rung-0 ranking carries no signal about the eventual
    winner, which is the fact the docstring's recommendation rests on.
    Were that to stop being true, this test should fail and the default
    should be revisited, not the test relaxed.
    """
    def f(x):
        return float(
            -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / 2.0))
            - np.exp(np.sum(np.cos(2.0 * np.pi * x)) / 2.0)
            + 20.0 + np.e
        )

    bounds = [(-5.0, 5.0)] * 2
    starts = pounce.generate_starts(27, bounds=bounds, seed=0)

    # Ground truth: run every start to convergence and see which wins.
    finals = [float(pounce.minimize(f, s, bounds=bounds).fun) for s in starts]
    winner = int(np.argmin(finals))
    assert finals[winner] < 1e-8, "no start reached the global minimum"

    _best, rep = _halving(f, starts, bounds=bounds, iters=40, top=1,
                          return_report=True)
    cut = rep.candidates[winner]
    assert cut.eliminated_round == 0, (
        f"candidate {winner} reaches {finals[winner]:.3g} at full effort but "
        f"the ladder kept it past rung 0 ({cut.reason!r}) — if the rung-0 "
        "ranking has become informative here, revisit the default policy"
    )
    assert "below halving cut" in cut.reason

    # And the default policy does not have this failure mode: it spends
    # the same budget on every start, so the winner is still in the field.
    best_fixed = pounce.race_starts(f, starts, bounds=bounds, iters=40, top=1)
    assert float(pounce.minimize(f, best_fixed[0].x, bounds=bounds).fun) < 1e-8


def test_fixed_policy_refuses_the_post_610_arguments():
    """``hess=`` / ``args=`` are #610 additions. Accepting them on the
    frozen policy and quietly ignoring them would make "reproducible
    baseline" false in the one situation where it matters."""
    f, g, bounds, cons = _double_well()
    with pytest.raises(TypeError, match="frozen baseline"):
        pounce.race_starts(f, np.array([[1.4], [-1.4]]), jac=g,
                           bounds=bounds, constraints=cons, policy="fixed",
                           hess=lambda x: np.array([[1.0]]))


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


def test_unknown_policy_is_rejected():
    f, g, bounds, _ = _double_well()
    with pytest.raises(ValueError, match="unknown policy"):
        pounce.race_starts(f, np.array([[1.0]]), jac=g, bounds=bounds,
                           policy="hyperband")


def test_halving_refuses_convex_routing_instead_of_losing_the_session():
    f, g, bounds, _ = _double_well()
    with pytest.raises(ValueError, match="NLP path"):
        _halving(f, np.array([[1.4], [-1.4]]), jac=g, bounds=bounds,
                           options={"solver_selection": "auto"})


def test_a_single_start_still_works():
    f, g, bounds, _ = _double_well()
    best, rep = _halving(f, np.array([[1.4]]), jac=g, bounds=bounds,
                                   iters=8, return_report=True)
    assert len(best) == 1
    assert rep.n_rounds >= 1
    assert rep.candidates[0].reason

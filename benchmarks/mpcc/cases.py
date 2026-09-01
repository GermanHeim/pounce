"""The benchmark ladder.

Every case is small enough that its global solution can be derived by
hand, and every derivation is written out in the factory's docstring so
a reviewer can check the asserted number rather than trust it. Nothing
is taken on trust twice: `oracle.enumerate_branches` recomputes the same
global solution by solving each complementarity branch as a smooth
program with SciPy, and `selftest` fails when the two disagree.

Two conventions hold throughout and both matter when reading a result:

* **Nonnegativity of a pair side lives in the pair, never in a variable
  bound.** ``lb`` is ``-inf`` for a variable whose only lower bound
  would be the ``G_i(x) >= 0`` it already appears in. Stating it twice
  would make MPCC-LICQ fail at every solution by construction, and the
  ladder needs cases where it holds and cases where it does not.

* **A case's ``Expected`` describes the source MPCC, not any lowering.**
  The relaxed problem at ``tau > 0`` is a different problem with a
  different optimum -- ``ralph2``'s is ``-2*tau``, below the MPCC's own
  ``0`` -- and the runner compares against the source.

Class coverage required by gh#794 (enforced in `selftest`): regular,
biactive, degenerate, infeasible, selector, and a pinned MacMPEC-style
subset.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from .spec import Affine, Expected, MpccCase, Pair, Quad, Row

INF = np.inf


def _pair(name, G, H, gz="", hz=""):
    return Pair(name, G, H, gz, hz)


# --------------------------------------------------------------------
# regular
# --------------------------------------------------------------------


def regular_strict() -> MpccCase:
    """Strict complementarity, unique solution, MPCC-LICQ holds.

    ``min (x1-1)^2 + (x2-2)^2  s.t.  0 <= x1 _|_ x2 >= 0``

    The feasible set is the two nonnegative axes. On ``x2 = 0`` the
    objective is ``(x1-1)^2 + 4``, minimised at ``x1 = 1`` with value
    ``4``; on ``x1 = 0`` it is ``1 + (x2-2)^2``, minimised at ``x2 = 2``
    with value ``1``. So the global solution is ``(0, 2)``, ``f* = 1``,
    and it is strict: ``G = 0``, ``H = 2 > 0``.

    Active gradients there are ``{grad G} = {(1,0)}`` -- rank 1 of one
    vector, so MPCC-LICQ holds. With no biactive pair the four
    stationarity classes coincide and the point is S-stationary
    (``nu = -2``, ``w = 0``).

    This is the case every route is expected to solve. A route that
    fails here has a defect that needs no MPCC to explain it.
    """
    P = np.diag([2.0, 2.0])
    f = Quad(P, np.array([-2.0, -4.0]), 5.0)
    pairs = (
        _pair(
            "p1",
            Affine([1.0, 0.0]),
            Affine([0.0, 1.0]),
            gz="x1 absent",
            hz="x2 absent",
        ),
    )
    return MpccCase(
        name="regular_strict",
        klass="regular",
        objective=f,
        rows=(),
        pairs=pairs,
        lb=np.array([-INF, -INF]),
        ub=np.array([INF, INF]),
        starts={
            "origin": np.array([0.0, 0.0]),
            "wrong_branch": np.array([2.0, 0.0]),
            "interior": np.array([1.0, 1.0]),
        },
        expected=Expected(
            feasible=True,
            obj=1.0,
            x=np.array([0.0, 2.0]),
            stationarity="S",
            n_biactive=0,
            mpcc_licq=True,
            notes=(
                "Every route should reach f=1 from every start. The "
                "'wrong_branch' start sits on the other axis at the other "
                "branch's local minimum (f=4); a route that stops there has "
                "found a genuine local solution, which is reported, not "
                "scored as a failure -- POUNCE is a local solver."
            ),
        ),
        provenance="derived in-repo; the standard two-axis textbook MPCC.",
    )


# --------------------------------------------------------------------
# biactive
# --------------------------------------------------------------------


def biactive_positive() -> MpccCase:
    """Biactive solution with both MPCC multipliers strictly positive.

    ``min x1 + x2  s.t.  0 <= x1 _|_ x2 >= 0``

    Feasible set is the two nonnegative axes, on which ``f = x1 + x2 >=
    0``; the minimum is ``0`` at the origin, where both pair sides
    vanish -- one biactive pair.

    Stationarity with ``L = f - nu*G - w*H``: ``(1,1) - nu(1,0) -
    w(0,1) = 0`` gives ``nu = w = 1 > 0``, the unique multiplier. Both
    strictly positive, so the point is S-stationary and therefore also
    M-, C- and weakly stationary. MPCC-LICQ holds (``(1,0)``, ``(0,1)``
    are independent).

    The value of this case is that it is biactive *and* benign: it
    separates "the route cannot handle a biactive point" from "the route
    cannot handle a hard biactive point".
    """
    f = Quad(np.zeros((2, 2)), np.array([1.0, 1.0]), 0.0)
    pairs = (_pair("p1", Affine([1.0, 0.0]), Affine([0.0, 1.0])),)
    return MpccCase(
        name="biactive_positive",
        klass="biactive",
        objective=f,
        rows=(),
        pairs=pairs,
        lb=np.array([-INF, -INF]),
        ub=np.array([10.0, 10.0]),
        starts={
            "origin": np.array([0.0, 0.0]),
            "far": np.array([5.0, 5.0]),
            "axis": np.array([3.0, 0.0]),
        },
        expected=Expected(
            feasible=True,
            obj=0.0,
            x=np.array([0.0, 0.0]),
            stationarity="S",
            n_biactive=1,
            mpcc_licq=True,
            notes=(
                "The relaxed problem at tau > 0 has the same optimum (0 at "
                "the origin), so Scholtes continuation should be flat in tau "
                "here; a route whose stage count grows as tau shrinks is "
                "paying for the relaxation, not for the problem."
            ),
        ),
        provenance="derived in-repo.",
    )


# --------------------------------------------------------------------
# degenerate
# --------------------------------------------------------------------


def ralph1() -> MpccCase:
    """MPCC-LICQ fails; the solution is M-stationary but not S-stationary.

    ``min 2x - y  s.t.  x >= 0,  0 <= y _|_ (y - x) >= 0``

    Feasibility: ``y >= 0``, ``y - x >= 0`` and ``y(y-x) = 0``. Either
    ``y = 0``, which with ``y >= x >= 0`` forces ``x = 0``; or ``y = x``
    with ``x >= 0``. So the feasible set is the ray ``{(t,t) : t >= 0}``
    together with the origin, and ``f = 2t - t = t >= 0``. The global
    solution is the origin with ``f* = 0``, and both pair sides vanish
    there.

    Active gradients at the origin are ``grad G = (0,1)``, ``grad H =
    (-1,1)`` and the bound ``x >= 0`` -- three vectors in ``R^2``, so
    **MPCC-LICQ fails**.

    Stationarity, ``L = f - nu*G - w*H - z*x`` with ``z >= 0``:

        x:   2 + w - z = 0
        y:  -1 - nu - w = 0     =>  nu = -1 - w

    S-stationarity needs ``nu >= 0`` and ``w >= 0``, impossible since
    ``nu = -1 - w <= -1``. M-stationarity is attained at ``w = 0``,
    ``nu = -1`` (the product vanishes), with ``z = 2 >= 0``. So the
    classifier must return **M** here, not S -- and a classifier that
    reads a single least-squares multiplier vector instead of searching
    the multiplier *set* can easily return C and be wrong.

    This is the case that decides whether the classifier is worth
    anything.
    """
    f = Quad(np.zeros((2, 2)), np.array([2.0, -1.0]), 0.0)
    pairs = (
        _pair("p1", Affine([0.0, 1.0]), Affine([-1.0, 1.0])),
    )
    return MpccCase(
        name="ralph1",
        klass="degenerate",
        objective=f,
        rows=(),
        pairs=pairs,
        lb=np.array([0.0, -INF]),
        ub=np.array([INF, INF]),
        starts={
            "origin": np.array([0.0, 0.0]),
            "ray": np.array([1.0, 1.0]),
            "off_ray": np.array([1.0, 2.0]),
        },
        expected=Expected(
            feasible=True,
            obj=0.0,
            x=np.array([0.0, 0.0]),
            stationarity="M",
            n_biactive=1,
            mpcc_licq=False,
            notes=(
                "Known-hard: MPCC-LICQ fails and no S-stationary point "
                "exists, so a route that reports a converged NLP KKT point "
                "here is reporting on its own reformulation, not on the "
                "MPCC. Routes are expected to differ; the interesting "
                "record is which ones stop at the origin and what "
                "stationarity class they can certify."
            ),
        ),
        provenance=(
            "derived in-repo; the instance is the classical `ralph1` shape "
            "used throughout the MPEC literature (Ralph & Wright)."
        ),
    )


def ctrap() -> MpccCase:
    """A C-stationary point that is not a local minimiser.

    ``min x1^2 + x2^2 - x1 - x2  s.t.  0 <= x1 _|_ x2 >= 0``

    On ``x2 = 0`` the objective is ``x1^2 - x1``, minimised at ``x1 =
    1/2`` with value ``-1/4``; symmetrically on the other axis. So there
    are two global minimisers, ``(1/2, 0)`` and ``(0, 1/2)``, both with
    ``f* = -1/4`` and both strictly complementary.

    The origin is feasible and biactive. Its unique multiplier is
    ``nu = w = -1`` (``grad f(0) = (-1,-1)``), so ``nu*w = 1 > 0``: the
    origin is **C-stationary but neither M- nor S-stationary**, and it is
    not a local minimiser -- the objective strictly decreases along both
    axes leaving it.

    That is the whole point of the case. A route that stops at the origin
    has converged to a point where every ordinary NLP residual is small
    and the answer is 0.25 above the optimum, and only the MPCC
    stationarity class says so. Started from the origin, the exact
    product lowerings have nothing to move them off it; Scholtes
    continuation at ``tau > 0`` does, because the relaxed feasible set
    has interior near the origin.
    """
    f = Quad(np.diag([2.0, 2.0]), np.array([-1.0, -1.0]), 0.0)
    pairs = (_pair("p1", Affine([1.0, 0.0]), Affine([0.0, 1.0])),)
    return MpccCase(
        name="ctrap",
        klass="degenerate",
        objective=f,
        rows=(),
        pairs=pairs,
        lb=np.array([-INF, -INF]),
        ub=np.array([5.0, 5.0]),
        starts={
            "origin": np.array([0.0, 0.0]),
            "axis": np.array([1.0, 0.0]),
            "sym": np.array([0.5, 0.5]),
        },
        expected=Expected(
            feasible=True,
            obj=-0.25,
            x=None,  # two global minimisers, related by the x1<->x2 symmetry
            stationarity="S",
            n_biactive=0,
            mpcc_licq=True,
            notes=(
                "Two global minimisers. The reportable outcome is whether a "
                "route stopped at the origin (f=0, C-stationary, not a local "
                "minimiser) or at f=-0.25. `landed_at_trap` in the "
                "validation block says which."
            ),
        ),
        provenance="derived in-repo.",
        validators=(_ctrap_validator,),
    )


def _ctrap_validator(case: MpccCase, x: np.ndarray) -> dict:
    """Did this route stop at the C-stationary origin?

    Read off the **pair values**, not off ``x``. ``G`` and ``H`` are
    invariant under `MpccCase.rescale` while the coordinates are not, so
    a norm on ``x`` would answer a different question on each scaling
    leg -- the "absolute threshold on a scale-dependent quantity" trap.
    """
    g, h = case.pair_values(x)
    at_trap = bool(np.max(np.maximum(np.abs(g), np.abs(h))) <= 1e-4)
    return {
        "landed_at_trap": at_trap,
        "reached_global_ok": bool(abs(case.objective.value(x) + 0.25) <= 1e-6),
    }


# --------------------------------------------------------------------
# infeasible
# --------------------------------------------------------------------


def infeasible_pair() -> MpccCase:
    """No feasible point, with an analytically known relaxation crossover.

    ``min x1^2 + x2^2  s.t.  x1 + x2 = 1,  x1 - x2 = 0,
                             0 <= x1 _|_ x2 >= 0``

    The two equalities force ``x = (1/2, 1/2)``, where the
    complementarity product is ``1/4``. The MPCC is therefore
    **infeasible**, while the Scholtes relaxation ``x1 x2 <= tau`` is
    feasible exactly for ``tau >= 1/4`` and infeasible below it. The
    continuation is expected to accept its first stages and then fail at
    a stage crossing ``tau = 0.25``; the stage index at which it fails is
    a property of the schedule, and the harness records it.

    An MPCC benchmark without this case cannot distinguish "the route is
    robust" from "the route never says no".
    """
    f = Quad(np.diag([2.0, 2.0]), np.zeros(2), 0.0)
    rows = (
        Row("sum", Affine([1.0, 1.0]).as_quad(), 1.0, 1.0),
        Row("diff", Affine([1.0, -1.0]).as_quad(), 0.0, 0.0),
    )
    pairs = (_pair("p1", Affine([1.0, 0.0]), Affine([0.0, 1.0])),)
    return MpccCase(
        name="infeasible_pair",
        klass="infeasible",
        objective=f,
        rows=rows,
        pairs=pairs,
        lb=np.array([-INF, -INF]),
        ub=np.array([INF, INF]),
        starts={
            "origin": np.array([0.0, 0.0]),
            "midpoint": np.array([0.5, 0.5]),
        },
        expected=Expected(
            feasible=False,
            obj=None,
            x=None,
            stationarity=None,
            n_biactive=0,
            mpcc_licq=None,
            notes=(
                "Expected failure. A route that returns Solve_Succeeded here "
                "is wrong regardless of its residuals; a route that returns "
                "an infeasibility status, or a restoration failure, is "
                "correct. The l1 routes exist for exactly this shape and are "
                "expected to report infeasibility rather than diverge."
            ),
        ),
        provenance="derived in-repo.",
        # No case-level validator: the `infeasible` class validator in
        # `validate.py` already checks the only thing there is to check
        # here -- that the returned point is not source-feasible, because
        # no such point exists.
    )


# --------------------------------------------------------------------
# selector (symmetric Boolean selector / branch change)
# --------------------------------------------------------------------


def _selector(theta: float) -> MpccCase:
    """A one-hot Boolean selector driven by a parameter.

    ``min (1-theta) y1 + theta y2  s.t.  y1 + y2 = 1,
                                          0 <= y1 _|_ y2 >= 0``

    The equality plus complementarity force ``y in {(1,0), (0,1)}``, so
    the pair *is* a Boolean selector with no integer variable. The
    objective is ``1-theta`` on the first branch and ``theta`` on the
    second, so the optimal branch flips at ``theta = 1/2``, where the two
    are exactly tied and the MPCC has two global solutions.

    The relaxation is what makes this interesting: ``y1 y2 <= tau`` with
    ``y1 + y2 = 1`` admits the symmetric point ``(1/2, 1/2)`` whenever
    ``tau >= 1/4``, and the objective there is ``1/2`` for every theta.
    A continuation therefore starts on a fractional, branch-free point
    and has to commit to a branch as tau falls through ``1/4``. At
    ``theta = 1/2`` there is nothing to commit *to*: which branch comes
    out is decided by the start and by arithmetic, and that is the
    reportable quantity.
    """
    name = f"selector_theta_{int(round(theta * 100)):03d}"
    f = Quad(np.zeros((2, 2)), np.array([1.0 - theta, theta]), 0.0)
    rows = (Row("onehot", Affine([1.0, 1.0]).as_quad(), 1.0, 1.0),)
    pairs = (
        _pair(
            "sel",
            Affine([1.0, 0.0]),
            Affine([0.0, 1.0]),
            gz="branch B selected (y2 = 1)",
            hz="branch A selected (y1 = 1)",
        ),
    )
    tie = abs(theta - 0.5) < 1e-12
    best = min(theta, 1.0 - theta)
    if tie:
        xstar = None
    else:
        xstar = np.array([1.0, 0.0]) if theta > 0.5 else np.array([0.0, 1.0])
    return MpccCase(
        name=name,
        klass="selector",
        objective=f,
        rows=rows,
        pairs=pairs,
        lb=np.array([-INF, -INF]),
        # No explicit upper bound: `y1 + y2 = 1` with `y >= 0` from the
        # pair already implies `y <= 1`, and stating it again would be
        # active at every solution and make MPCC-LICQ fail there for a
        # reason that has nothing to do with the selector.
        ub=np.array([INF, INF]),
        starts={
            "fractional": np.array([0.5, 0.5]),
            "branch_A": np.array([1.0, 0.0]),
            "branch_B": np.array([0.0, 1.0]),
        },
        expected=Expected(
            feasible=True,
            obj=best,
            x=xstar,
            stationarity="S",
            n_biactive=0,
            mpcc_licq=True,
            notes=(
                "Tie: both branches are globally optimal and the branch "
                "returned is start- and arithmetic-dependent. Recorded, not "
                "scored."
                if tie
                else "The optimal branch is unique; a route that returns the "
                "other branch has found the other local solution, which is "
                "reported with its objective."
            ),
        ),
        provenance="derived in-repo.",
        validators=(_selector_validator,),
    )


def _selector_validator(case: MpccCase, x: np.ndarray) -> dict:
    """Which branch, and is it a clean one-hot.

    Commitment as such is the `selector` class validator's job
    (`validate._selector`); what is case-specific here is *which* branch
    came out, which is the reportable quantity at the tie and the thing
    a branch-change comparison across theta is made of.
    """
    # Both quantities come off the pair values, which `rescale` leaves
    # alone; ``x`` itself does not survive the skew leg (the solution
    # (0, 1) becomes (0, 1e-3) there, and a test against 1 fails for no
    # reason but the units).
    y1, y2 = case.pair_values(x)
    y1, y2 = float(y1[0]), float(y2[0])
    onehot = (abs(y1 - 1.0) < 1e-5 and abs(y2) < 1e-5) or (
        abs(y1) < 1e-5 and abs(y2 - 1.0) < 1e-5
    )
    return {
        "branch": "A" if y1 > y2 else "B",
        "one_hot_ok": bool(onehot),
    }


# --------------------------------------------------------------------
# MacMPEC-style pinned subset
# --------------------------------------------------------------------


def ralph2() -> MpccCase:
    """The relaxed optimum approaches the MPCC optimum *from below*.

    ``min x1^2 + x2^2 - 4 x1 x2  s.t.  0 <= x1 _|_ x2 >= 0``

    On either axis the objective is ``t^2 >= 0``, so ``f* = 0`` at the
    origin, biactive, with ``grad f(0) = 0`` -- every multiplier vector
    is zero and the point is S-stationary. MPCC-LICQ holds.

    The relaxation is the point of the case. At ``x1 = x2 = t`` the
    objective is ``-2 t^2`` while the relaxed row gives ``t^2 <= tau``,
    so the ``tau``-problem's optimum is exactly ``-2 tau``, attained on
    the relaxation boundary and **below** the MPCC's own optimum for
    every ``tau > 0``. Continuation therefore approaches ``0`` from
    below through a sequence of points that are not MPCC-feasible, and a
    harness that reads a relaxed objective as an MPCC objective reports a
    better-than-optimal answer. The runner evaluates every stage's source
    feasibility for this reason.
    """
    f = Quad(np.array([[2.0, -4.0], [-4.0, 2.0]]), np.zeros(2), 0.0)
    pairs = (_pair("p1", Affine([1.0, 0.0]), Affine([0.0, 1.0])),)
    return MpccCase(
        name="ralph2",
        klass="macmpec",
        objective=f,
        rows=(),
        pairs=pairs,
        lb=np.array([-INF, -INF]),
        ub=np.array([10.0, 10.0]),
        starts={
            "origin": np.array([0.0, 0.0]),
            "diag": np.array([1.0, 1.0]),
            "axis": np.array([2.0, 0.0]),
        },
        expected=Expected(
            feasible=True,
            obj=0.0,
            x=np.array([0.0, 0.0]),
            stationarity="S",
            n_biactive=1,
            mpcc_licq=True,
            notes=(
                "Expect the final relaxed objective to sit at about -2*tau_min "
                "for the continuation routes and at 0 for the exact-product "
                "routes. The gap is the relaxation's, not the solver's."
            ),
        ),
        provenance=(
            "derived in-repo; the instance is the classical `ralph2` shape "
            "from the MPEC regularisation literature (Scholtes 2001)."
        ),
    )


def scholtes4() -> MpccCase:
    """No S-stationary point at the solution; two active ordinary rows.

    ``min x1 + x2 - x3
       s.t. -4 x1 + x3 <= 0,  -4 x2 + x3 <= 0,
            0 <= x1 _|_ x2 >= 0``

    Complementarity forces ``min(x1,x2) = 0``, and the two rows then
    force ``x3 <= 0``, so ``f = x1 + x2 - x3 >= 0`` with equality only at
    the origin: ``f* = 0``, uniquely, biactive.

    Stationarity at the origin, with ``lambda >= 0`` on the two active
    rows:

        x1:  1 - 4 l1 - nu = 0
        x2:  1 - 4 l2 - w  = 0
        x3: -1 + l1 + l2   = 0   =>  l1 + l2 = 1

    S-stationarity needs ``nu, w >= 0``, i.e. ``l1, l2 <= 1/4``, which
    contradicts ``l1 + l2 = 1``: **the origin is not S-stationary**. It
    is M-stationary, via ``l1 = 1/4``, ``l2 = 3/4`` giving ``nu = 0``,
    ``w = -2`` (the product vanishes), and C-stationary via ``l1 = l2 =
    1/2`` giving ``nu = w = -1``.

    The multiplier set here is a line, and which class you report depends
    on searching it rather than on picking one point of it. That is why
    the classifier enumerates branches instead of taking a least-squares
    multiplier at face value.
    """
    f = Quad(np.zeros((3, 3)), np.array([1.0, 1.0, -1.0]), 0.0)
    rows = (
        Row("r1", Affine([-4.0, 0.0, 1.0]).as_quad(), -INF, 0.0),
        Row("r2", Affine([0.0, -4.0, 1.0]).as_quad(), -INF, 0.0),
    )
    pairs = (
        _pair("p1", Affine([1.0, 0.0, 0.0]), Affine([0.0, 1.0, 0.0])),
    )
    return MpccCase(
        name="scholtes4",
        klass="macmpec",
        objective=f,
        rows=rows,
        pairs=pairs,
        lb=np.array([-INF, -INF, -INF]),
        ub=np.array([INF, INF, INF]),
        starts={
            "origin": np.array([0.0, 0.0, 0.0]),
            "offset": np.array([1.0, 1.0, 1.0]),
            "neg_x3": np.array([0.5, 0.5, -1.0]),
        },
        expected=Expected(
            feasible=True,
            obj=0.0,
            x=np.zeros(3),
            stationarity="M",
            n_biactive=1,
            # Four active gradients -- both rows, grad G and grad H -- in
            # R^3, so MPCC-LICQ fails here too, for a different reason
            # than ralph1's: the rows do it, not the bound.
            mpcc_licq=False,
            notes=(
                "The classifier must report M and refuse S. A route "
                "returning Solve_Succeeded on the NCP-equality lowering here "
                "is returning an NLP KKT point of a problem whose "
                "constraints are degenerate at every feasible point; the "
                "stationarity block is the only field that says so."
            ),
        ),
        provenance=(
            "derived in-repo; the instance is the classical `scholtes4` shape "
            "from the MPEC regularisation literature."
        ),
    )


def qpec_small() -> MpccCase:
    """A two-pair QPEC: one strict pair, one biactive pair.

    Upper level over ``x`` with a lower-level LCP in ``y``::

        min (x-1)^2 + (y1-1)^2 + y2^2
        s.t. 0 <= x <= 2
             0 <= y1 _|_ (2 y1 - 1 - x) >= 0
             0 <= y2 _|_ (2 y2 - 1 + x) >= 0

    The LCP solves in closed form: ``y1 = max(0, (1+x)/2)`` and
    ``y2 = max(0, (1-x)/2)``. For ``x`` in ``[0,2]`` that is
    ``y1 = (1+x)/2`` and ``y2 = (1-x)/2`` for ``x <= 1``, else ``0``.
    Substituting, the upper objective is ``1.5 (x-1)^2`` for ``x <= 1``
    and ``1.25 (x-1)^2`` above, so ``f* = 0`` at ``x = 1``, giving
    ``(x, y1, y2) = (1, 1, 0)``.

    At that point pair 1 is strict (``y1 = 1 > 0``, ``2y1-1-x = 0``) and
    pair 2 is **biactive** (``y2 = 0`` and ``2y2-1+x = 0``). ``grad f``
    vanishes there, so the zero multiplier vector is admissible and the
    point is S-stationary; the three active gradients ``(-1,2,0)``,
    ``(0,0,1)``, ``(1,0,2)`` have determinant 2, so MPCC-LICQ holds.

    It is the only case in the ladder with more than one pair, which is
    what makes it the one that can catch a lowering that indexes its
    product rows wrongly.
    """
    P = np.diag([2.0, 2.0, 2.0])
    f = Quad(P, np.array([-2.0, -2.0, 0.0]), 2.0)
    pairs = (
        _pair("lcp1", Affine([0.0, 1.0, 0.0]), Affine([-1.0, 2.0, 0.0], -1.0)),
        _pair("lcp2", Affine([0.0, 0.0, 1.0]), Affine([1.0, 0.0, 2.0], -1.0)),
    )
    return MpccCase(
        name="qpec_small",
        klass="macmpec",
        objective=f,
        rows=(),
        pairs=pairs,
        lb=np.array([0.0, -INF, -INF]),
        ub=np.array([2.0, INF, INF]),
        starts={
            "origin": np.array([0.0, 0.5, 0.5]),
            "upper_left": np.array([0.0, 0.0, 0.0]),
            "upper_right": np.array([2.0, 1.5, 0.0]),
        },
        expected=Expected(
            feasible=True,
            obj=0.0,
            x=np.array([1.0, 1.0, 0.0]),
            stationarity="S",
            n_biactive=1,
            mpcc_licq=True,
            notes=(
                "Two pairs, so a lowering that builds q product rows but "
                "wires the wrong (G,H) into one of them still produces a "
                "plausible answer on every one-pair case in the ladder and "
                "fails here."
            ),
        ),
        provenance="derived in-repo; a minimal LCP-constrained QPEC.",
    )


# --------------------------------------------------------------------
# registry
# --------------------------------------------------------------------

_FACTORIES: Dict[str, Callable[[], MpccCase]] = {
    "regular_strict": regular_strict,
    "biactive_positive": biactive_positive,
    "ralph1": ralph1,
    "ctrap": ctrap,
    "infeasible_pair": infeasible_pair,
    "selector_theta_025": lambda: _selector(0.25),
    "selector_theta_050": lambda: _selector(0.50),
    "selector_theta_075": lambda: _selector(0.75),
    "ralph2": ralph2,
    "scholtes4": scholtes4,
    "qpec_small": qpec_small,
}

REGISTRY: List[str] = list(_FACTORIES)

#: The deterministic smoke subset: one case per benchmark class, chosen
#: to be the cheapest member of that class that still exercises it.
#: `run --smoke` runs exactly these and asserts against the manifest, so
#: it is the subset that has to stay fast.
SMOKE: List[str] = [
    "regular_strict",
    "biactive_positive",
    "ralph1",
    "infeasible_pair",
    "selector_theta_050",
    "ralph2",
]


def make(name: str) -> MpccCase:
    return _FACTORIES[name]()


def all_cases() -> List[MpccCase]:
    return [make(n) for n in REGISTRY]

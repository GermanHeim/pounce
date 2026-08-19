#!/usr/bin/env python3
"""Generate the large-scale synthetic NLP suite as AMPL ``.nl`` files.

These five problems were originally hand-written as Rust ``TNLP`` structs
(the retired ``pounce-large-scale`` crate). They are large, sparse,
parameterised NLPs that stress the sparse linear-algebra path and
workspace sizing of both POUNCE and Ipopt. This script reproduces the
exact same math in Pyomo and emits one ``.nl`` per problem so the suite
runs through the standard ``.nl`` driver
(``benchmarks/scripts/run_nl_bench.sh``) like every other suite — no
compiled Rust harness, no libipopt FFI.

The first five (default sizes mirror the old Rust harness defaults):

  rosenbrock   Generalized/chained Rosenbrock (CUTE GENROSE), unconstrained,
               nonlinear, tridiagonal Hessian.            n = 2000
  bratu        1-D Bratu BVP, -u'' = λ e^u, 3-point stencil, feasibility
               (objective ≡ 0), nonlinear equality constraints.   n = 10000
  optcontrol   Discretised linear-quadratic optimal control; quadratic
               objective, linear dynamics constraints.    T = 50000  (n = 100001)
  poisson      2-D Poisson boundary control on a K×K grid; quadratic
               objective, linear 5-point-stencil constraints.  K = 200 (n = 80000)
  sparseqp     Convex sparse QP, tridiagonal Q, cyclic 3-term inequality
               rows, box bounds.                          n = 50000

The sixth was added for pounce #698 and is a different kind of problem —
nonconvex, implicit, and with a large active set — because the five above
have between them no nonlinear inequality, no restoration entry and no
dense Jacobian block, and that gap let #684 through a clean scaling probe:

  laptime      Minimum-lap-time vehicle trajectory, Radau direct
               collocation on a closed circuit; nonconvex objective,
               saturating tyre curves, friction-ellipse path constraint,
               periodic boundary.        N = 1000, 8 lag states (n = 58014)

Usage:
    python3 generate_nl.py                 # all problems, default sizes
    python3 generate_nl.py --scale 0.1     # 10% of every default size (quick)
    python3 generate_nl.py rosenbrock bratu # only the named problems
    python3 generate_nl.py --out-dir nl    # output directory (default: ./nl)

Per-problem sizes can also be overridden individually:
    python3 generate_nl.py --rosenbrock-n 500 --optcontrol-t 1000

`laptime` has a second dial that --scale does NOT touch, because adding
states is not a discretisation refinement:
    python3 generate_nl.py laptime --laptime-n 1400 --laptime-states 12

The ``.nl`` files (and matching ``.row``/``.col`` name maps) land in
``--out-dir`` and are regenerated locally — they are not tracked in git.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    Constraint,
    RangeSet,
    Reals,
    Set,
    minimize,
    atan,
    cos,
    exp,
    sin,
)

# Default sizes — mirror the retired Rust harness
# (benchmarks/large_scale/src/bin/large_scale_suite.rs). Rosenbrock is
# capped lower because chained Rosenbrock is fundamentally O(n) Newton
# iterations regardless of solver.
DEFAULTS = {
    "rosenbrock_n": 2000,
    "bratu_n": 10000,
    "optcontrol_t": 50000,
    "poisson_k": 200,
    "sparseqp_n": 50000,
    "laptime_n": 1000,
}

# Sizes that are not the primary `--scale` dial for their family. `laptime`
# grows two ways -- more grid points, or more states per grid point -- and
# only the first is a discretisation refinement, so only the first scales.
EXTRA_DEFAULTS = {
    "laptime_states": 8,
}

# Extra builder arguments, applied positionally after the size.
EXTRA_OPTS = {
    "laptime": ("laptime_states",),
}


def build_rosenbrock(n: int) -> ConcreteModel:
    """min 1 + Σ_{i=1}^{n-1} [100 (x_{i+1} - x_i²)² + (1 - x_{i+1})²].

    Unconstrained. Canonical CUTE GENROSE start x_i = i/(n+1).
    """
    m = ConcreteModel(name=f"rosenbrock_n{n}")
    m.I = RangeSet(1, n)
    m.x = Var(m.I, domain=Reals, initialize=lambda _m, i: i / (n + 1.0))

    def obj(m):
        return 1.0 + sum(
            100.0 * (m.x[i + 1] - m.x[i] ** 2) ** 2 + (1.0 - m.x[i + 1]) ** 2
            for i in range(1, n)
        )

    m.obj = Objective(rule=obj, sense=minimize)
    return m


def build_bratu(n: int) -> ConcreteModel:
    """Feasibility (obj ≡ 0) Bratu BVP -u'' = λ e^u on [0,1], u(0)=u(1)=0.

    Variables x_1..x_n; x_1 and x_n fixed to 0 (Dirichlet). Interior
    residual at i=2..n-1:  (-x_{i-1} + 2 x_i - x_{i+1})/h² - λ e^{x_i} = 0.
    """
    h = 1.0 / (n + 1.0)
    lam = 1.0
    m = ConcreteModel(name=f"bratu_n{n}")
    m.I = RangeSet(1, n)
    m.x = Var(m.I, domain=Reals, initialize=0.0)
    # Dirichlet boundary conditions baked into bounds.
    m.x[1].fix(0.0)
    m.x[n].fix(0.0)

    m.Interior = RangeSet(2, n - 1)

    def residual(m, i):
        return (-m.x[i - 1] + 2.0 * m.x[i] - m.x[i + 1]) / (h * h) - lam * exp(m.x[i]) == 0

    m.pde = Constraint(m.Interior, rule=residual)
    m.obj = Objective(expr=0.0, sense=minimize)
    return m


def build_optcontrol(t: int) -> ConcreteModel:
    """Discretised LQ optimal control.

    min h Σ_{i=0}^{T} (y_i - 1)² + α h Σ_{i=0}^{T-1} u_i²
    s.t. y_0 = 0;  y_{i+1} = (1-h) y_i + h u_i,  i = 0..T-1.
    """
    h = 1.0 / t
    alpha = 0.01
    m = ConcreteModel(name=f"optcontrol_t{t}")
    m.Iy = RangeSet(0, t)
    m.Iu = RangeSet(0, t - 1)
    m.y = Var(m.Iy, domain=Reals, initialize=0.0)
    m.u = Var(m.Iu, domain=Reals, initialize=0.0)

    def obj(m):
        return h * sum((m.y[i] - 1.0) ** 2 for i in range(t + 1)) + alpha * h * sum(
            m.u[i] ** 2 for i in range(t)
        )

    m.obj = Objective(rule=obj, sense=minimize)

    m.y0 = Constraint(expr=m.y[0] == 0.0)

    def dynamics(m, i):
        return m.y[i + 1] - (1.0 - h) * m.y[i] - h * m.u[i] == 0

    m.dyn = Constraint(m.Iu, rule=dynamics)
    return m


def build_poisson(k: int) -> ConcreteModel:
    """2-D Poisson boundary control on a K×K interior grid.

    min Σ_{ij} ½ h² (u_{ij} - u_d)² + ½ α h² f_{ij}²
    s.t. (4 u_{ij} - neighbours)/h² - f_{ij} = 0  (5-point stencil, Dirichlet 0).
    u_d(x,y) = sin(πx) sin(πy), x=(i+1)h, y=(j+1)h, h = 1/(K+1).
    """
    h = 1.0 / (k + 1.0)
    alpha = 0.01
    m = ConcreteModel(name=f"poisson_k{k}")
    m.I = RangeSet(0, k - 1)
    m.J = RangeSet(0, k - 1)
    m.u = Var(m.I, m.J, domain=Reals, initialize=0.0)
    m.f = Var(m.I, m.J, domain=Reals, initialize=0.0)

    def u_desired(i, j):
        x = (i + 1.0) * h
        y = (j + 1.0) * h
        return math.sin(math.pi * x) * math.sin(math.pi * y)

    def obj(m):
        return sum(
            0.5 * h * h * (m.u[i, j] - u_desired(i, j)) ** 2
            + 0.5 * alpha * h * h * m.f[i, j] ** 2
            for i in range(k)
            for j in range(k)
        )

    m.obj = Objective(rule=obj, sense=minimize)

    def pde(m, i, j):
        lap = 4.0 * m.u[i, j]
        if i > 0:
            lap -= m.u[i - 1, j]
        if i < k - 1:
            lap -= m.u[i + 1, j]
        if j > 0:
            lap -= m.u[i, j - 1]
        if j < k - 1:
            lap -= m.u[i, j + 1]
        return lap / (h * h) - m.f[i, j] == 0

    m.pde = Constraint(m.I, m.J, rule=pde)
    return m


def build_sparseqp(n: int) -> ConcreteModel:
    """Convex sparse QP with cyclic 3-term inequality rows and box bounds.

    min ½ xᵀQx - Σ x_i,  Q tridiagonal (4 on diagonal, -1 off-diagonal)
    s.t. x_j + x_{(j+1) mod n} + x_{(j+2) mod n} ≤ 2.5,  0 ≤ x_i ≤ 10.

    ½ xᵀQx expands to Σ 2 x_i² - Σ_{i=1}^{n-1} x_i x_{i+1}.
    """
    m = ConcreteModel(name=f"sparseqp_n{n}")
    m.I = RangeSet(1, n)
    m.x = Var(m.I, domain=Reals, bounds=(0.0, 10.0), initialize=0.5)

    def obj(m):
        quad = sum(2.0 * m.x[i] ** 2 for i in range(1, n + 1))
        offdiag = sum(m.x[i] * m.x[i + 1] for i in range(1, n))
        linear = sum(m.x[i] for i in range(1, n + 1))
        return quad - offdiag - linear

    m.obj = Objective(rule=obj, sense=minimize)

    def threesum(m, j):
        # 0-based cyclic indices j, j+1, j+2 → 1-based variable keys.
        a = j
        b = (j % n) + 1
        c = ((j + 1) % n) + 1
        return m.x[a] + m.x[b] + m.x[c] <= 2.5

    m.row = Constraint(m.I, rule=threesum)
    return m


# --- Minimum-lap-time direct collocation (pounce #698) -------------------
#
# WHY THIS FAMILY EXISTS. The other five problems here are large and sparse,
# but none of them is shaped like the models that have actually broken the
# limited-memory path. `optcontrol` is the closest and it is a single-state,
# linear-dynamics, convex QP: two Jacobian entries per row, no active
# inequalities, no restoration, and an exact Hessian available for free.
#
# That gap is not hypothetical. `scripts/scaling-probe.sh` measured the
# limited-memory path as linear from n = 2,000 to n = 128,000 on these
# families and reported no hidden quadratic -- and pounce #684 was, at that
# moment, allocating a dense n(n+1)/2 Hessian triangle the instant
# restoration was entered under `hessian_approximation=limited-memory`. The
# probe was right about what it measured and blind to the defect, because
# none of its problems enters restoration.
#
# This family is the shape that found #677, #684, #686 and #688: a
# minimum-lap-time vehicle trajectory problem, transcribed by direct
# collocation, reported against pounce by a user running a 60,000-variable
# model built with CasADi's `DaeBuilder` from an FMU that exposes analytic
# Jacobians and no analytic Hessian (#698). It reproduces the structural
# features that matter, none of which the other five have together:
#
#   * nonlinear implicit dynamics in residual form -- the collocation
#     equations, not an explicit Euler recurrence;
#   * many states per grid point, so the Jacobian block coupling consecutive
#     points is a dense n_x x n_x block rather than two entries;
#   * a nonconvex objective (minimum time) whose optimum sits on the
#     intersection of several active nonlinear inequalities;
#   * a friction-ellipse path constraint active over most of the horizon,
#     so the active set is large and churns;
#   * saturating tyre curves (Pacejka sin-atan), which are nonconvex in a way
#     that quadratic penalties are not;
#   * a periodicity row closing the lap, which puts a dense-ish coupling
#     between the first and last block into an otherwise banded KKT matrix.
#
# It is deliberately NOT a copy of the reporter's model, which is
# proprietary: it is an independent problem of the same kind, built from
# published vehicle-dynamics modelling, so it can be committed and shared.
#
# FORMULATION. Distance along the centreline `s` is the independent
# variable, not time -- standard for lap-time problems because the track
# geometry is then a known function of the independent variable. State
# derivatives convert as dq/ds = qdot / sdot, with
#
#     sdot = (vx cos(xi) - vy sin(xi)) / (1 - n kappa(s))
#
# and lap time enters as a state `t` with dt/ds = 1/sdot, minimised at
# s = L. The curvature kappa(s) is a truncated Fourier series with a 2*pi/L
# offset, so the track is C-infinity and closes into a genuine circuit
# (integral of kappa over the lap = 2*pi).
#
# SIZING. Variables = (N+1)*n_x + N*d*n_x + N*n_u with d = 3 Radau points,
# n_u = 2 and n_x = 6 + `lag_states`. The defaults (N = 1000, 8 lag states,
# so n_x = 14) give 58,014 variables and 56,014 equalities -- 2,000 degrees
# of freedom, exactly the control count, which is the correct DOF for a
# transcribed optimal control problem. That lands in the reporter's
# 60,000-80,000 band at their 1,000 grid points.
#
# The `lag_states` dial is the honest way to grow n_x: each state is one
# stage of a first-order lag cascade on the steering command, which is a
# real actuator/compliance model, and the per-stage time constant is
# TAU_STEER/K so the *physics* stays put as the state count changes. Adding
# states therefore changes the linear-algebra size without changing the
# trajectory being asked for -- which is what you want from a size dial.

_RADAU_ROOTS = {
    1: [1.0],
    2: [1.0 / 3.0, 1.0],
    3: [0.15505102572168222, 0.6449489742783178, 1.0],
}

# Vehicle and circuit parameters. A ~800 kg, ~400 kW car on a 4 km circuit;
# the numbers are representative rather than drawn from any real vehicle.
_LAP_L = 4000.0        # circuit length [m]
_LAP_MASS = 800.0      # mass [kg]
_LAP_IZ = 1000.0       # yaw inertia [kg m^2]
_LAP_LF = 1.6          # CoG to front axle [m]
_LAP_LR = 1.4          # CoG to rear axle [m]
_LAP_MU = 1.6          # tyre-road friction coefficient
_LAP_G = 9.81
_LAP_BF = _LAP_BR = 10.0   # Pacejka stiffness factor
_LAP_CF = _LAP_CR = 1.9    # Pacejka shape factor
_LAP_KDRAG = 0.9       # 0.5 * rho * Cd * A  [N s^2/m^2]
_LAP_FXMAX = 8000.0    # peak longitudinal force [N]
_LAP_PMAX = 4.0e5      # peak power [W]
_LAP_TAU_STEER = 0.05  # total steering lag [s]
_LAP_WTRACK = 5.0      # usable half-width [m]
_LAP_VMIN, _LAP_VMAX = 5.0, 120.0
_LAP_V0 = 45.0         # initial-guess speed [m/s]
_LAP_WREG = 1.0e-2     # control-rate regularisation weight

# Peak lateral force per axle, split by static weight distribution.
_LAP_DF = _LAP_MU * _LAP_MASS * _LAP_G * _LAP_LR / (_LAP_LF + _LAP_LR)
_LAP_DR = _LAP_MU * _LAP_MASS * _LAP_G * _LAP_LF / (_LAP_LF + _LAP_LR)


def _polymul(p, q):
    """Multiply two polynomials, highest-degree coefficient first."""
    out = [0.0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] += a * b
    return out


def _polyval(p, x):
    v = 0.0
    for c in p:
        v = v * x + c
    return v


def _polyder(p):
    n = len(p) - 1
    return [p[i] * (n - i) for i in range(n)] if n > 0 else [0.0]


def _collocation_coeffs(d):
    """Radau IIA differentiation matrix `C` and continuity vector `D`.

    Same construction CasADi's `direct_collocation` example uses: build the
    Lagrange basis on [0, tau_1, .., tau_d], then `C[j][r] = L_j'(tau_r)`
    and `D[j] = L_j(1)`. For Radau IIA the last point is 1.0, so `D` comes
    out `[0, .., 0, 1]` and continuity reduces to "the interval ends at its
    last collocation point" -- but the general form is what is emitted, so
    the transcription stays correct if the scheme is ever changed.
    """
    tau = [0.0] + _RADAU_ROOTS[d]
    cmat = [[0.0] * (d + 1) for _ in range(d + 1)]
    dvec = [0.0] * (d + 1)
    for j in range(d + 1):
        poly = [1.0]
        for r in range(d + 1):
            if r != j:
                den = tau[j] - tau[r]
                poly = _polymul(poly, [1.0 / den, -tau[r] / den])
        dvec[j] = _polyval(poly, 1.0)
        pder = _polyder(poly)
        for r in range(d + 1):
            cmat[j][r] = _polyval(pder, tau[r])
    return tau, cmat, dvec


def _lap_kappa(s):
    """Track curvature [1/m] at centreline distance `s`.

    Truncated Fourier series plus a 2*pi/L offset so the heading turns
    through exactly one revolution over the lap -- a closed circuit rather
    than a wiggly line. Smooth by construction, which is the point: the
    reporter's track surface is differentiable for the same reason.
    """
    w = 2.0 * math.pi / _LAP_L
    return (w
            + 0.0040 * math.sin(w * s)
            + 0.0030 * math.sin(2.0 * w * s + 0.7)
            + 0.0020 * math.sin(3.0 * w * s + 2.1))


def build_laptime(n_intervals: int, lag_states: int = 8) -> ConcreteModel:
    """Minimum-lap-time trajectory by Radau direct collocation.

    States: `t` (elapsed time), `n` (lateral offset from the centreline),
    `xi` (heading relative to the centreline tangent), `vx`, `vy` (body-frame
    velocities), `r` (yaw rate), then `lag_states` stages of a steering lag
    cascade. Controls: steering command and normalised longitudinal force.
    """
    d = 3
    n_int = int(n_intervals)
    n_lag = max(1, int(lag_states))
    tau, cmat, dvec = _collocation_coeffs(d)
    h = _LAP_L / n_int
    tau_stage = _LAP_TAU_STEER / n_lag

    lag = ["z%d" % k for k in range(1, n_lag + 1)]
    states = ["t", "n", "xi", "vx", "vy", "r"] + lag
    controls = ["delta", "fx"]

    lb = {"t": 0.0, "n": -_LAP_WTRACK, "xi": -1.2, "vx": _LAP_VMIN,
          "vy": -25.0, "r": -2.5}
    ub = {"t": None, "n": _LAP_WTRACK, "xi": 1.2, "vx": _LAP_VMAX,
          "vy": 25.0, "r": 2.5}
    for z in lag:
        lb[z], ub[z] = -0.6, 0.6
    ulb = {"delta": -0.6, "fx": -1.0}
    uub = {"delta": 0.6, "fx": 1.0}

    def guess(name, s):
        """Constant-speed lap on the centreline -- feasible in the bounds,
        badly infeasible in the dynamics, which is the realistic starting
        point for this class of problem."""
        if name == "t":
            return s / _LAP_V0
        if name == "vx":
            return _LAP_V0
        if name == "r":
            return _lap_kappa(s) * _LAP_V0
        if name in lag or name == "delta":
            # kinematic steer angle for the instantaneous radius
            return (_LAP_LF + _LAP_LR) * _lap_kappa(s)
        return 0.0

    m = ConcreteModel(name="laptime_n%d_x%d" % (n_int, 6 + n_lag))
    m.K = RangeSet(0, n_int)
    m.Kc = RangeSet(0, n_int - 1)
    m.J = RangeSet(1, d)
    m.S = Set(initialize=states, ordered=True)
    m.C = Set(initialize=controls, ordered=True)

    m.X = Var(m.K, m.S,
              bounds=lambda m, k, i: (lb[i], ub[i]),
              initialize=lambda m, k, i: guess(i, k * h))
    m.Xc = Var(m.Kc, m.J, m.S,
               bounds=lambda m, k, j, i: (lb[i], ub[i]),
               initialize=lambda m, k, j, i: guess(i, (k + tau[j]) * h))
    m.U = Var(m.Kc, m.C,
              bounds=lambda m, k, c: (ulb[c], uub[c]),
              initialize=lambda m, k, c: guess(c, k * h) if c == "delta" else 0.0)

    def rhs(st, u, kappa):
        """d(state)/ds and the forces the path constraints need."""
        vx, vy, r, n, xi = st["vx"], st["vy"], st["r"], st["n"], st["xi"]
        delta = st[lag[-1]]          # steering after the lag cascade
        fx = _LAP_FXMAX * u["fx"]

        # Slip angles and saturating (Pacejka) axle forces.
        af = atan((vy + _LAP_LF * r) / vx) - delta
        ar = atan((vy - _LAP_LR * r) / vx)
        fyf = -_LAP_DF * sin(_LAP_CF * atan(_LAP_BF * af))
        fyr = -_LAP_DR * sin(_LAP_CR * atan(_LAP_BR * ar))
        fdrag = _LAP_KDRAG * vx * vx

        # Body-frame rigid-body dynamics (time domain).
        vxdot = (fx - fdrag - fyf * sin(delta)) / _LAP_MASS + vy * r
        vydot = (fyf * cos(delta) + fyr) / _LAP_MASS - vx * r
        rdot = (_LAP_LF * fyf * cos(delta) - _LAP_LR * fyr) / _LAP_IZ

        # Curvilinear kinematics; sdot is the change of variable to `s`.
        sdot = (vx * cos(xi) - vy * sin(xi)) / (1.0 - n * kappa)
        ndot = vx * sin(xi) + vy * cos(xi)
        xidot = r - kappa * sdot

        out = {"t": 1.0 / sdot,
               "n": ndot / sdot,
               "xi": xidot / sdot,
               "vx": vxdot / sdot,
               "vy": vydot / sdot,
               "r": rdot / sdot}
        prev = u["delta"]
        for z in lag:
            out[z] = ((prev - st[z]) / tau_stage) / sdot
            prev = st[z]
        return out, fx, fyf, fyr

    # Build each collocation point's residual once. Doing it per state would
    # rebuild the whole expression n_x times over.
    cache = {}
    for k in range(n_int):
        uk = {c: m.U[k, c] for c in controls}
        for j in range(1, d + 1):
            st = {i: m.Xc[k, j, i] for i in states}
            cache[k, j] = rhs(st, uk, _lap_kappa((k + tau[j]) * h))

    def xat(k, j, i):
        return m.X[k, i] if j == 0 else m.Xc[k, j, i]

    def colloc(m, k, j, i):
        """h * f_i == sum_j C[j][r] x_j -- the implicit residual form."""
        deriv = sum(cmat[jj][j] * xat(k, jj, i) for jj in range(d + 1))
        return h * cache[k, j][0][i] - deriv == 0

    m.colloc = Constraint(m.Kc, m.J, m.S, rule=colloc)

    def cont(m, k, i):
        return m.X[k + 1, i] - sum(dvec[jj] * xat(k, jj, i)
                                   for jj in range(d + 1)) == 0

    m.cont = Constraint(m.Kc, m.S, rule=cont)

    # Friction ellipse: the combined-slip limit. Active over most of a
    # minimum-time lap, which is what makes the active set large and mobile.
    flim = _LAP_MU * _LAP_MASS * _LAP_G

    def ellipse(m, k, j):
        _, fx, fyf, fyr = cache[k, j]
        return (fx / flim) ** 2 + ((fyf + fyr) / flim) ** 2 <= 1.0

    m.ellipse = Constraint(m.Kc, m.J, rule=ellipse)

    def power(m, k, j):
        _, fx, _fyf, _fyr = cache[k, j]
        return fx * m.Xc[k, j, "vx"] <= _LAP_PMAX

    m.power = Constraint(m.Kc, m.J, rule=power)

    # Close the lap: every state but elapsed time is periodic in s.
    m.periodic = Constraint(
        Set(initialize=[i for i in states if i != "t"], ordered=True),
        rule=lambda m, i: m.X[n_int, i] - m.X[0, i] == 0)

    m.tstart = Constraint(expr=m.X[0, "t"] == 0.0)

    def obj(m):
        laptime = m.X[n_int, "t"]
        rate = sum((m.U[(k + 1) % n_int, c] - m.U[k, c]) ** 2
                   for k in range(n_int) for c in controls)
        return laptime + _LAP_WREG * rate

    m.obj = Objective(rule=obj, sense=minimize)
    return m


BUILDERS = {
    "rosenbrock": ("rosenbrock_n", build_rosenbrock),
    "bratu": ("bratu_n", build_bratu),
    "optcontrol": ("optcontrol_t", build_optcontrol),
    "poisson": ("poisson_k", build_poisson),
    "sparseqp": ("sparseqp_n", build_sparseqp),
    "laptime": ("laptime_n", build_laptime),
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # NB: `choices=` is deliberately not used here. With `nargs="*"` argparse
    # validates the empty default against `choices` on Python >= 3.12, so the
    # documented no-argument invocation ("generate every problem") died with
    # `invalid choice: []`. Validated by hand below instead.
    p.add_argument("problems", nargs="*", metavar="PROBLEM",
                   help="problems to generate (default: all); one or more of: "
                        + ", ".join(BUILDERS))
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "nl"),
                   help="output directory for .nl files (default: ./nl)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="multiply every default size by this factor (e.g. 0.1)")
    for key, default in DEFAULTS.items():
        p.add_argument(f"--{key.replace('_', '-')}", type=int, default=None,
                       help=f"override size for {key} (default {default})")
    for key, default in EXTRA_DEFAULTS.items():
        p.add_argument(f"--{key.replace('_', '-')}", type=int, default=default,
                       help=f"{key} (default {default}); not affected by --scale")
    args = p.parse_args(argv)

    selected = args.problems or list(BUILDERS)
    unknown = [n for n in selected if n not in BUILDERS]
    if unknown:
        p.error("invalid problem(s): %s (choose from %s)"
                % (", ".join(unknown), ", ".join(BUILDERS)))
    os.makedirs(args.out_dir, exist_ok=True)

    for name in selected:
        size_key, builder = BUILDERS[name]
        override = getattr(args, size_key)
        if override is not None:
            size = override
        else:
            size = max(2, int(round(DEFAULTS[size_key] * args.scale)))
        extra = [getattr(args, k) for k in EXTRA_OPTS.get(name, ())]
        model = builder(size, *extra)
        stub = os.path.join(args.out_dir, name)
        path = stub + ".nl"
        model.write(path, format="nl",
                    io_options={"symbolic_solver_labels": True})
        nvars = sum(1 for _ in model.component_data_objects(Var))
        ncons = sum(1 for _ in model.component_data_objects(Constraint))
        print(f"wrote {path}  ({size_key}={size}, vars={nvars}, cons={ncons})")

    return 0


if __name__ == "__main__":
    sys.exit(main())

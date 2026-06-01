"""Inverse / uncertainty mapping as an ODE, integrated with diffrax over
a POUNCE sensitivity right-hand side (pounce#91; companion to notebook
13 from pounce#82).

The inverse-mapping method of Alves, Kitchin & Lima (AIChE J. 2023
e18119; 2025 e18940) maps a closed boundary in the **output** space of
an embedded optimizer back to the **input** (parameter) space by
integrating

    dθ/ds = (∂y/∂θ)^{-1} · dy/ds          (Eq. 3, 2025 paper)

along the prescribed output path ``y(s)`` — *no NLP inversion, no brute
force*. The right-hand side is a **parametric NLP sensitivity**: with
``y = output(x*(θ), θ)`` and ``x*(θ) = argmin_x f(x, θ)``, POUNCE supplies
``∂x*/∂θ`` from the held KKT factor and an off-the-shelf adaptive ODE
integrator (here diffrax) does the stepping.

The division of labour:

* POUNCE owns the **RHS** — :func:`pounce.jax.inverse_map_rhs` builds a
  ``f(s, θ) -> dθ/ds`` callable that, per evaluation, solves the NLP and
  reads ``∂x*/∂θ`` off the held factor
  (:meth:`JaxProblem.solve_with_jacobian`), then forms ``(∂y/∂θ)^{-1}
  dy/ds``. Note this is a **linear solve against** the sensitivity, not a
  ``J @ v`` product.
* diffrax owns the **stepping** — adaptive Dopri5 with a PID controller
  and dense output. The POUNCE solve inside the RHS rides a
  ``jax.pure_callback``, so it composes cleanly under the integrator.

When to reach for this vs. :class:`pounce.jax.PathFollower`: use this
smooth-ODE recipe when ``∂y/∂θ`` stays non-singular and the optimizer's
active set is fixed along the path. Switch to the predictor–corrector
``PathFollower`` when the path folds (``∂y/∂θ`` singular) or the active
set changes — it monitors both.

Run::

    python inverse_map_diffrax.py

Requires ``diffrax`` for the adaptive integration; falls back to a fixed
RK4 (and says so) if diffrax isn't installed, so the round-trip still
runs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax import config

config.update("jax_enable_x64", True)

from pounce.jax import JaxProblem, inverse_map_rhs


def build_problem() -> JaxProblem:
    """A small, smooth, *nonlinearly coupled* embedded optimizer.

    ``min_x (x0 - θ0)^2 + (x1 - θ1)^2 + 0.15 x0^2 x1^2`` — unconstrained,
    so the solution ``x*(θ)`` is 2-D and ``∂x*/∂θ`` is generically
    invertible (the inverse map is well-posed). The coupling term makes
    ``∂x*/∂θ`` vary along the path, so the sensitivity genuinely changes
    from step to step.
    """
    def f(x, p):
        return (x[0] - p[0]) ** 2 + (x[1] - p[1]) ** 2 + 0.15 * x[0] ** 2 * x[1] ** 2

    return JaxProblem(
        f=f, g=None, n=2, m=0, p_example=jnp.zeros(2),
        options={"tol": 1e-11, "print_level": 0, "sb": "yes"},
    )


def output_path(s):
    """Prescribed closed boundary in OUTPUT space: an ellipse traced once
    over ``s ∈ [0, 1]``."""
    ang = 2.0 * jnp.pi * s
    return jnp.array([0.6 + 0.35 * jnp.cos(ang), 0.4 + 0.25 * jnp.sin(ang)])


def output_velocity(s):
    """``dy/ds`` of :func:`output_path`."""
    ang = 2.0 * jnp.pi * s
    w = 2.0 * jnp.pi
    return jnp.array([-0.35 * w * jnp.sin(ang), 0.25 * w * jnp.cos(ang)])


def seed_theta(jp: JaxProblem, y0) -> jnp.ndarray:
    """Find θ0 with ``x*(θ0) = y0`` — the anchor for the ODE. A couple of
    Newton steps on ``x*(θ) − y0`` using the held-factor sensitivity
    ``∂x*/∂θ`` (which is the inverse-map RHS machinery, reused)."""
    theta = jnp.asarray(y0, dtype=jnp.float64)
    for _ in range(10):
        x_star, _duals, J = jp.solve_with_jacobian(theta, jnp.zeros(2))
        resid = x_star - jnp.asarray(y0)
        if float(jnp.max(jnp.abs(resid))) < 1e-12:
            break
        theta = theta - jnp.linalg.solve(J, resid)
    return theta


def integrate_diffrax(rhs, theta0):
    """Adaptive Dopri5 over the POUNCE sensitivity RHS. Returns the dense
    solution sampled on a uniform grid, or ``None`` if diffrax is absent."""
    try:
        import diffrax
    except ImportError:
        return None

    def vf(s, theta, args):
        return rhs(s, theta)

    term = diffrax.ODETerm(vf)
    solver = diffrax.Dopri5()
    controller = diffrax.PIDController(rtol=1e-9, atol=1e-11)
    ts = jnp.linspace(0.0, 1.0, 121)
    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=1.0, dt0=0.01,
        y0=jnp.asarray(theta0),
        stepsize_controller=controller,
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=100_000,
    )
    return np.asarray(ts), np.asarray(sol.ys)


def integrate_rk4(rhs, theta0, n_steps=240):
    """Fixed-step RK4 fallback (no external dependency)."""
    h = 1.0 / n_steps
    th = jnp.asarray(theta0, dtype=jnp.float64)
    ts = [0.0]
    ys = [np.asarray(th)]
    for i in range(n_steps):
        s = i * h
        k1 = rhs(s, th)
        k2 = rhs(s + h / 2, th + h / 2 * k1)
        k3 = rhs(s + h / 2, th + h / 2 * k2)
        k4 = rhs(s + h, th + h * k3)
        th = th + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ts.append((i + 1) * h)
        ys.append(np.asarray(th))
    return np.asarray(ts), np.asarray(ys)


def main() -> None:
    jp = build_problem()

    # POUNCE supplies the RHS; the output is the optimizer's solution
    # itself (identity), so ∂y/∂θ = ∂x*/∂θ and dθ/ds = J^{-1} dy/ds.
    # Pass warm=True to warm-start each inner solve from the previous
    # one (a modest ~1.3x; see python/benchmarks/inverse_map_warm.py).
    rhs = inverse_map_rhs(jp, output_velocity)

    y0 = output_path(0.0)
    theta0 = seed_theta(jp, y0)

    out = integrate_diffrax(rhs, theta0)
    if out is None:
        print("diffrax not installed — using fixed-step RK4 fallback.\n")
        ts, thetas = integrate_rk4(rhs, theta0)
    else:
        print("Integrated with diffrax Dopri5 (adaptive, rtol=1e-9).\n")
        ts, thetas = out

    # Validation 1 — closed output loop ⇒ closed input loop.
    loop_gap = float(np.max(np.abs(thetas[-1] - thetas[0])))

    # Validation 2 — round-trip: push the recovered input path θ(s) back
    # through the optimizer and check it lands on the output boundary y(s).
    rt = 0.0
    for k in range(len(ts)):
        x_star, *_ = jp.solve_with_jacobian(jnp.asarray(thetas[k]), jnp.zeros(2))
        rt = max(rt, float(np.max(np.abs(np.asarray(x_star) - np.asarray(output_path(ts[k]))))))

    print(f"recovered input loop closes to   : {loop_gap:.2e}")
    print(f"round-trip onto output boundary  : {rt:.2e}")
    print(f"input-space θ extent             : "
          f"θ0∈[{thetas[:,0].min():.3f}, {thetas[:,0].max():.3f}], "
          f"θ1∈[{thetas[:,1].min():.3f}, {thetas[:,1].max():.3f}]")
    print("\nThe inverse map ran on POUNCE's KKT sensitivity as the ODE "
          "RHS — no NLP inversion, one solve + back-solve per RHS eval.")


if __name__ == "__main__":
    main()

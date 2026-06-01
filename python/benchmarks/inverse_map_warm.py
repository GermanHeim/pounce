"""Benchmark: warm- vs cold-started inner solves in the inverse-map ODE
RHS (pounce#91, follow-up to the warm= option).

The pure inverse-map ODE (`inverse_map_rhs`) solves the embedded NLP at
*every* integrator stage. `warm=True` seeds each solve from the previous
evaluation's converged primal / duals / barrier μ (pounce#86) instead of
a cold start. This harness quantifies the payoff — both the
size-independent **iteration** reduction and the end-to-end **wall-clock**
— across NLP sizes, so the recipe's guidance is grounded in numbers
rather than a guess.

Headline finding (run on this machine; see the printed table):

* Warm-starting cuts IPM **iterations** by ~1.4-1.7x, roughly flat in n.
  This is the known interior-point warm-start ceiling: the barrier
  homotopy resists a boundary-proximate seed, so the win is modest
  (active-set / SQP methods warm-start far better).
* **Wall-clock** speedup is smaller still (~1.2-1.3x) and does *not* grow
  with n at these scales, because each RHS evaluation also rebuilds the
  full Jacobian `J = ∂x*/∂θ` (n held-factor back-solves) and pays
  per-call FFI / callback overhead — costs warm-starting doesn't touch.

Takeaway: `warm=True` is a minor tweak, worth flipping on for expensive
solves but never transformative. For a real speedup on a smooth map with
expensive solves, reach for `PathFollower`, whose predictor *skips* most
solves entirely instead of just making each one cheaper.

Run:
    python python/benchmarks/inverse_map_warm.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from pounce.jax import JaxProblem, inverse_map_rhs


def _make(n: int, c: float = 0.05) -> JaxProblem:
    # Convex, banded-nonlinear NLP; square inverse map (output = identity,
    # p = n). The quartic ring coupling makes the IPM take several
    # iterations cold (so warm-starting has something to save).
    def f(x, p):
        return jnp.sum((x - p) ** 2) + c * jnp.sum((x[:-1] * x[1:]) ** 2)

    return JaxProblem(
        f=f, g=None, n=n, m=0, p_example=jnp.zeros(n),
        options={"tol": 1e-9, "print_level": 0, "sb": "yes"},
    )


def _iteration_counts(jp: JaxProblem, thetas) -> tuple[int, int]:
    """Total IPM iterations along the θ sweep, cold vs warm — the
    size-independent lever, measured directly via warm_anchor."""
    n = jp._n
    cold = 0
    for th in thetas:
        st, info = jp.warm_anchor(th, jnp.zeros(n))
        cold += int(info["iter_count"])
        st.close()
    warm = 0
    x = np.zeros(n)
    duals = None
    mu = None
    for th in thetas:
        st, info = jp.warm_anchor(th, x, duals=duals, mu=mu)
        warm += int(info["iter_count"])
        x = np.asarray(st.x_star[0])
        duals = tuple(np.asarray(d[0]) for d in st.duals)
        mu = float(info["mu"])
        st.close()
    return cold, warm


def _wall(jp: JaxProblem, thetas, *, warm: bool) -> float:
    rhs = inverse_map_rhs(jp, lambda s: jnp.ones(jp._n), warm=warm)
    rhs(0.0, thetas[0])                       # warm up / JIT the callback
    t0 = time.time()
    for k, th in enumerate(thetas):
        rhs(k / len(thetas), th)
    return time.time() - t0


def main() -> None:
    K = 16
    print(f"{'n':>4}  {'iters cold/warm':>16}  {'iter x':>7}  "
          f"{'wall cold/warm (s)':>20}  {'wall x':>7}")
    print("-" * 66)
    for n in (2, 20, 60, 120):
        jp = _make(n)
        base = jnp.full(n, 0.2)
        thetas = [base + (0.5 * k / K) * jnp.ones(n) for k in range(K)]
        ci, wi = _iteration_counts(jp, thetas)
        tc = _wall(jp, thetas, warm=False)
        tw = _wall(jp, thetas, warm=True)
        print(f"{n:>4}  {f'{ci}/{wi}':>16}  {ci / max(wi, 1):>6.2f}x  "
              f"{f'{tc:.2f}/{tw:.2f}':>20}  {tc / max(tw, 1e-9):>6.2f}x")
    print("\nWarm-start is a modest, roughly size-flat tweak (IPM warm-start "
          "ceiling + Jacobian-build overhead). Use PathFollower to skip "
          "solves outright when the map is smooth and solves are expensive.")


if __name__ == "__main__":
    main()

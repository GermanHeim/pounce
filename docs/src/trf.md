# Glass Box / Black Box Optimization

`pounce.trf_minimize` solves problems where part of the model is an *equation*
and part is a *program*:

\\[
\\min_x f(x) \\quad \\text{s.t.} \\quad h(x)=0,\\; g(x)\\le 0,\\; y = d(w)
\\]

Here \\(f, h, g\\) are ordinary algebra with exact derivatives — the **glass
box** — while \\(d\\) is a **black box**: a CFD solve, a converged unit model, a
trained network. Something you can *call* but cannot hand to an NLP solver as
equations. \\(w\\) and \\(y\\) are subvectors of \\(x\\).

This is common in process engineering, where a flowsheet is algebraic except
for one unit that needs its own simulator.

## Why not just fit a surrogate and optimize that?

Because it does not work, and it fails quietly. Consider

\\[
\\min\\; x_1^2 + x_2^2 \\quad \\text{s.t.}\\quad x_2 = x_1^3 + x_1^2 + 1
\\]

whose solution is \\((0, 1)\\) with \\(f = 1\\). Replace the constraint with a
linear surrogate \\(x_2 = x_1 + b\\), fit \\(b\\) so the surrogate matches the
truth model at the current point, optimize, refit, repeat. Started *exactly at
the optimum*, this iteration walks away and converges to \\((-1, 1)\\), where
\\(f = 2\\) — a local **maximum** of the real problem (Biegler 2024, Fig. 2a).

The reason is that matching *values* gives feasibility but says nothing about
*optimality*, which is a statement about gradients. The trust-region filter
method fixes this by (a) correcting the surrogate to match the truth model in
both value and slope, and (b) confining each step to a region where that
correction is still valid.

## Quick start

```python
import numpy as np
import pounce

# Minimize (z-1)^2 + x^2 subject to z = sin(x), treating sin as a black box.
# The variable vector is v = [x, z]: v[0] is the black-box input w,
# v[1] is its output y.
res = pounce.trf_minimize(
    fun=lambda v: (v[1] - 1.0) ** 2 + v[0] ** 2,
    x0=[0.5, 0.0],
    truth_model=lambda w: np.sin(w),
    w_index=[0],
    y_index=[1],
    jac=lambda v: np.array([2 * v[0], 2 * (v[1] - 1.0)]),
    truth_jac=lambda w: np.cos(w).reshape(1, 1),
)
print(res.x, res.fun, res.n_truth_evals)
```

You do **not** write the relationship `y = d(w)` into `constraints` yourself —
`w_index` and `y_index` tell the method which variables they are, and it
installs the surrogate constraint into each subproblem.

A worked, runnable version of everything below is in
[`python/notebooks/29_trust_region_filter.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/29_trust_region_filter.ipynb):
the failure mode plotted against the objective contours, the ZOC/FOC identities
demonstrated on a deliberately bad basis, a basis cost comparison, and the
Eason & Biegler benchmark with its convergence traces.

## How it works

At each iteration the method:

1. **Builds a corrected surrogate** at the current point \\(w_k\\):
   \\[
   r_k(w) = \\bar r(w) + \\big(d(w_k) - \\bar r(w_k)\\big)
          + \\big(J_d(w_k) - J_{\\bar r}(w_k)\\big)(w - w_k)
   \\]
   The last two terms are the zero- and first-order corrections (ZOC/FOC).
   They guarantee \\(r_k(w_k) = d(w_k)\\) and \\(J_{r_k}(w_k) = J_d(w_k)\\)
   for *any* differentiable basis \\(\\bar r\\).

2. **Solves an ordinary NLP** with \\(d\\) replaced by \\(r_k\\), and the trust
   region imposed as bounds on the decision variables:
   \\(\\max(l, u_k - \\Delta) \\le u \\le \\min(b, u_k + \\Delta)\\).

3. **Evaluates the truth model** at the trial point and measures the mismatch
   \\(\\theta = \\lVert y - d(w) \\rVert\\).

4. **Accepts or rejects** via a filter on \\((\\theta, f)\\) — a Pareto front of
   trade-offs rather than a penalty function with a parameter to tune.

Because ZOC/FOC makes the surrogate κ-fully linear automatically, the
compatibility check and criticality phase of the original 2016 algorithm are
unnecessary and \\(\\Delta\\) need not shrink to zero.

## Choosing a basis

The basis \\(\\bar r\\) affects efficiency, not correctness — the corrections
hold regardless of how good it is.

| `basis` | Samples per iteration | Use when |
|---|---|---|
| `"zero"` (default) | 0 (base point only) | Always start here. The surrogate is a plain linearization; this is also what `pyomo.contrib.trustregion` does by default. |
| `"quadratic"` | \\((n_w{+}1)(n_w{+}2)/2\\) | Curvature matters and truth calls are cheap. Grows fast: 66 samples at \\(n_w = 10\\). |

You can also pass any object implementing the `Basis` protocol
(`fit`, `predict`, `jacobian`) — a low-fidelity physical model, a symbolic
regression fit, a Gaussian process. No adapter or registration is needed.

### There is no `"linear"`, and that is not an oversight

**Only curvature survives the correction.** Write the ZOC/FOC formula with an
affine basis \\(\\bar r(w) = a + B(w - w_{\\text{ref}})\\). Its Jacobian is
\\(B\\) everywhere, so the basis-dependent part is

\\[
\\bar r(w) - \\bar r(w_k) - B(w - w_k) = B(w-w_k) - B(w-w_k) = 0
\\]

leaving \\(r_k(w) = d(w_k) + J_d(w_k)(w - w_k)\\) — exactly the `"zero"` result.
An affine basis is therefore *provably* incapable of changing anything, while
costing \\(n_w + 1\\) truth-model calls per iteration to compute it. Before
adding any basis, check whether it is affine in \\(w\\); if it is, it cannot help.

Two findings from the literature worth internalizing before reaching for
something fancy:

- **Fit quality does not predict optimization performance.** Pedrozo et al.
  (2025) benchmarked five surrogate families on a CO₂ pooling problem. Radial
  basis functions had the best R² of any model and needed 8 TRF iterations;
  Kriging needed 2; global polynomials were worst at both.
- **Simple often wins outright.** On Williams-Otto, Eason & Biegler (2016)
  found linear interpolation beat Kriging by 91 truth-model calls to 3141.

## Fit the basis once and freeze it

By default a string basis is re-fitted from fresh truth-model samples every
iteration. That is the wrong trade when the truth model is expensive, and it is
not how the literature uses surrogates: Pedrozo et al. fit an ALAMO model *once*
from a designed dataset and then let ZOC/FOC re-anchor it at each new point.

Pass a pre-fitted object and `trf_minimize` freezes it — `fit` is never called,
no per-iteration sampling happens, and each iteration costs one truth-model
evaluation plus a gradient. On the `sin` example:

| configuration | iterations | truth evals in the loop |
|---|---|---|
| `"zero"` | 7 | 8 |
| `"quadratic"`, refit each iteration | 3 | 10 |
| `"quadratic"`, **frozen** | 3 | **4** (+3 upfront) |

Freezing keeps the quadratic's three-iteration convergence but drops the in-loop
cost from 10 calls to 4.

**Freezing beats refitting the same basis; it does not automatically beat
`"zero"`.** At \\(n_w = 3\\) on a similar problem the numbers come out:

| configuration | iterations | in-loop | upfront | total |
|---|---|---|---|---|
| `"zero"` | 9 | 10 | 0 | **10** |
| `"quadratic"`, refit | 4 | 41 | 0 | 41 |
| `"quadratic"`, frozen | 6 | 7 | 10 | 17 |

Freezing cuts the quadratic's cost by more than half (41 → 17), but the free
`"zero"` basis still wins outright. The upfront design is
\\((n_w{+}1)(n_w{+}2)/2\\) calls — 10 at \\(n_w=3\\), 66 at \\(n_w=10\\) — paid
whether or not the curvature turns out to help. It amortizes when the run is
long or each truth call is genuinely expensive, and not otherwise.

Start with `"zero"`. Reach for a frozen curved basis when you can see the
iteration count is the bottleneck and you have samples to spare.

```python
from pounce.trf import QuadraticBasis, quadratic_design

design = quadratic_design(w0, 0.05)
basis = QuadraticBasis().fit(design, np.vstack([truth_model(w) for w in design]))

res = pounce.trf_minimize(..., basis=basis)   # frozen automatically
```

The default is *auto*: string bases refit, user-supplied objects freeze — if
you fitted a model yourself it will not be clobbered. Override either way with
`refit_basis=True|False`.

This is also sound rather than merely cheap. Because ZOC/FOC forces
\\(r_k(w_k) = d(w_k)\\) and \\(J_{r_k}(w_k) = J_d(w_k)\\) at every new base
point, a basis fitted somewhere else entirely still converges to the truth
model's solution — it only contributes curvature.

## Supply `truth_jac` if you possibly can

It is the single highest-value option here. It removes \\(n_w\\) truth-model
calls per iteration and makes the surrogate exactly first-order accurate rather
than accurate to the finite-difference step. Many simulators expose it —
COMSOL's sensitivity module, Aspen's equation-oriented mode — and the ZOC/FOC
variant is designed around the assumption that it is available.

Without it, the method finite-differences at the *sampling* radius \\(\\sigma_k\\).
That radius is deliberately not machine epsilon: the right perturbation is a
property of the truth model. Eason & Biegler had to inflate it to
\\(\\min(0.1, 0.8\\Delta)\\) for a boiler model "to compensate for greater
numerical noise in the model outputs", against \\(10^{-5}\\) for smooth steam
tables.

## Two radii, not one

`trust_radius` (\\(\\Delta\\)) bounds the **step**. `sampling_radius`
(\\(\\sigma \\le \\Delta\\)) bounds where the surrogate is **fit**. Eason &
Biegler (2018) introduced this separation and reported it more than doubled the
number of problems their test set could solve: with one radius doing both jobs,
the algorithm is forced into tiny steps near the solution purely to keep the
model accurate.

## Convergence

`trf_minimize` reports success when **both**:

- \\(\\theta \\le\\) `feasibility_tol` — the surrogate agrees with the truth
  model, so the point is feasible for the real problem; and
- the step in \\(w\\) is below `criticality_tol` — so the FOC gradient match is
  still valid *at the solution*.

The second is not optional. ZOC/FOC pins \\(J_{r_k} = J_d\\) only at the base
point, so a large accepted step leaves the subproblem solving KKT conditions
with a stale Jacobian. Testing feasibility alone will report convergence at
points whose true gradient is of order one.

## Limitations

**Noise.** The truth model must be deterministic and smooth. Eason & Biegler
assume noise is negligible and list rigorous noise handling as open. For
optimization against physical measurements, use a noise-aware method such as
Bayesian optimization.

**Local.** Converges to a *local* KKT point. Pair it with
[`find_minima`](find-minima.md) for a multistart sweep.

**No restoration phase.** The published algorithm calls a restoration procedure
in two situations; this implementation approximates one and detects the other.

*Incompatible subproblem.* The glass-box constraints may have no solution inside
the current trust region — common on the first iteration, when the default
radius is simply too small for the constraints to be satisfiable. Contracting
would be exactly backwards, so `trf_minimize` **expands** the radius and retries,
logging the iteration as `incompatible`. On Eason's example 1 this fires three
times before the first real step:

```text
   0  incompatible (Infeasible_Problem_Detected); Delta -> 2.000e-01
   1  incompatible (Infeasible_Problem_Detected); Delta -> 4.000e-01
   2  incompatible (Infeasible_Problem_Detected); Delta -> 8.000e-01
   3  f= 0.2748077559  theta=4.513e-02  Delta=2.677e+00  f-step
```

If the radius reaches `trust_radius_max` and the subproblem is still
infeasible, the constraints are infeasible for reasons unrelated to the trust
region, and you get a clear error rather than a wrong answer.

*Blocked filter.* When the filter rejects every candidate, the iteration can
settle into an f-step / θ-step / rejected limit cycle in which θ creeps down but
the step length never shrinks. Real restoration would find a point acceptable to
the filter; without it, `trf_minimize` detects the stall and returns
`success=False` with a `Stalled:` message rather than silently burning its
iteration budget. If you hit it, try a larger `trust_radius`, a richer basis, or
a looser `feasibility_tol`.

## References

- Eason, J.P. & Biegler, L.T. *A trust region filter method for glass box/black
  box optimization.* AIChE J. **62**, 3124–3136 (2016).
- Eason, J.P. & Biegler, L.T. *Advanced trust region optimization strategies for
  glass box/black box models.* AIChE J. **64**, 3934–3943 (2018).
- Yoshio, N. & Biegler, L.T. *Demand-based optimization of a chlorobenzene
  process…* AIChE J. **67**, e17054 (2021).
- Biegler, L.T. *The trust region filter strategy.* Digital Chemical Engineering
  **13**, 100197 (2024).
- Pedrozo, H.A. et al. *Surrogate model optimization: a comparison case study
  with pooling problems of CO₂ point sources.* Comput. Chem. Eng. **200**,
  109199 (2025).

# Homogeneous self-dual embedding for the convex IPM — design note

**Status: Phases H2 + H3 landed — HSDE solves LP/QP/SOCP.**
Chosen as the foundation for Clarabel cone parity (see
`clarabel-parity.md`): reformulate the interior-point driver into a
homogeneous self-dual embedding (HSDE), prove it reproduces every existing
LP/QP/SOCP result and infeasibility certificate, switch over, and *then*
add the non-symmetric (exp/power) and PSD cones onto the uniform HSDE
driver — the structure Clarabel, SCS, and ECOS use.

## Why HSDE

The current driver (`ipm.rs`) is an infeasible-start primal–dual method
with a **bolt-on** verified certificate check (`detect_infeasibility`). It
works, but:

- infeasibility/unboundedness is detected by watching the iterate diverge
  along a Farkas/recession ray — robust but heuristic in *when* it fires;
- there is no single self-starting iterate that handles primal- and
  dual-infeasible problems uniformly;
- non-symmetric cones (exp, power) are far better behaved inside HSDE — the
  embedding bounds the iterates and gives a clean central path.

HSDE folds primal, dual, and the infeasibility certificates into **one**
self-dual system. Its solution either has `τ > 0` (recover the optimal
primal–dual point by dividing by `τ`) or `κ > 0` (a certificate of
primal or dual infeasibility) — decided *at convergence*, not by a side
test.

## What is reused (the whole point)

The per-cone math — `kkt_block` (NT scaling `W²`), `rhs_comp_term`,
`recover_ds`, `comp_residual{,_corrector}`, `max_step`, `mu` — is **reused
verbatim**. So is `KktStructure`: the embedding borders the existing
symmetric `(x, y, z)` block

```text
      ⎡ P+δI   Aᵀ      Gᵀ      ⎤
  M = ⎢ A      −δI     0       ⎥        (exactly today's KKT matrix)
      ⎣ G      0     −W²−δI    ⎦
```

with one extra scalar `τ` (and its complement `κ`). The bordered system is
solved by **two** back-solves through the *same* factorization of `M` plus
a scalar Schur complement (the SCS/ECOS scheme), so the factorization, AMD
ordering, refactor-per-iteration, and the SOC aux-variable trick are
untouched. What changes is the outer iteration: residuals, the τ/κ row,
the step combination, the step length, and termination.

## The embedding — linear conic case (P = 0)

For `min cᵀx  s.t.  Ax = b, Gx + s = h, s ∈ K` with conic dual
`z ∈ K*` and free equality dual `y`, the self-dual embedding introduces
`τ ≥ 0, κ ≥ 0`:

```text
 (1)  Aᵀy + Gᵀz + c τ            = 0          (r_x, length n)
 (2)  A x            − b τ        = 0          (r_y, length m_eq)
 (3)  G x + s        − h τ        = 0          (r_z, length m_ineq)
 (4)  −cᵀx − bᵀy − hᵀz       − κ = 0          (r_τ, scalar)
      s ∈ K,  z ∈ K*,  τ ≥ 0, κ ≥ 0,  sᵀz = 0,  τκ = 0
```

This system is **self-dual** (the matrix is skew-symmetric apart from the
cone block). Goldman–Tucker: it has a solution with `τ + κ > 0`, and

- `τ > 0, κ = 0` ⇒ `(x, y, z, s)/τ` is an optimal primal–dual point;
- `τ = 0, κ > 0` ⇒ `cᵀx + bᵀy + hᵀz < 0` is impossible, so either
  `bᵀy + hᵀz < 0` with `Aᵀy+Gᵀz = 0, z ∈ K*` (primal-infeasible Farkas
  certificate) or `cᵀx < 0` with `Ax = 0, Gx + s = 0, s ∈ K`
  (dual-infeasible / unbounded recession ray).

### Central path and the Newton step

Relax the two complementarity conditions to `s ∘ z = σμ e` and
`τκ = σμ`, with `μ = (sᵀz + τκ)/(degree + 1)`. The Newton system for
`(Δx, Δy, Δz, Δs, Δτ, Δκ)` is the embedding matrix linearized. Eliminating
`Δs` via the cone (`Δs = −W²Δz − rhs_comp`, exactly `recover_ds`) and `Δκ`
via `τΔκ + κΔτ = σμ − τκ`, the reduced system is the bordered

```text
  ⎡ M   ⎤ ⎡Δx⎤   ⎡ ... ⎤        with border column   bcol = (c, −b, −h)
  ⎢   b ⎥ ⎢Δy⎥ = ⎢     ⎥        and  Δτ closing row    (−cᵀ,−bᵀ,−hᵀ)·(Δx,Δy,Δz)
  ⎣ col ⎦ ⎣Δz⎦   ⎣  .  ⎦                                 − (κ/τ) Δτ = r_τ + σμ/τ − κ
```

i.e. `M·Δw + Δτ·bcol = rhs_w` and `bcolᵀ·Δw − (κ/τ)Δτ = rhs_τ` (signs as in
(1)–(4)). **Two-solve scheme** (one factorization of `M`):

```text
  solve  M p = bcol        (the "constant" direction; depends only on data + scaling)
  solve  M q = rhs_w        (the "residual" direction)
  Δτ = (rhs_τ − bcolᵀ q) / (−κ/τ − bcolᵀ p)
  Δw = q − Δτ · p
```

`p` can be reused between the predictor and corrector (same `M`, same
`bcol`); only `q` and the scalars differ. So HSDE costs **one extra
back-solve per iteration** over the current method — the factorization is
shared exactly as today.

### Initial point, step, termination

- **Self-start:** `x = 0, y = 0, s = z = e` (cone identity), `τ = κ = 1`.
  Perfectly centered (`s∘z = e, τκ = 1`); no infeasible-start needed.
- **Step length:** fraction-to-boundary over the cone (`max_step` on
  `s, z`) **and** the rays `τ, κ > 0` — `α` is the min of the cone step and
  the `τ/κ` steps. One shared `α` (HSDE is symmetric in primal/dual).
- **Termination** (Clarabel/SCS style), all relative:
  - **optimal:** primal res `‖Ax−bτ‖/τ`, dual res `‖Aᵀy+Gᵀz+cτ‖/τ`, and gap
    `|cᵀx + bᵀy + hᵀz|/τ` all below `tol` (the `/τ` un-homogenizes);
  - **primal infeasible:** `τ` small, `bᵀy + hᵀz < 0`, `‖Aᵀy+Gᵀz‖` small;
  - **dual infeasible:** `τ` small, `cᵀx < 0`, `‖Ax‖, ‖Gx+s‖` small.
  These are the *same* certificate inequalities `detect_infeasibility`
  already checks — now produced natively by the embedding instead of a side
  test, so that function retires once HSDE is the default.

## The quadratic objective (P ≠ 0)

With `P`, the embedding is no longer perfectly self-dual; we adopt
Clarabel's QP embedding. Stationarity (1) gains `Px`:

```text
 (1q)  P x + Aᵀy + Gᵀz + c τ = 0
 (4q)  κ = −(cᵀx + bᵀy + hᵀz) − xᵀP x / τ
```

(At `τ>0`, dividing recovers the QP duality-gap condition
`x̂ᵀPx̂ + cᵀx̂ + bᵀŷ + hᵀẑ = 0`.) **Landed (H3).** The Newton linearization
of (4q) shows the `P` coupling enters *only* the τ-row scalar:

- `ρ_τ = κ + cᵀx + bᵀy + hᵀz + xᵀPx/τ`,
- the τ-row gradient becomes `g̃ = (c + (2/τ)Px, b, h)` (used in `g̃ᵀp`,
  `g̃ᵀq`),
- the scalar Schur denominator gains a `−xᵀPx/τ²` term.

The border *column* is unchanged — `(1q)`'s τ-coefficient is still `c`, so
`p = M⁻¹(−c, b, h)` as in the linear case — and `P` already sits in `M`'s
`(x,x)` block and in `ρ_x`. Hence the two M-solves, the cone elimination,
and the step are **identical** to H2; only the τ-row scalar differs, and it
reduces to the linear case at `P = 0`. Validated against the direct driver
and closed-form optima (equality-constrained QP; box/inequality QP; QP with
a second-order cone) — all agree.

## Phased plan

| Phase | Scope | Risk |
|---|---|---|
| H1 | This note: exact embedding, two-solve scheme, termination. | low |
| **H2** | ✅ HSDE driver for **linear** conic (`P=0`): orthant + SOC, reusing `KktStructure`/`Cone`. `solve_conic_hsde` alongside the current solver. Validated optima + both certificates vs the existing solver. | med-high — embedding signs, two-solve combination |
| **H3** | ✅ Quadratic objective: the `(1q)/(4q)` τ-row with the `P` coupling. Validated on the QP suite (closed-form optima + QP-with-SOC) vs the direct driver. | high — τ-row P algebra |
| H4 | Make HSDE the default driver; retire `detect_infeasibility`; warm start re-expressed in embedded space (`τ,κ` recentering). Whole convex + CLI + Python/JAX suite green. | med |
| H5 | **Exponential cone** on HSDE: barrier oracles, non-symmetric scaling, third-order corrector, neighborhood line search. Known-optima (GP, logistic, entropy) + KKT-residual validation. | high |
| H6 | **Power cone** (exp machinery + new barrier). | low after H5 |
| H7 | **PSD cone**: pure-Rust symmetric eig, svec/smat, dense `W⊗ₛW` block; small dense SDPs first, chordal decomposition later. | med-high |
| H8 | Cone-aware differentiable backward (JAX) for each new cone, FD-validated, as separate follow-ups. | med-high |

Validation discipline is unchanged and intrinsic: the IPM reports
`Optimal` only at a verified KKT point; each phase adds known-optima tests
plus randomized KKT-residual checks, and the orthant/SOC results stay
identical to the current solver (the cross-check that guards H2–H4). The
existing direct driver stays in place until H4 flips the default, so there
is no window where the crate regresses.

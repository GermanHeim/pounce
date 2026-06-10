# pounce-convex

Interior-point solvers for POUNCE's convex problem classes: **LP and
convex QP** today, with cone-generic scaffolding for the conic family
(SOCP, exponential/power cones, SDP) planned.

This crate is Phase 2 of the LP/QP routing plan
(`dev-notes/lp-qp-routing.md`). It provides a bare primal-dual
interior-point method for convex QP in standard form:

```text
minimize    ½ xᵀP x + cᵀx
subject to  A x = b
            G x ≤ h
```

LP is the `P = 0` case and is solved by the same driver.

## Design

- **Cone-generic.** The interior-point iteration is built over a
  [`cones::Cone`] trait with only the nonnegative orthant
  (`cones::nonneg`) implemented. Later phases add SOC / PSD / exp / pow
  cones behind the same trait, so the driver is extended, not rewritten.
- **Shared factorization.** The symmetric indefinite KKT system is solved
  through `pounce_linsol::Factorization` — the same factor-once /
  solve-many handle the NLP path uses (feral by default, MA57 optional).
  No new linear-algebra dependency.
- **Mehrotra predictor-corrector with HSDE.** The iteration uses a
  Mehrotra predictor-corrector step over a homogeneous self-dual embedding
  (`use_hsde` defaults to `true` — see [`hsde`]/[`hsde_nonsym`]), with Ruiz
  equilibration ([`equilibrate`]) on the non-HSDE path for badly-scaled
  data. The SOC / exponential / power / PSD cones (`cones::{soc,exp,power,
  psd}`) are present as tested building blocks for the conic family.

## Status

Solves convex LP and QP correctly (validated against problems with
analytically known optima, and exercised by the head-to-head benchmark
suites `benchmarks/lp_convex` and `benchmarks/qp_convex` against the
general NLP path). **Wired into the CLI dispatch**
(`crates/pounce-cli/src/dispatch.rs`): `solver_selection=auto` routes LP
and convex QP here (and convex QCQP to the same crate's SOCP IPM), and
`solver_selection=lp-ipm | qp-ipm | socp` force it explicitly. The conic
(SOCP / exponential / power / PSD) family beyond the QP/LP path is still
under construction.

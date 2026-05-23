# pounce-qp

Sparse parametric active-set quadratic programming subproblem solver
for POUNCE. The QP subproblem solver inside the active-set SQP NLP
path (Phase 5b) and the corrector inside `pounce-sensitivity`
(Phase 5c).

Pure Rust, no FFI. Layered on `pounce-common` + `pounce-linalg` +
`pounce-linsol`; tests use `pounce-feral` as the LDLᵀ backend.

## Status

**Phase 5a — feature-complete on correctness.** Every problem
class identified in the design note (`docs/research/active-set-sqp-warm-start.md`)
solves end-to-end; infeasibility is certified honestly; warm-start
is wired. Five of the six analytical-ladder problems (§8.0) pass;
problem #4 (LICQ-violating redundant equality) is a documented
limitation that needs rank-detection beyond inertia control.

The SOTA-performance items from §4.2 (Schur-complement factor
updates) and full §4.4 (Gill-Murray-Saunders-Wright EXPAND
anti-cycling, beyond Bland's-rule fallback) are deferred to
Phase 5a.1 — they are performance refinements, not correctness
prerequisites. The current solver refactors per active-set change
and uses steepest-violation drop (Dantzig's rule, the qpOASES
default), which is correct and matches qpOASES's behavior on every
non-pathological problem.

## What works

| Problem class | Path | Tested |
|---|---|---|
| Unconstrained QP | `solve_equality_only` fast path (m=0, no bounds) | ladder #1 |
| Equality-only (free vars) | `solve_equality_only` | ladder #2 |
| Box-constrained (m=0) | `solve_box_constrained` | ladder #3 |
| Equality + bounds (bound-feasible eq solution) | `solve_equality_plus_bounds` | dedicated tests |
| General inequalities (cold-feasible) | `solve_general` | dedicated tests |
| Warm-start with arbitrary working set | `solve_general` | drop / one-iter tests |
| Bound- or inequality-infeasible cold start | `solve_elastic` (l1-elastic mode, §4.3) | dedicated tests |
| Certified infeasibility detection | `solve_elastic` + `is_feasible` check | ladder #5 |
| Indefinite H with PD reduced Hessian | inertia-controlled factor (§4.5) | ladder #6 |
| Indefinite H requiring shift | `factorize_with_inertia_control` retry loop | dedicated test |

| Algorithmic feature | Status |
|---|---|
| Sparse triplet KKT assembly over `SymTMatrix` / `GenTMatrix` | done |
| §4.5 inertia control via diagonal-shift retry | done |
| §4.3 l1-elastic mode | done |
| §4.7 iterative refinement (inherited from FERAL) | done |
| §4.4 anti-cycling: Bland's rule (`AntiCyclingChoice::Bland`) | done |
| §4.4 anti-cycling: full EXPAND (Gill-Murray-Saunders-Wright 1989) | **deferred** |
| §4.2 sparse Schur-complement factor updates (parametric homotopy) | **deferred** |
| §8.1 Maros-Mészáros .qps reader | done |
| §8.1 Maros-Mészáros oracle comparison (qpOASES / OSQP) | **deferred** (requires FFI; non-pure-Rust) |
| §8.2 LASSO / MPC scaling-sweep benchmarks | **deferred** |
| §8.7 per-module unit tests for `kkt`, `elastic`, `refinement` | done |

## Public API at a glance

```rust
use pounce_qp::{ParametricActiveSetSolver, QpProblem, QpSolver, QpOptions, QpWarmStart};
use pounce_feral::FeralSolverInterface;

let mut solver = ParametricActiveSetSolver::new(Box::new(FeralSolverInterface::new()));
let sol = solver.solve(&qp, Some(&ws), &QpOptions::default())?;
assert_eq!(sol.status, pounce_qp::QpStatus::Optimal);
```

For the QPS / Maros-Mészáros on-ramp:

```rust
use pounce_qp::parse_qps;
let model = parse_qps(qps_text)?;
// ... wrap model.h_irow / model.a_irow / model.g / model.bl ... into
// pounce-linalg SymTMatrix / GenTMatrix and pass to solver.solve.
```

## Tests

49 tests across 6 test modules:

- `tests/analytical.rs` (17) — §8.0 ladder + integration tests for
  every problem class, with hand-derived expected values.
- `tests/api.rs` (11) — type-plumbing invariants for `WorkingSet`,
  `QpProblem::validate`, default `QpOptions`.
- `tests/kkt_unit.rs` (5) — §8.7 unit tests for
  `KktTriplet::add_h_diagonal_shift`.
- `tests/elastic_unit.rs` (7) — §8.7 unit tests for
  `ElasticReformulation::build` and `initial_seed`.
- `tests/refinement_unit.rs` (2) — §4.7 pin that FERAL's iterative
  refinement is on by default and delivers near-machine-precision
  on representative SPD and indefinite-saddle KKT systems.
- `tests/qps_unit.rs` (5) — QPS parser + round-trip solve.

Run with: `cargo test -p pounce-qp`.

## Design reference

The full design note, including literature pinning for every
algorithmic choice, integration plan, and per-workload notes, is at
[`docs/research/active-set-sqp-warm-start.md`](../../docs/research/active-set-sqp-warm-start.md).

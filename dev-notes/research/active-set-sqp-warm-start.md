# Design note — Active-set SQP for warm-started NLP sequences

**Status: design / proposed. Not yet implemented.** Research → plan
half of the research → plan → implement workflow; written for review
before any solver code lands. This note records the decision to commit
to the C1 active-set SQP path of
[`future-work-roadmap.md`](future-work-roadmap.md) (§3.2, §5 Phase 5)
and lays out its architecture.

## 1. What this is

An **active-set sequential quadratic programming** algorithm — a second
solver inside pounce, sharing the model / derivative / linalg
foundation but with its own iteration skeleton — designed for **warm-
started sequences of related NLPs**:

- **Model predictive control (MPC):** re-solve a similar NLP every
  control step. The horizon shifts by one stage; the active set rarely
  changes.
- **MINLP branch-and-bound:** thousands of node relaxations differing
  by a few bound changes. Bounds-only active-set updates dominate.
- **Parametric homotopy / continuation:** trace the solution along a
  parameter path. Predictor (sensitivity) + corrector (SQP step from
  the predicted point) reuses the working set across path steps.

The motivation is documented in `future-work-roadmap.md:185-206`:
interior-point methods warm-start badly because the barrier pushes
iterates to the interior, so a near-optimal point from a previous
solve sits near the bound boundary and cannot be exploited. Active-set
methods, by contrast, carry the **working set** across solves; if the
optimal active set is unchanged, the next solve converges in O(1)
QP iterations.

## 2. The architectural mismatch (read this first)

`IpoptData` / `IpoptCalculatedQuantities` are shaped around primal-dual
interior-point variables — slacks `s`, barrier `μ`, bound multipliers
`z_l`/`z_u`, complementarity quantities. Active-set SQP has none of
these: it carries `(x, λ, A)` where `A` is the **working set** (the
indices of currently active inequalities and bounds), and globalizes
on a merit function or filter without a barrier at all.

This is therefore **a new `AlgorithmStrategy` end to end** — Tier 3 in
the roadmap's tier ladder
(`future-work-roadmap.md:290-300`) — and not an edit to the existing
loop. The existing IPM (`IpoptAlgorithm::optimize` in
`crates/pounce-algorithm/src/ipopt_alg.rs`) is left untouched and
remains the default solver. Active-set SQP is an opt-in alternative
selected via a top-level builder enum, parallel to the way the
roadmap proposes composite-step trust-region as an opt-in
globalization.

The dual-skeleton commitment is the cost; the warm-start strength is
the payoff.

## 3. What pounce already has that SQP can reuse

| Need | Existing component | Location |
|---|---|---|
| NLP model trait (`f`, `g`, `∇f`, `J`, `∇²ℒ`) | `IpoptNlp` / `TNLP` | `crates/pounce-algorithm/src/ipopt_nlp.rs`, `crates/pounce-nlp/` |
| `.nl` and CUTEst frontends | `pounce-cli`, `benchmarks/cutest` | unchanged |
| Sparse linear algebra (triplet, CSC, BLAS-1) | `pounce-linalg` | unchanged |
| Symmetric LDLᵀ factor (used for KKT of the QP) | `pounce-linsol` + `pounce-feral`/`pounce-hsl` | unchanged |
| Limited-memory BFGS | `hess/quasi_newton.rs` | reused for SQP Hessian approximation |
| Filter acceptor | `line_search/filter_ls_acceptor.rs` | dominance test reusable for merit / filter SQP |
| Convergence-check trait | `conv_check::trait::ConvCheck` | reused; KKT-error formula is identical |
| Option / journalist / iteration-output plumbing | `pounce-common` + `output/` | reused; new fields for working-set events |
| Warm-start primal/dual seeds from `TNLP` | `init/warm_start.rs:60-100` | reused for cold-warm bootstrap, but extended with working-set seeds (§6) |
| Parametric sensitivity (sIPOPT port) | `pounce-sensitivity` | provides the **predictor** for parametric-homotopy use case |
| Presolve hints / bound tightening | `pounce-presolve` | unchanged; supplies tightened bounds the working set indexes into |

The interfaces below pounce-nlp are stable enough that the SQP path
inherits the full derivative and linalg layer without modification.
Everything new lives at the algorithm / solver level.

## 4. New components required

### 4.1 New crate `pounce-qp`

A standalone QP-subproblem solver. Standalone because:

- The same QP solver is the corrector inside parametric sensitivity
  (`pounce-sensitivity`), which today does only a one-shot
  reduced-Hessian linear solve — a real corrector wants the working
  set updated.
- It is independently testable against canonical QP suites (Maros–Mészáros).
- Presolve could eventually use a QP for a tighter feasibility check.

**Public API sketch** (the only types other crates depend on):

```rust
pub struct QpProblem<'a> {
    pub n: usize,            // number of primal variables
    pub m: usize,            // number of general constraints
    pub h: SymCsc<'a>,       // Hessian (positive semidefinite for convex QP)
    pub g: &'a [f64],        // linear objective term
    pub a: Csc<'a>,          // constraint Jacobian (general inequalities + equalities)
    pub bl: &'a [f64],       // lower constraint bounds (-inf for one-sided)
    pub bu: &'a [f64],       // upper constraint bounds (+inf for one-sided)
    pub xl: &'a [f64],       // lower variable bounds
    pub xu: &'a [f64],       // upper variable bounds
}

pub struct QpWarmStart {
    pub x:        Vec<f64>,        // primal seed
    pub lambda:   Vec<f64>,        // multipliers (per constraint + per bound)
    pub working:  WorkingSet,      // {lower, upper, equality, inactive} per index
}

pub trait QpSolver {
    fn solve(
        &mut self,
        qp: &QpProblem,
        ws: Option<QpWarmStart>,
    ) -> Result<QpSolution, QpError>;
}
```

**Internal algorithm — null-space active-set.** The standard choice
for SQP subproblems with potentially indefinite Hessians (Gould,
Hribar, Nocedal 2001; Gill, Murray, Wright 1991):

- Maintain an LDLᵀ factor of the **reduced Hessian** `ZᵀHZ` where `Z`
  spans the null space of the active-constraint Jacobian.
- Each iteration solves one EQP (equality-constrained QP, the
  currently-working set treated as equality), takes a step `p`, and
  either:
  - hits a non-working constraint → add it to the working set,
    update factor by rank-1.
  - reaches the EQP optimum → check multiplier signs; if any are
    wrong, drop the most-wrong one from the working set, update
    factor by rank-1.
- Terminate when EQP optimum has all-correct-sign multipliers.

Rank-1 updates use `pounce-linalg` row-update primitives that already
exist for L-BFGS. The Goldfarb-Idnani algorithm is a cheaper
alternative when the Hessian is strictly positive definite — listed as
an open question (§9).

### 4.2 SQP iterate state

```rust
pub struct SqpIterates {
    pub x: Vec<f64>,             // primals
    pub lambda_g: Vec<f64>,      // general-constraint multipliers
    pub lambda_x: Vec<f64>,      // bound multipliers (z_l − z_u packed signed)
    pub working: WorkingSet,     // active-set membership for bounds + general inequalities
    pub h_approx: HessianStore,  // exact, L-BFGS, or damped BFGS
}

pub struct WorkingSet {
    pub bound_status: Vec<BoundStatus>,   // n entries: Inactive | AtLower | AtUpper | Fixed
    pub cons_status:  Vec<ConsStatus>,    // m entries: same shape for general ineqs; Eq for equalities
}
```

This is the analog of `IteratesVector` but holds discrete state
(`working`). It is **not** a parallel `IteratesVector` — there are no
slacks and no `mu`. A new struct is cleaner than retrofitting.

### 4.3 SQP main driver

```rust
pub struct SqpAlgorithm {
    qp: Box<dyn QpSolver>,
    merit: Box<dyn SqpMerit>,          // l1 (Han-Powell) or filter
    hess: Box<dyn HessianUpdater>,     // reuses existing trait
    conv: Box<dyn ConvCheck>,          // reuses existing trait
    // …
}
```

Iteration skeleton (in pseudocode):

```
loop {
    if conv.is_kkt_satisfied(x, λ) { return Solved }
    build_qp_subproblem(x, λ, h_approx) -> qp
    p, λ_qp, ws' = self.qp.solve(qp, Some(QpWarmStart{ x, λ, working }))
    α = merit.line_search(x, p, λ_qp)             // l1 merit or filter
    x_new = x + α p
    h_approx.update(x, x_new, ∇ℒ(x), ∇ℒ(x_new))   // BFGS / damped
    (x, λ, working) = (x_new, λ_qp, ws')
}
```

The QP subproblem is one half of the per-iteration cost; the other is
one Hessian / Jacobian evaluation. There is no barrier, no μ-update,
no bound-push, no inertia correction.

### 4.4 Globalization — merit function or filter

Default proposal: **l1 (Han-Powell) merit** with adaptive penalty
update à la Byrd-Nocedal:

```
φ(x; ν) = f(x) + ν · ‖c(x)‖₁
```

Backtracking with sufficient-reduction `φ(x + αp) ≤ φ(x) + η α m'(0)`.

Alternative: **filter SQP** à la Fletcher-Leyffer, reusing the existing
`FilterLsAcceptor` dominance test. Filter SQP is arguably cleaner and
matches the IPM globalization, so the default may flip after
prototyping — listed as an open question (§9).

### 4.5 Builder integration

Add a top-level `solver_strategy` enum at the builder layer (parallel
to `globalization` proposed in the composite-step note):

```rust
// crates/pounce-algorithm/src/alg_builder.rs
pub enum SolverStrategy {
    InteriorPoint,   // default; IpoptAlgorithm
    ActiveSetSqp,    // new; SqpAlgorithm
}
```

Dispatched in `AlgorithmBuilder::build_inner` (around line 332+).
Wired from options in `application.rs` via the existing options
registry.

## 5. The QP-subproblem warm-start interface — the deliverable

This is the architectural feature that justifies the whole effort.
Three things must be carried across calls to `SqpAlgorithm::optimize`:

1. **Primal-dual iterate** `(x, λ)` — already supported by the
   existing `init/warm_start.rs` machinery; reuse unchanged.
2. **Working set** `A_prev` — discrete state per bound and per general
   inequality. New: extend `WarmStartIterateInitializer` to accept and
   forward a `WorkingSet`, or add an `SqpWarmStartIterateInitializer`
   parallel to it.
3. **Hessian approximation** `H_prev` — already supported for L-BFGS;
   reuse the existing `hess/quasi_newton.rs` carry-forward path.

The QP solver consumes `(x, λ, A_prev)` as its initial guess. If
`A_prev` is feasible for the new problem and is the optimal active
set, the QP converges in **one EQP solve** (zero active-set
updates) — the warm-start best case. The cold-warm bootstrap (when no
prior working set is available) is to call `pounce-presolve`-style
bound activity heuristics to seed `A_0`.

## 6. Per-workload considerations

### 6.1 MPC

- Working-set shift: stages of an MPC NLP have block structure. The
  natural carry-over is `A_{k+1}[i] = A_k[i+1]` (one-stage shift), with
  the new terminal stage seeded cold.
- Hessian shift: same block-shift trick for L-BFGS.
- This is a **modeling-layer convention**, not a solver feature; the
  solver only needs the SQP warm-start API to be cheap to call with a
  user-constructed `WorkingSet`.

### 6.2 MINLP branch-and-bound

- Sibling/child relaxations differ in one bound. The previous solve's
  working set is feasible for the child if the bound change doesn't
  invalidate it; otherwise one active-set update fixes it.
- The B&B driver lives outside pounce; the deliverable is the warm-
  start API the driver calls.

### 6.3 Parametric homotopy / continuation

- Step in parameter `t`: `min f(x; t) s.t. g(x; t) ≤ 0`.
- Predictor: `pounce-sensitivity` already computes
  `dx/dt`, `dλ/dt` from the reduced Hessian at the previous solution.
  Reuse unchanged.
- Corrector: one SQP solve from `(x + Δt·dx/dt, λ + Δt·dλ/dt,
  A_prev)`. If `A_prev` is still optimal, one QP iteration.
- This is the workload where SQP **outperforms** even a well-warm-
  started IPM most clearly — it is the canonical demonstration target.

## 7. Phasing

The roadmap places this whole effort at Phase 5
(`future-work-roadmap.md:398-401`). Within it, the natural sub-phasing
splits the cost across milestones, each shippable on its own.

- **Phase 5a — `pounce-qp` standalone (3–4 weeks).** New crate with
  null-space active-set QP solver. Tests on Maros-Mészáros. No NLP
  integration yet. **Exit criterion:** match a reference QP solver
  (qpOASES or quadprog) on a curated subset to within tolerance, and
  demonstrate measured warm-start speed-up on a shifted-bounds QP
  sequence.
- **Phase 5b — SQP NLP driver, cold (2–3 weeks).** `SqpAlgorithm`
  wired into the builder, l1-merit globalization, exact Hessian.
  Tests on CUTEst small-NLP subset. **Exit criterion:** convergence
  on a curated CUTEst subset (no warm start yet), comparable
  iteration counts to filterSQP / SNOPT on the same problems.
- **Phase 5c — SQP warm-start API + L-BFGS (1–2 weeks).** Extend
  `WarmStartIterateInitializer` for the working set; carry L-BFGS
  across solves; add an `SqpReOptimize` entry point parallel to
  `ReOptimizeTNLP`. **Exit criterion:** measured iteration-count
  drop on a synthetic MPC sequence and a synthetic parametric
  sequence, vs cold SQP and vs warm-started IPM.
- **Phase 5d — filter alternative + tuning (optional, 1–2 weeks).**
  Filter-SQP acceptor as a globalization option. Compare to l1-merit
  on the same benchmarks.

Total: 7–11 weeks of focused work, gated phase-by-phase. Phases 5a and
5b have value on their own (a standalone QP solver; a cold SQP NLP
solver that some users will prefer to IPM on smaller dense problems);
5c is where the warm-start payoff actually lands.

## 8. Risk

- **Maintenance.** Two solver paths is a permanent maintenance
  liability. Mitigation: keep the IPM as default, freeze SQP's contact
  surface with shared code at well-defined trait boundaries.
- **Hessian indefiniteness.** SQP-with-exact-Hessian QPs are indefinite
  on nonconvex NLPs; the null-space active-set method needs negative-
  curvature handling (Gould-Hribar-Nocedal). Damped BFGS sidesteps the
  issue but throws away exact-Hessian accuracy.
- **Warm-start failure modes.** A stale working set can produce an
  infeasible QP. Recovery path: fall back to a phase-1 LP for
  feasibility restoration. Adds code but is well-trodden territory.
- **Benchmark target.** No MPC, MINLP, or parametric workload sits in
  `benchmarks/` today. Phase 5c must ship with at least one such
  benchmark to make the iteration-count improvement reproducible.

## 9. Open questions for review

- **QP algorithm choice.** Null-space active-set (Gould-Hribar-Nocedal,
  handles indefinite Hessians) vs Goldfarb-Idnani (faster, convex-only
  — would require damped BFGS to guarantee PSD Hessians, losing exact-
  Hessian accuracy on nonconvex NLPs). Default proposal: null-space;
  open to Goldfarb-Idnani if L-BFGS is good enough on the target
  workloads.
- **Globalization default.** l1-merit (Han-Powell, simple, well-tested)
  vs filter (matches IPM, no penalty parameter, more code). Default
  proposal: l1-merit, with filter as an opt-in.
- **Working-set representation across re-solves.** Two viable encodings:
  per-index status enum vs index-set + sign-list. Status enum is
  simpler to serialize / restart; index-set is more compact for sparse
  active sets. Default proposal: status enum.
- **Sensitivity-corrector integration.** Should `pounce-sensitivity`
  call into `pounce-qp` for its corrector immediately (5a), or stay
  one-shot until 5c? Default proposal: stay one-shot in 5a, integrate
  in 5c so the parametric workload becomes a real test.
- **Benchmark workloads.** What MPC / parametric problem sets become
  the regression suite? `benchmarks/` has none today; a Phase 5c
  prerequisite is committing one. Open question: which.
- **Crate dependency direction.** `pounce-qp` is below `pounce-algorithm`
  in the dependency graph. Should `pounce-sensitivity` also depend on
  it (to do active-set corrector), or stay independent? Default
  proposal: depend, in 5c.

## 10. References

- Nocedal, Wright, *Numerical Optimization* (2nd ed.), §18 (SQP),
  §16 (QP).
- Gill, Murray, Saunders, "SNOPT: An SQP algorithm for large-scale
  constrained optimization", *SIAM Rev.* 47 (2005).
- Fletcher, Leyffer, "Nonlinear programming without a penalty function",
  *Math. Prog.* 91 (2002) — filter SQP.
- Byrd, Nocedal, Waltz, "KNITRO: An Integrated Package for Nonlinear
  Optimization" (2006) — the Active-Set/SLQP algorithm.
- Gould, Hribar, Nocedal, "On the solution of equality constrained
  quadratic programming problems arising in optimization", *SIAM J.
  Sci. Comput.* 23 (2001).
- Goldfarb, Idnani, "A numerically stable dual method for solving
  strictly convex quadratic programs", *Math. Prog.* 27 (1983).
- Ferreau, Kirches, Potschka, Bock, Diehl, "qpOASES: a parametric
  active-set algorithm for quadratic programming", *Math. Prog.
  Comp.* 6 (2014) — the canonical warm-started QP solver for MPC.
- `future-work-roadmap.md:185-206` — the C1 entry this note operationalizes.

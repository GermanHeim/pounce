# Auxiliary equality preprocessing — port of ripopt #32

Status: design / not yet implemented.
Tracking issue: <https://github.com/jkitchin/pounce/issues/53>
Upstream reference: <https://github.com/jkitchin/ripopt/pull/32>

## Context

ripopt PR #32 (merged, by D. Bernal) added ~6,200 lines of structural
preprocessing for nonlinear-programming equality subsystems: incidence graph
→ Hopcroft-Karp matching → Dulmage-Mendelsohn partition → Tarjan SCC /
block-triangular form → per-block solve → reduced problem → recursive
standard presolve → postsolve with multiplier recovery. On
`gaslib11_steady.nl` the reduction goes from 204/200 vars/cons to 140/136,
and on the `tutorial_flow_density` fixtures the IPM finishes in 0 iterations
instead of 6–7.

pounce is the sibling Rust IPM solver. `crates/pounce-presolve` already has
a clean `TNLP`-wrapper presolve pipeline:

- **Phase 1** — Andersen-style bound tightening against linear rows.
- **Phase 2** — redundant-linear-row removal.
- **Phase 3** — structural LICQ check on surviving equalities.
- **Phase 4** — bound-multiplier warm-start hints.
- **Phase 5** — sensitivity-aware metadata passthrough.

What's missing is the *structural* layer — no incidence graph, no matching,
no DM/BTF, no per-block solve, no postsolve multiplier recovery. This doc
specifies how we port that layer in as a new Phase-0 pre-pass that runs
ahead of the existing Phases 1–5 in the same `PresolveTnlp`.

## Goals

1. Land structural equality-subsystem preprocessing in `pounce-presolve`,
   exposed behind options that default to **off** until benchmarks justify
   flipping the default.
2. Reuse pounce's existing `TNLP` wrapper machinery (row mapping,
   Jacobian/Hessian remap, metadata projection, scaling forwarding) rather
   than introducing a parallel `NlpProblem` style.
3. Preserve pounce's pattern of small, reviewable PRs and per-phase
   `dev-notes/` specs.
4. Match ripopt's correctness contracts: full-space residual check before
   accepting a reduction, postsolve multiplier recovery, and coupling-class
   gating.
5. Provide a tracking issue and benchmark fixtures so the work is auditable.

## Non-goals

- discopt-level presolve orchestrator, pass-delta protocol, Rust↔Python
  presolve handshake.
- tree-persistent bound tightening, MINLP reformulation.
- replacing or refactoring the existing Phases 1–5.
- linear-solver-level structural reordering (AMD/METIS choices in
  `pounce-linsol`).

## Module layout under `crates/pounce-presolve/src/`

Existing: `bound_tighten.rs`, `licq.rs`, `redundant.rs`, `options.rs`, `lib.rs`.

New (added incrementally, one per PR where possible):

| Module | Purpose | ripopt anchor |
|---|---|---|
| `incidence.rs` | `EqualityIncidence`: equality-row × variable bipartite adjacency from inner `eval_jac_g(Structure)` + `get_constraints_linearity` + bounds | `src/auxiliary_preprocessing.rs:2282-2318` |
| `matching.rs` | Hopcroft-Karp bipartite matching | `src/auxiliary_preprocessing.rs:2280-2318` |
| `dulmage_mendelsohn.rs` | DM partition into underdetermined / square-matched / overdetermined parts | `src/auxiliary_preprocessing.rs:2320-2413` |
| `components.rs` | weakly-connected component extraction | `src/auxiliary_preprocessing.rs:2416-2469` |
| `btf.rs` | Tarjan SCC + topological order → block-triangular form | `src/auxiliary_preprocessing.rs:2473-2552` |
| `coupling.rs` | `AuxiliaryCouplingClass` + classification via objective-gradient probe and inequality incidence | `src/auxiliary_preprocessing.rs:39-59, 1642-1687` |
| `block_solve.rs` | Lightweight damped Newton for 1–8-dim square blocks; `BlockSolver` trait for the fallback path | `src/auxiliary_preprocessing.rs:1078-1182` |
| `reduction_frame.rs` | `ReductionFrame { var_map, row_map, fixed_values, multiplier_recovery }` + `ReductionStack` for nested layers; dense LU stationarity solve for recovered multipliers | `src/reduction_frame.rs:101-231` |
| `auxiliary.rs` | Orchestrator: `find_presolve_candidates`, `find_postsolve_candidates`, `solve_auxiliary_blocks`, build a reduced view, expose diagnostics | `src/auxiliary_preprocessing.rs` (top-level) |
| `diagnostics.rs` | `AuxiliaryPreprocessingDiagnostics` (timings, counts, rejection reasons, residuals) | `src/result.rs:13-65` |

## Integration into `PresolveTnlp`

`PresolveTnlp::ensure_init` currently runs Phases 1→5 once. We add a
**Phase 0** that runs first:

1. Build incidence from the inner `eval_jac_g(Structure)` probe.
2. Find candidate blocks (presolve and postsolve sets).
3. Solve auxiliary blocks against the inner TNLP probe point.
4. Fix the solved variables (clamp `x_l[i] = x_u[i] = value`) and extend
   the existing `row_kept_inner` mask to drop the solved equality rows
   *before* Phases 1–5 execute.
5. Push a `ReductionFrame` onto a `ReductionStack` stored in
   `PresolveState`.

The downstream Phases 1–5 already handle row dropping, Jacobian/Hessian
remap, metadata projection, and scaling. The auxiliary pass only needs to:

- extend `rows_kept_inner` (and, by transit, `jac_kept_idx` / `jac_irow_outer`
  / `jac_jcol_outer` / `g_l` / `g_u`);
- maintain the reduction stack;
- contribute time/counter fields to `AuxiliaryPreprocessingDiagnostics`.

`finalize_solution` runs the existing inner-expansion first, then
`ReductionFrame::unmap_solution_with_options` on top: fill back the fixed
variables, recover their `z` from stationarity, and recover the dropped-row
multipliers via dense LU on the stationarity block.

This keeps the surface change minimal: one new pre-phase in `lib.rs`, one
new field on `PresolveState`, and full reuse of the existing remapping
arrays.

## Coupling policy (initial)

Mirror ripopt's defaults:

- **PureEquality** blocks: eliminated in presolve.
- **ObjectiveCoupled** blocks: kept on the full-space path in presolve;
  eligible only as postsolve candidates.
- **InequalityCoupled** and **ObjectiveAndInequalityCoupled** blocks: not
  eliminated in v1 (matches ripopt's conservative default).

Gate behind option `presolve_auxiliary_coupling = none | safe | aggressive`
where `safe` ≡ PureEquality only and `aggressive` ≡ PureEquality plus
ObjectiveCoupled-postsolve. The master switch `presolve_auxiliary` defaults
to `false`.

## Options to add (`options.rs`)

| Option | Default | Notes |
|---|---|---|
| `presolve_auxiliary` (bool) | `false` | Master switch for the new pass. |
| `presolve_auxiliary_tol` (Number) | `1e-8` | Residual tol for accepting a block solve. |
| `presolve_auxiliary_max_block_dim` (Index) | `8` | Lightweight Newton ceiling; larger blocks rejected until `BlockSolver` is wired. |
| `presolve_auxiliary_wall_time_fraction` (Number) | `0.1` | Time budget slice. |
| `presolve_auxiliary_coupling` (enum) | `safe` | `none` / `safe` / `aggressive`. |
| `presolve_auxiliary_diagnostics` (bool) | `false` | Emit the diagnostics struct via the existing diagnostics channel. |

## Phased rollout

One tracking issue, multiple PRs — mirrors ripopt's fork series
(#12–#21, #29–#34 in `bernalde/ripopt`). Each PR lands behind the master
switch so it stays a no-op for users until PR 12.

1. **PR 1 — scaffolding**: empty modules, options registered, no-op
   orchestrator, diagnostics struct stub.
2. **PR 2 — incidence + matching**: `incidence.rs`, `matching.rs` with unit
   tests for square / over / under cases.
3. **PR 3 — DM + components**: `dulmage_mendelsohn.rs`, `components.rs`
   with property tests on small synthetic graphs.
4. **PR 4 — BTF**: Tarjan SCC + topological order; tests on chain / cycle
   / star graphs.
5. **PR 5 — coupling classification**: `coupling.rs` with objective-gradient
   probe and inequality incidence; rejection-reason enum populated.
6. **PR 6 — block solve (Newton only)**: dense LU + damped Newton for
   ≤8-dim blocks; `BlockSolver` trait for the larger-block fallback.
7. **PR 7 — reduction frame + postsolve recovery**: `reduction_frame.rs`
   with multiplier recovery via stationarity LU; round-trip unit tests.
8. **PR 8 — orchestrator wiring into `PresolveTnlp`**: integrate the
   Phase-0 pre-pass; preserve Phases 1–5; full-space residual check before
   accepting any reduction.
9. **PR 9 — diagnostics + docs**: populate the diagnostics struct; add
   `docs/src/auxiliary-presolve.md`; update `docs/src/options.md`.
10. **PR 10 — benchmarks**: `benchmarks/preprocessing/run_preprocessing_benchmark.py`
    modeled on ripopt's; copy `tutorial_flow_density{,_perturbed}.nl`
    fixtures from ripopt with attribution.
11. **PR 11 — IPM block-solve fallback**: implement `BlockSolver` for
    blocks > 8-dim by re-entering `pounce-algorithm`'s IPM. Kept last
    because it introduces a workspace-internal dep that may need feature
    gating.
12. **PR 12 — default flip (deferred)**: only after benchmark evidence;
    default `presolve_auxiliary = true` for `coupling = safe`.

## Reused code (do not re-invent)

- `pounce_presolve::PresolveTnlp` state machine and Jacobian / Hessian /
  metadata remap helpers (`crates/pounce-presolve/src/lib.rs`).
- `pounce_presolve::licq::licq_check` for any post-reduction LICQ
  verification on the surviving equality block.
- `pounce_presolve::redundant::find_redundant_rows` — auxiliary reduction
  may expose new redundant rows for Phase 2 to catch.
- `pounce_nlp::tnlp::TNLP` for the wrapper trait.
- `pounce_common::reg_options::RegisteredOptions` and the `OptionsList`
  plumbing for option registration.
- Dense LU in `reduction_frame.rs`: prefer an existing utility in
  `pounce-linalg` if one exists; otherwise port ripopt's standalone
  Gaussian-elimination-with-partial-pivoting (the ripopt impl is
  intentionally self-contained, ~80 lines).

## Tests

Unit tests per module, matching the coverage in ripopt's
`auxiliary_preprocessing.rs:4250+`:

- Bipartite matching on square / non-square graphs.
- DM partition correctness on known textbook examples.
- BTF returns Tarjan SCC + correct topological order.
- Coupling classifier flags objective-coupled and inequality-coupled blocks
  correctly.
- Block-solve fallback triggers when Newton diverges.
- Round trip: reduce → solve → unmap reproduces the full-space KKT
  residual.

Integration tests (`tests/nl_integration.rs` style):

- Copy `tutorial_flow_density.nl` and `tutorial_flow_density_perturbed.nl`
  from ripopt; assert pounce solves them in 0 IPM iterations with
  `presolve_auxiliary=true`.
- `auxiliary_gate.nl` minimal fixture.

Regression guard: any problem where the auxiliary pass produces a worse
final KKT residual than the unreduced path is a test failure (the
full-space residual check lives in the orchestrator).

## Verification

After PR 8 lands:

```bash
cd /Users/jkitchin/projects/pounce
cargo test -p pounce-presolve
cargo test -p pounce-cli --test nl_integration
```

After PR 10:

```bash
python benchmarks/preprocessing/run_preprocessing_benchmark.py \
    --problem tutorial_flow_density.nl --compare on,off
cargo run -p pounce-cli -- benchmarks/gas/gaslib11_steady.nl \
    --option presolve_auxiliary=yes
```

Acceptance criteria:

- `tutorial_flow_density{,_perturbed}.nl`: 0 IPM iterations with the master
  switch on; same final objective vs the no-presolve path (to
  `auxiliary_tol`).
- `gaslib11_steady.nl`: model reduces and the IPM still converges to the
  same objective; iteration count within ±2 of the no-presolve path.
- No regression on the existing pounce-presolve integration tests.

## Risks and open questions

- **Block-solve recursion**: ripopt re-enters its own IPM for non-tiny
  blocks. In pounce the cleanest path is a `BlockSolver` trait implemented
  in `pounce-algorithm` and injected by the CLI / Python entry points.
  Deferring to PR 11 keeps PRs 1–10 free of upward workspace deps.
- **Coupling-policy correctness**: postsolve recovery for objective-coupled
  blocks needs careful KKT bookkeeping. Cover with property tests against
  the full-space solution.
- **Dense LU sizing**: stationarity recovery does an `(m_eq × m_eq)` dense
  solve where `m_eq` is the number of eliminated equalities. For very large
  blocks this is wasteful; flag a fallback to sparse LU if `m_eq > 200`
  (track as a follow-up issue, not a blocker).
- **Fixture licensing**: the ripopt fixtures are MIT-licensed; carry the
  file headers / attribution when copying.

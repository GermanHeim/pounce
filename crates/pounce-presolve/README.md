# pounce-presolve

Algorithmic NLP preprocessing for POUNCE, exposed as a composable
[`TNLP`](../pounce-nlp) wrapper. Wraps a user TNLP and applies presolve
passes before the IPM ever sees the problem; restores eliminated
quantities (multipliers, dropped-row Lagrange entries) on the way out
of `finalize_solution`.

Internal crate. Off by default — enabled by `presolve = yes` in the
options table (`SolverOptions::presolve = true`).

## Phases

| Phase | What it does | Option keys | Tracking issue |
|---|---|---|---|
| **Phase 0 — auxiliary-equality preprocessing** | Hopcroft-Karp matching → Dulmage-Mendelsohn partition → Tarjan SCC → block-triangular reduction → damped-Newton block solver. Eliminates variables fixed by square equality blocks and drops the corresponding rows. Multipliers for dropped rows are recovered in `finalize_solution`. | `presolve_auxiliary`, `presolve_auxiliary_coupling`, `presolve_auxiliary_diagnostics` | [#53](https://github.com/jkitchin/pounce/issues/53) |
| **Phase 1 — Andersen-style bound tightening** | Propagates implied bounds from linear-row activity onto variable boxes. | `presolve_bound_tightening`, `presolve_max_passes` | [#20](https://github.com/jkitchin/pounce/issues/20) |
| **Phase 1b — FBBT** | Feasibility-Based Bound Tightening via interval arithmetic over the constraint expression DAGs. No-op unless the inner TNLP supplies an `ExpressionProvider`. | `presolve_fbbt`, `fbbt_tol`, `fbbt_max_iter`, `fbbt_max_constraints` | [#62](https://github.com/jkitchin/pounce/issues/62) |
| **Phase 2 — redundant-row removal** | Drops linear rows whose activity interval is already implied by the (post-tightening) variable box. | `presolve_redundant_constraint_removal` | [#20](https://github.com/jkitchin/pounce/issues/20) |
| **Phase 3 — LICQ structural check** | Flags rank-deficient equality Jacobians. Exposed via `PresolveTnlp::licq_verdict`. | `presolve_licq_check`, `presolve_licq_action` | [#20](https://github.com/jkitchin/pounce/issues/20) |
| **Phase 4 — bound-multiplier warm start** | For variables whose bounds Phase 1 moved strictly inward, seed `z_l`/`z_u` at `bound_mult_init_val`. Overlay only — user-supplied values always win. | `presolve_warm_z_bounds` | [#20](https://github.com/jkitchin/pounce/issues/20) |
| **Phase 5 — sensitivity-aware passthrough** | Projects user-supplied constraint metadata and scaling through the row reduction on the way in, and expands outer → inner on the way out in `finalize_metadata`. | — | [#20](https://github.com/jkitchin/pounce/issues/20) |

The phases compose: each builds on the bounds and row-keep mask the
previous phases left behind. Phase 0 has a defence-in-depth rollback:
if Phase 1 later derives a contradiction on the rows Phase 0 left
behind, the elimination is rolled back for that solve and Phase 1
re-runs on the un-filtered linear rows.

## Public surface

```rust
use pounce_presolve::{wrap_with_presolve, wrap_with_presolve_provider, PresolveOptions};

// Plain wrap — FBBT becomes a silent no-op without an ExpressionProvider.
let wrapped = wrap_with_presolve(inner_tnlp, opts)?;

// Or, with structural-expression access for FBBT:
let wrapped = wrap_with_presolve_provider(inner_tnlp, expr_provider, opts)?;
```

For inspection (used by integration tests + diagnostics):
`PresolveTnlp::{tighten_report, fbbt_report, n_dropped_rows, licq_verdict, z_warm_starts, auxiliary_diagnostics, cached_bounds}`.

## Acknowledgments

Phase 0 (the auxiliary-equality pass) is a port of
[ripopt PR #32](https://github.com/jkitchin/ripopt/pull/32) by
**David Bernal Neira**
([@bernalde](https://github.com/bernalde)). The
`tutorial_flow_density{,_perturbed}.nl` and `gaslib11_steady.nl`
fixtures originate from that PR. See
[`docs/src/auxiliary-presolve.md`](../../docs/src/auxiliary-presolve.md)
for the user-facing write-up and
[`dev-notes/auxiliary-equality-preprocessing.md`](../../dev-notes/auxiliary-equality-preprocessing.md)
for the design note.

## License

EPL-2.0. See [LICENSE](../../LICENSE) at the repo root.

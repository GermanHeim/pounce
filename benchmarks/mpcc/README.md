# MPCC benchmark (`benchmarks/mpcc/`)

The Gate 0 evidence harness for
[gh#794](https://github.com/jkitchin/pounce/issues/794), which is the
first gate of the phase-changing-dynamics workstream in
[gh#776](https://github.com/jkitchin/pounce/issues/776).

Its job is narrow and worth stating first, because the temptation is to
widen it: **decide whether POUNCE has a supported route for small MPCCs,
and draw the boundary of that route, before anyone writes a flash, a
tray, or a column model.** gh#794's last acceptance criterion is that no
such work begins from this issue. Nothing here is a process model and
nothing here should become one.

The gate decision itself is written up in
[`dev-notes/mpcc-gate0-report.md`](../../dev-notes/mpcc-gate0-report.md).

## Running it

Needs the built Python extension (`make python-ext` from the repo root)
and SciPy. From `benchmarks/`:

```sh
python -m mpcc.selftest        # no solver needed; must pass before anything else
python -m mpcc.run --smoke     # deterministic asserted subset (~10 s)
python -m mpcc.run --full      # the full configuration, writes the report
python -m mpcc.run --write-manifest    # regenerate manifest.json after a corpus edit
```

or, from the repo root, `make -C benchmarks mpcc-selftest / mpcc-smoke /
mpcc-run`.

Results land in `results-<mode>.json` beside the harness, with a
markdown rendering next to them. Both are regenerated and gitignored;
`manifest.json` and `schema.json` are tracked.

Neither target is wired into CI, which matches how every other
benchmark harness in this tree is treated — `warmstart/` included. The
smoke subset is asserted and fast enough to be wired in later if the
MPCC route becomes something the solver is expected to keep working; at
Gate 0 it is evidence for a decision, not a regression guard.

## What is being compared

A *case* is a source MPCC. A *lowering* turns it into a smooth NLP. A
*route* is a lowering plus solver options plus, for the continuation
routes, a warm-start level. A *control* is a kill switch that disables
one solver mechanism. One cell of the matrix is
`(case, scaling, start, route, control)`.

| route | lowering | what it is |
|---|---|---|
| `direct` | `G*H <= 0` | the direct POUNCE NLP formulation |
| `ncp_eq` | `G*H = 0` | exact product / NCP equality, ordinary POUNCE |
| `ncp_eq_l1` | `G*H = 0` | + `l1_exact_penalty_barrier=yes` |
| `ncp_eq_l1_fallback` | `G*H = 0` | + `l1_fallback_on_restoration_failure=yes` |
| `ncp_eq_auto_l1` | `G*H = 0` | + presolve LICQ check routing to `auto_l1` |
| `scholtes_cold` | `G*H <= tau` | continuation, independent cold solves |
| `scholtes_warm_primal` | `G*H <= tau` | continuation, primal-only warm starts |
| `scholtes_warm_full` | `G*H <= tau` | continuation, full primal/dual/barrier warm starts |

Controls: `no_acceptable` (`acceptable_iter=0`), `no_scaling`
(`nlp_scaling_method=none`), `upstream_heuristics` (the three POUNCE-only
convergence mechanisms zeroed, which their own documentation describes
as restoring bit-for-bit upstream Ipopt behaviour), and `no_presolve`.
gh#794 requires these to be run before a failure is attributed to a new
mechanism, and the report lists only the cells whose verdict actually
moves under one.

## The ladder

Eleven cases covering the six classes gh#794 requires. Every expected
value is derived by hand in the case factory's docstring **and**
recomputed independently by `oracle.enumerate_branches`, which solves
every complementarity branch as a smooth program with SciPy;
`selftest` fails if the two disagree.

| case | class | what it is for |
|---|---|---|
| `regular_strict` | regular | strict complementarity, MPCC-LICQ holds; every route should solve it |
| `biactive_positive` | biactive | biactive with both MPCC multipliers strictly positive — benign |
| `ralph1` | degenerate | MPCC-LICQ fails; **no S-stationary point exists** |
| `ctrap` | degenerate | the origin is C-stationary, has zero residuals, and is not a local minimiser |
| `infeasible_pair` | infeasible | provably infeasible, with the relaxation's crossover known exactly (`tau = 1/4`) |
| `selector_theta_{025,050,075}` | selector | one-hot Boolean selector; the optimal branch flips at `theta = 1/2`, where it ties |
| `ralph2` | macmpec | the relaxed optimum is `-2*tau`, **below** the MPCC's own optimum |
| `scholtes4` | macmpec | two active rows; the solution is M- but not S-stationary |
| `qpec_small` | macmpec | two pairs (one strict, one biactive) — the only case that catches a mis-indexed product row |

## What a record contains, and the one rule about reading it

**Source-level quantities and POUNCE's NLP diagnostics never share a
column.** A lowering's NLP residual is a residual of a different
problem: on a Scholtes stage `final_constr_viol` measures `G*H <= tau`,
which points nowhere near the MPCC satisfy. So every record carries

* a `source` block — objective, row and bound violations, the sign
  violations of `G, H >= 0`, and the complementarity product
  `|G_i H_i|` — computed from the source MPCC at the returned point;
* a `stationarity` block — the MPCC class (S / M / C / W), the residual
  of every class tried, the multiplier vector, the biactive set, and
  MPCC-LICQ; and
* an `nlp` block — POUNCE's own scaled and unscaled KKT diagnostics.

plus iterations, outer stages, accepted/rejected stages, warm-start and
restart level, `mu` in and out with the reason each moved, the
relaxation parameter and why it moved, restoration counters, the
inertia-correction count read off the iteration table, and the
active-pair regime at every stage. `schema.json` is the contract.

Three fields are deliberately null rather than zero. `filter_resets` is
not emitted by POUNCE and is therefore *unmeasured*, which is a
different claim from measured-and-zero. `discopt.commit` is null with a
reason, so "no DiscOpt comparison was run" is distinguishable from "one
was run and not stamped". `ccopt.comparison_run` is false with the pin a
comparison would have used.

## Why the classifier searches the multiplier set

MPCC stationarity is a property of the multiplier *set*, not of one
multiplier vector. On `scholtes4` the set at the solution is a line: its
least-squares point gives `nu = w = -1` (C-stationary), while
`lambda = (1/4, 3/4)` gives `nu = 0` (M-stationary). A classifier that
read one least-squares vector would report **C** for a point that is
genuinely M-stationary. `stationarity.classify` therefore poses each
class as a bounded-least-squares *feasibility* question over an explicit
enumeration of the sign branches the class admits.

`selftest` checks the discriminations by name — `ralph1` and
`scholtes4` must come back M **and refuse S**, `ctrap`'s origin C **and
refuse M** — because a classifier that always answered "W" would satisfy
every residual check in the harness and be worthless.

## Limits of this corpus

Stated here because Gate 1 inherits them:

* **Every complementarity pair is affine and every objective and row is
  quadratic.** That is what makes the product row's derivatives exact
  and machine-checkable, and it means nothing here says anything about a
  *nonlinear* complementarity function.
* **Nothing is larger than three variables and two pairs.** No statement
  about conditioning, fill, or scaling at process-model size can be read
  off these numbers.
* **The scaling legs span six orders**, which is mild next to a real
  flash, and already enough to move a route's verdict.
* **No dynamics, no thermodynamics, no phase equilibrium.** The pair
  semantics fields in the manifest are there so Gate 1 has somewhere to
  put units and branch meanings; at Gate 0 they are documentation.

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
| **`scholtes_then_ncp`** | `G*H <= tau`, then `G*H = 0` | **the supported route**: continuation to locate the branch, then one exact-product solve seeded from it |

Controls: `no_acceptable` (`acceptable_iter=0`), `no_scaling`
(`nlp_scaling_method=none`), `upstream_heuristics` (the three POUNCE-only
convergence mechanisms zeroed, which their own documentation describes
as restoring bit-for-bit upstream Ipopt behaviour), and `no_presolve`.
gh#794 requires these to be run before a failure is attributed to a new
mechanism, and the report lists only the cells whose verdict actually
moves under one.

The composition is the route the Gate 0 report recommends, and neither
half is sufficient alone: the continuation always converges but its
answer is only ever feasible for `G*H <= tau`, and the exact-product
solve returns an MPCC-feasible point but fails from a cold start where a
pair is biactive. The finishing solve takes the **point and nothing
else** — the relaxed problem's duals are duals of a different constraint
(the product row is an active inequality at `G*H <= tau` and an equality
at `G*H = 0`), and seeding them measurably sent a three-variable finish
into a thrash that had not returned in 25 s.

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

## Source-level validation

Each of the six benchmark classes has its own validation function in
`validate.py`, applied to every case of that class: strict
complementarity for `regular`, a genuinely biactive pair for
`biactive`, source feasibility for `degenerate`, *in*feasibility for
`infeasible`, branch commitment for `selector`, and a signed gap
against the pinned optimum for `macmpec`. A few cases add their own on
top — which branch a selector chose, whether `ctrap` stopped at the
C-stationary origin.

They read the source MPCC and nothing else: no solver status, no NLP
residual, no lowering. A route can converge its reformulation and still
be wrong about the model, and only a check written against the model
says so. `selftest` requires every class to have one and requires each
to pass at its own cases' expected solutions — a validator no expected
point satisfies is describing a different class than the one it is
registered under.

## Comparability of the arms

All eight routes are the same POUNCE build solving through the same
linear algebra: one process, one extension module, the default linear
solver, no per-route backend selection. So no comparison in the result
file crosses a linear-algebra boundary and none needs the disclosure
gh#794 asks for. The optional CCOpt comparison would cross one, which is
part of why it is pinned rather than merely named.

## The complementarity tolerance floor

`G*H` is quadratically flat at the corner, so a solve that drives the
product to `eps` pins each side of the pair only to `sqrt(eps)`. Two
things follow, and both are visible throughout the results:

* **an objective is only as good as `sqrt(tol)`.** At the pinned
  `tol = 1e-8` that is `1e-4`, whatever route produced it — `ralph1` at
  a residual of `2.6e-09` sits `5.07e-05` below `f*`, against
  `sqrt(2.6e-09) = 5.1e-05`.
* **so is the stationarity class.** The MPCC multipliers a biactive pair
  generates at that residual are themselves `O(sqrt(eps))`, and S- and
  C-stationarity differ only in their signs.

`spec.pair_activity` is built on the same fact: membership is judged
against `max(ACTIVE_TOL, sqrt(tol) * term_scale)`, not a fixed
tolerance. A fixed `1e-6` classified twelve points that had reached the
optimum to nine digits as `none` — not even weakly stationary — and the
triage table read them as solver defects.

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

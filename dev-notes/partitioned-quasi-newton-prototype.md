# Partitioned quasi-Newton: a per-constraint element prototype, and what it measured

> Prototype and measurement for the question "can partitioned quasi-Newton
> (Asprion, Chinellato & Guzzella, *J. Appl. Math.* 2014,
> doi:10.1155/2014/341716) close the exact-vs-L-BFGS gap on
> direct-collocation trajectory optimization?"
>
> **Headline. Two decompositions were built and measured.
> *Per-constraint* elements scale the wrong way — they converge on the
> smaller mesh and fail one refinement later — and that is disqualifying,
> since scaling is the whole reason the gap exists. *Per-primal-block*
> elements, the paper's own partition, do the opposite: iteration count is
> nearly flat across a 4× problem growth (276 → 252 → 338) where
> limited-memory goes 126 → 246 → does-not-converge. It is not yet a win —
> it still fails at N = 320, and never beats limited-memory on wall time
> where both work — but it fails in restoration, on a downgrade this
> prototype chose, rather than in the curvature model. That is a fixable
> next step, and it is the first result here that points forward.**

> **Every number below predates the gh#818 line-search split reaching this
> mode.** `alpha_red_factor_min` now resolves to `0.05` for `partitioned`
> as well as `limited-memory` — quasi-Newton elements are the same update
> on a finer decomposition, so the criterion that put L-BFGS on
> safeguarded quadratic interpolation puts these there too — and that is a
> trajectory change on the very path this note measures. Nothing here has
> been re-measured against it, so treat the iteration counts as a record of
> what the decomposition did under the fixed geometric sequence, not as
> current. `scripts/sweep-fixtures.sh` does not cover the gap: it runs the
> `exact` and `lbfgs` legs only, so no partitioned leg is swept either way.
> If this mode is ever picked back up, re-measure first — interpolation is
> aimed squarely at the weakness these numbers document, so it may well
> move them in the mode's favour.

## Why this was worth building

`benchmarks/large_scale` `laptime` is a minimum-lap-time Radau collocation
model added for gh#698 — "a 60 000-variable collocation model with analytic
Jacobians and no analytic Hessian". At `N = 160` (n = 9 294, m = 9 934):

| leg | iterations | wall | objective |
|---|---|---|---|
| exact | 30 | 2.8 s | 65.371107 |
| limited-memory | 246 | 27.2 s | 65.370540 |

An 8.2× iteration gap and a 9.6× wall-clock gap, on a model whose user
cannot supply the exact Hessian. That gap is the prize.

Two structural levers in POUNCE are also gated on having a real sparse
Hessian, and are therefore unavailable to exactly the users who need them
most:

* `LowRankAugSystemSolver` substitutes a **diagonal** `B0` for `W`, so the
  factorization never sees the model's block structure and
  `OrderingMethod::External` has only the Jacobian to work with;
* the Schur path (`set_kkt_schur_block`, pounce#180 item 2) is **silently
  ignored** under limited-memory — `alg_builder.rs` takes the `is_lbfgs`
  branch first, and the Schur branch is an `else if` gated on the
  exact-Hessian feral path.

A quasi-Newton scheme that publishes a genuine sparse `SymTMatrix` unlocks
both. That is what this prototype does.

## What was built

`crates/pounce-algorithm/src/hess/partitioned_quasi_newton.rs`, reachable
as `hessian_approximation=partitioned`.

One small dense symmetric block `B_e` per **element function** — the
objective, and each constraint row — updated from that element's own
curvature pair, then scatter-added into a `SymTMatrix`:

```
W = ∇²f + Σ_j (y_c)_j ∇²c_j + Σ_j (y_d)_j ∇²d_j
```

Design choices worth recording:

* **Per constraint row, not per primal block.** Asprion et al. partition
  the Lagrangian by primal stage blocks; each block's target then moves as
  the multipliers move. Splitting per constraint gives each `B_e` a
  multiplier-independent target (`∇²c_j` is a property of the model), and
  needs no new user-facing structure hook — an element's support is a row
  of the Jacobian, whose pattern every TNLP already declares.
* **`y_e` costs nothing extra.** It is the change in a Jacobian row, and
  the limited-memory updater already caches the previous Jacobians
  (`last_jac_c` / `last_jac_d`).
* **SR1 by default.** An individual constraint is not convex, so damped
  BFGS would force each `∇²c_j` model PSD and then scale it by a
  multiplier of either sign. `partitioned_update_type=bfgs` is available
  for comparison.
* **Restoration is scoped out.** `run_inner_resto` downgrades
  `Partitioned` → `LimitedMemory`: the element table is built from the
  original NLP's Jacobian, whose row and column spaces the restoration
  sub-NLP does not share.
* **Bounded element size.** Elements wider than `partitioned_max_element`
  (default 64) degrade to a diagonal weak-secant approximation rather than
  being dropped.

Supporting change, worth keeping on its own merits:
`TNLPAdapter::objective_nonlinear_vars` reads the objective element's
support from `get_objective_variables_linearity` (a pounce extension
`pounce-nl` already implements). Without it the objective's support has to
come from the first `∇f`'s nonzeros — a **value-derived** pattern. On
`laptime`, which declares 321 objective gradient nonzeros, that fallback
captured 161.

## The oracle is the most valuable artifact here

`POUNCE_PARTITIONED_ORACLE=1` compares the assembled `W`, entry by entry,
against `cq.curr_exact_hessian()` at the same iterate with the same
`obj_factor` and the same multipliers. On any `.nl` model both are
available.

This is the only check in the module that reads a number the updater did
not produce. The unit tests pin the update formulas against themselves —
secant equation, weak secant condition, degenerate denominators — and a
self-consistently wrong assembly passes every one of them. It also reports
`max|extra-pattern|`: mass the per-constraint pattern carries where the
true Hessian is structurally zero, which must be ~0 and is a direct read
on whether the elements are inventing curvature.

It earned its keep immediately; every diagnosis below came from it.

## What it found

### 1. A zero-denominator NaN (fixed)

The SR1 safeguard `|den| < r‖s‖‖w‖` degenerates to `0 < 0` when `w` is
exactly zero — a **linear constraint row**, whose `y_e` is identically
zero, hits this on its first pair. The rank-1 term then divided 0 by 0 and
published a NaN Hessian; the IPM reported `Restoration Failed` with
nothing naming the cause. `sr1_skips_a_degenerate_denominator` covers it.

### 2. The opening model must not be zero (fixed)

Before any element has a pair there is no curvature anywhere. Publishing
the honest all-zero `W` hands the first KKT solve a `(1,1)` block with no
curvature and lets the inertia correction invent the scale: on
`simple.nl` the first step was `‖d‖ = 2.74` against exact's `0.666`.
Publishing `init_val · I` — the limited-memory path's own empty-history
model — makes iteration 1 match the exact path bit for bit.

### 3. The real finding: denominators, not data

With the NaN fixed, `laptime` still stalled — `inf_du` frozen at
**exactly 3.26e-03** for iteration after iteration, under *both* SR1 and
BFGS. Identical to three digits across two different update formulas
rules out the update formula.

The oracle and the per-element peak instrumentation together localized it:

| iter | max‖y_e‖/‖s_e‖ | max block delta | oracle max_abs_err | max\|extra-pattern\| |
|---|---|---|---|---|
| 2 | 26.3 | 2.3e6 | 5.5e5 | 2.0e6 |
| 3 | 27.7 | 1.2e8 | 1.2e7 | 3.5e7 |
| 4 | 33.0 | 1.7e8 | 3.2e6 | 2.0e6 |
| 5 | 29.5 | 1.7e8 | 6.7e5 | 2.5e4 |

The curvature **data** is healthy throughout — `‖y_e‖/‖s_e‖` stays at
9–33, the right order for this model, whose exact Hessian entries run
1e1–1e3. But single-update block changes reach **1.7e8**. O(10) secant
data producing 1e8 changes is a denominator problem, and both formulas
have one:

* SR1's `w wᵀ/wᵀs` is bounded only by `‖w‖/(r‖s‖)`, and `r = 1e-8`
  permits a correction 1e8× the implied curvature. Nocedal & Wright's
  `r ∈ [1e-8, 1e-4]` assumes a **trust region** absorbs the rest; there is
  none here, and the blocks are then multiplied by multipliers and summed.
* the BFGS path tested only `sᵀr > 0` — no relative floor at all, which is
  why it was *worse* (2.9e8). That test is safe in the limited-memory
  updater because its `s` is a whole primal step; restricted to one
  element's support the same quantity goes arbitrarily small.

### 4. Tightening the floors overshoots just as badly

Setting both floors to `1e-4` — the conservative end of the literature
range — rejected **9 949 of 9 950 updates**. Element supports here are
small (`k ≈ 9`) and `w` is routinely near-orthogonal to `s`, so the blocks
never learned anything and `W` read `0.14` where the exact Hessian read
`2.1e3`.

So the denominator test is a *direction* test and cannot be the magnitude
control. The magnitude control added instead is
`partitioned_curvature_cap`: a bound on one update's size as a multiple of
the curvature that element's own secant pair implies (`‖y_e‖/‖s_e‖`) —
i.e. a bound in the units of the thing being modelled.

### 5. The magnitude cap makes things worse, non-monotonically

`partitioned_curvature_cap` bounds one update's size as a multiple of the
curvature that element's own secant pair implies. It does bound it. It
also makes the solver worse at every finite setting tried. `laptime`,
`N = 80`, `max_iter = 1200`, true optimum 65.462928:

| cap | status | iters | wall | objective |
|---|---|---|---|---|
| 1e1 | ErrorInStepComputation | 1071 | 179 s | 65.518586 |
| 1e2 | MaxIter | 1200 | 214 s | 67.202124 |
| 1e6 | MaxIter | 1200 | 232 s | 80.398129 |
| **off** | **Optimal** | **559** | **50 s** | **65.462802** |

`1e6` worse than both `1e1` and off is the shape of the result: this is
not "less capping is better". Rejection is **selective** — it drops
precisely the elements whose curvature is moving fastest, leaving those
blocks stale while their neighbours update. The assembled `W` is then
internally inconsistent, and that costs more than a uniformly noisy but
coherent model. The cap ships **off**.

### 6. Two things this note got wrong first time, and how

Both were errors of inference from measurements that were themselves
fine, and both are worth keeping visible.

**Selecting a default on a truncated run.** The first cap sweep ran 400
iterations over `{1e1, 1e2, 1e3, 1e4}` and never tested *off*. At a fixed
truncation nothing has converged, so the configurations got ranked by
interim objective — which is not a convergence measure. `1e1` won that
ranking precisely because it suppresses the curvature model hardest and
therefore creeps most smoothly, and it was shipped as the default. It is
in fact the worst of the four, and the setting that actually converges was
outside the sweep entirely.

**Reading the oracle as a quality metric.** `rel_fro` was treated as a
proxy for solver health. It is not one. The configuration that converges
(uncapped) is the one whose blocks reach 1e8 and whose `rel_fro` runs
1e3; the configurations with a well-behaved `rel_fro ≈ 1` all stall. A
Hessian model can be far from exact in Frobenius norm and still drive
convergence, because the inertia correction regularizes what it is handed.
The oracle is excellent at finding **defects** — it found the NaN and the
denominator blow-up, neither of which any self-comparison would have
caught — and it does not rank working configurations. Use it as a bug
detector, not a scoreboard.

## The measurement

`scripts/partitioned-qn-sweep.sh` and direct runs, `max_iter = 1200`,
cap off (the shipping default). Legs run at their own default barrier
strategy, which is what a user actually gets — `limited-memory` switches
`mu_strategy` to adaptive on its own, exact and partitioned stay monotone.

**N = 80** (n = 4 654, m = 4 974), true optimum 65.462928:

| leg | status | iters | wall | objective |
|---|---|---|---|---|
| exact | Optimal | 29 | 1.4 s | 65.462928 |
| lbfgs | Optimal | 126 | 6.7 s | 65.462928 |
| lbfgs-monotone | Acceptable | 155 | 7.8 s | 65.462928 |
| **partitioned-sr1** | **Optimal** | **559** | **50.3 s** | **65.462802** |
| partitioned-bfgs | MaxIter | 1200 | 250.9 s | 89.285871 |

**N = 160** (n = 9 294, m = 9 934), true optimum 65.371107:

| leg | status | iters | wall | objective |
|---|---|---|---|---|
| exact | Optimal | 30 | 4.4 s | 65.371107 |
| lbfgs | Optimal | 246 | 40.2 s | 65.370540 |
| lbfgs-monotone | Optimal | 234 | 34.7 s | 65.370589 |
| partitioned-sr1 | **MaxIter** | 1200 | 458.7 s | 69.611866 |
| partitioned-bfgs | **MaxIter** | 1200 | 529.7 s | 89.289627 |

So: per-constraint SR1 **does** converge, on the smaller mesh, to the
right answer — at 4.4× the iterations and 7.5× the wall of the
limited-memory path it was meant to beat. One mesh refinement later it
does not converge at all, while limited-memory still does.

**It scales the wrong way, which is the disqualifying property.** The
whole motivation was the models where limited-memory degrades with mesh
refinement (`benchmarks/large_scale/README.md`: 71 → 144 → 276 → failure);
a replacement that degrades *faster* is not a replacement. Damped BFGS
elements never converge at either size.

## Why — and why it is not a coding defect## Why — and why it is not a coding defect

The implementation is validated where validation is possible.
`crates/pounce-wasm/tests/simple.nl`, whose constraint Hessian is the
constant `2I` on a 2-variable support, converges under `partitioned` in
**10 iterations with the same objective to every printed digit as the
exact path**. SR1 recovers a constant element Hessian exactly from one
pair, which is the textbook result.

What does not survive the jump to `laptime` is statistical, not
algorithmic. Determining a `k × k` element block needs `k` independent
curvature directions. Every iteration supplies each element exactly
**one** — the restriction of the single primal step to its support — and
those directions are strongly correlated across iterations, because they
are all Newton-ish steps toward the same solution. With ~5 000 elements of
`k ≈ 9`, and steps that shrink as the solve converges, most blocks never
accumulate enough independent directions to be determined. The pattern
count makes the same point from the other side: the per-constraint pattern
is `⋃_j supp(∇g_j) ⊗ supp(∇g_j)` = **146 267** nonzeros against the true
Hessian's **28 000** — the decomposition is over-parameterized by 5.2×
relative to what it is trying to learn.

That is a property of *per-constraint* elements. It is also, read
backwards, an argument for what the paper actually does: Asprion et al.
partition by **primal stage blocks** — far fewer, larger blocks, each
receiving the whole step's information every iteration, at the cost of a
multiplier-dependent target.

## The per-primal-block partition: the follow-up, measured

`partitioned_elements=blocks` implements the paper's own partition — one
element per contiguous block of primal variables, modelling the
**Lagrangian's** restriction to that block with damped BFGS. The element
gradient is the Lagrangian gradient (`∇f + J_cᵀy_c + J_dᵀy_d`) restricted,
formed once per call by the same convention the limited-memory updater
uses, and the assembly weight is 1 because the multiplier is already
inside the modelled function.

`laptime` has `n_x = 14` states, `d = 3` Radau points and `n_u = 2`
controls, so one stage contributes `14·4 + 2 = 58` variables.
`partitioned_block_size` sweep at N = 80 and N = 160:

| block size | N=80 | N=160 |
|---|---|---|
| 16 | MaxIter 600 | MaxIter 600 |
| 58 (one stage) | ErrorInStepComputation, 525 | MaxIter 662 |
| **116 (two stages)** | **Acceptable, 276** | **Optimal, 252** |

One stage per block is **not** the right width, and the reason is
structural: a collocation defect couples stage `k` to stage `k+1`, so
58-wide blocks capture none of the inter-stage coupling. Two stages per
block capture half of it, and that is the difference between failing and
converging to the exact objective to 12 digits.

### Iteration scaling — the property that was actually being chased

| mesh | n | exact | lbfgs | blocks-116 |
|---|---|---|---|---|
| N=80 | 4 654 | 29 | 126 | 276 (acceptable) |
| N=160 | 9 294 | 30 | 246 | **252 (optimal)** |
| N=320 | 18 574 | 57 | **1200, DNF** | 338, DNF |

Over a 4× growth in problem size: exact 1.97×, **blocks 1.22×**, lbfgs
**> 9.5×**. The block partition has the flat iteration scaling that
limited-memory lacks, which is the entire property the exercise was after.
At N = 320 limited-memory does not converge at all and the block partition
gets within 1e-4 of feasibility before failing.

### But it still fails at N=320, and the blocker is not the curvature model

The N = 320 block run converges cleanly to `inf_pr ≈ 3e-4` around
iteration 240, then **enters restoration and grinds there** to a
local-infeasibility exit at 338 iterations. Restoration under
`Partitioned` downgrades to the limited-memory updater — a scoping
decision taken in this prototype because the element table is built from
the original NLP's Jacobian, whose index spaces the restoration sub-NLP
does not share. So the binding constraint at N = 320 is the *restoration*
Hessian, not the block model that got it there. That is a fixable,
identified next step and not a property of the method.

### And the linear algebra becomes the pressure

Per-iteration cost against exact: 1.5× at N = 160 (0.222 s vs 0.148 s),
3.3× at N = 320 (0.72 s vs 0.221 s). The block pattern is ~19× denser than
the true Hessian — 116-wide dense blocks against a Hessian that is sparse
inside them.

**This is where the original `OrderingMethod::External` premise finally
becomes correct.** Under limited-memory there was no Hessian structure to
exploit, because the low-rank path hands the factorization a diagonal
`(1,1)` block; the ordering lever had only the Jacobian to work with, and
measurement confirmed the whole exact-vs-lbfgs gap was iteration count,
not factorization time. Under a block-partitioned Hessian the `(1,1)`
block is genuinely block-banded, dense inside each block, and it *is* the
growing cost. That is exactly the shape `External` orderings and the Schur
path (`set_kkt_schur_block`, pounce#180 item 2) exist for — and the Schur
path is reachable here, because this updater publishes a real
`SymTMatrix` rather than a low-rank object.

## Where this leaves the idea

Not refuted; re-scoped. The unmeasured variant is the paper's own:

1. **Per-primal-block elements.** Partition `x` by collocation stage,
   maintain one damped-BFGS block per stage against Lagrangian gradient
   differences. Many fewer blocks, each much better informed. This is the
   experiment that should be run next, and the harness, the option
   surface, the assembly path, and the oracle are all now in place for it —
   only the element-construction function changes.
2. **Share blocks across rows with identical support.** Collocation
   repeats structure; one block per *support class* rather than per row
   multiplies each block's direction coverage by the number of rows
   sharing it. Intermediate between (1) and what was built.
3. **Restrict the assembled pattern** to a declared Hessian pattern when
   the model has one. Cuts the 5.2× over-parameterization directly — but
   is unavailable to precisely the users this targets.

What should be kept regardless of which way that goes: the oracle, the
`objective_nonlinear_vars` hook, and the recorded fact that a relative
denominator floor cannot serve as a magnitude control on a per-element
update.

## Reproducing

```bash
cd benchmarks/large_scale && python3 generate_nl.py laptime --scale 0.08 --out-dir nl_0.08
cargo build --release -p pounce-cli
scripts/partitioned-qn-sweep.sh ./target/release/pounce benchmarks/large_scale/nl_0.08/laptime.nl

# the oracle and the per-element peaks
POUNCE_PARTITIONED_ORACLE=1 ./target/release/pounce \
  benchmarks/large_scale/nl_0.08/laptime.nl max_iter=6 hessian_approximation=partitioned
```

`POUNCE_PARTITIONED_DEBUG=1` prints the one-time structural census;
`POUNCE_PARTITIONED_DUMP=1` prints the whole assembled `W` for models
small enough to read.

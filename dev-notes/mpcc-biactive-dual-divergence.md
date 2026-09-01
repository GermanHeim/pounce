# gh#884: the biactive dual divergence, and the four fixes that did not work

Status: **fixed.** The route this note recommended — detect late, retry
cold — shipped; see "What shipped" at the end for what was built and
which of its open questions the build answered. The note keeps its
original shape because its value is the **negative** results: four
approaches ruled out by measurement, three in the *trigger* family and
one in the *policy* family. Each of them is an idea a reader will have
again, and each already has a number attached.

The title was "why the biactive dual divergence is not a targeted fix",
and an earlier draft argued no solver-observable trigger could satisfy
gh#884's criteria 1 and 3 together. That was wrong, and how it was wrong
is the lesson: the argument turned on a property of the *limit point*
(S-stationarity) that the solver cannot know, and the fix turned on a
property of the *iterate* (that it stopped moving) that the solver
observes every iteration. When an impossibility argument appeals to
something unknowable, check whether the thing you actually need is
observable instead.

**Provenance.** Ruled out 1–3, the mechanism trace and the `||d||`
separation were measured on `87402274`, whose solver source is identical
to `main` at `7c42947f` (this branch adds only a test file and this
note). **Ruled out 4 was measured on `d89771bc`** plus an unpushed
working-tree patch, and is the one section not reproducible from this
repository — see its own provenance paragraph. Re-measure rather than
trusting any of these numbers across a commit that touches the IPM.

Guards: `crates/pounce-algorithm/tests/issue_884_biactive_dual_divergence.rs`
(eight tests now — four invariants and four branch tests added with the
fix) and `crates/pounce-cli/tests/issue_884_biactive_dual_divergence.rs`
(the reproducer, end to end).

The original four tests are **invariants, not a description of the bug**. None of
them asserts that the current failure persists: a test that pinned the bug
would go red on a genuine fix, which is backwards, and would teach the next
reader to expect red in that file. The split follows an asymmetry —
`qpec_small` failing is the *bug*, so it is never asserted; `ralph1`
failing is *correct*, so it is. Everything below is measurement, and being
measurement is why it lives here rather than in an assertion.

## The mechanism, sharpened

gh#884 describes the symptom — the primal reaches the solution and the
dual diverges. To be precise about "reaches": the run stops at
`(1.0002321, 1.0001161, 2.67e-15)`, feasible to `2.2e-16` with
`f = 6.73e-8` against `f* = 0`. That is *near* the optimum, not at it, and
the `lambda = 0` certificate that holds exactly at `(1, 1, 0)` leaves an
objective-gradient residual of `4.64e-4` at the returned iterate. The
point is that the run gives up that close, not that it arrives.

The cause is one step further back, and it is a feedback loop.

`qpec_small` under `prod_eq` from `origin`, default options, traced per
iteration. `dual` is the `s_d`-normalised dual term that
`curr_barrier_error` actually reads; `kappa_eps_mu = barrier_tol_factor · mu`
is what it must fall below for `mu` to decrease:

| it | `mu` | `prim` | `compl` | `dual_raw` | `s_d` | `dual` | `kappa_eps_mu` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1e-1 | 2.0e-2 | 1.9e0 | 1.5e0 | 1.0e0 | 1.5e0 | 1.0 |
| 3 | 1e-1 | 8.1e-5 | 1.1e-2 | 1.8e3 | 1.8e1 | 9.8e1 | 1.0 |
| 5 | 1e-1 | 4.1e-6 | 8.0e-7 | 3.2e6 | 1.8e5 | 1.8e1 | 1.0 |
| 7 | 1e-1 | 5.0e-8 | 8.0e-11 | 6.3e10 | 1.7e9 | 3.6e1 | 1.0 |
| 9 | 1e-1 | 1.0e-11 | 6.8e-11 | 7.8e11 | 3.5e9 | **2.2e2** | 1.0 |
| 11 | 1e-1 | 1.0e-15 | 8.0e-11 | 9.1e7 | 2.7e9 | 3.4e-2 | 1.0 |

Read the columns together:

1. **Neither `prim` nor `compl` ever blocks `mu`.** Both are already
   below `kappa_eps_mu = 1` by iteration 3 (`8.1e-5` and `1.1e-2`), and
   reach `~1e-11` by iteration 9. Only the `dual` column is ever the
   binding term.
2. **`mu` is pinned at `1e-1` for eleven iterations**, because the
   barrier-subproblem test is `sub_err <= kappa_eps_mu` and `sub_err` is
   the `dual` column, which oscillates `1.5 → 98 → 18 → 36 → 222` and
   does not fall below `1.0` until iteration 11.
3. **The multipliers are `z = mu / s`.** The equality product row drags
   the slack of the *parallel* inequality row (`G2 >= 0`) to `~1e-15`
   while `mu` stays at `1e-1`, so `z ~ 1e14`.
4. `s_d` is the mean multiplier magnitude, so it grows in lockstep with
   the very multipliers that are exploding. `dual_raw / s_d` therefore
   stays `O(1..100)` — normalised, but never *converging*.

That is the loop: the dual error cannot fall because the multipliers are
exploding; the multipliers explode because `mu` is pinned; `mu` is pinned
because the dual error cannot fall.

The linear dependence that starts it is visible in the returned
multipliers. Rows 2 and 5 both restrict only `y2`; rows 1 and 4 are
likewise parallel:

| row | `|grad|_inf` | `lambda` | product |
|---|---:|---:|---:|
| 1 (`H1 >= 0`) | `2.0` | `-2.089e10` | `4.18e10` |
| 4 (`G1·H1 = 0`) | `2.0` | `+2.089e10` | `4.18e10` |
| 2 (`G2 >= 0`) | `1.0` | `-7.283e11` | `7.28e11` |
| 5 (`G2·H2 = 0`) | `2.3e-4` | `+1.737e15` | `4.03e11` |

`-7.283e11 + 4.03e11 = -3.25e11`, the reported residual. Note **no
regularization ever engages** — `reg = 0` at every iteration — because
the KKT matrix is singular only in the limit, never at any finite
iterate. The inertia-driven trigger cannot see this coming.

## Ruled out 1: the Hessian sparsity hypothesis. **Refuted.**

gh#884 records that the same model exits differently through the MPCC
harness's Python path (`Error_In_Step_Computation`, 118 iterations) than
through `.nl`/CLI (`Solved_To_Acceptable_Level`, 41), and names the
Hessian sparsity difference as the leading hypothesis — the harness hands
back a dense lower triangle with 6 nonzeros where the structural pattern
has 5, since `(2,1)` is identically zero. It marks this the cheap
prerequisite, to be done before the fix.

It is done, and it is not the cause. Driven from Rust with the two
patterns as the only difference, the results are **bit-identical**: same
status, same 14 iterations, same returned iterate to every digit. Pinned
by `a_structurally_zero_hessian_entry_does_not_change_the_solve`, which is
a contract in its own right: declaring an identically-zero Hessian entry
must be a no-op.

Whatever separates the two paths is somewhere else, and the ownership
bucket's description still needs re-checking against a harness
re-measurement — that part of gh#884 stands.

## Ruled out 2: detect the runaway and engage `delta_c` *in flight*. **Too late by construction.**

Dual regularization *is* the effective remedy. With `perturb_always_cd=yes`
the same model converges in 19 iterations to an **unscaled** KKT error of
`9.96e-8` at `x = (0.999994, 0.999997, 3.7e-6)` — an honest solve, not a
residual normalised away. So an answer is reachable and gh#884 is a real
POUNCE gap. Pinned by `dual_regularization_reaches_the_optimum_honestly`.

The obvious refinement is to engage it *only* when the runaway is
detected, leaving every other model alone. That does not work, and the
reason is structural rather than a matter of tuning. Engaging `delta_c`
from iteration `N`:

| engage from | outcome |
|---:|---|
| 3 | `SolveSucceeded`, 18 iters |
| 6 | `SolveSucceeded`, 42 iters |
| 8 | **`RestorationFailed`** |
| 9 | `SolveSucceeded`, 35 iters |
| 10 | **`RestorationFailed`** |
| 11 | **`RestorationFailed`** |

Non-monotone, and failing from exactly the region where the pattern first
becomes *visible*. By the time the multipliers have reached `~1e11` —
which is what a detector keyed on "the dual is running away" must wait
for — regularization can no longer recover **that iterate**.

Note precisely what this rules out: switching regularization on *within
the same run*, keeping the state the run has already reached. It says
nothing about discarding that state. See "Not ruled out" below.

## Ruled out 3: make `perturb_always_cd` the default. **Trades an honest failure for a silent wrong answer.**

If it must be engaged from the start, the question becomes whether it can
be on by default. It cannot.

`ralph1` under the `direct` lowering (`G·H <= 0`), `f* = 0` at the
origin, which is M-stationary but **not** S-stationary — NLP KKT *is*
S-stationarity, so no sign-feasible multiplier exists there and failing
is correct:

| | status | objective | unscaled KKT |
|---|---|---:|---:|
| default | `RestorationFailed` (39 iters) | `1.22e-3` | `1.89e10` |
| `perturb_always_cd=yes`, `max_iter=300` | `SolvedToAcceptableLevel` | `-3.81e-5` | `2.44e-6` |
| `perturb_always_cd=yes`, `max_iter=3000` | **`SolveSucceeded`** (356 iters) | **`-2.71e-5`** | `5.25e-7` |

The last row is the disqualifying one: plain `Solve_Succeeded` at an
objective **below `f* = 0`**, reachable only at a point the MPCC does not
contain. Raising the cap makes it worse, not better — the acceptable-level
exit at 300 iterations was the cap hiding it.

This is gh#884's own acceptance criterion firing ("`ralph1` must still
fail — a fix that greens all eight has over-fired"), and it is the P1a
lesson again: gh#884's current failure is *honest*, and this would
replace it with a success at the wrong answer.

Guarded by `ralph1_must_not_claim_success_where_no_multiplier_certifies_it`
— and note *how*. An earlier draft of that test asserted the bad outcome
directly: it ran `perturb_always_cd=yes` and required a success below `f*`.
That was wrong three ways. It would go red when someone *fixed* the
underlying problem. It pinned a status the measurements above show is a
function of `max_iter` (`SolvedToAcceptableLevel` at 300,
`Solve_Succeeded` at 3000), so it would fail for reasons its name does not
describe. And it made "expected red" the norm for the file.

The test now runs at **default options** and asserts the contract: this
model must not report success, and never below `f*`. That is
direction-correct — red exactly when the solver gets worse. It still
catches the trap, because the only way to *ship* this remedy is to change
the default, which the test would then be running under.
Mutation-checked: flip that fixture to `perturb_always_cd=yes` and this
test alone fails, on the objective bound, at `-2.71e-5`.

The general rule, worth stating because it is easy to get backwards on a
known-open bug: **pin the invariant, not the state.** A conditional of the
form "if the solver claims success, the claim must be true" is vacuous
while a model is broken, load-bearing once it is fixed, and red exactly
when someone fakes the fix — which is the failure mode gh#884 names in its
own text ("a fix that only changed the exit verdict would be treating the
symptom").

`min -exp(x) s.t. x >= 0` — gh#274's own reproducer — is unaffected
either way (`ErrorInStepComputation`, identical iterate). That criterion
was never the binding one.

## Ruled out 4: give the acceptable-level gate a dual ceiling. **Refuses the right point, and a large correct class with it.**

The first entry in the *policy* family rather than the trigger family —
the route gh#884's "Out of scope" section names and leaves undecided
("refusing that point needs a threshold moved … precedent: gh#532's
`dual_inf_scale_kappa` … `1e10` reads as a *disabled* cap rather than a
loose one. Not decided here").

Measured by @jkitchin in a separate session on `d89771bc` plus an
unpushed working-tree patch implementing the bound below. The table is a
direct in-session probe, **not a committed test**, so it is not
reproducible from this repository as it stands; the patch was not
proposed for merge.

The route: give the acceptable-level gate the dual ceiling it lacks,
referenced to `norm_inf(grad f)` rather than to `dual_scale`.

```text
acceptable_dual_inf_bound = min( acceptable_dual_inf_tol,
                                 max( dual_inf_tol,
                                      kappa * acceptable_tol * norm_inf(grad f) ) )
```

`grad f` rather than `dual_scale`, because gh#532's argument — "a
multiplier contributes to `dual_scale` whatever it contributes to
`grad L`, so it cannot buy the test" — holds for *one* multiplier and
fails for a **pair**. Two rows with parallel gradients admit multipliers
of any size whose contributions cancel, which is exactly what a biactive
pair produces: `dual_scale` runs away while `grad L` does not.
`norm_inf(grad f)` is the one term of `grad L` no multiplier can move.

On its face it clears gh#884. It refuses the `qpec_small`/`ncp_eq`/origin
certificate at the residual the issue names, leaves
`ralph1`/`direct`/origin byte-identical, converges nothing so it cannot
over-fire, and `scripts/sweep-fixtures.sh` diffs **empty across all 182
fixture-legs, both legs**.

It is still wrong. `crates/pounce-rs/tests/watchdog_trial_is_not_a_divergence_verdict.rs`,
`IllConditionedQuadratic`, `n = 12`, condition `1e14`, L-BFGS:

| | status | iters | objective | unscaled `norm_inf(grad L)` | `norm_inf(grad f)` | ratio |
|---|---|---:|---:|---:|---:|---:|
| ceiling on | `RestorationFailed` | 298 | `9.5916e8` | `8.4150e10` | `8.4150e10` | **1.0000** |
| ceiling off | `SolvedToAcceptableLevel` | 197 | `3.7375e-6` | `8.7467e1` | `8.7467e1` | **1.0000** |

The model is **unconstrained** — no rows, no bounds — so `grad L` is
identically `grad f` and the ratio the ceiling tests is `1` by
construction. The bound collapses to `max(1.0, 8.75e-5) = 1.0` and
refuses a residual of `87.5`: `acceptable_dual_inf_tol` is tightened from
`1e10` to `dual_inf_tol` for the whole unconstrained and bound-only
class, which has nothing to do with a biactive pair. The run then wanders
100 further iterations into restoration and ends 15 orders from `f*`.

**The defect is in the discriminator, not the constant.** The claim was
that `norm_inf(grad L)/norm_inf(grad f) ~ 1` is the signature of an
*unresolved* multiplier. It is equally the signature of an ordinary
not-yet-converged iterate with nothing to cancel. gh#884's criterion 2
asks for something "a multiplier of `1e9` on a `1e-9` gradient cannot
satisfy"; `grad f` clears that direction and says nothing about what else
it refuses. A discriminator has to be checked in **both** directions —
what it admits and what it turns away — and the corpus could not perform
the second check here, for the reason in the next section.

## The corpus cannot see the dimension a gate change acts on

The generalisable half of ruled-out 4, and it qualifies the evidence in
every other section of this note.

**An empty fixture sweep is not evidence about a change to the
acceptable-level dual gate.** No fixture in the corpus has an
acceptable-level exit whose unscaled dual residual exceeds
`dual_inf_tol`, so the corpus is uniform in exactly the dimension such a
change acts on, and reports clean no matter what the change does. Ruled
out 4 diffs empty across all 182 fixture-legs while tightening
`acceptable_dual_inf_tol` by ten orders for an entire problem class. The
model that caught it is not in the corpus at all.

This is CLAUDE.md's own branch rule — "a corpus that is uniform in the
dimension a change acts on reports 'small and mixed' no matter how large
its models are" — instanced on the **convergence gate** rather than on
iteration counts, which is where that rule has always been stated. It
joins the two cases CLAUDE.md already records: the convex arm's
cost-normalization (`σ`) path, and the bound-relax iteration cost.

It also bounds the retry route below. That route's "generality across the
corpus" caveat is measuring against the same corpus, so an empty sweep
there will be evidence about *trajectories* and not about verdicts. Any
gate-touching component of a gh#884 fix needs a purpose-built fixture
that reaches an acceptable-level exit above `dual_inf_tol`, because
nothing in `benchmarks/mpcc/` or the CLI corpus does.

## Not ruled out, and the most promising route: detect late, then retry cold

Raised in review of PR #885, and it is a real gap in the three experiments
above. They cover the sparsity hypothesis, engaging `delta_c` **in flight**
at sampled iterations, and flipping the default **globally**. None of them
covers *detecting late and then restarting from iteration 0 with
regularization enabled*. "Too late to recover this iterate" is not "too
late to act" if the action is to throw the iterate away.

And the separation such a detector would need is present and large.
Minimum `||d||` over each run, taken only while the primal is converged
(`inf_pr <= 1e-8`), measured on `87402274`:

| fixture | min `||d||` | at iteration | max `inf_du` | status |
|---|---:|---:|---:|---|
| `qpec_small` / `prod_eq` / origin | **`4.309e-8`** | 12 | `7.76e11` | `RestorationFailed` |
| `ralph1` / `direct` / origin | **`7.153e-3`** | 17 | `3.63e13` | `RestorationFailed` |

Five orders apart, so any threshold in `(4.3e-8, 7.2e-3)` separates them.
That is the difference between an iterate that has *settled* while only
its multipliers run away, and one still wandering — which is exactly the
"converged primal, unbounded multiplier" versus "diverging iterate"
discriminator gh#884 asks for.

Composed with the cold retry, on these two fixtures the policy gives:

- `qpec_small` — detector fires (`||d||` reaches `4.3e-8`); retry from
  iteration 0 with `delta_c` on reaches `SolveSucceeded` at an unscaled
  KKT error of `9.96e-8`, the honest solve recorded above;
- `ralph1` — detector never fires (`||d||` bottoms out at `7.2e-3`, five
  orders above any separating threshold), so no retry happens and the
  honest failure stands.

**Two of gh#884's five acceptance criteria, satisfied** — 1 (`qpec_small`
must not report success at `7.9e+04` unscaled) and 3 (`ralph1` must still
fail). The other three are untouched by this two-fixture experiment and
remain open against the route: **2**, a discriminator "checkable at the
gate" — a minimum `||d||` taken over history while `inf_pr <= 1e-8` is a
*trajectory statistic*, so a route built on it either has to argue the
criterion is met in spirit or propose reinterpreting it; **4**, evidence
it does not relabel any `-exp(x)`-shaped case (gh#274's reproducer); and
**5**, `scripts/sweep-fixtures.sh` across both legs. (Both were measured
against the shipped detector — see "What shipped".)

An earlier version of this note claimed no solver-observable trigger could
satisfy 1 and 3 *together*, on the grounds that the property separating
the two models is whether the limit point is S-stationary and a solver
cannot know that in advance. That
claim was too strong and is **retracted**: the solver does not need to
know S-stationarity, only to observe that the iterate stopped moving,
which it can.

What is *not* established, and would have to be before this ships:

- **Generality.** Two fixtures. Whether a `||d||` threshold separates the
  classes across `benchmarks/mpcc/`'s 79 fixture-legs, let alone the CLI
  corpus, is unmeasured — and per CLAUDE.md a threshold with no measured
  population behind it is exactly the kind that gets retuned later by
  someone who cannot see why it was chosen.
- **Cost, and it is not the one to worry about.** A cold retry doubles
  the work on every model that trips the detector, and that rate is
  unmeasured. But wall clock is the *cheap* half. The retry's remedy is
  `perturb_always_cd=yes` — the option **Ruled out 3** above rejects under
  the heading "trades an honest failure for a silent wrong answer",
  measured there at `SolveSucceeded` with `f = -2.71e-5`, below
  `f* = 0`, on `ralph1`. So a false positive does not cost a doubled solve; it routes
  the model into a configuration measured to return a wrong answer
  *reported as success* — which is strictly worse than the honest failure
  the detector exists to preserve, and is the same failure mode this note
  ruled the option out for. The detector is the entire barrier between
  those two outcomes, and it currently rests on two data points.

  The consequence for the prototype is concrete: **the retry's answer
  cannot be trusted on the strength of the detector alone.** It needs a
  post-check the retry has to pass before its verdict is returned — at
  minimum the contract in
  `crates/pounce-algorithm/tests/issue_884_biactive_dual_divergence.rs`'s
  `a_claimed_success_must_be_real` (a claimed success must hold in the
  model's own units), and on a model with a known bound, that the retried
  objective has not gone *below* what the un-retried run achieved. Without
  it, the route's failure mode on a false positive is silent.
- **Whether the retry converges in general.** It does here; it is a
  different trajectory on every other model.
- **`ralph1` is one of eight cells.** The other seven are `qpec_small`
  starts; the detector's behaviour on the remaining six is unmeasured.

So: promising, cheap to prototype, and the route this note recommended
trying first. It was, and it is what shipped.

## What shipped

The route above, with three changes forced by the build.

**The trigger is a per-iterate conjunction, not a trajectory statistic.**
The experiment used a minimum `||d||` taken over history while
`inf_pr <= 1e-8`, which is exactly what gh#884's criterion 2 rules out —
it is not checkable at the gate. The shipped detector evaluates three
things at *one and the same* iterate, and is sticky once true:

* `inf_pr <= 1e-8`;
* a **scale-relative** step, `maxᵢ |dᵢ| / (1 + |xᵢ|)` over the `x` and `s`
  blocks, at or below `dual_divergence_retry_step_tol` (default `1e-5`);
* the **unscaled** dual infeasibility at or above
  `dual_divergence_retry_du_floor` (default `1e2`), and finite.

The step is scale-relative rather than a bare norm because a bare `||d||`
threshold means different things at different variable scales, and the
detector's whole job is to separate two models by that number.

**Eligibility: `m >= 1`.** On an unconstrained model `∇L ≡ ∇f`, so the
third conjunct degenerates into a second, much looser copy of
`dual_inf_tol` and the detector would fire on any model with a large
gradient at a flat spot. This is the same collapse that killed ruled-out
4, arriving from a different direction.

**The promotion gate has five conjuncts, not one.** The base attempt saw
the signature; its status is `SolvedToAcceptableLevel` or
`RestorationFailed` (see the scope note below — never `SolveSucceeded`,
and deliberately not the generic exhaustion exits); the retry returns
`Solve_Succeeded`; the retry's **unscaled** KKT error *and* unscaled
constraint violation are within `acceptable_tol`; and the retry's
unscaled KKT error strictly beats the base attempt's. Conjunct 4 is the
one that matters: the defect *was* a status contradicted by its own
unscaled residual, so a gate reading the status alone reproduces the bug
one attempt later. On a refusal all three sinks are floored the way the
μ fallback floors them (pounce#870) — solution payload, certificate, and
the last trace row.

Deliberately **not** deferred to `TERMINATION_POLICY_OPTIONS` the way the
μ flip's acceptable-level trigger is (gh#757). That deferral exists
because the μ flip can hand back a different *local solution*; this retry
cannot, because conjunct 4 requires the promoted answer to satisfy the
KKT conditions in the model's own units.

### What the build answered from the open list

- **"The false-positive class is unquantified."** It stays unquantified
  in general, and the sweep found the corpus's one second instance —
  `deb7` under L-BFGS, where the *detector* is right and the *remedy* is
  not. That is the subject of the next subsection, and it is what set
  the status scope. Among the acceptable-level exits, which is the class
  gh#884 is about, there is still no second instance: the only one on
  either leg is `mu_fallback_point_floor`, which ends at an unscaled dual
  of `2.4e-11`, and the closest approach to the `1e2` floor from any
  direction is `eigena2` under L-BFGS at `37` with a settled step of
  `7.9e-9` — under the floor, and the floor is why it is excluded. So the
  guard against a false positive is not corpus breadth. It is conjunct 4:
  a false positive costs one solve and cannot change the answer.
- **"The route needs a guard that a claimed success is real."** Built,
  as conjunct 4, and reported: `dual_divergence_signature` and
  `dual_divergence_retry_promoted` reach `SolveStatistics`, the JSON
  report and the console.
- **"Whether the retry converges in general."** Still open, and still a
  different trajectory on every other model — which is why a
  non-converging retry is a no-op rather than a regression.
- **"`ralph1` is one of eight cells."** Still one cell. What changed is
  that the safety argument no longer rests on breadth: the shipped
  `the_detector_must_not_fire_on_ralph1` is mutation-checked (widen the
  step conjunct to `1e-1` and only that test goes red), and the remedy's
  danger is stated in the option help rather than left implicit —
  `perturb_always_cd=yes` takes `ralph1` to `Solve_Succeeded` at
  `f = -2.71e-5`, *below* `f* = 0`, with an unscaled KKT error of
  `5.25e-7`. That answer beats its base attempt on every number the gate
  reads. Detector specificity is the only thing standing between it and
  a promotion, which is why the threshold is an option with a documented
  off value.

### `deb7`: a true positive for the detector, and why the gate names statuses

The fixture sweep — 80 fixtures, both legs, against a `main` binary —
moved exactly one line:

```
lbfgs  deb7  nlp  ErrorInStepComputation  it=715 -> it=3000
```

same status, same objective (`101.0934371`). The detector fires there,
and it is **not** a false positive. Measured at iteration 346: a
scale-relative step of `6.5e-6`, `inf_pr` of `3.0e-12`, and an unscaled
`inf_du` of `9.2e+05` — an *order above* `qpec_small`'s `7.9e+04`.

The first instinct is to move a threshold, and it does not survive
being written out. Raising the dual floor cannot work in the right
direction at all: `deb7` is the *larger* of the two on that conjunct.
Tightening the step tolerance can — `qpec_small` settles to `4.3e-8`
against `deb7`'s `6.5e-6` — but only by fitting the default to one
fixture, with a factor of ~30 of margin left for every model neither
corpus contains, on a conjunct whose other job is to hold `ralph1`
(`7.2e-3`) out by five orders. Buying two orders of specificity by
spending an order and a half of margin is a bad trade, and it is the
wrong lesson besides: the detector is describing this iterate
*correctly*. What does not transfer is the *remedy*: the retry
ran the full 3000-iteration budget to `Maximum_Iterations_Exceeded` at an
unscaled KKT error of `6.7e+01`, against the base attempt's `9.9e+01` —
better, and nowhere near `acceptable_tol`, so conjunct 4 refused it
exactly as designed. The cost was real all the same: 4x the trajectory,
for an answer that did not change.

So the scope was cut by **status**, not by threshold, and the two kept
statuses are the ones a vanishing-gradient row produces *directly*:

- `Solved_To_Acceptable_Level` is gh#884 verbatim — the `.nl`/CLI path.
- `Restoration_Failed` is the same defect one step earlier, where the
  TNLP path lands, at an unscaled KKT error of `3.3e+11`.
- `Error_In_Step_Computation` and `Maximum_Iterations_Exceeded` are
  generic exhaustion exits. Every hard model in the corpus can reach
  them for reasons that have nothing to do with an arbitrary multiplier,
  which is precisely what `deb7` demonstrates.

Worst case is now one extra solve, under the caller's own `max_iter`, on
a model that was going to report a non-success verdict anyway. `deb7` is
back to 715 iterations and the sweep diff is empty. Pinned by
`a_generic_exhaustion_exit_does_not_buy_a_retry`, which asserts the
signature *does* fire there and that no retry runs — so re-widening
`retry_worthy` fails a test that names this measurement rather than
silently re-costing the trajectory.

**Scoping by status is only as complete as the status is stable, and on
this very fixture it is not.** `deb7` reaches `Error_In_Step_Computation`
at default options and is out of scope; under
`limited_memory_ls_failure_restarts=1` — gh#818's re-anchor rung, off by
default — it reaches `Restoration_Failed` instead, and is therefore *in*.
It then paid the cost this section is about: measured **6.08 s to 25.17 s**
wall clock for the same `Restoration_Failed` verdict, the same objective,
and a declined retry. That cost is now fixed rather than accepted — see
the next section — but the scoping lesson survives the fix, so it is left
standing here. Nothing is wrong with the answer and nothing in the
default corpus sees it — the sweep runs default options — but it is the
honest bound on the narrowing. What the status scope buys is that the
retry only ever runs on a solve already reporting failure; what it does
*not* buy is that the retry only runs where the remedy has something to
work on. That is a second question, it needs a second gate, and the next
section is that gate.

### The retry has to be spent on an answer, not a trajectory (gh#887)

The detector is a statement about an **iterate**. Nothing in it says the
solve *ends* at that iterate, and `deb7` on the L-BFGS leg is the corpus
case where it does not: the signature is real there and the base attempt
then works its way back down before giving up. Whatever the trajectory
did in the middle, the answer being reported need not be one
`perturb_always_cd` can repair. Under the gh#818 rung that fixture bought
a full cold re-solve to decline an answer it was never going to promote —
**6.08 s to 25.17 s** — which is what gh#887 filed.

So the retry also asks what gh#884's defect looks like **in the answer**.
It is a point converged *except* that one multiplier ran away: the primal
is exact, complementarity is met, and the whole residual is dual
infeasibility. That is a ratio within one answer, and it is
`runaway_is_the_whole_residual` in `application.rs`:

```rust
max(viol, compl) <= DUAL_DIV_RETRY_DOMINANCE * dual_inf   // 1e-6
```

| run | unscaled dual | viol | compl | ratio |
|---|---|---|---|---|
| reproducer, `.nl` route | `7.90e4` | `1.1e-16` | `1.1e-9` | `1.5e-14` |
| reproducer, TNLP route | `3.25e11` | `2.5e-16` | `2.8e-3` | `8.7e-15` |
| `deb7` + rung, macOS | `9.90e1` | `8.0e-13` | `4.65e0` | `4.7e-2` |
| `deb7` + rung, Linux | `5.5743e3` | `5.6e-14` | `2.08e-5` | `3.7e-9` |

#### `deb7` is not a portable witness, and finding that out is the lesson

The first three rows are the ones the gate was designed against, and they
separate by twelve orders. The fourth row is the one that cost a red CI.

`deb7` under the rung reaches a **materially different answer** on the two
platforms CI runs — objective `99.677` on macOS against `99.651` on Linux
— and on Linux that answer genuinely *is* gh#884's shape: scaled overall
error `5.28e-1` against unscaled `5.57e3`, which is the `s_d`
normalisation hiding a runaway exactly as it did on `qpec_small`, with
complementarity eight orders under its own dual residual. So the retry
there is the **designed cost, not the waste gh#887 filed**, and a CLI
assertion that `deb7` declines is false on Linux at *any* threshold.

That is worth stating plainly because the tempting reading of the red was
"the constant is too tight". It was not. The constant was right and the
fixture was wrong, and the way to tell those apart was to read what the
Linux base attempt actually reported rather than to fit a number until
the job went green.

The consequence is that the rule is pinned as **unit tests on the
predicate** (`a_converged_point_with_a_runaway_multiplier_opens_the_retry`,
`an_unconverged_point_does_not_open_the_retry`,
`the_threshold_is_where_the_constant_says_it_is`,
`what_we_cannot_measure_does_not_buy_a_retry` in `application.rs`),
carrying all four measured rows including the Linux one. The CLI file
keeps only what does not depend on a trajectory:
`the_gate_that_reads_the_answer_does_not_cost_the_reproducer_its_retry`.
Mutation-checked: widen the ratio and the second and third go red; the
fourth stays green, because it pins the finiteness conjunct and not this
one.

This generalises past gh#887, and it is the branch rule in `CLAUDE.md`
wearing different clothes: **a fixture is only evidence about the answer
it actually reaches**, and "the answer it reaches" can differ by platform
on a model that is this hard. Before pinning a *negative* on a fixture,
check that the fixture reaches the same class everywhere it runs — a
green local run says nothing about that, and the assertion that catches
it is the one that fails on the other machine.

#### Why the ratio, and what it replaced

**An absolute floor** on the reported residual — reusing the detector's
own `dual_divergence_retry_du_floor` — declines macOS `deb7` by a **one
percent** margin (`9.9e1` against `1e2`). A coincidence, not a
discriminator, and a threshold on a scale-dependent quantity, which is
entry 3 of this repository's own review checklist.

**A retention ratio** — the reported unscaled KKT against the runaway the
*detector* had fired on — separated the first three rows by five orders
and passed locally. It failed on CI for the same underlying reason as the
fixture did: it reads two numbers from a **trajectory**, and `deb7`'s
detector reports `9.2e5` on one attempt and `8.7e2` on another *in the
same run*.

The dominance ratio compares two residuals of **one** answer. It carries
no units, needs no re-fitting when a fixture moves, and cannot depend on
which attempt fired or on how a platform rounded. That property is why it
is the shipped gate; the twelve-order separation is a bonus.

`DUAL_DIV_RETRY_DOMINANCE` is a constant rather than an option on
purpose: it does not express a tolerance a caller trades against, it
expresses "this answer is gh#884's shape". The escape hatch for the whole
remedy is `dual_divergence_retry=no`.

The cost claim is now true as stated: one extra solve, on a run that
satisfied the three-way detector, reached a scoped failure verdict, *and*
whose reported answer still has the defect's shape. On macOS `deb7` under
the rung is back to 6.1 s. On Linux it still pays the retry, and that is
correct — the answer it reports there has the shape the remedy is for.

One thing this does **not** do is make the status scope redundant, and it
should not be read as licence to widen it. The two gates fail in different
directions: the status scope keeps the retry off runs that never reported
failure at all, and the dominance gate keeps it off runs whose failure has
nothing to do with a multiplier.

One reporting wrinkle to know about, unchanged by any of this and not
gh#884's to relitigate. `SolutionCertificate` floors residuals and
deliberately *not* `iteration_count` — the count describes what the
invocation did, and rewinding it would under-report work that really
happened (see its doc comment, and
`issue857_escalation_gated_quality_rung.rs`, where `deb7` at
`max_iter=100` is the case that settled it). But the field is
overwritten per attempt rather than accumulated, so a declined retry
reports the base attempt's status beside the *retry's* count, which is
neither attempt's total. The μ fallback and the second-opinion ladder
have the same shape; the narrowing only makes it rarer here. Worth a
look if the scope ever widens.

### Criterion 4, measured

gh#274's `unbounded_exp.nl` — `min -exp(x) s.t. x >= 0`, unbounded below
— is the corpus's closest thing to a false positive, and it is close.
It satisfies **two of the three conjuncts outright**: the constraint row
stays satisfied while the iterates run off, so `inf_pr` is at zero and
the unscaled dual infeasibility is `8.7e+20` at the exit.

The step conjunct is the *only* thing holding the detector off, which is
gh#884's discriminator stated as a measurement rather than an intention.
Measured both ways, so that "it does not fire" is evidence for the
reason claimed rather than for some other conjunct quietly doing the
work:

| `dual_divergence_retry_step_tol` | signature | promoted | status | objective |
|---|---|---|---|---|
| `1e-5` (default) | no | no | `Error_In_Step_Computation` | `-8.688703979461661e+20` |
| `1e30` (conjunct disabled) | **yes** | no | `Error_In_Step_Computation` | `-8.688703979461661e+20` |

Status and objective are identical across the two rows: a deceived
detector costs nothing here at all. It cannot, for two independent
reasons. This model exits `Error_In_Step_Computation`, which the status
scope above excludes, so no retry runs — and when an earlier draft did
retry on that status, the retry returned an unscaled KKT error of
`1.49e+19` and conjunct 4 refused it. Either barrier alone is enough.
Pinned by `a_diverging_iterate_is_not_the_signature`, which asserts both
the signature's absence at the default *and* its presence at `1e30`, so
"it does not fire" stays evidence for the reason claimed.

### What did not change

The three ruled-out trigger/policy routes stay ruled out, and none of the
measurements above is superseded: an *in-flight* switch keyed on the
runaway pattern, a global default flip, and a dual ceiling on the
acceptable-level gate are all still wrong, for the reasons measured here.
A late detector with a cold retry is a different mechanism, and the
numbers in those sections say nothing about it either way.

And the corpus caveat stands: per the corpus section above, an empty
sweep says nothing about a gate change, which is why the fix ships with
its own fixture —
`crates/pounce-cli/tests/fixtures/mpcc_qpec_small_biactive.nl`, the
corpus's first MPCC lowering — rather than resting on the sweep being
quiet.

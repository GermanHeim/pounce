# Crossover: identifying an exact active set

An interior-point method never puts an iterate *on* a constraint. The
fraction-to-boundary rule keeps every slack strictly positive, so when the
solve converges, "which constraints are active" is something you **infer from
a tolerance test**, not something the solve established.

Usually the inference is right. Where it is not — and the case where it is
not is sharply defined — POUNCE can run an opt-in **crossover** phase: after
the interior-point solve converges, it pivots to the active-set path and
returns a point at which a linearly independent set of constraints is
satisfied to *equality*, with multipliers that certify stationarity against
exactly that set.

```
crossover                yes | no     (default: no)
crossover_max_iter       integer      (default: 30)
crossover_mult_tol       number       (default: 1e-8)
crossover_primal_tol     number       (default: 1e-6)
```

> This is the NLP feature. The convex LP path has its own, separate
> `qp_crossover` option, which purifies an LP iterate to an exact *vertex*
> — different engine, different problem class. Setting one does not affect
> the other.

## When you need it

The discriminating property is **failure of strict complementarity**: a
constraint that is active but whose multiplier is zero. At such a point the
barrier's own geometry places the iterate `O(√μ)` from the constraint. With
`μ` around `1e-9` at termination, that is a distance of about `1e-5` — four
orders of magnitude *larger* than the `1e-8` tolerance the solve reports
converging at. No tolerance test applied afterwards can recover the answer,
because the information is not in the iterate.

Three parts of POUNCE already pay for this:

1. **Sensitivity.** `covariance()` classifies each constraint as STRONGLY
   ACTIVE / WEAKLY ACTIVE / **AMBIGUOUS (loosely converged)** / UNIDENTIFIED
   (see [Sensitivity Analysis](sensitivity.md)). The AMBIGUOUS class exists
   because the interior iterate cannot decide. Crossover collapses it.
2. **Degeneracy.** A degenerate solution collapses the reduced Hessian, and
   that has repeatedly surfaced as an inertia problem met with a
   perturbation-side heuristic. Crossover attacks the same thing
   structurally: it produces a *linearly independent* active set.
3. **Warm starts.** The active-set SQP could previously only warm-start from
   a previous *SQP* solve ([Active-Set SQP & Warm
   Starts](active-set-sqp.md)). After crossover,
   `last_sqp_working_set()` returns the identified set, so a sequence whose
   first solve wants the interior method — MPC with a cold first solve,
   parameter continuation — can hand off to the active-set path.

Symptoms that point here: a parameter estimate sitting exactly at a bound
with a confidence interval you do not trust; shadow prices off a degenerate
model; an over-modeled engineering model where several constraints bind at
once.

## When you do not

If your solution satisfies strict complementarity — the bulk of well-posed
models — crossover has nothing to correct. It will run, take one step, find
the point already satisfies the stopping tolerances, and return it. That
costs roughly one extra iteration and changes nothing. Leaving it off is the
right default.

It also does nothing useful on a solve that did not converge: an unconverged
interior point is not a KKT point, so there is no active set at it worth
identifying. Crossover is skipped unless the solve reached `Solve_Succeeded`
or `Solved_To_Acceptable_Level`.

## What it does

The phase follows Byrd, Nocedal & Waltz, *KNITRO: An Integrated Package for
Nonlinear Optimization* (2006), §7.

1. The interior-point method terminates at `(x, y, z)` within its tolerance.
2. **Estimate the active set** by a tolerance test on primal distance
   (`crossover_primal_tol`) and multiplier magnitude
   (`crossover_mult_tol`).
3. **Take one EQP-equivalent step** over that set, plus a line search on the
   ℓ₁ penalty model with `ν₀` set just above the largest `|multiplier|` at
   the interior solution. If the result satisfies the stopping tolerances,
   stop. This is the common path, and it solves no LPs.
4. Otherwise **run the full active-set SQP** from the interior iterate,
   seeded with the estimated set and the same `ν₀`, for at most
   `crossover_max_iter` outer iterations.

### It works against the bounds you declared

The interior method widens every bound by `bound_relax_factor` (default
`1e-8`) before it starts — that widening is what lets an iterate approach a
bound without ever being pinned to it. Crossover undoes it: the pivot and
the activity test both run against the box and row bounds as *written*.

This is not a detail. A point sitting exactly on a declared bound is a full
`1e-8` **inside** the relaxed one, so measured against the relaxed bounds
every binding constraint reads as inactive and the pivot stops just short of
each one. Crossover would run, succeed, and report an empty active set.

A consequence worth knowing: the crossed-over point can sit `~1e-8` closer
to a bound than the interior solution did, and on the bound rather than
inside it. That is the intended result, and it is inside `constr_viol_tol`
by construction — the relaxation is capped there. It also makes
`honor_original_bounds` a no-op on a crossed-over solution: the point is
already in the declared box.

### Where this departs from the paper

KNITRO's active-set path is SLQP — an LP phase picks the working set, an EQP
phase computes the step. POUNCE's is an ordinary line-search SQP over
`pounce-qp`'s working-set interface. So:

- Step 3 is one `pounce-qp` solve against the NLP linearization at the
  interior iterate, warm-started with the estimated set. That call
  factorizes the hinted set to recover a primal and then pivots, which is
  what "solve the EQP over `A`, and fix `A` where the tolerance test got it
  wrong" amounts to here. The paper's property that step 3 avoids an LP is
  preserved.
- Step 4's LP trust region (the paper's eq. 7.22, sized to exclude every
  inactive constraint) has **no analogue** in a line-search SQP and is not
  implemented. Only the `ν₀` half of that setup is reproduced.

## It cannot make a solve worse

Crossover is a refinement of a solve that already succeeded, so the bar is
not "did it solve" but "is this at least as good a KKT point". The
crossed-over point replaces the interior one only if **all three** hold
against the interior iterate:

- constraint violation no worse, allowing movement within
  `sqp_constr_viol_tol`;
- stationarity no worse, allowing movement within the stationarity
  tolerance;
- the objective did not increase beyond a small relative slack.

Any failure returns the interior solution untouched. The tolerances rather
than the raw residuals are the comparison point on purpose: crossover puts
the iterate *on* the active constraints, which can move a residual from
`1e-12` to `1e-10` while the point is unambiguously better identified.
Refusing that would reject exactly the cases the phase exists for. What the
gate still refuses is a residual crossing its own tolerance.

Because it runs strictly *after* convergence and is off by default, enabling
it moves no interior trajectory.

## The one number that gets worse

Crossover puts the iterate on the declared bounds, but the end-of-run
summary still measures against the **relaxed** ones. Those two frames
disagree by exactly the relaxation, and the disagreement lands entirely on
the complementarity term: at an active constraint the returned point has
slack `δ ≈ 1e-8` in the relaxed frame with the true multiplier `v` beside
it, so the product reads `v·δ` instead of `~μ`.

Measured on HS14 (strictly complementary, `v ≈ 1.85`):

| | without crossover | with crossover |
|---|---|---|
| Dual infeasibility | `1.9e-12` | `8.9e-16` |
| Constraint violation | `2.9e-13` | `2.2e-16` |
| Complementarity | `2.5e-09` | `1.9e-08` |

The point is three to four orders of magnitude better on stationarity and
feasibility, and about 7× "worse" on a complementarity residual computed
against a constraint the model does not contain. In the frame crossover
actually solves in — the declared bounds — that residual is zero, which is
why the never-regress gate (which measures there, via `check_kkt`) accepts
the point.

Two practical consequences:

- **`Overall NLP error` in the summary can exceed `tol`** on a solve that
  legitimately converged, because it is the max over the three and
  complementarity now dominates it. The exit status is unaffected: it is
  decided by the interior loop before crossover runs.
- **`kkt_fidelity_tol` is the exception.** That opt-in gate is applied
  *after* crossover, so a threshold set between the two numbers above will
  downgrade a crossed-over solve. If you use both, set it against the
  post-crossover figure.

This is a reporting-frame mismatch rather than a property of the returned
point, and fixing it means teaching the summary which frame a crossed-over
solve lives in — tracked separately from this feature.

## Reading the result

The returned solution — `x`, the objective, `g`, and every multiplier — is
the crossed-over point, reported through the same path as any other solve.
There is no separate "crossover solution" to fetch.

Beyond that, from Rust:

```rust
let status = app.optimize_tnlp(tnlp);

// None  ⇒ crossover never ran (option off, or the solve did not converge).
// Some  ⇒ it ran; `accepted()` says whether it replaced the interior point.
if let Some(r) = app.crossover_report() {
    println!("accepted: {}", r.accepted());
    println!("phase:    {:?}", r.phase);          // EqpStep | ActiveSet
    println!("declined: {:?}", r.declined);       // why, when it did not
    println!("active:   {} bounds, {} rows", r.active_bounds, r.active_constraints);
    println!("estimated {} active before pivoting", r.estimated_active);
    println!("KKT {:e} -> {:e}", r.kkt_before, r.kkt_after);
}

// The identified set, ready to seed an `algorithm=active-set-sqp` solve.
let ws = app.last_sqp_working_set();
```

"Crossover never ran" and "crossover ran and declined" are different facts
about a solve, and the reason `crossover_report()` distinguishes them: a
consumer reasoning about active-set certainty must not read a declined
crossover as a confirmed active set.

`estimated_active` versus `active_bounds + active_constraints` is the
measurement the phase exists to make. They differ exactly where the
tolerance test on the interior iterate was wrong.

## Scope

Crossover is an opt-in post-convergence phase. It does not add SLQP as an
algorithm, is not a default, and changes nothing about the interior
iteration itself.

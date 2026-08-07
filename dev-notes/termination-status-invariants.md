# Termination-status invariants

One invariant has now been violated at four independent sites, and each was
fixed in isolation as if it were a one-off. It is written here so the fifth one
gets recognised as a recurrence rather than rediscovered.

## The invariant

> **Never claim infeasibility at a point the solver's own convergence test
> would accept.**

It is already stated in the source, at `pounce-restoration/src/resto_inner_solver.rs:756`,
where it is enforced as a single admissibility guard over the six restoration
gates:

```rust
let solver_would_call_it_feasible = orig_inf_pr_scaled <= outer_tol;
let locally_infeasible = !solver_would_call_it_feasible && ( … six disjuncts … );
```

The comment on that guard is the point of this note:

> *"Applied once to the combination rather than to each disjunct: the two
> preceding safeguards in this area were each added to one path and not its
> twin, and a hole survived both times."*

It was applied to restoration and never generalised. Two more holes followed,
in two other files.

## The four holes

| # | site | mechanism | fixed |
|---|---|---|---|
| gh #372 / #376 | restoration gates | scaled/unscaled units mismatch on `max(100·tol, 1e-4)` floors | 2026-07-22, guard above |
| gh #385 / #390 | `conv_check` surrogate `‖Jᵀc‖/max(1,‖c‖)` | not scale-invariant; row scaling drives it to zero anywhere | no-descent confirmation |
| gh #508 | `ipopt_alg.rs` cycle exit | violation compared against a threshold built from `tol` | 097a4719 |
| gh #519 | `conv_check/opt_error.rs` rapid detector | violation compared against an unclamped `infeas_viol_kappa · constr_viol_tol` | PR #521, ef5d77e8 |

Each was found by a different route, none by the guard that already existed.
gh #505 came from an external reporter; gh #508 from an `/adversary` run
attacking the same code from the opposite side, the same week.

Note how the fourth one was finally stated. gh #505 is the *report* — a model
returning local infeasibility at a point Ipopt accepts. gh #519 is the
*defect*, filed separately once the arming condition was identified, and it is
what actually fixed the reported symptom. PR #506, opened first and against an
earlier hypothesis, was rescoped afterwards to the structural change it always
should have been (below) and explicitly disclaims the fix. **The bug report and
the defect are different objects; conflating them is what made #505 take four
diagnoses to converge.**

## Why gh #505 survived three releases

The rapid-infeasibility detector landed in `8a711f34` (2026-05-20) and shipped
in v0.7.0, v0.8.0 and v0.9.0. Four reinforcing reasons, each of which is a
class of blind spot rather than an accident:

1. **The defaults are safe by coincidence.** The detector's absolute arm fires
   above `infeas_viol_kappa · constr_viol_tol` = `1e2 · 1e-4` = **1e-2**, which
   is exactly `acceptable_constr_viol_tol` = **1e-2**. Two independently chosen
   option defaults landing on the same number. Nothing in the code enforces the
   equality; it just happens to hold, so at defaults the detector can only fire
   *outside* the acceptable band. Tighten `constr_viol_tol` and the floor drops
   inside the band while the band stays put — **asking for tighter feasibility
   manufactures an infeasibility verdict.**

   The strongest evidence was in the same file the whole time: the *relative*
   arm one function up had exactly this hazard and was deliberately clamped
   against it, `(100.0 · constr_viol_tol).max(1e-2)`, with a doc comment stating
   the rule — *"too-loose withholds a verdict, too-tight fabricates one."* The
   absolute arm was the same expression **without the `.max`**. Two arms of one
   predicate, one hardened and one free to slide three orders below the floor.

2. **Every test ran at that coincidence.** Not one CLI test set
   `constr_viol_tol` on a *feasible* model. `infeasible_status_tol_invariance.rs`
   and `issue_508_infeasibility_gap_status.rs` sweep tolerances, but only on
   infeasible models. The untested direction is precisely the one that broke,
   and **it is still untested at the CLI level**: PR #521 pins the clamp with
   two unit tests in `opt_error.rs`, and the integration fixture that PR #506
   originally carried was dropped when that PR was rescoped. Enforcement item 3
   below is the missing arm, not a hypothetical one.

3. **The detector is POUNCE-only.** Upstream Ipopt has no equivalent, so no
   differential test against Ipopt can see it at default options — the two
   agree — and the divergence appears only in a feature the oracle does not
   have.

4. **The trigger was a user's driver, not a user's choice.** The reporter's
   `paper_ocp.py` sets `constr_viol_tol=1e-6` on every solve. POUNCE's own
   tests never sample the option distributions real drivers impose.

## Enforcement

Three checks, in cost order. The first two would have caught gh #505 in May.

### 1. Global self-consistency postcondition (no oracle needed)

Generalise `resto_inner_solver.rs:756` to every terminal status. At exit, this
is a contradiction in POUNCE's own reported numbers:

```
status ∈ infeasible band
  ∧ final scaled NLP error   ≤ acceptable_tol
  ∧ final unscaled violation ≤ acceptable_constr_viol_tol
```

On gh #505's `f=1` reproducer, `main` before gh #519's floor reported local
infeasibility at a scaled NLP error of **4.89e-10** against
`acceptable_tol = 1e-3` — six orders inside the band. Ship it as a debug assert
plus a CI sweep over the fixture corpus.

### 2. Kill-switch ablation

Every POUNCE-only heuristic has an option that disables it; the detector's is
`infeas_stationarity_tol=0`. Run the corpus with each switch off and diff the
verdicts. **Any model where disabling a heuristic improves the verdict is a bug
candidate.** This single control settled gh #505: `infeas_stationarity_tol=0`
flipped the verdict on stock `main`, naming the detector as the mechanism in
one run, with no oracle and no code reading. It was not run until roughly
eighteen hours into the investigation, and three superseded diagnoses were
published before it. gh #519 later confirmed it three more ways —
`infeas_viol_kappa=1e4` (floor back to 1e-2), `infeas_max_streak=15` and
`acceptable_iter=5` each flip it too. This should be a scheduled job, not
something reached for during debugging.

### 3. Tolerance monotonicity, in the feasible direction

`infeasible_status_tol_invariance.rs` pins "an infeasible model's verdict must
not depend on the user's `tol`". The missing mirror: **on a feasible model,
tightening any tolerance must never produce an infeasible or error verdict.**
Tightening may legitimately cost iterations, or downgrade `Solved` → `Acceptable`
→ `MaxIter`. It must never cross into the 200 or 500 AMPL bands. Cost is the
existing fixture corpus × ~6 tolerance options × ~6 values.

## Two static audits worth doing once

**Tolerance provenance.** Every numeric threshold compared against a constraint
violation must be built from a constraint-violation tolerance; every threshold
on a KKT error, from `tol`. gh #508 was a violation compared against
`max(100·tol, 1e-4)`; gh #519 is the mirror — a violation compared against
`1e2·constr_viol_tol`, gating a status that the acceptable band owns. Enumerate
the `*_tol` reads in `conv_check/`, `ipopt_alg.rs` and `pounce-restoration/`
together with the quantity each is compared against; the remaining instances
are findable by inspection.

**Default coincidences.** Enumerate derived thresholds that are numerically
equal at defaults but not equal *by construction*. Each is a latent bug: safe
today, broken by any user who moves one option. Either assert the equality as a
test that explains why it must hold, or remove the coincidence by keying the
threshold on a single option.

gh #519's fix is worth reading as a worked example of the trade-off. It takes
the constant form — `MIN_INFEAS_VIOL_FLOOR: Number = 1e-2`, applied as
`(infeas_viol_kappa · constr_viol_tol).max(MIN_INFEAS_VIOL_FLOOR)` — rather than
keying the floor on `acceptable_constr_viol_tol`. That is the right call for
the reported failure: it is the smaller change, it mirrors the sibling relative
arm exactly, and it restores the monotonicity that was broken. But it hardens
the coincidence into a constant rather than removing the coupling. The floor
now tracks the *default* `acceptable_constr_viol_tol`, not the user's, so a
user who **raises** `acceptable_constr_viol_tol` above `1e-2` re-opens a narrow
version of the same window — the detector can arm on a violation the user has
declared acceptable. Nobody has reported that; it is the residual to check
first if this area produces a fifth hole.

## Route convergence

PR #506 adds `terminate_local_infeasibility` plus a structural test asserting
that no site in `ipopt_alg.rs` builds `IterateOutcome::Terminate(SolverReturn::LocalInfeasibility)`
directly — two of the three routes had independently shipped the same defect of
discarding the acceptable-point stash. That PR is scoped to this and nothing
else; it does not fix gh #505's symptom and does not claim to.

The guard is a substring scan of one file: `rustfmt` wrapping the expression, a
temporary binding, or a construction in another module all evade it, and
`application.rs` names the same variant on the SQP and ℓ₁ elastic paths, out of
scope by design. It is a tripwire, not a proof — what it buys is that the
obvious way to add a fourth bare exit fails loudly and points at the helper,
which is the mistake that was actually made twice.

`ErrorInStepComputation` has the same shape and the same history (gh #372,
gh #508) and has not had the equivalent treatment.

## Reporting

A POUNCE-only heuristic that overrides a user-configured exit should say so in
the output. Had the banner read

```
local infeasibility (rapid detection: streak 5/5,
  floor 1e2·constr_viol_tol = 1e-4 vs violation 1.9e-4)
```

gh #505 would have been diagnosed in one comment instead of twenty.

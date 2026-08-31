//! gh #877 — FBBT cut feasible points three independent ways.
//!
//! The audit (`adversary/audit-2026-08-31/findings/seam-f-fbbt.md`) found
//! three mechanisms, all present in v0.10.0 and all made *reachable* by
//! PR #864, which accepted hand-written tapes through the `pounce-rs`
//! builder. The `.nl` translator had held the two unstated invariants by
//! construction — it emits reachable slots only and caps `PowInt`'s `n` at
//! 64 — so nothing had ever exercised the code without them.
//!
//! | finding | mechanism | fixed in |
//! |---|---|---|
//! | F-3 | `reverse_pass` propagated out of **every** slot, including ones the root's value does not depend on | `reverse::influencing_slots` |
//! | F-2 | `round_down` / `round_up` passed non-finite endpoints through, so an overflowed interval degenerated to `[inf, inf]` and emptied every operand | `interval::{round_down, round_up}` |
//! | F-1 | `n`-th roots were `powf(1.0/n)` plus a one-ULP pad; `1.0/n` is relatively 5.55e-17 low, so the root is short by `\|ln z\| · 5.55e-17` — 10 ULP at `z = 2^90` | `reverse::nth_root_enclosure` |
//!
//! F-3 is the severe one: it produced a **wrong answer under
//! `SolveSucceeded`**, off by the entire feasible range, with
//! `infeasibility_witness: None` and no diagnostic. F-1 and F-2 produce a
//! false `infeasibility_witness` on a feasible problem, which is at least
//! loud.
//!
//! # Which branch each test reaches
//!
//! Per CLAUDE.md's gh #756 rule — a guard is evidence only about the branch
//! its fixture reaches — the F-3 rule is covered on **both** sides. It is
//! not enough to show that a dead slot stops tightening: a rule that simply
//! stopped tightening would pass that and be useless. `the_control_case_*`
//! is the test; the other two are the bug.
//!
//! - `a_dead_slot_no_longer_tightens` — unreachable slot (the headline case).
//! - `a_pow_zero_masked_slot_no_longer_tightens` — *reachable* slot whose
//!   value the root ignores, because `a⁰ == 1` for every `a`. This is why
//!   the rule is phrased as influence rather than reachability.
//! - `the_control_case_an_influencing_slot_still_tightens` — the same `Ln`
//!   slot, now feeding the root, must still tighten `x` to `x > 0`.
//!
//! # Mutation table (measured, not asserted)
//!
//! Each row is the fix reverted and this file re-run. A test that goes red
//! for no mutation is documentation, not a guard, and is labelled as such
//! rather than left to look like coverage.
//!
//! | reverted | red |
//! |---|---|
//! | reverse loop ignores `influencing_slots` | `a_dead_slot_no_longer_tightens`, `a_pow_zero_masked_slot_no_longer_tightens`, `influence_is_decided_per_tape_not_per_pool` |
//! | `PowInt(_, 0)` edge followed (reachability only) | `a_pow_zero_masked_slot_no_longer_tightens` |
//! | `round_down`/`round_up` pass non-finite through | `an_overflowed_forward_interval_is_not_an_infeasibility`, `a_nan_producing_row_widens_to_entire_rather_than_emptying`, `an_overflowing_sum_on_a_feasible_row_is_not_an_infeasibility`, `a_huge_exponent_does_not_wrap_into_a_negative_power` |
//! | roots back to `powf(1/n)` ± 1 ULP | `the_odd_root_encloses_an_exactly_representable_solution`, `a_root_near_the_top_of_the_range_is_still_enclosed` |
//! | `powi` back to `n as i32` | `a_huge_exponent_does_not_wrap_into_a_negative_power` |
//!
//! Two tests are red for **no** mutation, and are kept anyway as branch
//! documentation, honestly labelled: `the_even_root_encloses_both_signs` and
//! `the_odd_root_encloses_a_negative_solution` use perfect squares and cubes,
//! where `powf` happens to be exact, so the old code passed them too. They
//! record that the even and negative-odd branches exist and are reached; the
//! shortfall that F-1 is about needs a non-power-of-two root, which is what
//! the two tests in the row above use. `the_control_case_*` is likewise green
//! under every mutation — by construction: it is the case the fix must *not*
//! break, and its value is that a rule which simply disabled tightening would
//! fail it.
//!
//! # What this file is NOT evidence about
//!
//! - **Trajectory.** Both `presolve` and `presolve_fbbt` default to `no`, so
//!   no shipping default path changes and no fixture sweep is owed
//!   (`scripts/sweep-fixtures.sh` runs at the defaults).
//! - **The `.nl` path.** Its translator's invariants meant these reproducers
//!   could not be built there. The fixes make the engine sound for *any*
//!   producer; they do not change what the `.nl` producer emits.
//! - **The `n`-cap suggested on the issue.** `validate_fbbt_tape` still does
//!   not bound `PowInt`'s `n`, deliberately: the defect was `n as i32`
//!   wrapping, and `interval::powi` now takes `u32` and converts with a
//!   checked `i32::try_from`, falling back to `powf`. A cap would be a
//!   workaround for a conversion that is no longer wrong, and it would
//!   reject tapes that are now handled correctly —
//!   `a_huge_exponent_does_not_wrap_into_a_negative_power` is the pin.
//! - **Tape/`constraints()` agreement.** F-6: the builder's check samples two
//!   points to a relative tolerance of ~1.5e-8. That is unchanged; the docs
//!   now say so instead of claiming "exactly restate".

use pounce_nlp::expression_provider::{ExpressionProvider, FbbtOp, FbbtTape};
use pounce_presolve::fbbt::{FbbtConfig, FbbtReport, forward_pass, forward_result, run_fbbt};

const INF: f64 = f64::INFINITY;

struct OneRow(FbbtTape);

impl ExpressionProvider for OneRow {
    fn constraint_expression(&self, i: usize) -> Option<FbbtTape> {
        (i == 0).then(|| self.0.clone())
    }
}

/// Run one row of FBBT over `ops` and return the tightened box plus report.
fn tighten(
    ops: Vec<FbbtOp>,
    xl: &[f64],
    xu: &[f64],
    glo: f64,
    ghi: f64,
) -> (Vec<f64>, Vec<f64>, FbbtReport) {
    let mut lo = xl.to_vec();
    let mut hi = xu.to_vec();
    let provider = OneRow(FbbtTape { ops });
    let report = run_fbbt(
        &provider,
        xl.len(),
        1,
        &mut lo,
        &mut hi,
        &[glo],
        &[ghi],
        None,
        &FbbtConfig::default(),
    );
    (lo, hi, report)
}

// ---------------------------------------------------------------------------
// F-3 — propagating out of a slot the root's value does not depend on
// ---------------------------------------------------------------------------

/// The headline case. `g(x) = x + 0` on `x ∈ [-10, 10]`, with an unused
/// `Ln(x)` slot parked in the tape — exactly what a producer emitting one
/// tape per row from a shared CSE pool hands over.
///
/// The `Ln` slot's forward interval is clipped to the log domain, which is
/// the correct *forward* answer for a sub-expression that is part of the
/// constraint. Pushing that clip back out of a slot the root ignores
/// fabricates `x ≥ 0`, and the LP's optimum moves from `-10` to `≈0`.
#[test]
fn a_dead_slot_no_longer_tightens() {
    let (lo, hi, report) = tighten(
        vec![
            FbbtOp::Var(0),
            FbbtOp::Ln(0), // dead: nothing reads it
            FbbtOp::Const(0.0),
            FbbtOp::Add(0, 2), // root
        ],
        &[-10.0],
        &[10.0],
        -10.0,
        10.0,
    );
    assert_eq!(
        (lo[0], hi[0]),
        (-10.0, 10.0),
        "the dead Ln slot must not move the box; got [{}, {}] (pre-fix: lo == 0)",
        lo[0],
        hi[0]
    );
    assert_eq!(report.bound_updates, 0);
    assert_eq!(report.infeasibility_witness, None);
    // The point the pre-fix box cut, and the LP optimum it moved.
    assert!(lo[0] <= -10.0);
}

/// Reachability alone would not have fixed it. `f64::NAN.powi(0) == 1.0`, so
/// `PowInt(a, 0)` is the constant `1` for *every* `a` — the `Ln` slot below
/// is reachable from the root and still says nothing about `x`. The rule
/// cuts the dependency edge at a zero exponent for exactly this reason.
#[test]
fn a_pow_zero_masked_slot_no_longer_tightens() {
    // g(x) = ln(x)^0 = 1, asserted == 1. True for every x, including x < 0.
    let (lo, hi, report) = tighten(
        vec![
            FbbtOp::Var(0),
            FbbtOp::Ln(0),
            FbbtOp::PowInt(1, 0), // root: == 1 whatever slot 1 holds
        ],
        &[-10.0],
        &[10.0],
        1.0,
        1.0,
    );
    assert_eq!(
        (lo[0], hi[0]),
        (-10.0, 10.0),
        "a ^0-masked slot must not move the box; got [{}, {}] (pre-fix cut x = -5)",
        lo[0],
        hi[0]
    );
    assert!(lo[0] < -5.0, "x = -5 is feasible and must stay in the box");
    assert_eq!(report.infeasibility_witness, None);
}

/// **The control, and the actual test of the rule.** The identical `Ln(x)`
/// slot, now feeding the root: `ln(x) ≤ 5` genuinely implies `x > 0`, and
/// FBBT must still say so. A rule that merely stopped tightening would pass
/// the two tests above and fail here.
#[test]
fn the_control_case_an_influencing_slot_still_tightens() {
    let (lo, hi, report) = tighten(
        vec![FbbtOp::Var(0), FbbtOp::Ln(0)],
        &[-10.0],
        &[10.0],
        -INF,
        5.0,
    );
    assert!(
        lo[0] > -10.0,
        "an influencing Ln slot must still tighten the lower bound to the log domain; box is [{}, {}]",
        lo[0],
        hi[0]
    );
    assert!(
        lo[0] <= 0.0,
        "and must not cut x just above 0: lo = {}",
        lo[0]
    );
    assert!(report.bound_updates >= 1);
    assert_eq!(report.infeasibility_witness, None);
}

/// A slot can be dead on one row and live on another when the two share a
/// pool, so the marking must be per-tape rather than global.
///
/// The two rows are chosen so the mistake is *visible*: row 0's dead slot is
/// `sqrt(x - 5)`, which fabricates `x >= 5` if propagated; row 1's live
/// `ln(x) <= 5` legitimately gives `x > 0`. A per-pool (or absent) marking
/// leaves `lo == 5`, five units tighter than the truth, so the legitimate
/// tightening cannot mask the fabricated one.
#[test]
fn influence_is_decided_per_tape_not_per_pool() {
    struct TwoRows;
    impl ExpressionProvider for TwoRows {
        fn constraint_expression(&self, i: usize) -> Option<FbbtTape> {
            match i {
                // row 0: root = x + 0; slots 1..3 are dead sqrt(x - 5)
                0 => Some(FbbtTape {
                    ops: vec![
                        FbbtOp::Var(0),     // 0
                        FbbtOp::Const(5.0), // 1  dead
                        FbbtOp::Sub(0, 1),  // 2  dead
                        FbbtOp::Sqrt(2),    // 3  dead — would force x >= 5
                        FbbtOp::Const(0.0), // 4
                        FbbtOp::Add(0, 4),  // 5  root
                    ],
                }),
                // row 1: ln(x) <= 5, live — legitimately forces x > 0
                1 => Some(FbbtTape {
                    ops: vec![FbbtOp::Var(0), FbbtOp::Ln(0)],
                }),
                _ => None,
            }
        }
    }
    let mut lo = vec![-10.0];
    let mut hi = vec![10.0];
    let report = run_fbbt(
        &TwoRows,
        1,
        2,
        &mut lo,
        &mut hi,
        &[-10.0, -INF],
        &[10.0, 5.0],
        None,
        &FbbtConfig::default(),
    );
    assert!(
        lo[0] < 1.0,
        "row 0's dead sqrt(x - 5) fabricated a bound: lo = {} (correct: the log domain, ~0)",
        lo[0]
    );
    // Row 1 does tighten — the outward rounding puts the result one ULP
    // below zero (`-5e-324`), not at zero, which is the whole point of
    // rounding outward.
    assert!(
        lo[0] > -1e-300,
        "row 1 legitimately forces x > 0: lo = {}",
        lo[0]
    );
    assert!(hi[0] <= 10.0);
    assert_eq!(report.infeasibility_witness, None);
}

// ---------------------------------------------------------------------------
// F-2 — non-finite endpoints surviving the outward rounding
// ---------------------------------------------------------------------------

/// `exp(x)` on `[800, 900]` overflows: `exp(800) = +inf` in `f64`. The old
/// rounding left both endpoints alone, so the forward interval degenerated
/// to the *point* `[inf, inf]`, which intersects nothing and emptied every
/// operand. Rounding a `+inf` lower endpoint **down** to `f64::MAX` is the
/// sound repair — the true value is somewhere at or above `f64::MAX`.
///
/// The row here is free (`g ∈ [-inf, +inf]`), so no tightening whatsoever is
/// derivable and any witness is false by construction.
#[test]
fn an_overflowed_forward_interval_is_not_an_infeasibility() {
    let (lo, hi, report) = tighten(
        vec![FbbtOp::Var(0), FbbtOp::Exp(0)],
        &[800.0],
        &[900.0],
        -INF,
        INF,
    );
    assert_eq!(
        report.infeasibility_witness, None,
        "a free row cannot be infeasible"
    );
    assert_eq!((lo[0], hi[0]), (800.0, 900.0));

    // And the interval itself is a genuine enclosure, not a degenerate point.
    let slots = forward_pass(
        &FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::Exp(0)],
        },
        &[800.0],
        &[900.0],
    )
    .expect("forward pass");
    let root = forward_result(&slots);
    assert_eq!(
        root.lo,
        f64::MAX,
        "an overflowed lower endpoint rounds down to f64::MAX"
    );
    assert_eq!(root.hi, INF);
    assert!(root.lo < root.hi, "not the degenerate point [inf, inf]");
}

/// `∞ − ∞` is `NaN`, and a `NaN` endpoint had been propagated verbatim.
/// A `NaN` means the arithmetic destroyed the value, so the sound enclosure
/// is `[-inf, +inf]` — **not** `EMPTY`, which would claim infeasibility.
#[test]
fn a_nan_producing_row_widens_to_entire_rather_than_emptying() {
    for (name, ops) in [
        (
            "exp(x) - exp(x)",
            vec![FbbtOp::Var(0), FbbtOp::Exp(0), FbbtOp::Sub(1, 1)],
        ),
        (
            "exp(x) / exp(x)",
            vec![FbbtOp::Var(0), FbbtOp::Exp(0), FbbtOp::Div(1, 1)],
        ),
    ] {
        let slots =
            forward_pass(&FbbtTape { ops: ops.clone() }, &[800.0], &[900.0]).expect("forward pass");
        assert!(
            slots.iter().all(|i| !i.lo.is_nan() && !i.hi.is_nan()),
            "{name}: a NaN endpoint escaped the forward pass"
        );
        let root = forward_result(&slots);
        assert!(
            root.lo.is_infinite() && root.lo < 0.0,
            "{name}: root.lo = {}",
            root.lo
        );
        assert!(
            root.hi.is_infinite() && root.hi > 0.0,
            "{name}: root.hi = {}",
            root.hi
        );

        // `x - x == 1` is unsatisfiable in exact arithmetic, but FBBT is not
        // entitled to say so from an interval it could not evaluate: a
        // missed infeasibility is sound, a fabricated one is not.
        let (_, _, report) = tighten(ops, &[800.0], &[900.0], 1.0, 1.0);
        assert_eq!(report.infeasibility_witness, None, "{name}");
    }
}

/// The overflow does not need transcendentals: `x + y` on `[1e308, 1.5e308]²`
/// overflows in plain addition, and the row is feasible.
#[test]
fn an_overflowing_sum_on_a_feasible_row_is_not_an_infeasibility() {
    let (_, _, report) = tighten(
        vec![FbbtOp::Var(0), FbbtOp::Var(1), FbbtOp::Add(0, 1)],
        &[1e308, 1e308],
        &[1.5e308, 1.5e308],
        1.0,
        INF,
    );
    assert_eq!(report.infeasibility_witness, None);
}

// ---------------------------------------------------------------------------
// F-1 — `n`-th roots
// ---------------------------------------------------------------------------

/// `1.0/3.0` is relatively 5.55e-17 below the true third, and `powf` computes
/// `exp(ln(x)·(1/n))`, so the root comes back short by `|ln x| · 5.55e-17` —
/// far more than the one-ULP pad the old code applied. At `x = 1024`
/// (`ln x ≈ 6.93`) that is ~10 ULP, and the returned box cut the exact
/// solution out of a well-scaled, feasible problem.
#[test]
fn the_odd_root_encloses_an_exactly_representable_solution() {
    // 1024³ = 2³⁰ = 1073741824, exact in f64.
    let (lo, hi, report) = tighten(
        vec![FbbtOp::Var(0), FbbtOp::PowInt(0, 3)],
        &[0.0],
        &[2000.0],
        1073741824.0,
        1073741824.0,
    );
    assert_eq!(
        report.infeasibility_witness, None,
        "the problem is feasible at x = 1024"
    );
    assert!(
        lo[0] <= 1024.0 && hi[0] >= 1024.0,
        "x = 1024 was cut: box is [{}, {}] (pre-fix: [1023.99999999999955, 1023.99999999999977])",
        lo[0],
        hi[0]
    );
    // Still a tightening, not a giving-up.
    assert!(
        hi[0] - lo[0] < 1e-6,
        "enclosure is loose: width {}",
        hi[0] - lo[0]
    );
}

/// The even branch is separate code — it takes the root of `|z|` and mirrors
/// it — so it gets its own case, and both signs of the answer are checked.
#[test]
fn the_even_root_encloses_both_signs() {
    let (lo, hi, _) = tighten(
        vec![FbbtOp::Var(0), FbbtOp::PowInt(0, 2)],
        &[-100.0],
        &[100.0],
        4.0,
        4.0,
    );
    assert!(
        lo[0] <= -2.0 && hi[0] >= 2.0,
        "box [{}, {}] must contain ±2",
        lo[0],
        hi[0]
    );
}

/// The odd branch on a negative right-hand side: `x³ = -8`.
#[test]
fn the_odd_root_encloses_a_negative_solution() {
    let (lo, hi, report) = tighten(
        vec![FbbtOp::Var(0), FbbtOp::PowInt(0, 3)],
        &[-10.0],
        &[10.0],
        -8.0,
        -8.0,
    );
    assert_eq!(report.infeasibility_witness, None);
    assert!(
        lo[0] <= -2.0 && hi[0] >= -2.0,
        "box [{}, {}] must contain -2",
        lo[0],
        hi[0]
    );
}

/// A root at the top of the double range, where `|ln x|` is largest and the
/// old relative shortfall reached ~7.9e-14.
#[test]
fn a_root_near_the_top_of_the_range_is_still_enclosed() {
    let x = 2.0_f64.powi(90);
    let z = x * x * x; // 2²⁷⁰, exact
    let (lo, hi, report) = tighten(
        vec![FbbtOp::Var(0), FbbtOp::PowInt(0, 3)],
        &[0.0],
        &[1e30],
        z,
        z,
    );
    assert_eq!(report.infeasibility_witness, None);
    assert!(
        lo[0] <= x && hi[0] >= x,
        "box [{lo:?}, {hi:?}] must contain 2^90",
        lo = lo[0],
        hi = hi[0]
    );
}

/// `PowInt`'s `n` is a `u32`. The old code wrote `n as i32`, which wraps for
/// `n > i32::MAX` — `u32::MAX as i32 == -1`, turning `xⁿ` into `1/x` in the
/// forward pass. `validate_fbbt_tape` does not cap `n` (the `.nl` translator
/// does, at 64, which is why this never surfaced), so the conversion is
/// checked at the point of use instead.
///
/// `x^(2³¹+1) ≥ 1` on `[2, 3]` is a tautology: every point satisfies it, so
/// no witness and no tightening are derivable.
#[test]
fn a_huge_exponent_does_not_wrap_into_a_negative_power() {
    for n in [2u32.pow(31) + 1, u32::MAX] {
        let (lo, hi, report) = tighten(
            vec![FbbtOp::Var(0), FbbtOp::PowInt(0, n)],
            &[2.0],
            &[3.0],
            1.0,
            INF,
        );
        assert_eq!(
            report.infeasibility_witness, None,
            "n = {n}: x^n ≥ 1 holds everywhere on [2, 3] (pre-fix: n as i32 wrapped negative)"
        );
        assert!(
            lo[0] <= 2.0 && hi[0] >= 3.0,
            "n = {n}: box [{}, {}] narrowed",
            lo[0],
            hi[0]
        );
    }
}

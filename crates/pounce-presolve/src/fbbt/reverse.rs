//! Reverse propagation through an [`FbbtTape`] (issue [#62]).
//!
//! Given the per-slot interval bag produced by the forward pass and a
//! target interval on the *root* slot (the constraint's `[g_lb,
//! g_ub]`), the reverse pass walks the tape backwards. For each op,
//! we ask: "given the parent's tightened interval and each
//! operand's current forward interval, what tighter interval can
//! each operand have?" The result is a tightened per-slot interval
//! bag, from which the orchestrator reads back the tightened
//! variable bounds.
//!
//! ## Per-operator inverse rules
//!
//! Each rule below is sound — it returns intervals that contain
//! every feasible operand value, so we never drop a feasible point.
//! Some operators (sin, cos) don't have a tractable interval
//! inverse; the rules for those leave the operand unchanged
//! ("decline to tighten").
//!
//! See Belotti, Cafieri, Lee, Liberti (2010), §3, for the canonical
//! list. The implementations here intersect the inverse with the
//! current forward interval, which is the standard FBBT step.
//!
//! [#62]: https://github.com/jkitchin/pounce/issues/62
//! [`FbbtTape`]: pounce_nlp::FbbtTape

use pounce_common::types::Number;
use pounce_nlp::expression_provider::{FbbtOp, FbbtTape};

use crate::fbbt::interval::{Interval, powi, round_down, round_up};

/// Result of [`reverse_pass`].
#[derive(Debug, Clone, PartialEq)]
pub struct ReverseResult {
    /// Per-slot tightened interval. Same length as `tape.ops`. Entry
    /// `i` is the intersection of the forward interval with whatever
    /// constraints reverse-propagation pushed onto slot `i`.
    pub slots: Vec<Interval>,
    /// `true` if the root interval intersected with the constraint
    /// bound was empty — i.e. FBBT detected that **this constraint
    /// is infeasible at the current variable box**. The orchestrator
    /// flags this back to the caller as a presolve-detected
    /// infeasibility; downstream slots are irrelevant in that case.
    pub infeasible: bool,
}

/// Walk `tape` in reverse, propagating the constraint bound
/// `con_bound` (the `[g_lb, g_ub]` of the constraint this tape
/// represents) into each slot. Returns the per-slot tightened
/// intervals.
///
/// The forward pass MUST have been run first (`forward.len() ==
/// tape.ops.len()`); we do not recompute it here.
pub fn reverse_pass(tape: &FbbtTape, forward: &[Interval], con_bound: Interval) -> ReverseResult {
    assert_eq!(
        forward.len(),
        tape.ops.len(),
        "forward bag length must match tape"
    );
    if tape.ops.is_empty() {
        return ReverseResult {
            slots: Vec::new(),
            infeasible: con_bound.is_empty(),
        };
    }

    let mut slots = forward.to_vec();
    // Seed: intersect root with the constraint's bound.
    let root_idx = slots.len() - 1;
    let new_root = slots[root_idx].intersect(con_bound);
    if new_root.is_empty() {
        return ReverseResult {
            slots,
            infeasible: true,
        };
    }
    slots[root_idx] = new_root;

    // Walk backward, but only out of slots the root's value actually
    // depends on. See `influencing_slots` — propagating out of the others
    // is how gh #877 turned a one-variable LP with optimum `x = -10` into a
    // `SolveSucceeded` at `x = -1e-8`.
    let influences = influencing_slots(tape);
    for i in (0..tape.ops.len()).rev() {
        if !influences[i] {
            continue;
        }
        let parent = slots[i];
        if parent.is_empty() {
            // Infeasible somewhere; no point pushing further.
            return ReverseResult {
                slots,
                infeasible: true,
            };
        }
        apply_inverse(&tape.ops[i], parent, &mut slots);
    }
    ReverseResult {
        slots,
        infeasible: false,
    }
}

/// Which tape slots the **root's value** depends on.
///
/// `FbbtTape`'s contract (`pounce-nlp`'s `expression_provider`) is that the
/// value of the whole tape is the value at the last slot. A slot the root
/// does not depend on therefore contributes nothing to the constraint, and
/// reverse-propagating out of it asserts something the constraint does not
/// say. `reverse_pass` used to walk **every** slot unconditionally, and
/// `validate_fbbt_tape` requires only that operand references point
/// backward — it does not require reachability. That combination is
/// gh #877 F-3, and it is a wrong answer under `SolveSucceeded`:
///
/// ```text
/// minimize x   s.t.  g(x) = x ∈ [-10, 10],  x ∈ [-10, 10]
/// tape: 0: Var(0)   1: Ln(0) ← dead   2: Const(0.0)   3: Add(0,2) ← root
///
/// fbbt=no  : SolveSucceeded  x = -10.000000098702795
/// fbbt=yes : SolveSucceeded  x =  -9.990002698385514e-9
/// ```
///
/// The tape exactly restates `constraints()` at its root, so the builder's
/// value check passes; the dead `Ln` slot's forward interval is clipped to
/// the log domain (the right *forward* answer, and the standard FBBT
/// convention when the sub-expression is part of the constraint), and
/// pushing that clip back out of a slot the root ignores fabricates
/// `x ≥ 0`. Wrong by the entire feasible range, `infeasibility_witness =
/// None`, no diagnostic.
///
/// Dead slots are not exotic. The tape format is *sold* on folding common
/// subexpressions into a shared pool, and a producer that emits one tape per
/// row from one pool will routinely carry slots the current root does not
/// reach. The `.nl` translator emits reachable slots only, which is why the
/// engine's unstated assumption held until hand-written tapes were accepted.
///
/// **Reachability alone is not the whole fix**, which is why this is phrased
/// as dependence of the *value*. `PowInt(a, 0)` is `1` for every `a`
/// including `NaN` (`f64::NAN.powi(0) == 1.0`), so `ln(x)^0 == 1` is a
/// tautology whose slot `Ln(x)` *is* reachable from the root and still
/// carries no information about `x` — and it cut `x = -5` out of
/// `[-10, 10]` just as the dead slot did. `inverse_powint` already declines
/// to tighten `a` when `n == 0`; what it could not do is stop the loop
/// visiting `a`'s own slot afterwards with `a`'s forward interval as a
/// "tightened parent". So the marking below cuts the dependency at
/// `PowInt(_, 0)` rather than following it.
///
/// The control case is the point of the whole rule: `ln(x) <= 5` tightens
/// `x` to `x > 0` and that is *correct*. The same tightening is right in one
/// tape and wrong in the other, and the only thing that distinguishes them
/// is whether the root's value depends on the slot.
fn influencing_slots(tape: &FbbtTape) -> Vec<bool> {
    let n = tape.ops.len();
    let mut influences = vec![false; n];
    if n == 0 {
        return influences;
    }
    influences[n - 1] = true;
    // One backward sweep suffices: every operand reference points strictly
    // backward (`validate_fbbt_tape` enforces it, and `first_invalid_slot`
    // is how), so a slot's own mark is final by the time we read it.
    for i in (0..n).rev() {
        if !influences[i] {
            continue;
        }
        match tape.ops[i] {
            FbbtOp::Const(_) | FbbtOp::Var(_) | FbbtOp::Opaque => {}
            FbbtOp::Add(a, b) | FbbtOp::Sub(a, b) | FbbtOp::Mul(a, b) | FbbtOp::Div(a, b) => {
                if a < n {
                    influences[a] = true;
                }
                if b < n {
                    influences[b] = true;
                }
            }
            FbbtOp::Neg(a)
            | FbbtOp::Sqrt(a)
            | FbbtOp::Exp(a)
            | FbbtOp::Ln(a)
            | FbbtOp::Abs(a)
            | FbbtOp::Sin(a)
            | FbbtOp::Cos(a) => {
                if a < n {
                    influences[a] = true;
                }
            }
            // `a⁰ == 1` for every `a`, so the root's value does not depend
            // on `a` through this slot. Cutting the edge here is what makes
            // the rule "influences the value" rather than "is reachable".
            FbbtOp::PowInt(a, exponent) => {
                if exponent != 0 && a < n {
                    influences[a] = true;
                }
            }
        }
    }
    influences
}

/// Push the parent's tightened interval back into the operand slots
/// per the inverse rule for `op`. Mutates `slots` in place.
fn apply_inverse(op: &FbbtOp, parent: Interval, slots: &mut [Interval]) {
    match *op {
        FbbtOp::Const(_) | FbbtOp::Var(_) | FbbtOp::Opaque => {
            // Leaves and Opaque: nothing to push into.
        }
        FbbtOp::Add(a, b) => {
            // a + b = z → a ⊆ z - b, b ⊆ z - a.
            let ai = slots[a];
            let bi = slots[b];
            slots[a] = ai.intersect(parent.sub(bi));
            // Recompute the "b ⊆ z - a" arm with the freshly
            // tightened ai (Gauss-Seidel-style FBBT — Belotti §3.2).
            slots[b] = bi.intersect(parent.sub(slots[a]));
        }
        FbbtOp::Sub(a, b) => {
            // a - b = z → a ⊆ z + b, b ⊆ a - z.
            let ai = slots[a];
            let bi = slots[b];
            slots[a] = ai.intersect(parent.add(bi));
            slots[b] = bi.intersect(slots[a].sub(parent));
        }
        FbbtOp::Mul(a, b) => {
            // a * b = z → a ⊆ z / b (when 0 ∉ b), b ⊆ z / a.
            let ai = slots[a];
            let bi = slots[b];
            if !bi.contains_zero() {
                slots[a] = ai.intersect(parent.div(bi));
            }
            // Use the (possibly) tightened a.
            let ai2 = slots[a];
            if !ai2.contains_zero() {
                slots[b] = bi.intersect(parent.div(ai2));
            }
        }
        FbbtOp::Div(a, b) => {
            // a / b = z → a ⊆ z * b. The inverse for b is only
            // useful when 0 ∉ z, since `b ⊆ a / z` requires a
            // divisor disjoint from zero — same condition we already
            // imposed on the forward Div, modulo signs.
            let ai = slots[a];
            let bi = slots[b];
            slots[a] = ai.intersect(parent.mul(bi));
            if !parent.contains_zero() {
                slots[b] = bi.intersect(slots[a].div(parent));
            }
        }
        FbbtOp::Neg(a) => {
            let ai = slots[a];
            slots[a] = ai.intersect(parent.neg());
        }
        FbbtOp::Sqrt(a) => {
            // sqrt(a) = z, z ≥ 0 → a ⊆ z².
            let ai = slots[a];
            let z_pos = parent.intersect(Interval::new(0.0, Number::INFINITY));
            if z_pos.is_empty() {
                slots[a] = Interval::EMPTY;
            } else {
                slots[a] = ai.intersect(z_pos.pow_uint(2));
            }
        }
        FbbtOp::Exp(a) => {
            // exp(a) = z, z > 0 → a ⊆ ln(z).
            let ai = slots[a];
            let z_pos = parent.intersect(Interval::new(0.0, Number::INFINITY));
            if z_pos.is_empty() || z_pos.hi <= 0.0 {
                slots[a] = Interval::EMPTY;
            } else {
                slots[a] = ai.intersect(z_pos.ln());
            }
        }
        FbbtOp::Ln(a) => {
            // ln(a) = z → a ⊆ exp(z).
            let ai = slots[a];
            slots[a] = ai.intersect(parent.exp());
        }
        FbbtOp::Abs(a) => {
            // |a| = z, z ⊆ [0, ∞] → a ⊆ [-z.hi, z.hi].
            let ai = slots[a];
            let z_nonneg = parent.intersect(Interval::new(0.0, Number::INFINITY));
            if z_nonneg.is_empty() {
                slots[a] = Interval::EMPTY;
            } else {
                let envelope = Interval::new(-z_nonneg.hi, z_nonneg.hi);
                slots[a] = ai.intersect(envelope);
            }
        }
        FbbtOp::PowInt(a, n) => {
            let ai = slots[a];
            slots[a] = ai.intersect(inverse_powint(parent, n, ai));
        }
        FbbtOp::Sin(_) | FbbtOp::Cos(_) => {
            // Periodic, multi-branch inverse — defer (no tightening).
        }
    }
}

/// `a^n = z` → tightened envelope on `a`, intersected against the
/// *prior* interval for `a` (so we get the correct branch when `n`
/// is even). Returns the envelope (an interval to intersect with the
/// current operand value).
fn inverse_powint(z: Interval, n: u32, prior_a: Interval) -> Interval {
    if z.is_empty() {
        return Interval::EMPTY;
    }
    if n == 0 {
        // a^0 = 1 — the constraint cannot tell us anything about a.
        return Interval::ENTIRE;
    }
    if n == 1 {
        return z;
    }
    if n % 2 == 1 {
        // Odd: real-valued cube/quintic/... root is monotone. Outward-round
        // the endpoints — `powf` is round-to-nearest, so without nudging the
        // lower endpoint up / upper endpoint down by a ULP we could exclude a
        // feasible point (L44, soundness invariant).
        // The enclosures are already outward-rounded; take the outer end of
        // each so the union covers every `a` whose `aⁿ` can reach `z`.
        let lo = signed_nth_root_enclosure(z.lo, n).0;
        let hi = signed_nth_root_enclosure(z.hi, n).1;
        Interval::new(lo, hi)
    } else {
        // Even: z must be non-negative.
        let z_pos = z.intersect(Interval::new(0.0, Number::INFINITY));
        if z_pos.is_empty() {
            return Interval::EMPTY;
        }
        // |a| ∈ [sqrt(z.lo), sqrt(z.hi)] (with `^(1/n)` for general
        // even n). Outward-round: `powf` is round-to-nearest, so the lower
        // root must be nudged down and the upper root up, else the
        // over-approximation could over-tighten and drop a feasible point
        // (L44, the soundness invariant the interval module promises).
        let abs_lo = nth_root_enclosure(z_pos.lo, n).0.max(0.0);
        let abs_hi = nth_root_enclosure(z_pos.hi, n).1;
        // Two branches: a ∈ [-abs_hi, -abs_lo] ∪ [abs_lo, abs_hi].
        // We can't return a union, so pick the branch that
        // intersects `prior_a` (the orchestrator-typical case). If
        // both branches intersect, fall back to the convex hull
        // [-abs_hi, abs_hi].
        let pos_branch = Interval::new(abs_lo, abs_hi);
        let neg_branch = Interval::new(-abs_hi, -abs_lo);
        let pos_hit = !prior_a.intersect(pos_branch).is_empty();
        let neg_hit = !prior_a.intersect(neg_branch).is_empty();
        match (pos_hit, neg_hit) {
            (true, false) => pos_branch,
            (false, true) => neg_branch,
            // Both branches feasible — return their hull (the
            // smallest single interval containing both).
            (true, true) => Interval::new(-abs_hi, abs_hi),
            // Neither branch hits — operand is empty.
            (false, false) => Interval::EMPTY,
        }
    }
}

/// How many ULPs the verification loop in [`nth_root_enclosure`] will walk
/// before giving up and widening by a relative amount instead.
///
/// The Newton step lands within a couple of ULPs on every case measured, so
/// this is a termination guard and not a working budget. It exists because
/// `powi`'s own error grows with `n`, and for a large enough `n` no f64
/// endpoint satisfies the bracketing test exactly — walking ULPs forever
/// instead of returning a wider, still-sound answer would be the wrong
/// trade.
const ROOT_MAX_ULPS: u32 = 64;

/// Relative widening applied when the verification loop hits its cap. Chosen
/// to dominate `|ln x| · ε` at the top of the double range
/// (`709 · 1.1e-16 ≈ 7.9e-14`) with three orders to spare.
const ROOT_FALLBACK_REL: Number = 1e-10;

/// A **sound** f64 enclosure `[lo, hi]` of `x^(1/n)` for `x ≥ 0`, `n ≥ 1`:
/// `lo^n ≤ x ≤ hi^n` under the same `powi` the forward pass uses.
///
/// The old computation was `x.powf(1.0 / n as f64)` padded by one ULP, and
/// it is short by far more than one ULP for two compounding reasons
/// (gh #877 F-1):
///
/// 1. `1.0 / n` is not exact for any `n` that is not a power of two.
///    `1.0/3.0` is relatively `5.55e-17` **below** the true third, and
///    `powf` computes `exp(ln(x) · (1/n))`, so the answer is short by
///    relatively `|ln x| · 5.55e-17` — which is `10` ULP at `x = 2^90`, not
///    one, and grows to ~`7.9e-14` relative at the top of the range.
/// 2. `powf` itself is only round-to-nearest, adding its own ULP.
///
/// One ULP of padding therefore does not restore the enclosure, and the
/// result *cuts feasible points*: `x³ == 1073741824` on `x ∈ [0, 2000]`
/// returned `[1023.99999999999955, 1023.99999999999977]`, which excludes the
/// exact answer `x = 1024`, and the row was then reported infeasible.
///
/// The repair is the one the issue proposes — a Newton correction on the
/// f64 seed — followed by **verification**: each endpoint is walked outward
/// until it actually brackets `x` under `powi`, then one ULP further.
/// Verifying against `powi` rather than against a real-arithmetic ideal is
/// deliberate: `powi` is what the forward pass evaluates, so this makes the
/// reverse rule unable to cut a point the forward rule would have accepted,
/// which is the consistency FBBT's soundness argument actually needs.
fn nth_root_enclosure(x: Number, n: u32) -> (Number, Number) {
    debug_assert!(n >= 1);
    if x.is_nan() {
        return (Number::NEG_INFINITY, Number::INFINITY);
    }
    if x <= 0.0 {
        // Callers pass `x ≥ 0`; `0` and the `-0.0` that `next_down` can
        // produce both root to 0.
        return (0.0, 0.0);
    }
    if x.is_infinite() {
        return (Number::INFINITY, Number::INFINITY);
    }

    let seed = x.powf(1.0 / n as Number);
    if !seed.is_finite() || seed <= 0.0 {
        // Under/overflowed out of the f64 range; nothing to refine, and the
        // widening below would not be meaningful.
        return (seed, seed);
    }

    // One Newton step on `f(r) = rⁿ − x`: `r ← r − (rⁿ − x) / (n·rⁿ⁻¹)`.
    // From a seed already correct to ~1e-13 relative this lands within a
    // couple of ULPs. Every intermediate is checked, because `rⁿ⁻¹·n`
    // overflows for a large `n` and a `NaN` correction must not be taken.
    let mut r = seed;
    let f = powi(r, n) - x;
    let d = powi(r, n - 1) * n as Number;
    if f.is_finite() && d.is_finite() && d > 0.0 {
        let cand = r - f / d;
        if cand.is_finite() && cand > 0.0 {
            r = cand;
        }
    }

    let mut lo = r;
    let mut steps = 0;
    while lo > 0.0 && powi(lo, n) > x && steps < ROOT_MAX_ULPS {
        lo = lo.next_down();
        steps += 1;
    }
    if steps == ROOT_MAX_ULPS {
        lo = r * (1.0 - ROOT_FALLBACK_REL);
    }

    let mut hi = r;
    steps = 0;
    while hi.is_finite() && powi(hi, n) < x && steps < ROOT_MAX_ULPS {
        hi = hi.next_up();
        steps += 1;
    }
    if steps == ROOT_MAX_ULPS {
        hi = r * (1.0 + ROOT_FALLBACK_REL);
    }

    (round_down(lo), round_up(hi))
}

/// `signum(x) · |x|^(1/n)` — the real-valued nth root for odd `n`, as a
/// sound enclosure. Returns `±∞` unchanged.
fn signed_nth_root_enclosure(x: Number, n: u32) -> (Number, Number) {
    if !x.is_finite() {
        return (x, x);
    }
    let (lo, hi) = nth_root_enclosure(x.abs(), n);
    // Negating swaps which end is which.
    if x < 0.0 { (-hi, -lo) } else { (lo, hi) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(tape: &FbbtTape, forward: &[Interval], bound: Interval) -> ReverseResult {
        reverse_pass(tape, forward, bound)
    }

    /// `x + 1 ∈ [2, 4]` ⇒ `x ⊆ [1, 3]`.
    #[test]
    fn add_constant_tightens() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::Const(1.0), FbbtOp::Add(0, 1)],
        };
        let forward = vec![
            Interval::new(-10.0, 10.0),
            Interval::point(1.0),
            Interval::new(-9.0, 11.0),
        ];
        let bound = Interval::new(2.0, 4.0);
        let r = run(&tape, &forward, bound);
        assert!(!r.infeasible);
        // Slot 0 (Var(0)) must be tightened to [1, 3].
        let v0 = r.slots[0];
        assert!(v0.lo >= 1.0 - 1e-12, "v0.lo = {}", v0.lo);
        assert!(v0.hi <= 3.0 + 1e-12, "v0.hi = {}", v0.hi);
    }

    /// `2 * x ∈ [4, 10]` ⇒ `x ⊆ [2, 5]`.
    #[test]
    fn mul_constant_tightens() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Const(2.0), FbbtOp::Var(0), FbbtOp::Mul(0, 1)],
        };
        let forward = vec![
            Interval::point(2.0),
            Interval::new(-100.0, 100.0),
            Interval::new(-200.0, 200.0),
        ];
        let bound = Interval::new(4.0, 10.0);
        let r = run(&tape, &forward, bound);
        assert!(!r.infeasible);
        let v1 = r.slots[1];
        assert!(v1.lo >= 2.0 - 1e-12);
        assert!(v1.hi <= 5.0 + 1e-12);
    }

    /// `x² ∈ [4, 9]` with `x ∈ [-10, 0]` ⇒ `x ⊆ [-3, -2]`.
    #[test]
    fn even_pow_picks_negative_branch() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::PowInt(0, 2)],
        };
        // Forward: x ∈ [-10, 0] → x² ∈ [0, 100].
        let forward = vec![Interval::new(-10.0, 0.0), Interval::new(0.0, 100.0)];
        let r = run(&tape, &forward, Interval::new(4.0, 9.0));
        assert!(!r.infeasible);
        let v0 = r.slots[0];
        assert!(v0.lo >= -3.0 - 1e-9, "got {}", v0.lo);
        assert!(v0.hi <= -2.0 + 1e-9, "got {}", v0.hi);
    }

    /// `x³ ∈ [-8, 27]` with `x ∈ [-100, 100]` ⇒ `x ⊆ [-2, 3]`.
    #[test]
    fn odd_pow_inverts_monotonically() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::PowInt(0, 3)],
        };
        let forward = vec![Interval::new(-100.0, 100.0), Interval::new(-1e6, 1e6)];
        let r = run(&tape, &forward, Interval::new(-8.0, 27.0));
        assert!(!r.infeasible);
        let v0 = r.slots[0];
        assert!(v0.lo >= -2.0 - 1e-9, "got {}", v0.lo);
        assert!(v0.hi <= 3.0 + 1e-9, "got {}", v0.hi);
    }

    /// `sqrt(x) ∈ [1, 2]` ⇒ `x ⊆ [1, 4]`.
    #[test]
    fn sqrt_inverse() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::Sqrt(0)],
        };
        let forward = vec![Interval::new(-10.0, 100.0), Interval::new(0.0, 10.0)];
        let r = run(&tape, &forward, Interval::new(1.0, 2.0));
        assert!(!r.infeasible);
        let v0 = r.slots[0];
        assert!(v0.lo >= 1.0 - 1e-12);
        assert!(v0.hi <= 4.0 + 1e-12);
    }

    /// `exp(x) ∈ [1, e]` ⇒ `x ⊆ [0, 1]`.
    #[test]
    fn exp_inverse() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::Exp(0)],
        };
        let forward = vec![Interval::new(-10.0, 10.0), Interval::new(0.0, 1.0e5)];
        let r = run(&tape, &forward, Interval::new(1.0, std::f64::consts::E));
        assert!(!r.infeasible);
        let v0 = r.slots[0];
        assert!(v0.lo >= 0.0 - 1e-12);
        assert!(v0.hi <= 1.0 + 1e-12);
    }

    /// `ln(x) ∈ [0, 1]` ⇒ `x ⊆ [1, e]`.
    #[test]
    fn ln_inverse() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::Ln(0)],
        };
        let forward = vec![Interval::new(0.5, 100.0), Interval::new(-1.0, 5.0)];
        let r = run(&tape, &forward, Interval::new(0.0, 1.0));
        assert!(!r.infeasible);
        let v0 = r.slots[0];
        assert!(v0.lo >= 1.0 - 1e-12);
        assert!(v0.hi <= std::f64::consts::E + 1e-12);
    }

    /// `|x| ∈ [0, 2]` with `x ∈ [-10, 10]` ⇒ `x ⊆ [-2, 2]`.
    #[test]
    fn abs_inverse_envelope() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::Abs(0)],
        };
        let forward = vec![Interval::new(-10.0, 10.0), Interval::new(0.0, 10.0)];
        let r = run(&tape, &forward, Interval::new(0.0, 2.0));
        assert!(!r.infeasible);
        let v0 = r.slots[0];
        assert!(v0.lo >= -2.0 - 1e-12);
        assert!(v0.hi <= 2.0 + 1e-12);
    }

    /// `(x + y) ∈ [1, 1]` with `x, y ∈ [0, 1]` ⇒ both tighten to
    /// `[0, 1]`. Already at the box; reverse pass shouldn't widen.
    #[test]
    fn add_already_tight_does_not_widen() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::Var(1), FbbtOp::Add(0, 1)],
        };
        let forward = vec![
            Interval::new(0.0, 1.0),
            Interval::new(0.0, 1.0),
            Interval::new(0.0, 2.0),
        ];
        let r = run(&tape, &forward, Interval::point(1.0));
        assert!(!r.infeasible);
        assert!(r.slots[0].lo >= 0.0 && r.slots[0].hi <= 1.0);
        assert!(r.slots[1].lo >= 0.0 && r.slots[1].hi <= 1.0);
    }

    /// Infeasible: `x ∈ [10, 20]` but constraint says `x ∈ [1, 5]`.
    #[test]
    fn root_disjoint_from_bound_is_infeasible() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0)],
        };
        let forward = vec![Interval::new(10.0, 20.0)];
        let r = run(&tape, &forward, Interval::new(1.0, 5.0));
        assert!(r.infeasible);
    }

    /// Opaque slot blocks tightening.
    #[test]
    fn opaque_does_not_propagate() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::Opaque, FbbtOp::Add(0, 1)],
        };
        let forward = vec![Interval::new(0.0, 10.0), Interval::ENTIRE, Interval::ENTIRE];
        let r = run(&tape, &forward, Interval::new(5.0, 5.0));
        assert!(!r.infeasible);
        // Slot 0 still gets some info: x + (anything) = 5 → x ⊆ ?
        // Since opaque is ENTIRE, x is unconstrained — slot 0 stays
        // [0, 10] (the forward bound).
        assert_eq!(r.slots[0], Interval::new(0.0, 10.0));
    }

    /// Soundness fuzz: tighten and resample. Every sample that
    /// satisfies the constraint at the *original* box must still lie
    /// inside the *tightened* per-variable interval. (i.e. FBBT
    /// can't drop a feasible point.)
    #[test]
    fn fuzz_no_overtightening_quadratic_sum() {
        // (x² + y²) = 5, x ∈ [-3, 3], y ∈ [-3, 3].
        let tape = FbbtTape {
            ops: vec![
                FbbtOp::Var(0),
                FbbtOp::PowInt(0, 2),
                FbbtOp::Var(1),
                FbbtOp::PowInt(2, 2),
                FbbtOp::Add(1, 3),
            ],
        };
        let forward =
            crate::fbbt::forward::forward_pass(&tape, &[-3.0, -3.0], &[3.0, 3.0]).unwrap();
        let r = run(&tape, &forward, Interval::point(5.0));
        assert!(!r.infeasible);

        // For random (x, y) with x² + y² = 5 (sampled on the unit
        // circle, rescaled by sqrt(5)), check both fall in the
        // tightened envelope.
        let var0 = r.slots[0];
        let var1 = r.slots[2];
        let n_samples = 36;
        for k in 0..n_samples {
            let theta = (k as Number) * std::f64::consts::TAU / (n_samples as Number);
            let x = (5.0_f64).sqrt() * theta.cos();
            let y = (5.0_f64).sqrt() * theta.sin();
            assert!(
                var0.contains(x),
                "x={x:.3} not in {:?} (theta={theta})",
                var0
            );
            assert!(
                var1.contains(y),
                "y={y:.3} not in {:?} (theta={theta})",
                var1
            );
        }
    }

    /// L44: the even-`n` inverse-power root must be *outward*-rounded —
    /// `powf` is round-to-nearest, so without nudging the lower root down /
    /// the upper root up the over-approximation could exclude a feasible
    /// point. Using a perfect-square interval `[4, 9]` (roots exactly 2, 3)
    /// the bug returns `[2, 3]` exactly; the fix returns a strictly wider box.
    #[test]
    fn inverse_powint_even_branch_is_outward_rounded() {
        let raw_lo = 4.0_f64.powf(0.5);
        let raw_hi = 9.0_f64.powf(0.5);
        // prior_a = [0, ∞) selects the positive branch only.
        let r = inverse_powint(
            Interval::new(4.0, 9.0),
            2,
            Interval::new(0.0, Number::INFINITY),
        );
        assert!(
            r.lo < raw_lo,
            "lower root must be rounded below {raw_lo}, got {} (fail-first: assignment leaves it == {raw_lo})",
            r.lo,
        );
        assert!(
            r.hi > raw_hi,
            "upper root must be rounded above {raw_hi}, got {}",
            r.hi
        );
        // Still sound: contains the exact roots.
        assert!(r.contains(2.0) && r.contains(3.0));
    }

    /// The odd-`n` branch carries the same outward-rounding requirement.
    #[test]
    fn inverse_powint_odd_branch_is_outward_rounded() {
        // The un-rounded roots. `signed_nth_root` used to *be* this expression
        // plus one ULP; gh #877 replaced it with `signed_nth_root_enclosure`,
        // which already rounds outward, so naming the raw value here keeps the
        // assertion below a statement about the enclosure rather than a
        // tautology about itself. Both are exact in `f64` (checked: `powf`
        // returns exactly 2.0 and 3.0 for these arguments).
        let raw_lo = 8.0_f64.powf(1.0 / 3.0);
        let raw_hi = 27.0_f64.powf(1.0 / 3.0);
        let r = inverse_powint(Interval::new(8.0, 27.0), 3, Interval::ENTIRE);
        assert!(
            r.lo < raw_lo,
            "odd lower root not outward-rounded: {} vs {raw_lo}",
            r.lo
        );
        assert!(
            r.hi > raw_hi,
            "odd upper root not outward-rounded: {} vs {raw_hi}",
            r.hi
        );
    }
}

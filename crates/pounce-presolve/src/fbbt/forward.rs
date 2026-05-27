//! Forward interval pass over an [`FbbtTape`].
//!
//! Given current variable bounds `x_lo[i] ≤ x_i ≤ x_hi[i]` and a
//! constraint expression tape, computes an interval over-approximation
//! of the value at every slot of the tape. The output is a parallel
//! `Vec<Interval>` whose last entry is the over-approximation of the
//! whole constraint expression. The next phase (reverse pass, commit
//! 3 of [#62]) consumes this buffer.
//!
//! The pass is one linear scan over `ops`; each op consults the
//! intervals already computed for its operand slots, so the result is
//! `O(n)` in the tape length.
//!
//! [`FbbtOp::Opaque`] slots collapse to [`Interval::ENTIRE`]: we have
//! no structural information about that subexpression, so we cannot
//! tighten anything through it.
//!
//! [#62]: https://github.com/jkitchin/pounce/issues/62
//! [`FbbtTape`]: pounce_nlp::FbbtTape
//! [`FbbtOp::Opaque`]: pounce_nlp::FbbtOp::Opaque

use pounce_common::types::Number;
use pounce_nlp::expression_provider::{FbbtOp, FbbtTape};

use crate::fbbt::interval::Interval;

/// Why a forward pass might fail.
#[derive(Debug, Clone, PartialEq)]
pub enum ForwardError {
    /// Tape failed [`FbbtTape::first_invalid_slot`] validation. The
    /// payload is the slot index that referenced something it
    /// shouldn't have.
    MalformedTape(usize),
    /// Tape mentioned `Var(j)` with `j >= x_lo.len()`. The payload is
    /// the offending variable index.
    VariableIndexOutOfRange(usize),
    /// `x_lo` and `x_hi` had different lengths.
    BoundsLengthMismatch { lo: usize, hi: usize },
}

/// Compute the per-slot interval bag for `tape`, given current
/// variable bounds `x_lo` and `x_hi` (parallel arrays). Returns a
/// `Vec<Interval>` of the same length as `tape.ops`.
///
/// On a well-formed tape this never panics — domain violations
/// (e.g. `ln` of a fully-negative interval) produce
/// [`Interval::EMPTY`] in the corresponding slot, which the
/// orchestrator interprets as "infeasibility candidate; skip
/// tightening from this constraint."
pub fn forward_pass(
    tape: &FbbtTape,
    x_lo: &[Number],
    x_hi: &[Number],
) -> Result<Vec<Interval>, ForwardError> {
    if x_lo.len() != x_hi.len() {
        return Err(ForwardError::BoundsLengthMismatch {
            lo: x_lo.len(),
            hi: x_hi.len(),
        });
    }
    if let Some(bad) = tape.first_invalid_slot() {
        return Err(ForwardError::MalformedTape(bad));
    }

    let n_vars = x_lo.len();
    let mut vals: Vec<Interval> = Vec::with_capacity(tape.ops.len());
    for op in &tape.ops {
        let v = match *op {
            FbbtOp::Const(c) => Interval::point(c),
            FbbtOp::Var(i) => {
                if i >= n_vars {
                    return Err(ForwardError::VariableIndexOutOfRange(i));
                }
                Interval::new(x_lo[i], x_hi[i])
            }
            FbbtOp::Opaque => Interval::ENTIRE,
            FbbtOp::Add(a, b) => vals[a].add(vals[b]),
            FbbtOp::Sub(a, b) => vals[a].sub(vals[b]),
            FbbtOp::Mul(a, b) => vals[a].mul(vals[b]),
            FbbtOp::Div(a, b) => vals[a].div(vals[b]),
            FbbtOp::PowInt(a, n) => vals[a].pow_uint(n),
            FbbtOp::Neg(a) => vals[a].neg(),
            FbbtOp::Sqrt(a) => vals[a].sqrt(),
            FbbtOp::Exp(a) => vals[a].exp(),
            FbbtOp::Ln(a) => vals[a].ln(),
            FbbtOp::Abs(a) => vals[a].abs(),
            FbbtOp::Sin(a) => vals[a].sin(),
            FbbtOp::Cos(a) => vals[a].cos(),
        };
        vals.push(v);
    }
    Ok(vals)
}

/// Final result of [`forward_pass`] — the interval enclosing the
/// whole constraint expression. Empty tape returns
/// [`Interval::ENTIRE`].
pub fn forward_result(vals: &[Interval]) -> Interval {
    vals.last().copied().unwrap_or(Interval::ENTIRE)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `2 * x` for `x ∈ [-1, 3]`. Result: `[-2, 6]`.
    #[test]
    fn simple_linear_combination() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Const(2.0), FbbtOp::Var(0), FbbtOp::Mul(0, 1)],
        };
        let vals = forward_pass(&tape, &[-1.0], &[3.0]).unwrap();
        let res = forward_result(&vals);
        assert!(res.contains(-2.0));
        assert!(res.contains(6.0));
    }

    /// `x^2 + y^2` for `x ∈ [-2, 1]`, `y ∈ [0, 3]`.
    /// `x^2 ∈ [0, 4]`, `y^2 ∈ [0, 9]` → sum `[0, 13]`.
    #[test]
    fn quadratic_sum() {
        let tape = FbbtTape {
            ops: vec![
                FbbtOp::Var(0),         // x
                FbbtOp::PowInt(0, 2),   // x^2
                FbbtOp::Var(1),         // y
                FbbtOp::PowInt(2, 2),   // y^2
                FbbtOp::Add(1, 3),      // x^2 + y^2
            ],
        };
        let vals = forward_pass(&tape, &[-2.0, 0.0], &[1.0, 3.0]).unwrap();
        let res = forward_result(&vals);
        assert!(res.contains(0.0), "should contain min");
        assert!(res.contains(13.0), "should contain max");
        // Outward rounding may give slightly looser bounds.
        assert!(res.lo <= 0.0);
        assert!(res.hi >= 13.0);
    }

    /// `exp(x)` for `x ∈ [0, 1]` → `[1, e]`.
    #[test]
    fn exp_monotone() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::Exp(0)],
        };
        let vals = forward_pass(&tape, &[0.0], &[1.0]).unwrap();
        let res = forward_result(&vals);
        assert!(res.contains(1.0));
        assert!(res.contains(std::f64::consts::E));
    }

    /// `ln(x)` for `x ∈ [-1, 4]` — domain straddles zero.
    /// Forward pass should produce `[-∞, ln(4)]` (we clip `lo` to 0
    /// inside `ln`).
    #[test]
    fn ln_domain_clip() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::Ln(0)],
        };
        let vals = forward_pass(&tape, &[-1.0], &[4.0]).unwrap();
        let res = forward_result(&vals);
        assert_eq!(res.lo, Number::NEG_INFINITY);
        assert!(res.hi >= std::f64::consts::LN_2 * 2.0);
    }

    /// `ln(x)` for `x ∈ [-3, -1]` — fully outside the domain. The
    /// interval is EMPTY, signalling infeasibility from this branch.
    #[test]
    fn ln_fully_outside_domain_is_empty() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::Ln(0)],
        };
        let vals = forward_pass(&tape, &[-3.0], &[-1.0]).unwrap();
        let res = forward_result(&vals);
        assert!(res.is_empty());
    }

    /// CSE-style sharing: a single `x` slot reused by two ops should
    /// still produce the correct interval (the tape representation
    /// inherently shares — every reference to slot 0 sees the same
    /// `[x_lo, x_hi]` interval).
    #[test]
    fn cse_via_tape_slot_sharing() {
        // x * x for x ∈ [-2, 3] should give [-6, 9] via Mul (not
        // [0, 9] like PowInt(2) would). This is the natural over-
        // approximation when sharing is structural, not symbolic.
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::Mul(0, 0)],
        };
        let vals = forward_pass(&tape, &[-2.0], &[3.0]).unwrap();
        let res = forward_result(&vals);
        // Looser than PowInt: this is expected — the interval
        // arithmetic forgets the shared-operand constraint.
        assert!(res.contains(-6.0));
        assert!(res.contains(9.0));
    }

    /// Opaque slot → ENTIRE.
    #[test]
    fn opaque_yields_entire() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::Opaque, FbbtOp::Add(0, 1)],
        };
        let vals = forward_pass(&tape, &[1.0], &[2.0]).unwrap();
        let res = forward_result(&vals);
        // [1, 2] + ENTIRE = ENTIRE.
        assert_eq!(res.lo, Number::NEG_INFINITY);
        assert_eq!(res.hi, Number::INFINITY);
    }

    #[test]
    fn empty_tape_yields_entire() {
        let tape = FbbtTape::new();
        let vals = forward_pass(&tape, &[], &[]).unwrap();
        assert!(vals.is_empty());
        let res = forward_result(&vals);
        assert!(res.is_entire());
    }

    #[test]
    fn malformed_tape_rejected() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Add(0, 1), FbbtOp::Const(0.0)],
        };
        let err = forward_pass(&tape, &[], &[]).unwrap_err();
        assert_eq!(err, ForwardError::MalformedTape(0));
    }

    #[test]
    fn out_of_range_var_rejected() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(2)],
        };
        let err = forward_pass(&tape, &[0.0], &[1.0]).unwrap_err();
        assert_eq!(err, ForwardError::VariableIndexOutOfRange(2));
    }

    #[test]
    fn mismatched_bounds_lengths_rejected() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Const(0.0)],
        };
        let err = forward_pass(&tape, &[0.0], &[1.0, 2.0]).unwrap_err();
        assert!(matches!(err, ForwardError::BoundsLengthMismatch { .. }));
    }

    /// Soundness: for many sample points inside the variable box,
    /// the constraint value must fall inside the forward-pass result.
    #[test]
    fn fuzz_soundness_pointwise() {
        // f(x, y) = (x - 1) * (y + 2) + sqrt(x + 10)
        let tape = FbbtTape {
            ops: vec![
                FbbtOp::Var(0),           // x
                FbbtOp::Const(1.0),
                FbbtOp::Sub(0, 1),        // x - 1
                FbbtOp::Var(1),           // y
                FbbtOp::Const(2.0),
                FbbtOp::Add(3, 4),        // y + 2
                FbbtOp::Mul(2, 5),        // (x-1)*(y+2)
                FbbtOp::Const(10.0),
                FbbtOp::Add(0, 7),        // x + 10
                FbbtOp::Sqrt(8),          // sqrt(x+10)
                FbbtOp::Add(6, 9),        // f
            ],
        };
        let x_lo = [-2.0, -1.0];
        let x_hi = [3.0, 5.0];
        let res = forward_result(&forward_pass(&tape, &x_lo, &x_hi).unwrap());

        // 25 sample points on a 5x5 grid.
        for ix in 0..5 {
            for iy in 0..5 {
                let x = x_lo[0] + (x_hi[0] - x_lo[0]) * (ix as f64) / 4.0;
                let y = x_lo[1] + (x_hi[1] - x_lo[1]) * (iy as f64) / 4.0;
                let f = (x - 1.0) * (y + 2.0) + (x + 10.0).sqrt();
                assert!(
                    res.contains(f),
                    "x={x}, y={y}, f={f} not in {:?}",
                    res
                );
            }
        }
    }
}

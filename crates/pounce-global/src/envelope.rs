//! Tight polyhedral envelopes for univariate atoms — the heart of a strong
//! relaxation.
//!
//! For `w = f(t)` on `[l, u]`, a relaxation needs linear cuts that *under*- and
//! *over*-estimate `f`. The quality of the whole spatial branch-and-bound hinges
//! on how tight these are:
//!
//! * **Convex** `f` (exp, even powers): the convex envelope *is* `f`, so any
//!   number of tangent lines underestimate; the chord (secant) is the unique
//!   concave overestimator.
//! * **Concave** `f` (ln, √): mirror image — secant under, tangents over.
//! * **Single inflection** `f` (odd powers across 0; a monotone-curvature
//!   segment of sin/cos): the exact convex/concave envelope is a
//!   *tangent line from the far endpoint* to the opposite-curvature region,
//!   meeting `f` at a tangency point found by a small root solve, plus tangent
//!   cuts on the same-curvature side. This is the construction (Liberti &
//!   Pantelides) that real global solvers use; it is what turns the previous
//!   "interval box only" fallback for odd powers and trig into a genuinely
//!   tight relaxation.
//!
//! Every cut returned is a *valid global* bound on `f` over `[l, u]`; tightness
//! improves automatically as branch-and-bound shrinks the box (the envelope is
//! exact at the box ends).

use std::f64::consts::PI;

/// Number of tangent cuts used to approximate a convex/concave arc.
const N_TANGENT: usize = 5;

/// A linear cut `slope·t + intercept`, applied as `w ≥ ·` (under) or `w ≤ ·`
/// (over).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cut {
    pub slope: f64,
    pub intercept: f64,
}

/// Under- and over-estimating linear cuts for `w = f(t)` over a box.
#[derive(Default)]
pub(crate) struct Envelope {
    pub under: Vec<Cut>,
    pub over: Vec<Cut>,
}

fn tangent(f: &impl Fn(f64) -> f64, df: &impl Fn(f64) -> f64, c: f64) -> Cut {
    let s = df(c);
    Cut {
        slope: s,
        intercept: f(c) - s * c,
    }
}

fn secant(f: &impl Fn(f64) -> f64, l: f64, u: f64) -> Cut {
    if (u - l).abs() < 1e-12 {
        return Cut {
            slope: 0.0,
            intercept: f(l),
        };
    }
    let s = (f(u) - f(l)) / (u - l);
    Cut {
        slope: s,
        intercept: f(l) - s * l,
    }
}

fn sample(l: f64, u: f64, k: usize) -> Vec<f64> {
    if k <= 1 || u - l < 1e-12 {
        return vec![0.5 * (l + u)];
    }
    (0..k)
        .map(|i| l + (u - l) * i as f64 / (k - 1) as f64)
        .collect()
}

fn tangents(
    f: &impl Fn(f64) -> f64,
    df: &impl Fn(f64) -> f64,
    l: f64,
    u: f64,
    k: usize,
) -> Vec<Cut> {
    sample(l, u, k)
        .into_iter()
        .map(|c| tangent(f, df, c))
        .collect()
}

/// Convex `f`: tangents underestimate, the chord overestimates.
fn convex(f: &impl Fn(f64) -> f64, df: &impl Fn(f64) -> f64, l: f64, u: f64) -> Envelope {
    Envelope {
        under: tangents(f, df, l, u, N_TANGENT),
        over: vec![secant(f, l, u)],
    }
}

/// Concave `f`: the chord underestimates, tangents overestimate.
fn concave(f: &impl Fn(f64) -> f64, df: &impl Fn(f64) -> f64, l: f64, u: f64) -> Envelope {
    Envelope {
        under: vec![secant(f, l, u)],
        over: tangents(f, df, l, u, N_TANGENT),
    }
}

/// Root of `h` in `[a, b]` by bisection, assuming a sign change (≤ 60 steps).
fn bisect(h: impl Fn(f64) -> f64, mut a: f64, mut b: f64) -> Option<f64> {
    let (ha, hb) = (h(a), h(b));
    if ha == 0.0 {
        return Some(a);
    }
    if hb == 0.0 {
        return Some(b);
    }
    if ha * hb > 0.0 {
        return None;
    }
    let pos_at_a = ha > 0.0;
    for _ in 0..60 {
        let m = 0.5 * (a + b);
        let hm = h(m);
        if hm == 0.0 || (b - a).abs() < 1e-12 {
            return Some(m);
        }
        if (hm > 0.0) == pos_at_a {
            a = m;
        } else {
            b = m;
        }
    }
    Some(0.5 * (a + b))
}

/// Tangency point `t ∈ [lo, hi]` where the line through `(anchor, f(anchor))`
/// touches `f`: `f'(t)·(t − anchor) = f(t) − f(anchor)`.
fn tangent_from(
    f: &impl Fn(f64) -> f64,
    df: &impl Fn(f64) -> f64,
    anchor: f64,
    lo: f64,
    hi: f64,
) -> Option<f64> {
    let fa = f(anchor);
    bisect(|t| df(t) * (t - anchor) - (f(t) - fa), lo, hi)
}

/// Exact envelope of a function that is **concave on `[l, m]` then convex on
/// `[m, u]`** (e.g. an odd power across 0, with `m = 0`).
fn concave_then_convex(
    f: &impl Fn(f64) -> f64,
    df: &impl Fn(f64) -> f64,
    l: f64,
    u: f64,
    m: f64,
) -> Envelope {
    let mut env = Envelope::default();
    // Underestimator (convex envelope): a line from the left endpoint tangent
    // to the convex side, then tangents along the convex side.
    match tangent_from(f, df, l, m, u) {
        Some(t) => {
            env.under.push(tangent(f, df, t));
            for c in sample(t, u, N_TANGENT) {
                env.under.push(tangent(f, df, c));
            }
        }
        None => env.under.push(secant(f, l, u)),
    }
    // Overestimator (concave envelope): mirror — line from the right endpoint
    // tangent to the concave side, then tangents along the concave side.
    match tangent_from(f, df, u, l, m) {
        Some(s) => {
            env.over.push(tangent(f, df, s));
            for c in sample(l, s, N_TANGENT) {
                env.over.push(tangent(f, df, c));
            }
        }
        None => env.over.push(secant(f, l, u)),
    }
    env
}

/// Exact envelope of a function that is **convex on `[l, m]` then concave on
/// `[m, u]`** (the mirror case).
fn convex_then_concave(
    f: &impl Fn(f64) -> f64,
    df: &impl Fn(f64) -> f64,
    l: f64,
    u: f64,
    m: f64,
) -> Envelope {
    let mut env = Envelope::default();
    // Underestimator: line from the right endpoint tangent to the convex side.
    match tangent_from(f, df, u, l, m) {
        Some(t) => {
            env.under.push(tangent(f, df, t));
            for c in sample(l, t, N_TANGENT) {
                env.under.push(tangent(f, df, c));
            }
        }
        None => env.under.push(secant(f, l, u)),
    }
    // Overestimator: line from the left endpoint tangent to the concave side.
    match tangent_from(f, df, l, m, u) {
        Some(s) => {
            env.over.push(tangent(f, df, s));
            for c in sample(s, u, N_TANGENT) {
                env.over.push(tangent(f, df, c));
            }
        }
        None => env.over.push(secant(f, l, u)),
    }
    env
}

/// Envelope of `x^n` (`n ≥ 2`) over `[l, u]`.
pub(crate) fn power(n: u32, l: f64, u: f64) -> Envelope {
    let ni = n as i32;
    let f = move |t: f64| t.powi(ni);
    let df = move |t: f64| n as f64 * t.powi(ni - 1);
    if n.is_multiple_of(2) || l >= 0.0 {
        convex(&f, &df, l, u) // x^even, or x^odd on the nonnegative side
    } else if u <= 0.0 {
        concave(&f, &df, l, u) // x^odd on the nonpositive side
    } else {
        concave_then_convex(&f, &df, l, u, 0.0) // x^odd straddling 0
    }
}

/// Convex envelope `exp` over `[l, u]`.
pub(crate) fn exp(l: f64, u: f64) -> Envelope {
    convex(&|t: f64| t.exp(), &|t: f64| t.exp(), l, u)
}

/// Concave envelope `ln` over `[l, u]` (clamps the domain to `t > 0`).
pub(crate) fn ln(l: f64, u: f64) -> Envelope {
    let l = l.max(1e-12);
    concave(&|t: f64| t.ln(), &|t: f64| 1.0 / t, l, u.max(l))
}

/// Concave envelope `√` over `[l, u]` (clamps the domain to `t ≥ 0`).
pub(crate) fn sqrt(l: f64, u: f64) -> Envelope {
    let l = l.max(0.0);
    concave(
        &|t: f64| t.sqrt(),
        &|t: f64| 0.5 / t.max(1e-300).sqrt(),
        l,
        u.max(l),
    )
}

/// Envelope of `sin`/`cos` over `[l, u]`.
///
/// `sin''=−sin`, `cos''=−cos`, so curvature flips sign exactly at the function's
/// zeros: convex where the value is negative, concave where positive.
///
/// * **Width ≤ π** — at most one interior inflection, so the *exact* convex/
///   concave-hull construction applies (secant + tangent / tangent-from-endpoint
///   cuts), the tightest possible relaxation.
/// * **π < width ≤ `WIDE_TRIG_MAX`** — multiple inflections; build a valid (not
///   exact) relaxation from slope-sampled supporting lines (see [`wide_trig`]),
///   which couples the atom column to its argument far better than the box.
/// * **Wider** — `None`; the caller keeps the interval box bound (already near
///   the convex hull once the function oscillates several full periods), and
///   branching shrinks the box until a tighter case applies.
pub(crate) fn trig(is_sin: bool, l: f64, u: f64) -> Option<Envelope> {
    let w = u - l;
    if w < 1e-12 {
        return None;
    }
    let f = move |t: f64| if is_sin { t.sin() } else { t.cos() };
    let df = move |t: f64| if is_sin { t.cos() } else { -t.sin() };

    if w > PI + 1e-9 {
        return wide_trig(&f, &df, l, u);
    }

    // Interior inflection = an interior zero of f (at most one for width ≤ π).
    let infl = if f(l) * f(u) < 0.0 {
        bisect(f, l, u)
    } else {
        None
    };
    Some(match infl {
        None => {
            if f(0.5 * (l + u)) < 0.0 {
                convex(&f, &df, l, u)
            } else {
                concave(&f, &df, l, u)
            }
        }
        Some(m) => {
            if f(0.5 * (l + m)) < 0.0 {
                convex_then_concave(&f, &df, l, u, m) // convex left, concave right
            } else {
                concave_then_convex(&f, &df, l, u, m) // concave left, convex right
            }
        }
    })
}

/// Widest box for which [`wide_trig`] builds a sloped relaxation. Beyond this
/// the function oscillates ≳ 3 full periods and its convex hull is essentially
/// the `[−1, 1]` box, so the interval bound loses almost nothing.
const WIDE_TRIG_MAX: f64 = 6.0 * PI;

/// A valid (not necessarily tight) relaxation of a `sin`/`cos` arc spanning
/// multiple inflections, via slope-sampled supporting lines.
///
/// For any slope `m`, the line `m·t + bₘ` with `bₘ = minₜ (f(t) − m·t)` lies
/// weakly below `f` on `[l, u]` (it touches at the minimizer); `m·t + Bₘ` with
/// `Bₘ = maxₜ (f(t) − m·t)` lies weakly above. We sample slopes `m = f'(cₖ)` at
/// several anchors `cₖ` (slope 0 recovers the box floor/ceiling) and take the
/// exact offset extrema over a fine grid, padded by the curvature bound
/// `max|f''|·Δ²/8 ≤ Δ²/8` so the lines are rigorously valid *between* grid
/// points too. Each `(under, over)` pair couples the atom column to its
/// argument — strictly more than the decoupled box bound.
fn wide_trig(
    f: &impl Fn(f64) -> f64,
    df: &impl Fn(f64) -> f64,
    l: f64,
    u: f64,
) -> Option<Envelope> {
    let w = u - l;
    if w > WIDE_TRIG_MAX {
        return None;
    }
    // ~128 samples per π keeps the curvature padding tiny (≈ 1e-4 here).
    let samples = ((w / PI) * 128.0).ceil().max(256.0) as usize;
    let dt = w / samples as f64;
    let pad = dt * dt / 8.0; // ≥ max|f''|·Δ²/8 since |f''| ≤ 1 for sin/cos.

    // Exact-to-grid offset extrema of g(t) = f(t) − m·t over [l, u].
    let offset_extrema = |m: f64| -> (f64, f64) {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for k in 0..=samples {
            let t = l + dt * k as f64;
            let g = f(t) - m * t;
            lo = lo.min(g);
            hi = hi.max(g);
        }
        // Pad outward so the lines bound f between grid points as well.
        (lo - pad, hi + pad)
    };

    const ANCHORS: usize = 6;
    let mut env = Envelope::default();
    for k in 0..=ANCHORS {
        let c = l + w * k as f64 / ANCHORS as f64;
        let m = df(c);
        let (b_under, b_over) = offset_extrema(m);
        env.under.push(Cut {
            slope: m,
            intercept: b_under,
        });
        env.over.push(Cut {
            slope: m,
            intercept: b_over,
        });
    }
    Some(env)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every under-cut lies ≤ f and every over-cut ≥ f across the interval.
    fn assert_valid(env: &Envelope, f: impl Fn(f64) -> f64, l: f64, u: f64) {
        for i in 0..=200 {
            let t = l + (u - l) * i as f64 / 200.0;
            let ft = f(t);
            for c in &env.under {
                assert!(
                    c.slope * t + c.intercept <= ft + 1e-6,
                    "under cut above f at t={t}: {} > {ft}",
                    c.slope * t + c.intercept
                );
            }
            for c in &env.over {
                assert!(
                    c.slope * t + c.intercept >= ft - 1e-6,
                    "over cut below f at t={t}: {} < {ft}",
                    c.slope * t + c.intercept
                );
            }
        }
    }

    #[test]
    fn cube_straddling_zero_is_valid() {
        let env = power(3, -1.5, 2.0);
        assert_valid(&env, |t| t.powi(3), -1.5, 2.0);
        assert!(
            env.under.len() > 1 && env.over.len() > 1,
            "should be tight, not box"
        );
    }

    #[test]
    fn even_power_convex_valid() {
        let env = power(4, -2.0, 3.0);
        assert_valid(&env, |t| t.powi(4), -2.0, 3.0);
    }

    #[test]
    fn exp_ln_sqrt_valid() {
        assert_valid(&exp(-1.0, 2.0), |t| t.exp(), -1.0, 2.0);
        assert_valid(&ln(0.5, 5.0), |t| t.ln(), 0.5, 5.0);
        assert_valid(&sqrt(0.0, 9.0), |t| t.sqrt(), 0.0, 9.0);
    }

    #[test]
    fn sine_envelopes_valid_each_curvature() {
        // concave arc [0, 3], convex arc [-3, 0], and an inflection-spanning arc.
        assert_valid(&trig(true, 0.2, 3.0).unwrap(), |t| t.sin(), 0.2, 3.0);
        assert_valid(&trig(true, -3.0, -0.2).unwrap(), |t| t.sin(), -3.0, -0.2);
        assert_valid(&trig(true, -1.0, 2.0).unwrap(), |t| t.sin(), -1.0, 2.0);
        // cosine inflection-spanning arc around π/2.
        assert_valid(&trig(false, 0.5, 2.5).unwrap(), |t| t.cos(), 0.5, 2.5);
    }

    #[test]
    fn wide_trig_is_valid_and_couples() {
        // π < width ≤ 6π: a valid (sloped) relaxation, not just the box. Check
        // validity across several multi-inflection windows for sin and cos.
        for &(is_sin, l, u) in &[
            (true, 0.0, 4.0),  // ~1.27 periods
            (true, -2.0, 5.0), // straddles several zeros
            (false, 0.5, 7.0), // cosine, wide
            (true, -8.0, 8.0), // ~2.5 periods, near the cap
        ] {
            let env = trig(is_sin, l, u).unwrap();
            let f = |t: f64| if is_sin { t.sin() } else { t.cos() };
            assert_valid(&env, f, l, u);
            // Sloped cuts (some non-zero slope) — genuinely coupling w to the
            // argument, not the decoupled box bound.
            assert!(
                env.under.iter().any(|c| c.slope.abs() > 1e-6),
                "wide-trig under cuts should include sloped lines"
            );
        }
    }

    #[test]
    fn very_wide_trig_declines() {
        // Beyond ~3 full periods the box bound is near-optimal; decline.
        assert!(trig(true, 0.0, 7.0 * std::f64::consts::PI).is_none());
    }
}

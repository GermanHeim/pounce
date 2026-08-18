//! The Q4 constant-structure evaluator against the AD tape it replaces
//! (gh #588).
//!
//! Q4 stops building a tape for every objective and row the recognizer
//! proves is degree ≤ 2, and evaluates `g`, `∇g` and `∇²L` from the constant
//! matrix instead. If those two ways of computing a derivative ever disagree,
//! POUNCE hands the algorithm a wrong Hessian and converges somewhere else —
//! or fails to — on a model no fixture solves. The fixture sweep cannot see
//! that: it only reaches rows a *solve* depends on, and it reports one number
//! per model rather than one per matrix entry.
//!
//! So this is the differential check, modelled on Q3's
//! `quad_recognizer_differential.rs`: every `.nl` file in the repository is
//! loaded **twice** — once with the fast path, once with
//! `NlTnlp::try_new_with_quadratic(prob, false)`, which is what
//! `POUNCE_DBG_NO_QUAD=1` selects — and the two are compared entry by entry
//! at several points, on:
//!
//! * `eval_g` — the line search's inner loop;
//! * `eval_jac_g` — compared as a `(row, col) -> value` map, because the two
//!   paths need not agree on the *pattern*: a coefficient that cancels to
//!   exactly zero is dropped by the recognizer and kept by the tape's
//!   structural sparsity;
//! * `eval_h` — same, over the assembled lower triangle.
//!
//! ### Which comparisons are bitwise
//!
//! `∇²L` is: the tape's entry for `0.5·((c·xᵢ)·xⱼ)` is the product of
//! constants `0.5·c` and the decode adds `w·(0.5·c)`, while the scatter
//! computes `w·q_val` with the same `q_val`. Nothing about that is an
//! *approximation* of the other. **Measured, that is right for the shape the
//! note reasoned about and wrong in general**: over this corpus 27 174 of
//! 27 176 assembled Hessian entries are bit-identical, and the two that are
//! not (`eigena2.nl`, entry (8, 8)) differ by exactly **one ulp**.
//!
//! The mechanism is worth stating because it decides where else it can
//! happen: **an entry written by both paths at once**. `eigena2` has 55
//! quadratic rows and a non-quadratic objective, so `∇²L[8, 8]` takes a
//! constant contribution from a row and an `x`-dependent one from the
//! objective's tape. On the tape path both land in the same compressed
//! column pass and reach `values` as one add; on the fast path the row's
//! share is scattered first and the objective's decode adds on top. Same
//! terms, different association, one ulp — and only where a model mixes the
//! two, which is why the entry moves with `x` even though the row's Hessian
//! does not. A model whose objective and rows are all recognized (the whole
//! `qcqp` family) has no entry with a foot in both camps.
//!
//! The disagreement is therefore bounded, not asserted away: the worst ulp
//! distance over the corpus is pinned at `MAX_HESS_ULPS`. Q1's 2-ulp line is
//! why that is a pin and not a tolerance — a one-ulp coefficient difference
//! moved a fixture from 17 to 12 conic iterations, and only a differential
//! check saw it.
//!
//! `f` and `∇f` are compared too, and that is not decoration: on a model
//! whose *rows* are not quadratic the objective is the only thing the phase
//! touches, and `infeasible_equalities.nl` (cubic rows, quadratic objective)
//! is exactly such a model. Adding those two comparisons immediately found
//! the phase's one real accuracy defect — expanding `(x − 500000)²` cancels
//! five digits where the tape squares a small residual — which is why the
//! fast path is now gated on `is_expanded_quadratic` and takes only forms
//! the writer had already expanded.
//!
//! `g` and `∇g` are **not** bitwise and the design note says so in advance:
//! the tape sums one summand at a time in file order while the matvec sums a
//! merged row, so the association differs. They are held to a tight relative
//! tolerance, and the *observed* worst deviation over the corpus is pinned as
//! a number so that a future regression has to move it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use pounce_cli::nl_reader::{NlProblem, NlTnlp, read_nl_file};
use pounce_nlp::tnlp::{SparsityRequest, TNLP};

/// Relative tolerance for the two summation orders in `eval_g` / `eval_jac_g`.
///
/// Not a bound anyone derived — a ceiling well above what the corpus
/// actually produces (see `WORST_OBSERVED_REL`, asserted separately), set
/// loose enough that reassociating a long sum cannot trip it and tight
/// enough that a wrong coefficient cannot hide under it.
const REL_TOL: f64 = 1e-12;

/// The largest relative deviation any fixture actually produces, over every
/// `g` and `∇g` entry at every probe point. Pinned so that "within
/// tolerance" stays a measurement rather than a hope.
const WORST_OBSERVED_REL: f64 = 1e-14;

/// The worst Hessian disagreement the corpus produces, in representable
/// doubles. See the module docs: the design note forecast bit-identity here
/// and the corpus refutes it in general while confirming it on the shape the
/// note reasoned about.
const MAX_HESS_ULPS: u64 = 1;

// ---------------------------------------------------------------------
// Probing one model both ways
// ---------------------------------------------------------------------

/// A deterministic xorshift, so a failure is reproducible from its seed.
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        // [-2, 2), which keeps `x` away from the all-equal points where a
        // sign error in a cross term cancels itself.
        (self.0 >> 11) as f64 / (1u64 << 52) as f64 * 4.0 - 2.0
    }
}

/// The probe points: the model's own starting point, then pseudo-random
/// perturbations of it. `x0` alone is not enough — a quadratic row at a
/// point where half the variables are zero hides half its coefficients.
fn probe_points(prob: &NlProblem, k: usize) -> Vec<Vec<f64>> {
    let mut out = vec![prob.x0.clone()];
    let mut rng = Rng(0x5eed_1234_9e37_79b9);
    for _ in 0..k {
        out.push(prob.x0.iter().map(|&v| v + rng.next_f64()).collect());
    }
    out
}

/// Sparse triplets as a map, so two paths with different *patterns* can
/// still be compared entry by entry.
fn as_map(irow: &[i32], jcol: &[i32], values: &[f64]) -> BTreeMap<(i32, i32), f64> {
    let mut m = BTreeMap::new();
    for k in 0..values.len() {
        m.insert((irow[k], jcol[k]), values[k]);
    }
    m
}

/// Worst relative deviation between two values, treating an exact zero on
/// both sides as agreement and falling back to absolute error when the
/// reference is zero.
///
/// A probe point is random, so it can land outside a model's domain — a
/// `sqrt` of something negative, a `log` of zero. Both paths then produce
/// `NaN` and that is agreement, not a difference; **one** of them producing
/// `NaN` is the loudest possible disagreement and comes back infinite.
fn rel_dev(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return if a.is_nan() && b.is_nan() {
            0.0
        } else {
            f64::INFINITY
        };
    }
    if a == b {
        return 0.0;
    }
    if a == 0.0 {
        return b.abs();
    }
    ((a - b) / a).abs()
}

/// Bit equality, with the same `NaN` convention as [`rel_dev`]: two `NaN`s
/// agree even if their payloads differ, because neither path promises a
/// particular quiet-`NaN` bit pattern.
fn bit_equal(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    a.to_bits() == b.to_bits()
}

/// Everything one probe of one model reports.
#[derive(Default)]
struct Report {
    /// Models where the fast path was actually taken.
    models_with_quadratic: usize,
    /// Constraint rows (over all models) routed through a form.
    quadratic_rows: usize,
    /// Worst relative deviation seen on `g` or `∇g`.
    worst_rel: f64,
    /// Hessian entries compared, and how many were not bit-identical.
    hess_entries: usize,
    hess_bit_diffs: usize,
    worst_hess_ulps: u64,
    worst_hess_where: String,
    /// `eval_f` and `eval_grad_f` values compared, and how many were not
    /// bit-identical.
    obj_entries: usize,
    obj_bit_diffs: usize,
}

/// Distance in representable doubles between two finite values of the same
/// sign. Used only to *bound* a disagreement, so mixed signs and non-finite
/// values come back saturated rather than being reasoned about.
fn ulp_distance(a: f64, b: f64) -> u64 {
    if !a.is_finite() || !b.is_finite() || a.is_sign_negative() != b.is_sign_negative() {
        return u64::MAX;
    }
    a.to_bits().abs_diff(b.to_bits())
}

fn compare_model(path: &Path, rep: &mut Report) {
    let Ok(prob) = read_nl_file(path) else { return };
    // Recognition is what decides whether this model exercises anything; a
    // model with no quadratic part builds two identical TNLPs and the
    // comparison is vacuous but free.
    let n = prob.n;
    let m = prob.m;
    if n == 0 {
        return;
    }
    let points = probe_points(&prob, 3);

    let Ok(mut fast) = NlTnlp::try_new_with_quadratic(prob.clone(), true) else {
        return;
    };
    let Ok(mut slow) = NlTnlp::try_new_with_quadratic(prob.clone(), false) else {
        return;
    };

    let quad_rows = (0..m).filter(|&i| fast.quadratic_row(i)).count();
    let quad_obj = fast.quadratic_objective();
    if quad_rows == 0 && !quad_obj {
        return;
    }
    rep.models_with_quadratic += 1;
    rep.quadratic_rows += quad_rows;

    let name = path.display();

    // The Jacobian and Hessian patterns are asked for once each; they do not
    // depend on `x`.
    let (fast_jac, slow_jac) = (
        structure(&mut fast, Kind::Jac),
        structure(&mut slow, Kind::Jac),
    );
    let (fast_h, slow_h) = (
        structure(&mut fast, Kind::Hess),
        structure(&mut slow, Kind::Hess),
    );

    for (p, x) in points.iter().enumerate() {
        // --- eval_f / eval_grad_f ---
        // The objective is on the fast path independently of the rows, and
        // is the only thing that moves on a model whose *rows* are not
        // quadratic — which is how `infeasible_equalities` (cubic rows,
        // quadratic objective) turned up in the fixture sweep.
        let (ff, fs) = (
            fast.eval_f(x, true).expect("eval_f (fast)"),
            slow.eval_f(x, true).expect("eval_f (tape)"),
        );
        let d = rel_dev(fs, ff);
        rep.worst_rel = rep.worst_rel.max(d);
        assert!(
            d <= REL_TOL,
            "{name}: probe {p}: eval_f disagrees: tape {fs:?} vs quad {ff:?} (rel {d:.3e})"
        );
        rep.obj_entries += 1;
        if !bit_equal(ff, fs) {
            rep.obj_bit_diffs += 1;
        }

        let (mut gradf, mut grads) = (vec![0.0; n], vec![0.0; n]);
        assert!(
            fast.eval_grad_f(x, true, &mut gradf),
            "{name}: eval_grad_f (fast)"
        );
        assert!(
            slow.eval_grad_f(x, true, &mut grads),
            "{name}: eval_grad_f (tape)"
        );
        for j in 0..n {
            let d = rel_dev(grads[j], gradf[j]);
            rep.worst_rel = rep.worst_rel.max(d);
            assert!(
                d <= REL_TOL,
                "{name}: probe {p}: grad_f[{j}] disagrees: tape {:?} vs quad {:?} (rel {d:.3e})",
                grads[j],
                gradf[j]
            );
            rep.obj_entries += 1;
            if !bit_equal(gradf[j], grads[j]) {
                rep.obj_bit_diffs += 1;
            }
        }

        // --- eval_g ---
        let (mut gf, mut gs) = (vec![0.0; m], vec![0.0; m]);
        assert!(fast.eval_g(x, true, &mut gf), "{name}: eval_g (fast)");
        assert!(slow.eval_g(x, true, &mut gs), "{name}: eval_g (tape)");
        for i in 0..m {
            let d = rel_dev(gs[i], gf[i]);
            rep.worst_rel = rep.worst_rel.max(d);
            assert!(
                d <= REL_TOL,
                "{name}: probe {p}: eval_g row {i} disagrees: tape {:?} vs quad {:?} (rel {d:.3e})",
                gs[i],
                gf[i]
            );
        }

        // --- eval_jac_g ---
        let mut vf = vec![0.0; fast_jac.0.len()];
        let mut vs = vec![0.0; slow_jac.0.len()];
        assert!(
            fast.eval_jac_g(Some(x), true, SparsityRequest::Values { values: &mut vf }),
            "{name}: eval_jac_g (fast)"
        );
        assert!(
            slow.eval_jac_g(Some(x), true, SparsityRequest::Values { values: &mut vs }),
            "{name}: eval_jac_g (tape)"
        );
        let jf = as_map(&fast_jac.0, &fast_jac.1, &vf);
        let js = as_map(&slow_jac.0, &slow_jac.1, &vs);
        for key in jf.keys().chain(js.keys()) {
            let (a, b) = (
                js.get(key).copied().unwrap_or(0.0),
                jf.get(key).copied().unwrap_or(0.0),
            );
            let d = rel_dev(a, b);
            rep.worst_rel = rep.worst_rel.max(d);
            assert!(
                d <= REL_TOL,
                "{name}: probe {p}: jac {key:?} disagrees: tape {a:?} vs quad {b:?} (rel {d:.3e})"
            );
        }

        // --- eval_h ---
        // Multipliers that are neither all-ones nor all-equal: a sign or an
        // index error in the λ-weighting survives both of those.
        let obj_factor = 0.75;
        let lambda: Vec<f64> = (0..m).map(|i| 1.0 + (i % 7) as f64 * 0.5).collect();
        let mut hf = vec![0.0; fast_h.0.len()];
        let mut hs = vec![0.0; slow_h.0.len()];
        assert!(
            fast.eval_h(
                Some(x),
                true,
                obj_factor,
                Some(&lambda),
                true,
                SparsityRequest::Values { values: &mut hf }
            ),
            "{name}: eval_h (fast)"
        );
        assert!(
            slow.eval_h(
                Some(x),
                true,
                obj_factor,
                Some(&lambda),
                true,
                SparsityRequest::Values { values: &mut hs }
            ),
            "{name}: eval_h (tape)"
        );
        let mf = as_map(&fast_h.0, &fast_h.1, &hf);
        let ms = as_map(&slow_h.0, &slow_h.1, &hs);
        for key in mf.keys().chain(ms.keys()) {
            let (a, b) = (
                ms.get(key).copied().unwrap_or(0.0),
                mf.get(key).copied().unwrap_or(0.0),
            );
            rep.hess_entries += 1;
            if !bit_equal(a, b) {
                rep.hess_bit_diffs += 1;
                let u = ulp_distance(a, b);
                if u > rep.worst_hess_ulps {
                    rep.worst_hess_ulps = u;
                    rep.worst_hess_where = format!("{name}: probe {p}: hessian {key:?}");
                }
            }
        }
    }
}

enum Kind {
    Jac,
    Hess,
}

fn structure(t: &mut NlTnlp, kind: Kind) -> (Vec<i32>, Vec<i32>) {
    let info = t.get_nlp_info().expect("nlp info");
    let nnz = match kind {
        Kind::Jac => info.nnz_jac_g,
        Kind::Hess => info.nnz_h_lag,
    } as usize;
    let (mut irow, mut jcol) = (vec![0i32; nnz], vec![0i32; nnz]);
    let req = SparsityRequest::Structure {
        irow: &mut irow,
        jcol: &mut jcol,
    };
    let ok = match kind {
        Kind::Jac => t.eval_jac_g(None, true, req),
        Kind::Hess => t.eval_h(None, true, 1.0, None, true, req),
    };
    assert!(ok, "structure request declined");
    (irow, jcol)
}

// ---------------------------------------------------------------------
// The corpus
// ---------------------------------------------------------------------

fn all_fixtures() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "nl") {
                out.push(p);
            }
        }
    }
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut out = Vec::new();
    walk(&base.join("fixtures"), &mut out);
    walk(&base.join("fixtures_issue_49"), &mut out);
    out.sort();
    out
}

#[test]
fn every_quadratic_fixture_evaluates_the_same_both_ways() {
    let fixtures = all_fixtures();
    assert!(
        fixtures.len() >= 50,
        "expected the fixture corpus, found {} files",
        fixtures.len()
    );
    let mut rep = Report::default();
    for f in &fixtures {
        compare_model(f, &mut rep);
    }

    // A floor, not a target: if a refactor ever leaves this walking two
    // models it should fail rather than pass vacuously.
    assert!(
        rep.models_with_quadratic >= 20,
        "the corpus should exercise the fast path on many models, got {}",
        rep.models_with_quadratic
    );
    assert!(
        rep.hess_entries >= 1_000,
        "too few Hessian entries compared: {}",
        rep.hess_entries
    );
    eprintln!(
        "[quad differential] {} models, {} quadratic rows, {} hessian entries \
         ({} not bit-identical, worst {} ulp at {}), worst g/jac rel deviation {:.3e}",
        rep.models_with_quadratic,
        rep.quadratic_rows,
        rep.hess_entries,
        rep.hess_bit_diffs,
        rep.worst_hess_ulps,
        rep.worst_hess_where,
        rep.worst_rel
    );
    eprintln!(
        "[quad differential] objective: {} values compared, {} not bit-identical",
        rep.obj_entries, rep.obj_bit_diffs
    );
    assert!(
        rep.worst_hess_ulps <= MAX_HESS_ULPS,
        "hessian disagreement grew past what the corpus produced: {} ulp at {}",
        rep.worst_hess_ulps,
        rep.worst_hess_where
    );
    assert!(
        rep.hess_bit_diffs * 1000 <= rep.hess_entries,
        "too many Hessian entries stopped being bit-identical: {} of {}",
        rep.hess_bit_diffs,
        rep.hess_entries
    );
    assert!(
        rep.worst_rel <= WORST_OBSERVED_REL,
        "g/jac deviation grew past what the corpus produced: {:.3e} > {WORST_OBSERVED_REL:.0e}",
        rep.worst_rel
    );
}

/// A model with nothing quadratic in it must be untouched — same structures,
/// same values, bit for bit, on both constructions. This is what makes the
/// phase's blast radius statable: it is exactly the recognized set.
#[test]
fn a_model_with_no_quadratic_part_is_byte_identical_on_both_paths() {
    let mut checked = 0usize;
    for f in all_fixtures() {
        let Ok(prob) = read_nl_file(&f) else { continue };
        if prob.n == 0 {
            continue;
        }
        let Ok(mut fast) = NlTnlp::try_new_with_quadratic(prob.clone(), true) else {
            continue;
        };
        if fast.quadratic_objective() || (0..prob.m).any(|i| fast.quadratic_row(i)) {
            continue;
        }
        let Ok(mut slow) = NlTnlp::try_new_with_quadratic(prob.clone(), false) else {
            continue;
        };
        let (a, b) = (
            structure(&mut fast, Kind::Hess),
            structure(&mut slow, Kind::Hess),
        );
        assert_eq!(a, b, "{}: Hessian pattern moved", f.display());
        let x = prob.x0.clone();
        let (mut gf, mut gs) = (vec![0.0; prob.m], vec![0.0; prob.m]);
        assert!(fast.eval_g(&x, true, &mut gf));
        assert!(slow.eval_g(&x, true, &mut gs));
        for i in 0..prob.m {
            assert!(
                bit_equal(gf[i], gs[i]),
                "{}: row {i} moved on a model with nothing quadratic",
                f.display()
            );
        }
        checked += 1;
    }
    assert!(
        checked >= 5,
        "expected some non-quadratic models, got {checked}"
    );
}

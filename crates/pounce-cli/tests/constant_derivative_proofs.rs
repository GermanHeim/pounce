//! Q6's proofs, checked against the derivatives they are proofs *about*
//! (gh #588).
//!
//! Q6 lets POUNCE evaluate a derivative once and reuse the answer for the
//! rest of the solve, on the strength of a claim about the model's algebra:
//! a body the recognizer proves is degree ≤ 1 has a gradient that does not
//! move, and a model whose rows are all degree ≤ 1 has a `∇²L` that does not
//! move either. If that claim is ever wrong, POUNCE hands the algorithm a
//! stale matrix and converges somewhere else — silently, on a model no
//! fixture asserts iteration counts for. That is precisely the failure mode
//! upstream Ipopt ships by trusting the user, and the whole point of the
//! phase is not to ship it ourselves.
//!
//! So the proofs are not argued here, they are **measured**, in the shape of
//! Q3's, Q4's and Q5's differentials: every `.nl` file in the repository is
//! loaded, asked what it can prove, and then the derivative it made a claim
//! about is evaluated at several points — and, for `∇²L`, at several
//! multiplier vectors, because `∇²L = σ∇²f + Σᵢλᵢ∇²gᵢ` depends on `λ` and a
//! proof that ignored that would be a proof of the wrong thing.
//!
//! Both directions are checked, because both are load-bearing:
//!
//! * a `Constant` proof must hold **bit for bit** — a tolerance would be
//!   meaningless when the shipped behaviour is literally to return the same
//!   `Rc` again;
//! * a `Varying` proof must be **witnessed** — some pair of probe points
//!   where the derivative really does move. A spurious `Varying` is not a
//!   wrong answer, but it makes POUNCE refuse a user's true hint with a
//!   warning that says the model disproves it, which is its own way of
//!   lying.
//!
//! `Unknown` is deliberately unchecked: it asserts nothing, which is the
//! whole reason it exists as a third state.
//!
//! ### Why the proof is *not* gated on `is_expanded_quadratic`
//!
//! Q4's fast path is gated on already-expanded forms because *evaluating* a
//! factored quadratic from stored coefficients cancels — `(x − 500000)²`
//! loses five digits (gh #544). Q6 evaluates nothing from coefficients: it
//! reuses the number the model's own tape produced. So the question here is
//! only about the body's *degree*, and a factored `(x − a)²` answers it as
//! well as an expanded one. `airport.nl`, the model that forced Q4's gate,
//! is one of the models Q6 reaches — and this test is what turns that
//! argument into evidence.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use pounce_cli::nl_reader::{NlProblem, NlTnlp, parse_nl_text_with_quadratic, read_nl_file};
use pounce_nlp::constant_derivatives::{DerivativeProof, DerivativeProofs};
use pounce_nlp::tnlp::{SparsityRequest, TNLP};

// ---------------------------------------------------------------------
// Probing
// ---------------------------------------------------------------------

/// A deterministic xorshift, so a failure is reproducible from its seed.
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 52) as f64 * 4.0 - 2.0
    }
    fn vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next_f64()).collect()
    }
}

/// Probe points: the model's own starting point plus perturbations. `x0`
/// alone is not enough — a quadratic row probed where half its variables are
/// zero hides half its coefficients.
fn probe_points(prob: &NlProblem, k: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    let mut out = vec![prob.x0.clone()];
    for _ in 0..k {
        out.push(prob.x0.iter().map(|&v| v + rng.next_f64()).collect());
    }
    out
}

fn bit_equal(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    a.to_bits() == b.to_bits()
}

fn jac_structure(t: &mut NlTnlp, nnz: usize) -> (Vec<i32>, Vec<i32>) {
    let (mut irow, mut jcol) = (vec![0i32; nnz], vec![0i32; nnz]);
    assert!(t.eval_jac_g(
        None,
        true,
        SparsityRequest::Structure {
            irow: &mut irow,
            jcol: &mut jcol,
        },
    ));
    (irow, jcol)
}

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

/// Corpus-wide census, printed so the reach of the phase is a number in the
/// log rather than a claim in a commit message.
#[derive(Default)]
struct Census {
    models: usize,
    grad_f_constant: usize,
    grad_f_varying: usize,
    grad_f_unknown: usize,
    hessian_constant: usize,
    hessian_varying: usize,
    hessian_unknown: usize,
    rows_constant: usize,
    rows_varying: usize,
    rows_unknown: usize,
    /// Derivative values compared across probe points.
    values_compared: usize,
    /// `Varying` proofs for which a witness was found, and the ones for
    /// which the probe grid happened not to separate two points.
    witnessed: usize,
    unwitnessed: usize,
}

/// The whole phase, over the whole corpus: every `Constant` proof holds bit
/// for bit at every probe point, and every `Varying` proof is witnessed.
#[test]
fn every_proof_holds_over_the_fixture_corpus() {
    let mut census = Census::default();
    for path in all_fixtures() {
        check_model(&path, &mut census);
    }

    println!(
        "constant-derivative proofs over {} models:\n  \
         grad_f  constant {:>4}  varying {:>4}  unknown {:>4}\n  \
         hessian constant {:>4}  varying {:>4}  unknown {:>4}\n  \
         rows    constant {:>4}  varying {:>4}  unknown {:>4}\n  \
         {} derivative values compared across probe points; \
         {} varying proofs witnessed, {} not separated by the probe grid",
        census.models,
        census.grad_f_constant,
        census.grad_f_varying,
        census.grad_f_unknown,
        census.hessian_constant,
        census.hessian_varying,
        census.hessian_unknown,
        census.rows_constant,
        census.rows_varying,
        census.rows_unknown,
        census.values_compared,
        census.witnessed,
        census.unwitnessed,
    );

    // Reach floors. Not the point of the test, but a proof set that
    // silently collapsed to "everything Unknown" would still pass every
    // assertion above, and the phase would be a no-op nobody noticed.
    assert!(
        census.models >= 40,
        "corpus shrank: {} models",
        census.models
    );
    assert!(
        census.grad_f_constant + census.hessian_constant >= 10,
        "the objective-side proofs went quiet: {} + {}",
        census.grad_f_constant,
        census.hessian_constant
    );
    assert!(
        census.rows_constant >= 100,
        "row proofs went quiet: {}",
        census.rows_constant
    );
    assert!(
        census.values_compared >= 10_000,
        "not enough was actually compared: {}",
        census.values_compared
    );
}

fn check_model(path: &Path, census: &mut Census) {
    let Ok(prob) = read_nl_file(path) else { return };
    let (n, m) = (prob.n, prob.m);
    if n == 0 {
        return;
    }
    let Ok(mut t) = NlTnlp::try_new(prob.clone()) else {
        return;
    };
    let Some(info) = t.get_nlp_info() else { return };
    let proofs = t.derivative_proofs();
    census.models += 1;
    let name = path.display();

    let mut rng = Rng(0x9e37_79b9_5eed_1234);
    let points = probe_points(&prob, 4, &mut rng);
    // Multiplier vectors, so the `∇²L` claim is tested against the `λ` it
    // actually depends on and not only against `x`.
    let lambdas: Vec<Vec<f64>> = (0..points.len()).map(|_| rng.vec(m)).collect();

    match proofs.grad_f {
        DerivativeProof::Constant => census.grad_f_constant += 1,
        DerivativeProof::Varying => census.grad_f_varying += 1,
        DerivativeProof::Unknown => census.grad_f_unknown += 1,
    }
    match proofs.hessian {
        DerivativeProof::Constant => census.hessian_constant += 1,
        DerivativeProof::Varying => census.hessian_varying += 1,
        DerivativeProof::Unknown => census.hessian_unknown += 1,
    }
    for i in 0..m {
        match proofs.row(i) {
            DerivativeProof::Constant => census.rows_constant += 1,
            DerivativeProof::Varying => census.rows_varying += 1,
            DerivativeProof::Unknown => census.rows_unknown += 1,
        }
    }

    // ---- ∇f ----
    let grads: Vec<Vec<f64>> = points
        .iter()
        .map(|x| {
            let mut g = vec![0.0; n];
            assert!(t.eval_grad_f(x, true, &mut g), "{name}: eval_grad_f");
            g
        })
        .collect();
    check_claim(&format!("{name}: ∇f"), proofs.grad_f, &grads, census);

    // ---- ∇g, per row ----
    if m > 0 && info.nnz_jac_g > 0 {
        let (irow, _) = jac_structure(&mut t, info.nnz_jac_g as usize);
        let jacs: Vec<Vec<f64>> = points
            .iter()
            .map(|x| {
                let mut v = vec![0.0; info.nnz_jac_g as usize];
                assert!(
                    t.eval_jac_g(Some(x), true, SparsityRequest::Values { values: &mut v }),
                    "{name}: eval_jac_g"
                );
                v
            })
            .collect();
        // Group the flat triplet values by row once, then compare per row:
        // a per-row proof is only about that row's entries.
        let mut by_row: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
        for (k, &r) in irow.iter().enumerate() {
            by_row.entry(r).or_default().push(k);
        }
        for (row, slots) in &by_row {
            // `NlTnlp` reports `IndexStyle::C`, so the triplet rows are
            // already the model's own 0-based row numbers.
            let i = *row as usize;
            if i >= m {
                continue;
            }
            let series: Vec<Vec<f64>> = jacs
                .iter()
                .map(|v| slots.iter().map(|&k| v[k]).collect())
                .collect();
            check_claim(
                &format!("{name}: ∇g row {i}"),
                proofs.row(i),
                &series,
                census,
            );
        }
    }

    // ---- ∇²L ----
    if info.nnz_h_lag > 0 {
        let nnz = info.nnz_h_lag as usize;
        let mut hs: Vec<Vec<f64>> = Vec::new();
        let mut ok = true;
        for (p, x) in points.iter().enumerate() {
            let mut v = vec![0.0; nnz];
            let lam = &lambdas[p];
            if !t.eval_h(
                Some(x),
                true,
                1.0,
                Some(lam),
                true,
                SparsityRequest::Values { values: &mut v },
            ) {
                ok = false;
                break;
            }
            hs.push(v);
        }
        if ok {
            check_claim(&format!("{name}: ∇²L"), proofs.hessian, &hs, census);
        }
    }
}

/// One claim against one series of evaluations, one entry per probe point.
fn check_claim(what: &str, proof: DerivativeProof, series: &[Vec<f64>], census: &mut Census) {
    if series.len() < 2 || series[0].is_empty() {
        return;
    }
    let mut moved = false;
    for later in &series[1..] {
        for (k, (&a, &b)) in series[0].iter().zip(later.iter()).enumerate() {
            census.values_compared += 1;
            if bit_equal(a, b) {
                continue;
            }
            moved = true;
            assert_ne!(
                proof,
                DerivativeProof::Constant,
                "{what}: proved constant, but entry {k} moved between probe \
                 points: {a:?} -> {b:?}. Q6 would have reused the first value \
                 for the whole solve.",
            );
        }
    }
    match proof {
        DerivativeProof::Varying if moved => census.witnessed += 1,
        // Not a failure: a genuinely quadratic row whose only free
        // variables happen to be fixed at these probe points, or a `∇²L`
        // whose moving entries all cancel, will not separate. It is
        // recorded so the number is visible rather than assumed.
        DerivativeProof::Varying => census.unwitnessed += 1,
        _ => {}
    }
}

/// A `Varying` proof is what makes POUNCE refuse a user's hint, so at least
/// some of them have to be real, and demonstrably so. The corpus-wide test
/// counts them; this one fails if that count ever drops to nothing, which is
/// the state in which the divergence from upstream would be theatre.
#[test]
fn varying_proofs_are_witnessed_by_the_corpus() {
    let mut census = Census::default();
    for path in all_fixtures() {
        check_model(&path, &mut census);
    }
    assert!(
        census.witnessed >= 20,
        "only {} varying proofs were witnessed by a derivative that actually \
         moved; a warn-and-ignore that never fires on real algebra is not a \
         feature",
        census.witnessed
    );
}

/// The proofs must not depend on whether the parser recognized a body at
/// parse time — that is a memory optimization (Q5), not a change of algebra.
/// If the two disagreed, `POUNCE_DBG_NO_QUAD=1` would silently change how
/// often derivatives are evaluated, and the A/B switch every phase of this
/// series is measured with would stop being an A/B.
#[test]
fn parse_time_recognition_does_not_move_a_single_proof() {
    let mut checked = 0usize;
    for path in all_fixtures() {
        let Ok(txt) = std::fs::read_to_string(&path) else {
            continue;
        };
        let (Ok(pq), Ok(pt)) = (
            parse_nl_text_with_quadratic(&txt, true),
            parse_nl_text_with_quadratic(&txt, false),
        ) else {
            continue;
        };
        if pq.n == 0 {
            continue;
        }
        let (Ok(mut a), Ok(mut b)) = (NlTnlp::try_new(pq), NlTnlp::try_new(pt)) else {
            continue;
        };
        let (pa, pb): (DerivativeProofs, DerivativeProofs) =
            (a.derivative_proofs(), b.derivative_proofs());
        assert_eq!(
            pa,
            pb,
            "{}: proofs moved with parse-time recognition",
            path.display()
        );
        checked += 1;
    }
    assert!(checked >= 40, "only {checked} models compared");
}

/// The three states, spelled out on models small enough to read.
///
/// These are the cases the reconciliation table in
/// `pounce_nlp::constant_derivatives` is written against, so they are pinned
/// against the actual recognizer rather than against a mock of it.
#[test]
fn the_three_states_on_hand_written_models() {
    // `min x0` s.t. `x0·x1 = 1` — affine objective (constant ∇f), a
    // genuinely bilinear row (∇g varies, so ∇²L varies).
    let bilinear = "g3 0 1 0\n 2 1 1 0 1\n 1 1\n 0 1\n 2 2 2\n 0 0 0 1\n 0 0 0 0 0\n 2 1\n 0 0\n 0 0\n\
                    C0\no2\nv0\nv1\nO0 0\nn0\nx2\n0 1.0\n1 1.0\nr\n4 1\nb\n3\n3\nk1\n2\nJ0 2\n0 0\n1 0\nG0 2\n0 1\n1 0\n";
    let p = parse_nl_text_with_quadratic(bilinear, true).expect("parse bilinear");
    let mut t = NlTnlp::try_new(p).expect("build bilinear");
    let pr = t.derivative_proofs();
    assert_eq!(pr.grad_f, DerivativeProof::Constant, "affine objective");
    assert_eq!(pr.row(0), DerivativeProof::Varying, "bilinear row");
    assert_eq!(pr.hessian, DerivativeProof::Varying, "λ-dependent ∇²L");
}

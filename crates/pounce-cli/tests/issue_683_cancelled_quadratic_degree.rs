//! A degree-2 row whose quadratic coefficients cancel must not be reported
//! **proved affine** (gh #683).
//!
//! `NlBody::provably_affine` is a proof of *degree*, and Q6 (gh #588) uses
//! it to set `jac_c_constant` / `jac_d_constant`: a row it proves affine has
//! its Jacobian evaluated once and reused for the whole solve. The proof
//! was read off the recognizer's quadratic term map, which stores only
//! coefficients that are nonzero — and those coefficients are *floating-point
//! sums*, so a row that is genuinely degree 2 can leave the map empty and be
//! reported affine. Two ways in, both of them exercised here:
//!
//! * **cancellation** — `2⁵³·x₀² + x₀² − 2⁵³·x₀²` is `x₀²`, and folds to
//!   exactly `0.0` because `2⁵³ + 1` rounds back down to `2⁵³`;
//! * **underflow** — `(10⁻²⁰⁰·x₀)·(10⁻²⁰⁰·x₀)` is one monomial whose
//!   coefficient `10⁻⁴⁰⁰` is not representable and flushes to `0.0`.
//!
//! Neither needs a `POUNCE_DBG_*` or a non-default option: the second row of
//! each model is a `sin`, so the model classifies `NLP` and takes the
//! default route, where the frozen Jacobian is a silent wrong answer.
//!
//! The models are built here rather than committed under `tests/fixtures`
//! on purpose. The cancelling row is *also* admitted by Q4's exactness gate
//! (it is a flat sum of monomials), so evaluating it from its stored
//! coefficients gave `g = 0` where its own tape gives 16 — a second,
//! separate defect on the *storage* side of the same cancellation, reported
//! as gh #685 and fixed alongside this one in
//! `issue_685_cancelled_quadratic_evaluation`. Dropping the model into the
//! corpus would still be the wrong home for it: the row is chaotic on
//! either route, and the corpus is not the place to assert that two kinds
//! of wrong agree.

use std::path::PathBuf;
use std::process::Command;

use pounce_cli::nl_quadratic::recognize_expr;
use pounce_cli::nl_reader::{
    BinOp, Expr, NlProblem, NlProblemParts, NlTnlp, UnaryOp, parse_nl_text_with_quadratic,
};
use pounce_nlp::constant_derivatives::DerivativeProof;
use pounce_nlp::tnlp::{SparsityRequest, TNLP};

/// Two variables, two rows, a linear objective. Row 0 is `body ≤ 5` — an
/// **inequality**, so it lands in `jac_d` — and row 1 is `sin(x₁) = 0`,
/// which is what keeps the classifier on the general NLP route. `$BODY` is
/// the row-0 expression segment.
///
/// `nzc = 1` is the one `J`-segment entry, on row 1: a row whose Jacobian
/// pattern is empty is not a row whose Jacobian can be observed to move.
fn model(body: &str) -> String {
    format!(
        "g3 0 1 0\n\
         2 2 1 0 1\n\
         2 0\n\
         0 0\n\
         2 1 1\n\
         0 0 0 1\n\
         0 0 0 0 0\n\
         1 2\n\
         0 0\n\
         0 0 0 0 0\n\
         C0\n{body}\
         C1\n\
         o41\n\
         v1\n\
         O0 0\n\
         n0\n\
         x2\n\
         0 1.0\n\
         1 1.0\n\
         r\n\
         1 5.0\n\
         4 0.0\n\
         b\n\
         3\n\
         3\n\
         k1\n\
         0\n\
         J1 1\n\
         1 0\n\
         G0 2\n\
         0 1.0\n\
         1 1.0\n"
    )
}

/// `2⁵³·x₀² + x₀² − 2⁵³·x₀²`, as an `o54` sumlist over three monomials on
/// the same `x₀²`. The recognizer folds a sumlist **front to back** — the
/// order the tape sums it, which is what `cd15c16f` corrected — so it
/// accumulates `2⁵³`, then `2⁵³ + 1 → 2⁵³` (ties-to-even), then
/// `2⁵³ − 2⁵³ → 0`. The body is `x₀²`.
///
/// gh #683 gives the term order as `−2⁵³, 1, +2⁵³`, which is the order that
/// cancelled under the *old*, back-to-front fold; `cd15c16f` landed between
/// the report and this fix, so the reproducer is reversed and the defect is
/// otherwise untouched. Front to back, the issue's own order sums
/// `−2⁵³ + 1 = −(2⁵³ − 1)` exactly and never cancels.
///
/// The middle term is spelled `x₀^2` where the outer two are `x₀·x₀`, and
/// that is load-bearing rather than decorative: the tape hash-conses
/// identical subtrees, so three identically-spelled `x₀·x₀` terms share one
/// node whose adjoint accumulates `2⁵³ + 1 − 2⁵³ = 0` — leaving the *tape's*
/// `∂g/∂x₀` identically zero too, and nothing to witness. With the middle
/// term on its own node the tape gives `∂g/∂x₀ = 2x₀`, which is the column
/// the issue reports (`g = 16`, `∇g = 8` at `x₀ = 3`).
fn cancelling_body() -> String {
    let big = (1u64 << 53) as f64;
    format!(
        "o54\n3\n\
         o2\nn{big:.1}\no2\nv0\nv0\n\
         o5\nv0\nn2\n\
         o2\nn{neg:.1}\no2\nv0\nv0\n",
        neg = -big,
        big = big,
    )
}

/// `(10⁻²⁰⁰·x₀)·(10⁻²⁰⁰·x₀)`. A single monomial by `is_monomial`, degree 2,
/// coefficient `10⁻⁴⁰⁰`.
fn underflowing_body() -> &'static str {
    "o2\no2\nn1e-200\nv0\no2\nn1e-200\nv0\n"
}

/// Parse with parse-time quadratic recognition on and off — the two paths
/// that produce a `Quad` body and a `Tree` body — because
/// `provably_affine` answers from a different arm on each.
fn both_paths(txt: &str) -> [NlProblem; 2] {
    [
        parse_nl_text_with_quadratic(txt, true).expect("parse (recognizing)"),
        parse_nl_text_with_quadratic(txt, false).expect("parse (trees)"),
    ]
}

/// The regression, at the level the defect lives: the degree answer for a
/// row whose quadratic terms cancelled, or underflowed, may be `Some(false)`
/// or `None`, but never `Some(true)`.
///
/// Before the fix this asserted `Some(true)` on both models and on both
/// parse paths.
#[test]
fn a_cancelled_quadratic_row_is_never_proved_affine() {
    for (what, txt) in [
        ("cancellation", model(&cancelling_body())),
        ("underflow", model(underflowing_body())),
    ] {
        for (path, prob) in ["recognizing", "trees"].iter().zip(both_paths(&txt)) {
            let got = prob.con_nonlinear[0].provably_affine();
            assert_ne!(
                got,
                Some(true),
                "{what} ({path}): a degree-2 row was proved affine",
            );
        }
    }
}

/// The other side of the demotion, sharpened in gh #687: a row whose terms
/// cancel **exactly** keeps its proof. `x₀² − x₀²` is degree 0 — the add
/// `fl(1) + fl(−1)` loses nothing — and the tape agrees, so demoting it to
/// "not established" would give up a whole solve's worth of frozen
/// Jacobian for arithmetic that never rounded.
///
/// This is what separates the gh #687 gate from the drop-based one it
/// replaces: on the pre-#687 code every assertion below reads `None`.
#[test]
fn an_exactly_cancelled_quadratic_row_is_still_proved_affine() {
    // `x₀·x₀ − x₀^2`, the two spellings, so the tape does not hash-cons the
    // pair into one node and cancel the comparison away with them.
    let body = "o1\no2\nv0\nv0\no5\nv0\nn2\n";
    for (path, prob) in ["recognizing", "trees"]
        .iter()
        .zip(both_paths(&model(body)))
    {
        assert_eq!(
            prob.con_nonlinear[0].provably_affine(),
            Some(true),
            "{path}: an exactly cancelling row lost its degree proof",
        );
    }

    // And the claim is true of the body: the tape's ∂g₀/∂x really does hold
    // still, because `2x₀ − 2x₀` is `0` at every `x₀`.
    let prob = parse_nl_text_with_quadratic(&model(body), true).expect("parse");
    let mut t = NlTnlp::try_new(prob).expect("build TNLP");
    assert_eq!(
        t.derivative_proofs().row(0),
        DerivativeProof::Constant,
        "the row's Jacobian was not proved constant",
    );
}

/// A row the recognizer has nothing to say about is still `None` and not
/// `Some(false)` — the demotion must not have turned into "everything is
/// suspicious", which would be correct and useless.
#[test]
fn an_ordinary_quadratic_row_still_proves_its_degree() {
    let prob = parse_nl_text_with_quadratic(&model("o2\nv0\nv0\n"), true).expect("parse");
    assert_eq!(
        prob.con_nonlinear[0].provably_affine(),
        Some(false),
        "x₀² is proved degree 2",
    );
    // And the `sin` row is refused, not called affine.
    assert_eq!(prob.con_nonlinear[1].provably_affine(), None);
}

/// The claim Q6 makes is about the row's real derivative, so the real
/// derivative is what checks it: on the model's own **tape** — the same
/// bytes, `POUNCE_DBG_NO_QUAD`'s path — row 0's Jacobian moves between two
/// points, and a `Constant` proof would therefore have been a lie.
///
/// The tape and not the default route, and the distinction is not
/// bookkeeping. Q4's constant-structure evaluator used to read the *same*
/// cancelled coefficients (`is_expanded_quadratic` admits a flat sum of
/// monomials, and this row is one), so on the default route the row was
/// evaluated as identically zero — its Jacobian really was constant there,
/// because the evaluator had replaced the row. That is gh #685, fixed
/// separately; what is fixed *here* is the proof, and the proof is about
/// the body, not about Q4's stand-in for it. Asking the tape keeps this
/// test answering its own question either way.
#[test]
fn the_row_jacobian_is_not_frozen() {
    // Probe points per model. The underflowing row needs a large `x`: its
    // true coefficient is `10⁻⁴⁰⁰`, so `∂g/∂x₀ = 2·10⁻⁴⁰⁰·x₀` is
    // indistinguishable from zero until `x₀` is big enough to bring it back
    // into range — which is exactly when freezing it would bite.
    for (what, txt, xs) in [
        ("cancellation", model(&cancelling_body()), (1.0, 3.0)),
        ("underflow", model(underflowing_body()), (1.0, 1e120)),
    ] {
        let prob = parse_nl_text_with_quadratic(&txt, true).expect("parse");

        let mut t = NlTnlp::try_new(prob.clone()).expect("build TNLP");
        assert_ne!(
            t.derivative_proofs().row(0),
            DerivativeProof::Constant,
            "{what}: row 0's Jacobian was proved constant",
        );

        let mut tape = NlTnlp::try_new_with_quadratic(prob, false).expect("build tape TNLP");
        let nnz = tape.get_nlp_info().expect("nlp info").nnz_jac_g as usize;
        let (mut irow, mut jcol) = (vec![0i32; nnz], vec![0i32; nnz]);
        assert!(tape.eval_jac_g(
            None,
            true,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol,
            },
        ));
        let row0 = |t: &mut NlTnlp, x0: f64| -> Vec<f64> {
            let x = [x0, 0.0];
            let mut vals = vec![0.0; nnz];
            assert!(t.eval_jac_g(
                Some(&x),
                true,
                SparsityRequest::Values { values: &mut vals },
            ));
            (0..nnz)
                .filter(|&k| irow[k] == 0)
                .map(|k| vals[k])
                .collect()
        };
        let (a, b) = (row0(&mut tape, xs.0), row0(&mut tape, xs.1));
        assert_ne!(
            a, b,
            "{what}: the tape's ∂g₀/∂x is the same at x₀ = {} and {}",
            xs.0, xs.1,
        );
    }
}

/// The reported reproduction, run as reported: the CLI on the cancelling
/// model, with `POUNCE_DBG_CONSTDERIV=1` to make the resolved hints visible.
/// It printed `jac_d_constant proof=Constant … reused=true`.
#[test]
fn the_cli_does_not_reuse_the_inequality_jacobian() {
    let dir = std::env::temp_dir().join("pounce_issue_683");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let path = dir.join("cancels.nl");
    std::fs::write(&path, model(&cancelling_body())).expect("write model");

    let out = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")))
        .arg(&path)
        .env("POUNCE_DBG_CONSTDERIV", "1")
        .output()
        .expect("run pounce");
    let stderr = String::from_utf8_lossy(&out.stderr);
    let line = stderr
        .lines()
        .find(|l| l.contains("jac_d_constant"))
        .unwrap_or_else(|| panic!("no jac_d_constant line in:\n{stderr}"));
    assert!(
        line.contains("reused=false"),
        "the inequality Jacobian is frozen for the whole solve: {line}",
    );
}

// ---------------------------------------------------------------------
// A battery over ill-scaled bodies
// ---------------------------------------------------------------------
//
// The two models above are the reported instance. This is the class: any
// body the recognizer proves affine must have a Jacobian its own tape
// agrees is constant, and the coefficients are drawn from magnitudes chosen
// so that sums round, cancel and underflow — which is the corner no `.nl`
// file in the corpus visits, and the reason the defect survived per-phase
// review with a differential test per phase.
//
// Only the `Some(true)` direction is asserted. `Some(false)` claims a
// nonzero second derivative, which floating point is entitled to round away
// (`2⁵³·x² + x² − 2⁵³·x²` has a tape Jacobian of exactly `2x`, but a body
// whose repeated monomials hash-cons into one node can have one of exactly
// zero); `None` asserts nothing by construction.

/// How far a proved-affine row's tape Jacobian may drift between probes
/// before this battery calls it a moving derivative.
///
/// Bitwise equality is what almost every row gives, and what every row gave
/// before gh #687 admitted the exactly-cancelling ones. Exactly one body in
/// 3 000 seeds drifts at all — seed 1293, whose adjoint accumulation sums
/// `±10³⁰⁸` against `−10³⁰⁰` — and it drifts by `4.5e-12`. This is three
/// orders above that and eleven below the smallest change a *degree* defect
/// produces (those move a derivative off zero, or by factors of `x`).
const TAPE_WOBBLE_REL: f64 = 1e-9;

/// A deterministic xorshift, so a failure names a reproducible seed.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
    }
}

/// Coefficients that make the arithmetic misbehave: `2⁵³` and `1` round
/// against each other, `±10²⁰⁰` underflow when multiplied and overflow when
/// added to their opposites, and the small exact ones keep ordinary terms in
/// the mix so the battery is not one-sided.
const HARSH: [f64; 10] = [
    9007199254740992.0, // 2⁵³
    -9007199254740992.0,
    1.0,
    -1.0,
    1e-200,
    1e200,
    1e300,
    -1e300,
    0.5,
    3.0,
];

fn harsh_expr(rng: &mut Rng, depth: u32) -> Expr {
    if depth == 0 || rng.below(3) == 0 {
        return if rng.below(2) == 0 {
            Expr::Var(rng.below(3) as usize)
        } else {
            Expr::Const(HARSH[rng.below(HARSH.len() as u64) as usize])
        };
    }
    match rng.below(8) {
        0 => Expr::Binary(
            BinOp::Add,
            Box::new(harsh_expr(rng, depth - 1)),
            Box::new(harsh_expr(rng, depth - 1)),
        ),
        1 => Expr::Binary(
            BinOp::Sub,
            Box::new(harsh_expr(rng, depth - 1)),
            Box::new(harsh_expr(rng, depth - 1)),
        ),
        2 | 3 => Expr::Binary(
            BinOp::Mul,
            Box::new(harsh_expr(rng, depth - 1)),
            Box::new(harsh_expr(rng, depth - 1)),
        ),
        4 => Expr::Binary(
            BinOp::Div,
            Box::new(harsh_expr(rng, depth - 1)),
            Box::new(Expr::Const(HARSH[rng.below(HARSH.len() as u64) as usize])),
        ),
        5 => Expr::Binary(
            BinOp::Pow,
            Box::new(harsh_expr(rng, depth - 1)),
            Box::new(Expr::Const(2.0)),
        ),
        6 => Expr::Unary(UnaryOp::Neg, Box::new(harsh_expr(rng, depth - 1))),
        _ => {
            let n = 2 + rng.below(3) as usize;
            Expr::Sum((0..n).map(|_| harsh_expr(rng, depth - 1)).collect())
        }
    }
}

#[test]
fn a_proved_affine_body_has_a_jacobian_its_own_tape_holds_still() {
    const N: usize = 3;
    let xs: [[f64; N]; 4] = [
        [1.0, -2.0, 0.5],
        [3.0, 1.5, -4.0],
        [1e8, -1e8, 7.0],
        [1e-8, 2.0, 1e8],
    ];
    let (mut affine, mut quadratic, mut refused) = (0usize, 0usize, 0usize);
    // Probe entries skipped because the tape itself overflowed — see the
    // comment at the comparison.
    let mut non_finite = 0usize;
    let mut worst_rel = 0.0f64;
    // Of the refusals, how many are this fix talking — a form the recognizer
    // *did* lower, demoted because a term went missing — as opposed to a
    // body it never lowered at all (a degree-3 product).
    let mut demoted = 0usize;

    for seed in 1..=3_000u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
        let rows: Vec<Expr> = (0..3).map(|_| harsh_expr(&mut rng, 4)).collect();
        let Ok(prob) = NlProblem::from_expressions(NlProblemParts {
            minimize: true,
            objective: Expr::Const(0.0),
            obj_constant: 0.0,
            constraints: rows,
            x_l: vec![-1e19; N],
            x_u: vec![1e19; N],
            x0: vec![1.0; N],
            g_l: vec![-1e19; 3],
            g_u: vec![1.0; 3],
            var_names: Vec::new(),
            con_names: Vec::new(),
        }) else {
            continue;
        };

        let claims: Vec<Option<bool>> = prob
            .con_nonlinear
            .iter()
            .map(|b| b.provably_affine())
            .collect();
        for (b, c) in prob.con_nonlinear.iter().zip(&claims) {
            match c {
                Some(true) => affine += 1,
                Some(false) => quadratic += 1,
                None => {
                    refused += 1;
                    if b.tree()
                        .and_then(recognize_expr)
                        .is_some_and(|q| q.lost_terms())
                    {
                        demoted += 1;
                    }
                }
            }
        }
        if !claims.contains(&Some(true)) {
            continue;
        }

        // The tape, because the claim is about the body — see
        // `the_row_jacobian_is_not_frozen` for why the fast path is not the
        // arbiter of it.
        let mut t =
            NlTnlp::try_new_with_quadratic(prob, false).expect("build tape TNLP for the battery");
        let nnz = t.get_nlp_info().expect("nlp info").nnz_jac_g as usize;
        let (mut irow, mut jcol) = (vec![0i32; nnz], vec![0i32; nnz]);
        assert!(t.eval_jac_g(
            None,
            true,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol,
            },
        ));
        let jac_at = |t: &mut NlTnlp, x: &[f64]| {
            let mut vals = vec![0.0; nnz];
            assert!(t.eval_jac_g(Some(x), true, SparsityRequest::Values { values: &mut vals },));
            vals
        };
        let first = jac_at(&mut t, &xs[0]);
        for x in &xs[1..] {
            let here = jac_at(&mut t, x);
            for k in 0..nnz {
                let row = irow[k] as usize;
                if claims[row] != Some(true) {
                    continue;
                }
                // A probe where the *tape* overflowed is not evidence about
                // the body. Seed 565 is the case: `x₀²·((x₁ − x₁)·x₁)·10³⁰⁰`
                // is identically zero, and the recognizer proves it — the
                // `x₁ − x₁` cancels exactly and annihilates the product. Its
                // tape multiplies before it sums, so at `x₀ = 10⁸` the
                // adjoint reaches `10³⁰⁰ · 10¹⁶ = ∞` and then `∞ · 0 = NaN`,
                // where every finite probe reads `0`. A `NaN` is not a
                // derivative that moved; it is a derivative the tape could
                // not compute, and the frozen `0` is the value it gives
                // wherever it gives a number at all. Both directions of the
                // comparison have to be numbers for the comparison to mean
                // anything (gh #687 admitted these rows; before it they were
                // refused for the cancellation and never reached here).
                if !first[k].is_finite() || !here[k].is_finite() {
                    non_finite += 1;
                    continue;
                }
                if first[k].to_bits() == here[k].to_bits() {
                    continue;
                }
                // Bitwise is the assertion; this is the one escape, and it
                // is about the *tape's* conditioning rather than the claim.
                // Since gh #687 an exactly cancelling row is proved affine,
                // and the tape re-does that cancellation with `x`-scaled
                // adjoints: seed 1293's row sums `±10³⁰⁸` contributions that
                // cancel mathematically into a running total of `−10³⁰⁰`, so
                // the constant it lands on drifts a few ulps with `x`. The
                // *body* is affine, and the frozen derivative is the value
                // the drift is around. What this rules out is a derivative
                // that genuinely moves, which is orders of magnitude, not
                // parts in 10¹²: the gh #683 reproductions move from `0`.
                let rel = (first[k] - here[k]).abs()
                    / first[k].abs().max(here[k].abs()).max(f64::MIN_POSITIVE);
                worst_rel = worst_rel.max(rel);
                assert!(
                    rel <= TAPE_WOBBLE_REL,
                    "seed {seed}: row {row} was proved affine, but ∂g/∂x[{}] moved \
                     from {} to {} (rel {rel:e}) between {:?} and {x:?}",
                    jcol[k],
                    first[k],
                    here[k],
                    xs[0],
                );
            }
        }
    }

    eprintln!(
        "[gh683 battery] {affine} rows proved affine, {quadratic} proved degree 2, \
         {refused} not established ({demoted} of them demoted for a lost term); \
         {non_finite} probe entries skipped as non-finite on the tape; \
         worst surviving Jacobian drift {worst_rel:e}"
    );
    // All three verdicts have to be represented, or the battery is asserting
    // something about a set it never reached. `refused` in particular is the
    // state this fix *adds* rows to.
    assert!(
        affine > 200 && quadratic > 200 && refused > 200,
        "battery is one-sided: {affine} affine, {quadratic} quadratic, {refused} refused",
    );
}

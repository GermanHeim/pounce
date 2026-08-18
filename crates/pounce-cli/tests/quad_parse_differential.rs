//! The Q5 parse-time recognizer against the parse it replaces (gh #588).
//!
//! Q5 reads a degree-2 `C`/`O` body off the `.nl` token stream as
//! coefficients and never builds its `Expr` tree — that tree is what sets
//! peak RSS on a quadratic model. Two things therefore have to be true, and
//! neither is the sort of claim a fixture sweep can settle:
//!
//! 1. **The recognizers agree.** What the parser stores must be, bit for
//!    bit, what `recognize_expr` returns for the tree those same bytes parse
//!    to — and the *set* of bodies it recognizes must be exactly the set
//!    `is_expanded_quadratic` admits, restricted to degree 2. A parse-time
//!    recognizer that admitted one body more would put a form on Q4's
//!    constant-matrix path that Q4's accuracy gate refuses; `airport.nl`
//!    measured what that costs (16 iterations to the 300-iteration cap).
//!    A coefficient one ulp out would be a wrong Hessian on a model no
//!    fixture solves.
//! 2. **The tree is still available, unchanged.** Everything that genuinely
//!    needs an `Expr` — the `POUNCE_DBG_NO_QUAD` tape reference, FBBT, the
//!    debugger's renderer — gets one back from `con_expr`/`obj_expr`. It has
//!    to be *the* tree, not an equivalent one: a re-derived tree would be
//!    taped differently and this phase would stop being invisible from
//!    outside.
//!
//! So every `.nl` file in the repository is parsed **twice**, with
//! recognition on and off, and the two problems are compared body by body.
//! The design note (§9, Q5) says to assert this directly instead of running
//! a sweep; the sweep is run as well, but this is the artefact that covers
//! rows no fixture's solve reaches.
//!
//! Everything here is compared on **bit patterns**, never with a tolerance.
//! Q1's 2-ulp line is why: a one-ulp coefficient difference moved a fixture
//! from 17 to 12 conic iterations, and only a differential check saw it.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use pounce_cli::nl_reader::{
    Expr, FuncallArg, NlBody, NlProblem, collect_vars, parse_nl_text_with_quadratic,
};
use pounce_nl::nl_quadratic::{is_expanded_quadratic, recognize_expr};

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

/// Structural equality on `Expr`, with every constant compared as a **bit
/// pattern**: `0.0 == -0.0` and `NaN != NaN` are both wrong answers for a
/// test whose whole job is to notice that a coefficient moved.
fn same_expr(a: &Expr, b: &Expr) -> bool {
    match (a, b) {
        (Expr::Const(x), Expr::Const(y)) => x.to_bits() == y.to_bits(),
        (Expr::Var(i), Expr::Var(j)) => i == j,
        (Expr::Binary(o1, a1, b1), Expr::Binary(o2, a2, b2)) => {
            o1 == o2 && same_expr(a1, a2) && same_expr(b1, b2)
        }
        (Expr::Unary(o1, a1), Expr::Unary(o2, a2)) => o1 == o2 && same_expr(a1, a2),
        (Expr::Sum(x), Expr::Sum(y))
        | (Expr::MinList(x), Expr::MinList(y))
        | (Expr::MaxList(x), Expr::MaxList(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same_expr(p, q))
        }
        // Structural, because the two sides come from two independent
        // parses and no `Arc` is shared between them. That a *rebuilt* body
        // reuses its own parse's `Arc`s — which is what `HybridTape` keys
        // CSE sharing on — is a separate assertion, `rebuilt_cses_are_the_parses_own`.
        (Expr::Cse(x), Expr::Cse(y)) => same_expr(x, y),
        (Expr::Compare(o1, a1, b1), Expr::Compare(o2, a2, b2)) => {
            o1 == o2 && same_expr(a1, a2) && same_expr(b1, b2)
        }
        (Expr::And(a1, b1), Expr::And(a2, b2)) | (Expr::Or(a1, b1), Expr::Or(a2, b2)) => {
            same_expr(a1, a2) && same_expr(b1, b2)
        }
        (Expr::Not(a1), Expr::Not(a2)) => same_expr(a1, a2),
        (
            Expr::Cond {
                cond: c1,
                then_: t1,
                else_: e1,
            },
            Expr::Cond {
                cond: c2,
                then_: t2,
                else_: e2,
            },
        ) => same_expr(c1, c2) && same_expr(t1, t2) && same_expr(e1, e2),
        (Expr::Funcall { id: i1, args: a1 }, Expr::Funcall { id: i2, args: a2 }) => {
            i1 == i2
                && a1.len() == a2.len()
                && a1.iter().zip(a2).all(|(p, q)| match (p, q) {
                    (FuncallArg::Real(x), FuncallArg::Real(y)) => same_expr(x, y),
                    (FuncallArg::Str(x), FuncallArg::Str(y)) => x == y,
                    _ => false,
                })
        }
        _ => false,
    }
}

/// Compare two recognized forms on the bit patterns of every coefficient.
fn same_form(a: &pounce_nl::nl_quadratic::Quad2, b: &pounce_nl::nl_quadratic::Quad2) -> bool {
    let bits = |x: &f64| x.to_bits();
    a.constant().to_bits() == b.constant().to_bits()
        && a.linear().len() == b.linear().len()
        && a.linear()
            .iter()
            .zip(b.linear())
            .all(|((i, x), (j, y))| i == j && bits(x) == bits(y))
        && a.quadratic().len() == b.quadratic().len()
        && a.quadratic()
            .iter()
            .zip(b.quadratic())
            .all(|((i, x), (j, y))| i == j && bits(x) == bits(y))
}

#[derive(Default)]
struct Report {
    files: usize,
    bodies: usize,
    recognized: usize,
    rebuilt_nodes: usize,
}

/// One body, both ways.
fn check_body(prob_q: &NlProblem, prob_t: &NlProblem, body: usize, name: &str, rep: &mut Report) {
    let (q_body, tree, q_expr) = if body == usize::MAX {
        (&prob_q.obj_nonlinear, prob_t.obj_expr(), prob_q.obj_expr())
    } else {
        (
            &prob_q.con_nonlinear[body],
            prob_t.con_expr(body),
            prob_q.con_expr(body),
        )
    };
    rep.bodies += 1;

    // (2) The tree survives, byte for byte, whether it was kept or rebuilt.
    assert!(
        same_expr(&q_expr, &tree),
        "{name}: the rebuilt tree is not the tree a non-recognizing parse builds"
    );

    // The no-quad parse must keep every body as a tree — that is what
    // `POUNCE_DBG_NO_QUAD=1` promises the gh #540 guard.
    assert!(
        prob_t
            .con_nonlinear
            .iter()
            .chain(std::iter::once(&prob_t.obj_nonlinear))
            .all(|b| b.tree().is_some()),
        "{name}: recognition leaked into the POUNCE_DBG_NO_QUAD parse"
    );

    // (1) The recognized set, and the coefficients in it.
    let want = recognize_expr(&tree)
        .filter(|f| !f.quadratic().is_empty())
        .filter(|_| is_expanded_quadratic(&tree));
    match (q_body.quad(), &want) {
        (Some(got), Some(want)) => {
            assert!(
                same_form(got, want),
                "{name}: the parse-time form is not bit-identical to the tree's"
            );
            rep.recognized += 1;
        }
        (None, None) => {}
        (Some(_), None) => panic!(
            "{name}: the parser recognized a body the tree walk refuses — \
             this is the direction that puts a factored form on Q4's \
             constant-matrix path"
        ),
        (None, Some(_)) => panic!(
            "{name}: the parser rewound a body the tree walk admits — reach \
             lost, and Q4 would pick it up anyway, so the two are out of step"
        ),
    }

    if let NlBody::Quad(q) = q_body {
        // The stored support is the tree's, including a variable whose
        // coefficient cancelled to zero (which the form necessarily drops).
        let mut want_vars: BTreeSet<usize> = BTreeSet::new();
        collect_vars(&tree, &mut want_vars);
        let got_vars: BTreeSet<usize> = q.vars.iter().map(|&v| v as usize).collect();
        assert_eq!(got_vars, want_vars, "{name}: variable support");
        assert_eq!(expr_depth(&tree), q.depth, "{name}: recorded tree depth");
        rep.rebuilt_nodes += node_count(&tree);
    }
}

fn expr_depth(e: &Expr) -> u32 {
    let deepest = |kids: &mut dyn Iterator<Item = &Expr>| kids.fold(0, |a, k| a.max(expr_depth(k)));
    1 + match e {
        Expr::Const(_) | Expr::Var(_) => 0,
        Expr::Binary(_, a, b) | Expr::Compare(_, a, b) | Expr::And(a, b) | Expr::Or(a, b) => {
            deepest(&mut [&**a, &**b].into_iter())
        }
        Expr::Unary(_, a) | Expr::Not(a) => expr_depth(a),
        Expr::Sum(v) | Expr::MinList(v) | Expr::MaxList(v) => deepest(&mut v.iter()),
        Expr::Cond { cond, then_, else_ } => {
            deepest(&mut [&**cond, &**then_, &**else_].into_iter())
        }
        Expr::Funcall { args, .. } => deepest(&mut args.iter().filter_map(|a| match a {
            FuncallArg::Real(x) => Some(x),
            FuncallArg::Str(_) => None,
        })),
        Expr::Cse(b) => expr_depth(b),
    }
}

fn node_count(e: &Expr) -> usize {
    1 + match e {
        Expr::Const(_) | Expr::Var(_) => 0,
        Expr::Binary(_, a, b) | Expr::Compare(_, a, b) | Expr::And(a, b) | Expr::Or(a, b) => {
            node_count(a) + node_count(b)
        }
        Expr::Unary(_, a) | Expr::Not(a) => node_count(a),
        Expr::Cse(a) => node_count(a),
        Expr::Sum(v) | Expr::MinList(v) | Expr::MaxList(v) => v.iter().map(node_count).sum(),
        Expr::Cond { cond, then_, else_ } => {
            node_count(cond) + node_count(then_) + node_count(else_)
        }
        Expr::Funcall { args, .. } => args
            .iter()
            .map(|a| match a {
                FuncallArg::Real(x) => node_count(x),
                FuncallArg::Str(_) => 0,
            })
            .sum(),
    }
}

#[test]
fn every_fixture_parses_identically_with_and_without_recognition() {
    let fixtures = all_fixtures();
    assert!(
        fixtures.len() >= 50,
        "expected the fixture corpus, found {} files",
        fixtures.len()
    );
    let mut rep = Report::default();
    for f in &fixtures {
        let Ok(txt) = std::fs::read_to_string(f) else {
            continue;
        };
        let (Ok(prob_q), Ok(prob_t)) = (
            parse_nl_text_with_quadratic(&txt, true),
            parse_nl_text_with_quadratic(&txt, false),
        ) else {
            continue;
        };
        let name = f.display().to_string();
        rep.files += 1;

        // Everything outside the bodies is untouched, and that includes the
        // constant-row fold (gh #492), which shifts row bounds during the
        // parse and must not see a different set of bodies.
        assert_eq!(prob_q.n, prob_t.n, "{name}: n");
        assert_eq!(prob_q.m, prob_t.m, "{name}: m");
        assert_eq!(prob_q.minimize, prob_t.minimize, "{name}: sense");
        for (i, (x, y)) in prob_q.g_l.iter().zip(&prob_t.g_l).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{name}: g_l[{i}]");
        }
        for (i, (x, y)) in prob_q.g_u.iter().zip(&prob_t.g_u).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{name}: g_u[{i}]");
        }
        assert_eq!(prob_q.con_linear, prob_t.con_linear, "{name}: con_linear");
        assert_eq!(prob_q.obj_linear, prob_t.obj_linear, "{name}: obj_linear");

        check_body(
            &prob_q,
            &prob_t,
            usize::MAX,
            &format!("{name}: obj"),
            &mut rep,
        );
        for k in 0..prob_q.m {
            check_body(&prob_q, &prob_t, k, &format!("{name}: row {k}"), &mut rep);
        }
    }

    // Floors, not targets: a refactor that leaves this test walking a
    // handful of bodies, or recognizing none of them, should fail rather
    // than pass vacuously.
    assert!(rep.files >= 50, "files walked: {}", rep.files);
    assert!(rep.bodies >= 1000, "bodies walked: {}", rep.bodies);
    assert!(
        rep.recognized >= 200,
        "bodies recognized at parse time: {}",
        rep.recognized
    );
    eprintln!(
        "[quad parse differential] {} files, {} bodies, {} recognized, \
         {} Expr nodes not built",
        rep.files, rep.bodies, rep.recognized, rep.rebuilt_nodes
    );
}

/// A model with nothing recognized is the pre-Q5 problem in every respect,
/// including keeping no source: that is what makes the blast radius
/// statable as "exactly the degree-2 bodies".
#[test]
fn a_model_with_nothing_recognized_keeps_no_source() {
    // `min sin(x0) + sin(x1)` — transcendental, so nothing is degree 2.
    let nl = "g3 0 1 0\n2 0 1 0 0\n0 1\n0 0\n0 2 0\n0 0 0 1\n0 0 0 0 0\n0 0\n0 0\n\
              0 0 0 0 0\nO0 0\no0\no41\nv0\no41\nv1\nb\n3\n3\n";
    let p = parse_nl_text_with_quadratic(nl, true).expect("parse");
    assert!(
        p.src.is_none(),
        "no body was recognized, so no text is kept"
    );
    assert!(p.obj_nonlinear.tree().is_some());
}

/// A rebuilt body resolves its `v<i>` (`i >= n`) references to the `Arc`s
/// **this** parse produced, not to fresh allocations.
///
/// That is not cosmetic: `HybridTape::build_multi` keys CSE sharing on
/// pointer identity, so a rebuilt row whose bodies were freshly allocated
/// would be taped as if nothing were shared — a different tape for the same
/// model, on the `POUNCE_DBG_NO_QUAD` path that exists to be the reference.
#[test]
fn rebuilt_cses_are_the_parses_own() {
    let mut checked = 0usize;
    for f in all_fixtures() {
        let Ok(txt) = std::fs::read_to_string(&f) else {
            continue;
        };
        let Ok(p) = parse_nl_text_with_quadratic(&txt, true) else {
            continue;
        };
        if p.cse_bodies.is_empty() {
            continue;
        }
        let own: BTreeSet<usize> = p
            .cse_bodies
            .iter()
            .map(|b| std::sync::Arc::as_ptr(b) as usize)
            .collect();
        for k in 0..p.m {
            if p.con_nonlinear[k].quad().is_none() {
                continue;
            }
            let mut seen: Vec<usize> = Vec::new();
            collect_cse_ptrs(&p.con_expr(k), &mut seen);
            for ptr in seen {
                assert!(
                    own.contains(&ptr),
                    "{}: row {k} rebuilt a CSE body instead of reusing the parse's",
                    f.display()
                );
                checked += 1;
            }
        }
    }
    eprintln!("[quad parse differential] {checked} rebuilt CSE references checked");
}

fn collect_cse_ptrs(e: &Expr, out: &mut Vec<usize>) {
    match e {
        Expr::Const(_) | Expr::Var(_) => {}
        Expr::Binary(_, a, b) | Expr::Compare(_, a, b) | Expr::And(a, b) | Expr::Or(a, b) => {
            collect_cse_ptrs(a, out);
            collect_cse_ptrs(b, out);
        }
        Expr::Unary(_, a) | Expr::Not(a) => collect_cse_ptrs(a, out),
        Expr::Sum(v) | Expr::MinList(v) | Expr::MaxList(v) => {
            v.iter().for_each(|k| collect_cse_ptrs(k, out))
        }
        Expr::Cond { cond, then_, else_ } => {
            collect_cse_ptrs(cond, out);
            collect_cse_ptrs(then_, out);
            collect_cse_ptrs(else_, out);
        }
        Expr::Funcall { args, .. } => args.iter().for_each(|a| {
            if let FuncallArg::Real(x) = a {
                collect_cse_ptrs(x, out)
            }
        }),
        Expr::Cse(b) => out.push(std::sync::Arc::as_ptr(b) as usize),
    }
}

/// A `V`-segment quadratic, referenced **twice** — the case Q4 had to
/// refuse and Q5's memoization is what makes affordable.
///
/// `V1 = x0·x0`, row `C0 = v1 + v1`. Both references sit on the sum spine
/// and each is a flat monomial, so the body is an expanded quadratic and the
/// row is `2·x0²`. Q4's gate ended the walk at the second reference and the
/// row kept its tape; here it is recognized, from the token stream, with the
/// same coefficients the tree walk produces.
///
/// This also makes `rebuilt_cses_are_the_parses_own` non-vacuous — no
/// fixture in the repository has a recognized body that references a CSE at
/// all, which is worth knowing about the corpus.
const CSE_QUAD: &str = "g3 0 1 0
1 1 1 0 0
1 0
0 0
1 0 0
0 0 0 1
0 0 0 0 0
0 0
0 0
0 1 0 0 0
V1 0 0
o2
v0
v0
C0
o0
v1
v1
O0 0
n0
r
1 5.0
b
3
k0
";

#[test]
fn a_twice_referenced_cse_quadratic_is_recognized() {
    let q = parse_nl_text_with_quadratic(CSE_QUAD, true).expect("parse");
    let t = parse_nl_text_with_quadratic(CSE_QUAD, false).expect("parse");
    let form = q.con_nonlinear[0]
        .quad()
        .expect("v1 + v1 with v1 = x0² is an expanded quadratic");
    assert_eq!(
        form.quadratic().get(&(0, 0)).map(|c| c.to_bits()),
        Some(2.0_f64.to_bits()),
        "x0² + x0² = 2·x0²"
    );
    let want = recognize_expr(&t.con_expr(0)).expect("the tree walk agrees");
    assert!(same_form(form, &want), "parse-time form vs tree walk");
    assert!(
        is_expanded_quadratic(&t.con_expr(0)),
        "and the exactness gate admits it — which it did not before Q5"
    );
    assert!(
        same_expr(&q.con_expr(0), &t.con_expr(0)),
        "the rebuilt tree is the tree"
    );
    // The rebuilt body reuses this parse's CSE body rather than a fresh one.
    let mut ptrs = Vec::new();
    collect_cse_ptrs(&q.con_expr(0), &mut ptrs);
    assert_eq!(ptrs.len(), 2, "both references survive the rebuild");
    let own = std::sync::Arc::as_ptr(&q.cse_bodies[0]) as usize;
    assert!(ptrs.iter().all(|p| *p == own));
}

/// A `Pow` whose base is a defined variable stays on the tape, because
/// `(x0 + x1)²` is a *factored* form and expanding it cancels — the rule
/// Q4 paid `airport.nl` 16 → 300 iterations to learn. The parse-time
/// recognizer has to enforce it at the point where it is cheapest to get
/// wrong, since there is no tree left to re-check afterwards.
const CSE_FACTORED: &str = "g3 0 1 0
2 1 1 0 0
1 0
0 0
2 0 0
0 0 0 1
0 0 0 0 0
0 0
0 0
0 1 0 0 0
V2 0 0
o0
v0
v1
C0
o5
v2
n2
O0 0
n0
r
1 5.0
b
3
3
k1
0
";

#[test]
fn a_factored_defined_variable_is_not_recognized() {
    let q = parse_nl_text_with_quadratic(CSE_FACTORED, true).expect("parse");
    let t = parse_nl_text_with_quadratic(CSE_FACTORED, false).expect("parse");
    assert!(
        q.con_nonlinear[0].quad().is_none(),
        "(x0 + x1)² must keep its tree — expanding it is the gh #544 defect"
    );
    assert!(!is_expanded_quadratic(&t.con_expr(0)));
    // And it *is* algebraically quadratic, so the refusal is the gate
    // talking and not the recognizer giving up.
    assert!(recognize_expr(&t.con_expr(0)).is_some());
}

/// Summation order is observable, so the parse-time fold has to be the one
/// `recognize_expr` performs — which is **last operand first**, because its
/// work stack lowers a sumlist in reverse and then folds in `drain` order.
///
/// The row is `1e16·x0² + 1·x0² + 1·x0²`. Folding front to back gives
/// `1e16` (each `+1` is lost below the ulp); folding back to front gives
/// `1e16 + 2`. One of those is what the tape-free path stores and the other
/// is a silent one-ulp lie about a coefficient.
const SUM_ORDER: &str = "g3 0 1 0
1 1 1 0 0
1 0
0 0
1 0 0
0 0 0 1
0 0 0 0 0
0 0
0 0
0 0 0 0 0
C0
o54
3
o2
n1e16
o2
v0
v0
o2
n1
o2
v0
v0
o2
n1
o2
v0
v0
O0 0
n0
r
1 5.0
b
3
k0
";

#[test]
fn a_sumlist_folds_in_the_order_the_tree_walk_folds() {
    let q = parse_nl_text_with_quadratic(SUM_ORDER, true).expect("parse");
    let t = parse_nl_text_with_quadratic(SUM_ORDER, false).expect("parse");
    let form = q.con_nonlinear[0].quad().expect("expanded quadratic");
    let want = recognize_expr(&t.con_expr(0)).expect("tree walk");
    assert!(same_form(form, &want), "fold order");
    // Pinned as a number as well as differentially, so that a change to
    // *both* sides at once still has to be argued for.
    let got = form.quadratic()[&(0, 0)];
    assert_eq!(
        got.to_bits(),
        (1.0e16_f64 + 2.0).to_bits(),
        "expected the back-to-front fold, got {got:e}"
    );
    assert_ne!(got.to_bits(), 1.0e16_f64.to_bits());
}

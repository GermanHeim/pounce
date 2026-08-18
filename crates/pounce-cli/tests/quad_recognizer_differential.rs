//! The Q3 recognizer against the one it replaced, bit for bit (gh #588).
//!
//! Q3 moved degree-≤2 recognition out of `pounce-cli`'s `dispatch` and into
//! [`pounce_cli::nl_quadratic`], and swapped the `BTreeMap<Vec<usize>, f64>`
//! polynomial for a fixed-shape `Quad2`. The new recognizer is iterative
//! where the old one recursed, which is the point of the phase — but it is
//! also *arithmetic*, and the series has already been bitten once by
//! arithmetic that was equivalent rather than identical: in Q1 a one-ulp
//! difference in a single cone coefficient moved a fixture from 17 to 12
//! conic iterations, and only the fixture sweep saw it.
//!
//! So the pre-Q3 recognizer is kept here verbatim as an oracle, and the two
//! are compared **bitwise** — not to a tolerance — on:
//!
//! * every objective and every constraint row of every `.nl` file in the
//!   repository, which covers rows no fixture's *solve* reaches;
//! * a deterministic pseudo-random battery of small expression trees, which
//!   covers operator combinations the corpus happens not to contain.
//!
//! The oracle cannot be run at depth — recursing is exactly what it did
//! wrong — so the deep-tree tests live next to the new recognizer in
//! `pounce-nl`, where there is nothing to compare against.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use pounce_cli::nl_quadratic::{QuadForm, analyze_quadratic_full};
use pounce_cli::nl_reader::{BinOp, Expr, UnaryOp, read_nl_file};

// ---------------------------------------------------------------------
// The oracle: the recognizer as it stood at 688ccd45, verbatim except for
// the allocation counter and the dropped (already unused) `_n` argument.
// ---------------------------------------------------------------------

// Monomial keys the legacy recognizer put on the heap. Every `Vec` it
// allocates as a map key is counted once; `Vec::new()` for the constant term
// is not (it does not allocate) and neither is the reallocation
// `extend_from_slice` may do inside `mul`, so this is a lower bound on what
// the representation cost.
//
// Per thread, because the harness runs the tests in this file in parallel and
// a shared counter would be counting all three of them.
thread_local! {
    static LEGACY_MONOMIAL_ALLOCS: Cell<usize> = const { Cell::new(0) };
}

fn bump() {
    LEGACY_MONOMIAL_ALLOCS.with(|c| c.set(c.get() + 1));
}

fn allocs_so_far() -> usize {
    LEGACY_MONOMIAL_ALLOCS.with(Cell::get)
}

type LegacyHessian = BTreeMap<(usize, usize), f64>;

#[derive(Debug, Clone, Default)]
struct Poly {
    terms: BTreeMap<Vec<usize>, f64>,
}

impl Poly {
    fn constant(c: f64) -> Self {
        let mut terms = BTreeMap::new();
        if c != 0.0 {
            terms.insert(Vec::new(), c);
        }
        Poly { terms }
    }

    fn var(i: usize) -> Self {
        let mut terms = BTreeMap::new();
        bump();
        terms.insert(vec![i], 1.0);
        Poly { terms }
    }

    fn max_degree(&self) -> usize {
        self.terms.keys().map(|m| m.len()).max().unwrap_or(0)
    }

    fn as_constant(&self) -> Option<f64> {
        match self.terms.len() {
            0 => Some(0.0),
            1 => self.terms.get(&Vec::new()).copied(),
            _ => None,
        }
    }

    fn add(mut self, other: &Poly) -> Poly {
        for (m, c) in &other.terms {
            bump();
            *self.terms.entry(m.clone()).or_insert(0.0) += c;
        }
        self.prune();
        self
    }

    fn neg(mut self) -> Poly {
        for c in self.terms.values_mut() {
            *c = -*c;
        }
        self
    }

    fn scale(mut self, s: f64) -> Poly {
        if s == 0.0 {
            return Poly::default();
        }
        for c in self.terms.values_mut() {
            *c *= s;
        }
        self
    }

    fn mul(&self, other: &Poly) -> Option<Poly> {
        let mut out = Poly::default();
        for (ma, ca) in &self.terms {
            for (mb, cb) in &other.terms {
                if ma.len() + mb.len() > 2 {
                    return None;
                }
                bump();
                let mut m = ma.clone();
                m.extend_from_slice(mb);
                m.sort_unstable();
                *out.terms.entry(m).or_insert(0.0) += ca * cb;
            }
        }
        out.prune();
        Some(out)
    }

    fn prune(&mut self) {
        self.terms.retain(|_, c| c.abs() > 0.0);
    }
}

// The oracle is kept as it was written, including the shapes clippy would
// rather see collapsed: an edit made to quiet a lint is an edit that could
// change what it answers, which is the one thing it must not do.
#[allow(clippy::collapsible_match)]
fn to_poly(e: &Expr) -> Option<Poly> {
    match e {
        Expr::Const(c) => Some(Poly::constant(*c)),
        Expr::Var(i) => Some(Poly::var(*i)),
        Expr::Cse(body) => to_poly(body),
        Expr::Sum(items) => {
            let mut acc = Poly::default();
            for it in items {
                let p = to_poly(it)?;
                for (m, c) in &p.terms {
                    bump();
                    *acc.terms.entry(m.clone()).or_insert(0.0) += c;
                }
            }
            acc.prune();
            Some(acc)
        }
        Expr::Unary(op, a) => match op {
            UnaryOp::Neg => Some(to_poly(a)?.neg()),
            _ => None,
        },
        Expr::Binary(op, a, b) => {
            let pa = to_poly(a)?;
            let pb = to_poly(b)?;
            match op {
                BinOp::Add => Some(pa.add(&pb)),
                BinOp::Sub => Some(pa.add(&pb.neg())),
                BinOp::Mul => pa.mul(&pb),
                BinOp::Div => {
                    let d = pb.as_constant()?;
                    if d == 0.0 {
                        None
                    } else {
                        Some(pa.scale(1.0 / d))
                    }
                }
                BinOp::Pow => {
                    let exp = pb.as_constant()?;
                    if exp == 0.0 {
                        Some(Poly::constant(1.0))
                    } else if exp == 1.0 {
                        Some(pa)
                    } else if exp == 2.0 {
                        pa.mul(&pa)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
        Expr::Funcall { .. } => None,
        _ => None,
    }
}

fn legacy_analyze(e: &Expr) -> Option<QuadForm> {
    let poly = to_poly(e)?;
    if poly.max_degree() > 2 {
        return None;
    }
    let mut h: LegacyHessian = BTreeMap::new();
    let mut lin: Vec<(usize, f64)> = Vec::new();
    let mut constant = 0.0;
    for (vars, coef) in &poly.terms {
        match vars.as_slice() {
            [] => constant += *coef,
            [i] => lin.push((*i, *coef)),
            [i, j] => {
                let (i, j) = (*i.min(j), *i.max(j));
                let contrib = if i == j { 2.0 * coef } else { *coef };
                *h.entry((i, j)).or_insert(0.0) += contrib;
            }
            _ => return None,
        }
    }
    h.retain(|_, v| v.abs() > 0.0);
    Some((h, lin, constant))
}

// ---------------------------------------------------------------------
// Bitwise comparison
// ---------------------------------------------------------------------

/// Equality down to the bit pattern of every coefficient, including the
/// sign of zero. `assert_eq!` on `f64` would call `2.0 == 2.0000000000000004`
/// unequal too, but it would call `0.0 == -0.0` equal, and the whole point
/// of this file is that "close enough" is not the property being checked.
fn identical(a: &Option<QuadForm>, b: &Option<QuadForm>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some((ha, la, ca)), Some((hb, lb, cb))) => {
            ha.len() == hb.len()
                && ha
                    .iter()
                    .zip(hb.iter())
                    .all(|((ka, va), (kb, vb))| ka == kb && va.to_bits() == vb.to_bits())
                && la.len() == lb.len()
                && la
                    .iter()
                    .zip(lb.iter())
                    .all(|((ia, va), (ib, vb))| ia == ib && va.to_bits() == vb.to_bits())
                && ca.to_bits() == cb.to_bits()
        }
        _ => false,
    }
}

fn check(e: &Expr, what: &str) {
    let old = legacy_analyze(e);
    let new = analyze_quadratic_full(e);
    assert!(
        identical(&old, &new),
        "{what}: the Q3 recognizer disagrees with the one it replaced\n  \
         legacy = {old:?}\n  quad2  = {new:?}"
    );
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

/// Every nonlinear part of every model in the repository, read both ways.
///
/// This is deliberately wider than the fixture sweep: the sweep only sees
/// rows that a *solve* depends on, and a recognizer that got a row wrong in
/// a model that solves anyway would slip through it.
#[test]
fn every_fixture_row_reads_identically_under_both_recognizers() {
    let fixtures = all_fixtures();
    assert!(
        fixtures.len() >= 50,
        "expected the fixture corpus, found {} files",
        fixtures.len()
    );
    let mut rows = 0usize;
    let mut quadratic = 0usize;
    for f in &fixtures {
        let Ok(prob) = read_nl_file(f) else { continue };
        let name = f.display();
        // `obj_expr`/`con_expr` rather than the stored bodies: since
        // gh #588 Q5 the parser recognizes a degree-2 body from the token
        // stream and keeps no tree for it, and this test is about trees.
        // The rebuilt tree is the one the parser would have built — which
        // this test then also happens to exercise, on every corpus row.
        let obj = prob.obj_expr();
        check(&obj, &format!("{name}: objective"));
        rows += 1;
        if analyze_quadratic_full(&obj).is_some() {
            quadratic += 1;
        }
        for i in 0..prob.m {
            let c = prob.con_expr(i);
            check(&c, &format!("{name}: row {i}"));
            rows += 1;
            if analyze_quadratic_full(&c).is_some() {
                quadratic += 1;
            }
        }
    }
    // Not a target, a floor: if a refactor ever leaves this test walking a
    // handful of rows it should fail rather than pass vacuously.
    // Measured on this corpus: 1631 objectives and rows, of which 1131 are
    // recognized as degree ≤ 2 and 500 are not.
    assert!(
        rows > 1_500 && quadratic > 1_000 && rows - quadratic > 400,
        "the corpus should exercise both answers: {rows} rows, {quadratic} recognized"
    );
}

// ---------------------------------------------------------------------
// A random battery
// ---------------------------------------------------------------------

/// A deterministic xorshift, so a failure is reproducible from the seed
/// printed in the assertion rather than "sometimes".
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// A small random expression over 4 variables, deep enough to nest but
/// shallow enough that the recursive oracle survives it.
fn random_expr(rng: &mut Rng, depth: u32) -> Expr {
    // Constants are drawn from a set with exact and inexact members, so
    // both `x/3` (inexact reciprocal) and `x/2` (exact) get exercised.
    const CONSTS: [f64; 8] = [0.0, 1.0, 2.0, 3.0, -1.0, 0.5, 1e-8, 1e8];
    if depth == 0 || rng.below(4) == 0 {
        return if rng.below(2) == 0 {
            Expr::Var(rng.below(4) as usize)
        } else {
            Expr::Const(CONSTS[rng.below(8) as usize])
        };
    }
    match rng.below(10) {
        0 => Expr::Binary(
            BinOp::Add,
            Box::new(random_expr(rng, depth - 1)),
            Box::new(random_expr(rng, depth - 1)),
        ),
        1 => Expr::Binary(
            BinOp::Sub,
            Box::new(random_expr(rng, depth - 1)),
            Box::new(random_expr(rng, depth - 1)),
        ),
        2 => Expr::Binary(
            BinOp::Mul,
            Box::new(random_expr(rng, depth - 1)),
            Box::new(random_expr(rng, depth - 1)),
        ),
        3 => Expr::Binary(
            BinOp::Div,
            Box::new(random_expr(rng, depth - 1)),
            Box::new(random_expr(rng, depth - 1)),
        ),
        4 => Expr::Binary(
            BinOp::Pow,
            Box::new(random_expr(rng, depth - 1)),
            Box::new(random_expr(rng, depth - 1)),
        ),
        5 => Expr::Unary(UnaryOp::Neg, Box::new(random_expr(rng, depth - 1))),
        // A transcendental, so the bail-out paths are exercised too.
        6 => Expr::Unary(UnaryOp::Sin, Box::new(random_expr(rng, depth - 1))),
        7 => Expr::Cse(std::sync::Arc::new(random_expr(rng, depth - 1))),
        _ => {
            let n = 1 + rng.below(4) as usize;
            Expr::Sum((0..n).map(|_| random_expr(rng, depth - 1)).collect())
        }
    }
}

#[test]
fn random_expression_battery_reads_identically_under_both_recognizers() {
    let mut recognized = 0usize;
    let mut refused = 0usize;
    for seed in 1..=4_000u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
        let e = random_expr(&mut rng, 5);
        let old = legacy_analyze(&e);
        let new = analyze_quadratic_full(&e);
        assert!(
            identical(&old, &new),
            "seed {seed}: the Q3 recognizer disagrees with the one it replaced\n  \
             expr   = {e:?}\n  legacy = {old:?}\n  quad2  = {new:?}"
        );
        if new.is_some() {
            recognized += 1;
        } else {
            refused += 1;
        }
    }
    // Both verdicts have to be represented or the battery proves nothing.
    assert!(
        recognized > 200 && refused > 200,
        "battery is one-sided: {recognized} recognized, {refused} refused"
    );
}

// ---------------------------------------------------------------------
// What the representation cost
// ---------------------------------------------------------------------

/// The allocation the phase removed, stated as a number.
///
/// One dense quadratic row over 500 variables — the `qcqp500-3c` shape —
/// expands to ~n²/2 monomials, and the legacy representation put every one
/// of them on the heap as its own `Vec`, allocating another on every merge.
/// `Quad2` keys its terms on `usize` and `(usize, usize)` inline, so the
/// per-monomial allocation is not reduced, it is gone: the count below is
/// structurally zero for the new recognizer, which is why only the legacy
/// side is instrumented.
///
/// The classifier expands each row two or three times per solve (class,
/// convexity, extraction), and a benchmark instance has ten such rows —
/// which is where the design note's "2.32 M `Vec`s per pass" came from. The
/// single-row count asserted here is the part that can be measured in a
/// unit test on any machine.
#[test]
fn legacy_representation_allocated_one_vec_per_monomial() {
    let n = 500usize;
    // (Σ xᵢ)² — one dense row, n(n+1)/2 distinct monomials.
    let row = Expr::Binary(
        BinOp::Pow,
        Box::new(Expr::Sum((0..n).map(Expr::Var).collect())),
        Box::new(Expr::Const(2.0)),
    );

    let before = allocs_so_far();
    let old = legacy_analyze(&row);
    let allocs = allocs_so_far() - before;
    let new = analyze_quadratic_full(&row);
    assert!(identical(&old, &new), "dense row must read identically");

    let monomials = old.expect("dense row is quadratic").0.len();
    assert_eq!(monomials, n * (n + 1) / 2);
    // n² keys from the product (one per pair of factors, before any of them
    // are merged), plus n merging the sum, plus n for the variables
    // themselves. For a 500-variable row that is 251 000 heap allocations to
    // describe 125 250 distinct monomials — and the classifier expands each
    // row two or three times per solve.
    assert_eq!(
        allocs,
        n * n + 2 * n,
        "the legacy monomial-key allocation count is the number this phase removed"
    );
}

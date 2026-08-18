//! Degree-≤2 recognition over an [`Expr`] DAG.
//!
//! This is the classifier's "is this row a quadratic, and which one?"
//! question, answered once and reused. `pounce-cli`'s `dispatch` owns the
//! *routing* decision (which `ProblemClass`, which solver); what lives here
//! is only the algebra, so the consumers that are not the CLI —
//! `NlTnlp`'s constant-structure evaluation, the parse-time recognizer, the
//! `QcqpProblem` extractor — can reach it without depending on the
//! command-line driver. It moved out of `pounce-cli/src/dispatch.rs` in
//! Q3 of the #588 series; see
//! `dev-notes/quadratic-structure-exploitation.md`.
//!
//! ## Two properties this module is built around
//!
//! **It is iterative.** The walk carries its own work stack rather than
//! recursing, because the trees it is handed are not shallow: a `.nl`
//! writer that emits `o0` (binary `+`) chains for a long sum — Pyomo does —
//! produces a left-deep `Add` tree one level per term. The recursive
//! predecessor aborted the process somewhere between 4 000 and 6 000 terms
//! on a 2 MB thread (which is what a test gets, and where the crash was
//! first reproduced) and between 16 000 and 24 000 on the CLI's 8 MB main
//! thread. A stack overflow is an abort, not an error return, so the depth a
//! *recognizer* survives must not depend on which thread called it.
//!
//! It is worth knowing what this does **not** fix: `nl_reader`'s parser
//! recurses too, with a fatter frame — it gives out at ~6 000 on that same
//! 8 MB thread — so a deep `.nl` file still fails to load, and it fails
//! before reaching this module. What is fixed is every path where the tree
//! is already built (`NlProblem::from_expressions`, a model handed across
//! threads) and the ceiling this module used to impose on the parser's
//! successor.
//!
//! **It never allocates per monomial.** The predecessor keyed monomials on
//! `BTreeMap<Vec<usize>, f64>`, so every term cost a heap allocation and
//! every merge cloned one (`entry(m.clone())`). A degree-≤2 form has only
//! three shapes of term, so [`Quad2`] stores them in three fields with
//! inline keys and the allocation disappears. The same change removes the
//! `O(N²)` accumulation on `Add` chains: the old `add` re-scanned the whole
//! accumulated map for zeros on *every* merge, which is quadratic down a
//! left-deep chain. Zeros can only appear where a merge touched, so that is
//! all this one looks at.

use crate::nl_reader::{BinOp, Expr, UnaryOp};
use std::collections::BTreeMap;

/// The symmetric Hessian of a quadratic form, stored as a sparse upper-
/// triangular (i ≤ j) map of `(i, j) -> ∂²/∂xᵢ∂xⱼ`. Empty means the
/// expression is (at most) linear.
pub type QuadHessian = BTreeMap<(usize, usize), f64>;

/// Full quadratic read-out: `(Hessian, [(var, linear coef), …], constant)`.
/// The linear and constant parts are the pieces AMPL/Pyomo fold into the
/// nonlinear objective tree (see [`analyze_quadratic_full`]).
pub type QuadForm = (QuadHessian, Vec<(usize, f64)>, f64);

/// A polynomial of total degree ≤ 2 in its own shape: a constant, the
/// linear coefficients keyed by variable, and the quadratic coefficients
/// keyed by the (i ≤ j) variable pair.
///
/// This replaces a general `BTreeMap<Vec<usize>, f64>` polynomial. Degree
/// is a property of the *type* here rather than of the data, so the
/// "is it still quadratic?" test is three `is_empty()` calls instead of a
/// scan, and no monomial key is ever allocated or cloned.
///
/// ### Zero coefficients
///
/// Stored coefficients are nonzero: [`add`](Quad2::add) and
/// [`mul`](Quad2::mul) drop any entry they leave at exactly zero, which is
/// what makes [`degree`](Quad2::degree) and [`as_constant`](Quad2::as_constant)
/// answerable in `O(1)`. `constant` is the one exception and needs none:
/// both `0.0` and `-0.0` in that field mean "no constant term", every
/// consumer guards on `!= 0.0`, and [`analyze_quadratic_full`] normalizes
/// the sign on the way out.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Quad2 {
    constant: f64,
    linear: BTreeMap<usize, f64>,
    quadratic: QuadHessian,
}

impl Quad2 {
    /// The degree-0 term.
    pub fn constant(&self) -> f64 {
        self.constant
    }

    /// The degree-1 terms, ascending by variable index.
    pub fn linear(&self) -> &BTreeMap<usize, f64> {
        &self.linear
    }

    /// The degree-2 terms as *polynomial* coefficients keyed `(i ≤ j)` —
    /// the coefficient of `xᵢxⱼ`, **not** the Hessian entry (they differ by
    /// a factor of 2 on the diagonal; [`analyze_quadratic_full`] applies it).
    pub fn quadratic(&self) -> &QuadHessian {
        &self.quadratic
    }

    fn of_constant(c: f64) -> Self {
        Quad2 {
            // `-0.0` and `0.0` alike mean "no constant term"; normalizing
            // here keeps `Neg(Const(0.0))` from reporting `-0.0`.
            constant: if c != 0.0 { c } else { 0.0 },
            ..Quad2::default()
        }
    }

    fn of_var(i: usize) -> Self {
        let mut q = Quad2::default();
        q.linear.insert(i, 1.0);
        q
    }

    /// Total degree: 0, 1, or 2.
    fn degree(&self) -> usize {
        if !self.quadratic.is_empty() {
            2
        } else if !self.linear.is_empty() {
            1
        } else {
            0
        }
    }

    /// The value, when this form has no variables in it.
    fn as_constant(&self) -> Option<f64> {
        (self.degree() == 0).then_some(self.constant)
    }

    /// Number of stored (nonzero) variable terms.
    fn width(&self) -> usize {
        self.linear.len() + self.quadratic.len()
    }

    /// `a + b`.
    ///
    /// Two things here are not incidental. **Only the entries the smaller
    /// side contributes are re-checked for zero** — the predecessor
    /// re-scanned the whole accumulated map on every merge, which is `O(k²)`
    /// down a `k`-term `Add` chain, and an `o0` chain is how a long sum
    /// reaches this code. And **the smaller side is merged into the larger**,
    /// whichever operand it is, so the same chain costs `O(k log k)` leaning
    /// either way rather than only when it leans left. Choosing the direction
    /// is free of arithmetic consequence: IEEE addition is commutative bit
    /// for bit, `-0.0` included.
    fn add(a: Quad2, b: Quad2) -> Quad2 {
        let (mut acc, small) = if a.width() >= b.width() {
            (a, b)
        } else {
            (b, a)
        };
        if small.constant != 0.0 {
            acc.constant += small.constant;
        }
        for (i, c) in &small.linear {
            merge(&mut acc.linear, *i, *c);
        }
        for (k, c) in &small.quadratic {
            merge(&mut acc.quadratic, *k, *c);
        }
        acc
    }

    fn neg(mut self) -> Quad2 {
        self.constant = -self.constant;
        for c in self.linear.values_mut() {
            *c = -*c;
        }
        for c in self.quadratic.values_mut() {
            *c = -*c;
        }
        self
    }

    fn scale(mut self, s: f64) -> Quad2 {
        if s == 0.0 {
            return Quad2::default();
        }
        self.constant *= s;
        for c in self.linear.values_mut() {
            *c *= s;
        }
        for c in self.quadratic.values_mut() {
            *c *= s;
        }
        self
    }

    /// `self · other`, or `None` when the product would exceed total
    /// degree 2 — past that the recognizer gives up and the caller routes
    /// to the general NLP path.
    fn mul(&self, other: &Quad2) -> Option<Quad2> {
        if self.degree() + other.degree() > 2 {
            return None;
        }
        let mut out = Quad2::default();
        if self.constant != 0.0 && other.constant != 0.0 {
            out.constant = self.constant * other.constant;
        }
        // constant × (linear, quadratic), both ways round.
        for (a, b) in [(self, other), (other, self)] {
            if a.constant == 0.0 {
                continue;
            }
            for (i, c) in &b.linear {
                *out.linear.entry(*i).or_insert(0.0) += a.constant * c;
            }
            for (k, c) in &b.quadratic {
                *out.quadratic.entry(*k).or_insert(0.0) += a.constant * c;
            }
        }
        // linear × linear. The degree guard above means at most one of the
        // two operands carries quadratic terms, so this runs only when
        // neither does and no ordering question arises.
        for (i, a) in &self.linear {
            for (j, b) in &other.linear {
                let key = (*i.min(j), *i.max(j));
                *out.quadratic.entry(key).or_insert(0.0) += a * b;
            }
        }
        out.linear.retain(|_, c| is_live(*c));
        out.quadratic.retain(|_, c| is_live(*c));
        Some(out)
    }
}

/// Add `c` to `map[key]`, keeping the "no stored zeros" invariant.
///
/// Only the touched key can have become zero, which is what keeps a merge
/// proportional to what it merged rather than to what it merged *into*.
fn merge<K: Ord>(map: &mut BTreeMap<K, f64>, key: K, c: f64) {
    use std::collections::btree_map::Entry;
    match map.entry(key) {
        Entry::Occupied(mut e) => {
            let v = *e.get() + c;
            if is_live(v) {
                e.insert(v);
            } else {
                e.remove();
            }
        }
        Entry::Vacant(e) => {
            if is_live(c) {
                e.insert(c);
            }
        }
    }
}

/// Is this coefficient worth storing? Exact zeros are dropped so that
/// "has a quadratic term" is a structural question; `NaN` is dropped for
/// the same reason its predecessor's `c.abs() > 0.0` retention dropped it.
fn is_live(c: f64) -> bool {
    c.abs() > 0.0
}

/// One entry on the recognizer's explicit work stack.
enum Step<'a> {
    /// Lower this subexpression onto the value stack.
    Visit(&'a Expr),
    /// Combine values already on the value stack.
    Apply(Op),
}

/// A pending combination, popped once its operands have been lowered.
enum Op {
    Neg,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    /// n-ary sum over the top `n` values.
    Sum(usize),
}

/// Lower an [`Expr`] to a [`Quad2`], or `None` if it contains anything the
/// recognizer cannot prove is a degree-≤2 polynomial (transcendental ops,
/// division by a non-constant, `Pow` with an exponent ∉ {0, 1, 2}, products
/// of degree > 2, external calls, comparisons, `if-then-else`, `min`/`max`,
/// …). `None` ⇒ treat as general nonlinear.
///
/// `Cse` nodes are inlined: a reference is mathematically its body, and
/// every reference is an independent occurrence. A body reached `r` times
/// is therefore lowered `r` times, exactly as the recursive predecessor did
/// — memoizing on the `Arc` identity is a pure win still on the table for
/// Q5, not a behaviour change made here.
///
/// The walk is iterative. See the module docs for why that is a
/// correctness property and not a style choice.
pub fn recognize_expr(e: &Expr) -> Option<Quad2> {
    let mut work: Vec<Step<'_>> = vec![Step::Visit(e)];
    let mut vals: Vec<Quad2> = Vec::new();

    while let Some(step) = work.pop() {
        match step {
            Step::Visit(e) => match e {
                Expr::Const(c) => vals.push(Quad2::of_constant(*c)),
                Expr::Var(i) => vals.push(Quad2::of_var(*i)),
                Expr::Cse(body) => work.push(Step::Visit(body)),
                Expr::Sum(items) => {
                    work.push(Step::Apply(Op::Sum(items.len())));
                    // Pushed forward, so they pop back to front and land on
                    // the value stack with item 0 on top — the sum then
                    // accumulates front to back, the order the recursive
                    // version summed in.
                    for it in items {
                        work.push(Step::Visit(it));
                    }
                }
                Expr::Unary(UnaryOp::Neg, a) => {
                    work.push(Step::Apply(Op::Neg));
                    work.push(Step::Visit(a));
                }
                // Every other unary op is transcendental.
                Expr::Unary(..) => return None,
                Expr::Binary(op, a, b) => {
                    let op = match op {
                        BinOp::Add => Op::Add,
                        BinOp::Sub => Op::Sub,
                        BinOp::Mul => Op::Mul,
                        BinOp::Div => Op::Div,
                        BinOp::Pow => Op::Pow,
                        // atan2 and any other binary opcode.
                        _ => return None,
                    };
                    work.push(Step::Apply(op));
                    // `b` under `a`: `a` pops first and is lowered first.
                    work.push(Step::Visit(b));
                    work.push(Step::Visit(a));
                }
                // External calls are opaque; comparisons, logicals,
                // conditionals and n-ary min/max are the control-flow `.nl`
                // opcodes. None is provably polynomial ⇒ route to NLP.
                _ => return None,
            },
            Step::Apply(op) => {
                let combined = match op {
                    Op::Sum(n) => {
                        // The items are the top `n` values, item 0 on top.
                        let at = vals.len().checked_sub(n)?;
                        let mut acc = Quad2::default();
                        for p in vals.drain(at..) {
                            acc = Quad2::add(acc, p);
                        }
                        acc
                    }
                    Op::Neg => vals.pop()?.neg(),
                    Op::Add => {
                        let (a, b) = pop2(&mut vals)?;
                        Quad2::add(a, b)
                    }
                    Op::Sub => {
                        let (a, b) = pop2(&mut vals)?;
                        Quad2::add(a, b.neg())
                    }
                    Op::Mul => {
                        let (a, b) = pop2(&mut vals)?;
                        a.mul(&b)?
                    }
                    Op::Div => {
                        // Division is polynomial only by a nonzero constant,
                        // and scales by the reciprocal (not `c / d`) so the
                        // arithmetic matches what the recursive predecessor
                        // produced bit for bit.
                        let (a, b) = pop2(&mut vals)?;
                        let d = b.as_constant()?;
                        if d == 0.0 {
                            return None;
                        }
                        a.scale(1.0 / d)
                    }
                    Op::Pow => {
                        // Polynomial only for constant exponents in {0, 1, 2}.
                        let (a, b) = pop2(&mut vals)?;
                        let exp = b.as_constant()?;
                        if exp == 0.0 {
                            Quad2::of_constant(1.0)
                        } else if exp == 1.0 {
                            a
                        } else if exp == 2.0 {
                            a.mul(&a)?
                        } else {
                            return None;
                        }
                    }
                };
                vals.push(combined);
            }
        }
    }

    debug_assert_eq!(vals.len(), 1, "one value per lowered expression");
    vals.pop()
}

/// Pop a binary operator's two operands, left first.
fn pop2(vals: &mut Vec<Quad2>) -> Option<(Quad2, Quad2)> {
    let b = vals.pop()?;
    let a = vals.pop()?;
    Some((a, b))
}

/// Attempt to read an expression as a polynomial of total degree ≤ 2 and
/// return its Hessian (constant, since the form is quadratic). `None` if
/// the expression is not provably quadratic ⇒ treat as general nonlinear.
pub fn analyze_quadratic(e: &Expr) -> Option<QuadHessian> {
    analyze_quadratic_full(e).map(|(h, _, _)| h)
}

/// Like [`analyze_quadratic`] but also returns the degree-1 (linear)
/// coefficients *and* the degree-0 (constant) term of the form:
/// `(Hessian, [(var, coef), …], constant)`.
///
/// AMPL folds the linear part of a nonlinear term into the objective's
/// nonlinear expression tree (the `−6·x₀` of `(x₀−3)²`, say) rather than
/// the linear section. Callers building the QP objective vector `c` must
/// add these in, exactly as the NLP path's `eval_f` sums the linear
/// section *and* the nonlinear tree — otherwise the linear shift is
/// silently dropped and the convex solve minimizes the wrong objective.
///
/// The **constant** is returned for the same reason: AMPL/Pyomo also fold
/// the objective's degree-0 term into the nonlinear tree (the `+9` of
/// `(x₀−3)²`), where it does *not* land in `NlProblem::obj_constant`. It
/// is irrelevant to the minimizer but is part of the *reported objective
/// value*; dropping it makes the convex solve report an objective off by
/// that constant versus the NLP path.
pub fn analyze_quadratic_full(e: &Expr) -> Option<QuadForm> {
    let q = recognize_expr(e)?;
    // ∂²(c·xᵢxⱼ)/∂xᵢ∂xⱼ = c for i≠j; ∂²(c·xᵢ²)/∂xᵢ² = 2c.
    let mut h: QuadHessian = q
        .quadratic
        .iter()
        .map(|(&(i, j), c)| ((i, j), if i == j { 2.0 * c } else { *c }))
        .collect();
    // Drop explicit zeros so `is_empty()` means "linear".
    h.retain(|_, v| v.abs() > 0.0);
    let lin: Vec<(usize, f64)> = q.linear.iter().map(|(i, c)| (*i, *c)).collect();
    // `0.0 +` normalizes `-0.0`, which is how this form spells "absent".
    Some((h, lin, 0.0 + q.constant))
}

/// True if the expression is the literal constant zero the `.nl` reader
/// uses for "no nonlinear part".
pub fn is_trivially_zero(e: &Expr) -> bool {
    matches!(e, Expr::Const(c) if *c == 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sq(i: usize) -> Expr {
        Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(i)),
            Box::new(Expr::Const(2.0)),
        )
    }

    #[test]
    fn quadratic_diagonal() {
        // (x0 - 1)^2  =>  x0^2 - 2 x0 + 1
        let e = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Binary(
                BinOp::Sub,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(1.0)),
            )),
            Box::new(Expr::Const(2.0)),
        );
        let (h, lin, c) = analyze_quadratic_full(&e).expect("degree-2 polynomial");
        assert_eq!(h.get(&(0, 0)), Some(&2.0));
        assert_eq!(lin, vec![(0, -2.0)]);
        assert_eq!(c, 1.0);
    }

    #[test]
    fn cross_term_hessian() {
        // x0 · x1 => H[0,1] = 1
        let e = Expr::Binary(BinOp::Mul, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        let h = analyze_quadratic(&e).expect("degree-2");
        assert_eq!(h.get(&(0, 1)), Some(&1.0));
    }

    #[test]
    fn rejects_transcendental_and_cubic() {
        assert!(analyze_quadratic(&Expr::Unary(UnaryOp::Sin, Box::new(Expr::Var(0)))).is_none());
        let cubic = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(3.0)),
        );
        assert!(analyze_quadratic(&cubic).is_none());
        // x0² · x1 — degree 3 by multiplication rather than by exponent.
        let deg3 = Expr::Binary(BinOp::Mul, Box::new(sq(0)), Box::new(Expr::Var(1)));
        assert!(analyze_quadratic(&deg3).is_none());
    }

    #[test]
    fn division_by_a_constant_scales_by_the_reciprocal() {
        // x0² / 3 — the coefficient must be `2 · (1/3)`, which is what the
        // Hessian of `x0²/3` came out as before this module existed, and is
        // *not* bitwise `2/3`.
        let e = Expr::Binary(BinOp::Div, Box::new(sq(0)), Box::new(Expr::Const(3.0)));
        let h = analyze_quadratic(&e).expect("degree-2");
        assert_eq!(h.get(&(0, 0)), Some(&(2.0 * (1.0 / 3.0))));
        // Division by a variable is not polynomial.
        let e = Expr::Binary(BinOp::Div, Box::new(sq(0)), Box::new(Expr::Var(1)));
        assert!(analyze_quadratic(&e).is_none());
    }

    #[test]
    fn cancellation_drops_the_term_and_the_degree_with_it() {
        // x0² − x0² is linear (empty Hessian), not a quadratic with a zero
        // coefficient — otherwise `x0²−x0²` times `x1` would be refused as
        // degree 3.
        let zero = Expr::Binary(BinOp::Sub, Box::new(sq(0)), Box::new(sq(0)));
        let h = analyze_quadratic(&zero).expect("degree-2 at worst");
        assert!(h.is_empty());
        let times_x1 = Expr::Binary(BinOp::Mul, Box::new(zero), Box::new(Expr::Var(1)));
        assert!(analyze_quadratic(&times_x1).is_some());
    }

    #[test]
    fn cse_bodies_are_inlined_at_every_reference() {
        // c = x0; c · c is x0².
        let body = std::sync::Arc::new(Expr::Var(0));
        let e = Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Cse(body.clone())),
            Box::new(Expr::Cse(body)),
        );
        let h = analyze_quadratic(&e).expect("degree-2");
        assert_eq!(h.get(&(0, 0)), Some(&2.0));
    }

    #[test]
    fn nary_sum_accumulates_front_to_back() {
        // Σ xᵢ² over 5000 terms as one `o54` node.
        const N: usize = 5000;
        let e = Expr::Sum((0..N).map(sq).collect());
        let h = analyze_quadratic(&e).expect("sum of squares is a QP");
        assert_eq!(h.len(), N);
        assert_eq!(h.get(&(N - 1, N - 1)), Some(&2.0));
    }

    /// The reason this module is iterative.
    ///
    /// A `.nl` writer that emits `o0` (binary `+`) chains for a long sum
    /// hands the recognizer a left-deep `Add` tree one level deep per term.
    /// The recursive predecessor aborted the process — a stack overflow is
    /// not a catchable error — somewhere under 8 000 terms on a 2 MB thread
    /// and under 24 000 on the CLI's 8 MB main thread. The depth below is
    /// far past both, and the test runs on a **default-sized test thread**
    /// deliberately: what a recognizer survives must not depend on who
    /// called it.
    ///
    /// The tree is leaked rather than dropped, because `Expr`'s derived
    /// `Drop` is still recursive and would overflow tearing this down.
    /// That is a real and separate defect (pounce#472 works around it in
    /// the Python bindings with a big-stack worker thread); it is not what
    /// this test is measuring.
    #[test]
    fn deep_add_chain_does_not_overflow_the_stack() {
        const K: usize = 250_000;
        let mut e = sq(0);
        for i in 1..K {
            e = Expr::Binary(BinOp::Add, Box::new(e), Box::new(sq(i)));
        }
        let h = analyze_quadratic(&e).expect("a sum of squares is a QP at any depth");
        assert_eq!(h.len(), K, "every xᵢ² contributes one diagonal entry");
        assert_eq!(h.get(&(K - 1, K - 1)), Some(&2.0));
        std::mem::forget(e);
    }

    /// Same shape, right-deep — the value stack, not the work stack, is
    /// what grows here. Both live on the heap.
    #[test]
    fn deep_right_leaning_chain_does_not_overflow_the_stack() {
        const K: usize = 250_000;
        let mut e = sq(K - 1);
        for i in (0..K - 1).rev() {
            e = Expr::Binary(BinOp::Add, Box::new(sq(i)), Box::new(e));
        }
        let h = analyze_quadratic(&e).expect("a sum of squares is a QP at any depth");
        assert_eq!(h.len(), K);
        std::mem::forget(e);
    }

    /// A non-quadratic node deep inside a deep tree must return `None`
    /// rather than unwind — the bail-out path drops the work and value
    /// stacks, and neither is recursive.
    #[test]
    fn deep_chain_with_a_transcendental_bails_without_overflowing() {
        const K: usize = 250_000;
        let mut e = Expr::Unary(UnaryOp::Sin, Box::new(Expr::Var(0)));
        for i in 1..K {
            e = Expr::Binary(BinOp::Add, Box::new(e), Box::new(sq(i)));
        }
        assert!(analyze_quadratic(&e).is_none());
        std::mem::forget(e);
    }
}

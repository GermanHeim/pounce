//! McCormick polyhedral relaxation of a factorable problem over a box.
//!
//! Factorable programming: every tape slot becomes an auxiliary LP variable
//! `w_k`, constrained to the convex/concave envelopes of the operation that
//! produced it given **interval** bounds (from FBBT's forward pass) on its
//! operands. Affine ops are exact equalities; bilinear products get the four
//! McCormick inequalities; univariate convex/concave atoms (`x^n`, `√`, `exp`,
//! `ln`, `|·|`) get a secant on one side and tangent cuts on the other. The
//! result is a **linear program** whose optimum is a valid lower bound on the
//! true minimum over the box — and, because every envelope is exact at the box
//! corners, the bound tightens to the truth as the box shrinks (so spatial
//! branch-and-bound converges).
//!
//! Operations the relaxation cannot convexify cheaply (`sin`, `cos`, division
//! by an interval straddling zero, `Opaque`) contribute only their interval
//! box bound on `w_k` — valid, just weak, which branching then sharpens.

use crate::problem::GlobalProblem;
use pounce_convex::{QpProblem, Triplet};
use pounce_nlp::{FbbtOp, FbbtTape};
use pounce_presolve::fbbt::forward_pass;
use std::collections::BTreeMap;

/// Sentinel magnitude past which the convex solver treats a bound as infinite.
const INF: f64 = 1e20;

/// A tape slot's representation in the LP: either an LP column or a constant.
#[derive(Clone, Copy)]
enum Handle {
    Col(usize),
    Const(f64),
}

/// The relaxation LP plus the bookkeeping to read a solution back. The first
/// `n_vars` LP columns are the original problem variables.
pub(crate) struct Relaxation {
    pub qp: QpProblem,
    /// `true` if a constant constraint was found out of bounds — the box is
    /// then certifiably infeasible and the node can be pruned without solving.
    pub trivially_infeasible: bool,
}

/// Accumulates LP columns and rows while walking the tapes.
struct Builder {
    col_lo: Vec<f64>,
    col_hi: Vec<f64>,
    eq: Vec<Triplet>,
    eq_rhs: Vec<f64>,
    ineq: Vec<Triplet>,
    ineq_rhs: Vec<f64>,
    infeasible: bool,
}

fn clamp_inf(v: f64) -> f64 {
    v.clamp(-INF, INF)
}

impl Builder {
    fn new(x_lo: &[f64], x_hi: &[f64]) -> Self {
        Builder {
            col_lo: x_lo.iter().map(|&v| clamp_inf(v)).collect(),
            col_hi: x_hi.iter().map(|&v| clamp_inf(v)).collect(),
            eq: Vec::new(),
            eq_rhs: Vec::new(),
            ineq: Vec::new(),
            ineq_rhs: Vec::new(),
            infeasible: false,
        }
    }

    fn add_col(&mut self, lo: f64, hi: f64) -> usize {
        let c = self.col_lo.len();
        self.col_lo.push(clamp_inf(lo));
        self.col_hi.push(clamp_inf(hi));
        c
    }

    /// Push `Σ coeff·handle  (=|≤) rhs`, folding constants into the RHS and
    /// summing duplicate column coefficients so each row has unique columns.
    fn row(&self, terms: &[(Handle, f64)], rhs: f64) -> (BTreeMap<usize, f64>, f64) {
        let mut cols: BTreeMap<usize, f64> = BTreeMap::new();
        let mut r = rhs;
        for &(h, coeff) in terms {
            match h {
                Handle::Col(c) => *cols.entry(c).or_insert(0.0) += coeff,
                Handle::Const(v) => r -= coeff * v,
            }
        }
        (cols, r)
    }

    fn emit_eq(&mut self, terms: &[(Handle, f64)], rhs: f64) {
        let (cols, r) = self.row(terms, rhs);
        let row = self.eq_rhs.len();
        for (c, v) in cols {
            if v != 0.0 {
                self.eq.push(Triplet::new(row, c, v));
            }
        }
        self.eq_rhs.push(r);
    }

    fn emit_le(&mut self, terms: &[(Handle, f64)], rhs: f64) {
        let (cols, r) = self.row(terms, rhs);
        if cols.is_empty() {
            // Pure constant inequality: 0 ≤ r. If violated the box is infeasible.
            if r < -1e-9 {
                self.infeasible = true;
            }
            return;
        }
        let row = self.ineq_rhs.len();
        for (c, v) in cols {
            if v != 0.0 {
                self.ineq.push(Triplet::new(row, c, v));
            }
        }
        self.ineq_rhs.push(r);
    }

    /// Four McCormick inequalities for `p = u·v` over `[uL,uU]×[vL,vU]`.
    #[allow(clippy::too_many_arguments)]
    fn bilinear(&mut self, p: Handle, u: Handle, v: Handle, ul: f64, uu: f64, vl: f64, vu: f64) {
        // Skip if any factor range is non-finite (envelope coefficients blow up);
        // the box bound on `p` then carries whatever interval info exists.
        if ![ul, uu, vl, vu].iter().all(|x| x.is_finite()) {
            return;
        }
        // p ≥ uL·v + vL·u − uL·vL
        self.emit_le(&[(p, -1.0), (v, ul), (u, vl)], ul * vl);
        // p ≥ uU·v + vU·u − uU·vU
        self.emit_le(&[(p, -1.0), (v, uu), (u, vu)], uu * vu);
        // p ≤ uU·v + vL·u − uU·vL
        self.emit_le(&[(p, 1.0), (v, -uu), (u, -vl)], -uu * vl);
        // p ≤ uL·v + vU·u − uL·vU
        self.emit_le(&[(p, 1.0), (v, -ul), (u, -vu)], -ul * vu);
    }

    /// Secant + tangent envelope of a univariate `f` (convex or concave) on
    /// `[al, au]`, linking `w = f(a)`.
    #[allow(clippy::too_many_arguments)]
    fn univariate(
        &mut self,
        w: usize,
        a: Handle,
        al: f64,
        au: f64,
        convex: bool,
        f: impl Fn(f64) -> f64,
        df: impl Fn(f64) -> f64,
    ) {
        let wh = Handle::Col(w);
        // Constant operand ⇒ w is a constant.
        if let Handle::Const(v) = a {
            self.emit_eq(&[(wh, 1.0)], f(v));
            return;
        }
        if !al.is_finite() || !au.is_finite() || au - al < 1e-12 {
            // Degenerate or unbounded domain: pin to the point value if we can,
            // else rely on the interval box bound already on column `w`.
            if al.is_finite() && au.is_finite() {
                self.emit_eq(&[(wh, 1.0)], f(0.5 * (al + au)));
            }
            return;
        }
        let s = (f(au) - f(al)) / (au - al); // secant slope
        let tangents = [al, 0.5 * (al + au), au];
        if convex {
            // Secant overestimates: w ≤ f(al) + s·(a − al).
            self.emit_le(&[(wh, 1.0), (a, -s)], f(al) - s * al);
            // Tangents underestimate: w ≥ f(c) + df(c)·(a − c).
            for c in tangents {
                let g = df(c);
                self.emit_le(&[(wh, -1.0), (a, g)], g * c - f(c));
            }
        } else {
            // Secant underestimates: w ≥ f(al) + s·(a − al).
            self.emit_le(&[(wh, -1.0), (a, s)], -(f(al) - s * al));
            // Tangents overestimate: w ≤ f(c) + df(c)·(a − c).
            for c in tangents {
                let g = df(c);
                self.emit_le(&[(wh, 1.0), (a, -g)], f(c) - g * c);
            }
        }
    }
}

/// Process one tape, appending its relaxation to `b`; return the root slot's
/// handle (the LP representation of the whole expression's value).
fn process_tape(b: &mut Builder, tape: &FbbtTape, x_lo: &[f64], x_hi: &[f64]) -> Option<Handle> {
    if tape.is_empty() {
        return None;
    }
    let ivals = forward_pass(tape, x_lo, x_hi).ok()?;
    let mut handle: Vec<Handle> = Vec::with_capacity(tape.ops.len());

    // A fresh aux column for slot `k`, bounded by its interval.
    macro_rules! new_col {
        ($k:expr) => {{
            let iv = ivals[$k];
            b.add_col(iv.lo, iv.hi)
        }};
    }
    let bounds = |h: Handle, k: usize| -> (f64, f64) {
        match h {
            Handle::Const(v) => (v, v),
            Handle::Col(_) => (ivals[k].lo, ivals[k].hi),
        }
    };

    for (k, op) in tape.ops.iter().enumerate() {
        let h = match *op {
            FbbtOp::Const(c) => Handle::Const(c),
            FbbtOp::Var(i) => Handle::Col(i), // shared original-variable column
            FbbtOp::Add(a, c) => {
                let col = new_col!(k);
                let w = Handle::Col(col);
                b.emit_eq(&[(w, 1.0), (handle[a], -1.0), (handle[c], -1.0)], 0.0);
                w
            }
            FbbtOp::Sub(a, c) => {
                let col = new_col!(k);
                let w = Handle::Col(col);
                b.emit_eq(&[(w, 1.0), (handle[a], -1.0), (handle[c], 1.0)], 0.0);
                w
            }
            FbbtOp::Neg(a) => {
                let col = new_col!(k);
                let w = Handle::Col(col);
                b.emit_eq(&[(w, 1.0), (handle[a], 1.0)], 0.0);
                w
            }
            FbbtOp::Mul(a, c) => {
                let col = new_col!(k);
                let w = Handle::Col(col);
                let (ha, hc) = (handle[a], handle[c]);
                match (ha, hc) {
                    (Handle::Const(va), _) => b.emit_eq(&[(w, 1.0), (hc, -va)], 0.0),
                    (_, Handle::Const(vc)) => b.emit_eq(&[(w, 1.0), (ha, -vc)], 0.0),
                    _ => {
                        let (al, au) = bounds(ha, a);
                        let (cl, cu) = bounds(hc, c);
                        b.bilinear(w, ha, hc, al, au, cl, cu);
                    }
                }
                w
            }
            FbbtOp::Div(a, c) => {
                let col = new_col!(k);
                let w = Handle::Col(col);
                let (ha, hc) = (handle[a], handle[c]);
                if let Handle::Const(vc) = hc {
                    if vc != 0.0 {
                        b.emit_eq(&[(w, 1.0), (ha, -1.0 / vc)], 0.0);
                    }
                } else {
                    let (cl, cu) = bounds(hc, c);
                    let (wl, wu) = (ivals[k].lo, ivals[k].hi);
                    // a = w·c, relaxed by McCormick (only sound when c avoids 0).
                    if cl > 1e-12 || cu < -1e-12 {
                        b.bilinear(ha, w, hc, wl, wu, cl, cu);
                    }
                }
                w
            }
            FbbtOp::PowInt(a, n) => {
                let col = new_col!(k);
                let w = Handle::Col(col);
                let ha = handle[a];
                let (al, au) = bounds(ha, a);
                match n {
                    0 => b.emit_eq(&[(w, 1.0)], 1.0),
                    1 => b.emit_eq(&[(w, 1.0), (ha, -1.0)], 0.0),
                    _ => {
                        let ni = n as i32;
                        let f = move |t: f64| t.powi(ni);
                        let df = move |t: f64| n as f64 * t.powi(ni - 1);
                        let even = n % 2 == 0;
                        if even || al >= 0.0 {
                            b.univariate(col, ha, al, au, true, f, df); // convex
                        } else if au <= 0.0 {
                            b.univariate(col, ha, al, au, false, f, df); // concave
                        }
                        // else straddles 0 with odd n: nonconvex → box bound only.
                    }
                }
                w
            }
            FbbtOp::Sqrt(a) => {
                let col = new_col!(k);
                let ha = handle[a];
                let (al, au) = bounds(ha, a);
                let al = al.max(0.0);
                // √ is concave on [0, ∞).
                b.univariate(
                    col,
                    ha,
                    al,
                    au,
                    false,
                    |t| t.sqrt(),
                    |t| 0.5 / t.max(1e-300).sqrt(),
                );
                Handle::Col(col)
            }
            FbbtOp::Exp(a) => {
                let col = new_col!(k);
                let ha = handle[a];
                let (al, au) = bounds(ha, a);
                b.univariate(col, ha, al, au, true, |t| t.exp(), |t| t.exp()); // convex
                Handle::Col(col)
            }
            FbbtOp::Ln(a) => {
                let col = new_col!(k);
                let ha = handle[a];
                let (al, au) = bounds(ha, a);
                let al = al.max(1e-12);
                b.univariate(col, ha, al, au, false, |t| t.ln(), |t| 1.0 / t); // concave
                Handle::Col(col)
            }
            FbbtOp::Abs(a) => {
                let col = new_col!(k);
                let w = Handle::Col(col);
                let ha = handle[a];
                let (al, au) = bounds(ha, a);
                // |·| is convex: w ≥ a, w ≥ −a, secant overestimator.
                b.emit_le(&[(w, -1.0), (ha, 1.0)], 0.0);
                b.emit_le(&[(w, -1.0), (ha, -1.0)], 0.0);
                if al.is_finite() && au.is_finite() && au - al > 1e-12 {
                    let s = (au.abs() - al.abs()) / (au - al);
                    b.emit_le(&[(w, 1.0), (ha, -s)], al.abs() - s * al);
                }
                w
            }
            // Nonconvex / unsupported: interval box bound on the column only.
            FbbtOp::Sin(_) | FbbtOp::Cos(_) | FbbtOp::Opaque => Handle::Col(new_col!(k)),
        };
        handle.push(h);
    }
    handle.last().copied()
}

/// Build the relaxation LP for `prob` over the box `[x_lo, x_hi]`.
pub(crate) fn build_relaxation(prob: &GlobalProblem, x_lo: &[f64], x_hi: &[f64]) -> Relaxation {
    let mut b = Builder::new(x_lo, x_hi);

    // Objective → LP cost on its root handle.
    let obj_handle = process_tape(&mut b, &prob.objective, x_lo, x_hi);

    // Constraints: bracket each root handle by [lo, hi].
    for con in &prob.constraints {
        match process_tape(&mut b, &con.tape, x_lo, x_hi) {
            Some(Handle::Col(c)) => {
                let h = Handle::Col(c);
                if con.hi < INF {
                    b.emit_le(&[(h, 1.0)], con.hi); //  g ≤ hi
                }
                if con.lo > -INF {
                    b.emit_le(&[(h, -1.0)], -con.lo); // −g ≤ −lo
                }
            }
            Some(Handle::Const(v)) => {
                if v > con.hi + 1e-9 || v < con.lo - 1e-9 {
                    b.infeasible = true;
                }
            }
            None => {}
        }
    }

    let n_cols = b.col_lo.len();
    let mut c = vec![0.0; n_cols];
    // A constant or empty objective leaves the cost vector zero (the bound is
    // then just the constant / zero, refined by branching).
    if let Some(Handle::Col(col)) = obj_handle {
        c[col] = 1.0;
    }

    let qp = QpProblem {
        n: n_cols,
        p_lower: Vec::new(),
        c,
        a: b.eq,
        b: b.eq_rhs,
        g: b.ineq,
        h: b.ineq_rhs,
        lb: b.col_lo,
        ub: b.col_hi,
    };
    Relaxation {
        qp,
        trivially_infeasible: b.infeasible,
    }
}

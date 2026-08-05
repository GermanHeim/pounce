//! Adversarial QP instance generators.
//!
//! Every instance carries its own *constructive* verdict — a feasibility
//! witness, or an exact-arithmetic proof of infeasibility — so the
//! harness never has to ask pounce whether pounce was right. The
//! generators are additionally adjudicated by `scipy.optimize.linprog`
//! (see `runs/*_adjudicate.py`), which guards against a bug in the
//! generator itself rather than in the solver.
//!
//! The knobs are chosen to reproduce the geometry that broke the
//! l1-elastic infeasibility certificate:
//!
//! * **indefinite `H`, including zero diagonals** — the shape of an
//!   exact ∇²L, which is what makes the elastic subproblem nonconvex and
//!   its residual slacks meaningless as a certificate;
//! * **near-parallel rows** — the tangency at which the feasible wedge
//!   pinches, so a solver that stops early lands just outside it;
//! * **row scaling by 10^±k** — γ = 1e6 turns a small primal error into
//!   a large apparent objective, so scale is the amplifier;
//! * **witnesses snapped onto bounds** — degenerate active sets, where
//!   "at the boundary" and "just outside it" differ by rounding;
//! * **boxes far from the origin** — the elastic seed is `0` projected
//!   into the box, so a distant box makes that seed useless and forces
//!   the recovery path that was inert on equality rows.

use crate::rng::Rng;
use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};

/// What we know about the instance independently of any solver.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Truth {
    /// A feasible point is attached in `witness`.
    Feasible,
    /// Infeasible by exact arithmetic; `proof` says why.
    Infeasible,
}

#[derive(Clone)]
pub struct Instance {
    pub seed: u64,
    pub kind: &'static str,
    pub truth: Truth,
    pub proof: String,
    pub n: usize,
    pub m: usize,
    /// Lower triangle, `(i, j, v)` with `i >= j`, 0-based.
    pub h: Vec<(usize, usize, f64)>,
    pub g: Vec<f64>,
    /// Row-major `m × n`, dense.
    pub a: Vec<f64>,
    pub bl: Vec<f64>,
    pub bu: Vec<f64>,
    pub xl: Vec<f64>,
    pub xu: Vec<f64>,
    pub witness: Option<Vec<f64>>,
}

impl Instance {
    pub fn row(&self, i: usize) -> &[f64] {
        &self.a[i * self.n..(i + 1) * self.n]
    }

    pub fn dot_row(&self, i: usize, x: &[f64]) -> f64 {
        self.row(i).iter().zip(x).map(|(a, v)| a * v).sum()
    }

    /// Largest violation of rows or box at `x`, in absolute terms.
    pub fn violation(&self, x: &[f64]) -> f64 {
        let mut worst: f64 = 0.0;
        for i in 0..self.m {
            let r = self.dot_row(i, x);
            if self.bl[i] > NLP_LOWER_BOUND_INF {
                worst = worst.max(self.bl[i] - r);
            }
            if self.bu[i] < NLP_UPPER_BOUND_INF {
                worst = worst.max(r - self.bu[i]);
            }
        }
        for j in 0..self.n {
            worst = worst.max(self.xl[j] - x[j]);
            worst = worst.max(x[j] - self.xu[j]);
        }
        worst.max(0.0)
    }

    /// Exact range of row `i` over the variable box, `(min, max)`.
    /// A linear function over a box attains its extremes at corners, so
    /// this is exact, not a bound.
    pub fn row_range_over_box(&self, i: usize) -> (f64, f64) {
        let mut lo = 0.0;
        let mut hi = 0.0;
        for j in 0..self.n {
            let a = self.row(i)[j];
            let (p, q) = (a * self.xl[j], a * self.xu[j]);
            lo += p.min(q);
            hi += p.max(q);
        }
        (lo, hi)
    }

    pub fn to_json(&self) -> String {
        fn arr(v: &[f64]) -> String {
            let parts: Vec<String> = v.iter().map(|x| format!("{x:.17e}")).collect();
            format!("[{}]", parts.join(","))
        }
        let witness = match &self.witness {
            Some(w) => arr(w),
            None => "null".to_string(),
        };
        format!(
            r#"{{"seed":{},"kind":"{}","truth":"{:?}","proof":{:?},"n":{},"m":{},"g":{},"a":{},"bl":{},"bu":{},"xl":{},"xu":{},"witness":{}}}"#,
            self.seed,
            self.kind,
            self.truth,
            self.proof,
            self.n,
            self.m,
            arr(&self.g),
            arr(&self.a),
            arr(&self.bl),
            arr(&self.bu),
            arr(&self.xl),
            arr(&self.xu),
            witness,
        )
    }
}


/// A finite, sane point inside the box.
///
/// `0.5·(xl + xu)` is the obvious choice and is wrong the moment a bound
/// is infinite: with `xl = -1e19` it returns `-5e18`, and every row bound
/// derived from it lands at ~1e19. The instance is then nominally valid
/// and numerically absurd — scipy disagreed with the constructive verdict
/// on three of them, which is precisely what the adjudicator is for.
fn box_interior_point(xl: &[f64], xu: &[f64]) -> Vec<f64> {
    xl.iter()
        .zip(xu.iter())
        .map(|(&l, &u)| {
            let lo_fin = l > NLP_LOWER_BOUND_INF;
            let hi_fin = u < NLP_UPPER_BOUND_INF;
            match (lo_fin, hi_fin) {
                (true, true) => 0.5 * (l + u),
                (true, false) => l + 1.0,
                (false, true) => u - 1.0,
                (false, false) => 0.0,
            }
        })
        .collect()
}

/// Symmetric `H` that is deliberately indefinite, with zero diagonal
/// entries — the exact-∇²L shape. Stored as a full lower triangle so the
/// pattern includes the structural zeros too, exactly as HS071's does.
fn indefinite_hessian(rng: &mut Rng, n: usize) -> Vec<(usize, usize, f64)> {
    let mut h = Vec::new();
    for i in 0..n {
        for j in 0..=i {
            let v = if i == j {
                // A third of the diagonal is exactly zero, a third
                // negative: no chance of accidental positive definiteness.
                match rng.int(0, 2) {
                    0 => 0.0,
                    1 => rng.range(-5.0, -0.1),
                    _ => rng.range(0.1, 5.0),
                }
            } else if rng.chance(0.35) {
                0.0
            } else {
                rng.range(-3.0, 3.0)
            };
            h.push((i, j, v));
        }
    }
    h
}

fn random_box(rng: &mut Rng, n: usize) -> (Vec<f64>, Vec<f64>) {
    // A quarter of the boxes sit far from the origin. The elastic seed is
    // `0` projected into the box, so a distant box makes that seed a
    // corner rather than an interior guess.
    let shift = if rng.chance(0.25) {
        rng.range(50.0, 500.0)
    } else {
        0.0
    };
    let mut xl = Vec::with_capacity(n);
    let mut xu = Vec::with_capacity(n);
    for _ in 0..n {
        let c = shift + rng.range(-3.0, 3.0);
        let w = rng.range(0.2, 4.0);
        // A quarter of the coordinates are free on one side or both.
        // Unbounded variables are ordinary — an SQP step QP inherits one
        // from every unbounded NLP variable — and they are the case where
        // the box gives no bound on how far a feasible point can sit, so
        // any reasoning that leans on the box has to cope without it.
        match rng.int(0, 7) {
            0 => {
                xl.push(NLP_LOWER_BOUND_INF);
                xu.push(c + w);
            }
            1 => {
                xl.push(c - w);
                xu.push(NLP_UPPER_BOUND_INF);
            }
            2 => {
                xl.push(NLP_LOWER_BOUND_INF);
                xu.push(NLP_UPPER_BOUND_INF);
            }
            _ => {
                xl.push(c - w);
                xu.push(c + w);
            }
        }
    }
    (xl, xu)
}

fn random_rows(rng: &mut Rng, n: usize, m: usize) -> Vec<f64> {
    let mut a = vec![0.0; m * n];
    for i in 0..m {
        if i > 0 && rng.chance(0.3) {
            // Near-parallel to the previous row: the tangency geometry.
            let eps = 10f64.powf(rng.range(-9.0, -4.0));
            let scale = rng.range(0.5, 2.0);
            for j in 0..n {
                a[i * n + j] = scale * a[(i - 1) * n + j] + eps * rng.range(-1.0, 1.0);
            }
        } else {
            for j in 0..n {
                a[i * n + j] = rng.range(-3.0, 3.0);
            }
        }
        // Row scaling: γ = 1e6 makes scale the amplifier.
        if rng.chance(0.3) {
            let s = 10f64.powi(rng.int(0, 12) as i32 - 6);
            for j in 0..n {
                a[i * n + j] *= s;
            }
        }
        // Never emit an all-zero row: its feasibility is a triviality
        // that says nothing about the solver.
        if a[i * n..(i + 1) * n].iter().all(|v| v.abs() < 1e-300) {
            a[i * n] = 1.0;
        }
    }
    a
}

/// A QP with a feasibility **witness** built in: the row bounds are
/// derived from `A·x_w`, so `x_w` satisfies every row and sits in the
/// box by construction. No solver may call this infeasible.
pub fn feasible(rng: &mut Rng, seed: u64) -> Instance {
    let n = rng.int(2, 6);
    let m = rng.int(1, 5);
    let (xl, xu) = random_box(rng, n);

    let mut w = Vec::with_capacity(n);
    for j in 0..n {
        // 40% of coordinates snap exactly onto a bound: degenerate
        // active sets, where "on the boundary" and "just outside" are a
        // rounding apart.
        let lo_fin = xl[j] > NLP_LOWER_BOUND_INF;
        let hi_fin = xu[j] < NLP_UPPER_BOUND_INF;
        w.push(match (lo_fin, hi_fin) {
            // 40% of bounded coordinates snap exactly onto a bound:
            // degenerate active sets, where "on the boundary" and "just
            // outside" are a rounding apart.
            (true, true) if rng.chance(0.4) => {
                if rng.chance(0.5) { xl[j] } else { xu[j] }
            }
            (true, true) => rng.range(xl[j], xu[j]),
            (true, false) => xl[j] + rng.range(0.0, 4.0),
            (false, true) => xu[j] - rng.range(0.0, 4.0),
            (false, false) => rng.range(-4.0, 4.0),
        });
    }

    let a = random_rows(rng, n, m);
    let mut inst = Instance {
        seed,
        kind: "feasible",
        truth: Truth::Feasible,
        proof: "row bounds derived from A·x_w with x_w inside the box".into(),
        n,
        m,
        h: indefinite_hessian(rng, n),
        g: (0..n).map(|_| rng.range(-10.0, 10.0)).collect(),
        a,
        bl: vec![0.0; m],
        bu: vec![0.0; m],
        xl,
        xu,
        witness: None,
    };

    for i in 0..m {
        let r = inst.dot_row(i, &w);
        let mag = r.abs().max(1.0);
        match rng.int(0, 3) {
            // Equality row. The class the recovery path silently failed
            // to enforce.
            0 => {
                inst.bl[i] = r;
                inst.bu[i] = r;
            }
            // One-sided, witness exactly on the boundary (active).
            1 => {
                inst.bl[i] = r;
                inst.bu[i] = NLP_UPPER_BOUND_INF;
            }
            // One-sided with slack.
            2 => {
                inst.bl[i] = r - rng.range(1e-9, 1.0) * mag;
                inst.bu[i] = NLP_UPPER_BOUND_INF;
            }
            // Two-sided bracket, sometimes vanishingly tight.
            _ => {
                let s = rng.range(1e-9, 0.5) * mag;
                inst.bl[i] = r - s;
                inst.bu[i] = r + s;
            }
        }
    }
    inst.witness = Some(w);
    inst
}

/// A QP that is infeasible by **exact arithmetic**, in one of two ways
/// that need no solver to see.
pub fn infeasible(rng: &mut Rng, seed: u64) -> Instance {
    let n = rng.int(2, 6);
    let (xl, xu) = random_box(rng, n);
    // `max_box aᵀx` is only a *proof* when the box is finite; with a free
    // coordinate the row can be driven anywhere and the construction
    // proves nothing. Contradictory equalities are unaffected — they are
    // infeasible whatever the box — so route unbounded boxes there.
    let box_finite = xl
        .iter()
        .zip(xu.iter())
        .all(|(l, u)| *l > NLP_LOWER_BOUND_INF && *u < NLP_UPPER_BOUND_INF);

    if rng.chance(0.5) || !box_finite {
        // (a) Contradictory equalities: the same row `a` required to
        // equal two different values. `aᵀx` is a function, so no x can
        // satisfy both, whatever the box.
        let m = rng.int(2, 4);
        let mut a = random_rows(rng, n, m);
        let dup_src = 0usize;
        let dup_dst = rng.int(1, m - 1);
        for j in 0..n {
            a[dup_dst * n + j] = a[dup_src * n + j];
        }
        let mut inst = Instance {
            seed,
            kind: "infeasible-contradictory-equalities",
            truth: Truth::Infeasible,
            proof: format!("rows {dup_src} and {dup_dst} are identical but required to equal different values"),
            n,
            m,
            h: indefinite_hessian(rng, n),
            g: (0..n).map(|_| rng.range(-10.0, 10.0)).collect(),
            a,
            bl: vec![0.0; m],
            bu: vec![0.0; m],
            xl,
            xu,
            witness: None,
        };
        // Give the other rows an achievable value so the *only* source of
        // infeasibility is the contradiction we planted.
        let mid = box_interior_point(&inst.xl, &inst.xu);
        for i in 0..m {
            let r = inst.dot_row(i, &mid);
            inst.bl[i] = r;
            inst.bu[i] = r;
        }
        let c1 = inst.bl[dup_src];
        let gap = c1.abs().max(1.0) * rng.range(0.5, 5.0);
        inst.bl[dup_dst] = c1 + gap;
        inst.bu[dup_dst] = c1 + gap;
        inst
    } else {
        // (b) A row required to exceed what it can attain. The range of a
        // linear function over a box is attained at corners, so
        // `max_box aᵀx = hi` exactly; demanding `aᵀx >= hi + δ` is
        // unsatisfiable.
        let m = rng.int(1, 4);
        let a = random_rows(rng, n, m);
        let mut inst = Instance {
            seed,
            kind: "infeasible-box-range",
            truth: Truth::Infeasible,
            proof: String::new(),
            n,
            m,
            h: indefinite_hessian(rng, n),
            g: (0..n).map(|_| rng.range(-10.0, 10.0)).collect(),
            a,
            bl: vec![0.0; m],
            bu: vec![0.0; m],
            xl,
            xu,
            witness: None,
        };
        let mid = box_interior_point(&inst.xl, &inst.xu);
        for i in 0..m {
            let r = inst.dot_row(i, &mid);
            inst.bl[i] = r - r.abs().max(1.0);
            inst.bu[i] = NLP_UPPER_BOUND_INF;
        }
        let bad = rng.int(0, m - 1);
        let (_, hi) = inst.row_range_over_box(bad);
        let delta = hi.abs().max(1.0) * rng.range(0.1, 2.0);
        inst.bl[bad] = hi + delta;
        inst.bu[bad] = NLP_UPPER_BOUND_INF;
        inst.proof = format!(
            "row {bad} attains at most {hi:.6e} over the box but is required to be >= {:.6e}",
            inst.bl[bad]
        );
        inst
    }
}

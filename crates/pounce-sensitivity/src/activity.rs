//! Post-solve activity classification (the covariance/information
//! roadmap's item 0, gh #362).
//!
//! Classifies every bounded variable and every finite-bounded inequality
//! row of a converged barrier solve into one of five statuses, keyed on
//! the ratio of barrier curvature to the objective's own curvature:
//!
//! ```text
//! r = Σ / q,   Σ = z/s summed over the sides that exist,
//!              q = |H_ii|                        (variable)
//!                  |∇dⱼᵀ H ∇dⱼ| / ‖∇dⱼ‖²         (inequality row)
//! ```
//!
//! `r` is `O(μ)` when the bound is inactive, `O(1)` when weakly active
//! (slack and multiplier vanish together), and `O(1/μ)` when strongly
//! active, so one ratio separates the regimes at any `μ` where a fixed
//! threshold on the slack or the multiplier alone cannot: both are
//! `O(√μ)` at weak activity, so any constant tracks the solve rather
//! than the geometry.
//!
//! Everything read here is retained by the converged state the
//! backsolver already holds: the bound multipliers on the iterate, the
//! solver's own slacks, `Σ` as `curr_sigma_x` / `curr_sigma_s`, the
//! barrier parameter, and the exact Lagrangian Hessian, so `H` is
//! never recovered from the barrier-augmented factor.

use std::rc::Rc;

use pounce_common::types::Number;
use pounce_linalg::Matrix;
use pounce_linalg::dense_vector::{DenseVector, DenseVectorSpace};
use pounce_linalg::expansion_matrix::ExpansionMatrix;

use crate::PdSensBacksolver;
use crate::vec_util::dense_to_vec;

/// No finite bound on this variable or row: nothing to classify.
pub const UNBOUNDED: i8 = -1;
/// `r = O(μ)`: the bound is not doing anything.
pub const INACTIVE: i8 = 0;
/// `r = O(1)`: slack and multiplier vanish together; kept, flagged.
pub const WEAKLY_ACTIVE: i8 = 1;
/// `r = O(1/μ)`: the bound holds the variable; projected out.
pub const STRONGLY_ACTIVE: i8 = 2;
/// `r` in a gap between the band and a `μ`-edge: undetermined at this
/// `μ`; re-solving tighter separates it.
pub const AMBIGUOUS: i8 = 3;
/// The curvature `q` is below noise scale: the bound question does not
/// arise, and the direction is poorly identified.
pub const UNIDENTIFIED: i8 = 4;

/// Per-variable and per-row classification of a converged solve.
///
/// Vectors are full-length (`n` variables, `m_d` inequality rows);
/// entries with no finite bound hold [`UNBOUNDED`] and `NaN` ratios.
pub struct ActivityReport {
    /// Barrier parameter of the converged iterate.
    pub mu: Number,
    /// Status per variable (codes above).
    pub var_status: Vec<i8>,
    /// `Σ_i / q_i` per variable; `NaN` where unbounded.
    pub var_ratio: Vec<Number>,
    /// Sign of the signed curvature `H_ii` (−1, 0, +1); the absolute
    /// value goes into `q`, so an indefinite direction is reported
    /// rather than hidden.
    pub var_q_sign: Vec<i8>,
    /// `s·z` differs from `μ` by more than a factor of ten on some
    /// side: off the central path, or the bound was relaxed.
    pub var_off_central_path: Vec<bool>,
    /// Classified inactive yet `r` non-negligible: barrier curvature
    /// where none should be.
    pub var_contaminated: Vec<bool>,
    /// Status per inequality row.
    pub row_status: Vec<i8>,
    /// `Σ_j / q_j` per row; `NaN` where the row has no finite bound.
    pub row_ratio: Vec<Number>,
    /// Sign of the signed row curvature `∇dⱼᵀ H ∇dⱼ`.
    pub row_q_sign: Vec<i8>,
    /// Central-path check per row, as for variables.
    pub row_off_central_path: Vec<bool>,
}

/// The classification rule of the roadmap's item 0.
fn classify(r: Number, mu: Number) -> i8 {
    if mu > 1e-4 {
        // the μ-edges have closed inside the fixed band: only the two
        // clear calls are made, everything else is honest refusal
        if r < 1e-1 {
            INACTIVE
        } else if r > 1e1 {
            STRONGLY_ACTIVE
        } else {
            AMBIGUOUS
        }
    } else if r < mu.sqrt() {
        INACTIVE
    } else if r > 1.0 / mu.sqrt() {
        STRONGLY_ACTIVE
    } else if (1e-1..=1e1).contains(&r) {
        WEAKLY_ACTIVE
    } else {
        AMBIGUOUS
    }
}

fn sign_of(x: Number) -> i8 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

/// Scatter a compressed (bounded-entries-only) vector to full length
/// through its expansion matrix. Entries without that bound stay 0.
fn expand(compressed: &[Number], px: &Rc<dyn Matrix>, n: usize) -> Vec<Number> {
    let mut full = vec![0.0; n];
    if let Some(em) = px.as_any().downcast_ref::<ExpansionMatrix>() {
        for (k, &pos) in em.expanded_pos_indices().iter().enumerate() {
            if k < compressed.len() {
                full[pos as usize] = compressed[k];
            }
        }
    }
    full
}

/// Presence mask for a bound side, from the same expansion.
fn present(px: &Rc<dyn Matrix>, n: usize) -> Vec<bool> {
    let mut mask = vec![false; n];
    if let Some(em) = px.as_any().downcast_ref::<ExpansionMatrix>() {
        for &pos in em.expanded_pos_indices() {
            mask[pos as usize] = true;
        }
    }
    mask
}

/// `y = H · e_i` restricted to entry `i`: the exact Hessian diagonal,
/// one sparse product per bounded variable.
fn hessian_diag_entry(
    hess: &Rc<dyn pounce_linalg::SymMatrix>,
    i: usize,
    n: usize,
    work_in: &mut DenseVector,
    work_out: &mut DenseVector,
) -> Number {
    work_in.values_mut().fill(0.0);
    work_in.values_mut()[i] = 1.0;
    work_out.values_mut().fill(0.0);
    hess.mult_vector(1.0, work_in, 0.0, work_out);
    // values_mut, not values: a zero product may have left the output
    // homogeneous (empty backing slice); this materializes it
    let out = work_out.values_mut();
    debug_assert_eq!(out.len(), n);
    out[i]
}

/// Central-path check for one side: `s·z` within a factor of ten of `μ`.
fn off_path(s: Number, z: Number, mu: Number) -> bool {
    let comp = s * z;
    comp > 10.0 * mu || comp < 0.1 * mu
}

/// `r` above this while classified inactive flags contamination.
const CONTAMINATION_FLOOR: Number = 1e-3;

pub(crate) fn compute(bs: &PdSensBacksolver) -> ActivityReport {
    let (data, cq, nlp) = bs.activity_handles();

    // scoped borrows: the Cq getters below re-borrow the NLP (mutably,
    // for lazy evaluation) and the data, so nothing here may hold
    // either across a Cq call
    let (mu, mult_z_l, mult_z_u, mult_v_l, mult_v_u, n, m_d) = {
        let d = data.borrow();
        let curr = d.curr.as_ref().expect("converged state has an iterate");
        (
            d.curr_mu,
            Rc::clone(&curr.z_l),
            Rc::clone(&curr.z_u),
            Rc::clone(&curr.v_l),
            Rc::clone(&curr.v_u),
            curr.x.dim() as usize,
            curr.s.dim() as usize,
        )
    };
    let (px_l, px_u, pd_l, pd_u) = {
        let nl = nlp.borrow();
        (nl.px_l(), nl.px_u(), nl.pd_l(), nl.pd_u())
    };
    let cq = cq.borrow();

    // --- variables -----------------------------------------------------
    let has_l = present(&px_l, n);
    let has_u = present(&px_u, n);
    let z_l = expand(&dense_to_vec(mult_z_l.as_ref()), &px_l, n);
    let z_u = expand(&dense_to_vec(mult_z_u.as_ref()), &px_u, n);
    let s_l = expand(&dense_to_vec(cq.curr_slack_x_l().as_ref()), &px_l, n);
    let s_u = expand(&dense_to_vec(cq.curr_slack_x_u().as_ref()), &px_u, n);
    let sigma_x = dense_to_vec(cq.curr_sigma_x().as_ref());

    let hess = cq.curr_exact_hessian();
    let space = DenseVectorSpace::new(n as i32);
    let mut w_in = DenseVector::new(space.clone());
    let mut w_out = DenseVector::new(space);

    // the identification floor is relative to the largest curvature
    // anywhere on the diagonal, not just the bounded entries, so a
    // row-only model still measures q against the model's own scale
    let mut diag = vec![0.0; n];
    let mut max_abs_diag: Number = 0.0;
    for i in 0..n {
        diag[i] = hessian_diag_entry(&hess, i, n, &mut w_in, &mut w_out);
        max_abs_diag = max_abs_diag.max(diag[i].abs());
    }
    let floor = Number::EPSILON.sqrt() * max_abs_diag.max(1.0);

    let mut var_status = vec![UNBOUNDED; n];
    let mut var_ratio = vec![Number::NAN; n];
    let mut var_q_sign = vec![0i8; n];
    let mut var_off = vec![false; n];
    let mut var_cont = vec![false; n];
    for i in 0..n {
        if !(has_l[i] || has_u[i]) {
            continue;
        }
        var_q_sign[i] = sign_of(diag[i]);
        let q = diag[i].abs();
        if q < floor {
            var_status[i] = UNIDENTIFIED;
            var_ratio[i] = sigma_x[i] / floor;
            continue;
        }
        let r = sigma_x[i] / q;
        var_ratio[i] = r;
        var_status[i] = classify(r, mu);
        var_off[i] = (has_l[i] && off_path(s_l[i], z_l[i], mu))
            || (has_u[i] && off_path(s_u[i], z_u[i], mu));
        var_cont[i] = var_status[i] == INACTIVE && r > CONTAMINATION_FLOOR;
    }

    // --- inequality rows ----------------------------------------------
    let rhas_l = present(&pd_l, m_d);
    let rhas_u = present(&pd_u, m_d);
    let v_l = expand(&dense_to_vec(mult_v_l.as_ref()), &pd_l, m_d);
    let v_u = expand(&dense_to_vec(mult_v_u.as_ref()), &pd_u, m_d);
    let rs_l = expand(&dense_to_vec(cq.curr_slack_s_l().as_ref()), &pd_l, m_d);
    let rs_u = expand(&dense_to_vec(cq.curr_slack_s_u().as_ref()), &pd_u, m_d);
    let sigma_s = dense_to_vec(cq.curr_sigma_s().as_ref());

    let jac_d = cq.curr_jac_d();
    let mspace = DenseVectorSpace::new(m_d as i32);
    let mut e_row = DenseVector::new(mspace);
    let nspace = DenseVectorSpace::new(n as i32);
    let mut grad = DenseVector::new(nspace.clone());
    let mut hgrad = DenseVector::new(nspace);

    let mut row_status = vec![UNBOUNDED; m_d];
    let mut row_ratio = vec![Number::NAN; m_d];
    let mut row_q_sign = vec![0i8; m_d];
    let mut row_off = vec![false; m_d];
    for j in 0..m_d {
        if !(rhas_l[j] || rhas_u[j]) {
            continue;
        }
        // ∇dⱼ = Jdᵀ eⱼ, then the curvature along the normal;
        // values_mut throughout because a zero product may leave the
        // output homogeneous (empty backing slice behind values())
        e_row.values_mut().fill(0.0);
        e_row.values_mut()[j] = 1.0;
        grad.values_mut().fill(0.0);
        jac_d.trans_mult_vector(1.0, &e_row, 0.0, &mut grad);
        let norm2: Number = grad.values_mut().iter().map(|g| *g * *g).sum();
        if norm2 <= 0.0 {
            continue;
        }
        hgrad.values_mut().fill(0.0);
        hess.mult_vector(1.0, &grad, 0.0, &mut hgrad);
        let ghg: Number = {
            let h = hgrad.values_mut();
            grad.values_mut()
                .iter()
                .zip(h.iter())
                .map(|(g, h)| g * h)
                .sum()
        };
        let q_signed = ghg / norm2;
        row_q_sign[j] = sign_of(q_signed);
        let q = q_signed.abs();
        if q < floor {
            row_status[j] = UNIDENTIFIED;
            row_ratio[j] = sigma_s[j] / floor;
            continue;
        }
        let r = sigma_s[j] / q;
        row_ratio[j] = r;
        row_status[j] = classify(r, mu);
        row_off[j] = (rhas_l[j] && off_path(rs_l[j], v_l[j], mu))
            || (rhas_u[j] && off_path(rs_u[j], v_u[j], mu));
    }

    ActivityReport {
        mu,
        var_status,
        var_ratio,
        var_q_sign,
        var_off_central_path: var_off,
        var_contaminated: var_cont,
        row_status,
        row_ratio,
        row_q_sign,
        row_off_central_path: row_off,
    }
}

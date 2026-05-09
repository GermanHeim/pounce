//! Limited-memory quasi-Newton (L-BFGS / SR1) — port of
//! `Algorithm/IpLimMemQuasiNewtonUpdater.{hpp,cpp}`. **Phase 8.**
//!
//! Update strategy is selected by the `limited_memory_update_type`
//! option (`bfgs` or `sr1`) per `MAIN_LOOP.md`.
//!
//! Phase 8a (this cut) ships a **dense-rebuild** assembler: at every
//! `update_hessian` call we walk the curvature-pair history (oldest to
//! newest) applying the rank-2 BFGS / rank-1 SR1 formulas in place on a
//! dense `n×n` work buffer, then project the result onto the
//! [`SymTMatrixSpace`] sparsity that [`crate::kkt::std_aug_system_solver`]
//! pinned at the first solve. The dense walk is `O(m · n²)` per
//! iteration (with `m = max_history`); for the small-NLP gold suite
//! that's negligible. The Sherman-Morrison-Woodbury / `LowRankUpdateSymMatrix`
//! shortcut wired through [`crate::kkt::low_rank_aug_system_solver`] is
//! the Phase-8b optimisation; mathematically the two paths agree on the
//! action of `B`, so HS071-class problems converge identically with
//! either backend.
//!
//! Update kernels:
//!   - [`initial_hessian_scalar`] (sigma per `LIM_MEM_INIT`)
//!   - [`powell_damping_theta`] (modified-y damping for BFGS)
//!   - [`bfgs_curvature_pair_ok`] (skip-criterion for L-BFGS)
//!   - [`sr1_denominator_ok`] (skip-criterion for SR1)

use crate::hess::r#trait::HessianUpdater;
use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use pounce_common::types::{Index, Number};
use pounce_linalg::dense_vector::DenseVector;
use pounce_linalg::triplet::{SymTMatrix, SymTMatrixSpace};
use pounce_linalg::Vector;
use std::rc::Rc;

/// One curvature pair `(s, y)` plus the cached `||s||`, `||y||`, `s·y`
/// scalars the BFGS / SR1 update kernels need on every history walk.
#[derive(Debug, Clone)]
pub struct CurvaturePair {
    pub s: Rc<dyn Vector>,
    pub y: Rc<dyn Vector>,
    pub s_dot_y: Number,
    pub s_norm: Number,
    pub y_norm: Number,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateType {
    Bfgs,
    Sr1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitialApprox {
    Identity,
    Scalar1,
    Scalar2,
}

pub struct LimMemQuasiNewtonUpdater {
    pub update_type: UpdateType,
    pub initial_approx: InitialApprox,
    pub max_history: i32,
    /// Powell-damping threshold. Default per upstream
    /// `IpLimMemQuasiNewtonUpdater.cpp:RegisterOptions`:
    /// `limited_memory_init_val_max=1e8` (clamp on initial sigma);
    /// the damping coefficient is hard-coded at 0.2 in the BFGS path.
    pub init_val_max: Number,
    pub init_val_min: Number,
    /// Rolling FIFO of curvature pairs, oldest at index 0. Capped at
    /// `max_history`; insertion drops the front.
    pub history: Vec<CurvaturePair>,
    /// `x` from the previous `update_hessian` call. None on the first
    /// iteration.
    pub last_x: Option<Rc<dyn Vector>>,
    /// `∇f(x_prev)` cached for the upstream y-difference formula
    /// (`IpLimMemQuasiNewtonUpdater.cpp:284`).
    pub last_grad_f: Option<Rc<dyn Vector>>,
    /// `J_c(x_prev)` cached for `J_c_prev^T · y_c_curr` in the
    /// y-difference. Stored as the trait object so we can call
    /// `trans_mult_vector` against the *current* multipliers — the
    /// upstream formula evaluates both Jacobians against `y_c_curr`
    /// (NOT `y_c_prev`).
    pub last_jac_c: Option<Rc<dyn pounce_linalg::matrix::Matrix>>,
    pub last_jac_d: Option<Rc<dyn pounce_linalg::matrix::Matrix>>,
    /// Cached `SymTMatrixSpace` we reuse to publish `data.w`. Filled
    /// lazily on the first `update_hessian` call by snapshotting the
    /// space of `cq.curr_exact_hessian()` so our matrices share the
    /// sparsity that the aug-system solver pinned during the
    /// least-square-multipliers init pass. Phase-8a limitation: when
    /// the user-declared sparsity is sparser than full-dense, the
    /// dense L-BFGS approximation is projected onto it (entries
    /// outside the declared pattern are dropped).
    pub h_space: Option<Rc<SymTMatrixSpace>>,
}

impl Default for LimMemQuasiNewtonUpdater {
    fn default() -> Self {
        Self {
            update_type: UpdateType::Bfgs,
            initial_approx: InitialApprox::Scalar2,
            max_history: 6,
            init_val_max: 1e8,
            init_val_min: 1e-8,
            history: Vec::new(),
            last_x: None,
            last_grad_f: None,
            last_jac_c: None,
            last_jac_d: None,
            h_space: None,
        }
    }
}

impl LimMemQuasiNewtonUpdater {
    pub fn new() -> Self {
        Self::default()
    }

    /// Try to absorb a new curvature pair. Returns `true` when the
    /// pair was accepted (and pushed to history), `false` when the
    /// skip-criterion rejected it. The caller owns `s` and `y` as
    /// `Rc<dyn Vector>` so the history can retain them cheaply.
    ///
    /// This matches the per-iteration path in
    /// `IpLimMemQuasiNewtonUpdater.cpp:Update` after the `(x, ∇L)`
    /// difference has been formed: skip-or-keep, then push, then
    /// trim the history to `max_history`.
    pub fn ingest_pair(&mut self, s: Rc<dyn Vector>, y: Rc<dyn Vector>) -> bool {
        let s_dot_y = s.dot(&*y);
        let s_norm = s.nrm2();
        let y_norm = y.nrm2();
        let accept = match self.update_type {
            UpdateType::Bfgs => bfgs_curvature_pair_ok(s_dot_y, s_norm, y_norm),
            UpdateType::Sr1 => {
                // SR1's skip-criterion is `(y - Bs)^T s` not `s^T y`;
                // without `B` available here we use the upstream
                // fallback of `s^T y` magnitude as the gating heuristic
                // (a more accurate test lands once the low-rank matrix
                // is wired in).
                sr1_denominator_ok(s_dot_y, s_norm, y_norm)
            }
        };
        if !accept {
            return false;
        }
        self.history.push(CurvaturePair {
            s,
            y,
            s_dot_y,
            s_norm,
            y_norm,
        });
        // Drop oldest pairs to honor the memory budget.
        while self.history.len() > self.max_history.max(0) as usize {
            self.history.remove(0);
        }
        true
    }
}

impl HessianUpdater for LimMemQuasiNewtonUpdater {
    /// Snapshot the current `(x, ∇_x L)` pair, build `s = x − x_prev`
    /// and `y = ∇L − ∇L_prev`, ingest into history (skip per the
    /// BFGS / SR1 acceptance criterion), then rebuild `B = σ I + Σ
    /// rank-2 updates` from the rolling history and publish it as
    /// `data.w`. Mirrors `IpLimMemQuasiNewtonUpdater::Update` minus the
    /// SMW shortcut (Phase-8a; the `LowRankUpdateSymMatrix` path lands
    /// in Phase-8b).
    fn update_hessian(&mut self, data: &IpoptDataHandle, cq: &IpoptCqHandle) -> bool {
        let (curr_x, curr_y_c, curr_y_d) = match data.borrow().curr.as_ref() {
            Some(c) => (c.x.clone(), c.y_c.clone(), c.y_d.clone()),
            None => return true,
        };
        let curr_grad_f = cq.borrow().curr_grad_f();
        let curr_jac_c = cq.borrow().curr_jac_c();
        let curr_jac_d = cq.borrow().curr_jac_d();

        // Lazily snapshot the SymTMatrix space pinned by upstream's
        // first aug-system call (least-square-multipliers init), so our
        // rebuilt `W` matrices reuse exactly the irow/jcol pattern the
        // solver already committed to.
        if self.h_space.is_none() {
            let h = cq.borrow().curr_exact_hessian();
            if let Some(symt) = h.as_any().downcast_ref::<SymTMatrix>() {
                self.h_space = Some(Rc::clone(symt.space()));
            } else {
                self.h_space = Some(make_full_lower_triangle_space(curr_x.dim()));
            }
        }

        // Upstream y formula (`IpLimMemQuasiNewtonUpdater.cpp:284-308`):
        //   y = (∇f_curr − ∇f_last)
        //     + (J_c_curr^T − J_c_last^T) · y_c_curr
        //     + (J_d_curr^T − J_d_last^T) · y_d_curr
        // i.e. the change in the *NLP* Lagrangian gradient (no bound
        // multipliers) where BOTH Jacobians are dotted against the
        // CURRENT y_c/y_d. Using `curr_grad_lag_x` here would inject
        // the bound-multiplier delta into y, which collapses spuriously
        // when μ drops and corrupts the BFGS update.
        if let (Some(prev_x), Some(prev_grad_f), Some(prev_jac_c), Some(prev_jac_d)) = (
            self.last_x.clone(),
            self.last_grad_f.clone(),
            self.last_jac_c.clone(),
            self.last_jac_d.clone(),
        ) {
            let mut s = curr_x.make_new();
            s.add_two_vectors(1.0, &*curr_x, -1.0, &*prev_x, 0.0);

            let mut y = curr_x.make_new();
            // y = ∇f_curr − ∇f_last
            y.add_two_vectors(1.0, &*curr_grad_f, -1.0, &*prev_grad_f, 0.0);
            // y += J_c_curr^T y_c_curr  −  J_c_last^T y_c_curr
            curr_jac_c.trans_mult_vector(1.0, &*curr_y_c, 1.0, &mut *y);
            prev_jac_c.trans_mult_vector(-1.0, &*curr_y_c, 1.0, &mut *y);
            // y += J_d_curr^T y_d_curr  −  J_d_last^T y_d_curr
            curr_jac_d.trans_mult_vector(1.0, &*curr_y_d, 1.0, &mut *y);
            prev_jac_d.trans_mult_vector(-1.0, &*curr_y_d, 1.0, &mut *y);

            self.ingest_pair(Rc::from(s), Rc::from(y));
        }
        self.last_x = Some(Rc::clone(&curr_x));
        self.last_grad_f = Some(Rc::clone(&curr_grad_f));
        self.last_jac_c = Some(Rc::clone(&curr_jac_c));
        self.last_jac_d = Some(Rc::clone(&curr_jac_d));

        let n = curr_x.dim() as usize;
        let sigma = match self.update_type {
            UpdateType::Bfgs => self.compute_sigma_bfgs(),
            // SR1 always uses identity start unless the user picked
            // `Scalar1`/`Scalar2`. Upstream's SR1 path matches BFGS for
            // the `LIM_MEM_INIT` sigma source.
            UpdateType::Sr1 => self.compute_sigma_bfgs(),
        };
        let mut dense = vec![0.0_f64; n * n];
        for i in 0..n {
            dense[i * n + i] = sigma;
        }
        match self.update_type {
            UpdateType::Bfgs => apply_bfgs_history(&mut dense, n, &self.history),
            UpdateType::Sr1 => apply_sr1_history(&mut dense, n, &self.history),
        }

        let space = Rc::clone(self.h_space.as_ref().unwrap());
        let mut mat = SymTMatrix::new(Rc::clone(&space));
        let irows = space.irows();
        let jcols = space.jcols();
        let mut vals = vec![0.0_f64; irows.len()];
        for k in 0..irows.len() {
            // Triplet indices are 1-based per upstream / MUMPS convention.
            let i = (irows[k] - 1) as usize;
            let j = (jcols[k] - 1) as usize;
            vals[k] = dense[i * n + j];
        }
        mat.set_values(&vals);
        data.borrow_mut().w = Some(Rc::new(mat));
        true
    }
}

impl LimMemQuasiNewtonUpdater {
    fn compute_sigma_bfgs(&self) -> Number {
        if self.history.is_empty() {
            return 1.0;
        }
        let last = self.history.last().unwrap();
        let s_dot_s = last.s_norm * last.s_norm;
        let y_dot_y = last.y_norm * last.y_norm;
        initial_hessian_scalar(
            self.initial_approx,
            s_dot_s,
            last.s_dot_y,
            y_dot_y,
            self.init_val_min,
            self.init_val_max,
        )
    }
}

/// Construct a `SymTMatrixSpace` whose pattern is the full lower
/// triangle (1-based) of an `n × n` matrix. Used as a fallback when
/// `cq.curr_exact_hessian()` cannot supply a `SymTMatrix` (e.g. the
/// TNLP elects not to implement `eval_h`).
fn make_full_lower_triangle_space(n: Index) -> Rc<SymTMatrixSpace> {
    let nz = (n as usize) * ((n as usize) + 1) / 2;
    let mut irows: Vec<Index> = Vec::with_capacity(nz);
    let mut jcols: Vec<Index> = Vec::with_capacity(nz);
    for i in 1..=n {
        for j in 1..=i {
            irows.push(i);
            jcols.push(j);
        }
    }
    SymTMatrixSpace::new(n, irows, jcols)
}

fn dense_from_vec(v: &dyn Vector, n: usize) -> Vec<Number> {
    if let Some(dv) = v.as_any().downcast_ref::<DenseVector>() {
        let ev = dv.expanded_values();
        debug_assert_eq!(ev.len(), n);
        return ev;
    }
    panic!("LimMemQuasiNewtonUpdater: curvature pairs must be DenseVector-backed");
}

/// In-place BFGS history walk on a column-major-or-row-major (it
/// doesn't matter; B is symmetric) dense `n × n` buffer. For each
/// curvature pair `(s, y)`, applies Powell damping when the curvature
/// is too negative, then writes
/// `B ← B + (r r^T)/(s^T r) − (B s)(B s)^T / (s^T B s)`
/// where `r = θ y + (1−θ) B s`. Mirrors
/// `IpLimMemQuasiNewtonUpdater.cpp::Update_BFGS` semantics.
fn apply_bfgs_history(b: &mut [Number], n: usize, history: &[CurvaturePair]) {
    if n == 0 {
        return;
    }
    let mut bs = vec![0.0_f64; n];
    let mut r = vec![0.0_f64; n];
    for pair in history {
        let s = dense_from_vec(pair.s.as_ref(), n);
        let y = dense_from_vec(pair.y.as_ref(), n);
        // bs = B s.
        for i in 0..n {
            let row = &b[i * n..(i + 1) * n];
            let mut acc = 0.0;
            for j in 0..n {
                acc += row[j] * s[j];
            }
            bs[i] = acc;
        }
        let s_bs: Number = (0..n).map(|i| s[i] * bs[i]).sum();
        if s_bs <= 0.0 {
            continue;
        }
        let sy = pair.s_dot_y;
        let theta = powell_damping_theta(sy, s_bs);
        for i in 0..n {
            r[i] = theta * y[i] + (1.0 - theta) * bs[i];
        }
        let sr: Number = theta * sy + (1.0 - theta) * s_bs;
        if sr <= 0.0 {
            continue;
        }
        for i in 0..n {
            let r_i = r[i];
            let bs_i = bs[i];
            let row = &mut b[i * n..(i + 1) * n];
            for j in 0..n {
                row[j] += r_i * r[j] / sr - bs_i * bs[j] / s_bs;
            }
        }
    }
}

/// In-place SR1 history walk: applies the rank-1 update
/// `B ← B + ((y − Bs)(y − Bs)^T) / ((y − Bs)^T s)` for each curvature
/// pair, skipping when `|denom|` falls below `1e-8 ||s|| ||y − Bs||`
/// per [`sr1_denominator_ok`]. Mirrors
/// `IpLimMemQuasiNewtonUpdater.cpp::Update_SR1`.
fn apply_sr1_history(b: &mut [Number], n: usize, history: &[CurvaturePair]) {
    if n == 0 {
        return;
    }
    let mut bs = vec![0.0_f64; n];
    let mut yms = vec![0.0_f64; n];
    for pair in history {
        let s = dense_from_vec(pair.s.as_ref(), n);
        let y = dense_from_vec(pair.y.as_ref(), n);
        for i in 0..n {
            let row = &b[i * n..(i + 1) * n];
            let mut acc = 0.0;
            for j in 0..n {
                acc += row[j] * s[j];
            }
            bs[i] = acc;
        }
        for i in 0..n {
            yms[i] = y[i] - bs[i];
        }
        let denom: Number = (0..n).map(|i| yms[i] * s[i]).sum();
        let yms_norm: Number = (0..n).map(|i| yms[i] * yms[i]).sum::<Number>().sqrt();
        if !sr1_denominator_ok(denom, pair.s_norm, yms_norm) {
            continue;
        }
        for i in 0..n {
            let yms_i = yms[i];
            let row = &mut b[i * n..(i + 1) * n];
            for j in 0..n {
                row[j] += yms_i * yms[j] / denom;
            }
        }
    }
}

/// Initial Hessian scalar used as the diagonal of `B_0` before the
/// rank-2 updates are applied. Mirrors upstream's three options
/// (`limited_memory_initialization` in
/// `IpLimMemQuasiNewtonUpdater.cpp`):
///
/// * `Identity` → `1.0`
/// * `Scalar1` → `(s^T y) / (s^T s)`
/// * `Scalar2` → `(y^T y) / (s^T y)`
///
/// Result is clamped to `[min_val, max_val]` per upstream's
/// `limited_memory_init_val_{min,max}` defaults.
pub fn initial_hessian_scalar(
    init: InitialApprox,
    s_dot_s: Number,
    s_dot_y: Number,
    y_dot_y: Number,
    min_val: Number,
    max_val: Number,
) -> Number {
    let raw = match init {
        InitialApprox::Identity => 1.0,
        InitialApprox::Scalar1 => {
            if s_dot_s > 0.0 {
                s_dot_y / s_dot_s
            } else {
                1.0
            }
        }
        InitialApprox::Scalar2 => {
            if s_dot_y > 0.0 {
                y_dot_y / s_dot_y
            } else {
                1.0
            }
        }
    };
    raw.clamp(min_val, max_val)
}

/// Powell damping coefficient `theta` for the modified-y BFGS update.
/// When the curvature pair `(s, y)` violates `s^T y >= 0.2 * s^T B s`,
/// we replace `y` by `y_bar = theta * y + (1 - theta) * B s` so that
/// the resulting update is positive-definite.
///
/// ```text
///   if s^T y >= 0.2 * s^T B s:  theta = 1
///   else:                        theta = (0.8 * s^T B s) / (s^T B s - s^T y)
/// ```
///
/// Mirrors upstream's `IpLimMemQuasiNewtonUpdater.cpp:PowellDamping`.
pub fn powell_damping_theta(s_dot_y: Number, s_dot_b_s: Number) -> Number {
    if s_dot_y >= 0.2 * s_dot_b_s {
        1.0
    } else {
        let denom = s_dot_b_s - s_dot_y;
        if denom > 0.0 {
            0.8 * s_dot_b_s / denom
        } else {
            1.0
        }
    }
}

/// L-BFGS curvature-pair acceptance: include `(s, y)` in history iff
/// `s^T y > eps * ||s|| ||y||`. Mirrors upstream's skip-criterion
/// (`IpLimMemQuasiNewtonUpdater.cpp` ~line 750: `eps = 1e-8`).
pub fn bfgs_curvature_pair_ok(s_dot_y: Number, s_norm: Number, y_norm: Number) -> bool {
    let eps = 1e-8_f64;
    s_dot_y > eps * s_norm * y_norm
}

/// SR1 acceptance: the SR1 update divides by `(y - Bs)^T s`, so we
/// need `|(y - Bs)^T s| > eps * ||s|| ||y - Bs||`. Mirrors upstream's
/// `IpLimMemQuasiNewtonUpdater.cpp` SR1 skip-criterion.
pub fn sr1_denominator_ok(yms_dot_s: Number, s_norm: Number, yms_norm: Number) -> bool {
    let eps = 1e-8_f64;
    yms_dot_s.abs() > eps * s_norm * yms_norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_init_returns_one() {
        assert_eq!(
            initial_hessian_scalar(InitialApprox::Identity, 1.0, 1.0, 1.0, 1e-8, 1e8),
            1.0
        );
    }

    #[test]
    fn scalar1_init_is_sy_over_ss() {
        // s_dot_s=4, s_dot_y=2 → 2/4 = 0.5.
        let v = initial_hessian_scalar(InitialApprox::Scalar1, 4.0, 2.0, 0.0, 1e-8, 1e8);
        assert!((v - 0.5).abs() < 1e-15);
    }

    #[test]
    fn scalar2_init_is_yy_over_sy() {
        // y_dot_y=8, s_dot_y=2 → 4.
        let v = initial_hessian_scalar(InitialApprox::Scalar2, 0.0, 2.0, 8.0, 1e-8, 1e8);
        assert!((v - 4.0).abs() < 1e-15);
    }

    #[test]
    fn init_clamped_to_max() {
        let v = initial_hessian_scalar(InitialApprox::Scalar2, 0.0, 1e-20, 1.0, 1e-8, 1e8);
        assert_eq!(v, 1e8);
    }

    #[test]
    fn init_clamped_to_min() {
        let v = initial_hessian_scalar(InitialApprox::Scalar2, 0.0, 1e20, 1.0, 1e-8, 1e8);
        assert_eq!(v, 1e-8);
    }

    #[test]
    fn powell_no_damping_when_curvature_ok() {
        // s^T y = 1, s^T B s = 1; 1 >= 0.2 * 1 → theta = 1.
        assert_eq!(powell_damping_theta(1.0, 1.0), 1.0);
    }

    #[test]
    fn powell_damps_when_curvature_violated() {
        // s^T y = 0.1, s^T B s = 1; 0.1 < 0.2 → theta = 0.8/(1-0.1) = 8/9.
        let theta = powell_damping_theta(0.1, 1.0);
        assert!((theta - 8.0 / 9.0).abs() < 1e-15);
    }

    #[test]
    fn bfgs_skip_criterion() {
        // s_dot_y = 1, ||s|| = 1, ||y|| = 1 → 1 > 1e-8: ok.
        assert!(bfgs_curvature_pair_ok(1.0, 1.0, 1.0));
        // s_dot_y = 1e-10, ||s|| = 1, ||y|| = 1 → 1e-10 < 1e-8: skip.
        assert!(!bfgs_curvature_pair_ok(1e-10, 1.0, 1.0));
    }

    #[test]
    fn sr1_skip_criterion_uses_absolute_value() {
        // Negative numerator is fine for SR1 (rank-1 update can have either sign).
        assert!(sr1_denominator_ok(-1.0, 1.0, 1.0));
        assert!(!sr1_denominator_ok(1e-10, 1.0, 1.0));
    }

    fn rcv(values: &[Number]) -> Rc<dyn Vector> {
        let mut v = pounce_linalg::dense_vector::DenseVectorSpace::new(values.len() as i32)
            .make_new_dense();
        v.set(0.0);
        v.values_mut().copy_from_slice(values);
        Rc::new(v)
    }

    #[test]
    fn ingest_pair_accepts_well_curved_pair() {
        let mut updater = LimMemQuasiNewtonUpdater::new();
        // s = (1, 0), y = (1, 0); s·y = 1 > 1e-8.
        let accepted = updater.ingest_pair(rcv(&[1.0, 0.0]), rcv(&[1.0, 0.0]));
        assert!(accepted);
        assert_eq!(updater.history.len(), 1);
        let pair = &updater.history[0];
        assert!((pair.s_dot_y - 1.0).abs() < 1e-15);
        assert!((pair.s_norm - 1.0).abs() < 1e-15);
        assert!((pair.y_norm - 1.0).abs() < 1e-15);
    }

    #[test]
    fn ingest_pair_skips_zero_curvature() {
        let mut updater = LimMemQuasiNewtonUpdater::new();
        // s · y = 0 ⇒ skip per BFGS criterion (eps · ||s|| · ||y||).
        let accepted = updater.ingest_pair(rcv(&[1.0]), rcv(&[0.0]));
        assert!(!accepted);
        assert!(updater.history.is_empty());
    }

    #[test]
    fn history_caps_at_max_history() {
        let mut updater = LimMemQuasiNewtonUpdater {
            max_history: 2,
            ..LimMemQuasiNewtonUpdater::default()
        };
        for _ in 0..5 {
            updater.ingest_pair(rcv(&[1.0]), rcv(&[1.0]));
        }
        assert_eq!(updater.history.len(), 2);
    }

    #[test]
    fn sr1_path_routes_through_sr1_skip() {
        let mut updater = LimMemQuasiNewtonUpdater {
            update_type: UpdateType::Sr1,
            ..LimMemQuasiNewtonUpdater::default()
        };
        // SR1's heuristic accepts negative s·y (rank-1 sign-indefinite).
        assert!(updater.ingest_pair(rcv(&[1.0]), rcv(&[-1.0])));
    }

    fn pair(s: &[Number], y: &[Number]) -> CurvaturePair {
        let s_rc = rcv(s);
        let y_rc = rcv(y);
        let s_dot_y = s_rc.dot(&*y_rc);
        let s_norm = s_rc.nrm2();
        let y_norm = y_rc.nrm2();
        CurvaturePair {
            s: s_rc,
            y: y_rc,
            s_dot_y,
            s_norm,
            y_norm,
        }
    }

    #[test]
    fn bfgs_quadratic_recovers_hessian() {
        // For a strictly-convex quadratic f(x) = ½ xᵀ A x with A SPD,
        // a single BFGS update from B₀ = I along a curvature pair
        // (s, y = A s) reproduces A on the s-direction:
        //   B₁ s = y = A s.
        // Use A = diag(2, 5), s = (1, 1), so y = (2, 5).
        let mut b = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let p = pair(&[1.0, 1.0], &[2.0, 5.0]);
        apply_bfgs_history(&mut b, 2, std::slice::from_ref(&p));
        // (B s)[0] should equal y[0]=2, (B s)[1] should equal y[1]=5.
        let bs0 = b[0] * 1.0 + b[1] * 1.0;
        let bs1 = b[2] * 1.0 + b[3] * 1.0;
        assert!((bs0 - 2.0).abs() < 1e-12, "Bs[0]={}", bs0);
        assert!((bs1 - 5.0).abs() < 1e-12, "Bs[1]={}", bs1);
    }

    #[test]
    fn bfgs_history_keeps_symmetry() {
        let mut b = vec![3.0, 0.0, 0.0, 3.0];
        let pairs = vec![
            pair(&[1.0, 0.5], &[2.0, 1.0]),
            pair(&[0.7, 1.2], &[1.0, 2.5]),
        ];
        apply_bfgs_history(&mut b, 2, &pairs);
        assert!((b[1] - b[2]).abs() < 1e-12);
    }

    #[test]
    fn sr1_quadratic_one_pair_recovers_hessian_action() {
        // SR1 update with B₀ = I, s = (1, 1), y = (2, 5):
        // y - B s = (1, 4); denom = (1, 4)·(1, 1) = 5;
        // ΔB = (1, 4)(1, 4)ᵀ / 5 = [[0.2, 0.8], [0.8, 3.2]]
        // B₁ = [[1.2, 0.8], [0.8, 4.2]]; B₁ s = (2.0, 5.0) = y. ✓
        let mut b = vec![1.0, 0.0, 0.0, 1.0];
        let p = pair(&[1.0, 1.0], &[2.0, 5.0]);
        apply_sr1_history(&mut b, 2, std::slice::from_ref(&p));
        let bs0 = b[0] + b[1];
        let bs1 = b[2] + b[3];
        assert!((bs0 - 2.0).abs() < 1e-12);
        assert!((bs1 - 5.0).abs() < 1e-12);
    }

    #[test]
    fn full_lower_triangle_space_has_n_n_plus_1_over_2() {
        let s = make_full_lower_triangle_space(4);
        assert_eq!(s.dim(), 4);
        assert_eq!(s.nonzeros(), 10);
        // First few entries: (1,1), (2,1), (2,2), (3,1), ...
        assert_eq!(s.irows()[0], 1);
        assert_eq!(s.jcols()[0], 1);
        assert_eq!(s.irows()[1], 2);
        assert_eq!(s.jcols()[1], 1);
        assert_eq!(s.irows()[2], 2);
        assert_eq!(s.jcols()[2], 2);
    }
}

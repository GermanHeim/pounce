//! `PdSensBacksolver` — `SensBacksolver` adapter over the converged
//! `PdFullSpaceSolver` from `pounce-algorithm`.
//!
//! This is the Phase B.2 piece tracked in
//! [pounce#16](https://github.com/jkitchin/pounce/issues/16): it lets
//! `pounce-sensitivity` drive backsolves against the real converged
//! KKT factor, replacing the synthetic [`crate::DenseLuBacksolver`]
//! used by Phase B.1 unit tests.
//!
//! # Use
//!
//! 1. Register an `on_converged` callback on `IpoptApplication` via
//!    [`pounce_algorithm::application::IpoptApplication::set_on_converged`].
//! 2. Inside the callback, build a `PdSensBacksolver` from the four
//!    handles passed in (`data`, `cq`, `nlp`, `&mut pd_solver`).
//! 3. Hand it to [`crate::SensApplication`] / a `SensStepCalc` /
//!    [`crate::compute_reduced_hessian`] like any other
//!    [`SensBacksolver`].
//!
//! Upstream `SensSimpleBacksolver`
//! ([`ref/Ipopt/contrib/sIPOPT/src/SensSimpleBacksolver.cpp`](../../../ref/Ipopt/contrib/sIPOPT/src/SensSimpleBacksolver.cpp))
//! is the analogous wrapper around `IpoptCalculatedQuantities` +
//! `PDSystemSolver` upstream.
//!
//! # Flat-slice ↔ `IteratesVector` mapping
//!
//! The full primal-dual state of pounce's IPM is the eight-block
//! compound `(x, s, λ_c, λ_d, z_l, z_u, v_l, v_u)` (see
//! [`pounce_algorithm::iterates_vector::IteratesVector`]). This
//! adapter packs / unpacks the flat slices that
//! [`crate::SensBacksolver`] takes as the concatenation
//! `x || s || λ_c || λ_d || z_l || z_u || v_l || v_u`, mirroring
//! upstream's `CompoundVector` layout (`IpCompoundVector.hpp`).
//!
//! # Reference
//!
//! Pirnay, H.; López-Negrete, R.; Biegler, L. T. (2012). *Optimal
//! sensitivity based on IPOPT*. Mathematical Programming Computation,
//! **4**(4), 307–331. DOI:
//! [10.1007/s12532-012-0043-2](https://doi.org/10.1007/s12532-012-0043-2).
//! Verified via Crossref on 2026-05-13.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::ipopt_cq::IpoptCqHandle;
use pounce_algorithm::ipopt_data::IpoptDataHandle;
use pounce_algorithm::iterates_vector::{IteratesVector, IteratesVectorMut};
use pounce_algorithm::kkt::pd_full_space_solver::{PdFullSpaceSolver, SigmaOverride};
use pounce_common::types::{Index, Number};
use pounce_linalg::dense_vector::DenseVector;
use pounce_nlp::ipopt_nlp::IpoptNlp;

use crate::backsolver::SensBacksolver;

/// Adapter from `PdFullSpaceSolver` to [`SensBacksolver`]. Holds
/// owning clones of the four pieces of the algorithm's converged
/// state, plus the 8-block iterate template used to allocate fresh
/// RHS / LHS vectors.
///
/// The PD solver lives behind an `Rc<RefCell<…>>` because
/// [`SensBacksolver::solve`] is `&self` but the upstream signature
/// for `PdFullSpaceSolver::solve` is `&mut self` (it caches the
/// last-solve dependency tags and the augsys-improved flag). The
/// `RefCell` is single-thread-only, single-borrow, exactly matching
/// the call pattern from `pounce-sensitivity`'s pipeline.
///
/// Owning (rather than borrowing) the four handles is what lets a
/// `PdSensBacksolver` outlive the `on_converged` callback frame —
/// required by the public `Solver` session API in `pounce-algorithm`,
/// which retains the backsolver for repeated `parametric_step` /
/// `kkt_solve` / `compute_reduced_hessian` calls after the IPM has
/// returned. The data, cq, and nlp handles are already
/// `Rc<RefCell<…>>` cheap-clone handles upstream, so this carries no
/// allocation overhead.
#[derive(Clone)]
pub struct PdSensBacksolver {
    /// Shared, interior-mutable handle to the converged PD solver.
    /// Cloned from `PdSearchDirCalc::pd_solver_rc()` at construction.
    pd: Rc<RefCell<PdFullSpaceSolver>>,
    data: IpoptDataHandle,
    cq: IpoptCqHandle,
    nlp: Rc<RefCell<dyn IpoptNlp>>,
    /// Block dimensions in `(x, s, y_c, y_d, z_l, z_u, v_l, v_u)` order.
    dims: [usize; 8],
    /// 8-block prototype used to mint fresh vectors with the same
    /// `VectorSpace`s as the converged iterate; cloned from
    /// `data.borrow().curr`.
    template: IteratesVector,
    /// Natural-units row/column scaling pair (pounce#128). The IPM's
    /// KKT factor is held in the NLP's internally **scaled** space
    /// (objective factor `df`, per-row constraint factors `dc` / `dd`
    /// from `nlp_scaling_method`; scaled multipliers `ỹ = (df/dc)·y`,
    /// `z̃ = df·z`, `ṽ = (df/dd)·v`, scaled slack `s̃ = dd·s`). The
    /// scaled 8-block primal-dual system is the two-sided diagonal
    /// scaling `K̃ = E K F` of the natural-units system, with
    /// per-block entries
    ///
    /// ```text
    ///        x      s        y_c      y_d     z_l/z_u   v_l/v_u
    /// E  =   df     df/dd_i  dc_i     dd_i    df        df
    /// F  =   1      1/dd_i   dc_i/df  dd_i/df 1/df      dd_r(j)/df
    /// ```
    ///
    /// (`dd_r(j)` = the d-row scaling of the j-th finite d-bound,
    /// through the `pd_l` / `pd_u` expansion). Hence
    /// `K⁻¹ = F K̃⁻¹ E`: scale the RHS by `E`, back-solve against the
    /// held factor, scale the result by `F`. Unlike a symmetric
    /// congruence this needs no square root, so it covers a negative
    /// `obj_scaling_factor` (maximization) and covers the z/v
    /// bound-multiplier rows exactly (those rows admit no symmetric
    /// diagonal: `K̃_{z,x} = df·Z·Pᵀ` but `K̃_{z,z} = X − x_L` is
    /// unscaled). `None` ⇔ scaling inactive, identity.
    ///
    /// **Variable scaling** (gh#486 stage 3) multiplies into the same
    /// pair. A change of variables `x̃ = d ⊙ x` contributes
    ///
    /// ```text
    ///        x      z_l/z_u          everything else
    /// E  =   1/d    1                1
    /// F  =   1/d    d_{px(j)}        1
    /// ```
    ///
    /// (`d_{px(j)}` = the factor of the variable carrying the j-th
    /// finite bound, through the `px_l` / `px_u` expansion). The `s`,
    /// `y_c`, `y_d`, `v_l` and `v_u` blocks are untouched because the
    /// substitution leaves `c`, `d` and their multipliers alone. The
    /// two contributions compose by elementwise product, in either
    /// order, because both are diagonal.
    conj: Option<Rc<ConjPair>>,
    /// Per-variable factors `d` the solve ran under (gh#486), in the
    /// algorithm's **var-x** space — i.e. already projected through
    /// the fixed-variable map, so entry `i` matches KKT row `i`.
    /// `None` ⇔ no variable scaling. Folded into [`Self::conj`] for
    /// the back-solves; kept here for the consumers that read the
    /// converged iterate and the model's matrices directly rather than
    /// through the factor (see [`crate::activity`]).
    d_var: Option<Rc<Vec<Number>>>,
    /// The same factors in the user TNLP's **full-x** space, the shape
    /// `finalize_solution_z_l` / `n_full_x`-length reports come in.
    /// `None` alongside [`Self::d_var`].
    d_full: Option<Rc<Vec<Number>>>,
    /// Var-x row of the variable each bound multiplier constrains,
    /// `z_l` entries then `z_u` entries, read off the `px_l` / `px_u`
    /// expansions. `None` when either expansion is not an
    /// `ExpansionMatrix` and the map cannot be recovered.
    bound_vars: Option<Rc<Vec<crate::backsolver::BoundRow>>>,
    /// The barrier geometry re-measured against the bounds the model
    /// declares, for a held iterate that came from crossover. `None` —
    /// meaning "read the calculated quantities as they stand" — on
    /// every solve that ended on an interior point. See
    /// [`DeclaredFrameBarrier`].
    declared: Option<Rc<DeclaredFrameBarrier>>,
}

/// `Σ` and the active-bound slacks of a **crossed-over** iterate,
/// measured against the bounds the user declared rather than the ones
/// the barrier ran against (gh#654).
///
/// `bound_relax_factor` (default `1e-8`) widens every bound by `δ`
/// before the solve, and crossover (gh#612) then parks the iterate
/// exactly on the *declared* bound — a full `δ` inside the live relaxed
/// one. So the calculated quantities report a slack of exactly `δ` at
/// every active bound, where an interior iterate would have carried
/// `μ/z`, and the barrier diagonal `Σ = z/s` comes out as `z/δ` instead
/// of `z²/μ`. Since `δ` is capped at `constr_viol_tol` and `μ` ends near
/// `tol/(barrier_tol_factor+1)`, that is *looser* whenever `z·δ/μ > 1`,
/// which is the ordinary case.
///
/// `Σ` is the stiffness with which the barrier holds a bounded variable,
/// and a reduced Hessian read off the held factor carries a residual
/// error of exactly `O(1/Σ)` — the leftover of that pin being finite. So
/// the looser reading is a measurably less accurate covariance: on the
/// gh#654 fixture, `18x` at a bound multiplier of `4.5` and `396x` at
/// `994.5`, tracking `z·δ/μ` exactly.
///
/// The correction is the same one gh#646 applied to the reported
/// residuals: measure the crossed-over point in the frame it was solved
/// in. Nothing on the live iterate is touched — the relaxed bounds are
/// still what the algorithm ran against, and un-relaxing them after the
/// fact would mean replacing the NLP's bound `Rc`s and invalidating
/// every tag-keyed cache built on them. This is the consumer boundary
/// instead, where the question "how stiffly is this point held" is
/// actually being asked.
struct DeclaredFrameBarrier {
    /// Variable-bound contribution to the `x` diagonal.
    sigma_x: Rc<dyn pounce_linalg::Vector>,
    /// Inequality-row-bound contribution to the `s` diagonal. Relaxation
    /// widens `d_L` / `d_U` too, and crossover puts `s = d(x)` on the
    /// declared row bounds, so the row half of the defect is identical
    /// to the variable half.
    sigma_s: Rc<dyn pounce_linalg::Vector>,
    /// The declared-frame `x` slacks behind `sigma_x`, compressed in the
    /// `px_l` / `px_u` spaces. Kept because a *released* solve rebuilds
    /// one variable's `Σ` entry from the sides that stay active and has
    /// to rebuild it in the same frame.
    slack_x_l: Vec<Number>,
    slack_x_u: Vec<Number>,
}

/// Left/right diagonal pair for the natural-units back-solve; see the
/// `conj` field doc on [`PdSensBacksolver`]. Both vectors are
/// flat-KKT-length, in the `x‖s‖y_c‖y_d‖z_l‖z_u‖v_l‖v_u` packing.
struct ConjPair {
    /// `E`: multiplied into the RHS before the scaled-space solve.
    e: Vec<Number>,
    /// `F`: multiplied into the solution after the scaled-space solve.
    f: Vec<Number>,
}

impl PdSensBacksolver {
    /// The retained handles the activity classifier reads
    /// (crate-internal; see [`crate::activity`]).
    pub(crate) fn activity_handles(
        &self,
    ) -> (&IpoptDataHandle, &IpoptCqHandle, &Rc<RefCell<dyn IpoptNlp>>) {
        (&self.data, &self.cq, &self.nlp)
    }

    /// Construct from the four handles handed in by the `on_converged`
    /// callback. Errors if `data` has no `curr` (i.e. the algorithm
    /// never reached an iterate — should not happen on
    /// `SolveSucceeded`) or the NLP reports scaling data inconsistent
    /// with the converged iterate (see [`Self::natural_units_conj`]).
    pub fn new(
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        pd: Rc<RefCell<PdFullSpaceSolver>>,
    ) -> Result<Self, String> {
        let curr = data
            .borrow()
            .curr
            .clone()
            .ok_or_else(|| "no current iterate at convergence".to_string())?;
        let dims = [
            curr.x.dim() as usize,
            curr.s.dim() as usize,
            curr.y_c.dim() as usize,
            curr.y_d.dim() as usize,
            curr.z_l.dim() as usize,
            curr.z_u.dim() as usize,
            curr.v_l.dim() as usize,
            curr.v_u.dim() as usize,
        ];
        let (d_var, d_full) = Self::variable_factors(nlp, &dims)?;
        let conj = Self::natural_units_conj(nlp, &dims, d_var.as_ref().map(|v| v.as_slice()))?;
        let bound_vars = Self::bound_variable_rows(nlp, &dims);
        let declared = Self::declared_frame_barrier(data, nlp, &dims);
        Ok(Self {
            pd,
            data: Rc::clone(data),
            cq: Rc::clone(cq),
            nlp: Rc::clone(nlp),
            dims,
            template: curr,
            conj,
            d_var,
            d_full,
            bound_vars,
            declared,
        })
    }

    /// Re-measure `Σ` against the declared bounds when the held iterate
    /// came from crossover; `None` otherwise, and `None` whenever the
    /// NLP does not report its declared box (the fallback is then the
    /// calculated quantities, i.e. the pre-gh#654 behaviour).
    ///
    /// Deliberately gated on crossover rather than applied everywhere:
    /// an *interior* iterate is not near the declared bounds in any
    /// useful sense — it can sit up to `δ` **outside** one — and its
    /// `μ/z` standoff is the barrier's own geometry, which is the right
    /// thing to read. Only a purified point has the declared frame as
    /// its own.
    fn declared_frame_barrier(
        data: &IpoptDataHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        dims: &[usize; 8],
    ) -> Option<Rc<DeclaredFrameBarrier>> {
        let (curr, from_crossover) = {
            let d = data.borrow();
            (d.curr.clone()?, d.curr_from_crossover)
        };
        if !from_crossover {
            return None;
        }
        let nlp_ref = nlp.borrow();
        let (x_l, x_u) = nlp_ref.declared_x_bounds()?;
        let (d_l, d_u) = nlp_ref.declared_d_bounds()?;
        if x_l.len() != dims[4]
            || x_u.len() != dims[5]
            || d_l.len() != dims[6]
            || d_u.len() != dims[7]
        {
            return None;
        }
        let (sigma_x, slack_x_l, slack_x_u) = declared_frame_sigma(
            &*nlp_ref.px_l(),
            &*nlp_ref.px_u(),
            &*curr.x,
            &x_l,
            &x_u,
            &*curr.z_l,
            &*curr.z_u,
            dims[0],
        );
        let (sigma_s, _, _) = declared_frame_sigma(
            &*nlp_ref.pd_l(),
            &*nlp_ref.pd_u(),
            &*curr.s,
            &d_l,
            &d_u,
            &*curr.v_l,
            &*curr.v_u,
            dims[1],
        );
        Some(Rc::new(DeclaredFrameBarrier {
            sigma_x,
            sigma_s,
            slack_x_l,
            slack_x_u,
        }))
    }

    /// The barrier diagonals to factor with: the declared-frame pair
    /// when the held iterate came from crossover, otherwise the
    /// calculated quantities (which is what [`SigmaOverride::default`]
    /// selects).
    fn sigma_override(&self) -> SigmaOverride {
        match self.declared.as_ref() {
            None => SigmaOverride::default(),
            Some(d) => SigmaOverride {
                x: Some(Rc::clone(&d.sigma_x)),
                s: Some(Rc::clone(&d.sigma_s)),
            },
        }
    }

    /// The `x`-block barrier diagonal in the frame the held iterate
    /// belongs to. Crate-internal because [`crate::activity`] reads `Σ`
    /// straight off the iterate rather than through the factor, and the
    /// two must not disagree about which bounds the point is measured
    /// against.
    pub(crate) fn barrier_sigma_x(&self) -> Rc<dyn pounce_linalg::Vector> {
        match self.declared.as_ref() {
            Some(d) => Rc::clone(&d.sigma_x),
            None => self.cq.borrow().curr_sigma_x(),
        }
    }

    /// [`Self::barrier_sigma_x`] for the `s` block.
    pub(crate) fn barrier_sigma_s(&self) -> Rc<dyn pounce_linalg::Vector> {
        match self.declared.as_ref() {
            Some(d) => Rc::clone(&d.sigma_s),
            None => self.cq.borrow().curr_sigma_s(),
        }
    }

    /// Shared body of the two released solves. `shift` moves the
    /// released multipliers onto their x rows, which the step needs and
    /// a Schur complement's unit vectors must not get.
    fn solve_released_inner(
        &self,
        released: &[usize],
        rhs: &[Number],
        lhs: &mut [Number],
        shift: bool,
    ) -> bool {
        if rhs.len() != self.dim() || lhs.len() != self.dim() {
            return false;
        }
        // Nothing released is an ordinary solve. Taking this early lets
        // callers route every solve through here without paying a
        // re-factorization for a step that releases nothing.
        if released.is_empty() {
            return self.solve(rhs, lhs);
        }
        let Some(sigma) = self.released_sigma_x(released) else {
            return false;
        };
        self.solve_released_prebuilt(released, sigma, rhs, lhs, shift)
    }

    /// [`Self::solve_released_inner`] with the released `Σ` supplied by
    /// the caller. Repeated solves against ONE released operator must
    /// pass the same `Rc` every time: the factorization cache keys on
    /// the sigma object's tag, so a sigma rebuilt per call forces a
    /// re-factorization per call, while a held one factorizes once and
    /// back-solves thereafter.
    pub(crate) fn solve_released_prebuilt(
        &self,
        released: &[usize],
        sigma: Rc<dyn pounce_linalg::Vector>,
        rhs: &[Number],
        lhs: &mut [Number],
        shift: bool,
    ) -> bool {
        if rhs.len() != self.dim() || lhs.len() != self.dim() {
            return false;
        }
        let mut scaled: Vec<Number> = match self.conj.as_ref() {
            Some(c) => rhs.iter().zip(c.e.iter()).map(|(&r, &e)| r * e).collect(),
            None => rhs.to_vec(),
        };
        // A released bound's multiplier is fixed at zero, so its row
        // has no equation and the right-hand side there is meaningless.
        // It is also dangerous: the elimination folds a multiplier
        // row's entry in as r_z / s, and at a tightly active bound the
        // barrier-correction term every parametric right-hand side
        // carries, -mu on each bound row, folds to -mu / s = -z by
        // complementarity, an order-one injection into the released
        // variable's equation. Measured on a two-variable QP that bent
        // the released direction from the analytic [1.227, 0.454] to
        // [1.154, 0.194].
        for &r in released {
            if r < scaled.len() {
                scaled[r] = 0.0;
            }
        }
        if shift && !self.shift_released_rhs(released, &mut scaled) {
            return false;
        }
        if !self.solve_released_scaled(sigma, &scaled, lhs) {
            return false;
        }
        if let Some(c) = self.conj.as_ref() {
            for (l, &f) in lhs.iter_mut().zip(c.f.iter()) {
                *l *= f;
            }
        }
        true
    }

    /// The barrier's `x` diagonal with each released bound's own `z / s`
    /// taken off the variable it constrains -- subtracted rather than
    /// zeroed, since a variable bounded on both sides contributes twice
    /// and only one side is being released.
    ///
    /// Both the starting diagonal and the slacks the surviving sides are
    /// rebuilt from come from the frame the held iterate belongs to
    /// (gh#654): mixing a declared-frame `Σ` with relaxed-frame slacks
    /// would leave the released variable pinned in one frame and its
    /// neighbours in the other.
    pub(crate) fn released_sigma_x(
        &self,
        released: &[usize],
    ) -> Option<Rc<dyn pounce_linalg::Vector>> {
        self.active_set_sigma_x(released, &[])
    }

    /// [`Self::released_sigma_x`], also raising the diagonal on
    /// variables a step brings onto a bound.
    ///
    /// The two directions an active set can move are the same
    /// modification to one diagonal. A bound that leaves has its
    /// `z / s` taken off the variable it constrains, and a bound that
    /// becomes active has one put on, so a single vector describes
    /// both and a single factorization serves the whole correction.
    /// The two arguments are in different index spaces and both are
    /// `usize`: `released` holds compound KKT rows of bound
    /// multipliers, `pinned` holds var-x rows. Passing one where the
    /// other belongs is not a type error and will not be caught here.
    pub(crate) fn active_set_sigma_x(
        &self,
        released: &[usize],
        pinned: &[(usize, Number)],
    ) -> Option<Rc<dyn pounce_linalg::Vector>> {
        use pounce_linalg::dense_vector::DenseVectorSpace;
        let rows = self.bound_vars.as_deref()?;
        let base_row = rows.first()?.row;
        let dense = |v: Rc<dyn pounce_linalg::Vector>| -> Option<Vec<Number>> {
            v.as_any()
                .downcast_ref::<DenseVector>()
                .map(|d| d.expanded_values())
        };
        let mut sigma = dense(self.barrier_sigma_x())?;
        let (slack_l, slack_u) = match self.declared.as_ref() {
            Some(d) => (d.slack_x_l.clone(), d.slack_x_u.clone()),
            None => {
                let cq_ref = self.cq.borrow();
                let l = dense(cq_ref.curr_slack_x_l())?;
                let u = dense(cq_ref.curr_slack_x_u())?;
                (l, u)
            }
        };
        let (z_l, z_u) = {
            let d = self.data.borrow();
            let curr = d.curr.as_ref()?;
            (dense(Rc::clone(&curr.z_l))?, dense(Rc::clone(&curr.z_u))?)
        };
        // Rebuild each released variable's entry from its bounds that
        // stay active, rather than subtracting the released bound's
        // `z / s` from the cached total. The subtraction differences
        // two numbers of order `z / s`, 1e7 and up at a tightly
        // active bound, and its correctness rests on the cache having
        // built its total from bitwise-identical products. Rebuilding
        // makes the released side an exact zero by construction and
        // depends on nothing about the cache.
        for &r in released {
            let br = rows.iter().find(|b| b.row == r)?;
            let mut fresh = 0.0;
            for other in rows.iter().filter(|b| b.var_row == br.var_row) {
                if released.contains(&other.row) {
                    continue;
                }
                let k = other.row - base_row - if other.lower { 0 } else { self.dims[4] };
                let (z, s) = if other.lower {
                    (*z_l.get(k)?, *slack_l.get(k)?)
                } else {
                    (*z_u.get(k)?, *slack_u.get(k)?)
                };
                if s == 0.0 || !s.is_finite() {
                    return None;
                }
                fresh += z / s;
            }
            *sigma.get_mut(br.var_row)? = fresh;
        }
        for &(var_row, add) in pinned {
            *sigma.get_mut(var_row)? += add;
        }
        let space = DenseVectorSpace::new(sigma.len() as Index);
        let mut out = DenseVector::new(space);
        out.values_mut().copy_from_slice(&sigma);
        Some(Rc::new(out) as Rc<dyn pounce_linalg::Vector>)
    }

    /// Zero the released multipliers' own rows of a scaled-space
    /// right-hand side and move each multiplier onto its variable's x
    /// row.
    ///
    /// Dropping `sigma` gives the released *matrix*; the released
    /// right-hand side still wants the multiplier moved across. The
    /// elimination folds a multiplier row in as `r_z / s`, so zeroing
    /// that row and adding the multiplier to the x row directly reaches
    /// the same place without `s` appearing at all -- which is the
    /// whole point of re-factoring rather than downdating.
    fn shift_released_rhs(&self, released: &[usize], rhs: &mut [Number]) -> bool {
        let Some(rows) = self.bound_vars.as_deref() else {
            return false;
        };
        let Some(base_row) = rows.first().map(|b| b.row) else {
            return false;
        };
        let d = self.data.borrow();
        let Some(curr) = d.curr.as_ref() else {
            return false;
        };
        let dense = |v: &Rc<dyn pounce_linalg::Vector>| -> Option<Vec<Number>> {
            v.as_any()
                .downcast_ref::<DenseVector>()
                .map(|x| x.expanded_values())
        };
        let (Some(z_l), Some(z_u)) = (dense(&curr.z_l), dense(&curr.z_u)) else {
            return false;
        };
        for &r in released {
            let Some(br) = rows.iter().find(|b| b.row == r) else {
                return false;
            };
            let k = r - base_row - if br.lower { 0 } else { self.dims[4] };
            let Some(&z) = (if br.lower { z_l.get(k) } else { z_u.get(k) }) else {
                return false;
            };
            if r >= rhs.len() || br.var_row >= rhs.len() {
                return false;
            }
            rhs[r] = 0.0;
            // the sign the x row carries this side's multiplier with
            rhs[br.var_row] += if br.lower { -z } else { z };
        }
        true
    }

    /// One back-solve against the re-factored system, in the solver's
    /// internal scaled space.
    fn solve_released_scaled(
        &self,
        sigma: Rc<dyn pounce_linalg::Vector>,
        rhs: &[Number],
        lhs: &mut [Number],
    ) -> bool {
        let off = self.offsets();
        let rhs_mut0 = self.template.make_new_zeroed();
        let mut rhs_iv = rhs_mut0.freeze();
        let mut res_iv = self.template.make_new_zeroed();
        if !(write_rhs_block(&mut rhs_iv.x, &rhs[off[0]..off[1]])
            && write_rhs_block(&mut rhs_iv.s, &rhs[off[1]..off[2]])
            && write_rhs_block(&mut rhs_iv.y_c, &rhs[off[2]..off[3]])
            && write_rhs_block(&mut rhs_iv.y_d, &rhs[off[3]..off[4]])
            && write_rhs_block(&mut rhs_iv.z_l, &rhs[off[4]..off[5]])
            && write_rhs_block(&mut rhs_iv.z_u, &rhs[off[5]..off[6]])
            && write_rhs_block(&mut rhs_iv.v_l, &rhs[off[6]..off[7]])
            && write_rhs_block(&mut rhs_iv.v_u, &rhs[off[7]..off[8]]))
        {
            return false;
        }
        if !self.pd.borrow_mut().solve_with_sigma(
            &self.data,
            &self.cq,
            &self.nlp,
            1.0,
            0.0,
            &rhs_iv,
            &mut res_iv,
            /* allow_inexact = */ true,
            /* improve_solution = */ false,
            SigmaOverride {
                x: Some(sigma),
                // The release is an x-block operation; the row block
                // keeps whichever frame the held iterate belongs to.
                s: self.sigma_override().s,
            },
        ) {
            return false;
        }
        read_res_block(&*res_iv.x, &mut lhs[off[0]..off[1]])
            && read_res_block(&*res_iv.s, &mut lhs[off[1]..off[2]])
            && read_res_block(&*res_iv.y_c, &mut lhs[off[2]..off[3]])
            && read_res_block(&*res_iv.y_d, &mut lhs[off[3]..off[4]])
            && read_res_block(&*res_iv.z_l, &mut lhs[off[4]..off[5]])
            && read_res_block(&*res_iv.z_u, &mut lhs[off[5]..off[6]])
            && read_res_block(&*res_iv.v_l, &mut lhs[off[6]..off[7]])
            && read_res_block(&*res_iv.v_u, &mut lhs[off[7]..off[8]])
    }

    /// Var-x row behind each `z_l` then `z_u` entry, through the
    /// `px_l` / `px_u` expansions. `None` when either is not an
    /// `ExpansionMatrix` or reports the wrong length -- the release
    /// half then stays off rather than guessing a mapping.
    fn bound_variable_rows(
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        dims: &[usize; 8],
    ) -> Option<Rc<Vec<crate::backsolver::BoundRow>>> {
        let nlp_ref = nlp.borrow();
        let z_l_off = dims[0] + dims[1] + dims[2] + dims[3];
        let mut out = Vec::with_capacity(dims[4] + dims[5]);
        for (pm, n_v, off, lower) in [
            (nlp_ref.px_l(), dims[4], z_l_off, true),
            (nlp_ref.px_u(), dims[5], z_l_off + dims[4], false),
        ] {
            if n_v == 0 {
                continue;
            }
            let em = pm
                .as_any()
                .downcast_ref::<pounce_linalg::expansion_matrix::ExpansionMatrix>()?;
            let pos = em.expanded_pos_indices();
            if pos.len() != n_v {
                return None;
            }
            for (k, &p) in pos.iter().enumerate() {
                let p = p as usize;
                if p >= dims[0] {
                    return None;
                }
                out.push(crate::backsolver::BoundRow {
                    row: off + k,
                    var_row: p,
                    lower,
                });
            }
        }
        Some(Rc::new(out))
    }

    /// Read the variable factors the solve ran under off the NLP and
    /// project them into the algorithm's var-x space (gh#486 stage 3).
    ///
    /// Returns `(None, None)` when no variable scaling is active.
    /// Errors when the reported vector does not match the NLP's own
    /// full-x width, or when the projection does not fill the `x`
    /// block: either would silently mis-pair a factor with a variable,
    /// which is the whole failure mode this plumbing exists to avoid.
    #[allow(clippy::type_complexity)]
    fn variable_factors(
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        dims: &[usize; 8],
    ) -> Result<(Option<Rc<Vec<Number>>>, Option<Rc<Vec<Number>>>), String> {
        let nlp_ref = nlp.borrow();
        let Some(d_full) = nlp_ref.variable_scaling() else {
            return Ok((None, None));
        };
        let n_full = nlp_ref.n_full_x() as usize;
        if d_full.len() != n_full {
            return Err(format!(
                "variable scaling length {} != n_full_x {}",
                d_full.len(),
                n_full
            ));
        }
        // NaN is not a "no-op" factor and neither is zero; the wrapper
        // refuses both at setup, so seeing one here means the vector
        // did not come from the wrapper that ran.
        if let Some(bad) = d_full.iter().find(|v| !v.is_finite() || **v <= 0.0) {
            return Err(format!(
                "variable scaling factor {bad} is not finite and positive"
            ));
        }
        let mut d_var = vec![Number::NAN; dims[0]];
        for (full, &factor) in d_full.iter().enumerate() {
            if let Some(var) = nlp_ref.full_x_to_var_x(full as Index) {
                let slot = d_var.get_mut(var as usize).ok_or_else(|| {
                    format!("var-x index {var} outside x block of width {}", dims[0])
                })?;
                *slot = factor;
            }
        }
        if let Some(pos) = d_var.iter().position(|v| v.is_nan()) {
            return Err(format!(
                "variable scaling left var-x column {pos} of {} unmapped",
                dims[0]
            ));
        }
        Ok((Some(Rc::new(d_var)), Some(Rc::new(d_full))))
    }

    /// The per-variable factors the held solve ran under, in the
    /// algorithm's **var-x** space (one entry per `x`-block KKT row),
    /// or `None` when no variable scaling was active (gh#486).
    pub fn variable_scaling(&self) -> Option<&[Number]> {
        self.d_var.as_deref().map(|v| v.as_slice())
    }

    /// [`Self::variable_scaling`] in the user TNLP's **full-x** space:
    /// the shape of an `n_full_x`-length report, with the columns the
    /// solve dropped as fixed still present.
    pub fn variable_scaling_full(&self) -> Option<&[Number]> {
        self.d_full.as_deref().map(|v| v.as_slice())
    }

    /// Build the natural-units scaling pair `(E, F)` from the NLP's
    /// effective scaling and the variable factors `d_var` the solve
    /// ran under (see the field doc on [`Self::conj`]).
    /// Returns `Ok(None)` when no scaling is active. Errors when the
    /// NLP reports scaling data inconsistent with the converged
    /// iterate's block dimensions (would silently corrupt every
    /// back-solve) or a zero/non-finite `df`.
    fn natural_units_conj(
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        dims: &[usize; 8],
        d_var: Option<&[Number]>,
    ) -> Result<Option<Rc<ConjPair>>, String> {
        let nlp_ref = nlp.borrow();
        let df = nlp_ref.obj_scaling_factor();
        let dc = nlp_ref.c_scale_vec();
        let dd = nlp_ref.d_scale_vec();
        // `d_var` counts as active scaling on its own: a solve with
        // unit objective and row factors but a change of variables
        // still holds its factor in scaled coordinates.
        if df == 1.0 && dc.is_none() && dd.is_none() && d_var.is_none() {
            return Ok(None);
        }
        // df may be negative (obj_scaling_factor < 0 means maximize);
        // the two-sided scaling needs no square root, only df ≠ 0.
        if !df.is_finite() || df == 0.0 {
            return Err(format!("invalid obj_scaling_factor {df}"));
        }
        if let Some(v) = &dc {
            if v.len() != dims[2] {
                return Err(format!("c_scale length {} != y_c dim {}", v.len(), dims[2]));
            }
        }
        if let Some(v) = &dd {
            if v.len() != dims[3] || dims[1] != dims[3] {
                return Err(format!(
                    "d_scale length {} != y_d dim {} (s dim {})",
                    v.len(),
                    dims[3],
                    dims[1]
                ));
            }
        }
        if let Some(d) = d_var {
            if d.len() != dims[0] {
                return Err(format!(
                    "variable scaling length {} != x dim {}",
                    d.len(),
                    dims[0]
                ));
            }
        }
        // Per-entry source scale for a compressed bound-multiplier
        // block: entry j of z_l / v_l covers the row
        // `px_l.expanded_pos[j]` / `pd_l.expanded_pos[j]` of `src`.
        // Used for the v blocks (source `d_scale`, indexed by
        // inequality row) and the z blocks (source `d_var`, indexed by
        // var-x column).
        let bound_row_scale = |pm: Rc<dyn pounce_linalg::matrix::Matrix>,
                               src: Option<&[Number]>,
                               n_v: usize,
                               which: &str|
         -> Result<Vec<Number>, String> {
            let Some(vals) = src else {
                return Ok(vec![1.0; n_v]);
            };
            if n_v == 0 {
                return Ok(Vec::new());
            }
            let Some(em) = pm
                .as_any()
                .downcast_ref::<pounce_linalg::expansion_matrix::ExpansionMatrix>()
            else {
                return Err(format!("{which} is not an ExpansionMatrix"));
            };
            let pos = em.expanded_pos_indices();
            if pos.len() != n_v {
                return Err(format!(
                    "{which} expansion length {} != {} block dim {}",
                    pos.len(),
                    which,
                    n_v
                ));
            }
            pos.iter()
                .map(|&r| {
                    vals.get(r as usize).copied().ok_or_else(|| {
                        format!(
                            "{which} expansion row {r} out of scale-vector range {}",
                            vals.len()
                        )
                    })
                })
                .collect()
        };
        let vl_dd = bound_row_scale(nlp_ref.pd_l(), dd.as_deref(), dims[6], "pd_l")?;
        let vu_dd = bound_row_scale(nlp_ref.pd_u(), dd.as_deref(), dims[7], "pd_u")?;
        // The variable factor carried by each finite x-bound, through
        // the same expansion (gh#486). `d_var` indexes var-x columns,
        // and `px_l` / `px_u` say which column each z entry belongs to.
        let zl_dx = bound_row_scale(nlp_ref.px_l(), d_var, dims[4], "px_l")?;
        let zu_dx = bound_row_scale(nlp_ref.px_u(), d_var, dims[5], "px_u")?;
        drop(nlp_ref);

        let total: usize = dims.iter().sum();
        let mut e = Vec::with_capacity(total);
        let mut f = Vec::with_capacity(total);
        // x block: E = df/d_i, F = 1/d_i. `df` is the objective scale;
        // the `1/d_i` on both sides is the change of variables —
        // `∇f̃ = ∇f ⊘ d` puts the RHS in scaled units and `x = x̃ ⊘ d`
        // brings the solution back.
        match d_var {
            Some(d) => {
                e.extend(d.iter().map(|&di| df / di));
                f.extend(d.iter().map(|&di| 1.0 / di));
            }
            None => {
                e.extend(std::iter::repeat_n(df, dims[0]));
                f.extend(std::iter::repeat_n(1.0, dims[0]));
            }
        }
        // s block: E = df/dd_i, F = 1/dd_i (slacks live in scaled d-space).
        match &dd {
            Some(v) => {
                e.extend(v.iter().map(|&ddi| df / ddi));
                f.extend(v.iter().map(|&ddi| 1.0 / ddi));
            }
            None => {
                e.extend(std::iter::repeat_n(df, dims[1]));
                f.extend(std::iter::repeat_n(1.0, dims[1]));
            }
        }
        // y_c block: E = dc_i, F = dc_i/df.
        match &dc {
            Some(v) => {
                e.extend(v.iter().copied());
                f.extend(v.iter().map(|&dci| dci / df));
            }
            None => {
                e.extend(std::iter::repeat_n(1.0, dims[2]));
                f.extend(std::iter::repeat_n(1.0 / df, dims[2]));
            }
        }
        // y_d block: E = dd_i, F = dd_i/df.
        match &dd {
            Some(v) => {
                e.extend(v.iter().copied());
                f.extend(v.iter().map(|&ddi| ddi / df));
            }
            None => {
                e.extend(std::iter::repeat_n(1.0, dims[3]));
                f.extend(std::iter::repeat_n(1.0 / df, dims[3]));
            }
        }
        // z_l / z_u blocks: E = df, F = d_{px(j)}/df (z̃ = (df/d)·z,
        // and the slack diagonal x̃ − x̃_L = d·(x − x_L) carries the
        // variable factor, so the two cancel in the row and leave it
        // identical to the natural one — E takes no `d` at all).
        // Without variable scaling this is the pre-#486 `F = 1/df`:
        // bounds on x are unscaled and the slack diagonal is shared.
        e.extend(std::iter::repeat_n(df, dims[4] + dims[5]));
        f.extend(zl_dx.iter().map(|&dx| dx / df));
        f.extend(zu_dx.iter().map(|&dx| dx / df));
        // v_l / v_u blocks: E = df, F = dd_r/df (ṽ = (df/dd)·v and the
        // slack diagonal s̃ − d̃_l = dd·(s − d_l) carries the d-row
        // scale).
        e.extend(std::iter::repeat_n(df, dims[6] + dims[7]));
        f.extend(vl_dd.iter().map(|&ddr| ddr / df));
        f.extend(vu_dd.iter().map(|&ddr| ddr / df));
        Ok(Some(Rc::new(ConjPair { e, f })))
    }

    /// Effective objective scaling factor `df` of the converged NLP
    /// (1.0 when no scaling is active).
    pub fn obj_scaling_factor(&self) -> Number {
        self.nlp.borrow().obj_scaling_factor()
    }

    /// Effective NLP scaling at convergence:
    /// `(obj_scaling_factor, c_scale, d_scale)`. The vectors are
    /// `None` when the corresponding block carries no row scaling.
    pub fn nlp_scaling(&self) -> (Number, Option<Vec<Number>>, Option<Vec<Number>>) {
        let n = self.nlp.borrow();
        (n.obj_scaling_factor(), n.c_scale_vec(), n.d_scale_vec())
    }

    /// Inertia-correction perturbations `(δ_x, δ_s, δ_c, δ_d)` baked
    /// into the held KKT factor (the IPM's `current_perturbation`
    /// state at convergence). All zero ⇔ the final factorization was
    /// unregularized and the natural-units back-solves invert the
    /// exact KKT matrix. Nonzero ⇔ the factor carries a (scaled-space)
    /// regularization, so sensitivity outputs — covariance in
    /// particular — are perturbed and no longer exactly
    /// scaling-invariant; consumers should check this before trusting
    /// `-inv(reduced_hessian)` on ill-conditioned problems
    /// (pounce#128 follow-up).
    pub fn kkt_perturbations(&self) -> [Number; 4] {
        let p = self.data.borrow().perturbations;
        [p.delta_x, p.delta_s, p.delta_c, p.delta_d]
    }

    /// Map user-facing 0-based `g(x)` indices of parameter-pin
    /// equality constraints to flat KKT rows **and** the pin rows'
    /// `dc_i` scaling factors, in one pass. The KKT row of pin `i` is
    /// `n_x + n_s + c_block_idx`, i.e. the matching `y_c` slot, found
    /// through `IpoptNlp::full_g_to_c_block` so the c/d split's row
    /// permutation is honored (pounce#128 follow-up: the previous
    /// direct `n_x + n_s + g_idx` mapping silently picked wrong rows
    /// when inequalities preceded the pins). The scales are 1.0 when
    /// no constraint scaling is active; they relate the natural and
    /// solver-space reduced Hessians via
    /// `H̃_ij = (df / (dc_i·dc_j)) · H_ij`. Errors when a pin index
    /// is out of range or refers to an inequality row.
    pub fn pin_rows_and_c_scales(
        &self,
        pin_g_indices: &[Index],
    ) -> Result<(Vec<Index>, Vec<Number>), String> {
        let y_c_offset = (self.dims[0] + self.dims[1]) as Index;
        let nlp = self.nlp.borrow();
        let dc = nlp.c_scale_vec();
        let n_full_g = nlp.n_full_g();
        let mut rows = Vec::with_capacity(pin_g_indices.len());
        let mut scales = Vec::with_capacity(pin_g_indices.len());
        for &gi in pin_g_indices {
            // n_full_g() defaults to 0 for IpoptNlp impls that don't
            // report it; only range-check when it's meaningful.
            if gi < 0 || (n_full_g > 0 && gi >= n_full_g) {
                return Err(format!(
                    "pin constraint index {gi} out of range [0, m={n_full_g})"
                ));
            }
            let Some(ci) = nlp.full_g_to_c_block(gi) else {
                return Err(format!(
                    "pin constraint index {gi} is an inequality (not an equality row); \
                     parameter pins must be exact equalities"
                ));
            };
            rows.push(y_c_offset + ci);
            scales.push(dc.as_ref().map(|v| v[ci as usize]).unwrap_or(1.0));
        }
        Ok((rows, scales))
    }

    /// KKT-row half of [`Self::pin_rows_and_c_scales`].
    pub fn map_pin_g_to_kkt_rows(&self, pin_g_indices: &[Index]) -> Result<Vec<Index>, String> {
        Ok(self.pin_rows_and_c_scales(pin_g_indices)?.0)
    }

    /// Scaling half of [`Self::pin_rows_and_c_scales`].
    pub fn pin_c_scales(&self, pin_g_indices: &[Index]) -> Result<Vec<Number>, String> {
        Ok(self.pin_rows_and_c_scales(pin_g_indices)?.1)
    }

    /// Block dimensions of the compound KKT vector at convergence, in
    /// `(x, s, y_c, y_d, z_l, z_u, v_l, v_u)` order. Sum equals
    /// [`SensBacksolver::dim`]. Useful when a caller needs to compute
    /// the flat offset of a non-x block (e.g. `n_x + n_s` for the
    /// start of the equality-multiplier `y_c` block).
    pub fn block_dims(&self) -> [usize; 8] {
        self.dims
    }

    /// Map a 0-based **full-g** index (user-TNLP `g(x)` order) to its
    /// 0-based position in the equality-multiplier `y_c` block, or
    /// `None` when the constraint is an inequality (it lives in the `d`
    /// block, not `y_c`). Delegates to the held NLP's c/d-split map.
    ///
    /// Pin-row construction must route through this: the flat KKT row of
    /// a pinned equality is `n_x + n_s + full_g_to_c_block(g)`, NOT
    /// `n_x + n_s + g` — those differ whenever any inequality precedes
    /// the pinned equality in `g(x)`.
    pub fn full_g_to_c_block(&self, full_idx: Index) -> Option<Index> {
        self.nlp.borrow().full_g_to_c_block(full_idx)
    }

    /// Map a 0-based **full-x** index (user-TNLP variable order) to its
    /// 0-based position in the algorithm-side `x` block, or `None` when
    /// the solve removed the column because `x_l == x_u` under
    /// `fixed_variable_treatment = make_parameter`. Delegates to the
    /// held NLP's fixed-variable map.
    ///
    /// The `x` counterpart of [`Self::full_g_to_c_block`], and it must
    /// be routed through for the same reason: the flat KKT row of a
    /// user variable is `full_x_to_var_x(i)`, NOT `i` — those differ
    /// whenever any fixed variable precedes it in the user's `x`.
    /// Reports and iterates are in full-x, the factor is in var-x, and
    /// nothing about the two spaces is distinguishable by length alone
    /// on a model that happens to have no fixed variables.
    pub fn full_x_to_var_x(&self, full_idx: Index) -> Option<Index> {
        self.nlp.borrow().full_x_to_var_x(full_idx)
    }

    /// The user TNLP's variable count, the domain of
    /// [`Self::full_x_to_var_x`]. Distinct from the `x` block width
    /// whenever the solve removed a fixed variable.
    pub fn n_full_x(&self) -> Index {
        self.nlp.borrow().n_full_x()
    }

    /// [`Self::offsets`], for the corrector's residual assembly, which
    /// writes one calculated-quantity block at a time.
    pub(crate) fn offsets_public(&self) -> [usize; 9] {
        self.offsets()
    }

    /// [`Self::pack`], for the corrector, which builds a trial iterate
    /// from the flat point it is stepping.
    pub(crate) fn pack_public(&self, flat: &[Number]) -> Result<IteratesVectorMut, ()> {
        self.pack(flat)
    }

    /// The converged iterate, flattened into the compound layout.
    ///
    /// The corrector steps a point rather than a step, so it needs the
    /// iterate the step is measured from, in the same layout the step
    /// arrives in.
    pub(crate) fn curr_flat(&self, out: &mut [Number]) -> Result<(), ()> {
        if out.len() != self.dim() {
            return Err(());
        }
        let curr = {
            let d = self.data.borrow();
            d.curr.clone().ok_or(())?
        };
        let off = self.offsets();
        let blocks: [&Rc<dyn pounce_linalg::vector::Vector>; 8] = [
            &curr.x, &curr.s, &curr.y_c, &curr.y_d, &curr.z_l, &curr.z_u, &curr.v_l, &curr.v_u,
        ];
        for (i, b) in blocks.iter().enumerate() {
            let vals = crate::vec_util::dense_to_vec(&***b);
            let (a, e) = (off[i], off[i + 1]);
            if vals.len() != e - a {
                return Err(());
            }
            out[a..e].copy_from_slice(&vals);
        }
        Ok(())
    }

    /// Cumulative block offsets: `offset(i)` is the start index of
    /// block `i` in the flat slice.
    fn offsets(&self) -> [usize; 9] {
        let mut o = [0usize; 9];
        for i in 0..8 {
            o[i + 1] = o[i] + self.dims[i];
        }
        o
    }

    /// Pack a flat slice into a freshly-allocated `IteratesVectorMut`
    /// shaped like the converged iterate.
    fn pack(&self, flat: &[Number]) -> Result<IteratesVectorMut, ()> {
        let mut out = self.template.make_new_zeroed();
        let off = self.offsets();
        let blocks: [&mut Box<dyn pounce_linalg::vector::Vector>; 8] = [
            &mut out.x,
            &mut out.s,
            &mut out.y_c,
            &mut out.y_d,
            &mut out.z_l,
            &mut out.z_u,
            &mut out.v_l,
            &mut out.v_u,
        ];
        for (i, blk) in blocks.into_iter().enumerate() {
            let slice = &flat[off[i]..off[i + 1]];
            let dv = blk.as_any_mut().downcast_mut::<DenseVector>().ok_or(())?;
            dv.set_values(slice);
        }
        Ok(out)
    }

    /// Read an `IteratesVectorMut` into a flat slice. Uses
    /// [`DenseVector::expanded_values`] rather than `values()` so
    /// blocks that the IPM left in homogeneous-scalar form (typical
    /// for empty z_l/z_u/v_l/v_u when the TNLP has no bounds) are
    /// materialized rather than panicking.
    fn unpack(&self, iv: &IteratesVectorMut, out: &mut [Number]) -> Result<(), ()> {
        let off = self.offsets();
        let blocks: [&Box<dyn pounce_linalg::vector::Vector>; 8] = [
            &iv.x, &iv.s, &iv.y_c, &iv.y_d, &iv.z_l, &iv.z_u, &iv.v_l, &iv.v_u,
        ];
        for (i, blk) in blocks.into_iter().enumerate() {
            let dst = &mut out[off[i]..off[i + 1]];
            if dst.is_empty() {
                continue;
            }
            let dv = (**blk).as_any().downcast_ref::<DenseVector>().ok_or(())?;
            let ev = dv.expanded_values();
            dst.copy_from_slice(&ev);
        }
        Ok(())
    }
}

impl PdSensBacksolver {
    /// Batched-RHS back-solve over the held factor. `rhs_flat` and
    /// `lhs_flat` are row-major `(n_rhs, dim)` buffers. Equivalent to
    /// looping [`SensBacksolver::solve`] over each row but reuses one
    /// frozen `IteratesVector` for the RHS and one `IteratesVectorMut`
    /// for the result across all `n_rhs` calls into
    /// [`PdFullSpaceSolver::solve`]. The pack step writes into the
    /// existing `DenseVector` storage via `Rc::get_mut` +
    /// `set_values`, and the unpack step reads it back via `values()`
    /// /`scalar()` — skipping the per-call 8-block `make_new_zeroed`
    /// (Box alloc) in `pack` and the per-block `expanded_values()` Vec
    /// alloc in `unpack` that otherwise dominate the held-factor
    /// back-solve cost under `jax.jacrev` over a JaxProblem solve
    /// (pounce#77 follow-up).
    ///
    /// The matrix and perturbation state inside `PdFullSpaceSolver`
    /// are unchanged across calls, so each iteration hits the cached
    /// fast path in `solve_once` (`uptodate && !pretend_singular`).
    ///
    /// Like [`SensBacksolver::solve`], results are in **natural
    /// (unscaled) units** — see [`Self::solve_many_scaled_space`] for
    /// the raw solver-space back-solve.
    pub fn solve_many(&self, rhs_flat: &[Number], lhs_flat: &mut [Number], n_rhs: usize) -> bool {
        match &self.conj {
            None => self.solve_many_scaled_space(rhs_flat, lhs_flat, n_rhs),
            Some(c) => {
                let total = self.dim();
                if rhs_flat.len() != n_rhs * total || lhs_flat.len() != n_rhs * total {
                    return false;
                }
                let mut rhs_scaled = rhs_flat.to_vec();
                for row in rhs_scaled.chunks_mut(total) {
                    for (r, &ei) in row.iter_mut().zip(c.e.iter()) {
                        *r *= ei;
                    }
                }
                if !self.solve_many_scaled_space(&rhs_scaled, lhs_flat, n_rhs) {
                    return false;
                }
                for row in lhs_flat.chunks_mut(total) {
                    for (l, &fi) in row.iter_mut().zip(c.f.iter()) {
                        *l *= fi;
                    }
                }
                true
            }
        }
    }

    /// Batched-RHS back-solve against the held factor in the solver's
    /// internal **scaled** space (no natural-units conjugation). Same
    /// buffer contract as [`Self::solve_many`].
    pub fn solve_many_scaled_space(
        &self,
        rhs_flat: &[Number],
        lhs_flat: &mut [Number],
        n_rhs: usize,
    ) -> bool {
        let total = self.dim();
        if rhs_flat.len() != n_rhs * total || lhs_flat.len() != n_rhs * total {
            return false;
        }
        if n_rhs == 0 {
            return true;
        }
        let off = self.offsets();

        // Both cached tiers below assemble their elimination from the
        // *calculated* `Σ` and slacks, and fire against whatever factor
        // the last solve left behind. Neither is the right system when
        // the held iterate came from crossover and the declared-frame
        // diagonal is in force (gh#654), and the tag check cannot be
        // relied on to decline: on the first call after convergence the
        // cached tags are the algorithm's own final solve, which used
        // exactly the diagonal being corrected. So skip them outright
        // and take the per-RHS path, which carries the override. That
        // costs the batched path its inlining on a crossed-over solve —
        // one factorization, then a back-substitution per RHS, against
        // the tag cache that the shared override vector keeps warm.
        let sigma = self.sigma_override();
        let corrected = self.declared.is_some();

        // Tier 1: fully-inline flat-slice path. `PdFullSpaceSolver::
        // solve_many_cached_flat` downcasts the slack / z / v vectors to
        // `DenseVector` and the bound-expansion matrices to
        // `ExpansionMatrix` once at the top, then runs Phase 1 / Phase 3
        // as raw scatter-add / divide loops on flat slices with no dyn
        // dispatch in the per-RHS inner loops. Returns `None` if a
        // downcast fails (homogeneous-on-non-empty block, unusual matrix
        // type) — we fall to Tier 2.
        if !corrected {
            let mut pd_ref = self.pd.borrow_mut();
            let fast_flat = pd_ref.solve_many_cached_flat(
                &self.data, &self.cq, &self.nlp, n_rhs, rhs_flat, lhs_flat, self.dims,
            );
            match fast_flat {
                Some(true) => return true,
                Some(false) => return false,
                None => { /* fall through to Tier 2 */ }
            }
        }

        // Tier 2: closure-based cached-factor path. Same single
        // back-substitution through the linsol, but Phase 1 / Phase 3
        // go through `dyn Vector` / `dyn Matrix` ops on a per-RHS
        // `IteratesVectorMut`. Slower than Tier 1 but correct for
        // homogeneous DenseVectors and non-`ExpansionMatrix` bound
        // expansions.
        if !corrected {
            let mut pd_ref = self.pd.borrow_mut();
            let fast = pd_ref.solve_many_cached(
                &self.data,
                &self.cq,
                &self.nlp,
                n_rhs,
                |k, iv| {
                    let row = &rhs_flat[k * total..(k + 1) * total];
                    let _ = write_rhs_box(&mut iv.x, &row[off[0]..off[1]])
                        && write_rhs_box(&mut iv.s, &row[off[1]..off[2]])
                        && write_rhs_box(&mut iv.y_c, &row[off[2]..off[3]])
                        && write_rhs_box(&mut iv.y_d, &row[off[3]..off[4]])
                        && write_rhs_box(&mut iv.z_l, &row[off[4]..off[5]])
                        && write_rhs_box(&mut iv.z_u, &row[off[5]..off[6]])
                        && write_rhs_box(&mut iv.v_l, &row[off[6]..off[7]])
                        && write_rhs_box(&mut iv.v_u, &row[off[7]..off[8]]);
                },
                |k, iv| {
                    let row = &mut lhs_flat[k * total..(k + 1) * total];
                    let _ = read_res_block(&*iv.x, &mut row[off[0]..off[1]])
                        && read_res_block(&*iv.s, &mut row[off[1]..off[2]])
                        && read_res_block(&*iv.y_c, &mut row[off[2]..off[3]])
                        && read_res_block(&*iv.y_d, &mut row[off[3]..off[4]])
                        && read_res_block(&*iv.z_l, &mut row[off[4]..off[5]])
                        && read_res_block(&*iv.z_u, &mut row[off[5]..off[6]])
                        && read_res_block(&*iv.v_l, &mut row[off[6]..off[7]])
                        && read_res_block(&*iv.v_u, &mut row[off[7]..off[8]]);
                },
            );
            match fast {
                Some(true) => return true,
                Some(false) => return false,
                None => { /* fall through to per-RHS loop */ }
            }
        }

        // Per-RHS fallback: reuse one frozen rhs and one mut sol across
        // all n_rhs `solve` calls.
        let rhs_mut0 = self.template.make_new_zeroed();
        let mut rhs_iv = rhs_mut0.freeze();
        let mut res_iv = self.template.make_new_zeroed();

        let mut pd_ref = self.pd.borrow_mut();
        for k in 0..n_rhs {
            let rhs_row = &rhs_flat[k * total..(k + 1) * total];
            let lhs_row = &mut lhs_flat[k * total..(k + 1) * total];

            if !write_rhs_block(&mut rhs_iv.x, &rhs_row[off[0]..off[1]])
                || !write_rhs_block(&mut rhs_iv.s, &rhs_row[off[1]..off[2]])
                || !write_rhs_block(&mut rhs_iv.y_c, &rhs_row[off[2]..off[3]])
                || !write_rhs_block(&mut rhs_iv.y_d, &rhs_row[off[3]..off[4]])
                || !write_rhs_block(&mut rhs_iv.z_l, &rhs_row[off[4]..off[5]])
                || !write_rhs_block(&mut rhs_iv.z_u, &rhs_row[off[5]..off[6]])
                || !write_rhs_block(&mut rhs_iv.v_l, &rhs_row[off[6]..off[7]])
                || !write_rhs_block(&mut rhs_iv.v_u, &rhs_row[off[7]..off[8]])
            {
                return false;
            }

            let ok = pd_ref.solve_with_sigma(
                &self.data,
                &self.cq,
                &self.nlp,
                1.0,
                0.0,
                &rhs_iv,
                &mut res_iv,
                /* allow_inexact = */ true,
                /* improve_solution = */ false,
                sigma.clone(),
            );
            if !ok {
                return false;
            }

            if !read_res_block(&*res_iv.x, &mut lhs_row[off[0]..off[1]])
                || !read_res_block(&*res_iv.s, &mut lhs_row[off[1]..off[2]])
                || !read_res_block(&*res_iv.y_c, &mut lhs_row[off[2]..off[3]])
                || !read_res_block(&*res_iv.y_d, &mut lhs_row[off[3]..off[4]])
                || !read_res_block(&*res_iv.z_l, &mut lhs_row[off[4]..off[5]])
                || !read_res_block(&*res_iv.z_u, &mut lhs_row[off[5]..off[6]])
                || !read_res_block(&*res_iv.v_l, &mut lhs_row[off[6]..off[7]])
                || !read_res_block(&*res_iv.v_u, &mut lhs_row[off[7]..off[8]])
            {
                return false;
            }
        }
        true
    }
}

/// Headroom on the representability half of [`declared_slack_floor`],
/// matching `ipopt_cq`'s `SIGMA_OVERFLOW_HEADROOM` (gh#655) because it
/// bounds the same quantity for the same reason: `Σ_x` sums a lower and
/// an upper ratio into one diagonal entry, so bounding each by `MAX/4`
/// bounds the sum by `MAX/2` with room for the rounding in the divide.
const SIGMA_OVERFLOW_HEADROOM: Number = 4.0;

/// The smallest offset from a bound that the declared-frame slack is
/// allowed to be: the larger of what double precision tells apart from
/// the bound itself, and what keeps `Σ = z/s` inside the double range.
///
/// A crossed-over point is *on* its active bounds, so its declared-frame
/// slack is the residual of the QP step and the line search — measured
/// around `1.8e-12` (gh#653), comfortably above both, so this does
/// nothing on the ordinary path. Each half covers a different corner:
///
/// * **`eps·max(1,|bound|)`** — a pivot landing on the bound exactly.
///   `Σ = z/0` is not a stiffer pin, it is a `NaN` in the KKT matrix.
///   Flooring says the honest thing instead: below this distance the
///   point *is* the bound, and the slack carries no further information
///   about how far inside it sits.
/// * **`z_max/(f64::MAX/4)`** — a multiplier large enough that `z/s`
///   leaves the double range even at a slack this frame considers
///   resolvable. This is gh#655's floor, which the live path gained in
///   `CalculateSafeSlack`; the declared frame does not go through that
///   function, so without carrying the bound here explicitly the
///   guarantee would stop at the frame boundary. It takes `max_i z_i`
///   over the block rather than each bound's own `z`, matching gh#655
///   and conservative in the harmless direction — raising a slack only
///   lowers `Σ`. A non-finite `z` leaves the floor alone; the iterate
///   finiteness checks own that case.
///
/// What is deliberately *not* borrowed is the rest of
/// `CalculateSafeSlack`: it raises a below-floor slack to
/// `max(μ/z, s_min)`, i.e. straight back to the `μ/z` standoff crossover
/// exists to remove (the same reason gh#646 declined it for the residual
/// report). Only the representability bound crosses over, because that
/// one is about what a double can hold, not about where the barrier
/// would have put the point.
fn declared_slack_floor(bound: Number, z_max: Number) -> Number {
    let resolvable = Number::EPSILON * bound.abs().max(1.0);
    if z_max.is_finite() && z_max > 0.0 {
        // Divide before multiplying so a `z_max` near `f64::MAX` cannot
        // overflow the floor itself.
        resolvable.max(z_max / (Number::MAX / SIGMA_OVERFLOW_HEADROOM))
    } else {
        resolvable
    }
}

/// `Σ = Σ_l z_l/s_l + Σ_u z_u/s_u` for one primal block, with the slacks
/// taken against `b_l` / `b_u` instead of the NLP's live (relaxed)
/// bounds. Returns the diagonal plus the two slack vectors it was built
/// from, compressed in the `p_l` / `p_u` spaces.
///
/// Mirrors `IpoptCalculatedQuantities`' own `curr_sigma_x` /
/// `curr_sigma_s` — the same `Pᵀx − b` slack and the same
/// `add_m_sinv_z` accumulation — so the only difference between this and
/// the cached value is which bounds it measured against, and the floor
/// above in place of the safe-slack correction.
#[allow(clippy::too_many_arguments)]
fn declared_frame_sigma(
    p_l: &dyn pounce_linalg::Matrix,
    p_u: &dyn pounce_linalg::Matrix,
    primal: &dyn pounce_linalg::Vector,
    b_l: &[Number],
    b_u: &[Number],
    z_l: &dyn pounce_linalg::Vector,
    z_u: &dyn pounce_linalg::Vector,
    n: usize,
) -> (Rc<dyn pounce_linalg::Vector>, Vec<Number>, Vec<Number>) {
    use pounce_linalg::Vector;
    use pounce_linalg::dense_vector::DenseVectorSpace;

    // One scalar over both sides of the block, as gh#655 does: the two
    // ratios land in the same `Σ` entry, so the bound that keeps their
    // sum representable has to be taken over both.
    let z_max = z_l.amax().max(z_u.amax());

    // `lower`: s = Pᵀx − b_l. Otherwise: s = b_u − Pᵀx.
    let slack = |p: &dyn pounce_linalg::Matrix, b: &[Number], lower: bool| -> DenseVector {
        let mut v = DenseVector::new(DenseVectorSpace::new(b.len() as Index));
        if !b.is_empty() {
            v.values_mut().copy_from_slice(b);
        }
        let (alpha, beta) = if lower { (1.0, -1.0) } else { (-1.0, 1.0) };
        p.trans_mult_vector(alpha, primal, beta, &mut v);
        for (s, &bi) in v.values_mut().iter_mut().zip(b.iter()) {
            *s = s.max(declared_slack_floor(bi, z_max));
        }
        v
    };
    let s_l = slack(p_l, b_l, true);
    let s_u = slack(p_u, b_u, false);

    let mut sigma = DenseVector::new(DenseVectorSpace::new(n as Index));
    sigma.set(0.0);
    p_l.add_m_sinv_z(1.0, &s_l, z_l, &mut sigma);
    p_u.add_m_sinv_z(1.0, &s_u, z_u, &mut sigma);
    (
        Rc::new(sigma) as Rc<dyn pounce_linalg::Vector>,
        s_l.expanded_values(),
        s_u.expanded_values(),
    )
}

/// Write `slice` into the `DenseVector` behind `b` in place. Used by
/// the fast path's `write_rhs` closure, where the new
/// `PdFullSpaceSolver::solve_many_cached` API hands back an
/// `IteratesVectorMut` (Box-backed blocks).
fn write_rhs_box(b: &mut Box<dyn pounce_linalg::vector::Vector>, slice: &[Number]) -> bool {
    if slice.is_empty() {
        return true;
    }
    let Some(dv) = b.as_any_mut().downcast_mut::<DenseVector>() else {
        return false;
    };
    dv.set_values(slice);
    true
}

/// Write `slice` into the `DenseVector` behind `rc` in place. Returns
/// `false` if the Rc is unexpectedly shared (would indicate a bug in
/// `PdFullSpaceSolver::solve`'s borrow discipline — it should never
/// `Rc::clone` from the rhs vector) or if the block is not a
/// `DenseVector`.
fn write_rhs_block(rc: &mut Rc<dyn pounce_linalg::vector::Vector>, slice: &[Number]) -> bool {
    if slice.is_empty() {
        return true;
    }
    let Some(v) = Rc::get_mut(rc) else {
        return false;
    };
    let Some(dv) = v.as_any_mut().downcast_mut::<DenseVector>() else {
        return false;
    };
    dv.set_values(slice);
    true
}

/// Read the `DenseVector` behind `blk` into `dst`. Handles the
/// homogeneous case (empty z/v blocks for a TNLP with no bounds) by
/// broadcasting the scalar rather than calling `expanded_values()`,
/// which would allocate a fresh `Vec<Number>` every call.
fn read_res_block(blk: &dyn pounce_linalg::vector::Vector, dst: &mut [Number]) -> bool {
    if dst.is_empty() {
        return true;
    }
    let Some(dv) = blk.as_any().downcast_ref::<DenseVector>() else {
        return false;
    };
    if dv.is_homogeneous() {
        let s = dv.scalar();
        for x in dst.iter_mut() {
            *x = s;
        }
    } else {
        dst.copy_from_slice(dv.values());
    }
    true
}

impl PdSensBacksolver {
    /// Single-RHS back-solve against the held factor in the solver's
    /// internal **scaled** space (no natural-units conjugation). This
    /// is the value [`SensBacksolver::solve`] returned before
    /// pounce#128; kept for callers that want the raw factor.
    pub fn solve_scaled_space(&self, rhs: &[Number], lhs: &mut [Number]) -> bool {
        let total = self.dim();
        if rhs.len() != total || lhs.len() != total {
            return false;
        }
        // Pack rhs into block form.
        let rhs_mut = match self.pack(rhs) {
            Ok(v) => v,
            Err(()) => return false,
        };
        let rhs_iv = rhs_mut.freeze();
        // Fresh result slot, zeroed.
        let mut res_iv = self.template.make_new_zeroed();

        // K · lhs = rhs   ⇒   solve(α=1, β=0, rhs, res) writes
        // res = K⁻¹ · rhs.
        //
        // `allow_inexact=true` mirrors upstream sIPOPT's
        // `SensSimpleBacksolver`: skip `PdFullSpaceSolver`'s iterative-
        // refinement loop and accept the first back-solve against the
        // held factor. The IPM-level refinement (`min_refinement_steps
        // = 1`, residual_ratio_max = 1e-10`) is there to clean up
        // numerical noise during forward IPM steps; for the held-factor
        // back-solve used by sens / JaxProblem bwd, it ~doubles the
        // per-call cost and produces gains that are below `tol`. Under
        // `jax.jacrev` over a JaxProblem solve this dominates the wall
        // time at moderate `n+m` (pounce#77 follow-up).
        let ok = {
            let mut pd_ref = self.pd.borrow_mut();
            pd_ref.solve_with_sigma(
                &self.data,
                &self.cq,
                &self.nlp,
                1.0,
                0.0,
                &rhs_iv,
                &mut res_iv,
                /* allow_inexact = */ true,
                /* improve_solution = */ false,
                // Identity on every ordinary solve; the declared-frame
                // diagonal when the held iterate came from crossover
                // (gh#654).
                self.sigma_override(),
            )
        };
        if !ok {
            return false;
        }
        self.unpack(&res_iv, lhs).is_ok()
    }
}

impl SensBacksolver for PdSensBacksolver {
    fn dim(&self) -> usize {
        self.dims.iter().sum()
    }

    /// `F` itself, the vector [`Self::solve`] post-multiplies its
    /// result by, so a caller converting an iterate quantity uses the
    /// same numbers the back-solve used.
    fn natural_units_factor(&self) -> Option<&[Number]> {
        self.conj.as_ref().map(|c| c.f.as_slice())
    }

    fn bound_rows(&self) -> Option<&[crate::backsolver::BoundRow]> {
        self.bound_vars.as_deref().map(|v| v.as_slice())
    }

    fn supports_release(&self) -> bool {
        self.bound_vars.is_some()
    }

    fn solve_released(&self, released: &[usize], rhs: &[Number], lhs: &mut [Number]) -> bool {
        self.solve_released_inner(released, rhs, lhs, false)
    }

    fn solve_released_step(&self, released: &[usize], rhs: &[Number], lhs: &mut [Number]) -> bool {
        self.solve_released_inner(released, rhs, lhs, true)
    }

    /// Solve `K · lhs = rhs` against the converged factor, in
    /// **natural (unscaled) units** (pounce#128): when the NLP carries
    /// active scaling (`nlp_scaling_method`, `obj_scaling_factor`,
    /// user scaling) the RHS is pre-multiplied by `E` and the result
    /// post-multiplied by `F` (see the `conj` field doc), so
    /// `lhs = K_natural⁻¹ rhs` for **all eight blocks** — including
    /// the z/v bound-multiplier rows. Use
    /// [`Self::solve_scaled_space`] for the raw factor.
    fn solve(&self, rhs: &[Number], lhs: &mut [Number]) -> bool {
        match &self.conj {
            None => self.solve_scaled_space(rhs, lhs),
            Some(c) => {
                let total = self.dim();
                if rhs.len() != total || lhs.len() != total {
                    return false;
                }
                let rhs_scaled: Vec<Number> =
                    rhs.iter().zip(c.e.iter()).map(|(&r, &ei)| r * ei).collect();
                if !self.solve_scaled_space(&rhs_scaled, lhs) {
                    return false;
                }
                for (l, &fi) in lhs.iter_mut().zip(c.f.iter()) {
                    *l *= fi;
                }
                true
            }
        }
    }
}

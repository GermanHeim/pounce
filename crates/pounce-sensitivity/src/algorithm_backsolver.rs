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
use pounce_algorithm::kkt::pd_full_space_solver::PdFullSpaceSolver;
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
        })
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

        // Tier 1: fully-inline flat-slice path. `PdFullSpaceSolver::
        // solve_many_cached_flat` downcasts the slack / z / v vectors to
        // `DenseVector` and the bound-expansion matrices to
        // `ExpansionMatrix` once at the top, then runs Phase 1 / Phase 3
        // as raw scatter-add / divide loops on flat slices with no dyn
        // dispatch in the per-RHS inner loops. Returns `None` if a
        // downcast fails (homogeneous-on-non-empty block, unusual matrix
        // type) — we fall to Tier 2.
        {
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
        {
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

            let ok = pd_ref.solve(
                &self.data,
                &self.cq,
                &self.nlp,
                1.0,
                0.0,
                &rhs_iv,
                &mut res_iv,
                /* allow_inexact = */ true,
                /* improve_solution = */ false,
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
            pd_ref.solve(
                &self.data,
                &self.cq,
                &self.nlp,
                1.0,
                0.0,
                &rhs_iv,
                &mut res_iv,
                /* allow_inexact = */ true,
                /* improve_solution = */ false,
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

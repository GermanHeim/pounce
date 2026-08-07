//! Per-variable scaling as a TNLP wrapper (issue #486 stage 2).
//!
//! The solver's own scaling handles the objective and the constraint
//! rows but has no representation for a change of variables, so a
//! per-variable `scaling_factor` reached the core and was rejected
//! (issue #483 stage 1). This wrapper supplies the missing piece by
//! substituting variables one level below the algorithm, the same way
//! `PresolveTnlp` transforms coordinates and inverts the transform in
//! `finalize_solution`.
//!
//! # Convention
//!
//! `d_i > 0` is the factor for variable `i`, following Pyomo's
//! `scaling_factor` suffix: the scaled variable is `x̃ = d ⊙ x`, so
//! the algorithm sees `x̃` and the inner TNLP always sees `x = x̃ ⊘ d`.
//!
//! # Transforms
//!
//! Diagonal scaling leaves sparsity unchanged, so every transform is
//! an elementwise multiply or divide:
//!
//! | quantity | transform |
//! |---|---|
//! | bounds, starting point | `× d` |
//! | objective, constraint values | unchanged |
//! | objective gradient | `÷ d` |
//! | Jacobian entry | `÷ d[col]` |
//! | Hessian entry | `÷ (d[row] · d[col])` |
//! | solution `x` | `÷ d` |
//! | bound multipliers `z_L`, `z_U` | `× d` on the way out, `÷ d` in |
//! | constraint multipliers `λ` | unchanged |
//!
//! The multiplier rule follows from stationarity. Writing the scaled
//! condition `∇f̃ - J̃ᵀλ - z̃_L + z̃_U = 0` and substituting
//! `∇f̃ = ∇f ⊘ d`, `J̃ᵀλ = (Jᵀλ) ⊘ d` gives `z_L = d ⊙ z̃_L`, with `λ`
//! untouched because `g` is untouched.
//!
//! # Factors come from the inner TNLP
//!
//! [`wrap_with_scaling`] calls the inner TNLP's
//! [`TNLP::get_scaling_parameters`] once, keeps the variable factors
//! for itself, and forwards the objective and constraint factors
//! upward unchanged, since the core already models those. Without
//! non-trivial variable factors it returns `Ok(None)` and the caller
//! keeps the bare TNLP, so an unscaled solve pays nothing.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_common::types::Number;

use crate::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, MetaData, NlpInfo, ScalingRequest, Solution,
    SparsityRequest, StartingPoint, TNLP,
};

/// Smallest variable factor accepted. Anything below this makes the
/// substitution numerically meaningless in the direction it scales.
const MIN_FACTOR: Number = 1e-12;

/// A TNLP presenting `x̃ = d ⊙ x` to the algorithm. See the module
/// documentation for the transform table.
pub struct ScalingTnlp {
    inner: Rc<RefCell<dyn TNLP>>,
    /// Per-variable factors, all finite and `>= MIN_FACTOR`.
    d: Vec<Number>,
    /// Objective and constraint scaling read off the inner TNLP and
    /// passed upward for the core to apply.
    obj_scaling: Number,
    g_scaling: Option<Vec<Number>>,
    /// Column of each Jacobian nonzero, and (row, column) of each
    /// Hessian nonzero, captured on the structure call so the values
    /// call knows which factors to divide by. Zero-based regardless
    /// of the inner TNLP's index style.
    jac_cols: Vec<usize>,
    hess_rc: Vec<(usize, usize)>,
    index_offset: usize,
    /// Scratch for the unscaled `x` handed to the inner TNLP.
    x_scratch: Vec<Number>,
}

impl ScalingTnlp {
    /// Fill the Jacobian column cache from the inner TNLP if a values
    /// call arrives before a structure call. The algorithm always asks
    /// for structure first, but a wrapper that panics when it does not
    /// is worse than one that asks for itself.
    fn ensure_jac_cols(&mut self) -> bool {
        if !self.jac_cols.is_empty() {
            return true;
        }
        let Some(info) = self.inner.borrow_mut().get_nlp_info() else {
            return false;
        };
        let nnz = info.nnz_jac_g as usize;
        let (mut irow, mut jcol) = (vec![0; nnz], vec![0; nnz]);
        let ok = self.inner.borrow_mut().eval_jac_g(
            None,
            true,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol,
            },
        );
        if ok {
            self.jac_cols = jcol
                .iter()
                .map(|&c| c as usize - self.index_offset)
                .collect();
        }
        ok
    }

    /// The Hessian counterpart of [`Self::ensure_jac_cols`].
    fn ensure_hess_rc(&mut self) -> bool {
        if !self.hess_rc.is_empty() {
            return true;
        }
        let Some(info) = self.inner.borrow_mut().get_nlp_info() else {
            return false;
        };
        let nnz = info.nnz_h_lag as usize;
        let (mut irow, mut jcol) = (vec![0; nnz], vec![0; nnz]);
        let ok = self.inner.borrow_mut().eval_h(
            None,
            true,
            1.0,
            None,
            true,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol,
            },
        );
        if ok {
            self.hess_rc = irow
                .iter()
                .zip(jcol.iter())
                .map(|(&r, &c)| {
                    (
                        r as usize - self.index_offset,
                        c as usize - self.index_offset,
                    )
                })
                .collect();
        }
        ok
    }

    /// Unscale the algorithm's `x̃` into the scratch buffer and return
    /// it, so the inner TNLP always sees the user's own coordinates.
    fn unscale_x(&mut self, x: &[Number]) -> &[Number] {
        let n = self.d.len().min(x.len());
        self.x_scratch.clear();
        self.x_scratch.extend((0..n).map(|i| x[i] / self.d[i]));
        &self.x_scratch
    }
}

/// Wrap `inner` when it requests non-trivial per-variable scaling.
///
/// Returns `Ok(None)` when no variable scaling is requested, in which
/// case the caller should use `inner` unchanged. Returns `Err` when a
/// factor is not finite, not positive, or below [`MIN_FACTOR`]: a
/// negative factor would reverse a variable's direction and swap its
/// bounds, which no caller has asked for and which is better refused
/// than silently applied.
pub fn wrap_with_scaling(
    inner: Rc<RefCell<dyn TNLP>>,
) -> Result<Option<Rc<RefCell<dyn TNLP>>>, String> {
    let info = inner
        .borrow_mut()
        .get_nlp_info()
        .ok_or_else(|| "scaling: get_nlp_info failed on the inner problem".to_string())?;
    let n = info.n as usize;
    let m = info.m as usize;

    let mut obj_scaling: Number = 1.0;
    let mut use_x = false;
    let mut x_scaling = vec![1.0; n];
    let mut use_g = false;
    let mut g_scaling = vec![1.0; m];
    let supplied = inner.borrow_mut().get_scaling_parameters(ScalingRequest {
        obj_scaling: &mut obj_scaling,
        use_x_scaling: &mut use_x,
        x_scaling: &mut x_scaling,
        use_g_scaling: &mut use_g,
        g_scaling: &mut g_scaling,
    });
    if !supplied || !use_x {
        return Ok(None);
    }
    if x_scaling.len() != n {
        return Err(format!(
            "scaling: x_scaling has {} entries for {n} variables",
            x_scaling.len()
        ));
    }
    if x_scaling.iter().all(|&s| s == 1.0) {
        return Ok(None);
    }
    for (i, &s) in x_scaling.iter().enumerate() {
        if !s.is_finite() || s <= 0.0 {
            return Err(format!(
                "scaling: variable {i} has scaling factor {s}; factors must be \
                 finite and positive (a negative factor would reverse the \
                 variable and swap its bounds)"
            ));
        }
        if s < MIN_FACTOR {
            return Err(format!(
                "scaling: variable {i} has scaling factor {s}, below the {MIN_FACTOR:e} \
                 floor; such a factor makes the substitution numerically meaningless"
            ));
        }
    }

    let index_offset = match info.index_style {
        IndexStyle::C => 0,
        IndexStyle::Fortran => 1,
    };
    Ok(Some(Rc::new(RefCell::new(ScalingTnlp {
        inner,
        d: x_scaling,
        obj_scaling,
        g_scaling: if use_g && g_scaling.len() == m {
            Some(g_scaling)
        } else {
            None
        },
        jac_cols: Vec::new(),
        hess_rc: Vec::new(),
        index_offset,
        x_scratch: Vec::with_capacity(n),
    }))))
}

/// The factors a [`ScalingTnlp`] applies, if `tnlp` is one.
///
/// Lets a caller that reads the algorithm's own iterate rather than
/// the `finalize_solution` payload undo the substitution itself.
pub fn factors_of(tnlp: &Rc<RefCell<dyn TNLP>>) -> Option<Vec<Number>> {
    tnlp.borrow().scaling_factors()
}

impl TNLP for ScalingTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        self.inner.borrow_mut().get_nlp_info()
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        let BoundsInfo { x_l, x_u, g_l, g_u } = b;
        let ok = self
            .inner
            .borrow_mut()
            .get_bounds_info(BoundsInfo { x_l, x_u, g_l, g_u });
        if !ok {
            return false;
        }
        // Positive factors preserve the order of the bounds, so no
        // swap is needed; infinities scale to themselves.
        for (i, &s) in self.d.iter().enumerate() {
            x_l[i] *= s;
            x_u[i] *= s;
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        let StartingPoint {
            init_x,
            x,
            init_z,
            z_l,
            z_u,
            init_lambda,
            lambda,
        } = sp;
        let ok = self.inner.borrow_mut().get_starting_point(StartingPoint {
            init_x,
            x,
            init_z,
            z_l,
            z_u,
            init_lambda,
            lambda,
        });
        if !ok {
            return false;
        }
        if init_x {
            for (i, &s) in self.d.iter().enumerate() {
                x[i] *= s;
            }
        }
        if init_z {
            // z_L = d ⊙ z̃_L, so the warm start divides on the way in.
            for (i, &s) in self.d.iter().enumerate() {
                z_l[i] /= s;
                z_u[i] /= s;
            }
        }
        let _ = (init_lambda, lambda); // λ is unchanged by the substitution
        true
    }

    fn eval_f(&mut self, x: &[Number], new_x: bool) -> Option<Number> {
        let xu = self.unscale_x(x).to_vec();
        self.inner.borrow_mut().eval_f(&xu, new_x)
    }

    fn eval_grad_f(&mut self, x: &[Number], new_x: bool, grad_f: &mut [Number]) -> bool {
        let xu = self.unscale_x(x).to_vec();
        if !self.inner.borrow_mut().eval_grad_f(&xu, new_x, grad_f) {
            return false;
        }
        for (i, &s) in self.d.iter().enumerate() {
            grad_f[i] /= s;
        }
        true
    }

    fn eval_g(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
        let xu = self.unscale_x(x).to_vec();
        self.inner.borrow_mut().eval_g(&xu, new_x, g)
    }

    fn eval_jac_g(&mut self, x: Option<&[Number]>, new_x: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let ok = self.inner.borrow_mut().eval_jac_g(
                    x,
                    new_x,
                    SparsityRequest::Structure { irow, jcol },
                );
                if ok {
                    // Remember each nonzero's column so the values
                    // call knows which factor divides it.
                    self.jac_cols = jcol
                        .iter()
                        .map(|&c| c as usize - self.index_offset)
                        .collect();
                }
                ok
            }
            SparsityRequest::Values { values } => {
                let xu = x.map(|xs| self.unscale_x(xs).to_vec());
                let ok = self.inner.borrow_mut().eval_jac_g(
                    xu.as_deref(),
                    new_x,
                    SparsityRequest::Values { values },
                );
                if !ok || !self.ensure_jac_cols() {
                    return false;
                }
                for (k, v) in values.iter_mut().enumerate() {
                    *v /= self.d[self.jac_cols[k]];
                }
                true
            }
        }
    }

    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let ok = self.inner.borrow_mut().eval_h(
                    x,
                    new_x,
                    obj_factor,
                    lambda,
                    new_lambda,
                    SparsityRequest::Structure { irow, jcol },
                );
                if ok {
                    self.hess_rc = irow
                        .iter()
                        .zip(jcol.iter())
                        .map(|(&r, &c)| {
                            (
                                r as usize - self.index_offset,
                                c as usize - self.index_offset,
                            )
                        })
                        .collect();
                }
                ok
            }
            SparsityRequest::Values { values } => {
                let xu = x.map(|xs| self.unscale_x(xs).to_vec());
                let ok = self.inner.borrow_mut().eval_h(
                    xu.as_deref(),
                    new_x,
                    obj_factor,
                    lambda,
                    new_lambda,
                    SparsityRequest::Values { values },
                );
                if !ok || !self.ensure_hess_rc() {
                    return false;
                }
                for (k, v) in values.iter_mut().enumerate() {
                    let (r, c) = self.hess_rc[k];
                    *v /= self.d[r] * self.d[c];
                }
                true
            }
        }
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, ip_data: &IpoptData, ip_cq: &IpoptCq) {
        let x: Vec<Number> = sol
            .x
            .iter()
            .zip(self.d.iter())
            .map(|(&v, &s)| v / s)
            .collect();
        let z_l: Vec<Number> = sol
            .z_l
            .iter()
            .zip(self.d.iter())
            .map(|(&v, &s)| v * s)
            .collect();
        let z_u: Vec<Number> = sol
            .z_u
            .iter()
            .zip(self.d.iter())
            .map(|(&v, &s)| v * s)
            .collect();
        self.inner.borrow_mut().finalize_solution(
            Solution {
                status: sol.status,
                x: &x,
                z_l: &z_l,
                z_u: &z_u,
                g: sol.g,
                lambda: sol.lambda,
                obj_value: sol.obj_value,
            },
            ip_data,
            ip_cq,
        );
    }

    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        // The variable factors were consumed at construction; the
        // objective and constraint factors belong to the core, which
        // already models them.
        *req.obj_scaling = self.obj_scaling;
        *req.use_x_scaling = false;
        match &self.g_scaling {
            Some(g) if g.len() == req.g_scaling.len() => {
                req.g_scaling.copy_from_slice(g);
                *req.use_g_scaling = true;
            }
            _ => *req.use_g_scaling = false,
        }
        true
    }

    fn get_var_con_metadata(&mut self, var: &mut MetaData, con: &mut MetaData) -> bool {
        self.inner.borrow_mut().get_var_con_metadata(var, con)
    }

    fn finalize_metadata(&mut self, var: &MetaData, con: &MetaData) {
        self.inner.borrow_mut().finalize_metadata(var, con);
    }

    fn get_variables_linearity(&mut self, types: &mut [crate::tnlp::Linearity]) -> bool {
        // A change of variables by a positive diagonal preserves
        // linearity of every row and of the objective.
        self.inner.borrow_mut().get_variables_linearity(types)
    }

    fn get_constraints_linearity(&mut self, types: &mut [crate::tnlp::Linearity]) -> bool {
        self.inner.borrow_mut().get_constraints_linearity(types)
    }

    fn get_objective_variables_linearity(&mut self, types: &mut [crate::tnlp::Linearity]) -> bool {
        self.inner
            .borrow_mut()
            .get_objective_variables_linearity(types)
    }

    fn scaling_factors(&self) -> Option<Vec<Number>> {
        Some(self.d.clone())
    }

    fn is_presolve_wrapper(&self) -> bool {
        // Forward rather than claim: this flag asks whether a generic
        // PRESOLVE wrapper sits below, so that the solve entry point
        // does not add a second one. Answering `true` here would make
        // presolve decline every scaled problem.
        self.inner.borrow().is_presolve_wrapper()
    }
}

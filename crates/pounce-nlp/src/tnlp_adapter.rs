//! TNLP → IpoptNLP adapter — Phase-3-scope port of
//! `Interfaces/IpTNLPAdapter.{hpp,cpp}`.
//!
//! Splits a user-facing [`TNLP`] (mixed bounds, mixed equality /
//! inequality constraints) into the separated form
//!     min  f(x)
//!     s.t. c(x) = 0    (equality)
//!          d(x) - s = 0,  d_L ≤ s ≤ d_U   (inequality with slacks)
//!          x_L ≤ x ≤ x_U
//! used by the algorithm. This file ships only the **classification**
//! piece — bounds and constraints are sorted into eq/ineq/{lower,upper}
//! sets and the corresponding index maps are computed. The full adapter
//! (function-evaluation routing, sparsity propagation, fixed-variable
//! treatment, scaling) lands with Phase 5 when `IpoptNLP` and
//! `ExpansionMatrix` are wired up.

use crate::tnlp::{BoundsInfo, NlpInfo, TNLP};
use pounce_common::exception::{ExceptionKind, SolverException};
use pounce_common::types::{Index, Number};
use std::cell::RefCell;
use std::rc::Rc;

/// Default infinity threshold for variable / constraint bounds. Matches
/// the `nlp_lower_bound_inf` / `nlp_upper_bound_inf` registered option
/// defaults in upstream Ipopt (`±1e19`).
pub const DEFAULT_NLP_LOWER_BOUND_INF: Number = -1.0e19;
pub const DEFAULT_NLP_UPPER_BOUND_INF: Number = 1.0e19;

/// Sorted decomposition of a TNLP's bounds and constraints. All `*_map`
/// vectors carry **0-based** indices into the full TNLP space.
#[derive(Debug, Clone)]
pub struct BoundClassification {
    pub n_full_x: Index,
    pub n_full_g: Index,
    /// Number of variables with `x_l == x_u`. With
    /// `fixed_variable_treatment = make_parameter` (the upstream
    /// default — and pounce's only mode today) these are removed from
    /// the active set; their indices live in `x_fixed_map` and their
    /// values in `x_fixed_vals`.
    pub n_x_fixed: Index,
    /// Indices in `[0, n_full_x)` that are not fixed (`x_l < x_u`).
    /// Length is `n_x_var = n_full_x - n_x_fixed`.
    pub x_not_fixed_map: Vec<Index>,
    /// Indices in `[0, n_full_x)` that ARE fixed. Length `n_x_fixed`.
    pub x_fixed_map: Vec<Index>,
    /// Fixed values (== `x_l[i] == x_u[i]`) for each entry of
    /// `x_fixed_map`. Used by `OrigIpoptNlp::lift_x_to_full` to insert
    /// the correct constant into the full-x array before calling the
    /// user's TNLP.
    pub x_fixed_vals: Vec<Number>,
    /// Maps full-x index → var-x index, with `-1` for fixed entries.
    /// Used by sparsity filtering for the Jacobian / Hessian.
    pub full_to_var: Vec<Index>,
    /// Subset of `x_not_fixed_map`'s domain (i.e. positions in `x_var`)
    /// where a finite lower bound is present.
    pub x_l_map: Vec<Index>,
    /// Same for finite upper bounds.
    pub x_u_map: Vec<Index>,
    /// Equality constraint count and indices into `[0, n_full_g)`.
    pub n_c: Index,
    pub c_map: Vec<Index>,
    /// Inequality constraint count and indices into `[0, n_full_g)`.
    pub n_d: Index,
    pub d_map: Vec<Index>,
    /// Subset of `[0, n_d)` with a finite lower bound.
    pub d_l_map: Vec<Index>,
    /// Subset of `[0, n_d)` with a finite upper bound.
    pub d_u_map: Vec<Index>,
}

impl BoundClassification {
    pub fn n_x_var(&self) -> Index {
        self.x_not_fixed_map.len() as Index
    }
    pub fn n_x_l(&self) -> Index {
        self.x_l_map.len() as Index
    }
    pub fn n_x_u(&self) -> Index {
        self.x_u_map.len() as Index
    }
    pub fn n_d_l(&self) -> Index {
        self.d_l_map.len() as Index
    }
    pub fn n_d_u(&self) -> Index {
        self.d_u_map.len() as Index
    }
}

/// Phase-3 TNLP wrapper. Holds shared ownership of the user's TNLP and
/// the cached problem dimensions / decomposition. Phase 5 will extend
/// this struct with cached scaled/unscaled `f`, `g`, `grad_f`, `jac_g`
/// and a `new_x` flag.
pub struct TNLPAdapter {
    tnlp: Rc<RefCell<dyn TNLP>>,
    info: NlpInfo,
    classification: BoundClassification,
    nlp_lower_bound_inf: Number,
    nlp_upper_bound_inf: Number,
}

impl std::fmt::Debug for TNLPAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TNLPAdapter")
            .field("info", &self.info)
            .field("classification", &self.classification)
            .field("nlp_lower_bound_inf", &self.nlp_lower_bound_inf)
            .field("nlp_upper_bound_inf", &self.nlp_upper_bound_inf)
            .finish_non_exhaustive()
    }
}

impl TNLPAdapter {
    /// Construct an adapter from a TNLP. Reads `get_nlp_info` and
    /// `get_bounds_info`, performs bound + constraint classification,
    /// and stores the result. Uses the default `±1e19` infinity
    /// thresholds.
    pub fn new(tnlp: Rc<RefCell<dyn TNLP>>) -> Result<Self, SolverException> {
        Self::new_with_inf(
            tnlp,
            DEFAULT_NLP_LOWER_BOUND_INF,
            DEFAULT_NLP_UPPER_BOUND_INF,
        )
    }

    /// Construct an adapter with custom infinity thresholds (the user
    /// can override these via `nlp_lower_bound_inf` / `nlp_upper_bound_inf`).
    pub fn new_with_inf(
        tnlp: Rc<RefCell<dyn TNLP>>,
        nlp_lower_bound_inf: Number,
        nlp_upper_bound_inf: Number,
    ) -> Result<Self, SolverException> {
        if nlp_lower_bound_inf >= nlp_upper_bound_inf {
            return Err(SolverException::new(
                ExceptionKind::OPTION_INVALID,
                "Option \"nlp_lower_bound_inf\" must be smaller than \
                 \"nlp_upper_bound_inf\".",
                file!(),
                line!() as Index,
            ));
        }

        let info = {
            let mut t = tnlp.borrow_mut();
            t.get_nlp_info().ok_or_else(|| {
                SolverException::new(
                    ExceptionKind::INVALID_TNLP,
                    "TNLP::get_nlp_info returned None.",
                    file!(),
                    line!() as Index,
                )
            })?
        };

        if info.n <= 0 {
            return Err(SolverException::new(
                ExceptionKind::INVALID_TNLP,
                format!("TNLP::get_nlp_info reported n = {} (must be > 0).", info.n),
                file!(),
                line!() as Index,
            ));
        }
        if info.m < 0 {
            return Err(SolverException::new(
                ExceptionKind::INVALID_TNLP,
                format!("TNLP::get_nlp_info reported m = {} (must be ≥ 0).", info.m),
                file!(),
                line!() as Index,
            ));
        }

        let n_full_x = info.n;
        let n_full_g = info.m;

        let mut x_l = vec![0.0; n_full_x as usize];
        let mut x_u = vec![0.0; n_full_x as usize];
        let mut g_l = vec![0.0; n_full_g as usize];
        let mut g_u = vec![0.0; n_full_g as usize];

        {
            let mut t = tnlp.borrow_mut();
            let ok = t.get_bounds_info(BoundsInfo {
                x_l: &mut x_l,
                x_u: &mut x_u,
                g_l: &mut g_l,
                g_u: &mut g_u,
            });
            if !ok {
                return Err(SolverException::new(
                    ExceptionKind::INVALID_TNLP,
                    "TNLP::get_bounds_info returned false.",
                    file!(),
                    line!() as Index,
                ));
            }
        }

        let classification = classify_bounds(
            n_full_x,
            n_full_g,
            &x_l,
            &x_u,
            &g_l,
            &g_u,
            nlp_lower_bound_inf,
            nlp_upper_bound_inf,
        )?;

        Ok(Self {
            tnlp,
            info,
            classification,
            nlp_lower_bound_inf,
            nlp_upper_bound_inf,
        })
    }

    pub fn nlp_info(&self) -> &NlpInfo {
        &self.info
    }

    pub fn classification(&self) -> &BoundClassification {
        &self.classification
    }

    pub fn nlp_lower_bound_inf(&self) -> Number {
        self.nlp_lower_bound_inf
    }

    pub fn nlp_upper_bound_inf(&self) -> Number {
        self.nlp_upper_bound_inf
    }

    pub fn tnlp(&self) -> &Rc<RefCell<dyn TNLP>> {
        &self.tnlp
    }
}

#[allow(clippy::too_many_arguments)]
fn classify_bounds(
    n_full_x: Index,
    n_full_g: Index,
    x_l: &[Number],
    x_u: &[Number],
    g_l: &[Number],
    g_u: &[Number],
    lo_inf: Number,
    up_inf: Number,
) -> Result<BoundClassification, SolverException> {
    let nx = n_full_x as usize;
    let ng = n_full_g as usize;

    // --- Variables ---------------------------------------------------
    let mut x_not_fixed_map: Vec<Index> = Vec::with_capacity(nx);
    let mut x_fixed_map: Vec<Index> = Vec::new();
    let mut x_fixed_vals: Vec<Number> = Vec::new();
    let mut full_to_var: Vec<Index> = vec![-1; nx];
    let mut x_l_map: Vec<Index> = Vec::new();
    let mut x_u_map: Vec<Index> = Vec::new();
    let mut n_x_fixed: Index = 0;

    for i in 0..nx {
        let lo = x_l[i];
        let hi = x_u[i];
        if lo > hi {
            return Err(SolverException::new(
                ExceptionKind::INCONSISTENT_BOUNDS,
                format!(
                    "There are inconsistent bounds on variable {i}: lower = {lo:25.16e} \
                     and upper = {hi:25.16e}."
                ),
                file!(),
                line!() as Index,
            ));
        }
        if lo == hi {
            // `fixed_variable_treatment = make_parameter` (upstream
            // default): drop fixed vars from x_var entirely. Their
            // values are spliced back into the full-x array each time
            // we call into the user's TNLP (see
            // `OrigIpoptNlp::lift_x_to_full`).
            n_x_fixed += 1;
            x_fixed_map.push(i as Index);
            x_fixed_vals.push(lo);
            continue;
        }
        let var_idx = x_not_fixed_map.len() as Index;
        x_not_fixed_map.push(i as Index);
        full_to_var[i] = var_idx;
        if lo > lo_inf {
            x_l_map.push(var_idx);
        }
        if hi < up_inf {
            x_u_map.push(var_idx);
        }
    }

    // --- Constraints -------------------------------------------------
    let mut c_map: Vec<Index> = Vec::new();
    let mut d_map: Vec<Index> = Vec::new();
    let mut d_l_map: Vec<Index> = Vec::new();
    let mut d_u_map: Vec<Index> = Vec::new();

    for i in 0..ng {
        let lo = g_l[i];
        let hi = g_u[i];
        if lo == hi {
            c_map.push(i as Index);
        } else if lo > hi {
            return Err(SolverException::new(
                ExceptionKind::INCONSISTENT_BOUNDS,
                format!(
                    "There are inconsistent bounds on constraint function {i}: \
                     lower = {lo:25.16e} and upper = {hi:25.16e}."
                ),
                file!(),
                line!() as Index,
            ));
        } else {
            let d_idx = d_map.len() as Index;
            d_map.push(i as Index);
            if lo > lo_inf {
                d_l_map.push(d_idx);
            }
            if hi < up_inf {
                d_u_map.push(d_idx);
            }
        }
    }

    let n_c = c_map.len() as Index;
    let n_d = d_map.len() as Index;

    Ok(BoundClassification {
        n_full_x,
        n_full_g,
        n_x_fixed,
        x_not_fixed_map,
        x_fixed_map,
        x_fixed_vals,
        full_to_var,
        x_l_map,
        x_u_map,
        n_c,
        c_map,
        n_d,
        d_map,
        d_l_map,
        d_u_map,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tnlp::{IndexStyle, IpoptCq, IpoptData, Solution, SparsityRequest, StartingPoint};

    /// HS071: min x[0]*x[3]*(x[0]+x[1]+x[2]) + x[2]
    /// s.t.   x[0]*x[1]*x[2]*x[3] >= 25                (inequality)
    ///        x[0]^2 + x[1]^2 + x[2]^2 + x[3]^2 == 40  (equality)
    ///        1 <= x[i] <= 5
    struct Hs071;
    impl TNLP for Hs071 {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 4,
                m: 2,
                nnz_jac_g: 8,
                nnz_h_lag: 10,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l.copy_from_slice(&[1.0; 4]);
            b.x_u.copy_from_slice(&[5.0; 4]);
            // Constraint 0: 25 <= g_0 <= +inf  (inequality, finite lower only)
            // Constraint 1: 40 == g_1 == 40    (equality)
            b.g_l.copy_from_slice(&[25.0, 40.0]);
            b.g_u.copy_from_slice(&[2.0e19, 40.0]);
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x.copy_from_slice(&[1.0, 5.0, 5.0, 1.0]);
            true
        }
        fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
            Some(x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2])
        }
        fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g.fill(0.0);
            true
        }
        fn eval_g(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g.fill(0.0);
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            if let SparsityRequest::Structure { irow, jcol } = mode {
                irow.copy_from_slice(&[0, 0, 0, 0, 1, 1, 1, 1]);
                jcol.copy_from_slice(&[0, 1, 2, 3, 0, 1, 2, 3]);
            }
            true
        }
        fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
    }

    #[test]
    fn hs071_decomposes_to_one_eq_one_ineq() {
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Hs071));
        let adapter = TNLPAdapter::new(tnlp).unwrap();
        let c = adapter.classification();
        assert_eq!(c.n_full_x, 4);
        assert_eq!(c.n_full_g, 2);
        assert_eq!(c.n_x_fixed, 0);
        assert_eq!(c.n_x_var(), 4);
        assert!(c.x_fixed_map.is_empty());
        assert_eq!(c.full_to_var, vec![0, 1, 2, 3]);
        // All four variables have both finite bounds.
        assert_eq!(c.x_l_map, vec![0, 1, 2, 3]);
        assert_eq!(c.x_u_map, vec![0, 1, 2, 3]);
        // Constraint #0 is the inequality, #1 is the equality.
        assert_eq!(c.n_c, 1);
        assert_eq!(c.c_map, vec![1]);
        assert_eq!(c.n_d, 1);
        assert_eq!(c.d_map, vec![0]);
        // The single inequality has a finite lower bound (25) and an
        // infinite upper bound (2e19 == nlp_upper_bound_inf).
        assert_eq!(c.d_l_map, vec![0]);
        assert!(c.d_u_map.is_empty());
        assert_eq!(adapter.nlp_info().nnz_jac_g, 8);
    }

    /// Variant with one fixed variable (x[0] in [3,3]) and one free
    /// variable (x[1] in [-inf, +inf]) to exercise the bound-only and
    /// fixed paths.
    struct Mixed;
    impl TNLP for Mixed {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 3,
                m: 2,
                nnz_jac_g: 6,
                nnz_h_lag: 0,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            // x[0] fixed at 3, x[1] free, x[2] upper-only at 7.
            b.x_l.copy_from_slice(&[3.0, -2.0e19, -2.0e19]);
            b.x_u.copy_from_slice(&[3.0, 2.0e19, 7.0]);
            // g[0]: 0 <= ... <= 1 (two-sided ineq)
            // g[1]: free constraint (-inf, +inf) — still classified as ineq.
            b.g_l.copy_from_slice(&[0.0, -2.0e19]);
            b.g_u.copy_from_slice(&[1.0, 2.0e19]);
            true
        }
        fn get_starting_point(&mut self, _sp: StartingPoint<'_>) -> bool {
            true
        }
        fn eval_f(&mut self, _x: &[Number], _new_x: bool) -> Option<Number> {
            Some(0.0)
        }
        fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g.fill(0.0);
            true
        }
        fn eval_g(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g.fill(0.0);
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            _m: SparsityRequest<'_>,
        ) -> bool {
            true
        }
        fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
    }

    #[test]
    fn mixed_bounds_classifies_correctly() {
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Mixed));
        let adapter = TNLPAdapter::new(tnlp).unwrap();
        let c = adapter.classification();
        assert_eq!(c.n_full_x, 3);
        assert_eq!(c.n_x_fixed, 1);
        // x[0] fixed at 3 → removed from x_var (make_parameter).
        // x[1] free, x[2] upper-only → both in x_var.
        assert_eq!(c.n_x_var(), 2);
        assert_eq!(c.x_not_fixed_map, vec![1, 2]);
        assert_eq!(c.x_fixed_map, vec![0]);
        assert_eq!(c.x_fixed_vals, vec![3.0]);
        assert_eq!(c.full_to_var, vec![-1, 0, 1]);
        // After fixed-var removal, x[1] (now var idx 0) is fully free,
        // x[2] (now var idx 1) has only an upper bound.
        assert!(c.x_l_map.is_empty());
        assert_eq!(c.x_u_map, vec![1]);
        // No equalities; both constraints are classified as inequalities.
        assert_eq!(c.n_c, 0);
        assert_eq!(c.n_d, 2);
        assert_eq!(c.d_map, vec![0, 1]);
        // d[0] has finite lower (0) and finite upper (1).
        // d[1] is fully free — neither bound finite.
        assert_eq!(c.d_l_map, vec![0]);
        assert_eq!(c.d_u_map, vec![0]);
    }

    /// Inconsistent bounds (lo > hi) should error.
    struct Bad;
    impl TNLP for Bad {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 1,
                m: 0,
                nnz_jac_g: 0,
                nnz_h_lag: 0,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l[0] = 5.0;
            b.x_u[0] = 1.0;
            true
        }
        fn get_starting_point(&mut self, _sp: StartingPoint<'_>) -> bool {
            true
        }
        fn eval_f(&mut self, _x: &[Number], _new_x: bool) -> Option<Number> {
            Some(0.0)
        }
        fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g.fill(0.0);
            true
        }
        fn eval_g(&mut self, _x: &[Number], _new_x: bool, _g: &mut [Number]) -> bool {
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            _m: SparsityRequest<'_>,
        ) -> bool {
            true
        }
        fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
    }

    #[test]
    fn inconsistent_variable_bounds_is_rejected() {
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Bad));
        let err = TNLPAdapter::new(tnlp).unwrap_err();
        assert_eq!(err.kind, ExceptionKind::INCONSISTENT_BOUNDS);
    }
}

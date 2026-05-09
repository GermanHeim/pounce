//! POUNCE C ABI — port of `Interfaces/IpStdCInterface.{h,cpp}`.
//!
//! Provides the `IpoptCreate / IpoptSolve / IpoptFreeProblem` C entry
//! points that existing PyIpopt / cyipopt / JuMP wrappers link
//! against. Function names and signatures match upstream exactly so
//! consumers can swap `libipopt.{dylib,so}` for `libpounce_cinterface`
//! without rebuilding.
//!
//! Phase 11 ships:
//!
//! * the FFI surface,
//! * a heap-allocated `IpoptProblemInfo` carrying an `IpoptApplication`
//!   plus the user callback table and bounds,
//! * working `Add*Option` setters that forward to the application's
//!   `OptionsList`,
//! * working `FreeIpoptProblem` and `SetIntermediateCallback`,
//! * `IpoptSolve` stub (returns `InternalError`) until the algorithm
//!   side drives a TNLP-backed solve end-to-end.
//!
//! All entry points are `extern "C"` and `#[no_mangle]`. Pointers are
//! raw and the caller is responsible for lifetime; the `IpoptProblem`
//! handle is opaque (`*mut c_void` from C's perspective).

#![allow(non_camel_case_types, non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn, dead_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod fortran;

use pounce_algorithm::application::IpoptApplication;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use std::cell::RefCell;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::rc::Rc;

/// Mirrors C `Number` typedef in `IpStdCInterface.h`.
pub type Number = f64;
/// Mirrors C `Index`.
pub type Index = c_int;
/// Mirrors C `Bool`.
pub type Bool = c_int;

const TRUE: Bool = 1;
const FALSE: Bool = 0;

/// Internal owned state behind the opaque `IpoptProblem` handle.
/// `#[repr(C)]` is unnecessary because C only sees the pointer.
pub struct IpoptProblemInfo {
    app: IpoptApplication,
    n: Index,
    m: Index,
    nele_jac: Index,
    nele_hess: Index,
    index_style: Index,
    x_l: Vec<Number>,
    x_u: Vec<Number>,
    g_l: Vec<Number>,
    g_u: Vec<Number>,
    eval_f: Option<Eval_F_CB>,
    eval_g: Option<Eval_G_CB>,
    eval_grad_f: Option<Eval_Grad_F_CB>,
    eval_jac_g: Option<Eval_Jac_G_CB>,
    eval_h: Option<Eval_H_CB>,
    intermediate_cb: Option<Intermediate_CB>,
}

pub type IpoptProblem = *mut IpoptProblemInfo;

// User-callback function pointer types — match
// `IpStdCInterface.h:Eval_F_CB` etc. byte for byte.

pub type Eval_F_CB = unsafe extern "C" fn(
    n: Index,
    x: *const Number,
    new_x: Bool,
    obj_value: *mut Number,
    user_data: *mut c_void,
) -> Bool;

pub type Eval_Grad_F_CB = unsafe extern "C" fn(
    n: Index,
    x: *const Number,
    new_x: Bool,
    grad_f: *mut Number,
    user_data: *mut c_void,
) -> Bool;

pub type Eval_G_CB = unsafe extern "C" fn(
    n: Index,
    x: *const Number,
    new_x: Bool,
    m: Index,
    g: *mut Number,
    user_data: *mut c_void,
) -> Bool;

pub type Eval_Jac_G_CB = unsafe extern "C" fn(
    n: Index,
    x: *const Number,
    new_x: Bool,
    m: Index,
    nele_jac: Index,
    iRow: *mut Index,
    jCol: *mut Index,
    values: *mut Number,
    user_data: *mut c_void,
) -> Bool;

pub type Eval_H_CB = unsafe extern "C" fn(
    n: Index,
    x: *const Number,
    new_x: Bool,
    obj_factor: Number,
    m: Index,
    lambda: *const Number,
    new_lambda: Bool,
    nele_hess: Index,
    iRow: *mut Index,
    jCol: *mut Index,
    values: *mut Number,
    user_data: *mut c_void,
) -> Bool;

pub type Intermediate_CB = unsafe extern "C" fn(
    alg_mod: Index,
    iter_count: Index,
    obj_value: Number,
    inf_pr: Number,
    inf_du: Number,
    mu: Number,
    d_norm: Number,
    regularization_size: Number,
    alpha_du: Number,
    alpha_pr: Number,
    ls_trials: Index,
    user_data: *mut c_void,
) -> Bool;

/// Port of `IpStdCInterface.cpp:CreateIpoptProblem`. Returns NULL on
/// invalid arguments (negative n/m, missing required callbacks, NULL
/// bound pointers when the corresponding dimension is positive).
///
/// # Safety
///
/// `x_L`, `x_U` must be valid pointers to `n` `Number`s when `n > 0`.
/// `g_L`, `g_U` must be valid pointers to `m` `Number`s when `m > 0`.
/// The callback function pointers must be valid for the lifetime of
/// the returned [`IpoptProblem`].
#[no_mangle]
pub unsafe extern "C" fn CreateIpoptProblem(
    n: Index,
    x_L: *const Number,
    x_U: *const Number,
    m: Index,
    g_L: *const Number,
    g_U: *const Number,
    nele_jac: Index,
    nele_hess: Index,
    index_style: Index,
    eval_f: Option<Eval_F_CB>,
    eval_g: Option<Eval_G_CB>,
    eval_grad_f: Option<Eval_Grad_F_CB>,
    eval_jac_g: Option<Eval_Jac_G_CB>,
    eval_h: Option<Eval_H_CB>,
) -> IpoptProblem {
    if n < 0 || m < 0 || nele_jac < 0 || nele_hess < 0 {
        return std::ptr::null_mut();
    }
    if !(0..=1).contains(&index_style) {
        return std::ptr::null_mut();
    }
    if eval_f.is_none() || eval_grad_f.is_none() {
        return std::ptr::null_mut();
    }
    if m > 0 && (eval_g.is_none() || eval_jac_g.is_none()) {
        return std::ptr::null_mut();
    }
    if n > 0 && (x_L.is_null() || x_U.is_null()) {
        return std::ptr::null_mut();
    }
    if m > 0 && (g_L.is_null() || g_U.is_null()) {
        return std::ptr::null_mut();
    }

    let x_l = if n > 0 {
        std::slice::from_raw_parts(x_L, n as usize).to_vec()
    } else {
        Vec::new()
    };
    let x_u = if n > 0 {
        std::slice::from_raw_parts(x_U, n as usize).to_vec()
    } else {
        Vec::new()
    };
    let g_l_vec = if m > 0 {
        std::slice::from_raw_parts(g_L, m as usize).to_vec()
    } else {
        Vec::new()
    };
    let g_u_vec = if m > 0 {
        std::slice::from_raw_parts(g_U, m as usize).to_vec()
    } else {
        Vec::new()
    };

    let info = Box::new(IpoptProblemInfo {
        app: IpoptApplication::new(),
        n,
        m,
        nele_jac,
        nele_hess,
        index_style,
        x_l,
        x_u,
        g_l: g_l_vec,
        g_u: g_u_vec,
        eval_f,
        eval_g,
        eval_grad_f,
        eval_jac_g,
        eval_h,
        intermediate_cb: None,
    });
    Box::into_raw(info)
}

/// Port of `IpStdCInterface.cpp:FreeIpoptProblem`.
///
/// # Safety
///
/// `ipopt_problem` must be a pointer previously returned by
/// [`CreateIpoptProblem`] and not yet freed, or NULL.
#[no_mangle]
pub unsafe extern "C" fn FreeIpoptProblem(ipopt_problem: IpoptProblem) {
    if ipopt_problem.is_null() {
        return;
    }
    drop(Box::from_raw(ipopt_problem));
}

unsafe fn keyword_str<'a>(keyword: *const c_char) -> Option<&'a str> {
    if keyword.is_null() {
        return None;
    }
    CStr::from_ptr(keyword).to_str().ok()
}

/// Port of `IpStdCInterface.cpp:AddIpoptStrOption`.
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem`. `keyword` and `val`
/// must be valid NUL-terminated strings.
#[no_mangle]
pub unsafe extern "C" fn AddIpoptStrOption(
    ipopt_problem: IpoptProblem,
    keyword: *const c_char,
    val: *const c_char,
) -> Bool {
    if ipopt_problem.is_null() {
        return FALSE;
    }
    let info = &mut *ipopt_problem;
    let Some(k) = keyword_str(keyword) else {
        return FALSE;
    };
    if val.is_null() {
        return FALSE;
    }
    let Ok(v) = CStr::from_ptr(val).to_str() else {
        return FALSE;
    };
    match info.app.options_mut().set_string_value(k, v, true, false) {
        Ok(_) => TRUE,
        Err(_) => FALSE,
    }
}

/// Port of `AddIpoptNumOption`.
///
/// # Safety
///
/// `keyword` must be a valid NUL-terminated string and
/// `ipopt_problem` must be a valid `IpoptProblem`.
#[no_mangle]
pub unsafe extern "C" fn AddIpoptNumOption(
    ipopt_problem: IpoptProblem,
    keyword: *const c_char,
    val: Number,
) -> Bool {
    if ipopt_problem.is_null() {
        return FALSE;
    }
    let info = &mut *ipopt_problem;
    let Some(k) = keyword_str(keyword) else {
        return FALSE;
    };
    match info.app.options_mut().set_numeric_value(k, val, true, false) {
        Ok(_) => TRUE,
        Err(_) => FALSE,
    }
}

/// Port of `AddIpoptIntOption`.
///
/// # Safety
///
/// `keyword` must be a valid NUL-terminated string and
/// `ipopt_problem` must be a valid `IpoptProblem`.
#[no_mangle]
pub unsafe extern "C" fn AddIpoptIntOption(
    ipopt_problem: IpoptProblem,
    keyword: *const c_char,
    val: Index,
) -> Bool {
    if ipopt_problem.is_null() {
        return FALSE;
    }
    let info = &mut *ipopt_problem;
    let Some(k) = keyword_str(keyword) else {
        return FALSE;
    };
    match info
        .app
        .options_mut()
        .set_integer_value(k, val as pounce_common::types::Index, true, false)
    {
        Ok(_) => TRUE,
        Err(_) => FALSE,
    }
}

/// Port of `IpStdCInterface.cpp:IpoptSolve`. Returns the
/// `ApplicationReturnStatus` integer.
///
/// Builds a [`CCallbackTnlp`] from the user-supplied callback table
/// and bounds, runs it through [`IpoptApplication::optimize_tnlp`],
/// and writes back the final iterate.
///
/// # Safety
///
/// All pointer arguments are read/written per the
/// `IpStdCInterface.h` contract: `x` is in/out (size `n`); `g`,
/// `mult_g`, `mult_x_L`, `mult_x_U` are out-only (sizes `m, m, n, n`)
/// and may be NULL when the corresponding output is not desired.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn IpoptSolve(
    ipopt_problem: IpoptProblem,
    x: *mut Number,
    g: *mut Number,
    obj_val: *mut Number,
    mult_g: *mut Number,
    mult_x_L: *mut Number,
    mult_x_U: *mut Number,
    user_data: *mut c_void,
) -> Index {
    if ipopt_problem.is_null() {
        return ApplicationReturnStatus::InternalError as Index;
    }
    let info = &mut *ipopt_problem;
    if info.n < 0 || info.m < 0 {
        return ApplicationReturnStatus::InvalidProblemDefinition as Index;
    }
    if info.n > 0 && x.is_null() {
        return ApplicationReturnStatus::InvalidProblemDefinition as Index;
    }

    let n_us = info.n as usize;
    let m_us = info.m as usize;
    let initial_x = if n_us > 0 {
        std::slice::from_raw_parts(x, n_us).to_vec()
    } else {
        Vec::new()
    };

    let bridge = Rc::new(RefCell::new(CCallbackTnlp {
        n: info.n,
        m: info.m,
        nele_jac: info.nele_jac,
        nele_hess: info.nele_hess,
        index_style: info.index_style,
        x_l: info.x_l.clone(),
        x_u: info.x_u.clone(),
        g_l: info.g_l.clone(),
        g_u: info.g_u.clone(),
        initial_x,
        eval_f: info.eval_f,
        eval_grad_f: info.eval_grad_f,
        eval_g: info.eval_g,
        eval_jac_g: info.eval_jac_g,
        eval_h: info.eval_h,
        user_data,
        final_status: None,
        final_x: vec![0.0; n_us],
        final_z_l: vec![0.0; n_us],
        final_z_u: vec![0.0; n_us],
        final_g: vec![0.0; m_us],
        final_lambda: vec![0.0; m_us],
        final_obj: 0.0,
    }));

    let bridge_for_solve: Rc<RefCell<dyn TNLP>> = bridge.clone();
    let status = info.app.optimize_tnlp(bridge_for_solve);

    let bridge_ref = bridge.borrow();
    if !x.is_null() && n_us > 0 {
        std::ptr::copy_nonoverlapping(bridge_ref.final_x.as_ptr(), x, n_us);
    }
    if !g.is_null() && m_us > 0 {
        std::ptr::copy_nonoverlapping(bridge_ref.final_g.as_ptr(), g, m_us);
    }
    if !obj_val.is_null() {
        *obj_val = bridge_ref.final_obj;
    }
    if !mult_g.is_null() && m_us > 0 {
        std::ptr::copy_nonoverlapping(bridge_ref.final_lambda.as_ptr(), mult_g, m_us);
    }
    if !mult_x_L.is_null() && n_us > 0 {
        std::ptr::copy_nonoverlapping(bridge_ref.final_z_l.as_ptr(), mult_x_L, n_us);
    }
    if !mult_x_U.is_null() && n_us > 0 {
        std::ptr::copy_nonoverlapping(bridge_ref.final_z_u.as_ptr(), mult_x_U, n_us);
    }
    status as Index
}

/// Port of `SetIntermediateCallback`.
///
/// # Safety
///
/// `ipopt_problem` must be valid.
#[no_mangle]
pub unsafe extern "C" fn SetIntermediateCallback(
    ipopt_problem: IpoptProblem,
    intermediate_cb: Option<Intermediate_CB>,
) -> Bool {
    if ipopt_problem.is_null() {
        return FALSE;
    }
    let info = &mut *ipopt_problem;
    info.intermediate_cb = intermediate_cb;
    TRUE
}

/// Adapter that bridges the user-supplied C callback table to the
/// in-crate [`TNLP`] trait. Mirrors `Interfaces/IpStdInterfaceTNLP.cpp`
/// (`StdInterfaceTNLP`); each TNLP method forwards to the matching
/// `Eval_*_CB` and propagates `false` returns up so the algorithm
/// layer can map them to `Invalid_Number_Detected`.
///
/// Holds a snapshot of bounds and the initial `x`. After `optimize_tnlp`
/// finishes, `finalize_solution` is called by the algorithm layer; the
/// adapter records the final iterate in `final_*` fields, which the
/// outer [`IpoptSolve`] copies back into the caller's buffers.
struct CCallbackTnlp {
    n: Index,
    m: Index,
    nele_jac: Index,
    nele_hess: Index,
    index_style: Index,
    x_l: Vec<Number>,
    x_u: Vec<Number>,
    g_l: Vec<Number>,
    g_u: Vec<Number>,
    initial_x: Vec<Number>,
    eval_f: Option<Eval_F_CB>,
    eval_grad_f: Option<Eval_Grad_F_CB>,
    eval_g: Option<Eval_G_CB>,
    eval_jac_g: Option<Eval_Jac_G_CB>,
    eval_h: Option<Eval_H_CB>,
    user_data: *mut c_void,
    final_status: Option<pounce_nlp::alg_types::SolverReturn>,
    final_x: Vec<Number>,
    final_z_l: Vec<Number>,
    final_z_u: Vec<Number>,
    final_g: Vec<Number>,
    final_lambda: Vec<Number>,
    final_obj: Number,
}

impl TNLP for CCallbackTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.n as pounce_common::types::Index,
            m: self.m as pounce_common::types::Index,
            nnz_jac_g: self.nele_jac as pounce_common::types::Index,
            nnz_h_lag: self.nele_hess as pounce_common::types::Index,
            index_style: if self.index_style == 1 {
                IndexStyle::Fortran
            } else {
                IndexStyle::C
            },
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        if !self.x_l.is_empty() {
            b.x_l.copy_from_slice(&self.x_l);
        }
        if !self.x_u.is_empty() {
            b.x_u.copy_from_slice(&self.x_u);
        }
        if !self.g_l.is_empty() {
            b.g_l.copy_from_slice(&self.g_l);
        }
        if !self.g_u.is_empty() {
            b.g_u.copy_from_slice(&self.g_u);
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        if !self.initial_x.is_empty() {
            sp.x.copy_from_slice(&self.initial_x);
        }
        true
    }

    fn eval_f(&mut self, x: &[Number], new_x: bool) -> Option<Number> {
        let cb = self.eval_f?;
        let mut obj = 0.0;
        let ok = unsafe {
            cb(
                self.n,
                x.as_ptr() as *mut Number,
                if new_x { TRUE } else { FALSE },
                &mut obj,
                self.user_data,
            )
        };
        if ok != FALSE {
            Some(obj)
        } else {
            None
        }
    }

    fn eval_grad_f(&mut self, x: &[Number], new_x: bool, grad_f: &mut [Number]) -> bool {
        let Some(cb) = self.eval_grad_f else {
            return false;
        };
        let ok = unsafe {
            cb(
                self.n,
                x.as_ptr() as *mut Number,
                if new_x { TRUE } else { FALSE },
                grad_f.as_mut_ptr(),
                self.user_data,
            )
        };
        ok != FALSE
    }

    fn eval_g(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
        if self.m == 0 {
            return true;
        }
        let Some(cb) = self.eval_g else {
            return false;
        };
        let ok = unsafe {
            cb(
                self.n,
                x.as_ptr() as *mut Number,
                if new_x { TRUE } else { FALSE },
                self.m,
                g.as_mut_ptr(),
                self.user_data,
            )
        };
        ok != FALSE
    }

    fn eval_jac_g(
        &mut self,
        x: Option<&[Number]>,
        new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        if self.m == 0 || self.nele_jac == 0 {
            return true;
        }
        let Some(cb) = self.eval_jac_g else {
            return false;
        };
        let x_ptr = x.map(|s| s.as_ptr() as *mut Number).unwrap_or(std::ptr::null_mut());
        let ok = match mode {
            SparsityRequest::Structure { irow, jcol } => unsafe {
                cb(
                    self.n,
                    x_ptr,
                    if new_x { TRUE } else { FALSE },
                    self.m,
                    self.nele_jac,
                    irow.as_mut_ptr(),
                    jcol.as_mut_ptr(),
                    std::ptr::null_mut(),
                    self.user_data,
                )
            },
            SparsityRequest::Values { values } => unsafe {
                cb(
                    self.n,
                    x_ptr,
                    if new_x { TRUE } else { FALSE },
                    self.m,
                    self.nele_jac,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    values.as_mut_ptr(),
                    self.user_data,
                )
            },
        };
        ok != FALSE
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
        let Some(cb) = self.eval_h else {
            return false;
        };
        if self.nele_hess == 0 {
            return true;
        }
        let x_ptr = x.map(|s| s.as_ptr() as *mut Number).unwrap_or(std::ptr::null_mut());
        let lambda_ptr = lambda
            .map(|s| s.as_ptr() as *mut Number)
            .unwrap_or(std::ptr::null_mut());
        let ok = match mode {
            SparsityRequest::Structure { irow, jcol } => unsafe {
                cb(
                    self.n,
                    x_ptr,
                    if new_x { TRUE } else { FALSE },
                    obj_factor,
                    self.m,
                    lambda_ptr,
                    if new_lambda { TRUE } else { FALSE },
                    self.nele_hess,
                    irow.as_mut_ptr(),
                    jcol.as_mut_ptr(),
                    std::ptr::null_mut(),
                    self.user_data,
                )
            },
            SparsityRequest::Values { values } => unsafe {
                cb(
                    self.n,
                    x_ptr,
                    if new_x { TRUE } else { FALSE },
                    obj_factor,
                    self.m,
                    lambda_ptr,
                    if new_lambda { TRUE } else { FALSE },
                    self.nele_hess,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    values.as_mut_ptr(),
                    self.user_data,
                )
            },
        };
        ok != FALSE
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.final_status = Some(sol.status);
        if !sol.x.is_empty() {
            self.final_x.copy_from_slice(sol.x);
        }
        if !sol.z_l.is_empty() {
            self.final_z_l.copy_from_slice(sol.z_l);
        }
        if !sol.z_u.is_empty() {
            self.final_z_u.copy_from_slice(sol.z_u);
        }
        if !sol.g.is_empty() {
            self.final_g.copy_from_slice(sol.g);
        }
        if !sol.lambda.is_empty() {
            self.final_lambda.copy_from_slice(sol.lambda);
        }
        self.final_obj = sol.obj_value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    unsafe extern "C" fn dummy_eval_f(
        _n: Index,
        _x: *const Number,
        _new_x: Bool,
        _obj_value: *mut Number,
        _user_data: *mut c_void,
    ) -> Bool {
        TRUE
    }
    unsafe extern "C" fn dummy_eval_grad_f(
        _n: Index,
        _x: *const Number,
        _new_x: Bool,
        _grad_f: *mut Number,
        _user_data: *mut c_void,
    ) -> Bool {
        TRUE
    }

    fn create_unconstrained() -> IpoptProblem {
        let xl = [-1.0; 4];
        let xu = [1.0; 4];
        unsafe {
            CreateIpoptProblem(
                4,
                xl.as_ptr(),
                xu.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                0,
                10,
                0,
                Some(dummy_eval_f),
                None,
                Some(dummy_eval_grad_f),
                None,
                None,
            )
        }
    }

    #[test]
    fn create_succeeds_for_unconstrained_problem() {
        let p = create_unconstrained();
        assert!(!p.is_null());
        unsafe { FreeIpoptProblem(p) };
    }

    #[test]
    fn create_returns_null_on_missing_required_callbacks() {
        let xl = [-1.0; 4];
        let xu = [1.0; 4];
        let p = unsafe {
            CreateIpoptProblem(
                4,
                xl.as_ptr(),
                xu.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                0,
                10,
                0,
                None, // missing eval_f
                None,
                Some(dummy_eval_grad_f),
                None,
                None,
            )
        };
        assert!(p.is_null());
    }

    #[test]
    fn create_returns_null_on_negative_n() {
        let p = unsafe {
            CreateIpoptProblem(
                -1,
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                0,
                10,
                0,
                Some(dummy_eval_f),
                None,
                Some(dummy_eval_grad_f),
                None,
                None,
            )
        };
        assert!(p.is_null());
    }

    #[test]
    fn create_returns_null_on_invalid_index_style() {
        let xl = [0.0; 1];
        let xu = [1.0; 1];
        let p = unsafe {
            CreateIpoptProblem(
                1,
                xl.as_ptr(),
                xu.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                0,
                1,
                2, // valid values are 0 and 1
                Some(dummy_eval_f),
                None,
                Some(dummy_eval_grad_f),
                None,
                None,
            )
        };
        assert!(p.is_null());
    }

    #[test]
    fn add_int_option_forwards_to_application() {
        let p = create_unconstrained();
        let key = CString::new("print_level").unwrap();
        let ok = unsafe { AddIpoptIntOption(p, key.as_ptr(), 5) };
        assert_eq!(ok, TRUE);
        let info = unsafe { &*p };
        let (level, found) = info
            .app
            .options()
            .get_integer_value("print_level", "")
            .unwrap();
        assert!(found);
        assert_eq!(level, 5);
        unsafe { FreeIpoptProblem(p) };
    }

    #[test]
    fn add_str_option_with_invalid_key_returns_false() {
        let p = create_unconstrained();
        let key = CString::new("totally_unknown_option").unwrap();
        let val = CString::new("yes").unwrap();
        let ok = unsafe { AddIpoptStrOption(p, key.as_ptr(), val.as_ptr()) };
        assert_eq!(ok, FALSE);
        unsafe { FreeIpoptProblem(p) };
    }

    #[test]
    fn add_options_on_null_problem_returns_false() {
        let key = CString::new("print_level").unwrap();
        let v = CString::new("yes").unwrap();
        unsafe {
            assert_eq!(
                AddIpoptIntOption(std::ptr::null_mut(), key.as_ptr(), 5),
                FALSE
            );
            assert_eq!(
                AddIpoptNumOption(std::ptr::null_mut(), key.as_ptr(), 1.0),
                FALSE
            );
            assert_eq!(
                AddIpoptStrOption(std::ptr::null_mut(), key.as_ptr(), v.as_ptr()),
                FALSE
            );
        }
    }

    unsafe extern "C" fn dummy_intermediate(
        _alg_mod: Index,
        _iter_count: Index,
        _obj_value: Number,
        _inf_pr: Number,
        _inf_du: Number,
        _mu: Number,
        _d_norm: Number,
        _regularization_size: Number,
        _alpha_du: Number,
        _alpha_pr: Number,
        _ls_trials: Index,
        _user_data: *mut c_void,
    ) -> Bool {
        TRUE
    }

    #[test]
    fn set_intermediate_callback_stores_pointer() {
        let p = create_unconstrained();
        let ok = unsafe { SetIntermediateCallback(p, Some(dummy_intermediate)) };
        assert_eq!(ok, TRUE);
        let info = unsafe { &*p };
        assert!(info.intermediate_cb.is_some());
        unsafe { FreeIpoptProblem(p) };
    }

    #[test]
    fn solve_returns_internal_error_on_null_problem() {
        let rc = unsafe {
            IpoptSolve(
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, -199);
    }

    #[test]
    fn free_null_is_safe() {
        unsafe { FreeIpoptProblem(std::ptr::null_mut()) };
    }

    // ---- End-to-end bridge: 1-D unconstrained quadratic ----
    //
    // f(x) = (x - 2)^2, no bounds, no constraints. Newton driver
    // converges in one step.

    unsafe extern "C" fn quad_eval_f(
        _n: Index,
        x: *const Number,
        _new_x: Bool,
        obj_value: *mut Number,
        _user_data: *mut c_void,
    ) -> Bool {
        let v = *x.offset(0);
        *obj_value = (v - 2.0) * (v - 2.0);
        TRUE
    }
    unsafe extern "C" fn quad_eval_grad_f(
        _n: Index,
        x: *const Number,
        _new_x: Bool,
        grad: *mut Number,
        _user_data: *mut c_void,
    ) -> Bool {
        let v = *x.offset(0);
        *grad.offset(0) = 2.0 * (v - 2.0);
        TRUE
    }
    unsafe extern "C" fn quad_eval_h(
        _n: Index,
        _x: *const Number,
        _new_x: Bool,
        obj_factor: Number,
        _m: Index,
        _lambda: *const Number,
        _new_lambda: Bool,
        _nele_hess: Index,
        irow: *mut Index,
        jcol: *mut Index,
        values: *mut Number,
        _user_data: *mut c_void,
    ) -> Bool {
        if !irow.is_null() && !jcol.is_null() && values.is_null() {
            *irow.offset(0) = 0;
            *jcol.offset(0) = 0;
        } else if irow.is_null() && jcol.is_null() && !values.is_null() {
            *values.offset(0) = 2.0 * obj_factor;
        } else {
            return FALSE;
        }
        TRUE
    }

    #[test]
    fn solve_drives_unconstrained_quadratic_through_bridge() {
        // Bounds wide open (kappa1 push won't move us off 0.0 since
        // |0| < 1e19, but the Newton step lands us at 2.0 anyway).
        let xl = [-1.0e20];
        let xu = [1.0e20];
        let p = unsafe {
            CreateIpoptProblem(
                1,
                xl.as_ptr(),
                xu.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                0,
                1,
                0,
                Some(quad_eval_f),
                None,
                Some(quad_eval_grad_f),
                None,
                Some(quad_eval_h),
            )
        };
        assert!(!p.is_null());
        let mut x = [0.0_f64];
        let mut obj = 0.0_f64;
        let rc = unsafe {
            IpoptSolve(
                p,
                x.as_mut_ptr(),
                std::ptr::null_mut(),
                &mut obj,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, ApplicationReturnStatus::SolveSucceeded as Index);
        assert!((x[0] - 2.0).abs() < 1e-6, "x[0] = {}", x[0]);
        assert!(obj.abs() < 1e-10, "obj = {}", obj);
        unsafe { FreeIpoptProblem(p) };
    }

    #[test]
    fn solve_invalid_problem_definition_when_x_null() {
        let p = create_unconstrained();
        let rc = unsafe {
            IpoptSolve(
                p,
                std::ptr::null_mut(), // x null but n > 0
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, ApplicationReturnStatus::InvalidProblemDefinition as Index);
        unsafe { FreeIpoptProblem(p) };
    }
}

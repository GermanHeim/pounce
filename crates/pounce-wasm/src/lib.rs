//! WebAssembly entry points for POUNCE.
//!
//! POUNCE's solver core is pure Rust with no C/Fortran dependency on the
//! default build, so the whole `.nl` → AD tape → interior-point pipeline
//! compiles to `wasm32-wasip1` unchanged. This crate is the thin C-ABI
//! shim a browser page talks to: hand it the bytes of an AMPL `.nl` file,
//! get back a JSON problem summary, then ask it to solve.
//!
//! # ABI
//!
//! Everything crosses the boundary as UTF-8 in linear memory, because that
//! is all the raw `WebAssembly` API can express without a bindings
//! generator (no `wasm-bindgen` dependency — see `web/README.md`):
//!
//! * [`pounce_alloc`] / [`pounce_dealloc`] — let JS place input bytes in
//!   wasm memory.
//! * [`pounce_load`] — parse a `.nl` (plus optional `.col` / `.row` name
//!   files), keep the built [`NlTnlp`] in a per-instance slot, and return
//!   a JSON summary. Returns `{"error": …}` on a bad file.
//! * [`pounce_solve`] — solve the loaded problem with an `ipopt.opt`-style
//!   options string, returning a JSON result.
//! * [`pounce_free_string`] — release a returned string.
//!
//! Returned strings are NUL-terminated so the caller can find their length
//! without a second call. Every entry point catches panics, so a malformed
//! model surfaces as a JSON error rather than a trapped instance the page
//! would have to rebuild.
//!
//! The solver's own console output (banner, iteration table, exit line) is
//! written to stdout as usual; under WASI the host shim receives it through
//! `fd_write`, which is how the demo page streams the live iteration log.

use std::cell::RefCell;
use std::ffi::{CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_nl::nl_reader::{NlProblem, NlTnlp, parse_nl_text};
use pounce_nlp::expression_provider::ExpressionProvider;
use pounce_nlp::tnlp::{Linearity, TNLP};

/// Bound on how many per-variable / per-constraint entries a JSON payload
/// carries. A million-variable model would otherwise serialize a JSON array
/// the page can neither render nor afford to parse; the arrays are for
/// display, and the summary reports the true counts separately.
const PREVIEW_LIMIT: usize = 2000;

thread_local! {
    /// The currently loaded model. A wasm instance drives one model at a
    /// time (the demo runs one instance per worker), so a single slot is
    /// enough and keeps the ABI handle-free.
    static LOADED: RefCell<Option<Rc<RefCell<NlTnlp>>>> = const { RefCell::new(None) };
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

/// Allocate `len` bytes in wasm linear memory for the caller to write into.
/// Returns null when `len` is 0 or the allocation fails.
///
/// # Safety
/// The returned pointer must be released with [`pounce_dealloc`] using the
/// same `len`, or handed to [`pounce_load`], which takes no ownership.
#[unsafe(no_mangle)]
pub extern "C" fn pounce_alloc(len: usize) -> *mut u8 {
    if len == 0 {
        return std::ptr::null_mut();
    }
    let mut buf = Vec::<u8>::new();
    if buf.try_reserve_exact(len).is_err() {
        return std::ptr::null_mut();
    }
    buf.resize(len, 0);
    let mut buf = std::mem::ManuallyDrop::new(buf);
    buf.as_mut_ptr()
}

/// Release a buffer obtained from [`pounce_alloc`].
///
/// # Safety
/// `ptr`/`len` must come from a single [`pounce_alloc`] call and must not
/// have been freed already.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pounce_dealloc(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    // SAFETY: the caller guarantees ptr/len came from `pounce_alloc`, which
    // built the allocation with exactly this length and capacity.
    drop(unsafe { Vec::from_raw_parts(ptr, len, len) });
}

/// Release a string returned by [`pounce_load`] / [`pounce_solve`].
///
/// # Safety
/// `ptr` must be a pointer this module returned and not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pounce_free_string(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: the caller guarantees this pointer came from `CString::into_raw`
    // in `to_c_string` below.
    drop(unsafe { CString::from_raw(ptr) });
}

/// Move a Rust string into a NUL-terminated allocation the caller owns.
/// Interior NULs are impossible here (all payloads are `serde_json` output),
/// but the fallback keeps the function total rather than panicking.
fn to_c_string(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(c) => c.into_raw(),
        Err(_) => match CString::new(r#"{"error":"internal: NUL in payload"}"#) {
            Ok(c) => c.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
    }
}

fn error_json(msg: impl std::fmt::Display) -> *mut c_char {
    to_c_string(serde_json::json!({ "error": msg.to_string() }).to_string())
}

/// Borrow `len` bytes at `ptr` as `&str`. Empty when `ptr` is null or `len`
/// is 0, so optional inputs can be passed as `(0, 0)`.
///
/// # Safety
/// `ptr`/`len` must describe an initialized, readable region that stays
/// valid for the call.
unsafe fn str_from_parts<'a>(ptr: *const u8, len: usize) -> Result<&'a str, String> {
    if ptr.is_null() || len == 0 {
        return Ok("");
    }
    // SAFETY: the caller guarantees the region is valid and initialized.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    std::str::from_utf8(bytes).map_err(|e| format!("input is not valid UTF-8: {e}"))
}

/// Run `f`, converting a panic into a JSON error string. `.nl` input is
/// arbitrary user data; a panic inside the parser or the solver must not
/// poison the wasm instance for the rest of the page's lifetime.
fn guarded(what: &str, f: impl FnOnce() -> *mut c_char) -> *mut c_char {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(p) => p,
        Err(_) => error_json(format!("{what} panicked (see the console log for details)")),
    }
}

// ---------------------------------------------------------------------------
// Load + summarize
// ---------------------------------------------------------------------------

/// Parse a `.nl` file and report what is in it.
///
/// `col_*` / `row_*` are the optional sibling `.col` / `.row` name files
/// AMPL writes under `option auxfiles rc;`; pass `(null, 0)` when absent.
/// The parsed model is retained for a following [`pounce_solve`].
///
/// # Safety
/// Each pointer/length pair must describe a readable region valid for the
/// call, or be `(null, 0)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pounce_load(
    nl_ptr: *const u8,
    nl_len: usize,
    col_ptr: *const u8,
    col_len: usize,
    row_ptr: *const u8,
    row_len: usize,
) -> *mut c_char {
    guarded("load", || {
        // SAFETY: forwarded from this function's own safety contract.
        let (nl, col, row) = unsafe {
            (
                str_from_parts(nl_ptr, nl_len),
                str_from_parts(col_ptr, col_len),
                str_from_parts(row_ptr, row_len),
            )
        };
        let (nl, col, row) = match (nl, col, row) {
            (Ok(a), Ok(b), Ok(c)) => (a, b, c),
            (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => return error_json(e),
        };
        if nl.trim().is_empty() {
            return error_json("empty .nl input");
        }

        let mut prob = match parse_nl_text(nl) {
            Ok(p) => p,
            Err(e) => return error_json(format!("could not parse .nl file: {e}")),
        };
        attach_names(&mut prob, col, row);

        let tnlp = match NlTnlp::try_new(prob) {
            Ok(t) => t,
            Err(e) => return error_json(e),
        };
        let tnlp = Rc::new(RefCell::new(tnlp));
        let summary = summarize(&mut tnlp.borrow_mut());
        LOADED.with(|slot| *slot.borrow_mut() = Some(Rc::clone(&tnlp)));
        to_c_string(summary.to_string())
    })
}

/// Scatter `.col` / `.row` line-per-name text onto the parsed problem, the
/// same way [`pounce_nl::nl_reader::read_nl_file`] does for files on disk.
/// A name file of the wrong length is ignored rather than mislabeling rows.
fn attach_names(prob: &mut NlProblem, col: &str, row: &str) {
    let lines = |txt: &str| -> Vec<String> {
        txt.lines()
            .map(str::trim_end)
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect()
    };
    let var_names = lines(col);
    if var_names.len() == prob.n {
        prob.var_names = var_names;
    }
    let con_names = lines(row);
    if con_names.len() == prob.m {
        prob.con_names = con_names;
    }
}

/// Classification of a `[lo, hi]` pair, shared by the variable and
/// constraint tallies. `INF` here is AMPL's 1e19 sentinel convention.
const INF: f64 = 1.0e19;

#[derive(Default)]
struct BoundTally {
    free: usize,
    lower_only: usize,
    upper_only: usize,
    boxed: usize,
    fixed: usize,
}

impl BoundTally {
    fn count(lo: &[f64], hi: &[f64]) -> Self {
        let mut t = Self::default();
        for (l, u) in lo.iter().zip(hi.iter()) {
            let has_l = *l > -INF;
            let has_u = *u < INF;
            match (has_l, has_u) {
                (false, false) => t.free += 1,
                (true, false) => t.lower_only += 1,
                (false, true) => t.upper_only += 1,
                (true, true) if l == u => t.fixed += 1,
                (true, true) => t.boxed += 1,
            }
        }
        t
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "free": self.free,
            "lower_only": self.lower_only,
            "upper_only": self.upper_only,
            "boxed": self.boxed,
            "fixed": self.fixed,
        })
    }
}

fn preview<T: Clone>(v: &[T]) -> (&[T], bool) {
    if v.len() > PREVIEW_LIMIT {
        (&v[..PREVIEW_LIMIT], true)
    } else {
        (v, false)
    }
}

/// Build the JSON problem summary shown before a solve: sizes, sparsity,
/// how the bounds break down, and how much of the model is nonlinear.
fn summarize(tnlp: &mut NlTnlp) -> serde_json::Value {
    let info = tnlp.get_nlp_info();
    let (n, m, nnz_jac, nnz_hess) = match info {
        Some(i) => (
            i.n as usize,
            i.m as usize,
            i.nnz_jac_g as usize,
            i.nnz_h_lag as usize,
        ),
        None => (0, 0, 0, 0),
    };

    let mut var_lin = vec![Linearity::Linear; n];
    let n_nonlinear_vars = if tnlp.get_variables_linearity(&mut var_lin) {
        var_lin
            .iter()
            .filter(|l| **l == Linearity::NonLinear)
            .count()
    } else {
        0
    };
    let mut con_lin = vec![Linearity::Linear; m];
    let n_nonlinear_cons = if tnlp.get_constraints_linearity(&mut con_lin) {
        con_lin
            .iter()
            .filter(|l| **l == Linearity::NonLinear)
            .count()
    } else {
        0
    };

    let prob = tnlp.problem();
    let var_bounds = BoundTally::count(&prob.x_l, &prob.x_u);
    let con_bounds = BoundTally::count(&prob.g_l, &prob.g_u);
    // A constraint whose bounds coincide is an equality; the rest of the
    // tally reads as inequalities (one-sided or ranged).
    let n_equality = con_bounds.fixed;

    let (var_names, var_names_truncated) = preview(&prob.var_names);
    let (con_names, con_names_truncated) = preview(&prob.con_names);
    let (x0, x0_truncated) = preview(&prob.x0);

    serde_json::json!({
        "n_vars": n,
        "n_cons": m,
        "n_objs": prob.num_obj,
        "sense": if prob.minimize { "minimize" } else { "maximize" },
        "nnz_jac": nnz_jac,
        "nnz_hess": nnz_hess,
        "jac_density": if n * m > 0 { nnz_jac as f64 / (n as f64 * m as f64) } else { 0.0 },
        "n_nonlinear_vars": n_nonlinear_vars,
        "n_nonlinear_cons": n_nonlinear_cons,
        "n_equality_cons": n_equality,
        "n_inequality_cons": m - n_equality,
        "degrees_of_freedom": n as i64 - n_equality as i64,
        "var_bounds": var_bounds.to_json(),
        "con_bounds": con_bounds.to_json(),
        "external_funcs": prob.imported_funcs.iter().map(|f| f.name.clone()).collect::<Vec<_>>(),
        "var_names": var_names,
        "con_names": con_names,
        "x0": x0,
        "truncated": var_names_truncated || con_names_truncated || x0_truncated,
        "preview_limit": PREVIEW_LIMIT,
    })
}

// ---------------------------------------------------------------------------
// Solve
// ---------------------------------------------------------------------------

/// Solve the model most recently loaded by [`pounce_load`].
///
/// `opts_*` is `ipopt.opt`-style text (`name value` per line, `#` comments)
/// — the same option names the CLI and the Python API take. Pass
/// `(null, 0)` for defaults.
///
/// # Safety
/// `opts_ptr`/`opts_len` must describe a readable region valid for the
/// call, or be `(null, 0)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pounce_solve(opts_ptr: *const u8, opts_len: usize) -> *mut c_char {
    guarded("solve", || {
        // SAFETY: forwarded from this function's own safety contract.
        let opts = match unsafe { str_from_parts(opts_ptr, opts_len) } {
            Ok(s) => s,
            Err(e) => return error_json(e),
        };
        let Some(tnlp) = LOADED.with(|slot| slot.borrow().clone()) else {
            return error_json("no model loaded — call pounce_load first");
        };
        to_c_string(solve_loaded(tnlp, opts).to_string())
    })
}

fn solve_loaded(tnlp: Rc<RefCell<NlTnlp>>, opts: &str) -> serde_json::Value {
    let mut app = IpoptApplication::new();
    if let Err(e) = app.initialize_with_options_str(opts) {
        return serde_json::json!({ "error": format!("bad options: {}", e.message) });
    }

    // Wrap in the auxiliary presolve exactly as the CLI does, so a model
    // solved in the browser takes the same path — and reaches the same
    // answer — as `pounce model.nl` on the command line. `NlTnlp` doubles as
    // the `ExpressionProvider` presolve needs for FBBT.
    let presolve_opts = match pounce_presolve::PresolveOptions::from_options_list(app.options()) {
        Ok(o) => o,
        Err(e) => return serde_json::json!({ "error": format!("presolve setup failed: {e}") }),
    };
    let presolve = presolve_opts.enabled.then(|| {
        app.set_presolve_already_applied(true);
        Rc::new(RefCell::new(
            pounce_presolve::PresolveTnlp::with_expression_provider(
                Rc::clone(&tnlp) as Rc<RefCell<dyn TNLP>>,
                Rc::clone(&tnlp) as Rc<RefCell<dyn ExpressionProvider>>,
                presolve_opts,
            ),
        ))
    });
    let target: Rc<RefCell<dyn TNLP>> = match &presolve {
        Some(p) => Rc::clone(p) as Rc<RefCell<dyn TNLP>>,
        None => Rc::clone(&tnlp) as Rc<RefCell<dyn TNLP>>,
    };

    let status = app.optimize_tnlp(target);
    let stats = app.statistics();
    let presolve_report = presolve.as_ref().map(|p| {
        let h = p.borrow();
        let tr = h.tighten_report();
        serde_json::json!({
            "tightened_bounds": tr.n_tightened,
            "newly_finite_bounds": tr.n_new_finite,
            "dropped_rows": h.n_dropped_rows(),
        })
    });

    let mut t = tnlp.borrow_mut();
    let x: Vec<f64> = t.final_x().map(<[f64]>::to_vec).unwrap_or_default();
    let objective = t.final_obj();
    // Constraint values at the returned point, so the page can show which
    // rows are tight or violated without re-evaluating the model in JS.
    let m = t.problem().m;
    let mut g = vec![0.0; m];
    if x.is_empty() || !t.eval_g(&x, true, &mut g) {
        g.clear();
    }
    let (g_l, g_u) = (t.problem().g_l.clone(), t.problem().g_u.clone());

    let (x_prev, x_truncated) = preview(&x);
    let (g_prev, g_truncated) = preview(&g);
    let (g_l_prev, _) = preview(&g_l);
    let (g_u_prev, _) = preview(&g_u);

    serde_json::json!({
        "status": format!("{status:?}"),
        "status_code": status.as_int(),
        "success": status.as_int() >= 0,
        "objective": objective,
        "iterations": stats.iteration_count,
        "wall_time_secs": stats.total_wallclock_time_secs,
        "dual_infeasibility": stats.final_unscaled_dual_inf,
        "constraint_violation": stats.final_unscaled_constr_viol,
        "complementarity": stats.final_unscaled_compl,
        "kkt_error": stats.final_unscaled_kkt_error,
        "restoration_calls": stats.restoration_calls,
        "presolve": presolve_report,
        "evals": {
            "objective": stats.num_obj_evals,
            "objective_grad": stats.num_obj_grad_evals,
            "constraints": stats.num_constr_evals,
            "constraint_jac": stats.num_constr_jac_evals,
            "hessian": stats.num_hess_evals,
        },
        "x": x_prev,
        "g": g_prev,
        "g_l": g_l_prev,
        "g_u": g_u_prev,
        "truncated": x_truncated || g_truncated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two variables, one nonlinear equality, one linear inequality —
    /// small enough to keep inline, big enough that every field of the
    /// summary has something to say.
    const SIMPLE_NL: &str = include_str!("../tests/simple.nl");

    /// Call the C ABI the way JS does — bytes in, JSON out — and hand back
    /// the parsed payload.
    fn call_load(nl: &str, col: &str, row: &str) -> serde_json::Value {
        let ptr = unsafe {
            pounce_load(
                nl.as_ptr(),
                nl.len(),
                col.as_ptr(),
                col.len(),
                row.as_ptr(),
                row.len(),
            )
        };
        let s = unsafe { std::ffi::CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned();
        unsafe { pounce_free_string(ptr) };
        serde_json::from_str(&s).expect("entry points must return JSON")
    }

    fn call_solve(opts: &str) -> serde_json::Value {
        let ptr = unsafe { pounce_solve(opts.as_ptr(), opts.len()) };
        let s = unsafe { std::ffi::CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned();
        unsafe { pounce_free_string(ptr) };
        serde_json::from_str(&s).expect("entry points must return JSON")
    }

    #[test]
    fn load_reports_problem_shape() {
        let s = call_load(SIMPLE_NL, "alpha\nbeta\n", "ring\nline\n");
        assert_eq!(s["n_vars"], 2);
        assert_eq!(s["n_cons"], 2);
        assert_eq!(s["sense"], "minimize");
        assert_eq!(s["var_names"][0], "alpha");
        assert_eq!(s["con_names"][1], "line");
        // One nonlinear row (the circle), one linear row.
        assert_eq!(s["n_nonlinear_cons"], 1);
        assert!(s["nnz_jac"].as_u64().unwrap_or(0) > 0);
    }

    #[test]
    fn name_files_of_the_wrong_length_are_ignored() {
        let s = call_load(SIMPLE_NL, "only_one\n", "");
        assert!(
            s["var_names"]
                .as_array()
                .map(Vec::is_empty)
                .unwrap_or(false),
            "a .col file that does not match n must not be applied"
        );
    }

    #[test]
    fn bad_input_is_an_error_not_a_panic() {
        assert!(call_load("not an nl file at all", "", "")["error"].is_string());
        assert!(call_load("", "", "")["error"].is_string());
    }

    #[test]
    fn solve_runs_the_loaded_model() {
        call_load(SIMPLE_NL, "", "");
        let r = call_solve("print_level 0\n");
        assert_eq!(r["success"], true, "solve payload: {r}");
        assert!(r["iterations"].as_i64().unwrap_or(0) > 0);
        // min x0 s.t. x0^2 + x1^2 == 1, x0 + x1 >= 0  ⇒  x0 = -1/√2.
        let obj = r["objective"].as_f64().unwrap_or(f64::NAN);
        assert!(
            (obj + 0.5f64.sqrt()).abs() < 1e-6,
            "unexpected objective {obj}"
        );
        assert_eq!(r["x"].as_array().map(Vec::len), Some(2));
        assert_eq!(r["g"].as_array().map(Vec::len), Some(2));
    }

    #[test]
    fn solving_with_no_model_loaded_is_an_error() {
        LOADED.with(|slot| *slot.borrow_mut() = None);
        assert!(call_solve("")["error"].is_string());
    }

    #[test]
    fn bad_options_are_reported() {
        call_load(SIMPLE_NL, "", "");
        assert!(call_solve("max_iter not_an_integer\n")["error"].is_string());
    }
}

//! `pounce_sens` — AMPL driver that runs a pounce solve, then a
//! post-optimal sensitivity step via `pounce-sensitivity`, and writes
//! the perturbed primal back into an AMPL `.sol` file as a
//! `sens_sol_state_N` suffix.
//!
//! Mirror of upstream sIPOPT's `ipopt_sens` AMPL binary
//! ([`ref/Ipopt/contrib/sIPOPT/src/AmplTNLP.cpp` etc.](../../../ref/Ipopt/contrib/sIPOPT/)),
//! limited to the metadata-measurement path that the
//! `parametric_ampl` example exercises.
//!
//! # Usage
//!
//! ```text
//! pounce_sens <input.nl> [<output.sol>]
//! ```
//!
//! The output path defaults to `<input>.sol` if omitted, matching
//! AMPL's convention. The input file must declare three suffixes
//! (otherwise `pounce_sens` just runs a normal solve and writes the
//! nominal solution):
//!
//! * `sens_state_1` — integer var-suffix tagging each parameter
//!   (value = 1..n_params, 0 for non-parameters).
//! * `sens_state_value_1` — real var-suffix carrying the perturbed
//!   parameter values.
//! * `sens_init_constr` — integer con-suffix tagging which
//!   constraint pins each parameter to its nominal value (value =
//!   1..n_params, 0 otherwise).
//!
//! See [`ref/Ipopt/contrib/sIPOPT/examples/parametric_cpp/parametricTNLP.cpp`](../../../ref/Ipopt/contrib/sIPOPT/examples/parametric_cpp/parametricTNLP.cpp)
//! `get_var_con_metadata` for the canonical suffix shape upstream
//! emits, and pounce#16's `parametric_cpp.rs` for an end-to-end
//! cross-check against upstream's golden output.

use std::cell::RefCell;
use std::path::PathBuf;
use std::process::ExitCode;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_cli::nl_reader::{self, NlSuffixes};
use pounce_cli::nl_writer::{
    format_sol, SolSuffix, SolSuffixTarget, SolSuffixValues, SolutionFile,
};
use pounce_cli::solve_report::{
    write_report_file, InputDescriptor, ReportBuilder, ReportDetail, SolutionSuffix,
};
use pounce_common::types::{Index, Number};
use pounce_linalg::dense_vector::DenseVector;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::TNLP;
use pounce_sensitivity::{
    IndexSchurData, PdSensBacksolver, SensApplication, SensBacksolver, SensOptions,
};

const USAGE: &str = "\
Usage: pounce_sens <input.nl> [<output.sol>] [OPTIONS]

Reads an AMPL `.nl` file declaring sIPOPT-style suffixes
(`sens_state_1`, `sens_state_value_1`, `sens_init_constr`), runs a
pounce solve, performs the post-optimal sensitivity step, and writes
the perturbed primal back into the `.sol` as `sens_sol_state_1`.

If <output.sol> is omitted it defaults to <input>.sol (same directory,
extension swapped).

Options:
  --json-output <path>      write a structured JSON solve report to PATH
                            after the solve (pounce#8 — FAIR-aligned)
  --json-detail LEVEL       summary | full (default: summary). `full` adds
                            per-iteration trajectory and suffix blocks.
  --sens-boundcheck         clamp the perturbed primal x* + Δx onto the
                            declared `[x_l, x_u]` (single-pass clamp;
                            upstream's iterative Schur refinement is a
                            future enhancement). Mirrors `sens_boundcheck`.
  --sens-bound-eps EPS      tolerance for `--sens-boundcheck` (default 1e-3).
  --compute-red-hessian     after the solve, compute the reduced Hessian
                            over the variables tagged by the `red_hessian`
                            integer var-suffix in the input .nl. The matrix
                            is printed to stderr and, with `--json-detail full`,
                            written as `_red_hessian` in the JSON report.
                            Mirrors upstream sIPOPT's `compute_red_hessian`.
  --rh-eigendecomp          also compute the eigendecomposition of the
                            reduced Hessian. Implies --compute-red-hessian.
                            Mirrors upstream `rh_eigendecomp`.
  --help, -h                print this message and exit
  --version, -V             print version and exit
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    // Handle --help / --version before the positional parse so they
    // work even without a `.nl` argument.
    for a in args.iter().skip(1) {
        match a.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            "-V" | "--version" => {
                println!("pounce_sens {}", env!("CARGO_PKG_VERSION"));
                return ExitCode::SUCCESS;
            }
            _ => {}
        }
    }

    let parsed = match parse_args(&args) {
        Ok(p) => p,
        Err(msg) => {
            eprintln!("pounce_sens: {msg}");
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let in_path = parsed.in_path;
    let out_path = parsed.out_path;
    let json_output = parsed.json_output;
    let json_detail = parsed.json_detail;
    let compute_red_hessian = parsed.compute_red_hessian;
    let rh_eigendecomp = parsed.rh_eigendecomp;
    let sens_boundcheck = parsed.sens_boundcheck;
    let sens_bound_eps = parsed.sens_bound_eps;

    // 1. Parse the .nl. Keep `suffixes` separate from the consumed
    //    NlProblem so the on_converged closure can read them after
    //    NlTnlp takes ownership.
    let prob = match nl_reader::read_nl_file(&in_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("pounce_sens: read {}: {e}", in_path.display());
            return ExitCode::from(2);
        }
    };
    let suffixes = prob.suffixes.clone();
    let n_full = prob.n;
    let m_full = prob.m;
    let tnlp_concrete = nl_reader::NlTnlp::new(prob);
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(tnlp_concrete));

    // 2. Set up the IPM. Default options + register `run_sens` so the
    //    keys roundtrip through ipopt.opt-style flags if the user
    //    plumbed one (we do not currently auto-load ipopt.opt; pounce
    //    binary handles that and pounce_sens follows the same pattern
    //    of reading argv only).
    let mut app = IpoptApplication::new();
    if let Err(e) = app.initialize() {
        eprintln!("pounce_sens: initialize: {e}");
        return ExitCode::from(2);
    }
    if json_output.is_some() && matches!(json_detail, ReportDetail::Full) {
        app.enable_iter_history();
    }

    // 3. Sensitivity callback: stashes the perturbed-x slice into
    //    `sens_sol_state_1` for the .sol writer below. When the
    //    required suffixes aren't present, the callback writes
    //    `None` and main() falls back to writing just the nominal
    //    solution.
    let sens_out: Rc<RefCell<Option<Vec<Number>>>> = Rc::new(RefCell::new(None));
    let nominal_x_out: Rc<RefCell<Option<Vec<Number>>>> = Rc::new(RefCell::new(None));
    let lambda_out: Rc<RefCell<Option<Vec<Number>>>> = Rc::new(RefCell::new(None));
    let red_hessian_out: Rc<RefCell<Option<RedHessianResult>>> = Rc::new(RefCell::new(None));

    let sens_out_cb = Rc::clone(&sens_out);
    let nominal_x_cb = Rc::clone(&nominal_x_out);
    let lambda_cb = Rc::clone(&lambda_out);
    let red_hessian_cb = Rc::clone(&red_hessian_out);
    let suffixes_cb = suffixes.clone();
    app.set_on_converged(Box::new(move |data, cq, nlp, pd| {
        // Capture nominal primal / dual once for the .sol writer. The
        // .sol's x block must be full-length (`n_full`); when fixed
        // variables were eliminated, the IPM's `curr.x` carries only
        // the `n_x_var` free coordinates — `lift_x_to_full` reinflates.
        let curr = data.borrow().curr.clone().expect("curr at convergence");
        let x_full: Vec<Number> = nlp.borrow().lift_x_to_full(&*curr.x);
        *nominal_x_cb.borrow_mut() = Some(x_full.clone());

        let yc_dense = curr.y_c.as_any().downcast_ref::<DenseVector>();
        let yd_dense = curr.y_d.as_any().downcast_ref::<DenseVector>();
        let n_c = curr.y_c.dim() as usize;
        let n_d = curr.y_d.dim() as usize;
        let mut lambda = Vec::with_capacity(n_c + n_d);
        if let Some(dy) = yc_dense {
            lambda.extend_from_slice(&dy.expanded_values());
        } else {
            lambda.extend(std::iter::repeat(0.0).take(n_c));
        }
        if let Some(dy) = yd_dense {
            lambda.extend_from_slice(&dy.expanded_values());
        } else {
            lambda.extend(std::iter::repeat(0.0).take(n_d));
        }
        *lambda_cb.borrow_mut() = Some(lambda);

        // Try to run the sensitivity step. Bail quietly (so the
        // nominal-solve solution still writes out) when any required
        // suffix is missing.
        if let Some(mut dx) =
            try_compute_sens_step(data, cq, nlp, pd, &suffixes_cb, n_full, m_full, &x_full)
        {
            let n_x = curr.x.dim() as usize;
            if sens_boundcheck {
                // Single-pass clamp of the primal step before
                // scattering onto the full-x grid; see
                // pounce_sensitivity::boundcheck for the algorithm.
                use pounce_linalg::dense_vector::DenseVector;
                let x_curr_compressed: Vec<Number> = curr
                    .x
                    .as_any()
                    .downcast_ref::<DenseVector>()
                    .map(|d| d.values().to_vec())
                    .unwrap_or_default();
                let mut dx_primal = dx[..n_x].to_vec();
                let n_clamped = pounce_sensitivity::boundcheck::clamp_with_nlp(
                    &*nlp.borrow(),
                    &x_curr_compressed,
                    &mut dx_primal,
                    sens_bound_eps,
                );
                if n_clamped > 0 {
                    eprintln!(
                        "pounce_sens: --sens-boundcheck clamped {n_clamped} primal coordinate(s)"
                    );
                    dx[..n_x].copy_from_slice(&dx_primal);
                }
            }
            // Scatter the compressed primal step `dx[0..n_x_var]` back
            // onto the full-x grid; fixed variables stay at their
            // nominal (== fixed) values.
            let mut x_pert = x_full.clone();
            let nlp_ref = nlp.borrow();
            for var_idx in 0..n_x {
                let full_idx = nlp_ref.var_x_to_full_x(var_idx as Index) as usize;
                x_pert[full_idx] += dx[var_idx];
            }
            *sens_out_cb.borrow_mut() = Some(x_pert);
        }

        if compute_red_hessian {
            match try_compute_red_hessian(
                data,
                cq,
                nlp,
                pd,
                &suffixes_cb,
                rh_eigendecomp,
            ) {
                Some(r) => *red_hessian_cb.borrow_mut() = Some(r),
                None => {
                    eprintln!(
                        "pounce_sens: --compute-red-hessian requested but `red_hessian` suffix \
                         is missing or empty in {}",
                        "the input .nl"
                    );
                }
            }
        }
    }));

    let status = app.optimize_tnlp(Rc::clone(&tnlp));

    // 4. Assemble the .sol. Always emit the nominal block (so AMPL's
    //    reader sees something), and attach the sens_sol_state_1
    //    suffix when the sensitivity step ran.
    let x_nominal = nominal_x_out
        .borrow()
        .clone()
        .unwrap_or_else(|| vec![0.0; n_full]);
    let lambda = lambda_out
        .borrow()
        .clone()
        .unwrap_or_else(|| vec![0.0; m_full]);
    let sens_sol: Option<Vec<Number>> = sens_out.borrow().clone();

    let mut suffixes_out: Vec<SolSuffix> = Vec::new();
    if let Some(xp) = sens_sol {
        suffixes_out.push(SolSuffix {
            name: "sens_sol_state_1".to_string(),
            target: SolSuffixTarget::Var,
            values: SolSuffixValues::Real(xp),
        });
    }

    let message = format!("POUNCE_SENS {}: {status:?}", env!("CARGO_PKG_VERSION"));
    let payload = SolutionFile {
        message: &message,
        x: &x_nominal,
        lambda: &lambda,
        solve_result_num: status_to_solve_result_num(status),
        suffixes: &suffixes_out,
    };
    let sol_text = format_sol(&payload);
    if let Err(e) = std::fs::write(&out_path, sol_text.as_bytes()) {
        eprintln!("pounce_sens: write {}: {e}", out_path.display());
        return ExitCode::from(2);
    }

    eprintln!("pounce_sens: wrote {}", out_path.display());

    // Print the reduced Hessian (and optional eigendecomp) to stderr
    // when requested. Mirrors upstream sIPOPT's `J_INSUPPRESSIBLE`
    // RedHessian / Eigenvalues prints in `SensReducedHessianCalculator.cpp`.
    if let Some(rh) = red_hessian_out.borrow().as_ref() {
        print_red_hessian_to_stderr(rh);
    } else if compute_red_hessian {
        eprintln!(
            "pounce_sens: --compute-red-hessian requested but the reduced \
             Hessian was not produced (see warnings above)."
        );
    }

    // Optional JSON report (pounce#8). Carries everything in the .sol
    // plus FAIR provenance + per-iter history (when --json-detail
    // full was requested).
    if let Some(jpath) = &json_output {
        let input = InputDescriptor::NlFile {
            path: in_path.clone(),
            size_bytes: std::fs::metadata(&in_path).ok().map(|m| m.len()),
        };
        let mut builder = ReportBuilder::new(json_detail, input);
        builder.problem.n_variables = n_full as Index;
        builder.problem.n_constraints = m_full as Index;
        builder.problem.n_objectives = 1;
        builder.solution.status = status;
        builder.solution.solve_result_num = status_to_solve_result_num(status);
        builder.solution.objective = app.statistics().final_objective;
        builder.solution.x = x_nominal.clone();
        builder.solution.lambda = lambda.clone();
        if matches!(json_detail, ReportDetail::Full) {
            for s in &suffixes_out {
                builder.solution.suffixes.push(sol_suffix_to_report(s));
            }
            if let Some(rh) = red_hessian_out.borrow().as_ref() {
                // Pack the reduced Hessian as a problem-level suffix so
                // downstream consumers can fish it out alongside the
                // standard solution suffixes. Layout: a single
                // length-n² real vector in column-major order, with the
                // ordering of variables recorded in `_red_hessian_vars`.
                builder.solution.suffixes.push(SolutionSuffix {
                    name: "_red_hessian".to_string(),
                    target: "problem".to_string(),
                    kind: "real".to_string(),
                    values: rh.hr.clone(),
                    int_values: Vec::new(),
                });
                builder.solution.suffixes.push(SolutionSuffix {
                    name: "_red_hessian_vars".to_string(),
                    target: "problem".to_string(),
                    kind: "int".to_string(),
                    values: Vec::new(),
                    int_values: rh.var_indices.iter().map(|&v| v as i32).collect(),
                });
                if let Some(w) = &rh.eigenvalues {
                    builder.solution.suffixes.push(SolutionSuffix {
                        name: "_red_hessian_eigenvalues".to_string(),
                        target: "problem".to_string(),
                        kind: "real".to_string(),
                        values: w.clone(),
                        int_values: Vec::new(),
                    });
                }
                if let Some(v) = &rh.eigenvectors {
                    builder.solution.suffixes.push(SolutionSuffix {
                        name: "_red_hessian_eigenvectors".to_string(),
                        target: "problem".to_string(),
                        kind: "real".to_string(),
                        values: v.clone(),
                        int_values: Vec::new(),
                    });
                }
            }
        }
        builder.ingest_stats(&app.statistics());
        let report = builder.finish();
        if let Err(e) = write_report_file(jpath, &report) {
            eprintln!(
                "pounce_sens: failed to write JSON report to {}: {e}",
                jpath.display()
            );
        } else {
            eprintln!("pounce_sens: wrote {}", jpath.display());
        }
    }

    match status {
        ApplicationReturnStatus::SolveSucceeded
        | ApplicationReturnStatus::SolvedToAcceptableLevel => ExitCode::SUCCESS,
        _ => ExitCode::from(1),
    }
}

/// Convert a `.sol`-shaped suffix block into the JSON report's flat
/// representation.
fn sol_suffix_to_report(s: &SolSuffix) -> SolutionSuffix {
    let target = match s.target {
        SolSuffixTarget::Var => "var",
        SolSuffixTarget::Con => "con",
        SolSuffixTarget::Obj => "obj",
        SolSuffixTarget::Problem => "problem",
    }
    .to_string();
    let (kind, values, int_values) = match &s.values {
        SolSuffixValues::Real(v) => ("real".to_string(), v.clone(), Vec::new()),
        SolSuffixValues::Int(v) => ("int".to_string(), Vec::new(), v.clone()),
        SolSuffixValues::ProblemReal(v) => ("real".to_string(), vec![*v], Vec::new()),
        SolSuffixValues::ProblemInt(v) => ("int".to_string(), Vec::new(), vec![*v]),
    };
    SolutionSuffix {
        name: s.name.clone(),
        target,
        kind,
        values,
        int_values,
    }
}

struct ParsedArgs {
    in_path: PathBuf,
    out_path: PathBuf,
    json_output: Option<PathBuf>,
    json_detail: ReportDetail,
    compute_red_hessian: bool,
    rh_eigendecomp: bool,
    sens_boundcheck: bool,
    sens_bound_eps: Number,
}

/// Read argv. Positional: `<input.nl> [<output.sol>]`. Flags:
/// `--json-output PATH` / `--json-detail summary|full` (pounce#8).
/// The output path defaults to the input with `.sol` swapped in for
/// `.nl`.
fn parse_args(args: &[String]) -> Result<ParsedArgs, String> {
    let mut positional: Vec<PathBuf> = Vec::new();
    let mut json_output: Option<PathBuf> = None;
    let mut json_detail = ReportDetail::Summary;
    let mut compute_red_hessian = false;
    let mut rh_eigendecomp = false;
    let mut sens_boundcheck = false;
    let mut sens_bound_eps: Number = 1e-3;

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--json-output" => {
                let v = it
                    .next()
                    .ok_or_else(|| "--json-output requires a value".to_string())?;
                json_output = Some(PathBuf::from(v));
            }
            "--json-detail" => {
                let v = it
                    .next()
                    .ok_or_else(|| "--json-detail requires a value".to_string())?;
                json_detail = ReportDetail::parse(v)?;
            }
            "--compute-red-hessian" => compute_red_hessian = true,
            "--rh-eigendecomp" => {
                rh_eigendecomp = true;
                compute_red_hessian = true;
            }
            "--sens-boundcheck" => sens_boundcheck = true,
            "--sens-bound-eps" => {
                let v = it
                    .next()
                    .ok_or_else(|| "--sens-bound-eps requires a value".to_string())?;
                sens_bound_eps = v
                    .parse::<Number>()
                    .map_err(|e| format!("--sens-bound-eps: {e}"))?;
                sens_boundcheck = true;
            }
            other if other.starts_with("--") => {
                return Err(format!("unrecognized flag '{other}'"));
            }
            other => positional.push(PathBuf::from(other)),
        }
    }

    if positional.is_empty() || positional.len() > 2 {
        return Err(format!(
            "expected 1 or 2 positional args, got {}",
            positional.len()
        ));
    }
    let in_path = positional.remove(0);
    let out_path = if let Some(p) = positional.pop() {
        p
    } else {
        let mut p = in_path.clone();
        p.set_extension("sol");
        p
    };
    Ok(ParsedArgs {
        in_path,
        out_path,
        json_output,
        json_detail,
        compute_red_hessian,
        rh_eigendecomp,
        sens_boundcheck,
        sens_bound_eps,
    })
}

/// Format a reduced Hessian (and optional eigendecomp) onto stderr.
/// Matches the style of upstream sIPOPT's
/// `SensReducedHessianCalculator.cpp` `S->Print(...)` /
/// `eigenvalues->Print(...)` calls — informational, not parsed.
fn print_red_hessian_to_stderr(rh: &RedHessianResult) {
    let n = rh.var_indices.len();
    eprintln!("\n=== Reduced Hessian (n={n}) ===");
    eprintln!("var indices: {:?}", rh.var_indices);
    for i in 0..n {
        let mut row = String::new();
        for j in 0..n {
            // column-major: hr[i + n*j]
            row.push_str(&format!(" {:>14.6e}", rh.hr[i + n * j]));
        }
        eprintln!(" [{i:>3}]{row}");
    }
    if let Some(w) = &rh.eigenvalues {
        eprintln!("\n=== Reduced-Hessian eigenvalues (ascending) ===");
        for (k, v) in w.iter().enumerate() {
            eprintln!(" [{k:>3}] {v:>14.6e}");
        }
    }
    eprintln!();
}

/// Outputs of [`try_compute_red_hessian`]: the column-major `n × n`
/// reduced Hessian (`hr`), the variable indices `var_indices` that
/// label its rows/cols (so a downstream JSON consumer can map back to
/// AMPL var names), and the optional eigendecomposition.
struct RedHessianResult {
    /// var-x indices (algorithm-side, length `n`) that label the
    /// rows/cols of `hr`, ordered by the 1..n slot from the AMPL
    /// `red_hessian` suffix. Fixed variables are skipped (they cannot
    /// participate in the reduced Hessian).
    var_indices: Vec<usize>,
    /// Column-major `n × n` reduced Hessian.
    hr: Vec<Number>,
    /// Optional ascending eigenvalues (length `n`).
    eigenvalues: Option<Vec<Number>>,
    /// Optional column-major eigenvectors (length `n²`).
    eigenvectors: Option<Vec<Number>>,
}

/// Read the AMPL `red_hessian` integer var-suffix from `.nl`, select
/// the tagged free variables, and compute the reduced Hessian via
/// [`SensApplication::compute_reduced_hessian`] (optionally also the
/// eigendecomposition). Returns `None` (quietly) when the suffix is
/// missing or empty.
///
/// Mirrors the `compute_red_hessian=yes` branch of upstream
/// [`SensBuilder::BuildRedHessCalc`](../../../ref/Ipopt/contrib/sIPOPT/src/SensBuilder.cpp).
fn try_compute_red_hessian(
    data: &pounce_algorithm::ipopt_data::IpoptDataHandle,
    cq: &pounce_algorithm::ipopt_cq::IpoptCqHandle,
    nlp: &Rc<RefCell<dyn pounce_nlp::ipopt_nlp::IpoptNlp>>,
    pd: &mut pounce_algorithm::kkt::pd_full_space_solver::PdFullSpaceSolver,
    suffixes: &NlSuffixes,
    compute_eigen: bool,
) -> Option<RedHessianResult> {
    let red_hessian_tags = suffixes.var_int.get("red_hessian")?;
    let max_slot = red_hessian_tags.iter().copied().max().unwrap_or(0);
    if max_slot <= 0 {
        return None;
    }
    let n_slots = max_slot as usize;

    // For each slot 1..n_slots, look up the full-x index, then map to
    // the var-x index via the IpoptNlp trait. Fixed variables (no
    // var-x mapping) are skipped with a warning.
    let nlp_ref = nlp.borrow();
    let mut full_for_slot: Vec<Option<usize>> = vec![None; n_slots];
    for (full_idx, &slot) in red_hessian_tags.iter().enumerate() {
        if slot > 0 {
            let s = slot as usize;
            if s <= n_slots {
                full_for_slot[s - 1] = Some(full_idx);
            }
        }
    }
    let mut var_indices: Vec<usize> = Vec::with_capacity(n_slots);
    for (k, slot) in full_for_slot.iter().enumerate() {
        let full_idx = match slot {
            Some(i) => *i,
            None => {
                eprintln!(
                    "pounce_sens: red_hessian slot {} has no tagged variable",
                    k + 1
                );
                return None;
            }
        };
        match nlp_ref.full_x_to_var_x(full_idx as Index) {
            Some(v) => var_indices.push(v as usize),
            None => {
                eprintln!(
                    "pounce_sens: red_hessian slot {} tags fixed variable {} (skipping)",
                    k + 1,
                    full_idx
                );
                return None;
            }
        }
    }
    drop(nlp_ref);

    // Build the row-selector SchurData over the var-x rows directly
    // (the x block starts at compound-vector index 0).
    let rows: Vec<Index> = var_indices.iter().map(|&v| v as Index).collect();
    let signs: Vec<Index> = vec![1; var_indices.len()];
    let a_data = IndexSchurData::from_parts(rows, signs).ok()?;

    let backsolver = PdSensBacksolver::new(data, cq, nlp, pd).ok()?;
    let opts = SensOptions {
        compute_red_hessian: true,
        rh_eigendecomp: compute_eigen,
        ..SensOptions::default()
    };
    let mut app = SensApplication::new(a_data, backsolver, opts);
    let n = var_indices.len();
    let mut hr = vec![0.0; n * n];
    let (eigenvalues, eigenvectors) = if compute_eigen {
        let mut w = vec![0.0; n];
        let mut v = vec![0.0; n * n];
        if !app.compute_reduced_hessian_eigen(&mut hr, &mut w, &mut v) {
            eprintln!("pounce_sens: reduced-Hessian eigendecomp failed");
            return None;
        }
        (Some(w), Some(v))
    } else {
        if !app.compute_reduced_hessian(&mut hr) {
            eprintln!("pounce_sens: reduced-Hessian computation failed");
            return None;
        }
        (None, None)
    };
    let _ = cq;
    Some(RedHessianResult {
        var_indices,
        hr,
        eigenvalues,
        eigenvectors,
    })
}

/// Try to compute the parametric sensitivity step from the suffixes
/// declared in the input `.nl`. Returns `None` (quietly) when any
/// required suffix is missing — typical for `.nl` files that aren't
/// sensitivity inputs.
#[allow(clippy::too_many_arguments)]
fn try_compute_sens_step(
    data: &pounce_algorithm::ipopt_data::IpoptDataHandle,
    cq: &pounce_algorithm::ipopt_cq::IpoptCqHandle,
    nlp: &Rc<RefCell<dyn pounce_nlp::ipopt_nlp::IpoptNlp>>,
    pd: &mut pounce_algorithm::kkt::pd_full_space_solver::PdFullSpaceSolver,
    suffixes: &NlSuffixes,
    n_full: usize,
    _m_full: usize,
    x_nominal: &[Number],
) -> Option<Vec<Number>> {
    // Required suffixes. The "_1" suffix tier corresponds to upstream
    // sIPOPT's `n_sens_steps=1` mode. Higher tiers (sens_state_2 etc.)
    // are a Phase-2 follow-up.
    let sens_state = suffixes.var_int.get("sens_state_1")?;
    let sens_state_value = suffixes.var_real.get("sens_state_value_1")?;
    let sens_init_constr = suffixes.con_int.get("sens_init_constr")?;

    if sens_state.len() != n_full || sens_state_value.len() != n_full {
        eprintln!(
            "pounce_sens: sens_state_1 / sens_state_value_1 length mismatch (expected {n_full})"
        );
        return None;
    }

    // Number of parameters and per-parameter (var_idx, constraint_idx)
    // pairs. The integer suffix value is 1..n_params, indexing which
    // parameter slot each variable / constraint maps to.
    let n_params = sens_state.iter().copied().max().unwrap_or(0).max(0) as usize;
    if n_params == 0 {
        return None;
    }

    // For each parameter slot, find its variable index (from
    // sens_state_1) and its pinning-constraint index (from
    // sens_init_constr).
    let mut param_var_idx: Vec<Option<usize>> = vec![None; n_params];
    for (var_idx, &slot) in sens_state.iter().enumerate() {
        if slot > 0 {
            let s = slot as usize;
            if s <= n_params {
                param_var_idx[s - 1] = Some(var_idx);
            }
        }
    }
    let mut param_con_idx: Vec<Option<usize>> = vec![None; n_params];
    for (con_idx, &slot) in sens_init_constr.iter().enumerate() {
        if slot > 0 {
            let s = slot as usize;
            if s <= n_params {
                param_con_idx[s - 1] = Some(con_idx);
            }
        }
    }
    for k in 0..n_params {
        if param_var_idx[k].is_none() || param_con_idx[k].is_none() {
            eprintln!(
                "pounce_sens: parameter {} missing sens_state_1 or sens_init_constr tag",
                k + 1
            );
            return None;
        }
    }

    // Build the SchurData rows: flat compound-vector index for each
    // pinning constraint = n_x + n_s + c_block_idx (i.e. y_c[…] slot).
    // Pounce's compound layout matches upstream's
    // `MetadataMeasurement::GetInitialEqConstraints`
    // (`ref/Ipopt/contrib/sIPOPT/src/SensMetadataMeasurement.cpp:69-83`).
    //
    // Two coordinate transforms are needed when `n_x != n_full` (fixed
    // variables present) or when the c/d split reorders constraints:
    //   * full-x index → var-x index via `IpoptNlp::full_x_to_var_x`
    //   * full-g index → c-block index via `IpoptNlp::full_g_to_c_block`
    let curr = data.borrow().curr.clone()?;
    let n_x = curr.x.dim() as usize;
    let n_s = curr.s.dim() as usize;
    let nlp_ref = nlp.borrow();
    let y_c_offset = n_x + n_s;
    let mut rows: Vec<Index> = Vec::with_capacity(n_params);
    for k in 0..n_params {
        let full_ci = param_con_idx[k].unwrap();
        match nlp_ref.full_g_to_c_block(full_ci as Index) {
            Some(c_idx) => rows.push(y_c_offset as Index + c_idx),
            None => {
                eprintln!(
                    "pounce_sens: parameter {} pinning constraint #{} is an inequality (not in the c block)",
                    k + 1,
                    full_ci
                );
                return None;
            }
        }
    }
    let signs: Vec<Index> = vec![1; n_params];
    let a_data = IndexSchurData::from_parts(rows, signs).ok()?;

    // Δp[k] = perturbed_value - current_value for parameter k. Both
    // sides are read from the user's full-x array (length `n_full`); the
    // caller passes `x_nominal` already lifted via
    // `IpoptNlp::lift_x_to_full`, so indexing by the full-x var index
    // works whether or not other variables were eliminated.
    let mut delta_p: Vec<Number> = Vec::with_capacity(n_params);
    for k in 0..n_params {
        let vi = param_var_idx[k].unwrap();
        delta_p.push(sens_state_value[vi] - x_nominal[vi]);
    }
    drop(nlp_ref);

    let backsolver = PdSensBacksolver::new(data, cq, nlp, pd).ok()?;
    let n_full_pd = backsolver.dim();
    let mut rhs_full = vec![0.0; n_full_pd];
    {
        use pounce_sensitivity::SchurData;
        a_data
            .trans_multiply(&delta_p, &mut rhs_full)
            .map_err(|e| eprintln!("pounce_sens: trans_multiply error: {e:?}"))
            .ok()?;
    }
    let mut dx_full = vec![0.0; n_full_pd];
    if !backsolver.solve(&rhs_full, &mut dx_full) {
        eprintln!("pounce_sens: KKT backsolve failed");
        return None;
    }
    Some(dx_full)
}

/// Map a pounce `ApplicationReturnStatus` onto an AMPL-style
/// `solve_result_num` per
/// <https://ampl.com/REFS/hooking2.pdf> §5 (table p. 23): 0 solved,
/// 100-range solved-with-warning, 200-range infeasible,
/// 400-range limit-reached, 500-range failure.
fn status_to_solve_result_num(status: ApplicationReturnStatus) -> i32 {
    use ApplicationReturnStatus::*;
    match status {
        SolveSucceeded => 0,
        SolvedToAcceptableLevel => 100,
        FeasiblePointFound => 100,
        InfeasibleProblemDetected => 200,
        SearchDirectionBecomesTooSmall => 400,
        DivergingIterates => 401,
        MaximumIterationsExceeded => 400,
        MaximumCpuTimeExceeded => 400,
        MaximumWallTimeExceeded => 400,
        UserRequestedStop => 502,
        RestorationFailed => 500,
        ErrorInStepComputation => 500,
        InvalidNumberDetected => 500,
        InternalError => 500,
        UnrecoverableException => 500,
        NonIpoptExceptionThrown => 500,
        InsufficientMemory => 503,
        InvalidProblemDefinition => 504,
        InvalidOption => 504,
        NotEnoughDegreesOfFreedom => 504,
    }
}

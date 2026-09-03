//! Which engine answers a sIPOPT sensitivity request on a convex QP — and the
//! evidence that both give the same answer.
//!
//! # The contract this file pins, and the one it replaced
//!
//! Issue #196 was a *silent drop*: a `.nl` carrying the sIPOPT suffixes routed
//! to the convex fast path, which had no sensitivity machinery, and no
//! `sens_sol_state_1` came back. The fix at the time was to **decline the fast
//! path** — `auto` rerouted the whole model to the general NLP filter-IPM, and
//! an explicit `solver_selection=qp-ipm` warned and skipped. Correct, and
//! expensive: an LP or convex QP paid the general engine's cost for a question
//! the specialized engine can now answer itself.
//!
//! The convex arm has a parametric step now
//! (`pounce_convex::QpSensitivity`, reached through
//! `pounce_cli::convex_sens`), so the decline is narrowed to the capability
//! that is still missing rather than applied to every post-optimal request.
//! The contract is now:
//!
//! | request | engine | why |
//! |---|---|---|
//! | parametric step, LP / convex QP, equality pins | **convex** | it can express it |
//! | parametric step, pin is not a unit equality row | NLP (`auto`) / skipped (forced) | `parametric_step` perturbs `b`; an inequality lives in `h` |
//! | reduced Hessian | NLP | a *different computation* behind the same word — null-space projection vs sIPOPT's Schur route |
//! | conic (`SocpIpm`) | NLP | `build_conic` can answer, but the CLI's conic route has its own provenance map and that plumbing is not written |
//!
//! # Why the parity test is the load-bearing one
//!
//! `auto_serves_sens_on_the_convex_path` alone would pass if the convex arm
//! returned a plausible number that is not a derivative — which is exactly the
//! class this whole effort has been refusing. What makes it evidence is
//! `both_engines_agree_on_the_same_model`: the same `.nl`, the two independent
//! engines, the same answer. That is the in-tree pattern of
//! `crates/pounce-cli/tests/cblib_vs_nlp.rs`, and it is the only assertion here
//! that could catch a self-consistently wrong step.
//!
//! # Mutation evidence
//!
//! Each row was **run**: the mutation applied, compile-checked (a mutation that
//! does not build emits no failures and reads exactly like one nothing
//! catches), and `--test qp_sens_dispatch --lib --no-fail-fast`.
//!
//! | mutation | red here | note |
//! |---|---|---|
//! | `convex_can_serve_sens` hard-coded `false` | `auto_serves_sens_on_the_convex_path`, `the_banner_reports_the_convex_engine`, `an_explicit_convex_force_is_served_not_skipped` | the whole routing change, reverted |
//! | drop the `!wants_red_hessian` guard | `a_reduced_hessian_request_still_routes_to_the_nlp_path` | and only that, which is the point — narrowing the decline must not widen it |
//! | `resolve_pins` uses the `.nl` constraint index instead of `ConRowMap`'s equality row | `the_nl_constraint_index_is_not_the_equality_row_index` (unit) | `/sens-review` entry 1 in this arm's space |
//! | the unit-coefficient check disabled | `a_non_unit_pin_row_is_refused` (unit) | |
//! | the parameter delta's sign flipped | `auto_serves_sens_on_the_convex_path`, `an_explicit_convex_force_is_served_not_skipped`, `both_engines_agree_on_the_same_model` | the parity test earning its place |
//! | `presolve_on && !convex_can_serve_sens` → `presolve_on` | `serving_sens_switches_the_convex_presolve_off` | **this one was green on the first run.** The guard for the phase's stated largest risk had no failing test; the test was written to close that, and it asserts what was measured (an accuracy cost) rather than what had been assumed (a broken index space) |
//!
//! Fixture `convex_qp_sens.nl` is issue #196's reproduction — a pure convex QP
//!   min (x - p)^2 + y^2   s.t.   p == 1.0
//! carrying the three sIPOPT suffixes, with p -> 1.5. The analytic sensitivity
//! is dx*/dp = 1, so the perturbed primal has x -> 1.5.
//!
//! # What this file is NOT evidence about
//!
//! - **The convex step's numerics.** `crates/pounce-convex/tests/` owns those;
//!   here the step is a black box behind the CLI.
//! - **What presolve would do to the *active set*.** A run that serves a
//!   sensitivity request switches the convex presolve off.
//!   `serving_sens_switches_the_convex_presolve_off` pins that and pins the
//!   accuracy it buys, but the reason the guard exists is a question this
//!   fixture cannot answer: presolve fixes the pinned parameter and drops its
//!   row, and whether a reconstructed bound multiplier can move the active set
//!   the sensitivity infers needs a model where that active set is nontrivial.
//!   This one's is not.
//! - **Any model with more than one parameter**, or with a pin whose row the
//!   extractor reorders. `convex_sens`'s unit tests own the index-space rule;
//!   this file drives one fixture end to end.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("convex_qp_sens.nl");
    p
}

/// Copy the fixture next to a fresh temp path so each test writes its own
/// sibling `.sol` (the AMPL convention) without racing the others.
fn staged_nl(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("pounce_issue196_{tag}"));
    std::fs::create_dir_all(&dir).expect("mkdir temp");
    let dst = dir.join("convex_qp_sens.nl");
    std::fs::copy(fixture(), &dst).expect("copy fixture");
    // Remove any stale .sol from a previous run.
    let _ = std::fs::remove_file(dir.join("convex_qp_sens.sol"));
    dst
}

/// Parse the `sens_sol_state_1` real-var suffix block out of a `.sol` file.
/// Returns index -> value for the listed entries, or None if the suffix is
/// absent.
fn parse_sens_sol_state_1(sol: &str) -> Option<HashMap<usize, f64>> {
    let mut lines = sol.lines();
    while let Some(line) = lines.next() {
        if let Some(rest) = line.strip_prefix("suffix ") {
            // AMPL `.sol` suffix header:
            //   "<kind> <nvalues> <namelen> <tablen> <tabline>"
            // then the suffix name, any table lines, then nvalues "<idx> <val>".
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if parts.len() < 5 {
                continue;
            }
            let count: usize = parts[1].parse().ok()?;
            let tabline: usize = parts[4].parse().ok()?;
            let name = lines.next()?.trim().to_string();
            if name != "sens_sol_state_1" {
                // Skip this suffix's table + value lines and keep scanning.
                for _ in 0..(tabline + count) {
                    lines.next();
                }
                continue;
            }
            for _ in 0..tabline {
                lines.next();
            }
            let mut out = HashMap::new();
            for _ in 0..count {
                let l = lines.next()?;
                let mut it = l.split_whitespace();
                let idx: usize = it.next()?.parse().ok()?;
                let val: f64 = it.next()?.parse().ok()?;
                out.insert(idx, val);
            }
            return Some(out);
        }
    }
    None
}

/// `auto` must honor the sensitivity request by routing to the NLP path and
/// writing `sens_sol_state_1` with the correct perturbed primal (x -> 1.5).
/// Run `pounce` on a staged copy of the fixture and return `(stderr, sol)`.
fn run(tag: &str, args: &[&str]) -> (String, String) {
    let nl = staged_nl(tag);
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(&nl);
    for a in args {
        cmd.arg(a);
    }
    let out = cmd.output().expect("spawn pounce");
    assert_eq!(out.status.code(), Some(0), "solve should succeed");
    (
        String::from_utf8_lossy(&out.stderr).into_owned(),
        std::fs::read_to_string(nl.with_extension("sol")).expect("read .sol"),
    )
}

/// The headline. `auto` keeps an LP/QP parametric-step request on the convex
/// path and answers it there — where before it rerouted the whole model to the
/// general NLP engine to get the same number.
#[test]
fn auto_serves_sens_on_the_convex_path() {
    let (stderr, sol) = run("auto", &[]);
    assert!(
        !stderr.contains("routing to the general NLP"),
        "the reroute must no longer fire for a servable request; stderr=\n{stderr}"
    );
    assert!(
        stderr.contains("computes it directly"),
        "the routing change should be announced, not silent; stderr=\n{stderr}"
    );
    let sens = parse_sens_sol_state_1(&sol)
        .expect("sens_sol_state_1 must be present — served on the convex path");
    let x = *sens.get(&0).expect("perturbed x (index 0)");
    assert!(
        (x - 1.5).abs() < 1e-6,
        "dx*/dp = 1 so p 1.0 -> 1.5 gives x* -> 1.5; got {x}"
    );
}

/// And the banner names the engine that actually ran, so a user comparing two
/// versions can see the routing change rather than infer it. Before this, the
/// same input reported the NLP path here.
#[test]
fn the_banner_reports_the_convex_engine() {
    let nl = staged_nl("banner");
    let out = Command::new(pounce_exe())
        .arg(&nl)
        .output()
        .expect("spawn pounce");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.starts_with("Problem class:"))
        .expect("the routing line is printed");
    assert!(
        line.contains("convex QP interior-point"),
        "the run that answers must be the one named; got: {line}"
    );
}

/// **The load-bearing test.** Two independent engines, one model, one answer.
///
/// Every other assertion in this file would pass on a convex step that is
/// self-consistently wrong — it solves the KKT it was handed, and the KKT is
/// the thing that could be wrong. Only a second engine's answer can catch that,
/// and this is the CLI-level form of the cross-arm check
/// `crates/pounce-cli/tests/cblib_vs_nlp.rs` makes for the solve itself.
#[test]
fn both_engines_agree_on_the_same_model() {
    let (_, convex_sol) = run("parity_convex", &[]);
    let (_, nlp_sol) = run("parity_nlp", &["solver_selection=nlp"]);

    let convex = parse_sens_sol_state_1(&convex_sol).expect("convex path writes the suffix");
    let nlp = parse_sens_sol_state_1(&nlp_sol).expect("NLP path writes the suffix");

    assert_eq!(
        convex.len(),
        nlp.len(),
        "the two engines must report the same entries, not merely both report something"
    );
    for (idx, cx) in &convex {
        let nx = nlp
            .get(idx)
            .unwrap_or_else(|| panic!("NLP path is missing index {idx}"));
        assert!(
            (cx - nx).abs() < 1e-7,
            "the two engines disagree at index {idx}: convex {cx}, nlp {nx}"
        );
    }
}

/// An explicit convex force is served too, now that the engine can answer.
/// This is the assertion that flipped: the old contract warned that the request
/// "will be skipped", which would now be a lie.
#[test]
fn an_explicit_convex_force_is_served_not_skipped() {
    let (stderr, sol) = run("qp_ipm", &["solver_selection=qp-ipm"]);
    assert!(
        !stderr.contains("will be skipped"),
        "the forced convex path computes the step now; stderr=\n{stderr}"
    );
    let sens = parse_sens_sol_state_1(&sol)
        .expect("a forced convex solve writes sens_sol_state_1 like any other");
    let x = *sens.get(&0).expect("perturbed x (index 0)");
    assert!((x - 1.5).abs() < 1e-6, "expected x* -> 1.5; got {x}");
}

/// A reduced-Hessian request still reroutes, and that is a capability
/// statement rather than an oversight: `QpSensitivity::reduced_hessian` exists
/// but computes a null-space projection where the CLI's sIPOPT path takes the
/// Schur route. Serving it here would silently change which number
/// `--compute-red-hessian` returns.
#[test]
fn a_reduced_hessian_request_still_routes_to_the_nlp_path() {
    let nl = staged_nl("redhess");
    let out = Command::new(pounce_exe())
        .arg(&nl)
        .arg("--compute-red-hessian")
        .output()
        .expect("spawn pounce");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("routing to the general NLP"),
        "a reduced-Hessian request is still the general path\'s; stderr=\n{stderr}"
    );
}

/// The convex presolve is off for a run that serves a sensitivity request, and
/// this is the test that says so — because nothing else did.
///
/// Found by mutation: removing the guard (`presolve_on && !convex_can_serve_sens`
/// → `presolve_on`) left the whole suite green, which is the same shape of gap
/// the PSD curvature fixture had in the phase before this one. A guard with no
/// failing test is a guard nobody can change safely.
///
/// What it asserts is what was measured, not what was assumed. Presolve does
/// **not** break the pin indices — this driver postsolves back to the
/// extracted-QP space, so the step is still within `1e-6` of the NLP path's
/// with presolve left on. What it costs is accuracy: on this fixture presolve
/// fixes the pinned parameter and drops its row (`3 → 2 vars, 1 → 0 rows`), so
/// the sensitivity reads a postsolve reconstruction rather than the converged
/// KKT, and the step lands `5.0e-11` from the analytic answer instead of
/// `6.2e-15`. The threshold below sits between those two measurements.
#[test]
fn serving_sens_switches_the_convex_presolve_off() {
    let nl = staged_nl("presolve");
    let out = Command::new(pounce_exe())
        .arg(&nl)
        .output()
        .expect("spawn pounce");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.lines().any(|l| l.starts_with("Presolve:")),
        "a run serving a sensitivity request must not presolve; stdout=\n{stdout}"
    );

    let sol = std::fs::read_to_string(nl.with_extension("sol")).expect("read .sol");
    let sens = parse_sens_sol_state_1(&sol).expect("the step is still produced");
    let x = *sens.get(&0).expect("perturbed x (index 0)");
    assert!(
        (x - 1.5).abs() < 1e-12,
        "reading the converged KKT rather than a postsolve reconstruction is worth four \
         orders here: measured 6.2e-15 with the guard, 5.0e-11 without. got |x - 1.5| = {:e}",
        (x - 1.5f64).abs()
    );
}

/// Control / no-regression: the general NLP path is unchanged.
#[test]
fn nlp_path_writes_sens_suffix() {
    let (_, sol) = run("nlp", &["solver_selection=nlp"]);
    let sens = parse_sens_sol_state_1(&sol).expect("sens_sol_state_1 present on NLP path");
    let x = *sens.get(&0).expect("perturbed x (index 0)");
    assert!((x - 1.5).abs() < 1e-6, "expected x* -> 1.5; got {x}");
}

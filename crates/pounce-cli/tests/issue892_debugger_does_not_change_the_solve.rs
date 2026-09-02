//! gh #892 — end to end, attaching the debugger must not change the solve.
//!
//! `crates/pounce-convex/tests/issue892_debug_routes_like_the_plain_solve.rs`
//! pins the library-level routing. This file pins what the reporter actually
//! ran: the `pounce` binary, with and without a `--debug-script` containing
//! nothing but `continue`, which is a pure no-op on the trajectory.
//!
//! It exists because the CLI carried the *second* half of the defect, and the
//! library legs cannot see it. Until this issue the convex entry points
//! skipped presolve while a debugger was attached — deliberately, so the
//! inspected blocks were the user's rows rather than a reduced set — which
//! meant the debugged run solved a different, smaller problem. Both halves
//! have to be closed for "attaching the debugger changes nothing" to be true
//! of the command the user types, and only a process-level test says so.
//!
//! What each check reaches, in the terms `CLAUDE.md` sets for branch coverage:
//!
//! | check | path | the half it covers |
//! | --- | --- | --- |
//! | `conic_qcqp_is_unmoved_by_the_debugger` | `solver_selection=socp` | driver substitution (the reported symptom) |
//! | `conic_qcqp_is_unmoved_under_auto` | `solver_selection=auto` | the same, plus the NLP fallback that masked it |
//! | `presolve_still_runs_under_the_debugger` | conic, presolve reduces | the CLI carve-out |
//! | `convex_qp_is_unmoved_by_the_debugger` | `solver_selection=qp-ipm` | the LP/QP twin of the same carve-out |
//!
//! **Mutation table** — measured, not asserted:
//!
//! | mutation | red |
//! | --- | --- |
//! | reinstate the CLI presolve carve-out (a `debug_hook` arm ahead of `presolve_on` in both `run_convex_qp` and `run_convex_socp`) | `presolve_still_runs_under_the_debugger`, `convex_qp_is_unmoved_by_the_debugger` |
//! | `solve_socp_ipm_debug` builds its own iteration for symmetric cones | `conic_qcqp_is_unmoved_by_the_debugger`, `conic_qcqp_is_unmoved_under_auto` |
//!
//! The two halves are independent, and each mutation leaves the other half's
//! checks green — which is the argument for keeping both files.
//!
//! Not evidence about: the NLP filter-IPM (which never had the carve-out),
//! the active-set engine (where the debugger deliberately does not engage and
//! the caller says so), or `--debug-json`.

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

/// A `--debug-script` holding the issue's script: `continue`, and nothing
/// else. Written under the target dir so the test needs no fixture of its own.
fn continue_script() -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pounce_issue892_continue_{}.pdbg",
        std::process::id()
    ));
    let mut f = std::fs::File::create(&p).expect("write debug script");
    writeln!(f, "continue").expect("write debug script");
    p
}

/// The verdict line the CLI prints, reduced to the three fields a trajectory
/// change shows up in: status, objective and iteration count. The wall-clock
/// suffix is stripped — it moves run to run and says nothing about the solve.
///
/// e.g. `POUNCE (convex QCQP conic IPM, pounce-convex): Optimal Solution
/// Found.  obj=0.46851408  iters=38  (0.049s)`
fn verdict(stdout: &str) -> String {
    let line = stdout
        .lines()
        .find(|l| l.starts_with("POUNCE (") && l.contains("iters="))
        .unwrap_or_else(|| panic!("no verdict line in stdout:\n{stdout}"));
    let cut = line.rfind("  (").unwrap_or(line.len());
    line[..cut].to_string()
}

/// Run the binary on `model` with `opts`, optionally with the debugger
/// attached. Returns `(verdict line, exit code)`.
fn run(model: &str, opts: &[&str], script: Option<&PathBuf>) -> (String, Option<i32>) {
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture(model)).arg("--no-sol");
    for o in opts {
        cmd.arg(o);
    }
    if let Some(s) = script {
        cmd.arg("--debug-script").arg(s);
    }
    cmd.stdin(std::process::Stdio::null());
    let out = cmd.output().expect("spawn pounce");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    (verdict(&stdout), out.status.code())
}

/// Plain and debugged runs must agree on status, objective, iteration count
/// **and exit code**. The iteration count is the sensitive field: gh #892's
/// two `Optimal` instances carried the same substituted driver as its two
/// failing ones, and only the count showed it.
fn assert_unmoved(model: &str, opts: &[&str]) {
    let script = continue_script();
    let (plain, plain_rc) = run(model, opts, None);
    let (dbg, dbg_rc) = run(model, opts, Some(&script));
    let _ = std::fs::remove_file(&script);
    assert_eq!(
        plain, dbg,
        "{model} {opts:?}: the debugger changed the solve\n  plain: {plain}\n  debug: {dbg}"
    );
    assert_eq!(
        plain_rc, dbg_rc,
        "{model} {opts:?}: the debugger changed the exit code ({plain_rc:?} vs {dbg_rc:?})"
    );
}

/// The reported symptom. On `d32204e` the plain run returned `Optimal` and the
/// debugged one `Numerical failure (no verified KKT point)` with exit 1,
/// because the debug entry point ran the direct IPM where the plain one ran
/// the HSDE embedding.
#[test]
fn conic_qcqp_is_unmoved_by_the_debugger() {
    assert_unmoved("qcqp_ball.nl", &["solver_selection=socp"]);
    assert_unmoved("qcqp_columns_illcond.nl", &["solver_selection=socp"]);
}

/// Under `auto` the same conic failure was *masked* by the NLP fallback, at
/// the cost of a silent second solve — so the run disagreed with the plain one
/// in engine and iteration count even where the objective survived.
#[test]
fn conic_qcqp_is_unmoved_under_auto() {
    assert_unmoved("qcqp_ball.nl", &["solver_selection=auto"]);
}

/// The CLI half. `qcqp_shared_linear_rows.nl` is the fixture whose rows
/// `presolve_conic` is the correct entry point for, so presolve has something
/// to do on it; before this issue the debugged run skipped the reduction and
/// solved a different problem. Checked both ways round, because
/// `qp_presolve=no` is the documented escape hatch for inspecting unreduced
/// blocks and it has to agree too.
#[test]
fn presolve_still_runs_under_the_debugger() {
    assert_unmoved("qcqp_shared_linear_rows.nl", &["solver_selection=socp"]);
    assert_unmoved(
        "qcqp_shared_linear_rows.nl",
        &["solver_selection=socp", "qp_presolve=no"],
    );
}

/// The LP/QP entry point, which carried the same CLI carve-out. The library
/// legs report this path as already agreeing on the *driver*; presolve is the
/// part only a process-level run can see.
#[test]
fn convex_qp_is_unmoved_by_the_debugger() {
    // `convex_qp_share1b.nl` is the reducible one — presolve takes it
    // 225 -> 208 vars and 117 -> 98 rows, so the debugged run was solving a
    // visibly different problem. `bound_active_qp.nl` reduces by nothing,
    // and is here as the control: the two runs have to agree there too, for
    // a reason that has nothing to do with presolve.
    assert_unmoved("convex_qp_share1b.nl", &["solver_selection=qp-ipm"]);
    assert_unmoved(
        "convex_qp_share1b.nl",
        &["solver_selection=qp-ipm", "qp_presolve=no"],
    );
    assert_unmoved("bound_active_qp.nl", &["solver_selection=qp-ipm"]);
}

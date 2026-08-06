//! `presolve_linear_eq_reduction` end to end through the `.nl` path
//! (issue #487).
//!
//! The fixture is `min (x0−1)² + (x2−3)²` subject to `x0 − 2·x1 = 0` and
//! `x1² + x2² = 2`, over `x2 ∈ [0, 10]`, `x0, x1 ∈ [−10, 10]`. The first
//! row is a free/free two-variable linear equality — no bound pins either
//! column — so it is exactly the shape the reduction exists for, and one
//! column plus one row must disappear.
//!
//! What is actually at stake here is not the reduction but the *reporting*.
//! AMPL and Pyomo read the `.sol` primal and dual blocks positionally
//! against the originating `.nl`, so a solver that quietly hands back a
//! shorter vector produces silently misattributed values rather than an
//! error. Both blocks must come back at full length, in the original
//! order, with the eliminated row's multiplier recovered rather than
//! zeroed.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

struct Run {
    stdout: String,
    /// The `.sol` value block, in `.nl` order: duals first, then primals.
    values: Vec<f64>,
}

fn run(tag: &str, opts: &[&str]) -> Run {
    let dir = std::env::temp_dir().join(format!("pounce_lin_eq_reduction_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let mut fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fixture.push("tests/fixtures/linear_eq_aggregation.nl");
    let nl = dir.join("m.nl");
    std::fs::copy(&fixture, &nl).expect("copy fixture");

    let mut cmd = Command::new(pounce_exe());
    cmd.current_dir(&dir).arg("m").arg("-AMPL");
    for o in opts {
        cmd.arg(o);
    }
    let out = cmd.output().expect("run pounce");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let sol = std::fs::read_to_string(dir.join("m.sol")).expect("read .sol");

    // Everything numeric before `objno` is the `Options` preamble followed
    // by the dual block then the primal block. The fixture has 2 rows and
    // 3 columns, so the trailing five entries are the ones under test.
    let numeric: Vec<f64> = sol
        .lines()
        .take_while(|l| !l.starts_with("objno"))
        .filter_map(|l| l.trim().parse::<f64>().ok())
        .collect();
    assert!(
        numeric.len() >= 5,
        "expected 2 duals + 3 primals in the .sol body, got {numeric:?}\n{sol}"
    );
    let values = numeric[numeric.len() - 5..].to_vec();
    Run { stdout, values }
}

#[test]
fn the_reduction_reports_what_it_removed() {
    let on = run("on", &["presolve=yes", "presolve_linear_eq_reduction=yes"]);
    assert!(
        on.stdout.contains("eliminated 1 columns"),
        "no reduction summary on stdout:\n{}",
        on.stdout
    );
    assert!(
        on.stdout.contains("Optimal Solution Found"),
        "{}",
        on.stdout
    );
}

#[test]
fn the_option_is_off_by_default() {
    let on = run("default", &["presolve=yes"]);
    assert!(
        !on.stdout.contains("linear-equality reduction"),
        "the reduction ran without being asked for:\n{}",
        on.stdout
    );
}

#[test]
fn the_sol_file_keeps_the_original_shape_and_values() {
    let base = run("base", &[]);
    let reduced = run(
        "reduced",
        &["presolve=yes", "presolve_linear_eq_reduction=yes"],
    );

    assert_eq!(
        base.values.len(),
        reduced.values.len(),
        "the .sol body changed length: base={:?} reduced={:?}",
        base.values,
        reduced.values
    );
    for (k, (b, r)) in base.values.iter().zip(reduced.values.iter()).enumerate() {
        assert!(
            (b - r).abs() < 1e-6,
            ".sol entry {k} diverged: base={b} reduced={r}"
        );
    }

    // Entries 0..2 are the duals. The first row is the one the reduction
    // consumed; a zero there would mean the recovery silently gave up.
    assert!(
        reduced.values[0].abs() > 1e-6,
        "the consumed row came back with a zero multiplier: {:?}",
        reduced.values
    );

    // Entries 2..5 are the primals in `.nl` order (v0 = x2, v1 = x1,
    // v2 = x0), so the consumed row `x0 = 2·x1` must hold among them.
    let (x2, x1, x0) = (reduced.values[2], reduced.values[3], reduced.values[4]);
    assert!(
        (x0 - 2.0 * x1).abs() < 1e-9,
        "the eliminated row is violated in the reported point: {x0} != 2 * {x1}"
    );
    assert!(
        (x1 * x1 + x2 * x2 - 2.0).abs() < 1e-6,
        "the surviving row is violated: x1={x1} x2={x2}"
    );
}

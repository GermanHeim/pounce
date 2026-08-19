//! gh#588 (Q9b) — the convex-path presolve switch reaches the **conic**
//! driver, and reaches it through the cone-aware entry point.
//!
//! Q1 extracted `presolve_conic` and then deliberately did not wire it into
//! `run_convex_socp`: "with no benchmark instance on the conic path the change
//! is unmeasurable, and the safety work it requires … is already scoped into
//! Q9". The consequence was that `qp_presolve` — a registered, documented
//! option — was silently ignored on every convex QCQP. This is that wiring.
//!
//! Per gh#551 each claim is shown by driving the real CLI rather than by
//! inspecting a field assignment. Two things have to be true:
//!
//! 1. **The switch is live on the conic driver.** At the parent commit no
//!    presolve runs on this route at all, so `default_reduces_a_conic_model`
//!    fails there for exactly that reason: no `Presolve:` line is ever
//!    printed, whatever the option says.
//! 2. **The reduction does not corrupt a cone.** The fixture is built so that
//!    an unprotected merge *would* corrupt one — see
//!    `crates/pounce-convex/tests/presolve_conic_quadratic_rows.rs`, which
//!    shows that same shape returning −17.35 for a problem whose optimum is
//!    −10.40 — so an objective that survives the reduction is evidence, not
//!    decoration.
//!
//! The witness is `qcqp_shared_linear_rows`, added with this test. It is the
//! only conic fixture in the corpus that presolve can act on at all: `airport`
//! and `qcqp_ball` are pure box-plus-cone models with **no orthant rows**, so
//! there is nothing for a row reduction to remove. See the third test.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(format!("{name}.nl"));
    p
}

fn tmp_path(tag: &str, ext: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pounce_gh588_q9b_{}_{seq}_{tag}.{ext}",
        std::process::id()
    ));
    p
}

/// Solve `model` with `opts` appended verbatim, returning the CLI's stdout.
/// The `Presolve:` line is stdout-only — `SolveReport` carries no presolve
/// block — so the transcript is what has to be read.
fn run(model: &str, opts: &[&str]) -> String {
    let sol = tmp_path(model, "sol");
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture(model)).arg("--sol-output").arg(&sol);
    for o in opts {
        cmd.arg(o);
    }
    let out = cmd.output().expect("spawn pounce");
    let _ = std::fs::remove_file(&sol);
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// The status line's `obj=` field, which is where the conic driver reports.
fn objective(stdout: &str) -> f64 {
    let line = stdout
        .lines()
        .find(|l| l.starts_with("POUNCE (") && l.contains("conic IPM"))
        .unwrap_or_else(|| panic!("no conic status line in:\n{stdout}"));
    let tok = line
        .split_whitespace()
        .find(|t| t.starts_with("obj="))
        .expect("no obj= field");
    tok["obj=".len()..].parse().expect("parse obj")
}

/// The analytic optimum of `qcqp_shared_linear_rows`. Maximizing
/// `s = x0 + x1` subject to `2x0² ≤ 20 − s` and `2x1² ≤ 20 − s` gives
/// `s² + 2s − 40 ≤ 0`, so `s = √41 − 1`; with `x2 + x3 = 5` the objective is
/// `−(√41 − 1) − 5 = −10.403124…`.
fn true_optimum() -> f64 {
    -(41f64.sqrt() - 1.0) - 5.0
}

/// Claim 1: the switch is live on the conic driver.
///
/// The model carries three orthant rows that are scalar multiples of one
/// another (`x2 + x3 ≤ 5` twice, and `2x2 + 2x3 ≤ 10`), so a working presolve
/// must drop two of them — 9 → 7 inequality rows. At the parent commit
/// `run_convex_socp` does not presolve at all and this assertion fails on an
/// absent line.
#[test]
fn the_conic_driver_honours_the_presolve_switch() {
    let on = run("qcqp_shared_linear_rows", &[]);
    let line = on
        .lines()
        .find(|l| l.starts_with("Presolve:"))
        .unwrap_or_else(|| {
            panic!("the conic driver ignored qp_presolve — no Presolve: line in:\n{on}")
        });
    assert!(
        line.contains("9 → 7 rows"),
        "expected the two parallel orthant rows to go, got: {line}"
    );

    let off = run("qcqp_shared_linear_rows", &["qp_presolve=no"]);
    assert!(
        !off.lines().any(|l| l.starts_with("Presolve:")),
        "qp_presolve=no still presolved:\n{off}"
    );
}

/// Claim 2: the reduction reaches the right answer, on both settings.
///
/// `qcqp_shared_linear_rows` has two quadratic rows — `2x0² + x0 + x1 ≤ 20`
/// and `2x1² + x0 + x1 ≤ 20` — that differ **only** in `Q`. Their cone blocks'
/// first two rows are byte-identical, which is the §7 collision. A presolve
/// that merged them would report `Optimal` with the objective 67% off, so
/// pinning the objective here is pinning the protection.
#[test]
fn the_shared_linear_part_does_not_lose_a_quadratic_row() {
    let with = objective(&run("qcqp_shared_linear_rows", &[]));
    let without = objective(&run("qcqp_shared_linear_rows", &["qp_presolve=no"]));
    for (label, got) in [("presolved", with), ("bare", without)] {
        assert!(
            (got - true_optimum()).abs() < 1e-6,
            "{label} objective {got} != analytic optimum {}",
            true_optimum()
        );
    }
}

/// Claim 3, and the honest limit of this phase: on the two conic fixtures
/// that predate it, the wiring changes nothing, because there is nothing for
/// it to change. `qcqp_ball` is a two-variable ball-constrained QP — one cone
/// block, a box, and **no orthant rows at all** — so every row is protected by
/// construction and presolve has an empty catalog to work with.
///
/// This is pinned rather than left implicit because it is the measurement
/// behind the phase's claim that the wiring is unmeasurable on the real
/// corpus: the reduction is not declining to fire, it has nothing to fire on.
#[test]
fn a_cone_only_model_is_untouched_by_the_wiring() {
    let out = run("qcqp_ball", &[]);
    assert!(
        !out.lines().any(|l| l.starts_with("Presolve:")),
        "qcqp_ball has no orthant rows; nothing should have been reduced:\n{out}"
    );
    assert!(
        (objective(&out) - (-10.0)).abs() < 1e-6,
        "qcqp_ball objective moved: {}",
        objective(&out)
    );
}

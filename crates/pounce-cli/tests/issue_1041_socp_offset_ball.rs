//! Regression for pounce#1041: the `.nl` → SOCP conic path stalled at an
//! *infeasible* point on a convex QCQP whose ball sits far from the origin.
//!
//! `qcqp_offset_ball_1d.nl` is `min x s.t. (x − 1000)² ≤ 1`, so `x* = 999`.
//! The reformulation used to put `½xᵀQx + aᵀx + b ≤ 0` into the rotated cone
//! as `(s+1, s−1, √2·Fx)` with `s = −(aᵀx + b)`: at the optimum `‖Fx‖²` and
//! `2s` are both ~2·10⁶ and the constraint is their difference, so the cone
//! was pinned to its boundary to six digits. The conic IPM stopped at
//! `x ≈ 1003.41` with a numerical failure (exit 1). The cone is now centred
//! by completing the square, which turns the same row into `(2, 0, 2(x − 1000))`.
//!
//! `qcqp_offset_ellipse_coupled.nl` reaches the *other* branch of the
//! centring, the normal-equations solve for a non-diagonal `Q`, and keeps a
//! linear term outside `range(Q)` (the `−z`) so the `a_⊥` half is exercised
//! too: `min x + 2y + z s.t. (x + y − 2000)² + 3(x − y)² − z ≤ 1, z ≥ 0`.
//! Substituting `u = x + y − 2000`, `v = x − y`, the objective is
//! `3000 + 1.5u − 0.5v + z` over `u² + 3v² ≤ 1 + z`; at `z = 0` its minimum is
//! `3000 − √(1.5² + 0.5²/3) = 3000 − √(7/3)`, and the marginal value of `z`
//! (`1 − √(7/3)/2 > 0`) keeps `z* = 0`.

use pounce_solve_report::SolveReport;
use std::path::PathBuf;
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

fn solve(name: &str, selection: &str) -> (SolveReport, Option<i32>, String) {
    let json = std::env::temp_dir().join(format!("pounce_issue_1041_{name}_{selection}.json"));
    let _ = std::fs::remove_file(&json);
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(fixture(name))
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .arg(format!("solver_selection={selection}"))
        .output()
        .expect("spawn pounce");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let text = std::fs::read_to_string(&json)
        .unwrap_or_else(|e| panic!("read JSON report ({e}); stdout=\n{stdout}"));
    let _ = std::fs::remove_file(&json);
    let report = serde_json::from_str(&text).expect("parse JSON report");
    (report, out.status.code(), stdout)
}

fn assert_conic_solves(name: &str, x_star: &[f64], f_star: f64) {
    for selection in ["socp", "auto"] {
        let (report, code, stdout) = solve(name, selection);
        assert_eq!(
            code,
            Some(0),
            "{name} [{selection}] must exit 0; stdout=\n{stdout}"
        );
        let status = format!("{:?}", report.solution.status);
        assert_eq!(
            status, "SolveSucceeded",
            "{name} [{selection}]; stdout=\n{stdout}"
        );
        // Under `auto` a conic failure would be rescued by the NLP fallback,
        // which is exactly what must not be needed here.
        assert!(
            stdout.contains("pounce-convex") && !stdout.contains("re-solved on the general NLP"),
            "{name} [{selection}] must be answered by the conic IPM; stdout=\n{stdout}"
        );
        let x = &report.solution.x;
        for (i, (xi, xs)) in x.iter().zip(x_star).enumerate() {
            assert!(
                (xi - xs).abs() < 1e-5,
                "{name} [{selection}] x[{i}] = {xi}, expected {xs}"
            );
        }
        assert!(
            (report.solution.objective - f_star).abs() < 1e-6,
            "{name} [{selection}] objective {} expected {f_star}",
            report.solution.objective
        );
    }
}

#[test]
fn ball_far_from_origin_solves_on_the_conic_path() {
    assert_conic_solves("qcqp_offset_ball_1d.nl", &[999.0], 999.0);
}

#[test]
fn coupled_ellipse_far_from_origin_solves_on_the_conic_path() {
    let r = (7.0_f64 / 3.0).sqrt();
    // The minimizer of gᵀ(u, v), g = (1.5, −0.5), on u² + 3v² = 1 is
    // −D⁻¹g/r with D = diag(1, 3): (u, v) = (−1.5, 0.5/3)/r.
    let (u, v) = (-1.5 / r, 0.5 / 3.0 / r);
    let x = (u + 2000.0 + v) / 2.0;
    let y = (u + 2000.0 - v) / 2.0;
    assert_conic_solves("qcqp_offset_ellipse_coupled.nl", &[x, y, 0.0], 3000.0 - r);
}

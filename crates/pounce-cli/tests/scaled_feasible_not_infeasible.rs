//! A feasible model whose rows carry heavy scaling must not be reported
//! infeasible.
//!
//! Regression for a defect introduced by the #376 fix. That change made the
//! restoration locally-infeasible gates measure the constraint violation
//! **unscaled**, which was correct — the floors they compare against
//! (`max(100*tol, 1e-4)`) are absolute, user-facing magnitudes. But
//! `inner_kkt_err` and the inner convergence test remain in the *scaled* space,
//! so the fix moved the units mismatch rather than removing it.
//!
//! On a model whose rows are scaled down, the inner restoration can stop at a
//! point it considers feasible — scaled violation below `tol` — while the
//! unscaled violation still clears the `1e-4` floor. A gate then fires and a
//! feasible model is reported `Infeasible_Problem_Detected`. `scaled_feasible_a`
//! made the mechanism plain: `inner_kkt_err = 3.249625e-9` against
//! `orig_inf_pr = 3.249625e-3` — the same mantissa, scaled by `1e6`, the row
//! scaling factor.
//!
//! Both fixtures are verified feasible by `pounce verify` with exactly `0.000e0`
//! constraint and bound violation, and POUNCE's own convex QP route solves both
//! to optimality. Before #376 they returned `Solve_Succeeded` (a) and
//! `Solved_To_Acceptable_Level` (b); after #376 both returned 200.
//!
//! The fix guards all five gates in one place: never claim infeasibility at a
//! point the solver's own convergence test would accept as feasible.

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

fn solve_result_num(text: &str) -> i32 {
    for line in text.lines() {
        if let Some(rest) = line.trim().strip_prefix("objno ") {
            if let Some(code) = rest.split_whitespace().nth(1) {
                return code.parse().expect("objno code parses");
            }
        }
    }
    panic!("no `objno` line in .sol:\n{text}");
}

fn solve(model: &str, tag: &str) -> String {
    let sol = std::env::temp_dir().join(format!("pounce_scaled_feas_{tag}.sol"));
    let _ = std::fs::remove_file(&sol);
    let out = Command::new(pounce_exe())
        .arg(fixture(model))
        .arg("-AMPL")
        .arg("--sol-output")
        .arg(&sol)
        .arg("print_level=0")
        .arg("solver_selection=nlp")
        .arg("presolve=no")
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(0), "-AMPL must exit 0");
    std::fs::read_to_string(&sol).expect("read .sol")
}

#[test]
fn heavily_scaled_feasible_models_are_not_reported_infeasible() {
    for (model, tag) in [("scaled_feasible_a.nl", "a"), ("scaled_feasible_b.nl", "b")] {
        let text = solve(model, tag);
        let srn = solve_result_num(&text);
        assert!(
            !(200..300).contains(&srn),
            "{model} is feasible — `pounce verify` reports 0.000e0 constraint \
             and bound violation on the convex route's solution — but the NLP \
             path reported the AMPL infeasible band (solve_result_num={srn}). \
             This is the scaled/unscaled units mismatch: the inner stops at a \
             point it considers feasible in scaled terms while the unscaled \
             violation clears the 1e-4 floor.\n{text}"
        );
    }
}

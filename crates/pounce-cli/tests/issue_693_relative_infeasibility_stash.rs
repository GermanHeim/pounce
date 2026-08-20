//! The acceptable-point stash must never hold a *relatively* infeasible point.
//!
//! `min x²  s.t.  1e-8·x >= 2e-8,  0 <= x <= 1` is `x >= 2` over `x ∈ [0, 1]`
//! with every row multiplied by `1e-8`. Multiplying a row by a positive
//! constant leaves the feasible set exactly unchanged, so the verdict must be
//! `Infeasible_Problem_Detected` at this scale exactly as it is at unit scale.
//!
//! It was not. `OptErrorConvCheck::current_is_acceptable_with_state` gates the
//! stash on the scale-relative feasibility measure — a row violated by more
//! than `relative_viol_threshold` of its own magnitude is not an acceptable
//! point and must not become a rollback target — but that gate carried
//! `VETO_MAX_EXTRA_ITERS`, the *certificate* veto's iteration budget. The
//! budget exists to bound how long a veto can keep a run alive; declining to
//! stash extends the run by nothing, so on this gate it bounded no cost and
//! only ever expired. Once it did, the 99.998%-violated iterate went into the
//! stash, and `ConvergenceStatus::LocallyInfeasible` — which consults the
//! stash (gh #505) — handed it back as `Solved_To_Acceptable_Level`.
//!
//! Reached here through the `feral_scaling=mc64` rung of the second-opinion
//! ladder, which needs 288 iterations on this model and so blows the 60-
//! iteration budget; `feral_scaling=identity` reaches it directly. Both are
//! pinned below, because a fix that only satisfies the default path leaves the
//! same wrong answer one option away.
//!
//! `solver_selection=nlp` on purpose: the convex-QP presolve proves this model
//! infeasible outright, which is correct and is what a default run gets — and
//! would hide the NLP-path defect this test exists for.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("infeasible_row_scaled_1em8.nl");
    p
}

/// AMPL `solve_result_num`: 200..=299 is the infeasible band.
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

fn solve(tag: &str, extra: &[&str]) -> i32 {
    let sol = std::env::temp_dir().join(format!("pounce_gh693_stash_{tag}.sol"));
    let _ = std::fs::remove_file(&sol);

    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("-AMPL")
        .arg("--sol-output")
        .arg(&sol)
        .arg("solver_selection=nlp")
        .arg("print_level=0")
        .args(extra)
        .output()
        .expect("pounce runs");
    assert!(
        out.status.success(),
        "pounce exited {:?}\nstderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );

    let text = std::fs::read_to_string(&sol).expect("a .sol was written");
    let _ = std::fs::remove_file(&sol);
    solve_result_num(&text)
}

#[test]
fn a_row_scaled_infeasibility_is_not_reported_solved_to_acceptable_level() {
    for (tag, extra) in [
        ("default", &[][..]),
        ("identity", &["feral_scaling=identity"][..]),
        ("mc64", &["feral_scaling=mc64"][..]),
        ("infnorm", &["feral_scaling=infnorm"][..]),
    ] {
        let code = solve(tag, extra);
        assert!(
            (200..300).contains(&code),
            "`1e-8·x >= 2e-8` over `x ∈ [0, 1]` is the same empty feasible set \
             at every row scaling, but {tag} reported solve_result_num={code} \
             (200..299 is the infeasible band). A code under 200 means the \
             acceptable-point stash handed back a point whose only row is \
             violated by ~100% of its own magnitude."
        );
    }
}

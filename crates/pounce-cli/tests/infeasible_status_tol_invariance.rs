//! An infeasible model's *verdict* must not depend on the user's `tol`.
//!
//! Follow-up to gh #372, which fixed one instance of this: a one-variable
//! contradiction reported `Restoration_Failed` at `tol=1e-10` but
//! `Infeasible_Problem_Detected` at the default `tol=1e-8`. Sweeping `tol`
//! across the other infeasible fixtures found a second, independent instance
//! in `infeasible_equalities.nl` — pre-existing, and triggered from the
//! opposite end:
//!
//! ```text
//!   tol >= 3e-7   ->  Error_In_Step_Computation   (AMPL 500)
//!   tol <= 1e-7   ->  Infeasible_Problem_Detected (AMPL 200)
//! ```
//!
//! Non-monotonic (`1e-7` passed, `3e-7` failed), and the 500 range surfaces
//! through Pyomo as `internalSolverError` — indistinguishable from a solver
//! bug on a model whose true constraint violation is `2.0`.
//!
//! Root cause was a units mismatch in two places. The constraint violation was
//! measured **scaled** (`eval_c` returns `dc ⊙ c_user`; `curr_constraint_violation`
//! likewise) but compared against *absolute* floors — `max(100·outer_tol, 1e-4)`.
//! Those floors are user-facing magnitudes meaning "the violation is
//! meaningfully nonzero", so the comparison mixed unit systems. NLP scaling
//! shrinks this fixture's rows by ~3e6, so a violation of `2.0` read as
//! `6.67e-7` and could never clear a `1e-4` floor.
//!
//! Two sites had to change, because the fixture is a *square* 2x2 system:
//!   1. `resto_inner_solver::eval_orig_inf_pr_at_inner_curr` — feeds all the
//!      restoration locally-infeasible gates.
//!   2. `ipopt_alg`'s outer `cycle_exit` — square problems are carved out of
//!      the `strict` restoration gate, so this cycle exit is their only
//!      locally-infeasible safety net, and the unit mismatch had disabled it.
//!
//! The tests below pin the invariant rather than the specific tolerances, so
//! they keep their value if the detection path is reworked again.

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

fn solve_at_tol(model: &str, tag: &str, tol: &str) -> String {
    let sol = std::env::temp_dir().join(format!("pounce_inf_tol_{tag}.sol"));
    let _ = std::fs::remove_file(&sol);

    let out = Command::new(pounce_exe())
        .arg(fixture(model))
        .arg("-AMPL")
        .arg("--sol-output")
        .arg(&sol)
        .arg("print_level=0")
        .arg(format!("tol={tol}"))
        .output()
        .expect("spawn pounce");

    assert_eq!(out.status.code(), Some(0), "-AMPL must exit 0");
    std::fs::read_to_string(&sol).expect("read .sol")
}

/// `tol` values spanning four orders of magnitude either side of the default,
/// including `3e-7` — the value that failed while its neighbour `1e-7` passed.
const TOLS: &[(&str, &str)] = &[
    ("t3", "1e-3"),
    ("t4", "1e-4"),
    ("t5", "1e-5"),
    ("t6", "1e-6"),
    ("t3e7", "3e-7"),
    ("t7", "1e-7"),
    ("t8", "1e-8"),
    ("t10", "1e-10"),
    ("t12", "1e-12"),
];

/// The regression this fixes: a square, badly-scaled, genuinely infeasible
/// system must land in the AMPL infeasible range at every `tol`.
#[test]
fn infeasible_equalities_verdict_is_tol_invariant() {
    for (tag, tol) in TOLS {
        let text = solve_at_tol("infeasible_equalities.nl", &format!("eq_{tag}"), tol);
        let srn = solve_result_num(&text);
        assert!(
            (200..300).contains(&srn),
            "tol={tol}: expected the AMPL infeasible range (200..299), got \
             solve_result_num={srn}. This model's true constraint violation is \
             2.0; NLP scaling reports it as ~6.67e-7. A verdict that depends on \
             the user's tolerance makes an infeasible model look like an \
             internal solver error to Pyomo:\n{text}"
        );
    }
}

/// The gh #372 shape, swept over the same range — guards the first instance
/// from the opposite direction than `issue_372_infeasible_bounds_status.rs`
/// covers.
#[test]
fn issue_372_bounds_verdict_is_tol_invariant() {
    for (tag, tol) in TOLS {
        let text = solve_at_tol("issue_372_infeasible_bounds.nl", &format!("b_{tag}"), tol);
        let srn = solve_result_num(&text);
        assert!(
            (200..300).contains(&srn),
            "tol={tol}: expected the AMPL infeasible range (200..299), got \
             solve_result_num={srn}:\n{text}"
        );
    }
}

/// Whatever the tolerance, an infeasible model must never be advertised in the
/// AMPL *solved* family — that is what makes Pyomo load the returned point as
/// `optimal`. Separate from the range assertion above so a future change that
/// trades one wrong verdict for a worse one is caught explicitly.
#[test]
fn infeasible_models_are_never_reported_solved() {
    for model in [
        "infeasible_equalities.nl",
        "issue_372_infeasible_bounds.nl",
        "infeasible_qp.nl",
        "inconsistent_eq_qp.nl",
    ] {
        for (tag, tol) in TOLS {
            let stem = model.trim_end_matches(".nl");
            let text = solve_at_tol(model, &format!("ns_{stem}_{tag}"), tol);
            let srn = solve_result_num(&text);
            assert!(
                !(0..200).contains(&srn),
                "{model} at tol={tol} reported in the AMPL solved family \
                 (solve_result_num={srn}); 0..99 = solved and 100..199 = \
                 solved-with-warning both make Pyomo report \
                 TerminationCondition.optimal:\n{text}"
            );
        }
    }
}

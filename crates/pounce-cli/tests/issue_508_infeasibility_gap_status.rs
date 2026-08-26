//! gh #508 — the "is this violation real?" question must be asked with
//! `constr_viol_tol`, not with `tol`.
//!
//! When restoration cycles without progress on an infeasible model, `ipopt_alg`
//! chooses between `LocalInfeasibility` (AMPL 200 — "your model has no
//! solution") and `ErrorInStepComputation` (AMPL 500, Pyomo
//! `internalSolverError` — "your solver broke"). That choice was made against
//! `max(100·tol, 1e-4)`. `tol` is a tolerance on the **KKT error**; the
//! quantity being tested is a constraint violation. Different quantity,
//! different units, and `constr_viol_tol` — the option that declares what a
//! violated constraint *is* — never entered into it.
//!
//! The probe model makes the threshold measurable to the digit:
//!
//! ```text
//!   min (x - 5)^2   s.t.   x^2 + delta == 0
//! ```
//!
//! Infeasible for every `delta > 0`. `½‖c‖²` has a strict local minimum at
//! `x = 0` where the violation is exactly `delta` and `Jᵀc = 2x(x²+δ) = 0`, so
//! restoration converges there, no local move reduces the violation, and the
//! honest verdict at every `delta` is "converged to a point of local
//! infeasibility". Two measured symptoms, both reproduced by the fixtures here:
//!
//! 1. sweeping `constr_viol_tol` over four orders moved the boundary *not at
//!    all* — at `constr_viol_tol=1e-3` a violation of `1e-4`, comfortably
//!    inside the user's own feasibility tolerance, still exited 500;
//! 2. sweeping `tol` moved it a great deal and in the wrong direction: at
//!    `tol=1e-4` the threshold widened to `1e-2` and swallowed every gap from
//!    `3e-4` up — including a model infeasible by a full percent. Loosening
//!    `tol` is the standard user reaction to a struggling solve, so the failure
//!    band widened exactly when the user tried to help.
//!
//! The fix reads `constr_viol_tol` and compares with `>=`, so the invariant the
//! tests below pin is the one that matters rather than any particular constant:
//! **a violation at or above the user's declared feasibility tolerance lands in
//! the AMPL infeasible range, at every `tol`.**
//!
//! Sibling coverage: `infeasible_status_tol_invariance.rs` pins the same
//! `tol`-independence for gh #372/#446's fixtures, which reach the verdict
//! through the restoration-side gates rather than through this cycle exit.

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

/// Run the model and return the `.sol` text. `extra` carries the option
/// assignments under test.
fn solve(model: &str, tag: &str, extra: &[String]) -> String {
    let sol = std::env::temp_dir().join(format!("pounce_issue_508_{tag}.sol"));
    let _ = std::fs::remove_file(&sol);

    let out = Command::new(pounce_exe())
        .arg(fixture(model))
        .arg("-AMPL")
        .arg("--sol-output")
        .arg(&sol)
        .arg("print_level=0")
        // No acceptable point exists anywhere on this model's trajectory, so
        // the acceptable-level exit cannot mask which of the two failure
        // statuses the decision site picks.
        .arg("acceptable_tol=1e-12")
        .args(extra)
        .output()
        .expect("spawn pounce");

    assert_eq!(out.status.code(), Some(0), "-AMPL must exit 0");
    std::fs::read_to_string(&sol).expect("read .sol")
}

/// `tol` spanning four orders either side of the default. `1e-4` and `1e-5` are
/// the values that widened the old `max(100·tol, 1e-4)` threshold to `1e-2` and
/// `1e-3`.
const TOLS: &[(&str, &str)] = &[
    ("t3", "1e-3"),
    ("t4", "1e-4"),
    ("t5", "1e-5"),
    ("t6", "1e-6"),
    ("t7", "1e-7"),
    ("t8", "1e-8"),
    ("t10", "1e-10"),
];

/// A model infeasible by `1e-2` — two orders above the default
/// `constr_viol_tol` — must land in the AMPL infeasible range whatever the user
/// sets `tol` to. At `tol=1e-4` and `tol=1e-5` this returned 500.
#[test]
fn a_percent_wide_infeasibility_gap_is_tol_invariant() {
    for (tag, tol) in TOLS {
        let text = solve(
            "issue_508_infeasible_gap_1em2.nl",
            &format!("gap2_{tag}"),
            &[format!("tol={tol}")],
        );
        let srn = solve_result_num(&text);
        assert!(
            (200..300).contains(&srn),
            "tol={tol}: expected the AMPL infeasible range (200..299), got \
             solve_result_num={srn}. This model is infeasible by 1e-2 — a full \
             percent — and 500 surfaces through Pyomo as internalSolverError, \
             indistinguishable from a solver bug:\n{text}"
        );
    }
}

/// The boundary cell. The reported violation is exactly `1e-4`, exactly the
/// default `constr_viol_tol`; the old `>` comparison took the failure branch on
/// it. A violation *at* the user's declared tolerance is a violation.
#[test]
fn a_violation_exactly_at_constr_viol_tol_is_infeasible_not_an_internal_error() {
    for (tag, tol) in TOLS {
        let text = solve(
            "issue_508_infeasible_gap_1em4.nl",
            &format!("gap4_{tag}"),
            &[format!("tol={tol}"), "constr_viol_tol=1e-4".into()],
        );
        let srn = solve_result_num(&text);
        assert!(
            (200..300).contains(&srn),
            "tol={tol}, constr_viol_tol=1e-4: the reported violation is exactly \
             1e-4, so it is at the tolerance the user declared to be too much; \
             expected the AMPL infeasible range (200..299), got \
             solve_result_num={srn}:\n{text}"
        );
    }
}

/// `constr_viol_tol` is the knob that decides this, so moving it must move the
/// verdict. Previously all four settings gave a bit-identical column.
#[test]
fn constr_viol_tol_moves_the_boundary() {
    // Tightened well below the 1e-4 gap: the violation is a violation, and the
    // model is infeasible.
    for (tag, cvt) in [
        ("c8", "1e-8"),
        ("c6", "1e-6"),
        ("c5", "1e-5"),
        ("c4", "1e-4"),
    ] {
        let text = solve(
            "issue_508_infeasible_gap_1em4.nl",
            &format!("cvt_{tag}"),
            &["tol=1e-6".into(), format!("constr_viol_tol={cvt}")],
        );
        let srn = solve_result_num(&text);
        assert!(
            (200..300).contains(&srn),
            "constr_viol_tol={cvt}: a 1e-4 violation is at or above the declared \
             feasibility tolerance, so the verdict must be infeasible \
             (200..299); got solve_result_num={srn}:\n{text}"
        );
    }
}

/// The other half of that claim: with `constr_viol_tol` widened *past* the gap,
/// the solver has no evidence of infeasibility to report — the iterate is
/// primal-feasible by the user's own declaration — so it must not claim any.
/// Pins the direction of the branch, not just its sensitivity.
#[test]
fn a_gap_inside_constr_viol_tol_is_not_claimed_infeasible() {
    let text = solve(
        "issue_508_infeasible_gap_1em4.nl",
        "cvt_wide",
        &["tol=1e-6".into(), "constr_viol_tol=1e-3".into()],
    );
    let srn = solve_result_num(&text);
    assert!(
        !(200..300).contains(&srn),
        "constr_viol_tol=1e-3: a 1e-4 violation is inside the tolerance the user \
         declared feasible, so POUNCE has no infeasibility to certify and must \
         not report one; got solve_result_num={srn}:\n{text}"
    );
}

/// Neither fixture may ever be advertised as solved — that is what makes Pyomo
/// load the returned point as `optimal`. Separate assertion so a future change
/// that trades one wrong verdict for a worse one is caught explicitly.
#[test]
fn the_gap_models_are_never_reported_solved() {
    for model in [
        "issue_508_infeasible_gap_1em4.nl",
        "issue_508_infeasible_gap_1em2.nl",
    ] {
        for (tag, tol) in TOLS {
            let stem = model.trim_end_matches(".nl");
            let text = solve(model, &format!("ns_{stem}_{tag}"), &[format!("tol={tol}")]);
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

/// Secondary defect from the same report: when the local-infeasibility
/// second-opinion re-solve runs and is *not* promoted, the `.sol` keeps the
/// first solve's verdict while the terminal's last `EXIT:` banner was the
/// retry's — so the console and the modelling layer disagreed about one solve.
/// Several banners is expected and announced; them disagreeing is not, and
/// `validation/p3_control.py` reads exactly this way (last `EXIT:` line, paired
/// with the `.sol`).
///
/// The retry is a ladder as of gh #524 (`feral_scaling=mc64`, then
/// `mu_strategy=adaptive`, then `start_point_perturbation=1e-2`), so this now
/// covers a run that emits *four* end-of-run banners rather than two — which is
/// strictly more of the failure mode the invariant is about. The path is
/// detected off the ladder's own "keeping the original … verdict" line rather
/// than any one rung's message, so adding or reordering rungs cannot silently
/// turn this test into a no-op the way naming a rung would.
///
/// That line names the status it kept rather than saying "local infeasibility"
/// in prose, because the third rung also fires on `Invalid_Number_Detected` and
/// the old wording would have been a lie on that path. The sentinel below spells
/// out the status this fixture reaches, so it still fails loudly — rather than
/// passing vacuously — if the fixture stops exercising the non-promoted retry.
///
/// **Known gap, stated deliberately: this is an invariant guard, not a
/// bite-on-parent regression pin.** It asserts the right thing and it does
/// exercise the non-promoted retry path (the first assertion fails loudly if a
/// future change stops it doing so), but it passes on the pre-fix commit,
/// because every non-promoted retry reachable from this repo's fixture corpus
/// happens to return `InfeasibleProblemDetected` — the same verdict the `.sol`
/// keeps, so the banners agree by luck rather than by construction. The
/// reporter observed the mismatch on their own `.nl` at `tol=1e-4`
/// (`Error in step computation.` at δ=1e-9, `Maximum Number of Iterations
/// Exceeded.` at δ=1e-1, both over a `.sol` that said locally infeasible); a
/// sweep over `tol`, `max_iter` and every `.nl` under `tests/fixtures/` did not
/// reproduce a divergent retry status here. The fix does not depend on
/// reproducing one — the retry's status is not constrained to match the kept
/// verdict, and `scaling_retry_promoted` is false for most statuses, so the
/// mismatch is a reachable state of the code whether or not a fixture reaches
/// it. Vendoring a model that does would upgrade this to a real pin.
#[test]
fn the_last_exit_banner_matches_the_sol_after_a_non_promoted_second_opinion() {
    let sol = std::env::temp_dir().join("pounce_issue_508_banner.sol");
    let _ = std::fs::remove_file(&sol);

    let out = Command::new(pounce_exe())
        .arg(fixture("issue_508_infeasible_gap_1em2.nl"))
        .arg("-AMPL")
        .arg("--sol-output")
        .arg(&sol)
        .arg("tol=1e-8")
        .arg("acceptable_tol=1e-12")
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(0), "-AMPL must exit 0");

    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        stderr.contains("keeping the original Infeasible_Problem_Detected verdict"),
        "fixture no longer exercises the non-promoted retry path, so this test \
         proves nothing — pick a model/tol that still does:\nstderr:\n{stderr}"
    );

    let text = std::fs::read_to_string(&sol).expect("read .sol");
    let srn = solve_result_num(&text);
    assert!(
        (200..300).contains(&srn),
        "expected the kept local-infeasibility verdict, got \
         solve_result_num={srn}:\n{text}"
    );

    let last_exit = stdout
        .lines()
        .filter(|l| l.starts_with("EXIT:"))
        .next_back()
        .expect("at least one EXIT: banner on stdout");
    assert!(
        last_exit.contains("local infeasibility"),
        "the terminal's final word must be the verdict that shipped in the \
         `.sol` (solve_result_num={srn}); last EXIT: banner was {last_exit:?}. \
         A consumer that keeps the last EXIT: line — p3_control.py does — pairs \
         it with a `.sol` that never held it:\nstdout:\n{stdout}"
    );
}

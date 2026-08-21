//! The square-problem "feasible point found" verdict must be scale-invariant.
//!
//! `min x²+y²  s.t.  s·(x·y) == s·1,  s·(x+y) == s·0.5,  x,y ∈ [-10,10]` has
//! no real solution at any `s`: `x+y = 0.5` and `x·y = 1` need roots of
//! `t² - 0.5t + 1`, whose discriminant is `-3.75`. Multiplying both rows by a
//! positive constant leaves the feasible set exactly unchanged, so the verdict
//! must be the infeasible band at every `s` exactly as it is at `s = 1`. The
//! fixture here is `s = 1e-4`.
//!
//! It was not. gh#508 added Ipopt's square-problem path — two variables, two
//! equality rows, so `IsSquareProblem()` holds — which lets restoration hand
//! back its least-infeasible point as `Feasible_Point_Found` (AMPL success
//! band) when the *original* NLP violation at that point is under
//! `constr_viol_tol`. That test was absolute only, on a residual that carries
//! `s`. At `s = 1e-4` the least-infeasible point misses the product row by
//! ~94% of the row, but `0.94 · 1e-4 = 9.375e-5` is under the default
//! `constr_viol_tol = 1e-4`, so the run reported a feasible point for a system
//! that has none. Real Ipopt 3.14.19 does exactly the same thing here; this is
//! the gh#387 / gh#390 / gh#391 defect class — a scale-dependent quantity
//! compared against an absolute threshold — and pounce refuses it.
//!
//! The fix pairs the absolute test with the gh#390 declared-RHS relative
//! measure at the shared `1e-2` band, so the point must be small against
//! `constr_viol_tol` *and* small against each row's own declared magnitude.
//!
//! Both the default solver selection and the forced NLP path are pinned: the
//! defect reaches through both, and a fix that only satisfies one leaves the
//! same wrong answer one option away.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("infeasible_square_scaled_1em4.nl");
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
    let sol = std::env::temp_dir().join(format!("pounce_gh508_square_{tag}.sol"));
    let _ = std::fs::remove_file(&sol);

    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("-AMPL")
        .arg("--sol-output")
        .arg(&sol)
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
fn a_row_scaled_square_infeasibility_is_not_reported_feasible_point_found() {
    for (tag, extra) in [
        ("default", &[][..]),
        ("nlp", &["solver_selection=nlp"][..]),
        (
            "identity",
            &["solver_selection=nlp", "feral_scaling=identity"][..],
        ),
        ("mc64", &["solver_selection=nlp", "feral_scaling=mc64"][..]),
    ] {
        let code = solve(tag, extra);
        assert!(
            (200..300).contains(&code),
            "`1e-4·(x·y) == 1e-4, 1e-4·(x+y) == 0.5e-4` over `x,y ∈ [-10,10]` \
             is the same empty feasible set at every row scaling, but {tag} \
             reported solve_result_num={code} (200..299 is the infeasible \
             band). A code under 200 means the square-problem path called a \
             point feasible because its violation was small in absolute terms \
             at a scale where every row is small in absolute terms."
        );
    }
}

/// The other side of the guard: it defends the *default* tolerance, and stands
/// down when the author widens it.
///
/// `constr_viol_tol` out of the box is a solver-chosen `1e-4`, picked with no
/// knowledge of this model's units, so pairing it with the relative measure is
/// what makes the verdict above scale-invariant. A tolerance set wider than
/// that default cannot have come from the solver — it is the author declaring
/// how much violation this model tolerates, chosen with the model in hand — and
/// upstream honours it at every setting. The sibling invariant lives in
/// `issue_508_infeasibility_gap_status.rs::a_gap_inside_constr_viol_tol_is_not_claimed_infeasible`;
/// pinned here too because it is *this* gate that decides both.
///
/// `constr_viol_tol=1e-4` is included on purpose: restating the default is not
/// widening it, so it must answer exactly as the default does. Reading the
/// option's provenance instead of its value would split those two, and would
/// hand any frontend that echoes defaults back a silently disabled guard.
#[test]
fn a_widened_constr_viol_tol_is_honored_and_a_restated_default_is_not() {
    assert!(
        !(200..300).contains(&solve("wide", &["constr_viol_tol=1e-3"])),
        "constr_viol_tol=1e-3 puts the ~9.4e-5 violation inside the band the \
         author declared feasible, so POUNCE has no infeasibility to certify"
    );
    for cvt in ["1e-4", "1e-6"] {
        let code = solve(&format!("cvt{cvt}"), &[&format!("constr_viol_tol={cvt}")]);
        assert!(
            (200..300).contains(&code),
            "constr_viol_tol={cvt} is not wider than the default, so the \
             scale-relative guard still applies and the verdict must stay in \
             the infeasible band; got solve_result_num={code}"
        );
    }
}

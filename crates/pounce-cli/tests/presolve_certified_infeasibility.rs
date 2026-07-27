//! Presolve can *prove* a feasible region empty; that proof is now the verdict.
//!
//! Bound propagation and FBBT both establish emptiness exactly — for a linear
//! row over a box, propagation is a decision procedure, and FBBT's interval
//! arithmetic is outward-rounded, so an empty computed interval means the true
//! range is empty. Previously presolve had nowhere to report this: it logged a
//! warning, discarded the result, and let the IPM re-derive a strictly weaker
//! numerical verdict — a stationary point of the constraint violation, which on
//! a nonconvex problem proves nothing globally.
//!
//! The two verdicts are now distinguishable:
//!
//! ```text
//!   proved   -> solve_result_num 201, "... (proved by presolve: <how>)"
//!   local    -> solve_result_num 200, "InfeasibleProblemDetected"
//! ```
//!
//! Both sit in AMPL's 200..299 "infeasible" band, so every band-reading
//! consumer is unaffected — Pyomo maps the whole range to
//! `TerminationCondition.infeasible` in both of its SOL readers. Sub-coding
//! within a band is the AMPL-native idiom; Ipopt does the same with 500/501/502
//! in the failure band.

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

fn solve(tag: &str, extra: &[&str]) -> String {
    let sol = std::env::temp_dir().join(format!("pounce_presolve_cert_{tag}.sol"));
    let _ = std::fs::remove_file(&sol);

    let out = Command::new(pounce_exe())
        .arg(fixture("issue_372_infeasible_bounds.nl"))
        .arg("-AMPL")
        .arg("--sol-output")
        .arg(&sol)
        .arg("print_level=0")
        .args(extra)
        .output()
        .expect("spawn pounce");

    assert_eq!(out.status.code(), Some(0), "-AMPL must exit 0");
    std::fs::read_to_string(&sol).expect("read .sol")
}

/// `0 <= x <= 0.6` with `x >= 0.7` is a one-row contradiction over a box, which
/// bound propagation decides exactly. With presolve on it must be reported as
/// *proved*, not merely locally infeasible.
#[test]
fn presolve_proves_the_contradiction_and_says_so() {
    let text = solve("on", &["presolve=yes"]);
    let srn = solve_result_num(&text);

    assert_eq!(
        srn, 201,
        "a presolve-proved empty feasible region must report the certified \
         sub-code 201, not the generic local-infeasibility 200:\n{text}"
    );
    assert!(
        text.contains("proved by presolve"),
        "the message must say the verdict is a proof, not a local \
         verdict:\n{text}"
    );
    assert!(
        text.contains("bound propagation"),
        "the message must name *how* it was proved, so the claim is \
         checkable:\n{text}"
    );
}

/// With presolve off the numerical path is unchanged — still infeasible, but
/// the honest weaker verdict and the original 200. Guards against the
/// certified path quietly becoming the only one.
#[test]
fn without_presolve_the_numerical_verdict_is_unchanged() {
    let text = solve("off", &["presolve=no"]);
    let srn = solve_result_num(&text);

    assert_eq!(
        srn, 200,
        "without presolve the verdict is the numerical local one (200):\n{text}"
    );
    assert!(
        !text.contains("proved by presolve"),
        "the IPM's local verdict must not claim to be a proof:\n{text}"
    );
}

/// Whichever path produced it, the answer stays inside AMPL's infeasible band.
/// This is what every downstream consumer actually keys on — Pyomo included —
/// so the new sub-code must not escape the range.
#[test]
fn both_paths_stay_in_the_ampl_infeasible_band() {
    for (tag, opt) in [("band_on", "presolve=yes"), ("band_off", "presolve=no")] {
        let text = solve(tag, &[opt]);
        let srn = solve_result_num(&text);
        assert!(
            (200..300).contains(&srn),
            "{opt}: solve_result_num={srn} escaped the AMPL infeasible band \
             (200..299); Pyomo and every other band-reading consumer would \
             misclassify it:\n{text}"
        );
    }
}

/// A proof is not something a different matrix scaling can overturn, so the
/// MC64 re-solve guard — which exists to second-guess a *numerical* local
/// infeasibility — must not fire and burn a second solve.
#[test]
fn certified_infeasibility_skips_the_mc64_second_opinion() {
    let sol = std::env::temp_dir().join("pounce_presolve_cert_mc64.sol");
    let _ = std::fs::remove_file(&sol);
    let out = Command::new(pounce_exe())
        .arg(fixture("issue_372_infeasible_bounds.nl"))
        .arg("-AMPL")
        .arg("--sol-output")
        .arg(&sol)
        .arg("print_level=0")
        .arg("presolve=yes")
        .output()
        .expect("spawn pounce");

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !combined.contains("re-solving once with MC64"),
        "the MC64 second-opinion retry re-derives a verdict that scaling \
         cannot affect; it must be skipped for a proof:\n{combined}"
    );
}

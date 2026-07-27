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

// ---------------------------------------------------------------------------
// Adversarial regressions. Both were found by trying to break the certificate
// rather than confirm it, and both were real.
// ---------------------------------------------------------------------------

/// A model infeasible only by floating-point noise must NOT be certified.
///
/// `presolve_float_trap.nl` is `x >= 0.1 + 0.2` with `x <= 0.3`. In binary
/// floating point `0.1 + 0.2 == 0.30000000000000004`, so it is infeasible — by
/// `5.5e-17`. A modeller writing that means `x >= 0.3`, and POUNCE agrees:
/// `presolve=no` returns `Solve_Succeeded` and the LP route reports "Optimal
/// Solution Found".
///
/// Before the fix, `presolve=yes presolve_fbbt=yes` reported it **proved
/// infeasible** — the strongest claim the solver can make, on the flimsiest
/// possible margin, contradicting three other routes on the same model. The
/// cause was an asymmetry between the two proof paths: Phase-1 bound
/// propagation requires the crossing to exceed `1e-12`, while FBBT's emptiness
/// tests carry no margin at all. `FBBT_CERTIFY_MARGIN` closes it.
///
/// A second, pre-existing defect sat underneath: Phase-1 declines to call a
/// sub-margin crossing infeasible but still *wrote* the crossed bounds, so the
/// solver received `x_l > x_u` and rejected it as `Invalid_Problem_Definition`
/// (504) — a well-posed model failing outright the moment presolve was on.
#[test]
fn float_noise_infeasibility_is_never_certified() {
    for (tag, opts) in [
        ("ft_none", vec!["presolve=no"]),
        ("ft_p", vec!["presolve=yes", "presolve_fbbt=no"]),
        ("ft_pf", vec!["presolve=yes", "presolve_fbbt=yes"]),
    ] {
        let sol = std::env::temp_dir().join(format!("pounce_{tag}.sol"));
        let _ = std::fs::remove_file(&sol);
        let out = Command::new(pounce_exe())
            .arg(fixture("presolve_float_trap.nl"))
            .arg("-AMPL")
            .arg("--sol-output")
            .arg(&sol)
            .arg("print_level=0")
            .arg("solver_selection=nlp")
            .args(&opts)
            .output()
            .expect("spawn pounce");
        assert_eq!(out.status.code(), Some(0), "-AMPL must exit 0");
        let text = std::fs::read_to_string(&sol).expect("read .sol");
        let srn = solve_result_num(&text);

        assert!(
            !text.contains("proved by presolve"),
            "{opts:?}: a model infeasible by 5.5e-17 must never be reported as \
             *proved* infeasible — POUNCE solves it successfully on every other \
             route:\n{text}"
        );
        assert_eq!(
            srn, 0,
            "{opts:?}: expected the same Solve_Succeeded (0) every other route \
             gives, got solve_result_num={srn}. 504 here means presolve handed \
             the solver a crossed box `x_l > x_u`:\n{text}"
        );
    }
}

/// The JSON report and the `.sol` must never disagree about the same run.
///
/// The certified sub-code was initially applied only in the `.sol` writer, so a
/// single run reported `201` in one output and `200` in the other — a
/// contradiction a caller reading both has no way to reconcile.
#[test]
fn json_report_and_sol_agree_on_the_code() {
    for (tag, opt) in [("cmp_on", "presolve=yes"), ("cmp_off", "presolve=no")] {
        let sol = std::env::temp_dir().join(format!("pounce_{tag}.sol"));
        let json = std::env::temp_dir().join(format!("pounce_{tag}.json"));
        let _ = std::fs::remove_file(&sol);
        let _ = std::fs::remove_file(&json);
        let out = Command::new(pounce_exe())
            .arg(fixture("issue_372_infeasible_bounds.nl"))
            .arg("-AMPL")
            .arg("--sol-output")
            .arg(&sol)
            .arg("--json-output")
            .arg(&json)
            .arg("print_level=0")
            .arg(opt)
            .output()
            .expect("spawn pounce");
        assert_eq!(out.status.code(), Some(0), "-AMPL must exit 0");

        let text = std::fs::read_to_string(&sol).expect("read .sol");
        let sol_srn = solve_result_num(&text);
        let raw = std::fs::read_to_string(&json).expect("read json");
        let json_srn: i32 = raw
            .split("\"solve_result_num\"")
            .nth(1)
            .and_then(|s| s.split(&[':', ',', '}'][..]).nth(1))
            .and_then(|s| s.trim().parse().ok())
            .expect("solve_result_num in JSON report");

        assert_eq!(
            sol_srn, json_srn,
            "{opt}: the .sol says {sol_srn} and the JSON report says \
             {json_srn} for the same run"
        );
    }
}

/// Interval-arithmetic **overflow** must never be mistaken for a proof.
///
/// `presolve_overflow_feasible.nl` is `x in [-1e300, 1e300]` with
/// `1e300*x >= 1e300` — i.e. `x >= 1`, plainly feasible (`x = 1`). But
/// `1e300 * 1e300` overflows to infinity, and `Interval::is_empty` treats a NaN
/// endpoint as an empty range, so FBBT reported "this constraint cannot be
/// satisfied" and it was certified as **proved infeasible** — a false proof on
/// a feasible model, the worst failure this feature can produce. Meanwhile
/// `presolve_fbbt=no` solved the same model, so POUNCE contradicted itself.
///
/// The margin probe cannot catch this: an overflow is unaffected by widening
/// bounds `1e-9`. The guard is therefore on the *inputs* — a finite magnitude
/// at or beyond the INF sentinel is where the arithmetic stops being
/// representable, so no proof may rest on it. Genuine infinities (`x >= 5` has
/// `g_u = +inf`) are ordinary and still certify normally.
#[test]
fn interval_overflow_is_never_mistaken_for_a_proof() {
    let sol = std::env::temp_dir().join("pounce_presolve_overflow.sol");
    let _ = std::fs::remove_file(&sol);
    let out = Command::new(pounce_exe())
        .arg(fixture("presolve_overflow_feasible.nl"))
        .arg("-AMPL")
        .arg("--sol-output")
        .arg(&sol)
        .arg("print_level=0")
        .arg("solver_selection=nlp")
        .arg("presolve=yes")
        .arg("presolve_fbbt=yes")
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(0), "-AMPL must exit 0");
    let text = std::fs::read_to_string(&sol).expect("read .sol");
    let srn = solve_result_num(&text);

    assert!(
        !text.contains("proved by presolve"),
        "a feasible model (x >= 1) was reported as *proved* infeasible because \
         1e300*1e300 overflowed to inf:\n{text}"
    );
    assert!(
        !(200..300).contains(&srn),
        "feasible model reported in the AMPL infeasible band \
         (solve_result_num={srn}):\n{text}"
    );
}

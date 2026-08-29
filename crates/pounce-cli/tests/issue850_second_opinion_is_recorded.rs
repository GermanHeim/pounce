//! A promoted second-opinion re-solve must leave a trace (gh #850).
//!
//! # What was invisible
//!
//! When the base solve fails and a ladder rung recovers it, the report's
//! `status` and `statistics.iteration_count` both become the **promoted rung's**
//! and nothing else says the base solver failed. So a fixture that *lost* its
//! baseline solve and is now only rescued by a retry reads in
//! `scripts/sweep-fixtures.sh` as a large improvement:
//!
//! ```text
//!                                          status             iters
//!   v0.10.0, defaults                      SolveSucceeded      116
//!   HEAD, defaults                         SolveSucceeded       54
//!   HEAD, infeasibility_perturbed_start_retry=no   RestorationFailed   131
//! ```
//!
//! `v0.10.0` does not have `infeasibility_perturbed_start_retry` — it rejects
//! the option outright — so that 116 is the *base solver* converging, and
//! HEAD's base solver no longer does. The only thing between the user and a
//! `RestorationFailed` is a rung added in the same release. The sweep read it
//! as `116 -> 54`, **a 2× improvement**.
//!
//! That is worse than a gap in the evidence: `scripts/sweep-fixtures.sh` is the
//! repo's primary trajectory guard and CLAUDE.md makes it the required evidence
//! for a trajectory change, so a guard that converts a lost solve into a
//! recorded win produces positive evidence for the wrong conclusion.
//!
//! Three things had to line up, and they did: the narration is on stdout at the
//! default `print_level`, but `sweep_leg` captures that stdout only to scrape
//! the engine out of it and then deletes it; and the JSON report had no field
//! for any of it. This file pins the JSON half — the sweep column is built from
//! it.
//!
//! # The cost was understated too
//!
//! `statistics.iteration_count` is the promoted rung's **alone**. The 131
//! iterations that failed first are not in it, so the fixture's true cost is
//! `131 + 54 = 185`, 3.4× what the report says.

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

/// Solve `name` and return its JSON report as untyped text.
fn report(name: &str, extra: &[&str]) -> String {
    let dir = std::env::temp_dir();
    let sol = dir.join(format!("pounce_850_{name}.sol"));
    let json = dir.join(format!("pounce_850_{name}.json"));
    let out = Command::new(pounce_exe())
        .arg(fixture(name))
        .arg(&sol)
        .arg("--json-output")
        .arg(&json)
        .args(extra)
        .output()
        .expect("run pounce");
    assert!(
        json.exists(),
        "no JSON written for {name}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::read_to_string(&json).expect("read json")
}

/// Pull `"key": <number>` out of the JSON text. Enough for this file, and it
/// keeps the test free of a serde dependency on the report crate's shape.
fn number(json: &str, key: &str) -> Option<i64> {
    let pat = format!("\"{key}\":");
    let at = json.find(&pat)? + pat.len();
    let rest = json[at..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '-')?;
    rest[..end].parse().ok()
}

fn string(json: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":");
    let at = json.find(&pat)? + pat.len();
    let rest = json[at..].trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

const PROMOTING: &str = "square_flowsheet_resto.nl";

/// The premise, asserted rather than assumed: this fixture's base solver really
/// does fail, and the ladder really is what rescues it. Turning the rung off is
/// how the report separates the two.
#[test]
fn the_base_solver_alone_does_not_solve_this_fixture() {
    let json = report(PROMOTING, &["infeasibility_perturbed_start_retry=no"]);
    let status = string(&json, "status").expect("a status");
    assert_eq!(
        status, "RestorationFailed",
        "premise: without the rung this fixture fails; if it now solves, this \
         file's subject has been fixed and the fixture needs replacing"
    );
}

/// The fix: a promotion is recorded, and it names the base solve's verdict —
/// the fact the report used to lose entirely.
#[test]
fn a_promotion_records_what_the_base_solve_did() {
    let json = report(PROMOTING, &[]);
    assert_eq!(string(&json, "status").as_deref(), Some("SolveSucceeded"));
    assert!(
        json.contains("\"second_opinion\""),
        "a promoted solve must carry a second_opinion block:\n{json}"
    );
    assert_eq!(
        string(&json, "base_status").as_deref(),
        Some("Restoration_Failed"),
        "the base solve's own verdict is the thing that was invisible"
    );
    assert_eq!(
        string(&json, "promoted_by").as_deref(),
        Some("start_point_perturbation=1e-2"),
    );
}

/// The cost, which is understated on the same line. `iteration_count` is the
/// promoted rung's alone; `total_iteration_count` is what the solve spent.
#[test]
fn the_true_cost_is_recorded_not_just_the_promoted_rungs() {
    let json = report(PROMOTING, &[]);
    let reported = number(&json, "iteration_count").expect("iteration_count");
    let base = number(&json, "base_iteration_count").expect("base_iteration_count");
    let total = number(&json, "total_iteration_count").expect("total_iteration_count");
    assert!(
        base > 0 && total > base,
        "base {base} and total {total} must both be real and total must \
         include the base solve"
    );
    assert!(
        total > reported,
        "the reported {reported} iterations understate the true cost {total}, \
         which is the point of recording it"
    );
    assert_eq!(
        total,
        base + reported,
        "with one rung promoted, total is exactly base + the promoted rung"
    );
}

/// The other direction, which "always emit the block" would not distinguish: a
/// solve the ladder never opens must carry no `second_opinion` at all, so the
/// field's *presence* is itself the signal.
#[test]
fn an_ordinary_solve_carries_no_second_opinion_block() {
    let json = report("boxed_qp_min.nl", &["solver_selection=nlp"]);
    assert_eq!(string(&json, "status").as_deref(), Some("SolveSucceeded"));
    assert!(
        !json.contains("\"second_opinion\""),
        "a solve that opened no ladder must not carry the block:\n{json}"
    );
}

/// A ladder that runs and promotes nothing is a third case, and it is not the
/// same as no ladder: `it=` is then the base solve's, but the rungs still cost
/// something and that cost is otherwise invisible too.
#[test]
fn a_ladder_that_promotes_nothing_still_records_what_it_spent() {
    let json = report("infeasible_equalities.nl", &[]);
    assert!(
        json.contains("\"second_opinion\""),
        "the ladder ran here, so it must be recorded:\n{json}"
    );
    assert!(
        !json.contains("\"promoted_by\""),
        "nothing was promoted, so the field is absent rather than null"
    );
    let reported = number(&json, "iteration_count").expect("iteration_count");
    let total = number(&json, "total_iteration_count").expect("total_iteration_count");
    assert!(
        total > reported,
        "three rungs ran; their cost ({total} total against a reported \
         {reported}) is exactly what a reader cannot otherwise see"
    );
}

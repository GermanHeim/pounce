//! gh#1009 — `--debug*` on the active-set SQP solver says it does not engage.
//!
//! The debug hook is consumed only by the filter-IPM; `optimize_sqp_tnlp`
//! never installs it. Before this fix an `algorithm=active-set-sqp` run with
//! `--debug-script` printed the "debugger enabled" banner, then ran to
//! completion with no pause, no `[source]` echo and no explanation — and under
//! `--debug-on-error` a failing SQP solve could never break in. The active-set
//! QP engine already printed an IPM-only note; the SQP route now prints its own.
//!
//! What each check reaches:
//!
//! | check | route | why it is here |
//! | --- | --- | --- |
//! | `sqp_says_the_debugger_does_not_engage` | NLP → SQP driver, `--debug-script` | the reported symptom |
//! | `sqp_says_so_under_debug_on_error` | NLP → SQP driver, `--debug-on-error` | the mode whose silence costs most |
//! | `ipm_still_engages_and_prints_no_note` | NLP → filter-IPM | the note must not leak onto the engine that does engage |
//! | `an_lp_under_active_set_sqp_still_engages` | LP → convex IPM despite `algorithm=active-set-sqp` | the `algorithm` option alone does not mean the SQP driver ran |
//! | `no_debugger_no_note` | NLP → SQP driver, no `--debug*` | the note is about the debugger, not about SQP |
//!
//! The model is `nonconvex_qp.nl`: an indefinite QP, which `auto` sends to the
//! general NLP route, so `algorithm` alone decides SQP versus filter-IPM. It
//! converges on the first attempt either way. That matters because an attached
//! debugger turns off the second-opinion retry ladder, so a model that needs a
//! retry (`degenerate_start_hs008.nl`, for one) exits 1 under `--debug*` for a
//! reason unrelated to this issue, and the exit-code checks would read it as one.
//!
//! **Mutation table** — measured:
//!
//! | mutation | red |
//! | --- | --- |
//! | delete the note | `sqp_says_the_debugger_does_not_engage`, `sqp_says_so_under_debug_on_error` |
//! | print it ahead of the convex dispatch, gated on the `algorithm` option | `an_lp_under_active_set_sqp_still_engages` |
//! | drop the `debug_hook.is_some()` gate | `no_debugger_no_note` |

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

const NOTE: &str = "does not engage on the active-set SQP solver";
/// The pause banner the REPL prints at every checkpoint it stops at.
const PAUSED: &str = "── pounce-dbg ──";

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

/// The issue's script. Named per call, not per process: the tests below run as
/// threads of one process and each deletes its script when done.
fn script() -> PathBuf {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let p = std::env::temp_dir().join(format!(
        "pounce_gh1009_{}_{}.pdbg",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let mut f = std::fs::File::create(&p).expect("write debug script");
    writeln!(f, "print mu\nstep\ncontinue").expect("write debug script");
    p
}

/// Run the binary and return `(stdout + stderr, exit code)`.
fn run(model: &str, opts: &[&str]) -> (String, Option<i32>) {
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(fixture(model))
        .arg("--no-sol")
        .args(opts)
        .arg("print_level=0")
        .stdin(Stdio::null())
        .output()
        .expect("spawn pounce");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (text, out.status.code())
}

fn run_with_script(model: &str, opts: &[&str]) -> (String, Option<i32>) {
    let s = script();
    let mut all: Vec<&str> = opts.to_vec();
    let s_str = s.to_str().unwrap().to_owned();
    all.push("--debug-script");
    all.push(&s_str);
    let r = run(model, &all);
    let _ = std::fs::remove_file(&s);
    r
}

#[test]
fn sqp_says_the_debugger_does_not_engage() {
    let (out, rc) = run_with_script("nonconvex_qp.nl", &["algorithm=active-set-sqp"]);
    assert!(
        out.contains("active-set SQP"),
        "did not reach the SQP driver:\n{out}"
    );
    assert!(
        out.contains(NOTE),
        "no IPM-only note on the SQP route:\n{out}"
    );
    assert!(
        !out.contains(PAUSED),
        "the SQP route paused, so the note is false:\n{out}"
    );
    assert_eq!(rc, Some(0), "{out}");
}

#[test]
fn sqp_says_so_under_debug_on_error() {
    let (out, _) = run(
        "nonconvex_qp.nl",
        &["algorithm=active-set-sqp", "--debug-on-error"],
    );
    assert!(
        out.contains(NOTE),
        "no IPM-only note under --debug-on-error:\n{out}"
    );
}

#[test]
fn ipm_still_engages_and_prints_no_note() {
    let (out, rc) = run_with_script("nonconvex_qp.nl", &[]);
    assert!(out.contains(PAUSED), "the filter-IPM did not pause:\n{out}");
    assert!(!out.contains(NOTE), "IPM run carries the SQP note:\n{out}");
    assert_eq!(rc, Some(0), "{out}");
}

/// `algorithm=active-set-sqp` on an LP under `auto` is routed to the convex
/// IPM and never reaches the SQP driver, so the debugger engages there and a
/// note claiming otherwise would be the opposite of what happened.
#[test]
fn an_lp_under_active_set_sqp_still_engages() {
    let (out, rc) = run_with_script("lp_afiro.nl", &["algorithm=active-set-sqp"]);
    assert!(
        out.contains("convex QP interior-point"),
        "routing moved:\n{out}"
    );
    assert!(out.contains(PAUSED), "the convex IPM did not pause:\n{out}");
    assert!(
        !out.contains(NOTE),
        "note printed on a route that engages:\n{out}"
    );
    assert_eq!(rc, Some(0), "{out}");
}

#[test]
fn no_debugger_no_note() {
    let (out, _) = run("nonconvex_qp.nl", &["algorithm=active-set-sqp"]);
    assert!(
        !out.contains(NOTE),
        "note printed with no debugger attached:\n{out}"
    );
}

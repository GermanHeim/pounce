//! Options files actually configure the run (gh#518).
//!
//! `option_file_name=` used to be accepted and do nothing, and there was
//! no implicit `ipopt.opt` lookup at all, so a run configured entirely
//! through an options file executed at stock defaults *and reported
//! success* — which silently invalidates any benchmark set up that way.
//!
//! Everything here is end-to-end for that reason: what matters is not
//! that a file parses but that its settings reach the solve, so each
//! test picks an option whose effect is unmistakable in the output
//! (`max_iter 1` on a fixture that needs 11 iterations) and reads the
//! verdict back off the console.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

/// Fresh scratch dir holding `m.nl`, a fixture that needs 11 iterations
/// (so `max_iter 1` is a loud, deterministic signal). Runs happen *in*
/// this directory: the implicit lookup probes the working directory.
fn scratch(tag: &str) -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let seq = N.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("pounce_gh518_{}_{seq}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fixture.push("tests/fixtures/hs71_obj1e8.nl");
    std::fs::copy(&fixture, dir.join("m.nl")).expect("copy fixture");
    dir
}

fn write(dir: &Path, name: &str, body: &str) {
    std::fs::write(dir.join(name), body).unwrap();
}

/// Run `pounce m.nl --no-sol <args>` with `dir` as the working
/// directory. Returns (exit code, stdout+stderr).
fn run(dir: &Path, args: &[&str]) -> (Option<i32>, String) {
    let out = Command::new(pounce_exe())
        .current_dir(dir)
        .arg("m.nl")
        .arg("--no-sol")
        .args(args)
        .output()
        .expect("spawn pounce");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    (out.status.code(), combined)
}

const CAPPED: &str = "Maximum Number of Iterations Exceeded";

#[test]
fn option_file_name_is_honored() {
    let dir = scratch("named");
    write(&dir, "tiny.opt", "max_iter 1\n");
    let (_, out) = run(&dir, &["option_file_name=tiny.opt"]);
    assert!(out.contains(CAPPED), "tiny.opt was ignored; output:\n{out}");
    assert!(
        out.contains("Using option file \"tiny.opt\""),
        "the run should say which file configured it; output:\n{out}"
    );
}

/// The `--options-file` spelling of the same thing, which was the only
/// way in before gh#518.
#[test]
fn options_file_flag_is_honored() {
    let dir = scratch("flag");
    write(&dir, "tiny.opt", "max_iter 1\n");
    let (_, out) = run(&dir, &["--options-file", "tiny.opt"]);
    assert!(out.contains(CAPPED), "tiny.opt was ignored; output:\n{out}");
}

/// Ipopt's documented default options file, read from the working
/// directory with nothing on the command line pointing at it. This is
/// the divergence from Ipopt the issue found most visible.
#[test]
fn implicit_ipopt_opt_is_read() {
    let dir = scratch("implicit_ipopt");
    write(&dir, "ipopt.opt", "max_iter 1\n");
    let (_, out) = run(&dir, &[]);
    assert!(
        out.contains(CAPPED),
        "./ipopt.opt was ignored; output:\n{out}"
    );
    assert!(
        out.contains("Using option file \"ipopt.opt\""),
        "a discovered file must announce itself — nothing on the command \
         line hints it is steering the solve; output:\n{out}"
    );
}

/// pounce's own name is probed first, and the shadowed file is named
/// rather than left to look applied.
#[test]
fn pounce_opt_wins_over_ipopt_opt_and_says_so() {
    let dir = scratch("both");
    write(&dir, "pounce.opt", "max_iter 1\n");
    write(&dir, "ipopt.opt", "max_iter 500\n");
    let (_, out) = run(&dir, &[]);
    assert!(
        out.contains(CAPPED),
        "pounce.opt should have won; output:\n{out}"
    );
    assert!(
        out.contains("both present") && out.contains("ipopt.opt"),
        "the unread file should be named; output:\n{out}"
    );
}

/// Precedence, the direction that matters: a `key=value` typed on the
/// command line beats the file, so an options file cannot silently
/// override an explicit request.
#[test]
fn command_line_overrides_the_option_file() {
    let dir = scratch("override");
    write(&dir, "ipopt.opt", "max_iter 1\n");
    let (code, out) = run(&dir, &["max_iter=3000"]);
    assert_eq!(code, Some(0), "override solve failed; output:\n{out}");
    assert!(
        !out.contains(CAPPED),
        "the file's max_iter beat the command line; output:\n{out}"
    );
}

/// `$pounce_options` is the same layer as the command line, so it wins
/// over the file too (AMPL passes directives that way).
#[test]
fn pounce_options_env_overrides_the_option_file() {
    let dir = scratch("env");
    write(&dir, "ipopt.opt", "max_iter 1\n");
    let out = Command::new(pounce_exe())
        .current_dir(&dir)
        .arg("m.nl")
        .arg("--no-sol")
        .env("pounce_options", "max_iter=3000")
        .output()
        .expect("spawn pounce");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !combined.contains(CAPPED),
        "the file's max_iter beat $pounce_options; output:\n{combined}"
    );
}

/// The failure mode the issue is about, in its remaining form: a file
/// that was named but cannot be read configures nothing, so the run
/// must stop rather than quietly proceed at defaults. (Upstream opens
/// it with a bare ifstream and shrugs; this is the one deliberate
/// divergence.)
#[test]
fn a_named_options_file_that_does_not_exist_is_an_error() {
    let dir = scratch("missing");
    let (code, out) = run(&dir, &["option_file_name=nope.opt"]);
    assert_eq!(code, Some(2), "should fail; output:\n{out}");
    assert!(
        out.contains("nope.opt") && out.contains("does not exist"),
        "the message should name the file; output:\n{out}"
    );
    assert!(
        !out.contains("EXIT:"),
        "the solve must not run; output:\n{out}"
    );
}

/// A bad option name in a discovered file is rejected the same way one
/// on the command line is — the file is genuinely read, not scanned.
#[test]
fn an_invalid_option_in_a_discovered_file_is_rejected() {
    let dir = scratch("badopt");
    write(&dir, "ipopt.opt", "definitely_not_an_option 1\n");
    let (code, out) = run(&dir, &[]);
    assert_eq!(code, Some(2), "should fail; output:\n{out}");
    assert!(
        out.contains("definitely_not_an_option"),
        "the message should name the option; output:\n{out}"
    );
}

/// `option_file_name` *inside* an options file chains nowhere — the
/// file has already been chosen by then. Upstream ignores it silently;
/// gh#518 is a report about exactly that class of silence.
#[test]
fn option_file_name_set_inside_a_file_warns() {
    let dir = scratch("chain");
    write(&dir, "ipopt.opt", "option_file_name other.opt\n");
    write(&dir, "other.opt", "max_iter 1\n");
    let (_, out) = run(&dir, &[]);
    assert!(
        out.contains("no effect") && out.contains("other.opt"),
        "chaining should be called out; output:\n{out}"
    );
    assert!(
        !out.contains(CAPPED),
        "other.opt must not actually be read; output:\n{out}"
    );
}

/// The escape hatch: a stale options file in the working directory must
/// not be inescapable now that it is picked up automatically.
#[test]
fn no_options_file_skips_the_implicit_lookup() {
    let dir = scratch("suppressed");
    write(&dir, "ipopt.opt", "max_iter 1\n");
    let (code, out) = run(&dir, &["--no-options-file"]);
    assert_eq!(code, Some(0), "solve failed; output:\n{out}");
    assert!(
        !out.contains(CAPPED) && !out.contains("Using option file"),
        "ipopt.opt should have been skipped; output:\n{out}"
    );
}

#[test]
fn no_options_file_conflicts_with_a_named_one() {
    let dir = scratch("conflict");
    write(&dir, "tiny.opt", "max_iter 1\n");
    let (code, out) = run(&dir, &["--no-options-file", "option_file_name=tiny.opt"]);
    assert_eq!(code, Some(2), "should fail; output:\n{out}");
    assert!(out.contains("conflicts"), "output:\n{out}");
}

/// `sb yes` silences the banner; the option-file line rides the same
/// gate, so a driver that suppresses the preamble still gets clean
/// output.
#[test]
fn the_option_file_line_respects_sb() {
    let dir = scratch("sb");
    write(&dir, "ipopt.opt", "sb yes\n");
    let (_, out) = run(&dir, &[]);
    assert!(
        !out.contains("Using option file"),
        "sb yes should silence it; output:\n{out}"
    );
}

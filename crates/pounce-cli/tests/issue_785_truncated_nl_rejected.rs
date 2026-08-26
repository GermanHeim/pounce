//! gh #785 — a `.nl` file that ends before one of its trailing segments must
//! be **rejected**, not solved.
//!
//! The reported file was truncated just before its `r` (constraint-bounds)
//! segment, which took `r`, `b`, `k` and `J` with it. Nothing in the parser
//! required those segments, so each one silently reverted to its default —
//! rows at ±1e19 (unconstrained), variables free, every row's linear part
//! empty — and POUNCE reported `Optimal Solution Found` / `SolveSucceeded`
//! with exit 0 and `obj=0.0` on a model whose optimum is `12.5`. That is the
//! worst possible failure mode for a corrupt input, and an outlier: an empty
//! file, non-UTF-8 garbage and a missing file each already exit 2.
//!
//! The fixture is the issue's model,
//!
//! ```text
//! min (x0-3)^2 + (x1+2)^2   s.t.  x0 + x1 == 6,  -10 <= x <= 10
//! ```
//!
//! whose optimum is closed-form: stationarity gives `x0-3 = x1+2`, so
//! `x* = (5.5, 0.5)` and `f* = 12.5`.
//!
//! The three cut points below are the three distinct ways the truncation can
//! land, and each is caught by a different check, so all three are asserted:
//!
//! | cut before | first segment lost | caught by |
//! |---|---|---|
//! | `r` | constraint bounds | the missing-`r` check |
//! | `b` | variable bounds   | the missing-`b` check |
//! | `k` | Jacobian entries  | declared `nzc` vs. parsed J entries |
//!
//! The last one is the check the issue asked for, and the only one that fires
//! when the bounds are all present and only the coefficients are gone.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture_text() -> String {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("issue_785_truncated_source.nl");
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

/// Run `pounce <file> --no-sol print_level=0`, returning `(exit code, stdout
/// + stderr)`.
fn run(nl: &std::path::Path) -> (i32, String) {
    let out = Command::new(pounce_exe())
        .arg(nl)
        .arg("--no-sol")
        .arg("print_level=0")
        .output()
        .expect("run pounce");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.code().unwrap_or(-1), text)
}

/// Write `text` to a uniquely named temp `.nl` and solve it.
fn run_text(tag: &str, text: &str) -> (i32, String) {
    let path =
        std::env::temp_dir().join(format!("pounce_issue785_{}_{tag}.nl", std::process::id()));
    std::fs::write(&path, text).expect("write temp .nl");
    let r = run(&path);
    let _ = std::fs::remove_file(&path);
    r
}

/// The complete file is the oracle: it must still solve, to `f* = 12.5`.
/// Without this the truncation assertions below would pass just as well
/// against a parser that rejects the *whole* fixture.
#[test]
fn complete_file_still_solves_to_the_closed_form_optimum() {
    let (code, text) = run_text("complete", &fixture_text());
    assert_eq!(code, 0, "complete file should solve:\n{text}");
    assert!(
        text.contains("obj=12.50000000"),
        "complete file should reach f*=12.5:\n{text}"
    );
}

/// Truncation before `r`: the case as reported.
#[test]
fn truncated_before_r_is_rejected() {
    let src = fixture_text();
    let cut = src.find("\nr\n").expect("fixture has an r segment") + 1;
    let (code, text) = run_text("before_r", &src[..cut]);
    assert_eq!(code, 2, "truncated file must exit 2, got {code}:\n{text}");
    assert!(
        text.contains("`r` (constraint-bounds) segment"),
        "error should name the missing segment:\n{text}"
    );
    // The whole point: no confident wrong answer.
    assert!(
        !text.contains("Optimal Solution Found"),
        "truncated file must not report success:\n{text}"
    );
}

/// Truncation between `r` and `b`: the bounds on the row survived, the ones
/// on the variables did not.
#[test]
fn truncated_before_b_is_rejected() {
    let src = fixture_text();
    let cut = src.find("\nb\n").expect("fixture has a b segment") + 1;
    let (code, text) = run_text("before_b", &src[..cut]);
    assert_eq!(code, 2, "truncated file must exit 2, got {code}:\n{text}");
    assert!(
        text.contains("`b` (variable-bounds) segment"),
        "error should name the missing segment:\n{text}"
    );
    assert!(
        !text.contains("Optimal Solution Found"),
        "truncated file must not report success:\n{text}"
    );
}

/// Truncation after every bound segment, so only the Jacobian is gone. This
/// is the declared-vs-parsed nonzero cross-check on its own — the bound
/// checks above cannot see this one.
#[test]
fn truncated_before_the_jacobian_is_rejected() {
    let src = fixture_text();
    let cut = src.find("\nk1\n").expect("fixture has a k segment") + 1;
    let (code, text) = run_text("before_j", &src[..cut]);
    assert_eq!(code, 2, "truncated file must exit 2, got {code}:\n{text}");
    assert!(
        text.contains("Jacobian nonzero"),
        "error should report the declared-vs-parsed mismatch:\n{text}"
    );
    assert!(
        !text.contains("Optimal Solution Found"),
        "truncated file must not report success:\n{text}"
    );
}

/// Dropping only the `J` segment, keeping `r`, `b` and `k`, is the issue's
/// isolation variant. It previously reported `InfeasibleProblemDetected` —
/// a claim about the *model*, made from a corrupt file. The header says the
/// row has two coefficients, so this is a parse error like the rest.
#[test]
fn dropping_only_the_jacobian_segment_is_a_parse_error_not_an_infeasibility() {
    let src = fixture_text().replace("J0 2\n0 1\n1 1\n", "");
    let (code, text) = run_text("no_j", &src);
    assert_eq!(code, 2, "corrupt file must exit 2, got {code}:\n{text}");
    assert!(
        text.contains("Jacobian nonzero"),
        "error should report the declared-vs-parsed mismatch:\n{text}"
    );
}

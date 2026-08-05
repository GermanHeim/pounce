//! `linear_solver` accepted every backend name and silently ran FERAL
//! (gh#483 follow-up).
//!
//! The option's valid-value list is a faithful port of upstream Ipopt's
//! — `ma27`, `ma57`, `ma77`, `ma86`, `ma97`, `mumps`, `pardiso`,
//! `pardisomkl`, `spral`, `wsmp`, `custom`, `feral` — so an `ipopt.opt`
//! written for Ipopt parses here unchanged. pounce implements two of
//! them. The resolver mapped everything else through a `_ =>` arm to
//! FERAL, so `linear_solver=ma97` "worked": a run that reported success
//! while using a backend the binary does not contain, and a benchmark
//! comparing linear solvers that compared FERAL against itself.
//!
//! Two boundaries this pins, because both are load-bearing:
//!
//! * The **registered default is `ma57`** (upstream's), not a user
//!   request. On the pure-Rust build it resolves to FERAL and always
//!   has, so a default run must stay silent — erroring on it would fail
//!   every solve.
//! * **Explicit `ma57` on a build without the feature** still falls back.
//!   That fallback is reported in the banner rather than hidden, and
//!   failing a portable `ipopt.opt` over a build flag would cost more
//!   than it buys.
//!
//! The refusal is checked before routing, so it does not depend on
//! whether the model classifies into the NLP path or the convex one —
//! the convex dispatch never reaches `optimize_tnlp`, where the sibling
//! guard for library consumers lives.

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

/// Run `fixture_name` with `opts`; returns `(exit code, stderr)`.
fn run(fixture_name: &str, tag: &str, opts: &[&str]) -> (Option<i32>, String) {
    let dir = std::env::temp_dir().join(format!("pounce_linsolsel_{tag}"));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let nl = dir.join(fixture_name);
    std::fs::copy(fixture(fixture_name), &nl).expect("copy fixture");
    let out = Command::new(pounce_exe())
        .arg(&nl)
        .args(opts)
        .arg("print_level=0")
        .output()
        .expect("run pounce");
    let _ = std::fs::remove_dir_all(&dir);
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Every backend pounce does not implement is refused, by name, with a
/// message that says what to use instead. Pre-fix each of these exited
/// 0 having run FERAL.
#[test]
fn unimplemented_backends_are_refused() {
    for (i, name) in [
        "ma27",
        "ma77",
        "ma86",
        "ma97",
        "mumps",
        "pardiso",
        "pardisomkl",
        "spral",
        "wsmp",
        "custom",
    ]
    .into_iter()
    .enumerate()
    {
        let (code, err) = run(
            "user_scaling_suffix.nl",
            &format!("bad{i}"),
            &[&format!("linear_solver={name}")],
        );
        assert_eq!(code, Some(2), "linear_solver={name} should fail; {err}");
        assert!(
            err.contains(&format!("linear_solver={name} is not implemented")),
            "the refusal must name the backend; stderr:\n{err}",
        );
        assert!(
            err.contains("linear_solver=feral"),
            "the refusal must say what to use instead; stderr:\n{err}",
        );
    }
}

/// The option value is case-insensitive in the registry, so the guard
/// must be too — `MUMPS` cannot be a way around it.
#[test]
fn the_refusal_is_case_insensitive() {
    let (code, err) = run("user_scaling_suffix.nl", "upper", &["linear_solver=MUMPS"]);
    assert_eq!(code, Some(2), "stderr:\n{err}");
}

/// The two pounce implements are accepted. `ma57` is accepted on any
/// build: without the feature it falls back to FERAL and the banner says
/// so, which is a reported substitution rather than a hidden one.
#[test]
fn implemented_backends_are_accepted() {
    for name in ["feral", "ma57"] {
        let (code, err) = run(
            "user_scaling_suffix.nl",
            name,
            &[&format!("linear_solver={name}")],
        );
        assert_eq!(code, Some(0), "linear_solver={name} should solve; {err}");
    }
}

/// The registered default is upstream's `ma57`, which is not a user
/// request — a plain run must be unaffected. This is the assertion that
/// would have caught a guard written without the `found` gate, which
/// would fail every default solve on the pure-Rust build.
#[test]
fn the_registered_default_is_not_treated_as_a_request() {
    let (code, err) = run("user_scaling_suffix.nl", "default", &[]);
    assert_eq!(code, Some(0), "a default run must not be refused; {err}");
    assert!(
        !err.contains("not implemented"),
        "a default run must not warn either; stderr:\n{err}",
    );
}

/// The guard runs before solver routing: a model that classifies as a
/// convex QP dispatches to `pounce-convex`, which never reaches the
/// library-side guard in `optimize_tnlp`.
#[test]
fn the_refusal_covers_the_convex_route() {
    let (code, err) = run("boxed_qp_min.nl", "convex", &["linear_solver=mumps"]);
    assert_eq!(code, Some(2), "stderr:\n{err}");
    assert!(err.contains("not implemented"), "stderr:\n{err}");
}

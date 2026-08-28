//! An `--features ma57` build of `pounce` must actually run, and its
//! `ma57_*` options must actually do something.
//!
//! Two defects, one test file, because they were found together and
//! neither is visible from the other's vantage point.
//!
//! **gh#811 — the binary would not start.** `crates/pounce-hsl/build.rs`
//! emitted `cargo:rustc-link-arg=-Wl,-rpath,<coinhsl>/lib`, and that
//! directive applies only to targets in the package that emits it.
//! pounce-hsl is a library with no binary, so the flag reached nothing:
//! `cargo build -p pounce-cli --features ma57` linked fine and the
//! resulting `pounce` died at process start with
//!
//! ```text
//! dyld: Library not loaded: @rpath/libcoinhsl.dylib
//!   Reason: no LC_RPATH's found
//! ```
//!
//! It could not even be patched afterwards — a release link leaves no
//! header padding, so `install_name_tool -add_rpath` refuses. The fix
//! passes the directory to dependents as `links` metadata
//! (`DEP_COINHSL_RPATH`), which `crates/pounce-cli/build.rs` re-emits.
//! Linux with `libcoinhsl.so` on the loader path never showed it, which
//! is why it went unnoticed.
//!
//! **gh#825 — the options did nothing.** All nine `ma57_*` options were
//! registered, documented, accepted, and discarded, because every
//! production construction of the backend called
//! `Ma57SolverInterface::new()`. No warning, no diagnostic, no journal
//! message: two solves whose option blocks differed by eight orders of
//! magnitude in `ma57_pivtol` printed identical logs.
//!
//! Only runs under `--features ma57`; CI cannot link CoinHSL. The
//! always-runnable companions are
//! `pounce-algorithm/tests/no_production_site_builds_ma57_with_defaults.rs`
//! (gh#825's signature, checked in source) and the fact that
//! `default_backend_factory` now requires a `Ma57Config` argument.
#![cfg(feature = "ma57")]

use std::path::PathBuf;
use std::process::Command;

fn pounce_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

/// gh#811. The dynamic loader has to resolve `@rpath/libcoinhsl.dylib`
/// before `main` runs, so this fails at *process start* — no output, a
/// signal or a nonzero status, and a dyld message on stderr — rather
/// than anywhere the rest of the suite would notice.
#[test]
fn the_ma57_binary_starts() {
    let out = Command::new(pounce_bin())
        .arg("--version")
        .output()
        .expect("spawn pounce");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("Library not loaded") && !stderr.contains("LC_RPATH"),
        "the ma57 build of `pounce` cannot start — the CoinHSL rpath did not reach the \
         binary (gh#811). Check that crates/pounce-hsl/build.rs still emits \
         `cargo:rpath=` and that crates/pounce-cli/build.rs still re-emits it from \
         DEP_COINHSL_RPATH.\nstderr: {stderr}"
    );
    assert!(out.status.success(), "`pounce --version` failed: {stderr}");
}

/// The binary agrees it has MA57 compiled in. Guards against the test
/// silently degrading into a check of a FERAL-only build, which would
/// pass every assertion in this file for the wrong reason.
#[test]
fn the_binary_reports_ma57_enabled() {
    let out = Command::new(pounce_bin())
        .arg("--about")
        .output()
        .expect("spawn pounce");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("ma57:           enabled"),
        "built with --features ma57 but `--about` does not report it:\n{stdout}"
    );
}

/// A run's outcome: the iteration count of each solve it performed, in
/// order, plus the status it finally reported.
///
/// A *vector*, because a POUNCE run is not always one solve — deb7 ends
/// `Solved_To_Acceptable_Level`, which opens the auto-fallback retry, so
/// a default run reports two blocks. Reading only the first or only the
/// last would compare one arm's original solve against another arm's
/// retry. The whole sequence is the trajectory signature.
fn run(args: &[&str]) -> (Vec<u32>, String) {
    let out = Command::new(pounce_bin())
        .arg(fixture("deb7.nl"))
        .arg("linear_solver=ma57")
        .args(args)
        .output()
        .expect("spawn pounce");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr);
    let status = stdout
        .lines()
        .find_map(|l| l.trim().strip_prefix("Status:"))
        .unwrap_or_else(|| {
            panic!("{args:?} produced no status line\nstdout: {stdout}\nstderr: {stderr}")
        })
        .trim()
        .to_string();
    let iters = stdout
        .lines()
        .filter_map(|l| l.trim().strip_prefix("Number of Iterations....:"))
        .filter_map(|v| v.trim().parse::<u32>().ok())
        .collect::<Vec<_>>();
    assert!(
        !iters.is_empty(),
        "{args:?} produced no iteration counts:\n{stdout}"
    );
    (iters, status)
}

fn solved(status: &str) -> bool {
    matches!(status, "Solve_Succeeded" | "Solved_To_Acceptable_Level")
}

/// The dylib loads and MA57 actually solves — a stronger statement than
/// "the process started", since `libcoinhsl`'s own `@rpath` dependencies
/// (openblas, metis, libgfortran, libgomp) are resolved lazily and a
/// missing one would surface only here.
#[test]
fn ma57_solves_a_model() {
    let (_, status) = run(&[]);
    assert!(
        solved(&status),
        "linear_solver=ma57 did not solve deb7: {status}"
    );
}

/// gh#825, end to end and in the user's own terms: setting an `ma57_*`
/// option changes the solve.
///
/// `ma57_pivtol` goes from the default `1e-8` — pivot for sparsity — to
/// `0.5`, LAPACK's maximum-stability threshold. That is eight orders of
/// magnitude on `CNTL(1)`; it changes which entries MA57 accepts as
/// pivots and therefore the rounding of every factorization, so it
/// cannot leave the trajectory untouched. Before the fix these two runs
/// were identical to all seventeen digits of the objective, which is
/// exactly what the issue reported.
///
/// Asserted as "the signatures differ", never as specific numbers: the
/// property under test is that the option is *connected*, and pinning a
/// count here would make an unrelated trajectory change fail this test
/// for a reason its name does not describe.
#[test]
fn an_ma57_option_changes_the_solve() {
    let (baseline, base_status) = run(&[]);
    let (stabilized, stab_status) = run(&["ma57_pivtol=0.5"]);
    assert!(solved(&base_status) && solved(&stab_status));
    assert_ne!(
        baseline, stabilized,
        "ma57_pivtol=0.5 left the solve identical to the 1e-8 default. Either the option is \
         being discarded again (gh#825), or it is reaching MA57 and genuinely changing \
         nothing — the first is a bug and the second is worth understanding before this \
         assertion is relaxed."
    );
}

/// The `"resto."` prefix reaches the restoration sub-IPM's backend, and
/// only that one.
///
/// Upstream's `Ma57TSolverInterface::InitializeImpl` takes a prefix and
/// pounce's reader always mirrored it — but nothing called the reader,
/// so the facility did not exist in practice (gh#825, impact 4). The
/// restoration sub-IPM builds its *own* backend, through its own
/// `InnerBackendFactoryFactory`, which is what makes the two separately
/// configurable at all.
///
/// Both halves are asserted, because either alone passes while the other
/// is broken:
///
/// * against the baseline — a prefixed value that reached nothing would
///   leave the run untouched;
/// * against the *un*-prefixed arm — a prefix that were simply ignored
///   would make `resto.ma57_pivtol` a synonym for `ma57_pivtol` and
///   reproduce that arm instead.
#[test]
fn the_resto_prefix_reaches_only_the_restoration_subsolve() {
    let (baseline, base_status) = run(&[]);
    let (unprefixed, _) = run(&["ma57_pivtol=0.5"]);
    let (prefixed, prefixed_status) = run(&["resto.ma57_pivtol=0.5"]);
    assert!(solved(&base_status) && solved(&prefixed_status));

    assert_ne!(
        baseline, prefixed,
        "`resto.ma57_pivtol=0.5` did not move the run. This fixture is only evidence while \
         its solve actually enters restoration — if deb7's trajectory has changed so that it \
         no longer does, this test needs a model that does, not a relaxed assertion."
    );
    assert_ne!(
        unprefixed, prefixed,
        "`resto.ma57_pivtol=0.5` reproduced the un-prefixed arm exactly, which is what an \
         ignored prefix looks like: the value reached the main IPM instead of the \
         restoration sub-IPM."
    );
}

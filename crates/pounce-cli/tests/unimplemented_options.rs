//! Options naming features pounce does not implement are refused
//! (gh#483 follow-up, continuing #191).
//!
//! `upstream_options.rs` registers every name Ipopt registers, so an
//! `ipopt.opt` written for Ipopt parses unchanged — a real compatibility
//! benefit that also turned ~200 knobs into silent no-ops, because
//! registering an option says nothing about implementing it.
//!
//! #191 fixed the half where the feature runs and only the read site was
//! missing, and explicitly scoped out "feature genuinely unimplemented —
//! expected no-ops". This is that other half.
//!
//! The table's membership rules, and why an explicitly-set *default* is
//! still allowed, live in `pounce-algorithm/src/unimplemented_options.rs`
//! alongside the unit tests for the predicate. What is checked here is
//! the CLI's end of it: the exit code, that the message reaches stderr,
//! and — the part only an end-to-end test can show — that the guard runs
//! before solver routing.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn run(fixture_name: &str, tag: &str, opts: &[&str]) -> (Option<i32>, String) {
    let dir = std::env::temp_dir().join(format!("pounce_unimplopt_{tag}"));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let nl = dir.join(fixture_name);
    let mut fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fixture.push("tests/fixtures");
    fixture.push(fixture_name);
    std::fs::copy(&fixture, &nl).expect("copy fixture");
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

/// One representative per feature group, so a group dropped from the
/// table fails here rather than going quiet again.
#[test]
fn requesting_an_unimplemented_feature_fails_with_an_explanation() {
    for (i, (opt, needle)) in [
        ("penalty_init_max=42", "CG-penalty"),
        (
            "gradient_approximation=finite-difference-values",
            "finite differences",
        ),
        ("dependency_detector=mumps", "linear-dependency detection"),
        ("check_derivatives_for_naninf=yes", "NaN/Inf"),
        ("magic_steps=yes", "magic steps"),
        // #551: the two line-search knobs whose *feature* is missing.
        // `theta_min` is the CG-penalty acceptor's threshold, not the
        // filter's (the filter derives its own from `theta_min_fact`);
        // `alpha_for_y_tol` only configures the `primal-and-full` /
        // `dual-and-full` multiplier-step rules, which pounce does not
        // have.
        ("theta_min=1e-5", "CG-penalty acceptor"),
        ("alpha_for_y_tol=1e-3", "primal-and-full"),
        ("suppress_all_output=yes", "output controls"),
        ("hsllib=libcoinhsl.so", "HSL loader"),
    ]
    .into_iter()
    .enumerate()
    {
        let (code, err) = run("user_scaling_suffix.nl", &format!("g{i}"), &[opt]);
        assert_eq!(code, Some(2), "`{opt}` should fail; stderr:\n{err}");
        assert!(
            err.contains(needle),
            "`{opt}` should mention `{needle}`; stderr:\n{err}",
        );
        // Every group names its tracking issue; the older ones are
        // gh#483, the #551 line-search pair carry 551.
        assert!(err.contains("483") || err.contains("551"), "stderr:\n{err}",);
    }
}

/// The message names the option, not just the feature — with ~200
/// registered names, "some option is unsupported" would be useless.
#[test]
fn the_refusal_names_the_offending_option() {
    let (_, err) = run("user_scaling_suffix.nl", "named", &["vartheta=0.9"]);
    assert!(err.contains("`vartheta`"), "stderr:\n{err}");
}

/// Setting an option to its registered default asks for nothing. A
/// generated `ipopt.opt` spells out defaults, and failing on that would
/// break the compatibility the registry exists to provide.
#[test]
fn explicitly_setting_a_default_still_solves() {
    for (i, opt) in ["dependency_detector=none", "magic_steps=no", "recalc_y=no"]
        .into_iter()
        .enumerate()
    {
        let (code, err) = run("user_scaling_suffix.nl", &format!("d{i}"), &[opt]);
        assert_eq!(code, Some(0), "`{opt}` asks for nothing; stderr:\n{err}");
        assert!(!err.contains("does not implement"), "stderr:\n{err}");
    }
}

/// Options whose feature *runs* and only whose read site is missing must
/// keep solving — refusing them would fail solves that are correct
/// today. Wiring them is separate work.
#[test]
fn knobs_on_implemented_features_still_solve() {
    for (i, opt) in [
        "max_resto_iter=17",
        "accept_after_max_steps=3",
        "limited_memory_max_skipping=4",
        "corrector_type=affine",
        // #677: `recalc_y` was refused as unimplemented until the
        // least-square multiplier recalculation landed. It is a real
        // feature now, so asking for it must solve rather than fail.
        "recalc_y=yes",
        "recalc_y_feas_tol=1e-4",
    ]
    .into_iter()
    .enumerate()
    {
        let (code, err) = run("user_scaling_suffix.nl", &format!("b{i}"), &[opt]);
        assert_eq!(
            code,
            Some(0),
            "`{opt}` configures an implemented feature; stderr:\n{err}",
        );
    }
}

/// The four constant-derivative hints used to warn "pounce does not
/// exploit this" and re-evaluate anyway. gh #588 Q6 exploits them, and
/// this pin flips with it: what a hint earns now is decided by the
/// model's own algebra, and what it earns is *measured in evaluations*
/// rather than asserted from a warning string.
///
/// `user_scaling_suffix.nl` has a `∇²L` POUNCE proves is not constant,
/// so asserting `hessian_constant=yes` on it is the case upstream Ipopt
/// honours — reusing a Hessian that genuinely moves — and the case
/// POUNCE refuses. The solve must still succeed: the option is ignored,
/// not fatal.
#[test]
fn a_disproved_caching_hint_is_refused_with_a_warning() {
    let (code, err) = run("user_scaling_suffix.nl", "hint", &["hessian_constant=yes"]);
    assert_eq!(code, Some(0), "stderr:\n{err}");
    assert!(err.contains("warning"), "stderr:\n{err}");
    assert!(err.contains("hessian_constant"), "stderr:\n{err}");
    assert!(
        err.contains("ignoring"),
        "the warning must say the hint was refused, not merely unused; \
         stderr:\n{err}"
    );
}

/// The other half of the flip, and the half a string match cannot reach:
/// a model whose `∇²L` and Jacobian POUNCE *proves* constant is evaluated
/// **once**, with no option set at all.
///
/// `nonconvex_qp.nl` is a quadratic objective over linear rows forced
/// down the NLP path (the convex-QP route would not exercise this), so
/// `∇²L = σ∇²f` and every row's gradient is a constant vector. The
/// assertion is `num_hess_evals == 1` against an iteration count well
/// above 1 — before Q6 both counters tracked the iterations.
#[test]
fn a_proved_constant_hessian_is_evaluated_once() {
    let dir = std::env::temp_dir().join("pounce_unimplopt_constderiv");
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let nl = dir.join("nonconvex_qp.nl");
    let mut fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fixture.push("tests/fixtures/nonconvex_qp.nl");
    std::fs::copy(&fixture, &nl).expect("copy fixture");
    let json = dir.join("out.json");
    let out = Command::new(pounce_exe())
        .arg(&nl)
        .arg("--json-output")
        .arg(&json)
        .arg("print_level=0")
        .output()
        .expect("run pounce");
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = std::fs::read_to_string(&json).expect("json report");
    let _ = std::fs::remove_dir_all(&dir);

    // Small, flat JSON: pull the three integers out without a parser
    // dependency this test crate does not otherwise have.
    fn field(text: &str, key: &str) -> i64 {
        let at = text
            .find(&format!("\"{key}\""))
            .unwrap_or_else(|| panic!("`{key}` missing from report:\n{text}"));
        let rest = &text[at + key.len() + 2..];
        let start = rest.find(':').expect("colon") + 1;
        let end = rest[start..]
            .find([',', '}'])
            .map(|e| start + e)
            .unwrap_or(rest.len());
        rest[start..end].trim().parse().expect("integer field")
    }
    let iters = field(&text, "iteration_count");
    let hess = field(&text, "num_hess_evals");
    let jac = field(&text, "num_constr_jac_evals");
    assert!(iters > 1, "expected a multi-iteration solve, got {iters}");
    assert_eq!(
        hess, 1,
        "`∇²L` is provably constant on this model; it must be evaluated \
         once and reused for all {iters} iterations"
    );
    assert_eq!(
        jac, 1,
        "every row is linear; the Jacobian must be evaluated once (got \
         {jac} over {iters} iterations)"
    );
}

/// The guard runs before routing: a convex-QP model dispatches to
/// `pounce-convex` and never reaches the library-side guard.
#[test]
fn the_refusal_covers_the_convex_route() {
    let (code, err) = run("boxed_qp_min.nl", "convex", &["magic_steps=yes"]);
    assert_eq!(code, Some(2), "stderr:\n{err}");
    assert!(err.contains("magic steps"), "stderr:\n{err}");
}

/// A plain run is untouched — no refusal, no warning.
#[test]
fn a_default_run_is_silent() {
    let (code, err) = run("user_scaling_suffix.nl", "plain", &[]);
    assert_eq!(code, Some(0), "stderr:\n{err}");
    assert!(!err.contains("does not implement"), "stderr:\n{err}");
    assert!(!err.contains("warning:"), "stderr:\n{err}");
}

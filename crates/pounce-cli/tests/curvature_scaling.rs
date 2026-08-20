//! `nlp_scaling_method=curvature-based` end-to-end (gh #703).
//!
//! The default `gradient-based` scaling is a point sample: it reads the
//! Jacobian once at x₀ and cannot see a row whose derivative vanishes
//! there, nor a column imbalance that the sample happens not to expose.
//! `curvature-based` derives the factors from the model's quadratic
//! coefficients instead — one joint variable scaling `D` equilibrated
//! across `Q₀ + Σλᵢ Qᵢ`, then a per-row `eᵢ = 1/max(‖D Qᵢ D‖_∞,
//! ‖D aᵢ‖_∞, |bᵢ|)`.
//!
//! What is pinned here:
//!
//! * the **invariance** claim, against a fixture pair that is one problem
//!   in two coordinate systems (see `fixtures/README-qcqp-columns.md`):
//!   `curvature-based` returns the well-conditioned answer bit for bit on
//!   the ill-conditioned twin, where `gradient-based` loses five digits;
//! * that it **refuses** a model it is not defined for rather than solving
//!   it unscaled, which is the gh #483 failure this option must not repeat;
//! * that it is **off unless asked for** — the default path is untouched.
//!
//! Every solve passes `solver_selection=nlp`. At this size both fixtures
//! clear the conic guards and would route to the SOCP driver; gh #703 is
//! about the NLP path, and pinning the route is what keeps this test
//! measuring the thing it names.

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

fn tmp(suffix: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("pounce_curv_{}_{suffix}", std::process::id()));
    p
}

/// Solve `nl` and return `(status, iterations, objective)`.
fn solve(nl: &str, extra: &[&str]) -> (String, u64, f64) {
    let sol = tmp(&format!("{nl}.sol"));
    let json = tmp(&format!("{nl}.json"));
    let out = Command::new(pounce_exe())
        .arg(fixture(nl))
        .arg(&sol)
        .arg("--json-output")
        .arg(&json)
        .arg("print_level=0")
        .arg("solver_selection=nlp")
        .args(extra)
        .output()
        .expect("spawn pounce");
    assert_eq!(
        out.status.code(),
        Some(0),
        "solve of {nl} {extra:?} failed\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let text = std::fs::read_to_string(&json).expect("json report");
    let v: serde_json::Value = serde_json::from_str(&text).expect("json parses");
    let status = v["solution"]["status"].as_str().unwrap_or("?").to_string();
    let iters = v["statistics"]["iteration_count"].as_u64().unwrap_or(0);
    let obj = v["statistics"]["final_objective"]
        .as_f64()
        .unwrap_or(f64::NAN);
    let _ = std::fs::remove_file(&sol);
    let _ = std::fs::remove_file(&json);
    (status, iters, obj)
}

/// The headline claim: the scheme recovers the injected column scaling, so
/// the solve stops depending on which coordinates the model was written in.
#[test]
fn curvature_scaling_is_invariant_to_a_column_imbalance() {
    let (s_well, it_well, obj_well) = solve("qcqp_columns_wellcond.nl", &[]);
    assert_eq!(s_well, "SolveSucceeded");

    // The default loses digits on the ill-conditioned twin: same problem,
    // different coordinates, a visibly different answer.
    let (s_ill, it_ill, obj_ill) = solve("qcqp_columns_illcond.nl", &[]);
    assert_eq!(s_ill, "SolveSucceeded");
    assert!(
        (obj_ill - obj_well).abs() > 1e-6,
        "the fixture pair no longer exercises the failure it was built for: \
         gradient-based got {obj_ill} vs {obj_well}"
    );
    assert!(
        it_ill > it_well,
        "expected the ill-conditioned twin to cost more iterations under \
         the default ({it_ill} vs {it_well})"
    );

    // curvature-based reproduces the well-conditioned answer on the
    // ill-conditioned model — to the last bit, not merely to a tolerance.
    let (s_a, it_a, obj_a) = solve(
        "qcqp_columns_wellcond.nl",
        &["nlp_scaling_method=curvature-based"],
    );
    let (s_b, it_b, obj_b) = solve(
        "qcqp_columns_illcond.nl",
        &["nlp_scaling_method=curvature-based"],
    );
    assert_eq!(s_a, "SolveSucceeded");
    assert_eq!(s_b, "SolveSucceeded");
    // Not bit equality: `D` recovers the injected `c` up to how the Ruiz
    // sweep's factors round, so the two solves agree to **one ulp** rather
    // than exactly. Pinned as the measured distance, not as a tolerance —
    // the claim is that the coordinate system stops mattering, and a
    // regression that made it matter would have to move this number.
    let ulps = obj_a.to_bits().abs_diff(obj_b.to_bits());
    assert!(
        ulps <= 2,
        "curvature-based should make the two coordinate systems the same \
         solve to within an ulp: {obj_a} vs {obj_b} ({ulps} ulp)"
    );
    assert_eq!(it_a, it_b, "…including the trajectory length");
    assert!(
        (obj_a - obj_well).abs() < 1e-8,
        "and it should agree with the well-conditioned reference \
         ({obj_a} vs {obj_well})"
    );
    assert!(
        it_b < it_ill,
        "expected fewer iterations than the default on the ill-conditioned \
         twin ({it_b} vs {it_ill})"
    );
}

/// A model with a genuine nonlinearity has no constant `Qᵢ`, so the
/// magnitude envelope this scheme equilibrates does not exist. It must say
/// so rather than accept the option and solve unscaled — the shape of
/// gh #483, which is why the message names the alternatives.
#[test]
fn a_model_without_constant_curvature_is_refused_readably() {
    let sol = tmp("refuse.sol");
    let out = Command::new(pounce_exe())
        .arg(fixture("cresc4.nl"))
        .arg(&sol)
        .arg("print_level=0")
        .arg("nlp_scaling_method=curvature-based")
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(2), "expected a usage-style refusal");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("curvature-based") && err.contains("degree <= 2"),
        "refusal should name the option and the reason, got: {err}"
    );
    assert!(
        err.contains("gradient-based") && err.contains("user-scaling"),
        "refusal should name what to use instead, got: {err}"
    );
}

/// The option is opt-in: without it, the ill-conditioned fixture solves
/// exactly as it did before this feature existed.
#[test]
fn the_default_path_is_untouched() {
    let (s1, it1, obj1) = solve("qcqp_columns_illcond.nl", &[]);
    let (s2, it2, obj2) = solve(
        "qcqp_columns_illcond.nl",
        &["nlp_scaling_method=gradient-based"],
    );
    assert_eq!((s1, it1, obj1.to_bits()), (s2, it2, obj2.to_bits()));
}

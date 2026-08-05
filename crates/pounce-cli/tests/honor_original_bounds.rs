//! `honor_original_bounds` was registered and never read (gh#483
//! follow-up), so the reported solution could sit outside the bounds the
//! user declared with no way to turn that off.
//!
//! `bound_relax_factor` (default `1e-8`) widens the variable box before
//! the solve — upstream does this too — so a solution pinned to a bound
//! comes back just past it. On
//! `min (x−3)² + (y+2)²  s.t.  x ∈ [0,1], y ∈ [−1,1]`, whose optimum
//! pins both, pounce reported
//!
//! ```text
//! x = 1.00000000937320332      (upper bound 1)
//! y = -1.00000000874562045     (lower bound -1)
//! ```
//!
//! Upstream registers `honor_original_bounds` precisely to project that
//! back; pounce accepted the option and did nothing, so there was no way
//! to get a point inside the declared box. That is not cosmetic — the
//! value flows into a downstream `sqrt(1 − x)`, a domain assertion, or a
//! Pyomo `Var` whose bounds it is loaded back into.
//!
//! The default stays `no`, matching upstream: this test pins both the
//! unchanged default and the now-working opt-in.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

/// Solve `bound_active_qp.nl` and return its `(x, y)` from the `.sol`.
/// `solver_selection=nlp` because bound relaxation — and therefore the
/// projection — belongs to the NLP interior-point path.
fn solve(tag: &str, opts: &[&str]) -> (f64, f64) {
    let dir = std::env::temp_dir().join(format!("pounce_honorbounds_{tag}"));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let nl = dir.join("bound_active_qp.nl");
    let mut fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fixture.push("tests/fixtures/bound_active_qp.nl");
    std::fs::copy(&fixture, &nl).expect("copy fixture");

    let out = Command::new(pounce_exe())
        .arg(&nl)
        .arg("solver_selection=nlp")
        .arg("print_level=0")
        .args(opts)
        .output()
        .expect("run pounce");
    assert!(
        out.status.success(),
        "solve failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let sol = std::fs::read_to_string(nl.with_extension("sol")).expect("read .sol");
    let _ = std::fs::remove_dir_all(&dir);
    // m = 0, n = 2: the two numeric lines after the header are x and y.
    let vals: Vec<f64> = sol
        .lines()
        .filter_map(|l| l.trim().parse::<f64>().ok())
        .collect();
    let n = vals.len();
    (vals[n - 2], vals[n - 1])
}

/// Default (`no`): the relaxed point is reported as-is, unchanged from
/// before this option was wired and matching upstream's default.
#[test]
fn default_reports_the_unprojected_point() {
    let (x, y) = solve("default", &[]);
    assert!(x > 1.0, "expected x just past its upper bound, got {x}");
    assert!(y < -1.0, "expected y just past its lower bound, got {y}");
    // …but only by the relaxation, not by anything larger.
    assert!((x - 1.0).abs() < 1e-6 && (y + 1.0).abs() < 1e-6);
}

/// `honor_original_bounds=yes`: the reported point is inside the box the
/// user declared, exactly on the active bounds. Pre-fix this was
/// byte-identical to the default.
#[test]
fn opting_in_projects_back_into_the_declared_bounds() {
    let (x, y) = solve("honored", &["honor_original_bounds=yes"]);
    assert!(x <= 1.0, "x = {x} is still outside its upper bound 1");
    assert!(y >= -1.0, "y = {y} is still below its lower bound -1");
    assert_eq!(x, 1.0, "an active bound should project exactly onto it");
    assert_eq!(y, -1.0);
}

/// The projection only clamps: a component strictly inside its bounds
/// is untouched, so turning the option on cannot move an interior
/// solution.
#[test]
fn an_interior_solution_is_unmoved() {
    let dir = std::env::temp_dir().join("pounce_honorbounds_interior");
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let nl = dir.join("boxed_qp_min.nl");
    let mut fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fixture.push("tests/fixtures/user_scaling_suffix.nl");
    std::fs::copy(&fixture, &nl).expect("copy fixture");

    let read = |opts: &[&str]| -> Vec<f64> {
        let out = Command::new(pounce_exe())
            .arg(&nl)
            .arg("solver_selection=nlp")
            .arg("print_level=0")
            .args(opts)
            .output()
            .expect("run pounce");
        assert!(out.status.success());
        std::fs::read_to_string(nl.with_extension("sol"))
            .expect("read .sol")
            .lines()
            .filter_map(|l| l.trim().parse::<f64>().ok())
            .collect()
    };
    // `user_scaling_suffix.nl` is unbounded in x, so nothing can clamp.
    let plain = read(&[]);
    let honored = read(&["honor_original_bounds=yes"]);
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(plain, honored);
}

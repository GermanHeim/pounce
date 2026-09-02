//! gh #880 at the CLI, which is where the demotion's *consequences* live.
//!
//! The convex `σ` cascade can return a point it declined to certify. That now
//! comes back as `OptimalInaccurate` rather than a bare `Optimal`, and inside
//! the library that is only a label. At the CLI it is three things:
//!
//! 1. `lp_declines_to_nlp` reroutes the model to the general NLP arm
//!    (`main.rs`, gh #535 / #888);
//! 2. `qp_status_to_ars` maps it to `SolvedToAcceptableLevel`;
//! 3. `convex_status_report` returns AMPL `solve_result_num` `1`, not `0`.
//!
//! **None of that is reachable from the fixture corpus.**
//! `scripts/sweep-fixtures.sh` is empty across all 79 fixtures, and that is
//! not evidence of safety here — exactly one fixture reaches `σ` at all
//! (`qcqp_columns_illcond`), and it is a QCQP, which the cascade's
//! non-orthant early return excludes from the recording site entirely. So the
//! corpus has *zero* coverage of this population, which is the shape CLAUDE.md
//! warns about ("an empty sweep is not evidence about it"). This file is the
//! coverage.
//!
//! `issue880_sigma_uncertified.nl` is the crate's own
//! `coupled_qp([1.0, 1e12], [3.0, 0.5], 1.0)` written out as an `.nl`: a
//! 2-variable unconstrained QP whose Hessian is `diag(1, 1e12)` rotated by
//! `(cos, sin) = (0.6, 0.8)`, so the exact minimiser is `(3.0, 0.5)` by
//! construction and the objective there is `-2.205e12`.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

fn run(opts: &[&str]) -> pounce_cli::solve_report::SolveReport {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("pounce_issue880_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/issue880_sigma_uncertified.nl");
    std::fs::copy(&src, dir.join("m.nl")).expect("copy fixture");
    let json = dir.join("r.json");
    let mut cmd = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")));
    cmd.current_dir(&dir)
        .arg("m.nl")
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json);
    for o in opts {
        cmd.arg(o);
    }
    let _ = cmd.output().expect("run pounce");
    let text = std::fs::read_to_string(&json).expect("read json report");
    serde_json::from_str(&text).expect("deserialize SolveReport")
}

/// Under `auto` the demotion routes the model to the NLP arm, which certifies
/// it. The engine column is the assertion that matters: status and objective
/// would both be satisfied by the convex arm's own answer, so a routing
/// regression would be invisible without it — the gh #760 lesson.
#[test]
fn a_demoted_convex_solve_reroutes_to_the_nlp_arm() {
    let v = run(&[]);
    assert_eq!(
        v.solution.engine, "nlp",
        "the σ demotion must reach `lp_declines_to_nlp`; if this reads \
         `cvx-qp` the reroute stopped firing for this population"
    );
    assert_eq!(format!("{:?}", v.solution.status), "SolveSucceeded");
    // `f* = −½ tᵀPt` with `t = (3, 0.5)`: the rotated spectrum puts `4.41e12`
    // of that on the stiff eigenvector, so `f* = −2.205e12` exactly by
    // construction. (The CLI's banner prints the *scaled* objective beside it;
    // the report carries the unscaled one.)
    let obj = v.solution.objective;
    assert!(
        (obj - -2.205e12).abs() / 2.205e12 < 1e-9,
        "objective {obj:e} against the constructed optimum -2.205e12"
    );
}

/// Pinned to the convex arm there is no reroute available, and the honest
/// status is itself the deliverable — including the AMPL `solve_result_num`,
/// which a driver reads to decide whether the solve succeeded.
///
/// | change | effect |
/// |---|---|
/// | drop the demotion in `solve_qp_ipm` | this reads `SolveSucceeded` / `0`, and the test above reads `cvx-qp` |
#[test]
fn pinned_to_the_convex_arm_the_status_is_the_deliverable() {
    let v = run(&["solver_selection=qp-ipm"]);
    assert_eq!(v.solution.engine, "cvx-qp");
    assert_eq!(
        format!("{:?}", v.solution.status),
        "SolvedToAcceptableLevel",
        "the cascade could not certify this point; the CLI must say so"
    );
    assert_eq!(
        v.solution.solve_result_num, 1,
        "AMPL solve_result_num 1 (acceptable), not 0 (optimal)"
    );
}

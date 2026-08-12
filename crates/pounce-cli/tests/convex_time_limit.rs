use std::path::PathBuf;
use std::process::Command;

use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_solve_report::SolveReport;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn assert_timeout(name: &str, selection: &str) {
    let json = std::env::temp_dir().join(format!(
        "pounce-convex-time-limit-{}-{}-{}.json",
        std::process::id(),
        name.replace('.', "-"),
        selection
    ));
    let _ = std::fs::remove_file(&json);
    let out = Command::new(pounce_exe())
        .arg(fixture(name))
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .arg(format!("solver_selection={selection}"))
        .arg("max_wall_time=1e-12")
        .output()
        .expect("spawn pounce");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stdout.contains("Maximum wallclock time exceeded."),
        "selection={selection}; stdout=\n{stdout}\nstderr=\n{stderr}"
    );
    assert!(
        !stderr.contains("being re-solved on the general NLP"),
        "a timed-out convex solve must not fall back; stderr=\n{stderr}"
    );
    let report: SolveReport =
        serde_json::from_str(&std::fs::read_to_string(&json).expect("read JSON report"))
            .expect("parse JSON report");
    assert_eq!(
        report.solution.status,
        ApplicationReturnStatus::MaximumWallTimeExceeded
    );
    assert_eq!(report.solution.solve_result_num, 400);
    let _ = std::fs::remove_file(json);
}

#[test]
fn auto_qp_timeout_is_final() {
    assert_timeout("convex_qp.nl", "auto");
}

#[test]
fn active_set_qp_timeout_is_final() {
    assert_timeout("convex_qp.nl", "qp-active-set");
}

#[test]
fn auto_socp_timeout_is_final() {
    assert_timeout("qcqp_ball.nl", "auto");
}

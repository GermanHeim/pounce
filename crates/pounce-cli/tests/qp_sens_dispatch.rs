//! Regression test for issue #196: sIPOPT sensitivity suffixes must not be
//! silently dropped when a problem classifies as a convex QP.
//!
//! Fixture `convex_qp_sens.nl` is the issue's reproduction — a *pure convex QP*
//!   min (x - p)^2 + y^2   s.t.   p == 1.0
//! carrying the three sIPOPT suffixes (`sens_state_1`, `sens_state_value_1`,
//! `sens_init_constr`) that request a post-optimal parametric sensitivity step,
//! with the perturbed parameter value p -> 1.5. Because it is a convex QP,
//! `solver_selection=auto` classifies it as such; the specialized pounce-convex
//! fast path has no sensitivity machinery, so pre-fix it returned no
//! `sens_sol_state_1` suffix at all (the silent drop the issue reports).
//!
//! The analytical sensitivity is dx*/dp = 1, so the perturbed primal has
//! x -> 1.5 (and p -> 1.5). Generated with pyomo 6.10 (quadratic objective +
//! linear equality + INT/FLOAT sIPOPT export suffixes); see the issue thread.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("convex_qp_sens.nl");
    p
}

/// Copy the fixture next to a fresh temp path so each test writes its own
/// sibling `.sol` (the AMPL convention) without racing the others.
fn staged_nl(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("pounce_issue196_{tag}"));
    std::fs::create_dir_all(&dir).expect("mkdir temp");
    let dst = dir.join("convex_qp_sens.nl");
    std::fs::copy(fixture(), &dst).expect("copy fixture");
    // Remove any stale .sol from a previous run.
    let _ = std::fs::remove_file(dir.join("convex_qp_sens.sol"));
    dst
}

/// Parse the `sens_sol_state_1` real-var suffix block out of a `.sol` file.
/// Returns index -> value for the listed entries, or None if the suffix is
/// absent.
fn parse_sens_sol_state_1(sol: &str) -> Option<HashMap<usize, f64>> {
    let mut lines = sol.lines();
    while let Some(line) = lines.next() {
        if let Some(rest) = line.strip_prefix("suffix ") {
            // AMPL `.sol` suffix header:
            //   "<kind> <nvalues> <namelen> <tablen> <tabline>"
            // then the suffix name, any table lines, then nvalues "<idx> <val>".
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if parts.len() < 5 {
                continue;
            }
            let count: usize = parts[1].parse().ok()?;
            let tabline: usize = parts[4].parse().ok()?;
            let name = lines.next()?.trim().to_string();
            if name != "sens_sol_state_1" {
                // Skip this suffix's table + value lines and keep scanning.
                for _ in 0..(tabline + count) {
                    lines.next();
                }
                continue;
            }
            for _ in 0..tabline {
                lines.next();
            }
            let mut out = HashMap::new();
            for _ in 0..count {
                let l = lines.next()?;
                let mut it = l.split_whitespace();
                let idx: usize = it.next()?.parse().ok()?;
                let val: f64 = it.next()?.parse().ok()?;
                out.insert(idx, val);
            }
            return Some(out);
        }
    }
    None
}

/// `auto` must honor the sensitivity request by routing to the NLP path and
/// writing `sens_sol_state_1` with the correct perturbed primal (x -> 1.5).
#[test]
fn auto_routes_sens_qp_to_nlp_and_writes_sens_suffix() {
    let nl = staged_nl("auto");
    let out = Command::new(pounce_exe())
        .arg(&nl)
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(0), "solve should succeed");

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("routing to the general NLP"),
        "auto should announce the reroute to NLP; stderr=\n{stderr}"
    );

    let sol_path = nl.with_extension("sol");
    let sol = std::fs::read_to_string(&sol_path).expect("read .sol");
    let sens = parse_sens_sol_state_1(&sol)
        .expect("sens_sol_state_1 must be present after auto reroute (issue #196)");
    let x = *sens.get(&0).expect("perturbed x (index 0)");
    assert!(
        (x - 1.5).abs() < 1e-6,
        "dx*/dp = 1 so p 1.0 -> 1.5 gives x* -> 1.5; got {x}"
    );
}

/// An *explicit* convex force must be respected, but the dropped sensitivity
/// request must be surfaced as a warning (not silent), and no
/// `sens_sol_state_1` is written.
#[test]
fn explicit_qp_ipm_warns_and_skips_sens() {
    let nl = staged_nl("qp_ipm");
    let out = Command::new(pounce_exe())
        .arg(&nl)
        .arg("solver_selection=qp-ipm")
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(0), "solve should succeed");

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("warning:") && stderr.contains("will be skipped"),
        "explicit convex force must warn that sensitivity is skipped; stderr=\n{stderr}"
    );

    let sol = std::fs::read_to_string(nl.with_extension("sol")).expect("read .sol");
    assert!(
        parse_sens_sol_state_1(&sol).is_none(),
        "forced convex path does not compute sensitivity, so no sens_sol_state_1"
    );
}

/// Control / no-regression: the general NLP path still writes the suffix.
#[test]
fn nlp_path_writes_sens_suffix() {
    let nl = staged_nl("nlp");
    let out = Command::new(pounce_exe())
        .arg(&nl)
        .arg("solver_selection=nlp")
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(0), "solve should succeed");

    let sol = std::fs::read_to_string(nl.with_extension("sol")).expect("read .sol");
    let sens = parse_sens_sol_state_1(&sol).expect("sens_sol_state_1 present on NLP path");
    let x = *sens.get(&0).expect("perturbed x (index 0)");
    assert!((x - 1.5).abs() < 1e-6, "expected x* -> 1.5; got {x}");
}

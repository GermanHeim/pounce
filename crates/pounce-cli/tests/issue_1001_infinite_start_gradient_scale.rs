//! gh#1001 item 4: an infinite objective gradient at the start must not
//! freeze gradient-based scaling at its floor.
//!
//! `gibbs_xlogx_origin.nl` is the Gibbs energy of an ideal four-species
//! mixture, `min sum_i x_i (ln x_i + c_i)`, `c = (0, 1, 2, 3)`, on the simplex
//! `sum_i x_i = 1`, `0 <= x_i <= 10`, started at `x0 = 0` (optbook wexi32-1).
//! The closed form is `G* = -ln sum_i exp(-c_i) = -0.44018969856119533`.
//!
//! Scaling samples `grad f` at the raw `x0`, before the bound push, where
//! `d/dx (x ln x) = ln x + 1 = -inf`. `100 / inf` floored at
//! `nlp_scaling_min_value` froze the objective factor at `1e-8`; the first
//! solve then ran 16 iterations to an unscaled KKT error of `9.1e-6`
//! (`|G - G*| = 1.3e-9`), and the gh#983 audit had to re-solve (8 more) to
//! recover. A non-finite start gradient carries no scale, so it now takes
//! upstream's failed-evaluation branch (`df = 1`) -- what Ipopt under ASL
//! does on this file, which rejects `log(0)`: 10 iterations, one solve.

use std::path::PathBuf;
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

#[test]
fn an_infinite_start_gradient_leaves_the_objective_unscaled() {
    let json =
        std::env::temp_dir().join(format!("pounce_issue1001_i4_{}.json", std::process::id()));
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(fixture("gibbs_xlogx_origin.nl"))
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .arg("solver_selection=nlp")
        .output()
        .expect("spawn pounce");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let text = std::fs::read_to_string(&json).expect("read json report");
    let _ = std::fs::remove_file(&json);
    let r: serde_json::Value = serde_json::from_str(&text).expect("parse json report");

    assert_eq!(r["solution"]["status"], "SolveSucceeded", "{stdout}");
    let g_star = -(0..4).map(|i| (-(i as f64)).exp()).sum::<f64>().ln();
    let f = r["solution"]["objective"].as_f64().unwrap();
    assert!(
        (f - g_star).abs() < 1e-12,
        "|G - G*| = {:.2e}",
        (f - g_star).abs()
    );
    let factor = r["statistics"]["objective_scaling"]["factor"]
        .as_f64()
        .unwrap();
    assert_eq!(factor, 1.0, "objective factor frozen at {factor:e}");
    // One solve: the audit has nothing to repair.
    assert!(!stdout.contains("gh#983 scale audit"), "{stdout}");
    assert_eq!(
        stdout.matches("Number of Iterations").count(),
        1,
        "{stdout}"
    );
}

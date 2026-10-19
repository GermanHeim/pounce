//! The active-set SQP arm reports the same bound multipliers as the NLP arm.
//!
//! Two defects, found auditing the multiplier boundaries next to gh#1033:
//!
//! * `finalize_via_sqp` (and the live-inspector iterate) split the SQP's
//!   signed per-variable `lambda_x` into per-variable `z_L` / `z_U` and fed
//!   them to `finalize_solution_z_*`, which read the *compressed* x_l / x_u
//!   spaces (one entry per finite bound). With an unbounded variable ahead
//!   of a bounded one, each multiplier landed on the wrong variable or was
//!   dropped: here the active `z_U` on `x1` read 0.
//! * The CLI backfilled the SQP solution's `x` / `lambda` for the `.sol`
//!   writer but not its bound multipliers, so the `.sol` had no
//!   `ipopt_zL_out` / `ipopt_zU_out` entries at all.
//!
//! Model: `min (x0-3)^2 + (x1-3)^2  s.t.  x0 + x1 <= 2.5,  x1 <= 1`, with
//! `x0` free, so `x1` is the only variable with a bound. At the optimum
//! (1.5, 1) the row dual is 3 and `z_U[x1] = 1` (`ipopt_zU_out = -1`).

use std::path::PathBuf;
use std::process::Command;

const NL: &str = "g3 1 1 0
 2 1 1 0 0
 0 1 0 0 0 0
 0 0
 0 2 0
 0 0 0 1
 0 0 0 0 0
 2 2
 4 1
 0 0 0 0 0
C0
n0
O0 0
o0
o5
v0
n2
o5
v1
n2
r
1 2.5
b
3
1 1
k1
1
J0 2
0 1
1 1
G0 2
0 -6
1 -6
";

fn zu_out_x1(tag: &str, opts: &[&str]) -> Option<f64> {
    let dir = std::env::temp_dir().join(format!("pounce_sqp_zmult_{tag}"));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let nl = dir.join("q.nl");
    std::fs::write(&nl, NL).expect("write nl");
    let sol = dir.join("q.sol");
    let st = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")))
        .arg(&nl)
        .arg(&sol)
        .arg("solver_selection=nlp")
        .args(opts)
        .status()
        .expect("spawn pounce");
    assert!(st.success(), "{tag}: solve failed");
    let text = std::fs::read_to_string(&sol).expect("read sol");
    let lines: Vec<&str> = text.lines().collect();
    let at = lines.iter().position(|l| l.trim() == "ipopt_zU_out")?;
    lines[at + 1..]
        .iter()
        .take_while(|l| !l.starts_with("suffix"))
        .find_map(|l| {
            let mut it = l.split_whitespace();
            (it.next() == Some("1")).then(|| it.next().unwrap().parse().unwrap())
        })
}

#[test]
fn sqp_reports_the_active_upper_bound_multiplier() {
    let nlp = zu_out_x1("nlp", &[]).expect("NLP arm writes ipopt_zU_out[x1]");
    assert!(
        (nlp + 1.0).abs() < 1e-6,
        "NLP arm: ipopt_zU_out[x1] = {nlp}"
    );
    let sqp = zu_out_x1("sqp", &["algorithm=active-set-sqp"])
        .expect("SQP arm must write ipopt_zU_out[x1] (was missing)");
    assert!(
        (sqp - nlp).abs() < 1e-6,
        "SQP arm ipopt_zU_out[x1] = {sqp}, NLP arm {nlp}"
    );
}

//! A primal-only warm start reads the same through presolve as without it.
//!
//! `OrigIpoptNlp` hands `get_starting_point` bound-multiplier buffers filled
//! with NaN, meaning "unseeded" (gh#622): the warm-start initializer gives
//! such an entry `bound_mult_init_val`, while a supplied zero is floored at
//! `warm_start_mult_bound_push`. `PresolveTnlp` and `LinearEqElimTnlp`
//! each allocated their own zero-filled buffers and copied them back over the
//! marker, so with presolve on, a start that supplied no multipliers read as
//! one supplying `z = 0` (1e-3 here instead of 1.0). Found auditing the
//! multiplier boundaries next to gh#1033.
//!
//! Model: `min (x0-3)^2 + (x1-1)^2  s.t.  x0 - x1 = 0,  0 <= x1 <= 10`, so
//! the linear-equality pass aggregates one column and keeps the bounded one.

use std::path::PathBuf;
use std::process::Command;

const NL: &str = "g3 1 1 0
 2 1 1 0 1
 0 1
 0 0
 0 2 0
 0 0 0 1
 0 0 0 0 0
 2 2
 0 0
 0 0 0 0 0
C0
n0
O0 0
o0
o5
o0
v0
n-3
n2
o5
o0
v1
n-1
n2
x2
0 1.0
1 1.0
r
4 0
b
3
0 0 10
k1
1
J0 2
0 1
1 -1
";

/// The `(ipopt_zL_out, ipopt_zU_out)` entries of variable 1 after zero
/// iterations of a primal-only warm start.
fn z_x1(tag: &str, presolve: &[&str]) -> (f64, f64) {
    let dir = std::env::temp_dir().join(format!("pounce_presolve_unseeded_{tag}"));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let nl = dir.join("e.nl");
    std::fs::write(&nl, NL).expect("write nl");
    let st = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")))
        .arg(&nl)
        .arg("-AMPL")
        .args([
            "solver_selection=nlp",
            "warm_start_init_point=yes",
            "warm_start_recentering=none",
            "max_iter=0",
        ])
        .args(presolve)
        .status()
        .expect("spawn pounce");
    assert!(st.code().is_some(), "{tag}: pounce did not exit");
    let text = std::fs::read_to_string(dir.join("e.sol")).expect("read sol");
    let lines: Vec<&str> = text.lines().collect();
    let entry = |name: &str| -> f64 {
        let at = lines
            .iter()
            .position(|l| l.trim() == name)
            .unwrap_or_else(|| panic!("{tag}: no {name} block"));
        lines[at + 1..]
            .iter()
            .take_while(|l| !l.starts_with("suffix"))
            .find_map(|l| {
                let mut it = l.split_whitespace();
                (it.next() == Some("1")).then(|| it.next().unwrap().parse().unwrap())
            })
            .unwrap_or_else(|| panic!("{tag}: no variable-1 entry in {name}"))
    };
    (entry("ipopt_zL_out"), entry("ipopt_zU_out"))
}

#[test]
fn presolve_keeps_the_unseeded_marker() {
    let base = z_x1("off", &["presolve=no"]);
    assert_eq!(base, (1.0, -1.0), "bound_mult_init_val, as unseeded");
    for (tag, opts) in [
        ("presolve", &["presolve=yes"][..]),
        (
            "linear_eq",
            &["presolve=yes", "presolve_linear_eq_reduction=yes"][..],
        ),
    ] {
        let got = z_x1(tag, opts);
        assert_eq!(got, base, "{tag}: presolve changed an unseeded start");
    }
}

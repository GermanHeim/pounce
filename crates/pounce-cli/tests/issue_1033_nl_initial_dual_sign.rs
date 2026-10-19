//! Regression test for gh#1033: the `.nl` `d` segment's initial duals are in
//! AMPL's sign (`d obj / d b`, the sign a `.sol` dual block carries), so the
//! reader must negate them into pounce's `L = f + lambda' g` multiplier just
//! as `sol_writer` negates on the way out. Before the fix a warm start with
//! `max_iter=0` handed a `d` entry of 5 back as -5; Ipopt's AMPL driver
//! gives back 5.
//!
//! The model is `min (x - 1)^2  s.t.  exp(x) >= 8,  0 <= x <= 10`, started
//! at x = 3. A full solve's AMPL dual is `2 (x* - 1) / exp(x*) ≈ 0.27 > 0`,
//! which pins the sign convention of the `.sol` side independently.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

/// `sense` is the `O` segment's kind: `0` minimize, `1` maximize. The
/// maximize body is the negated objective, so both senses describe the
/// same optimum and the same `.sol` dual.
fn nl_text(dual: &str, sense: u8) -> String {
    nl_text_with(dual, sense, "")
}

/// As [`nl_text`], with `extra` (e.g. `S` segments) appended.
fn nl_text_with(dual: &str, sense: u8, extra: &str) -> String {
    let obj = if sense == 0 {
        "o5\no0\nv0\nn-1\nn2"
    } else {
        "o16\no5\no0\nv0\nn-1\nn2"
    };
    format!(
        "g3 1 1 0
 1 1 1 0 0
 1 1
 0 0
 1 1 1
 0 0 0 1
 0 0 0 0 0
 1 1
 0 0
 0 0 0 0 0
C0
o44
v0
O0 {sense}
{obj}
d1
0 {dual}
x1
0 3.0
r
2 8
b
0 0 10
k0
J0 1
0 0
G0 1
0 0
{extra}"
    )
}

/// Solve with `-AMPL` and return the single constraint dual from the `.sol`.
fn sol_dual(tag: &str, dual: &str, sense: u8, opts: &[&str]) -> f64 {
    let text = solve_sol(tag, &nl_text(dual, sense), opts);
    // Layout: message, blank, "Options", option count k, k option lines,
    // then four counts (m, m, n, n), then the m duals.
    let lines: Vec<&str> = text.lines().collect();
    let o = lines
        .iter()
        .position(|l| l.trim() == "Options")
        .expect("Options line");
    let k: usize = lines[o + 1].trim().parse().expect("option count");
    let counts = o + 2 + k;
    let m: usize = lines[counts].trim().parse().expect("m");
    assert_eq!(m, 1, "one constraint dual expected");
    lines[counts + 4].trim().parse().expect("dual value")
}

/// The value of variable 0 in the `.sol` suffix block named `name`.
fn sol_suffix_x0(text: &str, name: &str) -> f64 {
    let lines: Vec<&str> = text.lines().collect();
    let at = lines
        .iter()
        .position(|l| l.trim() == name)
        .unwrap_or_else(|| panic!("no {name} block in .sol:\n{text}"));
    let mut it = lines[at + 1].split_whitespace();
    assert_eq!(it.next(), Some("0"), "{name}: first entry is variable 0");
    it.next().expect("value").parse().expect("f64")
}

/// Solve `nl` with `-AMPL` and return the `.sol` text.
fn solve_sol(tag: &str, nl_body: &str, opts: &[&str]) -> String {
    let dir = std::env::temp_dir().join(format!("pounce_issue_1033_{tag}"));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let nl = dir.join("d.nl");
    std::fs::write(&nl, nl_body).expect("write nl");
    let sol = dir.join("d.sol");
    let _ = std::fs::remove_file(&sol);
    let out = Command::new(pounce_exe())
        .arg(&nl)
        .arg("-AMPL")
        .args(opts)
        .output()
        .expect("spawn pounce");
    let text = std::fs::read_to_string(&sol).unwrap_or_else(|e| {
        panic!(
            "no .sol written ({e}); stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        )
    });
    text
}

#[test]
fn warm_start_round_trips_the_nl_initial_dual() {
    // Maximize too: the `.sol` dual of a maximize model is `-sigma*lambda`,
    // so the reader's inverse must carry the objective sense as well.
    for (tag, d, sense, want) in [
        ("pos", "5.0", 0, 5.0),
        ("neg", "-5.0", 0, -5.0),
        ("max_pos", "5.0", 1, 5.0),
        ("max_neg", "-5.0", 1, -5.0),
    ] {
        let got = sol_dual(tag, d, sense, &["warm_start_init_point=yes", "max_iter=0"]);
        assert!(
            (got - want).abs() < 1e-12,
            "d entry {d}: .sol dual after zero iterations is {got}, expected {want} \
             (a .nl initial dual and a .sol dual must share one sign)"
        );
    }
}

#[test]
fn full_solve_dual_is_in_ampl_sign() {
    // Independent anchor on the `.sol` side: d obj / d b > 0 here.
    // Minimize (x-1)^2: d obj / d b = 2 (x* - 1) / exp(x*) > 0. Maximize
    // -(x-1)^2: the same optimum, d obj / d b negated.
    let xs = 8f64.ln();
    let want = 2.0 * (xs - 1.0) / 8.0;
    for (tag, sense, want) in [("full", 0, want), ("full_max", 1, -want)] {
        let got = sol_dual(tag, "0.0", sense, &[]);
        assert!(
            (got - want).abs() < 1e-6,
            "{tag}: full-solve AMPL dual {got}, expected {want}"
        );
    }
}

#[test]
fn warm_start_round_trips_the_bound_multiplier_suffixes() {
    // Ipopt's AMPL driver seeds z from `ipopt_zL_in` / `ipopt_zU_in`
    // (`z_L = sigma*zL_in`, `z_U = -sigma*zU_in`), the inverse of the
    // `ipopt_z*_out` blocks; the reader used to ignore them. After zero
    // iterations the `.sol` must hand them back as written, in both senses.
    // `warm_start_recentering=none` keeps gh#606's residual recentering from
    // re-deriving the bound multipliers, so the round trip is the identity.
    //
    // The values are the `.sol` signs of a genuine multiplier for each sense:
    // `ipopt_zL_out = sigma*z_L` is >= 0 on a minimize model and <= 0 on a
    // maximize one (and `ipopt_zU_out` the reverse), so the maximize case
    // carries them negated — the engine clamps a wrong-signed z.
    for (tag, sense, zl_in, zu_in) in [("z_min", 0, 7.0, -9.0), ("z_max", 1, -7.0, 9.0)] {
        let sfx = format!("S4 1 ipopt_zL_in\n0 {zl_in}\nS4 1 ipopt_zU_in\n0 {zu_in}\n");
        let text = solve_sol(
            tag,
            &nl_text_with("0.0", sense, &sfx),
            &[
                "warm_start_init_point=yes",
                "warm_start_recentering=none",
                "max_iter=0",
            ],
        );
        let zl = sol_suffix_x0(&text, "ipopt_zL_out");
        let zu = sol_suffix_x0(&text, "ipopt_zU_out");
        assert!(
            (zl - zl_in).abs() < 1e-12,
            "{tag}: ipopt_zL_out {zl}, expected {zl_in}"
        );
        assert!(
            (zu - zu_in).abs() < 1e-12,
            "{tag}: ipopt_zU_out {zu}, expected {zu_in}"
        );
    }
}

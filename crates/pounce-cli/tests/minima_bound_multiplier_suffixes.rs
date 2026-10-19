//! `pounce --minima` writes the `ipopt_zL_out` / `ipopt_zU_out` suffix blocks
//! a plain solve's `.sol` carries (gh #296), with the same signs.
//!
//! Before this, `write_sol_files` passed `suffixes: &[]`, so a Pyomo
//! `model.ipopt_zL_out` / AMPL `.rc` read after a `--minima` run came back
//! empty while the same read after a plain solve of the same file did not.
//!
//! The model is `(x0 + 1)² + (x1 − 3)²` on `[0, 2]²`, whose minimizer
//! `(0, 2)` has `x0` on its **lower** bound and `x1` on its **upper** one,
//! both with a nonzero multiplier: `∂f/∂x0 = 2`, `∂f/∂x1 = −2`. So a plain
//! minimize solve writes `ipopt_zL_out[0] = 2` and `ipopt_zU_out[1] = −2`, and
//! the maximize twin (`max −f`, same point) writes both negated (gh#959). The
//! expected values are those analytic numbers, not just agreement between two
//! POUNCE surfaces, so a uniform sign flip on both paths fails here too.
//!
//! Two shapes, because the dual recovery branches on `m`: with a constraint
//! (an inactive `x0 + x1 ≤ 3`) it always re-solves at the minimum; without one
//! it re-solves only when the minimum sits on a bound (`recover_duals`).

use std::path::{Path, PathBuf};
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

/// The `.nl` text. `maximize` writes `max −f` (`O0 1` over `o16`, unary
/// minus); `constrained` adds the inactive row `x0 + x1 ≤ 3`.
fn model(maximize: bool, constrained: bool) -> String {
    let m = usize::from(constrained);
    let mut s = String::new();
    s.push_str("g3 1 1 0\t# problem minima_bound_active\n");
    s.push_str(&format!(" 2 {m} 1 0 0\n"));
    s.push_str(" 0 1 0 0 0 0\n 0 0\n 0 2 0\n 0 0 0 1\n 0 0 0 0 0\n");
    s.push_str(&format!(" {} 2\n", 2 * m));
    s.push_str(" 0 0\n 0 0 0 0 0\n");
    if constrained {
        s.push_str("C0\nn0\n");
    }
    s.push_str(if maximize { "O0 1\no16\n" } else { "O0 0\n" });
    s.push_str("o0\no5\no0\nv0\nn1\nn2\no5\no0\nv1\nn-3\nn2\n");
    s.push_str("x2\n0 1\n1 1\n");
    if constrained {
        s.push_str("r\n1 3\n");
    }
    s.push_str("b\n0 0 2\n0 0 2\n");
    s.push_str(if constrained { "k1\n1\n" } else { "k1\n0\n" });
    if constrained {
        s.push_str("J0 2\n0 1\n1 1\n");
    }
    s.push_str("G0 2\n0 0\n1 0\n");
    s
}

fn run(nl: &Path, sol: &Path, extra: &[&str]) {
    let out = Command::new(pounce_exe())
        .arg(nl)
        .arg("--sol-output")
        .arg(sol)
        .args(extra)
        .output()
        .expect("spawn pounce");
    assert!(
        out.status.success(),
        "pounce {extra:?} failed; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Read a named `.sol` suffix block as a dense length-`n` vector (zero
/// entries are trimmed on write, so an absent index is a zero multiplier).
/// `None` if the block is absent.
fn suffix(text: &str, name: &str, n: usize) -> Option<Vec<f64>> {
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        if !line.trim_start().starts_with("suffix ") {
            continue;
        }
        let count: usize = line
            .split_whitespace()
            .nth(2)
            .and_then(|v| v.parse().ok())
            .expect("suffix header carries a count");
        let this = lines.next().expect("suffix header is followed by a name");
        let mut dense = vec![0.0; n];
        for _ in 0..count {
            let row = lines.next().expect("truncated suffix block");
            let mut it = row.split_whitespace();
            let i: usize = it.next().unwrap().parse().unwrap();
            dense[i] = it.next().unwrap().parse().unwrap();
        }
        if this.trim() == name {
            return Some(dense);
        }
    }
    None
}

fn check(maximize: bool, constrained: bool) {
    let tag = format!(
        "{}_{}",
        if maximize { "max" } else { "min" },
        if constrained { "m1" } else { "m0" }
    );
    let dir = std::env::temp_dir().join(format!(
        "pounce_minima_zsuffix_{}_{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let nl = dir.join("m.nl");
    std::fs::write(&nl, model(maximize, constrained)).expect("write .nl");
    let plain = dir.join("plain.sol");
    let minima = dir.join("minima.sol");

    run(&nl, &plain, &["solver_selection=nlp"]);
    run(
        &nl,
        &minima,
        &[
            "--minima",
            "multistart",
            "--n-minima",
            "2",
            "--max-solves",
            "6",
            "--seed",
            "0",
        ],
    );

    let sigma = if maximize { -1.0 } else { 1.0 };
    let want_zl = [2.0 * sigma, 0.0];
    let want_zu = [0.0, -2.0 * sigma];
    let plain_text = std::fs::read_to_string(&plain).expect("read plain .sol");
    let minima_text = std::fs::read_to_string(&minima).expect("read --minima .sol");
    for (name, want) in [("ipopt_zL_out", want_zl), ("ipopt_zU_out", want_zu)] {
        let p = suffix(&plain_text, name, 2)
            .unwrap_or_else(|| panic!("{tag}: plain .sol has no {name}:\n{plain_text}"));
        let q = suffix(&minima_text, name, 2)
            .unwrap_or_else(|| panic!("{tag}: --minima .sol has no {name}:\n{minima_text}"));
        for i in 0..2 {
            assert!(
                (p[i] - want[i]).abs() < 1e-5,
                "{tag}: plain {name}[{i}] = {} , want {}",
                p[i],
                want[i]
            );
            assert!(
                (q[i] - p[i]).abs() < 1e-5,
                "{tag}: --minima {name}[{i}] = {} but a plain solve writes {}",
                q[i],
                p[i]
            );
        }
    }
}

#[test]
fn minima_sol_bound_multipliers_match_a_plain_solve_minimize() {
    check(false, true);
}

#[test]
fn minima_sol_bound_multipliers_match_a_plain_solve_maximize() {
    check(true, true);
}

#[test]
fn minima_sol_bound_multipliers_without_constraints_minimize() {
    check(false, false);
}

#[test]
fn minima_sol_bound_multipliers_without_constraints_maximize() {
    check(true, false);
}

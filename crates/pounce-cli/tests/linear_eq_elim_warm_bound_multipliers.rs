//! A bound multiplier `presolve_linear_eq_reduction` moved off an eliminated
//! column round-trips through a warm start (gh#493, inbound half).
//!
//! Outbound, `attribute_bound_multiplier` reports the multiplier of a bound
//! the plan *transferred* onto a survivor against the eliminated column that
//! declared it, scaled by `1/|α|` and side-flipped when `α < 0`. Inbound,
//! `get_starting_point` used to read only the survivor's own entry, so feeding
//! that `.sol` back in seeded the reduced bound as unsupplied (NaN, so
//! `bound_mult_init_val = 1`) and the multiplier came back as 1/|α| instead of
//! 10.
//!
//! Model: `min x0^2 + x1^2  s.t.  x0 + c*x1 = 0,  x0 >= 4`, `c = ∓2`. The
//! comparable coefficients eliminate `x0 = -x1/c`, so `x0 >= 4` becomes the
//! survivor's lower bound (`c < 0`) or upper bound (`c > 0`). At `x0 = 4`,
//! `z_L(x0) = 10`. `presolve_bound_tightening=no`, because otherwise Phase 1
//! derives the same bound on `x1` from the row first and the transfer ties
//! (the tie keeps the multiplier on the survivor, a separate story: M24).

use std::path::PathBuf;
use std::process::Command;

fn nl(c: f64, x: [f64; 2], extra: &str) -> String {
    format!(
        "g3 1 1 0
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
v0
n2
o5
v1
n2
x2
0 {}
1 {}
r
4 0
b
2 4
3
k1
1
J0 2
0 1
1 {c}
{extra}",
        x[0], x[1]
    )
}

struct Sol {
    dual: f64,
    x: [f64; 2],
    zl: Vec<(usize, f64)>,
    zu: Vec<(usize, f64)>,
}

fn solve(tag: &str, body: &str, opts: &[&str]) -> Sol {
    let dir = std::env::temp_dir().join(format!("pounce_elim_warm_z_{tag}"));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let nl = dir.join("e.nl");
    std::fs::write(&nl, body).expect("write nl");
    let sol = dir.join("e.sol");
    let _ = std::fs::remove_file(&sol);
    let out = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")))
        .arg(&nl)
        .arg("-AMPL")
        .args(["solver_selection=nlp", "presolve=yes"])
        .args([
            "presolve_linear_eq_reduction=yes",
            "presolve_bound_tightening=no",
        ])
        .args(opts)
        .output()
        .expect("spawn pounce");
    let text = std::fs::read_to_string(&sol).unwrap_or_else(|e| {
        panic!(
            "{tag}: no .sol ({e}); stderr {}",
            String::from_utf8_lossy(&out.stderr)
        )
    });
    let lines: Vec<&str> = text.lines().collect();
    let o = lines.iter().position(|l| l.trim() == "Options").unwrap();
    let k: usize = lines[o + 1].trim().parse().unwrap();
    let c = o + 2 + k;
    let num = |i: usize| -> f64 { lines[i].trim().parse().unwrap() };
    let block = |name: &str| -> Vec<(usize, f64)> {
        let Some(at) = lines.iter().position(|l| l.trim() == name) else {
            return Vec::new();
        };
        lines[at + 1..]
            .iter()
            .take_while(|l| !l.starts_with("suffix"))
            .map(|l| {
                let mut it = l.split_whitespace();
                (
                    it.next().unwrap().parse().unwrap(),
                    it.next().unwrap().parse().unwrap(),
                )
            })
            .collect()
    };
    Sol {
        dual: num(c + 4),
        x: [num(c + 5), num(c + 6)],
        zl: block("ipopt_zL_out"),
        zu: block("ipopt_zU_out"),
    }
}

fn entry(v: &[(usize, f64)], j: usize) -> f64 {
    v.iter().find(|(i, _)| *i == j).map_or(0.0, |&(_, z)| z)
}

#[test]
fn a_transferred_bound_multiplier_round_trips_through_a_warm_start() {
    for (tag, c) in [("neg", -2.0), ("pos", 2.0)] {
        let full = solve(&format!("{tag}_full"), &nl(c, [4.0, 2.0], ""), &[]);
        assert!(
            (entry(&full.zl, 0) - 10.0).abs() < 1e-6,
            "{tag}: the full solve reports z_L(x0) = 10, got {:?}",
            full.zl
        );
        // Feed the `.sol` straight back: x, the AMPL-sign dual, and the
        // bound multipliers as the `ipopt_z*_in` suffixes.
        let mut sfx = format!("d1\n0 {:?}\n", full.dual);
        for (name, blk) in [("ipopt_zL_in", &full.zl), ("ipopt_zU_in", &full.zu)] {
            if !blk.is_empty() {
                sfx += &format!("S4 {} {name}\n", blk.len());
                for (j, z) in blk.iter() {
                    sfx += &format!("{j} {z:?}\n");
                }
            }
        }
        let warm = solve(
            &format!("{tag}_warm"),
            &nl(c, full.x, &sfx),
            &[
                "warm_start_init_point=yes",
                "warm_start_recentering=none",
                "max_iter=0",
            ],
        );
        assert!(
            (entry(&warm.zl, 0) - entry(&full.zl, 0)).abs() < 1e-9,
            "{tag}: z_L(x0) {:?} after zero warm iterations, supplied {:?}",
            warm.zl,
            full.zl
        );
        assert_eq!(
            entry(&warm.zl, 1),
            0.0,
            "{tag}: nothing on x1: {:?}",
            warm.zl
        );
        assert_eq!(
            entry(&warm.zu, 1),
            0.0,
            "{tag}: nothing on x1: {:?}",
            warm.zu
        );
    }
}

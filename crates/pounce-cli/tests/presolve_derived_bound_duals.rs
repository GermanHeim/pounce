//! Presolve reports the original problem's multipliers when the solve leans
//! on a bound presolve *derived* from rows (M24).
//!
//! Phase 1 / FBBT derive variable bounds from rows — `x0 = 2·x1, x0 ≥ 4`
//! implies `x1 ≥ 2` — and when such a bound is active the reduced solve may
//! put the multiplier on it, against a bound the model never declared.
//! Measured before the fix on that model with
//! `presolve_linear_eq_reduction=yes`: `.sol` dual `8`, `ipopt_zL_out(x1) =
//! 20`, against `presolve=no`'s `-2` and `z_L(x0) = 10` — duals that
//! `pounce verify --require-optimal` rejects (dual infeasibility `3.2e1`).
//! Plain `presolve=yes` split it three ways (`2.99`, `5.01`, `9.98`; `1.2e1`).
//!
//! `PresolveTnlp::finalize_solution` now zeroes a multiplier on a derived
//! side the original model does not have active and recovers the original
//! problem's multipliers on the active rows and declared bounds the residual
//! reaches (`dual_recovery`). Where those are unique this is the
//! `presolve=no` answer; every case below is unique except
//! `coincident_singleton`, which is the degenerate pair the rule leaves on
//! the declared bound, and is checked with `verify --require-optimal`.

use std::path::PathBuf;
use std::process::Command;

/// One `.nl` model: `min Σ_j (x_j − c_j)²` over linear rows.
struct Model {
    name: &'static str,
    centers: &'static [f64],
    /// `(r-section line, [(col, coeff)])` per row.
    rows: &'static [(&'static str, &'static [(usize, f64)])],
    /// `b`-section line per variable.
    bounds: &'static [&'static str],
}

fn nl(m: &Model, maximize: bool, x0: &[f64], extra: &str) -> String {
    let n = m.centers.len();
    let nr = m.rows.len();
    let neq = m.rows.iter().filter(|r| r.0.starts_with('4')).count();
    let nrange = m.rows.iter().filter(|r| r.0.starts_with('0')).count();
    let nnz: usize = m.rows.iter().map(|r| r.1.len()).sum();
    let (sense, neg) = if maximize { (1, "o16\n") } else { (0, "") };
    let mut s = format!(
        "g3 1 1 0\n {n} {nr} 1 {nrange} {neq}\n 0 1\n 0 0\n 0 {n} 0\n 0 0 0 1\n 0 0 0 0 0\n {nnz} {n}\n 0 0\n 0 0 0 0 0\n"
    );
    for i in 0..nr {
        s += &format!("C{i}\nn0\n");
    }
    s += &format!("O0 {sense}\n{neg}o54\n{n}\n");
    for (j, c) in m.centers.iter().enumerate() {
        s += &format!("o5\no0\nv{j}\nn{}\nn2\n", -c);
    }
    s += &format!("x{n}\n");
    for (j, v) in x0.iter().enumerate() {
        s += &format!("{j} {v:?}\n");
    }
    s += "r\n";
    for r in m.rows {
        s += r.0;
        s += "\n";
    }
    s += "b\n";
    for b in m.bounds {
        s += b;
        s += "\n";
    }
    s += &format!("k{}\n", n - 1);
    let mut cum = 0;
    for c in 0..n - 1 {
        cum += m
            .rows
            .iter()
            .flat_map(|r| r.1.iter())
            .filter(|e| e.0 == c)
            .count();
        s += &format!("{cum}\n");
    }
    for (i, r) in m.rows.iter().enumerate() {
        s += &format!("J{i} {}\n", r.1.len());
        for (j, a) in r.1 {
            s += &format!("{j} {a}\n");
        }
    }
    s += &format!("G0 {n}\n");
    for j in 0..n {
        s += &format!("{j} 0\n");
    }
    s + extra
}

/// `min x0² + x1²  s.t.  x0 − 2·x1 = 0,  x0 ≥ 4` — the λ = 8 case.
const ELIM_NEG: Model = Model {
    name: "elim_neg",
    centers: &[0.0, 0.0],
    rows: &[("4 0", &[(0, 1.0), (1, -2.0)])],
    bounds: &["2 4", "3"],
};
/// The same with `x0 + 2·x1 = 0`: the derived bound is `x1 ≤ −2`.
const ELIM_POS: Model = Model {
    name: "elim_pos",
    centers: &[0.0, 0.0],
    rows: &[("4 0", &[(0, 1.0), (1, 2.0)])],
    bounds: &["2 4", "3"],
};
/// `C0: x0 = 0.5`, `C1: x0 + x1 ≤ 2`, `min (x0+1)² + (x1−3)²`. Presolve
/// fixes x0 from C0, derives `x1 ≤ 1.5` from C1 and drops both rows; the
/// original multipliers are `λ1 = 3` (C1, `.sol` −3) and `λ0 = −6`.
const CHAIN: Model = Model {
    name: "chain",
    centers: &[-1.0, 3.0],
    rows: &[("4 0.5", &[(0, 1.0)]), ("1 2", &[(0, 1.0), (1, 1.0)])],
    bounds: &["3", "3"],
};
/// `C0: x0 + x1 ≥ 3`, `x0 ≤ 1`, `min (x0−2)² + x1²`: `x1 ≥ 2` is derived
/// from C0 with x0 at its declared upper bound. `x* = (1, 2)`, `λ = −4`,
/// `z_U(x0) = 6` — the chain rule through a variable used at its bound.
const VIA_BOUND: Model = Model {
    name: "via_bound",
    centers: &[2.0, 0.0],
    rows: &[("2 3", &[(0, 1.0), (1, 1.0)])],
    bounds: &["1 1", "3"],
};
/// `C0: x0 ≥ 0` with `x0 ≥ 0` declared too, `min (x0+1)² + (x1−1)²` plus a
/// row that keeps the model from collapsing (`C1: x0 + x1 ≤ 5`). The row
/// and the bound are a degenerate pair: any split of the multiplier 2 is a
/// valid KKT point. `presolve=yes` leaves it all on the declared bound.
const COINCIDENT: Model = Model {
    name: "coincident_singleton",
    centers: &[-1.0, 1.0],
    rows: &[("2 0", &[(0, 1.0)]), ("1 5", &[(0, 1.0), (1, 1.0)])],
    bounds: &["2 0", "3"],
};

#[derive(Debug, Clone)]
struct Sol {
    duals: Vec<f64>,
    x: Vec<f64>,
    zl: Vec<(usize, f64)>,
    zu: Vec<(usize, f64)>,
}

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("pounce_presolve_derived_duals_{tag}"));
    std::fs::create_dir_all(&d).expect("mkdir");
    d
}

fn solve(tag: &str, body: &str, opts: &[&str]) -> (Sol, PathBuf, PathBuf) {
    let d = dir(tag);
    let nl = d.join("m.nl");
    std::fs::write(&nl, body).expect("write nl");
    let sol = d.join("m.sol");
    let _ = std::fs::remove_file(&sol);
    let out = Command::new(bin())
        .arg(&nl)
        .arg("-AMPL")
        .args(opts)
        .output()
        .expect("spawn pounce");
    let text = std::fs::read_to_string(&sol).unwrap_or_else(|e| {
        panic!(
            "{tag}: no .sol ({e}); stdout {} stderr {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    });
    let lines: Vec<&str> = text.lines().collect();
    let o = lines.iter().position(|l| l.trim() == "Options").unwrap();
    let k: usize = lines[o + 1].trim().parse().unwrap();
    let c = o + 2 + k;
    let num = |i: usize| -> f64 { lines[i].trim().parse().unwrap() };
    let (m, n): (usize, usize) = (
        lines[c].trim().parse().unwrap(),
        lines[c + 2].trim().parse().unwrap(),
    );
    let block = |name: &str| -> Vec<(usize, f64)> {
        let Some(at) = lines.iter().position(|l| l.trim() == name) else {
            return Vec::new();
        };
        lines[at + 1..]
            .iter()
            .take_while(|l| !l.starts_with("suffix") && !l.trim().is_empty())
            .map(|l| {
                let mut it = l.split_whitespace();
                (
                    it.next().unwrap().parse().unwrap(),
                    it.next().unwrap().parse().unwrap(),
                )
            })
            .collect()
    };
    let s = Sol {
        duals: (0..m).map(|i| num(c + 4 + i)).collect(),
        x: (0..n).map(|i| num(c + 4 + m + i)).collect(),
        zl: block("ipopt_zL_out"),
        zu: block("ipopt_zU_out"),
    };
    (s, nl, sol)
}

/// `pounce verify --require-optimal`: stationarity with the bound
/// multipliers against the canonical model.
fn verified(nl: &PathBuf, sol: &PathBuf) -> (bool, String) {
    let out = Command::new(bin())
        .arg("verify")
        .arg(nl)
        .arg(sol)
        .arg("--require-optimal")
        .output()
        .expect("spawn verify");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}

fn entry(v: &[(usize, f64)], j: usize) -> f64 {
    v.iter().find(|(i, _)| *i == j).map_or(0.0, |&(_, z)| z)
}

fn same(a: &Sol, b: &Sol, tol: f64) -> bool {
    let close = |p: f64, q: f64| (p - q).abs() <= tol * (1.0 + q.abs());
    a.duals.len() == b.duals.len()
        && a.duals.iter().zip(&b.duals).all(|(p, q)| close(*p, *q))
        && a.x.iter().zip(&b.x).all(|(p, q)| close(*p, *q))
        && (0..a.x.len()).all(|j| {
            close(entry(&a.zl, j), entry(&b.zl, j)) && close(entry(&a.zu, j), entry(&b.zu, j))
        })
}

const ARMS: [(&str, &[&str]); 2] = [
    ("nlp", &["solver_selection=nlp"]),
    ("sqp", &["solver_selection=nlp", "algorithm=active-set-sqp"]),
];

const PRESOLVE: [(&str, &[&str]); 4] = [
    ("on", &["presolve=yes"]),
    ("on_nobt", &["presolve=yes", "presolve_bound_tightening=no"]),
    (
        "elim",
        &["presolve=yes", "presolve_linear_eq_reduction=yes"],
    ),
    (
        "elim_nobt",
        &[
            "presolve=yes",
            "presolve_linear_eq_reduction=yes",
            "presolve_bound_tightening=no",
        ],
    ),
];

/// Every presolve setting writes the `presolve=no` duals on every model
/// whose multipliers are unique, on both arms, in both senses — and every
/// `.sol` passes `verify --require-optimal`.
#[test]
fn derived_bound_duals_match_presolve_off() {
    for model in [&ELIM_NEG, &ELIM_POS, &CHAIN, &VIA_BOUND] {
        let x0 = vec![0.1; model.centers.len()];
        for maximize in [false, true] {
            let body = nl(model, maximize, &x0, "");
            let sense = if maximize { "max" } else { "min" };
            for (arm, arm_opts) in ARMS {
                let tag = format!("{}_{sense}_{arm}", model.name);
                let (off, nl_off, sol_off) = solve(
                    &format!("{tag}_off"),
                    &body,
                    &[arm_opts, &["presolve=no"]].concat(),
                );
                let (ok, rep) = verified(&nl_off, &sol_off);
                assert!(ok, "{tag}: presolve=no .sol fails verify\n{rep}");
                for (ps, ps_opts) in PRESOLVE {
                    let t = format!("{tag}_{ps}");
                    let (on, nl_on, sol_on) = solve(&t, &body, &[arm_opts, ps_opts].concat());
                    let (ok, rep) = verified(&nl_on, &sol_on);
                    assert!(
                        ok,
                        "{t}: .sol fails verify --require-optimal: {on:?}\n{rep}"
                    );
                    assert!(
                        same(&on, &off, 1e-6),
                        "{t}: presolve {on:?}\n   vs presolve=no {off:?}"
                    );
                }
            }
        }
    }
}

/// The pinned numbers for the case that motivated this: `.sol` dual `-2`
/// (`8` before), `z_L(x0) = 10` and nothing on x1 (`20` before).
#[test]
fn the_lambda_eight_case() {
    let body = nl(&ELIM_NEG, false, &[1.0, 1.0], "");
    let (s, _, _) = solve(
        "lambda8",
        &body,
        &[
            "solver_selection=nlp",
            "presolve=yes",
            "presolve_linear_eq_reduction=yes",
        ],
    );
    assert!((s.duals[0] + 2.0).abs() < 1e-6, "{s:?}");
    assert!((entry(&s.zl, 0) - 10.0).abs() < 1e-6, "{s:?}");
    assert_eq!(entry(&s.zl, 1), 0.0, "{s:?}");
}

/// The degenerate pair: a singleton row coincident with the model's own
/// bound. `presolve=no`'s interior point splits the multiplier between them;
/// presolve drops the row and the declared bound — the incumbent — keeps it
/// all, row dual 0. Both are valid KKT points of the original problem, and
/// the presolve one is pinned here so a change of rule is deliberate.
#[test]
fn a_coincident_singleton_row_leaves_its_multiplier_on_the_declared_bound() {
    for maximize in [false, true] {
        let body = nl(&COINCIDENT, maximize, &[0.1, 0.1], "");
        let sign = if maximize { -1.0 } else { 1.0 };
        for (arm, arm_opts) in ARMS {
            for (ps, ps_opts) in PRESOLVE {
                let t = format!("coincident_{maximize}_{arm}_{ps}");
                let (on, nl_on, sol_on) = solve(&t, &body, &[arm_opts, ps_opts].concat());
                let (ok, rep) = verified(&nl_on, &sol_on);
                assert!(ok, "{t}: {on:?}\n{rep}");
                assert!(on.duals[0].abs() < 1e-9, "{t}: row dual {on:?}");
                assert!(
                    (entry(&on.zl, 0) - sign * 2.0).abs() < 1e-6,
                    "{t}: z_L(x0) {on:?}"
                );
            }
            let t = format!("coincident_{maximize}_{arm}_off");
            let (off, nl_off, sol_off) = solve(&t, &body, &[arm_opts, &["presolve=no"]].concat());
            let (ok, rep) = verified(&nl_off, &sol_off);
            assert!(ok, "{t}: {off:?}\n{rep}");
            // Whatever the split, the two carry the same total.
            // Ipopt `∇f + λ − z_L = 0` at x0, and the `.sol` dual is `−λ`.
            let total = entry(&off.zl, 0) + off.duals[0];
            assert!((total - sign * 2.0).abs() < 1e-6, "{t}: {off:?}");
        }
    }
}

/// The `.sol` duals and `ipopt_zL_out` / `ipopt_zU_out` of `a` and `b` agree
/// over all `n` variables, whatever their `x`.
fn same_duals(a: &Sol, b: &Sol, n: usize, tol: f64) -> bool {
    let close = |p: f64, q: f64| (p - q).abs() <= tol * (1.0 + q.abs());
    a.duals.len() == b.duals.len()
        && a.duals.iter().zip(&b.duals).all(|(p, q)| close(*p, *q))
        && (0..n).all(|j| {
            close(entry(&a.zl, j), entry(&b.zl, j)) && close(entry(&a.zu, j), entry(&b.zu, j))
        })
}

/// The warm starts the round trip runs: `max_iter=0` under the default
/// warm-start push (the interior-point start moves `x` about `1e-3` off
/// every bound it was supplied on) and under a near-zero one that keeps it
/// there, and the same start left to iterate, where the solver moves the
/// duals and postsolve's own recovery has to agree with `presolve=no`.
const WARM_RUNS: [(&str, &[&str]); 3] = [
    ("default_push", &["max_iter=0"]),
    (
        "tiny_push",
        &[
            "max_iter=0",
            "warm_start_bound_push=1e-14",
            "warm_start_bound_frac=1e-14",
            "warm_start_slack_bound_push=1e-14",
            "warm_start_slack_bound_frac=1e-14",
        ],
    ),
    ("iterated", &[]),
];

/// A `.sol` written through presolve, fed back as a warm start, comes back
/// with exactly the duals `presolve=no` returns for the same warm start —
/// which, with zero iterations, are the supplied ones.
///
/// Before the fix the default push failed 4 of the 12 `max_iter=0` cases
/// (ELIM_NEG under `elim`: supplied `λ = -2`, returned `7e-16`; under
/// `elim_nobt`, `-1.992`; CHAIN under `on` and `elim`: supplied `[6, -3]`,
/// returned `[3, -0]` with `z_U(x1) = 3` on a bound the model does not
/// have): postsolve re-derived the multipliers the reduced problem does not
/// carry from stationarity at the pushed `x`, where `∇f` is not the one the
/// supplied duals balance. The 1e-14 push hid that by keeping `x` on its
/// bounds. Now each presolve layer returns the duals it was handed when the
/// solver hands back the ones it served (`warm_echo`).
#[test]
fn derived_bound_duals_round_trip_through_a_warm_start() {
    let mut failures = Vec::new();
    for model in [&ELIM_NEG, &CHAIN, &VIA_BOUND] {
        let n = model.centers.len();
        for (ps, ps_opts) in PRESOLVE {
            let opts = [&["solver_selection=nlp"][..], ps_opts].concat();
            let (full, _, _) = solve(
                &format!("rt_{}_{ps}_full", model.name),
                &nl(model, false, &vec![0.1; n], ""),
                &opts,
            );
            let mut extra = format!("d{}\n", full.duals.len());
            for (i, d) in full.duals.iter().enumerate() {
                extra += &format!("{i} {d:?}\n");
            }
            for (sfx, blk) in [("ipopt_zL_in", &full.zl), ("ipopt_zU_in", &full.zu)] {
                if !blk.is_empty() {
                    extra += &format!("S4 {} {sfx}\n", blk.len());
                    for (j, z) in blk.iter() {
                        extra += &format!("{j} {z:?}\n");
                    }
                }
            }
            let body = nl(model, false, &full.x, &extra);
            for (run, run_opts) in WARM_RUNS {
                let warm_opts = |presolve: &[&'static str]| -> Vec<&'static str> {
                    [
                        &["solver_selection=nlp"][..],
                        presolve,
                        &["warm_start_init_point=yes", "warm_start_recentering=none"],
                        run_opts,
                    ]
                    .concat()
                };
                let t = format!("rt_{}_{ps}_{run}", model.name);
                let (warm, _, _) = solve(&format!("{t}_warm"), &body, &warm_opts(ps_opts));
                let (reference, _, _) =
                    solve(&format!("{t}_ref"), &body, &warm_opts(&["presolve=no"]));
                // Duals only: the interior-point start moves x off the bound
                // by `warm_start_bound_push`, presolve or not.
                if !same_duals(&reference, &full, n, 1e-6) {
                    failures.push(format!(
                        "{t}: presolve=no warm {reference:?}\n   vs supplied {full:?}"
                    ));
                }
                if !same_duals(&warm, &reference, n, 1e-6) {
                    failures.push(format!(
                        "{t}: warm {warm:?}\n   vs presolve=no warm {reference:?}"
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

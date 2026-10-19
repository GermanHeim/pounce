//! The active-set SQP arm's `.sol` is in the model's units under
//! `nlp_scaling_method=user-scaling`, as the NLP arm's is.
//!
//! The SQP path bypasses the IPM's `on_converged` hook, so the CLI
//! backfills `x` and the bound multipliers from what `CountingTnlp`
//! captured at `finalize_solution`. `CountingTnlp` sits *inside* the gh#486
//! variable-scaling wrapper, so that payload has already had `x /= d`,
//! `z *= d` applied — and the CLI's own gh#486 undo then applied the factor
//! a second time. With `scaling_factor[x1] = 10` the `.sol` read `x1 = 0.1`
//! and `ipopt_zU_out = -10` where both arms should read `1` and `-1`.
//!
//! Model (the `sqp_bound_multipliers.rs` one, plus suffixes):
//! `min (x0-3)^2 + (x1-3)^2  s.t.  x0 + x1 <= 2.5,  x1 <= 1`, with
//! `scaling_factor[x1] = 10` on the bounded variable and
//! `scaling_factor[c0] = 5` on the active row, so a row-scaling slip in the
//! dual block would show too. Optimum (1.5, 1), row dual 3, `z_U[x1] = 1`.

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
S4 1 scaling_factor
1 10
S5 1 scaling_factor
0 5
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

/// The parts of a `.sol` this test compares.
#[derive(Debug)]
struct Sol {
    duals: Vec<f64>,
    x: Vec<f64>,
    z_l: Vec<(usize, f64)>,
    z_u: Vec<(usize, f64)>,
}

fn suffix(lines: &[&str], name: &str) -> Vec<(usize, f64)> {
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
}

fn solve(tag: &str, opts: &[&str]) -> Sol {
    let dir = std::env::temp_dir().join(format!("pounce_sqp_user_scaling_{tag}"));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let nl = dir.join("q.nl");
    std::fs::write(&nl, NL).expect("write nl");
    let sol = dir.join("q.sol");
    let st = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")))
        .arg(&nl)
        .arg(&sol)
        .arg("solver_selection=nlp")
        .arg("nlp_scaling_method=user-scaling")
        .args(opts)
        .output()
        .expect("spawn pounce");
    assert!(
        st.status.success(),
        "{tag}: solve failed\n{}",
        String::from_utf8_lossy(&st.stderr)
    );
    let text = std::fs::read_to_string(&sol).expect("read sol");
    let lines: Vec<&str> = text.lines().collect();
    // Header: message, blank, `Options`, its count and values, then
    // `m m_dual n n_primal`, then the dual and primal blocks.
    let opt = lines.iter().position(|l| l.trim() == "Options").unwrap();
    let nopt: usize = lines[opt + 1].trim().parse().unwrap();
    let dims = opt + 2 + nopt;
    let m: usize = lines[dims + 1].trim().parse().unwrap();
    let n: usize = lines[dims + 3].trim().parse().unwrap();
    let start = dims + 4;
    let num = |l: &str| l.trim().parse::<f64>().unwrap();
    Sol {
        duals: lines[start..start + m].iter().map(|l| num(l)).collect(),
        x: lines[start + m..start + m + n]
            .iter()
            .map(|l| num(l))
            .collect(),
        z_l: suffix(&lines, "ipopt_zL_out"),
        z_u: suffix(&lines, "ipopt_zU_out"),
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-5 * (1.0 + b.abs())
}

#[test]
fn sqp_sol_matches_the_nlp_arm_under_user_scaling() {
    let nlp = solve("nlp", &[]);
    let sqp = solve("sqp", &["algorithm=active-set-sqp"]);

    // Anchor the reference to the analytic answer first, so the parity
    // below is not two arms agreeing on a wrong number.
    assert!(
        close(nlp.x[0], 1.5) && close(nlp.x[1], 1.0),
        "NLP x: {nlp:?}"
    );
    assert_eq!(nlp.z_u.len(), 1, "NLP zU: {nlp:?}");
    assert_eq!(nlp.z_u[0].0, 1);
    assert!(close(nlp.z_u[0].1, -1.0), "NLP zU: {nlp:?}");
    assert!(close(nlp.duals[0].abs(), 3.0), "NLP dual: {nlp:?}");

    for (i, (s, r)) in sqp.x.iter().zip(&nlp.x).enumerate() {
        assert!(
            close(*s, *r),
            "x[{i}]: SQP {s} vs NLP {r} — scaling applied twice?\nSQP {sqp:?}\nNLP {nlp:?}"
        );
    }
    for (i, (s, r)) in sqp.duals.iter().zip(&nlp.duals).enumerate() {
        assert!(close(*s, *r), "dual[{i}]: SQP {s} vs NLP {r}");
    }
    for (name, s, r) in [("zL", &sqp.z_l, &nlp.z_l), ("zU", &sqp.z_u, &nlp.z_u)] {
        assert_eq!(
            s.iter().map(|e| e.0).collect::<Vec<_>>(),
            r.iter().map(|e| e.0).collect::<Vec<_>>(),
            "ipopt_{name}_out indices: SQP {s:?} vs NLP {r:?}"
        );
        for ((j, sv), (_, rv)) in s.iter().zip(r.iter()) {
            assert!(
                close(*sv, *rv),
                "ipopt_{name}_out[{j}]: SQP {sv} vs NLP {rv}"
            );
        }
    }
}

//! The square-system stage of the second-opinion ladder: when a square,
//! equality-only, constant-objective model's interior-point solve fails, and
//! every option rung that applies has failed too, POUNCE solves `c(x) = 0`
//! directly by bounded Levenberg–Marquardt on the constraint Jacobian
//! (`crates/pounce-algorithm/src/square_system.rs`).
//!
//! # The case
//!
//! `square_flowsheet_resto` (32×32, `f ≡ 0`, feasible by construction, its
//! Jacobian nearly singular even at the solution). Under
//! `hessian_approximation=limited-memory` on FERAL PR#222 the IPM runs to the
//! 3000-iteration cap and rung 4 fails as well; the LM solves it in a handful
//! of iterations. On the published FERAL 0.20 the same leg is rescued earlier,
//! by rung 4 (`feral_increase_quality=no`), so the LM never runs there by
//! default — the stage is last on purpose, so that every rescue the ladder
//! already had keeps its answer.
//!
//! # Why the recovery arm turns rung 4 off and caps `max_iter`
//!
//! To reach the stage on **both** FERAL versions with one assertion. With
//! `feral_increase_quality_retry=no` no option rung opens on a budget exit,
//! so the IPM's failure goes straight to the LM wherever it happens. The cap
//! at 500 is `issue857_escalation_gated_quality_rung`'s, for that file's
//! reason: the free-running lbfgs leg exits `Maximum_Iterations_Exceeded` on
//! macOS and `Infeasible_Problem_Detected` on linux, and the cap fires long
//! before the two part company.
//!
//! # The branches the gate takes, each with its own fixture
//!
//! A gate that is only ever seen opening says nothing about the conditions
//! that close it, so each closing condition has a model that *reaches the
//! ladder* (its solve fails) and differs from an opening one in exactly that
//! condition:
//!
//! * **non-square** — 3 free variables, 2 equality rows, constant objective,
//!   infeasible (written inline below);
//! * **non-constant objective** — `infeasible_square_scaled_1em4` and
//!   `infeasible_equalities`, square and equality-only with a real objective;
//! * **the IPM already solves it** — a square, constant-objective 2×2 model
//!   (inline) whose solve succeeds, so the ladder never opens;
//! * **an option rung promotes first** — `degenerate_start_hs008`, square
//!   with a constant objective, whose base solve fails and whose rung 3
//!   recovers it, so the LM never runs.
//!
//! The assertions read the JSON report's `second_opinion.tried`, which names
//! every stage that actually ran, so a gate that opened on a model it should
//! not have is red even where the LM would then have failed honestly.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

const LABEL: &str = "square_system_lm";

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

struct Run {
    out: String,
    report: serde_json::Value,
    sol: String,
}

impl Run {
    fn status(&self) -> &str {
        self.report["solution"]["status"].as_str().unwrap_or("?")
    }
    fn tried(&self) -> Vec<String> {
        self.report["second_opinion"]["tried"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    }
    fn promoted_by(&self) -> Option<&str> {
        self.report["second_opinion"]["promoted_by"].as_str()
    }
    fn engine(&self) -> &str {
        self.report["solution"]["engine"].as_str().unwrap_or("")
    }
    fn vec(&self, key: &str) -> Vec<f64> {
        self.report["solution"][key]
            .as_array()
            .unwrap_or_else(|| panic!("solution.{key} in report:\n{}", self.out))
            .iter()
            .map(|v| v.as_f64().expect("number"))
            .collect()
    }
}

fn solve_path(model: &std::path::Path, extra: &[&str]) -> Run {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir();
    let pid = std::process::id();
    let stem = model.file_stem().unwrap().to_string_lossy().to_string();
    let sol = dir.join(format!("pounce_sqlm_{pid}_{seq}_{stem}.sol"));
    let json = dir.join(format!("pounce_sqlm_{pid}_{seq}_{stem}.json"));
    let out = Command::new(pounce_exe())
        .arg(model)
        .arg(&sol)
        .arg("--json-output")
        .arg(&json)
        .args(extra)
        .output()
        .expect("run pounce");
    let out = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let text = std::fs::read_to_string(&json)
        .unwrap_or_else(|e| panic!("json report written ({e}):\n{out}"));
    let report = serde_json::from_str(&text).expect("valid json report");
    let sol_text = std::fs::read_to_string(&sol).unwrap_or_default();
    let _ = std::fs::remove_file(&sol);
    let _ = std::fs::remove_file(&json);
    Run {
        out,
        report,
        sol: sol_text,
    }
}

fn solve(model: &str, extra: &[&str]) -> Run {
    solve_path(&fixture(model), extra)
}

/// Write an inline `.nl` to the temp dir — kept out of `tests/fixtures/`, so
/// the sweep corpus is unchanged by this file.
fn inline_nl(name: &str, body: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("pounce_sqlm_{}_{name}.nl", std::process::id()));
    std::fs::write(&p, body).expect("write inline .nl");
    p
}

/// An independent evaluation of `square_flowsheet_resto`'s 32 rows, from
/// `dev-notes/square_flowsheet_resto_gen.py` (each row is
/// `expr(x) − expr(x*)`), in the `.nl`'s variable order: `F[0..8]`,
/// `y[8..16]`, `P[16..24]`, `T[24..32]`. Nothing here goes through the
/// solver's `.nl` reader or expression evaluator.
fn flowsheet_residual(x: &[f64]) -> Vec<f64> {
    const NS: usize = 8;
    let star = |i: usize| {
        let i = i as f64;
        (
            3.0e6 - 1.0e4 * i,
            30.0 + 0.7 * i,
            1.0e-6 * (1.0 + 0.3 * i),
            600.0 + 5.0 * i,
        )
    };
    let rows = |p: &dyn Fn(usize) -> f64,
                f: &dyn Fn(usize) -> f64,
                y: &dyn Fn(usize) -> f64,
                t: &dyn Fn(usize) -> f64| {
        let mut r = Vec::with_capacity(4 * NS);
        for i in 0..NS {
            let q = (i + NS - 1) % NS;
            r.push(f(i) - 0.97 * f(q) - 0.5 * y(q) * f(q));
        }
        for i in 0..NS {
            let q = (i + NS - 1) % NS;
            r.push(p(i) - p(q) + 1.5e3 * (f(i) / 30.0).powi(2));
        }
        for i in 0..NS {
            r.push(y(i) * p(i) - 1.0e-3 * (-6000.0 / t(i)).exp() * f(i));
        }
        for i in 0..NS {
            let q = (i + NS - 1) % NS;
            r.push(
                75.0 * f(i) * (t(i) - 298.15) - 75.0 * f(q) * (t(q) - 298.15) + 2.2e5 * y(i) * f(i),
            );
        }
        r
    };
    let at = rows(&|i| x[16 + i], &|i| x[i], &|i| x[8 + i], &|i| x[24 + i]);
    let rhs = rows(&|i| star(i).0, &|i| star(i).1, &|i| star(i).2, &|i| {
        star(i).3
    });
    at.iter().zip(&rhs).map(|(a, b)| a - b).collect()
}

const FAILING_LBFGS: &[&str] = &[
    "hessian_approximation=limited-memory",
    "feral_increase_quality_retry=no",
    "max_iter=500",
];

/// The recovery: the IPM fails, no option rung opens, the LM solves it, and
/// the answer holds up under an evaluation that shares no code with the
/// solver.
#[test]
fn lbfgs_square_flowsheet_is_recovered_by_the_square_system_stage() {
    let r = solve("square_flowsheet_resto.nl", FAILING_LBFGS);
    assert_eq!(r.status(), "SolveSucceeded", "{}", r.out);
    assert_eq!(r.promoted_by(), Some(LABEL), "{}", r.out);
    assert_eq!(r.engine(), "square-lm", "the point is LM's, not the IPM's");
    assert_eq!(
        r.tried(),
        [LABEL],
        "no option rung should open here:\n{}",
        r.out
    );
    let base = r.report["second_opinion"]["base_status"]
        .as_str()
        .unwrap_or("?");
    assert!(
        matches!(
            base,
            "Maximum_Iterations_Exceeded" | "Infeasible_Problem_Detected"
        ),
        "the stage recovers a failed IPM solve, not a converged one: base {base}"
    );
    assert!(
        r.out.contains("EXIT: Optimal Solution Found."),
        "the console verdict is the promoted one:\n{}",
        r.out
    );

    let x = r.vec("x");
    assert_eq!(x.len(), 32);
    let c = flowsheet_residual(&x);
    let worst = c.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    // constr_viol_tol is 1e-4 and tol 1e-8 against gradient-scaled rows;
    // the unscaled 1e-8 is stricter than either and the LM reaches ~3e-10.
    assert!(worst <= 1e-8, "independent max|c| = {worst:e}");

    // Declared box, exactly: F ∈ [1e-3, 1e4], y ∈ [1e-12, 1], P ∈ [1e5, 1e7],
    // T ∈ [300, 900].
    let boxes = [(1e-3, 1e4), (1e-12, 1.0), (1e5, 1e7), (300.0, 900.0)];
    for (k, &v) in x.iter().enumerate() {
        let (lo, hi) = boxes[k / 8];
        assert!(lo <= v && v <= hi, "x[{k}] = {v} outside [{lo}, {hi}]");
    }
    // The multipliers the stage documents: zero, which at ∇f = 0 satisfy
    // stationarity exactly.
    assert!(
        r.vec("lambda").iter().all(|&l| l == 0.0),
        "λ reported as zero"
    );

    // The `.sol` carries the same point the report does (the CLI's capture
    // is overwritten for this promotion — see `square_system_promoted`).
    let tail: Vec<f64> = r
        .sol
        .lines()
        .filter_map(|l| l.trim().parse::<f64>().ok())
        .collect();
    assert!(tail.len() >= 64, ".sol has a dual and a primal block");
    let xs = &tail[tail.len() - 32..];
    let worst_sol = flowsheet_residual(xs)
        .iter()
        .fold(0.0f64, |a, v| a.max(v.abs()));
    assert!(worst_sol <= 1e-8, ".sol max|c| = {worst_sol:e}");
}

/// The opt-out restores the IPM's verdict untouched.
#[test]
fn the_stage_can_be_switched_off() {
    let mut opts = FAILING_LBFGS.to_vec();
    opts.push("square_system_recovery=no");
    let r = solve("square_flowsheet_resto.nl", &opts);
    assert_ne!(r.status(), "SolveSucceeded", "{}", r.out);
    assert!(!r.tried().iter().any(|t| t == LABEL), "{:?}", r.tried());
}

/// Non-square: 3 free variables, 2 equality rows, constant objective,
/// infeasible (`x0² + x1² + x2² = −1`, `x0 = 0`), so its solve fails and the
/// ladder opens — and the square gate alone keeps the stage out.
#[test]
fn does_not_fire_on_a_non_square_model() {
    let nl = inline_nl(
        "nonsquare",
        "g3 1 1 0\t# 3 vars, 2 equality rows, constant objective, infeasible\n \
         3 2 1 0 2\n 1 0\n 0 0\n 3 0 0\n 0 0 0 1\n 0 0 0 0 0\n 4 0\n 0 0\n 0 0 0 0 0\n\
         C0\no54\n3\no5\nv0\nn2\no5\nv1\nn2\no5\nv2\nn2\n\
         C1\nn0\n\
         O0 0\nn0\n\
         x3\n0 0.5\n1 0.5\n2 0.5\n\
         r\n4 -1\n4 0\n\
         b\n3\n3\n3\n\
         k2\n2\n3\n\
         J0 3\n0 0\n1 0\n2 0\n\
         J1 1\n0 1\n",
    );
    let r = solve_path(&nl, &[]);
    let _ = std::fs::remove_file(&nl);
    assert_ne!(r.status(), "SolveSucceeded", "{}", r.out);
    assert!(
        !r.tried().is_empty(),
        "the ladder must open, or this test says nothing about the gate:\n{}",
        r.out
    );
    assert!(
        !r.tried().iter().any(|t| t == LABEL),
        "{:?}\n{}",
        r.tried(),
        r.out
    );
}

/// Square and equality-only, but with a real objective: the ladder opens on
/// each model and the constant-objective gate keeps the stage out.
///
/// The gate has two halves and each needs its own model. Both fixtures have
/// quadratic objectives, which the `.nl` degree recognizer *proves* varying,
/// so they are stopped by the `derivative_proofs` half before anything is
/// evaluated — measured: with the evaluated half disabled they stay green.
/// The inline model's objective is `exp(x0)`, which the recognizer declines
/// (`Unknown`), so only the evaluated `∇f(x₀) = 0` test can keep it out.
#[test]
fn does_not_fire_on_a_non_constant_objective() {
    let exp_obj = inline_nl(
        "exp_objective",
        "g3 1 1 0\t# square 2x2 infeasible rows, objective exp(x0)\n \
         2 2 1 0 2\n 1 1\n 0 0\n 2 1 1\n 0 0 0 1\n 0 0 0 0 0\n 4 1\n 0 0\n 0 0 0 0 0\n\
         C0\no0\no5\nv0\nn2\no5\nv1\nn2\n\
         C1\nn0\n\
         O0 0\no44\nv0\n\
         x2\n0 0.5\n1 0.5\n\
         r\n4 -1\n4 0\n\
         b\n3\n3\n\
         k1\n2\n\
         J0 2\n0 0\n1 0\n\
         J1 2\n0 1\n1 -1\n\
         G0 1\n0 0\n",
    );
    let models = [
        fixture("infeasible_square_scaled_1em4.nl"),
        fixture("infeasible_equalities.nl"),
        exp_obj.clone(),
    ];
    for model in &models {
        let r = solve_path(model, &[]);
        let name = model.display();
        assert!(
            !r.tried().is_empty(),
            "{name}: ladder should open:\n{}",
            r.out
        );
        assert!(
            !r.tried().iter().any(|t| t == LABEL),
            "{name}: {:?}\n{}",
            r.tried(),
            r.out
        );
    }
    let _ = std::fs::remove_file(&exp_obj);
}

/// Square with a constant objective, and the IPM solves it: no ladder, no
/// stage, the IPM's own answer (`x0·x1 = 2`, `x0 − x1 = 1` from `(1.5, 1.5)`).
#[test]
fn does_not_fire_when_the_ipm_already_solves_the_square_model() {
    let nl = inline_nl(
        "square_ok",
        "g3 1 1 0\t# square 2x2, constant objective, feasible\n \
         2 2 1 0 2\n 1 0\n 0 0\n 2 0 0\n 0 0 0 1\n 0 0 0 0 0\n 4 0\n 0 0\n 0 0 0 0 0\n\
         C0\no2\nv0\nv1\n\
         C1\nn0\n\
         O0 0\nn0\n\
         x2\n0 1.5\n1 1.5\n\
         r\n4 2\n4 1\n\
         b\n3\n3\n\
         k1\n2\n\
         J0 2\n0 0\n1 0\n\
         J1 2\n0 1\n1 -1\n",
    );
    for leg in [&[][..], &["hessian_approximation=limited-memory"][..]] {
        let r = solve_path(&nl, leg);
        assert_eq!(r.status(), "SolveSucceeded", "{leg:?}\n{}", r.out);
        assert!(
            r.report["second_opinion"].is_null(),
            "{leg:?}: no ladder\n{}",
            r.out
        );
        let x = r.vec("x");
        assert!((x[0] * x[1] - 2.0).abs() < 1e-7 && (x[0] - x[1] - 1.0).abs() < 1e-7);
    }
    let _ = std::fs::remove_file(&nl);
}

/// Square with a constant objective, the base solve fails, and an option
/// rung recovers it: the stage is last, so it never runs and the rung's
/// answer stands.
#[test]
fn an_option_rung_that_recovers_the_model_pre_empts_the_stage() {
    let r = solve("degenerate_start_hs008.nl", &[]);
    assert_eq!(r.status(), "SolveSucceeded", "{}", r.out);
    let promoted = r.promoted_by().unwrap_or("-");
    assert_ne!(promoted, LABEL, "{}", r.out);
    assert!(!r.tried().iter().any(|t| t == LABEL), "{:?}", r.tried());
    assert_eq!(r.engine(), "nlp", "a rung's answer is the IPM's");
}

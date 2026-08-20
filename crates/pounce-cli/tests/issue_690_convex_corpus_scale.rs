//! gh #690 (closing note) — convex fixtures large enough to measure a
//! step-rule change on.
//!
//! #690 measured an adaptive-τ tail for the HSDE corrector three times and
//! declined it the third time, not because the number was bad (−4.2% exact /
//! −1.7% L-BFGS across the corpus, no status changes, no objective
//! regressions) but because the population producing the number could not
//! answer the question:
//!
//! > The corpus's substantial models — `deb7` (813 vars), `eigena2`/`eigenb2`
//! > (110), `autocorr_bern55-06` (56), `pooling_rt2stp` (46) — are all NLP
//! > class and the convex driver never sees them. `airport` (84 vars), the
//! > only convex fixture above 32 variables, does not move at all. […] Net of
//! > the one pathological ill-scaled stress fixture, the corpus-wide saving is
//! > 90 iterations across 59 problems, almost entirely `8 → 5` on models of
//! > one to three variables.
//!
//! Every convex fixture in the corpus was a routing or verdict witness — built
//! to prove *which engine ran* or *what it reported*, which two or three
//! variables do perfectly well. None was a trajectory witness. So
//! `scripts/sweep-fixtures.sh`, the tool CLAUDE.md requires before any
//! trajectory change merges, covered the HSDE driver with arithmetic that was
//! correct and evidence that was not there.
//!
//! These four fixtures are that missing half. They are 4×–17× the variable
//! count of `lp_afiro`, they take 15–32 HSDE iterations instead of 8, and each
//! carries a published optimum from its source collection, so a trajectory
//! change that moves them can be checked against an answer nobody in this
//! repository chose.
//!
//! | fixture | source | n | m | optimum | iters |
//! |---|---|---|---|---|---|
//! | `lp_degen2` | NETLIB `degen2` | 534 | 444 | −1435.178 | 15 |
//! | `lp_share1b` | NETLIB `share1b` | 225 | 117 | −76589.319 | 32 |
//! | `lp_israel` | NETLIB `israel` | 142 | 174 | −896644.822 | 29 |
//! | `convex_qp_share1b` | Maros–Mészáros `QSHARE1B` | 225 | 117 | 720078.318 | 28 |
//!
//! Each was chosen for a distinct pathology rather than for size alone —
//! bulk would only have made the sweep slower:
//!
//! * `degen2` is massively primal-degenerate, which is the gh #535 / gh #133
//!   population: the LPs where strict complementarity fails and a pure IPM
//!   struggles to certify the vertex. It is also the largest, at 534 columns.
//! * `share1b` is the classic ill-conditioned NETLIB instance and takes the
//!   longest trajectory of the four (32 iterations), so it is the most
//!   sensitive of them to a change in how far a step is allowed to go.
//! * `israel` has dense columns, which stress the KKT factorization and its
//!   ordering rather than the barrier trajectory — a different way for a step
//!   rule to go wrong.
//! * `QSHARE1B` is `share1b` with a quadratic objective bolted on, from the
//!   Maros–Mészáros convex-QP set. Same sparsity, same bounds, non-zero `P`:
//!   it is the convex-QP branch of the same driver measured against an LP
//!   control that differs in one term.
//!
//! **Provenance.** All four are regenerated from cached upstream data by the
//! benchmark harnesses already in the tree — nothing here was hand-built:
//!
//! ```sh
//! # NETLIB LP (netlib.org/lp/data, expanded with netlib's own `emps`)
//! cd benchmarks/lp && python3 generate_nl.py --netlib-only degen2 share1b israel
//! cp nl/degen2.nl   ../../crates/pounce-cli/tests/fixtures/lp_degen2.nl
//! cp nl/share1b.nl  ../../crates/pounce-cli/tests/fixtures/lp_share1b.nl
//! cp nl/israel.nl   ../../crates/pounce-cli/tests/fixtures/lp_israel.nl
//!
//! # Maros–Mészáros convex QP (qpsolvers/maros_meszaros_qpbenchmark mirror)
//! cd benchmarks/qp && python3 generate_nl.py QSHARE1B
//! cp nl/QSHARE1B.nl ../../crates/pounce-cli/tests/fixtures/convex_qp_share1b.nl
//! ```
//!
//! `lp_afiro` came from the same NETLIB set and stays where it is: it is the
//! gh #535 / gh #588 witness and several tests name it directly. These join
//! it; they do not replace it.
//!
//! **What these tests assert** is that the fixtures keep being what they were
//! added to be — convex-path, non-trivial, and pinned to a published answer.
//! The measurement itself is the sweep's job, not a test's: absolute iteration
//! counts are the most platform-sensitive numbers in this repository (see
//! `issue_588_gondzio_correctors.rs`), so nothing below asserts one.

use std::path::PathBuf;
use std::process::Command;

use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_solve_report::SolveReport;

/// One fixture and what is known about it independently of this solver.
struct Model {
    /// Fixture stem under `tests/fixtures`.
    name: &'static str,
    /// Columns, from the `.nl` header.
    n: usize,
    /// Rows, from the `.nl` header.
    m: usize,
    /// The published optimum of the upstream instance.
    optimum: f64,
    /// The routing banner this model must produce — the whole point of the
    /// addition is that these run on the convex driver.
    class: &'static str,
}

const MODELS: [Model; 4] = [
    Model {
        name: "lp_degen2",
        n: 534,
        m: 444,
        optimum: -1435.178_000_0,
        class: "LP",
    },
    Model {
        name: "lp_share1b",
        n: 225,
        m: 117,
        optimum: -76589.318_579,
        class: "LP",
    },
    Model {
        name: "lp_israel",
        n: 142,
        m: 174,
        optimum: -896_644.821_86,
        class: "LP",
    },
    Model {
        name: "convex_qp_share1b",
        n: 225,
        m: 117,
        optimum: 720_078.318_2,
        class: "convex QP",
    },
];

/// `lp_afiro`'s column count — the ceiling the convex half of the corpus sat
/// under when #690 was closed, and the number these fixtures exist to clear.
const AFIRO_N: usize = 32;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(format!("{name}.nl"));
    p
}

fn tmp_path(tag: &str, ext: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pounce_gh690_{}_{seq}_{tag}.{ext}",
        std::process::id()
    ));
    p
}

/// Solve `model` with `opts` appended verbatim, returning the JSON report and
/// stdout (the routing banner is only on stdout).
fn solve(model: &str, opts: &[&str]) -> (SolveReport, String) {
    let tag = format!("{model}_{}", opts.join("_").replace(['=', '.'], "-"));
    let json = tmp_path(&tag, "json");
    // Explicit, so a solved fixture does not drop a `.sol` beside the `.nl`.
    let sol = tmp_path(&tag, "sol");
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture(model))
        .arg("--sol-output")
        .arg(&sol)
        .arg("--json-output")
        .arg(&json);
    for o in opts {
        cmd.arg(o);
    }
    let out = cmd.output().expect("spawn pounce");
    let text = std::fs::read_to_string(&json).unwrap_or_else(|e| {
        panic!(
            "no report for {model} @ {opts:?} (exit {:?}, {e}); stderr:\n{}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr),
        )
    });
    let _ = std::fs::remove_file(&json);
    let _ = std::fs::remove_file(&sol);
    (
        serde_json::from_str(&text).expect("parse SolveReport JSON"),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}

fn iters(r: &SolveReport) -> usize {
    r.statistics.iteration_count as usize
}

/// The answers. A trajectory sentinel is only useful if a change that breaks
/// it is visible as *wrong* and not merely as *different*, which is what
/// having a published optimum buys.
#[test]
fn each_fixture_reaches_its_published_optimum() {
    for m in &MODELS {
        let (r, stdout) = solve(m.name, &[]);
        assert_eq!(
            r.solution.status,
            ApplicationReturnStatus::SolveSucceeded,
            "{}: stdout=\n{stdout}",
            m.name
        );
        let obj = r.solution.objective;
        let rel = ((obj - m.optimum) / m.optimum).abs();
        assert!(
            rel < 1e-6,
            "{}: objective {obj} is {rel:.2e} from the published optimum {}; \
             stdout=\n{stdout}",
            m.name,
            m.optimum
        );
    }
}

/// …reached **on the convex driver**. If a routing change quietly sends these
/// to the filter-IPM they still solve, still match the published optimum, and
/// stop covering the thing they were added for — which is exactly how the gap
/// #690 closed on went unnoticed in the first place.
#[test]
fn each_fixture_runs_on_the_convex_driver() {
    for m in &MODELS {
        let (_, stdout) = solve(m.name, &[]);
        assert!(
            stdout.contains(&format!("Problem class: {}.", m.class)),
            "{}: must classify as {}; stdout=\n{stdout}",
            m.name,
            m.class
        );
        assert!(
            stdout.contains("pounce-convex"),
            "{}: the convex engine must be the one that reports; stdout=\n{stdout}",
            m.name
        );
    }
}

/// The size claim, asserted rather than described. Dimensions come from the
/// report so that swapping a fixture for a smaller model of the same name
/// fails here instead of silently shrinking the corpus back to where #690
/// found it.
#[test]
fn each_fixture_is_larger_than_the_convex_corpus_it_joins() {
    for m in &MODELS {
        let (r, stdout) = solve(m.name, &[]);
        assert_eq!(
            (
                r.problem.n_variables as usize,
                r.problem.n_constraints as usize
            ),
            (m.n, m.m),
            "{}: dimensions changed; stdout=\n{stdout}",
            m.name
        );
        assert!(
            m.n > 4 * AFIRO_N,
            "{}: {} columns is not meaningfully past `lp_afiro`'s {AFIRO_N}",
            m.name,
            m.n
        );
    }
}

/// The property that makes them sentinels: the HSDE step rule can be *seen*
/// here. `qp_tau` is the fraction-to-boundary parameter — the one knob that
/// changes how far each step is allowed to go without changing the model, the
/// convergence test, or the engine — and every one of these fixtures responds
/// to it.
///
/// Only the response is asserted, never a count. On the corpus as #690 left
/// it, this same perturbation moved most convex fixtures by `8 → 5` on one to
/// three variables, which is arithmetic, not evidence.
#[test]
fn a_step_rule_change_is_visible_on_each_fixture() {
    for m in &MODELS {
        let (base, _) = solve(m.name, &[]);
        let (perturbed, stdout) = solve(m.name, &["qp_tau=0.99999"]);
        assert_eq!(
            perturbed.solution.status,
            ApplicationReturnStatus::SolveSucceeded,
            "{}: the perturbed solve must still converge; stdout=\n{stdout}",
            m.name
        );
        assert_ne!(
            iters(&base),
            iters(&perturbed),
            "{}: a fixture whose trajectory does not move under the \
             fraction-to-boundary parameter cannot measure a step-rule change",
            m.name
        );
        // …and it moves without moving the answer. A sentinel that reports a
        // different optimum under a step-rule change is reporting a bug, and
        // this is where that would surface first.
        let rel = ((perturbed.solution.objective - m.optimum) / m.optimum).abs();
        assert!(
            rel < 1e-6,
            "{}: perturbed objective {} is {rel:.2e} from the published \
             optimum {}; stdout=\n{stdout}",
            m.name,
            perturbed.solution.objective,
            m.optimum
        );
    }
}

//! gh #760 — a convex fixture on which relaxing the bounds is *expensive*.
//!
//! `4c02817d` ("Apply bound_relax_factor on the convex arm too") cost **+515
//! iterations across the 138-problem Maros–Mészáros suite**, 4.4× on the
//! QSCFXM family, and nothing in the CLI fixture corpus predicted that.
//!
//! The widening is no longer the convex arm's default — it moved the ANSWER,
//! not just the trajectory, and by `delta` times the bound's multiplier with
//! nothing bounding that product (`LISWET1`: 33%, against a HiGHS- and
//! DOC 97/6-confirmed optimum). So this fixture now measures the cost of
//! asking for it: `bound_relax_factor=1e-8` against the default. The property
//! it pins is unchanged and so is the reason it exists — a corpus in which
//! the relaxation is cheap everywhere is the corpus gh #760 was filed about.
//!
//! The reason is not that the sweep skips the convex arm — it does not; both
//! legs of `scripts/sweep-fixtures.sh` run at `solver_selection=auto` and 42
//! of 79 fixtures never touch the NLP arm. The reason is that **no convex
//! fixture in the corpus was one on which the relaxation is expensive.**
//! Measured on `fdea82b5`, sweeping the whole corpus twice (default vs
//! `bound_relax_factor=0`), the convex lines that move at all move like this:
//!
//! | fixture | relaxed | `bound_relax_factor=0` |
//! |---|---|---|
//! | `lp_degen2` (534 cols, the largest well-posed convex fixture) | 18 | 15 |
//! | `feasible_x0_sentinel_bound` | 27 | 25 |
//! | `feasible_x0_extreme_row` | 32 | 33 |
//! | `scaled_feasible_b` | 44 | 47 |
//! | `qcqp_ball` | 12 | 17 |
//!
//! Tens of percent, in both directions, plus two ill-scaled stress fixtures
//! (`scaled_feasible_a`, `feasible_x0_wide_scale`) whose numbers are about
//! their scaling rather than about the relaxation. A reviewer reading that
//! diff learns "small and mixed" — which is true of the corpus and false of
//! the suite.
//!
//! `convex_qp_qscfxm1` is the missing row. Same measurement, same binary:
//!
//! | fixture | relaxed | `bound_relax_factor=0` |
//! |---|---|---|
//! | `convex_qp_qscfxm1` | 131 | 30 |
//!
//! 4.4×, on both sweep legs, which is the suite's own signature rather than a
//! scaled-down analogue of it. It is `QSCFXM1`, the smallest member of the
//! family the benchmark measured: 457 columns and 0.41 s, against `QSCFXM3`'s
//! 1371 columns and 1.7 s. Measured on this host, the sweep pays 1.0 s for it:
//! 10.9 s -> 11.9 s over both legs (three runs each, spread under 0.3 s).
//!
//! **Provenance.** Regenerated from the cached upstream mirror by a harness
//! already in the tree — nothing here was hand-built:
//!
//! ```sh
//! # Maros–Mészáros convex QP (qpsolvers/maros_meszaros_qpbenchmark mirror)
//! cd benchmarks/qp && python3 generate_nl.py QSCFXM1
//! cp nl/QSCFXM1.nl ../../crates/pounce-cli/tests/fixtures/convex_qp_qscfxm1.nl
//! ```
//!
//! **What this file asserts, and what it deliberately does not.** It pins the
//! routing, the dimensions, the published optimum, and the *ratio* under the
//! relaxation. It pins no absolute iteration count: those are the most
//! platform-sensitive numbers in this repository (see
//! `issue_588_gondzio_correctors.rs` and the same note in
//! `issue_690_convex_corpus_scale.rs`). Measuring is the sweep's job. The
//! threshold below is 3×, against a measured 4.4×, so the test asks only
//! whether this fixture is still in a different magnitude class from the rest
//! of the convex corpus.
//!
//! Full record of the measurement: `dev-notes/qp-bound-relax-iteration-cost.md`.

use std::path::PathBuf;
use std::process::Command;

use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_solve_report::SolveReport;

/// Fixture stem under `tests/fixtures`.
const FIXTURE: &str = "convex_qp_qscfxm1";

/// Columns and rows, from the `.nl` header — and from DOC 97/6's own table,
/// which lists `qscfxm1` at n=457 / m=330.
const N: usize = 457;
const M: usize = 330;

/// The published optimum, Maros & Mészáros DOC 97/6 (`1.68826917D+07`, BPMPD
/// at default settings). Nobody in this repository chose this number, which is
/// what makes the fixture a sentinel rather than a snapshot.
const OPTIMUM: f64 = 1.688_269_17e7;

/// How much more expensive the relaxed box is here, at minimum. Measured 4.4×
/// (131 vs 30) on `fdea82b5`, on both sweep legs. The margin is deliberate:
/// the claim under test is "different magnitude class", not "4.4".
/// Opt in to the widening: this fixture's whole subject is what that
/// costs, and it is no longer what the default does.
const RELAX: &str = "bound_relax_factor=1e-8";

const MIN_RELAX_RATIO: f64 = 3.0;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(format!("{FIXTURE}.nl"));
    p
}

fn tmp_path(tag: &str, ext: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pounce_gh760_{}_{seq}_{tag}.{ext}",
        std::process::id()
    ));
    p
}

/// Solve the fixture with `opts` appended verbatim, returning the JSON report
/// and stdout (the routing banner is only on stdout).
fn solve(opts: &[&str]) -> (SolveReport, String) {
    let tag = opts.join("_").replace(['=', '.', '-'], "_");
    let json = tmp_path(&tag, "json");
    // Explicit, so a solved fixture does not drop a `.sol` beside the `.nl`.
    let sol = tmp_path(&tag, "sol");
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture())
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
            "no report for {FIXTURE} @ {opts:?} (exit {:?}, {e}); stderr:\n{}",
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

fn assert_at_optimum(r: &SolveReport, what: &str, stdout: &str) {
    assert_eq!(
        r.solution.status,
        ApplicationReturnStatus::SolveSucceeded,
        "{FIXTURE} ({what}): must solve; stdout=\n{stdout}"
    );
    let rel = ((r.solution.objective - OPTIMUM) / OPTIMUM).abs();
    assert!(
        rel < 1e-6,
        "{FIXTURE} ({what}): objective {} is {rel:.2e} from the published \
         optimum {OPTIMUM}; stdout=\n{stdout}",
        r.solution.objective
    );
}

/// The fixture is what it was added to be: a convex QP, on the convex driver,
/// at its published answer. If a routing change quietly sends it to the
/// filter-IPM it still solves and still matches DOC 97/6 — and stops measuring
/// the arm gh #760 is about.
#[test]
fn the_fixture_is_a_convex_qp_at_its_published_optimum() {
    let (r, stdout) = solve(&[]);
    assert_at_optimum(&r, "default", &stdout);
    assert!(
        stdout.contains("Problem class: convex QP."),
        "{FIXTURE}: must classify as a convex QP; stdout=\n{stdout}"
    );
    assert!(
        stdout.contains("pounce-convex"),
        "{FIXTURE}: the convex engine must be the one that reports; \
         stdout=\n{stdout}"
    );
    assert_eq!(
        (
            r.problem.n_variables as usize,
            r.problem.n_constraints as usize
        ),
        (N, M),
        "{FIXTURE}: dimensions changed — a fixture swapped for a smaller model \
         of the same name shrinks the corpus back to where gh #760 found it; \
         stdout=\n{stdout}"
    );
}

/// The property the fixture exists for. Relaxing the bounds by
/// `bound_relax_factor` is the change `4c02817d` made to the convex arm, and
/// here it costs a *multiple* of the trajectory rather than a percentage of
/// it — which is what the rest of the convex corpus reports and what made the
/// benchmark's +515 iterations a surprise.
///
/// Note the direction: the relaxed solve is the expensive one, and it is also
/// the *less accurate* one. Against HiGHS on this model's family the widened
/// answer is the one that misses (QSCFXM1: `9.1e-09` widened vs `1.8e-13`
/// declared, on 131 iterations against 30). Both halves of that used to be
/// read the other way round.
#[test]
fn relaxing_the_bounds_costs_a_multiple_of_the_trajectory_here() {
    let (relaxed, relaxed_out) = solve(&[RELAX]);
    let (verbatim, verbatim_out) = solve(&[]);

    assert_at_optimum(&relaxed, "relaxed", &relaxed_out);
    assert_at_optimum(&verbatim, "declared (default)", &verbatim_out);

    let ratio = iters(&relaxed) as f64 / iters(&verbatim).max(1) as f64;
    assert!(
        ratio >= MIN_RELAX_RATIO,
        "{FIXTURE}: relaxed {} iters vs verbatim {} iters is {ratio:.2}×, \
         under the {MIN_RELAX_RATIO}× this fixture was added to hold. A convex \
         corpus in which the bound relaxation is cheap everywhere is the \
         corpus gh #760 was filed about.",
        iters(&relaxed),
        iters(&verbatim),
    );
}

/// …and on the limited-memory leg too, which is the one the Python frontend
/// and the CasADi plugin select on their own. The convex driver does not use a
/// Lagrangian Hessian, so the two legs should agree exactly here; asserting it
/// is how a future routing change that makes them disagree becomes visible.
#[test]
fn the_cost_is_the_same_on_the_limited_memory_leg() {
    const LBFGS: &str = "hessian_approximation=limited-memory";

    let (exact_relaxed, out) = solve(&[RELAX]);
    assert_at_optimum(&exact_relaxed, "relaxed", &out);
    let (lbfgs_relaxed, out) = solve(&[LBFGS, RELAX]);
    assert_at_optimum(&lbfgs_relaxed, "lbfgs relaxed", &out);
    let (lbfgs_verbatim, out) = solve(&[LBFGS]);
    assert_at_optimum(&lbfgs_verbatim, "lbfgs declared (default)", &out);

    assert_eq!(
        iters(&exact_relaxed),
        iters(&lbfgs_relaxed),
        "{FIXTURE}: the convex driver takes no Lagrangian Hessian, so the two \
         sweep legs must walk the same trajectory here"
    );
    let ratio = iters(&lbfgs_relaxed) as f64 / iters(&lbfgs_verbatim).max(1) as f64;
    assert!(
        ratio >= MIN_RELAX_RATIO,
        "{FIXTURE}: on the lbfgs leg, relaxed {} iters vs verbatim {} iters is \
         {ratio:.2}×, under the {MIN_RELAX_RATIO}× this fixture holds",
        iters(&lbfgs_relaxed),
        iters(&lbfgs_verbatim),
    );
}

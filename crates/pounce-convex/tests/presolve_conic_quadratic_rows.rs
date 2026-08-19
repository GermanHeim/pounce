//! Presolve validity for **quadratic** rows on the conic path (gh #588 Q9b,
//! §7 of `dev-notes/quadratic-structure-exploitation.md`).
//!
//! §7 names one reduction as the silent-wrong-answer risk: parallel/duplicate
//! row removal. `parallel_signature` builds a row's signature from its
//! *linear* triplets, so "two quadratic rows with identical `aᵢ` and different
//! `Qᵢ` hash equal and one is dropped".
//!
//! That table was written for the native `QcqpProblem` of Q7, which does not
//! exist. It reaches the **conic** path anyway, and by a sharper route than
//! §7 describes: `extract_socp_with_map` writes each quadratic row
//! `½xᵀQᵢx + aᵢᵀx + bᵢ ≤ 0` as a second-order cone block whose rows 0 and 1
//! are `aᵢ` *verbatim* (`h = 1 − bᵢ` and `h = −(1 + bᵢ)`), and only whose
//! rows 2.. carry `Qᵢ`, as `−√2·Fᵢ` with `FᵢᵀFᵢ = Qᵢ`. Two quadratic rows that
//! share a linear part therefore produce **byte-identical** `G` rows in
//! *different* cones — the collision §7 predicts, with no quantization slack
//! needed to reach it.
//!
//! The guard is `presolve_conic`, which marks every non-orthant row
//! `protected`; `dedup_rows` excludes protected rows from grouping entirely,
//! so they are "never dropped and never drop others". These tests hold that
//! guard down from both sides: what the *unprotected* entry point does to
//! this shape (a wrong answer reported as `Optimal`), and that the cone-aware
//! one does not do it.

use pounce_convex::presolve::{PresolveOutcome, presolve_conic};
use pounce_convex::{
    ConeSpec, QpOptions, QpProblem, QpSolution, QpStatus, Triplet, solve_socp_ipm,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn solve(prob: &QpProblem, cones: &[ConeSpec]) -> QpSolution {
    solve_socp_ipm(prob, cones, &QpOptions::default(), backend)
}

/// The exact conic problem `extract_socp_with_map` builds for the fixture
/// `crates/pounce-cli/tests/fixtures/qcqp_shared_linear_rows.nl`:
///
/// ```text
/// min  −x0 − x1 − x2 − x3        x ∈ [−10, 10]^4
///  c0: 2x0² + x0 + x1 ≤ 20       (quadratic)
///  c1: 2x1² + x0 + x1 ≤ 20       (quadratic — same linear part as c0)
///  c2: x2 + x3 ≤ 5
///  c3: x2 + x3 ≤ 5               (exact duplicate of c2)
///  c4: 2x2 + 2x3 ≤ 10            (a positive multiple of c2)
/// ```
///
/// `c0` and `c1` differ **only** in `Q` — `diag(4,0,0,0)` against
/// `diag(0,4,0,0)` — and share `a = (1,1,0,0)` and `b_eff = −20`. Their cone
/// blocks' rows 0 and 1 are identical in coefficients *and* right-hand side.
///
/// Row order matches the extractor's: the orthant block first (c2, c3, c4),
/// then one SOC block per quadratic row.
fn shared_linear_part_qcqp() -> (QpProblem, Vec<ConeSpec>) {
    let sqrt2 = std::f64::consts::SQRT_2;
    // Fᵢ for Qᵢ = diag(4, ...) is the 1×1 factor [2], so the cone tail row is
    // −√2·2·x_k. Each block is rank 1 + 2 = 3 rows.
    let f = 2.0;
    let prob = QpProblem {
        n: 4,
        p_lower: vec![],
        c: vec![-1.0, -1.0, -1.0, -1.0],
        a: vec![],
        b: vec![],
        g: vec![
            // --- orthant block: c2, c3, c4 ---
            Triplet::new(0, 2, 1.0),
            Triplet::new(0, 3, 1.0),
            Triplet::new(1, 2, 1.0),
            Triplet::new(1, 3, 1.0),
            Triplet::new(2, 2, 2.0),
            Triplet::new(2, 3, 2.0),
            // --- SOC block for c0: rows 3,4 are `a`; row 5 is −√2·F ---
            Triplet::new(3, 0, 1.0),
            Triplet::new(3, 1, 1.0),
            Triplet::new(4, 0, 1.0),
            Triplet::new(4, 1, 1.0),
            Triplet::new(5, 0, -sqrt2 * f),
            // --- SOC block for c1: rows 6,7 are the SAME `a`; row 8 is −√2·F ---
            Triplet::new(6, 0, 1.0),
            Triplet::new(6, 1, 1.0),
            Triplet::new(7, 0, 1.0),
            Triplet::new(7, 1, 1.0),
            Triplet::new(8, 1, -sqrt2 * f),
        ],
        //                c2   c3   c4    c0 block      c1 block
        h: vec![5.0, 5.0, 10.0, 21.0, 19.0, 0.0, 21.0, 19.0, 0.0],
        lb: vec![-10.0; 4],
        ub: vec![10.0; 4],
    };
    let cones = vec![
        ConeSpec::Nonneg(3),
        ConeSpec::SecondOrder(3),
        ConeSpec::SecondOrder(3),
    ];
    (prob, cones)
}

/// The true optimum, derived by hand. Maximizing `s = x0 + x1` subject to
/// `2x0² ≤ 20 − s` and `2x1² ≤ 20 − s` gives `s ≤ √(2(20 − s))`, i.e.
/// `s² + 2s − 40 ≤ 0`, so `s = √41 − 1`. With `x2 + x3 = 5` the objective is
/// `−(√41 − 1) − 5`.
fn true_optimum() -> f64 {
    -(41f64.sqrt() - 1.0) - 5.0
}

/// **The defect, demonstrated.** Run the reduction pass without telling it
/// which rows are cone rows and it hashes all four `a` rows — both blocks'
/// rows 0 and 1 — into one parallel group, keeps the most restrictive, and
/// drops the other three. Nothing reports an error: the reduced problem is a
/// perfectly well-formed conic program, it solves to `Optimal`, and the
/// objective it returns is wrong by 67% (−17.3508 against −10.4031).
///
/// This is why the wiring in `run_convex_socp` calls `presolve_conic` with
/// the real cone list and never the orthant `presolve`. It is a
/// characterization test, not a regression test: until this branch no
/// production caller pointed *any* presolve at a conic problem, so the hazard
/// was latent rather than live. What it pins is that the protection is
/// load-bearing — remove it and this is the answer the solver returns.
#[test]
fn the_unprotected_entry_point_deletes_half_a_quadratic_rows_cone() {
    let (prob, cones) = shared_linear_part_qcqp();

    // The same single pass, told that every row is an ordinary orthant row —
    // i.e. exactly what a presolve that does not know which rows came from a
    // quadratic constraint runs. Nothing else about the call changes, so the
    // difference between this test and the next one is the protection and
    // nothing else.
    let blind = [ConeSpec::Nonneg(prob.m_ineq())];
    let ps = match presolve_conic(&prob, &blind) {
        PresolveOutcome::Reduced(ps) => ps,
        _ => panic!("expected Reduced"),
    };

    // Two of the three orthant rows are genuine duplicates and go legally.
    // The other two losses are cone rows. All four `a` rows — both blocks'
    // rows 0 and 1 — normalize to the same pattern `(1,1)` and land in one
    // parallel group, so the merge keeps the single most restrictive of them
    // (`h = 19`) and drops the other three.
    assert_eq!(ps.reduced.m_ineq(), 4, "9 → 4 rows");
    let reduced_cones = ps.reduced_cones(&cones);
    assert_eq!(
        reduced_cones,
        vec![
            ConeSpec::Nonneg(1),
            ConeSpec::SecondOrder(2),
            ConeSpec::SecondOrder(1),
        ],
        "both quadratic rows' cone blocks have been cut — 3 rows → 2 and \
         3 rows → 1. Neither encodes the constraint it was built for any more"
    );

    // And the answer that comes out of it. `SecondOrder(1)` is `{s ≥ 0}`,
    // i.e. `x1 ≥ 0`, standing where `2x1² + x0 + x1 ≤ 20` used to be, and
    // the surviving `SecondOrder(2)` is `19 − x0 − x1 ≥ 2√2|x0|` rather than
    // `2x0² + x0 + x1 ≤ 20`. Both are *relaxations*, so the optimum moves
    // down and away.
    let red = solve(&ps.reduced, &reduced_cones);
    let sol = ps.postsolve(&red);
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "the corrupted problem does not fail — that is the whole point"
    );
    let wrong = sol.obj;
    assert!(
        wrong < true_optimum() - 5.0,
        "the reported objective ({wrong}) should be far below the true \
         optimum ({}) — a silently wrong answer, reported as Optimal",
        true_optimum()
    );
    // Pinned so a change in the merge rule cannot quietly turn this into a
    // *different* wrong answer and still pass.
    assert!(
        (wrong - (-17.35083)).abs() < 1e-4,
        "expected the wrong answer −17.35083, got {wrong}"
    );
}

/// The guard. `presolve_conic` protects every non-orthant row, so the two
/// cone blocks survive whole while the orthant duplicates still go — and the
/// postsolved answer is the true optimum.
#[test]
fn the_cone_aware_entry_point_keeps_every_quadratic_block_whole() {
    let (prob, cones) = shared_linear_part_qcqp();

    let ps = match presolve_conic(&prob, &cones) {
        PresolveOutcome::Reduced(ps) => ps,
        _ => panic!("expected Reduced"),
    };

    assert_eq!(
        ps.reduced.m_ineq(),
        7,
        "9 → 7: exactly the two orthant duplicates, and nothing from a cone"
    );
    let reduced_cones = ps.reduced_cones(&cones);
    assert_eq!(
        reduced_cones,
        vec![
            ConeSpec::Nonneg(1),
            ConeSpec::SecondOrder(3),
            ConeSpec::SecondOrder(3),
        ],
        "both quadratic rows must keep their full 3-row block"
    );

    let red = solve(&ps.reduced, &reduced_cones);
    let sol = ps.postsolve(&red);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!(
        (sol.obj - true_optimum()).abs() < 1e-6,
        "postsolved objective {} != analytic optimum {}",
        sol.obj,
        true_optimum()
    );

    // And it agrees with the same problem solved without presolve at all.
    let bare = solve(&prob, &cones);
    assert_eq!(bare.status, QpStatus::Optimal);
    assert!(
        (sol.obj - bare.obj).abs() < 1e-6,
        "presolved {} vs bare {}",
        sol.obj,
        bare.obj
    );
}

/// A tiny reproducible PRNG (SplitMix64), so the battery below is a fixed
/// sequence rather than a different problem set on every run. Same pattern as
/// `quad_evaluator_differential.rs`.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform on `[-1, 1)`.
    fn signed(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// The differential battery §7's warning actually calls for. The fixture
/// corpus is what missed gh #683 and gh #685, so the evidence here is a
/// synthetic sweep instead: random QCQP-shaped conic problems, each solved
/// twice — bare, and through `presolve_conic` + postsolve — with the
/// objectives required to agree.
///
/// Every instance is built the way `extract_socp_with_map` builds one: a
/// diagonal `Fᵢ` on a random support, the linear part `aᵢ` repeated as the
/// block's first two rows, and — the shape that matters — **a linear part
/// deliberately shared between quadratic rows**, plus duplicate and parallel
/// orthant rows for presolve to have something legal to remove.
#[test]
fn presolve_conic_agrees_with_no_presolve_over_a_random_qcqp_battery() {
    let sqrt2 = std::f64::consts::SQRT_2;
    let mut agreed = 0usize;
    let mut reduced_something = 0usize;
    for seed in 0..64u64 {
        let mut rng = Rng(seed.wrapping_mul(0x2545F4914F6CDD1D) ^ 0xA5A5);
        let n = 4 + rng.below(5); // 4..8 variables
        let n_quad = 2 + rng.below(3); // 2..4 quadratic rows
        let n_lin = 2 + rng.below(4); // 2..5 orthant rows before duplication

        // One linear part, shared by every quadratic row — the collision.
        let shared: Vec<f64> = (0..n).map(|_| rng.signed()).collect();

        let mut g: Vec<Triplet> = Vec::new();
        let mut h: Vec<f64> = Vec::new();
        let mut row = 0usize;

        // --- orthant block: each row emitted twice (exact duplicate) and
        //     once more as a positive multiple, all three legally mergeable.
        for _ in 0..n_lin {
            let coeffs: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
            let rhs = 2.0 + rng.signed().abs() * 3.0;
            for scale in [1.0, 1.0, 2.5] {
                for (col, &v) in coeffs.iter().enumerate() {
                    if v != 0.0 {
                        g.push(Triplet::new(row, col, scale * v));
                    }
                }
                h.push(scale * rhs);
                row += 1;
            }
        }
        let n_orthant = row;

        // --- one SOC block per quadratic row ---
        let mut cones = vec![ConeSpec::Nonneg(n_orthant)];
        for q in 0..n_quad {
            // Qᵢ = diag on a support unique to this row, so the blocks differ
            // in Q and in nothing else.
            let support: Vec<usize> = (0..n).filter(|c| (c + q) % n_quad == 0).collect();
            let b_eff = -(5.0 + rng.signed().abs() * 5.0);
            for hv in [1.0 - b_eff, -(1.0 + b_eff)] {
                for (col, &v) in shared.iter().enumerate() {
                    if v != 0.0 {
                        g.push(Triplet::new(row, col, v));
                    }
                }
                h.push(hv);
                row += 1;
            }
            for &col in &support {
                let fv = 0.5 + rng.signed().abs();
                g.push(Triplet::new(row, col, -sqrt2 * fv));
                h.push(0.0);
                row += 1;
            }
            cones.push(ConeSpec::SecondOrder(support.len() + 2));
        }

        let prob = QpProblem {
            n,
            p_lower: (0..n).map(|i| Triplet::new(i, i, 1.0)).collect(),
            c: (0..n).map(|_| rng.signed()).collect(),
            a: vec![],
            b: vec![],
            g,
            h,
            lb: vec![-8.0; n],
            ub: vec![8.0; n],
        };

        let bare = solve(&prob, &cones);
        if bare.status != QpStatus::Optimal {
            // Nothing to compare against; the battery is about agreement, not
            // about every random instance being solvable.
            continue;
        }
        let ps = match presolve_conic(&prob, &cones) {
            PresolveOutcome::Reduced(ps) => ps,
            other => panic!(
                "seed {seed}: presolve_conic returned {} on a strictly feasible \
                 instance the bare solve calls Optimal",
                if matches!(other, PresolveOutcome::Unbounded) {
                    "Unbounded"
                } else {
                    "Infeasible"
                }
            ),
        };
        // Whatever it removed, every cone block must still be whole.
        let reduced_cones = ps.reduced_cones(&cones);
        for (before, after) in cones.iter().zip(&reduced_cones) {
            if let (ConeSpec::SecondOrder(a), ConeSpec::SecondOrder(b)) = (before, after) {
                assert_eq!(a, b, "seed {seed}: a cone block lost rows");
            }
        }
        assert_eq!(
            reduced_cones.len(),
            cones.len(),
            "seed {seed}: a whole cone block vanished"
        );
        if ps.stats().reduced_anything() {
            reduced_something += 1;
        }
        let red = solve(&ps.reduced, &reduced_cones);
        let sol = ps.postsolve(&red);
        assert_eq!(sol.status, QpStatus::Optimal, "seed {seed}");
        assert!(
            (sol.obj - bare.obj).abs() <= 1e-6 * (1.0 + bare.obj.abs()),
            "seed {seed}: presolved objective {} != bare {}",
            sol.obj,
            bare.obj
        );
        agreed += 1;
    }
    // The battery is worthless if it never exercised the reduction, and worth
    // little if only a handful of instances were comparable.
    assert!(agreed >= 40, "only {agreed} instances were comparable");
    assert!(
        reduced_something >= 40,
        "presolve reduced only {reduced_something} of {agreed} instances — the \
         battery is not exercising the reduction it is meant to guard"
    );
}

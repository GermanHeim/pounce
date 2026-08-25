//! Solving a parametric family through [`ActiveSetSession`] (gh #769).
//!
//! The active-set engine's headline capability is *parametric* reuse: given a
//! solved neighbouring QP it traces a homotopy to the new one instead of
//! starting over. Reaching that from the convex path used to require
//! restating the private convex → `pounce-qp` translation, because
//! [`solve_qp_active_set`] builds it in locals and drops it — so every solve
//! through the convex API was cold, however close the problems were.
//!
//! This example is the measurement behind the session: the same path of QPs
//! solved cold and through a session, printing the wall clock and the engine's
//! own cost measure (working-set changes) per solve. On a family the homotopy
//! traces from the box relaxation, the *cold* solve already reports zero
//! working-set changes — the whole path is the cold start — so the wall clock
//! is what separates the two arms there.
//!
//! Run: `cargo run -p pounce-convex --example active_set_session`

use pounce_convex::{
    ActiveSetOverrides, ActiveSetSession, QpOptions, QpProblem, Triplet, solve_qp_active_set,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// `min ½ xᵀ diag(d) x + cᵀx  s.t.  Σx ≤ cap,  0 ≤ x ≤ 10`, with a wide
/// eigenvalue spread so a cold solve has enough work in it for reuse to show.
///
/// Only `c` and `cap` move across the family — `P`, `G` and the box are fixed,
/// which is the shape the homotopy interpolates (`g` and the row bounds).
fn capped_qp(c: &[f64], cap: f64) -> QpProblem {
    let n = c.len();
    let cond = 1e4_f64;
    let p_lower: Vec<Triplet> = (0..n)
        .map(|i| {
            let t = i as f64 / (n.max(2) as f64 - 1.0);
            Triplet::new(i, i, 2.0 * cond.powf(t))
        })
        .collect();
    QpProblem {
        n,
        p_lower,
        c: c.to_vec(),
        a: vec![],
        b: vec![],
        g: (0..n).map(|i| Triplet::new(0, i, 1.0)).collect(),
        h: vec![cap],
        lb: vec![0.0; n],
        ub: vec![10.0; n],
    }
}

fn main() {
    let opts = QpOptions::default();
    let n = 40;
    let base_c: Vec<f64> = (0..n).map(|i| -1.0 - (i as f64) * 0.05).collect();
    let steps = 8;

    // Presolve off on both arms so the two columns differ by reuse alone.
    let mut session = ActiveSetSession::new(backend)
        .with_presolve(false)
        .with_options(opts);

    let mut cold_total = 0usize;
    let mut warm_total = 0usize;

    let mut cold_time = 0.0;
    let mut warm_time = 0.0;
    println!(
        "{:<5} {:>9} {:>9} {:>10} {:>10} {:>15}  {}",
        "step", "cold_wsc", "sess_wsc", "cold_ms", "sess_ms", "reuse", "objective"
    );
    for k in 0..steps {
        let scale = 1.0 + 0.005 * (k as f64 + 1.0);
        let c: Vec<f64> = base_c.iter().map(|v| v * scale).collect();
        let prob = capped_qp(&c, 5.0 + 0.02 * (k as f64 + 1.0));

        let mut mk = backend;
        let t0 = std::time::Instant::now();
        let cold = solve_qp_active_set(&prob, &opts, &ActiveSetOverrides::default(), &mut mk);
        let cold_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = std::time::Instant::now();
        let warm = session.solve(&prob);
        let warm_ms = t1.elapsed().as_secs_f64() * 1e3;

        // The two arms must agree on the answer; reuse is a cost claim, not a
        // licence to report something else.
        assert_eq!(warm.status, cold.status);
        assert!(
            (warm.obj - cold.obj).abs() <= 1e-7 * (1.0 + cold.obj.abs()),
            "step {k}: session {} vs cold {}",
            warm.obj,
            cold.obj
        );
        println!(
            "{:<5} {:>9} {:>9} {:>10.2} {:>10.2} {:>15}  {:.8e}",
            k,
            cold.iters,
            warm.iters,
            cold_ms,
            warm_ms,
            format!("{:?}", session.last_reuse()),
            warm.obj,
        );
        cold_total += cold.iters;
        warm_total += warm.iters;
        cold_time += cold_ms;
        warm_time += warm_ms;
    }

    let st = session.stats();
    println!(
        "\nworking-set changes: cold={cold_total} session={warm_total}\n\
         wall clock: cold={cold_time:.1}ms session={warm_time:.1}ms\n\
         attempts={} accepted={} (homotopy={} working-set={} engine-cold={}) \
         cold_solves={}",
        st.parametric_attempts,
        st.attempts_accepted(),
        st.homotopy_accepted,
        st.working_set_accepted,
        st.engine_cold_accepted,
        st.cold_solves
    );
    assert_eq!(
        st.attempts_accepted() + st.cold_solves,
        steps,
        "every step is either reused or solved cold"
    );
    // The point of the run is the homotopy, so say so where it can fail. An
    // accepted attempt is not evidence the path was traced: `solve_parametric`
    // declines on a changed `H` or topology and answers from the working set
    // instead, which is warm but not what this family is demonstrating
    // (gh #769, review). Without this the timing table could report a win from
    // a route the text does not describe.
    assert_eq!(
        st.homotopy_accepted, st.parametric_attempts,
        "this family moves only `c`, so every attempt should trace the path"
    );
}

//! Which orientation does `Solver::compute_reduced_hessian` report?
//!
//! Run with:
//!
//! ```text
//! cargo run --release -p pounce-sensitivity --example rh_orientation_check
//! ```
//!
//! gh#937. The pin path returns `−H_R`, and until that issue the sign
//! was recorded only inside two crossover test files. This example is
//! the one-command demonstration: it prints the matrix and its spectrum
//! on a model whose reduced Hessian is known exactly, and scores it
//! against all three candidates a reader might assume — `+H_R`, `−H_R`
//! and `H_R⁻¹`. Scoring against three rather than two is what makes it
//! a check rather than a sign convention restated: `+H_R` and `−H_R`
//! differ only in sign, but `H_R⁻¹` differs in *magnitude*, so a run
//! that merely negated would still be caught.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_sensitivity::Solver;

/// `min x0² + x1² + x0·x1`  s.t.  `g0: x0 = p0`, `g1: x1 = p1`.
///
/// The objective Hessian is `H = [[2, 1], [1, 2]]`. Both variables are
/// pinned, so the null space of the active constraints is `{0}` and the
/// reduced Hessian over the two pin rows is `H` itself — which is the
/// point: there is no projection to get wrong, so whatever the accessor
/// returns is `±H` or `H⁻¹` and nothing else.
struct PinnedQuadratic {
    p0: Number,
    p1: Number,
}

impl TNLP for PinnedQuadratic {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 2,
            nnz_jac_g: 2,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for k in 0..2 {
            b.x_l[k] = -1.0e19;
            b.x_u[k] = 1.0e19;
        }
        b.g_l[0] = self.p0;
        b.g_u[0] = self.p0;
        b.g_l[1] = self.p1;
        b.g_u[1] = self.p1;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = self.p0;
        sp.x[1] = self.p1;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[0] + x[1] * x[1] + x[0] * x[1])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * x[0] + x[1];
        g[1] = 2.0 * x[1] + x[0];
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0];
        g[1] = x[1];
        true
    }

    fn eval_jac_g(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0 as Index, 1 as Index]);
                jcol.copy_from_slice(&[0 as Index, 1 as Index]);
            }
            SparsityRequest::Values { values } => values.copy_from_slice(&[1.0, 1.0]),
        }
        true
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                // lower triangle of [[2, 1], [1, 2]]
                irow.copy_from_slice(&[0 as Index, 1 as Index, 1 as Index]);
                jcol.copy_from_slice(&[0 as Index, 0 as Index, 1 as Index]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[2.0 * obj_factor, obj_factor, 2.0 * obj_factor]);
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// `H` itself, column-major, since every variable is pinned.
const H: [Number; 4] = [2.0, 1.0, 1.0, 2.0];

fn main() {
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(PinnedQuadratic { p0: 1.0, p1: 2.0 }));
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.initialize().unwrap();

    let mut solver = Solver::new(app, tnlp);
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "solve failed: {status:?}"
    );

    let (hr, vals, vecs) = solver
        .compute_reduced_hessian_eigen(&[0, 1], 1.0)
        .expect("reduced Hessian");

    println!("model:  min x0² + x1² + x0·x1  s.t.  x0 = 1, x1 = 2");
    println!("H    = [[2, 1], [1, 2]]   (eigenvalues 1 and 3)");
    println!();
    println!("compute_reduced_hessian(pins=[0, 1]):");
    for i in 0..2 {
        println!("  [{:>9.6}, {:>9.6}]", hr[i], hr[i + 2]);
    }
    println!(
        "  eigenvalues (ascending) = [{:>9.6}, {:>9.6}]",
        vals[0], vals[1]
    );
    for j in 0..2 {
        println!(
            "  eigenvector[{j}]          = [{:>9.6}, {:>9.6}]",
            vecs[2 * j],
            vecs[2 * j + 1]
        );
    }
    println!();

    // Three candidates, discriminated by magnitude as well as sign:
    // inv(H) = [[2/3, -1/3], [-1/3, 2/3]].
    let h_inv: [Number; 4] = [2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0];
    let candidates: [(&str, [Number; 4]); 3] = [
        ("+H_R", H),
        ("-H_R", [-H[0], -H[1], -H[2], -H[3]]),
        ("H_R⁻¹", h_inv),
    ];
    for (name, want) in candidates {
        let err = (0..4)
            .map(|k| (hr[k] - want[k]).abs())
            .fold(0.0 as Number, Number::max);
        println!(
            "  vs {name:<6} max|Δ| = {err:.3e}  {}",
            if err < 1e-7 { "← MATCH" } else { "" }
        );
    }
    println!();
    println!(
        "So the ascending spectrum runs STIFFEST first: {:.6} is the curvature-3",
        vals[0]
    );
    println!(
        "mode and {:.6} the curvature-1 (soft) one — the reverse of the",
        vals[1]
    );
    println!("order a caller reading `+H_R` would assume.");
}

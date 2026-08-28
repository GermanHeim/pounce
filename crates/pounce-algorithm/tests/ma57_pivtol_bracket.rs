//! `ma57_pivtolmax` below `ma57_pivtol` is refused, not silently lifted.
//!
//! `ma57_pivtolmax` is the ceiling MA57 may raise the pivot tolerance to
//! when `increase_quality` escalates for accuracy; `ma57_pivtol` is where
//! it starts. A ceiling below the floor is a contradiction, and upstream
//! refuses it (`IpMa57TSolverInterface.cpp:311-320`, `OPTION_INVALID`):
//!
//! ```cpp
//! if( options.GetNumericValue("ma57_pivtolmax", pivtolmax_, prefix) )
//!    ASSERT_EXCEPTION(pivtolmax_ >= pivtol_, OPTION_INVALID, ...);
//! else
//!    pivtolmax_ = Max(pivtolmax_, pivtol_);
//! ```
//!
//! pounce's reader applied that `Max` unconditionally, so an explicitly
//! set `ma57_pivtolmax` under `ma57_pivtol` was quietly rewritten to
//! `ma57_pivtol` — a self-contradictory pair accepted with no diagnostic.
//! Harmless while gh#825 was live, because no `ma57_*` value reached the
//! backend at all; reachable the moment that was fixed.
//!
//! **The rule branches, and both branches are tested here**, because
//! either one alone passes while the other is broken:
//!
//! * *explicitly set* and below `ma57_pivtol` → refused;
//! * *unset* and below `ma57_pivtol` → lifted, and the solve proceeds.
//!
//! The second is not a formality. `ma57_pivtolmax` is registered with a
//! default of `1e-4`, so **any** `ma57_pivtol` above `1e-4` puts the
//! default below the floor. A rule written as "refuse whenever
//! `pivtolmax < pivtol`" would reject `ma57_pivtol 0.5` on its own —
//! the single most ordinary thing a user tuning MA57 does.
//!
//! This file is **not** feature-gated, and that is the point: the check
//! is a comparison of two numbers the user wrote, needs no HSL to
//! perform, and CI cannot link CoinHSL. It is the one part of the
//! gh#825 work that CI exercises end to end.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// `min (x - 1)^2`, unconstrained. As small as a TNLP gets: the refusal
/// fires before any numerical work, so the model only has to exist.
struct Quadratic;

impl TNLP for Quadratic {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 1,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: 1,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-2.0e19]);
        b.x_u.copy_from_slice(&[2.0e19]);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.0]);
        true
    }
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] - 1.0) * (x[0] - 1.0))
    }
    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 1.0);
        true
    }
    fn eval_g(&mut self, _x: &[Number], _new_x: bool, _g: &mut [Number]) -> bool {
        true
    }
    fn eval_jac_g(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        _mode: SparsityRequest<'_>,
    ) -> bool {
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        of: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let z: [Index; 1] = [0];
                irow.copy_from_slice(&z);
                jcol.copy_from_slice(&z);
            }
            SparsityRequest::Values { values, .. } => values[0] = of * 2.0,
        }
        true
    }
    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn solve_with(options: &str) -> ApplicationReturnStatus {
    let mut app = IpoptApplication::new();
    app.initialize().expect("registry initializes");
    app.initialize_with_options_str(&format!("print_level 0\n{options}"))
        .unwrap_or_else(|e| panic!("options rejected at set time: {e:?}\n{options}"));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Quadratic));
    app.optimize_tnlp(tnlp)
}

fn solved(s: ApplicationReturnStatus) -> bool {
    matches!(
        s,
        ApplicationReturnStatus::SolveSucceeded | ApplicationReturnStatus::SolvedToAcceptableLevel
    )
}

/// The refusing branch. Both values are individually legal — the
/// registry bounds each to `(0, 1]` — so nothing rejects them at set
/// time, and before this check nothing rejected the pair either.
#[test]
fn an_explicit_pivtolmax_below_pivtol_is_refused() {
    let status = solve_with("ma57_pivtol 0.5\nma57_pivtolmax 1e-9\n");
    assert_eq!(
        status,
        ApplicationReturnStatus::InvalidOption,
        "an explicit ma57_pivtolmax under ma57_pivtol must be refused, not lifted"
    );
}

/// The same pair under `"resto."`. The restoration sub-IPM configures
/// its own MA57 backend from `resto.`-scoped options (gh#825), so this
/// pair can contradict itself while the un-prefixed one is fine — and a
/// check that looked at only one prefix would pass here.
#[test]
fn the_resto_prefixed_pair_is_refused_too() {
    let status = solve_with("resto.ma57_pivtol 0.5\nresto.ma57_pivtolmax 1e-9\n");
    assert_eq!(
        status,
        ApplicationReturnStatus::InvalidOption,
        "the `resto.` prefix has its own MA57 backend and its own pair to keep consistent"
    );
}

/// The lifting branch, and the reason the refusal is conditioned on the
/// option being *explicitly set*.
///
/// `ma57_pivtolmax` defaults to `1e-4`, which is below this `ma57_pivtol`
/// — so a rule that just compared the two numbers would reject the most
/// ordinary MA57 tuning there is. Upstream lifts the default instead,
/// and so must pounce.
#[test]
fn an_unset_pivtolmax_is_lifted_not_refused() {
    let status = solve_with("ma57_pivtol 0.5\n");
    assert!(
        solved(status),
        "raising ma57_pivtol above the ma57_pivtolmax *default* must lift the default, \
         not refuse the solve — got {status:?}"
    );
}

/// The boundary. Upstream asserts `pivtolmax >= pivtol`, so equality is
/// legal; an off-by-one written as `>` would fail here and nowhere else.
#[test]
fn an_explicit_pivtolmax_equal_to_pivtol_is_accepted() {
    let status = solve_with("ma57_pivtol 0.5\nma57_pivtolmax 0.5\n");
    assert!(
        solved(status),
        "pivtolmax == pivtol is legal — got {status:?}"
    );
}

/// An ordinary explicit pair, above the floor.
#[test]
fn an_explicit_pivtolmax_above_pivtol_is_accepted() {
    let status = solve_with("ma57_pivtol 1e-6\nma57_pivtolmax 0.5\n");
    assert!(solved(status), "status = {status:?}");
}

/// And the check does not fire on a solve that never mentions MA57 —
/// the guard against a refusal that quietly rejects everything.
#[test]
fn a_solve_with_no_ma57_options_is_unaffected() {
    let status = solve_with("");
    assert!(solved(status), "status = {status:?}");
}

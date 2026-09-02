//! The parametric sensitivity step on the **convex** dispatch path.
//!
//! [`crate::sens`] is the NLP arm's producer: it reads the filter-IPM's
//! converged KKT factor through `PdSensBacksolver` and is hard-wired to
//! `pounce_algorithm` types. This module is its convex counterpart, built on
//! `pounce_convex::QpSensitivity` — the same sIPOPT computation over the
//! active-set KKT the convex IPM's solution defines.
//!
//! # Why the CLI needed one at all
//!
//! Before this, a `.nl` carrying the sIPOPT suffixes made `auto` *decline* the
//! convex fast path outright (issue #196) and pay the general engine's cost for
//! a problem the specialized one solves, because only the general engine could
//! answer the question. Under an explicit `solver_selection=qp-ipm` the request
//! was warned about and dropped. Now an LP or convex QP whose pins the convex
//! arm can express is served where it was solved.
//!
//! # The index space, which is the whole risk here
//!
//! The request arrives in the **`.nl`'s own** indices:
//!
//! * `sens_state_1` — var-int, one slot per original variable;
//! * `sens_state_value_1` — var-real, the perturbed value;
//! * `sens_init_constr` — con-int, which original constraint pins each
//!   parameter.
//!
//! `QpSensitivity::parametric_step` takes indices into the extracted QP's
//! **equality right-hand side `b`**, which is a different space: the extractor
//! splits ranges, drops empty rows, and orders equalities and inequalities into
//! separate blocks. [`qp_extract::ConRowMap`](crate::qp_extract::ConRowMap) is
//! the single source of truth for that map, and this module reads it rather
//! than reconstructing the correspondence — `/sens-review` entry 1, in the
//! space that entry was written about.
//!
//! Two hazards the NLP arm has and this one does not, worth naming so nobody
//! goes looking for them:
//!
//! * **No var-x / full-x split.** The extractor keeps variables 1:1 with the
//!   `.nl` (`qp.n == prob.n`), including fixed ones, so there is no
//!   `lift_x_to_full` and no gh#450 to reproduce. `the_convex_arm_has_no_var_x_split`
//!   asserts that rather than leaving it as a reading of the extractor.
//! * **No presolve row space** — but not for the reason it looks like. The
//!   convex driver postsolves back to the extracted-QP space before anything
//!   downstream runs, so the pins stay valid even with presolve on; that was
//!   measured rather than assumed. Presolve is switched off anyway, because on
//!   the one fixture that exercises it presolve *fixes the parameter the pin
//!   parametrizes* and drops its row, leaving the sensitivity to read a
//!   postsolve reconstruction instead of the converged KKT — four orders of
//!   accuracy on the step, and an unmeasured question about whether the
//!   reconstructed bound multipliers can move the inferred active set. See the
//!   call site in `main.rs` for the numbers.

use pounce_common::types::Number;
use pounce_convex::qp::{QpProblem, QpSolution};
use pounce_convex::sensitivity::QpSensitivity;
use pounce_convex::{QpOptions, QpStatus};
use pounce_linsol::SparseSymLinearSolverInterface;

use crate::nl_reader::NlSuffixes;
use crate::nl_writer::{SolSuffix, SolSuffixTarget, SolSuffixValues};
use crate::qp_extract::ConRowMap;

/// A parametric-step request resolved into the extracted QP's own indices.
#[derive(Debug, Clone, PartialEq)]
pub struct SensPins {
    /// Row of `A` (index into `b`) pinning each parameter.
    pub pin_rows: Vec<usize>,
    /// The parameter's variable index — identical in the `.nl` and the QP.
    pub param_vars: Vec<usize>,
    /// The perturbed value the `.nl` asks for, per parameter.
    pub target: Vec<Number>,
}

/// Why the convex arm cannot express this request. Carrying the reason (rather
/// than an `Option`) is what lets the caller print a message a user can act on
/// — and what keeps "the convex arm declined" distinguishable from "the convex
/// arm answered zero".
#[derive(Debug, Clone, PartialEq)]
pub enum PinRefusal {
    /// A required suffix is absent or the wrong length.
    Suffixes(String),
    /// A parameter has no `sens_state_1` or no `sens_init_constr` tag.
    UntaggedParameter(usize),
    /// The pinning constraint is an inequality (or a range). `parametric_step`
    /// perturbs the equality right-hand side `b`; an inequality lives in
    /// `h`/`G`, which is a different perturbation with a different meaning.
    PinIsNotAnEquality(usize),
    /// The pinning row is not `x_p = value` with a unit coefficient.
    ///
    /// The NLP arm assumes this shape without checking it — `try_compute_sens_step`
    /// passes `signs = vec![1; n_params]` — so a row with any other coefficient
    /// would make the two arms disagree. Refusing here hands such a model back
    /// to the path that has always handled it, rather than introducing a second
    /// answer.
    PinRowIsNotUnit { con: usize, coefficient: Number },
}

impl PinRefusal {
    /// A sentence for the "routing to the general NLP path" note.
    pub fn describe(&self) -> String {
        match self {
            PinRefusal::Suffixes(m) => m.clone(),
            PinRefusal::UntaggedParameter(k) => format!(
                "parameter {} carries no sens_state_1 or sens_init_constr tag",
                k + 1
            ),
            PinRefusal::PinIsNotAnEquality(c) => format!(
                "constraint {} pins a parameter but is an inequality or range, and the \
                 convex parametric step perturbs the equality right-hand side",
                c + 1
            ),
            PinRefusal::PinRowIsNotUnit { con, coefficient } => format!(
                "constraint {} pins a parameter with coefficient {coefficient} rather \
                 than 1, which the sIPOPT suffix convention does not describe",
                con + 1
            ),
        }
    }
}

/// Resolve the `.nl`'s sIPOPT suffixes into pins on the extracted QP.
///
/// Runs **before** the solve, so a refusal can hand the model back to the NLP
/// path with nothing printed and no `.sol` written.
pub fn resolve_pins(
    suffixes: &NlSuffixes,
    con_map: &[ConRowMap],
    qp: &QpProblem,
    n_full: usize,
) -> Result<SensPins, PinRefusal> {
    let missing = |what: &str| PinRefusal::Suffixes(format!("the .nl declares no `{what}` suffix"));
    let sens_state = suffixes
        .var_int
        .get("sens_state_1")
        .ok_or_else(|| missing("sens_state_1"))?;
    let sens_state_value = suffixes
        .var_real
        .get("sens_state_value_1")
        .ok_or_else(|| missing("sens_state_value_1"))?;
    let sens_init_constr = suffixes
        .con_int
        .get("sens_init_constr")
        .ok_or_else(|| missing("sens_init_constr"))?;

    if sens_state.len() != n_full || sens_state_value.len() != n_full {
        return Err(PinRefusal::Suffixes(format!(
            "sens_state_1 / sens_state_value_1 length mismatch (expected {n_full})"
        )));
    }

    let n_params = sens_state.iter().copied().max().unwrap_or(0).max(0) as usize;
    if n_params == 0 {
        return Err(PinRefusal::Suffixes(
            "sens_state_1 tags no parameters".to_string(),
        ));
    }

    let mut param_var: Vec<Option<usize>> = vec![None; n_params];
    for (var_idx, &slot) in sens_state.iter().enumerate() {
        if slot > 0 && (slot as usize) <= n_params {
            param_var[slot as usize - 1] = Some(var_idx);
        }
    }
    let mut param_con: Vec<Option<usize>> = vec![None; n_params];
    for (con_idx, &slot) in sens_init_constr.iter().enumerate() {
        if slot > 0 && (slot as usize) <= n_params {
            param_con[slot as usize - 1] = Some(con_idx);
        }
    }

    let mut pins = SensPins {
        pin_rows: Vec::with_capacity(n_params),
        param_vars: Vec::with_capacity(n_params),
        target: Vec::with_capacity(n_params),
    };
    for k in 0..n_params {
        let (Some(vi), Some(ci)) = (param_var[k], param_con[k]) else {
            return Err(PinRefusal::UntaggedParameter(k));
        };
        // The `.nl` constraint index into the extractor's provenance map. A
        // constraint the extractor dropped has no entry, which is itself a
        // refusal rather than an index to guess at.
        let row = match con_map.get(ci) {
            Some(ConRowMap::Eq { a_row }) => *a_row,
            _ => return Err(PinRefusal::PinIsNotAnEquality(ci)),
        };
        // The suffix convention describes `x_p = p₀`; anything else and the
        // delta below would be in the wrong units. Read the coefficient off the
        // extracted row rather than trusting the shape.
        let coefficient = row_coefficient(qp, row, vi);
        if (coefficient - 1.0).abs() > 1e-12 {
            return Err(PinRefusal::PinRowIsNotUnit {
                con: ci,
                coefficient,
            });
        }
        pins.pin_rows.push(row);
        pins.param_vars.push(vi);
        pins.target.push(sens_state_value[vi]);
    }
    Ok(pins)
}

/// The coefficient of variable `var` in equality row `row` of `A`.
fn row_coefficient(qp: &QpProblem, row: usize, var: usize) -> Number {
    qp.a.iter()
        .filter(|t| t.row == row && t.col == var)
        .map(|t| t.val)
        .sum()
}

/// Take the step and return the perturbed primal, in the `.nl`'s own variable
/// order.
///
/// `None` when the sensitivity could not be built or the solve is not a point
/// to differentiate at — the caller reports it; there is no silent zero.
pub fn perturbed_x<F>(
    qp: &QpProblem,
    sol: &QpSolution,
    opts: &QpOptions,
    pins: &SensPins,
    make_backend: F,
) -> Result<Vec<Number>, String>
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface> + Copy,
{
    if sol.status != QpStatus::Optimal {
        return Err(format!(
            "the solve finished {:?}, so there is no optimum to differentiate at",
            sol.status
        ));
    }
    let mut sens = QpSensitivity::build(qp, sol, opts, 1e-7, make_backend)
        .map_err(|e| format!("could not build the convex sensitivity: {e:?}"))?;
    // `Δp[k] = perturbed − current`, both read off the solved primal. The QP's
    // variables are the `.nl`'s, so `param_vars` indexes `sol.x` directly —
    // this is the step the NLP arm needs `lift_x_to_full` for.
    let deltas: Vec<Number> = pins
        .param_vars
        .iter()
        .zip(&pins.target)
        .map(|(&vi, &t)| t - sol.x[vi])
        .collect();
    let dx = sens.parametric_step(&pins.pin_rows, &deltas);
    if sens.ill_conditioned() {
        return Err(format!(
            "the active-set KKT is too ill-conditioned for the step to be meaningful \
             (condition estimate {:.3e})",
            sens.kkt_cond_estimate()
        ));
    }
    Ok(sol.x.iter().zip(&dx).map(|(a, b)| a + b).collect())
}

/// The `.sol` block the NLP path writes under the same name, so a consumer
/// cannot tell which engine produced it — which is the point.
pub fn sens_suffix(x_pert: Vec<Number>) -> SolSuffix {
    SolSuffix {
        name: "sens_sol_state_1".to_string(),
        target: SolSuffixTarget::Var,
        values: SolSuffixValues::Real(x_pert),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_convex::qp::Triplet;
    use std::collections::BTreeMap;

    /// `min ½‖x‖² s.t. x₀ + x₁ = 1 (row 0), p = 1 (row 1)` with three
    /// variables `(x₀, x₁, p)`. Row 1 is the pin.
    fn qp() -> QpProblem {
        QpProblem {
            n: 3,
            p_lower: (0..3).map(|j| Triplet::new(j, j, 1.0)).collect(),
            c: vec![0.0; 3],
            a: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(1, 2, 1.0),
            ],
            b: vec![1.0, 1.0],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        }
    }

    /// The `.nl` tags variable 2 as parameter 1 and constraint 1 as its pin.
    fn suffixes(state: Vec<i32>, value: Vec<f64>, con: Vec<i32>) -> NlSuffixes {
        let mut s = NlSuffixes::default();
        s.var_int.insert("sens_state_1".into(), state);
        s.var_real.insert("sens_state_value_1".into(), value);
        s.con_int.insert("sens_init_constr".into(), con);
        s
    }

    fn good() -> NlSuffixes {
        suffixes(vec![0, 0, 1], vec![0.0, 0.0, 1.5], vec![0, 1])
    }

    #[test]
    fn a_well_formed_request_resolves_to_the_equality_row() {
        let pins = resolve_pins(
            &good(),
            &[ConRowMap::Eq { a_row: 0 }, ConRowMap::Eq { a_row: 1 }],
            &qp(),
            3,
        )
        .expect("a unit equality pin is expressible");
        assert_eq!(
            pins,
            SensPins {
                pin_rows: vec![1],
                param_vars: vec![2],
                target: vec![1.5],
            }
        );
    }

    /// The whole point of reading `ConRowMap`: the `.nl`'s constraint index and
    /// the QP's equality-row index are **different numbers**, and taking one
    /// for the other returns a neighbouring row's answer — plausible and wrong.
    /// This is `/sens-review` entry 1 in the convex arm's own space.
    #[test]
    fn the_nl_constraint_index_is_not_the_equality_row_index() {
        // Constraint 0 is an inequality, so the pin at constraint 1 lands on
        // equality row **0**, not row 1.
        let con_map = [
            ConRowMap::Ineq {
                upper: Some(0),
                lower: None,
            },
            ConRowMap::Eq { a_row: 0 },
        ];
        let mut q = qp();
        // Row 0 of `A` now carries the pin.
        q.a = vec![Triplet::new(0, 2, 1.0)];
        q.b = vec![1.0];
        let pins = resolve_pins(&good(), &con_map, &q, 3).expect("still expressible");
        assert_eq!(
            pins.pin_rows,
            vec![0],
            "constraint 1 pins equality row 0 here; using the .nl index would perturb a \
             row that does not exist"
        );
    }

    /// `parametric_step` perturbs `b`. An inequality pin lives in `h`/`G`,
    /// which is a different perturbation with a different meaning, so the model
    /// goes back to the path that expresses it.
    #[test]
    fn an_inequality_pin_is_refused() {
        let con_map = [
            ConRowMap::Eq { a_row: 0 },
            ConRowMap::Ineq {
                upper: Some(0),
                lower: None,
            },
        ];
        assert_eq!(
            resolve_pins(&good(), &con_map, &qp(), 3),
            Err(PinRefusal::PinIsNotAnEquality(1))
        );
    }

    /// The sIPOPT suffix convention describes `x_p = p₀`. The NLP arm assumes
    /// that shape without checking (`signs = vec![1; n_params]`), so a row with
    /// another coefficient is where the two arms would disagree — and the point
    /// of the check is that they do not.
    #[test]
    fn a_non_unit_pin_row_is_refused() {
        let mut q = qp();
        q.a = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(1, 2, -1.0),
        ];
        assert_eq!(
            resolve_pins(
                &good(),
                &[ConRowMap::Eq { a_row: 0 }, ConRowMap::Eq { a_row: 1 }],
                &q,
                3
            ),
            Err(PinRefusal::PinRowIsNotUnit {
                con: 1,
                coefficient: -1.0
            })
        );
    }

    #[test]
    fn a_parameter_with_no_pinning_constraint_is_refused() {
        let s = suffixes(vec![0, 0, 1], vec![0.0, 0.0, 1.5], vec![0, 0]);
        assert_eq!(
            resolve_pins(
                &s,
                &[ConRowMap::Eq { a_row: 0 }, ConRowMap::Eq { a_row: 1 }],
                &qp(),
                3
            ),
            Err(PinRefusal::UntaggedParameter(0))
        );
    }

    #[test]
    fn a_length_mismatch_is_refused_rather_than_indexed_past() {
        let s = suffixes(vec![0, 1], vec![0.0, 1.5], vec![0, 1]);
        assert!(matches!(
            resolve_pins(
                &s,
                &[ConRowMap::Eq { a_row: 0 }, ConRowMap::Eq { a_row: 1 }],
                &qp(),
                3
            ),
            Err(PinRefusal::Suffixes(_))
        ));
    }

    /// The hazard the NLP arm has and this one does not, asserted rather than
    /// left as a reading of the extractor: `resolve_pins` indexes `sol.x` by
    /// the `.nl`'s own variable number, which is only sound because the convex
    /// extractor keeps variables 1:1 — no `lift_x_to_full`, so no gh#450 to
    /// reproduce here.
    #[test]
    fn the_convex_arm_has_no_var_x_split() {
        let q = qp();
        assert_eq!(
            q.n, 3,
            "the extracted QP keeps every .nl variable, fixed ones included"
        );
        let pins = resolve_pins(
            &good(),
            &[ConRowMap::Eq { a_row: 0 }, ConRowMap::Eq { a_row: 1 }],
            &q,
            q.n,
        )
        .unwrap();
        assert!(
            pins.param_vars.iter().all(|&v| v < q.n),
            "parameter variable indices are QP indices and .nl indices at once"
        );
    }

    #[test]
    fn every_refusal_describes_itself_without_panicking() {
        let cases = [
            PinRefusal::Suffixes("x".into()),
            PinRefusal::UntaggedParameter(0),
            PinRefusal::PinIsNotAnEquality(1),
            PinRefusal::PinRowIsNotUnit {
                con: 1,
                coefficient: -1.0,
            },
        ];
        for c in cases {
            assert!(!c.describe().is_empty());
        }
        let _ = BTreeMap::<String, u8>::new();
    }
}

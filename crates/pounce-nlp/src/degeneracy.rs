//! Non-finite and structural-degeneracy diagnostics for a point and for the
//! constraint Jacobian evaluated there.
//!
//! Two questions this module answers, both about a *specific point* rather
//! than about the problem:
//!
//! 1. **Which value is not finite?** [`audit_point`] scans `x`, `f`, `∇f`,
//!    `g` and the Jacobian values and names the variable, row or nonzero that
//!    carries the `NaN` / `±inf`. `Invalid_Number_Detected` on its own tells a
//!    user that *something* was not a number; it does not tell them whether
//!    their starting point contained a `NaN`, or their objective divided by
//!    zero at it. Those have different fixes.
//!
//! 2. **Is the constraint Jacobian rank-deficient here?**
//!    [`jacobian_degeneracy`] reports rows and columns that are structurally
//!    present but numerically zero. A zero row is a constraint with no local
//!    first-order handle: no step improves it to first order, which is exactly
//!    the configuration in which a filter line-search IPM converges to a
//!    stationary point of the constraint violation and reports local
//!    infeasibility on a problem that is perfectly feasible.
//!
//! Neither function decides anything. They produce findings for the caller to
//! print alongside a verdict it has already reached, so nothing here can move
//! a trajectory.
//!
//! # Why zero rows and columns, rather than a rank estimate
//!
//! A genuine numerical rank needs an SVD or a rank-revealing QR of the whole
//! Jacobian, which is not affordable to run speculatively at the end of every
//! failed solve on a model of any size. Zero rows and columns are `O(nnz)`,
//! they need no factorization, and they catch the cases that motivated this:
//!
//! * The squared-slack reformulation `g(x) ≤ 0  ↦  g(x) + s² = 0` has
//!   `∂(s²)/∂s = 0` at `s = 0`, so **every** active row's slack column is zero
//!   exactly on the active set — the point the solve is trying to reach.
//! * A start at the origin of a model whose constraints are homogeneous
//!   quadratics (`x₁² + x₂² = 25`, `x₁x₂ = 9`) has an identically zero
//!   Jacobian, every row and every column.
//!
//! Structural absence is *not* reported. A variable that appears in no
//! constraint has an empty column and that is ordinary — an unconstrained
//! variable is not a degeneracy. Only a column the model declared and then
//! evaluated to zero is a finding.

use pounce_common::types::{Index, Number};

/// Which quantity a non-finite value turned up in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantity {
    /// An entry of the starting point `x`.
    StartingPoint,
    /// The objective value `f(x)`.
    Objective,
    /// An entry of `∇f(x)`.
    ObjectiveGradient,
    /// An entry of `g(x)`.
    Constraint,
    /// A nonzero of `∇g(x)`, reported by its `(row, column)`.
    Jacobian,
}

impl Quantity {
    /// How this quantity is named in a diagnostic line.
    pub fn label(self) -> &'static str {
        match self {
            Self::StartingPoint => "starting point x",
            Self::Objective => "objective f(x)",
            Self::ObjectiveGradient => "objective gradient grad f(x)",
            Self::Constraint => "constraint value g(x)",
            Self::Jacobian => "constraint Jacobian",
        }
    }
}

/// One non-finite value, located.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonFinite {
    pub quantity: Quantity,
    /// Zero-based index: the variable for `x` / `∇f`, the row for `g`, the
    /// Jacobian row for a Jacobian nonzero. Always `0` for the scalar `f`.
    pub index: usize,
    /// The Jacobian column, for [`Quantity::Jacobian`] only.
    pub column: Option<usize>,
    pub value: Number,
}

impl NonFinite {
    /// A one-line human-readable rendering, e.g.
    /// `starting point x[3] = NaN`.
    pub fn describe(&self) -> String {
        let v = if self.value.is_nan() {
            "NaN".to_string()
        } else if self.value > 0.0 {
            "+inf".to_string()
        } else {
            "-inf".to_string()
        };
        match (self.quantity, self.column) {
            (Quantity::Objective, _) => format!("{} = {v}", self.quantity.label()),
            (Quantity::Jacobian, Some(col)) => format!(
                "{}[row {}, column {}] = {v}",
                self.quantity.label(),
                self.index,
                col
            ),
            _ => format!("{}[{}] = {v}", self.quantity.label(), self.index),
        }
    }
}

/// Findings from [`audit_point`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PointAudit {
    /// Located non-finite values, in scan order, capped at the caller's limit.
    pub non_finite: Vec<NonFinite>,
    /// How many further non-finite values were found past the cap. The count
    /// is exact — a capped report still tells the truth about the scale.
    pub suppressed: usize,
}

impl PointAudit {
    /// Nothing wrong here.
    pub fn is_clean(&self) -> bool {
        self.non_finite.is_empty() && self.suppressed == 0
    }

    /// Total non-finite values found, including those past the cap.
    pub fn total(&self) -> usize {
        self.non_finite.len() + self.suppressed
    }

    /// A one-line report, or `None` when clean.
    ///
    /// A *fragment*, not a message: no leading indent, no trailing newline, no
    /// terminating period, so a caller can set it inside a sentence of its own.
    /// It was multi-line and indented until the first caller wanted it mid-
    /// sentence and got a report that stopped at the first newline in every
    /// line-oriented log reader downstream.
    pub fn describe(&self) -> Option<String> {
        if self.is_clean() {
            return None;
        }
        let mut parts: Vec<String> = self.non_finite.iter().map(|f| f.describe()).collect();
        if self.suppressed > 0 {
            parts.push(format!("and {} more", self.suppressed));
        }
        Some(parts.join("; "))
    }
}

/// The values of one point, as far as they were evaluated.
///
/// Every field is optional because the audit has to work at a point where
/// evaluation *failed*: if `eval_f` returned `None` there is no `f` to scan,
/// and the audit still has to be able to report the `NaN` in `x` that caused
/// it.
#[derive(Debug, Clone, Copy, Default)]
pub struct PointValues<'a> {
    pub x: Option<&'a [Number]>,
    pub f: Option<Number>,
    pub grad_f: Option<&'a [Number]>,
    pub g: Option<&'a [Number]>,
    /// Jacobian nonzeros as `(values, rows, cols)`, zero-based.
    pub jac: Option<(&'a [Number], &'a [Index], &'a [Index])>,
}

/// Locate every non-finite value in `values`, up to `limit` reported findings.
///
/// Scan order is `x`, `f`, `∇f`, `g`, Jacobian — cheapest and most likely
/// first, so a truncated report leads with the input the user controls.
pub fn audit_point(values: PointValues<'_>, limit: usize) -> PointAudit {
    let mut audit = PointAudit::default();
    let mut push = |quantity: Quantity, index: usize, column: Option<usize>, value: Number| {
        if audit.non_finite.len() < limit {
            audit.non_finite.push(NonFinite {
                quantity,
                index,
                column,
                value,
            });
        } else {
            audit.suppressed += 1;
        }
    };

    if let Some(x) = values.x {
        for (i, v) in x.iter().enumerate() {
            if !v.is_finite() {
                push(Quantity::StartingPoint, i, None, *v);
            }
        }
    }
    if let Some(f) = values.f
        && !f.is_finite()
    {
        push(Quantity::Objective, 0, None, f);
    }
    if let Some(grad) = values.grad_f {
        for (i, v) in grad.iter().enumerate() {
            if !v.is_finite() {
                push(Quantity::ObjectiveGradient, i, None, *v);
            }
        }
    }
    if let Some(g) = values.g {
        for (i, v) in g.iter().enumerate() {
            if !v.is_finite() {
                push(Quantity::Constraint, i, None, *v);
            }
        }
    }
    if let Some((vals, rows, cols)) = values.jac {
        for (k, v) in vals.iter().enumerate() {
            if !v.is_finite() {
                let row = rows.get(k).copied().unwrap_or(0).max(0) as usize;
                let col = cols.get(k).copied().unwrap_or(0).max(0) as usize;
                push(Quantity::Jacobian, row, Some(col), *v);
            }
        }
    }
    audit
}

/// Rows and columns that the model declared and then evaluated to zero.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JacobianDegeneracy {
    /// Constraint rows with at least one structural nonzero, all of whose
    /// values are at or below the tolerance.
    pub zero_rows: Vec<usize>,
    /// Variable columns with at least one structural nonzero, all of whose
    /// values are at or below the tolerance.
    pub zero_cols: Vec<usize>,
    /// Rows / columns that have any structural entry at all. The denominators
    /// for the counts above — "3 of 5 rows" means something, "3 rows" does not.
    pub structural_rows: usize,
    pub structural_cols: usize,
}

impl JacobianDegeneracy {
    /// Did anything degenerate turn up?
    pub fn is_degenerate(&self) -> bool {
        !self.zero_rows.is_empty() || !self.zero_cols.is_empty()
    }

    /// A short report naming up to `limit` indices per category, or `None`
    /// when the Jacobian is structurally sound at this point.
    ///
    /// A *fragment*, like [`PointAudit::describe`]: single line, no indent, no
    /// trailing newline or period.
    pub fn describe(&self, limit: usize) -> Option<String> {
        if !self.is_degenerate() {
            return None;
        }
        fn list(v: &[usize], limit: usize) -> String {
            let shown: Vec<String> = v.iter().take(limit).map(|i| i.to_string()).collect();
            if v.len() > limit {
                format!("{}, … ({} total)", shown.join(", "), v.len())
            } else {
                shown.join(", ")
            }
        }
        let mut parts: Vec<String> = Vec::new();
        if !self.zero_rows.is_empty() {
            parts.push(format!(
                "{} of {} constraint rows have an identically zero gradient here (rows {})",
                self.zero_rows.len(),
                self.structural_rows,
                list(&self.zero_rows, limit)
            ));
        }
        if !self.zero_cols.is_empty() {
            parts.push(format!(
                "{} of {} variable columns are identically zero here (variables {})",
                self.zero_cols.len(),
                self.structural_cols,
                list(&self.zero_cols, limit)
            ));
        }
        Some(parts.join("; "))
    }
}

/// Find structurally-present, numerically-zero rows and columns of a sparse
/// Jacobian given in triplet form.
///
/// `rows` and `cols` are zero-based and parallel to `values`. Entries whose
/// index falls outside `m` / `n` are skipped rather than panicking: this runs
/// on the failure path, where the last thing a user needs is the diagnostic
/// itself aborting.
///
/// `tol` is an absolute threshold. Zero is the meaningful default — the cases
/// this exists for produce exact zeros, and a loose tolerance would start
/// reporting merely small gradients as degenerate.
pub fn jacobian_degeneracy(
    m: usize,
    n: usize,
    values: &[Number],
    rows: &[Index],
    cols: &[Index],
    tol: Number,
) -> JacobianDegeneracy {
    let mut row_present = vec![false; m];
    let mut col_present = vec![false; n];
    let mut row_nonzero = vec![false; m];
    let mut col_nonzero = vec![false; n];

    for (k, v) in values.iter().enumerate() {
        let (Some(&r), Some(&c)) = (rows.get(k), cols.get(k)) else {
            continue;
        };
        if r < 0 || c < 0 {
            continue;
        }
        let (r, c) = (r as usize, c as usize);
        if r >= m || c >= n {
            continue;
        }
        row_present[r] = true;
        col_present[c] = true;
        // A non-finite entry is not a zero. It is a different failure, which
        // `audit_point` reports; treating it as nonzero here keeps the two
        // diagnostics from contradicting each other on the same matrix.
        if !v.is_finite() || v.abs() > tol {
            row_nonzero[r] = true;
            col_nonzero[c] = true;
        }
    }

    JacobianDegeneracy {
        zero_rows: (0..m)
            .filter(|&i| row_present[i] && !row_nonzero[i])
            .collect(),
        zero_cols: (0..n)
            .filter(|&j| col_present[j] && !col_nonzero[j])
            .collect(),
        structural_rows: row_present.iter().filter(|p| **p).count(),
        structural_cols: col_present.iter().filter(|p| **p).count(),
    }
}

// ---------------------------------------------------------------------------
// Driving the audit through a TNLP
// ---------------------------------------------------------------------------

/// What a failing solve's starting point looks like: the non-finite audit and,
/// when the model has constraints, the Jacobian's structural degeneracy there.
#[derive(Debug, Clone)]
pub struct StartPointDiagnosis {
    pub audit: PointAudit,
    /// `None` for an unconstrained model, or when the model refused to
    /// evaluate its Jacobian at the point — there is nothing to say either way.
    pub jacobian: Option<JacobianDegeneracy>,
}

impl StartPointDiagnosis {
    /// Nothing worth telling the user about.
    pub fn is_clean(&self) -> bool {
        self.audit.is_clean() && !self.jacobian.as_ref().is_some_and(|j| j.is_degenerate())
    }
}

/// Evaluate a model at its own starting point and report what is wrong there.
///
/// Runs on the **failure path only** — one objective, one gradient, one
/// constraint and one Jacobian evaluation, spent to explain a solve that has
/// already failed. Nothing here changes a status, a trajectory, or an option;
/// the caller reports it and moves on.
///
/// Call this on the **user's** TNLP, before any presolve or scaling wrapper.
/// The report names variables and rows by index, and a wrapper renumbers both:
/// naming `x[3]` of a presolved model to a user reading their own file points
/// at a neighbouring variable's answer — plausible and wrong, which is the
/// gh#450 failure mode and not one worth reproducing in a diagnostic.
///
/// Returns `None` if the model will not describe itself (`get_nlp_info` or
/// `get_starting_point` refused), because then there is no point to audit.
pub fn diagnose_start_point(
    tnlp: &std::rc::Rc<std::cell::RefCell<dyn crate::tnlp::TNLP>>,
    limit: usize,
) -> Option<StartPointDiagnosis> {
    use crate::tnlp::{IndexStyle, SparsityRequest, StartingPoint};

    let info = tnlp.borrow_mut().get_nlp_info()?;
    let n = info.n.max(0) as usize;
    let m = info.m.max(0) as usize;
    let nnz = info.nnz_jac_g.max(0) as usize;

    let mut x = vec![0.0; n];
    let (mut z_l, mut z_u, mut lambda) = (vec![0.0; n], vec![0.0; n], vec![0.0; m]);
    if !tnlp.borrow_mut().get_starting_point(StartingPoint {
        init_x: true,
        x: &mut x,
        init_z: false,
        z_l: &mut z_l,
        z_u: &mut z_u,
        init_lambda: false,
        lambda: &mut lambda,
    }) {
        return None;
    }

    // Everything downstream of `x` is evaluated AT `x`, so a model that
    // cannot be evaluated there will decline — or produce garbage — for
    // reasons the audit of `x` itself already explains. Ask anyway and record
    // only what comes back: an evaluator that refuses contributes nothing
    // rather than a fabricated finding.
    let f = tnlp.borrow_mut().eval_f(&x, true);
    let mut grad_f = vec![0.0; n];
    let have_grad = tnlp.borrow_mut().eval_grad_f(&x, false, &mut grad_f);
    let mut g = vec![0.0; m];
    let have_g = m > 0 && tnlp.borrow_mut().eval_g(&x, false, &mut g);

    let (mut rows, mut cols, mut vals) =
        (vec![0 as Index; nnz], vec![0 as Index; nnz], vec![0.0; nnz]);
    let have_jac = m > 0
        && tnlp.borrow_mut().eval_jac_g(
            None,
            false,
            SparsityRequest::Structure {
                irow: &mut rows,
                jcol: &mut cols,
            },
        )
        && tnlp.borrow_mut().eval_jac_g(
            Some(&x),
            false,
            SparsityRequest::Values { values: &mut vals },
        );
    if have_jac && info.index_style == IndexStyle::Fortran {
        // `audit_point` and `jacobian_degeneracy` both index zero-based.
        for idx in rows.iter_mut().chain(cols.iter_mut()) {
            *idx -= 1;
        }
    }

    let audit = audit_point(
        PointValues {
            x: Some(&x),
            f,
            grad_f: have_grad.then_some(&grad_f[..]),
            g: have_g.then_some(&g[..]),
            jac: have_jac.then_some((&vals[..], &rows[..], &cols[..])),
        },
        limit,
    );
    let jacobian = have_jac.then(|| jacobian_degeneracy(m, n, &vals, &rows, &cols, 0.0));
    Some(StartPointDiagnosis { audit, jacobian })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_the_variable_carrying_a_nan_in_the_starting_point() {
        // `hong`'s bundled start is literally [NaN, NaN, NaN, NaN, 0, 0, …].
        // The whole point of the audit is that the user is told "x[0]", not
        // just that some number somewhere was invalid.
        let x = [f64::NAN, 1.0, f64::NAN, 0.0];
        let audit = audit_point(
            PointValues {
                x: Some(&x),
                ..Default::default()
            },
            8,
        );
        assert_eq!(audit.total(), 2);
        assert_eq!(audit.non_finite[0].index, 0);
        assert_eq!(audit.non_finite[1].index, 2);
        assert_eq!(audit.non_finite[0].quantity, Quantity::StartingPoint);
        assert_eq!(audit.non_finite[0].describe(), "starting point x[0] = NaN");
    }

    #[test]
    fn distinguishes_a_bad_start_from_a_bad_objective_at_a_good_start() {
        // Same status upstream, different fix: one is the user's data, the
        // other is their model evaluating to 0/0 at a legitimate point.
        let x = [0.0, 0.0];
        let bad_obj = audit_point(
            PointValues {
                x: Some(&x),
                f: Some(f64::NAN),
                ..Default::default()
            },
            8,
        );
        assert_eq!(bad_obj.non_finite.len(), 1);
        assert_eq!(bad_obj.non_finite[0].quantity, Quantity::Objective);
        assert_eq!(bad_obj.non_finite[0].describe(), "objective f(x) = NaN");
    }

    #[test]
    fn reports_infinities_with_their_sign() {
        let g = [f64::INFINITY, f64::NEG_INFINITY];
        let audit = audit_point(
            PointValues {
                g: Some(&g),
                ..Default::default()
            },
            8,
        );
        assert!(audit.non_finite[0].describe().ends_with("= +inf"));
        assert!(audit.non_finite[1].describe().ends_with("= -inf"));
    }

    #[test]
    fn locates_a_jacobian_nonzero_by_row_and_column() {
        let vals = [1.0, f64::NAN];
        let rows = [0, 1];
        let cols = [0, 2];
        let audit = audit_point(
            PointValues {
                jac: Some((&vals, &rows, &cols)),
                ..Default::default()
            },
            8,
        );
        assert_eq!(audit.non_finite.len(), 1);
        assert_eq!(audit.non_finite[0].index, 1);
        assert_eq!(audit.non_finite[0].column, Some(2));
        assert_eq!(
            audit.non_finite[0].describe(),
            "constraint Jacobian[row 1, column 2] = NaN"
        );
    }

    #[test]
    fn caps_the_report_but_counts_the_rest_exactly() {
        // A truncated report must not understate the scale of the problem:
        // "2 shown, 1198 more" and "2 found" call for different responses.
        let x = vec![f64::NAN; 1200];
        let audit = audit_point(
            PointValues {
                x: Some(&x),
                ..Default::default()
            },
            2,
        );
        assert_eq!(audit.non_finite.len(), 2);
        assert_eq!(audit.suppressed, 1198);
        assert_eq!(audit.total(), 1200);
        assert!(audit.describe().unwrap().contains("and 1198 more"));
    }

    #[test]
    fn a_finite_point_is_clean_and_describes_as_nothing() {
        let x = [1.0, 2.0];
        let grad = [0.5, -0.5];
        let audit = audit_point(
            PointValues {
                x: Some(&x),
                f: Some(3.0),
                grad_f: Some(&grad),
                ..Default::default()
            },
            8,
        );
        assert!(audit.is_clean());
        assert_eq!(audit.describe(), None);
    }

    #[test]
    fn an_identically_zero_jacobian_reports_every_row_and_column() {
        // HS008 (x₁²+x₂²=25, x₁x₂=9) started at the origin: the Jacobian is
        // [[2x₁, 2x₂], [x₂, x₁]] = 0. POUNCE calls this locally infeasible;
        // the problem has four solutions.
        let vals = [0.0, 0.0, 0.0, 0.0];
        let rows = [0, 0, 1, 1];
        let cols = [0, 1, 0, 1];
        let d = jacobian_degeneracy(2, 2, &vals, &rows, &cols, 0.0);
        assert!(d.is_degenerate());
        assert_eq!(d.zero_rows, vec![0, 1]);
        assert_eq!(d.zero_cols, vec![0, 1]);
        assert_eq!(d.structural_rows, 2);
        assert_eq!(d.structural_cols, 2);
    }

    #[test]
    fn a_squared_slack_at_zero_shows_up_as_a_zero_column() {
        // g(x) + s² = 0 with s = 0: the row still has a handle through x, so
        // it is not a zero row — but the slack's column is exactly zero, which
        // is the rank deficiency that costs the solve.
        //   row 0: d/dx = 1.0 (col 0), d/ds = 2s = 0 (col 1)
        let vals = [1.0, 0.0];
        let rows = [0, 0];
        let cols = [0, 1];
        let d = jacobian_degeneracy(1, 2, &vals, &rows, &cols, 0.0);
        assert!(d.is_degenerate());
        assert!(d.zero_rows.is_empty());
        assert_eq!(d.zero_cols, vec![1]);
    }

    #[test]
    fn a_structurally_absent_column_is_not_a_finding() {
        // Variable 2 appears in no constraint. That is an ordinary
        // unconstrained variable, not a degeneracy, and reporting it would
        // fire on most models in the corpus.
        let vals = [1.0, 1.0];
        let rows = [0, 1];
        let cols = [0, 1];
        let d = jacobian_degeneracy(2, 3, &vals, &rows, &cols, 0.0);
        assert!(!d.is_degenerate());
        assert_eq!(d.structural_cols, 2);
    }

    #[test]
    fn a_non_finite_entry_is_not_counted_as_a_zero() {
        // Otherwise the two diagnostics contradict each other on one matrix:
        // `audit_point` calls the entry NaN while this one calls its row zero.
        let vals = [f64::NAN];
        let rows = [0];
        let cols = [0];
        let d = jacobian_degeneracy(1, 1, &vals, &rows, &cols, 0.0);
        assert!(!d.is_degenerate());
    }

    #[test]
    fn out_of_range_and_negative_indices_are_skipped_not_panicked_on() {
        // This runs on the failure path. A diagnostic that panics turns a
        // reportable verdict into a crash.
        let vals = [1.0, 2.0, 3.0];
        let rows = [0, 9, -1];
        let cols = [0, 0, 0];
        let d = jacobian_degeneracy(1, 1, &vals, &rows, &cols, 0.0);
        assert!(!d.is_degenerate());
        assert_eq!(d.structural_rows, 1);
    }

    #[test]
    fn the_tolerance_is_absolute_and_zero_by_default_semantics() {
        let vals = [1e-14];
        let rows = [0];
        let cols = [0];
        assert!(!jacobian_degeneracy(1, 1, &vals, &rows, &cols, 0.0).is_degenerate());
        assert!(jacobian_degeneracy(1, 1, &vals, &rows, &cols, 1e-12).is_degenerate());
    }

    #[test]
    fn the_row_and_column_lists_are_truncated_with_a_total() {
        let m = 50;
        let vals = vec![0.0; m];
        let rows: Vec<Index> = (0..m as Index).collect();
        let cols = vec![0; m];
        let d = jacobian_degeneracy(m, 1, &vals, &rows, &cols, 0.0);
        let text = d.describe(3).unwrap();
        assert!(text.contains("(50 total)"), "{text}");
        assert!(text.contains("50 of 50 constraint rows"), "{text}");
    }
}

//! Restoration-phase iteration output — port of
//! `Algorithm/IpRestoIterationOutput.{hpp,cpp}`.
//!
//! Differences from `OrigIterationOutput` (`pounce-algorithm/src/output/orig.rs`):
//!
//! * the iteration counter is followed by a literal `r` (not a space) so
//!   restoration rows are visually distinct in a unified log;
//! * the objective column reports the **original** NLP's unscaled
//!   trial objective at the resto-trial point (after copying the
//!   `orig_x` slice of the resto iterate into the orig trial), not the
//!   resto problem's own `f`;
//! * the `inf_pr` column reports either the resto problem's primal
//!   infeasibility (`InfPrTag::Internal`) or the original NLP's
//!   unscaled constraint violation at the trial (`InfPrTag::Original`);
//! * `inf_du`, `mu`, `‖d‖`, `lg(rg)`, `alpha_*`, and the LS count come
//!   from the **resto** `IpoptData`/CQ exactly as in the orig formatter.
//!
//! The plumbing that supplies the orig-NLP numbers depends on
//! `RestoIpoptNlp::orig_ip_nlp/orig_ip_cq` being wired up. Until then,
//! [`RestoIterationOutput::format_row_explicit`] takes the relevant
//! numbers as parameters so this module can be unit-tested standalone.

/// Mirrors `IpOrigIterationOutput.hpp:InfPrOutput` for the resto
/// formatter. The `inf_pr` column either reports the resto problem's
/// internal primal infeasibility or the original NLP's unscaled
/// constraint violation at the trial point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InfPrTag {
    Internal,
    Original,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrintInfoString {
    Yes,
    No,
}

pub struct RestoIterationOutput {
    pub print_info_string: PrintInfoString,
    pub inf_pr_output: InfPrTag,
    pub print_frequency_iter: i32,
    pub print_frequency_time: f64,
}

impl Default for RestoIterationOutput {
    fn default() -> Self {
        // Defaults mirror upstream `RegisterOptions` (inherited from
        // `OrigIterationOutput`).
        Self {
            print_info_string: PrintInfoString::No,
            inf_pr_output: InfPrTag::Original,
            print_frequency_iter: 1,
            print_frequency_time: 0.0,
        }
    }
}

impl RestoIterationOutput {
    pub fn new() -> Self {
        Self::default()
    }

    /// Header line — identical literal to the orig formatter
    /// (`IpRestoIterationOutput.cpp:71`).
    pub const HEADER: &'static str =
        "iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls\n";

    /// Build the single-line restoration iteration row, mirroring the
    /// `Snprintf` block at `IpRestoIterationOutput.cpp:156`:
    /// `"%4d r%14.7e %7.2e %7.2e %5.1f %7.2e %5s %7.2e %7.2e%c%3d"`.
    ///
    /// `iter` and `f` come from the resto-side `IpData` and the **orig**
    /// CQ respectively (orig `unscaled_trial_f` after the orig trial
    /// has been set to the resto-trial's `x_only`/`s_only` slices).
    /// `inf_pr` is whichever of the two infeasibilities the
    /// `inf_pr_output` setting selects (the caller has already chosen).
    /// `inf_du`, `mu`, `dnrm`, `regu_x`, `alpha_*`, `alpha_char`, and
    /// `ls_count` are all resto-side.
    #[allow(clippy::too_many_arguments)]
    pub fn format_row_explicit(
        &self,
        iter: i32,
        f: f64,
        inf_pr: f64,
        inf_du: f64,
        mu: f64,
        dnrm: f64,
        regu_x: f64,
        alpha_dual: f64,
        alpha_primal: f64,
        alpha_char: char,
        ls_count: i32,
    ) -> String {
        let lg_mu = mu.log10();
        let regu_str: String = if regu_x == 0.0 {
            "   - ".to_string()
        } else {
            format!("{:5.1}", regu_x.log10())
        };
        format!(
            "{:>4}r{:14.7e} {:7.2e} {:7.2e} {:5.1} {:7.2e} {:>5} {:7.2e} {:7.2e}{}{:>3}",
            iter,
            f,
            inf_pr,
            inf_du,
            lg_mu,
            dnrm,
            regu_str,
            alpha_dual,
            alpha_primal,
            alpha_char,
            ls_count,
        )
    }
}

/// Adapter wiring [`RestoIterationOutput`] into the algorithm-side
/// [`pounce_algorithm::output::r#trait::IterationOutput`] trait.
///
/// Pulls every numeric column from the inner-IPM data / CQ exactly as
/// `OrigIterationOutput::format_row` does, then routes through
/// [`RestoIterationOutput::format_row_explicit`] so the row gets the
/// literal `r` suffix on the iter index.
///
/// When `orig_nlp` is wired ([`Self::with_orig_nlp`]), the `inf_pr`
/// and `objective` columns report the **original** NLP's unscaled
/// constraint violation (`max(||c(x_orig)||∞, ||d(x_orig) − s||∞)`)
/// and unscaled objective at `x_orig` — matching upstream
/// `IpRestoIterationOutput.cpp:106-156`. Without it, both fall back
/// to the resto NLP's own values (the original v0.1 behavior).
pub struct RestoIterationOutputAdapter {
    pub inner: RestoIterationOutput,
    orig_nlp: Option<std::rc::Rc<std::cell::RefCell<dyn pounce_nlp::ipopt_nlp::IpoptNlp>>>,
}

impl RestoIterationOutputAdapter {
    pub fn new() -> Self {
        Self {
            inner: RestoIterationOutput::new(),
            orig_nlp: None,
        }
    }

    /// Wire the original NLP so `format_row` can report orig-NLP
    /// `inf_pr` / objective on resto rows. Mirrors upstream's
    /// `RestoIterationOutput::WriteOutput` reading `orig_ip_cq`.
    pub fn with_orig_nlp(
        mut self,
        orig: std::rc::Rc<std::cell::RefCell<dyn pounce_nlp::ipopt_nlp::IpoptNlp>>,
    ) -> Self {
        self.orig_nlp = Some(orig);
        self
    }
}

impl Default for RestoIterationOutputAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl pounce_algorithm::output::r#trait::IterationOutput for RestoIterationOutputAdapter {
    fn format_row(
        &mut self,
        data: &pounce_algorithm::ipopt_data::IpoptDataHandle,
        cq: &pounce_algorithm::ipopt_cq::IpoptCqHandle,
    ) -> String {
        let d = data.borrow();
        let c = cq.borrow();

        let iter = d.iter_count;
        // Resto-NLP fallbacks; overridden below when `orig_nlp` is
        // wired so we report orig-NLP `f` and `inf_pr` at the
        // `(x_orig, s)` slice of the resto iterate (upstream
        // `IpRestoIterationOutput.cpp:106-156`).
        let mut unscaled_f = c.curr_f();
        let mut inf_pr = c.curr_primal_infeasibility_max();
        if let Some(orig_rc) = &self.orig_nlp {
            if let Some(curr) = d.curr.clone() {
                if let Some((f_orig, viol_orig)) =
                    eval_orig_at_inner_curr(&curr, orig_rc)
                {
                    unscaled_f = f_orig;
                    inf_pr = viol_orig;
                }
            }
        }
        let inf_du = c.curr_dual_infeasibility_max();
        let mu = d.curr_mu;

        let dnrm = match &d.delta {
            Some(delta) => delta.x.amax().max(delta.s.amax()),
            None => 0.0,
        };

        let regu_x = d.info_regu_x;
        let alpha_dual = d.info_alpha_dual;
        let alpha_primal = d.info_alpha_primal;
        let alpha_char = d.info_alpha_primal_char;
        let ls_count = d.info_ls_count;

        self.inner.format_row_explicit(
            iter,
            unscaled_f,
            inf_pr,
            inf_du,
            mu,
            dnrm,
            regu_x,
            alpha_dual,
            alpha_primal,
            alpha_char,
            ls_count,
        )
    }
}

/// Evaluate the original NLP's unscaled `f(x_orig)` and constraint
/// violation `max(||c(x_orig)||∞, ||d(x_orig) − s||∞)` at the resto
/// iterate's `x_orig` slice (block 0 of the compound `x`) and the
/// inner `s` vector. Returns `None` if any of the downcasts /
/// dimension reads fail (in which case the caller falls back to
/// resto-NLP values).
fn eval_orig_at_inner_curr(
    curr: &pounce_algorithm::iterates_vector::IteratesVector,
    orig_rc: &std::rc::Rc<std::cell::RefCell<dyn pounce_nlp::ipopt_nlp::IpoptNlp>>,
) -> Option<(f64, f64)> {
    use pounce_linalg::dense_vector::DenseVectorSpace;
    use pounce_linalg::{CompoundVector, Vector};

    let xc = curr.x.as_any().downcast_ref::<CompoundVector>()?;
    let x_orig = xc.comp(crate::resto_nlp::BLOCK_X);
    let s_inner = &*curr.s;

    let mut orig = orig_rc.borrow_mut();
    let m_eq = orig.m_eq();
    let m_ineq = orig.m_ineq();

    // c(x_orig)
    let c_amax = if m_eq > 0 {
        let mut c_buf = DenseVectorSpace::new(m_eq).make_new_dense();
        orig.eval_c(x_orig, &mut c_buf);
        c_buf.amax()
    } else {
        0.0
    };

    // d(x_orig) − s
    let d_minus_s_amax = if m_ineq > 0 {
        let mut d_buf = DenseVectorSpace::new(m_ineq).make_new_dense();
        orig.eval_d(x_orig, &mut d_buf);
        d_buf.axpy(-1.0, s_inner);
        d_buf.amax()
    } else {
        0.0
    };

    let f = orig.eval_f(x_orig);
    Some((f, c_amax.max(d_minus_s_amax)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_matches_upstream_literal() {
        assert_eq!(
            RestoIterationOutput::HEADER,
            "iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls\n"
        );
    }

    #[test]
    fn row_has_r_suffix_after_iter_field() {
        let out = RestoIterationOutput::new();
        let row = out.format_row_explicit(
            7, 1.234567e+0, 1e-3, 1e-4, 0.1, 1e-2, 0.0, 1.0, 0.5, 'f', 2,
        );
        // Iter is right-justified in width 4 then the literal 'r', so
        // `"   7r"` is the prefix.
        assert!(row.starts_with("   7r"), "row = {row:?}");
    }

    #[test]
    fn regu_field_dashes_when_zero() {
        let out = RestoIterationOutput::new();
        let row = out.format_row_explicit(
            0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, ' ', 0,
        );
        // Look for the exact "   - " regu segment in the row.
        assert!(row.contains("   - "), "row = {row:?}");
    }

    #[test]
    fn regu_field_logs_value_when_nonzero() {
        let out = RestoIterationOutput::new();
        let row = out.format_row_explicit(
            0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e-3, 1.0, 1.0, ' ', 0,
        );
        // log10(1e-3) = -3.0 → " -3.0".
        assert!(row.contains(" -3.0"), "row = {row:?}");
    }

    #[test]
    fn alpha_char_appears_immediately_before_ls_field() {
        let out = RestoIterationOutput::new();
        let row = out.format_row_explicit(
            0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 'h', 12,
        );
        // …7.2e + 'h' + 3-wide ls = "1.0000e0h 12" (no space between
        // alpha_pr and the char).
        assert!(row.ends_with("h 12"), "row = {row:?}");
    }

    // The adapter's `IterationOutput::format_row` impl just plumbs
    // numeric fields from the inner data/CQ into
    // `RestoIterationOutput::format_row_explicit`, which the
    // `format_row_explicit` tests above already cover field-by-field.
    // End-to-end coverage of the adapter through a live inner IPM is
    // provided by the `restoration_triggers` integration test.
}

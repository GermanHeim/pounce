//! Options registered for `ipopt.opt` compatibility whose *feature*
//! pounce does not implement (gh#483 follow-up, continuing #191).
//!
//! # Why these are refused rather than ignored
//!
//! `upstream_options.rs` is a faithful port of Ipopt's option registry:
//! every name Ipopt registers is registered here, so an `ipopt.opt`
//! written for Ipopt parses unchanged. That is a real compatibility
//! benefit — and it silently turned ~200 knobs into no-ops, because
//! registering an option says nothing about implementing it. Setting one
//! did exactly nothing and said exactly nothing.
//!
//! Issue #191 audited this class and fixed the half where the *feature*
//! runs and only the option's read site was missing. It explicitly
//! scoped out "feature genuinely unimplemented — expected no-ops". This
//! module closes that half: an option naming a feature pounce does not
//! have is now an error, not a shrug.
//!
//! # What is and is not in the table
//!
//! Membership was established per option, not guessed:
//!
//! 1. the option's name appears in **no** crate source outside the
//!    registry (whole-word — `penalty_max` does not count as present
//!    because `l1_penalty_max` exists), **and**
//! 2. the *feature* it configures is absent too.
//!
//! Both are needed. An option whose name is unread but whose feature
//! runs — `max_resto_iter`, the `limited_memory_*` tail, the corrector
//! knobs — is a missing read site, not a missing feature; refusing those
//! would fail solves whose current answers are already correct. They are
//! deliberately **not** here; wiring them is the other half of the work.
//!
//! The clearest case is the penalty line search. pounce implements
//! `IpPenaltyLSAcceptor` (`line_search_method=penalty`), so its knobs
//! (`nu_init`, `nu_inc`, `rho`, `eta_penalty`) are read sites to add.
//! Ipopt's *other* penalty acceptor — the CG-penalty / inexact-Newton
//! one — has no counterpart here at all, and the port registered its
//! whole option set. Those are refused.
//!
//! # The default gate
//!
//! Only an explicit value **different from the registered default** is
//! refused. `expect_infeasible_problem_ctol` left alone, or an
//! `ipopt.opt` that spells out defaults, must keep working: those ask
//! for nothing. Refusing them would break the very compatibility the
//! registry exists to provide.

use pounce_common::options_list::OptionsList;
use pounce_common::reg_options::{DefaultValue, RegisteredOptions};

/// One unimplemented feature and the options that configure it.
pub struct UnimplementedFeature {
    /// Named in the error, e.g. "the CG-penalty / inexact-Newton line search".
    pub feature: &'static str,
    /// What the caller can do instead. Empty when there is nothing.
    pub advice: &'static str,
    /// The options that belong to it.
    pub options: &'static [&'static str],
}

/// Feature groups pounce does not implement. Refused when set.
pub const UNIMPLEMENTED_FEATURES: &[UnimplementedFeature] = &[
    UnimplementedFeature {
        feature: "the Chen-Goldfarb (CG-penalty) / inexact-Newton line search \
                  — Ipopt's `CGPenaltyLSAcceptor`",
        advice: "pounce implements the filter line search (the default) and \
                 `line_search_method=penalty` (`IpPenaltyLSAcceptor`); tune \
                 those instead",
        options: &[
            "chi_cup",
            "chi_hat",
            "chi_tilde",
            "delta_y_max",
            "epsilon_c",
            "eta_min",
            "fast_des_fact",
            "gamma_hat",
            "gamma_tilde",
            "kappa_x_dis",
            "kappa_y_dis",
            "min_alpha_primal",
            "mult_diverg_feasibility_tol",
            "mult_diverg_y_tol",
            "never_use_fact_cgpen_direction",
            "never_use_piecewise_penalty_ls",
            "pen_des_fact",
            "pen_init_fac",
            "pen_theta_max_fact",
            "penalty_init_max",
            "penalty_init_min",
            "penalty_max",
            "penalty_update_compl_tol",
            "penalty_update_infeasibility_tol",
            "piecewisepenalty_gamma_infeasi",
            "piecewisepenalty_gamma_obj",
            "vartheta",
            "inexact_algorithm",
            "fast_step_computation",
        ],
    },
    UnimplementedFeature {
        feature: "derivative approximation by finite differences",
        advice: "supply `eval_grad_f` / `eval_jac_g` / `eval_h`, and check them \
                 with `derivative_test=first-order`",
        options: &[
            "gradient_approximation",
            "jacobian_approximation",
            "findiff_perturbation",
        ],
    },
    UnimplementedFeature {
        feature: "linear-dependency detection on the equality constraints",
        advice: "pounce's presolve removes structurally redundant rows; see \
                 `presolve`",
        options: &[
            "dependency_detector",
            "dependency_detection_with_rhs",
            "ma28_pivtol",
        ],
    },
    UnimplementedFeature {
        feature: "the per-iteration NaN/Inf check on derivative matrices",
        advice: "`derivative_test=first-order` checks the derivatives once, at \
                 the starting point",
        options: &["check_derivatives_for_naninf"],
    },
    UnimplementedFeature {
        feature: "multiplier recalculation by least squares",
        advice: "",
        options: &["recalc_y", "recalc_y_feas_tol"],
    },
    UnimplementedFeature {
        feature: "a selectable constraint-violation norm",
        advice: "pounce measures the violation in the 2-norm throughout",
        options: &["constraint_violation_norm_type"],
    },
    UnimplementedFeature {
        feature: "magic steps",
        advice: "",
        options: &["magic_steps"],
    },
    UnimplementedFeature {
        feature: "bound replacement on the original problem",
        advice: "",
        options: &["replace_bounds"],
    },
    UnimplementedFeature {
        feature: "the L-BFGS augmented-system and space variants",
        advice: "`hessian_approximation=limited-memory` uses the low-rank \
                 augmented system unconditionally",
        options: &["hessian_approximation_space", "limited_memory_aug_solver"],
    },
    UnimplementedFeature {
        feature: "the linear-variable count hint for L-BFGS",
        advice: "",
        options: &["num_linear_variables"],
    },
    UnimplementedFeature {
        feature: "reading options from a file",
        advice: "pass options on the command line (`pounce model.nl key=value`) \
                 or, from a library, via `initialize_with_options_str`",
        options: &["option_file_name"],
    },
    UnimplementedFeature {
        feature: "skipping the finalize-solution callback",
        advice: "",
        options: &["skip_finalize_solution_call"],
    },
    UnimplementedFeature {
        feature: "the dynamic HSL loader",
        advice: "MA57 is linked at build time with `--features ma57`",
        options: &["hsllib"],
    },
    UnimplementedFeature {
        feature: "these output controls",
        advice: "use `print_level` (0 silences the solver) and `sb=yes` to \
                 suppress the banner",
        options: &["suppress_all_output", "debug_print_level"],
    },
    UnimplementedFeature {
        feature: "a randomly perturbed evaluation point for the derivative \
                  checker",
        advice: "pounce's checker tests at the (bound-projected) starting point, \
                 which is where the solve actually begins",
        options: &["point_perturbation_radius"],
    },
];

/// Options that *are* honored in the sense that matters — the answer is
/// unaffected — but whose performance hint pounce does not exploit.
/// These warn rather than fail: refusing them would stop a solve that
/// returns the right result today, only a little slower.
pub const UNEXPLOITED_HINTS: &[&str] = &[
    "grad_f_constant",
    "hessian_constant",
    "jac_c_constant",
    "jac_d_constant",
];

/// An option set to something the registry says is not its default.
///
/// Both halves matter. `found` alone would fire on an `ipopt.opt` that
/// spells out a default; comparing values alone would fire on nothing,
/// since an unset option *reads back* as its default.
fn set_to_a_non_default(options: &OptionsList, reg: &RegisteredOptions, name: &str) -> bool {
    let Some(opt) = reg.get_option(name) else {
        return false;
    };
    match &opt.default {
        // Bools are registered as `yes`/`no` string options, so this arm
        // covers them too.
        DefaultValue::String(d) => {
            matches!(options.get_string_value(name, ""), Ok((v, true)) if !v.eq_ignore_ascii_case(d))
        }
        DefaultValue::Number(d) => {
            matches!(options.get_numeric_value(name, ""), Ok((v, true)) if v != *d)
        }
        DefaultValue::Integer(d) => {
            matches!(options.get_integer_value(name, ""), Ok((v, true)) if v != *d)
        }
        DefaultValue::None => false,
    }
}

/// The first unimplemented-feature option the caller set, with the
/// message it earns. `None` when nothing in the table was touched.
pub fn refusal(options: &OptionsList, reg: &RegisteredOptions) -> Option<String> {
    for group in UNIMPLEMENTED_FEATURES {
        for name in group.options {
            if set_to_a_non_default(options, reg, name) {
                let advice = if group.advice.is_empty() {
                    String::new()
                } else {
                    format!(" Instead: {}.", group.advice)
                };
                return Some(format!(
                    "pounce: `{name}` configures {}, which pounce does not \
                     implement. It is registered so an ipopt.opt written for \
                     Ipopt still parses, but setting it used to do nothing at \
                     all — silently — so it is refused instead.{advice} \
                     Remove it to run. Tracking issue: \
                     https://github.com/jkitchin/pounce/issues/483",
                    group.feature
                ));
            }
        }
    }
    None
}

/// Warnings for hints pounce does not exploit. Never blocks a solve.
pub fn hint_warnings(options: &OptionsList, reg: &RegisteredOptions) -> Vec<String> {
    UNEXPLOITED_HINTS
        .iter()
        .filter(|name| set_to_a_non_default(options, reg, name))
        .map(|name| {
            format!(
                "pounce: warning: `{name}` is a caching hint pounce does not \
                 exploit — it re-evaluates each iteration regardless. Your \
                 answer is unaffected; only the evaluation count is. \
                 (gh#483)"
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn registry() -> std::rc::Rc<RegisteredOptions> {
        let r = RegisteredOptions::new();
        crate::upstream_options::register_all_upstream_options(&r).expect("register");
        r
    }

    /// A fresh options list over the shared registry, plus a handle on
    /// the registry itself for the default lookups.
    fn fixture() -> (OptionsList, std::rc::Rc<RegisteredOptions>) {
        let reg = registry();
        (OptionsList::with_registered(std::rc::Rc::clone(&reg)), reg)
    }

    /// Every name in the table must actually be registered — a typo
    /// would make its entry dead code that silently never fires, which
    /// is the exact failure mode this module exists to remove.
    #[test]
    fn every_listed_option_is_registered() {
        let (_, reg) = fixture();
        for group in UNIMPLEMENTED_FEATURES {
            for name in group.options {
                assert!(
                    reg.get_option(name).is_some(),
                    "`{name}` is in the refusal table but is not registered",
                );
            }
        }
        for name in UNEXPLOITED_HINTS {
            assert!(
                reg.get_option(name).is_some(),
                "`{name}` is in the hint table but is not registered",
            );
        }
    }

    /// No option may appear twice — once in two feature groups, or in
    /// both tables — or the message a user gets would depend on table
    /// order.
    #[test]
    fn the_tables_do_not_overlap() {
        let mut seen = BTreeSet::new();
        for name in UNIMPLEMENTED_FEATURES
            .iter()
            .flat_map(|g| g.options.iter())
            .chain(UNEXPLOITED_HINTS.iter())
        {
            assert!(seen.insert(*name), "`{name}` is listed twice");
        }
    }

    /// A pristine options list touches nothing.
    #[test]
    fn defaults_are_not_refused() {
        let (opts, reg) = fixture();
        assert_eq!(refusal(&opts, &reg), None);
        assert!(hint_warnings(&opts, &reg).is_empty());
    }

    /// Explicitly writing a default is how a generated `ipopt.opt` looks;
    /// it asks for nothing and must not fail.
    #[test]
    fn explicitly_setting_the_default_is_not_refused() {
        let (mut opts, reg) = fixture();
        // `dependency_detector` defaults to "none"; `magic_steps` to "no".
        opts.set_string_value("dependency_detector", "none", true, false)
            .unwrap();
        opts.set_string_value("magic_steps", "no", true, false)
            .unwrap();
        assert_eq!(refusal(&opts, &reg), None);
    }

    /// …but asking for the feature is refused, by name, with a pointer.
    #[test]
    fn requesting_an_unimplemented_feature_is_refused() {
        let (mut opts, reg) = fixture();
        opts.set_string_value("dependency_detector", "mumps", true, false)
            .unwrap();
        let msg = refusal(&opts, &reg).expect("must refuse");
        assert!(msg.contains("dependency_detector"), "{msg}");
        assert!(msg.contains("linear-dependency detection"), "{msg}");
        assert!(msg.contains("483"), "{msg}");
    }

    /// Numeric knobs of an absent feature are refused the same way.
    #[test]
    fn a_numeric_knob_of_an_absent_feature_is_refused() {
        let (mut opts, reg) = fixture();
        opts.set_numeric_value("penalty_init_max", 42.0, true, false)
            .unwrap();
        let msg = refusal(&opts, &reg).expect("must refuse");
        assert!(msg.contains("CG-penalty"), "{msg}");
    }

    /// Hints warn instead of failing: the answer is the same either way,
    /// so blocking the solve would cost the user more than the silence
    /// did.
    #[test]
    fn caching_hints_warn_but_do_not_refuse() {
        let (mut opts, reg) = fixture();
        opts.set_string_value("hessian_constant", "yes", true, false)
            .unwrap();
        assert_eq!(refusal(&opts, &reg), None, "a hint must not block a solve");
        let warnings = hint_warnings(&opts, &reg);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("hessian_constant"));
    }

    /// Options whose *feature* runs and only whose read site is missing
    /// must stay out of the table — refusing them would fail solves that
    /// are correct today. This pins the boundary the triage drew.
    #[test]
    fn options_on_implemented_features_are_not_refused() {
        for (name, value) in [
            // restoration runs; these are missing read sites (#191 round 2)
            ("max_resto_iter", "17"),
            // the filter line search runs
            ("accept_after_max_steps", "3"),
            // L-BFGS runs
            ("limited_memory_max_skipping", "4"),
            // the Mehrotra corrector runs
            ("corrector_type", "affine"),
        ] {
            let (mut opts, reg) = fixture();
            // The table mixes string, integer and numeric options; try
            // each setter until one takes the value.
            let set = opts.set_string_value(name, value, true, false).is_ok()
                || value
                    .parse::<i32>()
                    .ok()
                    .is_some_and(|v| opts.set_integer_value(name, v, true, false).is_ok())
                || value
                    .parse::<f64>()
                    .ok()
                    .is_some_and(|v| opts.set_numeric_value(name, v, true, false).is_ok());
            assert!(set, "could not set `{name}` to `{value}`");
            assert_eq!(
                refusal(&opts, &reg),
                None,
                "`{name}` configures a feature pounce implements; it needs a \
                 read site, not a refusal",
            );
        }
    }
}

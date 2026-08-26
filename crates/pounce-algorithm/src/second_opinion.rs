//! The second-opinion ladder: re-solving a failure along a different
//! trajectory before believing it.
//!
//! A local-infeasibility verdict on a nonconvex problem is a *local*
//! statement, and an `Invalid_Number_Detected` is a statement about the
//! callbacks at one point. Both are frequently artifacts of the trajectory
//! the solve happened to take, or of the point it started from, rather than
//! facts about the model. The ladder re-solves along up to three deliberately
//! different trajectories and promotes a re-solve only if it converges.
//!
//! This module holds the **policy** — which rungs apply to which failure, what
//! each rung changes, and how a ladder's outcome resolves. It reads options
//! and returns descriptions; it runs no solves. The **driver** that applies a
//! rung and calls back into the solver lives in `pounce-restoration`, because
//! each rung has to rebuild the restoration sub-IPM's factory provider and
//! that provider is defined one crate up the dependency graph.
//!
//! Split out of `pounce-cli`'s `main.rs` so the Python, C and Rust embedding
//! surfaces get the same ladder rather than each re-deriving it — the
//! asymmetry mattered, because a caller driving POUNCE from a modelling layer
//! is precisely the one most likely to hand over an uninitialized (and so
//! all-zero, and so possibly rank-deficient) starting point.

use pounce_common::options_list::OptionsList;
use pounce_nlp::SolveStatistics;
use pounce_nlp::return_codes::ApplicationReturnStatus;

/// One rung of the local-infeasibility second-opinion ladder: a label for the
/// console plus the option assignments that define this re-solve's trajectory.
///
/// Assignments are applied on top of the *baseline* options, not on top of the
/// previous rung — see `second_opinion_rungs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecondOpinionRung {
    pub label: &'static str,
    /// The knob this rung varies, and *only* that knob — one
    /// `read_from_str` line per assignment.
    ///
    /// A rung does not carry lines undoing the earlier rungs. It used to,
    /// and that was a defect: the undo was written as the baseline's
    /// *resolved* value, so a knob the caller never set came back **set**.
    /// `mu_strategy` is the one that bites — `is_mu_strategy_fallback_enabled`
    /// is default-on only while `mu_strategy` is unset, so rung 3 writing
    /// back a resolved `monotone` silently switched off pounce's own
    /// μ-strategy stall retry for the duration of the rung. Measured on
    /// KRONOS `a18_ackley1`: the displaced solve stalls at `max_iter`, the
    /// flip that would have certified it in 237 iterations never fires, and
    /// the ladder reports no recovery. Restoring by *value* is not the same
    /// as restoring by *set-ness*.
    ///
    /// The driver restores the baseline with
    /// `OptionSnapshot::apply` before every rung, which does honour
    /// set-ness, so a rung starts from the true baseline by construction.
    pub assignments: Vec<String>,
}

/// Which failure opened the ladder. Not every rung is evidence about every
/// failure: an `Invalid_Number_Detected` is a statement about the *callbacks*
/// at a point, and re-running the same callbacks at the same point under a
/// different linear-solver scaling or a different barrier strategy evaluates
/// the same non-finite quantity again. Only the rung that moves the point
/// applies there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecondOpinionTrigger {
    /// `Infeasible_Problem_Detected` — a *local* statement about a nonconvex
    /// problem, which every rung is evidence against.
    LocalInfeasibility,
    /// `Invalid_Number_Detected` — a NaN or infinity out of the model.
    InvalidNumber,
}

/// What the baseline options already provide, so a rung that would be a no-op
/// can be dropped instead of burning a solve to re-derive the same answer.
#[derive(Debug, Clone, Copy)]
pub struct SecondOpinionAvailability {
    pub trigger: SecondOpinionTrigger,
    pub scaling_retry_enabled: bool,
    pub mu_retry_enabled: bool,
    pub perturbed_start_retry_enabled: bool,
    pub already_mc64: bool,
    pub already_adaptive: bool,
    /// The baseline already displaces the start, so there is no displacement
    /// left for the third rung to add that the failing solve did not have.
    pub already_perturbed: bool,
    /// `feral_scaling` tag naming the baseline's *resolved* scaling strategy.
    /// `None` under `ScalingStrategy::External`, which drops rungs 2 and 3.
    ///
    /// Now purely a gate. It was the tag those rungs wrote back to undo rung
    /// 1, and `None` meant there was nothing to write; the driver's snapshot
    /// restores by set-ness instead, so no tag is needed and the External
    /// case would in fact be safe to run. The gate is kept because dropping
    /// it adds two rungs on externally-scaled models — a trajectory change,
    /// and a separate decision from this bug fix.
    pub baseline_scaling: Option<&'static str>,
}

/// Build the ladder of second-opinion re-solves for a failing verdict, in the
/// order they should be tried. Each rung varies exactly one knob from the
/// baseline options.
///
/// 1. **`feral_scaling=mc64` — numerical diversity**
///    (`feral_infeasibility_scaling_retry`, on by default). Some KKT
///    trajectories are chaotic: under two equally backward-stable
///    linear-solver scalings the iterates stay bit-identical for many
///    iterations, then diverge by ~1 ULP and fall into different basins — one
///    optimal, the other a spurious stationary point of the constraint
///    violation (`discs.nl`: InfNorm → infeasible, MC64/Identity/MA57/IPOPT →
///    optimal). Sensitive dependence, not a bad solve, so the a-priori
///    scaling router cannot tell the two apart and no per-factor residual
///    flags it; the only reliable signal is the whole-solve verdict.
///
/// 2. **`mu_strategy=adaptive` — algorithmic diversity**
///    (`infeasibility_mu_strategy_retry`, on by default). Rung 1 perturbs only
///    the linear algebra, so it is evidence *only* when the trajectory is
///    ULP-hypersensitive. When it is not, MC64 retraces the same iterates and
///    agrees for the same reason the first solve was wrong — on gh #524
///    (`cresc4`, 6 vars / 8 constraints, feasible, Ipopt solves it in 71
///    iterations) the MC64 re-solve reproduced the original trajectory
///    bit-identically and "corroborated" the false verdict. A different
///    barrier strategy changes the iterate sequence itself, which is what the
///    monotone-µ default gets wrong here: adaptive µ walks to the known
///    optimum. This is also the remedy IPOPT's own documentation gives a user
///    who gets an infeasibility verdict on a problem they believe is feasible;
///    running it automatically just spares them the round trip.
///
/// 3. **`start_point_perturbation=1e-2` — a different starting point**
///    (`infeasibility_perturbed_start_retry`, on by default). The only rung
///    that moves the point rather than the path, and so the only one that is
///    evidence about an `Invalid_Number_Detected` — see
///    [`SecondOpinionTrigger`].
///
/// Rungs are **not** cumulative: the driver restores the baseline before each
/// rung, so rung 2 runs without rung 1's scaling and rung 3 without either
/// earlier knob. That reset is load-bearing, not tidiness: on gh #524's
/// `cresc4`, `mu_strategy=adaptive` recovers the optimum but
/// `mu_strategy=adaptive` with `feral_scaling=mc64` still reports local
/// infeasibility, so a cumulative ladder would have discarded the fix.
pub fn second_opinion_rungs(avail: SecondOpinionAvailability) -> Vec<SecondOpinionRung> {
    let mut rungs = Vec::new();
    let infeasible = avail.trigger == SecondOpinionTrigger::LocalInfeasibility;
    if infeasible && avail.scaling_retry_enabled && !avail.already_mc64 {
        rungs.push(SecondOpinionRung {
            label: "feral_scaling=mc64",
            assignments: vec!["feral_scaling mc64\n".to_string()],
        });
    }
    if avail.baseline_scaling.is_some()
        && infeasible
        && avail.mu_retry_enabled
        && !avail.already_adaptive
    {
        rungs.push(SecondOpinionRung {
            label: "mu_strategy=adaptive",
            assignments: vec!["mu_strategy adaptive\n".to_string()],
        });
    }
    // Rung 3 varies exactly one knob from the *baseline*, so both earlier
    // rungs' knobs must be undone first, not inherited — gh #524 is the case
    // where stacking two of them threw the fix away. That undo is the
    // driver's job (`OptionSnapshot::apply` before each rung), not an
    // assignment here; see the note on `assignments`.
    if avail.baseline_scaling.is_some()
        && avail.perturbed_start_retry_enabled
        && !avail.already_perturbed
    {
        rungs.push(SecondOpinionRung {
            label: "start_point_perturbation=1e-2",
            assignments: vec!["start_point_perturbation 1e-2\n".to_string()],
        });
    }
    rungs
}

/// Whether the ladder's narration should reach the console.
///
/// `print_level 0` is a request for silence, and the ladder's running
/// commentary is no more exempt from it than the `EXIT:` block — the C
/// interface is an Ipopt drop-in, where `print_level=0 sb=yes` is the
/// documented way to get a quiet solve, and up to five unexpected `pounce:`
/// lines on a failing one is exactly what that asks not to happen. The ladder
/// still *runs*; only the console is quiet.
///
/// Lives here rather than at each call site because the CLI and the C
/// interface both need it and a duplicated branch is a branch that can drift
/// — one of them silently losing the gate would look identical to the other
/// keeping it. `pounce-rs` and Python do not call this: they never print,
/// they hand the narration back to the caller.
///
/// Only an *explicit* `print_level` silences: a level nobody set narrates,
/// and so does an unregistered or unreadable one. That is the pre-gate
/// behaviour and the safe direction — too much on a failing solve, never too
/// little.
pub fn narration_is_wanted(options: &OptionsList) -> bool {
    options
        .get_integer_value("print_level", "")
        .map(|(level, found)| !found || level >= 1)
        .unwrap_or(true)
}

/// Did a second-opinion re-solve converge well enough to overturn the original
/// local-infeasibility verdict? Only a clean or acceptable-level solve
/// promotes; everything else (including a second infeasibility verdict) leaves
/// the original verdict standing.
pub fn scaling_retry_promoted(retry_status: ApplicationReturnStatus) -> bool {
    matches!(
        retry_status,
        ApplicationReturnStatus::SolveSucceeded | ApplicationReturnStatus::SolvedToAcceptableLevel
    )
}

/// Resolve the final `(status, statistics)` after an MC64 hypersensitivity
/// re-solve (code review L23).
///
/// On promotion the retry is the authoritative solve, so its status **and** its
/// statistics are reported together. Otherwise the original local-infeasibility
/// verdict is kept — and so are the *original* solve's statistics, so the
/// summary / JSON report never pair the original verdict with the failed
/// retry's iteration count or objective. The pre-fix code reverted `status` to
/// `InfeasibleProblemDetected` but read `app.statistics()` *after* the retry,
/// leaking the retry solve's stats into a report labeled with the original
/// verdict.
pub fn resolve_scaling_retry_outcome(
    original_status: ApplicationReturnStatus,
    retry_status: ApplicationReturnStatus,
    original_stats: SolveStatistics,
    retry_stats: SolveStatistics,
) -> (ApplicationReturnStatus, SolveStatistics) {
    if scaling_retry_promoted(retry_status) {
        (retry_status, retry_stats)
    } else {
        (original_status, original_stats)
    }
}

impl SecondOpinionTrigger {
    /// Which ladder, if any, a finished solve's verdict opens.
    ///
    /// Only these two statuses open one. In particular an iteration- or
    /// time-limit exit does not: the answer there is a bigger budget, and a
    /// re-solve from a different trajectory would burn the same budget again
    /// to reach the same wall.
    pub fn for_status(status: ApplicationReturnStatus) -> Option<Self> {
        match status {
            ApplicationReturnStatus::InfeasibleProblemDetected => {
                Some(SecondOpinionTrigger::LocalInfeasibility)
            }
            ApplicationReturnStatus::InvalidNumberDetected => {
                Some(SecondOpinionTrigger::InvalidNumber)
            }
            _ => None,
        }
    }

    /// The word for this trigger in a console line.
    pub fn describe(self) -> &'static str {
        match self {
            SecondOpinionTrigger::LocalInfeasibility => "local infeasibility",
            SecondOpinionTrigger::InvalidNumber => "invalid number",
        }
    }
}

impl SecondOpinionAvailability {
    /// Read everything the ladder needs to know about the baseline solve out
    /// of the options it ran under.
    ///
    /// The scaling and barrier tags are read from the **resolved** strategy,
    /// not from the option strings. `feral_scaling` is applied only when set
    /// explicitly and otherwise `FeralConfig::from_env()` governs via
    /// `POUNCE_FERAL_SCALING`, so the option string reads `auto` for an
    /// env-configured run; writing that back would silently override the
    /// environment on the retry instead of restoring it. `External` is
    /// unreachable from the string option, and if it ever arrives here there
    /// is no tag to write, so the rungs that need one are dropped rather than
    /// guessed at.
    pub fn from_options(options: &OptionsList, trigger: SecondOpinionTrigger) -> Self {
        // Each of the three `*_retry` flags is read with its tag written out
        // as a literal rather than through a shared closure: `init_options_wiring`
        // proves every registered Initialization option is actually consumed by
        // scanning the source for `get_*_value("<tag>"`, and a closure taking
        // the tag as a parameter is invisible to that scan — which would leave
        // `infeasibility_perturbed_start_retry` looking like a knob that
        // validates, accepts a value and does nothing.
        let scaling = crate::application::feral_config_from_options(options).scaling;
        let baseline_scaling = match scaling {
            pounce_feral::ScalingStrategy::Auto => Some("auto"),
            pounce_feral::ScalingStrategy::InfNorm => Some("infnorm"),
            pounce_feral::ScalingStrategy::Mc64Symmetric => Some("mc64"),
            pounce_feral::ScalingStrategy::Identity => Some("identity"),
            pounce_feral::ScalingStrategy::External(_) => None,
        };
        let already_adaptive = options
            .get_string_value("mu_strategy", "")
            .map(|(v, _found)| v == "adaptive")
            .unwrap_or(false);
        Self {
            trigger,
            scaling_retry_enabled: options
                .get_bool_value("feral_infeasibility_scaling_retry", "")
                .map(|(v, _found)| v)
                .unwrap_or(true),
            mu_retry_enabled: options
                .get_bool_value("infeasibility_mu_strategy_retry", "")
                .map(|(v, _found)| v)
                .unwrap_or(true),
            perturbed_start_retry_enabled: options
                .get_bool_value("infeasibility_perturbed_start_retry", "")
                .map(|(v, _found)| v)
                .unwrap_or(true),
            already_mc64: matches!(scaling, pounce_feral::ScalingStrategy::Mc64Symmetric),
            already_adaptive,
            already_perturbed: options
                .get_numeric_value("start_point_perturbation", "")
                .map(|(v, _found)| v > 0.0)
                .unwrap_or(false),
            baseline_scaling,
            // `mu_strategy` has exactly two registered values, so "not
            // adaptive" is "monotone" and there is no third case to guess at.
        }
    }
}

#[cfg(test)]
mod scaling_retry_tests {
    use super::{
        SecondOpinionAvailability, SecondOpinionTrigger, narration_is_wanted,
        resolve_scaling_retry_outcome, scaling_retry_promoted, second_opinion_rungs,
    };
    use pounce_common::options_list::OptionsList;
    use pounce_nlp::SolveStatistics;
    use pounce_nlp::return_codes::ApplicationReturnStatus;

    fn avail() -> SecondOpinionAvailability {
        SecondOpinionAvailability {
            trigger: SecondOpinionTrigger::LocalInfeasibility,
            scaling_retry_enabled: true,
            mu_retry_enabled: true,
            perturbed_start_retry_enabled: true,
            already_mc64: false,
            already_adaptive: false,
            already_perturbed: false,
            baseline_scaling: Some("auto"),
        }
    }

    /// The default ladder is three rungs, in increasing order of how much
    /// they change: linear algebra, then barrier trajectory, then the point
    /// the trajectory starts from.
    #[test]
    fn default_ladder_is_scaling_then_barrier_strategy_then_start() {
        let rungs = second_opinion_rungs(avail());
        let labels: Vec<_> = rungs.iter().map(|r| r.label).collect();
        assert_eq!(
            labels,
            [
                "feral_scaling=mc64",
                "mu_strategy=adaptive",
                "start_point_perturbation=1e-2"
            ]
        );
    }

    /// gh #524: the rungs are applied to the *baseline*, not stacked, because
    /// on `cresc4` `mu_strategy=adaptive` recovers the optimum while
    /// `mu_strategy=adaptive` together with `feral_scaling=mc64` still reports
    /// local infeasibility — a cumulative ladder would throw the fix away.
    ///
    /// The barrier rung therefore carries `mu_strategy` and nothing else;
    /// rung 1's scaling is undone by the driver re-applying its snapshot, and
    /// `second_opinion_driver::tests::each_rung_starts_from_the_baseline`
    /// is where the undo itself is pinned.
    #[test]
    fn barrier_rung_varies_only_the_barrier_strategy() {
        for baseline in ["auto", "infnorm"] {
            let rungs = second_opinion_rungs(SecondOpinionAvailability {
                baseline_scaling: Some(baseline),
                ..avail()
            });
            let barrier = rungs
                .iter()
                .find(|r| r.label == "mu_strategy=adaptive")
                .expect("barrier rung present");
            let assigned: Vec<_> = barrier.assignments.iter().map(|a| a.trim()).collect();
            assert_eq!(assigned, ["mu_strategy adaptive"]);
        }
    }

    /// A rung that cannot change anything is dropped rather than burning a
    /// whole solve to re-derive the same answer.
    #[test]
    fn rungs_already_satisfied_at_baseline_are_dropped() {
        let only_barrier = second_opinion_rungs(SecondOpinionAvailability {
            already_mc64: true,
            ..avail()
        });
        assert_eq!(
            only_barrier.iter().map(|r| r.label).collect::<Vec<_>>(),
            ["mu_strategy=adaptive", "start_point_perturbation=1e-2"],
        );

        let only_scaling = second_opinion_rungs(SecondOpinionAvailability {
            already_adaptive: true,
            ..avail()
        });
        assert_eq!(
            only_scaling.iter().map(|r| r.label).collect::<Vec<_>>(),
            ["feral_scaling=mc64", "start_point_perturbation=1e-2"],
        );

        assert!(
            second_opinion_rungs(SecondOpinionAvailability {
                already_mc64: true,
                already_adaptive: true,
                already_perturbed: true,
                ..avail()
            })
            .is_empty(),
            "nothing left to vary means no ladder at all",
        );
    }

    /// A resolved scaling with no `feral_scaling` tag
    /// (`ScalingStrategy::External`) drops the barrier rung; the scaling rung
    /// is unaffected. The gate is now conservatism rather than necessity —
    /// the driver restores by set-ness, so a missing tag no longer strands
    /// the rung under rung 1's scaling — and it is kept because lifting it
    /// would add rungs on externally-scaled models, which is a trajectory
    /// change. See the note on `baseline_scaling`.
    #[test]
    fn barrier_rung_is_dropped_when_the_baseline_scaling_has_no_tag() {
        let rungs = second_opinion_rungs(SecondOpinionAvailability {
            baseline_scaling: None,
            ..avail()
        });
        assert_eq!(
            rungs.iter().map(|r| r.label).collect::<Vec<_>>(),
            ["feral_scaling=mc64"],
        );
    }

    /// Each rung has its own opt-out, and turning both off restores upstream
    /// IPOPT's behaviour of shipping the first verdict.
    #[test]
    fn each_rung_can_be_disabled_independently() {
        assert_eq!(
            second_opinion_rungs(SecondOpinionAvailability {
                scaling_retry_enabled: false,
                ..avail()
            })
            .iter()
            .map(|r| r.label)
            .collect::<Vec<_>>(),
            ["mu_strategy=adaptive", "start_point_perturbation=1e-2"],
        );
        assert_eq!(
            second_opinion_rungs(SecondOpinionAvailability {
                mu_retry_enabled: false,
                ..avail()
            })
            .iter()
            .map(|r| r.label)
            .collect::<Vec<_>>(),
            ["feral_scaling=mc64", "start_point_perturbation=1e-2"],
        );
        assert_eq!(
            second_opinion_rungs(SecondOpinionAvailability {
                perturbed_start_retry_enabled: false,
                ..avail()
            })
            .iter()
            .map(|r| r.label)
            .collect::<Vec<_>>(),
            ["feral_scaling=mc64", "mu_strategy=adaptive"],
        );
        assert!(
            second_opinion_rungs(SecondOpinionAvailability {
                scaling_retry_enabled: false,
                mu_retry_enabled: false,
                perturbed_start_retry_enabled: false,
                ..avail()
            })
            .is_empty(),
        );
    }

    /// gh #524's lesson applied to the third rung: it varies exactly one thing
    /// from the *baseline*. It undoes the earlier rungs by having the driver
    /// re-apply the snapshot, so its own assignment list is the displacement
    /// and nothing else.
    #[test]
    fn start_rung_assigns_only_the_displacement() {
        for baseline_scaling in ["auto", "infnorm"] {
            let rungs = second_opinion_rungs(SecondOpinionAvailability {
                baseline_scaling: Some(baseline_scaling),
                ..avail()
            });
            let start = rungs
                .iter()
                .find(|r| r.label == "start_point_perturbation=1e-2")
                .expect("start rung present");
            let assigned: Vec<_> = start.assignments.iter().map(|a| a.trim()).collect();
            assert_eq!(assigned, ["start_point_perturbation 1e-2"]);
        }
    }

    /// The regression this file exists to prevent a second time.
    ///
    /// Rungs 2 and 3 used to undo their predecessors by writing the
    /// baseline's *resolved* value back. For a knob the caller never set that
    /// is a no-op by value and a change by set-ness, and
    /// `is_mu_strategy_fallback_enabled` reads set-ness: it is default-on
    /// only while `mu_strategy` is unset. So rung 3 re-asserting a resolved
    /// `monotone` turned pounce's own μ-strategy stall retry off for the
    /// length of the rung. On KRONOS `a18_ackley1` that is the difference
    /// between `Solve_Succeeded` in 237 iterations and
    /// `Maximum_Iterations_Exceeded` at 3000.
    ///
    /// No rung may name a knob it is not there to vary — restoring is the
    /// driver's job, because only the snapshot knows set-ness.
    #[test]
    fn no_rung_writes_back_a_knob_it_does_not_vary() {
        for avail in [
            avail(),
            SecondOpinionAvailability {
                already_adaptive: true,
                ..avail()
            },
            SecondOpinionAvailability {
                trigger: SecondOpinionTrigger::InvalidNumber,
                ..avail()
            },
        ] {
            for rung in second_opinion_rungs(avail) {
                let varies = rung.label.split('=').next().expect("label has a tag");
                for a in &rung.assignments {
                    let tag = a.trim().split_whitespace().next().expect("tag");
                    assert_eq!(
                        tag,
                        varies,
                        "rung `{}` writes `{}`, which it does not vary",
                        rung.label,
                        a.trim(),
                    );
                }
            }
        }
    }

    /// Rung 3 sits behind the same `baseline_scaling` gate as rung 2 and is
    /// dropped with it, for the same reason: kept as conservatism about a
    /// trajectory change on externally-scaled models, not because the rung
    /// needs a tag to put back.
    #[test]
    fn start_rung_is_dropped_when_the_baseline_scaling_has_no_tag() {
        let rungs = second_opinion_rungs(SecondOpinionAvailability {
            baseline_scaling: None,
            ..avail()
        });
        assert!(
            !rungs
                .iter()
                .any(|r| r.label == "start_point_perturbation=1e-2"),
            "{:?}",
            rungs.iter().map(|r| r.label).collect::<Vec<_>>(),
        );
    }

    /// The console gate, which both printing surfaces share. An explicit `0`
    /// is silence; anything above it, or no setting at all, stays loud — a
    /// caller who did not ask for quiet is not asking for less.
    #[test]
    fn narration_follows_print_level() {
        let mut opts = OptionsList::new();
        // Unset: narrate, the pre-gate behaviour.
        assert!(narration_is_wanted(&opts));
        // …and an unset level narrates whatever the registered default reads
        // as, which is what distinguishes "nobody asked" from "asked for 0".
        for (level, want) in [(0, false), (1, true), (5, true), (12, true)] {
            opts.set_integer_value("print_level", level, true, true)
                .unwrap();
            assert_eq!(narration_is_wanted(&opts), want, "print_level={level}");
        }
    }

    /// An `Invalid_Number_Detected` reaches only the rung that moves the
    /// point. Re-running the same callbacks at the same point under a
    /// different linear-solver scaling or a different barrier strategy
    /// evaluates the same non-finite quantity again, so those two rungs are
    /// not evidence about this failure and would only burn solves.
    #[test]
    fn an_invalid_number_reaches_only_the_start_rung() {
        let rungs = second_opinion_rungs(SecondOpinionAvailability {
            trigger: SecondOpinionTrigger::InvalidNumber,
            ..avail()
        });
        assert_eq!(
            rungs.iter().map(|r| r.label).collect::<Vec<_>>(),
            ["start_point_perturbation=1e-2"],
        );
    }

    /// …and disabling that rung leaves an invalid-number run with no ladder at
    /// all, rather than falling back to the two rungs that cannot help.
    #[test]
    fn an_invalid_number_with_the_start_rung_off_has_no_ladder() {
        assert!(
            second_opinion_rungs(SecondOpinionAvailability {
                trigger: SecondOpinionTrigger::InvalidNumber,
                perturbed_start_retry_enabled: false,
                ..avail()
            })
            .is_empty(),
        );
    }

    /// A baseline that already displaces the start has nothing left for rung 3
    /// to add: re-running with the same displacement reproduces the failing
    /// solve.
    #[test]
    fn a_baseline_that_already_perturbs_drops_the_start_rung() {
        let rungs = second_opinion_rungs(SecondOpinionAvailability {
            already_perturbed: true,
            ..avail()
        });
        assert_eq!(
            rungs.iter().map(|r| r.label).collect::<Vec<_>>(),
            ["feral_scaling=mc64", "mu_strategy=adaptive"],
        );
    }

    /// The verdict a failed ladder keeps is the one the solve actually
    /// shipped. Before the ladder took `Invalid_Number_Detected` as a trigger
    /// this function hard-coded `Infeasible_Problem_Detected`, which for the
    /// new trigger would have reported the wrong failure.
    #[test]
    fn a_failed_ladder_keeps_whichever_verdict_opened_it() {
        for original in [
            ApplicationReturnStatus::InfeasibleProblemDetected,
            ApplicationReturnStatus::InvalidNumberDetected,
        ] {
            let (status, stats) = resolve_scaling_retry_outcome(
                original,
                ApplicationReturnStatus::MaximumIterationsExceeded,
                stats_with_iters(7),
                stats_with_iters(42),
            );
            assert_eq!(status, original);
            assert_eq!(stats.iteration_count, 7);
        }
    }

    fn stats_with_iters(n: i32) -> SolveStatistics {
        SolveStatistics {
            iteration_count: n,
            final_objective: n as f64,
            ..SolveStatistics::default()
        }
    }

    /// Code review L23: when the MC64 hypersensitivity re-solve does **not**
    /// recover, the verdict reverts to the original local-infeasibility status
    /// — and the reported statistics must revert with it, not leak the failed
    /// retry's iteration count / objective.
    #[test]
    fn failed_retry_keeps_original_status_and_stats() {
        let original = stats_with_iters(7);
        let retry = stats_with_iters(42);
        for retry_status in [
            ApplicationReturnStatus::InfeasibleProblemDetected,
            ApplicationReturnStatus::MaximumIterationsExceeded,
            ApplicationReturnStatus::RestorationFailed,
        ] {
            assert!(!scaling_retry_promoted(retry_status));
            let (status, stats) = resolve_scaling_retry_outcome(
                ApplicationReturnStatus::InfeasibleProblemDetected,
                retry_status,
                original.clone(),
                retry.clone(),
            );
            assert_eq!(
                status,
                ApplicationReturnStatus::InfeasibleProblemDetected,
                "a non-promoting retry ({retry_status:?}) keeps the original verdict"
            );
            assert_eq!(
                stats.iteration_count, 7,
                "stats must stay the original solve's, not the failed retry's"
            );
            assert_eq!(stats.final_objective, 7.0);
        }
    }

    /// On promotion the retry is authoritative: its status AND its statistics
    /// are reported together.
    #[test]
    fn promoted_retry_adopts_retry_status_and_stats() {
        let original = stats_with_iters(7);
        let retry = stats_with_iters(42);
        for retry_status in [
            ApplicationReturnStatus::SolveSucceeded,
            ApplicationReturnStatus::SolvedToAcceptableLevel,
        ] {
            assert!(scaling_retry_promoted(retry_status));
            let (status, stats) = resolve_scaling_retry_outcome(
                ApplicationReturnStatus::InfeasibleProblemDetected,
                retry_status,
                original.clone(),
                retry.clone(),
            );
            assert_eq!(status, retry_status, "a promoting retry adopts its verdict");
            assert_eq!(
                stats.iteration_count, 42,
                "promoted: stats must be the retry solve's"
            );
            assert_eq!(stats.final_objective, 42.0);
        }
    }
}

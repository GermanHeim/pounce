//! The four upstream constant-derivative hints, and what pounce can
//! *prove* about them.
//!
//! Ipopt registers `grad_f_constant`, `hessian_constant`,
//! `jac_c_constant` and `jac_d_constant` as unchecked user assertions:
//! setting one makes the solver evaluate that derivative once and reuse
//! it forever, and if the assertion is false the answer is silently
//! wrong — a Hessian frozen at the starting point is still *a* matrix,
//! and the algorithm converges to something with it. pounce until now
//! did the opposite and equally unhelpful thing: it warned that the hint
//! was ignored and re-evaluated regardless (gh #483,
//! `unimplemented_options::UNEXPLOITED_HINTS`).
//!
//! This module is the middle road, and it is a deliberate divergence
//! from upstream (gh #588, phase Q6). A model that knows its own algebra
//! — an `.nl` file, whose bodies the degree-≤2 recognizer reads exactly —
//! can *prove* the hint, and then the hint needs no user at all. It can
//! also prove the hint **false**, and then honouring it would produce
//! the wrong Hessian on purpose.
//!
//! # Three states, not two
//!
//! The reconciliation is over [`DerivativeProof`], which is three-valued,
//! because the two-valued reading is a bug in each direction:
//!
//! | proof | user asserted | pounce does |
//! |---|---|---|
//! | `Constant` | yes or no | **reuses** the derivative — the option is not needed |
//! | `Varying` | no | evaluates every iterate |
//! | `Varying` | yes | **warns and ignores** — the divergence |
//! | `Unknown` | no | evaluates every iterate |
//! | `Unknown` | yes | **honours it on trust** — upstream's contract |
//!
//! The last row is the one that keeps this honest. A callback TNLP, the
//! C interface and both GAMS links hand pounce numbers, not algebra:
//! there is nothing to read a proof off, and `Unknown` is the truthful
//! answer. Overriding a user there — refusing a hint pounce merely
//! cannot confirm — would be its own silent wrong answer, one level up.
//! `Unknown` is emphatically **not** "varies"; it is "not established".
//!
//! # What counts as proof
//!
//! Degree. A body the recognizer proves is degree ≤ 1 has a constant
//! gradient and a zero Hessian; one it proves is degree 2 has a Hessian
//! that is nonzero, hence a gradient that genuinely moves. A body it
//! refuses is `Unknown` — the refusal is not evidence of nonlinearity
//! (`is_expanded_quadratic` refuses `2·(x + 1)`, which is affine).
//!
//! Note which gate this is. Q4's `is_expanded_quadratic` gate exists
//! because *evaluating* a factored form from stored coefficients cancels
//! (`(x − 500000)²` loses five digits, gh #544). Q6 evaluates nothing
//! from coefficients: it reuses the value the model's own tape produced
//! at the first call. So the gate that applies here is the weaker,
//! ungated degree question — which is why `airport.nl`, a factored
//! model Q4 must refuse, is one of the models Q6 reaches.

use pounce_common::types::Index;

/// What a model can prove about one derivative's constancy.
///
/// Three-valued on purpose; see the module docs. `Unknown` is the
/// default because it is what a model that has not been asked, or
/// cannot answer, truthfully reports.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DerivativeProof {
    /// No proof either way. **Not** a claim that the derivative varies.
    #[default]
    Unknown,
    /// Proved to be the same function value at every point pounce will
    /// evaluate it at.
    Constant,
    /// Proved to depend on its arguments.
    Varying,
}

impl DerivativeProof {
    /// The proof for a derivative assembled from several independent
    /// pieces — a Jacobian block over its rows, or `∇²L` over the
    /// objective and the rows it sums.
    ///
    /// One proved-varying piece proves the whole thing varies, whatever
    /// the others say; otherwise every piece must be proved constant,
    /// because a single `Unknown` leaves the sum unestablished.
    pub fn all(pieces: impl IntoIterator<Item = DerivativeProof>) -> DerivativeProof {
        let mut out = DerivativeProof::Constant;
        for p in pieces {
            match p {
                DerivativeProof::Varying => return DerivativeProof::Varying,
                DerivativeProof::Unknown => out = DerivativeProof::Unknown,
                DerivativeProof::Constant => {}
            }
        }
        out
    }

    /// Weaken a `Varying` proof to `Unknown`, keeping the other two.
    ///
    /// Used where a transformation below the proof can only *remove*
    /// variation — fixed-variable elimination is the case that matters:
    /// a row `x·y` with `y` fixed to a parameter has a constant gradient
    /// in the reduced space the algorithm actually sees, so the
    /// full-space proof that it varies no longer holds. Constancy
    /// survives such a transformation; a proof of variation does not.
    pub fn forget_variation(self) -> DerivativeProof {
        match self {
            DerivativeProof::Varying => DerivativeProof::Unknown,
            other => other,
        }
    }
}

/// What one model proves about its own derivatives, in the model's own
/// (full, pre-split) row order.
///
/// Returned by [`crate::tnlp::TNLP::derivative_proofs`]. The default —
/// everything `Unknown`, no rows — is what a TNLP that has only
/// callbacks to offer honestly reports, and is what every TNLP reports
/// until it overrides the method.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DerivativeProofs {
    /// Proof for `∇f`. The `grad_f_constant` hint.
    pub grad_f: DerivativeProof,
    /// Proof for `∇²L` **as a whole** — objective and rows together.
    /// The `hessian_constant` hint.
    pub hessian: DerivativeProof,
    /// Proof for `∇gᵢ`, one entry per constraint row in the TNLP's own
    /// row order. Empty means "declined", which is the same answer as
    /// `m` `Unknown`s and costs no allocation.
    pub jac: Vec<DerivativeProof>,
}

impl DerivativeProofs {
    /// The proof for row `i`, treating an empty [`Self::jac`] as
    /// `Unknown`.
    pub fn row(&self, i: usize) -> DerivativeProof {
        self.jac.get(i).copied().unwrap_or_default()
    }
}

/// Which derivatives one solve will reuse across iterates.
///
/// This is the *resolved* answer — proof reconciled with the user's four
/// options — and the only thing [`crate::orig_ipopt_nlp::OrigIpoptNlp`]
/// reads. Default: reuse nothing, i.e. exactly the behaviour every
/// pounce release before gh #588 Q6 had.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConstantDerivatives {
    pub grad_f: bool,
    pub hessian: bool,
    pub jac_c: bool,
    pub jac_d: bool,
}

impl ConstantDerivatives {
    /// True when nothing is reused — the pre-Q6 behaviour, and the case
    /// every hot path can skip.
    pub fn is_empty(&self) -> bool {
        !(self.grad_f || self.hessian || self.jac_c || self.jac_d)
    }
}

/// The registered option names, in the order [`reconcile`] takes and
/// returns them.
pub const HINT_OPTIONS: [&str; 4] = [
    "grad_f_constant",
    "hessian_constant",
    "jac_c_constant",
    "jac_d_constant",
];

/// What became of one hint, for reporting to the user.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HintOutcome {
    /// The registered option name.
    pub name: &'static str,
    /// Whether the user set it to a non-default (i.e. asserted it).
    pub asserted: bool,
    /// What the model proved.
    pub proof: DerivativeProof,
    /// Whether pounce will reuse the derivative.
    pub honoured: bool,
}

impl HintOutcome {
    /// The user asserted a hint the model disproves. This is the case
    /// upstream honours silently and pounce refuses.
    pub fn contradicted(&self) -> bool {
        self.asserted && self.proof == DerivativeProof::Varying
    }

    /// The user asserted a hint pounce can neither confirm nor refute,
    /// and it is being taken on trust.
    pub fn trusted(&self) -> bool {
        self.asserted && self.proof == DerivativeProof::Unknown
    }

    /// pounce proved the hint without being asked. Nothing to report to
    /// the user; worth a debug line.
    pub fn auto_detected(&self) -> bool {
        !self.asserted && self.proof == DerivativeProof::Constant
    }

    /// The message a contradicted hint earns, or `None`.
    pub fn warning(&self) -> Option<String> {
        if !self.contradicted() {
            return None;
        }
        Some(format!(
            "pounce: warning: ignoring `{}=yes` — pounce proved from the \
             model's own algebra that this derivative is not constant, so \
             reusing it would return a wrong answer rather than a slow one. \
             Ipopt honours this hint without checking it. Remove the option; \
             pounce detects a genuinely constant derivative on its own and \
             needs no hint to reuse it. (gh#588)",
            self.name
        ))
    }
}

/// Reconcile what the model proved with what the user asserted.
///
/// `proofs` and `asserted` are both in [`HINT_OPTIONS`] order. The rule
/// is the table in the module docs: proof wins in both directions where
/// there is one, and the user's assertion decides only where there is
/// none.
pub fn reconcile(
    proofs: [DerivativeProof; 4],
    asserted: [bool; 4],
) -> ([HintOutcome; 4], ConstantDerivatives) {
    let mut outcomes = [HintOutcome {
        name: "",
        asserted: false,
        proof: DerivativeProof::Unknown,
        honoured: false,
    }; 4];
    for k in 0..4 {
        let honoured = match proofs[k] {
            DerivativeProof::Constant => true,
            DerivativeProof::Varying => false,
            DerivativeProof::Unknown => asserted[k],
        };
        outcomes[k] = HintOutcome {
            name: HINT_OPTIONS[k],
            asserted: asserted[k],
            proof: proofs[k],
            honoured,
        };
    }
    let enabled = ConstantDerivatives {
        grad_f: outcomes[0].honoured,
        hessian: outcomes[1].honoured,
        jac_c: outcomes[2].honoured,
        jac_d: outcomes[3].honoured,
    };
    (outcomes, enabled)
}

/// Fold per-row proofs into the equality / inequality split the
/// algorithm actually evaluates.
///
/// `rows` are indices into the model's own row order — `c_map` or
/// `d_map` from the bound classification. An empty subsystem is
/// vacuously constant: there is no Jacobian block to re-evaluate.
pub fn subsystem_proof(proofs: &DerivativeProofs, rows: &[Index]) -> DerivativeProof {
    DerivativeProof::all(rows.iter().map(|&i| proofs.row(i as usize)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_is_dominated_by_a_proof_of_variation() {
        use DerivativeProof::*;
        assert_eq!(DerivativeProof::all([Constant, Constant]), Constant);
        assert_eq!(DerivativeProof::all([Constant, Unknown]), Unknown);
        assert_eq!(DerivativeProof::all([Unknown, Varying]), Varying);
        assert_eq!(DerivativeProof::all([Varying, Unknown]), Varying);
        // Vacuous: nothing to re-evaluate.
        assert_eq!(DerivativeProof::all([]), Constant);
    }

    #[test]
    fn forgetting_variation_keeps_the_other_two() {
        use DerivativeProof::*;
        assert_eq!(Varying.forget_variation(), Unknown);
        assert_eq!(Constant.forget_variation(), Constant);
        assert_eq!(Unknown.forget_variation(), Unknown);
    }

    /// The heart of the phase: a proof beats the user in both
    /// directions, and only where there is no proof does the user decide.
    #[test]
    fn proof_beats_the_user_in_both_directions() {
        use DerivativeProof::*;
        let (out, en) = reconcile(
            [Constant, Varying, Unknown, Unknown],
            [false, true, true, false],
        );
        // Proved constant without being asked: reused anyway.
        assert!(out[0].honoured && out[0].auto_detected());
        // Proved varying and asserted: refused, and it warns.
        assert!(!out[1].honoured);
        assert!(out[1].contradicted());
        assert!(
            out[1]
                .warning()
                .is_some_and(|w| w.contains("hessian_constant"))
        );
        // Unknown and asserted: honoured on trust, silently.
        assert!(out[2].honoured && out[2].trusted());
        assert!(out[2].warning().is_none());
        // Unknown and not asserted: nothing happens.
        assert!(!out[3].honoured);
        assert_eq!(
            en,
            ConstantDerivatives {
                grad_f: true,
                hessian: false,
                jac_c: true,
                jac_d: false,
            }
        );
        assert!(!en.is_empty());
    }

    #[test]
    fn a_default_options_list_over_a_silent_model_reuses_nothing() {
        let (out, en) = reconcile([DerivativeProof::Unknown; 4], [false; 4]);
        assert!(en.is_empty());
        assert!(out.iter().all(|o| o.warning().is_none()));
    }

    #[test]
    fn an_empty_subsystem_is_vacuously_constant() {
        let p = DerivativeProofs {
            jac: vec![DerivativeProof::Varying, DerivativeProof::Constant],
            ..Default::default()
        };
        assert_eq!(subsystem_proof(&p, &[]), DerivativeProof::Constant);
        assert_eq!(subsystem_proof(&p, &[1]), DerivativeProof::Constant);
        assert_eq!(subsystem_proof(&p, &[0, 1]), DerivativeProof::Varying);
        // Out of range reads as Unknown rather than panicking: a model
        // that answered for fewer rows than it has has declined for the
        // rest.
        assert_eq!(subsystem_proof(&p, &[5]), DerivativeProof::Unknown);
    }
}

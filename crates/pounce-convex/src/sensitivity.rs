//! Post-optimal sensitivity for the convex QP — the sIPOPT analog.
//!
//! Given a converged [`QpSolution`] to
//!
//! ```text
//!   min ½xᵀPx + cᵀx  s.t.  Ax = b,  Gx ≤ h,  lb ≤ x ≤ ub,
//! ```
//!
//! the first-order change of the primal–dual solution under a small
//! perturbation of the problem data — *holding the active set fixed* — is
//! the solution of the **active-set KKT system**
//!
//! ```text
//!   ⎡ P    Aᵀ   B_aᵀ ⎤ ⎡ dx  ⎤   ⎡ −dc                  ⎤
//!   ⎢ A    0    0    ⎥ ⎢ dy  ⎥ = ⎢  db                  ⎥
//!   ⎣ B_a  0    0    ⎦ ⎣ dz_a⎦   ⎣  dr_a                ⎦
//! ```
//!
//! where `B_a` stacks the **active** inequality rows of `G` and the active
//! variable-bound rows (`eⱼᵀ`), and the right-hand side is the parameter
//! derivative of the KKT residual. This is exactly the predictor used by
//! Ipopt's sIPOPT (Pirnay, López-Negrete & Biegler 2012) specialized to a
//! quadratic program, where the Lagrangian Hessian is the constant `P`.
//!
//! [`QpSensitivity`] assembles and factors this symmetric, indefinite
//! system **once** at the optimum; each [`QpSensitivity::parametric_step`]
//! is then a single back-substitution, so a parametric sweep costs one
//! solve per query (the build-once / solve-many idiom of the NLP
//! `Solver`). A tiny static regularization `δ` (the QP solver's own `reg`,
//! default `1e-10`) is placed on the diagonal so the indefinite factor is
//! stable.
//!
//! # Near-singular (near-LICQ) KKT: refinement + a conditioning diagnostic
//!
//! When the active-constraint gradients are *nearly* rank-deficient (LICQ
//! almost fails — e.g. two nearly-parallel equality rows) the KKT matrix is
//! near-singular. A single regularized back-solve then **over-damps**
//! `dx/db` toward a smooth but badly wrong value, silently, because the
//! static `δ` floors the smallest KKT singular value (gh #284). Two
//! defenses close that gap:
//!
//! 1. **Iterative refinement against the *unregularized* KKT.** Each solve
//!    refines its back-substitution against the true (`δ`-free) KKT matrix,
//!    so the `O(δ)` regularization bias is removed wherever the information
//!    is still present in double precision — recovering LU-quality `dx/db`.
//!    On a well-conditioned KKT the first residual is already at round-off
//!    and refinement is a no-op, so this never perturbs the good cases.
//! 2. **A two-part conditioning diagnostic.**
//!    [`QpSensitivity::kkt_cond_estimate`] is a cheap Hager 1-norm estimate of
//!    `κ₁` of the factored KKT; [`QpSensitivity::ill_conditioned`] fires when it
//!    is huge **or** when the most recent step's refinement residual is large.
//!    The condition estimate alone has a blind spot: it measures the
//!    *regularized* factor, whose smallest singular value is floored at `δ`, so
//!    on a well-scaled `P` (e.g. `P = I`, `‖K‖₁ ≈ O(1)`) it saturates near
//!    `‖K‖₁ / δ` and never reaches its threshold, even when the true KKT is
//!    numerically singular — so a purely near-parallel *constraint* Jacobian
//!    slips past it (gh #328). The per-step residual closes that gap: refinement
//!    against the true KKT *cannot* solve an unrecoverable step, so it stalls at
//!    a large relative residual ([`QpSensitivity::last_step_residual`]), which
//!    fires the flag. Between the two, a caller can always *detect* that `dx/db`
//!    is untrustworthy instead of consuming a silently-damped value — whether
//!    the near-singularity shows up in the condition estimate (badly-scaled `P`)
//!    or only in the stalled residual (well-scaled `P`, near-LICQ constraints).

use crate::activity::{ConvexActivityReport, classify_all, curvature_floor};
use crate::cones::ConeSpec;
use crate::ipm::QpOptions;
use crate::qp::{BOUND_INF, QpProblem, QpSolution, QpStatus, Triplet};
use pounce_common::types::{Index, Number};
use pounce_linalg::symmetric_eigen;
use pounce_linsol::{Factorization, SparseSymLinearSolverInterface};
use pounce_sens_core::backsolver::{BoundRow, SensBacksolver};
use pounce_sens_core::boundcheck::{
    BoundMultiplier, PathSegment, RefineStop, refine_step_onto_bounds, step_along_path,
};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;

/// Group a constraint matrix's triplets by row, so an active-set assembly
/// can read a row's `(col, val)` entries directly. Without this, both the
/// KKT build and the reduced-Hessian assembly re-scanned *all* of `G` once
/// per active row (`O(n_active · nnz(G))`); the grouping is a single
/// `O(nnz(G))` pass and each lookup is then proportional to that row's
/// own nonzeros. `n_rows` is the number of inequality rows (`m_ineq`), so
/// every `t.row` is a valid index.
pub(crate) fn group_rows_by_index(triplets: &[Triplet], n_rows: usize) -> Vec<Vec<(usize, f64)>> {
    let mut rows = vec![Vec::new(); n_rows];
    for t in triplets {
        rows[t.row].push((t.col, t.val));
    }
    rows
}

/// A reason a [`QpSensitivity`] could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SensError {
    /// The solution was not optimal, so the active set is undefined.
    NotOptimal,
    /// The active-set KKT factorization failed (e.g. the active constraint
    /// gradients are rank-deficient, so the parametric step is not unique).
    FactorizationFailed,
    /// A symmetric eigensolve did not converge while forming the reduced
    /// Hessian, so its rank / null-space (and hence the result) cannot be
    /// trusted. Only [`reduced_hessian`](QpSensitivity::reduced_hessian) can
    /// raise this; the parametric step does not eigendecompose.
    EigenFailed,
    /// The solution's inequality block does not complement **row by row**, so
    /// it is not a solution of an orthant-only problem and
    /// [`build`](QpSensitivity::build) must not read it as one.
    ///
    /// [`solve_socp_ipm`](crate::solve_socp_ipm) returns the same
    /// [`QpSolution`] type as [`solve_qp_ipm`](crate::solve_qp_ipm), and the
    /// cone partition travels beside it as a separate `&[ConeSpec]` argument
    /// that `build` never sees. Without this check a solved SOCP is accepted
    /// and every cone row is silently read as an orthant row — a wrong
    /// `dx/db` reported as a good one. A conic solution complements only as a
    /// *block* inner product `⟨s, z⟩ = 0`; row-wise `sᵢ·zᵢ` is generally
    /// nonzero and `z` generally has negative entries, which is what this
    /// detects. Use [`build_conic`](QpSensitivity::build_conic) for a problem
    /// that carries cones.
    NotOrthantComplementary {
        /// The inequality row whose evidence triggered the refusal.
        row: usize,
        /// Which test failed, for a diagnosable message.
        what: &'static str,
    },
    /// The bound refinement could not run — a shape mismatch or a back-solve
    /// failure inside `pounce-sens-core`, carrying that layer's own message.
    Refinement(String),
    /// The problem carries a cone family whose sensitivity is not implemented.
    ///
    /// Raised only by [`build_conic`](QpSensitivity::build_conic). Refusing is
    /// deliberate: the alternative is a number that looks like a derivative and
    /// is not one.
    /// `ConeSpec` is deliberately not `Eq` (it carries `Power(f64)`), so the
    /// family travels as its name rather than the spec — keeping `SensError`
    /// `Eq`, which it has always been.
    UnsupportedCone {
        /// Index into the `&[ConeSpec]` slice.
        block: usize,
        /// The cone family that is not yet supported, e.g. `"SecondOrder"`.
        family: &'static str,
    },
    /// The cone partition handed to
    /// [`build_conic`](QpSensitivity::build_conic) does not cover the
    /// inequality block exactly, so the caller and the builder disagree about
    /// which rows are which.
    ConePartitionMismatch {
        /// Rows the partition accounts for (`Σ ConeSpec::dim`).
        covered: usize,
        /// Rows the problem actually has.
        m_ineq: usize,
    },
}

/// Refuse a solution whose inequality block does not complement row by row.
///
/// `gx` is `G·x` at the solution, so `sᵢ = hᵢ − gxᵢ` is row `i`'s slack. For a
/// nonnegative-orthant row at an optimum, `sᵢ ≥ 0`, `zᵢ ≥ 0` and `sᵢzᵢ = μ ≈ 0`
/// — all three hold *per row*. A conic block satisfies only `⟨s, z⟩ = 0` over
/// the block, so a second-order-cone row on its boundary has `s = (t, u)` with
/// `t = ‖u‖ > 0` and `z ∝ (t, −u)`: the tail entries of `z` are negative and
/// `s₀z₀ = c·t² > 0`. Either signal is decisive.
///
/// # What this cannot catch
///
/// A second-order cone at its **apex** with `z_{1:} = 0` — `s = 0` and
/// `z = (z₀, 0, …, 0)` — passes every test here, because row-wise it is
/// indistinguishable from a degenerate orthant block, which is a legitimate
/// input. That case needs the cone partition, which is what
/// [`QpSensitivity::build_conic`] takes. This guard is the safety net for a
/// caller who never mentions cones; it is not a substitute for telling the
/// builder what the problem is.
fn check_orthant_complementarity(
    prob: &QpProblem,
    sol: &QpSolution,
    gx: &[f64],
    primal_scale: f64,
    dual_scale: f64,
) -> Result<(), SensError> {
    let sign_tol = ORTHANT_GUARD_REL * dual_scale;
    let slack_tol = ORTHANT_GUARD_REL * primal_scale;
    let comp_tol = ORTHANT_GUARD_REL * primal_scale * dual_scale;
    for (i, (&h_i, &gx_i)) in prob.h.iter().zip(gx.iter()).enumerate() {
        let s = h_i - gx_i;
        let z = sol.z[i];
        // A dual entry that is negative by more than round-off is not an
        // orthant multiplier at all — the SOC dual's tail is the common case.
        if z < -sign_tol {
            return Err(SensError::NotOrthantComplementary {
                row: i,
                what: "the inequality multiplier is negative, which the nonnegative orthant \
                       forbids; a second-order or exponential cone's dual has negative entries",
            });
        }
        // Likewise a slack outside the orthant.
        if s < -slack_tol {
            return Err(SensError::NotOrthantComplementary {
                row: i,
                what: "the inequality slack is negative beyond the solve's own tolerance, so \
                       this row is not a satisfied orthant row",
            });
        }
        if (s * z).abs() > comp_tol {
            return Err(SensError::NotOrthantComplementary {
                row: i,
                what: "slack and multiplier are both away from zero, so the row does not \
                       complement; a conic block complements only as a block inner product",
            });
        }
    }
    Ok(())
}

/// The `ConeSpec` variant name, for [`SensError::UnsupportedCone`].
fn cone_family(spec: &ConeSpec) -> &'static str {
    match spec {
        ConeSpec::Nonneg(_) => "Nonneg",
        ConeSpec::SecondOrder(_) => "SecondOrder",
        ConeSpec::Exponential => "Exponential",
        ConeSpec::Power(_) => "Power",
        ConeSpec::Psd(_) => "Psd",
    }
}

/// Post-optimal sensitivity for a solved convex QP.
///
/// Holds the factored active-set KKT system at the optimum. Build it once
/// from a [`QpProblem`] and its [`QpSolution`], then call
/// [`parametric_step`](Self::parametric_step) for each parameter
/// perturbation — the factorization is reused across queries.
pub struct QpSensitivity {
    n: usize,
    m_eq: usize,
    /// KKT dimension `n + m_eq + n_active`.
    dim: usize,
    /// Shared so [`backsolver`](QpSensitivity::backsolver) can hand the same
    /// factorization to the core's machinery — see [`QpKktBacksolver`].
    fact: Rc<RefCell<Factorization>>,
    /// Problem data, retained for the reduced-Hessian projection.
    prob: QpProblem,
    /// Active inequality rows (indices into `G`).
    active_ineq: Vec<usize>,
    /// Variables whose bound is active (one row each).
    active_bound_vars: Vec<usize>,
    /// The solution this sensitivity was built at, retained because the bound
    /// refinement measures the step against the base point and the duality
    /// measure needs the slacks.
    base_x: Vec<f64>,
    base_z: Vec<f64>,
    base_z_lb: Vec<f64>,
    base_z_ub: Vec<f64>,
    /// The same set, carrying which side is active: `true` = lower bound.
    /// The orientation is what `assemble_kkt` signs the row by and what the
    /// recovered `dz_a` block is read against.
    active_bounds: Vec<(usize, bool)>,
    /// Each active bound's multiplier at the base point, oriented so it is the
    /// non-negative multiplier of the row `assemble_kkt` emitted. Needed by the
    /// release half of fix-relax, which has to move a released multiplier onto
    /// its variable's `x` row.
    bound_base_mult: Vec<f64>,
    /// `true` when this is a pure LP (`P = 0`) whose solve did not run
    /// crossover — see [`lp_without_crossover`](QpSensitivity::lp_without_crossover).
    lp_without_crossover: bool,
    /// Inequality rows at which strict complementarity fails (gh #219).
    weakly_active_ineq: Vec<usize>,
    /// Variables whose bound is weakly active.
    weakly_active_bound_vars: Vec<usize>,
    /// Lower-triangle KKT pattern (1-based), shared by the factored
    /// (regularized) matrix and the unregularized values below.
    kkt_airn: Vec<Index>,
    kkt_ajcn: Vec<Index>,
    /// Unregularized KKT values (the `δ`-free matrix) for the refinement
    /// residual — see [`solve_refined`] (gh #284).
    kkt_vals_true: Vec<f64>,
    /// Hager 1-norm estimate of `κ₁` of the factored KKT (gh #284).
    kkt_cond_estimate: f64,
    /// Relative KKT residual of the most recent parametric step, or `None`
    /// before any step has been taken (gh #284).
    last_residual: Option<f64>,
    /// Reusable iterative-refinement buffers.
    ir_scratch: IrScratch,
    /// Factored (regularized) KKT values; a release starts from these.
    kkt_vals_reg: Rc<Vec<f64>>,
    /// Per active bound, the value slots a release neutralizes.
    release_slots: Rc<Vec<(usize, usize)>>,
    /// One spare linear-solver instance, drawn from the caller's factory at
    /// build, reserved for the released system.
    ///
    /// Storing an *instance* rather than the factory is what keeps
    /// [`build`](QpSensitivity::build)'s signature unchanged — boxing the
    /// factory would have needed an `F: 'static` bound on a published method.
    /// One instance is enough because at most one released `Factorization` is
    /// ever created: a different released set refactors it in place, the
    /// sparsity pattern being identical.
    release_backend: Rc<RefCell<Option<Box<dyn SparseSymLinearSolverInterface>>>>,
}

/// Relative threshold below which a slack or a multiplier counts as zero for
/// the weak-activity screen (see [`QpSensitivity::weakly_active_ineq`]).
///
/// Deliberately loose, because it is the *conjunction* that carries the signal:
/// a constraint must be binding in the primal **and** carry a negligible dual
/// at the same time. Either alone is ordinary — every active constraint has
/// zero slack, every inactive one has zero multiplier — while both at once is
/// exactly the non-strict complementarity that makes `dx/db` one-sided.
///
/// The magnitude is set by how these quantities actually behave. At a
/// degenerate optimum both collapse together at roughly `√tol`: on gh #219's
/// QP the multiplier and slack measure `(3.8e-5, 1.7e-4)` at `tol = 1e-8`,
/// `(2.0e-7, 9.0e-7)` at `1e-12`, and `(2.9e-8, 1.3e-7)` at `1e-14` — their
/// ratio pinned near 0.22 across six orders of magnitude. A threshold tight
/// enough to look precise would simply miss the default-tolerance case, which
/// is the one users hit.
const WEAK_ACTIVE_REL: f64 = 1e-3;

/// Relative margin for the orthant guard's three row-wise tests
/// ([`SensError::NotOrthantComplementary`]).
///
/// Deliberately loose. The guard separates two *categorically* different
/// inputs, not two nearby numbers: a converged orthant row has `sᵢ·zᵢ ≈ μ`,
/// which is at round-off relative to `‖s‖∞·‖z‖∞`, while a second-order-cone
/// row on its boundary has `s₀z₀ = c·s₀²`, which is `O(1)` on the same scale.
///
/// # The measured populations on each side
///
/// Worst row of each fixture, as the ratio this constant is compared against:
///
/// | fixture | `|sᵢzᵢ| / (‖s‖∞‖z‖∞)` | `max(−zᵢ) / ‖z‖∞` | verdict |
/// |---|---|---|---|
/// | `SecondOrder(3)` through `solve_socp_ipm` | **1.3e-1** | **1.2e-1** | must refuse |
/// | convex QP, one active inequality | 1.7e-9 | 0 | must accept |
/// | weakly-active (non-strictly-complementary) QP | 6.5e-9 | 0 | must accept |
///
/// The gap is **7.9 orders wide** and `1e-4` sits near its centre — about five
/// orders below anything that must be refused and about five above anything
/// that must be accepted. The weakly-active row is the one worth checking,
/// since non-strict complementarity is the accept-side case that comes closest
/// to the boundary, and it is still five orders clear.
///
/// The asymmetry is deliberate: a false *positive* is the failure that would
/// matter, because this guard sits on the path of every convex sensitivity
/// build, so the threshold is placed far from the accept side.
const ORTHANT_GUARD_REL: f64 = 1e-4;

/// Above this 1-norm condition estimate of the factored KKT the parametric
/// step is reported [`ill_conditioned`](QpSensitivity::ill_conditioned).
///
/// Calibrated against gh #284's near-LICQ sweep. With the static `δ = 1e-10`
/// flooring the smallest KKT singular value, `κ₁` saturates near `1e16` on a
/// numerically singular KKT while the genuinely well-conditioned sensitivity
/// cases sit at `κ₁ ≈ 3–8e9`. Iterative refinement (see the module doc)
/// recovers a correct `dx/db` up to `κ₁ ≈ 6e13`; past that the information is
/// below the double-precision floor and refinement cannot help. The threshold
/// sits in the wide gap between those regimes, so it fires exactly on the
/// unrecoverable cases and stays quiet on every case refinement rescues — no
/// false alarm on the well-conditioned equality-only or active-set paths.
///
/// This condition estimate is a *build-time* screen, and by itself it has a
/// blind spot: it is the `κ₁` of the **regularized** factor, whose smallest
/// singular value is floored at `δ`, so on a *well-scaled* `P` (e.g. `P = I`,
/// `‖K‖₁ ≈ O(1)`) it saturates near `‖K‖₁ / δ ≈ 3e10` no matter how nearly
/// parallel the active rows become — never reaching this threshold even when
/// the true KKT is numerically singular (gh #328). The per-step residual gate
/// below closes that gap.
const KKT_ILL_CONDITIONED_THRESHOLD: f64 = 1e14;

/// Relative KKT residual above which the most recent parametric step is treated
/// as *unreliable*, so [`ill_conditioned`](QpSensitivity::ill_conditioned)
/// fires on it (gh #328).
///
/// This is the companion signal to the build-time condition estimate and covers
/// its blind spot. When the active-constraint Jacobian is near-LICQ but `P` is
/// well scaled, the saturating [`KKT_ILL_CONDITIONED_THRESHOLD`] never trips,
/// yet iterative refinement against the true (`δ`-free) KKT *cannot* solve the
/// step — it stalls at a large relative residual (`≈ 3e-2` at `κ(A) ≈ 2e5`,
/// `≈ 0.25` at `κ(A) ≈ 2e7`). A well-solved step, by contrast, refines to
/// round-off (`≲ 1e-8`). The two regimes are separated by many orders of
/// magnitude, so this threshold sits comfortably in the gap: it flags exactly
/// the steps whose returned `dx/db` does not satisfy the true KKT, and stays
/// quiet on every accurately recovered step.
const STEP_UNRELIABLE_RESIDUAL: f64 = 1e-6;

/// Iterative-refinement passes for a parametric step (mirrors the HSDE
/// solve's `IR_MAX_PASSES`). A handful suffices: refinement against the
/// unregularized KKT converges geometrically until it hits the near-singular
/// floor, where it stagnates and stops.
/// Margin below which a bound multiplier counts as driven negative by a step.
///
/// The NLP arm derives this from `bound_relax_factor`, since there the solve's
/// own bound relaxation sets the scale. The convex path does not relax bounds
/// in that sense, so this is the floor that derivation bottoms out at: `1e-9`,
/// which upstream also uses when `bound_relax_factor` is unset or unreadable.
const RELEASE_FLOOR: f64 = 1e-9;

const IR_MAX_PASSES: usize = 5;

/// Relative-residual target below which refinement stops early (the KKT step
/// is solved to working precision).
const IR_RELTOL: f64 = 1e-12;

/// Hager/Higham 1-norm power iterations for the `‖K⁻¹‖₁` estimate. Five is
/// LAPACK's `dlacon` default and is more than enough here (the estimate
/// matched the exact 1-norm condition on gh #284's sweep).
const HAGER_ITERS: usize = 5;

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// Symmetric matvec `y ← K x` for lower-triangle KKT triplets (`airn`/`ajcn`
/// 1-based, `row ≥ col`). Each strictly-lower entry hits both `y[i]` and
/// `y[j]`; the diagonal once. Mirrors the HSDE solver's `kkt_matvec` — used
/// to form the residual `rhs − K u` that drives refinement.
fn kkt_matvec(airn: &[Index], ajcn: &[Index], vals: &[f64], x: &[f64], y: &mut [f64]) {
    for v in y.iter_mut() {
        *v = 0.0;
    }
    for k in 0..vals.len() {
        let i = (airn[k] - 1) as usize;
        let j = (ajcn[k] - 1) as usize;
        let v = vals[k];
        y[i] += v * x[j];
        if i != j {
            y[j] += v * x[i];
        }
    }
}

/// 1-norm `‖K‖₁ = maxⱼ Σᵢ |Kᵢⱼ|` of a symmetric matrix from its lower-triangle
/// triplets (equal to `‖K‖∞`). Each off-diagonal entry contributes to both its
/// row and column absolute sums.
fn one_norm_sym(dim: usize, airn: &[Index], ajcn: &[Index], vals: &[f64]) -> f64 {
    let mut colsum = vec![0.0_f64; dim];
    for k in 0..vals.len() {
        let i = (airn[k] - 1) as usize;
        let j = (ajcn[k] - 1) as usize;
        let a = vals[k].abs();
        colsum[i] += a;
        if i != j {
            colsum[j] += a;
        }
    }
    colsum.into_iter().fold(0.0_f64, f64::max)
}

/// Hager/Higham lower-bound estimate of `‖K⁻¹‖₁` using only back-solves
/// against the cached factor. `K` is symmetric, so `K⁻ᵀ = K⁻¹` and a single
/// factor drives both half-steps. Returns `∞` if a back-solve fails (the
/// caller then reports an infinite condition estimate — the safe direction).
fn estimate_inv_norm1(fact: &mut Factorization, dim: usize) -> f64 {
    if dim == 0 {
        return 0.0;
    }
    let mut x = vec![1.0 / dim as f64; dim];
    let mut est = 0.0_f64;
    let mut prev_j = usize::MAX;
    for _ in 0..HAGER_ITERS {
        // y = K⁻¹ x; the 1-norm of y is the running estimate of ‖K⁻¹‖₁.
        let mut y = x.clone();
        if fact.solve_one(&mut y).is_err() {
            return f64::INFINITY;
        }
        est = y.iter().map(|v| v.abs()).sum();
        // z = K⁻¹ sign(y)  (K symmetric ⇒ K⁻ᵀ = K⁻¹).
        let mut z: Vec<f64> = y
            .iter()
            .map(|v| if *v >= 0.0 { 1.0 } else { -1.0 })
            .collect();
        if fact.solve_one(&mut z).is_err() {
            return f64::INFINITY;
        }
        let (j, zmax) = z
            .iter()
            .enumerate()
            .fold((0usize, 0.0_f64), |(bi, bm), (i, v)| {
                if v.abs() > bm { (i, v.abs()) } else { (bi, bm) }
            });
        let ztx: f64 = z.iter().zip(&x).map(|(a, b)| a * b).sum();
        // Higham's stopping test: no coordinate of z beats the current
        // direction, or we would revisit a unit vector (a cycle).
        if zmax <= ztx || j == prev_j {
            break;
        }
        prev_j = j;
        x = vec![0.0; dim];
        x[j] = 1.0;
    }
    est
}

/// Solve `K u = rhs` against the cached (regularized) factor, then refine `u`
/// against the **unregularized** KKT triplets `(airn, ajcn, vals_true)` to
/// strip the `O(δ)` regularization bias. Overwrites `rhs` with `u` and returns
/// the final **relative** residual `‖rhs₀ − K u‖∞ / ‖rhs₀‖∞` — the reliability
/// signal a caller reads back as
/// [`last_step_residual`](QpSensitivity::last_step_residual).
///
/// The residual is normalized by `‖rhs₀‖∞` (not `1 + ‖rhs₀‖∞`) so it is a true
/// *relative* residual, invariant to the magnitude of the perturbation. The
/// `1 +` floor of the earlier form (gh #284) silently masked a failed solve
/// whenever the perturbation was small: a step scaled by e.g. `1e-6` shrank
/// both `‖r‖` and `‖rhs₀‖` by `1e-6`, but the `1 +` left the denominator at
/// `≈ 1`, so a fully over-damped step (true relative residual `≈ 0.25`) read
/// back as `≈ 2.5e-7` — small enough to look solved (gh #328).
/// The assembled lower triangle of the active-set KKT, in the 1-based triplet
/// form [`Factorization`] takes.
///
/// `vals_reg` is what gets factored (the `δ`-stabilized, indefinite matrix);
/// `vals_true` is the same pattern with the regularization removed and is what
/// iterative refinement measures its residual against (gh #284). They share one
/// sparsity pattern by construction, which is what lets a release re-use the
/// symbolic factor.
struct KktPattern {
    airn: Vec<Index>,
    ajcn: Vec<Index>,
    vals_true: Vec<f64>,
    vals_reg: Vec<f64>,
    dim: usize,
}

/// Assemble the active-set KKT
///
/// ```text
///   ⎡ P    Aᵀ   B_aᵀ ⎤
///   ⎢ A    0    0    ⎥
///   ⎣ B_a  0    0    ⎦
/// ```
///
/// `B_a` stacks the active inequality rows of `G` then one row per active
/// variable bound, in the order given. Every diagonal slot is materialized
/// (even where `P` is zero) so `vals_true` and `vals_reg` share one pattern.
///
/// # Bound-row orientation
///
/// `active_bounds` carries `(variable, is_lower)`, and the sign matters. In the
/// `Gx ≤ h` orientation the convex form uses, a lower bound `lb ≤ xⱼ` is the row
/// `−eⱼᵀ` (its multiplier is `z_lb ≥ 0`, entering stationarity as `−z_lb`), and
/// an upper bound `xⱼ ≤ ub` is `+eⱼᵀ`. Emitting `+1` for both — as this code did
/// before the sign was needed — is invisible while the active block's
/// right-hand side is zero, because `eⱼᵀ dx = 0` and `−eⱼᵀ dx = 0` are the same
/// constraint. It stops being invisible the moment anything *reads* the
/// recovered multiplier block `dz_a`, where the lower-bound entries come back
/// negated. `the_recovered_bound_multipliers_carry_the_solutions_sign` pins it.
fn assemble_kkt(
    prob: &QpProblem,
    active_ineq: &[usize],
    active_bounds: &[(usize, bool)],
    reg: f64,
) -> KktPattern {
    let n = prob.n;
    let m_eq = prob.m_eq();
    let n_active = active_ineq.len() + active_bounds.len();
    let dim = n + m_eq + n_active;

    let mut entries: BTreeMap<(usize, usize), (f64, f64)> = BTreeMap::new();
    let mut add = |r: usize, c: usize, v: f64, reg_off: f64| {
        let (r, c) = if r >= c { (r, c) } else { (c, r) };
        let e = entries.entry((r, c)).or_insert((0.0, 0.0));
        e.0 += v;
        e.1 += reg_off;
    };

    // (x,x): P, with +δ on the diagonal for the factor only.
    for t in &prob.p_lower {
        add(t.row, t.col, t.val, 0.0);
    }
    for i in 0..n {
        add(i, i, 0.0, reg);
    }
    // (y,x): A; (y,y): −δI (factor only).
    for t in &prob.a {
        add(n + t.row, t.col, t.val, 0.0);
    }
    for i in 0..m_eq {
        add(n + i, n + i, 0.0, -reg);
    }
    // Active-row block `B_a`: active inequality rows, then active bound rows.
    let abase = n + m_eq;
    let g_rows = group_rows_by_index(&prob.g, prob.m_ineq());
    for (k, &i) in active_ineq.iter().enumerate() {
        for &(col, val) in &g_rows[i] {
            add(abase + k, col, val, 0.0);
        }
    }
    for (k, &(j, is_lower)) in active_bounds.iter().enumerate() {
        let sign = if is_lower { -1.0 } else { 1.0 };
        add(abase + active_ineq.len() + k, j, sign, 0.0);
    }
    for k in 0..n_active {
        add(abase + k, abase + k, 0.0, -reg);
    }

    let nnz = entries.len();
    let mut airn = Vec::with_capacity(nnz);
    let mut ajcn = Vec::with_capacity(nnz);
    let mut vals_true = Vec::with_capacity(nnz);
    let mut vals_reg = Vec::with_capacity(nnz);
    for ((r, c), (v_true, v_reg_off)) in entries {
        airn.push((r + 1) as Index);
        ajcn.push((c + 1) as Index);
        vals_true.push(v_true);
        vals_reg.push(v_true + v_reg_off);
    }
    KktPattern {
        airn,
        ajcn,
        vals_true,
        vals_reg,
        dim,
    }
}

/// For each active bound, the two value-array slots a release has to touch:
/// the `±1` coupling the multiplier row to its variable, and that row's own
/// diagonal.
///
/// Precomputed once at build so a release is an array rewrite plus a
/// [`Factorization::refactor`] — the sparsity pattern is unchanged by
/// neutralizing a row, so the symbolic factorization is reused.
fn release_slots(
    pat: &KktPattern,
    abase: usize,
    n_ineq_active: usize,
    active_bounds: &[(usize, bool)],
) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(active_bounds.len());
    for (k, &(j, _)) in active_bounds.iter().enumerate() {
        let row = abase + n_ineq_active + k;
        // 1-based in the pattern arrays.
        let (r1, j1) = ((row + 1) as Index, (j + 1) as Index);
        let mut coupling = usize::MAX;
        let mut diagonal = usize::MAX;
        for (idx, (&a, &b)) in pat.airn.iter().zip(pat.ajcn.iter()).enumerate() {
            if a == r1 && b == j1 {
                coupling = idx;
            } else if a == r1 && b == r1 {
                diagonal = idx;
            }
        }
        debug_assert!(
            coupling != usize::MAX && diagonal != usize::MAX,
            "every active bound row has a coupling entry and a diagonal"
        );
        out.push((coupling, diagonal));
    }
    out
}

/// Reusable buffers for [`solve_refined`].
///
/// Hoisted out of the function because [`QpKktBacksolver::solve`] is called
/// once per candidate row per refinement pass — `refine_step_onto_bounds` makes
/// `k + 1` back-solves per pass — so a per-call `Vec` allocation turns `O(1)`
/// into `O(k)` for no reason.
#[derive(Default, Clone)]
struct IrScratch {
    b: Vec<f64>,
    r: Vec<f64>,
}

impl IrScratch {
    fn ready(&mut self, dim: usize) {
        self.b.clear();
        self.b.resize(dim, 0.0);
        self.r.clear();
        self.r.resize(dim, 0.0);
    }
}

fn solve_refined(
    fact: &mut Factorization,
    airn: &[Index],
    ajcn: &[Index],
    vals_true: &[f64],
    rhs: &mut [f64],
    scratch: &mut IrScratch,
) -> Result<f64, ()> {
    let dim = rhs.len();
    scratch.ready(dim);
    let b = &mut scratch.b;
    b.copy_from_slice(rhs);
    let b = &*b;
    fact.solve_one(rhs).map_err(|_| ())?;
    // True relative residual: divide by ‖rhs₀‖, flooring only a genuinely zero
    // RHS (whose exact solution is the zero step, residual zero) to avoid 0/0.
    let bnorm = {
        let bn = inf_norm(&b);
        if bn > 0.0 { bn } else { 1.0 }
    };
    let r = &mut scratch.r;
    let mut res = f64::INFINITY;
    for _ in 0..IR_MAX_PASSES {
        kkt_matvec(airn, ajcn, vals_true, rhs, r);
        for k in 0..dim {
            r[k] = b[k] - r[k];
        }
        let new_res = inf_norm(r) / bnorm;
        // Stop when solved to working precision, or when refinement stops
        // making progress — the latter is the near-singular floor and its
        // residual is exactly the "step is unreliable" signal.
        if new_res <= IR_RELTOL || new_res >= res {
            res = new_res;
            break;
        }
        res = new_res;
        fact.solve_one(r).map_err(|_| ())?;
        for k in 0..dim {
            rhs[k] += r[k];
        }
    }
    Ok(res)
}

impl QpSensitivity {
    /// Build the active-set sensitivity for `sol` (a solution of `prob`).
    ///
    /// The active set is read from the dual certificate: an inequality row
    /// `i` is active when `zᵢ > active_tol`, a lower bound on `xⱼ` when
    /// `z_lbⱼ > active_tol`, an upper bound when `z_ubⱼ > active_tol`. A
    /// good default for `active_tol` is `1e-7` (see
    /// [`build_default`](Self::build_default)).
    ///
    /// Returns [`SensError::NotOptimal`] unless `sol` is `Optimal` or
    /// `OptimalInaccurate` (see gh #880), or
    /// [`SensError::FactorizationFailed`] if the active-set KKT is singular.
    pub fn build<F>(
        prob: &QpProblem,
        sol: &QpSolution,
        opts: &QpOptions,
        active_tol: f64,
        mut make_backend: F,
    ) -> Result<Self, SensError>
    where
        F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
    {
        // gh #880 admits `OptimalInaccurate` here. That status now also means
        // "the `σ` cascade could not certify this point to `tol`", and such a
        // point previously arrived as a clean `Optimal` and built a
        // sensitivity object. The issue is about a wrong *label*; narrowing
        // what the library will do for that population would be a second,
        // unrelated change, and it is the one the repo has already ruled on
        // in the other direction — `_curve_fit.py` (gh #119 / #123) records
        // that treating `Solved_To_Acceptable_Level` as a non-success
        // "reported `success=False` at a verified optimum ... and callers
        // gating on `.success` discarded valid fits". The derivative at a
        // less accurate point is less accurate, exactly as it was before.
        if !matches!(sol.status, QpStatus::Optimal | QpStatus::OptimalInaccurate) {
            return Err(SensError::NotOptimal);
        }
        let n = prob.n;
        let m_eq = prob.m_eq();
        let reg = opts.reg;

        // Active set: which inequality rows and which variable bounds bind.
        let active_ineq: Vec<usize> = (0..prob.m_ineq())
            .filter(|&i| sol.z[i] > active_tol)
            .collect();
        // A bound contributes one row per variable, oriented by which side is
        // active: `−eⱼᵀ` for a lower bound, `+eⱼᵀ` for an upper one (see
        // `assemble_kkt`). A variable active on both sides is a fixed variable
        // (`lb == ub`); it gets one row, and the larger multiplier decides the
        // orientation, which is the side the solve actually leaned on.
        let active_bounds: Vec<(usize, bool)> = (0..n)
            .filter(|&j| sol.z_lb[j] > active_tol || sol.z_ub[j] > active_tol)
            .map(|j| (j, sol.z_lb[j] >= sol.z_ub[j]))
            .collect();
        let active_bound_vars: Vec<usize> = active_bounds.iter().map(|&(j, _)| j).collect();
        let n_active = active_ineq.len() + active_bounds.len();
        let dim = n + m_eq + n_active;

        // Weak activity (gh #219): binding in the primal *and* negligible in
        // the dual, i.e. non-strict complementarity. Classical post-optimal
        // sensitivity (Fiacco) assumes this never happens; where it does, the
        // perturbation changes the active set and `dx/db` is a one-sided
        // derivative with another, equally valid, value on the other side.
        //
        // Both tests are relative to the natural scale of their own quantity,
        // so the screen is invariant to a rescaling of the problem data.
        let inf_norm = |v: &[f64]| v.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        let dual_scale = inf_norm(&sol.y)
            .max(inf_norm(&sol.z))
            .max(inf_norm(&sol.z_lb))
            .max(inf_norm(&sol.z_ub))
            .max(1.0);
        let mut gx = vec![0.0; prob.m_ineq()];
        prob.g_mul(&sol.x, &mut gx);
        let primal_scale = inf_norm(&prob.h).max(inf_norm(&gx)).max(1.0);

        // Before reading a single row as an orthant row, check that it is one.
        // `solve_socp_ipm` hands back this very `QpSolution` type and the cone
        // partition travels beside it, so without this a solved SOCP is
        // accepted here and answered wrongly in silence.
        check_orthant_complementarity(prob, sol, &gx, primal_scale, dual_scale)?;

        let dual_zero = WEAK_ACTIVE_REL * dual_scale;
        let primal_zero = WEAK_ACTIVE_REL * primal_scale;

        let weakly_active_ineq: Vec<usize> = (0..prob.m_ineq())
            .filter(|&i| (prob.h[i] - gx[i]).abs() <= primal_zero && sol.z[i] <= dual_zero)
            .collect();
        let x_scale = inf_norm(&sol.x).max(1.0);
        let bound_zero = WEAK_ACTIVE_REL * x_scale;
        let weakly_active_bound_vars: Vec<usize> = (0..n)
            .filter(|&j| {
                // `lb`/`ub` may be empty (= unbounded), and a "present" bound
                // is one inside the `BOUND_INF` sentinel band, matching
                // `QpProblem::has_bounds`.
                let (lb, ub) = (prob.lb_of(j), prob.ub_of(j));
                let lb_weak = lb > -BOUND_INF
                    && (sol.x[j] - lb).abs() <= bound_zero
                    && sol.z_lb[j] <= dual_zero;
                let ub_weak = ub < BOUND_INF
                    && (ub - sol.x[j]).abs() <= bound_zero
                    && sol.z_ub[j] <= dual_zero;
                lb_weak || ub_weak
            })
            .collect();

        let pat = assemble_kkt(prob, &active_ineq, &active_bounds, reg);
        let release_slots = release_slots(&pat, n + m_eq, active_ineq.len(), &active_bounds);
        let KktPattern {
            airn: kkt_airn,
            ajcn: kkt_ajcn,
            vals_true: kkt_vals_true,
            vals_reg: values_reg,
            ..
        } = pat;
        // Kept for the release path, which starts from the unreleased values.
        let values_reg_kept = values_reg.clone();

        // 1-norm of the factored (regularized) KKT, for the condition estimate.
        let kkt_norm1 = one_norm_sym(dim, &kkt_airn, &kkt_ajcn, &values_reg);

        let mut fact = Factorization::new(
            dim as Index,
            kkt_airn.clone(),
            kkt_ajcn.clone(),
            values_reg,
            make_backend(),
        )
        .map_err(|_| SensError::FactorizationFailed)?;

        // Hager estimate of κ₁ = ‖K‖₁·‖K⁻¹‖₁ (gh #284). Reuses the factor, so
        // it costs only a handful of back-solves.
        let inv_norm1 = estimate_inv_norm1(&mut fact, dim);
        let kkt_cond_estimate = kkt_norm1 * inv_norm1;

        // Oriented base multipliers, in the row order `assemble_kkt` used.
        let bound_base_mult: Vec<f64> = active_bounds
            .iter()
            .map(|&(j, is_lower)| if is_lower { sol.z_lb[j] } else { sol.z_ub[j] })
            .collect();
        let lp_without_crossover = prob.p_lower.is_empty() && !opts.crossover;

        Ok(QpSensitivity {
            n,
            m_eq,
            base_x: sol.x.clone(),
            base_z: sol.z.clone(),
            base_z_lb: sol.z_lb.clone(),
            base_z_ub: sol.z_ub.clone(),
            active_bounds,
            bound_base_mult,
            lp_without_crossover,
            dim,
            prob: prob.clone(),
            active_ineq,
            active_bound_vars,
            weakly_active_ineq,
            weakly_active_bound_vars,
            kkt_airn,
            kkt_ajcn,
            kkt_vals_true,
            kkt_cond_estimate,
            last_residual: None,
            ir_scratch: IrScratch::default(),
            kkt_vals_reg: Rc::new(values_reg_kept),
            release_slots: Rc::new(release_slots),
            release_backend: Rc::new(RefCell::new(Some(make_backend()))),
            fact: Rc::new(RefCell::new(fact)),
        })
    }

    /// [`build`](Self::build) with the QP's default options and an active-set
    /// tolerance of `1e-7`.
    pub fn build_default<F>(
        prob: &QpProblem,
        sol: &QpSolution,
        make_backend: F,
    ) -> Result<Self, SensError>
    where
        F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
    {
        Self::build(prob, sol, &QpOptions::default(), 1e-7, make_backend)
    }

    /// [`build`](Self::build) for a problem that carries a cone partition —
    /// the entry point for a solution produced by
    /// [`solve_socp_ipm`](crate::solve_socp_ipm).
    ///
    /// `cones` is the same slice handed to the solve; its blocks stack in
    /// order to cover the `m_ineq` inequality rows.
    ///
    /// An all-[`ConeSpec::Nonneg`] partition *is* the orthant problem, so it
    /// delegates to [`build`](Self::build) and behaves identically. Every other
    /// cone family currently returns [`SensError::UnsupportedCone`].
    ///
    /// The refusal is the feature. A cone's active object is not a set of rows
    /// but the tangent/normal decomposition of the face its slack sits on, and
    /// reading its rows as orthant rows produces a plausible number that is not
    /// a derivative. Refusing names the gap; answering hides it.
    pub fn build_conic<F>(
        prob: &QpProblem,
        cones: &[ConeSpec],
        sol: &QpSolution,
        opts: &QpOptions,
        active_tol: f64,
        make_backend: F,
    ) -> Result<Self, SensError>
    where
        F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
    {
        // The partition must cover the inequality block exactly, or the caller
        // and the builder disagree about which rows are which. A short
        // partition is the dangerous direction: `[Nonneg(2)]` on a problem
        // whose rows 2.. are really a cone would pass the family check below
        // and hand a conic solution to the orthant path. `build`'s row-wise
        // guard would still catch that, but a caller error deserves to be
        // reported as one rather than as a complementarity failure.
        let covered: usize = cones.iter().map(ConeSpec::dim).sum();
        if covered != prob.m_ineq() {
            return Err(SensError::ConePartitionMismatch {
                covered,
                m_ineq: prob.m_ineq(),
            });
        }
        for (block, spec) in cones.iter().enumerate() {
            if !matches!(spec, ConeSpec::Nonneg(_)) {
                return Err(SensError::UnsupportedCone {
                    block,
                    family: cone_family(spec),
                });
            }
        }
        Self::build(prob, sol, opts, active_tol, make_backend)
    }

    /// First-order primal step `dx ≈ x*(b + Δb) − x*(b)` for a perturbation
    /// of the **equality right-hand side** `b`, the direct QP analog of
    /// sIPOPT's "pin a constraint, perturb its value". Constraint
    /// `pin_constraint_indices[k]` (an index into `b`) is perturbed by
    /// `deltas[k]`; all others are held fixed.
    ///
    /// Returns the length-`n` primal sensitivity, so `x* + dx` predicts the
    /// solution of the perturbed QP (exact to first order while the active
    /// set is unchanged). The factorization is reused, so repeated calls
    /// (e.g. a continuation sweep) cost one back-substitution each.
    ///
    /// # Panics
    ///
    /// Panics if `pin_constraint_indices` and `deltas` differ in length, or
    /// if any pin index is `≥ m_eq`.
    pub fn parametric_step(
        &mut self,
        pin_constraint_indices: &[usize],
        deltas: &[f64],
    ) -> Vec<f64> {
        assert_eq!(
            pin_constraint_indices.len(),
            deltas.len(),
            "pin_constraint_indices and deltas must have equal length"
        );
        let mut db = vec![0.0; self.m_eq];
        for (&i, &d) in pin_constraint_indices.iter().zip(deltas) {
            assert!(
                i < self.m_eq,
                "pin constraint index {i} out of range (m_eq = {})",
                self.m_eq
            );
            db[i] += d;
        }
        self.step_from_db(&db)
    }

    /// Primal sensitivity for a full equality-RHS perturbation `db` (length
    /// `m_eq`): solves the active-set KKT with right-hand side `[0; db; 0]`
    /// and returns `dx = step[0..n]`.
    ///
    /// The back-solve is refined against the **unregularized** KKT
    /// ([`solve_refined`]) so the `O(δ)` regularization bias is stripped
    /// wherever the information survives in double precision; the achieved
    /// relative residual is recorded for
    /// [`last_step_residual`](Self::last_step_residual) (gh #284).
    pub fn step_from_db(&mut self, db: &[f64]) -> Vec<f64> {
        assert_eq!(db.len(), self.m_eq, "db must have length m_eq");
        let mut rhs = vec![0.0 as Number; self.dim];
        rhs[self.n..self.n + self.m_eq].copy_from_slice(db);
        // A singular factor would have been caught at build; a back-solve
        // failure here is not recoverable, so surface a zero step.
        let mut fact = self.fact.borrow_mut();
        match solve_refined(
            &mut fact,
            &self.kkt_airn,
            &self.kkt_ajcn,
            &self.kkt_vals_true,
            &mut rhs,
            &mut self.ir_scratch,
        ) {
            Ok(res) => self.last_residual = Some(res),
            Err(()) => return vec![0.0; self.n],
        }
        rhs.truncate(self.n);
        rhs
    }

    /// The full compound KKT step for an equality-RHS perturbation `db`,
    /// length [`kkt_dim`](Self::kkt_dim) rather than truncated to `n`.
    ///
    /// [`step_from_db`](Self::step_from_db) returns the primal block; the
    /// bound refinement needs the whole vector, because the multiplier rows are
    /// what tell it which bounds the step drives negative.
    fn full_step_from_db(&mut self, db: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
        assert_eq!(db.len(), self.m_eq, "db must have length m_eq");
        let mut rhs = vec![0.0 as Number; self.dim];
        rhs[self.n..self.n + self.m_eq].copy_from_slice(db);
        let rhs_plain = rhs.clone();
        let mut fact = self.fact.borrow_mut();
        match solve_refined(
            &mut fact,
            &self.kkt_airn,
            &self.kkt_ajcn,
            &self.kkt_vals_true,
            &mut rhs,
            &mut self.ir_scratch,
        ) {
            Ok(res) => {
                drop(fact);
                self.last_residual = Some(res);
                Some((rhs, rhs_plain))
            }
            Err(()) => None,
        }
    }

    /// The active-set KKT as a [`SensBacksolver`], sharing this object's
    /// factorization.
    ///
    /// This is the seam onto `pounce-sens-core`: anything there that is generic
    /// over the trait works on a convex QP through this handle.
    pub fn backsolver(&self) -> QpKktBacksolver {
        let abase = self.n + self.m_eq + self.active_ineq.len();
        let bound_rows: Vec<BoundRow> = self
            .active_bounds
            .iter()
            .enumerate()
            .map(|(k, &(j, is_lower))| BoundRow {
                row: abase + k,
                var_row: j,
                lower: is_lower,
            })
            .collect();
        QpKktBacksolver {
            fact: Rc::clone(&self.fact),
            airn: Rc::new(self.kkt_airn.clone()),
            ajcn: Rc::new(self.kkt_ajcn.clone()),
            vals_true: Rc::new(self.kkt_vals_true.clone()),
            scratch: Rc::new(RefCell::new(IrScratch::default())),
            dim: self.dim,
            bound_rows: Rc::new(bound_rows),
            last_residual: Rc::new(Cell::new(f64::NAN)),
            vals_reg: Rc::clone(&self.kkt_vals_reg),
            slots: Rc::clone(&self.release_slots),
            base_mult: Rc::new(self.bound_base_mult.clone()),
            released: Rc::new(RefCell::new(None)),
            release_backend: Rc::clone(&self.release_backend),
        }
    }

    /// [`parametric_step`](Self::parametric_step), refined to respect the
    /// variable bounds — the convex arm's `fix_relax`.
    ///
    /// The plain step is a linear predictor and can point outside the box.
    /// Clipping the offending coordinate is cheap but leaves every other one at
    /// its unclipped value, so the result satisfies the bounds and no longer
    /// satisfies the constraints. This instead pins the crossing coordinate at
    /// its bound and re-solves, so the others move with it.
    ///
    /// Returns `(dx, pinned_variables, stop_reason)`. The computation is
    /// `pounce_sens_core::boundcheck::refine_step_onto_bounds` — the same code
    /// the NLP arm runs, reached through [`backsolver`](Self::backsolver)
    /// rather than reimplemented.
    ///
    /// In this phase the refinement **pins only**: a bound whose multiplier the
    /// step drives negative is not released (see
    /// [`QpKktBacksolver::supports_release`]), so a perturbation pulling a
    /// variable off a bound is still held there.
    pub fn parametric_step_bounded(
        &mut self,
        pin_constraint_indices: &[usize],
        deltas: &[f64],
        bound_eps: f64,
        max_iter: usize,
    ) -> Result<(Vec<f64>, Vec<usize>, RefineStop), SensError> {
        assert_eq!(
            pin_constraint_indices.len(),
            deltas.len(),
            "pin_constraint_indices and deltas must have equal length"
        );
        let mut db = vec![0.0; self.m_eq];
        for (&i, &d) in pin_constraint_indices.iter().zip(deltas) {
            assert!(i < self.m_eq, "pin constraint index {i} out of range");
            db[i] += d;
        }
        let (dx_full, rhs_plain) = self
            .full_step_from_db(&db)
            .ok_or(SensError::FactorizationFailed)?;

        let n = self.n;
        let (_, x_curr, lo, hi, multipliers) = self.path_inputs(pin_constraint_indices, deltas);
        let bs = self.backsolver();
        let (dx, pinned, stop) = refine_step_onto_bounds(
            &bs,
            &dx_full,
            &x_curr,
            &lo,
            &hi,
            &multipliers,
            &rhs_plain,
            bound_eps,
            RELEASE_FLOOR,
            max_iter,
        )
        .map_err(SensError::Refinement)?;
        self.last_residual = Some(bs.last_residual());
        Ok((dx[..n].to_vec(), pinned, stop))
    }

    /// The perturbation applied **a little at a time**, stopping at each point
    /// where the active set changes.
    ///
    /// `parametric_step_bounded` repairs one linear predictor. This instead
    /// walks the perturbation, re-solving at every breakpoint where a variable
    /// reaches a bound or a bound's multiplier hits zero, so the answer follows
    /// the piecewise-affine solution path rather than extrapolating across its
    /// kinks. For a QP the path *is* piecewise affine, so within a segment the
    /// walk is exact.
    ///
    /// Returns `(dx, segments)`; each [`PathSegment`] records the fraction of
    /// the perturbation at which a bound was reached or released, and which
    /// variable it belonged to.
    ///
    /// Runs `pounce_sens_core::boundcheck::step_along_path` — the NLP arm's
    /// code, reached through [`backsolver`](Self::backsolver).
    pub fn parametric_step_path(
        &mut self,
        pin_constraint_indices: &[usize],
        deltas: &[f64],
        max_iter: usize,
    ) -> Result<(Vec<f64>, Vec<PathSegment>), SensError> {
        let (rhs_plain, x_curr, lo, hi, multipliers) =
            self.path_inputs(pin_constraint_indices, deltas);
        let bs = self.backsolver();
        let (dx, segments) = step_along_path(
            &bs,
            &rhs_plain,
            &x_curr,
            &lo,
            &hi,
            &multipliers,
            max_iter,
            &[],
            &[],
            &[],
        )
        .map_err(SensError::Refinement)?;
        self.last_residual = Some(bs.last_residual());
        Ok((dx[..self.n].to_vec(), segments))
    }

    /// The shared inputs the path and bounded modes both need: the compound
    /// right-hand side, the base point, its bounds, and the active bounds'
    /// multiplier rows.
    fn path_inputs(
        &self,
        pin_constraint_indices: &[usize],
        deltas: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<BoundMultiplier>) {
        assert_eq!(
            pin_constraint_indices.len(),
            deltas.len(),
            "pin_constraint_indices and deltas must have equal length"
        );
        let mut rhs_plain = vec![0.0 as Number; self.dim];
        for (&i, &d) in pin_constraint_indices.iter().zip(deltas) {
            assert!(i < self.m_eq, "pin constraint index {i} out of range");
            rhs_plain[self.n + i] += d;
        }
        let n = self.n;
        let lo: Vec<f64> = (0..n).map(|j| self.prob.lb_of(j)).collect();
        let hi: Vec<f64> = (0..n).map(|j| self.prob.ub_of(j)).collect();
        let abase = n + self.m_eq + self.active_ineq.len();
        let multipliers: Vec<BoundMultiplier> = self
            .bound_base_mult
            .iter()
            .enumerate()
            .map(|(k, &base)| BoundMultiplier {
                row: abase + k,
                base,
            })
            .collect();
        (rhs_plain, self.base_x.clone(), lo, hi, multipliers)
    }

    /// The achieved complementarity `⟨s, z⟩ / degree` at the solution.
    ///
    /// The μ-scaled activity classification the NLP arm uses needs a barrier
    /// parameter, and `QpSolution` does not carry one. It does not need to:
    /// everything required is derivable from `(prob, sol)`, which is why this
    /// is a method rather than a new field — the type is all-public, not
    /// `#[non_exhaustive]`, and constructed by literal in dozens of places.
    ///
    /// Caveat worth knowing before comparing arms: this is the *achieved*
    /// complementarity at the returned point, not the barrier parameter the
    /// last interior-point iteration actually ran at. They agree to within the
    /// centering ratio at convergence, and the classification bands are decades
    /// wide, so it is fine for classifying — but a cross-arm test must not
    /// demand bit-agreement with the NLP's `barrier_mu()`.
    pub fn duality_measure(&self) -> f64 {
        let prob = &self.prob;
        let mut gx = vec![0.0; prob.m_ineq()];
        prob.g_mul(&self.base_x, &mut gx);
        let mut total = 0.0;
        let mut degree = 0usize;
        for (i, &gx_i) in gx.iter().enumerate() {
            total += (prob.h[i] - gx_i) * self.base_z[i];
            degree += 1;
        }
        for j in 0..prob.n {
            let (lb, ub) = (prob.lb_of(j), prob.ub_of(j));
            if lb > -BOUND_INF {
                total += (self.base_x[j] - lb) * self.base_z_lb[j];
                degree += 1;
            }
            if ub < BOUND_INF {
                total += (ub - self.base_x[j]) * self.base_z_ub[j];
                degree += 1;
            }
        }
        if degree == 0 {
            0.0
        } else {
            total / degree as f64
        }
    }

    /// What each bound is doing at this optimum: holding its coordinate,
    /// inactive, or vanishing together with its multiplier at a kink.
    ///
    /// The decision is `pounce_sens_core::activity_kernel`, the same rule the
    /// NLP arm applies, so the two arms cannot drift on what a kink is. What
    /// differs is the derivation of the inputs, and that is
    /// [`crate::activity`]'s subject.
    ///
    /// # Read the caveat before reading a status
    ///
    /// [`AMBIGUOUS`](pounce_sens_core::activity_kernel::AMBIGUOUS) does **not**
    /// mean "probably not a kink". A genuine kink lands there whenever its
    /// coordinate is coupled, because the cheap curvature is a diagonal (for a
    /// variable) or a directional one (for a row) while the multiplier is
    /// generated by the *reduced* curvature. The ratio is then
    /// `reduced/diagonal`, which is μ-independent — so re-solving tighter does
    /// not separate it. Treating the class as a proxy for kink-ness is the
    /// error that shipped gh#763.
    pub fn activity(&self) -> ConvexActivityReport {
        let floor = curvature_floor(&self.prob);
        let sol = QpSolution {
            status: QpStatus::Optimal,
            x: self.base_x.clone(),
            y: vec![],
            z: self.base_z.clone(),
            z_lb: self.base_z_lb.clone(),
            z_ub: self.base_z_ub.clone(),
            obj: 0.0,
            iters: 0,
            iterates: vec![],
        };
        classify_all(&self.prob, &sol, self.duality_measure(), floor)
    }

    /// `true` when this is a pure LP (`P = 0`) whose solve did not run
    /// crossover, so a **degenerate** optimal vertex is a real hazard here.
    ///
    /// At a degenerate LP vertex more constraints are active than there are
    /// variables, the active-set KKT is rank-deficient, and `dx/db` is not
    /// single-valued — measured on a two-variable example, the step comes back
    /// summing to half the perturbation it should. That case *is* caught, by
    /// [`ill_conditioned`](Self::ill_conditioned); this flag names the cause and
    /// the remedy, which is to solve with `qp_crossover=yes` so the interior
    /// point is pivoted to an exact vertex basis first.
    ///
    /// Reading `opts.crossover` means the `opts` handed to
    /// [`build`](Self::build) must be **the options the solve actually ran
    /// with**. Passing a fresh `QpOptions::default()` to `build` after solving
    /// with crossover on would report `true` here and be wrong.
    pub fn lp_without_crossover(&self) -> bool {
        self.lp_without_crossover
    }

    /// Hager 1-norm estimate of the condition number `κ₁` of the (factored,
    /// regularized) active-set KKT.
    ///
    /// A large value warns that the sensitivity system is near-singular — the
    /// active-constraint gradients are nearly rank-deficient (near-LICQ) — so
    /// the parametric step may be untrustworthy even though the solve reports
    /// success (gh #284). This is the quantitative companion to the boolean
    /// [`ill_conditioned`](Self::ill_conditioned); see also the per-step
    /// [`last_step_residual`](Self::last_step_residual). Well-conditioned
    /// sensitivities report a modest `κ₁` (a few `×10⁹` on the badly-scaled
    /// gh #284 QPs); a numerically singular one saturates near `1e16`.
    pub fn kkt_cond_estimate(&self) -> f64 {
        self.kkt_cond_estimate
    }

    /// Whether the KKT/sensitivity system is ill-conditioned enough that
    /// [`parametric_step`](Self::parametric_step) may be unreliable even after
    /// refinement.
    ///
    /// `true` when **either**
    ///
    /// * the build-time [`kkt_cond_estimate`](Self::kkt_cond_estimate) exceeds
    ///   [`KKT_ILL_CONDITIONED_THRESHOLD`] (gh #284) — catches a numerically
    ///   singular KKT before any step is taken; **or**
    /// * the most recent [`parametric_step`](Self::parametric_step) refined to a
    ///   relative KKT residual above [`STEP_UNRELIABLE_RESIDUAL`] (gh #328) —
    ///   catches the near-LICQ case the saturating condition estimate misses
    ///   (well-scaled `P`, near-parallel active rows), where the returned
    ///   `dx/db` does not actually satisfy the true sensitivity system.
    ///
    /// The second clause is what makes the diagnostic honest across the whole
    /// near-LICQ family: on a well-scaled `P` the condition estimate saturates
    /// below its threshold (see [`KKT_ILL_CONDITIONED_THRESHOLD`]), so before
    /// gh #328 an over-damped, ~3300×-wrong step reported `ill_conditioned =
    /// false`. Now the stalled refinement residual fires the flag instead of
    /// letting a silently-damped value pass. On the well-conditioned
    /// equality-only and active-set cases both clauses stay quiet.
    pub fn ill_conditioned(&self) -> bool {
        self.kkt_cond_estimate > KKT_ILL_CONDITIONED_THRESHOLD
            || self
                .last_residual
                .is_some_and(|r| r > STEP_UNRELIABLE_RESIDUAL)
    }

    /// Relative KKT residual `‖rhs − K·step‖∞ / ‖rhs‖∞` achieved by the most
    /// recent [`parametric_step`](Self::parametric_step) /
    /// [`step_from_db`](Self::step_from_db), or `None` before any step.
    ///
    /// Measured against the **unregularized** KKT, so it reflects how well the
    /// returned step actually satisfies the true sensitivity system. A tiny
    /// value (round-off level) means the step is trustworthy; a large one
    /// means refinement could not solve the near-singular system and the step
    /// is unreliable (gh #284). Because it is a true *relative* residual it is
    /// invariant to the magnitude of the perturbation, so it exposes a stalled
    /// solve even for a small `db` — the case the earlier `1 + ‖rhs‖` floor
    /// masked (gh #328). A value above [`STEP_UNRELIABLE_RESIDUAL`] fires
    /// [`ill_conditioned`](Self::ill_conditioned).
    pub fn last_step_residual(&self) -> Option<f64> {
        self.last_residual
    }

    /// The active-set KKT dimension `n + m_eq + n_active`.
    pub fn kkt_dim(&self) -> usize {
        self.dim
    }

    /// Inequality rows (indices into `G`) in the active set.
    pub fn active_ineq(&self) -> &[usize] {
        &self.active_ineq
    }

    /// Variables whose bound is in the active set.
    pub fn active_bound_vars(&self) -> &[usize] {
        &self.active_bound_vars
    }

    /// Inequality rows at which **strict complementarity fails**: binding in
    /// the primal while carrying a negligible multiplier (gh #219).
    ///
    /// This is the precondition check for
    /// [`parametric_step`](Self::parametric_step). That predictor is exact only
    /// while the active set is unchanged; at a weakly active constraint the
    /// perturbation changes it, so `dx/db` is a genuine one-sided derivative
    /// and the opposite direction has a different — equally correct — value.
    /// On gh #219's QP the two branches differ by 33%, and which one is
    /// reported turns on the solver's `tol`.
    ///
    /// A non-empty result does not invalidate anything already returned: both
    /// branches are real derivatives. It means the caller should not assume the
    /// predictor extrapolates in both directions, and should probe the
    /// direction it actually cares about. The screen is deliberately
    /// conservative (see `WEAK_ACTIVE_REL`) — a near-degenerate constraint is
    /// flagged too, which is the useful behaviour for a diagnostic.
    pub fn weakly_active_ineq(&self) -> &[usize] {
        &self.weakly_active_ineq
    }

    /// Variables whose bound is weakly active — the bound analog of
    /// [`weakly_active_ineq`](Self::weakly_active_ineq).
    pub fn weakly_active_bound_vars(&self) -> &[usize] {
        &self.weakly_active_bound_vars
    }

    /// Reduced Hessian of the QP at the optimum: the objective Hessian `P`
    /// projected onto the null space of the **active constraints**
    /// `B = [A; active G rows; active bound rows]`. If `Z` is an
    /// orthonormal basis of `null(B)` (the feasible directions / degrees of
    /// freedom), the reduced Hessian is `H_R = Zᵀ P Z`. Its eigenvalues are
    /// the objective's curvatures along feasible directions: all positive
    /// ⟺ a strict second-order minimizer (always so for a strictly convex
    /// `P`), and their spread is the conditioning of the QP on the active
    /// manifold. This mirrors the NLP `Solver.reduced_hessian` /
    /// `solve_with_sens(compute_reduced_hessian=True)`.
    ///
    /// The basis `Z` is the null space of `B`, obtained from the
    /// eigenvectors of `BᵀB` whose eigenvalue is below `rank_tol · λ_max`
    /// (squared singular values; the count above the threshold is
    /// `rank(B)`, so the degrees of freedom are `n − rank(B)`). The
    /// computation densifies `B` and `P`, so it is `O(n³)` — intended, like
    /// sIPOPT's reduced Hessian, for QPs with a modest number of variables
    /// (the parametric step stays sparse and is the workhorse for large
    /// problems).
    ///
    /// # Errors
    ///
    /// Returns [`SensError::EigenFailed`] if either symmetric eigensolve (the
    /// one that extracts `Z` from `BᵀB`, or the final one on `H_R`) does not
    /// converge — its rank / null-space, and hence the result, cannot be
    /// trusted, so a wrong answer is never returned silently.
    pub fn reduced_hessian(&self, rank_tol: f64) -> Result<ReducedHessian, SensError> {
        let n = self.n;

        // Active Jacobian B (m_act × n), dense row-major: equality rows,
        // then active inequality rows, then active variable-bound rows.
        let m_act = self.m_eq + self.active_ineq.len() + self.active_bound_vars.len();
        let mut b = vec![0.0; m_act * n];
        for t in &self.prob.a {
            b[t.row * n + t.col] += t.val;
        }
        let g_rows = group_rows_by_index(&self.prob.g, self.prob.m_ineq());
        let mut row = self.m_eq;
        for &i in &self.active_ineq {
            for &(col, val) in &g_rows[i] {
                b[row * n + col] += val;
            }
            row += 1;
        }
        for &j in &self.active_bound_vars {
            b[row * n + j] += 1.0;
            row += 1;
        }

        // Null space of B from the eigenvectors of BᵀB (symmetric, n×n,
        // column-major for `symmetric_eigen`). BᵀB[a,c] = Σ_r B[r,a]·B[r,c].
        let mut btb = vec![0.0; n * n];
        for r in 0..m_act {
            for a in 0..n {
                let bra = b[r * n + a];
                if bra == 0.0 {
                    continue;
                }
                for c in 0..n {
                    btb[a * n + c] += bra * b[r * n + c];
                }
            }
        }
        let mut sv = vec![0.0; n];
        let mut vecs = vec![0.0; n * n];
        // Ascending eigenvalues. A failed eigensolve makes the rank/null-space
        // count below meaningless, so refuse rather than return garbage.
        if !symmetric_eigen(&btb, n, &mut sv, &mut vecs) {
            return Err(SensError::EigenFailed);
        }

        // rank(B) = # squared-singular-values above the relative threshold;
        // the null space is spanned by the eigenvectors of the rest (the
        // smallest, ≈ 0). With ascending order those are the first columns.
        let max_sv = sv.last().copied().unwrap_or(0.0).max(0.0);
        let thresh = rank_tol * max_sv;
        let rank = sv.iter().filter(|&&l| l > thresh).count();
        let n_dof = n - rank;

        // Dense symmetric P (n×n) from its lower triangle.
        let mut p = vec![0.0; n * n];
        for t in &self.prob.p_lower {
            p[t.row * n + t.col] += t.val;
            if t.row != t.col {
                p[t.col * n + t.row] += t.val;
            }
        }

        // H_R = Zᵀ P Z, with Z = first `n_dof` columns of `vecs` (the null
        // space). Column-major throughout: column j of Z is vecs[j*n + ·].
        let z = |j: usize, r: usize| vecs[j * n + r];
        // PZ (n × n_dof), column-major.
        let mut pz = vec![0.0; n * n_dof];
        for j in 0..n_dof {
            for (r, pzr) in pz[j * n..(j + 1) * n].iter_mut().enumerate() {
                let mut acc = 0.0;
                for c in 0..n {
                    acc += p[r * n + c] * z(j, c);
                }
                *pzr = acc;
            }
        }
        // H_R (n_dof × n_dof), column-major: H_R[i,j] = z_iᵀ (P z_j).
        let mut hr = vec![0.0; n_dof * n_dof];
        for j in 0..n_dof {
            for i in 0..n_dof {
                let mut acc = 0.0;
                for r in 0..n {
                    acc += z(i, r) * pz[j * n + r];
                }
                hr[j * n_dof + i] = acc;
            }
        }

        // Eigendecompose the (small) reduced Hessian.
        let mut eigenvalues = vec![0.0; n_dof];
        let mut eigenvectors = vec![0.0; n_dof * n_dof];
        if !symmetric_eigen(&hr, n_dof, &mut eigenvalues, &mut eigenvectors) {
            return Err(SensError::EigenFailed);
        }

        Ok(ReducedHessian {
            n_dof,
            matrix: hr,
            eigenvalues,
            eigenvectors,
        })
    }

    /// [`reduced_hessian`](Self::reduced_hessian) with a relative rank
    /// tolerance of `1e-9`.
    pub fn reduced_hessian_default(&self) -> Result<ReducedHessian, SensError> {
        self.reduced_hessian(1e-9)
    }
}

/// The reduced Hessian `H_R = Zᵀ P Z` of a QP on its active manifold, with
/// its eigendecomposition. All matrices are column-major and `n_dof × n_dof`
/// (`n_dof` = degrees of freedom = `n − rank` of the active Jacobian).
#[derive(Debug, Clone, PartialEq)]
pub struct ReducedHessian {
    /// Degrees of freedom: the dimension of every field here.
    pub n_dof: usize,
    /// The reduced Hessian `H_R`, column-major `n_dof × n_dof` (symmetric).
    pub matrix: Vec<f64>,
    /// Eigenvalues of `H_R`, ascending (length `n_dof`).
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors, column-major `n_dof × n_dof`; column `j` pairs with
    /// `eigenvalues[j]`. Signs are pinned (largest-magnitude component
    /// positive) so a column read as a direction reproduces. Note these
    /// live in the null-space basis `Z`, which is itself only fixed up
    /// to a rotation within any degenerate eigenspace of `BᵀB`.
    pub eigenvectors: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipm::solve_qp_ipm;
    use crate::qp::Triplet;
    use pounce_feral::FeralSolverInterface;

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(FeralSolverInterface::new())
    }

    /// gh #219's degenerate QP: `min ½‖x‖² s.t. x₀ + x₁ = 1, x₀ − 2x₁ ≤ h`.
    /// At `h = −½` the equality-only optimum `(½, ½)` hits the inequality
    /// *exactly*, so strict complementarity fails; other `h` give a strictly
    /// active (`h = −0.9`) or strictly inactive (`h = 0.5`) constraint.
    fn weakly_active_qp(h: f64) -> QpProblem {
        QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
            c: vec![0.0, 0.0],
            a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            b: vec![1.0],
            g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, -2.0)],
            h: vec![h],
            lb: vec![f64::NEG_INFINITY; 2],
            ub: vec![f64::INFINITY; 2],
        }
    }

    #[test]
    fn weak_activity_is_detected_independently_of_solver_tol() {
        // The point of the flag (gh #219). At this optimum `dx/db` is
        // two-valued — (2/3, 1/3) on the minus side, (1/2, 1/2) on the plus
        // side, 33% apart — and *which* one `parametric_step` reports turns on
        // `tol`, an otherwise unrelated setting: the multiplier and the slack
        // both collapse at ~√tol, so `active_tol` slices the pair differently
        // at different `tol`.
        //
        // `kkt_dim` therefore flips 4 → 3 across this sweep while the geometry
        // does not change at all. The weak-activity flag is the stable signal:
        // it must fire at every tolerance, including the ones where the
        // constraint *is* in the active set.
        let prob = weakly_active_qp(-0.5);
        let mut saw_in_active_set = false;
        let mut saw_out_of_active_set = false;
        for tol in [1e-8, 1e-12, 1e-14] {
            let opts = QpOptions {
                tol,
                ..QpOptions::default()
            };
            let sol = solve_qp_ipm(&prob, &opts, backend);
            assert_eq!(sol.status, QpStatus::Optimal);
            let sens = QpSensitivity::build(&prob, &sol, &opts, 1e-7, backend).unwrap();
            assert_eq!(
                sens.weakly_active_ineq(),
                &[0],
                "tol {tol:e}: weak activity missed (kkt_dim {})",
                sens.kkt_dim()
            );
            match sens.active_ineq() {
                [] => saw_out_of_active_set = true,
                [0] => saw_in_active_set = true,
                other => panic!("tol {tol:e}: unexpected active set {other:?}"),
            }
        }
        // Guards the premise: if the sweep stopped straddling the active-set
        // boundary the test would still pass while no longer testing anything.
        assert!(
            saw_in_active_set && saw_out_of_active_set,
            "sweep no longer straddles the active-set boundary, so this test \
             no longer demonstrates tol-independence"
        );
    }

    #[test]
    fn strictly_complementary_constraints_are_not_flagged_weak() {
        // The false-positive guard. A screen that fired on every active
        // constraint, or on every constraint with a small multiplier, would
        // pass the test above while being useless.
        //
        // `h = −0.9`: the constraint binds with multiplier ~8.9e-2 — strictly
        // active, `dx/db` two-sided. `h = 0.5`: the constraint is slack at the
        // optimum — strictly inactive. Neither is degenerate.
        for (h, expect_active) in [(-0.9, true), (0.5, false)] {
            let prob = weakly_active_qp(h);
            let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
            assert_eq!(sol.status, QpStatus::Optimal);
            let sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();
            assert!(
                sens.weakly_active_ineq().is_empty(),
                "h = {h}: strictly complementary constraint flagged as weakly active"
            );
            assert_eq!(
                !sens.active_ineq().is_empty(),
                expect_active,
                "h = {h}: active set {:?}",
                sens.active_ineq()
            );
        }
    }

    /// `min ½‖x‖²  s.t.  x₀ + x₁ = b` (b = 2). The optimum is the projection
    /// of the origin onto the line: `x = (b/2, b/2)`, so `dx/db = (½, ½)`
    /// exactly. The parametric step for `Δb` must reproduce that.
    #[test]
    fn parametric_step_matches_closed_form_equality() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
            c: vec![0.0, 0.0],
            a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            b: vec![2.0],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-7 && (sol.x[1] - 1.0).abs() < 1e-7);

        let mut sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();
        let dx = sens.parametric_step(&[0], &[1.0]); // Δb = +1
        assert!((dx[0] - 0.5).abs() < 1e-6, "dx0 = {}", dx[0]);
        assert!((dx[1] - 0.5).abs() < 1e-6, "dx1 = {}", dx[1]);

        // Predictor lands on the exact re-solve for the perturbed b.
        let mut prob2 = prob.clone();
        prob2.b = vec![3.0];
        let sol2 = solve_qp_ipm(&prob2, &QpOptions::default(), backend);
        assert!((sol.x[0] + dx[0] - sol2.x[0]).abs() < 1e-6);
        assert!((sol.x[1] + dx[1] - sol2.x[1]).abs() < 1e-6);
    }

    /// With an **active inequality** in the active set, the predictor must
    /// still match the re-solve. `min ½‖x‖² s.t. x₀+x₁ = b, x₀ ≥ 1`. At
    /// b = 1 the unconstrained projection would be (0.5, 0.5) but `x₀ ≥ 1`
    /// binds, giving `x = (1, 0)`. Perturbing b shifts along the active
    /// face: `x = (1, b−1)`, so `dx/db = (0, 1)`.
    #[test]
    fn parametric_step_with_active_inequality() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
            c: vec![0.0, 0.0],
            a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            b: vec![1.0],
            g: vec![Triplet::new(0, 0, -1.0)], // −x₀ ≤ −1  ⇔  x₀ ≥ 1
            h: vec![-1.0],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-6 && sol.x[1].abs() < 1e-6);
        assert!(sol.z[0] > 1e-6, "inequality should be active");

        let mut sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();
        let dx = sens.parametric_step(&[0], &[0.5]);
        assert!(dx[0].abs() < 1e-6, "dx0 = {} (should stay on x₀=1)", dx[0]);
        assert!((dx[1] - 0.5).abs() < 1e-6, "dx1 = {}", dx[1]);
    }

    // ---- gh #284: near-LICQ conditioning diagnostic + refinement ----------

    /// The gh #284 Hessian `P = D·H₆·D` (Hilbert matrix `H₆`, `D =
    /// diag(1e3,…,1e-2)`; `cond(P) ≈ 7e15`) and its linear term. Shared by the
    /// conditioning tests below.
    fn hilbert_p_and_c() -> (Vec<Triplet>, Vec<f64>) {
        let d = [1e3, 1e2, 1e1, 1.0, 1e-1, 1e-2];
        let mut p_lower = Vec::new();
        for i in 0..6 {
            for j in 0..=i {
                let hij = 1.0 / ((i + j + 1) as f64);
                p_lower.push(Triplet::new(i, j, d[i] * hij * d[j]));
            }
        }
        (p_lower, vec![1.0, -2.0, 3.0, -1.0, 0.5, -0.25])
    }

    /// Two equality rows that are all-ones except the last entry of row 1,
    /// which differs by `eps` — nearly parallel, so LICQ nearly fails.
    fn near_parallel_rows(eps: f64) -> Vec<Triplet> {
        let mut a = Vec::new();
        for j in 0..6 {
            a.push(Triplet::new(0, j, 1.0));
            a.push(Triplet::new(1, j, if j == 5 { 1.0 + eps } else { 1.0 }));
        }
        a
    }

    /// Dense LU with partial pivoting — the float64 reference the gh #284 issue
    /// uses (`numpy.linalg.solve`) to show `dx/db` survives in double precision.
    /// `a` is row-major `n×n`; solves `a x = b`.
    fn dense_lu_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
        let n = b.len();
        for k in 0..n {
            let mut piv = k;
            for i in (k + 1)..n {
                if a[i][k].abs() > a[piv][k].abs() {
                    piv = i;
                }
            }
            a.swap(k, piv);
            b.swap(k, piv);
            for i in (k + 1)..n {
                let f = a[i][k] / a[k][k];
                for j in k..n {
                    a[i][j] -= f * a[k][j];
                }
                b[i] -= f * b[k];
            }
        }
        let mut x = vec![0.0; n];
        for k in (0..n).rev() {
            let mut s = b[k];
            for j in (k + 1)..n {
                s -= a[k][j] * x[j];
            }
            x[k] = s / a[k][k];
        }
        x
    }

    /// Dense true (δ-free) equality KKT `[[P, Aᵀ], [A, 0]]`, row-major.
    fn dense_eq_kkt(p_lower: &[Triplet], a: &[Triplet], n: usize, m: usize) -> Vec<Vec<f64>> {
        let dim = n + m;
        let mut k = vec![vec![0.0; dim]; dim];
        for t in p_lower {
            k[t.row][t.col] += t.val;
            if t.row != t.col {
                k[t.col][t.row] += t.val;
            }
        }
        for t in a {
            k[n + t.row][t.col] += t.val;
            k[t.col][n + t.row] += t.val;
        }
        k
    }

    /// dx/db reference for the equality-only KKT: the x-block of the true-KKT
    /// solve with rhs `[0; e_{pin}]`, by dense float64 LU (independent of the
    /// factored/regularized path).
    fn dxdb_reference(prob: &QpProblem, pin: usize) -> Vec<f64> {
        let (n, m) = (prob.n, prob.m_eq());
        let kkt = dense_eq_kkt(&prob.p_lower, &prob.a, n, m);
        let mut rhs = vec![0.0; n + m];
        rhs[n + pin] = 1.0;
        dense_lu_solve(kkt, rhs)[..n].to_vec()
    }

    fn rel_err(a: &[f64], b: &[f64]) -> f64 {
        let scale = b.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        a.iter()
            .zip(b)
            .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
            / scale
    }

    /// A near-LICQ sensitivity (two equality rows differing by `1e-9`) is
    /// **detectably** untrustworthy (gh #284). Before the fix, `dx/db`
    /// collapsed to a smoothly over-damped, ~98%-wrong value while every
    /// existing signal (`weakly_active`, `kkt_dim`, status) looked ordinary and
    /// no exception was raised — the caller had no way to know. The
    /// conditioning diagnostic must fire, and the refinement residual must
    /// expose that the near-singular step was not solved.
    #[test]
    fn near_licq_sensitivity_is_flagged_ill_conditioned() {
        let (p_lower, c) = hilbert_p_and_c();
        let prob = QpProblem {
            n: 6,
            p_lower,
            c,
            a: near_parallel_rows(1e-10),
            b: vec![1.0, 1.0],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        let mut sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();

        // The old signals stay silent: nothing weakly active, full KKT dim.
        assert!(sens.weakly_active_ineq().is_empty());
        assert!(sens.weakly_active_bound_vars().is_empty());
        // The new diagnostic fires.
        assert!(
            sens.ill_conditioned(),
            "near-LICQ KKT must be flagged (κ₁ = {:.3e})",
            sens.kkt_cond_estimate()
        );
        assert!(
            sens.kkt_cond_estimate() > KKT_ILL_CONDITIONED_THRESHOLD,
            "κ₁ = {:.3e}",
            sens.kkt_cond_estimate()
        );

        // The step's residual against the true KKT is large: refinement could
        // not solve the near-singular system, so the step is unreliable.
        let _ = sens.parametric_step(&[0], &[1.0]);
        let res = sens.last_step_residual().expect("a step was taken");
        assert!(
            res > 1e-6,
            "expected a large refinement residual, got {res:.3e}"
        );
    }

    /// The false-alarm guard: the gh #284 *well-conditioned* case — the same
    /// badly-scaled `P = D·H₆·D` with a full-rank (if badly scaled) `A` whose
    /// rows differ by orders of magnitude, `cond(KKT) ≈ 5e9`. The diagnostic
    /// must stay quiet, and `dx/db` must match a dense float64 LU reference to
    /// ~1e-7 (the regularization introduces no detectable bias here).
    #[test]
    fn well_conditioned_sensitivity_not_flagged_and_accurate() {
        let (p_lower, c) = hilbert_p_and_c();
        // Rows: [1,1,1,1,1,1] and [1e4,1,1,1,1,1e-4] — badly scaled, full rank.
        let a = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(0, 2, 1.0),
            Triplet::new(0, 3, 1.0),
            Triplet::new(0, 4, 1.0),
            Triplet::new(0, 5, 1.0),
            Triplet::new(1, 0, 1e4),
            Triplet::new(1, 1, 1.0),
            Triplet::new(1, 2, 1.0),
            Triplet::new(1, 3, 1.0),
            Triplet::new(1, 4, 1.0),
            Triplet::new(1, 5, 1e-4),
        ];
        let prob = QpProblem {
            n: 6,
            p_lower,
            c,
            a,
            b: vec![1.0, 2.0],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        let mut sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();

        assert!(
            !sens.ill_conditioned(),
            "well-conditioned KKT must NOT be flagged (κ₁ = {:.3e})",
            sens.kkt_cond_estimate()
        );
        assert!(
            sens.kkt_cond_estimate() < 1e12,
            "κ₁ = {:.3e}",
            sens.kkt_cond_estimate()
        );

        let dx = sens.parametric_step(&[0], &[1.0]);
        let reference = dxdb_reference(&prob, 0);
        let err = rel_err(&dx, &reference);
        assert!(err < 1e-7, "dx/db rel err vs float64 LU = {err:.3e}");
        assert!(
            sens.last_step_residual().unwrap() < 1e-8,
            "residual = {:?}",
            sens.last_step_residual()
        );
    }

    /// Refinement recovers accuracy where the information survives in double
    /// precision (gh #284). At `eps = 1e-6` the KKT is near-LICQ enough that a
    /// single regularized back-solve over-damps `dx/db` to ~4e-5 relative
    /// error, yet a plain float64 LU recovers it. Refinement against the
    /// unregularized KKT must close that gap — matching the LU reference far
    /// better than the un-refined solve could — while the conditioning flag
    /// stays quiet (the step *is* reliable here).
    #[test]
    fn refinement_recovers_dxdb_where_information_survives() {
        let (p_lower, c) = hilbert_p_and_c();
        let prob = QpProblem {
            n: 6,
            p_lower,
            c,
            a: near_parallel_rows(1e-6),
            b: vec![1.0, 1.0],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        let mut sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();

        // Recoverable, so not flagged.
        assert!(
            !sens.ill_conditioned(),
            "κ₁ = {:.3e}",
            sens.kkt_cond_estimate()
        );

        let dx = sens.parametric_step(&[0], &[1.0]);
        let reference = dxdb_reference(&prob, 0);
        let err = rel_err(&dx, &reference);
        // Comfortably better than the ~4e-5 an un-refined regularized solve
        // yields here: refinement did its job.
        assert!(
            err < 1e-6,
            "refined dx/db rel err vs float64 LU = {err:.3e}"
        );
        assert!(
            sens.last_step_residual().unwrap() < 1e-6,
            "residual = {:?}",
            sens.last_step_residual()
        );
    }

    /// gh #328: a **well-scaled** `P` (`P = I`) with a near-LICQ *constraint*
    /// Jacobian must never return a silently-wrong `dx/db`. `A = [[1,0],[1,ε]]`
    /// fully pins `x`, so the exact sensitivity is `dx/db = A⁻¹` with no
    /// truncation — `dx/db[:,0] = [1, −1/ε]`. As `ε → 0` the two equality rows
    /// become parallel and the KKT goes numerically singular, but because `P`
    /// is well scaled the build-time condition estimate saturates (`κ₁ ≈ 3e10`)
    /// and never reaches its threshold — the blind spot the old
    /// `ill_conditioned` had. The regression bar: for **every** `ε` the step is
    /// either accurate to a reasonable relative tolerance **or**
    /// `ill_conditioned` is `true`; a catastrophically over-damped step with
    /// `ill_conditioned == false` (the gh #328 failure — `−2999` where the truth
    /// is `−1e7`) must never happen. The well-conditioned `ε = 1e-3` end must
    /// stay accurate *and* unflagged.
    #[test]
    fn near_licq_constraint_jacobian_never_silently_wrong() {
        // Perturbing b0 by a small δb, then dividing out δb, is exactly how a
        // caller reads dx/db — and the small δb is what the old (1 + ‖rhs‖)
        // residual floor masked, so exercise that path here (δb = 1e-6).
        let db = 1e-6;
        let mut saw_flagged = false;
        for eps in [1e-3, 1e-5, 1e-7, 1e-9] {
            let prob = QpProblem {
                n: 2,
                p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
                c: vec![0.0, 0.0],
                // Rows [1,0] and [1,ε]: near-parallel as ε → 0.
                a: vec![
                    Triplet::new(0, 0, 1.0),
                    Triplet::new(1, 0, 1.0),
                    Triplet::new(1, 1, eps),
                ],
                b: vec![1.0, 1.0],
                g: vec![],
                h: vec![],
                lb: vec![],
                ub: vec![],
            };
            let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
            assert_eq!(sol.status, QpStatus::Optimal, "eps {eps:e}");
            let mut sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();

            // dx/db[:,0] via a pinned step, divided back out.
            let step = sens.parametric_step(&[0], &[db]);
            let dxdb = [step[0] / db, step[1] / db];
            // Exact reference: A⁻¹[:,0] = [1, −1/ε].
            let exact = [1.0, -1.0 / eps];
            let rel = rel_err(&dxdb, &exact);
            let accurate = rel < 1e-3;

            // The acceptance bar: accurate OR honestly flagged. The forbidden
            // state is exactly the gh #328 bug — wrong AND unflagged.
            assert!(
                accurate || sens.ill_conditioned(),
                "eps {eps:e}: silently wrong dx/db = {dxdb:?} (exact {exact:?}, \
                 rel err {rel:.3e}) with ill_conditioned = false, \
                 kkt_cond = {:.3e}, residual = {:?}",
                sens.kkt_cond_estimate(),
                sens.last_step_residual(),
            );

            if eps == 1e-3 {
                // The well-conditioned end must be accurate *and* trusted.
                assert!(accurate, "eps {eps:e}: dx/db = {dxdb:?} rel err {rel:.3e}");
                assert!(
                    !sens.ill_conditioned(),
                    "eps {eps:e}: well-conditioned case falsely flagged \
                     (kkt_cond = {:.3e}, residual = {:?})",
                    sens.kkt_cond_estimate(),
                    sens.last_step_residual(),
                );
            } else {
                // Whenever the step is *not* accurate here, the flag must fire —
                // this is the clause that was silently false before the fix.
                if !accurate {
                    assert!(
                        sens.ill_conditioned(),
                        "eps {eps:e}: over-damped dx/db = {dxdb:?} (rel err \
                         {rel:.3e}) not flagged; residual = {:?}",
                        sens.last_step_residual(),
                    );
                    saw_flagged = true;
                }
            }
        }
        // Guards the premise: the sweep must actually reach the unrecoverable
        // regime, otherwise it no longer exercises the #328 flag path.
        assert!(
            saw_flagged,
            "sweep never hit an over-damped step, so the ill-conditioned flag \
             path was not exercised"
        );
    }

    /// A non-optimal solution has no well-defined active set.
    #[test]
    fn build_rejects_non_optimal() {
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![-1.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, -1.0)],
            h: vec![0.0], // x ≥ 0, min −x ⇒ unbounded
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_ne!(sol.status, QpStatus::Optimal);
        assert!(matches!(
            QpSensitivity::build_default(&prob, &sol, backend),
            Err(SensError::NotOptimal)
        ));
    }

    /// Unconstrained-direction reduced Hessian equals `P` itself: with no
    /// active constraints the null space is all of ℝⁿ, so `H_R = ZᵀPZ = P`
    /// (up to an orthonormal rotation, hence the eigenvalues match `P`).
    /// `min ½(2x₀² + 3x₁²)` has no binding constraints; eigenvalues = {2, 3}.
    #[test]
    fn reduced_hessian_unconstrained_is_p() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 3.0)],
            c: vec![0.0, 0.0],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        let sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();
        let rh = sens
            .reduced_hessian_default()
            .expect("eigensolve converges");
        assert_eq!(rh.n_dof, 2);
        assert!(
            (rh.eigenvalues[0] - 2.0).abs() < 1e-9,
            "{:?}",
            rh.eigenvalues
        );
        assert!(
            (rh.eigenvalues[1] - 3.0).abs() < 1e-9,
            "{:?}",
            rh.eigenvalues
        );
    }

    /// One equality constraint removes one degree of freedom. `min ½‖x‖²`
    /// (P = I) on the 3-D space with `x₀ + x₁ + x₂ = b` leaves a 2-D null
    /// space; the reduced Hessian is the 2×2 identity (both curvatures = 1).
    #[test]
    fn reduced_hessian_drops_one_dof_per_active_constraint() {
        let prob = QpProblem {
            n: 3,
            p_lower: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(1, 1, 1.0),
                Triplet::new(2, 2, 1.0),
            ],
            c: vec![0.0, 0.0, 0.0],
            a: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(0, 2, 1.0),
            ],
            b: vec![1.0],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        let sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();
        let rh = sens
            .reduced_hessian_default()
            .expect("eigensolve converges");
        assert_eq!(rh.n_dof, 2, "one equality ⇒ 2 DOF");
        for &ev in &rh.eigenvalues {
            assert!((ev - 1.0).abs() < 1e-9, "eig {ev}");
        }
    }

    /// A non-identity reduced Hessian: `min ½xᵀPx` with a coupled `P` and an
    /// equality that pins the sum, cross-checked against the hand-computed
    /// `ZᵀPZ` for the unit null-space direction `z = (1,−1)/√2`.
    #[test]
    fn reduced_hessian_value_matches_hand_projection() {
        // P = [[3, 1], [1, 2]]; constraint x₀ + x₁ = 0 ⇒ Z = (1,−1)/√2.
        // zᵀPz = (3 − 1 − 1 + 2)/2 = 3/2.
        let prob = QpProblem {
            n: 2,
            p_lower: vec![
                Triplet::new(0, 0, 3.0),
                Triplet::new(1, 0, 1.0),
                Triplet::new(1, 1, 2.0),
            ],
            c: vec![0.0, 0.0],
            a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            b: vec![0.0],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        let sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();
        let rh = sens
            .reduced_hessian_default()
            .expect("eigensolve converges");
        assert_eq!(rh.n_dof, 1);
        assert!(
            (rh.eigenvalues[0] - 1.5).abs() < 1e-9,
            "H_R = {:?}",
            rh.eigenvalues
        );
        assert!((rh.matrix[0] - 1.5).abs() < 1e-9);
    }

    /// Two **simultaneously active** inequality rows, each with *multiple*
    /// nonzeros and a **shared column**, so both the KKT build and the
    /// reduced-Hessian assembly must read each active row's full set of
    /// `(col, val)` entries — and must not let one row's entries leak into
    /// the other (col 1 appears in both). The single-triplet active-row
    /// fixtures elsewhere never exercise the per-row grouping; this is the
    /// guard for the `group_rows_by_index` assembly.
    ///
    /// `min ½‖x‖² − 2·𝟙ᵀx` (unconstrained min at `(2,2,2)`) with
    /// `x₀+x₁ ≤ 1` and `x₁+x₂ ≤ 1`. Both bind at the optimum `(1,0,1)`
    /// with equal positive multipliers (λ = 1), so `B = [[1,1,0],[0,1,1]]`
    /// has rank 2 → one degree of freedom. The null space is spanned by
    /// `(−1,1,−1)/√3`, so `H_R = ZᵀIZ = 1`.
    #[test]
    fn reduced_hessian_two_active_multi_triplet_rows() {
        let prob = QpProblem {
            n: 3,
            p_lower: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(1, 1, 1.0),
                Triplet::new(2, 2, 1.0),
            ],
            c: vec![-2.0, -2.0, -2.0],
            a: vec![],
            b: vec![],
            // Row 0: x₀ + x₁ (cols 0,1); row 1: x₁ + x₂ (cols 1,2).
            g: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(1, 1, 1.0),
                Triplet::new(1, 2, 1.0),
            ],
            h: vec![1.0, 1.0],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!(
            (sol.x[0] - 1.0).abs() < 1e-5 && sol.x[1].abs() < 1e-5 && (sol.x[2] - 1.0).abs() < 1e-5,
            "x = {:?} (expected (1, 0, 1))",
            sol.x,
        );
        assert!(
            sol.z[0] > 1e-6 && sol.z[1] > 1e-6,
            "both inequalities should be active: z = {:?}",
            sol.z,
        );

        let sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();
        let rh = sens
            .reduced_hessian_default()
            .expect("eigensolve converges");
        assert_eq!(rh.n_dof, 1, "rank-2 active Jacobian on n=3 ⇒ 1 DOF");
        assert!(
            (rh.eigenvalues[0] - 1.0).abs() < 1e-7,
            "H_R = {:?} (expected eigenvalue 1)",
            rh.eigenvalues,
        );

        // The build's KKT must also see both active rows: a free RHS over
        // the (empty) equality block leaves dx = 0, but the factorization
        // having succeeded with dim = n + 0 + 2 confirms both rows entered.
        assert_eq!(sens.kkt_dim(), 3 + 0 + 2);
    }

    /// `reduced_hessian` now *returns* an eigensolve-convergence verdict
    /// instead of silently ignoring it: on a well-formed QP both internal
    /// symmetric eigensolves (the `BᵀB` rank/null-space split and the final
    /// `H_R` decomposition) converge, so the call must yield `Ok` with the
    /// hand-checked reduced Hessian.
    ///
    /// The `Err(EigenFailed)` branch is a defensive consistency guard: it can
    /// only trip if `symmetric_eigen` exhausts its sweeps, which a modest,
    /// well-conditioned reduced Hessian like this one never does — so the
    /// failure path is not reachable through the public solver here and is not
    /// exercised by a fixture (the same limitation noted for the underlying
    /// `symmetric_eigen` convergence flag). This test pins the `Ok` contract;
    /// before the fix the function returned a bare `ReducedHessian` and a
    /// non-converged solve would have been published as if trustworthy.
    #[test]
    fn reduced_hessian_returns_ok_on_convergent_eigensolve() {
        // min ½‖x‖² − 2·𝟙ᵀx with x₀ + x₁ ≤ 1 (active at (0.5, 0.5)); the
        // single active row has rank 1 on n=2 ⇒ 1 DOF, null space (1,−1)/√2,
        // so H_R = ZᵀIZ = 1.
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
            c: vec![-2.0, -2.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            h: vec![1.0],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        let sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();

        // The verdict is surfaced, not discarded: matching on the Result is
        // the behavior L40 introduced.
        let rh = match sens.reduced_hessian_default() {
            Ok(rh) => rh,
            Err(e) => panic!("convergent eigensolve must yield Ok, got {e:?}"),
        };
        assert_eq!(rh.n_dof, 1, "rank-1 active Jacobian on n=2 ⇒ 1 DOF");
        assert!(
            (rh.eigenvalues[0] - 1.0).abs() < 1e-7,
            "H_R = {:?} (expected eigenvalue 1)",
            rh.eigenvalues,
        );
        // The explicit-tolerance entry point carries the same contract.
        assert!(sens.reduced_hessian(1e-9).is_ok());
    }

    /// Pin the regularization value the module doc cites. The default-built
    /// sensitivity (`build_default` → `QpOptions::default()`) places
    /// `opts.reg` on the KKT diagonal (see `build`, `let reg = opts.reg`), so
    /// the "default `δ`" the module-level doc names *is* `QpOptions::default()
    /// .reg`. That default was retuned `1e-8 → 1e-10` (ipm.rs: `1e-8` stalls
    /// `adlittle`), but the doc kept saying `1e-8` (L42). This guards the doc
    /// against silent drift: if the default reg changes again, this fails and
    /// forces the module doc to be updated in lockstep.
    #[test]
    fn module_doc_regularization_matches_qp_options_default() {
        assert_eq!(
            QpOptions::default().reg,
            1e-10,
            "module doc names this as the default sensitivity regularization δ",
        );
    }
}

// ---------------------------------------------------------------------------
// The `SensBacksolver` adapter.
// ---------------------------------------------------------------------------

/// The convex active-set KKT, as a [`SensBacksolver`].
///
/// This is what lets the convex arm reach the parametric machinery in
/// `pounce-sens-core` — fix-relax refinement, and in due course path following
/// and the directional derivative — instead of reimplementing it. That
/// machinery is already generic over this trait, derives its variable count
/// from slice lengths, and reads bounds only through [`BoundRow`], so the only
/// thing it needed was an implementation over a different KKT.
///
/// # Why `Rc<RefCell<…>>`
///
/// Not a style choice. [`Factorization::solve_one`] takes `&mut self` while
/// [`SensBacksolver::solve`] takes `&self`, and `boundcheck` additionally
/// requires `B: Clone` while `Factorization` owns a non-`Clone`
/// `TSymLinearSolver`. Shared interior mutability is the only shape that
/// satisfies all three — and it is exactly the shape the NLP arm's
/// `PdSensBacksolver` already uses for the same reasons.
///
/// # Block layout
///
/// The compound vector is `(x, y, z_a)`: `n` primal rows, `m_eq` equality
/// multipliers, then one row per active constraint (inequality rows first, then
/// bound rows). The NLP arm's is an eight-block iterate instead, which is fine —
/// the shared machinery assumes only that block 0 is `x`.
#[derive(Clone)]
pub struct QpKktBacksolver {
    fact: Rc<RefCell<Factorization>>,
    airn: Rc<Vec<Index>>,
    ajcn: Rc<Vec<Index>>,
    vals_true: Rc<Vec<f64>>,
    scratch: Rc<RefCell<IrScratch>>,
    dim: usize,
    bound_rows: Rc<Vec<BoundRow>>,
    /// Relative residual of the most recent `solve`, so the caller's
    /// `ill_conditioned` reporting keeps working across boundcheck-driven
    /// solves. A `Cell` because `solve` takes `&self`.
    last_residual: Rc<Cell<f64>>,
    // --- release support ---
    /// The factored (regularized) values of the *unreleased* system; a release
    /// starts from these and neutralizes the released rows.
    vals_reg: Rc<Vec<f64>>,
    /// Per active bound, the `(coupling, diagonal)` value slots to neutralize.
    slots: Rc<Vec<(usize, usize)>>,
    /// Per active bound, its oriented base multiplier — what
    /// `solve_released_step` moves onto the variable's `x` row.
    base_mult: Rc<Vec<f64>>,
    /// The most recent released system, kept so a repeat costs nothing and a
    /// *different* release costs one numeric refactorization rather than a new
    /// symbolic one (the pattern never changes).
    released: Rc<RefCell<Option<ReleasedFactor>>>,
    release_backend: Rc<RefCell<Option<Box<dyn SparseSymLinearSolverInterface>>>>,
}

/// A factored released system, with the released row set it belongs to.
struct ReleasedFactor {
    key: Vec<usize>,
    fact: Factorization,
    vals_true: Vec<f64>,
}

impl QpKktBacksolver {
    /// Relative KKT residual of the most recent [`SensBacksolver::solve`],
    /// or `f64::NAN` if none has run.
    pub fn last_residual(&self) -> f64 {
        self.last_residual.get()
    }
}

impl SensBacksolver for QpKktBacksolver {
    fn dim(&self) -> usize {
        self.dim
    }

    fn solve(&self, rhs: &[Number], lhs: &mut [Number]) -> bool {
        if rhs.len() != self.dim || lhs.len() != self.dim {
            return false;
        }
        lhs.copy_from_slice(rhs);
        let mut fact = self.fact.borrow_mut();
        let mut scratch = self.scratch.borrow_mut();
        match solve_refined(
            &mut fact,
            &self.airn,
            &self.ajcn,
            &self.vals_true,
            lhs,
            &mut scratch,
        ) {
            Ok(res) => {
                self.last_residual.set(res);
                true
            }
            Err(()) => false,
        }
    }

    /// `None`, and that is correct rather than unimplemented.
    ///
    /// The factor this trait answers in and the multipliers a caller reads off
    /// `QpSolution` are already in one frame: Ruiz equilibration lives inside
    /// `solve_qp_ipm` and is undone by `Scaling::unscale_solution` before the
    /// solution is returned, and [`assemble_kkt`] builds from the raw
    /// `QpProblem`. So the per-row factor is the identity.
    ///
    /// That is an **invariant, not a fact of nature**. If anyone later builds
    /// the sensitivity KKT from equilibrated data — a plausible "make the
    /// factorization faster" change — `F` silently becomes the Ruiz diagonal,
    /// and every pin and release would read a mis-scaled multiplier against a
    /// natural-units step.
    ///
    /// **And it is currently unguarded, measured rather than assumed.** Making
    /// this return a non-identity vector turns *nothing* in the crate red,
    /// because the only consumer is the release half of `refine_step_onto_bounds`
    /// and [`supports_release`](Self::supports_release) is `false` here. So the
    /// value is correct and untested at the same time. The phase that turns
    /// release on is the phase that makes it load-bearing, and it owes this a
    /// guard — a leg comparing a released step against a re-solve, which cannot
    /// pass with a mis-scaled `F`.
    ///
    /// `the_step_is_unmoved_by_internal_equilibration` guards the neighbouring
    /// and weaker claim that `dx/db` itself does not depend on whether the
    /// solve equilibrated internally. That is worth having, but it exercises
    /// `parametric_step`, which never reads this factor.
    fn natural_units_factor(&self) -> Option<&[Number]> {
        None
    }

    fn bound_rows(&self) -> Option<&[BoundRow]> {
        Some(&self.bound_rows)
    }

    /// Both halves of fix-relax are available on this arm.
    ///
    /// Releasing is cheaper here than on the NLP one, and for a structural
    /// reason worth stating: the NLP's release *must* re-factor because an
    /// active bound puts `σ = z/s` on the `x` diagonal, and on a tightly
    /// converged bound that term destroys the released system's information in
    /// the converged factor — the better the solve converged, the worse a
    /// recovered answer would be. The convex active-set KKT carries no barrier
    /// term at all; the bound is an explicit row. Releasing it is exact, and
    /// costs one numeric refactorization against an unchanged sparsity pattern.
    fn supports_release(&self) -> bool {
        true
    }

    fn solve_released(&self, released: &[usize], rhs: &[Number], lhs: &mut [Number]) -> bool {
        self.solve_released_inner(released, rhs, lhs, false)
    }

    fn solve_released_step(&self, released: &[usize], rhs: &[Number], lhs: &mut [Number]) -> bool {
        self.solve_released_inner(released, rhs, lhs, true)
    }
}

impl QpKktBacksolver {
    /// Index of the active bound whose multiplier lives at compound row `row`.
    fn bound_at(&self, row: usize) -> Option<usize> {
        self.bound_rows.iter().position(|b| b.row == row)
    }

    /// Make the released system current, factoring it if the cached one is for
    /// a different released set.
    ///
    /// Neutralizing a row means zeroing its `±1` coupling to the variable and
    /// setting its diagonal to `−1`: the row then reads `−dz_a = rhs`, so the
    /// multiplier decouples and the variable is free. The **sparsity pattern is
    /// untouched**, which is the point — `refactor` reuses the symbolic
    /// factorization, so a release costs a numeric factorization rather than a
    /// fresh analyse.
    fn ensure_released(&self, key: &[usize]) -> bool {
        {
            let cached = self.released.borrow();
            if cached.as_ref().is_some_and(|rf| rf.key == key) {
                return true;
            }
        }
        let mut vals_reg = (*self.vals_reg).clone();
        let mut vals_true = (*self.vals_true).clone();
        for &row in key {
            let Some(k) = self.bound_at(row) else {
                return false;
            };
            let (coupling, diagonal) = self.slots[k];
            vals_reg[coupling] = 0.0;
            vals_true[coupling] = 0.0;
            vals_reg[diagonal] = -1.0;
            vals_true[diagonal] = -1.0;
        }
        let mut slot = self.released.borrow_mut();
        match slot.as_mut() {
            Some(rf) => {
                if rf.fact.refactor(&vals_reg).is_err() {
                    return false;
                }
                rf.key = key.to_vec();
                rf.vals_true = vals_true;
            }
            None => {
                let Some(backend) = self.release_backend.borrow_mut().take() else {
                    // Already consumed, which cannot happen: the `None` arm
                    // runs at most once, and the factor it builds is refactored
                    // thereafter.
                    return false;
                };
                let Ok(fact) = Factorization::new(
                    self.dim as Index,
                    (*self.airn).clone(),
                    (*self.ajcn).clone(),
                    vals_reg,
                    backend,
                ) else {
                    return false;
                };
                *slot = Some(ReleasedFactor {
                    key: key.to_vec(),
                    fact,
                    vals_true,
                });
            }
        }
        true
    }

    fn solve_released_inner(
        &self,
        released: &[usize],
        rhs: &[Number],
        lhs: &mut [Number],
        shift: bool,
    ) -> bool {
        if rhs.len() != self.dim || lhs.len() != self.dim {
            return false;
        }
        if released.is_empty() {
            return self.solve(rhs, lhs);
        }
        let mut key = released.to_vec();
        key.sort_unstable();
        key.dedup();
        if !self.ensure_released(&key) {
            return false;
        }

        lhs.copy_from_slice(rhs);
        if shift {
            // The released multiplier no longer acts on its variable, so its
            // base value moves onto that variable's `x` row with the sign that
            // row carries the bound's side with — `−z` for a lower bound,
            // `+z` for an upper one. Mirrors the NLP arm's
            // `shift_released_rhs`, which is the reference for this convention.
            for &row in &key {
                let Some(k) = self.bound_at(row) else {
                    return false;
                };
                let b = &self.bound_rows[k];
                if b.var_row >= self.dim {
                    return false;
                }
                lhs[row] = 0.0;
                let z = self.base_mult[k];
                lhs[b.var_row] += if b.lower { -z } else { z };
            }
        }

        let mut slot = self.released.borrow_mut();
        let Some(rf) = slot.as_mut() else {
            return false;
        };
        let mut scratch = self.scratch.borrow_mut();
        let vals_true = rf.vals_true.clone();
        match solve_refined(
            &mut rf.fact,
            &self.airn,
            &self.ajcn,
            &vals_true,
            lhs,
            &mut scratch,
        ) {
            Ok(res) => {
                self.last_residual.set(res);
                true
            }
            Err(()) => false,
        }
    }
}

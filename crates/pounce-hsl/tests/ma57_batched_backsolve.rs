//! `ma57_batched_backsolve`: the gate, and what opening it costs.
//!
//! MA57 answers
//! [`SparseSymLinearSolverInterface::multi_solve_matches_single_solve`]
//! from an option rather than from a measurement, which is the opposite
//! of what feral does, and the reason is here in this file's third test.
//! feral's blocked substitution really is a column-at-a-time cascade
//! below a threshold, so it can affirm bit-identity and re-measure the
//! claim on every run. MA57C blocks across columns, so the honest
//! factual answer is `false` at every width and the option is a
//! permission the user grants rather than a capability the backend has.
//!
//! Needs a linked `libcoinhsl`, like everything else in this crate, so
//! none of it runs in CI. The registry half — that the option exists and
//! defaults to `no` — is
//! `pounce-algorithm/tests/ma57_batched_backsolve_is_opt_in.rs`, which
//! does run there.
//!
//! **What this file is not evidence about.** The fixture below is a
//! 500-row banded SPD matrix, which is not the population the option was
//! measured on. The ~1-ulp trajectory divergence that makes the option
//! opt-in was observed on gh#809's review model — a 118276-row KKT
//! system under the limited-memory quasi-Newton path — and the
//! *absence* of a bit-level difference here would say nothing about
//! that: MA57 chooses its solve kernel from the factor's shape and the
//! right-hand-side count, so a small well-conditioned band may take the
//! unblocked path at every width this file uses.
//! `the_batched_and_per_column_answers_agree_to_tolerance` therefore
//! asserts agreement and *reports* the bit-level distance rather than
//! asserting either way. If someone with an HSL licence establishes a
//! width below which MA57C genuinely reproduces the per-column result on
//! representative KKT systems, that measurement belongs here, and it
//! would turn the option into a ceiling of feral's kind.

#![allow(clippy::unwrap_used)]

use pounce_common::options_list::OptionsList;
use pounce_common::types::{Index, Number};
use pounce_hsl::{Ma57Options, Ma57SolverInterface};
use pounce_linsol::{ESymSolverStatus, SparseSymLinearSolverInterface};

const N: Index = 500;
const NRHS: usize = 8;

/// Lower-triangle triplet (1-based) of a deterministic banded, strictly
/// diagonally dominant symmetric matrix. Dominant so no pivoting choice
/// depends on rounding: the two arms below must share a factorization
/// bit-for-bit for the back-substitution comparison to mean anything.
fn fixture() -> (Vec<Index>, Vec<Index>, Vec<Number>) {
    let (mut ia, mut ja, mut a) = (Vec::new(), Vec::new(), Vec::new());
    for i in 1..=N {
        ia.push(i);
        ja.push(i);
        a.push(8.0 + Number::from(i % 7));
        if i > 1 {
            ia.push(i);
            ja.push(i - 1);
            a.push(-1.0);
        }
        if i > 10 {
            ia.push(i);
            ja.push(i - 10);
            a.push(-0.5);
        }
    }
    (ia, ja, a)
}

/// Deterministic right-hand sides, packed column-major as the interface
/// wants them. A fixed LCG rather than a constant vector: identical
/// columns would make a reassociating kernel and a per-column loop agree
/// by symmetry and hide the thing this file is here to look at.
fn rhs_block(nrhs: usize) -> Vec<Number> {
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut v = Vec::with_capacity(N as usize * nrhs);
    for _ in 0..(N as usize * nrhs) {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        v.push(((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0);
    }
    v
}

fn backend(batched: bool) -> Ma57SolverInterface {
    let mut opts = OptionsList::default();
    if batched {
        opts.read_from_str("ma57_batched_backsolve yes", true)
            .expect("`yes` is a legal bool");
    }
    Ma57SolverInterface::with_options(Ma57Options::from_options_list(&opts, ""))
}

fn factored(batched: bool) -> Ma57SolverInterface {
    let (ia, ja, a) = fixture();
    let mut s = backend(batched);
    assert_eq!(
        s.initialize_structure(N, ia.len() as Index, &ia, &ja),
        ESymSolverStatus::Success
    );
    s.values_array_mut().copy_from_slice(&a);
    s
}

/// The default backend declines at every width — including the widths
/// `LowRankAugSystemSolver` actually offers, which are small.
///
/// The trait default is already `false`
/// (`pounce-linsol/src/sparse_sym_iface.rs`), so this would pass on a
/// backend with no override at all. That is deliberate: it is the
/// property a user relies on, and it must survive the override being
/// added, not just the override being absent.
#[test]
fn the_gate_is_closed_by_default() {
    let s = factored(false);
    for nrhs in [1usize, 2, 4, 8, 16, 64] {
        assert!(
            !s.multi_solve_matches_single_solve(nrhs),
            "nrhs={nrhs}: an unconfigured MA57 must not affirm bit-identity"
        );
    }
    assert!(!Ma57SolverInterface::new().multi_solve_matches_single_solve(2));
}

/// And the option opens it — at every width, with no ceiling.
///
/// The absence of a ceiling is the assertion, not an omission. feral
/// affirms only up to `FERAL_BITWISE_MULTI_SOLVE_MAX_NRHS` because it
/// knows where its own kernel switches; nothing in this repository can
/// establish that number for MA57, so inventing one would be a constant
/// no test could re-derive.
#[test]
fn the_option_opens_the_gate() {
    let s = factored(true);
    for nrhs in [1usize, 2, 4, 8, 16, 64, 1024] {
        assert!(
            s.multi_solve_matches_single_solve(nrhs),
            "nrhs={nrhs}: the option is the whole rule, so it cannot taper off"
        );
    }
}

/// The option is a **gate and nothing else**: at the same `nrhs`, the two
/// settings must produce bit-identical numbers.
///
/// This is the one that would catch the option being wired somewhere it
/// does not belong — an `ICNTL` entry in `apply_icntl`, a branch in
/// `backsolve`. Such a wiring would look like an optimization and would
/// change the answer of *every* MA57 solve, including the ones that never
/// consult the gate at all, because `multi_solve` is on the main KKT path
/// and the SMW batch is not.
#[test]
fn the_option_does_not_change_what_multi_solve_computes() {
    for nrhs in [1usize, NRHS] {
        let mut closed = factored(false);
        let mut open = factored(true);
        let (mut b_closed, mut b_open) = (rhs_block(nrhs), rhs_block(nrhs));
        assert_eq!(
            closed.multi_solve(true, &[], &[], nrhs as Index, &mut b_closed, false, 0),
            ESymSolverStatus::Success
        );
        assert_eq!(
            open.multi_solve(true, &[], &[], nrhs as Index, &mut b_open, false, 0),
            ESymSolverStatus::Success
        );
        assert_eq!(
            b_closed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b_open.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "nrhs={nrhs}: ma57_batched_backsolve changed the numbers MA57 \
             returns. It must only change the answer to \
             `multi_solve_matches_single_solve`; anything else moves every \
             MA57 solve, not just the batched ones."
        );
    }
}

/// The probe behind the option: batched versus per-column, on one
/// factor.
///
/// Asserts they agree to solve tolerance — a disagreement larger than
/// that is a real defect in the marshalling, not a reassociation — and
/// **reports** the bit-level distance without asserting it, for the
/// reason in this file's header: this fixture is not the population the
/// ~1-ulp divergence was measured on, so neither outcome here is
/// evidence about it. Run with `--nocapture` to read the number.
#[test]
fn the_batched_and_per_column_answers_agree_to_tolerance() {
    let mut batched = factored(false);
    let mut b = rhs_block(NRHS);
    assert_eq!(
        batched.multi_solve(true, &[], &[], NRHS as Index, &mut b, false, 0),
        ESymSolverStatus::Success
    );

    // Same matrix, same options, one column at a time against a single
    // factorization: `new_matrix` is true only for the first call.
    let mut per_col = factored(false);
    let src = rhs_block(NRHS);
    let mut one = Vec::with_capacity(N as usize * NRHS);
    for j in 0..NRHS {
        let mut col = src[j * N as usize..(j + 1) * N as usize].to_vec();
        assert_eq!(
            per_col.multi_solve(j == 0, &[], &[], 1, &mut col, false, 0),
            ESymSolverStatus::Success
        );
        one.extend_from_slice(&col);
    }

    let mut worst_rel = 0.0f64;
    let mut worst_ulps = 0i64;
    let mut differing = 0usize;
    for (x, y) in b.iter().zip(one.iter()) {
        let scale = x.abs().max(y.abs()).max(1e-300);
        worst_rel = worst_rel.max((x - y).abs() / scale);
        if x.to_bits() != y.to_bits() {
            differing += 1;
            worst_ulps = worst_ulps.max((x.to_bits() as i64 - y.to_bits() as i64).abs());
        }
    }
    println!(
        "ma57 nrhs={NRHS} on a {N}-row band: {differing}/{} entries differ, \
         worst {worst_ulps} ulp, worst relative {worst_rel:.3e}",
        b.len()
    );
    assert!(
        worst_rel < 1e-12,
        "batched and per-column MA57 disagree by {worst_rel:.3e} relative, \
         far past reassociation — this is a marshalling defect, not the \
         one-ulp effect ma57_batched_backsolve exists to gate"
    );
}

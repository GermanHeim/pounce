//! Diagnostic tool: read augmented systems dumped by
//! `POUNCE_DUMP_KKT=…` and compare FERAL, MA57, and ground-truth dense
//! LAPACK eigendecomposition on each.
//!
//! Reports for each system:
//!   - `n`, `nnz`
//!   - LAPACK's true negative-eigenvalue count + smallest-|eig|
//!   - FERAL's reported inertia + factor status + ‖A x − b‖∞
//!   - MA57's reported inertia + factor status + ‖A x − b‖∞
//!
//! Build: `cargo run --release -p pounce-cutest --features ma57 --bin kkt_compare -- /tmp/heart6_kkt.jsonl`

use pounce_common::types::Index;
use pounce_feral::FeralSolverInterface;
use pounce_hsl::Ma57SolverInterface;
use pounce_linsol::sparse_sym_iface::SparseSymLinearSolverInterface;
use pounce_linsol::ESymSolverStatus;
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Dump {
    n: i32,
    check_neg_evals: bool,
    num_neg_evals_expected: i32,
    num_neg_evals_actual: i32,
    status: String,
    irn: Vec<i32>,
    jcn: Vec<i32>,
    vals: Vec<f64>,
    rhs: Vec<f64>,
    sol: Vec<f64>,
}

// ---- LAPACK dsyev FFI (macOS Accelerate / system LAPACK) ----
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn dsyev_(
        jobz: *const u8,
        uplo: *const u8,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        w: *mut f64,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
}

/// Compute all eigenvalues of an n×n symmetric matrix stored in
/// column-major form (lower triangle used). Returns the sorted
/// eigenvalues (ascending).
fn dense_eigvals(n: usize, a: &mut [f64]) -> Vec<f64> {
    assert_eq!(a.len(), n * n);
    let n_i = n as i32;
    let lda = n_i;
    let mut w = vec![0.0; n];
    let mut info: i32 = 0;
    // Workspace query.
    let mut work_query = [0.0f64; 1];
    let lwork_query: i32 = -1;
    unsafe {
        dsyev_(
            b"N".as_ptr(),
            b"L".as_ptr(),
            &n_i,
            a.as_mut_ptr(),
            &lda,
            w.as_mut_ptr(),
            work_query.as_mut_ptr(),
            &lwork_query,
            &mut info,
        );
    }
    let lwork = work_query[0] as i32;
    let mut work = vec![0.0; lwork as usize];
    unsafe {
        dsyev_(
            b"N".as_ptr(),
            b"L".as_ptr(),
            &n_i,
            a.as_mut_ptr(),
            &lda,
            w.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork,
            &mut info,
        );
    }
    assert_eq!(info, 0, "dsyev failed: info={}", info);
    w
}

/// Build a dense column-major n×n matrix from 1-based lower-triangular triplets.
fn assemble_dense(n: usize, irn: &[i32], jcn: &[i32], vals: &[f64]) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for k in 0..vals.len() {
        let i = (irn[k] - 1) as usize;
        let j = (jcn[k] - 1) as usize;
        a[j * n + i] += vals[k];
        if i != j {
            a[i * n + j] += vals[k];
        }
    }
    a
}

fn matvec(n: usize, irn: &[i32], jcn: &[i32], vals: &[f64], x: &[f64], y: &mut [f64]) {
    y.iter_mut().for_each(|v| *v = 0.0);
    for k in 0..vals.len() {
        let i = (irn[k] - 1) as usize;
        let j = (jcn[k] - 1) as usize;
        y[i] += vals[k] * x[j];
        if i != j {
            y[j] += vals[k] * x[i];
        }
    }
    let _ = n;
}

fn run_backend<B: SparseSymLinearSolverInterface>(
    backend: &mut B,
    n: i32,
    irn: &[i32],
    jcn: &[i32],
    vals: &[f64],
    rhs: &[f64],
    expected_neg: i32,
) -> (String, i32, f64) {
    let st = backend.initialize_structure(n, vals.len() as Index, irn, jcn);
    if st != ESymSolverStatus::Success {
        return (format!("init_{:?}", st), -1, f64::NAN);
    }
    backend.values_array_mut().copy_from_slice(vals);

    let mut sol = rhs.to_vec();
    let st = backend.multi_solve(true, irn, jcn, 1, &mut sol, true, expected_neg);
    let neg = if backend.provides_inertia() {
        backend.number_of_neg_evals()
    } else {
        -1
    };
    let mut ax = vec![0.0; n as usize];
    matvec(n as usize, irn, jcn, vals, &sol, &mut ax);
    let resid: f64 = ax
        .iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    (format!("{:?}", st), neg, resid)
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: kkt_compare <jsonl>");
    let only_distinct: bool = std::env::var("DISTINCT").is_ok();
    let max_rows: usize = std::env::var("MAX_ROWS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let txt = std::fs::read_to_string(&path).expect("read jsonl");
    let mut seen = BTreeMap::<String, usize>::new();
    let mut printed_header = false;

    for (idx, line) in txt.lines().enumerate() {
        if idx >= max_rows {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let r: Dump = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("row {idx}: parse error: {e}");
                continue;
            }
        };

        // Optional dedup: bucket by status,n,expected,actual to avoid printing 300 nearly identical rows.
        if only_distinct {
            let key = format!(
                "{}_{}_e{}_a{}",
                r.status, r.n, r.num_neg_evals_expected, r.num_neg_evals_actual
            );
            let count = seen.entry(key.clone()).or_insert(0);
            *count += 1;
            if *count > 1 {
                continue;
            }
        }

        if !printed_header {
            println!(
                "{:>4} {:>3} {:>4} {:>2} {:>2}  {:>14}  {:<22}  {:<22}",
                "row",
                "n",
                "nnz",
                "exp",
                "logged",
                "lapack(neg|min|cond)",
                "FERAL(stat|neg|res)",
                "MA57(stat|neg|res)"
            );
            printed_header = true;
        }

        let n = r.n as usize;
        let nnz = r.vals.len();

        // LAPACK ground truth.
        let mut dense = assemble_dense(n, &r.irn, &r.jcn, &r.vals);
        let evals = dense_eigvals(n, &mut dense);
        let lapack_neg = evals.iter().filter(|&&e| e < 0.0).count();
        let lapack_min_abs = evals.iter().map(|e| e.abs()).fold(f64::INFINITY, f64::min);
        let lapack_max_abs = evals.iter().map(|e| e.abs()).fold(0.0_f64, f64::max);
        let cond = lapack_max_abs / lapack_min_abs;

        // FERAL
        let mut feral = FeralSolverInterface::new();
        let (fst, fneg, fres) = run_backend(
            &mut feral,
            r.n,
            &r.irn,
            &r.jcn,
            &r.vals,
            &r.rhs,
            r.num_neg_evals_expected,
        );

        // MA57
        let mut ma57 = Ma57SolverInterface::new();
        let (mst, mneg, mres) = run_backend(
            &mut ma57,
            r.n,
            &r.irn,
            &r.jcn,
            &r.vals,
            &r.rhs,
            r.num_neg_evals_expected,
        );

        let lapack_field = format!("{}|{:.1e}|{:.1e}", lapack_neg, lapack_min_abs, cond);
        let feral_field = format!("{}|{}|{:.1e}", fst, fneg, fres);
        let ma57_field = format!("{}|{}|{:.1e}", mst, mneg, mres);
        println!(
            "{:>4} {:>3} {:>4} {:>2} {:>2}  {:<14}  {:<22}  {:<22}",
            idx,
            n,
            nnz,
            r.num_neg_evals_expected,
            r.num_neg_evals_actual,
            lapack_field,
            feral_field,
            ma57_field
        );

        // Note dump-time vs offline mismatch.
        if r.status != fst {
            println!(
                "    note: dump-time pounce-FERAL status={}, offline rerun={}",
                r.status, fst
            );
        }
    }
}

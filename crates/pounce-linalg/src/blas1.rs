//! Reference BLAS-1 in Rust.
//!
//! Mirrors `LinAlg/IpBlas.{hpp,cpp}` BLAS-1 entry points. To preserve
//! bit-equivalence with upstream Ipopt, the loop structure here
//! follows the netlib reference Fortran BLAS exactly:
//!
//! - `dot`, `axpy`, `copy`, `scal`: simple scalar loop, no unrolling
//!   (we deliberately do **not** use SIMD intrinsics or `mul_add`).
//! - `nrm2`: the scale/ssq accumulation from netlib `dnrm2`,
//!   bit-identical to a reference-BLAS build of upstream Ipopt.
//! - `iamax`: tie-break on first occurrence (matches `idamax`).
//! - `asum`: straight scalar sum, no unroll.
//!
//! All routines accept `incX != 1` strides like Fortran BLAS, but the
//! common `incX = incY = 1` case is still a tight loop.

use pounce_common::types::{Index, Number};

#[inline]
fn step(idx: usize, inc: Index) -> usize {
    // Fortran BLAS computes `1 + (k-1) * INC`, but with INC<=0 the
    // start offset is `(1-N)*INC + 1`. Ipopt only ever calls with
    // `inc>0` here; the FFI fallback in `IpBlas.cpp` handles inc<0
    // separately. We just take inc as a positive stride.
    idx * (inc as usize)
}

/// Dot product `x · y`. Equivalent to `IpBlasDot` / Fortran `DDOT`.
pub fn dot(x: &[Number], inc_x: Index, y: &[Number], inc_y: Index, n: Index) -> Number {
    let n = n.max(0) as usize;
    let mut s = 0.0;
    if inc_x == 1 && inc_y == 1 {
        for k in 0..n {
            s += x[k] * y[k];
        }
    } else {
        for k in 0..n {
            s += x[step(k, inc_x)] * y[step(k, inc_y)];
        }
    }
    s
}

/// 2-norm `‖x‖₂` via the netlib scale/ssq scheme — overflow-safe and
/// bit-identical to reference `DNRM2`.
pub fn nrm2(x: &[Number], inc_x: Index, n: Index) -> Number {
    let n = n.max(0) as usize;
    if n == 0 || inc_x < 1 {
        return 0.0;
    }
    if n == 1 {
        return x[0].abs();
    }
    let mut scale = 0.0_f64;
    let mut ssq = 1.0_f64;
    for k in 0..n {
        let v = x[step(k, inc_x)];
        if v != 0.0 {
            let absxi = v.abs();
            if scale < absxi {
                let r = scale / absxi;
                ssq = 1.0 + ssq * (r * r);
                scale = absxi;
            } else {
                let r = absxi / scale;
                ssq += r * r;
            }
        }
    }
    scale * ssq.sqrt()
}

/// 1-norm `Σ |xᵢ|`. Equivalent to `IpBlasAsum` / `DASUM`.
pub fn asum(x: &[Number], inc_x: Index, n: Index) -> Number {
    let n = n.max(0) as usize;
    let mut s = 0.0;
    if inc_x == 1 {
        for v in &x[..n] {
            s += v.abs();
        }
    } else {
        for k in 0..n {
            s += x[step(k, inc_x)].abs();
        }
    }
    s
}

/// 0-based index of the largest-magnitude entry. Equivalent to
/// `IpBlasIamax` − 1 (Fortran `IDAMAX` returns 1-based; Ipopt's
/// `IpBlasIamax` returns the Fortran value, so callers add/subtract
/// 1 as needed). We return 0-based indexing throughout POUNCE.
///
/// Tie-break: first occurrence wins, matching `IDAMAX`.
pub fn iamax(x: &[Number], inc_x: Index, n: Index) -> Index {
    let n = n.max(0) as usize;
    if n == 0 {
        return 0;
    }
    let mut best_idx: usize = 0;
    let mut best_val = x[0].abs();
    for k in 1..n {
        let v = x[step(k, inc_x)].abs();
        if v > best_val {
            best_val = v;
            best_idx = k;
        }
    }
    best_idx as Index
}

/// `y ← x`. Equivalent to `IpBlasCopy` / `DCOPY`.
pub fn copy(x: &[Number], inc_x: Index, y: &mut [Number], inc_y: Index, n: Index) {
    let n = n.max(0) as usize;
    if inc_x == 1 && inc_y == 1 {
        y[..n].copy_from_slice(&x[..n]);
        return;
    }
    for k in 0..n {
        y[step(k, inc_y)] = x[step(k, inc_x)];
    }
}

/// `y ← α x + y`. Equivalent to `IpBlasAxpy` / `DAXPY`.
pub fn axpy(alpha: Number, x: &[Number], inc_x: Index, y: &mut [Number], inc_y: Index, n: Index) {
    let n = n.max(0) as usize;
    if alpha == 0.0 {
        return;
    }
    if inc_x == 1 && inc_y == 1 {
        for k in 0..n {
            y[k] += alpha * x[k];
        }
    } else {
        for k in 0..n {
            y[step(k, inc_y)] += alpha * x[step(k, inc_x)];
        }
    }
}

/// `x ← α x`. Equivalent to `IpBlasScal` / `DSCAL`.
pub fn scal(alpha: Number, x: &mut [Number], inc_x: Index, n: Index) {
    let n = n.max(0) as usize;
    if inc_x == 1 {
        for v in &mut x[..n] {
            *v *= alpha;
        }
    } else {
        for k in 0..n {
            x[step(k, inc_x)] *= alpha;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_basic() {
        let x = [1.0, 2.0, 3.0];
        let y = [10.0, 20.0, 30.0];
        assert_eq!(dot(&x, 1, &y, 1, 3), 1.0 * 10.0 + 2.0 * 20.0 + 3.0 * 30.0);
    }

    #[test]
    fn dot_strided() {
        let x = [1.0, 99.0, 2.0, 99.0, 3.0];
        let y = [10.0, 20.0, 30.0];
        assert_eq!(dot(&x, 2, &y, 1, 3), 10.0 + 40.0 + 90.0);
    }

    #[test]
    fn nrm2_zero_size() {
        assert_eq!(nrm2(&[1.0], 1, 0), 0.0);
    }

    #[test]
    fn nrm2_canonical() {
        let x = [3.0, 4.0];
        assert!((nrm2(&x, 1, 2) - 5.0).abs() < 1e-15);
    }

    #[test]
    fn nrm2_overflow_safe() {
        // sqrt(2) * 1e200 is well within f64 range, but the naive
        // sum-of-squares would overflow at 1e200^2 = 1e400.
        let x = [1e200, 1e200];
        let n = nrm2(&x, 1, 2);
        let expected = 2.0_f64.sqrt() * 1e200;
        assert!(((n - expected) / expected).abs() < 1e-15);
    }

    #[test]
    fn iamax_first_of_ties() {
        let x = [1.0, -3.0, 3.0, 2.0, -3.0];
        assert_eq!(iamax(&x, 1, 5), 1);
    }

    #[test]
    fn axpy_basic() {
        let x = [1.0, 2.0, 3.0];
        let mut y = [10.0, 20.0, 30.0];
        axpy(2.0, &x, 1, &mut y, 1, 3);
        assert_eq!(y, [12.0, 24.0, 36.0]);
    }

    #[test]
    fn axpy_alpha_zero_is_identity() {
        let x = [f64::NAN, f64::INFINITY, 0.0];
        let mut y = [1.0, 2.0, 3.0];
        axpy(0.0, &x, 1, &mut y, 1, 3);
        assert_eq!(y, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn scal_zero_is_explicit() {
        let mut x = [1.0, 2.0, 3.0];
        scal(0.0, &mut x, 1, 3);
        assert_eq!(x, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn copy_strided() {
        let x = [1.0, 2.0, 3.0];
        let mut y = [0.0; 6];
        copy(&x, 1, &mut y, 2, 3);
        assert_eq!(y, [1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn asum_basic() {
        let x = [-1.0, 2.0, -3.0, 4.0];
        assert_eq!(asum(&x, 1, 4), 10.0);
    }
}

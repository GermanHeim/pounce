//! Sparse finite-difference Lagrangian Hessian, recovered by graph
//! coloring from the **analytic Jacobian** (Curtis, Powell & Reid 1974;
//! Coleman & Moré 1983).
//!
//! # Why this exists
//!
//! The models this targets — a direct-collocation transcription built
//! from an FMU or a CasADi `DaeBuilder` — supply analytic first
//! derivatives and no second derivatives. Today that leaves only
//! [`crate::hess::lim_mem_quasi_newton`], and on
//! `benchmarks/large_scale` `laptime` the difference is stark: the exact
//! Hessian converges in 30 iterations where limited-memory takes 246,
//! and one mesh refinement later limited-memory does not converge at all.
//!
//! But an analytic Jacobian is already enough to *build* the Hessian.
//! The Lagrangian gradient
//!
//! ```text
//!     ∇ₓL(x, y) = ∇f(x) + J_c(x)ᵀ y_c + J_d(x)ᵀ y_d
//! ```
//!
//! is available in closed form, so its directional derivative
//!
//! ```text
//!     ∇²ₓₓL · d ≈ [ ∇ₓL(x + d, y) − ∇ₓL(x, y) ] / h
//! ```
//!
//! costs one gradient and one Jacobian evaluation — and with a known
//! sparsity pattern, one probe recovers a whole *group* of structurally
//! orthogonal columns at once rather than a single one.
//!
//! # Why it is affordable
//!
//! Measured on `laptime` at `N = 160`: a Jacobian evaluation costs
//! 5.4 ms against a 92.6 ms iteration, and the Hessian pattern has
//! `rho_max = 15` with a mean row of 5.68 — **unchanged at `N = 320`**
//! (`POUNCE_HESS_PATTERN_CENSUS`). The row width is set by the
//! per-stage stencil, not the horizon, so the number of probes does not
//! grow with the mesh. That is the property that makes the whole scheme
//! viable: the cost per Hessian is constant in problem size while the
//! iteration count it buys is the exact path's.
//!
//! # The partition
//!
//! Columns are grouped so that no two columns in a group share a row of
//! the pattern (Curtis-Powell-Reid structural orthogonality). Probing
//! group `g` with `d = Σ_{j∈g} h_j e_j` then gives, for every row `i`,
//!
//! ```text
//!     w_i = Σ_{j∈g} H_ij h_j = H_ij h_j   for the unique j ∈ g with H_ij ≠ 0
//! ```
//!
//! so each entry is read off directly with no linear solve. A *star*
//! coloring would exploit symmetry to use roughly half as many groups;
//! CPR is used here because it is straightforward to get right, and the
//! measurement below is therefore a conservative bound on what the
//! technique can do.
//!
//! # The pattern
//!
//! Two sources, selected by `fd_hessian_pattern`:
//!
//! * `declared` — the TNLP's own Hessian sparsity, when it declares one
//!   (every `.nl` does, through AMPL's AD). Requires no values, only the
//!   structure call, so it is available to a model that cannot evaluate
//!   second derivatives.
//! * `jacobian` — derived as `⋃_j supp(∇g_j) ⊗ supp(∇g_j)`, which needs
//!   nothing beyond the Jacobian pattern every TNLP must declare. This
//!   is a strict **superset** of the true pattern, which is safe (a
//!   superset costs extra groups, never a wrong answer) but not free: on
//!   `laptime` it is 146 267 nonzeros against the true 28 000.
//!
//! A superset is always safe; a subset would silently drop curvature, so
//! there is no fallback that guesses.

use crate::hess::r#trait::HessianUpdater;
use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use pounce_common::types::{Index, Number};
use pounce_linalg::Vector;
use pounce_linalg::compound_vector::CompoundVector;
use pounce_linalg::dense_vector::DenseVector;
use pounce_linalg::triplet::{GenTMatrix, SymTMatrix, SymTMatrixSpace};
use std::rc::Rc;

/// Relative finite-difference step. `sqrt(eps)` is the classic
/// forward-difference optimum: truncation error is `O(h)` and round-off
/// `O(eps/h)`, and they balance there.
const FD_REL_STEP: Number = 1.4901161193847656e-8;

/// Where the Hessian sparsity pattern comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FdPatternSource {
    /// The TNLP's declared Hessian structure, when it has one.
    Declared,
    /// `⋃_j supp(∇g_j) ⊗ supp(∇g_j)`, from the Jacobian pattern alone.
    Jacobian,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FdStats {
    pub n: usize,
    pub nnz: usize,
    pub groups: usize,
    pub rho_max: usize,
    /// Probes per Hessian as a fraction of `n` — the quantity a dense
    /// finite-difference scheme would pay in full.
    pub compression: f64,
}

pub struct FdHessianUpdater {
    pub pattern_source: FdPatternSource,
    /// Assembled pattern, lower triangle, 1-based (the exact-Hessian
    /// path's own convention).
    space: Option<Rc<SymTMatrixSpace>>,
    /// Columns of each structurally orthogonal group.
    groups: Vec<Vec<Index>>,
    /// For every stored (lower-triangle) entry `k`, the 0-based
    /// `(row, col)` it holds.
    entries: Vec<(Index, Index)>,
    /// Stored-entry indices owned by each column, i.e. recovered from the
    /// probe of that column's group. An entry `(i, j)` is recovered from
    /// `j`'s probe by reading component `i`.
    by_col: Vec<Vec<u32>>,
    stats: FdStats,
    reported: bool,
}

impl FdHessianUpdater {
    pub fn new(pattern_source: FdPatternSource) -> Self {
        Self {
            pattern_source,
            space: None,
            groups: Vec::new(),
            entries: Vec::new(),
            by_col: Vec::new(),
            stats: FdStats::default(),
            reported: false,
        }
    }

    pub fn stats(&self) -> FdStats {
        self.stats
    }

    /// Group columns so that no two in a group share a row — greedy
    /// coloring of the column intersection graph, columns taken in
    /// descending degree (largest-first, which is what makes greedy
    /// competitive with the optimum on sparse patterns).
    fn build_groups(n: usize, cols_of_row: &[Vec<Index>], rows_of_col: &[Vec<Index>]) -> Vec<Vec<Index>> {
        let mut order: Vec<Index> = (0..n as Index).collect();
        order.sort_unstable_by_key(|&j| std::cmp::Reverse(rows_of_col[j as usize].len()));

        let mut color = vec![usize::MAX; n];
        // Reused scratch: `forbidden[c] == j+1` marks colour `c` taken by
        // a neighbour of the column being processed, without clearing the
        // whole array each time.
        let mut forbidden = vec![0usize; n + 1];
        let mut n_colors = 0usize;

        for &j in &order {
            let stamp = j as usize + 1;
            for &i in &rows_of_col[j as usize] {
                for &k in &cols_of_row[i as usize] {
                    let c = color[k as usize];
                    if c != usize::MAX {
                        forbidden[c] = stamp;
                    }
                }
            }
            let mut c = 0usize;
            while c < n_colors && forbidden[c] == stamp {
                c += 1;
            }
            if c == n_colors {
                n_colors += 1;
            }
            color[j as usize] = c;
        }

        let mut groups = vec![Vec::new(); n_colors];
        for j in 0..n {
            groups[color[j]].push(j as Index);
        }
        groups
    }

    fn build_structure(
        &mut self,
        n: usize,
        declared: Option<&(Vec<Index>, Vec<Index>)>,
        jac_c: &GenTMatrix,
        jac_d: &GenTMatrix,
    ) {
        // ---- lower-triangle pattern ---------------------------------
        let mut pairs: Vec<(Index, Index)> = Vec::new();
        match (self.pattern_source, declared) {
            (FdPatternSource::Declared, Some((ir, jc))) => {
                for (&i, &j) in ir.iter().zip(jc.iter()) {
                    let (a, b) = (i - 1, j - 1);
                    pairs.push(if a >= b { (a, b) } else { (b, a) });
                }
            }
            _ => {
                // `⋃_j supp(∇g_j) ⊗ supp(∇g_j)` over both Jacobians.
                for jac in [jac_c, jac_d] {
                    let n_rows = jac.space().n_rows() as usize;
                    let mut by_row: Vec<Vec<Index>> = vec![Vec::new(); n_rows + 1];
                    for (&i, &j) in jac.irows().iter().zip(jac.jcols().iter()) {
                        by_row[i as usize].push(j - 1);
                    }
                    for row in by_row.iter_mut() {
                        row.sort_unstable();
                        row.dedup();
                        for (a, &ca) in row.iter().enumerate() {
                            for &cb in row.iter().take(a + 1) {
                                pairs.push((ca, cb));
                            }
                        }
                    }
                }
            }
        }
        // The diagonal always belongs: a structurally empty `(1,1)` row
        // is carried by the barrier term alone and costs the
        // factorization a near-singular pivot on every one of them.
        for i in 0..n {
            pairs.push((i as Index, i as Index));
        }
        pairs.sort_unstable();
        pairs.dedup();

        // ---- adjacency over the FULL symmetric pattern --------------
        //
        // Orthogonality and recovery both need both triangles: probing
        // column `j` moves every row `i` with `H_ij ≠ 0`, whichever
        // triangle that entry is stored in.
        let mut rows_of_col: Vec<Vec<Index>> = vec![Vec::new(); n];
        let mut cols_of_row: Vec<Vec<Index>> = vec![Vec::new(); n];
        for &(i, j) in &pairs {
            rows_of_col[j as usize].push(i);
            cols_of_row[i as usize].push(j);
            if i != j {
                rows_of_col[i as usize].push(j);
                cols_of_row[j as usize].push(i);
            }
        }
        let rho_max = cols_of_row.iter().map(|r| r.len()).max().unwrap_or(0);

        let groups = Self::build_groups(n, &cols_of_row, &rows_of_col);

        // ---- recovery map -------------------------------------------
        //
        // Entry `(i, j)` is recovered from column `j`'s probe by reading
        // component `i`. Each stored entry is owned by exactly one
        // column, so every entry is written exactly once per Hessian.
        let mut by_col: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (k, &(_, j)) in pairs.iter().enumerate() {
            by_col[j as usize].push(k as u32);
        }

        self.stats = FdStats {
            n,
            nnz: pairs.len(),
            groups: groups.len(),
            rho_max,
            compression: groups.len() as f64 / n.max(1) as f64,
        };
        let irows: Vec<Index> = pairs.iter().map(|&(i, _)| i + 1).collect();
        let jcols: Vec<Index> = pairs.iter().map(|&(_, j)| j + 1).collect();
        self.space = Some(SymTMatrixSpace::new(n as Index, irows, jcols));
        self.entries = pairs;
        self.groups = groups;
        self.by_col = by_col;
    }
}

impl HessianUpdater for FdHessianUpdater {
    fn update_hessian(&mut self, data: &IpoptDataHandle, cq: &IpoptCqHandle) -> bool {
        let (curr_x, curr_y_c, curr_y_d) = match data.borrow().curr.as_ref() {
            Some(c) => (c.x.clone(), c.y_c.clone(), c.y_d.clone()),
            None => return true,
        };
        let nlp = Rc::clone(cq.borrow().nlp());

        let base_grad_f = cq.borrow().curr_grad_f();
        let base_jac_c = cq.borrow().curr_jac_c();
        let base_jac_d = cq.borrow().curr_jac_d();
        let (Some(jc), Some(jd)) = (
            base_jac_c.as_any().downcast_ref::<GenTMatrix>(),
            base_jac_d.as_any().downcast_ref::<GenTMatrix>(),
        ) else {
            return false;
        };

        let x = flat(&*curr_x);
        let n = x.len();
        if self.space.is_none() {
            let declared = nlp.borrow().uninitialized_h();
            let declared_pat = declared
                .as_any()
                .downcast_ref::<SymTMatrix>()
                .filter(|t| t.nonzeros() > 0)
                .map(|t| (t.irows().to_vec(), t.jcols().to_vec()));
            self.build_structure(n, declared_pat.as_ref(), jc, jd);
            if !self.reported && std::env::var("POUNCE_FD_HESSIAN_DEBUG").is_ok() {
                self.reported = true;
                eprintln!("fd-hessian: {:?}", self.stats);
            }
        }

        // Baseline `∇ₓL` at the current iterate, from quantities the
        // algorithm has already evaluated — no extra NLP call.
        let mut base = curr_x.make_new();
        base.copy(&*base_grad_f);
        base_jac_c.trans_mult_vector(1.0, &*curr_y_c, 1.0, &mut *base);
        base_jac_d.trans_mult_vector(1.0, &*curr_y_d, 1.0, &mut *base);
        let base = flat(&*base);

        // Per-column step, `sqrt(eps)` scaled by the variable's own
        // magnitude.
        //
        // No bound guard is attempted from `nlp.x_l()` / `x_u()`: those
        // live in Ipopt's *compressed* bounded-variable space, one entry
        // per variable that HAS that bound, not one per variable, so
        // indexing them by variable index is simply wrong (it panicked
        // here before this was understood). The protection that matters
        // is applied below instead, and is stronger: a probe that lands
        // outside a `sqrt` or `log` domain returns NaN rather than an
        // error, so each group's result is checked for finiteness and
        // retried with the step reversed.
        let steps: Vec<Number> = (0..n)
            .map(|j| FD_REL_STEP * x[j].abs().max(1.0))
            .collect();

        let space = Rc::clone(self.space.as_ref().expect("structure built above"));
        let mut w = SymTMatrix::new(Rc::clone(&space));
        {
            let vals = w.values_mut();
            vals.iter_mut().for_each(|v| *v = 0.0);

            let mut probe = curr_x.make_new();
            let mut gl = curr_x.make_new();
            let mut xp = x.clone();
            for group in &self.groups {
                // Forward step first; on a non-finite result the whole
                // group is retried backwards. A collocation model is full
                // of `sqrt` and `log`, and a probe that leaves the domain
                // yields NaN silently — which would then be scattered
                // straight into `W`.
                let mut sign = 1.0;
                let mut g1;
                loop {
                    xp.copy_from_slice(&x);
                    for &j in group {
                        xp[j as usize] += sign * steps[j as usize];
                    }
                    set_expanded(probe.as_mut(), &xp);

                    nlp.borrow_mut().eval_grad_f(&*probe, &mut *gl);
                    let pj_c = nlp.borrow_mut().eval_jac_c(&*probe);
                    pj_c.trans_mult_vector(1.0, &*curr_y_c, 1.0, &mut *gl);
                    let pj_d = nlp.borrow_mut().eval_jac_d(&*probe);
                    pj_d.trans_mult_vector(1.0, &*curr_y_d, 1.0, &mut *gl);
                    g1 = flat(&*gl);

                    if g1.iter().all(|v| v.is_finite()) {
                        break;
                    }
                    if sign < 0.0 {
                        // Both directions leave the domain. Publishing a
                        // NaN block would be reported as a converged
                        // restoration failure with nothing naming the
                        // cause, so fail loudly instead.
                        return false;
                    }
                    sign = -1.0;
                }

                for &j in group {
                    let hj = sign * steps[j as usize];
                    for &k in &self.by_col[j as usize] {
                        let (i, _) = self.entries[k as usize];
                        vals[k as usize] = (g1[i as usize] - base[i as usize]) / hj;
                    }
                }
            }
        }
        data.borrow_mut().w = Some(Rc::new(w));
        true
    }
}

fn flat(v: &dyn Vector) -> Vec<Number> {
    if let Some(dv) = v.as_any().downcast_ref::<DenseVector>() {
        return dv.expanded_values();
    }
    if let Some(cv) = v.as_any().downcast_ref::<CompoundVector>() {
        let mut out = Vec::with_capacity(cv.dim() as usize);
        for i in 0..cv.n_comps() {
            out.extend(flat(cv.comp(i)));
        }
        return out;
    }
    panic!("FdHessianUpdater: unsupported primal vector type");
}

fn set_expanded(dst: &mut dyn Vector, values: &[Number]) {
    if let Some(dv) = dst.as_any_mut().downcast_mut::<DenseVector>() {
        dv.set_values(values);
        return;
    }
    if let Some(cv) = dst.as_any_mut().downcast_mut::<CompoundVector>() {
        let dims: Vec<usize> = (0..cv.n_comps()).map(|i| cv.comp(i).dim() as usize).collect();
        let mut off = 0usize;
        for (i, &d) in dims.iter().enumerate() {
            set_expanded(cv.comp_mut(i as Index), &values[off..off + d]);
            off += d;
        }
        return;
    }
    panic!("FdHessianUpdater: unsupported primal vector type");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tridiagonal pattern must group into exactly the number of
    /// colours a structurally orthogonal partition needs, and the groups
    /// must actually be orthogonal — two columns sharing a row in the
    /// same group would make the recovery read a sum of two entries as if
    /// it were one, silently.
    #[test]
    fn groups_are_structurally_orthogonal() {
        let n = 12usize;
        let mut pairs: Vec<(Index, Index)> = Vec::new();
        for i in 0..n as Index {
            pairs.push((i, i));
            if i > 0 {
                pairs.push((i, i - 1));
            }
        }
        let mut rows_of_col: Vec<Vec<Index>> = vec![Vec::new(); n];
        let mut cols_of_row: Vec<Vec<Index>> = vec![Vec::new(); n];
        for &(i, j) in &pairs {
            rows_of_col[j as usize].push(i);
            cols_of_row[i as usize].push(j);
            if i != j {
                rows_of_col[i as usize].push(j);
                cols_of_row[j as usize].push(i);
            }
        }
        let groups = FdHessianUpdater::build_groups(n, &cols_of_row, &rows_of_col);

        // every column appears exactly once
        let mut seen = vec![0usize; n];
        for g in &groups {
            for &j in g {
                seen[j as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&c| c == 1), "columns not partitioned: {seen:?}");

        // and no two columns in a group share a row
        for g in &groups {
            for (a, &ja) in g.iter().enumerate() {
                for &jb in g.iter().skip(a + 1) {
                    let ra: std::collections::HashSet<_> =
                        rows_of_col[ja as usize].iter().collect();
                    for r in &rows_of_col[jb as usize] {
                        assert!(!ra.contains(r), "columns {ja} and {jb} share row {r}");
                    }
                }
            }
        }
        // a tridiagonal matrix needs 3 groups, not 12
        assert_eq!(groups.len(), 3, "groups: {}", groups.len());
    }

    /// A dense row forces every other column apart from it: the group
    /// count is then bounded below by that row's width, which is the
    /// `rho_max` the cost model is built on.
    #[test]
    fn a_dense_row_forces_its_width_in_groups() {
        let n = 8usize;
        let mut pairs: Vec<(Index, Index)> = Vec::new();
        for i in 0..n as Index {
            pairs.push((i, i));
        }
        // row 7 touches every column
        for j in 0..7 as Index {
            pairs.push((7, j));
        }
        let mut rows_of_col: Vec<Vec<Index>> = vec![Vec::new(); n];
        let mut cols_of_row: Vec<Vec<Index>> = vec![Vec::new(); n];
        for &(i, j) in &pairs {
            rows_of_col[j as usize].push(i);
            cols_of_row[i as usize].push(j);
            if i != j {
                rows_of_col[i as usize].push(j);
                cols_of_row[j as usize].push(i);
            }
        }
        let groups = FdHessianUpdater::build_groups(n, &cols_of_row, &rows_of_col);
        assert_eq!(groups.len(), n, "a dense row must separate every column");
    }
}

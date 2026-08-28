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
//! so each entry is read off directly with no linear solve.
//!
//! A *star* colouring (`fd_hessian_coloring=star`) lets an entry be read
//! from **either** endpoint's probe and so needs fewer groups: 76 → 42 on
//! the Jacobian-derived pattern here, 17 → 16 on the declared one. Its
//! recovery is algebraically exact — `overlapping_cliques_are_validated_not_assumed`
//! verifies that by recovering a known matrix through it.
//!
//! **And it is still the wrong choice on a dense pattern, which is why CPR
//! is the default.** On `laptime`, star colouring over the Jacobian-derived
//! pattern takes 404 iterations to an objective of 65.368334 where CPR takes
//! 38 to 65.371106.
//!
//! The cause is not group size — the measurement rules that out, since
//! `declared/star` packs the *largest* groups of the four (580 columns per
//! probe against `jacobian/cpr`'s 122) and converges in 30 iterations with
//! the exact objective:
//!
//! | pattern / colouring | groups | cols per group | result |
//! |---|---|---|---|
//! | declared / cpr | 17 | 546 | Optimal, 30 it |
//! | declared / star | 16 | 580 | Optimal, 30 it |
//! | jacobian / cpr | 76 | 122 | Optimal, 38 it |
//! | jacobian / star | 42 | 221 | Acceptable, 404 it, wrong objective |
//!
//! The cause is the **finite-difference remainder**. Direct-recovery theory
//! assumes exact Hessian-vector products; a forward difference also carries
//! `½ Σ_{m,p ∈ g} T_imp h_m h_p` into row `i`, where `T` is the third
//! derivative. `T_imp ≠ 0` needs `i`, `m` and `p` in a common constraint's
//! support, hence `H_im ≠ 0` **and** `H_ip ≠ 0`. CPR's distance-2 property
//! forbids two such columns in one group, so those cross terms vanish
//! structurally. A star colouring only guarantees the single-neighbour
//! property for the pair being recovered, so the cross terms survive — and
//! they matter exactly when the pattern is dense (`rho_max` 59 here against
//! the declared pattern's 15).
//!
//! So star colouring is safe on a sparse declared pattern and unsafe on a
//! Jacobian-derived one, which is the mode most models need. It stays
//! opt-in. Every colouring is additionally validated entry by entry before
//! use, unconditionally — that check began as a `debug_assert`, which is
//! compiled out in release, and a Hessian that is wrong but plausible is the
//! failure this module is most exposed to.
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

/// How columns are grouped into probes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FdColoring {
    /// Curtis-Powell-Reid: no two columns in a group share a row. Treats
    /// the Hessian as a general matrix, so it ignores symmetry and pays
    /// for it — this is a distance-2 colouring of the adjacency graph.
    Cpr,
    /// Star colouring: a proper colouring in which every path on four
    /// vertices uses at least three colours, i.e. every bichromatic
    /// component is a star (Coleman & Moré 1983; Gebremedhin, Manne &
    /// Pothen 2005). Exploits symmetry — `H_ij` may be read from the
    /// probe of *either* endpoint's colour — so it needs materially
    /// fewer groups than CPR for the same pattern.
    Star,
}

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
    /// Whether the requested star colouring failed validation and CPR was
    /// substituted.
    pub coloring_fell_back: bool,
}

pub struct FdHessianUpdater {
    pub pattern_source: FdPatternSource,
    pub coloring: FdColoring,
    /// Reuse the previous Hessian when neither the primal iterate nor the
    /// multipliers have moved by more than this, relative to their own
    /// magnitude (`fd_hessian_reuse_tol`). `0` rebuilds every iteration.
    ///
    /// **Both** are tested, not just `x`: `∇²L = ∇²f + Σ yⱼ ∇²cⱼ` depends
    /// on the multipliers too, so a cached Hessian is stale the moment `y`
    /// moves even if `x` has not.
    pub reuse_tol: Number,
    /// Assembled pattern, lower triangle, 1-based (the exact-Hessian
    /// path's own convention).
    space: Option<Rc<SymTMatrixSpace>>,
    /// Columns of each probe group.
    groups: Vec<Vec<Index>>,
    /// For every stored (lower-triangle) entry `k`: which group's probe
    /// carries it, which component of that probe to read, and which
    /// column's step it must be divided by.
    ///
    /// Under CPR this is always `(colour(j), i, j)`. Under a star
    /// colouring it is that *or* `(colour(i), j, i)`, whichever endpoint
    /// is the leaf of the bichromatic star — which is the whole reason a
    /// star colouring needs fewer groups.
    recovery: Vec<(u32, u32, u32)>,
    /// Stored-entry indices carried by each group's probe.
    by_group: Vec<Vec<u32>>,
    stats: FdStats,
    reported: bool,
    /// Cached iterate and Hessian for the reuse test.
    prev_x: Option<Vec<Number>>,
    prev_y: Option<Vec<Number>>,
    prev_w: Option<Rc<SymTMatrix>>,
    pub reused: u64,
    pub rebuilt: u64,
}

impl FdHessianUpdater {
    pub fn new(pattern_source: FdPatternSource) -> Self {
        Self {
            pattern_source,
            coloring: FdColoring::Cpr,
            reuse_tol: 0.0,
            space: None,
            groups: Vec::new(),
            recovery: Vec::new(),
            by_group: Vec::new(),
            stats: FdStats::default(),
            reported: false,
            prev_x: None,
            prev_y: None,
            prev_w: None,
            reused: 0,
            rebuilt: 0,
        }
    }

    pub fn stats(&self) -> FdStats {
        self.stats
    }

    /// Curtis-Powell-Reid grouping: no two columns in a group share a
    /// row. Greedy, largest-degree-first, over the column intersection
    /// graph — which for a symmetric pattern is distance-2 adjacency.
    fn color_cpr(n: usize, cols_of_row: &[Vec<Index>], rows_of_col: &[Vec<Index>]) -> Vec<usize> {
        let mut order: Vec<Index> = (0..n as Index).collect();
        order.sort_unstable_by_key(|&j| std::cmp::Reverse(rows_of_col[j as usize].len()));

        let mut color = vec![usize::MAX; n];
        let mut forbidden = vec![usize::MAX; n + 1];
        let mut n_colors = 0usize;

        for &j in &order {
            let stamp = j as usize;
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
        color
    }

    /// Star colouring of the adjacency graph: a proper colouring in which
    /// no path on four vertices is bichromatic, so every bichromatic
    /// component is a star (Gebremedhin, Manne & Pothen 2005, Alg. 4).
    ///
    /// This is what lets symmetry be exploited. Under CPR an entry must
    /// come from its column's probe; under a star colouring it may come
    /// from *either* endpoint's, and in every bichromatic star the leaf
    /// end always has exactly one neighbour of the centre's colour — so
    /// direct recovery is always available from one side or the other.
    /// `adj` excludes self-loops.
    fn color_star(n: usize, adj: &[Vec<Index>]) -> Vec<usize> {
        let mut order: Vec<Index> = (0..n as Index).collect();
        order.sort_unstable_by_key(|&v| std::cmp::Reverse(adj[v as usize].len()));

        let mut color = vec![usize::MAX; n];
        let mut forbidden = vec![usize::MAX; n + 2];
        let mut n_colors = 0usize;

        for &v in &order {
            let stamp = v as usize;
            for &w in &adj[v as usize] {
                let cw = color[w as usize];
                if cw != usize::MAX {
                    // A proper colouring forbids a neighbour's colour.
                    forbidden[cw] = stamp;
                    // And a bichromatic P4 `v-w-x-y` would be created by
                    // giving `v` the colour of an `x` two hops away whose
                    // own neighbourhood already carries `w`'s colour.
                    for &x in &adj[w as usize] {
                        if x == v {
                            continue;
                        }
                        let cx = color[x as usize];
                        if cx == usize::MAX {
                            continue;
                        }
                        for &y in &adj[x as usize] {
                            if y != w && color[y as usize] == cw {
                                forbidden[cx] = stamp;
                                break;
                            }
                        }
                    }
                } else {
                    // `w` uncolored: `v-w-x` with `x` colored would leave
                    // a P4 realizable later, so keep `v` off `x`'s colour.
                    for &x in &adj[w as usize] {
                        if x == v {
                            continue;
                        }
                        let cx = color[x as usize];
                        if cx != usize::MAX {
                            forbidden[cx] = stamp;
                        }
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
            color[v as usize] = c;
        }
        color
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

        // Adjacency without self-loops, for the star colouring.
        let mut adj: Vec<Vec<Index>> = vec![Vec::new(); n];
        for &(i, j) in &pairs {
            if i != j {
                adj[i as usize].push(j);
                adj[j as usize].push(i);
            }
        }

        // A colouring is only usable if EVERY entry has an endpoint with
        // exactly one neighbour in the other endpoint's colour — otherwise
        // the probe component that entry is read from carries a *sum* of
        // several entries, and reading it as one is silently wrong.
        //
        // This is validated rather than assumed, and the validation is
        // unconditional. It was a `debug_assert` first, which is compiled
        // out in release: the star colouring below is NOT valid on the
        // Jacobian-derived pattern, and the resulting wrong Hessian showed
        // up only as `laptime` taking 404 iterations to an objective of
        // 65.368334 where the CPR colouring takes 38 to 65.371106. A
        // Hessian that is wrong but plausible is the failure this module is
        // most exposed to, so an invalid colouring falls back to CPR, which
        // is correct by construction.
        let validate = |color: &[usize]| -> bool {
            let count_in_color = |v: Index, c: usize| -> usize {
                adj[v as usize]
                    .iter()
                    .filter(|&&w| color[w as usize] == c)
                    .count()
            };
            pairs.iter().all(|&(i, j)| {
                i == j
                    || count_in_color(i, color[j as usize]) == 1
                    || count_in_color(j, color[i as usize]) == 1
            })
        };
        let mut color = match self.coloring {
            FdColoring::Cpr => Self::color_cpr(n, &cols_of_row, &rows_of_col),
            FdColoring::Star => Self::color_star(n, &adj),
        };
        let mut fell_back = false;
        if self.coloring == FdColoring::Star && !validate(&color) {
            color = Self::color_cpr(n, &cols_of_row, &rows_of_col);
            fell_back = true;
            debug_assert!(validate(&color), "CPR colouring must always be recoverable");
        }
        let n_colors = color.iter().copied().max().map(|c| c + 1).unwrap_or(0);
        let mut groups = vec![Vec::new(); n_colors];
        for (j, &c) in color.iter().enumerate() {
            groups[c].push(j as Index);
        }

        // ---- recovery map -------------------------------------------
        //
        // `H_ij` is read from the probe of some group `g` at component
        // `p`, divided by the step of column `q`. Validity requires that
        // `p` have exactly ONE neighbour in group `g` — otherwise the
        // probe component is a sum of several entries and reading it as
        // one is silently wrong.
        //
        // Under CPR that is guaranteed for `(colour(j), i, j)` by
        // construction. Under a star colouring it holds for at least one
        // of the two endpoints — the leaf of the bichromatic star — so
        // both are tried and the valid one taken.
        let count_in_color = |v: Index, c: usize| -> usize {
            adj[v as usize]
                .iter()
                .filter(|&&w| color[w as usize] == c)
                .count()
        };
        let mut recovery: Vec<(u32, u32, u32)> = Vec::with_capacity(pairs.len());
        for &(i, j) in &pairs {
            if i == j {
                // A proper colouring gives `i` no neighbour of its own
                // colour, so the diagonal is always directly readable.
                recovery.push((color[i as usize] as u32, i as u32, i as u32));
                continue;
            }
            let (ci, cj) = (color[i as usize], color[j as usize]);
            if count_in_color(i, cj) == 1 {
                recovery.push((cj as u32, i as u32, j as u32));
            } else {
                // Guaranteed reachable by the validation above.
                recovery.push((ci as u32, j as u32, i as u32));
            }
        }
        let mut by_group: Vec<Vec<u32>> = vec![Vec::new(); n_colors];
        for (k, &(g, _, _)) in recovery.iter().enumerate() {
            by_group[g as usize].push(k as u32);
        }

        self.stats = FdStats {
            n,
            nnz: pairs.len(),
            groups: groups.len(),
            rho_max,
            compression: groups.len() as f64 / n.max(1) as f64,
            coloring_fell_back: fell_back,
        };
        let irows: Vec<Index> = pairs.iter().map(|&(i, _)| i + 1).collect();
        let jcols: Vec<Index> = pairs.iter().map(|&(_, j)| j + 1).collect();
        self.space = Some(SymTMatrixSpace::new(n as Index, irows, jcols));
        self.groups = groups;
        self.recovery = recovery;
        self.by_group = by_group;
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

        // Reuse the cached Hessian when neither the iterate nor the
        // multipliers have moved. `∇²L = ∇²f + Σ yⱼ ∇²cⱼ` depends on both,
        // so testing `x` alone would hand back a stale Hessian every time
        // the duals moved on a short step — which is exactly what the
        // endgame of an interior-point solve does.
        if self.reuse_tol > 0.0 {
            let y_now: Vec<Number> = flat(&*curr_y_c)
                .into_iter()
                .chain(flat(&*curr_y_d))
                .collect();
            if let (Some(px), Some(py), Some(pw)) =
                (self.prev_x.as_ref(), self.prev_y.as_ref(), self.prev_w.as_ref())
            {
                let rel = |a: &[Number], b: &[Number]| -> Number {
                    let (mut d, mut m) = (0.0_f64, 1.0_f64);
                    for (u, v) in a.iter().zip(b.iter()) {
                        d = d.max((u - v).abs());
                        m = m.max(u.abs());
                    }
                    d / m
                };
                if px.len() == x.len()
                    && py.len() == y_now.len()
                    && rel(&x, px) <= self.reuse_tol
                    && rel(&y_now, py) <= self.reuse_tol
                {
                    self.reused += 1;
                    data.borrow_mut().w = Some(Rc::clone(pw) as Rc<dyn pounce_linalg::SymMatrix>);
                    return true;
                }
            }
            self.prev_x = Some(x.clone());
            self.prev_y = Some(y_now);
        }
        self.rebuilt += 1;

        let space = Rc::clone(self.space.as_ref().expect("structure built above"));
        let mut w = SymTMatrix::new(Rc::clone(&space));
        {
            let vals = w.values_mut();
            vals.iter_mut().for_each(|v| *v = 0.0);

            let mut probe = curr_x.make_new();
            let mut gl = curr_x.make_new();
            let mut xp = x.clone();
            for (gi, group) in self.groups.iter().enumerate() {
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

                for &k in &self.by_group[gi] {
                    let (_, read, col) = self.recovery[k as usize];
                    let hq = sign * steps[col as usize];
                    vals[k as usize] = (g1[read as usize] - base[read as usize]) / hq;
                }
            }
        }
        let w = Rc::new(w);
        if self.reuse_tol > 0.0 {
            self.prev_w = Some(Rc::clone(&w));
        }
        data.borrow_mut().w = Some(w as Rc<dyn pounce_linalg::SymMatrix>);
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

    /// Build lower-triangle pairs for a symmetric banded pattern, plus
    /// the adjacency and row/column incidence the colourings need.
    fn banded(n: usize, half_band: usize) -> (Vec<(Index, Index)>, Vec<Vec<Index>>, Vec<Vec<Index>>, Vec<Vec<Index>>) {
        let mut pairs = Vec::new();
        for i in 0..n {
            for j in i.saturating_sub(half_band)..=i {
                pairs.push((i as Index, j as Index));
            }
        }
        let mut rows_of_col: Vec<Vec<Index>> = vec![Vec::new(); n];
        let mut cols_of_row: Vec<Vec<Index>> = vec![Vec::new(); n];
        let mut adj: Vec<Vec<Index>> = vec![Vec::new(); n];
        for &(i, j) in &pairs {
            rows_of_col[j as usize].push(i);
            cols_of_row[i as usize].push(j);
            if i != j {
                rows_of_col[i as usize].push(j);
                cols_of_row[j as usize].push(i);
                adj[i as usize].push(j);
                adj[j as usize].push(i);
            }
        }
        (pairs, rows_of_col, cols_of_row, adj)
    }

    /// The property both colourings must have, and the only one that
    /// makes direct recovery sound: for every entry there is a probe
    /// component that carries **exactly one** entry, so reading it as a
    /// single value is not reading a sum. This is checked by actually
    /// recovering a known matrix from simulated probes rather than by
    /// inspecting the colouring — a colouring can look plausible and
    /// still make the recovery read two entries as one, silently.
    fn recovers_exactly(coloring: FdColoring, n: usize, half_band: usize) -> usize {
        let (pairs, rows_of_col, cols_of_row, adj) = banded(n, half_band);
        let color = match coloring {
            FdColoring::Cpr => FdHessianUpdater::color_cpr(n, &cols_of_row, &rows_of_col),
            FdColoring::Star => FdHessianUpdater::color_star(n, &adj),
        };
        let n_colors = color.iter().copied().max().unwrap() + 1;

        // A known symmetric matrix on that pattern.
        let val = |i: Index, j: Index| -> Number {
            1.0 + (i as Number) * 0.5 - (j as Number) * 0.25 + ((i + j) as Number).sin()
        };
        let mut dense = vec![vec![0.0 as Number; n]; n];
        for &(i, j) in &pairs {
            let v = val(i.max(j), i.min(j));
            dense[i as usize][j as usize] = v;
            dense[j as usize][i as usize] = v;
        }

        // Exact probes: b_g = H · Σ_{m∈g} e_m  (unit steps).
        let mut probes = vec![vec![0.0 as Number; n]; n_colors];
        for (m, &c) in color.iter().enumerate() {
            for i in 0..n {
                probes[c][i] += dense[i][m];
            }
        }

        // Recovery, mirroring `build_structure` exactly.
        let count_in_color = |v: Index, c: usize| -> usize {
            adj[v as usize].iter().filter(|&&w| color[w as usize] == c).count()
        };
        for &(i, j) in &pairs {
            let (g, read) = if i == j {
                (color[i as usize], i)
            } else {
                let (ci, cj) = (color[i as usize], color[j as usize]);
                if count_in_color(i, cj) == 1 {
                    (cj, i)
                } else {
                    assert_eq!(
                        count_in_color(j, ci),
                        1,
                        "{coloring:?}: neither endpoint of ({i},{j}) is directly recoverable"
                    );
                    (ci, j)
                }
            };
            let got = probes[g][read as usize];
            let want = dense[i as usize][j as usize];
            assert!(
                (got - want).abs() < 1e-12,
                "{coloring:?}: entry ({i},{j}) recovered {got}, want {want}                  — the probe component carried a sum, not one entry"
            );
        }
        n_colors
    }

    #[test]
    fn cpr_recovers_a_banded_matrix_exactly() {
        for hb in 1..=4 {
            recovers_exactly(FdColoring::Cpr, 40, hb);
        }
    }

    #[test]
    fn star_recovers_a_banded_matrix_exactly() {
        for hb in 1..=4 {
            recovers_exactly(FdColoring::Star, 40, hb);
        }
    }

    /// Star colouring should need no more groups than CPR on a banded
    /// pattern — that is the entire reason to pay for the extra
    /// bookkeeping. It is asserted as `<=` rather than as a ratio because
    /// how much it saves is a property of the pattern, not a guarantee:
    /// where the pattern contains a dense `k × k` clique, the clique
    /// number lower-bounds *any* colouring and star wins nothing.
    #[test]
    fn star_never_needs_more_groups_than_cpr() {
        for hb in [1usize, 2, 4, 8] {
            let star = recovers_exactly(FdColoring::Star, 60, hb);
            let cpr = recovers_exactly(FdColoring::Cpr, 60, hb);
            assert!(star <= cpr, "half-band {hb}: star {star} > cpr {cpr}");
        }
    }

    /// **The test that the banded ones missed.** The Jacobian-derived
    /// pattern is not banded — it is a union of OVERLAPPING CLIQUES, one
    /// per constraint row, since `supp(∇g_j) ⊗ supp(∇g_j)` is dense. The
    /// greedy star colouring is not valid on that shape, and the banded
    /// fixtures never exposed it: on `laptime` it silently produced a
    /// wrong Hessian that cost 404 iterations and a wrong objective.
    ///
    /// What is asserted here is the *validation*, not the colouring:
    /// whatever colouring is produced, every entry must be directly
    /// recoverable, or the caller must fall back. This is the invariant
    /// the recovery depends on, and it is checked on the shape that
    /// actually breaks it.
    #[test]
    fn overlapping_cliques_are_validated_not_assumed() {
        // Five 6-wide "constraint rows", each overlapping the next by 3 —
        // the shape a collocation Jacobian produces.
        let n = 18usize;
        let mut set = std::collections::BTreeSet::new();
        for start in (0..n - 5).step_by(3) {
            let cols: Vec<Index> = (start..start + 6).map(|v| v as Index).collect();
            for (a, &ca) in cols.iter().enumerate() {
                for &cb in cols.iter().take(a + 1) {
                    set.insert((ca, cb));
                }
            }
        }
        for i in 0..n as Index {
            set.insert((i, i));
        }
        let pairs: Vec<(Index, Index)> = set.into_iter().collect();

        let mut rows_of_col: Vec<Vec<Index>> = vec![Vec::new(); n];
        let mut cols_of_row: Vec<Vec<Index>> = vec![Vec::new(); n];
        let mut adj: Vec<Vec<Index>> = vec![Vec::new(); n];
        for &(i, j) in &pairs {
            rows_of_col[j as usize].push(i);
            cols_of_row[i as usize].push(j);
            if i != j {
                rows_of_col[i as usize].push(j);
                cols_of_row[j as usize].push(i);
                adj[i as usize].push(j);
                adj[j as usize].push(i);
            }
        }

        let recoverable = |color: &[usize]| -> bool {
            let cnt = |v: Index, c: usize| {
                adj[v as usize].iter().filter(|&&w| color[w as usize] == c).count()
            };
            pairs.iter().all(|&(i, j)| {
                i == j
                    || cnt(i, color[j as usize]) == 1
                    || cnt(j, color[i as usize]) == 1
            })
        };

        // CPR is correct by construction on any pattern.
        let cpr = FdHessianUpdater::color_cpr(n, &cols_of_row, &rows_of_col);
        assert!(recoverable(&cpr), "CPR must always be directly recoverable");

        // Now the decisive part: actually RECOVER a known matrix through
        // each colouring on this shape. The predicate above is necessary
        // but was not shown to be sufficient — on `laptime` the star
        // colouring passed it (`coloring_fell_back: false`) and still
        // produced a Hessian wrong enough to cost 404 iterations. This
        // reproduces that in-process instead of only in a solve.
        let val = |i: Index, j: Index| -> Number {
            1.0 + (i as Number) * 0.5 - (j as Number) * 0.25 + ((i * 7 + j) as Number).sin()
        };
        let mut dense = vec![vec![0.0 as Number; n]; n];
        for &(i, j) in &pairs {
            let v = val(i.max(j), i.min(j));
            dense[i as usize][j as usize] = v;
            dense[j as usize][i as usize] = v;
        }
        let check = |color: &[usize], name: &str| -> Result<(), String> {
            let n_colors = color.iter().copied().max().unwrap() + 1;
            let mut probes = vec![vec![0.0 as Number; n]; n_colors];
            for (m, &c) in color.iter().enumerate() {
                for i in 0..n {
                    probes[c][i] += dense[i][m];
                }
            }
            let cnt = |v: Index, c: usize| {
                adj[v as usize].iter().filter(|&&w| color[w as usize] == c).count()
            };
            for &(i, j) in &pairs {
                let (g, read) = if i == j {
                    (color[i as usize], i)
                } else if cnt(i, color[j as usize]) == 1 {
                    (color[j as usize], i)
                } else {
                    (color[i as usize], j)
                };
                let (got, want) = (probes[g][read as usize], dense[i as usize][j as usize]);
                if (got - want).abs() > 1e-12 {
                    return Err(format!("{name}: ({i},{j}) recovered {got}, want {want}"));
                }
            }
            Ok(())
        };
        check(&cpr, "cpr").expect("CPR recovery must be exact on overlapping cliques");

        // The star colouring is NOT asserted correct here: it is not, and
        // that is the recorded finding. What is asserted is that whenever
        // recovery would be wrong, the predicate `build_structure` gates on
        // rejects it — i.e. the predicate is not weaker than the truth.
        let star = FdHessianUpdater::color_star(n, &adj);
        if check(&star, "star").is_err() {
            assert!(
                !recoverable(&star),
                "star recovery is wrong on this pattern yet the validation \
                 predicate accepts it — the predicate is unsound, and \
                 `build_structure` would ship a silently wrong Hessian"
            );
        }
    }

    /// A dense row makes every column adjacent to it, so the pattern
    /// contains a clique and no colouring can do better than its size.
    /// This is the case where star colouring is *not* a win, and the
    /// Jacobian-derived Hessian pattern is exactly this shape.
    #[test]
    fn a_clique_forces_its_size_in_groups_under_either_coloring() {
        let n = 10usize;
        let mut pairs = Vec::new();
        for i in 0..n as Index {
            for j in 0..=i {
                pairs.push((i, j));
            }
        }
        let mut rows_of_col: Vec<Vec<Index>> = vec![Vec::new(); n];
        let mut cols_of_row: Vec<Vec<Index>> = vec![Vec::new(); n];
        let mut adj: Vec<Vec<Index>> = vec![Vec::new(); n];
        for &(i, j) in &pairs {
            rows_of_col[j as usize].push(i);
            cols_of_row[i as usize].push(j);
            if i != j {
                rows_of_col[i as usize].push(j);
                cols_of_row[j as usize].push(i);
                adj[i as usize].push(j);
                adj[j as usize].push(i);
            }
        }
        let star = FdHessianUpdater::color_star(n, &adj);
        let cpr = FdHessianUpdater::color_cpr(n, &cols_of_row, &rows_of_col);
        assert_eq!(star.iter().copied().max().unwrap() + 1, n);
        assert_eq!(cpr.iter().copied().max().unwrap() + 1, n);
    }
}

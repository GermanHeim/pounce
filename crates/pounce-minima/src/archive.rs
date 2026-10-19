//! Archive of accepted minima with per-dimension-scaled dedup, mirroring
//! `MinimaArchive` in `python/pounce/_minima.py`.
//!
//! Two points are "the same" when their Euclidean distance in the
//! per-dimension scaled space `‖(a−b)/L‖` is within `dedup`, where `L` is
//! the box width per variable (1.0 for unbounded dims). This makes `dedup`
//! scale-free and keeps it consistent with the anisotropic repulsion widths.

use pounce_common::types::Number;

/// Scaled Euclidean distance `‖(a−b)/L‖`.
pub fn scaled_distance(a: &[Number], b: &[Number], l: &[Number]) -> Number {
    let mut acc = 0.0;
    for i in 0..a.len() {
        let d = (a[i] - b[i]) / l[i];
        acc += d * d;
    }
    acc.sqrt()
}

/// The base-problem multipliers recovered at an accepted minimum.
#[derive(Clone, Debug, Default)]
pub struct Duals {
    /// Constraint duals, length `m`, in the user's `∇f + λ·∇g + z = 0`
    /// convention.
    pub lambda: Vec<Number>,
    /// Bound multipliers, length `n` (empty when unavailable), `≥ 0` at an
    /// active bound, in the frame of the problem the engine solved.
    pub z_l: Vec<Number>,
    pub z_u: Vec<Number>,
}

/// Accepted minima plus the dedup test.
pub struct Archive {
    dedup: Number,
    /// Per-dimension scale `L` for the dedup metric.
    l: Vec<Number>,
    pub xs: Vec<Vec<Number>>,
    pub fs: Vec<Number>,
    /// Per-minimum base-problem constraint duals (length `m`), recovered by a
    /// clean re-solve at the accepted point (issue #196, related). Parallel to
    /// `xs` / `fs`.
    pub ls: Vec<Vec<Number>>,
    /// Per-minimum lower / upper bound multipliers (length `n`, or empty when
    /// unavailable), recovered alongside `ls`. Parallel to `xs` / `fs`.
    pub zls: Vec<Vec<Number>>,
    pub zus: Vec<Vec<Number>>,
}

impl Archive {
    pub fn new(dedup: Number, l: Vec<Number>) -> Self {
        Self {
            dedup,
            l,
            xs: Vec::new(),
            fs: Vec::new(),
            ls: Vec::new(),
            zls: Vec::new(),
            zus: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.xs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.xs.is_empty()
    }

    /// Is `x` within `dedup` of any already-accepted minimum?
    pub fn is_known(&self, x: &[Number]) -> bool {
        self.xs
            .iter()
            .any(|m| scaled_distance(x, m, &self.l) <= self.dedup)
    }

    /// Is `x` within `radius` of any accepted minimum (MLSL clustering)?
    pub fn near_any(&self, x: &[Number], radius: Number) -> bool {
        self.xs
            .iter()
            .any(|m| scaled_distance(x, m, &self.l) <= radius)
    }

    pub fn add(&mut self, x: Vec<Number>, duals: Duals, f: Number) {
        self.xs.push(x);
        self.ls.push(duals.lambda);
        self.zls.push(duals.z_l);
        self.zus.push(duals.z_u);
        self.fs.push(f);
    }

    /// Indices of the accepted minima ordered by ascending objective.
    pub fn order_by_objective(&self) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.fs.len()).collect();
        idx.sort_by(|&a, &b| {
            self.fs[a]
                .partial_cmp(&self.fs[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx
    }
}

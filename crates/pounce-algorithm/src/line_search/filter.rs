//! Two-dimensional filter — port of `Algorithm/IpFilter.{hpp,cpp}`.
//!
//! Stores `(theta, phi)` pairs and answers "is this point dominated
//! by any pair already in the filter?". Bit-exact with upstream's
//! dominance check (`<= ` not `<`).

use pounce_common::types::Number;

/// One filter entry — port of `Ipopt::FilterEntry`.
#[derive(Debug, Clone, Copy)]
pub struct FilterEntry {
    pub theta: Number,
    pub phi: Number,
    /// Iteration index this entry was added at — used for clearing
    /// on restoration return; mirrors upstream's `iter_` field.
    pub iter: i32,
}

impl FilterEntry {
    pub fn new(theta: Number, phi: Number, iter: i32) -> Self {
        Self { theta, phi, iter }
    }
}

/// Two-dimensional filter. Backed by a `Vec` for predictable
/// iteration order — matches upstream `std::list<FilterEntry>`
/// insertion behavior.
#[derive(Debug, Default, Clone)]
pub struct Filter {
    entries: Vec<FilterEntry>,
}

impl Filter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn entries(&self) -> &[FilterEntry] {
        &self.entries
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// True if `(theta, phi)` is dominated by *any* existing entry,
    /// using upstream's dominance rule. A point is **not acceptable**
    /// iff some entry beats it strictly in BOTH coordinates:
    ///
    /// ```text
    ///   theta_new > e.theta  AND  phi_new > e.phi
    /// ```
    ///
    /// This is the negation of `IpFilter.cpp:FilterEntry::Acceptable`,
    /// which returns true when *any* coord satisfies `vals[i] <= vals_[i]`.
    /// Strict `>` matters at exact ties: upstream considers `vals == entry`
    /// acceptable (1.0 <= 1.0 → true at first coord). Using `>=` here
    /// would over-reject and trigger spurious adaptive-μ free→fixed
    /// transitions when the iterate is approximately constant.
    pub fn dominated_by_any(&self, theta: Number, phi: Number) -> bool {
        self.entries
            .iter()
            .any(|e| theta > e.theta && phi > e.phi)
    }

    /// Add the entry and prune any existing entries that the new one
    /// strictly dominates. Mirrors `IpFilter::AddEntry`.
    pub fn add(&mut self, theta: Number, phi: Number, iter: i32) {
        self.entries
            .retain(|e| !(theta <= e.theta && phi <= e.phi));
        self.entries.push(FilterEntry::new(theta, phi, iter));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_filter_dominates_nothing() {
        let f = Filter::new();
        assert!(!f.dominated_by_any(1.0, 1.0));
    }

    #[test]
    fn add_entry_then_dominated_check() {
        let mut f = Filter::new();
        f.add(1.0, 5.0, 0);
        // Strictly worse in both → dominated.
        assert!(f.dominated_by_any(2.0, 6.0));
        // Exact tie → acceptable per upstream (`vals[i] <= vals_[i]`
        // satisfied at i=0 → entry returns Acceptable=true).
        assert!(!f.dominated_by_any(1.0, 5.0));
        // Smaller in either dimension → not dominated.
        assert!(!f.dominated_by_any(0.5, 5.0));
        assert!(!f.dominated_by_any(1.0, 4.9));
    }

    #[test]
    fn add_entry_prunes_dominated_existing_entries() {
        let mut f = Filter::new();
        f.add(2.0, 6.0, 0);
        f.add(1.0, 5.0, 1);
        // The first entry was dominated by the second and should
        // have been pruned.
        assert_eq!(f.entries().len(), 1);
        assert_eq!(f.entries()[0].iter, 1);
    }

    #[test]
    fn pareto_set_is_preserved() {
        let mut f = Filter::new();
        f.add(1.0, 5.0, 0);
        f.add(2.0, 3.0, 1); // smaller phi, larger theta → not dominated
        f.add(3.0, 1.0, 2);
        assert_eq!(f.entries().len(), 3);
    }
}

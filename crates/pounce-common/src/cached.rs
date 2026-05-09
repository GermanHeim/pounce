//! Tag-keyed result cache.
//!
//! Mirrors `Common/IpCachedResults.hpp`. Ipopt's `CachedResults<T>`
//! stores an LRU list of (dependency tags, scalar dependencies, value)
//! tuples; on lookup it returns the value whose dependency vector
//! matches the current tags of the depended-on `TaggedObject`s.
//!
//! Differences from upstream:
//! - We do not implement the `Observer/Subject` invalidation push
//!   path. Upstream uses it to mark stale entries early; correctness
//!   only requires the pull-side check at lookup time, which we keep.
//! - Negative `max_cache_size` (= unbounded) is supported with
//!   `Cache::unbounded()` — same semantics as Ipopt's negative size.

use crate::tagged::{Tag, TaggedObject};
use crate::types::Number;
use std::collections::VecDeque;

/// One entry in the cache: stored value plus the dependency tags and
/// scalar dependencies it was computed against.
#[derive(Debug, Clone)]
struct Entry<T> {
    value: T,
    dep_tags: Vec<Tag>,
    scalar_deps: Vec<Number>,
}

impl<T> Entry<T> {
    fn matches(&self, dep_tags: &[Tag], scalar_deps: &[Number]) -> bool {
        if self.dep_tags.len() != dep_tags.len()
            || self.scalar_deps.len() != scalar_deps.len()
        {
            return false;
        }
        for (a, b) in self.dep_tags.iter().zip(dep_tags.iter()) {
            if a != b {
                return false;
            }
        }
        for (a, b) in self.scalar_deps.iter().zip(scalar_deps.iter()) {
            // Matches Ipopt: bit-equality via float `!=` comparison.
            if a != b {
                return false;
            }
        }
        true
    }
}

/// LRU cache keyed on dependency tags + scalar dependencies. `T` is
/// the cached value type (`Number`, a `Vec<Number>`, an `Rc<Vector>`,
/// ...). Equivalent to `Ipopt::CachedResults<T>`.
#[derive(Debug)]
pub struct Cache<T> {
    /// `None` means unbounded.
    max_size: Option<usize>,
    /// Front = most-recently inserted, matching Ipopt's `push_front`.
    entries: VecDeque<Entry<T>>,
}

impl<T: Clone> Cache<T> {
    /// Bounded cache holding up to `max_size` entries.
    pub fn new(max_size: usize) -> Self {
        Self { max_size: Some(max_size), entries: VecDeque::new() }
    }

    /// Equivalent to Ipopt's negative `max_cache_size` (no eviction).
    pub fn unbounded() -> Self {
        Self { max_size: None, entries: VecDeque::new() }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Generic add — equivalent to `AddCachedResult(result, dependents, scalar_dependents)`.
    pub fn add(
        &mut self,
        value: T,
        dependents: &[&dyn TaggedObject],
        scalar_dependents: &[Number],
    ) {
        let dep_tags: Vec<Tag> = dependents.iter().map(|d| d.get_tag()).collect();
        self.add_with_tags(value, dep_tags, scalar_dependents.to_vec());
    }

    fn add_with_tags(&mut self, value: T, dep_tags: Vec<Tag>, scalar_deps: Vec<Number>) {
        self.entries.push_front(Entry { value, dep_tags, scalar_deps });
        if let Some(max) = self.max_size {
            while self.entries.len() > max {
                self.entries.pop_back();
            }
        }
    }

    /// Generic lookup — equivalent to `GetCachedResult(...)`. Returns
    /// `Some(value)` if a stored entry's dependency tags exactly match
    /// the current tags of `dependents` and the scalar deps match.
    pub fn get(
        &self,
        dependents: &[&dyn TaggedObject],
        scalar_dependents: &[Number],
    ) -> Option<T> {
        let dep_tags: Vec<Tag> = dependents.iter().map(|d| d.get_tag()).collect();
        for e in &self.entries {
            if e.matches(&dep_tags, scalar_dependents) {
                return Some(e.value.clone());
            }
        }
        None
    }

    pub fn add_1dep(&mut self, value: T, dep: &dyn TaggedObject) {
        self.add(value, &[dep], &[]);
    }

    pub fn get_1dep(&self, dep: &dyn TaggedObject) -> Option<T> {
        self.get(&[dep], &[])
    }

    pub fn add_2dep(&mut self, value: T, d1: &dyn TaggedObject, d2: &dyn TaggedObject) {
        self.add(value, &[d1, d2], &[]);
    }

    pub fn get_2dep(&self, d1: &dyn TaggedObject, d2: &dyn TaggedObject) -> Option<T> {
        self.get(&[d1, d2], &[])
    }

    pub fn add_3dep(
        &mut self,
        value: T,
        d1: &dyn TaggedObject,
        d2: &dyn TaggedObject,
        d3: &dyn TaggedObject,
    ) {
        self.add(value, &[d1, d2, d3], &[]);
    }

    pub fn get_3dep(
        &self,
        d1: &dyn TaggedObject,
        d2: &dyn TaggedObject,
        d3: &dyn TaggedObject,
    ) -> Option<T> {
        self.get(&[d1, d2, d3], &[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tagged::TaggedCell;

    #[test]
    fn hit_then_miss_after_bump() {
        let dep = TaggedCell::new();
        let mut cache: Cache<f64> = Cache::new(4);
        cache.add_1dep(2.5, &dep);
        assert_eq!(cache.get_1dep(&dep), Some(2.5));
        dep.bump();
        assert_eq!(cache.get_1dep(&dep), None);
    }

    #[test]
    fn lru_evicts_oldest() {
        let d1 = TaggedCell::new();
        let d2 = TaggedCell::new();
        let d3 = TaggedCell::new();
        let mut cache: Cache<i32> = Cache::new(2);
        cache.add_1dep(1, &d1);
        cache.add_1dep(2, &d2);
        cache.add_1dep(3, &d3);
        assert_eq!(cache.get_1dep(&d1), None); // evicted
        assert_eq!(cache.get_1dep(&d2), Some(2));
        assert_eq!(cache.get_1dep(&d3), Some(3));
    }

    #[test]
    fn unbounded_keeps_all() {
        let deps: Vec<TaggedCell> = (0..32).map(|_| TaggedCell::new()).collect();
        let mut cache: Cache<i32> = Cache::unbounded();
        for (i, d) in deps.iter().enumerate() {
            cache.add_1dep(i as i32, d);
        }
        for (i, d) in deps.iter().enumerate() {
            assert_eq!(cache.get_1dep(d), Some(i as i32));
        }
    }

    #[test]
    fn scalar_dep_distinguishes_entries() {
        let dep = TaggedCell::new();
        let mut cache: Cache<i32> = Cache::new(8);
        cache.add(10, &[&dep], &[1.0]);
        cache.add(20, &[&dep], &[2.0]);
        assert_eq!(cache.get(&[&dep], &[1.0]), Some(10));
        assert_eq!(cache.get(&[&dep], &[2.0]), Some(20));
        assert_eq!(cache.get(&[&dep], &[3.0]), None);
    }
}

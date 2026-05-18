//! Tagged-object change tracking.
//!
//! Mirrors `Common/IpTaggedObject.{hpp,cpp}`. Each `TaggedObject`
//! exposes a `Tag` which is bumped from a thread-local counter every
//! time `object_changed()` is called. Cached results compare stored
//! tags against current tags to decide whether to recompute.
//!
//! Ipopt's implementation is `unsigned int` per-thread starting at 1.
//! We keep the same semantics with a `u64` counter — the underlying
//! type is wider (u32 wraparound is reachable in long restoration runs
//! per Ipopt's own DBG_ASSERT) but the equality semantics are
//! identical: two `Tag` values compare equal iff they came from the
//! same `object_changed()` call.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

/// Per-`TaggedObject` change tag. Equivalent to `TaggedObject::Tag`
/// (`unsigned int` upstream).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Tag(pub u64);

impl Tag {
    pub const NONE: Tag = Tag(0);
}

thread_local! {
    /// Mirrors the file-static `IPOPT_THREAD_LOCAL TaggedObject::Tag unique_tag = 1`
    /// in `IpTaggedObject.cpp`.
    static UNIQUE_TAG: Cell<u64> = const { Cell::new(1) };
}

/// Allocate a fresh, never-before-used tag from this thread's counter.
pub fn next_tag() -> Tag {
    UNIQUE_TAG.with(|c| {
        let t = c.get();
        c.set(t.wrapping_add(1));
        Tag(t)
    })
}

/// Cross-thread fallback used by `TaggedCell` when a TaggedObject is
/// shared via `Arc` and may be mutated from any thread.
static GLOBAL_UNIQUE_TAG: AtomicU64 = AtomicU64::new(1);

/// Allocate a fresh tag from the cross-thread global counter.
pub fn next_tag_global() -> Tag {
    Tag(GLOBAL_UNIQUE_TAG.fetch_add(1, Ordering::Relaxed))
}

/// Embeddable tag holder. A struct that wants Ipopt's tagged-object
/// behavior holds a `TaggedCell` and calls `.bump()` from inside any
/// state-mutating method, the same way Ipopt classes call
/// `ObjectChanged()` from inside their setters.
#[derive(Debug)]
pub struct TaggedCell {
    tag: Cell<Tag>,
}

impl TaggedCell {
    /// Construct with an initial tag (matches Ipopt's constructor which
    /// calls `ObjectChanged()` once).
    pub fn new() -> Self {
        Self {
            tag: Cell::new(next_tag()),
        }
    }

    /// Current tag — equivalent to `TaggedObject::GetTag`.
    pub fn tag(&self) -> Tag {
        self.tag.get()
    }

    /// Equivalent to `TaggedObject::HasChanged(comparison_tag)`.
    pub fn has_changed(&self, comparison_tag: Tag) -> bool {
        self.tag.get() != comparison_tag
    }

    /// Equivalent to `TaggedObject::ObjectChanged()`. Bumps the
    /// thread-local counter and stores the new tag.
    pub fn bump(&self) {
        self.tag.set(next_tag());
    }
}

impl Default for TaggedCell {
    fn default() -> Self {
        Self::new()
    }
}

/// Object-safe trait so [`crate::cached::CachedResults`] can take
/// dependencies as `&dyn TaggedObject`.
pub trait TaggedObject {
    fn get_tag(&self) -> Tag;
}

impl TaggedObject for TaggedCell {
    fn get_tag(&self) -> Tag {
        self.tag()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_tags_are_distinct() {
        let a = next_tag();
        let b = next_tag();
        assert_ne!(a, b);
    }

    #[test]
    fn bump_changes_tag() {
        let c = TaggedCell::new();
        let t0 = c.tag();
        assert!(!c.has_changed(t0));
        c.bump();
        assert!(c.has_changed(t0));
        let t1 = c.tag();
        assert_ne!(t0, t1);
    }

    #[test]
    fn none_never_matches_a_real_tag() {
        let c = TaggedCell::new();
        assert!(c.has_changed(Tag::NONE));
    }
}

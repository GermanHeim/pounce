//! Re-export shim. The canonical [`TimingStatistics`] type lives in
//! [`pounce_common::timing`] so that crates lower in the dep graph
//! (`pounce-nlp`, `pounce-linsol`) can populate it without depending
//! on `pounce-algorithm`. Mirrors `Algorithm/IpTimingStatistics.{hpp,cpp}`.

pub use pounce_common::timing::TimingStatistics;

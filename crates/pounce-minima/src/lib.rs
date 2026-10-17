//! The building blocks of POUNCE's multistart / find-minima global search:
//! the deduplicating archive of found minima, the start sampler (seedable
//! ChaCha8 + Owen-scrambled Sobol'), the penalty / tunnel TNLP wrappers the
//! repulsion strategies solve, and the search options.
//!
//! Extracted from `pounce-cli` (gh#966) so the search can be shared by every
//! frontend instead of being re-implemented in each. The search driver still
//! lives in `pounce_cli::minima` and moves here next; until then this crate is
//! the parts that do not depend on the solver application.

pub mod archive;
pub mod options;
pub mod penalty_tnlp;
pub mod sampling;

pub use options::{Method, MinimaOptions};

//! NLP traits used by the algorithm — port of `IpNLP.hpp` and
//! `IpIpoptNLP.hpp`.
//!
//! The traits themselves live in `pounce_nlp::ipopt_nlp` (alongside
//! `OrigIpoptNlp`, which composes `TNLPAdapter` + `NLPScalingObject`).
//! They are re-exported from this module so existing imports of the
//! form `crate::ipopt_nlp::IpoptNlp` / `pounce_algorithm::ipopt_nlp::Nlp`
//! continue to compile. The relocation avoids a `pounce-algorithm` ←→
//! `pounce-nlp` cycle that would otherwise arise once the concrete
//! `OrigIpoptNlp` (which depends on `TNLPAdapter`) wants to implement
//! the algorithm-side trait.

pub use pounce_nlp::ipopt_nlp::{IpoptNlp, Nlp};

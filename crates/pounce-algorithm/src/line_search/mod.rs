//! Line-search subsystem — port of
//! `Algorithm/IpLineSearch.hpp`, `IpBacktrackingLineSearch.{hpp,cpp}`,
//! `IpFilter.{hpp,cpp}`, `IpFilterLSAcceptor.{hpp,cpp}`,
//! `IpBacktrackingLSAcceptor.hpp`. Penalty / CG-penalty acceptors
//! (Phase 10) live here too once they land.

pub mod backtracking;
pub mod filter;
pub mod filter_acceptor;
pub mod ls_acceptor;
pub mod penalty_acceptor;

pub use filter::{Filter, FilterEntry};

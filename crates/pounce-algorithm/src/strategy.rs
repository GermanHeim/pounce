//! Algorithm-strategy base trait — port of `IpAlgStrategy.hpp`.
//!
//! Every strategy object (line search, mu update, conv check, etc.)
//! implements this so `IpoptAlgorithm` can call `initialize` on it
//! after option-loading and before iteration.

use pounce_common::exception::SolverException;
use pounce_common::journalist::Journalist;
use pounce_common::options_list::OptionsList;
use std::rc::Rc;

/// Ports `Ipopt::AlgorithmStrategyObject`. The `_jnlst` etc. arguments
/// match the upstream signature; concrete strategies pull what they
/// need from the four shared handles.
pub trait AlgorithmStrategy {
    fn initialize(
        &mut self,
        jnlst: &Rc<Journalist>,
        options: &OptionsList,
        prefix: &str,
    ) -> Result<(), SolverException>;
}

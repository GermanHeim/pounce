//! MA57 option registration — port of
//! `Ma57TSolverInterface::RegisterOptions` in
//! `IpMa57TSolverInterface.cpp`.

use pounce_common::exception::SolverException;
use pounce_common::reg_options::RegisteredOptions;

/// Register MA57's options on `reg`. Idempotent within a fresh
/// registry; re-registering raises `OPTION_ALREADY_REGISTERED`.
///
/// Defaults and bounds come from `IpMa57TSolverInterface.cpp:191-263`.
/// `ma57_pivot_order` defaults to 5 (METIS) on the non-Matlab build —
/// upstream's `FUNNY_MA57_FINT` branch is irrelevant here.
pub fn register_options_ma57(reg: &RegisteredOptions) -> Result<(), SolverException> {
    reg.set_registering_category("MA57 Linear Solver");

    reg.add_lower_bounded_integer_option(
        "ma57_print_level",
        "Debug printing level for the linear solver MA57",
        0,
        0,
        "0: no printing; 1: Error messages only; 2: Error and warning messages; \
         3: Error and warning messages and terse monitoring; >=4: All information.",
    )?;

    reg.add_bounded_number_option(
        "ma57_pivtol",
        "Pivot tolerance for the linear solver MA57.",
        0.0,
        true,
        1.0,
        true,
        1e-8,
        "A smaller number pivots for sparsity, a larger number pivots for stability.",
    )?;

    reg.add_bounded_number_option(
        "ma57_pivtolmax",
        "Maximum pivot tolerance for the linear solver MA57.",
        0.0,
        true,
        1.0,
        true,
        1e-4,
        "Ipopt may increase pivtol as high as ma57_pivtolmax to get a more accurate \
         solution to the linear system.",
    )?;

    reg.add_lower_bounded_number_option(
        "ma57_pre_alloc",
        "Safety factor for work space memory allocation for the linear solver MA57.",
        1.0,
        false,
        1.05,
        "If 1 is chosen, the suggested amount of work space is used. However, \
         choosing a larger number might avoid reallocation if the suggest values \
         do not suffice.",
    )?;

    reg.add_bounded_integer_option(
        "ma57_pivot_order",
        "Controls pivot order in MA57",
        0,
        5,
        5,
        "This is ICNTL(6) in MA57.",
    )?;

    reg.add_bool_option(
        "ma57_automatic_scaling",
        "Controls whether to enable automatic scaling in MA57",
        false,
        "For higher reliability of the MA57 solver, you may want to set this \
         option to yes. This is ICNTL(15) in MA57.",
    )?;

    reg.add_lower_bounded_integer_option(
        "ma57_block_size",
        "Controls block size used by Level 3 BLAS in MA57BD",
        1,
        16,
        "This is ICNTL(11) in MA57.",
    )?;

    reg.add_lower_bounded_integer_option(
        "ma57_node_amalgamation",
        "Node amalgamation parameter",
        1,
        16,
        "This is ICNTL(12) in MA57.",
    )?;

    reg.add_bounded_integer_option(
        "ma57_small_pivot_flag",
        "Handling of small pivots",
        0,
        1,
        0,
        "If set to 1, then when small entries defined by CNTL(2) are detected they are \
         removed and the corresponding pivots placed at the end of the factorization. \
         This can be particularly efficient if the matrix is highly rank deficient. \
         This is ICNTL(16) in MA57.",
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_all_ma57_options() {
        let reg = RegisteredOptions::default();
        register_options_ma57(&reg).unwrap();
        for name in [
            "ma57_print_level",
            "ma57_pivtol",
            "ma57_pivtolmax",
            "ma57_pre_alloc",
            "ma57_pivot_order",
            "ma57_automatic_scaling",
            "ma57_block_size",
            "ma57_node_amalgamation",
            "ma57_small_pivot_flag",
        ] {
            assert!(
                reg.get_option(name).is_some(),
                "missing registered option {name}"
            );
        }
    }

    #[test]
    fn double_registration_is_error() {
        let reg = RegisteredOptions::default();
        register_options_ma57(&reg).unwrap();
        assert!(register_options_ma57(&reg).is_err());
    }
}

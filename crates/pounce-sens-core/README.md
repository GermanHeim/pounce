# pounce-sens-core

The engine-agnostic core of POUNCE's sensitivity layer: the `SensBacksolver`
contract and everything that can be built on it without knowing which solver
produced the KKT system.

## What is here

Everything in this crate is written against one trait — `SensBacksolver`,
whose entire required surface is `dim()` and `solve(rhs, lhs)`. Any engine
that can back-solve against its converged factor gets the rest for free:

| module | what it provides |
|---|---|
| `backsolver` | the `SensBacksolver` contract, `BoundRow`, and the synthetic `DenseLuBacksolver` that validates the math without an IPM |
| `boundcheck` | fix-relax refinement, path following through active-set breakpoints, the directional derivative at a kink |
| `sens_app` | the sIPOPT `SensApplication` driver, reduced-Hessian entry point, option registration |
| `p_calculator`, `schur_data`, `schur_driver`, `step_calc`, `reduced_hessian` | the `P = K⁻¹A` / Schur-complement stack |

Two consumers exist in tree. [`pounce-sensitivity`](../pounce-sensitivity)
implements the trait over the NLP filter-IPM's KKT factor;
[`pounce-convex`](../pounce-convex) implements it over the convex active-set
KKT. Neither depends on the other, and this crate depends on neither — only
on `pounce-common` and `pounce-linalg`.

## What is deliberately not here

Two parts of the NLP arm's sensitivity layer are genuinely engine-coupled and
stay in `pounce-sensitivity`, so the boundary is a decision rather than an
oversight:

- **The corrector.** Its entry points take the concrete `PdSensBacksolver` and
  reach for `activity_handles()`, `offsets_public()`, `block_dims()`,
  `pack_natural()` and `corrector_sigma()` — none on the trait, several
  meaningful only for the filter-IPM's eight-block compound iterate.
- **Activity classification's plumbing.** It reads the filter-IPM's own
  iterate (`z_l`, `z_u`, `v_l`, `v_u`) out of an `IpoptData` handle. Its pure
  decision rule is portable; the plumbing around it is not.

Generalizing either means abstracting `IpoptData` / `CalculatedQuantities`
access behind another trait — a larger project than this crate.

## Algorithmic reference

> Pirnay, H., López-Negrete, R., and Biegler, L.T. (2012).
> *Optimal sensitivity based on IPOPT.*
> Mathematical Programming Computation, **4**(4), 307–331.
> DOI: [10.1007/s12532-012-0043-2](https://doi.org/10.1007/s12532-012-0043-2).

## Upstream source

The port follows the upstream sIPOPT contrib at
[`ref/Ipopt/contrib/sIPOPT/src/`][upstream-src] in this repo (EPL-2.0,
© Hans Pirnay 2009–2011 per the headers). The files mapped here moved out of
`pounce-sensitivity` when this crate was split off; the mapping is unchanged:

| pounce-sens-core                             | upstream                                                                                    |
|----------------------------------------------|---------------------------------------------------------------------------------------------|
| `src/schur_data.rs` — `SchurData` trait      | [`SensSchurData.hpp`](../../ref/Ipopt/contrib/sIPOPT/src/SensSchurData.hpp) (lines 17–177)   |
| `src/schur_data.rs` — `IndexSchurData`       | [`SensIndexSchurData.{hpp,cpp}`](../../ref/Ipopt/contrib/sIPOPT/src/SensIndexSchurData.hpp)  |
| `src/p_calculator.rs` — `PCalculator` trait  | [`SensPCalculator.hpp`](../../ref/Ipopt/contrib/sIPOPT/src/SensPCalculator.hpp) (lines 17–133) |
| `src/backsolver.rs` — `SensBacksolver` trait | [`SensBacksolver.hpp`](../../ref/Ipopt/contrib/sIPOPT/src/SensBacksolver.hpp)                |

Every public item in this crate documents the upstream symbol it mirrors,
with line numbers when they're stable.

## License

EPL-2.0, matching upstream Ipopt and the sIPOPT contrib.

[upstream-src]: ../../ref/Ipopt/contrib/sIPOPT/src/

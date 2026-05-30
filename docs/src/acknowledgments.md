# Acknowledgments

POUNCE is a Rust port of [Ipopt](https://github.com/coin-or/Ipopt),
the interior-point nonlinear programming solver by Andreas Wächter,
Lorenz T. Biegler, and the COIN-OR community. Its algorithm, console
output, and option semantics are modeled directly on that codebase,
which is released under the EPL-2.0.

It is a sibling of [ripopt](https://github.com/jkitchin/ripopt), an
earlier memory-safe interior-point NLP optimizer in Rust by the same
author (DOI
[10.5281/zenodo.19542664](https://doi.org/10.5281/zenodo.19542664)).

## Contributors

- **David Bernal Neira** ([@bernalde](https://github.com/bernalde))
  designed and prototyped the auxiliary-equality preprocessing pass
  in [ripopt PR #32](https://github.com/jkitchin/ripopt/pull/32).
  POUNCE's `pounce-presolve::auxiliary` Phase-0 orchestrator (issue
  [#53](https://github.com/jkitchin/pounce/issues/53)) is a port of
  that work — Hopcroft-Karp matching, Dulmage-Mendelsohn partition,
  Tarjan SCC, block-triangular reduction, damped-Newton block
  solver, reduction frame with multiplier recovery — and ships
  with the `tutorial_flow_density{,_perturbed}.nl` and
  `gaslib11_steady.nl` test fixtures David vendored.

## Key references

- Wächter, A., Biegler, L.T. "On the implementation of an
  interior-point filter line-search algorithm for large-scale
  nonlinear programming." *Mathematical Programming* 106(1), 25–57
  (2006). DOI
  [10.1007/s10107-004-0559-y](https://doi.org/10.1007/s10107-004-0559-y)
  — the algorithm POUNCE implements.
- Wächter, A., Biegler, L.T. "Line search filter methods for nonlinear
  programming: Motivation and global convergence." *SIAM Journal on
  Optimization* 16(1), 1–31 (2005). DOI
  [10.1137/S1052623403426556](https://doi.org/10.1137/S1052623403426556)
- Wächter, A., Biegler, L.T. "Line search filter methods for nonlinear
  programming: Local convergence." *SIAM Journal on Optimization*
  16(1), 32–48 (2005). DOI
  [10.1137/S1052623403426544](https://doi.org/10.1137/S1052623403426544)
- Fletcher, R., Leyffer, S. "Nonlinear programming without a penalty
  function." *Mathematical Programming* 91(2), 239–269 (2002). DOI
  [10.1007/s101070100244](https://doi.org/10.1007/s101070100244) — the
  filter concept underlying the line search.
- Pirnay, H., López-Negrete, R., Biegler, L.T. "Optimal sensitivity
  based on IPOPT." *Mathematical Programming Computation* 4(4),
  307–331 (2012). DOI
  [10.1007/s12532-012-0043-2](https://doi.org/10.1007/s12532-012-0043-2)
  — the sIPOPT method behind `pounce-sensitivity`.
- Duff, I.S. "MA57—a code for the solution of sparse symmetric
  definite and indefinite systems." *ACM Transactions on Mathematical
  Software* 30(2), 118–144 (2004). DOI
  [10.1145/992200.992202](https://doi.org/10.1145/992200.992202) — the
  optional `ma57` linear-solver backend.
- Wilkinson, M.D. et al. "The FAIR Guiding Principles for scientific
  data management and stewardship." *Scientific Data* 3, 160018
  (2016). DOI
  [10.1038/sdata.2016.18](https://doi.org/10.1038/sdata.2016.18) — the
  provenance model behind the [JSON solve report](json-output.md).

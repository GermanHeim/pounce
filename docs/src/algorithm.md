# Algorithm & Workspace

## Algorithm

POUNCE implements the interior-point filter line-search algorithm of
Wächter & Biegler (2006) — the same algorithm upstream Ipopt uses. A
solve proceeds as a sequence of barrier subproblems: for a decreasing
sequence of barrier parameters μ, it takes primal-dual Newton steps on
the perturbed KKT system, accepting each step through a filter
line-search that balances objective descent against constraint
infeasibility. When a regular step cannot be found, a restoration
phase minimizes constraint violation to return the iterate to a
filter-acceptable region.

See [Acknowledgments](acknowledgments.md) for the papers behind each
component.

## Workspace layout

POUNCE is a Cargo workspace. Each crate maps onto a part of the
upstream Ipopt source tree:

| Crate | Purpose |
|---|---|
| `pounce-common` | Types, exceptions, journalist, options, tagged objects, cached results (Ipopt `src/Common`). |
| `pounce-linalg` | BLAS-1, dense/compound vectors and matrices, triplet storage, CSC conversion (Ipopt `src/LinAlg`). |
| `pounce-linsol` | Symmetric linear-solver trait layer — no FFI; backends plug in below. |
| `pounce-feral` | Pure-Rust sparse symmetric LDLᵀ backend. The default. |
| `pounce-hsl` | MA57 backend via `libcoinhsl` (optional, behind the `ma57` feature). |
| `pounce-nlp` | TNLP trait, TNLPAdapter, `IpoptApplication` entry point (Ipopt `src/Interfaces`). |
| `pounce-algorithm` | IteratesVector, IpoptData, calculated quantities, KKT, line search, μ update, convergence check, main loop (Ipopt `src/Algorithm`). |
| `pounce-restoration` | Restoration phase (Ipopt `Algorithm/Resto*`). |
| `pounce-presolve` | Presolve / problem-reduction pass run before the IPM. |
| `pounce-l1penalty` | ℓ₁-exact penalty-barrier wrapper for degenerate / MPCC NLPs. |
| `pounce-sensitivity` | Parametric sensitivity (port of Ipopt `contrib/sIPOPT`). |
| `pounce-cinterface` | C ABI shim — `IpoptCreate` / `IpoptSolve` / `IpoptFreeProblem`. |
| `pounce-py` | Python bindings (the `pounce` Python package). |
| `pounce-cli` | The `pounce` command-line driver. |

The C ABI shim lets existing PyIpopt / cyipopt / JuMP / AMPL clients
link against POUNCE in place of Ipopt.

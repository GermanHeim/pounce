# Global Optimization

Most of POUNCE settles a problem at a **local** optimum (the NLP filter-IPM and
SQP) or exploits convexity so that local *is* global (the convex/conic IPM).
For a genuinely **nonconvex** problem, POUNCE offers one certified-global
route, and it is for **polynomials**:

- **The SOS / Lasserre hierarchy** (`pounce-convex`) — for **polynomial**
  problems, via a single semidefinite program. Callable from Rust
  (`sos_minimize`) and Python (`pounce.sos_minimize`).

It returns a result that is *certified*: a lower bound together with a moment
certificate that, when exact, pins the global minimum and recovers its
minimizer(s).

> **There is no general-purpose spatial branch-and-bound solver in POUNCE.**
> For a nonconvex problem that is *not* polynomial — anything with
> `exp`/`ln`/trig — POUNCE has no certified-global path. Use the local NLP
> solver from several starting points (see the multistart notebooks below), or
> reformulate into the convex cone library. A `pounce-global` crate was
> prototyped and removed from `main` before release; its design is recorded in
> `dev-notes/spatial-bnb-design.md`.

## The SOS / Lasserre path (polynomials)

When the objective and constraints are **polynomials**, the
sum-of-squares / moment approach in `pounce-convex` certifies the global
minimum from a *single* semidefinite program — no branching — by searching for
the largest `γ` such that `p(x) − γ` lies in the Putinar cone (a sum of squares
plus constraint multipliers). The SDP is solved by POUNCE's own convex conic
interior-point method; flat truncation of the resulting moment matrix certifies
when the bound is exact, and a **facial-reduction** step recovers every global
minimizer — even when the optimum is attained at several points.

From Python, a polynomial is a **dict mapping an exponent tuple to its
coefficient** (the all-zeros key is the constant term):

```python
from pounce.sos import sos_minimize

# x**4 - 2 x**2 + 3  ->  global minimum 2, attained at BOTH x = +1 and x = -1
r = sos_minimize({(4,): 1.0, (2,): -2.0, (0,): 3.0})
r.lower_bound       # ≈ 2.0
r.is_exact          # True — flat-truncation certificate: the bound is the minimum
r.minimizers        # both x = +1 and x = -1
```

Constraints are polynomials too, passed as `inequalities` (`g_i(x) ≥ 0`) and
`equalities` (`h_j(x) = 0`); raise the relaxation `order` to tighten the bound
(the Lasserre hierarchy) at the cost of a larger SDP. A runnable walkthrough —
double well, a constrained problem, and a 2-D example — is in
[`18_sos_global_optimization.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/18_sos_global_optimization.ipynb).

The same solver from Rust, via the `pounce-rs` facade with the `convex`
feature on (`pounce-rs = { version = "0.9", features = ["convex"] }`):

```rust
use pounce_rs::convex::{sos_minimize, PolyProblem, Polynomial};
use pounce_rs::linsol::backend;      // the sparse LDLᵀ factory the solver takes

// x⁴ − 2x² + 3 → global minimum 2 at x = ±1.
let p = Polynomial::new(1, vec![(vec![4], 1.0), (vec![2], -2.0), (vec![0], 3.0)]);
let sol = sos_minimize(&PolyProblem::new(p), None, backend);
// sol.lower_bound ≈ 2; when the moment matrix is flat, sol.minimizers holds
// the global minimizer(s) — here both x = +1 and x = −1.
```

The full treatment lives in the `pounce_convex::sos` module documentation —
reachable without a second dependency, since `pounce_rs::convex` re-exports
the `pounce_convex` crate itself for anything outside its curated surface.

**When SOS fits:** polynomials of modest degree and dimension — one SDP,
recovers all global minimizers, but the SDP grows with the relaxation order.

## When SOS does not fit

For a general factorable problem (`exp`/`ln`/trig), or a polynomial whose SDP
would be too large, the textbook tool is spatial branch-and-bound — and POUNCE
does not have one. Two things you *can* do:

- **Reformulate into the cone library.** If the model can be cast as an LP,
  convex QP, SOCP, or an exponential / power / PSD cone program, local *is*
  global and the guarantee comes for free. See
  [Choosing a Solver](choosing-a-solver.md).
- **Multistart the local solver.** Running the NLP filter-IPM from many
  starting points finds the low minima in practice, but certifies nothing —
  there is no bound proving you have the global one. Three notebooks work
  through the tactics: repulsion-based sampling
  ([`19_find_minima_repulsion.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/19_find_minima_repulsion.ipynb)),
  random restarts
  ([`20_find_minima_restart.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/20_find_minima_restart.ipynb)),
  and basin hopping
  ([`21_find_minima_hopping.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/21_find_minima_hopping.ipynb)).

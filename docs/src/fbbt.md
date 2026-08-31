# Feasibility-Based Bound Tightening (FBBT)

`pounce` supports feasibility-based bound tightening on nonlinear
constraints: interval-arithmetic propagation through the constraint
expression DAG to discover variable bounds the user did not write
down (e.g. `x² + y² ≤ 1` ⇒ `x ∈ [-1, 1]`, `exp(x) ≤ 10` ⇒
`x ≤ ln 10`). It pairs with the linear bound-tightening already in
the presolve pipeline (which only handles linear constraints).

Tracks [issue #62](https://github.com/jkitchin/pounce/issues/62).
References: Belotti, Cafieri, Lee, Liberti (2010).

## When it helps

- The Jacobian / objective row magnitudes are wildly different from
  what the user-declared bounds suggest.
- A nonlinear equality or one-sided inequality is much tighter than
  the user's `[lo, hi]` box.
- Loose bounds were inherited from a modeling tool that doesn't
  propagate constraints back to variable boxes (most modeling tools
  don't).

FBBT cannot help when:

- The TNLP has no structural-expression representation. `.nl`-loaded
  problems (`NlTnlp`) expose one, and `pounce-rs` builder problems can opt in
  with `Problem::constraint_expression`. Python (`PyTnlp`) and C-callback
  (`CCallbackTnlp`) problems still silently opt out.
- The expression uses operators FBBT doesn't reason about
  (`Funcall` to AMPL imported functions, variable-exponent powers,
  `sin` / `cos` reverse pass). Those subtrees become opaque and
  block tightening through them, but the rest of the constraint
  still propagates normally.

## Options

| Option | Default | Effect |
|---|---|---|
| `presolve_fbbt` | `no` | Master switch. Requires `presolve=yes` and an `ExpressionProvider`. |
| `fbbt_tol` | `1e-6` | Minimum per-variable bound improvement to keep iterating. |
| `fbbt_max_iter` | `10` | Outer-sweep cap. |
| `fbbt_max_constraints` | `0` | Per-sweep cap on constraints inspected (`0` = unlimited). |

FBBT runs after the linear bound-tightening (Phase 1) and before
the redundant-constraint pass (Phase 2), so any FBBT-derived
tightening feeds forward into row drops, the LICQ check, and the
bound-multiplier warm starts.

## Reading the presolve banner

With `presolve_fbbt=yes`, the per-solve presolve banner prints two
lines instead of one:

```text
Presolve: tightened 170 bounds (82 newly-finite), dropped 46 redundant rows, LICQ=Full
Presolve FBBT: 10 sweeps, 1362 variable tightenings (Σ|Δ|=7.5e20)
```

Fields:

- `sweeps` — number of outer iterations actually executed
  (≤ `fbbt_max_iter`). Hitting the cap is informational, not an
  error.
- `variable tightenings` — total count of per-variable
  `(x_lo[j], x_hi[j])` updates that strictly improved the box.
- `Σ|Δ|` — sum of absolute bound improvements across all updates.
  Provided as a coarse "how much did we move" signal — not part of
  the FBBT algorithm.

If FBBT detects infeasibility (the constraint bound is disjoint
from the interval enclosure at the current variable box), it stops
and emits `pounce: FBBT detected infeasibility (witness constraint
N)`. The solve continues with the partially-updated bounds — the
IPM will then report infeasibility through its own channels.

## Should I turn it on?

The issue's design says: default off until benchmark evidence
justifies a flip. Today's evidence:

- On small problems (e.g. `tutorial_flow_density.nl`): FBBT moves
  iteration count slightly, sometimes up, sometimes down.
- On larger problems (e.g. `gaslib11_steady.nl`): FBBT enables
  additional redundant-row drops and can promote the LICQ verdict
  from `StructuralRank` to `Full`, but the iteration count change
  is mixed.

So: try it on your problem. If you see fewer iterations or a
cleaner LICQ verdict, keep it on; if it costs iterations, turn it
off again. The cost of FBBT itself is small (one pass over the
expression DAGs per sweep, capped at `fbbt_max_iter`).

## Soundness guarantees

FBBT uses outward-rounded interval arithmetic. Every operation
widens its result by one ULP outward so accumulated floating-point
error always increases the interval, never shrinks it. Non-finite
endpoints round outward too: an overflowed `+inf` upper endpoint
stays `+inf`, an overflowed `-inf` lower endpoint stays `-inf`, and
an endpoint that arithmetic destroyed (`∞−∞`, `∞/∞`) widens to the
matching infinity rather than propagating a `NaN`.

The consequence: FBBT may produce a *looser* tightening than ideal,
but it does not drop a feasible point — **provided the tape it is
given obeys the tape contract below.** That proviso is not a
formality. It is the whole of gh #877: an earlier revision of this
crate rounded non-finite endpoints by leaving them alone, computed
`n`-th roots as `powf(1.0/n)` with a one-ULP pad, and reverse-
propagated out of *every* tape slot. Each of those turned a feasible
point into a cut one — the last of them silently, reporting
`Solve_Succeeded` on an answer wrong by the entire feasible range.

The contract a caller's tape must satisfy — enforced by construction
for `.nl` models, and the caller's responsibility when a
`Problem::constraint_expression` implementation hand-writes one:

1. **The tape restates the constraint.** Its root value must equal
   what `constraints()` returns at every point of the box, not only
   at the two points `pounce-rs` samples as a smoke test.
2. **Every slot influences the root's value.** A slot the root does
   not depend on carries no information about the constraint, so
   propagating a bound backwards out of it fabricates one. Since
   `PowInt(a, 0)` is `1` for every `a`, this is strictly stronger
   than reachability: the reverse pass computes influence itself
   (`influencing_slots`) and skips the rest, so a tape carrying dead
   or `^0`-masked slots is tightened conservatively rather than
   wrongly. Keeping them out of the tape is still cheaper.

The pointwise soundness fuzz tests in
`crates/pounce-presolve/src/fbbt/{forward,reverse,orchestrator}.rs`
verify the property on random sample grids, and
`crates/pounce-presolve/tests/issue877_fbbt_unsound_tightening.rs`
pins each of the three gh #877 mechanisms against the branch it
reaches.

## Operator support

Forward + reverse rules cover the operators that account for ~all
nonlinear constraints in practice:

| Operator | Forward | Reverse |
|---|---|---|
| `+ - * / neg` | ✓ | ✓ |
| `pow` (integer constant) | ✓ | ✓ (branch-selecting for even powers) |
| `pow` (variable / non-integer) | opaque | opaque |
| `sqrt exp ln abs` | ✓ | ✓ (with domain clipping) |
| `sin cos` | ✓ (loose) | declines to tighten |
| `log10` | rewritten as `ln / ln(10)` | follows the rewrite |
| AMPL imported `Funcall` | opaque | opaque |
| n-ary `Sum` | folded into binary `Add` | follows the fold |

`Opaque` slots evaluate to `[-∞, +∞]` on the forward pass and block
reverse propagation through them — they don't pollute the rest of
the constraint.

## Extending support to new TNLP sources

Builder users implement the optional hook and enable both `presolve` and
`presolve_fbbt`:

```rust
impl Problem for MyProblem {
    fn constraint_expression(&self, i: usize) -> Option<pounce_rs::FbbtTape> {
        Some(tapes[i].clone())
    }
}
```

The builder returns the resulting `FbbtReport` on `Solution::fbbt_report`.

The tape for row `i` must exactly restate `constraints(x)[i]`. A mismatch can
tighten bounds around the wrong function and cut off the true optimum.
`try_solve` rejects disagreements found at the starting point or box midpoint,
but this is only a smoke check, and a loose one: two points, compared to a
relative tolerance of `sqrt(f64::EPSILON)` (~1.5e-8). A mismatch elsewhere, or
a smaller one at those points, has no diagnostic. Generate the callback and
tape from one expression source when possible.

FBBT consumes the `pounce_nlp::expression_provider::ExpressionProvider`
trait. Any TNLP can opt in by implementing:

```rust
impl ExpressionProvider for MyTnlp {
    fn constraint_expression(&self, i: usize) -> Option<pounce_nlp::FbbtTape> {
        // Build a tape from your problem's symbolic structure.
        // Return None to decline (FBBT becomes a no-op on that
        // constraint).
    }
}
```

`FbbtTape` is a flat tape of `FbbtOp` nodes; the existing
`NlTnlp` implementation in `crates/pounce-cli/src/nl_fbbt_translate.rs`
is the canonical template (it walks an AMPL `Expr` tree, preserving
CSE sharing via `Rc::as_ptr` keying). Building a similar tape from
a Pyomo, JAX, or sympy expression is a finite-effort project.

## References

- Belotti, Cafieri, Lee, Liberti. *On feasibility based bounds
  tightening.* (2010).
  <https://enac.hal.science/hal-00935464v1/document>
- Liberti et al. *Feasibility-based bounds tightening via fixed
  points.* COCOA 2010.
  <https://www.lix.polytechnique.fr/~liberti/fbbt-cocoa10.pdf>
- Puranik, Sahinidis. *Domain reduction techniques for global NLP
  and MINLP optimization.* Constraints 22 (2017).
  <https://arxiv.org/pdf/1706.08601>
- pounce issue [#62](https://github.com/jkitchin/pounce/issues/62).

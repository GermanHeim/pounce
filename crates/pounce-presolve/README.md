# pounce-presolve

Algorithmic NLP preprocessing for POUNCE, exposed as a TNLP wrapper.

Internal crate. Wraps a user [`TNLP`](../pounce-nlp) and applies
presolve passes before the IPM ever sees the problem:

- **Bound tightening** — propagates implied bounds from linear
  constraints back onto variable boxes.
- **Redundant-constraint removal** — drops constraints that are
  implied by the variable bounds.
- **LICQ degeneracy detection** — flags rank-deficient constraint
  Jacobians so the algorithm can pick a more robust strategy (e.g.
  the ℓ₁ exact penalty-barrier in `pounce-l1penalty`).

Opt-in. Off by default; enable via `SolverOptions::presolve = true`.

## Status

Scaffolding. Bound tightening and redundant-row elimination land
first; the LICQ detector is wired against the algorithm crate's
inertia signal.

## License

EPL-2.0.

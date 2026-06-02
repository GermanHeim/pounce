# `find_minima`: a multiple-minima companion to `minimize`

Status: design brainstorm (not yet implemented)

## Motivation

`pounce.minimize` finds *one* local minimum from a starting point. A large
class of problems — molecular conformer search, parameter estimation with
multiple solutions, global optimization, bifurcation/equilibrium
enumeration — instead needs *many* distinct minima, or the global one among
them. The recurring trick across several fields is the same:

> solve → record the minimum → transform the problem so the solver can no
> longer settle there → re-solve.

This note designs a `find_minima` companion that packages that loop on top
of the existing `minimize` facade, with the escape mechanism pluggable so
flooding, deflation, and tunneling share one driver.

A working prototype of the flooding variant lives at
`python/examples/gaussian_hump_minima.py`.

## Prior art (verified references)

The "make the found minimum unattractive and re-solve" idea was invented
independently in optimization theory, molecular dynamics, and numerical
PDEs.

Optimization theory:

- Ge, R. "A filled function method for finding a global minimizer of a
  function of several variables." *Mathematical Programming* **46**,
  191–204 (1990). doi:10.1007/BF01585737 — the additive-bump-on-a-minimum
  method; the closest formal match to the flooding prototype.
- Levy, A.V. & Montalvo, A. "The tunneling algorithm for the global
  minimization of functions." *SIAM J. Sci. Stat. Comput.* **6**(1),
  15–29 (1985). doi:10.1137/0906002 — horizontal "tunnel" to an
  equal-height point past the barrier, using poles to repel known minima.

Molecular dynamics / chemistry:

- Laio, A. & Parrinello, M. "Escaping free-energy minima." *PNAS*
  **99**(20), 12562–12566 (2002). doi:10.1073/pnas.202427399 —
  metadynamics; history-dependent Gaussians in collective-variable space.
- Grubmüller, H. "Predicting slow structural transitions in macromolecular
  systems: Conformational flooding." *Phys. Rev. E* **52**(3), 2893–2906
  (1995). doi:10.1103/PhysRevE.52.2893 — independent, earlier Gaussian
  flooding potential on a found minimum.
- Huber, T., Torda, A.E. & van Gunsteren, W.F. "Local elevation: a method
  for improving the searching properties of molecular dynamics
  simulation." *J. Comput.-Aided Mol. Des.* **8**, 695–708 (1994).
  doi:10.1007/BF00124016 — the memory-based precursor to metadynamics.
- Barducci, A., Bussi, G. & Parrinello, M. "Well-tempered metadynamics: a
  smoothly converging and tunable free-energy method." *Phys. Rev. Lett.*
  **100**, 020603 (2008). doi:10.1103/PhysRevLett.100.020603 — the
  curvature-adaptive bump height (A ∝ σ²·λ_min); basis for the "auto"
  hyperparameter mode below.

Energy-landscape search (related; hop via MC/MD rather than filling):

- Li, Z. & Scheraga, H.A. "Monte Carlo-minimization approach to the
  multiple-minima problem in protein folding." *PNAS* **84**(19),
  6611–6615 (1987). doi:10.1073/pnas.84.19.6611 — the origin of
  perturb-then-locally-minimize-then-Metropolis-accept (basin-hopping).
- Wales, D.J. & Doye, J.P.K. "Global optimization by basin-hopping and the
  lowest energy structures of Lennard-Jones clusters containing up to 110
  atoms." *J. Phys. Chem. A* **101**(28), 5111–5116 (1997).
  doi:10.1021/jp970984n — the named, widely-used basin-hopping algorithm
  (scipy parity: `scipy.optimize.basinhopping`).
- Wales, D.J. & Scheraga, H.A. "Global optimization of clusters, crystals,
  and biomolecules." *Science* **285**(5432), 1368–1372 (1999).
  doi:10.1126/science.285.5432.1368 — survey of hypersurface-deformation
  and hopping methods.
- Goedecker, S. "Minima hopping: an efficient search method for the global
  minimum of the potential energy surface of complex molecular systems."
  *J. Chem. Phys.* **120**(21), 9911–9917 (2004). doi:10.1063/1.1724816 —
  keeps a history of visited minima and penalizes revisiting.

Restart and clustering (the multistart family):

- Rinnooy Kan, A.H.G. & Timmer, G.T. "Stochastic global optimization
  methods part I: Clustering methods." *Mathematical Programming* **39**,
  27–56 (1987). doi:10.1007/BF02592070.
- Rinnooy Kan, A.H.G. & Timmer, G.T. "Stochastic global optimization
  methods part II: Multi level methods." *Mathematical Programming* **39**,
  57–78 (1987). doi:10.1007/BF02592071 — Multi-Level Single Linkage
  (MLSL): start a local solve from a sample point only when no
  already-found basin is nearby, so each basin is descended ~once.

Deflation (the root-finding cousin; multiplicative repulsion for Newton):

- Brown, K.M. & Gearhart, W.B. "Deflation techniques for the calculation of
  further solutions of a nonlinear system." *Numerische Mathematik* **16**,
  334–342 (1971). doi:10.1007/BF02165004 — origin of deflation.
- Farrell, P.E., Birkisson, Á. & Funke, S.W. "Deflation techniques for
  finding distinct solutions of nonlinear partial differential equations."
  *SIAM J. Sci. Comput.* **37**(4), A2026–A2045 (2015).
  doi:10.1137/140984798 — deflation operator on the residual; most natural
  framing for a Newton-type NLP solver like pounce.
- (directly relevant prior art for the deflation-on-minimization path:
  "Deflation techniques for finding multiple local minima of a nonlinear
  least squares problem," arXiv:2409.14438, 2024.)

Adjacent paradigms (deliberately out of scope — see "Scope boundary"):

- Jones, D.R., Perttunen, C.D. & Stuckman, B.E. "Lipschitzian optimization
  without the Lipschitz constant." *J. Optim. Theory Appl.* **79**(1),
  157–181 (1993). doi:10.1007/BF00941892 — DIRECT; deterministic global
  via rectangle subdivision (scipy `method="direct"`).
- Breiding, P. & Timme, S. "HomotopyContinuation.jl: a package for homotopy
  continuation in Julia." *ICMS 2018*, LNCS 10931, 458–465 (2018).
  doi:10.1007/978-3-319-96418-8_54 — numerical polynomial homotopy
  continuation; finds *all* solutions of polynomial systems.

## Unifying observation

Every method here drives the same inner local solver (`minimize`) and
differs only in (a) how it transforms the problem between solves and/or
(b) how it picks the next start. That is the inclusion test for this family
(see "Scope boundary") and the axis the table organizes:

| method        | transform between solves                            | start policy                | family     |
|---------------|-----------------------------------------------------|-----------------------------|------------|
| flooding      | add Σ_k A_k·exp(−‖x−x*_k‖²/2σ_k²) to the objective   | roll from newest min        | repulsion  |
| deflation     | multiply ∇f (or residual) by ∏_k 1/‖x−x*_k‖^p        | reuse / warm start          | repulsion  |
| tunneling     | replace solve with root-find f(x)=f*, poles at x*_k  | tunneled point              | repulsion  |
| multistart    | none                                                 | random / Sobol box sample   | restart    |
| MLSL          | none                                                 | clustered sample (skip near-basin) | restart |
| basin-hopping | none                                                 | perturb current + Metropolis accept | hopping |

So one driver loop + a pluggable "escape strategy" covers all of them. Some
strategies are *stateful* (hopping carries a current point + temperature;
MLSL carries a sample pool + reduced distance threshold), so a strategy is
an object, not just a function. The verification of what counts as a real
minimum is identical across them and factors out.

## Proposed API

Mirror `minimize` so it is instantly familiar; add a `method=` selector in
the spirit of `scipy.optimize.minimize(method=...)`.

```python
result = pounce.find_minima(
    fun, x0,
    method="deflation",         # "deflation" | "flooding" | "tunneling"
                                #  | "multistart" | "mlsl" | "basinhopping"
    jac=None, hess=None,
    bounds=None, constraints=None,
    n_minima=10,                # TARGET: stop once this many distinct minima found
    max_solves=None,            # BUDGET: hard cap on solver calls (default 4*n_minima)
    patience=8,                 # GIVE-UP: stop after this many solves in a row with nothing new
    dedup=1e-4,                 # distance below which two minima are "the same"
    options=None,               # passed through to each minimize() (e.g. inner max_iter)
    strategy_kw=None,           # method-specific knobs (sigma, amplitude, pole power...)
    seed=None,
    callback=None,              # called with each accepted minimum
)
# result.minima  -> list[np.ndarray], sorted by objective
# result.values  -> list[float]
# result.x       -> best minimum
# result.results -> list[OptimizeResult] (one per accepted minimum)
# result.status  -> "target_reached" | "converged" | "budget_exhausted"
# result.trace   -> per-solve diagnostics
```

### Termination

The driver stops on whichever fires first:

| condition                              | meaning                          | `status`            |
|----------------------------------------|----------------------------------|---------------------|
| `len(minima) == n_minima`              | found what was asked for         | `"target_reached"`  |
| `patience` consecutive empty solves    | landscape appears exhausted      | `"converged"`       |
| `n_solves >= max_solves`               | spent the budget                 | `"budget_exhausted"`|

The "there aren't that many minima" case is exactly why `patience` exists. A
target alone never terminates when the function has fewer minima than
requested (restarts keep rediscovering the same ones); `max_solves` alone
terminates but wastes the whole remaining budget re-finding knowns before
giving up. `patience` is the efficient early exit: ask for 4, find 2, try
`patience` more times across the box, find nothing, return the 2 with
`status="converged"` — signalling *believed-exhaustive*, not *out of gas*.
`max_solves` remains as a hard backstop. `find_minima` always returns
however many it actually found; falling short of `n_minima` is not an error.

Granularity note: a *solve* is many function evaluations. The outer budget
is counted in solver calls (`max_solves`); a true per-evaluation ceiling
belongs inside each solve via `options` (e.g. `max_iter`), not conflated
into the outer loop.

## Architecture (four layers)

### 1. `MinimaArchive` — shared memory

Holds accepted minima (`x`, `f`, multipliers, local Hessian/curvature).
Responsibilities:

- dedup (Euclidean distance + objective-value agreement),
- expose centers/curvatures so strategies can build repulsion terms,
- a **pluggable distance metric** — the key MD-facing hook. Periodic-box
  (PBC) or symmetry-quotient distances drop in here without touching any
  strategy.

### 2. `EscapeStrategy` — the pluggable mechanism (`method=`)

Small protocol; a strategy implements whichever pair fits:

```python
class EscapeStrategy(Protocol):
    # A strategy may be stateful (hopping, MLSL) and implements whichever
    # hooks it needs; the driver calls the ones that are defined.
    def augment(self, fun, jac, hess, archive): ...   # flooding, deflation
        # -> (fun2, jac2, hess2)
    def subproblem(self, archive): ...                # tunneling
        # -> problem-shaped object for minimize()
    def propose_start(self, archive, rng, bounds): ...# restart / hopping
        # -> x0
    def accept(self, candidate, archive, rng): ...    # hopping (Metropolis)
        # -> bool  (default: always accept new, dedup elsewhere)
```

Concrete implementations:

*Repulsion (transform-and-resolve):*

- `Deflation` — multiply the gradient by ∏_k (1/‖x−x*_k‖^p + shift). Most
  Newton-natural; pairs with warm starts. Phase 1.
- `GaussianFlooding` — additive Gaussian bumps with analytic grad/Hess
  (already prototyped). `sigma`/`amplitude` from `strategy_kw`, or `"auto"`
  from local curvature (well-tempered: A ≈ c·σ²·λ_min). Phase 1.
- `Tunneling` — build the root-find f(x)=f* with poles; run as a
  feasibility solve. Yields a monotonically non-increasing minima sequence
  (good when the goal is the global min, not enumeration). Phase 3.

*Restart (smart start point, no transform):*

- `MultiStart` — sample the box (uniform or, better, Sobol/Halton
  low-discrepancy for coverage). The honest baseline. Phase 1.
- `MLSL` — Multi-Level Single Linkage: maintain a sample pool and start a
  local solve from a point only when no accepted minimum lies within a
  shrinking "reduced distance," so each basin is descended ~once. Provably
  efficient multistart. Phase 2.

*Hopping (Markov chain over minima):*

- `BasinHopping` — perturb the current minimum, local-minimize, accept by
  Metropolis on the objective. scipy parity; the chemistry/physics
  workhorse. Phase 2.

### 3. Acceptance / verification — shared

Independent of strategy:

- **polish** the candidate on the clean `fun` (no augmentation) so it lands
  in the true basin;
- **certify**: ‖∇f‖ ≈ 0 and Hessian PSD (smallest eigenvalue ≥ −tol) to
  reject saddles/maxima (when `hess` available);
- **feasibility / bounds** check for constrained problems.

"Search chooses where to look; verification decides what is real" — and it
lives in exactly one place.

### 4. Driver + `MinimaResult`

Loop: `propose_start → augment/subproblem → minimize → polish → verify →
archive`, until `n_minima`, `patience` (stagnation), or `max_solves` — see
the Termination table above. `MinimaResult` is a sorted list of
`OptimizeResult`-shaped entries plus a trace.

## Design decisions / trade-offs

- **Reuse `minimize`, don't reimplement.** Every strategy ultimately hands
  a wrapped `(fun, jac, hess)` (or a subproblem object) to the existing
  facade. Keeps `find_minima` a thin, pure-Python companion — no Rust
  changes for phase 1.
- **Auto hyperparameters.** `sigma/amplitude="auto"` from the local Hessian
  is well-tempered metadynamics; reuses the λ_min already computed for the
  saddle check. The escape condition is A/σ² > λ_min(∇²f(x*)) — flood
  harder than the basin curvature and the minimum turns into a saddle the
  solver leaves on its own.
- **Warm starts.** Consecutive flooded/deflated solves are near-identical
  problems. Strong synergy with the factor-once/solve-many `Solver` session
  (notebook 12) and batched warm start (notebook 11) — a real differentiator
  vs. a naive scipy loop.
- **Constraints.** Flooding passes through untouched (bumps touch only the
  objective; bounds/constraints carry over). Deflation under constraints
  means deflating the KKT residual — phase 2/advanced.
- **Parallelism.** `MultiStart` is embarrassingly parallel; flooding /
  tunneling / deflation are inherently sequential (each needs prior minima)
  but can parallelize within a round.
- **Relationship to scipy.** scipy has `basinhopping`, `shgo`,
  `dual_annealing`, `differential_evolution`, `direct` for the *global
  minimum*, but no tool that *enumerates distinct minima via
  deflation/flooding on a deterministic NLP solver*. That gap is the value
  proposition — lean into enumeration, not just "find the lowest." (We can
  still offer `basinhopping`/`multistart` for parity and as baselines.)

## Scope boundary

Inclusion test: **a method belongs in `find_minima` iff it drives the
existing local solver (`minimize`) as its inner loop**, differing only in
how it transforms the problem and/or picks the next start. That keeps the
package coherent and is exactly where it complements (rather than
duplicates) scipy. Deliberately *out* of scope, as different machinery:

- **Deterministic global / branch-and-bound** — DIRECT (Jones et al.
  1993), αBB, BARON. Need convex relaxations or Lipschitz sampling; they do
  not reuse the local solver. Rigorous global optimization is a separate
  effort.
- **Population / stochastic derivative-free globals** — differential
  evolution, dual/simulated annealing, CMA-ES, particle swarm. Already in
  scipy; do not exploit pounce's gradients/Hessians.
- **Homotopy continuation** — numerical polynomial homotopy (Breiding &
  Timme 2018) finds *all* stationary points, but only for polynomial /
  algebraic systems, via entirely separate machinery. The rigorous
  "all minima" cousin for that problem class.

These are worth a one-paragraph "see also" in user docs, not strategies.

## Known limitations

- A minimum hiding *within a few σ* of a placed bump is the blind spot: the
  bump distorts it and polishing rolls back to the flooded basin (dedup
  then rejects it). Resolving two very close minima needs a narrower σ.
- Starting exactly on a stationary point (saddle/max) leaves a Newton
  solver stuck; the Hessian certification filters these, and restarts avoid
  re-proposing them.
- No global guarantee of finding *all* minima — like every method here,
  coverage is heuristic and depends on σ, restarts, and budget.

## Suggested phasing

1. `Deflation` + `GaussianFlooding` + `MultiStart` (uniform/Sobol) +
   archive + verification, unconstrained and bound-constrained. Promote the
   prototype to `pounce/_minima.py`, export as `pounce.find_minima`. Tests
   on six-hump camel, Rastrigin, Ackley.
2. Warm-started re-solves via `Solver` sessions; `MLSL` and `BasinHopping`
   strategies (clustering restart + Metropolis hopping, scipy parity).
3. `Tunneling` + general constraints (deflate the KKT residual) + pluggable
   distance metrics (PBC / symmetry) for the MD audience.

# Covariance and information: v0.10 roadmap

**Status: roadmap proposal for pyomo-pounce, targeting v0.10.** This note
scopes the post-solve second-order surface in pyomo-pounce: `covariance()`,
shipping in v0.9, and the additions v0.10 should make around it. Nothing here
changes an existing signature. Companion to the active-set sensitivity roadmap
(`sensitivity-roadmap.md`), which extends `estimate()`; this note extends
`covariance()`.

## State of the art

Parameter covariance from the reduced Hessian is standard: at an estimation
optimum the covariance is the scaled inverse of the reduced Hessian of the
Lagrangian. The reduced Hessian is the object the sensitivity engines form
first. sIPOPT and k_aug both compute it, the information matrix, and hand it
back directly: k_aug writes the reduced Hessian to a file, and sIPOPT exposes
it through `rh_eigendecomp`. The parameter covariance is a downstream step on
top of it, invert and scale. The covariance-first tools sit a level up, at the
estimation frontend: Pyomo's `parmest` inverts the reduced Hessian to report a
covariance, and scipy's `curve_fit` returns the covariance (`pcov`) in the
Gauss-Newton form.

## Benefit hypothesis

The reduced Hessian and the covariance recipe are established, and pounce ships
them in its core, QP, and `curve_fit` interfaces (see Related reduced-Hessian
work below). What no pounce interface offers together, on a Pyomo model, is the
un-inverted object and a reduce-onto block chosen after the solve rather than
at declaration time. The next section states each against the shipped code.

Moving horizon estimation wants both at once, and is the worked case: its
information-form arrival cost is `Γ(x0) = ½ (x0 − x̂)ᵀ Π⁻¹ (x0 − x̂)`,
where `Π⁻¹` is the reduced Hessian of the one-step subproblem (the previous
arrival cost plus the stage leaving the window) marginalized onto the arrival
state. That is Lagrangian information, not covariance, on a block the solve did
not know about in advance.

## Where we are

Items 0, 1 and 2 are MERGED (jkitchin/pounce#371, #436 and #454): the
activity classifier ships in the Rust core as `Solver.classify_activity()`
plus `Solver.row_normal()` and `Solver.hessian_vec()`; `covariance()`
takes its membership from it, keeps a weakly active parameter at its true
variance, and projects binding rows; and `information()` ships as the
un-inverted sibling, built by tangent recovery rather than the Schur
construction sketched below (see item 2 for what the implementation
settled). The paragraphs below describe v0.9 as the baseline the roadmap
was written against; items 3 and 4 remain.

v0.9 shipped `covariance()` (in `pyomo-pounce/pyomo_pounce/sens.py`). You
`declare_fitted` a set of free variables, solve, and `covariance(model)`
returns their asymptotic covariance: the scaled parameter block of the
inverse KKT matrix, $2\sigma^2 (K^{-1})_{pp}$, where $\sigma^2$ is the residual
variance and the 2 is the sum-of-squares convention that makes the reduced
Hessian $2J^\top J$. Under the hood that block
is `inv(d2f*/dp2)`, the inverse reduced Hessian of the eliminated problem,
obtained by one backsolve per declared variable against the held
factorization. `hessian=` selects the Lagrangian (observed) or Gauss-Newton
(expected) form.

Two limits matter for what comes next:

1. **There is no un-inverted accessor.** `covariance()` returns the inverse
   reduced Hessian. A caller who wants the reduced Hessian itself, the
   information matrix, has to invert the covariance back. This is a
   pyomo-interface gap, not a pounce one: the core and QP interfaces already
   expose the reduced Hessian directly (see Related reduced-Hessian work
   below).
2. **The reduce-onto block is fixed at declaration time.** `covariance()`
   reports over the whole `declare_fitted` set. Asking about a different
   block means re-declaring and re-solving.

## Related reduced-Hessian work in other pounce interfaces

pounce already surfaces the reduced Hessian outside pyomo-pounce. The core
`Problem.solve_with_sens` returns it in natural (unscaled) units, so
`-inv(reduced_hessian)` is directly the covariance, with an eigendecomposition
(pounce#128, mirroring sIPOPT's `rh_eigendecomp`).
`QpSensitivity.reduced_hessian` mirrors it on the convex-QP side, and
`pounce.curve_fit` is the scipy-style covariance frontend for callable models,
of which `covariance()` is the Pyomo-model sibling.

## Roadmap

Items 2 to 4 are additive pyomo surface and can land on their own, running on
the shipped classifier, which is correct for interior solutions. Items 0 and 1
are what make the answer right at a bound, and item 1 consumes item 0.

Notation used throughout. `Σ_i = z^L_i/s^L_i + z^U_i/s^U_i` is the barrier
diagonal on variable `i`, summed over both bounds. $H$ is the Lagrangian
Hessian, so `H_ii` is that variable's objective curvature; the held factor
carries $W = H + \Sigma$, the barrier-augmented block. $F$ is
the set of variables the reduction keeps, $A$ the ones it projects out.

**0. The activity classifier → Rust core.** Which regime each bounded variable
and each inequality row is in, returned with the matrix by both accessors.

```
classify(i):                          # a bounded variable, or an inequality row
    Σ = z/s, summed over whichever sides exist
    q = |H_ii|                                  variable: curvature in that coordinate
        |∇gᵢᵀ H ∇gᵢ| / ‖∇gᵢ‖⁴                   row: see below

    if q < sqrt(eps_machine) * max(1, max_j |H_jj|):
                                      return unidentified, sign of q's value
    r = Σ / q

    if μ > 1e-4:                      # the μ-edges thin toward the band
                                      # (they meet it at μ = 1e-2); only
                                      # the two clear calls are made
        return inactive         if r < 1e-1
        return strongly active  if r > 1e1
        return ambiguous

    return inactive         if r < √μ
    return strongly active  if r > 1/√μ
    return weakly active    if 1e-1 ≤ r ≤ 1e1
    return ambiguous                  # in a gap between the band and an edge
```

The row denominator carries the fourth power so `r` is invariant to
rescaling the row: `d → c·d` sends `Σ → Σ/c²` while curvature along the
unit normal is unchanged, and `‖∇g‖⁴` restores the balance (equivalently,
the geometric weight `Σ‖∇g‖²` against curvature along the unit normal).
This also absorbs the solver's per-row `d_scale`; without it, the second
review of #371 measured an exactly-active row classifying `inactive` at
row coefficient 1000. The merged report is USER-SPACE indexed (a
`make_parameter`-removed variable reports `fixed` at its own index, an
equality row `equality`) and exports `var_sigma`/`row_sigma` and
`row_normal(j)` in natural units: classification runs on the solver's
scaled quantities (the ratio is scale-invariant), the exports follow
the sensitivity-output contract, and item 1 consumes them.

It requires `bound_relax_factor = 0` and checks it rather than documenting
it, since the shipped default lets a converged primal sit outside its bound
and a user hits that by doing nothing. Two more conditions are checked on
every call, not only in the tests: `s·z` away from `μ`, meaning the point is
off the central path or the bound was relaxed, and `Σ_i/|H_ii| > 100μ`
on a variable classified inactive, meaning barrier curvature survived where
none should be (a weakly active variable is kept with ratio near one by
design, so the check excludes it). The contamination threshold is
μ-relative because `inactive` means `r = O(μ)`: a fixed constant can
never sit below the inactive edge `√μ` at any converged μ, which the
second review of #371 showed made the earlier fixed floor structurally
dead.

Why it is shaped that way:

- **The ratio, not `s` and `z` separately.** Both are `O(√μ)` at weak
  activity, so any constant threshold on them tracks the solve rather than the
  geometry. `r` is `O(μ)`, `O(1)` and `O(1/μ)` across the three regimes.
- **Edges at `√μ` and `1/√μ`.** The geometric midpoints between those
  scalings, so the weakly active case sits at the centre with `0.5·|log μ|`
  decades of margin either side.
- **A fixed inner band.** It absorbs the coupling drift
  $\Sigma_i = H_{ii} + \sum_{j \ne i} H_{ij} x_j / x_i$, which is `O(1)` in
  `μ`. Scaling it with `μ` would widen it as the solve tightens, exactly where
  the classification should be sharpest.
- **The denominator guard first.** `|H_ii|` is a curvature scale only where
  the objective curves. An inactive bound a full unit from its variable
  classifies as `ambiguous` once `H_ii` reaches `1e-6` and as `strongly
  active` by `1e-13`, and tightening `μ` relocates the misfile rather than
  removing it. Below the square root of machine precision, relative to the
  block's largest curvature, a diagonal is noise.
- **Rows use the same `r`.** A row carries a slack and a multiplier with
  `s_j z_j = μ` exactly as a bound does, and its denominator reduces to
  `H_ii` when the row is a bound in disguise, so a limit written either way
  lands in the same regime by construction (jkitchin/pounce#362).

**1. Membership and dispositions.** What `covariance()` and `information()`
each return for a variable, given item 0's classification. Write
$S = H_{AA} - H_{AF} H_{FF}^{-1} H_{FA}$ for the reduction onto the pinned
set. Each accessor returns a matrix over the whole block; the columns are the
row a fitted variable $i$ gets in each:

| status | `s` | `z` | `Σ` as `μ → 0` | $i$ in | `covariance()` row | `information()` row |
|---|---|---|---|---|---|---|
| inactive | `O(1)` | `→ 0` | `μ/s² → 0` | $F$ | $2\sigma^2 (H_{FF}^{-1})_{iF}$ | $H_{iF}$ |
| strongly active | `→ 0` | `O(1)` | `z²/μ → ∞` | $A$ | $0$ | $S_{iA}$ |
| weakly active | `→ 0` | `→ 0` | finite, `O(1)` | $F$ | $2\sigma^2 (H_{FF}^{-1})_{iF}$ | $H_{iF}$ |
| ambiguous | n/a | n/a | ratio in a band gap | $F$ | $2\sigma^2 (H_{FF}^{-1})_{iF}$ | $H_{iF}$ |
| unidentified | n/a | n/a | curvature below scale | $F$ | $2\sigma^2 (H_{FF}^{-1})_{iF}$ | $H_{iF}$ |

The `s` and `z` columns say what each regime looks like, not how it is
detected: weak activity is the case where both vanish together, and item 0
classifies on `Σ/q` rather than on either alone. Membership runs that rule
at the reduced fitted block rather than reading item 0's per-coordinate
status: a fitted parameter in the residual-variable idiom has a zero
Lagrangian diagonal (its curvature reaches it through the residual
equalities), so the coordinate rule honestly reports `unidentified` there.
The reduced `q` is the factor's fitted-block diagonal with the parameter's
own `Σ` removed; `Σ` itself is retained by the solve, and item 0 reports it
(`var_sigma`, `row_sigma`). Rows classify the same way, along their normals
within the block. The `Σ` column shows how the
barrier diagonal gets where it does, through the slack when the bound is
inactive and through the multiplier when it is active, and is what using the
factor's $W$ in place of $H$ would cost.

$S$ carries one caveat the table cannot: it is conditional on the rest of
$A$. With more than one pinned variable, $S_{ii}$ holds the others at their
bounds rather than marginalizing over them.

$S$ is built from the exact Lagrangian Hessian (`curr_exact_hessian`, which
the solver retains), formed densely on the small fitted block: full precision
at any `μ`, no recovery from the barrier-augmented factor. Binding rows enter
the same dense construction: with $Z$ an orthonormal basis for the null space
of their normals on the fitted block, covariance is
$2\sigma^2 Z (Z^T H_{FF} Z)^{-1} Z^T$ and information is its pseudo-inverse
$Z (Z^T H_{FF} Z) Z^T$. No binding rows means $Z = I$ and both collapse to
the table.

The two constructions split by the size of `Σ`. On the free block `Σ` is at
most the curvature's own scale, so the correction is a benign subtraction
off the factor's reduced Hessian; on the pinned rows `Σ` is `z²/μ` and the
subtraction cancels, which is what reserves the dense exact-Hessian
construction for $S$. A binding row's barrier weight needs no removal on
the projected directions at all: $Z^T a = 0$ annihilates it exactly. Since
item 2 landed, the per-row conditional-information scalar comes from the
tangent-recovered reduced Hessian (accurate to ~1e-6 where the factor
subtraction lost `log10(Σ/q)` digits; the residue is the binding row's own
slack-barrier weight in the recovery), with the subtraction kept as the
fallback outside the square estimation structure.

Two mechanism facts the implementation settled (merged in #436). The
subtraction composes without scale factors because the classifier's
exports are natural units at the boundary; nothing in the pyomo layer
tracks `df` or `d_scale`. And the Gauss-Newton path needs no `Σ`
correction at all, by an exact identity: the residual rows of the
K-inverse columns are $J$ times the W-based parameter sensitivities,
$Z_r = J M$, so $Z_r M^{-1} = J$ and the factor's barrier weight cancels
regardless of `Σ`. The Lagrangian branch corrects $R_W$ because it uses
the W-based reduced Hessian; Gauss-Newton rebuilds from the exact $J$.

The last two rows warn as well as return: `ambiguous` that re-solving
tighter will settle it, which works because the drift into the band is
`O(1)` while the edges move as
`sqrt(μ)`; `unidentified` that the variance is large rather than small. $F$
is the conservative side for `covariance()` and the anti-conservative side
for `information()`, so those warnings are load-bearing rather than
decoration.

`covariance()` ships a slack-only test today (`sens.py:826-827`,
`tol = 1e-6 * (1.0 + abs(xv))`), which pins a weakly active variable and
deletes its information.

A strongly active constraint row projects too, on the null space of its
normal. `A ≤ cap` pins the coordinate `A` and the table's disposition states
the truth; `A + B ≤ 1` pins a combination, and no per-coordinate disposition
can. So the reduction happens on the null space of the binding row normals
restricted to the fitted block, pushed back to the original coordinates.
Both accessors stay full-block, one row per fitted variable, singular by
exactly the number of binding rows. `covariance()` reports zero variance
along the pinned combination, and the surviving correlation structure says
what the data still determines: with `A + B` binding, both variances shrink
and the correlation is `-1`, the data determines only the difference.
`information()` is the pseudo-inverse, the Hessian projected on both sides
onto the free directions. Each binding row is named in a warning, and the
conditional information of the pinned combination itself, the row analog of
a pinned variable's $S_{ii}$, is a per-row scalar reported with that
warning, since it belongs to no variable's row of the matrix. Where the
row's normal is a single fitted coordinate, as the bound rewrite in
jkitchin/pounce#357 produces, the projection reduces to the table's
disposition, so the two spellings of the same limit agree
(jkitchin/pounce#362) in the returned matrices, not only in item 0's
classification.

The restriction is honest only while the row's support outside the
fitted block is pinned (pin columns cannot move and count as inside).
A binding row that reaches the fitted block through FREE eliminated
variables pins a direction the restricted normal cannot represent:
`a + r_1 <= cap` with `r_1 = y_1 - a - b x_1` pins a `b`-direction
while the restricted normal reads `e_a`, and the reduced-level ratio is
equally blind, since the row's barrier weight lands through the
elimination away from the restricted direction. Such a row takes item
0's raw classification, is kept unprojected, and warns explicitly; its
general treatment is the reduced normal through the elimination, which
belongs with item 2's machinery.

**2. `information()`, the un-inverted sibling of `covariance()`.** MERGED
(#454). Natural units, the core's convention; pyomo `covariance()` carries
the `2σ²` on top. Same `hessian=` selector, Lagrangian (default) or
Gauss-Newton. Binding rows follow item 1's projection rule, identically in
both accessors, and membership is literally shared: the item-1
classification block is extracted into `_classify_fitted_block(who=)` and
consumed by both, so the accessors cannot drift.

The construction changed from the sketch above. Instead of a Schur
complement off the held factor (which would inherit the factor's barrier
weight by subtraction), the Lagrangian form is built by TANGENT RECOVERY:
the x-blocks of the K-inverse columns are `T·M`, so `T = Zx·inv(M)`
exactly, and `R = TᵀHT` with the exact Lagrangian Hessian through a new
core primitive, `Solver.hessian_vec(v)` (user-space, natural units). The
factor's barrier weight cancels multiplicatively. Accuracy boundary,
measured: machine-exact for equality and variable-bound activity
(everything in `W` cancels, pinned variables included; 1.9e-16 at
`Σ/q ≈ 3e10` where subtraction loses ten digits), but a binding
INEQUALITY row couples through its slack barrier and leaves ~1e-6
relative residue at practical `μ`, degrading as `μ` tightens. The
recovery requires the square estimation structure (equalities determine
the non-fitted variables given the block): guarded by
`_estimation_counts()`, refusing in `information()` and falling back to
the subtraction in `covariance()`'s binding-row scalar.

The three dispositions shipped as designed: Gauss-Newton sliced last (the
`JᵀJ` product is formed over all fitted variables, `J = Z_r·inv(M)`
exactly, no `Σ` correction needed); the pinned set returned as $S$, not
zeros; an indefinite Lagrangian block returned as computed with a warning
naming Gauss-Newton. The inertia-correction guardrail carries over and
its warning is honestly testable: two interchangeable fitted variables
beside a pinned one force real `δ_w` into the held factor and an exact
zero pivot in the free block, exercising both the guardrail and the
singular-$S$ refusal in one fixture.

One discipline item 3 inherits: every index into the factor must route
through `Solver.primal_rows` (gh #450, landed mid-implementation): the
`.col`-order rows the session holds are full-x, the factor's x block is
var-x, and they diverge exactly when the model carries a fixed variable.

**3. `wrt=` block selection.** `covariance(model, wrt=block)` and
`information(model, wrt=block)` reduce onto any block of free variables off
the held factor, post-solve, since the factor covers every free variable.
`declare_fitted` is the default block when `wrt=` is omitted, so
`covariance(model)` behaves exactly as in v0.9. The block accepts a slice
(`m.x[t, :]`) or a `(Var, time)` pair, not only a hand-listed VarData set.

Each call re-reduces onto its own argument, so one solve serves as many
blocks as you ask about, and each gets that block's marginal.

With one exception, which is returned rather than hidden. A strongly active
variable outside the block is not deleted: its `Σ` stays in the held factor
and drives the coupling through it to zero as `μ` falls, so the block
converges to the value conditional on that bound rather than the marginal
over it, and its numbers move with `μ` on the way. The active set comes back
with the matrix.

Item 2's machinery sets the constructions here. A block that parameterizes
the constraint manifold (block size equals `n_var - n_eq`, the same square
structure `_estimation_counts()` checks) gets the exact tangent-recovered
Lagrangian; any other block reduces off the held factor with the item-1
corrections and that route's documented precision. Gauss-Newton profiles
through the K-inverse chain (`J_B = Z_r·inv(M_B)`) for any block.

**4. `retain_kkt()`, a factor-retention switch decoupled from
declarations.** The solve factors the KKT to solve the NLP; the only question
is whether that factor is kept for post-solve queries. Today any declaration
keeps it (`declare_sens_param`, `declare_fitted`, or `declare_residual`).
`retain_kkt()` keeps it with no declaration at all, which is what item 3's
declaration-free flow needs: the MHE case, where the arrival state and the
parameters are each queried by `wrt=` and neither is a default. It defaults
off, so a solve with no sensitivity pays nothing.

| setup | factor kept | `covariance(model)` | `covariance(model, wrt=T)` |
|---|---|---|---|
| nothing | no | error | error |
| `declare_fitted(S)` | yes | over S | over T |
| `retain_kkt()` only | yes | error, no default | over T |
| `retain_kkt()` + `declare_fitted(S)` | yes | over S | over T |

The columns show `covariance()`; `information()` follows the same rows, since
factor retention and the default block are accessor-agnostic. `gradient()` and
`estimate()` are unaffected, since the `declare_sens_param` they require
already keeps the factor. A block queried under that declaration alone comes
out conditional on the pinned parameter, since fixing an input conditions
rather than marginalizes.

## Preconditions

Four conditions underwrite the whole surface, and the roadmap handles two of
them explicitly.

Strict complementarity failing is the weakly active case, which item 0 detects
and reports rather than assuming away. An active set that changes under
perturbation is item 3's conditional-versus-marginal distinction, which is
stated with the matrix.

The other two are assumed. LICQ is what makes the active-set KKT nonsingular,
and it is the precondition for the roadmap's own validation, since matching a
free block against the same model solved with a variable fixed is a
bounds-to-equalities substitution. Second-order sufficiency is what makes the
reduced Hessian positive definite on the free space; when it fails the result
is the indefinite free block item 2 returns with a warning, which is a
diagnostic rather than an error but is not a covariance.

## Scope and compatibility

Items 2 to 4 are additive: `information()` is a new function, `wrt=` (with its
slice and `(Var, time)` block forms) is a new optional keyword, and
`retain_kkt()` is new surface. No signature changes, and v0.9
`covariance(model)` with no `wrt=` reduces onto the declared set, which is
exactly the v0.10 no-argument default, so the v0.9 surface is a
forward-compatible subset.

Items 0 and 1 are the exception. Item 1 changes which variables
`covariance()` projects out, corrects a kept weakly active variable's value
from the factor's `2q` to the true `q`, and projects binding general rows
that v0.9 passes through unprojected, so a model in any of those cases gets
different numbers than v0.9 returns. Item 0 is Rust core work rather than
pyomo surface, because the bound multipliers, `μ` and the Hessian diagonal
are not reachable from the pyomo session: it carries only the primal and the
bounds, and the held factor is barrier-augmented, so `kkt_solve` inverts $W$
and neither `Σ` nor $H$ can be recovered above it. The Rust side retains
everything the classifier needs at the converged iterate, so item 0 is
exposure, not computation.

Until they land, `information()` inherits the shipped slack-only
classification, so items 2 to 4 are complete for interior solutions and misfile
a weakly active bound exactly as `covariance()` does now.

## Validation

- A `μ` sweep over all three regimes, checking together: item 0's ratio goes
  as `O(μ)`, `O(1)` and `O(1/μ)`; the weakly active value holds at `1` and its
  free-block diagonal matches the objective's curvature rather than twice it;
  every other free-block number moves by `O(μ)` and no more; and `Σ_i/|H_ii|`
  stays `O(μ)` on the variables the reduction keeps, since a non-negligible
  ratio there is barrier curvature surviving the projection. Against
  `diagnose_bounds` (`python/notebooks/barrier_curvature_sensitivity.ipynb`
  §5) on the same points, including one it calls `ambiguous`. That routine is
  the reference implementation of the shape, not a primitive to call.
- The classification itself across the `μ` sweep, not just the numbers: record
  every label change. An `ambiguous` variable must settle into a regime as `μ`
  tightens, which is what item 1's re-solve instruction promises, and a
  variable crossing between $F$ and $A$ changes the returned matrix's rank,
  which a caller holding a block across two solves has to see.
- Item 0 under variable scaling: hold the regime and move the objective's
  curvature away from the bound's scale. The weakly active ratio stays at `1`;
  the other two drift toward the band, and at loose `μ` they enter it, which
  must come back as `ambiguous` rather than as a confident wrong regime.
- One-sided finite differences at a transition, against the classification and
  against `covariance()`'s free block. A symmetric difference there returns the
  barrier's smoothing value rather than either true one-sided derivative, so it
  passes while hiding the active-set change; only the one-sided pair shows the
  two disagree. The declaration-triggered pyomo solve sets
  `bound_relax_factor = 0` itself, so the slacks being classified are
  distances to the user's bounds.
- `information()` on a problem where the solve reports non-zero
  `kkt_perturbations`: the inertia-correction guardrail fires, and the returned
  block differs from the unregularized one by the isotropic `δ_w` and nothing
  else.
- `information(...)` against `inv(covariance(...))` to tolerance on a
  well-conditioned block with no active bound and pooled residuals; the
  conditioning advantage on a deliberately ill-identified one. The identity
  holds only there: a bound active makes `covariance()` singular (the pinned
  row is zero) and grouped residuals of unequal variance make the two objects
  different.
- A strongly active fitted variable: the free block matches the same model solved
  with that variable fixed (a bounds-to-equalities substitution, so LICQ is
  assumed), and the pinned variable reports $S_{ii}$ with the activity
  classification. $S_{ii}$ is checkable directly: it is the second derivative
  of the objective minimized over the free variables with that variable held at
  a value and its own bound dropped, and the first derivative of that same
  function is the bound multiplier.
- Two strongly active fitted variables: the off-diagonal $S_{ij}$ matches the
  same construction, and $S_{ii}$ differs from the same variable's value when
  the other is free.
- The row spelling of a limit against its bound spelling: identical
  `covariance()` and `information()` matrices, item 0's agreement
  (jkitchin/pounce#362) carried into item 1's outputs. Then a genuinely
  two-coordinate binding row (`A + B ≤ 1`): rank drops by one, variance
  vanishes along the normal and survives along the difference with
  correlation `-1`, the free block matches the same model solved with the
  row as an equality, and the per-row conditional-information scalar matches
  that model's $S_{ii}$ construction applied to the combination.
- The `μ` sweep is necessary, not sufficient, on its own: the weakly-active
  case is `μ`-invariant and barrier-inflated at once, so it only certifies
  anything alongside item 0's classification. A block conditioned on a strongly
  active variable outside it does move with `μ`, which is correct.
- `s·z` against `μ` on every bounded variable. A mismatch means the point is
  off the central path or the solver relaxed the bound, and the slack being
  classified is not the true slack.
- An indefinite Lagrangian free block is returned as computed, with the
  warning, and is not silently replaced by the Gauss-Newton form.
- The marginal identity: `inv(state block of covariance(wrt={state,
  params}))` against `information(wrt=state)`, both the parameter-marginal
  state information.
- The conditional identity: the state block of `information(wrt={state,
  params})` against `information(wrt=state)` computed with the parameters
  fixed.
- Lagrangian versus Gauss-Newton agree on a linear model and in the
  small-residual limit; the Lagrangian can go indefinite where Gauss-Newton
  stays PSD.

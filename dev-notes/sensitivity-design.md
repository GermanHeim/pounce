# Active-set-aware parametric sensitivity: design notes

This document records the design of pyomo-pounce's active-set-aware
parametric sensitivity as it stands: the three modes of
`sens_solution()`, the report `sens_solution_report()` returns, the
record `sens_active_set_changes()` returns, the degeneracy decision
every mode consumes, and the corrector `corrector_iter` runs. Everything is
computed against the factorization one ordinary solve left behind. The
solver factorizes the KKT matrix to solve the NLP, the session keeps
that factorization, and every question below is a back-solve or a small
augmented solve against it.

User-facing documentation lives in `docs/src/sensitivity.md`, and
[`python/notebooks/36_active_set_parametric_sensitivity.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/36_active_set_parametric_sensitivity.ipynb)
demonstrates the whole surface.

Throughout, $\alpha$ is the fraction of a requested perturbation that
fits before the first bound is reached, and a *crossing* is a bound
whose status the perturbation changes, in either direction. The symbols
of the barrier problem are defined with the sensitivity system below.

## The surface

```python
sens_solution(model, perturb, clamp=True, mode="linear",
              predictor_iter=16, degeneracy="directional",
              degeneracy_iter=None, corrector_iter=0, bound_eps=None,
              max_pdpert=None)
sens_solution_report(model, perturb, max_iter=None,
                     degeneracy="directional", degeneracy_iter=None,
                     corrector_iter=0, mode="linear", predictor_iter=16,
                     bound_eps=None, max_pdpert=None)
sens_active_set_changes(model, perturb, predictor_iter=16,
                        degeneracy="directional", degeneracy_iter=None,
                        max_pdpert=None)
sens_jacobian(of=None, *, wrt, max_pdpert=None)
```

`degeneracy_iter=None` resolves to 16 inside. The sentinel exists so a
budget passed under an option that makes no decision can warn.
`sens_covariance()` and `sens_information()` take `max_pdpert` as well,
and the rest of their surface is recorded in
`covariance-information-design.md`.

`mode` is one ordered argument. Each mode handles everything the one
above it handles and more, at the cost in the last column. Costs are
back-solves against the held factorization, and one full re-solve is the
worst case.

| mode | what it does about a crossing | cost |
|---|---|---|
| `"linear"` | takes the step and clamps, so a crossing coordinate truncates and every other one keeps its predictor value | two back-solves, the step and the barrier correction |
| `"fix_relax"` | pins every crossing at its bound and re-solves, so the others move to stay consistent under the pins | one augmented solve per pass, each growing with the pin count |
| `"path"` | applies the perturbation in stretches, stopping where the active set changes and continuing under the changed set | one back-solve per stretch and one re-factorization per release along the way |

The default is `"linear"`, which is what sIPOPT does by default.

`bound_eps` is how far outside a variable bound a step has to end to
count as having left it, and one value of it decides what `fix_relax`
pins, what the clamp acts on, and what `crossed` reports. `max_pdpert`
raises when the converged factor carries an inertia correction above it,
since every output here inverts that factor.

## The report

`sens_solution_report()` measures the step `sens_solution()` takes for
the same arguments, and carries thirteen fields in two groups.

Eight fields describe the step, and move with `mode`:

- `alpha`, the fraction of the perturbation that fits before the first
  bound is reached, from a ratio test along the step, with `first` and
  `first_kind` naming what is reached there. Under `fix_relax` and
  `path` the step stops at the bound by construction, so `alpha` is 1.0
  and `crossed` is empty for every model. A `bound_eps` wide enough to
  cover the crossing is the exception, since the refinement then pins
  nothing and the step is the linear one.
- `crossed` and `crossed_rows`, the distance by which the step leaves
  each bound it leaves.
- `violation`, the constraint violation evaluated at the predicted
  point.
- `refine_stop`, why the `fix_relax` refinement stopped: `"settled"`,
  `"iteration_limit"`, `"degrees_of_freedom"` or `"worse_than_plain"`.
  A pass pins every crossing it sees, so the pin count tracks how many
  crossings the model has, and this field carries the stop reason.
- `corrector`, described below.

Five more describe the base point and the solve, and do not move with
`mode`. They are `activity` and `row_activity` from the activity
classification, `mu`, `perturbations` (the factor's inertia
corrections), and `bounds_relaxed`.

The report takes `mode` and `predictor_iter` so that it measures the
step the caller takes rather than the linear one. `mu`, `perturbations`
and `bounds_relaxed` are the three things that separate the estimate
from the exact value at the perturbed active set, and a caller comparing
an estimate against a re-solve reads them to tell which one explains the
difference.

## The interior point solution

The notation follows Section 6.3 of Biegler (2010). The solver is
given

```math
\min_x f(x) \quad \text{s.t.} \quad c(x; p) = 0, \quad x \ge 0,
```

where $p$ are the parameters. One lower bound at zero stands for every
bound here, since a general lower bound, an upper bound or an
inequality constraint is a shift of variables or a slack away from it.
It solves a sequence of barrier problems

```math
\min_x \varphi_\mu(x) = f(x) - \mu \sum_i \ln \, x_i
\quad \text{s.t.} \quad c(x; p) = 0,
```

with $\mu$ driven toward zero, and $\mu$ below is the value it finished
on. With $v$ the multipliers of the equality constraints and $u$ the
multipliers of the bounds, the Lagrangian is

```math
L(x, v, u) = f(x) + c(x; p)^\top v - u^\top x,
```

and the primal-dual form of the first-order conditions is

```math
F(w; p) =
\begin{bmatrix}
\nabla_x L \\
c(x; p) \\
X U e - \mu e
\end{bmatrix}
=
\begin{bmatrix}
\nabla f(x) + \nabla c(x; p) v - u \\
c(x; p) \\
X U e - \mu e
\end{bmatrix}
= 0,
\qquad w = (x, v, u),
```

where $X$ and $U$ are the diagonal matrices of $x$ and $u$, $e$ is the
vector of ones, and $\nabla c(x)$ is the matrix whose columns are the
constraint gradients. The three blocks say that the gradient of the
Lagrangian vanishes, that the constraints hold, and that each bound's
variable and multiplier satisfy $x_i u_i = \mu$, which is
complementarity $x_i u_i = 0$ with the barrier parameter in place of
zero. Newton's method on $F$ is what the solver ran. At an iterate
$w^k = (x^k, v^k, u^k)$ the step $d^k = (d_x, d_v, d_u)$ solves the
linearization of $F$ there,

```math
\begin{bmatrix}
W & \nabla c & -I \\
\nabla c^\top & 0 & 0 \\
U & 0 & X
\end{bmatrix}
\begin{bmatrix} d_x \\ d_v \\ d_u \end{bmatrix}
=
-\begin{bmatrix}
\nabla f(x) + \nabla c(x; p) v - u \\
c(x; p) \\
X U e - \mu e
\end{bmatrix},
\qquad W = \nabla^2_{xx} L,
```

everything evaluated at $w^k$, and the next iterate is
$w^{k+1} = w^k + \alpha^k d^k$ with step lengths in $(0, 1]$ from a
line search that keeps $x$ and $u$ positive.

The third block row gives $d_u = \mu X^{-1} e - u - \Sigma d_x$, and
substituting it into the first leaves the KKT matrix

```math
K =
\begin{bmatrix} W + \Sigma & \nabla c \\ \nabla c^\top & 0 \end{bmatrix},
\qquad \Sigma = X^{-1} U,
```

which is the system the solver factorizes. At convergence the session
keeps that factorization, and every question below is a back-solve
against it.

$K$ is symmetric and indefinite, with $n$ positive and $m$ negative
eigenvalues at a solution that satisfies the second-order conditions,
and the factorization is the symmetric indefinite one,
$K = L B L^\top$ after a symmetric permutation of rows and columns
chosen for sparsity and stability, with $L$ unit lower triangular and
$B$ block diagonal with $1 \times 1$ and $2 \times 2$ blocks. Computing
$L$ and $B$ is
the expensive step. Once they exist, solving $K d = r$ for any
right-hand side is two triangular substitutions and a diagonal solve,
which is the back-solve counted throughout this note, and its cost is
proportional to the nonzeros of $L$. The solver factorizes once per
Newton iteration and back-solves for the step. It also reads the inertia
of $K$ off the blocks of $B$, and when the inertia is wrong it adds
corrections $\delta_W$ and $\delta_A$ to the two diagonal blocks until
it is right. Those corrections are the `perturbations` the report
carries. For sensitivity, the factorization the solve finished with is
still valid at the solution, so every mode below answers from back-solves
against it. A new factorization is taken where the active set changes a
diagonal entry of $K$ itself, which a release does. The corrector is
the exception to answering from the held factor. It assembles and
factors its own $K$ at the predicted point, once per correction.

## The sensitivity system

Differentiating $F(w(p); p) = 0$ in $p$ (the implicit function theorem)
and eliminating $\Delta u$ the same way gives the sensitivity system

```math
K \begin{bmatrix} \Delta x \\ \Delta v \end{bmatrix}
= \begin{bmatrix} 0 \\ -\dfrac{\partial c}{\partial p} \Delta p \end{bmatrix},
\qquad \Delta u = -\Sigma \Delta x,
```

so the first-order change in the solution for a change $\Delta p$ in
the parameters is one back-solve against the factor the solve already
holds. That is the whole of sIPOPT's basic step, and it is what
`sens_jacobian()` and `mode="linear"` compute. Below, $\Delta w$ is the
solution $(\Delta x, \Delta v)$ of that system, and $r$ its right-hand
side.

The factorization is held at the final $\mu$, so that step predicts
where the barrier problem's solution moves, not where the original
problem's does. One more back-solve, with $\mu$ placed in the
complementarity rows of the right-hand side, carries the step toward
the $\mu = 0$ solution, eq. 11 of Pirnay et al. (2012). Every mode
adds that correction, which is why the linear step's price above is
two back-solves rather than one.

How $p$ enters is arranged so the right-hand side is simple. A declared
Param enters the model through one defining equality, a single variable
equal to the Param, written by the modeler or created once at
declaration for a Param that lacks the form. Perturbing the Param
shifts that row's right-hand side, so $\partial c / \partial p$ is a
signed unit vector in that row and $r$ is zero everywhere except there,
where it carries the shift.

$\Sigma$ is what makes the active set matter. With $\Sigma_i = u_i / x_i$
and $x_i u_i = \mu$:

- At a tightly active bound $x_i \to 0$ with $u_i > 0$, so
  $\Sigma_i = u_i^2 / \mu$ is enormous and the step leaves that
  coordinate where it is for any perturbation.
- At an inactive bound $u_i = \mu / x_i$ is tiny, so $\Sigma_i$ is
  negligible and the coordinate moves freely, bound or no bound.

So the linear step is exact while the active set does not change, and
cannot represent the change when it does. There are two ways it can be
wrong, and the rest of the design is about them:

- A coordinate that was free is carried past its bound, which shows up
  as a value outside the bound.
- A bound that was active should let go, which shows up as its
  multiplier $u_i$ driven negative.

## Linear: the whole step, then a clamp

`mode="linear"` is the step above.

- Solve $K \Delta w = r$ once. Add $\Delta x$ to the base point.
- Put any coordinate that ends outside a bound back on the bound.
  Every other coordinate keeps the value the step gave it.
- Warn when anything was clamped, since a coordinate outside its bound
  means the active set changed under a step that assumed it did not.

The clamp truncates one coordinate and leaves the rest at full length,
so the result is feasible but no longer solves the KKT system. It is
the default because it is what sIPOPT does, and it is the right answer
whenever nothing crosses.

Three details apply in every mode.

- `clamp` applies in all three modes, to whatever is still outside a
  bound at the end. Under `fix_relax` and `path` that is usually
  nothing, and when it is not, the warning names the stopping condition
  that left it. `clamp=False` returns the raw step.
- "Outside" is an absolute comparison against
  `max(|bound_relax_factor|, 1e-9)`, the same test the refinement uses,
  so the clamp and the pins agree on a coordinate of any magnitude.
- A bound written in terms of a declared Param is a constraint by solve
  time, so it moves with the perturbation and the linear step follows
  it to first order.

## Fix-relax: repair the active set, then re-solve

`mode="fix_relax"` takes the linear step, looks at what it broke, and
re-solves with the active set repaired, against the same factor.

The two repairs are two different linear-algebra operations.

**A pin** holds a coordinate at its bound. That is one extra row
$(\Delta x)_i = b_i - x_i$ added to the system. With $k$ pins collected
into the selector $P$, the augmented system is solved through the Schur
complement over the added rows,

```math
S = P K^{-1} P^\top, \qquad
S \lambda = P K^{-1} r - b, \qquad
\Delta w = K^{-1} (r - P^\top \lambda),
```

which is $k + 1$ back-solves against the held factor and one dense
$k \times k$ solve. The factorization is never rebuilt for a pin.

**A release** lets an active bound go. That removes its $\Sigma_i$ from
the diagonal and holds its multiplier at zero. Removing a term from the
diagonal is a change to $K$, so a release re-factorizes.

The loop runs as follows.

1. Take the linear step.
2. Collect every coordinate outside a bound by more than the margin,
   and every bound multiplier the step drives negative past the solve's
   own margin.
3. Pin the first group, release the second, re-solve. Pins survive a
   release, with their right-hand sides re-measured against the
   re-solved base.
4. Repeat from 2 until the list is empty.

It stops, and `refine_stop` says which, when the list is empty
(`settled`), at `predictor_iter` passes (`iteration_limit`), when the
pins have used up the problem's degrees of freedom and the augmented
system cannot be solved (`degrees_of_freedom`), or when the refined
step ends further outside the bounds than the linear step, in which
case the linear step is returned (`worse_than_plain`).

Four decisions in that loop are not the obvious ones:

- **The whole list per pass, not the worst crossing.** One crossing per
  pass made `predictor_iter` decide where the loop stopped and grew
  quadratically with the pin count. The whole list settles in a handful
  of passes, which is why `predictor_iter` is a safety limit rather
  than a budget.
- **A release batch is checked, a pin batch checks itself.** Too many
  pins come back as a system that cannot be solved. Too many releases
  solve anyway and carry variables off bounds they were sitting on, so
  a release batch is kept only when its step is no further outside the
  bounds than the one in hand, and otherwise the most negative
  multiplier goes alone.
- **A release re-factorizes rather than becoming a Schur row.** On a
  tightly converged bound $\Sigma_i$ is so large that computing the
  released step from the held factor loses digits in proportion to
  $\epsilon \Sigma_i$, and the loss grows as the solve converges. One
  re-factorization costs an order of magnitude less than a re-solve.
- **The release threshold is not the caller's margin.** `bound_eps`
  says what counts as on the bound, which says nothing about whether a
  multiplier has changed sign, so the release test keeps the solve's
  own margin.

## Path: each change applied where it happens

`mode="path"` treats the perturbation as a line from the base point to
the target and follows it, changing the active set at each point along
the line where it changes.

With the active set fixed, the solution is linear in the fraction
$t \in [0, 1]$ of the perturbation taken, $w(t) = w_0 + t \Delta w$.
It stops being linear at the first $t$ where a coordinate reaches a
bound or a multiplier reaches zero. Past that $t$ the active set is
different, $K$ is different, and the direction is different. The
solution along the whole line is therefore piecewise linear, with a
breakpoint at every active-set change.

The loop runs as follows.

1. From the current base, compute the direction $\Delta w$ under the
   current active set.
2. Find the smallest $t$ at which the direction carries a coordinate to
   a bound (a ratio test) or a multiplier to zero. If none is below the
   remaining fraction, take the rest of the perturbation and stop.
3. Advance to that $t$, apply the change (pin or release), and go to 1.

`predictor_iter` caps the number of changes. Past the cap the rest of
the perturbation is taken in one step under the active set reached,
which degrades smoothly, since $t$ advances toward 1 either way.

`fix_relax` decides every change from the step at the base point, and
`path` decides each one where it happens. They agree when nothing
crosses, and when the base point's direction stays right across the
whole change. They differ when bounds enter and leave the active set
repeatedly along the way, which a large perturbation does not produce
by itself.

`sens_active_set_changes()` returns the record `path` builds on the way:
which bound, which direction, at what fraction. It is the way to see
whether a given perturbation has any breakpoints at all, and the first
fraction says how much of the perturbation leaves the held solve's
active set unchanged.

## Degeneracy: decided for the perturbation's own direction

Every mode above assumes each bound is either clearly active or clearly
inactive at the base point. A bound can be neither. At a weakly active
bound the slack and the multiplier vanish together, the slack of order
$\sqrt{\mu / H}$ and the multiplier of order $\sqrt{\mu H}$, with $H$
the curvature reduced along that coordinate, and strict complementarity
fails. This is Biegler's weakly
active set (Section 4.3, at Definition 4.6), a bound that is active with
a zero multiplier, as the barrier solution sees it. The solution is then
not differentiable in $p$. It has two one-sided derivatives, one for each
direction the perturbation can push that bound, and the linear step
picks neither reliably. $\Sigma$ equals the reduced curvature $H$
there, the barrier holding the bound with the same stiffness the
problem carries along it, so what fails is the magnitude of the step,
not a coordinate being stuck.

The directional derivative still exists, and it is the solution of a
quadratic program in the direction, eq. 14 of Pirnay et al. (2012),
which decides for each weakly active bound whether it stays active or
releases for this direction. `degeneracy="directional"`, the default,
solves that QP on the held factor:

1. Release every weakly active bound at once, removing its $\Sigma$,
   in one factorization that serves the whole decision.
2. Compute the step $\Delta w_0$ of that released system.
3. The weak bounds that $\Delta x_0$ moves into violation are the ones
   that matter for this direction. Each is decided only if its bound is
   at a kink, which its own diagonal of $S$ measures. $\kappa_k =
   \Sigma_k S_{kk}$ is exactly 1 at a kink and falls as the squared
   ratio of the kink's barrier width $\sqrt{\mu / H}$ to the row's
   slack, so a row far below one is dropped and its plain movement
   stands. A coordinate an equality pins is the limiting case,
   $S_{kk}$ exactly zero. For the rows that remain, solve the small QP
   below in the multipliers $\lambda$ of the pin rows, whose
   optimality conditions are eq. 14's complementarity. Each of those
   bounds either holds with a nonnegative multiplier or releases and
   moves feasibly.
4. Check the decided direction against every weak bound. If a new one
   is violated, add it to that set and repeat 3.

The QP in step 3 is, with $a_k$ the signed unit vector of bound $k$,
$S_{jk} = a_j^\top K_{\text{rel}}^{-1} a_k$ and
$\beta_k = a_k^\top \Delta x_0$,

```math
\min_\lambda \tfrac{1}{2} \lambda^\top S \lambda + \beta^\top \lambda
\quad \text{s.t.} \quad \lambda \ge 0.
```

The drop in step 3 exists because the weak set is deliberately wider
than the kinks. Its membership comes from the activity classifier,
whose ambiguous class widens as curvature falls, so on a model with
low curvature along some coordinates it holds rows a large fraction of
their range from any bound. The QP treats a decided row as at its bound, and the error
of that model is the row's slack, so deciding such a row is wrong at
first order, while dropping a row near the threshold costs at most its
own slack, of order $\sqrt{\mu}$. $\kappa$ costs no extra work, since
the column that builds row $k$ of $S$ already carries $S_{kk}$.

`degeneracy_iter` budgets the back-solves this spends, and a budget it
cannot fit falls back to the one-sided step with a warning.
`degeneracy="one_sided"` turns the decision off and takes the
single-sided value. The decision is handed to whichever mode is running.

`degeneracy="release_all"` stops after step 2 and returns the
all-released direction with nothing decided, at one back-solve and no
QP. A weak bound the perturbation holds then shows up downstream. The
unconstrained minimum along such a coordinate sits on the infeasible
side, so the released step drives it through its bound, and the
violation is visible to whatever runs next. `fix_relax` pins it and
re-solves, so coupled neighbors are repaired too. `linear` clamps the
crossing coordinate and leaves its neighbors carrying the released
coupling. `path` is measured not to deliver the walk here. For a weak
bound the perturbation holds it applies no active-set change, the
clamp is what returns the crossing coordinate, and coupled neighbors
keep the one-sided value, under `"one_sided"` as well as here. A
correction acts on whatever the mode left. The releasing side needs
no repair, since the released direction is that side's exact
derivative. The cost is deterministic and independent of
`degeneracy_iter`, which makes this the treatment for a base point
whose weak set is too large for the decision's budget, where
`"directional"` pays the failed attempt and falls back to one-sided
anyway. The one-sided step hides both kink errors as
feasible partial movements a bound check cannot see, so releasing
converts them into either the exact answer or a repairable
violation. At an exact kink under `mode="linear"`, the holding side
is the released value until the clamp truncates it, where
`"directional"` decides it correctly, so the decision remains the
default.

`sens_solution()` names a perturbation and the QP takes $\Delta p$ as
input, so it returns the directional value for the direction asked and
needs no new argument. A weakly active bound is still active for the
first stretch of the perturbation, so `path` releases it where its
multiplier reaches zero rather than at the start, and releasing at the
start overshot tenfold.

`sens_jacobian()` has no direction, and $dx / dp$ has two values at a
kink. It returns a float, warns that the base point is degenerate and
the value one-sided, and leaves a caller who needs the other side to
ask through `sens_solution()`.

## The corrector

Every mode returns a first-order step, so at the perturbed parameters
the primal-dual equations are not quite satisfied.
`corrector_iter`, zero by default, runs that many iterations of the
solver's own Newton step at the perturbed parameters, against $K$
assembled and factored at the predicted point. The current iterate is
swapped to the stepped point for the correction, so the Hessian, the
constraint Jacobians, the bound quantities, and the barrier diagonal
all live in that one frame. On an exact-Hessian solve the Hessian is
re-evaluated there with the step's own clamped multipliers. A
`limited-memory` solve keeps its quasi-Newton matrix, since no exact
Hessian exists to evaluate anywhere. The diagonal is
$\Sigma_i = u_i / x_i$ from the predicted slacks and those clamped
multipliers, with two rules applied in the same frame. Each entry is
capped where eliminating through it would leave the variable's
constraint rows holding $a^2 / \Sigma_i$ below their own roundoff,
the coefficients $a$ read at the predicted point. And when the solve
relaxed the variable bounds internally and moved its answer back onto
the bounds the model declares, the slacks are measured against those
declared bounds, which follow the iterate because they are
constants.

```math
K \begin{bmatrix} d_x \\ d_v \end{bmatrix}
= -\begin{bmatrix} \nabla \varphi_\mu(x) + \nabla c(x; p + \Delta p) v \\
                   c(x; p + \Delta p) \end{bmatrix},
\qquad d_u = \mu X^{-1} e - u - \Sigma d_x,
```

so every correction costs one derivative evaluation and one
factorization, and each iteration after that costs one back-solve.
This is Newton's method with a frozen Jacobian, which converges
linearly at a rate set by the distance between $K$ and the KKT matrix
at the answer. The answer lies near the predicted point, so a $K$
assembled there is close to it, while one assembled at the base point
is off by however much the Hessian changes over the step.

The active set is the predictor's, applied to $K$ once before
iterating, because the iterations cannot change it afterwards. Every
direction they produce is shaped by the diagonal $K$ carries.

- A released bound comes out of $K$, with its multiplier held at zero
  and its complementarity row removed. At a bound the solve held
  tightly, complementarity gives $u_i x_i = \mu$, so
  $\Sigma_i = u_i / x_i = u_i^2 / \mu$, and a direction divided by
  that entry cannot move the coordinate off the bound in any number
  of iterations.
- A bound the step brought in enters $K$, with its diagonal raised to
  what the barrier assigns at the predicted slack, under the same
  cap, so the iterations hold the coordinate where the step put it.
- Every other row carries the predicted point's own term.

A release and a new pin are both a change to one diagonal entry, so
one factorization serves every iteration. `fix_relax` and `path` hand
over their active set, and `linear` hands over whatever the clamp left
on a bound. A release no step endpoint shows is applied by no mode.
The step's clamped multiplier leaves a weak diagonal entry at that
bound, the iterations can carry the variable partway off it, by a
delta-dependent margin and never to the re-solve, and the residual
they measure genuinely falls, so the warning below stays quiet. The
modes that decide the release cross exactly.

What to expect from it:

- It converges to the barrier solution at the $\mu$ the solve finished
  on, so that offset, $O(\mu)$ by Biegler's Theorem 6.7, is as close as
  it gets.
- The multipliers stay as the predictor extrapolated them over the whole
  perturbation, and once the perturbation is large that is the dominant
  error.
- How far it gets depends on how many crossings the predictor handed
  over rather than on the size of the perturbation, and
  `sens_solution()` warns when the correction ends without at least
  halving the residual.

The `corrector` block of the report carries:

- `stationarity`, `feasibility` and `complementarity`, the residual
  split by optimality condition, the three terms of Biegler's error
  measure. The last two say whether the values can be acted on,
  and the first can stay large while both are small.
- `released`, `pinned` and `active_set_changes`, what the predictor
  handed over and their total.
- `initial_residual`, measured where the iterations start, after the
  active set is applied and any coordinate outside a bound is put back
  inside.
- `iterations`, `residual` and `converged`, what the loop spent, where
  the residual ended, and whether the loop stopped because an
  iteration failed to improve rather than because the budget ran out.

## References

- Biegler, *Nonlinear Programming: Concepts, Algorithms, and
  Applications to Chemical Processes*, MOS-SIAM Series on Optimization,
  SIAM, 2010. [DOI](https://doi.org/10.1137/1.9780898719383).
  Sections 4.3, 5.2 and 6.3.
- Pirnay, López-Negrete, Biegler, *Optimal sensitivity based on IPOPT*,
  Math. Program. Comput. 4 (2012) 307–331.
  [DOI](https://doi.org/10.1007/s12532-012-0043-2)
- Fiacco, *Introduction to Sensitivity and Stability Analysis in Nonlinear
  Programming*, Academic Press, 1983.

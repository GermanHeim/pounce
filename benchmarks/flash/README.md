# Gate 1 flash fixture (`benchmarks/flash/`)

The phase-changing fixture
[gh#776](https://github.com/jkitchin/pounce/issues/776) gates its tray
and dynamic-column work behind, and the successor to the Gate 0 harness
in [`benchmarks/mpcc/`](../mpcc/README.md).

Its job is narrow, and worth stating first because the temptation is to
widen it: **solve one flash across a phase boundary, on the route Gate 0
established, and check the answer against a calculation that does not
share the first one's reasoning.** Gate 0's last acceptance criterion
was that no flash, tray or column work began from it. This is that work,
and it stops at one flash. Nothing here is a tray, and nothing here
should grow into one.

## Where things live

**The model is not in this directory.** The flash, its complementarity
pairs, the lowerings and the independent oracle are
`pounce.examples.flash_mpcc`, so that the wheel ships them and the
tutorial can import them the way the other three application notebooks
import theirs. This directory is the *evidence apparatus* around that
model — routes, the measurement protocol, the traversal, provenance and
the report — and keeping the two apart is what stops the harness from
becoming a second, subtly different copy of the physics, which is the
failure mode the cross-check exists to catch.

The narrative walk-through, with figures, is
[`python/notebooks/38_phase_changing_flash_mpcc.ipynb`](../../python/notebooks/38_phase_changing_flash_mpcc.ipynb);
the user-facing page is [`docs/src/flash-mpcc.md`](../../docs/src/flash-mpcc.md).

## Running it

Needs the built Python extension (`make python-ext` from the repo root),
SciPy and JAX. From `benchmarks/`:

```sh
python -m flash.selftest      # no solver needed; must pass before anything else
python -m flash.run --smoke   # deterministic asserted subset (~15 s)
python -m flash.run --full    # every leg and route; writes the report
```

or, from the repo root, `make -C benchmarks flash-selftest / flash-smoke
/ flash-run`.

Results land in `results-full.{json,md}` beside the harness. Both are
regenerated per run and gitignored; `schema.json` is tracked. The smoke
subset also runs in CI as `python/tests/test_flash_mpcc.py`, which is
the "fast regression" gh#776 asks Gate 1 to become.

## The model

Ethane / n-butane, equimolar feed, 10 bar, Peng--Robinson with classical
one-fluid mixing and `k_ij = 0`. Constants are the ethane and n-butane
rows of `phase_envelope.NATURAL_GAS`, so the numbers have one home in
this repository rather than two.

Unknowns are the vapor fraction and the two phases' mole numbers per
unit feed — five variables for two components:

```
z_i = (1 - beta) x_i + beta y_i                          balance
ln x_i + ln phi_i^L(x/Sx) = ln y_i + ln phi_i^V(y/Sy)    isofugacity

0 <= beta      _|_  1 - Sy >= 0                          pair V
0 <= 1 - beta  _|_  1 - Sx >= 0                          pair L
```

The pairs are **phase amount against stability slack**, which is the
guardrail gh#776 states — *not* `L ⟂ V`, which would say the two phases
cannot coexist. At `beta = 0` the balance forces `x = z`, the
isofugacity rows collapse to Michelsen's tangent-plane stationarity for
the trial vapor, and `Sy <= 1` is exactly `TPD >= 0`. The
complementarity is not an encoding trick: it *is* the stability test.
`pounce.examples.flash_mpcc` derives this at length, including where the
normalization goes
and what happens when it goes in the wrong place.

The two regime switches are the two biactive points: at the bubble point
`beta = 0` and `Sy = 1` together, at the dew point `beta = 1` and
`Sx = 1`. The degeneracy is the physics, not an artifact of writing it
this way.

## What is checked, and against what

The oracle in `pounce.examples.flash_mpcc` is a second calculation of the
same flash: Michelsen
tangent-plane stability with a multistart, per phase label, then
Rachford--Rice with a Newton polish. It shares exactly one thing with
the model — `phase_envelope.log_fugacity_coefficients` and the cubic
under it — and differs in everything above: phase detection, the
two-phase solve, and the strength of the stability claim. The oracle's
is the stronger one (every stationary point its multistart reaches,
against the model's one), which is the right way round: agreement means
the solver found the stationary point that matters.

`selftest.py` runs the checks that need no solver, and the load-bearing
one is that **the oracle's answer satisfies the model's own rows**. That
is what makes this a cross-check rather than two opinions, and it earned
its place immediately: the first implementation normalized inside the
logarithm as well as inside `phi`, which adds `ln(Sy/Sx)` to every
isofugacity row. That term **vanishes identically in the two-phase
region**, so the model solved and agreed with the oracle at every
two-phase temperature, and was wrong only in the single-phase regimes —
the ones the fixture exists to reach. A corpus that stopped at the
two-phase region would have shipped it.

## Results

The path is 34 temperatures from 230 to 360 K, crossing liquid →
two-phase → vapor with both switches interior (bubble 268.8896 K, dew
323.3849 K, located by bisecting the stability boundary). On
`scholtes_then_ncp`:

- all 34 points solve, in all four legs (up/down × cold/warm) — with
  the qualification in the next section about *which half* of the route
  actually ran;
- every regime label matches the oracle;
- `beta` and both phase sums agree to ~1e-11, complementarity products
  land at 1e-12 to 1e-18;
- the cold legs agree exactly with each other, and no leg is
  path-dependent: no hysteresis, no branch artifact.

## The route comparison, and the finding in it

Every route runs on the ascending cold leg. On this fixture:

| route | solved | all source checks pass | worst `|beta - beta_oracle|` |
|---|---:|---:|---:|
| `scholtes_then_ncp` (supported) | 34/34 | 34/34 | 8.4e-10 |
| `scholtes_warm_full` | 34/34 | 34/34 | 8.4e-10 |
| `direct` | 34/34 | 34/34 | **1.0e-13** |
| `ncp_eq` | **0/34** | 0/34 | - |
| `ncp_eq_l1` | 34/34 | 33/34 | 1.3e-06 |
| `ncp_eq_l1_fallback` | 34/34 | 33/34 | 1.3e-06 |

**Gate 0's supported route runs only its continuation half here, at
every single temperature.** `scholtes_then_ncp` is defined as a Scholtes
continuation followed by one exact-product NCP-equality solve seeded
from it — and that finishing solve is rejected structurally, with
`Not_Enough_Degrees_Of_Freedom` after zero iterations, at all 34 points.
The reason is not numerical: a square flash has no objective and no
slack, so making both product rows equalities gives **six equality rows
against five variables**. Gate 0's corpus had an objective and free
variables and never met this; every equilibrium-stage model does, by
construction.

The identical numbers for `scholtes_then_ncp` and `scholtes_warm_full`
in the table above are that finding restated: on this fixture they are
the same route. The answers are still right — the continuation reaches
`tau = 1e-8` and the source complementarity products land at 1e-12,
four orders inside the `sqrt(tol)` floor — but the half of the route
that exists to *guarantee* MPCC-feasibility never ran, and a report that
said "the supported route works on a phase-change model" without this
paragraph would be false in the way that matters.

`ncp_eq` fails for exactly the same structural reason, which is why it
is 0/34 rather than 34/34 with bad answers: the model is rejected before
the first iteration.

**What to use instead, measured:** `direct` — the `G*H <= 0` lowering,
which adds inequalities rather than equalities and so does not
over-determine the square system. It solves all 34 points on its own,
with the best agreement in the table (1e-13, three orders better than
the continuation routes) and without a continuation at all. That is a
concrete recommendation for the next model up rather than a preference:
on a square complementarity flash, the direct lowering is the one that
fits.

The single `ncp_eq_l1` miss is at 269.0 K, the path point nearest the
bubble at 268.89 K: `beta` off by 1.3e-6 against a 1e-6 threshold, with
the source complementarity at 2.8e-9 and status `Solve_Succeeded`. A
marginal accuracy shortfall at the hardest point on the path, not a
wrong phase state.

## Two more findings worth carrying forward

**Reverse-mode autodiff loses this model's Hessian at the bubble
point.** `jax.hessian` is `jacfwd(jacrev(.))`, and under `jax.jit` it is
wrong by O(20) at exactly the two path points straddling the bubble
temperature, and nowhere else — where the equation of state has a double
root and Cardano's trigonometric branch runs `arccos` into its endpoint
singularity. `jacfwd(jacfwd(.))` is exact there and costs the same.
The Jacobian is unaffected in every mode, so gradients, KKT residuals
and *converged answers* were all right: the full traversal agreed with
the oracle to 1e-11 at all 34 temperatures **with the wrong Hessian**.
It costs iterations and robustness at the one place the fixture exists
to test, and nothing reports it. The measurement table is in
`pounce.examples.flash_mpcc`.

**A present phase must sit at its lower-Gibbs cubic root; an incipient
one need not.** The first draft asserted the root guard of both phases
and failed at five temperatures. It is not a defect: Michelsen's trial
phase probes the tangent plane at the feed rather than its own
stability, and the incipient vapor takes the metastable root below 240 K
and the incipient liquid above 340 K. In the two-phase region, where
both phases are real, both pass at every point — which is a check on the
model, not an excuse. `validate.py` judges the present phases and
records the incipient one.

## What this fixture is not evidence about

- **More than one flash.** Two components, five variables, one
  equilibrium stage, one operating parameter. Nothing measured here
  bounds a tray, a column, or a dynamic transcription. gh#776 gates
  those on this result; this result does not reach them.
- **The DiscOpt half of Gate 1.** The reduced GDP/SOS1 regime
  cross-validation gh#776 asks for is blocked on
  [jkitchin/discopt#1123](https://github.com/jkitchin/discopt/issues/1123)
  (first-class complementarity provenance, and the local-versus-certified
  result contract). Every result file records that it was not run, and
  why, rather than omitting the field.
- **Supercritical mixture states.** The path stays far from the mixture
  critical point. Ethane is above its own `Tc` over the top third of it,
  which is ordinary and is not the same thing; the record carries the
  pure-component reduced temperatures so the two cannot be confused.
- **Trivial-solution or no-incipient-phase conditions.** Both are
  guarded and neither occurs on this path — `selftest` asserts it. The
  second matters structurally: with no incipient phase for either label,
  `x = y = z` is the only solution of the isofugacity rows, both pair
  slacks vanish, and `beta` is left undetermined — a one-parameter
  family rather than a point. The pinned path stays out of that region;
  a path that did not would be checking a family against a point.
- **Nonlinear complementarity functions.** The pairs are affine in the
  variables here, as they were in Gate 0. The nonlinearity in this
  fixture is in the ordinary rows.

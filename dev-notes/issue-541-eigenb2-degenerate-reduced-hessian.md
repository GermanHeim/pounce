# issue #541 — `eigenb2`: a degenerate reduced Hessian, and an inertia test that stops meaning anything

[#541](https://github.com/jkitchin/pounce/issues/541): `eigenb2` (Vanderbei)
exits `Solved To Acceptable Level` after 67 iterations where Ipopt certifies
`Optimal` in 21. The issue localises the divergence to iteration 3, where Ipopt
applies `delta_w = 10^2.9 ≈ 794` and POUNCE applies none, and proposes that both
this and the sister issue [#540](https://github.com/jkitchin/pounce/issues/540)
(`eigena2`) trace to one root cause in the inertia-correction update.

**Outcome: iteration 3 is not the cause of the 67 iterations, and the `delta_w`
update rule is not the bug — but the sister-issue conjecture in #541 was right,
and the shared root cause is real.** POUNCE's inertia at iteration 3 is
*correct* — verified against a dense eigendecomposition of the dumped KKT
matrix — and `PdPerturbationHandler` is a faithful line-by-line port of upstream
3.14. The run is slow because `eigenb2` is a **degenerate NLP**: the reduced
Hessian `Zᵀ W Z` has a smallest eigenvalue that collapses to zero, and the
Newton step blows up along a direction of numerically-zero curvature.

> **Superseded in part by [#544](https://github.com/jkitchin/pounce/pull/544).**
> This note originally concluded that the inertia test stays satisfied and
> `delta_w` stays zero for the whole run. That is wrong, and §2 below says why
> the sample that suggested it does not reach the part of the run that matters.
> On `270a0502` the failing tail is *full* of inertia activity: 20
> factorizations mismatch, with FERAL reporting 43…64 against an expected 55,
> and 11 iterations carry a nonzero `lg(rg)` — re-escalating 1.9, 1.4, 0.9, 0.5,
> 1.8, 1.3, 1.7 over iterations 61-67, which is the same signature #540 reported
> for `eigena2`. The KKT is singular to working precision there, so its
> negative-eigenvalue count is noise and `delta_w` is the wrong answer to it.
> #544 routes an unmeasurable inertia test to `delta_c` instead; its trigger
> fires **15 times on `eigenb2`**, more than the 5 on `eigena2`, the model it was
> written for. `eigenb2` now certifies `Optimal` in 68 iterations under stock
> defaults. §5's reading — that Ipopt escapes only because MA57's inertia is
> thresholded in practice — is exactly what #544 implements, and was reached
> here independently from a different model.

Everything below is measured on `270a0502` + the committed fixture
`crates/pounce-cli/tests/fixtures/eigenb2.nl`
(`sha256 4664e87e…9e1a37d3`, the reporter's own file), FERAL backend, release
build, `OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1`.

---

## The problem is bound-free and equality-only

```
Total number of variables............................:      110
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        0
                     variables with only upper bounds:        0
Total number of equality constraints.................:       55
Total number of inequality constraints...............:        0
```

No bounds, no inequalities. The barrier term is identically zero, `s`, `z_L`,
`z_U`, `v_L`, `v_U` and both `sigma` blocks are empty, and `alpha_du` is `1` on
every line. The algorithm is a filter line-search **Newton–SQP** on

    min f(x)   s.t.   c(x) = 0,     n = 110, m = 55

with a 165×165 KKT matrix `[[W, Jᵀ], [J, 0]]` and `num_neg_evals = 55`. The
constraints are quadratic, so `c(x + d) = c(x) + J d + ½ Q(d)` holds *exactly*.
That last fact matters in §4.

## 1. The reproduction is exact

```
iter      objective   inf_pr   inf_du lg(mu)    ||d|| lg(rg) alpha_du alpha_pr  ls
   1  5.9533607e+02 2.26e+00 2.37e+02   -1.0 5.56e-01    2.0 1.00e+00 1.00e+00f  1
   2  2.3547661e+02 3.97e-01 4.61e+02   -1.0 4.10e-01    3.3 1.00e+00 1.00e+00f  1
   3  1.5991320e+01 3.56e-02 9.09e+01   -1.0 3.28e-01      - 1.00e+00 1.00e+00f  1
```

Byte-identical to the issue, including `lg(rg) = -` at iteration 3 and the
67-iteration `Solved To Acceptable Level` exit.

## 2. POUNCE's inertia is the correct one — over the first five iterations

`--dump kkt:all` writes each factorization's triplets. Feeding them to a dense
`numpy.linalg.eigvalsh` gives the exact inertia of the matrix POUNCE actually
factored:

| dump | δ_x | FERAL `neg` | status | true (pos, neg, zero) |
|---|---|---|---|---|
| iter 0, factor 1 | 0 | 64 | WrongInertia | (101, 64, 0) |
| iter 0, factor 5 | 1e+2 | 55 | Success | (110, 55, 0) |
| iter 1, factor 1 | 0 | 64 | WrongInertia | (101, 64, 0) |
| iter 1, factor 4 | 2133 | 55 | Success | (110, 55, 0) |
| **iter 2, factor 1** | **0** | **55** | **Success** | **(110, 55, 0)** |
| iter 3, factor 1 | 0 | 55 | Success | (110, 55, 0) |
| iter 4, factor 1 | 0 | 55 | Success | (110, 55, 0) |

FERAL's count matches the exact spectrum on every factorization **in this
sample**.

> **Do not generalize this table to the run.** It covers iterations 0-4, where
> the KKT is still well conditioned and the count is a meaningful quantity. The
> stall lives in the tail, and there the matrix is singular to working precision
> and the count is noise — 20 mismatches across the run, FERAL reporting
> 43…64 against an expected 55. Reading "correct on every factorization" off
> these seven rows is what led the original version of this note to conclude
> that `delta_w` stayed zero throughout; it does not. See the banner at the top,
> and #544.

The escalation ladders match Ipopt's printed `lg(rg)` exactly where they overlap:
iteration 0 runs `0 → 1e-4 → 1e-2 → 1 → 100` (`lg(rg) = 2.0`), iteration 1 runs
`0 → 100/3 → 267 → 2133` (`lg(rg) = 3.3`). Ipopt's `2.9` at iteration 3 is
`2133/3 = 711`, i.e. exactly one `get_deltas_for_wrong_inertia` step from the
same `delta_x_last` — so Ipopt tested `δ_x = 0`, was told the inertia was wrong,
and escalated once.

At that iterate the smallest reduced-Hessian eigenvalue is **+135.5** against
`‖W‖ = 1117` and `κ(K) = 9.9e6`. The matrix is comfortably of correct inertia;
POUNCE is right and Ipopt/MA57 over-regularizes there. `IpPDPerturbationHandler.cpp`
(3.14) was diffed against `crates/pounce-common/src/pd_perturbation.rs`
statement by statement — `ConsiderNewSystem`, `PerturbForSingularity`,
`PerturbForWrongInertia`, `get_deltas_for_wrong_inertia` and `finalize_test` all
agree, including the detail that `finalize_test()` at the top of
`PerturbForWrongInertia` marks both flags `NOT_DEGENERATE` on the first
wrong-inertia event, which makes `hess_degenerate_ == DEGENERATE` effectively
unreachable in both codes. Why MA57 reports wrong inertia on that matrix is not
resolved here; §3 shows it does not matter.

## 3. Iteration 3 is not what costs the 40 extra iterations

A scratch hook forcing `WrongInertia` on the first factorization of iteration 2
reproduces Ipopt's choice (`δ_x = 711` at printed line 3):

| run | iterations | exit |
|---|---|---|
| default | 67 | Solved To Acceptable Level |
| forced `δ_w` at iteration 3 | **58** | Optimal Solution Found |

Optimality is recovered, but 58 ≫ 21 and the same stall is still there
(iterations 35–48 accept `alpha_pr` of 1/32–1/64 after 6–7 trials). Iteration 3
is a symptom of the same underlying degeneracy, not the cause of the run length.

## 4. The actual mechanism: a reduced Hessian that collapses to singular

Splitting each dumped KKT into `W = A[:110,:110]`, `J = A[110:,:110]`, taking an
orthonormal null basis `Z` of `J` from an SVD, and diagonalising `Zᵀ W Z`:

| iter | λ_min(ZᵀWZ) | λ_max | ‖W‖ | κ(K) | FERAL min pivot (scaled) |
|---|---|---|---|---|---|
| 2 | 1.355e+02 | 1.11e+03 | 1.12e+03 | 9.9e6 | 3.1e-1 |
| 4 | 2.587e+00 | 2.86e+02 | 2.87e+02 | 1.0e7 | 1.2e-1 |
| 6 | 1.737e-03 | 1.27e+02 | 1.28e+02 | 1.9e7 | 1.4e-3 |
| 8 | 2.714e-05 | 1.26e+02 | 1.26e+02 | 7.4e7 | 7.4e-5 |
| 10 | 1.413e-06 | 1.28e+02 | 1.28e+02 | 2.0e8 | 4.5e-6 |
| 13 | 4.227e-07 | 1.30e+02 | 1.30e+02 | 3.1e8 | 1.9e-6 |
| 19 | 1.452e-07 | 1.32e+02 | 1.33e+02 | 9.1e8 | 5.3e-7 |
| 25 | 1.231e-09 | 1.42e+02 | 1.43e+02 | 1.2e11 | ~1e-8 |
| 36 | 1.430e-11 | 1.60e+02 | 1.60e+02 | 3.8e14 | — |
| 60 | 7.236e-13 | 1.60e+02 | 1.60e+02 | 7.6e17 | — |

`λ_min` stays **positive** all the way down — the number of negative eigenvalues
is exactly 55 at every one of those iterations — so the inertia test is
satisfied and `δ_w = 0` for the whole run. But relative to `‖W‖` the smallest
curvature falls from 1e-1 to 1e-13. `eigenb2` has no strict second-order
sufficiency: it is an eigenvalue-decomposition model, and its solution set is a
manifold (rotation within a degenerate eigenspace), so `ZᵀWZ` is *singular at
the solution*.

Decomposing the computed step `dx` into the null space of `J` (tangential) and
its complement (normal), and projecting the tangential part onto the
reduced-Hessian eigenvectors:

| iter | ‖dx‖ | ‖tangential‖ | ‖normal‖ | λ₂ | share of tangential along v₂ |
|---|---|---|---|---|---|
| 9 | 2.19e-02 | 2.17e-02 | 2.94e-03 | 3.36e-06 | 99.6 % |
| 11 | 5.12e-03 | 5.10e-03 | 3.73e-04 | 9.46e-07 | 99.7 % |
| 13 | 7.04e-03 | 7.03e-03 | 4.30e-04 | 4.23e-07 | 99.7 % |
| 19 | 2.52e-03 | 2.51e-03 | 1.83e-04 | 1.45e-07 | 99.9 % |

The step is essentially `-g₂/λ₂ · v₂` with `λ₂ ~ 1e-7`: an unregularized Newton
step along a direction of numerically-zero curvature. Regularizing the Hessian
is precisely what bounds that quotient, and it never happens.

### How that turns into the stall

Because the constraints are quadratic, `c(x + d) = ½ Q(d)` exactly once
`J d = -c`. A tangential step that long makes the second-order term dominate:

```
[PN_TRIAL] iter=13 trial=0 alpha=1.0000e0   Reject  theta=5.242e-3  th_tr=3.090e-2
[PN_SOC]   iter=13 count_soc=0 a_soc=1.0e0  Reject  theta=5.242e-3  th_soc=5.385e-2
[PN_TRIAL] iter=13 trial=1 alpha=5.0000e-1  Reject                  th_tr=1.035e-2
[PN_TRIAL] iter=13 trial=2 alpha=2.5000e-1  Reject                  th_tr=5.863e-3
[PN_TRIAL] iter=13 trial=3 alpha=1.2500e-1  Accept                  th_tr=5.070e-3
```

The full step grows `theta` by 6×, so the filter rejects it. The second-order
correction fires (gate `trial == 0 && theta <= theta_trial`, matching upstream)
but **diverges**: `theta_soc = 5.4e-2` is worse than the uncorrected
`3.1e-2`, so the SOC loop's `theta_trial <= kappa_soc · theta_soc_old` guard
stops it after one try. SOC's fixed point only contracts when
`‖c(x+d)‖ ≪ ‖c(x)‖`; here `c(x+d)` is already 6× `c(x)`, so it cannot. This is
textbook Maratos-effect territory that the SOC is designed for, entered at a
step length the SOC cannot repair.

What follows is mechanical: ten iterations of `alpha_pr = 1/8 … 1/128` making
~1e-5 of progress per iteration in `phi`, a watchdog `w` break-out at 20,
another ten, a second break-out, and the acceptable-level exit. The line search
is behaving correctly given the step it is handed — the step is the problem, and
the step is bad because the reduced Hessian is numerically singular and nothing
regularizes it.

## 5. Why Ipopt does not hit this

Ipopt's `lg(rg)` on `eigenb2` is non-blank at essentially every iteration
(`2.0, 3.3, 2.9, …, -3.4, -3.8, -4.3, -4.8, -5.3`), i.e. MA57 reports
`WrongInertia` at `δ_x = 0` on every iteration and Ipopt always applies some
`δ_w`. Ipopt's advantage here is *not* algorithmic: MA57's `INFO(24)` is a
thresholded, pivoting-dependent quantity, and on a KKT whose smallest positive
curvature is 1e-9 to 1e-13 relative it stops being reliably signed. That
inaccuracy acts as an implicit regularizer that caps the null-direction step.
FERAL's inertia is *more* accurate, and on this problem the accuracy is what
hurts. The same reading explains #540 from the opposite side: `eigena2`'s
`lg(rg)` re-escalates from `10^-0.8` to `10^1.4` when POUNCE gets a
`WrongInertia` where Ipopt does not, on a KKT that is equally on the edge.

## 6. The knob that already exists, and why its default cannot simply be raised

`feral_singular_pivot_floor` was added for exactly this shape — its own option
text says "a numerically rank-deficient KKT system that happens to land on the
correct inertia produces a clean solve and the IPM never escalates delta_w".
Raising it from the `1e-20` default (MA57 `CNTL(2)`) does fix `eigenb2`:

| `feral_singular_pivot_floor` | iterations | exit |
|---|---|---|
| `1e-20` (default) | 67 | Solved To Acceptable Level |
| `1e-14` | 67 | Optimal Solution Found |
| `1e-12` | 67 | Optimal Solution Found |
| `1e-10` | 51 | Optimal Solution Found |
| `1e-9` | 43 | Optimal Solution Found |
| `1e-8` | **39** | Optimal Solution Found |
| `1e-7` | **33** | Optimal Solution Found |

and `feral_singular_pivot_floor=1e-8 mu_strategy=adaptive` reaches **30
iterations, Optimal** — within 1.5× of Ipopt's 21.

**But the default must not be raised globally.** Measuring FERAL's smallest
accepted pivot (scaled space; `max|pivot|` sits at ~2.0 throughout, so the
equilibration makes this effectively a relative quantity) across the whole
`crates/pounce-cli/tests/fixtures` corpus shows healthy solves living far below
any floor that would fire on `eigenb2`:

| fixture | min pivot | exit |
|---|---|---|
| `pooling_rt2stp.nl` | 3.1e-21 | Optimal Solution Found |
| `airport.nl` | 1.6e-14 | Optimal Solution Found |
| `feasible_x0_wide_scale.nl` | 4.0e-14 | (converges) |
| `jit1.nl` / `jit1_boxed.nl` / `jit1_node.nl` | 2.7e-12 … 4.3e-12 | Optimal Solution Found |
| `lp_afiro.nl` | 2.2e-10 | (converges) |
| `csfi2.nl` | 6.9e-10 | Solved To Acceptable Level |

A floor at `1e-8` would flag `airport`, `jit1`, `pooling_rt2stp` and `lp_afiro`
as singular on iterations where they are converging fine. The original design
note's warning holds: on a *bounded* problem the tiny pivot comes from the
barrier blocks (`Σ_x = z/x` as a bound activates) and is both legitimate and
harmless; on `eigenb2` it comes from the Hessian and is fatal. The pivot
magnitude alone does not distinguish the two — `eigenb2` only makes it look like
it does because the problem has no bounds at all, so the *only* source of a
tiny pivot is the Hessian.

## 7. A step-curvature guard was prototyped, and it does not work

The obvious candidate fix is to key off **the curvature along the computed
step** instead of the pivot magnitude. Ipopt already computes exactly the right
quantity for its inertia-free heuristic
(`IpPDFullSpaceSolver.cpp:599-623`):

    xWx = dxᵀW dx + Σ_x⊙dx·dx + Σ_s⊙ds·ds + δ_x‖dx‖² + δ_s‖ds‖²

Upstream uses it only to *ignore* a wrong inertia (`neg_curv_test_tol`, the
Chiang–Zavala inertia-free method, default off). The proposed generalisation is
to use it as a *floor*: after a `Success` factorization, if
`xWx < tol · ‖(dx,ds)‖² · ‖W‖`, report `WrongInertia` and escalate `δ_x`. It is
one `W·dx` product, and the `δ_x‖dx‖²` term makes the loop self-terminating.

On `eigenb2` the signal is beautifully clean. Relative curvature
`xWx / (‖(dx,ds)‖² ‖W‖)`, one line per accepted factorization:

```
iter  0  5.24e-1     iter  9  1.26e-3
iter  1  4.21e+0     iter 11  1.31e-4
iter  4  5.20e-1     iter 13  3.08e-5
iter  6  5.48e-1     iter 17  7.22e-6
iter  7  2.11e-1     iter 21  1.83e-7
```

Order 1 while the run is healthy, four to seven orders down once the reduced
Hessian degenerates — and turning the guard on fixes the problem:

| `curv_tol` | iterations | exit |
|---|---|---|
| off | 67 | Solved To Acceptable Level |
| `1e-8` | 50 | Optimal Solution Found |
| `1e-6` | 46 | Optimal Solution Found |
| `1e-3` | 28 | Optimal Solution Found |

**It nevertheless has to be rejected.** Replaying the whole
`crates/pounce-cli/tests/fixtures` corpus against the prototype (55 models,
2-minute cap, `iteration_count` and status compared to the guard-off baseline):

| fixture | off | `curv_tol=1e-8` | `1e-6` | `1e-4` |
|---|---|---|---|---|
| `jit1.nl` | 24 Optimal | 21 Optimal | 840 Acceptable | 3000 MaxIter |
| `jit1_boxed.nl` | 24 Optimal | 22 Optimal | 1023 Acceptable | 3000 MaxIter |
| `jit1_node.nl` | 24 Optimal | **246** Optimal | 3000 MaxIter | 3000 MaxIter |
| `cresc4.nl` | 81 Optimal | timeout | 3000 MaxIter | 3000 MaxIter |
| `csfi2.nl` | 35 Acceptable | 24 Optimal | 37 Optimal | 38 Optimal |
| `pooling_rt2stp.nl` | 206 Optimal | timeout | timeout | timeout |
| `deb7.nl`, `unbounded_exp.nl`, `infeasible_equalities.nl`, `issue_508_*` | — | timeout | timeout | timeout |

42 of 55 fixtures are untouched at every tolerance, but the 13 that move include
outright failures, and even the most conservative `1e-8` costs `jit1_node` ten
times the iterations and pushes several models past the time cap. The
false-positive mode is real: a Newton step whose normal (range-space) component
dominates has an unconstrained Rayleigh quotient, and the `Σ`-weighted terms do
not rescue it near an active bound. Restricting the test to the tangential
component would fix that, but computing a null-space projection per
factorization is not something this hot path can afford.

So the shape of the fix is understood and the signal is real, but the cheap
version of it is not shippable. What a working version needs is a curvature
measure confined to the tangential subspace (or an equivalent cheap surrogate),
plus a full-corpus run — all of `vanderbei`, `cute`, `mittelmann`, iteration
counts *and* statuses — to set a default. `$POUNCE_BENCH_DATA` is not available
in the container this was investigated in, so that gate has not been passed and
nothing is turned on. Until then §6's per-problem recipe is the answer, and it
is documented in `docs/src/troubleshooting.md`.

## Other options tried on `eigenb2`

| options | iterations | dual inf | exit |
|---|---|---|---|
| *(defaults)* | 67 | 4.69e-07 | Solved To Acceptable Level |
| `mu_strategy=adaptive` | 75 | 7.91e-11 | Optimal Solution Found |
| `feral_scaling=mc64` | 3000 | 9.09e+01 | Maximum Number of Iterations Exceeded |
| `feral_pivtol=1e-2` | 76 | 2.19e-03 | Error in step computation |
| `feral_static_pivoting=yes` | 67 | 4.69e-07 | Solved To Acceptable Level |
| `feral_singular_pivot_floor=1e-8` | 39 | 1.25e-09 | Optimal Solution Found |
| `feral_singular_pivot_floor=1e-8 mu_strategy=adaptive` | 30 | 4.98e-09 | Optimal Solution Found |

`feral_scaling=mc64` is worth flagging separately: it takes `eigenb2` from a
67-iteration acceptable solve to a 3000-iteration failure with `dual_inf` back
at 9e+01. That is a much larger regression than anything in this issue and is
not explained here.

## Reproducing

The fixture is committed, so no corpus is needed:

```
cargo build --release -p pounce-cli
OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  ./target/release/pounce crates/pounce-cli/tests/fixtures/eigenb2.nl

# the workaround
OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  ./target/release/pounce crates/pounce-cli/tests/fixtures/eigenb2.nl \
    feral_singular_pivot_floor=1e-8

# the reduced-Hessian spectrum
./target/release/pounce crates/pounce-cli/tests/fixtures/eigenb2.nl \
    --dump kkt:all --dump-dir /tmp/eigenb2-kkt
```

`crates/pounce-cli/tests/issue_541_eigenb2_degenerate_hessian.rs` pins the
post-#544 behaviour: the default solve certifies `Optimal`, and
`feral_singular_pivot_floor=1e-8` still reaches the optimum point.

> **Addendum, gh#693.** Every measurement above was taken before #693
> removed the Tikhonov `δ = 1e-8` from the least-squares
> equality-multiplier initializer, and they are left as recorded — this
> section is the #541 investigation, not a current-state reference. The
> current numbers on the same committed fixture are:
>
> | options | 0.10.0 | with #693 |
> |---|---|---|
> | *(defaults)* | 67 it, 3.504e-09, Optimal | **21 it, 2.712e-09, Optimal** |
> | `feral_singular_pivot_floor=1e-8` | 39 it, 7.806e-10, Optimal | 72 it, 2.394e-08, Acceptable |
> | `… + mu_strategy=adaptive` | 30 it, 3.11e-09, Optimal | 86 it, 1.768e-08, Acceptable |
> | `mu_strategy=adaptive` | 63 it, 7.763e-10, Optimal | 21 it, 2.712e-09, Optimal |
>
> The §6 conclusion that the knob rescues this model is therefore no
> longer the operative advice: the default now reaches the optimum in
> fewer iterations than the knob ever did, and the knob costs the
> certificate. §6's separate conclusion — that the *default* floor must
> not be raised globally, because healthy corpus solves live below any
> floor that fires on `eigenb2` — is untouched by #693 and still holds.
> The question this raised — whether the recipe is still correct advice
> for the *symptom* it is written for, which one fixture cannot answer —
> was settled against the benchmark corpus rather than left open. On the
> 110 hardest corpus problems (non-`Optimal` with `dual_inf > tol`, or
> 100+ iterations to certify), `feral_singular_pivot_floor=1e-8` is
> unchanged on 89, better on 10 (5 rescues, 5 speedups ≥20%) and worse on
> 11 (7 lost certificates or solves, 4 slowdowns ≥25%). A coin flip in
> aggregate, with large effects both ways: `britgas` goes
> `Restoration Failed`@2748 → `Optimal`@54, `twirism1` goes
> `Optimal`@178 → `Optimal`@1679. Five of the seven regressions are
> `Optimal → Solved To Acceptable Level` — the same shape as `eigenb2`'s,
> the right point without the certificate. `docs/src/troubleshooting.md`
> carries the table and the resulting advice: reach for it only when
> already losing, and check `dual_inf` against `tol` afterwards.

The first of those is the regression test for the `eigenb2` half of #544 —
that PR pins `eigena2` and found this model only through a corpus sweep, so
without this test its second claim rests on the sweep alone. The second keeps
the knob honest: it is no longer needed for correctness, but it remains the
fastest route through this model's degeneracy.

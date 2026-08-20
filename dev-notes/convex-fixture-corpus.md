# The convex half of the CLI fixture corpus

`scripts/sweep-fixtures.sh` is the tool CLAUDE.md requires before a trajectory
change merges, and it is only as good as the models it sweeps. This note is
about one population inside it: the fixtures that reach the **convex** driver
(`pounce-convex` — the HSDE loop and the direct driver behind it), as opposed
to the general filter-IPM.

## The gap, and how it was found

gh#690 measured an adaptive-τ tail for the HSDE corrector three times over
four months and declined it on the third pass. The number was not the problem:
−4.2% iterations on the exact leg, −1.7% on the L-BFGS leg, no status changes,
no objective regressions. The population was.

Every fixture that moved under the variant:

| moved | vars |
|---|---|
| `lp_afiro` | 32 |
| `scaled_feasible_a` / `_b` | 4 |
| `rankdef_eq_qp` | 3 |
| the other 13 | 1–2 |

The corpus's substantial models — `deb7` (813 vars), `eigena2`/`eigenb2` (110),
`autocorr_bern55-06` (56), `pooling_rt2stp` (46) — are all NLP class, and the
convex driver never sees them. `airport` (84 vars) was the only convex fixture
above 32 variables and did not move at all. Net of one pathological ill-scaled
stress fixture, the corpus-wide saving was **90 iterations across 59 problems,
almost entirely `8 → 5` on models of one to three variables.**

That is arithmetic over a population that cannot answer the question. #690's
closing note put the prerequisite plainly:

> The prerequisite is not more τ variants — it is convex fixtures large enough
> to measure. […] Until the convex path is exercised by something bigger than
> 32 variables, an HSDE step-rule change cannot be evaluated here at all, and
> that gap is worth more than this rule was.

The cause is worth naming, because it is not carelessness. Every convex fixture
in the corpus was added as a **routing or verdict witness** — to prove which
engine ran, or what it reported, or that a presolve line appeared exactly once.
Two variables do that perfectly well, and a bigger model would only have made
the test slower. Nothing was ever added as a **trajectory witness**, and the
sweep silently inherited the difference.

## What was added (gh#724 branch)

Four fixtures, chosen for distinct pathologies rather than for size alone:

| fixture | source | n | m | iters | qp_tau=0.99999 |
|---|---|---|---|---|---|
| `lp_degen2` | NETLIB `degen2` | 534 | 444 | 15 | 11 |
| `lp_share1b` | NETLIB `share1b` | 225 | 117 | 32 | 25 |
| `lp_israel` | NETLIB `israel` | 142 | 174 | 29 | 23 |
| `convex_qp_share1b` | Maros–Mészáros `QSHARE1B` | 225 | 117 | 28 | 22 |

* **`degen2`** — massively primal-degenerate, the gh#535 / gh#133 population:
  strict complementarity fails and a pure IPM struggles to certify the vertex.
  The largest of the four at 534 columns.
* **`share1b`** — the classic ill-conditioned NETLIB instance, and the longest
  trajectory of the four, so the most sensitive to how far a step may go.
* **`israel`** — dense columns, which stress the KKT factorization and its
  ordering rather than the barrier trajectory.
* **`QSHARE1B`** — `share1b` with a quadratic objective bolted on. Same
  sparsity, same bounds, non-zero `P`: it exercises the convex-QP branch of the
  same driver against an LP control differing in one term.

All four solve in under 0.11 s each, so both sweep legs stay cheap; together
they add ~115 KB to the repository.

The `qp_tau` column is the point. `qp_tau` is the fraction-to-boundary
parameter — the one knob that changes how far a step may go without changing
the model, the convergence test, or the engine — and it stands in here for "a
step-rule change". Before these four, the convex corpus answered that
perturbation with `8 → 5` on toy models. It now also answers with `32 → 25` on
an ill-conditioned 225-column LP whose optimum is published, and with `15 → 11`
on a 534-column degenerate one.

`lp_afiro` stays exactly where it is; it is the gh#535 and gh#588 witness and
several tests name it directly. These join it.

## Provenance

Nothing here is hand-built. All four are regenerated from cached upstream data
by benchmark harnesses already in the tree:

```sh
# NETLIB LP (netlib.org/lp/data, expanded with netlib's own `emps`)
cd benchmarks/lp && python3 generate_nl.py --netlib-only degen2 share1b israel
cp nl/degen2.nl   ../../crates/pounce-cli/tests/fixtures/lp_degen2.nl
cp nl/share1b.nl  ../../crates/pounce-cli/tests/fixtures/lp_share1b.nl
cp nl/israel.nl   ../../crates/pounce-cli/tests/fixtures/lp_israel.nl

# Maros–Mészáros convex QP (qpsolvers/maros_meszaros_qpbenchmark mirror)
cd benchmarks/qp && python3 generate_nl.py QSHARE1B
cp nl/QSHARE1B.nl ../../crates/pounce-cli/tests/fixtures/convex_qp_share1b.nl
```

Each carries a published optimum from its source collection, which is what
makes it a usable sentinel: a trajectory change that moves the answer can be
checked against a number nobody in this repository chose.

| fixture | published optimum | pounce (convex path) |
|---|---|---|
| `lp_degen2` | −1435.1780000 | −1435.17800002 |
| `lp_share1b` | −76589.318579 | −76589.31857895 |
| `lp_israel` | −896644.82186 | −896644.82186290 |
| `convex_qp_share1b` | 720078.3182 | 720078.31815401 |

`crates/pounce-cli/tests/issue_690_convex_corpus_scale.rs` pins all of it:
the published optimum, the convex routing, the dimensions (so a fixture cannot
be swapped for a smaller model of the same name), and the fact that each one
still *responds* to a step-rule perturbation. It deliberately asserts no
absolute iteration count — those are the most platform-sensitive numbers in the
repository. Measuring is the sweep's job.

## What this does not fix

The models gh#535 was actually filed on — NETLIB/Mészáros `gen`/`gen1`, where
the convex path burns its whole budget in ~191 s and the NLP path finishes in
19 iterations — are still not fixtures, and should not be: `gen.nl` is n=2560 /
m=769 / 63085 Jacobian nonzeros and takes three minutes to *fail*. The four
above are big enough to move under a step rule; they are not big enough to
reach the failure modes that motivated the LP→NLP reroute. Reaching those in
the corpus at a realistic tolerance remains open, and gh#724 documents the
sweep (all fixtures × `qp_tau ∈ {0.99, 0.999, 0.9999, 0.99999}`) that produced
no `NumericalFailure` on the corpus as it stood.

## Related

* gh#690 — the τ study, closed measured-and-declined; its closing comment is
  the source of the table above.
* gh#724 — the LP→NLP reroute gate omitted `NumericalFailure`; found by the
  same study, fixed alongside this corpus work.
* `dev-notes/trajectory-regressions-and-the-fixture-sweep.md` — why the sweep
  exists at all (gh#544, gh#592).

# adversary/

Working directory for the `/adversary` agent — automated correctness testing
of pounce's solver families against independent oracles.

**The agent never modifies pounce source.** Everything it produces lands here.

## Layout
- `log.org` — the running problem index + per-family coverage counts.
- `runs/` — one `*.py` cross-check script + one `*.org` report per problem,
  named `YYYY-MM-DD_<family>_<name>.{py,org}`.
- `fuzz/` — a Rust property-based harness, for properties that must hold over
  a *distribution* of instances rather than on one named problem. It links
  the crates directly (`pounce-qp`, `pounce-cinterface`), so it works without
  a built Python extension and can probe the C ABI as a C caller sees it.
  `./run.sh` reproduces a full run; see "Property-based probes" below.

## Running it
From the repo root, in Claude Code:

```
/adversary                       # 1 problem, least-tested family
/adversary 5                     # 5 problems, balanced across families
/adversary socp                  # 1 second-order-cone problem
/adversary 3 exp geometric programming
```

Families: `nlp`, `lp`, `qp`, `qp-active-set`, `socp`, `exp`, `power`, `sdp`,
`sos`, `autoroute`, `batch`, `diff`, `sensitivity`.

## Oracles
- `scipy`, `numpy`, `sympy`, `jax`, `torch` — preinstalled in `.venv-qa`.
- `cvxpy` (convex/conic gold standard) and `pyomo` (Ipopt-vs-pounce NLP path)
  are installed on demand into `.venv-qa` by the agent.
- `ipopt` binary at `/opt/homebrew/bin/ipopt`.
- `pounce verify <problem.nl> <claim.sol>` — solver-independent feasibility/KKT
  oracle (does not trust the solver that produced the `.sol`).

The full procedure lives in `.claude/commands/adversary.md`.

## Property-based probes (`fuzz/`)

Some defects are not reachable from a named textbook problem — they need a
distribution. Two probes live here, both deterministic from a seed (an
in-tree splitmix64, no `rand` dependency, so a run reproduces bit for bit):

```
./run.sh                          # both probes + the scipy adjudicator
SEED=12345 QP_N=1000 ./run.sh     # different seed / instance count
```

**`adversary-fuzz qp`** — soundness of the l1-elastic infeasibility
certificate. Generates QPs that carry their own verdict: feasible ones derive
their row bounds from `A·x_w` for a witness `x_w`, and infeasible ones are
infeasible by exact arithmetic (contradictory equalities, or a row required to
exceed its own maximum over the box). Asserts that `Infeasible` is never
returned on an instance with a witness, that `Optimal` never sits at a point
violating the problem, and tracks certification power separately as a quality
metric.

**`adversary-fuzz warmstart`** — answer transparency of the C warm-start ABI.
A working set is a hint about which constraints are active; it selects the
path, never the answer. Stages valid-but-deliberately-wrong working sets
before `IpoptSolve` and asserts the answer does not move. The strict form is
asserted only on convex instances, where the minimizer is unique; nonconvex
instances get the weaker invariant that a converged answer must be feasible,
which is what an SQP actually owes when it may land in a different local
minimum.

**The adjudicator is not optional.** `runs/2026-08-05_qp-active-set_adjudicate.py`
re-decides every generated instance with `scipy.optimize.linprog` (HiGHS).
Its job is to catch a bug in the *generator* before any conclusion is drawn
about the solver. If it reports `GENERATOR BUG`, nothing the probes say can
be trusted until that is fixed.

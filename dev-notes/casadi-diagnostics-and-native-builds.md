# CasADi plugin: what gh#634 asked for, what shipped, what did not

gh#634 (a CasADi user driving POUNCE from native C++ with FERAL) asked
for six things. Three of them were already there, three shipped with the
issue, and three did not. This note is the record of the last group, so
the next reader does not have to re-derive why.

Status of the ask, item by item:

| Ask | Status |
| --- | --- |
| Option pass-through | Already worked; typing bug found and fixed (below) |
| Final statistics in `stats()` | **Shipped** — `final_inf_pr` / `final_inf_du` / `final_compl_inf` |
| FERAL / linear-solver statistics | **Shipped, minus timings** — `stats()["linear_solver"]` |
| Richer live iteration information | **Shipped** — `stats()` is callable mid-callback; `current_violations` |
| Native GCC plugin builds | **Documented** — the Makefile already builds Python-free; MinGW/Windows declined |
| Prebuilt native release artifacts | **Not done** — see below |

## What shipped, and the two defects found on the way

The diagnostics were mostly a plumbing job: the C API already had
`GetIpoptPrimalInf` / `DualInf` / `ComplInf`, the linear-solver
post-mortem already existed as `LinearSolverSummary` feeding the CLI's
solve report, and `GetIpoptCurrentViolations` already worked live. The
new C entry points (`GetPounceLinearSolverStats`,
`GetPounceRestorationStats`, `GetPounceOptionType`) expose data POUNCE
already collected; none of them adds instrumentation.

Two real defects turned up while wiring it:

1. **The per-iteration trace accumulated across solves.** CasADi's ipopt
   plugin clears its trace vectors at the top of `solve()`; this plugin
   never did. One solver object called three times reported
   `iter_count = 7` beside a 23-entry `iterations` trace. Every
   receding-horizon loop — the issue author's use case — hit this.
2. **Options were typed off the Python literal, not off POUNCE's
   registry.** `{"tol": 1}` is an `int` in Python and a number to
   POUNCE, so it went to `AddIpoptIntOption`, which refuses it: `tol`
   silently kept its default while the script looked like it set it.
   `GetPounceOptionType` now decides, on both the interpreted and the
   code-generated path.

## Not done: per-iteration restoration flag (`alg_mod`)

The issue asked for an "algorithm mode or restoration-phase flag" per
iteration. Ipopt's intermediate callback carries it as the leading
`alg_mod` argument, POUNCE's C API passes it through, and the plugin
receives it — so exposing it looks like a one-line change.

It is not, because the value is a constant. `IpoptAlgorithm::
build_iter_stats` (`crates/pounce-algorithm/src/ipopt_alg.rs`) hardcodes
`mode: AlgorithmMode::RegularMode`, with a comment saying alg_mod
tracking is a follow-up, and the callback fires only from the outer
loop — restoration's inner solve does not drive it at all. Publishing
the field would have shipped a column that is always `0` and reads as
working restoration detection.

What shipped instead is `stats()["restoration"]` — calls, inner iters,
outer iters, wall seconds — which POUNCE does measure, and which answers
the question restoration flags are usually asked for ("did this solve
struggle, and how much of it was restoration?"). Verified non-zero on a
solve that enters restoration, zero on one that does not; both are in
the parity suite.

Making the per-iteration flag real means firing the user's callback from
the restoration inner solve with `mode = RestorationPhaseMode`, which
means threading the user TNLP into the restoration algorithm. That is a
behaviour change for every existing callback user (more fires per
solve), not just an added field, so it wants its own issue.

## Not done: linear-solver phase timings

The issue asked for symbolic-analysis, numeric-factorization and
back-solve times, then added — correctly — that it was "not a request to
add instrumentation that does not otherwise exist". POUNCE does not
instrument those phases separately: `LinearSolverSummary` carries
counts, fill, pivots and inertia, and no timers. So the timings are
absent from `stats()["linear_solver"]` rather than reported as zero,
which would read as "instantaneous" instead of "unmeasured".

Adding them means timers inside the FERAL backend around analyse /
factor / solve. Cheap in principle; it is new instrumentation on the
hottest path in the solver, so it needs its own issue and a measurement
that it costs nothing.

## Not done: native builds and release artifacts

The two big items, and they are a maintenance commitment rather than a
one-time build.

**Native CMake build.** Partly answered, cheaply, after the issue
author clarified on the PR that what their CI actually needs is a
documented way to build against a CasADi they build from source — not
maintained artifacts.

It turned out the Makefile already did this: `CASADI_LIB`, `CASADI_INC`,
`CASADI_VER` and `CASADI_SRC` are all `?=`, so their Python-derived
defaults are only defaults. Verified by building with
`PYTHON=/nonexistent` and explicit paths; the resulting plugin passes
all 73 parity checks. So the gap was documentation, plus two failure
modes worth catching:

* `CASADI_VER` empty (no Python to read `casadi.__version__` from) defined
  the version macros as nothing and died deep in the plugin source on
  `expected primary-expression before ';'`, naming neither the option
  nor the cause. `check-env` now stops with a message that does.
* CasADi's `INSTALL_INTERNAL_HEADERS` defaults to **OFF**, so a
  source-built CasADi's `make install` does not ship the internal
  headers a plugin subclasses — the `CASADI_SRC` error now says so and
  names both ways out.

What is still not done is the CMake *idiom*: a `find_package`-style
build taking `CASADI_ROOT` / `CASADI_DIR` / `CMAKE_PREFIX_PATH`. That is
a second build system for the same target, and both would have to keep
producing ABI-identical plugins — CasADi's loader does no version
handshake, so drift between them is a silent misbehaviour, not an error.
Worth doing only if the documented Makefile path proves awkward in a
real native CI.

**Windows is out of scope for this project.** The maintainer does not
support a Windows build, so the MinGW half of the request is declined
rather than deferred. Note also that it would not have been one build:
the wheel path targets MSVC (matching CasADi's own wheels,
`casadi/wheel/README.md`) while the issue asks for MinGW to match a
native CasADi distribution. Those are two incompatible binaries for one
platform.

**Release artifacts.** Nothing CasADi-related is published today, not
even the wheel (gh#626 left that open; gh#635 made the wheel's platform
tag correct, which was the blocker). Publishing per-platform native
archives adds a surface with its own (platform × CasADi minor ×
compiler) matrix. Per `CLAUDE.md` it would need a tag-triggered workflow
gated on `github.repository == 'jkitchin/pounce'`, and a decision on
whether `scripts/check-release-consistency.sh` covers it.

Both remain reasonable asks — and both mostly evaporate if the interface
goes upstream into CasADi, which the issue author offered to help
broker. `dev-notes/casadi-interface-options.md` is where that trade-off
is argued.

# Contributing to POUNCE

Thanks for working on POUNCE. This file captures the few conventions that keep
a multi-solver, multi-registry project from drifting. The release mechanics
live in `dev-notes/cargo-release.md` and `dev-notes/pypi-release.md`; this file
is about getting a change *merge-ready*.

Opening a PR fills in `.github/pull_request_template.md`, which scaffolds the
narrative these PRs are expected to carry — problem with numbers, cause, fix,
blast radius, tests — and checklists the definition of done below.

## Enable the git hooks (one-time)

```sh
git config core.hooksPath .githooks
```

The `pre-commit` hook runs `cargo fmt --all -- --check`, mirroring CI so
formatting drift never reaches `main`.

## Definition of done for a user-facing change

POUNCE is a family of solvers (NLP filter-IPM, active-set SQP, convex/conic
IPM, SOS/Lasserre), and the recurring failure mode is *"the feature exists but
isn't documented where a user looks."* To avoid it, a change that adds or
changes user-visible behavior is not done until **all three** of these land in
the same PR:

1. **Code + test.** The behavior, with a test that pins it (Rust `cargo test`
   and/or Python `pytest`).

2. **CHANGELOG entry.** Add a bullet under the `## [Unreleased]` section of
   `CHANGELOG.md` (create that section if it is absent — it sits directly above
   the most recent released version). One entry per feature, in the user's
   terms, naming the surface(s) it affects (CLI / Python / Pyomo). At release
   time the section is renamed to the version and dated.

3. **Book section.** Update the rendered book under `docs/src/`. A brand-new
   page **must** be linked from `docs/src/SUMMARY.md` — an unlinked page is
   invisible in the book, and `scripts/check-docs-consistency.sh` (run in CI)
   will fail the build until it is wired in.

   In particular, anything that changes **which solver handles which problem
   class** — a new solver, a new routing rule, a new `solver_selection` value,
   a class moving from local to global — must update the cross-solver landscape
   docs as a unit (see ownership below), not just the page for the one solver.

## Cross-solver documentation ownership

The cross-solver "landscape" docs are easy to update piecemeal and leave
inconsistent. These three pages must always agree and are owned, as a unit, by
the maintainer (**@jkitchin**) — flag a reviewer on any PR that touches solver
routing or problem-class coverage:

- `docs/src/choosing-a-solver.md` — the solver-landscape map and the
  "at a glance" table.
- `docs/src/lp-qp-routing.md` — how `auto` classifies a problem and the
  `solver_selection` values.
- `docs/src/python.md` — the Python entry points and auto-routing behavior.

When a solver lands or changes its problem-class coverage, update all three in
the same change so a reader gets one coherent story regardless of which page
they land on.

## CI guards worth knowing about

These run on every PR; run them locally before pushing to get fast feedback:

- `scripts/check-release-consistency.sh` — the three registry versions agree
  and the crates.io publish list matches the workspace in topological order.
- `scripts/check-docs-consistency.sh` — every `docs/src` page is reachable
  from `SUMMARY.md` and every TOC link resolves.
- `cargo fmt --all -- --check`, `cargo clippy`, `cargo test`, and the Python
  test suite (see `.github/workflows/ci.yml`).

## Measuring coverage (`make coverage`)

Use `make coverage` (or `make coverage-quick` to skip the slow pytest suite),
**not** `cargo llvm-cov --workspace`.

`cargo llvm-cov` instruments and runs only the Rust test suite. Large parts of
POUNCE are exercised solely through the Python extension (`pounce._pounce`) or
through the CLI driven by pytest/pyomo, and those paths read as 0% in a
Rust-only report. That makes the report actively misleading as a "what is
under-tested?" signal: it invents gaps that are in fact well covered. Nor can
`cargo llvm-cov report` fix this after the fact — it has no `--object` flag, so
it can never attribute the extension module's profile data.

`scripts/coverage-combined.sh` therefore drives `llvm-profdata` / `llvm-cov`
directly and passes every instrumented artifact — the Rust test binaries, the
CLI, and the installed `.so` — as an explicit `-object`. It needs
`rustup component add llvm-tools-preview`. Outputs land under
`target/coverage-combined/`:

- `summary.txt` — per-file table across all sources.
- `core.txt` — the numerical core only (`pounce-algorithm`, `pounce-qp`,
  `pounce-linsol`, …), ranked by uncovered regions. Diagnostics, dump, and
  binary paths are excluded deliberately: low coverage there is real but cannot
  corrupt a solve, and it would otherwise crowd out the gaps that can.
- `lcov.info` — for editor/CI consumption.

As a sanity check that the attribution actually worked: `crates/pounce-py/*`
is reachable *only* through the extension module, so those rows read 0% in a
Rust-only report and 60–90% here. If you see `pounce-py` at zero, the `.so`
did not get attributed and the whole report is suspect.

Three things to know before running it:

- **The run leaves `python/pounce/_pounce*.so` built with instrumentation**,
  which is slower and can upset timing-sensitive tests. Restore it with
  `make python-ext` (or `cd python && maturin develop --release`).
- **`test_qp_solve_releases_the_gil` fails during the run.** That is expected,
  not a regression: it asserts that a QP solve actually releases the GIL by
  timing concurrent threads, and the instrumentation overhead breaks the
  timing margin. It passes normally once the extension is restored. Any *other*
  failure is worth investigating.
- **Build everything under instrumentation first, then run, then report.**
  Rebuilding any artifact between profiling and reporting changes its
  coverage-mapping hash and silently yields a 0% report. The script already
  orders itself this way; keep that invariant if you edit it.

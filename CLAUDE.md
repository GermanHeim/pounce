# pounce — release / publishing facts

pounce ships to **three** registries on each release, all three automated by
GitHub Actions (each on its own tag prefix). The crates.io publish used to be
manual; it is now automated by `release-crates.yml` (tag-triggered).

## Surfaces (all must reach the same X.Y.Z)

A pre-tag guard, `scripts/check-release-consistency.sh` (run in CI on every
PR), fails the build unless all three versions below agree **and** the
crates.io publish list matches the workspace's publishable crates in
topological order. Run it before tagging.

1. **PyPI `pounce-solver`** — `.github/workflows/release-pounce.yml`, triggered
   by pushing a `python-vX.Y.Z` tag. Builds wheels (incl. Windows) + sdist,
   publishes to PyPI. Version: `python/pyproject.toml`.
2. **PyPI `pyomo-pounce`** — `.github/workflows/release-pyomo-pounce.yml`,
   triggered by a `pyomo-pounce-vX.Y.Z` tag. Version: `pyomo-pounce/pyproject.toml`.
3. **crates.io — 20 workspace crates** — automated by
   `.github/workflows/release-crates.yml`, triggered by a `vX.Y.Z` tag push
   (or run manually via `workflow_dispatch`, which defaults to a dry run). It
   runs `scripts/publish-crates.sh`, which publishes in topological order and
   is idempotent (skips any crate already live at the target version), so a
   re-run or resumed run is safe; resume a mid-batch failure with
   `--start-from <crate>`. New-crate rate limits apply on first publish only.
   Crates with `publish = false` (pounce-py, pounce-studio-pyo3, iter-diff)
   are intentionally excluded. Full procedure: `dev-notes/cargo-release.md`.
   Version: root `Cargo.toml` `[workspace.package]` (all crates inherit it via
   `version.workspace = true`).

   The CLI binary is also bundled inside the PyPI wheels, so an end user
   `pip install pounce-solver` does not require the crates.io publish — but the
   crates.io publish is still part of a complete release.

## Container images build for stable releases only

A fourth surface, `ghcr.io/jkitchin/pounce`, is **not** covered by the
version guard above and ships **only on a `v*` tag** (gh#599).
`.github/workflows/release-docker.yml` has no `main` trigger: the
source-built image no longer goes out as `:edge` / `:sha-<short>` on every
merge, so those tags sit frozen at whatever commit last published them and
must not be read as "tip of `main`". `make docker` is how you get an image
of an unreleased fix; a manual workflow run (`variant=source`,
`dry_run=false`) publishes one if a bug report needs it.

Every publishing workflow — the three registry releases, this one, and the
docs deploy — gates its first job on `github.repository ==
'jkitchin/pounce'`. Forks receive tags when they sync, and none of these can
succeed there, so without the gate a contributor's fork sync is a failed run
and an email. Keep the gate on the job every other job reaches through
`needs`; `pull_request` runs in the base repo's context, so PR checks are
unaffected. `coverage.yml` carries the same gate for the same reason — it is
not a publishing workflow, but it needs `secrets.CODECOV_TOKEN`, which no fork
has, and it fails loudly on a bad upload.

## Coverage is measured combined, never Rust-only

`.github/workflows/coverage.yml` runs `scripts/coverage-combined.sh` (i.e.
`make coverage`) and uploads to Codecov. It runs on **pushes to `main` only** —
plus `workflow_dispatch` for an on-demand run against a branch. It was on
`pull_request` too until the wait proved indefensible: 33–40 minutes against
ci.yml's ~10, holding every PR visibly un-green long after the checks that can
reject it had passed, for a number nothing blocks on. The cost of that choice
is the per-PR patch-coverage status, which was `informational` anyway; the
other cost is that a PR editing `coverage.yml` no longer exercises it, so
dispatch such a branch manually before merging. It does **not** run `cargo
llvm-cov --workspace`: large parts of POUNCE are reachable only through the
`pounce._pounce` extension module or the CLI, and a Rust-only report shows
those at 0% — inventing gaps in code that is well covered. The job asserts the
attribution worked before uploading, by checking that `crates/pounce-py/*`
does not read ~0% (it can only be reached through the `.so`). Requires the
`CODECOV_TOKEN` repository secret. Full rationale: CONTRIBUTING.md,
"Measuring coverage".

## Trajectory changes need the fixture sweep

A change that reroutes **which** correction the solver reaches for, or that
reorders/rescales the steps it takes, is a *trajectory* change. Run
`scripts/sweep-fixtures.sh` against a baseline binary and diff, **before
merge**, and be able to explain every line that moves.

The sweep runs **two legs** per fixture — `exact` (the default path) and
`lbfgs` (`hessian_approximation=limited-memory`) — each line prefixed with the
leg name. Both run by default; do not diff only one. The L-BFGS leg exists
because the corpus was exact-Hessian only, and that gap shipped gh#677: the
initial Hessian scalar used `scalar2` where Ipopt uses `scalar1`, because
`limited_memory_initialization` was registered and never read. L-BFGS is not
a rare opt-in — the Python frontend and the CasADi plugin both select it
automatically when no exact Lagrangian Hessian is available.

**The convex arm is covered — do not skip the sweep for a convex-path
change.** Both legs run at the default `solver_selection=auto`, which routes
to the most specialized engine available, so 40 of the 71 fixtures never touch
the NLP arm at all: 35 reach the convex QP interior-point and 5 the convex
QCQP conic one. gh#760 is the case for saying so explicitly — `4c02817d`
skipped the sweep on the reasoning that "this is a trajectory change on the
convex path, not the NLP path, so `scripts/sweep-fixtures.sh` does not cover
it", and substituted an objective-parity check over the convex corpus.
Objective parity is blind to trajectory by construction. Run across that
commit the sweep moves 52 fixture-legs and flips `scaled_feasible_a` on the
lbfgs leg from `MaximumIterationsExceeded`/199 to `SolveSucceeded`/69.

Each line also records **which engine solved the model** (`cvx-qp`,
`cvx-qcqp`, `nlp`). Status, objective and iteration count can all be unchanged
while a model silently changes arms, and the JSON report does not name the
engine, so a routing regression used to leave no trace in the diff. A line
whose only moving field is the engine is a routing change, and is as
reportable as a moved iteration count.

What the corpus still cannot give you is **magnitude on large degenerate
models**. Every convex fixture is 1–32 variables while the substantial models
(`deb7`, `eigena2`/`eigenb2`, `pooling_rt2stp`) are all NLP class — see
`dev-notes/convex-fixture-corpus.md`. Nothing in the corpus predicted the 4.4×
iteration cost `4c02817d` carried on the Maros-Meszaros QSCFXM family
(38 → 168). A moved convex line is a signal to go measure `benchmarks/qp`, not
a bound on what you will find there.

"It cannot produce a wrong answer" is **not** the relevant safety property
here, and that exact argument is what shipped gh#544 in 0.10.0: a trajectory
regression produces the *right* answer, slowly — or a differently-wrong
tolerance-legal one. gh#544 took `pooling_rt2stp` from 206 to 812 iterations;
the suite asserts status and objective, not iteration count, so nothing saw it.
It surfaced as a CI wall-clock timeout, was misattributed, the cap was raised
for good reasons, and the underlying defect shipped and came back as gh#592.

Related: a measured regression that gets recorded as "an accepted cost of the
fix" needs an issue and an owner. Without one it is indistinguishable from
noise to the next reader — which is how 206 → 812 sat in a commit message
through a release.

Full post-mortem: `dev-notes/trajectory-regressions-and-the-fixture-sweep.md`.

## Working GitHub issues

When opening a PR that fixes a filed issue, the PR **body** (not just the
title) must contain an actual GitHub closing keyword tied to the issue
number — `Fixes #123`, `Closes #123`, or `Resolves #123`. Putting the issue
number only in the PR title (e.g. `Fix foo (#123)`) does **not** trigger
GitHub's auto-close on merge — the issue is left open, dangling, even
though the fix is merged. Confirmed missing on PR #342 (fixed #339, no
closing keyword, issue had to be closed by hand after merge); PR #344 (#341)
did it correctly.

## GitHub Release

Created **by hand** (`gh release create vX.Y.Z --notes-file <file>`); no workflow
makes it. Body has historically been the matching `## [X.Y.Z]` section of
CHANGELOG.md. A git tag alone does NOT create a Release, and creating a Release
does NOT trigger any workflow (nothing has an `on: release` trigger).

## Checking what's published (don't get this wrong)

crates.io API needs a User-Agent or it silently looks unpublished:

    curl -s -H "User-Agent: pounce-release-check (jkitchin@andrew.cmu.edu)" \
      https://crates.io/api/v1/crates/<name> | python3 -c \
      "import sys,json; c=json.load(sys.stdin).get('crate'); print(c['max_version'] if c else 'NOT PUBLISHED')"

Sanity-check against `serde` first; if serde reads NOT PUBLISHED your request is
being rejected, not the crate missing.

## GAMS solver link — two routes

POUNCE registers with GAMS (`option nlp = pounce;`) two independent ways:

1. **pip (pure-Python, recommended for users)** — `pip install
   pounce-solver[gams]` + `pounce-gams register`. Lives in
   `python/pounce/gams/` (`gmo_translate.py`, `link.py`, `register.py`) with the
   `pounce-gams` CLI in `python/pounce/_gams_cli.py`. Built on GAMS's own
   `gamsapi[core]` PyPI bindings (which `dlopen` the user's GAMS libs) — **we
   redistribute nothing GAMS-owned**. POUNCE is a local NLP solver, so the link
   wires GMO's numerical evaluators straight into the cyipopt-style `Problem`
   callbacks (no opcode translator, unlike discopt's global solver). Registers a
   script solver via a `gamsconfig.yaml` `solverConfig` entry — no `sudo`, no
   system-dir writes, survives GAMS upgrades. The per-user config dir is
   OS-specific and **NOT XDG on macOS**: macOS `~/Library/Preferences/GAMS`,
   Linux `~/.config/GAMS`, Windows `%LOCALAPPDATA%\GAMS` (verify with `gamsinst
   -listdirs`). License-free unit tests in `python/tests/test_gams_link.py`
   drive a fake `GmoView`; the live `gamsapi` adapter is the only
   CI-untestable surface.
2. **native C link** — `gams/gams_pounce.c` + `make -C gams && sudo make -C gams
   install`. The authoritative reference for GMO call sequence, sign
   conventions, option keywords, and status mapping. Adds active-set-SQP
   working-set / state-file warm starts the pip link does not yet reproduce.

Docs: `docs/src/gams.md` (user-facing), `gams/README.md` (C link).

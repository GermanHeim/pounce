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

## Sensitivity changes need the invariance legs

`crates/pounce-sensitivity/tests/sens_invariance_legs.rs` runs a kink
fixture down **three** dimensions. Each leg exists because a defect shipped
through a corpus that was uniform in exactly that dimension, and each is
mutation-checked: reintroduce the historical defect and the matching leg —
and only that leg — goes red.

1. **Scaling.** Both arms solve under `user-scaling`; only the per-variable
   factors differ. Under `x̃ = d ⊙ x` the barrier diagonal carries `d⁻²`, so
   anything compared against a bare `Sigma` moves. `Sigma` is also
   proportional to the `mu` each run stopped at, so **the invariant is
   `Sigma/mu`, never `Sigma`** — `variable_scaling_sensitivity.rs` says the
   same thing about the classifier. This is what shipped `205bb67`: the
   corrector added the scaled iterate to bounds in the model's units, which
   "coincide only at unit scaling, which is every fixture it had."
2. **Perturbation magnitude.** `delta` over eight orders, both sides of the
   kink. gh#672 finding 4 put an absolute tolerance on a quantity that
   scales with the perturbation, so a `1e-10` step cleared feasibility
   everywhere and the holding side's derivative read `-1` instead of `0`.
   This leg runs over **two** fixtures, because the rule that engages a
   weak row branches on the activity class. The second is a *coupled*
   kink: `classify` divides `Sigma` by the Hessian's **diagonal**, but a
   kink's multiplier is generated by the curvature *reduced* along that
   coordinate, so the ratio is `reduced/diagonal` and equals 1 only when
   the coordinate is decoupled. Couple it and a genuine kink lands in
   `AMBIGUOUS`, not `WEAKLY_ACTIVE` — routine on a collocation model. A
   rule that is exact in one class and length-based in the other is
   invisible to a corpus carrying only the certified class.
3. **Fixed variable ahead of the kink.** full-x and var-x diverge from the
   first `make_parameter`-removed variable on, and reading one as the other
   returns a *neighbouring* variable's answer — plausible and wrong. That is
   gh#450, and gh#672 finding 1 shipped it again.

The legs compare **slopes, not `dx/delta`**. The parametric step is affine
in `delta`, not linear: it carries a base-point term of order `mu` that
dividing by `delta` inflates until, at `1e-10`, it is the whole answer.
`the_step_is_affine_in_delta` pins that term's size so a leg never fails for
a reason its name does not describe.

A change to membership, to the directional decision, or to any frame
conversion in `pounce-sensitivity` runs these before merge. A new
public accessor gets a row in each leg — the cost of leaving one out is
that the next defect in that dimension is invisible, which is the whole
history above.

**A leg is only evidence about the branch its fixture reaches.** If the
rule under test *branches* — on an activity class, on a status, on which
side of a threshold a quantity falls — then a fixture that always takes
one branch says nothing about the other, and it stays green while the
other branch is broken. This is not hypothetical: the legs above passed on
gh#756's head while that PR's defect was live, because their kink
certifies `WEAKLY_ACTIVE` and that PR's rule gave certified rows an
early return — so the code actually under review was never executed. The leg that found the defect
is the one whose fixture lands in the *other* class. Before trusting a
green leg on a change, check which branch its fixture takes, and add a
fixture for each branch the rule can take — the second fixture is the
test, not a duplicate of the first.

Corollary for the classifier specifically: `AMBIGUOUS` is not "probably
not a kink". A genuine kink lands there whenever its coordinate is
coupled, because the ratio is `reduced/diagonal` (gh#763). Never use the
activity class as a proxy for kink-ness — that inference is exactly what
produced the gh#756 defect.

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

# PyPI release setup

POUNCE ships two PyPI distributions:

| Distribution     | Import name     | Source                          | Workflow                          |
| ---------------- | --------------- | ------------------------------- | --------------------------------- |
| `pounce-solver`  | `pounce`        | `python/` + `crates/pounce-py/` | `release-pounce.yml`              |
| `pyomo-pounce`   | `pyomo_pounce`  | `pyomo-pounce/`                 | `release-pyomo-pounce.yml`        |

Both workflows use PyPI **Trusted Publishing** (OIDC), so no API tokens
need to be stored anywhere. This file is the one-time click-through to
make that work.

## Why `pounce-solver` instead of `pounce`?

`pounce` on PyPI is owned by an unrelated CLI downloader project
(github.com/darthlukan/pounce, last released v1.1). We publish under
`pounce-solver`; the Python import name stays `pounce` because
maturin's `module-name = "pounce._pounce"` plus `python-source = "."`
control what users actually type.

## One-time PyPI side setup

Do this **once per project** on PyPI (and once on TestPyPI if you want
the dry-run path).

### 1. Register the projects

If the project does not exist yet on PyPI, you have two options:

- **Pre-register the name via a manual upload.** Make a throwaway 0.0.0
  wheel locally, `twine upload` it once with an API token, then yank
  it. From then on, Trusted Publishing handles every release.
- **Use the "pending publisher" flow.** PyPI lets you configure a
  Trusted Publisher *before* the project exists. The first run of the
  workflow then creates the project. This is the cleaner path; instructions:
  https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/

### 2. Add the GitHub workflow as a Trusted Publisher

For each of `pounce-solver` and `pyomo-pounce`, on
https://pypi.org/manage/project/<name>/settings/publishing/ click
**Add a new pending publisher** (or **Add a new publisher** if the
project already exists) and fill in:

| Field                | `pounce-solver`             | `pyomo-pounce`                    |
| -------------------- | --------------------------- | --------------------------------- |
| PyPI Project Name    | `pounce-solver`             | `pyomo-pounce`                    |
| Owner                | `jkitchin`                  | `jkitchin`                        |
| Repository name      | `pounce`                    | `pounce`                          |
| Workflow filename    | `release-pounce.yml`        | `release-pyomo-pounce.yml`        |
| Environment name     | `pypi`                      | `pypi`                            |

Repeat on https://test.pypi.org/manage/account/publishing/ using
environment name `testpypi` if you want the workflow_dispatch dry-run
path to work.

### 3. Create the GitHub environments

In https://github.com/jkitchin/pounce/settings/environments create two
environments — `pypi` and `testpypi` — with no secrets. Optionally add
required reviewers / branch protection on `pypi` so a tag push pauses
for human confirmation before uploading.

## Cutting a release

### `pounce-solver` (the maturin wheel)

```sh
# 1. Bump version in python/pyproject.toml AND
#    workspace `version` in Cargo.toml if you want crates+python in lockstep.
# 2. Commit and tag:
git commit -am "release: pounce-solver v0.1.1"
git tag python-v0.1.1
git push origin main --tags
```

The `release-pounce` workflow fires on the tag, builds wheels for
linux x86_64 + aarch64, macOS x86_64 + arm64, and Windows x86_64, plus
an sdist, and uploads everything via Trusted Publishing.

abi3-py39 means one wheel per platform covers Python 3.9+. No need for
a per-Python-version matrix.

### `pyomo-pounce` (binary-bundling wheel)

```sh
# 1. Bump version in pyomo-pounce/pyproject.toml.
# 2. Commit and tag:
git commit -am "release: pyomo-pounce v0.1.1"
git tag pyomo-pounce-v0.1.1
git push origin main --tags
```

The `release-pyomo-pounce` workflow per platform: builds the `pounce`
CLI from source, drops it into `pyomo_pounce/bin/`, builds the wheel,
then re-tags it as platform-specific so PyPI accepts one wheel per
platform under the same version.

### Dry-run on TestPyPI

In GitHub → Actions → `release-pounce` (or `release-pyomo-pounce`) →
**Run workflow** → set `target` to `testpypi`. Same with `target: none`
to just produce build artifacts without uploading.

## Versioning policy

- **Tag prefix matters.** `python-v*` only fires the `pounce-solver`
  workflow. `pyomo-pounce-v*` only fires its workflow. The Rust
  crates use bare `v*` tags (or `pounce-algorithm-v*`-style if we
  ever publish them individually) — those don't collide.
- **abi3.** The maturin extension is abi3-py39, so a single wheel per
  platform serves Python 3.9 through the latest 3.x. Don't add a
  per-interpreter matrix unless we drop abi3.
- **HSL.** The `pounce-hsl` feature is *not* bundled into the wheels.
  HSL is third-party-licensed and must be installed separately by
  the user; the published wheel includes only the pure-Rust FERAL
  backend.

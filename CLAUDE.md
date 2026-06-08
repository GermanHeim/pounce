# pounce — release / publishing facts

pounce ships to **three** registries on each release. Two are automated by
GitHub Actions (tag-triggered); the **crates.io one is manual** and is the
easiest to forget — it is NOT triggered by pushing a tag or by creating a
GitHub Release.

## Surfaces (all must reach the same X.Y.Z)

1. **PyPI `pounce-solver`** — `.github/workflows/release-pounce.yml`, triggered
   by pushing a `python-vX.Y.Z` tag. Builds wheels (incl. Windows) + sdist,
   publishes to PyPI.
2. **PyPI `pyomo-pounce`** — `.github/workflows/release-pyomo-pounce.yml`,
   triggered by a `pyomo-pounce-vX.Y.Z` tag.
3. **crates.io — 16 workspace crates** — **MANUAL**, via
   `scripts/publish-crates.sh` (run locally). NO workflow does `cargo publish`.
   Full procedure in `dev-notes/cargo-release.md`. The script publishes in
   topological (dependency) order; resume a mid-batch failure with
   `--start-from <crate>`. New-crate rate limits apply on first publish only.
   Crates with `publish = false` (pounce-py, pounce-studio-*, iter-diff) are
   intentionally excluded.

   The CLI binary is also bundled inside the PyPI wheels, so an end user
   `pip install pounce-solver` does not require the crates.io publish — but the
   crates.io publish is still part of a complete release.

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

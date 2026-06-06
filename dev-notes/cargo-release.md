# crates.io release

POUNCE ships 18 Rust crates to crates.io. This file is the procedure.
For the PyPI side (`pounce-solver` + `pyomo-pounce`), see
`pypi-release.md`.

## What publishes, what does not

| Crate                  | Publishes? | Why                                          |
| ---------------------- | ---------- | -------------------------------------------- |
| `pounce-common`        | yes        | foundation                                   |
| `pounce-linalg`        | yes        |                                              |
| `pounce-linsol`        | yes        |                                              |
| `pounce-nlp`           | yes        |                                              |
| `pounce-nl`            | yes        | `.nl` reader; pounce-cli depends on it       |
| `pounce-feral`         | yes        | pure-Rust linear-solver backend              |
| `pounce-hsl`           | yes        | optional HSL/MA57 backend (user supplies HSL)|
| `pounce-l1penalty`     | yes        |                                              |
| `pounce-presolve`      | yes        |                                              |
| `pounce-algorithm`     | yes        | IPM core                                     |
| `pounce-restoration`   | yes        |                                              |
| `pounce-sensitivity`   | yes        | sIPOPT port                                  |
| `pounce-cinterface`    | yes        | C ABI (CreateIpoptProblem / IpoptSolve)      |
| `pounce-studio-core`   | yes        | solve-report parsers; pounce-cli dep (0.4.0+)|
| `pounce-cli`           | yes        | `pounce` and `pounce_sens` binaries          |
| `pounce-py`            | **no**     | ships on PyPI as `pounce-solver` via maturin |
| `pounce-studio-pyo3`   | **no**     | PyO3 wrapper; ships on PyPI                   |
| `pounce-cutest`        | **no**     | benchmark harness (gitignored crate too)     |
| `pounce-large-scale`   | **no**     | synthetic benchmark suite                    |
| `iter-diff`            | **no**     | internal Track-A validation tool             |

Each `publish = false` crate has that flag in its `Cargo.toml`. The
publish script enforces the same list and will skip them by construction.

## Dependency order

Layer 0: `pounce-common`, `pounce-studio-core` (leaf: serde only)
Layer 1: `pounce-linalg`
Layer 2: `pounce-linsol`, `pounce-nlp`
Layer 3: `pounce-nl`, `pounce-feral`, `pounce-hsl`, `pounce-l1penalty`, `pounce-presolve`
Layer 4: `pounce-algorithm`
Layer 5: `pounce-restoration`, `pounce-sensitivity`
Layer 6: `pounce-cinterface`, `pounce-cli`

`pounce-studio-core` is a leaf (serde/serde_json only); it can publish any
time before `pounce-cli`. `pounce-nl` depends on `pounce-common` +
`pounce-nlp`, so it sits in layer 3.

The script publishes one crate at a time in this layered order, not in
parallel — each crate must be live on crates.io before any dependent
crate can publish, and the index update is not instantaneous.

## Rate limits — read this before the first release

The first release is the painful one. crates.io rate-limits **new
crate names** to:

- **5 publishes burst**, then **1 per ~10 minutes**.
- New *versions* of *existing* crates: 1 per minute, burst 30. So
  follow-up releases (0.1.1, 0.2.0, …) are not affected — only the
  initial run for these 13 names is.

Untreated: 5 immediate publishes, then 8 × 10 min = ~80 min wall time
for the initial release. The publish script handles this if you set
`SLEEP=600`, but the better option is to request an exemption.

### Requesting a rate-limit exemption

Before the first release, email **help@crates.io** with:

> Subject: Rate-limit exemption request for batched workspace release
>
> Hi crates.io team,
>
> I am about to publish 13 new crates in a single coordinated release
> for the POUNCE project (https://github.com/jkitchin/pounce — a pure-
> Rust port of Ipopt). All crates share the `pounce-` prefix and will
> be released under my account `jkitchin`. Could I get a temporary
> exemption from the new-crate rate limit so the batch can land in one
> sitting?
>
> Crate list: pounce-common, pounce-linalg, pounce-linsol, pounce-nlp,
> pounce-feral, pounce-hsl, pounce-l1penalty, pounce-presolve,
> pounce-algorithm, pounce-restoration, pounce-sensitivity,
> pounce-cinterface, pounce-cli.
>
> Thanks!

They typically respond within a business day.

## Cutting a release

### Pre-flight

1. Make sure `cargo login` is set up (one-time: `cargo login <token>`
   from https://crates.io/me).
2. Bump the workspace version in `Cargo.toml` (root `[workspace.package]`
   and every entry in `[workspace.dependencies]` that points at one of
   our crates — they must all match the new version). If the version
   bump itself is non-trivial, do it as its own commit before tagging.
3. Bump `CITATION.cff` to match: set `version:` to the new release version
   and `date-released:` to the release date. GitHub's "Cite this
   repository" widget reads these, so a stale value misattributes the
   citation. (The `doi:` is the Zenodo *concept* DOI and stays put — it
   always resolves to the latest version.)
4. Run `scripts/publish-crates.sh --dry-run` to catch any missing
   metadata, broken links, or dirty working tree errors. **This
   completes the dry-run for every crate end-to-end**, so any breakage
   appears here, not three crates into the real release.

### Real release

```sh
# Option A: rate-limit exemption granted, no inter-crate sleep needed:
scripts/publish-crates.sh

# Option B: no exemption — space publishes out to dodge the rate limit:
SLEEP=600 scripts/publish-crates.sh
```

If a publish fails part-way through (network blip, transient 5xx,
intermittent toolchain error), fix the underlying issue and resume:

```sh
scripts/publish-crates.sh --start-from pounce-algorithm
```

### After publish

Tag the release in git so the release point is reproducible:

```sh
git tag v0.1.0 && git push origin v0.1.0
```

(The Python distributions use their own tag prefixes — `python-v*` and
`pyomo-pounce-v*` — so the bare `v*` tag namespace is reserved for the
Rust crates.)

## Yanking

If a release is broken, yank individual crates with
`cargo yank --version 0.1.0 -p pounce-common`. Yanking is reversible
(`cargo yank --undo …`) and does **not** delete the artifact — it just
prevents new builds from picking that version. There is no "yank the
whole workspace" command; iterate over the crate list manually if
needed.

## HSL note

`pounce-hsl` is publishable but does **not** ship HSL source — users
must license MA57 separately from STFC and set `COINHSL_DIR`. The
published crate is a thin FFI wrapper. The README spells this out;
flagging here so we do not accidentally pull HSL source into a future
release and create a licensing problem.

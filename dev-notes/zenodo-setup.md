# Zenodo archival setup

POUNCE auto-deposits to Zenodo on every GitHub release. The metadata
shown on the Zenodo page (title, authors, license, related identifiers)
is driven by `.zenodo.json` at the repo root.

## One-time setup (browser, ~2 min)

1. Sign in to <https://zenodo.org/> with your GitHub account
   (Login → "Log in with GitHub" → authorize).
2. Open <https://zenodo.org/account/settings/github/> and find the
   `jkitchin/pounce` repository in the list. Flip its toggle to **On**.
   - If POUNCE isn't listed, click **Sync now** at the top of the page.
3. That's it. Zenodo now listens for `published` release events on the
   repo.

## Triggering an archive

Cut a GitHub release (Releases → Draft a new release → pick or create a
tag → Publish). Within a few minutes Zenodo will:

- Fetch the source tarball for that tag.
- Apply the metadata in `.zenodo.json`.
- Mint two DOIs: a **version DOI** (this specific release) and a
  **concept DOI** (always resolves to the latest version).

The concept DOI is the one to put in the README badge — it auto-updates
when new releases land.

## Filling in the README badge

After the first release lands, find the concept DOI on the Zenodo
record (it's listed as "Cite all versions"). Replace the placeholder
in `README.md`:

```markdown
[![DOI](https://img.shields.io/badge/DOI-<CONCEPT_DOI_URLENCODED>-blue.svg)](https://doi.org/<CONCEPT_DOI>)
```

`<CONCEPT_DOI_URLENCODED>` is the DOI with its `/` written `%2F` — shields
reads `/` as a path separator, so `10.5281/zenodo.20387011` goes in as
`10.5281%2Fzenodo.20387011`. (Any literal `-` would likewise need doubling
to `--`; Zenodo DOIs don't contain one.)

**Do not** use the badge markdown Zenodo offers on the record page
(the "Cite as" widget on the right sidebar). It points at

```
https://zenodo.org/badge/DOI/<CONCEPT_DOI>.svg
```

which is Zenodo's own badge service, and that endpoint is unreliable enough
that the README badge has broken more than once. The failure is worse than a
missing image: GitHub proxies README images through Camo, and Camo *caches*
what it gets — including an error or a timeout — so one bad fetch leaves a
broken badge sitting on the repo front page long after Zenodo has recovered,
with nothing in the repo to explain it. The shields.io form above has no
Zenodo dependency at all; it is a static label, and the concept DOI is stable
by definition (that is the whole point of a concept DOI), so it cannot go
stale. The README's PyPI and download badges already come from shields, so
this also means one image host to trust instead of two.

Nothing needs updating here on a new release — the concept DOI does not
change. If a *version* DOI ever goes in the README, that one does.

## Editing the deposit metadata

Edit `.zenodo.json` and tag a new release. The new metadata is applied
on the next archive — past versions keep their original metadata, by
design.

## Related identifiers

`.zenodo.json` declares:

- `isContinuationOf` → `10.5281/zenodo.19542664` (the predecessor
  `ripopt` deposit; preserves project lineage)
- `isDerivedFrom` → `10.1007/s10107-004-0559-y` (the Wächter & Biegler
  IPOPT paper)
- `isDerivedFrom` → `github.com/coin-or/Ipopt` (the upstream C++ code)

Add more as the project picks up downstream uses (papers citing POUNCE,
forks, etc.).

## CITATION.cff

`CITATION.cff` powers GitHub's "Cite this repository" widget on the
repo home page. It's separate from Zenodo and used by tools like
Zotero's GitHub importer. Keep its `version:` field in sync with
`Cargo.toml` at release time.

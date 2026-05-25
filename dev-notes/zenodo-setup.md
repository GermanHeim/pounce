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
[![DOI](https://zenodo.org/badge/DOI/<CONCEPT_DOI>.svg)](https://doi.org/<CONCEPT_DOI>)
```

Or, easier: copy the **DOI badge** markdown directly from the Zenodo
record page (there's a "Cite as" / badge widget on the right sidebar).

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

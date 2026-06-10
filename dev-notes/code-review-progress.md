# Code-review remediation progress (dev-notes/code-review-2026-06.md)

Worked one issue per `/loop` iteration: verify by running code → write a
regression test that fails pre-fix and passes post-fix → fix → `cargo test`.

## Status

| ID | Title (short) | Status | Notes |
|----|---------------|--------|-------|
| C1 | presolve: Phase-2 redundancy mask misaligned after Phase-0 row drop | **FIXED** | `apply_redundant_verdicts` helper guards on `row_kept_inner`; rollback path rebuilds `linear_rows`. Test `c1_redundancy_mask_realigned_after_phase0_drop`. |

## C1 detail

- **Bug**: `redundant_mask` from `find_redundant_rows` is aligned to the
  *kept* linear rows (`linear_rows`, filtered by `row_kept_inner`), but the
  mapping loop advanced the mask iterator on *every* `Some(linear_row)`,
  including ones Phase 0 already dropped. Every kept linear row after a
  Phase-0-dropped linear row received its predecessor's verdict → a binding
  constraint silently dropped, reinstated at postsolve with λ=0 (wrong answer).
- **Fix**: extracted `apply_redundant_verdicts()` which advances the mask only
  on rows that are both `Some` *and* still `row_kept_inner[i]`. Also made
  `linear_rows` mutable and rebuilt it to the full set inside the Phase-0
  rollback path (lines ~556-583), so the mask stays aligned with the restored
  all-kept mask there too.
- **Test**: `c1_redundancy_mask_realigned_after_phase0_drop` builds a
  3-linear-row map with inner row 0 dropped by Phase 0 and a mask flagging the
  2nd *kept* row; asserts the fixed helper drops inner row 2 (correct) while the
  inlined old loop drops inner row 1 (the documented bug). Deterministic, no FFI.
- **Verified**: `cargo test -p pounce-presolve` → 202 unit + 1 e2e + 9 doc, all pass.

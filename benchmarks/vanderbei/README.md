# Vanderbei CUTE benchmark archive — AMPL `.nl` form

## Source

The original `.mod` files come from Robert Vanderbei's CUTE-in-AMPL collection:

  https://vanderbei.princeton.edu/bench.html

This is an AMPL transliteration of (most of) the CUTEst test set, maintained
by Vanderbei. Reference solutions for many of the problems are tabulated at:

  https://vanderbei.princeton.edu/cute_table.pdf

## What is here

733 `.nl` files, one per problem, alongside matching `.row` (constraint
names) and `.col` (variable names) files. The `.nl` is what any
AMPL-aware NLP solver — ipopt, knitro, baron, bonmin, pounce-cli — reads
directly; the `.row` / `.col` are optional name maps used only for
human-readable solver output.

## How they were produced

Each `.nl` was generated from the corresponding `.mod` with AMPL
(`/Applications/AMPL/ampl`, version 20260520), via a per-file driver:

```bash
# Strip the solve/display/print tail (AMPL would otherwise try to
# actually solve the problem when the model file is included).
awk '
  /^[[:space:]]*solve[[:space:]]*;/   {next}
  /^[[:space:]]*display[[:space:]]/   {skip=1}
  /^[[:space:]]*dispaly[[:space:]]/   {skip=1}   # tolerate typo in helix.mod
  /^[[:space:]]*print[[:space:]]/     {skip=1}
  /^[[:space:]]*printf[[:space:]]/    {skip=1}
  /^[[:space:]]*write[[:space:]]/     {skip=1}
  /^[[:space:]]*display[[:space:]]*;/ {next}
  {if(!skip) print; if(skip && /;/) skip=0}
' "$src.mod" > tmp.mod

# Drive AMPL to emit the .nl (and matching .row / .col via `auxfiles rc`).
# presolve 0 keeps the .nl identical to what the .mod declares, so the
# solver — not AMPL — gets to do bound tightening and substitution.
ampl <<EOF
option presolve 0;
option auxfiles rc;
model tmp.mod;
write g${stub};
EOF
```

The full driver script (`convert_one.sh`) and the batch loop that ran it
over all 733 files lived in the session that produced this archive.

## Caveats

- **`option presolve 0;`.** AMPL's default presolve aggressively eats
  fixed-variable constraints (e.g. `subject to cons: x = 0.1;`) and tight
  bound implications (e.g. `-1.5 <= x[2]` with the bound) before writing
  the `.nl`. Without `presolve 0`, the solver sees a simplified problem
  rather than the one the `.mod` declares. With `presolve 0`, the `.nl`
  is faithful to the source and the solver does its own presolve. Two
  visible consequences in this archive: `hs001.row` includes `constr`
  (the `-1.5 <= x[2]` row); `aircrfta.row` includes `cons1`, `cons2`,
  `cons3` (the elevator/aileron/rudder fixings). Without the option both
  would have collapsed into the objective or into variable bounds.

- **`solve;`/`display ...;`/typos stripped.** The Vanderbei `.mod` files
  end with execution statements meant for an interactive AMPL session
  (`solve;`, `display obj;`, etc.). Those are stripped before `write`
  because we are emitting a `.nl`, not solving. One file (`helix.mod`)
  has a typo `dispaly x;` instead of `display x;`; the filter tolerates
  it. Stripped statements are not part of the model; nothing numerical
  is lost.

- **One `.nl` is large.** `sensors.nl` is ~46 MB — by far the biggest
  problem in the set. Most `.nl` files are well under 1 MB.

- **Round-tripped through AMPL, not bit-identical to a hand-encoded
  expression graph.** The `.nl` is whatever AMPL's expression-graph
  writer produces from the parsed `.mod`. If you re-convert with a
  different AMPL version you may see tiny structural differences
  (variable / constraint orderings, common-subexpression placement)
  even though the math is the same.

- **Reference solutions live elsewhere.** See `cute_table.pdf` above.
  The `.nl` files contain starting points, bounds, and the
  expression graph but not the optimum.

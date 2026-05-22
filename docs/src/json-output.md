# JSON Solve Report

Pass `--json-output PATH` to write a structured solve report alongside
the regular console output:

```sh
pounce problem.nl --json-output result.json
pounce problem.nl --json-output result.json --json-detail full
```

The report carries everything an AMPL `.sol` file holds — status,
primal `x`, dual `lambda`, suffix blocks — plus FAIR-aligned
provenance metadata (Wilkinson et al. 2016, DOI
[10.1038/sdata.2016.18](https://doi.org/10.1038/sdata.2016.18)) and,
optionally, the per-iteration trajectory.

## Detail levels

| Level | Emits |
|---|---|
| `summary` (default) | FAIR metadata, problem dimensions, final solution, aggregate statistics. |
| `full` | The above plus the per-iteration trajectory (`iter`, `objective`, `inf_pr`, `inf_du`, `mu`, step norms, alphas, line-search trials) and sensitivity / suffix blocks. |

Choose `summary` for production logs and batch runs; `full` for
debugging — it is the JSON equivalent of upstream's `print_level=8`.

## Schema stability

The schema is versioned (`pounce.solve-report/v1`) so downstream
tooling can pin against a major version:

- **Adding fields** is non-breaking — consumers must tolerate unknown
  fields.
- **Removing or renaming** a field bumps the major version (`v1` →
  `v2`).

The [Schema v1 Reference](schema/solve-report-v1.md) documents every
field, the FAIR mapping, and the versioning policy in full.

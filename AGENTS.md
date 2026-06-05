# Agent guide for POUNCE

POUNCE is an interior-point NLP solver (a Rust port/reimagining of Ipopt
3.14). This file is the entry point for **LLM agents and automated tools**:
it points at the machine-driveable interfaces so you don't have to
rediscover them from source.

## Driving the solver programmatically

| You want to… | Use | Discover it via |
|---|---|---|
| Step/inspect/mutate a **live** solve | `pounce <model> --debug-json` | the `hello` handshake (self-describing) |
| Post-mortem a **finished** solve | pounce-studio MCP server, or `--json-output` | `studio/mcp/`, `docs/src/schema/` |
| Solve and get machine-readable output | `pounce <model> --json-output r.json --json-detail full` | `docs/src/schema/solve-report-v1.md` |

### Interactive debugger — `--debug-json` (live)

A *pdb for the interior-point loop*. Launch `pounce <model.nl> --debug-json`
(or `--problem <name>` for a built-in) with stdin/stdout piped. The **first
line is a `hello` event** that enumerates everything you can do —
`commands`, `events`, `checkpoints`, `metrics`, `blocks`, and a
`capabilities` map. Feature-detect off those lists, not the version string.
Then send `{"cmd":"…","id":N}` lines and read `pause` / `progress` /
`terminated` events; every event carries the scalar metrics under the names
listed in `hello.metrics` (`objective`, `mu`, `inf_pr`, `inf_du`,
`nlp_error`, `complementarity`, `iter`). Stop with `{"cmd":"continue"}`
(run to completion) or `{"cmd":"quit"}`.

Full contract and a worked transcript: **`docs/src/debugger.md`**
(see "For an LLM agent: the whole contract"). Human REPL variants:
`--debug`, `--debug-on-error`, `--debug-on-interrupt`,
`--debug-script <file>`.

### pounce-studio MCP server (post-mortem)

`studio/mcp/` is a FastMCP server exposing solve reports as callable tools
(`diagnose`, `find_stalls`, `restoration_windows`, `convergence_trace`,
`compare_runs`, `run_problem`, …). These analyze a **finished**
`pounce.solve-report/v1` JSON. For a **live** step-through, drive
`--debug-json` (above); the MCP `debug_session_guide` tool returns that
contract with a launch snippet.

## Repo conventions

- Build: `cargo build --release` (CLI binary at `target/release/pounce`).
- Test: `cargo test` (workspace) or `cargo test -p <crate>`.
- Docs: `make book` renders `docs/src/` (mdbook) to `docs/book/`.
- The user guide lives in `docs/src/`; `docs/src/SUMMARY.md` is its TOC.
- `gams/nlpbench/` and `benchmarks/` hold benchmark suites; the former is
  gitignored.

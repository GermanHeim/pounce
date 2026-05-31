# pounce-studio-pyo3

PyO3 bindings for [`pounce-studio-core`](../pounce-studio-core). Built
as the `_native` extension module of the `pounce-studio-mcp` Python
package (see `studio/mcp/pyproject.toml`).

The Python MCP server in `studio/mcp/pounce_studio_mcp/` delegates all
analysis through these bindings, so the Rust core stays the single
source of truth for derived series and diagnostics.

## FFI strategy

The bindings return **JSON strings** rather than Python objects; the
Python wrapper does the `json.loads`. This keeps the code trivial on
both sides, avoids a `pythonize` dependency, and is plenty fast at our
data sizes (a Full-detail solve report is a few hundred KB).

Method names intentionally omit a `_json` suffix: the Python wrapper
hides the marshalling, so `Report.summarize()` returns a parsed dict
even though the Rust side returns a string.

## Surface

- `Report` (`pyclass`) — a parsed `pounce.solve-report/v1` document.
  Construct via `from_bytes` / `from_path`. Methods: `summarize`,
  `convergence_trace`, `find_stalls`, `restoration_windows`,
  `diagnose`, `get_iterate`, `render_markdown`, `linear_solver_summary`.
  Parameter-less results are memoised per-instance for cheap repeat
  MCP-tool calls.
- `IterDump` (`pyclass`) — a parsed iter-dump trace. Construct via
  `from_bytes` / `from_path`. Methods: `header`, `records`,
  `record_count`.
- `compare_reports(pairs)` — compare a set of named reports.
- `_native` (`pymodule`) — the extension module that exposes the above.

## Build

`maturin` sets the `extension-module` feature when building the wheel;
plain `cargo build` leaves it off so workspace CI stays
Python-toolchain-free. The crate is `publish = false` — it ships only
inside the `pounce-studio-mcp` wheel.

```sh
cd studio/mcp && maturin develop
```

## License

EPL-2.0.

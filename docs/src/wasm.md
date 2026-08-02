# WebAssembly: POUNCE in the Browser

POUNCE's default build is pure Rust — no C, no Fortran, no BLAS to link —
so the entire solver compiles to WebAssembly and runs in a browser tab:
the AMPL `.nl` reader, the reverse-mode AD tape, the sparse LDL^T
factorization, and the interior-point algorithm. Nothing is sent to a
server.

**[Try it: jkitchin.github.io/pounce/demo](https://jkitchin.github.io/pounce/demo/)**
— drop a `.nl` file on the page, see what is in the model, and solve it. The
demo is published from this repository's `main` branch alongside these docs,
and it runs it locally in your tab.

To run the same page from a checkout:

```sh
rustup target add wasm32-wasip1        # once
crates/pounce-wasm/build.sh --serve    # http://localhost:8000
```

Or `make wasm` to build the module without serving it.

## Hosting it

The page is a static directory (`crates/pounce-wasm/web/`) — deploying it
anywhere is a copy. It needs no special server: no threads means no
`SharedArrayBuffer`, so none of the `Cross-Origin-Opener-Policy` /
`Cross-Origin-Embedder-Policy` headers that thread-enabled wasm requires,
and every URL the page fetches is relative, so it works under any base
path. If a host serves `.wasm` as something other than `application/wasm`,
the page falls back from streaming compilation to a buffered
`WebAssembly.instantiate` on its own.

GitHub Pages is what this repository uses: `.github/workflows/docs.yml`
builds the module and stages `crates/pounce-wasm/web/` into the docs site
at `/demo/`, so the live demo ships with every docs deployment from `main`.
The demo is version-independent — one live build, not one per archived
release tag.

## What you get

Dropping a model shows the problem summary POUNCE derives while building
its evaluator — sizes, degrees of freedom, how many rows are equalities,
how much of the model is nonlinear, Jacobian and Hessian sparsity, and how
the variable bounds break down. Solving streams the usual iteration table
into the page (that really is the solver's stdout) and reports the exit
status, KKT residuals, evaluation counts, and the solution vector next to
the `.col` / `.row` names when you drop those alongside the `.nl`.

Solve options are `ipopt.opt`-format text — the same option names the CLI
and the Python API take.

## Numerical parity with the native build

The wasm build runs the same code, so it produces the same answers. Over
all 37 `.nl` fixtures in `crates/pounce-cli/tests/fixtures`, driven through
the same entry points on both sides:

- exit status: identical on 37 of 37
- iteration count: identical on 37 of 37
- objective: bit-identical on 34 of the 36 that return one; the two
  exceptions (`scaled_feasible_a`, `feasible_x0_sentinel_bound`) differ by
  one ulp, at objectives of 4.5e-10 and 7.1e-11

The 37th (`presolve_overflow_feasible`) returns `InvalidNumberDetected`
with no objective on either side — that is the fixture's job.

Speed is what you would expect from wasm. Solver-internal wall time, same
build, same code path, native `x86_64` vs `wasm32-wasip1` under Node:

| model | n × m | native | wasm | ratio |
| --- | --- | --- | --- | --- |
| `pooling_rt2stp` | 46 × 72 | 9.9 ms | 40 ms | 4.0× |
| `jit1` | 25 × 32 | 8.1 ms | 43 ms | 5.3× |
| `airport` | 84 × 42 | 20 ms | 80 ms | 4.1× |
| `autocorr_bern55-06` | 56 × 1 | 50 ms | 101 ms | 2.0× |
| `deb7` | 813 × 897 | 461 ms | 556 ms | 1.2× |

The larger the model, the closer wasm gets: small solves are dominated by
per-call overhead, while big ones spend their time in the sparse
factorization, where the gap narrows. Nothing here is tuned — no SIMD, no
`wasm-opt`.

## How it is put together

| Piece | What it is |
| --- | --- |
| `crates/pounce-wasm` | C-ABI entry points (`pounce_load`, `pounce_solve`), bytes in / JSON out |
| `crates/pounce-wasm/web` | the demo page: `index.html`, `app.js`, `worker.js`, `wasi.js` |
| `crates/pounce-wasm/build.sh` | builds the module and stages it into `web/` |

The target is `wasm32-wasip1`, not `wasm32-unknown-unknown`. WASI gives the
solver a clock (`std::time::Instant::now()` panics on
`wasm32-unknown-unknown`, and POUNCE times every solve) and a stdout to
write its iteration table to. Browsers do not implement WASI, so the page
carries a ~60-line shim, `wasi.js`, which answers `clock_time_get` from
`performance.now()` and turns each `fd_write` into a line in the log pane.
That shim is the entire cost of the approach: no `wasm-bindgen`, no npm, no
build step beyond `cargo build`.

A solve is one synchronous call into wasm that can run for seconds, so the
module lives in a web worker and the page stays responsive.

## Limitations

- **Single-threaded.** No threads are spawned; rayon-parallel paths run
  serially. Results are unaffected.
- **No AMPL imported functions.** A model that calls compiled-C external
  functions (`funcadd_ASL` — IDAES property packages, for instance) needs a
  dynamic loader the browser sandbox does not provide. The summary flags
  such a model rather than failing mysteriously mid-solve.
- **No HSL.** The optional `ma57` backend links Fortran; the wasm build
  uses the default FERAL backend, like any stock `cargo build`.
- **2.4 MB module**, about 800 kB gzipped over the wire.

## Embedding it in your own page

`crates/pounce-wasm` is a thin shim you can copy or fork. The ABI is four
exports — allocate, load, solve, free — and every payload is JSON:

```js
const summary = fromWasm(wasm.pounce_load(nlPtr, nlLen, 0, 0, 0, 0));
const result  = fromWasm(wasm.pounce_solve(optsPtr, optsLen));
```

Both entry points catch panics and return `{"error": …}`, so a malformed
model cannot trap the instance. See `crates/pounce-wasm/web/README.md` for
the full walkthrough and `crates/pounce-wasm/tests/smoke.mjs` for a
headless (Node) driver of the same ABI.

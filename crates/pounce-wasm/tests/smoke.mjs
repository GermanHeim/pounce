// Smoke test for the wasm build: drive the same C ABI the browser page uses
// and check that a `.nl` file loads, summarizes, and solves.
//
//   crates/pounce-wasm/build.sh
//   node crates/pounce-wasm/tests/smoke.mjs
//
// Node's WASI stands in for the browser's `wasi.js` shim. What this catches
// that `cargo test -p pounce-wasm` cannot: a dependency that stops
// compiling for wasm32, or one that compiles but traps at run time (an
// unsupported syscall, a thread spawn, a missing clock).

import { readFileSync } from 'node:fs';
import { WASI } from 'node:wasi';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));
const wasmPath =
  process.argv[2] ?? join(here, '../../../target/wasm32-wasip1/release/pounce_wasm.wasm');

const wasi = new WASI({ version: 'preview1', args: [], env: {} });
const instance = await WebAssembly.instantiate(
  await WebAssembly.compile(readFileSync(wasmPath)),
  wasi.getImportObject(),
);
wasi.initialize(instance);
const wasm = instance.exports;

const encoder = new TextEncoder();
const decoder = new TextDecoder();

function intoWasm(str) {
  if (!str) return [0, 0];
  const bytes = encoder.encode(str);
  const ptr = wasm.pounce_alloc(bytes.length);
  assert.notEqual(ptr, 0, 'pounce_alloc returned null');
  new Uint8Array(wasm.memory.buffer, ptr, bytes.length).set(bytes);
  return [ptr, bytes.length];
}

function fromWasm(ptr) {
  const mem = new Uint8Array(wasm.memory.buffer);
  let end = ptr;
  while (mem[end] !== 0) end++;
  const json = JSON.parse(decoder.decode(mem.subarray(ptr, end)));
  wasm.pounce_free_string(ptr);
  return json;
}

function load(nl, col = '', row = '') {
  const args = [nl, col, row].map(intoWasm);
  try {
    return fromWasm(wasm.pounce_load(...args.flat()));
  } finally {
    for (const [ptr, len] of args) if (ptr) wasm.pounce_dealloc(ptr, len);
  }
}

function solve(options = '') {
  const [ptr, len] = intoWasm(options);
  try {
    return fromWasm(wasm.pounce_solve(ptr, len));
  } finally {
    if (ptr) wasm.pounce_dealloc(ptr, len);
  }
}

// min x0  s.t.  x0^2 + x1^2 == 1,  x0 + x1 >= 0   ⇒   x0 = -1/√2
const summary = load(readFileSync(join(here, 'simple.nl'), 'utf8'), 'alpha\nbeta\n', 'ring\nline\n');
assert.equal(summary.error, undefined, `load failed: ${summary.error}`);
assert.equal(summary.n_vars, 2);
assert.equal(summary.n_cons, 2);
assert.equal(summary.n_nonlinear_cons, 1);
assert.equal(summary.sense, 'minimize');
assert.deepEqual(summary.var_names, ['alpha', 'beta']);

const result = solve('print_level 0\n');
assert.equal(result.error, undefined, `solve failed: ${result.error}`);
assert.equal(result.success, true, `unexpected status ${result.status}`);
assert.ok(
  Math.abs(result.objective + Math.SQRT1_2) < 1e-6,
  `objective ${result.objective} != ${-Math.SQRT1_2}`,
);
assert.equal(result.x.length, 2);

// A malformed model must come back as JSON, not a trapped instance.
assert.equal(typeof load('this is not an .nl file').error, 'string');

console.log(
  `ok — solved in wasm: objective ${result.objective.toFixed(9)}, ` +
    `${result.iterations} iterations, ${(result.wall_time_secs * 1000).toFixed(1)} ms`,
);

// Worker for the Python app: Pyodide on one side, the POUNCE wasm module on
// the other, joined by a backend function the Python shim calls.
//
// Both are wasm, but they are separate instances with separate memories —
// nothing is shared but the strings that cross between them: Pyomo writes an
// `.nl`, POUNCE returns a JSON result plus the `.sol`, and the shim loads
// that back onto the Pyomo model.

import { createWasi } from './wasi.js';

// Pinned so a Pyodide release cannot change what this page runs. Override
// with ?pyodide=<base-url> to serve Pyodide yourself (offline, or behind a
// network that does not allow the CDN).
const PYODIDE_VERSION = '0.28.3';
const params = new URLSearchParams(self.location.search);
const PYODIDE_URL = params.get('pyodide') || `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

const say = (text) => self.postMessage({ type: 'status', text });
const out = (text) => self.postMessage({ type: 'stdout', text });

// --- the POUNCE module -----------------------------------------------------

let solver = null;
let solverLog = '';
const encoder = new TextEncoder();
const decoder = new TextDecoder();

const wasi = createWasi((text) => {
  solverLog += text;
  out(text);
});

async function loadSolver() {
  let instance;
  try {
    ({ instance } = await WebAssembly.instantiateStreaming(fetch('./pounce.wasm'), wasi.imports));
  } catch {
    const bytes = await (await fetch('./pounce.wasm')).arrayBuffer();
    ({ instance } = await WebAssembly.instantiate(bytes, wasi.imports));
  }
  wasi.bind(instance);
  solver = instance.exports;
}

function intoWasm(str) {
  if (!str) return [0, 0];
  const bytes = encoder.encode(str);
  const ptr = solver.pounce_alloc(bytes.length);
  if (!ptr) throw new Error('wasm allocation failed');
  new Uint8Array(solver.memory.buffer, ptr, bytes.length).set(bytes);
  return [ptr, bytes.length];
}

function fromWasm(ptr) {
  if (!ptr) return null;
  const len = new DataView(solver.memory.buffer).getUint32(ptr, true);
  const text = decoder.decode(new Uint8Array(solver.memory.buffer, ptr + 4, len));
  solver.pounce_free_payload(ptr);
  return text;
}

/**
 * The backend `pounce_browser.solve()` calls. Synchronous, because the wasm
 * solve is: Python blocks here until POUNCE returns, exactly as it would
 * waiting on a subprocess.
 *
 * Returns a JSON string — the Python side accepts one, which keeps the
 * boundary free of Pyodide proxy objects and their lifetimes.
 */
function solveNl(nlText, options) {
  solverLog = '';
  const nlArgs = intoWasm(nlText);
  let loaded;
  try {
    loaded = JSON.parse(fromWasm(solver.pounce_load(...nlArgs, 0, 0, 0, 0)));
  } finally {
    if (nlArgs[0]) solver.pounce_dealloc(...nlArgs);
  }
  if (loaded.error) return JSON.stringify({ result: loaded, log: solverLog });

  const optArgs = intoWasm(options || '');
  let result;
  try {
    result = JSON.parse(fromWasm(solver.pounce_solve(...optArgs)));
  } finally {
    if (optArgs[0]) solver.pounce_dealloc(...optArgs);
  }
  const sol = fromWasm(solver.pounce_solution_sol());
  return JSON.stringify({ result, sol, summary: loaded, log: solverLog });
}

// --- Pyodide ---------------------------------------------------------------

let pyodide = null;

const ready = (async () => {
  say('loading the POUNCE solver…');
  await loadSolver();

  say(`loading Pyodide ${PYODIDE_VERSION} (~10 MB, cached after the first run)…`);
  // `pyodide.mjs`, not `pyodide.js`: this is a module worker, where
  // `importScripts` does not exist.
  const { loadPyodide } = await import(`${PYODIDE_URL}pyodide.mjs`);
  pyodide = await loadPyodide({
    indexURL: PYODIDE_URL,
    stdout: (line) => out(line + '\n'),
    stderr: (line) => out(line + '\n'),
  });

  say('installing Pyomo…');
  await pyodide.loadPackage('micropip');
  const micropip = pyodide.pyimport('micropip');
  // Pyomo publishes a `py3-none-any` wheel, so micropip installs it as-is —
  // its compiled extensions are optional and unused here, and nothing has to
  // be built for wasm. `?pyomo=<url>[,<url>…]` points at self-hosted wheels
  // (offline, or where PyPI is unreachable); the default resolves from PyPI.
  const wheels = params.get('pyomo');
  await micropip.install(wheels ? wheels.split(',') : 'pyomo');

  say('wiring POUNCE into Python…');
  const shim = await (await fetch('./pounce_browser.py')).text();
  pyodide.FS.writeFile('/home/pyodide/pounce_browser.py', shim);
  self.pounceSolveNl = solveNl;
  await pyodide.runPythonAsync(`
import sys
sys.path.insert(0, "/home/pyodide")
import js, pounce_browser
pounce_browser.set_backend(lambda nl, options: js.pounceSolveNl(nl, options))
`);
  say('ready');
  self.postMessage({ type: 'ready' });
})().catch((err) => {
  self.postMessage({ type: 'fatal', message: String(err && err.message ? err.message : err) });
});

self.onmessage = async (event) => {
  if (event.data.type !== 'run') return;
  try {
    await ready;
    if (!pyodide) return;
    self.postMessage({ type: 'running' });
    const started = performance.now();
    await pyodide.runPythonAsync(event.data.code);
    self.postMessage({ type: 'done', ms: performance.now() - started });
  } catch (err) {
    // A Python exception arrives with its traceback in `message`; show it as
    // the script's own output rather than as a page-level failure.
    self.postMessage({ type: 'error', message: String(err && err.message ? err.message : err) });
  }
};

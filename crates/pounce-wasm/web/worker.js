// Worker that owns the POUNCE wasm instance.
//
// A solve is a single synchronous call into wasm that can run for seconds,
// so it lives off the main thread; the page stays responsive and the
// solver's console output is streamed back as it is produced.

import { createWasi } from './wasi.js';

let exports = null;
const encoder = new TextEncoder();
const decoder = new TextDecoder();

const wasi = createWasi((text) => self.postMessage({ type: 'log', text }));

const ready = (async () => {
  const response = fetch('./pounce.wasm');
  let instance;
  try {
    ({ instance } = await WebAssembly.instantiateStreaming(response, wasi.imports));
  } catch {
    // Streaming compilation needs `Content-Type: application/wasm`; not
    // every static server sets it, so fall back to buffering the module.
    const bytes = await (await fetch('./pounce.wasm')).arrayBuffer();
    ({ instance } = await WebAssembly.instantiate(bytes, wasi.imports));
  }
  wasi.bind(instance);
  exports = instance.exports;
})();

/** Copy a JS string into wasm memory. Returns [ptr, len]; ptr 0 for empty. */
function intoWasm(str) {
  if (!str) return [0, 0];
  const bytes = encoder.encode(str);
  const ptr = exports.pounce_alloc(bytes.length);
  if (!ptr) throw new Error('wasm allocation failed');
  new Uint8Array(exports.memory.buffer, ptr, bytes.length).set(bytes);
  return [ptr, bytes.length];
}

/** Read a NUL-terminated string the module returned, then free it. */
function fromWasm(ptr) {
  if (!ptr) throw new Error('wasm returned a null string');
  const mem = new Uint8Array(exports.memory.buffer);
  let end = ptr;
  while (mem[end] !== 0) end++;
  const text = decoder.decode(mem.subarray(ptr, end));
  exports.pounce_free_string(ptr);
  return JSON.parse(text);
}

self.onmessage = async (event) => {
  const msg = event.data;
  try {
    await ready;
    if (msg.type === 'load') {
      const args = [msg.nl, msg.col ?? '', msg.row ?? ''].map(intoWasm);
      try {
        const summary = fromWasm(exports.pounce_load(...args.flat()));
        self.postMessage({ type: 'summary', summary });
      } finally {
        for (const [ptr, len] of args) if (ptr) exports.pounce_dealloc(ptr, len);
      }
    } else if (msg.type === 'solve') {
      const [ptr, len] = intoWasm(msg.options ?? '');
      const started = performance.now();
      try {
        const result = fromWasm(exports.pounce_solve(ptr, len));
        result.browser_ms = performance.now() - started;
        self.postMessage({ type: 'result', result });
      } finally {
        if (ptr) exports.pounce_dealloc(ptr, len);
      }
    }
  } catch (err) {
    self.postMessage({ type: 'fatal', message: String(err && err.message ? err.message : err) });
  }
};

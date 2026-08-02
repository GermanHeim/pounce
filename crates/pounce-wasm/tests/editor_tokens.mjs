// Unit tests for the Python highlighter in web-python/editor.js.
//
//   node crates/pounce-wasm/tests/editor_tokens.mjs
//
// The tokenizer is one regex with alternation, which is exactly the kind of
// code that breaks quietly: a mis-numbered backreference once made every
// string containing an escape fall apart mid-token, and the page still
// rendered — just wrong. These cases are the ones that catch that.

import assert from 'node:assert/strict';
import { highlight } from '../web-python/editor.js';

/** The classes applied, in order, so a case reads as what it should colour. */
const classes = (code) => [...highlight(code).matchAll(/class="tok-(\w+)"/g)].map((m) => m[1]);

/** The text inside the first span of `kind`. */
function spanText(code, kind) {
  const m = highlight(code).match(new RegExp(`<span class="tok-${kind}">([\\s\\S]*?)</span>`));
  return m?.[1];
}

// A string is one token, escapes and all — including an escaped copy of its
// own delimiter, and a `\n` that is not a line break.
assert.equal(spanText(String.raw`opts = "print_level 5\ntol 1e-8"`, 'string'), String.raw`"print_level 5\ntol 1e-8"`);
assert.equal(spanText(String.raw`x = "a\"b"`, 'string'), String.raw`"a\"b"`);
assert.equal(spanText(String.raw`s = 'it\'s'`, 'string'), String.raw`'it\'s'`);

// Triple-quoted strings span lines; f-strings keep their prefix.
assert.equal(spanText('d = """tri\nple""" # after', 'string'), '"""tri\nple"""');
assert.deepEqual(classes('d = """tri\nple""" # after'), ['op', 'string', 'comment']);
assert.equal(spanText('f"{m.x[1]:.3f}"', 'string'), 'f"{m.x[1]:.3f}"');

// A quote inside a comment does not open a string, and a `#` inside a
// string does not open a comment.
assert.deepEqual(classes('# comment with "quote'), ['comment']);
assert.deepEqual(classes('s = "# not a comment"'), ['op', 'string']);

// Keywords, the name a `def` introduces, and the Pyomo vocabulary.
assert.deepEqual(classes('def solve(m): pass'), ['keyword', 'def', 'keyword']);
assert.deepEqual(classes('m.x = Var(initialize=1.0)'), ['op', 'builtin', 'op', 'number']);
assert.equal(spanText('for i in m.I:', 'keyword'), 'for');

// Numbers in the shapes a model file actually uses.
for (const n of ['0x1f', '1_000', '2.5e-3', '40', '1e19']) {
  assert.equal(spanText(`y = ${n}`, 'number'), n, `number: ${n}`);
}

// Markup in the source is escaped, never emitted as markup.
assert.ok(highlight('if a < b and c > d: pass').includes('&lt;'));
assert.ok(!highlight('x = "<script>"').includes('<script>'));

console.log('ok — highlighter: 20 cases');

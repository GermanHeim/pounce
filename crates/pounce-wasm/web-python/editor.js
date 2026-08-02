// A small Python editor: syntax highlighting, line numbers, and the two
// indentation habits a Python editor has to have.
//
// Written rather than imported. The alternative — CodeMirror or Ace from a
// CDN — buys bracket matching and autocomplete at the cost of a second
// version-pinned network dependency on a page whose whole point is that the
// solving happens locally, and it would stop working in exactly the
// self-hosted/offline setup `?pyodide=` exists to support. This is ~150
// lines and no dependency; if the page ever wants an IDE, swap this file.
//
// The technique is the usual one: a `<pre>` of highlighted markup sits
// underneath a transparent `<textarea>`, aligned character-for-character.
// The textarea keeps the caret, selection, undo history, IME, and
// accessibility that a `contenteditable` re-implementation would have to
// rebuild badly.

const KEYWORDS = new Set(
  ('False None True and as assert async await break class continue def del elif else except ' +
    'finally for from global if import in is lambda nonlocal not or pass raise return try ' +
    'while with yield match case')
    .split(' '),
);

// Builtins worth colouring on this page: Python's own, plus the Pyomo names
// a model is written out of — `Var`, `Constraint`, `RangeSet` read as
// vocabulary here, not as user identifiers.
const BUILTINS = new Set(
  ('abs all any bool dict enumerate filter float format int len list map max min print range ' +
    'repr reversed round set sorted str sum tuple type zip ' +
    'ConcreteModel AbstractModel Var Param Set RangeSet Constraint ConstraintList Objective ' +
    'Suffix Block Expression value minimize maximize exp log log10 sin cos tan sqrt ' +
    'SolverFactory TransformationFactory')
    .split(' '),
);

// One pass, longest-first: comments and strings must win over everything,
// and triple quotes over single. `f"…"` is matched as a string with its
// prefix so an f-string does not fall apart at the `f`.
const TOKEN = new RegExp(
  [
    /(?<comment>#[^\n]*)/,
    /(?<string>[rbuf]{0,2}(?<tq>"""|''')[\s\S]*?\k<tq>|[rbuf]{0,2}(?<q>"|')(?:\\.|(?!\k<q>)[^\n\\])*\k<q>?)/,
    /(?<decorator>@[A-Za-z_]\w*)/,
    /(?<number>\b\d[\d_]*\.?[\d_]*(?:[eE][+-]?\d+)?j?\b|\b0[xXoObB][\da-fA-F_]+\b)/,
    /(?<def>\b(?:def|class)\s+[A-Za-z_]\w*)/,
    /(?<word>\b[A-Za-z_]\w*\b)/,
    /(?<op>[+\-*/%=<>!&|^~@]+)/,
  ]
    .map((r) => r.source)
    .join('|'),
  'g',
);

const escapeHtml = (s) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/** Tokenize `code` into highlighted HTML. Exported for `tests/editor_tokens.mjs`. */
export function highlight(code) {
  let out = '';
  let last = 0;
  for (const m of code.matchAll(TOKEN)) {
    out += escapeHtml(code.slice(last, m.index));
    last = m.index + m[0].length;
    const g = m.groups;
    const text = escapeHtml(m[0]);
    if (g.comment) out += `<span class="tok-comment">${text}</span>`;
    else if (g.string) out += `<span class="tok-string">${text}</span>`;
    else if (g.decorator) out += `<span class="tok-decorator">${text}</span>`;
    else if (g.number) out += `<span class="tok-number">${text}</span>`;
    else if (g.def) {
      // `def foo` / `class Foo`: keyword, then the name it introduces.
      const [kw, name] = m[0].split(/(\s+)/).filter((s) => s.trim());
      out += `<span class="tok-keyword">${escapeHtml(kw)}</span> <span class="tok-def">${escapeHtml(name)}</span>`;
    } else if (g.word) {
      const cls = KEYWORDS.has(m[0])
        ? 'tok-keyword'
        : BUILTINS.has(m[0])
          ? 'tok-builtin'
          : null;
      out += cls ? `<span class="${cls}">${text}</span>` : text;
    } else if (g.op) out += `<span class="tok-op">${text}</span>`;
    else out += text;
  }
  out += escapeHtml(code.slice(last));
  return out;
}

/**
 * Turn a plain `<textarea>` into a highlighted editor in place. Returns a
 * `{ get, set }` handle; the textarea keeps working as itself, so a caller
 * that only reads `.value` needs no change and the page degrades to a plain
 * textarea if this module fails to load.
 */
export function attachEditor(textarea) {
  const wrap = document.createElement('div');
  wrap.className = 'editor';
  const gutter = document.createElement('div');
  gutter.className = 'editor-gutter';
  gutter.setAttribute('aria-hidden', 'true');
  const pre = document.createElement('pre');
  pre.className = 'editor-highlight';
  pre.setAttribute('aria-hidden', 'true');

  textarea.parentNode.insertBefore(wrap, textarea);
  wrap.append(gutter, pre, textarea);

  const render = () => {
    // The trailing newline keeps the last line's box alive so the caret on
    // an empty final line still has something behind it.
    pre.innerHTML = highlight(textarea.value) + '\n';
    const lines = textarea.value.split('\n').length;
    gutter.textContent = Array.from({ length: lines }, (_, i) => i + 1).join('\n');
  };

  const syncScroll = () => {
    pre.scrollTop = textarea.scrollTop;
    pre.scrollLeft = textarea.scrollLeft;
    gutter.scrollTop = textarea.scrollTop;
  };

  textarea.addEventListener('input', render);
  textarea.addEventListener('scroll', syncScroll);

  textarea.addEventListener('keydown', (e) => {
    const { value, selectionStart: start, selectionEnd: end } = textarea;

    if (e.key === 'Tab') {
      e.preventDefault();
      const lineStart = value.lastIndexOf('\n', start - 1) + 1;
      if (start !== end || e.shiftKey) {
        // Block indent / dedent over every line the selection touches.
        const blockEnd = value.indexOf('\n', end) === -1 ? value.length : value.indexOf('\n', end);
        const block = value.slice(lineStart, blockEnd);
        const shifted = e.shiftKey
          ? block.replace(/^ {1,4}/gm, '')
          : block.replace(/^/gm, '    ');
        replace(lineStart, blockEnd, shifted, lineStart, lineStart + shifted.length);
      } else {
        insert('    ');
      }
      return;
    }

    if (e.key === 'Enter') {
      // Keep the current line's indentation, and add a level after a colon
      // — the two things that make typing a `for` loop bearable.
      const lineStart = value.lastIndexOf('\n', start - 1) + 1;
      const line = value.slice(lineStart, start);
      const indent = (line.match(/^[ \t]*/) ?? [''])[0];
      const deeper = /:\s*$/.test(line) ? '    ' : '';
      e.preventDefault();
      insert('\n' + indent + deeper);
    }
  });

  function insert(text) {
    const { selectionStart: s, selectionEnd: e } = textarea;
    replace(s, e, text, s + text.length, s + text.length);
  }

  /** Edit through `execCommand` when available so the browser's own undo
   *  stack survives; fall back to a direct splice where it is not. */
  function replace(from, to, text, selStart, selEnd) {
    textarea.setSelectionRange(from, to);
    let ok = false;
    try {
      ok = document.execCommand('insertText', false, text);
    } catch {
      ok = false;
    }
    if (!ok) {
      textarea.value = textarea.value.slice(0, from) + text + textarea.value.slice(to);
    }
    textarea.setSelectionRange(selStart, selEnd);
    render();
  }

  render();
  return {
    get: () => textarea.value,
    set: (text) => {
      textarea.value = text;
      render();
      syncScroll();
    },
  };
}

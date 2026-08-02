// Page logic: take dropped files, ask the worker to load and solve, render.

const worker = new Worker('./worker.js', { type: 'module' });

const $ = (id) => document.getElementById(id);
const drop = $('drop');
const fileInput = $('files');
const loadError = $('loaderr');
const solveButton = $('solve');
const solveStatus = $('solve-status');
const log = $('log');

let summary = null;

// --- file intake -----------------------------------------------------------

['dragenter', 'dragover'].forEach((ev) =>
  drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.add('over');
  }),
);
['dragleave', 'drop'].forEach((ev) =>
  drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.remove('over');
  }),
);
drop.addEventListener('drop', (e) => takeFiles(e.dataTransfer.files));
fileInput.addEventListener('change', () => takeFiles(fileInput.files));

async function takeFiles(fileList) {
  const files = [...fileList];
  const pick = (ext) => files.find((f) => f.name.toLowerCase().endsWith(ext));
  // Anything that is not a .col/.row is treated as the model, so a file
  // named `stub` or `model.nl.txt` still works.
  const nl = pick('.nl') ?? files.find((f) => !/\.(col|row)$/i.test(f.name));
  if (!nl) {
    showLoadError('Drop an .nl file (optionally with its .col / .row name files).');
    return;
  }
  showLoadError(null);
  $('filename').textContent = `— ${nl.name} (${formatBytes(nl.size)})`;
  const [nlText, colText, rowText] = await Promise.all([
    nl.text(),
    pick('.col')?.text() ?? Promise.resolve(''),
    pick('.row')?.text() ?? Promise.resolve(''),
  ]);
  worker.postMessage({ type: 'load', nl: nlText, col: colText, row: rowText });
}

function showLoadError(message) {
  loadError.hidden = !message;
  loadError.textContent = message ?? '';
}

// --- worker plumbing -------------------------------------------------------

worker.onmessage = ({ data }) => {
  if (data.type === 'summary') {
    if (data.summary.error) {
      summary = null;
      $('summary-section').hidden = true;
      $('solve-section').hidden = true;
      showLoadError(data.summary.error);
      return;
    }
    summary = data.summary;
    renderSummary(summary);
    $('result-section').hidden = true;
    $('log-section').hidden = true;
    log.textContent = '';
  } else if (data.type === 'log') {
    log.textContent += data.text;
    log.scrollTop = log.scrollHeight;
  } else if (data.type === 'result') {
    solveButton.disabled = false;
    if (data.result.error) {
      solveStatus.textContent = data.result.error;
      return;
    }
    solveStatus.textContent = '';
    renderResult(data.result);
  } else if (data.type === 'fatal') {
    solveButton.disabled = false;
    showLoadError(data.message);
    solveStatus.textContent = data.message;
  }
};

solveButton.addEventListener('click', () => {
  solveButton.disabled = true;
  solveStatus.textContent = 'solving…';
  log.textContent = '';
  $('log-section').hidden = false;
  $('result-section').hidden = true;
  worker.postMessage({ type: 'solve', options: $('options').value });
});

// --- rendering -------------------------------------------------------------

const num = (v) => (v == null ? '—' : v.toLocaleString());
const sci = (v) => {
  if (v == null || Number.isNaN(v)) return '—';
  if (v === 0) return '0';
  return Math.abs(v) >= 1e-4 && Math.abs(v) < 1e6
    ? v.toPrecision(6).replace(/\.?0+$/, '')
    : v.toExponential(2);
};

function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} kB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function stats(target, entries) {
  target.innerHTML = '';
  for (const [label, value] of entries) {
    const wrap = document.createElement('div');
    const dt = document.createElement('dt');
    dt.textContent = label;
    const dd = document.createElement('dd');
    if (value instanceof Node) dd.append(value);
    else dd.textContent = value;
    wrap.append(dt, dd);
    target.append(wrap);
  }
}

function renderSummary(s) {
  const b = s.var_bounds;
  stats($('stats'), [
    ['Variables', num(s.n_vars)],
    ['Constraints', num(s.n_cons)],
    ['Objective', s.sense],
    ['Degrees of freedom', num(s.degrees_of_freedom)],
    ['Equalities', num(s.n_equality_cons)],
    ['Inequalities', num(s.n_inequality_cons)],
    ['Nonlinear rows', `${num(s.n_nonlinear_cons)} / ${num(s.n_cons)}`],
    ['Nonlinear vars', `${num(s.n_nonlinear_vars)} / ${num(s.n_vars)}`],
    ['Jacobian nnz', `${num(s.nnz_jac)} (${(s.jac_density * 100).toFixed(2)}%)`],
    ['Hessian nnz', num(s.nnz_hess)],
    ['Bounded vars', num(b.boxed + b.lower_only + b.upper_only)],
    ['Fixed vars', num(b.fixed)],
    ['Free vars', num(b.free)],
  ]);

  const notes = [];
  if (s.external_funcs.length) {
    notes.push(
      `Model declares AMPL imported functions (${s.external_funcs.join(', ')}), ` +
        `which need a native shared library — this build cannot evaluate them.`,
    );
  }
  if (s.truncated) {
    notes.push(`Name / value lists are truncated to the first ${num(s.preview_limit)} entries.`);
  }
  $('summary-note').textContent = notes.join(' ');

  $('summary-section').hidden = false;
  $('solve-section').hidden = false;
}

function renderResult(r) {
  const badge = document.createElement('span');
  badge.className = `tag ${r.success ? 'ok' : 'bad'}`;
  badge.textContent = spaced(r.status);

  stats($('result-stats'), [
    ['Status', badge],
    ['Objective', sci(r.objective)],
    ['Iterations', num(r.iterations)],
    ['Solve time', `${(r.wall_time_secs * 1000).toFixed(0)} ms`],
    ['Constraint violation', sci(r.constraint_violation)],
    ['Dual infeasibility', sci(r.dual_infeasibility)],
    ['Complementarity', sci(r.complementarity)],
    ['KKT error', sci(r.kkt_error)],
    ['f / ∇f evals', `${num(r.evals.objective)} / ${num(r.evals.objective_grad)}`],
    ['g / ∇g evals', `${num(r.evals.constraints)} / ${num(r.evals.constraint_jac)}`],
    ['Hessian evals', num(r.evals.hessian)],
    ['Restorations', num(r.restoration_calls)],
  ]);

  const names = summary?.var_names ?? [];
  table($('vars'), ['Variable', 'Value'],
    r.x.map((v, i) => [names[i] ?? `x[${i}]`, sci(v)]));

  const conNames = summary?.con_names ?? [];
  table($('cons'), ['Constraint', 'Lower', 'Value', 'Upper'],
    r.g.map((v, i) => [
      conNames[i] ?? `c[${i}]`,
      finite(r.g_l[i]) ? sci(r.g_l[i]) : '−∞',
      sci(v),
      finite(r.g_u[i]) ? sci(r.g_u[i]) : '∞',
    ]));

  $('result-section').hidden = false;
}

// AMPL writes ±1e19 for "no bound".
const finite = (v) => v != null && Math.abs(v) < 1e19;

// `SolvedToAcceptableLevel` → `Solved To Acceptable Level`.
const spaced = (s) => String(s).replace(/([a-z])([A-Z])/g, '$1 $2');

function table(el, headers, rows) {
  el.innerHTML = '';
  const thead = el.createTHead().insertRow();
  for (const h of headers) {
    const th = document.createElement('th');
    th.textContent = h;
    thead.append(th);
  }
  const body = el.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    for (const cell of row) tr.insertCell().textContent = cell;
  }
}

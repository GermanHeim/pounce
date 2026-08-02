// Page logic: pick an example, run it in the Pyodide worker, show what it
// printed. The worker owns both wasm instances; this side is a console.

const $ = (id) => document.getElementById(id);
const codeBox = $('code');
const outBox = $('out');
const runButton = $('run');
const statusLine = $('status');

const EXAMPLES = {
  'Constrained NLP': `# Minimize x1 on the unit circle, above the line x1 + x2 = 0.
from pyomo.environ import *
import pounce_browser

m = ConcreteModel()
m.x1 = Var(initialize=0.5, bounds=(-10, 10))
m.x2 = Var(initialize=0.5, bounds=(-10, 10))
m.circle = Constraint(expr=m.x1**2 + m.x2**2 == 1)
m.halfspace = Constraint(expr=m.x1 + m.x2 >= 0)
m.obj = Objective(expr=m.x1)
m.dual = Suffix(direction=Suffix.IMPORT)

res = pounce_browser.solve(m, options="print_level 5")

print(res)
print(f"x1 = {value(m.x1):.9f}   (exact: {-2**-0.5:.9f})")
print(f"x2 = {value(m.x2):.9f}")
print(f"dual[circle]    = {m.dual[m.circle]:.9f}")
print(f"dual[halfspace] = {m.dual[m.halfspace]:.9f}")
`,

  'HS071': `# Hock-Schittkowski 71 — the classic Ipopt example.
from pyomo.environ import *
import pounce_browser

m = ConcreteModel()
m.I = RangeSet(1, 4)
m.x = Var(m.I, bounds=(1, 5), initialize={1: 1, 2: 5, 3: 5, 4: 1})
m.obj = Objective(expr=m.x[1] * m.x[4] * (m.x[1] + m.x[2] + m.x[3]) + m.x[3])
m.c1 = Constraint(expr=m.x[1] * m.x[2] * m.x[3] * m.x[4] >= 25)
m.c2 = Constraint(expr=sum(m.x[i] ** 2 for i in m.I) == 40)

res = pounce_browser.solve(m, options="print_level 5\\ntol 1e-8")

print(res)
print("x* =", [round(value(m.x[i]), 9) for i in m.I])
print("known optimum: 17.0140173")
`,

  'Parameter estimation': `# Least-squares fit of a kinetic rate law to noisy data.
from pyomo.environ import *
import math, pounce_browser

# Synthetic data: y = A*exp(-k*t), A = 2.5, k = 0.8
A_true, k_true = 2.5, 0.8
ts = [0.1 * i for i in range(40)]
noise = [0.02 * math.sin(37.0 * t) for t in ts]          # deterministic "noise"
ys = [A_true * math.exp(-k_true * t) + n for t, n in zip(ts, noise)]

m = ConcreteModel()
m.A = Var(initialize=1.0, bounds=(0, 10))
m.k = Var(initialize=0.1, bounds=(0, 10))
m.obj = Objective(
    expr=sum((m.A * exp(-m.k * t) - y) ** 2 for t, y in zip(ts, ys))
)

res = pounce_browser.solve(m, options="print_level 0")

print(res)
print(f"A = {value(m.A):.6f}  (true {A_true})")
print(f"k = {value(m.k):.6f}  (true {k_true})")
`,

  'Larger: 2,000 variables': `# A separable NLP big enough to feel the solver work.
from pyomo.environ import *
import pounce_browser

N = 1000
m = ConcreteModel()
m.I = RangeSet(1, N)
m.x = Var(m.I, initialize=0.5, bounds=(0.01, 10))
m.y = Var(m.I, initialize=0.5, bounds=(0.01, 10))
m.link = Constraint(m.I, rule=lambda m, i: m.x[i] * m.y[i] == 1.0)
m.obj = Objective(expr=sum(m.x[i] ** 2 + 0.5 * m.y[i] ** 2 for i in m.I))

res = pounce_browser.solve(m, options="print_level 5")

print(res)
print(f"x[1] = {value(m.x[1]):.6f}, y[1] = {value(m.y[1]):.6f}")
print("each pair solves x^2 + 0.5/x^2 -> x = (0.5)**0.25 =", 0.5 ** 0.25)
`,
};

const select = $('example');
for (const name of Object.keys(EXAMPLES)) {
  const option = document.createElement('option');
  option.value = name;
  option.textContent = name;
  select.append(option);
}
select.addEventListener('change', () => {
  codeBox.value = EXAMPLES[select.value];
});
codeBox.value = EXAMPLES[select.value];

// --- worker ----------------------------------------------------------------

// The worker inherits this page's query string, so ?pyodide= / ?pyomo=
// overrides for a self-hosted deployment reach it.
const worker = new Worker(`./worker.js${self.location.search}`, { type: 'module' });

let ready = false;

worker.onmessage = ({ data }) => {
  if (data.type === 'status') {
    setStatus(data.text);
  } else if (data.type === 'ready') {
    ready = true;
    runButton.disabled = false;
    setStatus('ready — Python and POUNCE are loaded');
  } else if (data.type === 'stdout') {
    append(data.text);
  } else if (data.type === 'running') {
    outBox.textContent = '';
    setStatus('running…');
  } else if (data.type === 'done') {
    runButton.disabled = false;
    setStatus(`done in ${(data.ms / 1000).toFixed(2)} s`);
  } else if (data.type === 'error') {
    runButton.disabled = false;
    append(`\n${data.message}\n`);
    setStatus('the script raised an exception', true);
  } else if (data.type === 'fatal') {
    runButton.disabled = true;
    setStatus(data.message, true);
  }
};

function setStatus(text, isError = false) {
  statusLine.textContent = text;
  statusLine.classList.toggle('err', isError);
}

function append(text) {
  outBox.textContent += text;
  outBox.scrollTop = outBox.scrollHeight;
}

function run() {
  if (!ready) return;
  runButton.disabled = true;
  worker.postMessage({ type: 'run', code: codeBox.value });
}

runButton.addEventListener('click', run);
codeBox.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault();
    run();
  }
});

$('dl-code').addEventListener('click', () => {
  const url = URL.createObjectURL(new Blob([codeBox.value], { type: 'text/x-python' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = 'pounce_model.py';
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
});

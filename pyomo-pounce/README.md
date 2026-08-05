# pyomo-pounce

Pyomo solver plugin for [POUNCE](https://github.com/jkitchin/pounce), a
pure-Rust interior-point NLP solver (a Rust port of IPOPT).

POUNCE speaks the AMPL NL/SOL protocol, so Pyomo drives it through the
AMPL Solver Library interface — exactly how Pyomo integrates with IPOPT.

## Installation

```bash
pip install pyomo-pounce
```

That single command pulls in the `pounce-solver` dependency, which
ships a per-platform wheel bundling the `pounce` executable. After
install, `pounce` is on your `PATH` and Pyomo finds it automatically.

## Usage

```python
import pyomo_pounce  # registers the solver — REQUIRED before SolverFactory('pounce')
from pyomo.environ import *

model = ConcreteModel()
model.x = Var(initialize=0.5)
model.obj = Objective(expr=(model.x - 2)**2)

solver = SolverFactory('pounce')
result = solver.solve(model, tee=True)
print(f"x* = {value(model.x)}")  # 2.0
```

> **`import pyomo_pounce` is required.** Without it, `SolverFactory('pounce')`
> raises a clear `UnknownSolver` / "plugin not registered" error — it does not
> silently run some other `pounce`. With it imported, the plugin runs the
> `pounce` binary **bundled in the `pounce-solver` wheel**, independent of
> `PATH`. Only a source/dev install without that wheel falls back to a `pounce`
> on `PATH`, in which case the plugin warns.
>
> To see exactly which binary will run (and whether a stale or unrelated
> `pounce` earlier on `PATH` would shadow it), call:
>
> ```python
> import pyomo_pounce
> pyomo_pounce.check_binary()
> ```
>
> The check compares the git **commit** embedded in `pounce --about`, not the
> version string — two builds can share the same `X.Y.Z` while differing in
> behavior (as a binary from before/after a fix does).

## Solver Options

Pass options the same way as IPOPT:

```python
solver = SolverFactory('pounce')
solver.options['max_iter'] = 1000
solver.options['tol'] = 1e-10
solver.options['print_level'] = 5
```

Options are forwarded to POUNCE's `OptionsList` (ipopt.opt-compatible
keys).

## User scaling

The standard `scaling_factor` Suffix works as it does with IPOPT:

```python
model.scaling_factor = Suffix(direction=Suffix.EXPORT)
model.scaling_factor[model.obj] = 1e-3
model.scaling_factor[model.mass_balance] = 1e2

solver.solve(model, options={'nlp_scaling_method': 'user-scaling'})
```

Both halves are needed — the Suffix alone is inert (it also drives
Pyomo's own `core.scale_model`), and the option alone has nothing to
apply, which pyomo-pounce warns about. Untagged components are
unscaled; inactive constraints and fixed variables are skipped.

POUNCE models objective and constraint scaling only, so a
`scaling_factor` on a `Var` **raises** rather than being silently
dropped ([#483](https://github.com/jkitchin/pounce/issues/483)) —
rescale those variables in the model instead.

## Local development / unsupported platforms

If `pounce-solver` does not ship a wheel for your platform, the pip
install fails on the dependency. Two workarounds:

1. **Build POUNCE from source and put it on `PATH`** — the plugin
   resolves `pounce` via `shutil.which`, so any binary on `PATH`
   works:

   ```bash
   # in the pounce repo
   cargo build --release --bin pounce
   export PATH="$PWD/target/release:$PATH"
   pip install --no-deps pyomo-pounce pyomo
   ```

2. **Install `pounce-solver` from source** via maturin:

   ```bash
   cd pounce/python && maturin develop --release
   # then `cargo install --path ../crates/pounce-cli` to get the CLI
   # since maturin develop does not bundle the binary.
   pip install pyomo-pounce
   ```

## Running the tests locally

Which binary the tests exercise depends on whether a *bundled* one is
present: the plugin prefers `pounce/bin/pounce` inside the installed
`pounce-solver` package and falls back to `PATH` only when that is absent.
Neither setup above bundles anything, so a plain source checkout takes the
fallback path while CI takes the bundled one. Tests that describe the
bundled arrangement then skip, and the suite you run locally is not the
suite CI runs.

To match CI, stage the freshly built CLI into the package *before* building
the wheel — this is what `.github/workflows/ci.yml` does:

```bash
cargo build --release --bin pounce
mkdir -p python/pounce/bin
cp target/release/pounce python/pounce/bin/pounce   # the step that bundles it
(cd python && maturin build --release --out dist)
pip install python/dist/*.whl                       # pounce module + bundled CLI
pip install --no-deps -e pyomo-pounce
pytest pyomo-pounce/tests -q
```

`networkx` and `scipy` are needed for `test_block_init.py` and
`test_repair.py`; without them those two files skip.

If a test involving duals, multipliers, or bound reduced costs fails, check
the binary before the code:

```python
import pyomo_pounce; pyomo_pounce.check_binary()
```

It reports which executable will actually run, its build *commit*, and
whether anything on `PATH` shadows it. Two builds can share a version
string while differing in commit, so a stale binary reports a plausible
`X.Y.Z` while returning pre-fix results — that is the failure mode gh #315
added `check_binary()` for, and the one gh #366 turned out to be about.

## License

EPL-2.0, same as POUNCE.

# CasADi `Nlpsol` plugin for POUNCE

Builds `libcasadi_nlpsol_pounce.so` (`.dylib` on macOS), which registers
POUNCE as a CasADi NLP solver. Once it is on CasADi's plugin search path:

```python
import casadi as ca
solver = ca.nlpsol("solver", "pounce", nlp, {"pounce": {"tol": 1e-9}})
# …or, with Opti:
opti.solver("pounce", {}, {"tol": 1e-9})
```

User-facing documentation, including the option reference and worked
examples, is in [CasADi integration](../docs/src/casadi.md). This file
covers building and installing the plugin.

## Files

- `casadi_nlpsol_pounce.cpp` — the plugin. A `casadi::Nlpsol` subclass
  that wires CasADi's oracle functions into POUNCE through `pounce.h`,
  the Ipopt-3.14-compatible C API exported by `libpounce_cinterface`.
  Registration entry point: `casadi_register_nlpsol_pounce`.
- `Makefile` — build, install, test, run examples.
- `test_parity.py` — 73 checks cross-referencing POUNCE against CasADi's
  bundled `ipopt` plugin on the same models: primal, both multiplier sets and
  `lam_p`, solution-map derivatives and the bounded-variable gain trap,
  `Opti`, stats, live iteration callbacks and their throttle, diagnostics read
  from inside a callback, the final KKT errors, the linear-solver and
  restoration post-mortems, option typing, warm starts and the working-set
  carry, L-BFGS masks, user-supplied derivative functions, convexification, a
  save/load round trip, code generation (compiled and run against the
  interpreted solve), exception safety, and a threaded map. Run in CI by the
  `CasADi plugin parity` job.
- `pounce_runtime.hpp` — the C the plugin emits into generated code, the
  counterpart of CasADi's `ipopt_runtime.hpp`.
- `convexify_compat.hpp` — one shim, for one CasADi rename. See
  *Source compatibility across CasADi versions* below.
- `tests/convexify_compat/` — compiles that shim against every spelling,
  with no CasADi installed. `make compat`, and the `CasADi convexify name
  shim` job in CI.
- `examples/` — eight runnable scripts, from hello-world to embedded codegen.

## Build

You need three things: a C++ compiler, POUNCE's C library, and a CasADi
**source tree matching the installed CasADi version** (the pip wheel
ships only public headers — the internal ones a plugin subclasses are
not in it).

```bash
pip install casadi
cargo build --release -p pounce-cinterface   # from the repo root
cd casadi
make fetch-src        # clones casadi at exactly your installed version
make
```

`make fetch-src` clones into `casadi-src/`. If you already have a source
tree, point at it instead: `make CASADI_SRC=/path/to/casadi`.

### Against a CasADi you built yourself, with no Python

The defaults above read the installed *Python* CasADi, but nothing in
the build requires Python — every input is overridable, so a CI that
builds CasADi from source can build the plugin against it directly
(gh#634):

```bash
cargo build --release -p pounce-cinterface

make -C casadi \
  CASADI_LIB=/opt/casadi/lib \        # holds libcasadi.so
  CASADI_INC=/opt/casadi/include \    # public headers
  CASADI_SRC=/src/casadi \            # internal headers — see below
  CASADI_VER=3.7.2 \                  # no Python to ask, so say it
  CXX11_ABI=1
```

Four things to get right:

1. **`CASADI_SRC` is an include root, not a source tree per se** — it
   has to be the directory *containing* `casadi/core/nlpsol_impl.hpp`.
   CasADi's `INSTALL_INTERNAL_HEADERS` option defaults to **OFF**, so a
   stock `make install` of CasADi does *not* give you the internal
   headers a plugin subclasses. Either point `CASADI_SRC` at your CasADi
   source checkout (its repo root), or configure CasADi with
   `-DINSTALL_INTERNAL_HEADERS=ON` and point both `CASADI_SRC` and
   `CASADI_INC` at `<prefix>/include`.
2. **`CASADI_VER` must be set explicitly.** It normally comes from
   `casadi.__version__`, and with no Python it is empty. `check-env`
   stops the build and says so; left to run, it used to die deep in the
   plugin source on `expected primary-expression before ';'`, which
   names neither the option nor the cause.
3. **`CXX11_ABI=1` for a self-built CasADi.** The default is `0` to
   match the pip wheels' pre-C++11 libstdc++ string ABI. Getting this
   wrong shows up as an undefined-symbol error at link time — see
   [ABI](#abi-what-has-to-match-and-why).
4. `make abi-flags` prints the `-D` set the CasADi you are pointing at
   was actually built with, for comparison with `DEFS` in the Makefile.

`make install` targets the Python package's directory, so it does not
apply here: put the plugin on `CASADIPATH`, or beside the CasADi runtime
that will load it, and make `libpounce_cinterface` visible to the loader
(the build already adds both `-rpath` entries).

`make test` and `make examples` drive the plugin through CasADi's Python
bindings, so they run only on the pip-installed path. CI builds and
tests that path only; this one is verified by hand, most recently
against casadi 3.7.2 on Linux x86-64.

## Install

CasADi's plugin loader searches its own package directory first, so the
plugin only has to land there — no environment variable, and no `sudo`,
because that is the user's `site-packages`:

```bash
make install       # copies the plugin next to the casadi that will load it
```

For a system-wide CasADi you cannot write to, keep the plugin where it
is and point CasADi at it:

```bash
export CASADIPATH=/path/to/pounce/casadi
```

Uninstall with `make uninstall`.

## Verify

```bash
make test          # parity checks against ipopt — all should PASS
make examples      # runs every script in examples/
make compat        # the convexify name shim, against every CasADi spelling
```

`make compat` is the odd one out: it needs neither CasADi nor the solver
library, only a compiler. See below for what it is checking.

## Source compatibility across CasADi versions

CasADi renamed the runtime helper `convexify_eval` to
`casadi_convexify_eval` after 3.7.2 (gh#668). The codegen SYMBOL name and
the signature are unchanged, so the break is source-level only — but it is
unconditional: the call sits in `cb_h`, which compiles whether or not
`convexify_strategy` is ever used. A plugin built against current CasADi
master fails for users who never convexify anything.

Nothing in the version macros separates the two spellings — the nightly
that first carried the rename reports the same
`CASADI_MAJOR/MINOR/PATCH` as 3.7.2 — so `convexify_compat.hpp` detects
which name the installed CasADi declares, by overload resolution. One
source tree builds against 3.6, 3.7 and master with no build flags and no
version test to revise at the next CasADi release.

The catch, and the reason `tests/convexify_compat/` exists: a real CasADi
declares exactly one of the two names, so building against one CasADi
compiles one overload of the shim and leaves the other unchecked — and it
is the unchecked one that most users get, since the wheel ships plugins
for 3.6 and 3.7 and both take the fallback. `make compat` compiles both
against mock declarations, so neither branch can rot unnoticed. Checked
locally on g++ and clang; the MSVC leg — the toolchain the Windows wheel
is built with, and the likeliest of the three to disagree about a
dependent name — runs in CI. Add a compiler to `run.py` rather than
trusting the mechanism on a new one.

## ABI: what has to match, and why

CasADi's plugin loader does **no** version handshake — it does not
compare the plugin's `CASADI_VERSION` against its own. A plugin built
against the wrong CasADi will either fail to load with an
undefined-symbol error or, worse, load and misbehave. Three things have
to line up:

1. **CasADi version.** Rebuild the plugin for each CasADi minor version.
   The Makefile derives `CASADI_{MAJOR,MINOR,PATCH}_VERSION` from the
   installed package, so a mismatched `CASADI_SRC` usually shows up as a
   compile or link error rather than at runtime.
2. **libstdc++ string ABI.** The pip wheels are built with the
   pre-C++11 ABI, so the plugin is compiled `-D_GLIBCXX_USE_CXX11_ABI=0`
   by default. A CasADi you built yourself probably wants
   `make CXX11_ABI=1`. Getting this wrong looks like:

   ```
   undefined symbol: _ZNK6casadi16FunctionInternal16generate_options...
   ```
3. **The `-D` set.** `-DWITH_DL` is required to compile at all, and the
   rest affect struct layouts. `make abi-flags` prints the flags the
   installed CasADi was actually built with, for comparison with `DEFS`
   in the Makefile.

## Packaging

[`wheel/`](wheel/) packages the plugin as `pounce-casadi`, so that
installing it is all a user does:

```sh
cd wheel && ./build.sh          # builds for the installed casadi -> dist/*.whl
pip install dist/pounce_casadi-*.whl
```

```python
import casadi as ca
import pounce_casadi             # registers the plugin; nothing else needed
ca.nlpsol("solver", "pounce", nlp, {"pounce": {"tol": 1e-9}})
```

Importing it `dlopen`s the shipped plugin and calls its
`casadi_load_nlpsol_pounce` hook — the entry point CasADi's own loader
would call — so nothing is written into CasADi's installation, no
`CASADIPATH` is set, and CasADi's bundled plugins stay loadable
alongside. Nothing is published to PyPI yet; what a release build adds
(the manylinux / macOS / Windows matrix, one entry per CasADi minor
version) is in [`wheel/README.md`](wheel/README.md).

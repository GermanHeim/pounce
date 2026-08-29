# `pounce-casadi` — the plugin as a Python wheel

Packages the CasADi `nlpsol` plugin built in `../` so that installing it is
all a user does:

```sh
pip install pounce-casadi          # not published yet — see below
```

```python
import casadi as ca
import pounce_casadi               # registers the plugin

solver = ca.nlpsol("solver", "pounce", nlp, {"pounce": {"tol": 1e-9}})
```

Importing the package `dlopen`s the shipped plugin and calls its
`casadi_load_nlpsol_pounce` hook — the same entry point CasADi's own loader
calls after finding a plugin on its search path. Nothing is written into
CasADi's installation, no `CASADIPATH` is needed, and CasADi's bundled
plugins (Ipopt included) stay loadable side by side, which is what lets you
cross-check the two solvers in one process.

## Building it here

```sh
./build.sh          # builds the plugin for the installed casadi, writes dist/*.whl
```

Then, in a clean environment:

```sh
pip install casadi==<same minor> dist/pounce_casadi-*.whl
python -c "import casadi, pounce_casadi; print(casadi.nlpsol('S','pounce',
           {'x':casadi.MX.sym('x'),'f':casadi.MX.sym('x')**2}))"
```

## What a published wheel still needs

The plugin is a C++ extension of CasADi, so it is bound to a CasADi **minor**
version and to the platform's C++ ABI. CasADi performs no version handshake,
so this package refuses to guess: it ships one build per supported minor
version under `pounce_casadi/_plugins/<minor>/` and selects on
`casadi.__version__`, raising a clear `ImportError` when there is no match.

### The tag scheme

The two axes are not symmetric, and the wheel tag only carries one of them.

**Platform is in the tag.** The wheel is `py3-none-<platform>` —
`macosx_11_0_arm64`, `manylinux_2_28_x86_64`, `win_amd64`. Not `any`: it
carries a compiled plugin and the solver library, so a build for one
platform must not install on another, and `pip` is the right place to say
so. Not `cp311` either: nothing here links the CPython ABI — the payload is
loaded with `ctypes` — so one build per platform serves every Python 3, and
a CPython tag would strand users on the next Python release for no reason.
`setup.py` does both halves; `build.sh` and CI assert the result rather than
trust it, because the failure is silent at build time and only shows up on
a machine the packager does not have.

`POUNCE_CASADI_PLAT_NAME` overrides the platform half for a build whose
host is not its target: a manylinux tag (a raw `linux_x86_64` wheel is not
installable from PyPI, and `auditwheel repair` is the usual way to get one —
this is the escape hatch when it is not in the loop), or a macOS cross or
`universal2` build.

**CasADi minor is not in the tag, and cannot be.** CasADi is a runtime
dependency; there is no wheel tag that expresses "built against casadi
3.7.x". So it is resolved inside the wheel: one build per minor under
`_plugins/<minor>/`, selected on `casadi.__version__` at import, with a
plain `ImportError` when there is no match. One wheel per platform carries
every supported minor.

So the matrix collapses to: run `build.sh` once per (CasADi minor ×
platform), and per platform merge the staged `_plugins/<minor>/` trees into
a single wheel. Within one platform the runs share this tree and the staging
directory accumulates, so that is:

```sh
# in a platform's build image, once per supported casadi minor
pip install 'casadi==3.6.*' && POUNCE_CASADI_STAGE_ONLY=1 ./build.sh
pip install 'casadi==3.7.*' && POUNCE_CASADI_STAGE_ONLY=1 ./build.sh
pip install 'casadi==3.8.*' && POUNCE_CASADI_STAGE_ONLY=1 ./build.sh
./build.sh          # one wheel carrying all of them
```

The minors staged here and the `casadi` bound in `pyproject.toml` are one
statement made twice, and they have to agree. A minor inside the bound with
no `_plugins/<minor>/` staged installs cleanly and then fails at `import
pounce_casadi`, which is a worse failure than not resolving at all — pip has
already told the user it worked. 3.8 was added to both together (gh#782).

Per-platform build notes:

- **Linux** — inside a manylinux image, `-D_GLIBCXX_USE_CXX11_ABI` set to
  match the CasADi wheel being built against. Do not assume 0: casadi
  publishes both a `manylinux2014` build (old ABI) and, from 3.8.0, a
  `manylinux_2_28` build (new ABI) of the *same version*, and pip picks
  between them on the image's glibc (gh#782). The Makefile measures it from
  the installed `libcasadi.so` and `make -C casadi abi-flags` prints what it
  decided; a wheel staged against the wrong one links clean — a `-shared`
  link tolerates undefined symbols — and fails at the user's `dlopen` as
  `Plugin 'pounce' is not found`. Then `auditwheel` **excluding**
  `libcasadi` (it must resolve to the user's installed copy at runtime, not be
  vendored). `auditwheel repair` also does the manylinux retag, so
  `POUNCE_CASADI_PLAT_NAME` is not needed when it runs — but note it *refuses*
  a `py3-none-any` wheel outright, so before the tagging fix this step could
  not have run at all.
- **macOS** — x86_64 and arm64. The Makefile rewrites the plugin's reference
  to `libpounce_cinterface` to `@rpath` and adds `@loader_path`, because Rust
  stamps a cdylib's install name with its absolute *build* path; without that
  the staged plugin loads only on the build machine. CI asserts this on the
  `macos-latest` leg of the `CasADi plugin parity` job — `otool -L` must read
  `@rpath/libpounce_cinterface.dylib`, never an absolute path — and then hides
  the build tree and re-solves through the installed wheel, since a plugin
  that only *appears* relocatable passes every other test on the build
  machine.

  Watch the platform tag on macOS: it takes its OS version from
  `MACOSX_DEPLOYMENT_TARGET`, which defaults to the *building* machine's.
  A build on macOS 26 tags `macosx_26_0_arm64` and pip then declines it on
  every older macOS — a wheel that is correct and almost universally
  uninstallable. Set `MACOSX_DEPLOYMENT_TARGET=11.0` (or
  `POUNCE_CASADI_PLAT_NAME`) for a release build. CI does not catch this:
  its runner and its test environment are the same machine, so the tag is
  always satisfiable there.
- **Windows** — MSVC, matching CasADi's own toolchain.

CasADi minor releases are infrequent (3.6 in 2023, 3.7 in 2025), so the matrix
is small and mostly static. It is also exactly the maintenance that disappears
if the interface is contributed upstream — see
[`dev-notes/casadi-interface-options.md`](../../dev-notes/casadi-interface-options.md).

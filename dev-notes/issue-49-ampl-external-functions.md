# issue #49 — AMPL imported (external) functions

Status: implemented in `crates/pounce-cli` on
`worktree-issue-49-external-functions`.

## What this is

AMPL `.nl` files can declare imported (a.k.a. external) functions:

```
F0 1 -1 vf_hp        # F-segment: id=0, kind=1 (real), nargs=-1 (variadic), name
...
f0 4                 # expression node: call function id 0 with 4 args
h3:h2o               # string-literal arg (h<len>:<chars>)
v3                   # real arg
v4                   # real arg
h126:/path/to/...    # second string-literal arg
```

The function is implemented in a shared library named via the
`AMPLFUNC` environment variable (newline-separated list). The
library exposes a `funcadd_ASL(AmplExports*)` symbol that registers
each function through an `Addfunc` callback; pounce then calls back
with `Arglist` structs requesting value-only, value+gradient, or
value+gradient+packed-upper-Hessian.

Before this change pounce failed at the parser with
`Unknown expression token: 'f0 4'` on any such file.

## Design

Two new modules / types:

- `crates/pounce-cli/src/nl_external.rs` — libloading wrapper around
  the `funcadd_ASL` ABI. `ExternalLibrary` owns the loaded library
  and a heap-pinned `AmplExports`; `eval(name, args, want_derivs,
  want_hes)` is the single entry point and is serialised on a
  process-wide `Mutex` (AMPL's ABI is not thread-safe). The
  `ExternalResolver` maps funcall ids to `(Arc<ExternalLibrary>,
  String)` pairs.

- `nl_reader::Expr::Funcall { id, args: Vec<FuncallArg> }` —
  expression node. `FuncallArg::Real(Expr)` is a sub-expression;
  `FuncallArg::Str(String)` is an inline literal.

- `nl_tape::TapeOp::Funcall { lib: Arc<ExternalLibrary>, name,
  args: Vec<TapeFuncallArg> }` — tape op. `TapeFuncallArg::Tape(usize)`
  references a tape slot; `TapeFuncallArg::Str(String)` is the literal
  carried through.

### NlTnlp::new wiring

`NlTnlp::new` walks every nonlinear expression (`obj_nonlinear` plus
each constraint's `con_nonlinear`) collecting referenced funcall ids,
calls `ExternalResolver::build_for_problem`, then uses
`Tape::build_with_externals` instead of `Tape::build`. If no funcall
ids are referenced the resolver is empty and the path collapses to
the old `Tape::build` behaviour (zero overhead).

### AD sweeps with `Funcall`

For `F: R^nr → R` with `p_k = ∂F/∂ra[k]` and packed Hessian
`H_kl = ∂²F/∂ra[k]∂ra[l]`:

- **forward**: `vals[i] = F(call_args)`. Library called with
  `want_derivs=false, want_hes=false`.
- **reverse**: `adj[arg_k] += a * p_k`. Library called with
  `want_derivs=true, want_hes=false`.
- **forward_tangent**: `dot[i] = sum_k p_k * dot[arg_k]`. Library
  called with `want_derivs=true`.
- **hessian_accumulate / hessian_directional** (forward-over-reverse):
  - `adj[arg_k] += w * p_k`
  - `adj_dot[arg_k] += wd * p_k + w * sum_l H_kl * dot[arg_l]`
  - Library called with `want_derivs=true, want_hes=true`. Packed
    indexing: `hes[lo + hi*(hi+1)/2]` with `lo=min(k,l)`,
    `hi=max(k,l)`.
- **hessian_sparsity**: emit `emit_self(union of arg var-sets)` —
  every pair of real args contributes a structural Hessian entry.

### Paths that don't support `Funcall`

These panic on `TapeOp::Funcall`, because they're alternative AD
routes not used by `NlTnlp::new`:

- `nl_tape::HybridTape` (per-summand local tapes + shared CSE
  prelude — partial-separability optimization).
- `nl_hessian_program::HessianProgram` (JIT-style flat program for
  per-color Hessian directional derivatives).

If those paths are ever wired into the main flow, they'll need the
same Funcall arms the basic `Tape` path now carries.

## Tested against

`crates/pounce-cli/tests/issue_49_external_funcs.rs` runs the
`pounce` binary on a real IDAES Helmholtz fixture (3 vars, 3
equality constraints, 3 imported functions: `vf_hp`, `h_liq_hp`,
`h_vap_hp`) with `AMPLFUNC` pointed at
`~/.idaes/bin/general_helmholtz_external.dylib`. Expectations:

1. Optimal exit (`Optimal Solution Found`) — exercises full
   forward + reverse + Hessian through the live library.
2. Failure without `AMPLFUNC` — non-zero exit, message names
   `AMPLFUNC` / external functions.

Test is skipped when the dylib isn't installed locally; the
ripopt-style minimal `myfunc`-on-x0 fixture remains a unit-level
parser smoke test.

## Open items

- Errors from the resolver currently surface as a `panic!` from
  `NlTnlp::new` rather than a clean `Result` returned by the CLI
  layer. Acceptable because the panic message is precise, but a
  follow-up to make this a graceful CLI exit would be nicer.
- Coverage is one library (IDAES Helmholtz). Adding a second
  smaller library (e.g. an `amplgsl`-style trig function) would
  catch ABI regressions sooner.

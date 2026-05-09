# POUNCE iterate-trace binary format (`POUNCEIT`)

This document specifies the binary record stream produced by the
patched upstream Ipopt (branch `pounce-iter-dump`) and by the POUNCE
Rust port. A bit-equivalence comparator can read both streams with the
same parser.

This is the **format-version-1** specification.

## High-level contract

* One file per `Optimize()` call. Restoration sub-solves do not
  produce records (they would interleave the parent file).
* The file starts with a fixed header (see "Header"), followed by
  zero-or-more iteration records (see "Iteration record").
* Records are emitted at two points:
  1. **Iter 0**: after `InitializeIterates()` and before the main
     loop starts. Captures the initial point.
  2. **After every `AcceptTrialPoint`** in the main loop, after
     `IpData().Set_iter_count(iter+1)`, so the recorded iteration
     index matches `IpData().iter_count()` at that moment.
* All multi-byte integers are **unsigned little-endian**. All
  floating-point values are **IEEE 754 binary64 little-endian** (raw
  in-memory bytes on x86_64 / aarch64). No padding, no alignment.
* Vectors are written as `u32 len` followed by `len * 8` raw bytes
  copied directly from `DenseVector::Values()`. A homogeneous
  `DenseVector` (all entries equal a scalar) is expanded to `len`
  copies of that scalar.

## Activation

The dumper is gated by the environment variable `IPOPT_ITER_DUMP_PATH`
(set to an absolute path of the file to write). If the variable is
unset or empty, the patched Ipopt is bit-identical in behavior and
output to upstream (no file is opened, no work is performed).

The optional environment variable `IPOPT_ITER_DUMP_NAME` provides the
problem-name string for the header. If unset, the name is empty.

> **POUNCE-side coverage gap (v1).** On the POUNCE side, the writer is
> hooked inside `IpoptAlgorithm::optimize` (port of upstream's
> `Optimize()`). Unconstrained problems currently solved by the lighter
> `newton_driver` path (e.g. ROSENBR through the CUTEst harness in
> v1.0) bypass `IpoptAlgorithm::optimize` and therefore produce no
> trace file even when `IPOPT_ITER_DUMP_PATH` is set. The dumper
> activates once a constrained problem (or any TNLP wired through
> `IpoptApplication::optimize_tnlp`) reaches the IPM main loop. This
> gap closes when task #54 routes all paths through the IPM driver.

## Header

| Field          | Type    | Bytes | Description                              |
|----------------|---------|-------|------------------------------------------|
| `magic`        | `[u8;8]`| 8     | ASCII `POUNCEIT` (`50 4F 55 4E 43 45 49 54`) |
| `format_version` | `u32` | 4     | Currently `1`                            |
| `n`            | `u32`   | 4     | `dim(x)` = number of primal variables    |
| `m`            | `u32`   | 4     | `dim(y_c) + dim(y_d)` = number of constraints |
| `nnz_jac`      | `u32`   | 4     | Constraint Jacobian nnz, or `0` if not surfaced |
| `nnz_h`        | `u32`   | 4     | Lagrangian Hessian nnz, or `0` if not surfaced |
| `name_len`     | `u32`   | 4     | Length of the UTF-8 problem name        |
| `name`         | `[u8;name_len]` | name_len | UTF-8 problem name (no NUL terminator) |

Total fixed header size: 32 bytes + `name_len`.

> **Note on `nnz_jac` / `nnz_h`.** Upstream Ipopt does not surface the
> structural nnz counts on the `IpoptNLP` strategy interface, so the
> patched Ipopt always writes `0` for both fields. Comparators must
> tolerate `0` here as "unknown". POUNCE-side dumpers SHOULD write the
> real values.

## Iteration record

Each record is a sequence of fields, written contiguously:

### Scalar block (84 bytes)

| Offset | Field            | Type  | Notes                                    |
|--------|------------------|-------|------------------------------------------|
| +0     | `iter`           | `u32` | `IpData().iter_count()` at write time    |
| +4     | `status`         | `u32` | Always `0` ("in progress") in v1         |
| +8     | `mu`             | `f64` | `IpData().curr_mu()`                     |
| +16    | `tau`            | `f64` | `IpData().curr_tau()`                    |
| +24    | `alpha_pr`       | `f64` | `IpData().info_alpha_primal()`           |
| +32    | `alpha_du`       | `f64` | `IpData().info_alpha_dual()`             |
| +40    | `delta_x`        | `f64` | `IpData().info_regu_x()`                 |
| +48    | `delta_s`        | `f64` | Always `0.0` in v1 (see note)            |
| +56    | `delta_c`        | `f64` | Always `0.0` in v1 (see note)            |
| +64    | `delta_d`        | `f64` | Always `0.0` in v1 (see note)            |
| +72    | `inf_pr`         | `f64` | `IpCq().curr_primal_infeasibility(NORM_MAX)` |
| +80    | `inf_du`         | `f64` | `IpCq().curr_dual_infeasibility(NORM_MAX)`   |
| +88    | `constr_viol`    | `f64` | `IpCq().curr_constraint_violation()`     |
| +96    | `dual_inf`       | `f64` | `IpCq().curr_dual_infeasibility(NORM_MAX)` (= `inf_du`) |
| +104   | `complementarity`| `f64` | `IpCq().curr_complementarity(0.0, NORM_MAX)` |
| +112   | `f`              | `f64` | `IpCq().curr_f()` (scaled objective)     |

Scalar block total: **120 bytes**. (`u32 + u32 + 14 * f64`)

> **`delta_s` / `delta_c` / `delta_d`.** The PD perturbation handler
> stores these internally with no public accessor on `IpoptData`. They
> are recorded as `0.0` in format v1. The POUNCE-side Rust port has
> direct access to all four perturbations and may write the real
> values; bit-equivalence comparators MUST therefore treat these three
> fields as advisory in v1 (only `delta_x` is authoritative on both
> sides). A future format v2 will fix this.

### Iterate vector block (8 vectors)

Eight vectors are written contiguously, in this exact order:

1. `x`   — primal variables, `dim = n`
2. `s`   — slacks, `dim = n_d`
3. `y_c` — equality multipliers, `dim = m_c`
4. `y_d` — inequality multipliers, `dim = m_d`
5. `z_L` — lower bound multipliers on `x`, `dim = n_xL`
6. `z_U` — upper bound multipliers on `x`, `dim = n_xU`
7. `v_L` — lower bound multipliers on `s`, `dim = n_sL`
8. `v_U` — upper bound multipliers on `s`, `dim = n_sU`

Each vector is encoded as:

| Field   | Type             | Bytes  |
|---------|------------------|--------|
| `len`   | `u32`            | 4      |
| `values`| `[f64; len]`     | 8*len  |

The bytes of `values` are a verbatim copy of the in-memory
`DenseVector::Values()` array. On x86_64 / aarch64-darwin this is
little-endian IEEE 754 binary64. On a hypothetical big-endian host the
patched Ipopt would still write little-endian bytes (the writer
canonicalises via `memcpy` of the raw bytes; on the only currently
supported targets these are already little-endian).

A homogeneous `DenseVector` (which Ipopt uses to represent a constant
vector compactly) is expanded: `len` copies of the scalar value are
written, so the on-disk representation is identical regardless of
storage strategy.

### Filter block

| Field          | Type | Bytes |
|----------------|------|-------|
| `filter_count` | `u32`| 4     |
| `filter_entries` | `[(f64, f64); filter_count]` | 16 * filter_count |

Each entry is `(theta, phi)` in that order. In format v1 the patched
Ipopt always writes `filter_count = 0` because the filter is owned by
`FilterLSAcceptor` and not exposed on the algorithm-object surface.
The POUNCE port may write the real filter; bit-equivalence comparators
should treat this field similarly to the perturbations (advisory in
v1).

## Total record size

```
record_size = 120                                     # scalar block
            + sum_{v in 8 vectors} (4 + 8 * dim(v))   # iterate block
            + 4 + 16 * filter_count                   # filter block
```

For hs071 (n=4, m_c=1 equality, m_d=1 inequality with one slack and
one lower-bound multiplier on that slack) the per-record size is:

```
120 + (4 + 8*4) + (4 + 8*1)      # x, s              -> 36, 12
    + (4 + 8*1) + (4 + 8*1)      # y_c, y_d          -> 12, 12
    + (4 + 8*4) + (4 + 8*4)      # z_L, z_U          -> 36, 36
    + (4 + 8*1) + (4 + 0)        # v_L, v_U          -> 12,  4
    + 4                          # filter_count = 0  ->  4
  per record = 120 + 36 + 12 + 12 + 12 + 36 + 36 + 12 + 4 + 4
             = 284 bytes
```

A 3-iteration trace plus a 37-byte header (32 fixed + 5-byte name
"hs071") gives `37 + 3*284 = 889 bytes`, which matches the observed
file size byte-for-byte.

## Worked example (hs071)

A 2-iteration trace of hs071 (n=4, m=2) starts with bytes:

```
50 4F 55 4E 43 45 49 54   "POUNCEIT"
01 00 00 00               format_version = 1
04 00 00 00               n  = 4
02 00 00 00               m  = 2
00 00 00 00               nnz_jac = 0  (not surfaced)
00 00 00 00               nnz_h   = 0  (not surfaced)
00 00 00 00               name_len = 0
                          (no name bytes)
                          --- end of header (32 bytes) ---
00 00 00 00               iter   = 0
00 00 00 00               status = 0
... 14 * 8 bytes of scalars ...
04 00 00 00 <32 bytes>    x   = [..]    (n=4)
01 00 00 00 <8  bytes>    s   = [..]    (1 slack for the inequality)
01 00 00 00 <8  bytes>    y_c = [..]    (m_c=1, equality)
01 00 00 00 <8  bytes>    y_d = [..]    (m_d=1, inequality g>=25)
04 00 00 00 <32 bytes>    z_L = [..]    (4 lower bounds on x)
04 00 00 00 <32 bytes>    z_U = [..]    (4 upper bounds on x)
01 00 00 00 <8  bytes>    v_L = [..]    (slack lower bound on g)
00 00 00 00               v_U = []
00 00 00 00               filter_count = 0
                          --- end of iter-0 record ---
01 00 00 00               iter = 1
... ditto ...
```

A reference Python parser is in `tools/iter-dump/dump_inspect.py`.

Real verified output of the worked example (3-record file, 889 bytes):

```
file: /tmp/hs071.iter (889 bytes)
header: version=1 n=4 m=2 nnz_jac=0 nnz_h=0 name='hs071'
records: 3
  iter=0 |x|=4 |y_c|=1 |y_d|=1 |z_L|=4 |z_U|=4 |v_L|=1 |v_U|=0
  iter=1 ...
  iter=2 ...
```

Note that for hs071 the "inequality constraint" `g_2(x) = x_1*x_2*x_3*x_4 >= 25`
appears as a single y_d entry (and one v_L slack), and there is no
y_d upper-bound multiplier (v_U is empty).

## Compatibility / parsing rules

* Parsers MUST verify the `magic` field exactly. Reject if mismatched.
* Parsers MUST refuse to read a file with a `format_version` newer
  than they support.
* Future versions will add fields by **bumping the version** and
  appending. Parsers should not assume the iteration-record size is a
  function of the header alone — they must walk the variable-length
  vector and filter sections.

## Field provenance summary (for the Rust port)

| Format field      | C++ accessor                                      |
|-------------------|---------------------------------------------------|
| `iter`            | `IpData().iter_count()`                           |
| `mu`              | `IpData().curr_mu()`                              |
| `tau`             | `IpData().curr_tau()`                             |
| `alpha_pr`        | `IpData().info_alpha_primal()`                    |
| `alpha_du`        | `IpData().info_alpha_dual()`                      |
| `delta_x`         | `IpData().info_regu_x()`                          |
| `inf_pr`          | `IpCq().curr_primal_infeasibility(NORM_MAX)`      |
| `inf_du`          | `IpCq().curr_dual_infeasibility(NORM_MAX)`        |
| `constr_viol`     | `IpCq().curr_constraint_violation()`              |
| `complementarity` | `IpCq().curr_complementarity(0.0, NORM_MAX)`      |
| `f`               | `IpCq().curr_f()`                                 |
| `x` … `v_U`       | `IpData().curr()->{x,s,y_c,y_d,z_L,z_U,v_L,v_U}()`|

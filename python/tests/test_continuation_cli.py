"""Path manifest / repeated-solve protocol for the CLI (pounce#608).

pounce#608's scope note says CLI and GAMS "can follow using a path
manifest or repeated-solve protocol". These cover the manifest parser,
the `.nl` initial-point rewrite that carries a solved point into the
next model, and -- when a `pounce` binary and the CLI fixture corpus
are both present -- one end-to-end trace.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

try:
    from pounce import _continuation_cli as _cc
except ImportError:
    # The driver is pure Python, but `import pounce` loads the compiled
    # extension. Without it (a source checkout that was never built), load
    # the module file directly so the format tests below still run.
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "_continuation_cli",
        os.path.join(_ROOT, "python", "pounce", "_continuation_cli.py"),
    )
    _cc = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_cc)

PathManifest = _cc.PathManifest
parse_sol = _cc.parse_sol
parse_sol_suffixes = _cc.parse_sol_suffixes
rewrite_nl_initial_point = _cc.rewrite_nl_initial_point
trace_manifest = _cc.trace_manifest
_FIXTURES = os.path.join(_ROOT, "crates", "pounce-cli", "tests", "fixtures")


def _works(path):
    """A path is only a usable CLI if it actually runs.

    `shutil.which("pounce")` finds the wheel's *console-script shim*,
    which is present whenever pounce-solver is installed but only
    execs a real binary when the wheel was built with `pounce/bin/`
    staged. CI builds the wheel straight from `maturin-action` with no
    staging step, so the shim is on PATH, exits 1, and every solve in
    a trace silently returns nothing — which is how this arrived as
    `assert 0 == 2` rather than as a skip.
    """
    if path is None:
        return False
    try:
        proc = subprocess.run([path, "--version"], capture_output=True,
                              text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _binary():
    local = os.path.join(_ROOT, "target", "release", "pounce")
    if os.path.exists(local):
        return local
    bundled = os.path.join(_ROOT, "python", "pounce", "bin", "pounce")
    if os.path.exists(bundled):
        return bundled
    return shutil.which("pounce")


_CLI = _binary() if _works(_binary()) else None

needs_cli = pytest.mark.skipif(
    _CLI is None or not os.path.isdir(_FIXTURES),
    reason="needs a *working* pounce binary (a wheel built without "
           "pounce/bin/ staged leaves only a shim that exits 1) and the "
           "CLI fixture corpus",
)


# -- manifest ----------------------------------------------------------


def _write(tmp_path, obj, name="path.json"):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return str(p)


def test_manifest_round_trips(tmp_path):
    man = PathManifest.load(_write(tmp_path, {
        "version": 1,
        "points": [{"model": "a.nl", "theta": [1.0]},
                   {"model": "b.nl"}],
        "options": {"tol": "1e-8"},
    }))
    assert len(man.points) == 2
    assert man.warm is True
    assert man.options == {"tol": "1e-8"}
    # Model paths resolve against the manifest, not the cwd, so a
    # manifest is movable with its models.
    assert os.path.dirname(man.points[0]["model"]) == str(tmp_path)
    assert man.points[0]["theta"] == [1.0]
    assert man.points[1]["theta"] is None


def test_manifest_rejects_a_future_version(tmp_path):
    with pytest.raises(ValueError, match="unsupported version"):
        PathManifest.load(_write(tmp_path, {"version": 2, "points": []}))


def test_manifest_rejects_a_point_without_a_model(tmp_path):
    with pytest.raises(ValueError, match="no 'model'"):
        PathManifest.load(_write(tmp_path, {
            "version": 1, "points": [{"theta": [1.0]}]}))


def test_manifest_rejects_an_empty_path(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        PathManifest.load(_write(tmp_path, {"version": 1, "points": []}))


# -- .nl initial-point rewrite ----------------------------------------


_NL = """g3 1 1 0	# problem unknown
 2 1 1 0 1 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs
 0 0	# network constraints
 0 2 0 	# nonlinear vars
 0 0 0 1	# linear network variables
 0 0 0 0 0 	# discrete variables
 2 2 	# nonzeros
 0 0	# max name lengths
 0 0 0 0 0	# common exprs
x2
0 0.0
1 0.0
C0
n0
"""


def _src(tmp_path, text=_NL):
    p = tmp_path / "m.nl"
    p.write_text(text)
    return str(p)


def test_rewrite_replaces_the_initial_point(tmp_path):
    dst = str(tmp_path / "out.nl")
    rewrite_nl_initial_point(_src(tmp_path), dst, primals=[1.5, -2.5],
                             duals=[0.25])
    lines = open(dst).read().split("\n")
    heads = [ln for ln in lines if ln[:1] in ("x", "d") and ln[1:2].isdigit()]
    # Exactly one of each -- the old x segment is replaced, not appended
    # to, or the reader would see two starting points.
    assert heads == ["x2", "d1"]
    assert "0 1.5" in lines and "1 -2.5" in lines
    assert "0 0.25" in lines
    # The model body is untouched.
    assert "C0" in lines and "n0" in lines


def test_rewrite_omits_the_dual_segment_when_no_duals_are_given(tmp_path):
    dst = str(tmp_path / "out.nl")
    rewrite_nl_initial_point(_src(tmp_path), dst, primals=[1.0, 2.0])
    heads = [ln for ln in open(dst).read().split("\n")
             if ln[:1] in ("x", "d") and ln[1:2].isdigit()]
    assert heads == ["x2"]


def test_rewrite_checks_the_lengths_against_the_model(tmp_path):
    dst = str(tmp_path / "out.nl")
    with pytest.raises(ValueError, match="2 variables"):
        rewrite_nl_initial_point(_src(tmp_path), dst, primals=[1.0])
    with pytest.raises(ValueError, match="1 constraints"):
        rewrite_nl_initial_point(_src(tmp_path), dst, primals=[1.0, 2.0],
                                 duals=[0.1, 0.2])


def _segment(lines, header):
    """The ``index value`` lines of the segment headed `header`."""
    h = lines.index(header)
    count = int(header.split()[1])
    return lines[h + 1:h + 1 + count]


def test_rewrite_writes_the_bound_multiplier_suffixes(tmp_path):
    """gh#1033: the bound multipliers ride the real variable suffixes
    `ipopt_zL_in` / `ipopt_zU_in` (`S4 <count> <name>`, then sparse
    `index value` lines), replacing any segment of those names the source
    already had and leaving every other suffix alone."""
    text = _NL + ("S4 2 ipopt_zL_in\n0 9.0\n1 9.0\n"
                  "S0 1 sosno\n1 3\n")
    dst = str(tmp_path / "out.nl")
    rewrite_nl_initial_point(_src(tmp_path, text), dst, primals=[1.0, 2.0],
                             z_l=[2.0, 0.0], z_u=[0.0, -4.0])
    lines = open(dst).read().split("\n")
    heads = [ln for ln in lines if ln[:1] == "S" and ln[1:2].isdigit()]
    # One segment per name: the stale `ipopt_zL_in` is gone, not shadowed,
    # and the unrelated integer suffix is copied through with its entry.
    assert sorted(heads) == ["S0 1 sosno", "S4 1 ipopt_zL_in",
                             "S4 1 ipopt_zU_in"]
    assert _segment(lines, "S0 1 sosno") == ["1 3"]
    # Zero entries are the suffix default, so they are left out.
    assert _segment(lines, "S4 1 ipopt_zL_in") == ["0 2.0"]
    assert _segment(lines, "S4 1 ipopt_zU_in") == ["1 -4.0"]
    assert "9.0" not in " ".join(lines)


def test_rewrite_leaves_existing_bound_suffixes_when_none_are_given(tmp_path):
    text = _NL + "S4 1 ipopt_zL_in\n0 9.0\n"
    dst = str(tmp_path / "out.nl")
    rewrite_nl_initial_point(_src(tmp_path, text), dst, primals=[1.0, 2.0])
    lines = open(dst).read().split("\n")
    assert _segment(lines, "S4 1 ipopt_zL_in") == ["0 9.0"]


def test_rewrite_checks_the_bound_multiplier_lengths(tmp_path):
    with pytest.raises(ValueError, match="ipopt_zU_in"):
        rewrite_nl_initial_point(_src(tmp_path), str(tmp_path / "o.nl"),
                                 z_l=[0.0, 0.0], z_u=[1.0])


def test_rewrite_rejects_a_binary_nl(tmp_path):
    p = tmp_path / "b.nl"
    p.write_text("b3 1 1 0\n 2 1 1 0 1\n")
    with pytest.raises(ValueError, match="ASCII"):
        rewrite_nl_initial_point(str(p), str(tmp_path / "o.nl"),
                                 primals=[1.0, 2.0])


# -- .sol parsing ------------------------------------------------------


def test_parse_sol_splits_duals_from_primals(tmp_path):
    p = tmp_path / "s.sol"
    p.write_text("POUNCE: done\n\nOptions\n3\n0\n1\n0\n2\n2\n3\n3\n"
                 "10.0\n11.0\n1.0\n2.0\n3.0\n")
    duals, primals = parse_sol(str(p), 3, 2)
    assert duals == [10.0, 11.0]
    assert primals == [1.0, 2.0, 3.0]


def test_parse_sol_suffixes_reads_sparse_blocks(tmp_path):
    p = tmp_path / "s.sol"
    p.write_text("POUNCE: done\n\nOptions\n3\n0\n1\n0\n0\n0\n2\n2\n"
                 "1.0\n2.0\nobjno 0 0\n"
                 "suffix 4 1 13 0 0\nipopt_zL_out\n0 2.5e0\n"
                 "suffix 4 1 13 0 0\nipopt_zU_out\n1 -4.0e0\n")
    sfx = parse_sol_suffixes(str(p))
    assert sfx == {"ipopt_zL_out": {0: 2.5}, "ipopt_zU_out": {1: -4.0}}


def test_parse_sol_rejects_a_non_sol_file(tmp_path):
    p = tmp_path / "s.sol"
    p.write_text("not a sol file\n")
    with pytest.raises(ValueError, match="Options"):
        parse_sol(str(p), 1, 1)


# -- end to end --------------------------------------------------------


@needs_cli
def test_trace_manifest_solves_every_point(tmp_path):
    """A two-point path over the same model is the degenerate
    continuation: the second point starts from the first's answer."""
    src = os.path.join(_FIXTURES, "boxed_qp_min.nl")
    if not os.path.exists(src):
        pytest.skip("fixture not present")
    for name in ("p0.nl", "p1.nl"):
        shutil.copy(src, tmp_path / name)
    man = PathManifest.load(_write(tmp_path, {
        "version": 1,
        "points": [{"model": "p0.nl", "theta": [0.0]},
                   {"model": "p1.nl", "theta": [0.0]}],
        "options": {"print_level": "0", "sb": "yes"},
    }))
    trace = trace_manifest(man, binary=_CLI, workdir=str(tmp_path))
    assert trace["n_points"] == 2
    assert trace["n_converged"] == 2
    assert all(s["status"] in (0, 1) for s in trace["steps"])
    # The predictor is reported as absent rather than silently implied:
    # the KKT factor does not cross a process boundary.
    assert "process boundary" in trace["predictor"]
    assert trace["steps"][0]["predictor"] == "cold"
    assert trace["steps"][1]["predictor"] == "zero"


@needs_cli
def test_warm_transfer_writes_duals_in_the_sol_sign(tmp_path):
    """gh#1033: a `.nl` `d` segment is in AMPL's dual sign, the sign of a
    `.sol` dual block, and the reader negates it into pounce's multiplier.
    The trace takes the previous point's duals from the JSON report, in
    pounce's sign, so it must negate them on the way into the next `.nl`:
    the transferred `d` entry equals the previous `.sol` dual.
    `convex_qp.nl` has one equality with a nonzero multiplier (pounce
    lambda = -2, AMPL dual = +2), so a sign slip cannot hide at zero."""
    src = os.path.join(_FIXTURES, "convex_qp.nl")
    if not os.path.exists(src):
        pytest.skip("fixture not present")
    for name in ("p0.nl", "p1.nl"):
        shutil.copy(src, tmp_path / name)
    man = PathManifest.load(_write(tmp_path, {
        "version": 1,
        "points": [{"model": "p0.nl"}, {"model": "p1.nl"}],
        "options": {"print_level": "0", "sb": "yes",
                    "solver_selection": "nlp"},
    }))
    trace = trace_manifest(man, binary=_CLI, workdir=str(tmp_path))
    assert trace["n_converged"] == 2
    sol_duals, _ = parse_sol(str(tmp_path / "step0000.sol"), 1, 2)
    lines = (tmp_path / "step0001.nl").read_text().split("\n")
    d = lines.index("d1")
    transferred = float(lines[d + 1].split()[1])
    assert sol_duals[0] == pytest.approx(2.0, abs=1e-6)
    assert transferred == pytest.approx(sol_duals[0], abs=1e-12)


def _warm_bound_transfer(tmp_path, fixture, options):
    """Trace `fixture` twice and return ``(zL_out, zU_out)`` of the first
    point's `.sol` and ``(zL_in, zU_in)`` of the second point's `.nl`, all
    dense."""
    src = os.path.join(_FIXTURES, fixture)
    if not os.path.exists(src):
        pytest.skip("fixture not present")
    for name in ("p0.nl", "p1.nl"):
        shutil.copy(src, tmp_path / name)
    man = PathManifest.load(_write(tmp_path, {
        "version": 1,
        "points": [{"model": "p0.nl"}, {"model": "p1.nl"}],
        "options": options,
    }))
    trace = trace_manifest(man, binary=_CLI, workdir=str(tmp_path))
    assert trace["n_converged"] == 2
    lines = (tmp_path / "step0001.nl").read_text().split("\n")
    n = int(lines[1].split()[0])
    sol = parse_sol_suffixes(str(tmp_path / "step0000.sol"))

    def dense(entries):
        v = [0.0] * n
        for k, x in entries.items():
            v[k] = x
        return v

    def seg(name):
        head = [ln for ln in lines if ln.startswith("S4 ")
                and ln.split()[2] == name]
        assert len(head) == 1, f"expected one {name} segment: {head}"
        return dense({int(a): float(b) for a, b in
                      (ln.split() for ln in _segment(lines, head[0]))})

    return ((dense(sol["ipopt_zL_out"]), dense(sol["ipopt_zU_out"])),
            (seg("ipopt_zL_in"), seg("ipopt_zU_in")))


@needs_cli
def test_warm_transfer_writes_bound_multipliers_minimize(tmp_path):
    """gh#1033: the previous point's bound multipliers reach the next
    model as `ipopt_zL_in` / `ipopt_zU_in`, equal to the previous `.sol`'s
    `ipopt_zL_out` / `ipopt_zU_out` -- the two pairs share a sign, so a
    sign slip in the transfer shows as a mismatch. `boxed_qp_min.nl` is
    `min (x - 3)^2` on `[0, 1]`: the upper bound is active with
    `df/dx = -4`, so `ipopt_zU_out = -4` and cannot hide at zero."""
    (zl_out, zu_out), (zl_in, zu_in) = _warm_bound_transfer(
        tmp_path, "boxed_qp_min.nl",
        {"print_level": "0", "sb": "yes", "solver_selection": "nlp"})
    assert zu_out[0] == pytest.approx(-4.0, abs=1e-5)
    assert zl_in == pytest.approx(zl_out, abs=1e-12)
    assert zu_in == pytest.approx(zu_out, abs=1e-12)


@needs_cli
def test_warm_transfer_writes_bound_multipliers_maximize(tmp_path):
    """The maximize twin: `wyndor_max_boxed.nl` is `max 3x0 + 5x1` with
    `x0 <= 1.5` active and `df/dx0 = 3`, so `ipopt_zU_out[0] = +3` (gh#959)
    and the transfer must carry it unchanged -- the reader applies the
    sense, not the driver."""
    (zl_out, zu_out), (zl_in, zu_in) = _warm_bound_transfer(
        tmp_path, "wyndor_max_boxed.nl",
        {"print_level": "0", "sb": "yes", "solver_selection": "nlp"})
    assert zu_out[0] == pytest.approx(3.0, abs=1e-5)
    assert zl_in == pytest.approx(zl_out, abs=1e-12)
    assert zu_in == pytest.approx(zu_out, abs=1e-12)


@needs_cli
def test_cold_mode_does_not_transfer(tmp_path):
    src = os.path.join(_FIXTURES, "boxed_qp_min.nl")
    if not os.path.exists(src):
        pytest.skip("fixture not present")
    for name in ("p0.nl", "p1.nl"):
        shutil.copy(src, tmp_path / name)
    man = PathManifest.load(_write(tmp_path, {
        "version": 1,
        "points": [{"model": "p0.nl"}, {"model": "p1.nl"}],
        "options": {"print_level": "0", "sb": "yes"},
    }))
    trace = trace_manifest(man, binary=_CLI, cold=True,
                           workdir=str(tmp_path))
    assert trace["mode"] == "cold"
    assert [s["predictor"] for s in trace["steps"]] == ["cold", "cold"]


@needs_cli
def test_cli_entry_point_runs(tmp_path):
    src = os.path.join(_FIXTURES, "boxed_qp_min.nl")
    if not os.path.exists(src):
        pytest.skip("fixture not present")
    shutil.copy(src, tmp_path / "p0.nl")
    man = _write(tmp_path, {
        "version": 1, "points": [{"model": "p0.nl"}],
        "options": {"print_level": "0", "sb": "yes"},
    })
    out = str(tmp_path / "trace.json")
    proc = subprocess.run(
        [sys.executable, "-m", "pounce._continuation_cli", man,
         "--out", out, "--binary", _CLI],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    with open(out) as fh:
        trace = json.load(fh)
    assert trace["n_points"] == 1

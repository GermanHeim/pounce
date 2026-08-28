"""`available()` must never be a bare False (gh #816).

The report: ``SolverFactory("pounce").available(exception_flag=False)``
returned False on an installation where ``solve()`` returned ``optimal``.
Both halves were true, because they took different routes to a binary --
``solve()`` through the plugin's bundled-path resolution, ``available()``
through Pyomo's ASL layer running ``pounce -v`` on whatever PATH offered,
which in a ``maturin develop`` checkout is a console-script shim pointing
at a binary nobody had built.

The route is fixed elsewhere (the shim now finds the cargo build, and this
plugin resolves it directly). What is pinned here is the part that made the
report hard to file at all: a False that says which executable it ran and
what running it printed.
"""

import warnings

import pytest
from pyomo.common.errors import ApplicationError
from pyomo.opt import SolverFactory

import pyomo_pounce  # noqa: F401  (registers the `pounce` solver)
import pyomo_pounce.pounce_solver as ps


@pytest.fixture(autouse=True)
def _reset_warn_state(monkeypatch):
    """The plugin's warnings are one-shot per process; these tests each want
    to see their own."""
    monkeypatch.setattr(ps, "_unavailable_warned", set())
    monkeypatch.setattr(ps, "_checkout_warned", False)
    monkeypatch.setattr(ps, "_fallback_warned", False)


def _broken_shim(tmp_path):
    """The pre-fix console script: exits 1 with the bundled-binary error."""
    exe = tmp_path / "pounce"
    exe.write_text(
        "#!/bin/sh\n"
        "echo 'pounce: bundled CLI binary not found at "
        "/repo/python/pounce/bin/pounce.' >&2\n"
        "exit 1\n"
    )
    exe.chmod(0o755)
    return str(exe)


def _working_binary(tmp_path):
    exe = tmp_path / "pounce"
    exe.write_text("#!/bin/sh\necho 'pounce 0.10.0'\n")
    exe.chmod(0o755)
    return str(exe)


def _solver(monkeypatch, exe):
    monkeypatch.setattr(ps, "_bundled_path", lambda: exe)
    monkeypatch.setattr(ps, "_checkout_path", lambda: None)
    return SolverFactory("pounce")


def test_a_working_binary_is_available(monkeypatch, tmp_path):
    s = _solver(monkeypatch, _working_binary(tmp_path))
    assert s.available(exception_flag=False) is True
    assert s.available(exception_flag=True) is True


def test_an_unavailable_solver_says_which_binary_and_why(
    monkeypatch, tmp_path
):
    shim = _broken_shim(tmp_path)
    s = _solver(monkeypatch, shim)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert s.available(exception_flag=False) is False
    reason = str(caught[-1].message)
    assert shim in reason
    assert "bundled CLI binary not found" in reason
    assert "make dev" in reason


def test_the_reason_is_warned_once(monkeypatch, tmp_path):
    """`available(exception_flag=False)` is a predicate called in loops (test
    skips, preflight helpers). The reason is worth one line however many
    times it is asked."""
    s = _solver(monkeypatch, _broken_shim(tmp_path))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            assert s.available(exception_flag=False) is False
    assert len(caught) == 1


def test_the_default_flag_raises_with_the_same_reason(monkeypatch, tmp_path):
    """Pyomo's default is `exception_flag=True`, and its own message there is
    'No executable found' -- which is wrong here: one *was* found, it just
    does not run."""
    shim = _broken_shim(tmp_path)
    s = _solver(monkeypatch, shim)
    with pytest.raises(ApplicationError) as exc:
        s.available()
    assert shim in str(exc.value)


def test_no_binary_at_all_names_all_three_places_looked(monkeypatch):
    monkeypatch.setattr(ps, "_bundled_path", lambda: None)
    monkeypatch.setattr(ps, "_checkout_path", lambda: None)
    monkeypatch.setattr(ps.shutil, "which", lambda name: None)
    s = SolverFactory("pounce")
    with pytest.raises(ApplicationError) as exc:
        s.available()
    reason = str(exc.value)
    assert "pounce-solver" in reason and "make dev" in reason


def test_resolution_prefers_bundled_then_checkout_then_path(monkeypatch):
    monkeypatch.setattr(ps.shutil, "which", lambda name: "/usr/bin/pounce")
    monkeypatch.setattr(ps, "_build_id", lambda exe: "e17b0279")

    monkeypatch.setattr(ps, "_bundled_path", lambda: "/wheel/pounce")
    monkeypatch.setattr(ps, "_checkout_path", lambda: "/repo/target/x/pounce")
    assert SolverFactory("pounce")._default_executable() == "/wheel/pounce"

    monkeypatch.setattr(ps, "_bundled_path", lambda: None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = SolverFactory("pounce")._default_executable()
    assert got == "/repo/target/x/pounce"
    assert "cargo build" in str(caught[-1].message)

    monkeypatch.setattr(ps, "_checkout_path", lambda: None)
    monkeypatch.setattr(ps, "_checkout_warned", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = SolverFactory("pounce")._default_executable()
    assert got == "/usr/bin/pounce"
    assert "PATH executable" in str(caught[-1].message)


def test_check_binary_reports_the_checkout_rung(monkeypatch, capsys):
    monkeypatch.setattr(ps, "_bundled_path", lambda: None)
    monkeypatch.setattr(ps, "_checkout_path", lambda: "/repo/target/x/pounce")
    monkeypatch.setattr(ps, "_build_id", lambda exe: "e17b0279")
    monkeypatch.setattr(ps, "_all_path_pounce", lambda: [])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = pyomo_pounce.check_binary()
    out = capsys.readouterr().out
    assert info["checkout_executable"] == "/repo/target/x/pounce"
    assert info["using_checkout"] is True
    assert info["using_bundled"] is False
    assert "/repo/target/x/pounce" in out


# ── the v2 (pyomo.contrib.solver) interface takes the same three rungs ──────
#
# gh #558 is the precedent: a guard that covered the legacy interface only
# left the modern one with exactly the wrongness it existed to prevent. The
# v2 default resolved bundled-or-`"pounce"`, so in a `maturin develop`
# checkout it named the console-script shim by another route.


def test_v2_default_executable_prefers_bundled_then_checkout(monkeypatch):
    v2 = pytest.importorskip("pyomo_pounce.v2")

    monkeypatch.setattr(v2, "_bundled_path", lambda: "/wheel/pounce")
    monkeypatch.setattr(v2, "_checkout_path", lambda: "/repo/target/x/pounce")
    assert v2._default_executable() == "/wheel/pounce"

    monkeypatch.setattr(v2, "_bundled_path", lambda: None)
    assert v2._default_executable() == "/repo/target/x/pounce"


def test_v2_falls_back_to_a_path_lookup_with_neither(monkeypatch):
    v2 = pytest.importorskip("pyomo_pounce.v2")

    monkeypatch.setattr(v2, "_bundled_path", lambda: None)
    monkeypatch.setattr(v2, "_checkout_path", lambda: None)
    assert v2._default_executable() == "pounce"

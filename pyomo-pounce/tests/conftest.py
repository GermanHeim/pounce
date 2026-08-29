"""Environment guards for the pyomo-pounce suite (gh #366).

These tests shell out to a real `pounce` executable. Which one they get depends
on the environment, and the difference is invisible from inside the tests:

* **CI / an installed wheel** stage a binary at ``python/pounce/bin/pounce``
  (see ``ci.yml``: "Stage CLI binary into pounce-solver wheel"). The plugin
  resolves that *bundled* binary, which is by construction the build under test.
* **A bare source checkout** has no bundled binary, so the plugin falls back to
  whatever ``pounce`` is first on ``PATH`` — quite possibly an unrelated or
  older install from another project or venv.

`SolverFactory("pounce").available()` cannot tell these apart: it answers "is
there an executable?", not "does *this* executable work for what I am about to
ask". So a foreign binary sails past `available()` and then fails mid-test with
``ApplicationError: Solver (pounce) did not exit normally`` — which reads like a
solver defect but is an environment mismatch.

That produced the #366 confusion: a suite that is green in CI and red locally,
with the failing set varying by machine (it depends on what is on PATH), and one
of the casualties being a *correctness* test in ``sens.py``. Five expected
failures is exactly the number at which people stop reading the suite.

The fixtures below probe the resolved executable **once per session** by
actually exercising each capability, and skip with a precise reason when it is
missing. A genuine defect still fails; only an unusable environment skips.

To run the full suite locally, give the plugin the binary you just built::

    cargo build --release -p pounce-cli
    mkdir -p python/pounce/bin
    ln -sf "$PWD/target/release/pounce" python/pounce/bin/pounce
"""

import os
import shutil
import subprocess

import pytest


#: Escape hatch: name the executable explicitly and the guard steps aside.
#: Explicit beats bypassing — this records *which* binary you meant, where
#: `PATH` manipulation silently records nothing.
_EXE_ENV = "POUNCE_TEST_EXE"


def _resolve_pounce_exe():
    """Resolve the `pounce` executable for tests that shell out to it directly,
    and say why it is trustworthy — or refuse it (gh #403).

    Returns ``(path, reason_to_skip)``; exactly one is None.

    The suite's other guard (:func:`pytest_runtest_call` below, gh #366) covers
    tests that go through pyomo's ``SolverFactory``, and only fires when the
    solver produced *no result at all*. Neither half of that reaches a test that
    calls ``shutil.which("pounce")`` and ``subprocess.run`` itself: the plugin's
    bundled-binary resolution never runs, and a foreign binary that answers
    *incorrectly* exits cleanly, writes a valid ``.sol``, and sails through.

    That is not hypothetical. A pip-installed `pounce` from the previous day
    reported 49 of 200 feasible-by-construction models in the AMPL infeasible
    band — against a ratchet whose limit is 0. Both binaries said ``0.9.0``;
    only the embedded commit distinguished them (``10a6fe0c+dirty`` vs
    ``ad0991df``), which is exactly the discriminator ``_build_id`` exists to
    provide.

    The failure is silent in *both* directions, which is what makes it worth a
    guard: on a different `PATH` the same setup reports a comfortable green
    while the working tree is broken.
    """
    explicit = os.environ.get(_EXE_ENV)
    if explicit:
        if not os.path.isfile(explicit):
            return None, f"{_EXE_ENV}={explicit!r} is not a file"
        return explicit, None

    try:
        import pyomo_pounce
        from pyomo_pounce.pounce_solver import (
            _build_id,
            _bundled_path,
            _checkout_path,
        )
    except Exception as exc:  # noqa: BLE001 - a broken probe must not mislead
        return None, f"cannot import pyomo_pounce to resolve the binary: {exc}"
    del pyomo_pounce

    # Resolution order mirrors the plugin's: bundled, then the source
    # checkout's own cargo build, then PATH (gh #816). The middle rung is the
    # one a working tree usually lands on, and it is also the one most likely
    # to be *right* — but "most likely" is not "verified", so it goes through
    # the same build-id proof below as the other two.
    found = _bundled_path() or _checkout_path() or shutil.which("pounce")
    if found is None:
        return None, "no bundled binary and no pounce on PATH"

    # ...but *where* it came from does not make it trustworthy. The bundled
    # path is only "the build under test" when something staged it this run,
    # which is true in CI and false in a working tree, where
    # `python/pounce/bin/pounce` is gitignored and survives across days and
    # commits. The binary that produced this guard's motivating failure was
    # sitting there, six commits and a day stale. So verify the build itself.
    # Refuse only on a *proven* mismatch — both ids known and different.
    #
    # The asymmetry is deliberate. Skipping costs the ratchet: a skipped
    # `MAX_FALSE_POSITIVES = 0` proves nothing, and a suite that quietly stops
    # checking is the failure this guard exists to prevent. So an id we cannot
    # read must not skip. `crates/pounce-cli/build.rs` embeds "unknown" when it
    # builds outside a git checkout — true of a wheel built in a container
    # without `.git` — and CI is exactly where a silent skip would hurt most,
    # so unknown-vs-anything runs. What is caught is the case that motivated
    # this: two readable ids that disagree.
    want = _checkout_build_id()
    got = _build_id(found)
    if want and got and not _same_commit(got, want):
        return None, (
            f"refusing to measure {found!r}: it is build {got}, but this "
            f"checkout is {want}. These probes are correctness ratchets — a "
            f"different build's verdicts are not evidence either way, in "
            f"either direction. Rebuild and stage it (see this module's "
            f"docstring), or set {_EXE_ENV} to choose a binary deliberately."
        )
    return found, None


def _checkout_build_id():
    """This checkout's build identifier, in ``_build_id`` form, or None when
    we are not in a git checkout."""

    def _git(*args):
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

    try:
        head = _git("rev-parse", "--short=8", "HEAD")
        if head.returncode != 0:
            return None
        dirty = _git("status", "--porcelain")
    except Exception:  # noqa: BLE001
        return None
    commit = head.stdout.strip()
    if not commit:
        return None
    return commit + ("+dirty" if dirty.stdout.strip() else "")


def _same_commit(got, want):
    """Compare two build ids on the *commit* only.

    The ``+dirty`` flag is deliberately ignored. A binary built from a clean
    tree that you have since edited reports ``abc123`` against a checkout of
    ``abc123+dirty`` — a real staleness, but the narrow kind that the
    source-mtime guard already covers, and failing on it would make this guard
    fire constantly during ordinary edit-build-test work. What must not slip
    through is a *different commit*, which is the days-stale case that
    motivated this.
    """
    a = got.split("+")[0]
    b = want.split("+")[0]
    if not a or not b:
        return False
    n = min(len(a), len(b))
    return a[:n] == b[:n]


def solver_routes():
    """The `(name, solve)` pairs a "every route agrees" test should compare,
    or a skip when there is nothing to compare.

    Three entry points reach the same in-process sensitivity machinery: the
    legacy `SolverFactory("pounce")` plugin, the `pounce_v2` registration, and
    Pyomo's `contrib.solver` factory. The **last two are the same
    registration** — both come from `pyomo_pounce.v2` — so on a Pyomo older
    than 6.10.1 they are absent together, `SolverFactory("pounce_v2")` returns
    an `UnknownSolver` and `SF2("pounce")` returns `None`, and a test that
    calls `.solve()` on either dies with an unhelpful `AttributeError` or a
    `ValueError` about an `asl` executable. That is a missing *environment*,
    not a defect.

    Keyed on `pyomo_pounce.HAVE_V2_INTERFACE`, which is the package's own
    supported predicate for this. Deliberately **not** `try: import
    pyomo_pounce.v2` — `pyomo_pounce/__init__.py` says why in as many words: a
    try/except there would also swallow a genuine `ImportError` raised by a bug
    *inside* v2 and report the interface as merely unavailable.

    When only one route survives there is nothing for an agreement test to
    compare, so this skips rather than passing vacuously — a single-route
    "every route agrees" assertion is worse than no assertion. CI runs a Pyomo
    that has all three, so this never fires there.
    """
    import pyomo.environ as pyo
    import pyomo_pounce

    routes = [("legacy", lambda m: pyo.SolverFactory("pounce").solve(m))]
    if pyomo_pounce.HAVE_V2_INTERFACE:
        from pyomo.contrib.solver.common.factory import SolverFactory as SF2

        routes.append(("v2", lambda m: pyo.SolverFactory("pounce_v2").solve(m)))
        routes.append(("contrib", lambda m: SF2("pounce").solve(m)))
    if len(routes) < 2:
        import pyomo

        pytest.skip(
            f"only the legacy route is registered: the v2 and contrib routes "
            f"both come from pyomo_pounce.v2, which needs Pyomo 6.10.1+ "
            f"(this environment has {pyomo.version.version}). Nothing to "
            f"compare, so this is an environment gap rather than a result."
        )
    return routes


@pytest.fixture(scope="session")
def pounce_exe():
    """A `pounce` executable this checkout vouches for, or a skip."""
    exe, reason = _resolve_pounce_exe()
    if exe is None:
        pytest.skip(reason)
    return exe


def _using_bundled():
    """Is the plugin resolving the binary this checkout built?

    ``check_binary`` reports both the resolved and bundled executables; the
    tests are only trustworthy when those agree.
    """
    try:
        import pyomo_pounce

        info = pyomo_pounce.check_binary(verbose=False)
    except Exception:  # noqa: BLE001 - a broken probe must not break the suite
        return False, None
    return bool(info.get("using_bundled")), info.get("resolved_executable")


# The environment signature: pyomo raises this when the CLI it invoked did not
# produce a usable result — i.e. the solver never really ran.
_DID_NOT_RUN = "did not exit normally"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """Turn "the solver never ran" into a skip — but only in an environment
    that cannot be trusted to have the right binary.

    Deliberately narrow, because the opposite failure mode is worse than a red
    suite: a guard broad enough to swallow real defects makes the suite
    meaningless. Two conditions must both hold before anything is skipped.

    1. The error is pyomo's ``ApplicationError: Solver (pounce) did not exit
       normally`` — the solver produced no result at all. A wrong *answer*, a
       failed assertion, or any other exception still fails the test.
    2. The plugin is **not** using this checkout's bundled binary, so the thing
       that just failed is some other build we make no claims about.

    Condition 2 is the safety property: CI stages the bundled binary, so
    ``using_bundled`` is true there and this hook can never fire — CI cannot be
    silently masked by it. It only ever fires in a source checkout that has
    fallen back to a foreign `pounce` on PATH.
    """
    try:
        return (yield)
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is the env
        if _DID_NOT_RUN not in str(exc):
            raise
        bundled, resolved = _using_bundled()
        if bundled:
            raise
        pytest.skip(
            f"solver never ran, and the plugin is not using this checkout's "
            f"bundled binary (resolved: {resolved}). This is an environment "
            f"mismatch, not a solver defect -- see conftest.py for the "
            f"expected local setup. Original error: {exc}"
        )

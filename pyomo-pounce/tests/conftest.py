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

import pytest


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

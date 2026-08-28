"""The executable guard itself (gh #403).

``conftest._resolve_pounce_exe`` decides whether the probes in
``test_infeasibility_no_false_positives`` and ``test_scale_invariance`` are
measuring a binary this checkout vouches for. Both of those are correctness
ratchets, so the guard is load-bearing in *both* directions:

* refuse a binary we can prove is a different build — the case that motivated
  this, where a day-stale artifact reported 49 of 200 feasible models in the
  infeasible band against a limit of 0;
* but never skip merely because an id is unreadable, because a skipped ratchet
  proves nothing and that is the same silent loss by another route.

These tests pin that asymmetry.
"""

import conftest


def _resolve(monkeypatch, *, bundled, which, build_id, checkout_id,
             checkout=None, env=None):
    """Drive `_resolve_pounce_exe` with the environment fully stubbed.

    ``checkout`` is the cargo build of the surrounding source tree (the middle
    rung of the resolution order, gh #816); it defaults to absent so every
    test that predates it keeps describing the same two-rung situation.
    """
    monkeypatch.delenv(conftest._EXE_ENV, raising=False)
    if env is not None:
        monkeypatch.setenv(conftest._EXE_ENV, env)

    import pyomo_pounce.pounce_solver as ps

    monkeypatch.setattr(ps, "_bundled_path", lambda: bundled)
    monkeypatch.setattr(ps, "_checkout_path", lambda: checkout)
    monkeypatch.setattr(ps, "_build_id", lambda exe: build_id)
    monkeypatch.setattr(conftest.shutil, "which", lambda name: which)
    monkeypatch.setattr(conftest, "_checkout_build_id", lambda: checkout_id)
    return conftest._resolve_pounce_exe()


def test_a_proven_different_build_is_refused(monkeypatch):
    """The motivating case: two readable ids that disagree."""
    exe, reason = _resolve(
        monkeypatch,
        bundled="/repo/python/pounce/bin/pounce",
        which=None,
        build_id="10a6fe0c+dirty",
        checkout_id="e17b0279+dirty",
    )
    assert exe is None
    assert "10a6fe0c+dirty" in reason and "e17b0279+dirty" in reason


def test_a_matching_build_is_accepted(monkeypatch):
    exe, reason = _resolve(
        monkeypatch,
        bundled="/repo/python/pounce/bin/pounce",
        which=None,
        build_id="e17b0279",
        checkout_id="e17b0279",
    )
    assert reason is None
    assert exe == "/repo/python/pounce/bin/pounce"


def test_the_dirty_flag_alone_does_not_refuse(monkeypatch):
    """A clean build against an edited tree is stale only in the narrow way the
    source-mtime guard already covers. Refusing here would fire during ordinary
    edit-build-test work and train people to bypass the guard."""
    exe, reason = _resolve(
        monkeypatch,
        bundled="/repo/python/pounce/bin/pounce",
        which=None,
        build_id="e17b0279",
        checkout_id="e17b0279+dirty",
    )
    assert reason is None, reason
    assert exe is not None


def test_an_unreadable_binary_id_still_runs(monkeypatch):
    """`build.rs` embeds "unknown" outside a git checkout — true of a wheel
    built in a container without `.git`. Skipping there would silently disarm
    the ratchet in CI, which is worse than measuring an unverifiable binary
    that CI staged itself."""
    exe, reason = _resolve(
        monkeypatch,
        bundled="/wheel/pounce",
        which=None,
        build_id=None,  # `commit unknown` does not parse as a build id
        checkout_id="e17b0279",
    )
    assert reason is None, reason
    assert exe == "/wheel/pounce"


def test_a_non_git_checkout_still_runs(monkeypatch):
    """The mirror: an sdist or tarball has no HEAD to compare against."""
    exe, reason = _resolve(
        monkeypatch,
        bundled="/wheel/pounce",
        which=None,
        build_id="e17b0279",
        checkout_id=None,
    )
    assert reason is None, reason
    assert exe == "/wheel/pounce"


def test_path_fallback_is_still_checked(monkeypatch):
    """No bundled binary: the PATH one is subject to the same proof."""
    exe, reason = _resolve(
        monkeypatch,
        bundled=None,
        which="/usr/local/bin/pounce",
        build_id="10a6fe0c",
        checkout_id="e17b0279",
    )
    assert exe is None
    assert "/usr/local/bin/pounce" in reason


def test_no_binary_anywhere_is_reported(monkeypatch):
    exe, reason = _resolve(
        monkeypatch, bundled=None, which=None, build_id=None, checkout_id="e17b0279"
    )
    assert exe is None
    assert "no bundled binary" in reason


def test_an_explicit_choice_overrides_the_guard(monkeypatch, tmp_path):
    """Explicit beats bypassing: naming the binary records which one you meant,
    where `PATH` manipulation records nothing."""
    fake = tmp_path / "pounce"
    fake.write_text("#!/bin/sh\n")
    exe, reason = _resolve(
        monkeypatch,
        bundled="/repo/python/pounce/bin/pounce",
        which=None,
        build_id="10a6fe0c",  # would otherwise be refused
        checkout_id="e17b0279",
        env=str(fake),
    )
    assert reason is None, reason
    assert exe == str(fake)


def test_an_explicit_choice_that_does_not_exist_is_reported(monkeypatch):
    exe, reason = _resolve(
        monkeypatch,
        bundled=None,
        which=None,
        build_id=None,
        checkout_id=None,
        env="/nope/pounce",
    )
    assert exe is None
    assert "is not a file" in reason


def test_same_commit_compares_the_commit_only():
    assert conftest._same_commit("e17b0279", "e17b0279")
    assert conftest._same_commit("e17b0279", "e17b0279+dirty")
    assert conftest._same_commit("e17b0279+dirty", "e17b0279")
    # Differing widths compare on the shared prefix (short vs shorter).
    assert conftest._same_commit("e17b0279", "e17b027")
    assert not conftest._same_commit("10a6fe0c", "e17b0279")
    assert not conftest._same_commit("", "e17b0279")


def test_the_checkout_build_is_preferred_over_path(monkeypatch):
    """gh #816: a `maturin develop` tree has no bundled binary, and what sits
    on PATH there is that install's own console-script shim. The cargo build
    is both the honest answer and the one the plugin now resolves, so the
    guard has to look at the same one."""
    exe, reason = _resolve(
        monkeypatch,
        bundled=None,
        checkout="/repo/target/release/pounce",
        which="/venv/bin/pounce",
        build_id="e17b0279",
        checkout_id="e17b0279",
    )
    assert reason is None, reason
    assert exe == "/repo/target/release/pounce"


def test_the_bundled_binary_still_outranks_the_checkout(monkeypatch):
    exe, reason = _resolve(
        monkeypatch,
        bundled="/repo/python/pounce/bin/pounce",
        checkout="/repo/target/release/pounce",
        which=None,
        build_id="e17b0279",
        checkout_id="e17b0279",
    )
    assert reason is None, reason
    assert exe == "/repo/python/pounce/bin/pounce"


def test_the_checkout_build_is_still_proved(monkeypatch):
    """The new rung buys convenience, not trust: a `target/release` binary
    from before the change under test is exactly the stale artifact this
    guard exists for."""
    exe, reason = _resolve(
        monkeypatch,
        bundled=None,
        checkout="/repo/target/release/pounce",
        which=None,
        build_id="10a6fe0c",
        checkout_id="e17b0279",
    )
    assert exe is None
    assert "/repo/target/release/pounce" in reason

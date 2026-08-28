"""The console script has to work in a source checkout (gh #816).

`maturin develop` builds the extension module and nothing else, so
`pounce/bin/` stays empty and the `pounce` console script — a shim whose
whole job is to exec that binary — had nothing to exec. Every CLI
invocation failed, `--version` included, and the failure was quiet in the
way that costs time: Pyomo's ASL layer runs `pounce -v`, got no version
out of the shim's error message, and reported the solver *unavailable*
while an in-process solve worked fine.

These tests pin the three things that make the dev install usable again:
the checkout binary is found, the bundled one still outranks it, and the
fallback notice never contaminates the machine-read `-v` output.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from pounce import _cli

# `pyomo.opt.base.solvers._extract_version`, verbatim. Pyomo scans the
# *merged* stdout+stderr of `pounce -v` with this and takes the first hit,
# so anything the shim prints ahead of the version is a wrong answer, not
# noise.
_PYOMO_VERSION_RE = re.compile(r"[0-9]+(\.[0-9]+){1,3}")


def _fake_tree(root, *, built=(), cli_crate=True):
    """A checkout-shaped directory: workspace Cargo.toml, crates/pounce-cli,
    and whichever of release/debug binaries `built` names."""
    (root / "Cargo.toml").write_text("[workspace]\n")
    if cli_crate:
        crate = root / "crates" / "pounce-cli"
        crate.mkdir(parents=True)
        (crate / "Cargo.toml").write_text("[package]\nname='pounce-cli'\n")
    pkg = root / "python" / "pounce"
    pkg.mkdir(parents=True)
    for profile in built:
        d = root / "target" / profile
        d.mkdir(parents=True)
        exe = d / _cli._exe_name()
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)
    return pkg / "_cli.py"


def _with_file(monkeypatch, path):
    """Point the module's `__file__`-derived lookups at `path`."""
    monkeypatch.setattr(_cli, "__file__", str(path))


def test_the_repo_root_is_found_from_the_package(monkeypatch, tmp_path):
    _with_file(monkeypatch, _fake_tree(tmp_path, built=("release",)))
    assert _cli._repo_root() == tmp_path.resolve()


def test_an_installed_wheel_is_not_a_checkout(monkeypatch, tmp_path):
    """The narrowness is the safety property: an installed wheel must never
    quietly run something out of somebody's build directory."""
    pkg = tmp_path / "site-packages" / "pounce"
    pkg.mkdir(parents=True)
    _with_file(monkeypatch, pkg / "_cli.py")
    assert _cli._repo_root() is None
    assert _cli._checkout_binary() is None


def test_a_cargo_toml_alone_is_not_this_checkout(monkeypatch, tmp_path):
    """An unrelated Rust project above a site-packages tree is not POUNCE;
    `crates/pounce-cli/` is the half that says which project this is."""
    _with_file(monkeypatch, _fake_tree(tmp_path, built=("release",),
                                       cli_crate=False))
    assert _cli._repo_root() is None


def test_the_newest_build_wins(monkeypatch, tmp_path):
    """Newest rather than release-always: someone who just ran `cargo build`
    to test a fix means *that* binary, not a release artifact from weeks ago.
    The caller prints which one it took, so the choice stays visible."""
    _with_file(monkeypatch, _fake_tree(tmp_path, built=("release", "debug")))
    release = tmp_path / "target" / "release" / _cli._exe_name()
    debug = tmp_path / "target" / "debug" / _cli._exe_name()

    os.utime(release, (1_000, 1_000))
    os.utime(debug, (2_000, 2_000))
    assert _cli._checkout_binary() == debug

    os.utime(release, (3_000, 3_000))
    assert _cli._checkout_binary() == release


def test_cargo_target_dir_is_honored(monkeypatch, tmp_path):
    _with_file(monkeypatch, _fake_tree(tmp_path))
    elsewhere = tmp_path / "elsewhere" / "release"
    elsewhere.mkdir(parents=True)
    exe = elsewhere / _cli._exe_name()
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setenv("CARGO_TARGET_DIR", str(tmp_path / "elsewhere"))
    assert _cli._checkout_binary() == exe


def test_nothing_built_yet_resolves_to_nothing(monkeypatch, tmp_path):
    _with_file(monkeypatch, _fake_tree(tmp_path))
    assert _cli._checkout_binary() is None
    assert _cli.resolve_binary() == (None, None)


def test_the_bundled_binary_outranks_the_checkout(monkeypatch, tmp_path):
    """A wheel that also happens to sit in a checkout runs its own binary:
    the bundled one is the artifact under test everywhere else."""
    pkg_cli = _fake_tree(tmp_path, built=("release",))
    binary = pkg_cli.parent / "bin" / _cli._exe_name()
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    _with_file(monkeypatch, pkg_cli)

    resolved, origin = _cli.resolve_binary()
    assert origin == _cli.BUNDLED
    assert resolved == binary


def test_the_checkout_binary_is_used_when_nothing_is_bundled(
    monkeypatch, tmp_path
):
    _with_file(monkeypatch, _fake_tree(tmp_path, built=("release",)))
    resolved, origin = _cli.resolve_binary()
    assert origin == _cli.CHECKOUT
    assert resolved == tmp_path / "target" / "release" / _cli._exe_name()


def test_the_fallback_is_announced_on_a_real_invocation():
    notice = _cli._fallback_notice(Path("/repo/target/debug/pounce"),
                                   ["model.nl", "-AMPL"])
    assert "/repo/target/debug/pounce" in notice
    assert "make dev" in notice


@pytest.mark.parametrize("flag", ["-v", "-V", "--version", "--about",
                                  "-h", "--help"])
def test_the_notice_stays_out_of_machine_read_output(flag):
    """The notice names a path, and a path like `/opt/py3.11/...` matches
    Pyomo's version regex. Printing it ahead of `pounce -v` would not make
    the version unreadable — it would make Pyomo read `3.11` as POUNCE's
    version, which is worse."""
    assert _cli._fallback_notice(Path("/opt/py3.11/target/debug/pounce"),
                                 [flag]) == ""


def test_the_not_found_message_names_the_way_out(monkeypatch, tmp_path):
    _with_file(monkeypatch, _fake_tree(tmp_path))
    msg = _cli._not_found_message(_cli._bundled_binary())
    assert "make -C" in msg and "dev" in msg
    assert "check_binary" in msg


def test_a_wheel_with_no_binary_says_reinstall(monkeypatch, tmp_path):
    pkg = tmp_path / "site-packages" / "pounce"
    pkg.mkdir(parents=True)
    _with_file(monkeypatch, pkg / "_cli.py")
    msg = _cli._not_found_message(_cli._bundled_binary())
    assert "pip install --force-reinstall pounce-solver" in msg
    assert "make" not in msg


@pytest.mark.skipif(
    _cli.resolve_binary()[0] is None,
    reason="no pounce CLI binary built or bundled in this checkout",
)
def test_the_shim_answers_dash_v_with_exactly_a_version():
    """End to end, the way Pyomo asks: run the shim's `main()` with `-v` and
    read the stream Pyomo reads -- stderr folded into stdout, so anything the
    shim writes to stderr lands *ahead* of the version. This is the assertion
    the `available() is False while solve() works` report reduces to."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.argv = ['pounce', '-v'];"
            " from pounce._cli import main; sys.exit(main() or 0)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout
    # One line, and that line is the version: not "the version is in there
    # somewhere". Pyomo takes the *first* `N.N[.N]` in this stream, so a
    # leading notice is a wrong version rather than a noisy one.
    assert re.fullmatch(r"pounce \d+(\.\d+){1,3}", proc.stdout.strip()), (
        proc.stdout
    )
    assert _PYOMO_VERSION_RE.search(proc.stdout) is not None

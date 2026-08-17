"""Platform tagging for the ``pounce-casadi`` wheel.

Everything in this package is pure Python — it loads its payload with
``ctypes`` and defines no extension module — but the payload itself is
compiled: a CasADi ``nlpsol`` plugin and the POUNCE solver library, under
``pounce_casadi/_plugins/<casadi-minor>/``. setuptools sees no
``ext_modules``, concludes the distribution is pure, and tags the wheel
``py3-none-any``.

That tag is a promise the wheel cannot keep. ``pip`` reads it as "installs
anywhere", so a wheel built on macOS installs happily on Linux and the user
finds out at ``import pounce_casadi``, where the ``.dylib`` it wants is not
the ``.so`` the platform can load. The failure should happen at install
time, on the resolver, which is what a real platform tag buys.

Two adjustments, in opposite directions:

* ``root_is_pure = False`` moves the package into ``platlib`` and stamps the
  building platform onto the tag.
* ``get_tag`` puts the *Python* half back to ``py3``/``none``. The default
  for an impure wheel is ``cp311-cp311-…``, which is wrong the other way:
  nothing here is compiled against the CPython ABI, so one build genuinely
  does serve every Python 3 on that platform, and a ``cp311`` tag would
  make ``pip`` reject it on 3.12 for no reason.

The result is ``py3-none-<platform>``: one wheel per platform, each
carrying a build for every supported CasADi minor and choosing between
them at import. The CasADi axis of the matrix stays inside the wheel —
CasADi is a runtime dependency resolved by ``casadi.__version__``, and
there is no wheel tag that could express it.

``POUNCE_CASADI_PLAT_NAME`` overrides the platform half, for the two cases
where the building machine is not the target: a manylinux build (the raw
``linux_x86_64`` tag is not installable from PyPI — ``auditwheel repair``
normally retags, and this is the escape hatch when it is not in the loop)
and a macOS cross build (``macosx_11_0_universal2`` and friends).
"""

from __future__ import annotations

import os

from setuptools import setup
from setuptools.dist import Distribution

try:  # setuptools >= 70.1 vendors the command
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # pragma: no cover - older setuptools
    from wheel.bdist_wheel import bdist_wheel  # type: ignore[import-not-found]


class BinaryDistribution(Distribution):
    """Claim compiled content so ``bdist_wheel`` starts out impure.

    ``has_ext_modules`` is what setuptools consults, and it answers on
    ``ext_modules``, which is empty here. The compiled content is package
    data, which it does not look at.
    """

    def has_ext_modules(self) -> bool:
        return True


class PlatformTaggedWheel(bdist_wheel):
    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        _python, _abi, plat = super().get_tag()
        override = os.environ.get("POUNCE_CASADI_PLAT_NAME")
        if override:
            plat = override
        # Wheel filenames use `_` for every separator; distutils platform
        # names arrive with `-` and `.` in them (`macosx-11.0-arm64`).
        plat = plat.replace("-", "_").replace(".", "_")
        return "py3", "none", plat


setup(distclass=BinaryDistribution, cmdclass={"bdist_wheel": PlatformTaggedWheel})

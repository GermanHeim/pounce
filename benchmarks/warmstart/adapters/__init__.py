"""Solver adapters. Import the concrete one lazily so the core stays
importable (and the families' derivative self-test runnable) on a
machine with no solver installed."""

from __future__ import annotations

from .base import ARMS, REFERENCE_ARM, SolverAdapter, is_warm

__all__ = ["ARMS", "REFERENCE_ARM", "SolverAdapter", "is_warm", "get_adapter"]


#: Adapters the suite knows how to build. Listed here rather than
#: discovered, so `--solver ipopt` on a machine without cyipopt fails
#: with the install error rather than "unknown solver".
KNOWN = ("pounce", "ipopt")


def get_adapter(name: str, **kwargs) -> SolverAdapter:
    if name == "pounce":
        from .pounce_adapter import PounceAdapter

        return PounceAdapter(**kwargs)
    if name == "ipopt":
        try:
            import cyipopt  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "the ipopt adapter needs cyipopt, which is not installed.\n"
                "There is no cyipopt wheel on PyPI; it builds against a\n"
                "system Ipopt. On Debian/Ubuntu:\n"
                "    apt-get install -y coinor-libipopt-dev liblapack-dev "
                "libblas-dev\n"
                "    pip install cython && pip install --no-build-isolation "
                "cyipopt\n"
                f"(import failed with: {exc})"
            ) from exc
        from .ipopt_adapter import IpoptAdapter

        return IpoptAdapter(**kwargs)
    raise KeyError(
        f"unknown solver adapter {name!r} (known: {', '.join(KNOWN)})"
    )

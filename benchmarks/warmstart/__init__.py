"""Warm-start benchmark suite.

A benchmark for solving *sequences* of related NLPs, which is the only
setting in which warm starting means anything. See ``README.md`` for
the protocol, the arms, and what the numbers do and do not show.

Layout::

    spec.py       solver-agnostic core types (no pounce import)
    sparsity.py   dense family callbacks -> sparse, instrumented ones
    families/     the parametric families
    adapters/     solver plug-ins; adapters/pounce_adapter.py is the only
                  module that imports a solver
    runner.py     the measurement protocol
    report.py     JSON -> markdown
    run.py        command line entry point
    selftest.py   finite-difference derivative checks (no solver needed)
"""

__all__ = ["spec", "sparsity", "families", "adapters", "runner", "report"]

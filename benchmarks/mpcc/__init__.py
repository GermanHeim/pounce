"""MPCC benchmark harness (gh#794, Gate 0 of gh#776).

The smallest reproducible corpus that can tell three things apart when an
MPCC solve goes wrong:

* the **source formulation** is deficient (the MPCC has no S-stationary
  point, or none at all);
* the **lowering** to a smooth NLP is deficient (the product row is the
  wrong shape, the relaxation schedule is too coarse); or
* **POUNCE's inner NLP algorithm** is deficient.

Nothing here is a phase-change, flash, tray or column model, and nothing
here should grow into one: gh#794's last acceptance criterion is that no
such work begins from this issue. This is the evidence gate in front of
it.

Layout
------

``spec``          the model algebra (quadratic objective/rows, affine
                  complementarity pairs) and the record types.
``cases``         the benchmark ladder, one registry.
``lowering``      MPCC -> smooth NLP, one function per lowering.
``stationarity``  the MPCC stationarity classifier (S / M / C / W).
``validate``      source-level validation, one function per class.
``oracle``        branch-enumeration global oracle; optional CCOpt hook.
``routes``        the POUNCE configurations and the kill-switch controls.
``runner``        the measurement protocol, including the continuation
                  restart ladder.
``report``        markdown rendering.
``run``           the CLI.
``selftest``      everything checkable without a solver.
"""

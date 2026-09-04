"""Gate 1 flash fixture (gh#776), the successor to the Gate 0 harness.

Gate 0 (gh#794, `benchmarks/mpcc/`) established that POUNCE has a
supported route for small MPCCs -- `scholtes_then_ncp` -- and drew its
failure boundary on a corpus that was deliberately *not* a process
model: quadratic objectives, affine complementarity pairs, six variables
at most. Gate 1 is the first phase-changing model, and gh#776 scopes it
to exactly one: a single flash whose temperature path crosses
single-liquid, two-phase and single-vapor, validated against an
independent flash and stability calculation.

Where the model lives
---------------------

**Not here.** The flash, its complementarity pairs, the lowerings and
the independent oracle are `pounce.examples.flash_mpcc`, so that the
wheel ships them and notebook 38 can import them the way the other three
application tutorials import theirs. This directory is the *evidence
apparatus* around that model, and keeping the two apart is what stops
the harness from becoming a second, subtly different copy of the physics
-- which is the failure mode the whole cross-check exists to catch.

Layout
------

``routes``    the Gate 0 supported route, plus the arms Gate 1 compares
              against it, and the kill-switch controls.
``runner``    the measurement protocol for one solve, including the
              continuation restart ladder.
``path``      the temperature traversal: both directions, cold and warm.
``validate``  source-level checks at a returned point.
``stamp``     provenance, including the explicit record that no DiscOpt
              comparison was run and why.
``run``       the CLI and the markdown report.
``selftest``  everything checkable without the solver.
"""

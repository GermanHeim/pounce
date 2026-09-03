"""Gate 1 flash fixture (gh#776), the successor to the Gate 0 harness.

Gate 0 (gh#794, `benchmarks/mpcc/`) established that POUNCE has a
supported route for small MPCCs -- `scholtes_then_ncp` -- and drew its
failure boundary on a corpus that was deliberately *not* a process
model: quadratic objectives, affine complementarity pairs, six variables
at most. Gate 1 is the first phase-changing model, and gh#776 scopes it
to exactly one: a single flash whose temperature path crosses
single-liquid, two-phase and single-vapor, validated against an
independent flash and stability calculation.

Layout
------

``thermo``    Peng--Robinson layer over `pounce.examples.phase_envelope`,
              plus the cubic-root, trivial-solution and supercritical
              guards.
``spec``      the flash as an MPCC: variables, rows, the two
              complementarity pairs and what each branch means, and the
              source-level residuals.
``oracle``    the independent calculation: Michelsen tangent-plane
              stability with a multistart, Rachford--Rice with a Newton
              polish, and the switch-point bisection.
``lowering``  MPCC -> smooth NLP, with exact JAX derivatives.
``routes``    the Gate 0 supported route, plus the arms Gate 1 compares
              against it.
``path``      the temperature traversal: both directions, cold and warm.
``validate``  source-level checks at a returned point.
``run``       the CLI.
``selftest``  everything checkable without the solver.
"""

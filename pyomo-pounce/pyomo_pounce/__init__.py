"""Pyomo solver plugin for the POUNCE interior-point NLP solver.

Usage:
    import pyomo_pounce  # registers 'pounce' with SolverFactory
    from pyomo.environ import *
    solver = SolverFactory('pounce')

Initialization helpers (see the POUNCE docs' initialization chapter):
    report = pyomo_pounce.preflight(model)         # starting-point check
    pyomo_pounce.initialize_missing_values(model)  # fill unset Var values
    pyomo_pounce.block_initialize(model)           # experimental DM-ordered init
"""
from pyomo_pounce.block_init import BlockInitReport, block_initialize
from pyomo_pounce.pounce_solver import POUNCE
from pyomo_pounce.preflight import (
    PyomoPreflightReport,
    initialize_missing_values,
    preflight,
)

__all__ = [
    "POUNCE",
    "preflight",
    "PyomoPreflightReport",
    "initialize_missing_values",
    "block_initialize",
    "BlockInitReport",
]

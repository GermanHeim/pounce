"""Generate the IDAES Helmholtz external-function NL fixture for issue #15.

Original script written with GPT 5.4 and validated locally by CMarcher on
https://github.com/jkitchin/ripopt/issues/15. Saved here so the fixture can
be regenerated whenever IDAES + the general_helmholtz_external shared library
are available.

Running the script:
    python tests/fixtures/issue_15/generate_helmholtz_nl.py

Requires IDAES 2.x with `idaes get-extensions` already run so that
`~/.idaes/bin/general_helmholtz_external.{dylib,so,dll}` exists. Writes the
`.nl` file to tests/fixtures/issue_15/idaes_helmholtz.nl.
"""

from pathlib import Path
import shutil

import idaes
from idaes.core import FlowsheetBlock
from idaes.models.properties.general_helmholtz import (
    HelmholtzParameterBlock,
    PhaseType,
    StateVars,
    AmountBasis,
)
from idaes.models.unit_models.heater import Heater
from idaes.core.util.model_statistics import degrees_of_freedom
from pyomo.environ import ConcreteModel, SolverFactory, assert_optimal_termination, value


FIXTURE_NL = Path(__file__).with_name("idaes_helmholtz.nl")


def build_model():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = HelmholtzParameterBlock(
        pure_component="h2o",
        phase_presentation=PhaseType.LG,
        state_vars=StateVars.PH,
        amount_basis=AmountBasis.MOLE,
    )

    m.fs.heater = Heater(property_package=m.fs.properties)
    m.fs.heater.heat_duty.fix(0)
    m.fs.heater.inlet.flow_mol.fix(1)
    m.fs.heater.inlet.enth_mol.fix(1878.71)
    m.fs.heater.inlet.pressure.fix(101325)
    return m


def solve_with_ipopt(model):
    ipopt_path = Path(idaes.bin_directory) / "ipopt"
    if not ipopt_path.exists():
        raise FileNotFoundError(f"Expected ipopt at {ipopt_path}")

    solver = SolverFactory("ipopt", executable=str(ipopt_path))
    results = solver.solve(
        model,
        tee=False,
        keepfiles=True,
        symbolic_solver_labels=True,
    )
    assert_optimal_termination(results)

    nl_source = Path(solver._problem_files[0])
    shutil.copy2(nl_source, FIXTURE_NL)
    return nl_source


def main():
    model = build_model()
    dof = degrees_of_freedom(model)
    if dof != 0:
        raise AssertionError(f"Expected 0 degrees of freedom, got {dof}")

    model.fs.heater.initialize()
    nl_source = solve_with_ipopt(model)

    outlet_temp = value(model.fs.heater.control_volume.properties_out[0].temperature)
    if abs(outlet_temp - 298.0) > 1e-2:
        raise AssertionError(f"Unexpected outlet temperature: {outlet_temp}")

    print(f"IPOPT_NL_SOURCE={nl_source}")
    print(f"FIXTURE_NL={FIXTURE_NL}")


if __name__ == "__main__":
    main()

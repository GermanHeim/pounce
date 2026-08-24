# Uncertainty-Aware Catalyst-Pellet Inverse Design

Notebook
[`37_catalyst_pellet_inverse_design.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/37_catalyst_pellet_inverse_design.ipynb)
is a reproducible, activity-only inverse-design study for one spherical,
nonisothermal CO2-methanation pellet. It is intentionally smaller than a
pellet-in-reactor co-design: the point is to make the equations, exact
derivatives, physical checks, covariance model, and robust redesign auditable
in one POUNCE example.

The reusable implementation is in `pounce.catalyst_pellet`. The
notebook records the source commit, model revision, package versions, solver
tolerances, mesh, and activity basis in its saved output.

## Scope and source map

The chemical kinetics are the four-species CO2 methanation correlation of
Koschany, Schlereth, and Hinrichsen. The particle size, pressure, composition,
solid density, and thermal conductivity are anchored to the structured-particle
study of Zimmermann, Bremer, and Sundmacher. The reactor model in that paper is
not copied: the tutorial prescribes one bulk state and finite external films.

| Quantity | Tutorial value | Status |
| --- | ---: | --- |
| pellet radius | 1.25 mm | Zimmermann et al., 2.5 mm particle diameter |
| pressure | 5 bar | Zimmermann et al. |
| bulk mole fractions `(CO2, H2, CH4, H2O)` | `(0.2, 0.8, 0, 0)` | Zimmermann et al. inlet composition |
| bulk temperature | 555 K | Koschany kinetic reference temperature; replaces the reactor inlet temperature |
| solid density | 4500 kg m^-3 | Zimmermann et al. |
| effective thermal conductivity | 2.5 W m^-1 K^-1 | fixed-particle value used by Zimmermann et al. |
| pellet porosity | 0.35 | explicit tutorial assumption |
| effective diffusivities | `(1.0, 2.8, 1.2, 1.1)e-6` m^2 s^-1 | explicit tutorial assumptions for `(CO2, H2, CH4, H2O)` |
| external mass-transfer coefficients | `(0.08, 0.14, 0.09, 0.09)` m s^-1 | explicit tutorial assumptions |
| external heat-transfer coefficient | 250 W m^-2 K^-1 | explicit tutorial assumption; selected before optimization to keep the uniform reference on a steady low-temperature branch, not fitted to a target profile |
| reaction enthalpy | -164 kJ mol^-1 CO2 | fixed tutorial thermochemical approximation |
| temperature ceiling | 613 K | upper end of the published kinetic-correlation range, stricter than the 725 K particle-design limit in Zimmermann et al. |
| mean catalyst activity | 0.16 | explicit design inventory |

Every assumption above is a field of `PelletConfig`; none is hidden in the
optimizer. Replace the assumed transport data before treating the calculation
as a design for a particular support or reactor.

Primary references:

- F. Koschany, D. Schlereth, and O. Hinrichsen, *Applied Catalysis B* 181
  (2016) 504-516, [doi:10.1016/j.apcatb.2015.07.026](https://doi.org/10.1016/j.apcatb.2015.07.026).
- R. T. Zimmermann, J. Bremer, and K. Sundmacher, *Chemical Engineering
  Journal* 387 (2020) 123704,
  [doi:10.1016/j.cej.2019.123704](https://doi.org/10.1016/j.cej.2019.123704).
- R. Baratti, H. Wu, M. Morbidelli, and A. Varma, *Chemical Engineering
  Science* 48 (1993) 1869-1881,
  [doi:10.1016/0009-2509(93)80357-V](https://doi.org/10.1016/0009-2509(93)80357-V).

## Equations and units

For species `i`, positive `nu_i` denotes production. In a spherical pellet,

```text
(1/r^2) d/dr (r^2 D_i dc_i/dr) + nu_i rho_cat a(r) r_K(c,T) = 0
(1/r^2) d/dr (r^2 k_eff dT/dr)
    + (-Delta H) rho_cat a(r) r_K(c,T) = 0
```

with stoichiometry `nu = (-1, -4, 1, 2)`. Concentrations are mol m^-3,
temperature K, `D_i` m^2 s^-1, `k_eff` W m^-1 K^-1, and `r_K` mol CO2
(g_cat s)^-1. The ideal-gas relation converts cell concentrations to partial
pressures in bar for the kinetic law.

The Koschany rate is

```text
r_K = k sqrt(p_H2 p_CO2)
      [1 - p_CH4 p_H2O^2 / (K_eq p_CO2 p_H2^4)]
      / [1 + K_OH p_H2O/sqrt(p_H2)
           + K_H2 sqrt(p_H2) + K_mix sqrt(p_CO2)]^2.
```

The Arrhenius/van't Hoff constants and their units are collected in
`KoschanyKinetics`. At 555 K, 1 bar CO2, 4 bar H2, and zero products, the
implementation returns `9.084226002938914e-5 mol (g_cat s)^-1`; CI pins this
published-table calculation independently.

At `r=0`, every flux is zero. At `r=R`, finite films impose inward species
transfer `k_m,i (c_i,bulk - c_i,surface)` and outward heat transfer
`h (T_surface - T_bulk)`. The discretization uses equal-volume spherical
finite volumes. The center face has exactly zero area, so it never evaluates a
numerical `1/r` term. Summing the cell equations reproduces the external molar
and heat fluxes; the test and notebook report both closure errors.

## Validation ladder

The tutorial does not optimize until these checks pass:

1. `solve_first_order_sphere` reproduces the analytical sphere effectiveness
   factor

   ```text
   eta(phi) = (3/phi) [coth(phi) - 1/phi]
   ```

   from reaction-limited through diffusion-limited conditions.
2. The uniform four-species pellet closes each integrated species balance and
   the energy balance, stays positive, respects the 613 K ceiling, responds in
   the expected direction when the external film is slowed, and is re-solved
   on a finer mesh.
3. The implicit-function derivative
   `ds/da = -(dR/ds)^-1 (dR/da)` is checked against full central
   perturb-and-resolve calculations for production and peak temperature.
4. A small nested outer optimization and the simultaneous POUNCE NLP are timed
   and compared. They use the same fixed-mesh physics but different nonlinear
   algorithms. Agreement supplies an independent route check; the simultaneous
   form is retained because all balance equations, state bounds, the inventory,
   and the thermal ceiling remain explicit to POUNCE.

The optimized profile is always re-solved after interpolating its state to a
finer finite-volume mesh. That forward refinement is outside the optimization
NLP and catches basis/mesh artifacts.

## Nominal design problem

The pellet volume is divided into a small number of equal-volume activity
zones. The NLP maximizes normalized methane production with a quadratic
manufacturability penalty,

```text
maximize  production / production_uniform
          - lambda sum_j (a[j+1] - a[j])^2
subject to 0 <= a[j] <= 1
           sum_j volume_fraction[j] a[j] = 0.16
           species balances, energy balance, c_i >= 0, T <= 613 K.
```

`lambda=0.5` is fixed before the solve. The notebook compares equal-inventory
uniform, ideal step egg-shell, and regularized optimized profiles. The
unregularized bounded-loading limit is shell-like and step-like, consistent
with the classical loading result of Baratti et al.; regularization trades a
small amount of production for a less abrupt outer profile.

Zimmermann et al. obtained a different egg-yolk motif: active core plus an
inert, low-permeability shell, while jointly changing activity, permeability,
thermal conductivity, and a reactor trajectory. This tutorial holds
permeability and conductivity fixed and optimizes activity in one prescribed
bulk state, so its outer-active activity profile is **not** a reproduction of
their coupled optimum. The shared qualitative result is that bounds and
transport create structured, near-step radial designs; the opposite placement
is a documented model-form difference, not a parameter-tuning failure.

## Covariance and worst-case redesign

The study is labelled **synthetic calibration**, not experimental validation.
It creates log-rate observations for intrinsic powder and two pellet radii,
fits two interpretable log multipliers (intrinsic rate and CO2 effective
diffusivity), and obtains their covariance from POUNCE's reduced Hessian.
Intrinsic data break the rate/diffusion confounding that apparent pellet rates
alone would retain.

The uncertainty set contains the fitted mean and both directions of each
covariance principal axis at 1.645 standard deviations. The robust simultaneous
NLP adds one epigraph variable `q` and enforces

```text
production_scenario / production_reference >= q
```

for every scenario while sharing one activity profile. It maximizes `q` minus
the same manufacturability penalty. The reported `guaranteed_production_mol_s`
is therefore an enforced lower bound over this finite scenario set, not a distribution-
free or global guarantee. Full sampled nonlinear re-solves are compared with
delta-method standard deviations for production and peak temperature.

## Reproduce it

From the repository root, with the Python development environment installed:

```sh
PYTHONPATH=python python -m pytest -q \
  python/tests/test_catalyst_pellet_example.py
jupyter nbconvert --to notebook --execute --inplace \
  python/notebooks/37_catalyst_pellet_inverse_design.ipynb
```

## Limitations

- Every design is a local NLP solution. Two documented initializations agree
  in the short case, but neither that check nor POUNCE certifies a global
  optimum for this nonlinear model.
- Effective diffusivities, porosity, films, and reaction enthalpy are tutorial
  assumptions. Model-form uncertainty is not in the two-parameter covariance.
- Independent Fick diffusion omits Stefan-Maxwell coupling and pressure-driven
  pore transport; the prescribed bulk state omits axial reactor feedback.
- Activity is piecewise constant. No pore morphology, minimum physical feature
  size, thermal-conductivity design, permeability design, transient operation,
  or dead-core/free-boundary model is claimed.
- The finite principal-axis set and local delta method cover nearby parameter
  uncertainty only. Sampled re-solves validate that local approximation; they
  do not turn synthetic data into experimental evidence.

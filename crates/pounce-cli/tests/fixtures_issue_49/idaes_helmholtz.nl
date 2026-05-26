g3 1 1 0	# problem unknown
 3 3 0 0 3 	# vars, constraints, objectives, ranges, eqns
 2 0 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 3 0 0 	# nonlinear vars in constraints, objectives, both
 0 3 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 7 0 	# nonzeros in Jacobian, obj. gradient
 51 53	# max name lengths: constraints, variables
 0 5 0 6 0	# common exprs: b,c,o,c1,o1
F0 1 -1 vf_hp
F1 1 -1 h_liq_hp
F2 1 -1 h_vap_hp
V3 1 0	#fs.heater.control_volume.properties_out[0.0].h_kJ_per_kg
1 0.055508472036052976
n0
V4 1 0	#fs.heater.control_volume.properties_out[0.0].p_kPa
2 0.001
n0
V5 0 0	#fs.heater.control_volume.properties_out[0.0].vapor_frac
f0 4	#vf_hp_func
h3:h2o
v3	#fs.heater.control_volume.properties_out[0.0].h_kJ_per_kg
v4	#fs.heater.control_volume.properties_out[0.0].p_kPa
h126:/Users/jkitchin/Dropbox/uv/.venv/lib/python3.12/site-packages/idaes/models/properties/general_helmholtz/components/parameters/
V6 0 0	#fs.heater.control_volume.properties_out[0.0].phase_frac[Liq]
o0	#+
o16	#-
v5	#fs.heater.control_volume.properties_out[0.0].vapor_frac
n1.0
V7 0 0	#fs.heater.control_volume.properties_out[0.0].phase_frac[Vap]
f0 4	#vf_hp_func
h3:h2o
v3	#fs.heater.control_volume.properties_out[0.0].h_kJ_per_kg
v4	#fs.heater.control_volume.properties_out[0.0].p_kPa
h126:/Users/jkitchin/Dropbox/uv/.venv/lib/python3.12/site-packages/idaes/models/properties/general_helmholtz/components/parameters/
V8 0 1	#fs.heater.control_volume.properties_out[0.0].material_flow_terms[Liq]
o2	#*
v0	#fs.heater.control_volume.properties_out[0.0].flow_mol
v6	#fs.heater.control_volume.properties_out[0.0].phase_frac[Liq]
V9 0 1	#fs.heater.control_volume.properties_out[0.0].material_flow_terms[Vap]
o2	#*
v0	#fs.heater.control_volume.properties_out[0.0].flow_mol
v7	#fs.heater.control_volume.properties_out[0.0].phase_frac[Vap]
C0	#fs.heater.control_volume.material_balances[0.0,h2o]
o16	#-
o0	#+
v8	#fs.heater.control_volume.properties_out[0.0].material_flow_terms[Liq]
v9	#fs.heater.control_volume.properties_out[0.0].material_flow_terms[Vap]
V10 0 2	#fs.heater.control_volume.properties_out[0.0].enth_mol_phase[Liq]
o2	#*
n18.015268000000003
f1 4	#h_liq_hp_func
h3:h2o
v3	#fs.heater.control_volume.properties_out[0.0].h_kJ_per_kg
v4	#fs.heater.control_volume.properties_out[0.0].p_kPa
h126:/Users/jkitchin/Dropbox/uv/.venv/lib/python3.12/site-packages/idaes/models/properties/general_helmholtz/components/parameters/
V11 0 2	#fs.heater.control_volume.properties_out[0.0].enthalpy_flow_terms[Liq]
o2	#*
o2	#*
v10	#fs.heater.control_volume.properties_out[0.0].enth_mol_phase[Liq]
v6	#fs.heater.control_volume.properties_out[0.0].phase_frac[Liq]
v0	#fs.heater.control_volume.properties_out[0.0].flow_mol
V12 0 2	#fs.heater.control_volume.properties_out[0.0].enth_mol_phase[Vap]
o2	#*
n18.015268000000003
f2 4	#h_vap_hp_func
h3:h2o
v3	#fs.heater.control_volume.properties_out[0.0].h_kJ_per_kg
v4	#fs.heater.control_volume.properties_out[0.0].p_kPa
h126:/Users/jkitchin/Dropbox/uv/.venv/lib/python3.12/site-packages/idaes/models/properties/general_helmholtz/components/parameters/
V13 0 2	#fs.heater.control_volume.properties_out[0.0].enthalpy_flow_terms[Vap]
o2	#*
o2	#*
v12	#fs.heater.control_volume.properties_out[0.0].enth_mol_phase[Vap]
v7	#fs.heater.control_volume.properties_out[0.0].phase_frac[Vap]
v0	#fs.heater.control_volume.properties_out[0.0].flow_mol
C1	#fs.heater.control_volume.enthalpy_balances[0.0]
o16	#-
o0	#+
v11	#fs.heater.control_volume.properties_out[0.0].enthalpy_flow_terms[Liq]
v13	#fs.heater.control_volume.properties_out[0.0].enthalpy_flow_terms[Vap]
C2	#fs.heater.control_volume.pressure_balance[0.0]
n0
x3	# initial guess
0 1.0	#fs.heater.control_volume.properties_out[0.0].flow_mol
1 1878.71	#fs.heater.control_volume.properties_out[0.0].enth_mol
2 101325.0	#fs.heater.control_volume.properties_out[0.0].pressure
r	#3 ranges (rhs's)
4 -1.0	#fs.heater.control_volume.material_balances[0.0,h2o]
4 -1878.7099999999614	#fs.heater.control_volume.enthalpy_balances[0.0]
4 -101325	#fs.heater.control_volume.pressure_balance[0.0]
b	#3 bounds (on variables)
3	#fs.heater.control_volume.properties_out[0.0].flow_mol
0 0.011021386931383999 80765.34157807355	#fs.heater.control_volume.properties_out[0.0].enth_mol
0 1.0000000000000002e-06 1100000000.0	#fs.heater.control_volume.properties_out[0.0].pressure
k2	#intermediate Jacobian column lengths
2
4
J0 3	#fs.heater.control_volume.material_balances[0.0,h2o]
0 0
1 0
2 0
J1 3	#fs.heater.control_volume.enthalpy_balances[0.0]
0 0
1 0
2 0
J2 1	#fs.heater.control_volume.pressure_balance[0.0]
2 -1

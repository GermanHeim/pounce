g3 1 1 0	# problem unknown
 3 1 1 0 1 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 3 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 1 3 	# nonzeros in Jacobian, obj. gradient
 3 1	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
S0 1 sens_state_1
1 1
S4 1 sens_state_value_1
1 1.5
S1 1 sens_init_constr
0 1
C0	#pin
n0
O0 0	#obj
o0	#+
o5	#^
o0	#+
v0	#x
o2	#*
n-1
v1	#p
n2
o5	#^
v2	#y
n2
x3	# initial guess
0 0.0	#x
1 1.0	#p
2 0.0	#y
r	#1 ranges (rhs's)
4 1.0	#pin
b	#3 bounds (on variables)
3	#x
3	#p
3	#y
k2	#intermediate Jacobian column lengths
0
1
J0 1	#pin
1 1
G0 3	#obj
0 0
1 0
2 0

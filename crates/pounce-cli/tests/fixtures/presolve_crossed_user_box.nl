g3 1 1 0	# problem crossedbox
 1 0 1 0 0 	# vars, constraints, objectives, ranges, eqns
 0 1	# nonlinear constraints, objectives
 0 0	# network constraints: nonlinear, linear
 0 1 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 0 1 	# nonzeros in Jacobian, obj. gradient
 1 1	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
O0 0	#o
o5	#^
v0	#x
n2
x1	# initial guess
0 4.0	#x
r	#0 ranges
b	#1 bounds (on variables)
0 5 3	#x crossed: l=5 > u=3
k0	#intermediate Jacobian column lengths
G0 1	#o
0 0

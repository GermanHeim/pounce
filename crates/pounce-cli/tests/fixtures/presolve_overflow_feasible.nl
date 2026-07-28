g3 1 1 0	# problem unknown
 1 1 1 0 0 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 1 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 1 1 	# nonzeros in Jacobian, obj. gradient
 1 1	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c
n0
O0 0	#o
o5	#^
v0	#x
n2
x1	# initial guess
0 0.0	#x
r	#1 ranges (rhs's)
2 1e+300	#c
b	#1 bounds (on variables)
0 -1e+300 1e+300	#x
k0	#intermediate Jacobian column lengths
J0 1	#c
0 1e+300
G0 1	#o
0 0

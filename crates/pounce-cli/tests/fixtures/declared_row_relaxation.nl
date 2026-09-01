g3 1 1 0	# problem unknown
 2 1 1 0 0 	# vars, constraints, objectives, ranges, eqns
 0 0 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 0 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 2 2 	# nonzeros in Jacobian, obj. gradient
 3 1	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#cap
n0
O0 0	#obj
n0
x2	# initial guess
0 0.0	#x
1 0.0	#y
r	#1 ranges (rhs's)
1 500.0	#cap
b	#2 bounds (on variables)
0 0 1000	#x
0 0 1000	#y
k1	#intermediate Jacobian column lengths
1
J0 2	#cap
0 1
1 1
G0 2	#obj
0 -1
1 -1

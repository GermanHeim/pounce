g3 1 1 0	# problem unknown
 2 2 1 0 0 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 2 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 4 2 	# nonzeros in Jacobian, obj. gradient
 4 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c[1]
n0
C1	#c[2]
n0
O0 0	#o
o0	#+
o5	#^
o0	#+
v0	#x[0]
n-500000.0
n2
o5	#^
o0	#+
v1	#x[1]
n-500000.0
n2
x2	# initial guess
0 500000.0	#x[0]
1 500000.0	#x[1]
r	#2 ranges (rhs's)
2 -1e-06	#c[1]
2 -1.01	#c[2]
b	#2 bounds (on variables)
0 0.0 1000000.0	#x[0]
0 0.0 1000000.0	#x[1]
k1	#intermediate Jacobian column lengths
2
J0 2	#c[1]
0 -1e+30
1 1e+30
J1 2	#c[2]
0 -1e-08
1 -1e-08
G0 2	#o
0 0
1 0

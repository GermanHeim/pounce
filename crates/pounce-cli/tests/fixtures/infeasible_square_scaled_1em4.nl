g3 1 1 0	# problem unknown
 2 2 1 0 2 	# vars, constraints, objectives, ranges, eqns
 1 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 2 2 2 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 4 2 	# nonzeros in Jacobian, obj. gradient
 2 1	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#e1
o2	#*
n0.0001
o2	#*
v0	#x
v1	#y
C1	#e2
n0
O0 0	#o
o0	#+
o5	#^
v0	#x
n2
o5	#^
v1	#y
n2
x2	# initial guess
0 0.5	#x
1 -0.5	#y
r	#2 ranges (rhs's)
4 0.0001	#e1
4 5e-05	#e2
b	#2 bounds (on variables)
0 -10 10	#x
0 -10 10	#y
k1	#intermediate Jacobian column lengths
2
J0 2	#e1
0 0
1 0
J1 2	#e2
0 0.0001
1 0.0001
G0 2	#o
0 0
1 0

g3 1 1 0	# problem unknown
 2 1 1 0 0 	# vars, constraints, objectives, ranges, eqns
 1 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 2 2 2 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 2 2 	# nonzeros in Jacobian, obj. gradient
 1 1	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c
o2	#*
v0	#x
v1	#f
O0 0	#o
o0	#+
o5	#^
o0	#+
v0	#x
n-3
n2
o5	#^
o0	#+
v1	#f
n-1
n2
x2	# initial guess
0 0.5	#x
1 2.0	#f
r	#1 ranges (rhs's)
2 0.0	#c
b	#2 bounds (on variables)
0 0 1	#x
4 2.0	#f
k1	#intermediate Jacobian column lengths
1
J0 2	#c
0 0
1 0
G0 2	#o
0 0
1 0

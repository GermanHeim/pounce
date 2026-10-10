g3 1 1 0	# problem unknown
 2 2 1 0 2 	# vars, constraints, objectives, ranges, eqns
 2 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 2 2 2 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 4 2 	# nonzeros in Jacobian, obj. gradient
 2 2	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c1
o0	#+
o5	#^
v0	#x1
n2
o5	#^
v1	#x2
n2
C1	#c2
o16	#-
o5	#^
v0	#x1
n2
O0 0	#o
o0	#+
o5	#^
o0	#+
v0	#x1
n-1
n2
o5	#^
o0	#+
v1	#x2
n2
n2
x2	# initial guess
0 1.0	#x1
1 -0.5	#x2
r	#2 ranges (rhs's)
4 1	#c1
4 1.5	#c2
b	#2 bounds (on variables)
3	#x1
3	#x2
k1	#intermediate Jacobian column lengths
2
J0 2	#c1
0 0
1 0
J1 2	#c2
0 0
1 1
G0 2	#o
0 0
1 0

g3 1 1 0	# problem unknown
 2 2 1 0 0 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 2 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 2 2 	# nonzeros in Jacobian, obj. gradient
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
n0.5
n2
o5	#^
o0	#+
v1	#x[1]
n-499998.3031417761
n2
x2	# initial guess
0 -0.5	#x[0]
1 499998.3031417761	#x[1]
r	#2 ranges (rhs's)
2 49999830314176.61	#c[1]
1 34489200259824.066	#c[2]
b	#2 bounds (on variables)
0 -1.0 0.0	#x[0]
0 -1.696858223856672 999998.3031417761	#x[1]
k1	#intermediate Jacobian column lengths
0
J0 1	#c[1]
1 100000000.0
J1 1	#c[2]
1 68978634.61357497
G0 2	#o
0 0
1 0

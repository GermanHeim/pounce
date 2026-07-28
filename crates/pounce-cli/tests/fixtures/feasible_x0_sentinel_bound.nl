g3 1 1 0	# problem unknown
 3 3 1 0 0 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 3 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 5 3 	# nonzeros in Jacobian, obj. gradient
 4 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c[1]
n0
C1	#c[2]
n0
C2	#c[3]
n0
O0 0	#o
o54	# sumlist
3	# (n)
o5	#^
o0	#+
v0	#x[0]
n-5e-10
n2
o5	#^
o0	#+
v1	#x[1]
n3.9190147700811693
n2
o5	#^
o0	#+
v2	#x[2]
n1.2913940476175876
n2
x3	# initial guess
0 5e-10	#x[0]
1 -3.9190147700811693	#x[1]
2 -1.2913940476175876	#x[2]
r	#3 ranges (rhs's)
1 5.2104e-320	#c[1]
2 1.29139404770787e+18	#c[2]
1 -5.0000000000000007e+20	#c[3]
b	#3 bounds (on variables)
0 0.0 1e-09	#x[0]
0 -3.9190147705811693 -3.9190147695811692	#x[1]
0 -6.291394047617588 3.7086059523824124	#x[2]
k2	#intermediate Jacobian column lengths
2
3
J0 2	#c[1]
1 -1e-320
2 -1e-320
J1 2	#c[2]
0 1.8056493234311766e+17
2 -1e+18
J2 1	#c[3]
0 -1e+30
G0 3	#o
0 0
1 0
2 0

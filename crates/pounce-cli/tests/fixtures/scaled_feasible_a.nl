g3 1 1 0	# problem unknown
 4 3 1 0 0 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 4 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 9 4 	# nonzeros in Jacobian, obj. gradient
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
4	# (n)
o5	#^
o0	#+
v0	#x[0]
n3.8075263197246603
n2
o5	#^
o0	#+
v1	#x[1]
n-500000.0
n2
o5	#^
o0	#+
v2	#x[2]
n-0.5
n2
o5	#^
o0	#+
v3	#x[3]
n-499999.0
n2
x4	# initial guess
0 -3.8075263197246603	#x[0]
1 500000.0	#x[1]
2 0.5	#x[2]
3 499999.0	#x[3]
r	#3 ranges (rhs's)
1 530752631.97246605	#c[1]
2 -26547947618426.85	#c[2]
1 8.737549066070076e-07	#c[3]
b	#4 bounds (on variables)
0 -3.8075263202246603 -3.8075263192246602	#x[0]
0 0.0 1000000.0	#x[1]
0 0.0 1.0	#x[2]
0 -1.0 999999.0	#x[3]
k3	#intermediate Jacobian column lengths
1
4
6
J0 4	#c[1]
0 -100000000.0
1 100000000.0
2 100000000.0
3 -100000000.0
J1 2	#c[2]
1 46903904.763146296
3 -100000000.0
J2 3	#c[3]
1 1e-12
2 1e-12
3 7.475083082306318e-13
G0 4	#o
0 0
1 0
2 0
3 0

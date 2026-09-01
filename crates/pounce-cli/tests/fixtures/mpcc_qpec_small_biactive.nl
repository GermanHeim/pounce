g3 1 1 0	# problem unknown
 3 6 1 0 2 	# vars, constraints, objectives, ranges, eqns
 2 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 3 3 3 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 10 3 	# nonzeros in Jacobian, obj. gradient
 3 2	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#p1
o2	#*
v1	#y1
o54	# sumlist
3	# (n)
o2	#*
n-1
v0	#x
o2	#*
n2
v1	#y1
n-1
C1	#p2
o2	#*
v2	#y2
o54	# sumlist
3	# (n)
v0	#x
o2	#*
n2
v2	#y2
n-1
C2	#cG1
n0
C3	#cH1
n0
C4	#cG2
n0
C5	#cH2
n0
O0 0	#obj
o54	# sumlist
3	# (n)
o5	#^
o0	#+
v0	#x
n-1
n2
o5	#^
o0	#+
v1	#y1
n-1
n2
o5	#^
v2	#y2
n2
x3	# initial guess
0 0.0	#x
1 0.5	#y1
2 0.5	#y2
r	#6 ranges (rhs's)
4 0	#p1
4 0	#p2
2 0	#cG1
2 1	#cH1
2 0	#cG2
2 1	#cH2
b	#3 bounds (on variables)
0 0 2	#x
3	#y1
3	#y2
k2	#intermediate Jacobian column lengths
4
7
J0 2	#p1
0 0
1 0
J1 2	#p2
0 0
2 0
J2 1	#cG1
1 1
J3 2	#cH1
0 -1
1 2
J4 1	#cG2
2 1
J5 2	#cH2
0 1
2 2
G0 3	#obj
0 0
1 0
2 0

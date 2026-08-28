"""Square, badly-scaled, FEASIBLE-BY-CONSTRUCTION flowsheet-like model (gh#815 shape).

Each constraint is expr(x) == expr(x*), so x* is an exact solution.
Magnitudes span 1e-6 (trace fractions) to 3e6 (Pa), as in an IDAES flowsheet.
"""
import sys, math
import pyomo.environ as pyo

NS = 8

def xstar(i):
    return dict(P=3.0e6 - 1.0e4 * i, F=30.0 + 0.7 * i, y=1.0e-6 * (1.0 + 0.3 * i),
                T=600.0 + 5.0 * i)

def build():
    m = pyo.ConcreteModel()
    m.S = pyo.Set(initialize=list(range(NS)))
    xs = {i: xstar(i) for i in range(NS)}
    m.P = pyo.Var(m.S, initialize=lambda m, i: xs[i]['P'], bounds=(1e5, 1e7))
    m.F = pyo.Var(m.S, initialize=lambda m, i: xs[i]['F'], bounds=(1e-3, 1e4))
    m.y = pyo.Var(m.S, initialize=lambda m, i: xs[i]['y'], bounds=(1e-12, 1.0))
    m.T = pyo.Var(m.S, initialize=lambda m, i: xs[i]['T'], bounds=(300.0, 900.0))

    def _mass(P, F, y, T, i):
        p = (i - 1) % NS
        return F[i] - 0.97 * F[p] - 0.5 * y[p] * F[p]
    def _press(P, F, y, T, i):
        p = (i - 1) % NS
        return P[i] - P[p] + 1.5e3 * (F[i] / 30.0) ** 2
    def _equil(P, F, y, T, i):
        return y[i] * P[i] - 1.0e-3 * pyo.exp(-6000.0 / T[i]) * F[i]
    def _energy(P, F, y, T, i):
        p = (i - 1) % NS
        return 75.0 * F[i] * (T[i] - 298.15) - 75.0 * F[p] * (T[p] - 298.15) \
               + 2.2e5 * y[i] * F[i]

    Ps = {i: xs[i]['P'] for i in range(NS)}
    Fs = {i: xs[i]['F'] for i in range(NS)}
    ys = {i: xs[i]['y'] for i in range(NS)}
    Ts = {i: xs[i]['T'] for i in range(NS)}
    # exp() on floats for the rhs constants
    class _E:
        exp = staticmethod(math.exp)
    import types
    def rhs(fn, i):
        g = fn.__globals__
        old = g.get('pyo')
        g['pyo'] = _E
        try:
            return fn(Ps, Fs, ys, Ts, i)
        finally:
            g['pyo'] = old

    m.mass = pyo.Constraint(m.S, rule=lambda m, i: _mass(m.P, m.F, m.y, m.T, i) == rhs(_mass, i))
    m.press = pyo.Constraint(m.S, rule=lambda m, i: _press(m.P, m.F, m.y, m.T, i) == rhs(_press, i))
    m.equil = pyo.Constraint(m.S, rule=lambda m, i: _equil(m.P, m.F, m.y, m.T, i) == rhs(_equil, i))
    m.energy = pyo.Constraint(m.S, rule=lambda m, i: _energy(m.P, m.F, m.y, m.T, i) == rhs(_energy, i))
    m.obj = pyo.Objective(expr=0.0)
    return m

def perturb(m, fac):
    for i in m.S:
        xs = xstar(i)
        m.P[i].value = xs['P'] * fac
        m.F[i].value = xs['F'] / fac
        m.y[i].value = xs['y'] * fac
        m.T[i].value = xs['T'] * (1.0 + 0.05 * (fac - 1.0))

if __name__ == "__main__":
    solver = sys.argv[1]; fac = float(sys.argv[2])
    m = build()
    # residual at x* must be ~0
    perturb(m, 1.0)
    r = max(abs(pyo.value(c.body) - pyo.value(c.upper))
            for c in m.component_data_objects(pyo.Constraint, active=True))
    print(f"# max |residual| at x*: {r:.3e}")
    perturb(m, fac)
    r0 = max(abs(pyo.value(c.body) - pyo.value(c.upper))
             for c in m.component_data_objects(pyo.Constraint, active=True))
    print(f"# max |residual| at start (fac={fac}): {r0:.3e}")
    res = pyo.SolverFactory(solver).solve(m, tee=True)
    print("STATUS:", res.solver.status, "|", res.solver.termination_condition)

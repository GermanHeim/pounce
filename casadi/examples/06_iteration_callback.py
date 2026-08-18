#!/usr/bin/env python3
"""Watching a solve iterate by iterate, and stopping it early.

CasADi's `iteration_callback` is handed the full iterate — `x`, `f`, `g`,
`lam_x`, `lam_g` — once per iteration, and a nonzero return asks the
solver to stop.

This is worth calling out: with the bundled Ipopt plugin, a stock Ipopt
build cannot supply the iterate, and CasADi prints *"intermediate_callback
is disfunctional in your installation"* and passes the callback nothing
usable. POUNCE serves live iterates through its C API, so the callback
below sees real numbers with no special build.
"""

import casadi as ca

x = ca.MX.sym("x", 2)
f = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
g = x[0] ** 2 + x[1] ** 2 - 1.5
nlp = {"x": x, "f": f, "g": g}

NX, NG, NP = 2, 1, 0


class Watcher(ca.Callback):
    """Record the trajectory; optionally stop once the objective is small."""

    def __init__(self, name, stop_below=None, opts={}):
        ca.Callback.__init__(self)
        self.stop_below = stop_below
        self.trajectory = []
        self.construct(name, opts)

    # `iteration_callback` is called with the nlpsol *outputs*.
    def get_n_in(self):
        return ca.nlpsol_n_out()

    def get_n_out(self):
        return 1

    def get_name_in(self, i):
        return ca.nlpsol_out(i)

    def get_sparsity_in(self, i):
        name = ca.nlpsol_out(i)
        dims = {"f": 1, "x": NX, "g": NG, "lam_x": NX, "lam_g": NG, "lam_p": NP}
        n = dims.get(name, 0)
        return ca.Sparsity.dense(n, 1) if n else ca.Sparsity(0, 0)

    def eval(self, arg):
        named = dict(zip(ca.nlpsol_out(), arg))
        xk = named["x"].full().ravel()
        fk = float(named["f"])
        self.trajectory.append((xk.copy(), fk))
        print(f"  iter {len(self.trajectory) - 1:2d}: x = [{xk[0]:+.6f}, {xk[1]:+.6f}]  f = {fk:.6e}")
        # Returning nonzero requests termination (status User_Requested_Stop).
        if self.stop_below is not None and fk < self.stop_below:
            print("  -> objective below threshold, asking the solver to stop")
            return [1]
        return [0]


print("plain run:")
watcher = Watcher("watcher")
solver = ca.nlpsol("solver", "pounce", nlp, {
    "print_time": False,
    "iteration_callback": watcher,
    "pounce": {"print_level": 0},
})
sol = solver(x0=[-1.2, 1.0], lbg=-ca.inf, ubg=0)
print(f"  status = {solver.stats()['return_status']}, "
      f"{len(watcher.trajectory)} callback fires")

print("\nfull diagnostics from inside the callback:")


class Diagnostic(ca.Callback):
    """Everything POUNCE's iteration table shows, without parsing stdout.

    CasADi fixes the callback's inputs at (x, f, g, lam_x, lam_g), so the
    convergence metrics do not arrive as arguments. They are reachable
    anyway: `stats()` is callable from inside the callback, and mid-solve
    its per-iteration traces end on the iteration you are in.
    """

    def __init__(self, name, opts={}):
        ca.Callback.__init__(self)
        self.solver = None          # set after the solver is constructed
        self.construct(name, opts)

    def get_n_in(self):
        return ca.nlpsol_n_out()

    def get_n_out(self):
        return 1

    def get_name_in(self, i):
        return ca.nlpsol_out(i)

    def get_sparsity_in(self, i):
        name = ca.nlpsol_out(i)
        dims = {"f": 1, "x": NX, "g": NG, "lam_x": NX, "lam_g": NG, "lam_p": NP}
        n = dims.get(name, 0)
        return ca.Sparsity.dense(n, 1) if n else ca.Sparsity(0, 0)

    def eval(self, arg):
        it = self.solver.stats()["iterations"]
        k = len(it["inf_pr"]) - 1        # the entry for *this* iteration
        print(f"  iter {k:2d}: obj = {it['obj'][k]:+.6e}  inf_pr = {it['inf_pr'][k]:.2e}  "
              f"inf_du = {it['inf_du'][k]:.2e}  mu = {it['mu'][k]:.2e}  "
              f"|d| = {it['d_norm'][k]:.2e}  ls = {it['ls_trials'][k]}")

        # Present only while a solve is running: Ipopt's current-violation
        # field set, fetched on demand.
        viol = self.solver.stats().get("current_violations")
        if viol is not None and k in (0, 1):
            print(f"          |grad_lag_x| = {max(abs(v) for v in viol['grad_lag_x']):.2e}  "
                  f"g violation = {max((abs(v) for v in viol['nlp_constraint_violation']), default=0.0):.2e}")
        return [0]


diag = Diagnostic("diag")
solver3 = ca.nlpsol("solver3", "pounce", nlp, {
    "print_time": False,
    "iteration_callback": diag,
    "pounce": {"print_level": 0},
})
diag.solver = solver3
solver3(x0=[-1.2, 1.0], lbg=-ca.inf, ubg=0)

st = solver3.stats()
print(f"  final: inf_pr = {st['final_inf_pr']:.2e}  inf_du = {st['final_inf_du']:.2e}  "
      f"compl = {st['final_compl_inf']:.2e}")
print(f"  linear solver: {st['linear_solver']['solver_name']}, "
      f"{st['linear_solver']['n_factors']} factorizations "
      f"({st['linear_solver']['n_pattern_reuse']} reusing the pattern)")
print(f"  restoration: {st['restoration']['calls']} call(s), "
      f"{st['restoration']['wall_secs']:.4f}s")

print("\nearly stop at f < 0.05:")
stopper = Watcher("stopper", stop_below=0.05)
solver2 = ca.nlpsol("solver2", "pounce", nlp, {
    "print_time": False,
    "iteration_callback": stopper,
    "pounce": {"print_level": 0},
})
sol2 = solver2(x0=[-1.2, 1.0], lbg=-ca.inf, ubg=0)
print(f"  status = {solver2.stats()['return_status']}")
print(f"  stopped at x = {sol2['x'].full().ravel()}")

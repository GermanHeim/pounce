"""Warm-starting a parametric NLP sweep: where it's a wash, and where it pays.

Two mechanisms, two regimes:

PART 1 -- active-set SQP ``working_set`` carry-over (API-correctness demo).

   x*(p) = argmin 1/2 ||x - p||^2  s.t.  x >= 0,  sum(x_i) = 1
   (simplex projection -- the canonical "active set rotates with p" case).

   For a convex quadratic objective + linear constraints, active-set SQP
   lands at the optimum in ONE outer iteration whether cold or warm: the
   QP subproblem IS the original problem. The working-set carry-over is
   correct, but its payoff lives *inside* the QP solver (fewer active-set
   pivots), which the per-solve outer iter-count does not surface. So this
   part shows warm-starting being a wash -- and explains why.

PART 2 -- interior-point warm start across a receding-horizon sweep, where
   the payoff IS visible. We track a moving setpoint on a curved manifold

       x*(r) = argmin ||x - r||^2  s.t.  ||x||^2 = R^2,  -5 <= x_i <= 5,

   re-solving each step cold (from a fixed naive x0, i.e. no information)
   and warm (from the previous step's solution plus a tuned interior-point
   warm start -- small ``mu_init`` and tight ``warm_start_*_bound_push``).
   The per-step iteration count drops markedly once warm, the way an MPC
   loop benefits from carrying the previous solution forward.

Run with:

    python python/examples/sqp_warm_start_mpc.py

Requires ``pip install pounce`` (or ``maturin develop`` against the
workspace).
"""

import numpy as np

import pounce


# ===========================================================================
# Part 1 -- active-set SQP working-set carry-over (convex QP, a wash)
# ===========================================================================
class SimplexProjection:
    """min 1/2 ||x - p||^2 s.t. x >= 0, sum(x) = 1.

    With n variables and 1 equality constraint. The bound x >= 0 is the
    warm-start-relevant active set; the equality always binds.
    """

    def __init__(self, n):
        self.n = n
        self.p = np.zeros(n)

    def set_parameter(self, p):
        self.p = np.asarray(p, dtype=np.float64)

    def objective(self, x):
        d = x - self.p
        return 0.5 * float(d @ d)

    def gradient(self, x):
        return x - self.p

    def constraints(self, x):
        return np.array([float(np.sum(x))])

    def jacobianstructure(self):
        return (np.zeros(self.n, dtype=np.int64),
                np.arange(self.n, dtype=np.int64))

    def jacobian(self, x):
        return np.ones(self.n)

    def hessianstructure(self):
        idx = np.arange(self.n, dtype=np.int64)
        return (idx, idx)

    def hessian(self, x, lagrange, obj_factor):
        return np.full(self.n, obj_factor)


def make_sqp_problem(prob_obj, n):
    p = pounce.Problem(
        n=n, m=1, problem_obj=prob_obj,
        lb=[0.0] * n, ub=[1e20] * n,
        cl=[1.0], cu=[1.0],
    )
    p.add_option("algorithm", "active-set-sqp")
    p.add_option("print_level", 0)
    p.add_option("sqp_tol", 1e-9)
    return p


def working_set_demo():
    np.random.seed(0)
    n = 8
    n_steps = 20

    centre = np.full(n, 1.0 / n)
    direction = np.random.randn(n)
    direction -= direction.mean()
    direction /= np.linalg.norm(direction)
    radius = 0.2

    cold_iters = []
    warm_iters = []
    last_ws = None

    print(f"{'step':>4} {'cold iters':>10} {'warm iters':>10}  "
          f"{'||x* - x*_warm||':>18}")
    print("-" * 48)

    for k in range(n_steps):
        theta = 2.0 * np.pi * k / n_steps
        p_k = centre + radius * np.cos(theta) * direction

        cold_obj = SimplexProjection(n)
        cold_obj.set_parameter(p_k)
        x_cold, info_cold = make_sqp_problem(cold_obj, n).solve(x0=centre.copy())
        assert info_cold["status_msg"] == "Solve_Succeeded", \
            f"cold solve failed at step {k}: {info_cold['status_msg']}"
        cold_iters.append(info_cold["iter_count"])

        warm_obj = SimplexProjection(n)
        warm_obj.set_parameter(p_k)
        warm_prob = make_sqp_problem(warm_obj, n)
        kwargs = {}
        if last_ws is not None:
            kwargs["working_set"] = last_ws
        x_warm, info_warm = warm_prob.solve(x0=centre.copy(), **kwargs)
        assert info_warm["status_msg"] == "Solve_Succeeded", \
            f"warm solve failed at step {k}: {info_warm['status_msg']}"
        warm_iters.append(info_warm["iter_count"])
        last_ws = info_warm["working_set"]

        dx = float(np.linalg.norm(x_warm - x_cold))
        print(f"{k:>4} {cold_iters[-1]:>10} {warm_iters[-1]:>10}  {dx:>18.3e}")

    print("-" * 48)
    print(f"mean iter count: cold = {np.mean(cold_iters):.2f}, "
          f"warm = {np.mean(warm_iters):.2f}  "
          f"(speedup = {np.mean(cold_iters) / max(np.mean(warm_iters), 1e-9):.2f}x)")
    print("convex QP -> 1 outer iter either way; the working-set payoff is "
          "internal to the QP solve.")


# ===========================================================================
# Part 2 -- interior-point warm start across a receding horizon (a real win)
# ===========================================================================
class Tracking:
    """min ||x - r||^2 s.t. ||x||^2 = R^2 (nonlinear), -5 <= x_i <= 5."""

    def __init__(self, n):
        self.n = n
        self.r = np.zeros(n)

    def set_setpoint(self, r):
        self.r = np.asarray(r, dtype=np.float64)

    def objective(self, x):
        d = x - self.r
        return float(d @ d)

    def gradient(self, x):
        return 2.0 * (x - self.r)

    def constraints(self, x):
        return np.array([float(x @ x)])

    def jacobianstructure(self):
        return (np.zeros(self.n, dtype=np.int64),
                np.arange(self.n, dtype=np.int64))

    def jacobian(self, x):
        return 2.0 * x


def make_track_problem(prob_obj, n, r2, options=None):
    p = pounce.Problem(
        n=n, m=1, problem_obj=prob_obj,
        lb=[-5.0] * n, ub=[5.0] * n,
        cl=[r2], cu=[r2],
    )
    p.add_option("tol", 1e-8)
    p.add_option("print_level", 0)
    for k, v in (options or {}).items():
        p.add_option(k, v)
    return p


# Tuned interior-point warm start: small mu_init so the barrier starts near
# convergence, and tight bound pushes so the warm iterate stays put.
TUNED_WARM = {
    "warm_start_init_point": "yes",
    "mu_init": 1e-6,
    "warm_start_bound_push": 1e-8,
    "warm_start_bound_frac": 1e-8,
    "warm_start_slack_bound_push": 1e-8,
    "warm_start_slack_bound_frac": 1e-8,
    "warm_start_mult_bound_push": 1e-8,
}


def receding_horizon_demo():
    n = 4
    n_steps = 16
    r2 = 4.0
    naive_x0 = np.full(n, 1.0)  # fixed, uninformative cold start every step

    cold_iters = []
    warm_iters = []
    prev_x = None
    prev_info = None

    print(f"{'step':>4} {'cold iters':>10} {'warm iters':>10}")
    print("-" * 28)

    for k in range(n_steps):
        theta = 2.0 * np.pi * k / n_steps
        r_k = 1.5 * np.array([np.cos(theta + j) for j in range(n)])

        # Cold: from the fixed naive start, no warm-start data.
        cold_obj = Tracking(n)
        cold_obj.set_setpoint(r_k)
        _, info_cold = make_track_problem(cold_obj, n, r2).solve(x0=naive_x0.copy())
        cold_iters.append(info_cold["iter_count"])

        # Warm: from the previous step's solution + tuned IPM warm start.
        warm_obj = Tracking(n)
        warm_obj.set_setpoint(r_k)
        if prev_x is None:
            x_warm, info_warm = make_track_problem(
                warm_obj, n, r2).solve(x0=naive_x0.copy())
        else:
            x_warm, info_warm = make_track_problem(
                warm_obj, n, r2, TUNED_WARM).solve(
                x0=prev_x,
                lagrange=np.asarray(prev_info["mult_g"]),
                zl=np.asarray(prev_info["mult_x_L"]),
                zu=np.asarray(prev_info["mult_x_U"]),
            )
        warm_iters.append(info_warm["iter_count"])
        prev_x, prev_info = x_warm, info_warm

        print(f"{k:>4} {cold_iters[-1]:>10} {warm_iters[-1]:>10}")

    print("-" * 28)
    cold_total, warm_total = sum(cold_iters), sum(warm_iters)
    print(f"total iters: cold = {cold_total}, warm = {warm_total}  "
          f"(speedup = {cold_total / max(warm_total, 1):.2f}x)")
    print("nonlinear sweep -> carrying the previous solution forward "
          "visibly cuts the per-step iteration count.")


def main():
    print("== Part 1: active-set SQP working-set carry-over (convex QP) ==")
    working_set_demo()
    print()
    print("== Part 2: interior-point warm start, receding horizon (nonlinear) ==")
    receding_horizon_demo()


if __name__ == "__main__":
    main()

"""HS071 solved cold, then warm-started from the cold solution.

`warm_start_init_point=yes` forwards the primal point and the dual
seeds (`lagrange`, `zl`, `zu`) into the iterate initializer -- but on
its own it does NOT cut iterations. Two defaults cancel the warm start:

* `mu_init` (0.1) keeps the barrier parameter large, so the solver still
  walks mu down its full schedule even when started at x*.
* `warm_start_bound_push` / `_frac` (1e-2) shove the initial point off
  its bounds; HS071's x1 sits exactly on its lower bound, so the warm
  point is discarded.

To actually save iterations, pair the dual seeds with a small `mu_init`
and tight `warm_start_*_bound_push`/`_frac`. On HS071 that takes the
re-solve from 11 iterations down to 5.
"""

import numpy as np

import pounce


class HS071:
    def objective(self, x):
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    def gradient(self, x):
        return np.array([
            x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]),
            x[0] * x[3],
            x[0] * x[3] + 1.0,
            x[0] * (x[0] + x[1] + x[2]),
        ])

    def constraints(self, x):
        return np.array([np.prod(x), np.dot(x, x)])

    def jacobianstructure(self):
        return (np.repeat([0, 1], 4), np.tile([0, 1, 2, 3], 2))

    def jacobian(self, x):
        return np.array([
            x[1] * x[2] * x[3], x[0] * x[2] * x[3],
            x[0] * x[1] * x[3], x[0] * x[1] * x[2],
            2 * x[0], 2 * x[1], 2 * x[2], 2 * x[3],
        ])


def make_problem():
    p = pounce.Problem(
        n=4, m=2, problem_obj=HS071(),
        lb=[1.0] * 4, ub=[5.0] * 4,
        cl=[25.0, 40.0], cu=[2e19, 40.0],
    )
    p.add_option("tol", 1e-8)
    p.add_option("print_level", 0)
    return p


def tuned_warm_options():
    """Small mu_init + tight bound pushes so the warm point is honored."""
    return {
        "warm_start_init_point": "yes",
        "mu_init": 1e-7,
        "warm_start_bound_push": 1e-9,
        "warm_start_bound_frac": 1e-9,
        "warm_start_slack_bound_push": 1e-9,
        "warm_start_slack_bound_frac": 1e-9,
        "warm_start_mult_bound_push": 1e-9,
    }


def main():
    cold_x, cold_info = make_problem().solve(x0=np.array([1.0, 5.0, 5.0, 1.0]))
    print(f"cold:                   status={cold_info['status_msg']}, "
          f"iters={cold_info['iter_count']}")

    seeds = dict(
        lagrange=np.asarray(cold_info["mult_g"]),
        zl=np.asarray(cold_info["mult_x_L"]),
        zu=np.asarray(cold_info["mult_x_U"]),
    )

    # warm_start_init_point=yes ALONE -- no gain: the default mu_init and
    # warm_start_*_bound_push cancel the warm point.
    naive = make_problem()
    naive.add_option("warm_start_init_point", "yes")
    _, naive_info = naive.solve(x0=cold_x, **seeds)
    print(f"warm (init_point only): status={naive_info['status_msg']}, "
          f"iters={naive_info['iter_count']}   <- no gain")

    # The real warm start: small mu_init + tight bound pushes.
    warm = make_problem()
    for k, v in tuned_warm_options().items():
        warm.add_option(k, v)
    warm_x, warm_info = warm.solve(x0=cold_x, **seeds)
    print(f"warm (tuned mu_init):   status={warm_info['status_msg']}, "
          f"iters={warm_info['iter_count']}   <- {cold_info['iter_count']} "
          f"-> {warm_info['iter_count']}")
    print(f"      |delta obj| = {abs(warm_info['obj_val'] - cold_info['obj_val']):.2e}")
    print(f"      |delta x|   = {np.max(np.abs(warm_x - cold_x)):.2e}")


if __name__ == "__main__":
    main()

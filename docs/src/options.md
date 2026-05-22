# Solver Options

POUNCE accepts options the same way upstream Ipopt does. Option names
and semantics follow Ipopt's, so an existing Ipopt options file or
`KEY=VALUE` invocation works unchanged.

## Setting options

**On the command line** — append `KEY=VALUE` pairs after the input:

```sh
pounce problem.nl tol=1e-10 max_iter=500 print_level=8
```

**From an options file** — upstream `ipopt.opt` format:

```sh
pounce problem.nl --options-file ipopt.opt
```

Command-line `KEY=VALUE` pairs override values loaded from the options
file.

## Commonly used options

| Option | Meaning |
|---|---|
| `tol` | Overall convergence tolerance on the KKT error. |
| `max_iter` | Maximum number of outer iterations. |
| `print_level` | Console verbosity, 0 (silent) – 12 (maximum debug). |
| `linear_solver` | KKT linear-solver backend. `ma57` requires the `ma57` feature build. |
| `mu_strategy` | Barrier-parameter update strategy (`monotone` / `adaptive`). |

For the full upstream option catalogue, see the
[Ipopt options reference](https://coin-or.github.io/Ipopt/OPTIONS.html);
POUNCE reuses those names.

## ℓ₁ penalty-barrier wrapper options

These tune the degenerate-NLP wrapper described in
[Running Solves](cli.md). All are default-tuned and rarely need
overriding:

| Option | Default | Meaning |
|---|---|---|
| `l1_exact_penalty_barrier` | `no` | Run the ℓ₁-exact penalty-barrier wrapper unconditionally. |
| `l1_fallback_on_restoration_failure` | `no` | Retry with the wrapper only when the standard solve fails. |
| `l1_penalty_init` | `1.0` | Initial penalty weight ρ. |
| `l1_penalty_max` | `1e6` | Maximum penalty weight before declaring infeasibility. |
| `l1_penalty_increase_factor` | `8.0` | Multiplier applied to ρ each outer iteration. |
| `l1_penalty_max_outer_iter` | `8` | Maximum penalty outer iterations. |
| `l1_slack_tol` | `1e-6` | Slack tolerance for "constraints satisfied". |
| `l1_steering_factor` | `10.0` | Steering-rule factor for ρ escalation. |

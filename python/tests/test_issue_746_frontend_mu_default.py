"""pounce#746 — an auto-selected L-BFGS Hessian must not also pick the
barrier schedule.

gh#746 taught POUNCE what IpAlgBuilder.cpp:1059 does: substitute
`adaptive` for an unset `mu_strategy` whenever the Hessian is
limited-memory. Upstream that inference is safe, because the only way an
Ipopt caller gets limited-memory is by typing it — asking for L-BFGS is
a statement about the problem, and pairing a barrier schedule with it is
a second statement the caller is presumed to accept.

The Python frontend breaks that presumption. `Problem.solve` sets
`hessian_approximation = limited-memory` on its own whenever the problem
object exposes no `hessian` (crates/pounce-py/src/problem.rs), which is
the common case — a caller who never heard of L-BFGS gets it. Letting
gh#746's rule fire there reads a barrier-schedule preference out of a
choice POUNCE made for them.

It is not a free inference. Under adaptive, `test_warm_start_schema.py`'s
transferred start stopped beating a cold solve at every horizon of
gh#622's table — 86/65, 88/67, 76/70, 79/68 where the contract is
warm < cold — because free-mode adaptive recomputes mu from the current
point's complementarity and throws away the barrier state the transfer
carried. `test_trf.py` lost a start and `test_starts_racing.py` lost its
rung-0 cut with it: six failures, all from this one substitution.

So the frontend pins the registered `monotone` default explicitly. The
tests below pin *that*, and pin that an explicit request still wins.

Note what is not being claimed: monotone is not the better schedule
here. From `X0` adaptive converges in 8 iterations against monotone's
10; it is from the displaced `X0_FAR` that it costs (17 against 14).
The claim is only that the choice should be the caller's.
"""

import os

os.environ.setdefault("RUST_LOG", "off")

import pytest

from test_warm_start_schema import X0, X0_FAR, make


def _iters(x0, **opts):
    p = make(**opts)
    _, info = p.solve(x0=x0)
    assert info["status_msg"] == "Solve_Succeeded"
    return info["iter_count"]


@pytest.mark.parametrize("x0, name", [(X0, "X0"), (X0_FAR, "X0_FAR")])
def test_the_frontend_default_is_monotone_not_adaptive(x0, name):
    """The default trajectory is monotone's, step for step.

    Asserted as an equality against explicit monotone *and* an
    inequality against explicit adaptive. The equality alone would pass
    if the two schedules happened to agree on this fixture; the
    inequality is what proves the fixture can tell them apart at all.
    """
    default = _iters(x0)
    monotone = _iters(x0, mu_strategy="monotone")
    adaptive = _iters(x0, mu_strategy="adaptive")

    assert adaptive != monotone, (
        f"{name}: this fixture no longer separates the two schedules "
        f"(both {monotone}) — it cannot pin the default any more"
    )
    assert default == monotone, (
        f"{name}: default took {default} iterations, explicit monotone "
        f"{monotone}, explicit adaptive {adaptive} — the frontend's "
        f"auto-L-BFGS path has picked up gh#746's substitution again"
    )


def test_an_explicit_strategy_still_wins_on_the_auto_lbfgs_path():
    """The pin is a default, not an override.

    Someone who wants upstream's limited-memory/adaptive pairing asks
    for it and gets it — the frontend sets `mu_strategy` before the
    caller's own option list is applied, not after.
    """
    assert _iters(X0, mu_strategy="adaptive") != _iters(X0)

"""gh #862: the PSD pre-check must not mask the ``P``-shape guard.

The shape guard added for gh #113 lives in ``_validate``, which runs inside
``_build`` — *after* the PSD pre-check on every entry point that checks the
Hessian before it builds the problem. The pre-check hands ``_lower_triangle_coo``
the caller's ``n`` (from ``c``) while emitting index pairs from ``P``'s own
shape, so a ``P`` with more rows than ``n`` wrote out of bounds inside
``_min_eig_lower_coo`` and surfaced as a raw ``IndexError`` from numpy::

    IndexError: index 5 is out of bounds for axis 0 with size 5

The same malformed input with ``check_psd=False`` — which skips the pre-check
and lets ``_validate`` run — produced the precise, actionable message. Two
spellings of one bad input disagreed, and the default was the worse one.

Only ``P`` with **more rows than n** hit it: ``(3, 3)``, ``(4, 4)``, ``(5, 7)``
and ``(7, 5)`` all reached ``_validate`` intact, which is what made the
inconsistency visible rather than uniform. Those siblings are asserted here too,
so a future reordering cannot fix the reported shape while regressing them.

The fix validates the shape inside ``_psd_verdict`` before it reads ``P``, so
the single message is what every entry point raises, under either spelling.
"""

import numpy as np
import pytest

from pounce import (
    QpFactorization,
    QpSensitivity,
    solve_qp,
    solve_qp_batch,
    solve_qp_multi_rhs,
    solve_socp,
)

N = 5
# 7x7 while `c` fixes n = 5: more rows than `n` is the shape that reached numpy.
P_TOO_BIG = np.eye(7) * 2.0
C = np.ones(N)

EXPECTED = r"`P` has shape \(7, 7\) but must be \(5, 5\)"


def _entry_points():
    """Every public call that runs the PSD pre-check before ``_build``."""
    return {
        "solve_qp": lambda **kw: solve_qp(P=P_TOO_BIG, c=C, **kw),
        "solve_qp/active-set": lambda **kw: solve_qp(
            P=P_TOO_BIG, c=C, method="active-set", **kw
        ),
        "solve_qp_batch": lambda **kw: solve_qp_batch(
            [dict(P=P_TOO_BIG, c=C)], **kw
        ),
        "solve_qp_multi_rhs": lambda **kw: solve_qp_multi_rhs(
            P=P_TOO_BIG, cs=[C, 2.0 * C], **kw
        ),
        "solve_socp": lambda **kw: solve_socp(
            P=P_TOO_BIG, c=C, cones=[("nonneg", 1)], G=np.ones((1, N)),
            h=np.array([1.0]), **kw
        ),
        "QpFactorization": lambda **kw: QpFactorization(P=P_TOO_BIG, c=C, **kw),
        "QpSensitivity": lambda **kw: QpSensitivity(P=P_TOO_BIG, c=C, **kw),
    }


@pytest.mark.parametrize("name", sorted(_entry_points()))
def test_mis_shaped_P_raises_the_written_ValueError(name):
    call = _entry_points()[name]
    # The default path: `check_psd=None` runs the pre-check.
    with pytest.raises(ValueError, match=EXPECTED):
        call()
    # An IndexError is not a ValueError, so the `raises` above already rejects
    # the regression; assert the exact type too so the reason a failure happened
    # is legible rather than "did not raise ValueError".
    try:
        call()
    except ValueError:
        pass
    except IndexError as exc:  # pragma: no cover - the regression itself
        pytest.fail(f"{name}: PSD pre-check raised a raw IndexError: {exc}")


@pytest.mark.parametrize("name", sorted(_entry_points()))
def test_the_two_spellings_agree(name):
    """``check_psd=False`` skips the pre-check; it must reach the same error."""
    call = _entry_points()[name]
    with pytest.raises(ValueError, match=EXPECTED) as skipped:
        call(check_psd=False)
    with pytest.raises(ValueError, match=EXPECTED) as checked:
        call()
    assert str(skipped.value) == str(checked.value)


@pytest.mark.parametrize("shape", [(3, 3), (4, 4), (5, 7), (7, 5), (7, 7)])
def test_every_mis_shaped_P_is_rejected_uniformly(shape):
    """The siblings that were already correct stay correct.

    Only `(7, 7)` — more rows than `n` — reached numpy; the guard now speaks
    for all of them, so a shape that is wrong in any direction gets the one
    message naming what it is and what it must be.
    """
    P = np.eye(*shape) * 2.0
    with pytest.raises(
        ValueError,
        match=rf"`P` has shape \({shape[0]}, {shape[1]}\) but must be \(5, 5\)",
    ):
        solve_qp(P=P, c=C)


def test_a_correctly_shaped_P_still_solves():
    """The check runs before the guard, so it must not reject a good `P`."""
    r = solve_qp(P=np.eye(N) * 2.0, c=C)
    assert r.status == "optimal"
    np.testing.assert_allclose(r.x, -0.5 * np.ones(N), atol=1e-6)


def test_an_indefinite_P_of_the_right_shape_is_still_refused():
    """The shape check is placed before the PSD verdict, not instead of it."""
    P = np.eye(N) * 2.0
    P[0, 0] = -3.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        solve_qp(P=P, c=C)

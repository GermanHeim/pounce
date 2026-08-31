"""gh #874: the jax and torch QP layers must reject a mis-shaped ``P`` too.

``7dc03c66`` added :func:`pounce.qp._validate_p_shape` for gh #862 and landed
it in ``python/pounce/qp.py`` alone — ``grep -rn _validate_p_shape python/``
returned three lines, all in that one file. Its commit message's claim to cover
"every entry point that checks the Hessian before it builds" was false as
written: both differentiable frontends run their own ``_guard_psd``, which
reaches ``_check_psd(*_to_coo_lower(np.asarray(P)), n)`` with the shape already
discarded.

Measured at ``629535fa``, ``P = 2·I₃`` against a length-5 ``c``::

    pounce.qp.solve_qp(...)     ValueError: `P` has shape (3, 3) but must be (5, 5)
    pounce.torch.solve_qp(...)  -> [-0.5 -0.5 -0.5 -1.  -1. ]     NO ERROR
    pounce.jax.solve_qp(...)    -> [-0.5 -0.5 -0.5 -1.  -1. ]     NO ERROR

That vector is the exact optimum of the **zero-padded** 5x5 model, so it is a
silently substituted different problem rather than noise. An *oversized* ``P``
failed the other way, with the raw ``IndexError: index 3 is out of bounds for
axis 0 with size 3`` from inside numpy — verbatim the string ``7dc03c66``'s own
commit message quotes as the thing it fixed.

# Why these two frontends are the worse place for it

Both are differentiable layers. ``_kkt_backward`` inverts the padded KKT
system, so the gradients flowing back into a training loop are the padded
model's gradients. A user gets a silently wrong descent direction with no
exception anywhere to notice — the gh #845 shape, in the one place where the
wrong answer is consumed by an optimizer rather than read by a person.

# The two branches this file covers on purpose

Per CLAUDE.md's branch rule, an undersized and an oversized ``P`` are different
branches, not two spellings of one: the lower-triangle filter (``ri >= ci``)
means only an *oversized* ``P`` can emit an index past ``n``, so undersized
reached the solver intact and silently, while oversized raised from numpy.
A fix tested on one says nothing about the other.

``check_psd=False`` is a third branch and is covered here too. Shape validation
runs *above* that early return by design: ``check_psd`` says whether the caller
wants the definiteness precondition verified, and is not permission to solve a
different model than the one passed.
"""

import numpy as np
import pytest

import pounce.qp

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")

import pounce.jax  # noqa: E402
import pounce.torch  # noqa: E402

N = 5
C = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
LB = -np.ones(N)
UB = np.ones(N)

P_UNDER = 2.0 * np.eye(3)  # (3, 3) against n = 5 -- was silently zero-padded
P_OVER = np.eye(N)  # (5, 5) against n = 3 -- was a raw numpy IndexError

FRONTENDS = {
    "qp": pounce.qp,
    "torch": pounce.torch,
    "jax": pounce.jax,
}


@pytest.mark.parametrize("name", sorted(FRONTENDS))
def test_an_undersized_p_is_rejected_not_zero_padded(name):
    """The branch that had no error at all: `[-0.5, -0.5, -0.5, -1, -1]` was
    returned as if it answered the question that was asked."""
    with pytest.raises(
        ValueError, match=r"`P` has shape \(3, 3\) but must be \(5, 5\)"
    ):
        FRONTENDS[name].solve_qp(P=P_UNDER, c=C, lb=LB, ub=UB)


@pytest.mark.parametrize("name", sorted(FRONTENDS))
def test_an_oversized_p_gives_the_shared_message_not_a_numpy_indexerror(name):
    """The other branch: it *did* raise, but named an array the caller never
    created. All three frontends must give the one actionable message."""
    with pytest.raises(
        ValueError, match=r"`P` has shape \(5, 5\) but must be \(3, 3\)"
    ):
        FRONTENDS[name].solve_qp(
            P=P_OVER, c=np.ones(3), lb=-np.ones(3), ub=np.ones(3)
        )


@pytest.mark.parametrize("name", ["jax", "torch"])
def test_check_psd_false_is_not_a_way_past_the_shape_check(name):
    """`check_psd=False` turns off the *definiteness* check. It must not also
    turn off the shape check, or the documented escape hatch for a
    PSD-by-construction `P` doubles as an escape hatch for solving a different
    model."""
    with pytest.raises(
        ValueError, match=r"`P` has shape \(3, 3\) but must be \(5, 5\)"
    ):
        FRONTENDS[name].solve_qp(P=P_UNDER, c=C, lb=LB, ub=UB, check_psd=False)


@pytest.mark.parametrize("name", sorted(FRONTENDS))
def test_a_well_formed_model_still_solves(name):
    """The negative control. Without it, "raise on every P" passes everything
    above."""
    r = FRONTENDS[name].solve_qp(P=2.0 * np.eye(N), c=C, lb=LB, ub=UB)
    x = np.asarray(getattr(r, "x", r)).ravel()
    # min x'x + c'x over [-1, 1]^5  ->  x = clip(-c/2, -1, 1)
    assert np.allclose(x, np.clip(-C / 2.0, -1.0, 1.0), atol=1e-4), x


def test_the_error_is_raised_before_jax_wraps_it():
    """jax rewraps anything a host callback raises as
    ``JaxRuntimeError: INTERNAL: CpuCallback error``, burying the message in a
    nested traceback. ``P``'s shape is static at trace time, so the check runs
    at the entry point and the user sees the same plain ``ValueError`` the
    other two frontends give."""
    with pytest.raises(ValueError) as exc:
        pounce.jax.solve_qp(P=P_UNDER, c=C, lb=LB, ub=UB)
    assert not isinstance(exc.value, jax.errors.JaxRuntimeError), exc.value
    assert "must be (5, 5)" in str(exc.value)


# --------------------------------------------------------------------------
# The shape check runs at *trace* time on the jax path, so it must read `P`'s
# shape without materializing its value. The first draft of the fix reached
# `np.asarray(P)` inside `_mat_shape`, which raises
# `TracerArrayConversionError` on a tracer — turning the guard into a
# regression that broke every `jit`/`grad` call with a *correct* `P`. Eleven
# tests in `test_qp_jax.py` caught it; these two say so on purpose, so the
# next person to touch `_mat_shape` learns it from a name rather than from a
# distant suite.
# --------------------------------------------------------------------------


def test_a_correct_p_still_traces_under_jit_and_grad():
    import jax.numpy as jnp

    P = jnp.array([[3.0, 0.5], [0.5, 2.0]])
    c = jnp.array([-4.0, -1.0])

    def loss(Pm):
        # Bounds stay concrete: the layer reads them at trace time by design.
        return jnp.sum(pounce.jax.solve_qp(P=Pm, c=c, lb=-np.ones(2), ub=np.ones(2)) ** 2)

    # Both of these put a tracer, not an array, into `_validate_p_shape`.
    assert np.isfinite(float(jax.jit(loss)(P)))
    assert np.all(np.isfinite(np.asarray(jax.grad(loss)(P))))


def test_a_wrong_shaped_p_is_still_rejected_under_jit():
    import jax.numpy as jnp

    bad = jnp.eye(3)
    c = jnp.ones(5)

    def f(Pm):
        return pounce.jax.solve_qp(P=Pm, c=c, lb=-np.ones(5), ub=np.ones(5))

    with pytest.raises(ValueError, match=r"`P` has shape \(3, 3\) but must be \(5, 5\)"):
        jax.jit(f)(bad)

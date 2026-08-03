"""Tests for in-memory `.nl` model construction (issue #469).

Three surfaces, all of which used to be Rust-only:

* ``pounce.parse_nl_text`` — the same parser ``read_nl`` uses, fed a string
  instead of a path, so a frontend that generates `.nl` never touches disk.
* ``pounce.NlExpr`` / ``pounce.build_nl_problem`` — build the expression DAG
  directly and skip `.nl` entirely. This is the only route for operators
  `.nl` cannot carry (``atan2``, ``min``/``max``, ``erf``).
* ``NlProblem.hessian_vector_product`` — matrix-free ``∇²L · v``, for models
  where materializing the Lagrangian Hessian is impractical.
"""

import math

import numpy as np
import pytest

import pounce

# A complete `.nl` body: min (x0-1)^2 + (x1-2)^2, no constraints. Same
# fixture shape the Rust reader's unit tests use, kept here as text so the
# parse-from-string path has something to chew on that never hits disk.
SIMPLE_NL = """g3 0 1 0
2 0 1 0 0
0 1
0 0
0 2 0
0 0 0 1
0 0 0 0 0
0 0
0 0
0 0 0 0 0
O0 0
o0
o5
o1
v0
n1
n2
o5
o1
v1
n2
n2
b
3
3
"""


def test_parse_nl_text_matches_read_nl_semantics(tmp_path):
    """Parsing text and parsing the same bytes off disk agree exactly."""
    p_text = pounce.parse_nl_text(SIMPLE_NL)
    nl_file = tmp_path / "simple.nl"
    nl_file.write_text(SIMPLE_NL)
    p_file = pounce.read_nl(str(nl_file))

    assert (p_text.n, p_text.m) == (p_file.n, p_file.m) == (2, 0)
    x = np.array([0.3, 1.4])
    assert p_text.objective(x) == p_file.objective(x)
    np.testing.assert_array_equal(p_text.gradient(x), p_file.gradient(x))

    # (x0-1)^2 + (x1-2)^2 at (0.3, 1.4): 0.49 + 0.36
    assert p_text.objective(x) == pytest.approx(0.85)
    np.testing.assert_allclose(p_text.gradient(x), [-1.4, -1.2])


def test_parse_nl_text_accepts_names_and_validates_length():
    """There are no sibling `.col`/`.row` files, so names come as arguments."""
    p = pounce.parse_nl_text(SIMPLE_NL, var_names=["alpha", "beta"])
    assert p.var_names == ["alpha", "beta"]
    assert p.con_names == []

    with pytest.raises(ValueError, match="var_names"):
        pounce.parse_nl_text(SIMPLE_NL, var_names=["only_one"])


def test_parse_nl_text_reports_bad_input_as_valueerror():
    with pytest.raises(ValueError, match="parse_nl_text"):
        pounce.parse_nl_text("this is not an .nl file")


# ---- NlExpr / build_nl_problem ----------------------------------------


def test_build_nl_problem_rosenbrock_matches_analytic():
    x = pounce.NlExpr.vars(2)
    rosen = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    p = pounce.build_nl_problem(n=2, objective=rosen, x0=[-1.2, 1.0])

    assert (p.n, p.m) == (2, 0)
    assert p.minimize is True

    x0, x1 = -1.2, 1.0
    assert p.objective([x0, x1]) == pytest.approx(
        (1 - x0) ** 2 + 100 * (x1 - x0**2) ** 2
    )
    np.testing.assert_allclose(
        p.gradient([x0, x1]),
        [
            -2 * (1 - x0) - 400 * x0 * (x1 - x0**2),
            200 * (x1 - x0**2),
        ],
        rtol=1e-10,
    )


def test_build_nl_problem_constraints_and_structures():
    x = pounce.NlExpr.vars(3)
    p = pounce.build_nl_problem(
        n=3,
        objective=pounce.NlExpr.sum([x[0] ** 2, x[1] ** 2, x[2] ** 2]),
        constraints=[x[0] * x[1], x[1] + x[2]],
        g_l=[1.0, 0.0],
        g_u=[1.0, 5.0],
        x_l=[-10.0] * 3,
        x_u=[10.0] * 3,
        var_names=["a", "b", "c"],
        con_names=["prod", "lin"],
    )

    assert (p.n, p.m) == (3, 2)
    assert p.var_names == ["a", "b", "c"]
    assert p.con_names == ["prod", "lin"]
    np.testing.assert_allclose(p.g_l, [1.0, 0.0])
    np.testing.assert_allclose(p.g_u, [1.0, 5.0])

    pt = [2.0, 3.0, -1.0]
    np.testing.assert_allclose(p.constraints(pt), [6.0, 2.0])

    jr, jc = p.jacobian_structure()
    jv = p.jacobian(pt)
    assert jr.shape == jc.shape == jv.shape == (p.nnz_jac,)
    dense = np.zeros((p.m, p.n))
    dense[jr, jc] = jv
    np.testing.assert_allclose(dense, [[3.0, 2.0, 0.0], [0.0, 1.0, 1.0]])

    hr, hc = p.hessian_structure()
    assert np.all(hr >= hc), "Hessian structure must be the lower triangle"
    assert p.hessian(pt).shape == (p.nnz_hess,)


def test_build_nl_problem_defaults_are_unbounded_and_zero_start():
    p = pounce.build_nl_problem(n=2, objective=pounce.NlExpr.var(0))
    np.testing.assert_allclose(p.x0, [0.0, 0.0])
    assert np.all(np.asarray(p.x_l) <= -1e19)
    assert np.all(np.asarray(p.x_u) >= 1e19)


def test_build_nl_problem_maximize_negates_objective():
    x = pounce.NlExpr.var(0)
    p = pounce.build_nl_problem(n=1, objective=x**2, minimize=False)
    assert p.minimize is False
    # The evaluator hands back the minimization form.
    assert p.objective([3.0]) == pytest.approx(-9.0)
    np.testing.assert_allclose(p.gradient([3.0]), [-6.0])


def test_build_nl_problem_rejects_out_of_range_variable():
    with pytest.raises(ValueError, match=r"Var\(4\)"):
        pounce.build_nl_problem(n=2, objective=pounce.NlExpr.var(4))

    with pytest.raises(ValueError, match="constraint 0"):
        pounce.build_nl_problem(
            n=2, objective=pounce.NlExpr.var(0), constraints=[pounce.NlExpr.var(9)]
        )


def test_build_nl_problem_rejects_mismatched_vector_lengths():
    with pytest.raises(ValueError, match="x_l"):
        pounce.build_nl_problem(n=3, objective=pounce.NlExpr.var(0), x_l=[0.0, 1.0])


def test_build_nl_problem_solves_through_minimize():
    """End to end: an in-memory model reaches the solver, not just the
    evaluators."""
    x = pounce.NlExpr.vars(2)
    p = pounce.build_nl_problem(
        n=2,
        objective=(x[0] - 1) ** 2 + (x[1] + 2) ** 2,
        x0=[5.0, 5.0],
    )
    (x, info), = pounce.solve_nlp_batch([p])
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(x, [1.0, -2.0], atol=1e-6)


# ---- Operators `.nl` cannot carry -------------------------------------


def test_expressions_nl_cannot_export():
    """``atan2``, ``min``/``max`` and ``erf`` all evaluate here.

    A `.nl` round trip loses every one of them: writers have no two-argument
    funcall path for ``atan2``, ``min``/``max`` force a DNLP model type, and
    AMPL has no ``erf`` opcode at all.
    """
    x = pounce.NlExpr.vars(2)
    p = pounce.build_nl_problem(
        n=2,
        objective=pounce.NlExpr.sum(
            [
                pounce.NlExpr.atan2(x[0], x[1]),
                pounce.NlExpr.min(x[0], x[1]),
                pounce.NlExpr.max(x[0], x[1]),
                x[0].erf(),
            ]
        ),
    )
    a, b = 0.8, 1.5
    assert p.objective([a, b]) == pytest.approx(
        math.atan2(a, b) + min(a, b) + max(a, b) + math.erf(a)
    )


@pytest.mark.parametrize("value", [0.0, 0.5, -1.3, 2.0, -4.0])
def test_erf_value_and_derivative(value):
    x = pounce.NlExpr.var(0)
    p = pounce.build_nl_problem(n=1, objective=x.erf())
    assert p.objective([value]) == pytest.approx(math.erf(value), abs=1e-14)

    # erf'(u) = 2/sqrt(pi) * exp(-u^2)
    want = 2.0 / math.sqrt(math.pi) * math.exp(-(value**2))
    np.testing.assert_allclose(p.gradient([value]), [want], rtol=1e-12)

    # erf''(u) = -2u * erf'(u)
    hv = p.hessian([value])
    if hv.size:  # the structural entry exists for every nonconstant erf
        np.testing.assert_allclose(hv, [-2.0 * value * want], rtol=1e-9, atol=1e-12)


def test_select_and_compare_route_through_active_branch():
    x = pounce.NlExpr.vars(2)
    # f = x1^2 if x0 > 0 else x1^3 — value and derivative both follow the
    # live branch; the branch test itself contributes nothing.
    expr = pounce.NlExpr.select(
        pounce.NlExpr.compare(">", x[0], 0.0), x[1] ** 2, x[1] ** 3
    )
    p = pounce.build_nl_problem(n=2, objective=expr)

    assert p.objective([1.0, 3.0]) == pytest.approx(9.0)
    np.testing.assert_allclose(p.gradient([1.0, 3.0]), [0.0, 6.0])

    assert p.objective([-1.0, 3.0]) == pytest.approx(27.0)
    np.testing.assert_allclose(p.gradient([-1.0, 3.0]), [0.0, 27.0])


def test_compare_rejects_unknown_operator():
    with pytest.raises(ValueError, match="unknown operator"):
        pounce.NlExpr.compare("=<", pounce.NlExpr.var(0), 1.0)


# ---- NlExpr operator surface ------------------------------------------


def test_operator_coercion_both_directions():
    x = pounce.NlExpr.var(0)
    cases = {
        "add": (x + 3, 5.0),
        "radd": (3 + x, 5.0),
        "sub": (x - 3, -1.0),
        "rsub": (3 - x, 1.0),
        "mul": (x * 3, 6.0),
        "rmul": (3 * x, 6.0),
        "div": (x / 4, 0.5),
        "rdiv": (8 / x, 4.0),
        "pow": (x**3, 8.0),
        "rpow": (3**x, 9.0),
        "neg": (-x, -2.0),
        "pos": (+x, 2.0),
        "abs": (abs(-x), 2.0),
    }
    for name, (expr, want) in cases.items():
        assert expr.eval([2.0]) == pytest.approx(want), name


def test_unary_math_methods_match_stdlib():
    x = pounce.NlExpr.var(0)
    at = 0.4
    for name, ref in [
        ("sqrt", math.sqrt),
        ("exp", math.exp),
        ("log", math.log),
        ("log10", math.log10),
        ("sin", math.sin),
        ("cos", math.cos),
        ("tan", math.tan),
        ("asin", math.asin),
        ("acos", math.acos),
        ("atan", math.atan),
        ("sinh", math.sinh),
        ("cosh", math.cosh),
        ("tanh", math.tanh),
        ("asinh", math.asinh),
        ("atanh", math.atanh),
        ("erf", math.erf),
    ]:
        got = getattr(x, name)().eval([at])
        assert got == pytest.approx(ref(at), rel=1e-12), name
    assert x.acosh().eval([2.5]) == pytest.approx(math.acosh(2.5))


def test_expr_eval_gradient_and_variables():
    x = pounce.NlExpr.vars(3)
    e = x[0] * x[2]
    assert e.variables() == [0, 2]
    assert e.eval([2.0, 99.0, 5.0]) == pytest.approx(10.0)
    np.testing.assert_allclose(e.gradient([2.0, 99.0, 5.0]), [5.0, 0.0, 2.0])


def test_expr_eval_rejects_short_x():
    with pytest.raises(ValueError, match="variable 2"):
        pounce.NlExpr.var(2).eval([1.0, 2.0])


def test_expr_rejects_non_numeric_operand():
    x = pounce.NlExpr.var(0)
    with pytest.raises(TypeError):
        x + "1.0"
    with pytest.raises(TypeError):
        x * object()


def test_expr_repr_renders_small_expressions():
    x = pounce.NlExpr.vars(2)
    assert repr(x[0] + x[1]) == "NlExpr(x[0] + x[1])"
    # A large expression falls back to a node count rather than rendering
    # something unbounded.
    big = pounce.NlExpr.sum([x[0]] * 200)
    assert "nodes" in repr(big)


def test_min_max_require_operands():
    with pytest.raises(ValueError, match="at least one operand"):
        pounce.NlExpr.min()


# ---- Hessian-vector product -------------------------------------------


def _dense_hessian(p, x, lam=None, obj_factor=1.0):
    hr, hc = p.hessian_structure()
    hv = p.hessian(x, lam, obj_factor)
    dense = np.zeros((p.n, p.n))
    for i, j, v in zip(hr, hc, hv):
        dense[i, j] += v
        if i != j:
            dense[j, i] += v
    return dense


def test_hessian_vector_product_matches_dense():
    x = pounce.NlExpr.vars(3)
    p = pounce.build_nl_problem(
        n=3,
        objective=x[0] * x[1] * x[2] + (x[0] * x[1]).exp() + x[2].erf(),
        constraints=[x[0] ** 2 + x[2].sin(), x[1] * x[2]],
        g_l=[0.0, 0.0],
        g_u=[1.0, 1.0],
    )
    pt = np.array([0.3, -0.7, 1.1])
    lam = np.array([0.5, -1.25])
    obj_factor = 2.0

    dense = _dense_hessian(p, pt, lam, obj_factor)
    for v in (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.4, -1.3, 2.0]),
    ):
        got = p.hessian_vector_product(pt, v, lam, obj_factor)
        np.testing.assert_allclose(got, dense @ v, rtol=1e-9, atol=1e-12)


def test_hessian_vector_product_defaults_to_objective_block():
    x = pounce.NlExpr.vars(2)
    # f = x0^2 + 3 x0 x1  ->  H = [[2, 3], [3, 0]]
    p = pounce.build_nl_problem(
        n=2, objective=x[0] ** 2 + 3 * x[0] * x[1], constraints=[x[0] * x[1]]
    )
    got = p.hessian_vector_product([0.5, 2.0], [1.0, 1.0])
    np.testing.assert_allclose(got, [5.0, 3.0], atol=1e-12)


def test_hessian_vector_product_validates_lengths():
    p = pounce.build_nl_problem(n=2, objective=pounce.NlExpr.var(0) ** 2)
    with pytest.raises(ValueError, match="v"):
        p.hessian_vector_product([1.0, 1.0], [1.0])
    with pytest.raises(ValueError, match="x"):
        p.hessian_vector_product([1.0], [1.0, 1.0])


def test_hessian_vector_product_on_read_nl_model(tmp_path):
    """The HVP is a method on every ``NlProblem``, however it was built."""
    nl_file = tmp_path / "simple.nl"
    nl_file.write_text(SIMPLE_NL)
    p = pounce.read_nl(str(nl_file))
    # min (x0-1)^2 + (x1-2)^2  ->  H = 2I
    got = p.hessian_vector_product([0.0, 0.0], [1.5, -2.5])
    np.testing.assert_allclose(got, [3.0, -5.0], atol=1e-12)

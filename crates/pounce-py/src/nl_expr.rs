//! In-memory model construction for Python (issue #469).
//!
//! [`PyNlExpr`] is a thin handle on [`pounce_nl::nl_reader::Expr`], the same
//! expression DAG the `.nl` parser produces, with Python operators wired to
//! its nodes. [`build_nl_problem`] assembles a set of those expressions into
//! an [`crate::PyNlProblem`] — so a modeling frontend can go straight from
//! its own DAG to pounce's AD tape, with no `.nl` file on disk and no
//! parser in the middle:
//!
//! ```python
//! import pounce
//! x = pounce.NlExpr.vars(2)
//! rosen = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
//! p = pounce.build_nl_problem(
//!     n=2,
//!     objective=rosen,
//!     constraints=[x[0] ** 2 + x[1] ** 2],
//!     g_l=[0.0], g_u=[1.0],
//! )
//! p.gradient([0.5, 0.5])          # same evaluator `read_nl` returns
//! ```
//!
//! Going through `.nl` is not merely slower, it is lossy: `.nl` writers
//! commonly refuse `atan2` (no two-argument funcall path) and `min`/`max`
//! (they force a DNLP model type), and AMPL has no `erf` opcode at all —
//! yet the tape supports all four. Built here, they survive.
//!
//! [`PyNlExpr::eval`] and [`PyNlExpr::gradient`] expose
//! `pounce_nl::nl_tape::Tape::build` on a single scalar expression, which is
//! mostly useful for checking a subexpression in isolation before wiring it
//! into a model.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyString;

use pounce_common::types::Number;
use pounce_nl::nl_reader::{
    BinOp, CmpOp, Expr, NlProblem, NlProblemParts, NlTnlp, UnaryOp, render_expression,
};
use pounce_nl::nl_tape::Tape;

use crate::nl_problem::PyNlProblem;

/// The `.nl` reader's "unbounded" sentinel. Bounds at or beyond this
/// magnitude are treated as absent by the solver, so it is the right
/// default for an omitted bound vector.
const INF: Number = 1e19;

/// A node in an expression DAG, and the building block of
/// [`build_nl_problem`].
///
/// Cheap to clone: the payload is the same `Expr` tree the `.nl` parser
/// builds, and Python-level sharing (`t = x[0] * x[1]` used twice) deep-
/// copies the subtree rather than aliasing it. That keeps the semantics
/// obvious — every occurrence is an independent occurrence in the chain
/// rule, which is exactly how the tape treats a shared `Cse` body anyway —
/// at the cost of a larger tape for heavily-reused subexpressions.
#[pyclass(module = "pounce", name = "NlExpr")]
#[derive(Clone)]
pub struct PyNlExpr {
    pub(crate) inner: Expr,
}

impl PyNlExpr {
    fn wrap(inner: Expr) -> PyNlExpr {
        PyNlExpr { inner }
    }
}

/// Accept an `NlExpr` or any Python float/int as an expression operand, so
/// `2 * x[0]` and `x[0] ** 2` read the way a modeler expects.
fn coerce(v: &Bound<'_, PyAny>, what: &str) -> PyResult<Expr> {
    if let Ok(e) = v.extract::<PyRef<'_, PyNlExpr>>() {
        return Ok(e.inner.clone());
    }
    // Reject strings explicitly: Python would happily `float("nan")` some
    // of them via `extract`, and silently turning "1e-3" into a constant
    // hides a typo rather than reporting it.
    if v.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(format!(
            "{what}: expected NlExpr or a number, got str"
        )));
    }
    match v.extract::<Number>() {
        Ok(c) => Ok(Expr::Const(c)),
        Err(_) => Err(PyTypeError::new_err(format!(
            "{what}: expected NlExpr or a number, got {}",
            v.get_type().name()?
        ))),
    }
}

fn binary(op: BinOp, a: Expr, b: Expr) -> PyNlExpr {
    PyNlExpr::wrap(Expr::Binary(op, Box::new(a), Box::new(b)))
}

fn unary(op: UnaryOp, a: Expr) -> PyNlExpr {
    PyNlExpr::wrap(Expr::Unary(op, Box::new(a)))
}

/// Collect a Python iterable of operands into expressions.
fn coerce_all(items: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<Expr>> {
    let mut out = Vec::new();
    for item in items.iter()? {
        out.push(coerce(&item?, what)?);
    }
    Ok(out)
}

/// Number of distinct DAG nodes in `e`, stopping as soon as the count
/// exceeds `cap`. Used to keep `__repr__` from rendering a model-sized
/// expression (the renderer inlines shared bodies, so its output can be
/// exponentially larger than the DAG).
fn node_count_capped(e: &Expr, cap: usize, acc: &mut usize) {
    if *acc > cap {
        return;
    }
    *acc += 1;
    match e {
        Expr::Const(_) | Expr::Var(_) => {}
        Expr::Binary(_, a, b) | Expr::Compare(_, a, b) | Expr::And(a, b) | Expr::Or(a, b) => {
            node_count_capped(a, cap, acc);
            node_count_capped(b, cap, acc);
        }
        Expr::Unary(_, a) | Expr::Not(a) => node_count_capped(a, cap, acc),
        Expr::Cse(body) => node_count_capped(body, cap, acc),
        Expr::Sum(args) | Expr::MinList(args) | Expr::MaxList(args) => {
            for a in args {
                node_count_capped(a, cap, acc);
            }
        }
        Expr::Cond { cond, then_, else_ } => {
            node_count_capped(cond, cap, acc);
            node_count_capped(then_, cap, acc);
            node_count_capped(else_, cap, acc);
        }
        Expr::Funcall { args, .. } => {
            for a in args {
                if let pounce_nl::nl_reader::FuncallArg::Real(inner) = a {
                    node_count_capped(inner, cap, acc);
                }
            }
        }
    }
}

/// Map a Python comparison spelling onto a [`CmpOp`].
fn parse_cmp(op: &str) -> PyResult<CmpOp> {
    Ok(match op {
        "<" | "lt" => CmpOp::Lt,
        "<=" | "le" => CmpOp::Le,
        "==" | "eq" => CmpOp::Eq,
        ">=" | "ge" => CmpOp::Ge,
        ">" | "gt" => CmpOp::Gt,
        "!=" | "ne" => CmpOp::Ne,
        other => {
            return Err(PyValueError::new_err(format!(
                "compare: unknown operator {other:?}; expected one of \
                 '<', '<=', '==', '>=', '>', '!='"
            )));
        }
    })
}

#[pymethods]
impl PyNlExpr {
    /// Reference to problem variable `index` (0-based).
    #[staticmethod]
    fn var(index: usize) -> PyNlExpr {
        PyNlExpr::wrap(Expr::Var(index))
    }

    /// `[var(0), var(1), ..., var(n-1)]` — the usual first line of a model.
    #[staticmethod]
    fn vars(n: usize) -> Vec<PyNlExpr> {
        (0..n).map(PyNlExpr::var).collect()
    }

    /// A numeric literal. Rarely needed explicitly: plain Python numbers
    /// are accepted anywhere an `NlExpr` is.
    #[staticmethod]
    fn const_(value: Number) -> PyNlExpr {
        PyNlExpr::wrap(Expr::Const(value))
    }

    /// `sum(args)` as a single n-ary node — cheaper to build and to tape
    /// than a left-leaning chain of `+`.
    #[staticmethod]
    fn sum(args: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        Ok(PyNlExpr::wrap(Expr::Sum(coerce_all(args, "sum")?)))
    }

    /// `atan2(y, x)`, the two-argument arctangent. Has no `.nl` writer
    /// path in most frontends, which is one of the reasons this module
    /// exists.
    #[staticmethod]
    fn atan2(y: &Bound<'_, PyAny>, x: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        Ok(binary(
            BinOp::Atan2,
            coerce(y, "atan2: y")?,
            coerce(x, "atan2: x")?,
        ))
    }

    /// n-ary minimum. Piecewise linear: the derivative follows whichever
    /// operand is currently smallest (ties pick the first) and the second
    /// derivative is identically zero — the standard AD treatment.
    #[staticmethod]
    #[pyo3(signature = (*args))]
    fn min(args: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        let items = coerce_all(args, "min")?;
        if items.is_empty() {
            return Err(PyValueError::new_err("min: needs at least one operand"));
        }
        Ok(PyNlExpr::wrap(Expr::MinList(items)))
    }

    /// n-ary maximum; mirrors [`PyNlExpr::min`].
    #[staticmethod]
    #[pyo3(signature = (*args))]
    fn max(args: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        let items = coerce_all(args, "max")?;
        if items.is_empty() {
            return Err(PyValueError::new_err("max: needs at least one operand"));
        }
        Ok(PyNlExpr::wrap(Expr::MaxList(items)))
    }

    /// Relational test `a <op> b`, evaluating to `1.0` or `0.0`. `op` is
    /// one of `'<' '<=' '==' '>=' '>' '!='`.
    ///
    /// Spelled as a function rather than as Python's `<` operator because
    /// overloading comparison would break every ordinary use of an
    /// `NlExpr` in a container. The result is piecewise constant, hence
    /// zero-derivative; pair it with [`PyNlExpr::select`].
    #[staticmethod]
    fn compare(op: &str, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        Ok(PyNlExpr::wrap(Expr::Compare(
            parse_cmp(op)?,
            Box::new(coerce(a, "compare: a")?),
            Box::new(coerce(b, "compare: b")?),
        )))
    }

    /// `then_ if cond else else_`. The value and all derivatives flow
    /// through the active branch only; the branch switch itself is a
    /// non-smooth event the AD ignores, exactly as ASL/Ipopt treat `if`.
    #[staticmethod]
    fn select(
        cond: &Bound<'_, PyAny>,
        then_: &Bound<'_, PyAny>,
        else_: &Bound<'_, PyAny>,
    ) -> PyResult<PyNlExpr> {
        Ok(PyNlExpr::wrap(Expr::Cond {
            cond: Box::new(coerce(cond, "select: cond")?),
            then_: Box::new(coerce(then_, "select: then_")?),
            else_: Box::new(coerce(else_, "select: else_")?),
        }))
    }

    /// Logical AND: `1.0` iff both operands are nonzero. Zero derivative.
    #[staticmethod]
    fn logical_and(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        Ok(PyNlExpr::wrap(Expr::And(
            Box::new(coerce(a, "logical_and: a")?),
            Box::new(coerce(b, "logical_and: b")?),
        )))
    }

    /// Logical OR: `1.0` iff either operand is nonzero. Zero derivative.
    #[staticmethod]
    fn logical_or(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        Ok(PyNlExpr::wrap(Expr::Or(
            Box::new(coerce(a, "logical_or: a")?),
            Box::new(coerce(b, "logical_or: b")?),
        )))
    }

    /// Logical NOT: `1.0` iff the operand is zero. Zero derivative.
    #[staticmethod]
    fn logical_not(a: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        Ok(PyNlExpr::wrap(Expr::Not(Box::new(coerce(
            a,
            "logical_not: a",
        )?))))
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        Ok(binary(BinOp::Add, self.inner.clone(), coerce(other, "+")?))
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        Ok(binary(BinOp::Add, coerce(other, "+")?, self.inner.clone()))
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        Ok(binary(BinOp::Sub, self.inner.clone(), coerce(other, "-")?))
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        Ok(binary(BinOp::Sub, coerce(other, "-")?, self.inner.clone()))
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        Ok(binary(BinOp::Mul, self.inner.clone(), coerce(other, "*")?))
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        Ok(binary(BinOp::Mul, coerce(other, "*")?, self.inner.clone()))
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        Ok(binary(BinOp::Div, self.inner.clone(), coerce(other, "/")?))
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        Ok(binary(BinOp::Div, coerce(other, "/")?, self.inner.clone()))
    }

    fn __pow__(
        &self,
        other: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyNlExpr> {
        if modulo.is_some_and(|m| !m.is_none()) {
            return Err(PyValueError::new_err(
                "NlExpr ** exp % mod: three-argument pow is not supported",
            ));
        }
        Ok(binary(BinOp::Pow, self.inner.clone(), coerce(other, "**")?))
    }

    fn __rpow__(
        &self,
        other: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyNlExpr> {
        if modulo.is_some_and(|m| !m.is_none()) {
            return Err(PyValueError::new_err(
                "base ** NlExpr % mod: three-argument pow is not supported",
            ));
        }
        Ok(binary(BinOp::Pow, coerce(other, "**")?, self.inner.clone()))
    }

    fn __neg__(&self) -> PyNlExpr {
        unary(UnaryOp::Neg, self.inner.clone())
    }

    fn __pos__(&self) -> PyNlExpr {
        self.clone()
    }

    fn __abs__(&self) -> PyNlExpr {
        unary(UnaryOp::Abs, self.inner.clone())
    }

    fn sqrt(&self) -> PyNlExpr {
        unary(UnaryOp::Sqrt, self.inner.clone())
    }

    fn exp(&self) -> PyNlExpr {
        unary(UnaryOp::Exp, self.inner.clone())
    }

    /// Natural logarithm.
    fn log(&self) -> PyNlExpr {
        unary(UnaryOp::Log, self.inner.clone())
    }

    fn log10(&self) -> PyNlExpr {
        unary(UnaryOp::Log10, self.inner.clone())
    }

    fn sin(&self) -> PyNlExpr {
        unary(UnaryOp::Sin, self.inner.clone())
    }

    fn cos(&self) -> PyNlExpr {
        unary(UnaryOp::Cos, self.inner.clone())
    }

    fn tan(&self) -> PyNlExpr {
        unary(UnaryOp::Tan, self.inner.clone())
    }

    fn asin(&self) -> PyNlExpr {
        unary(UnaryOp::Asin, self.inner.clone())
    }

    fn acos(&self) -> PyNlExpr {
        unary(UnaryOp::Acos, self.inner.clone())
    }

    fn atan(&self) -> PyNlExpr {
        unary(UnaryOp::Atan, self.inner.clone())
    }

    fn sinh(&self) -> PyNlExpr {
        unary(UnaryOp::Sinh, self.inner.clone())
    }

    fn cosh(&self) -> PyNlExpr {
        unary(UnaryOp::Cosh, self.inner.clone())
    }

    fn tanh(&self) -> PyNlExpr {
        unary(UnaryOp::Tanh, self.inner.clone())
    }

    fn asinh(&self) -> PyNlExpr {
        unary(UnaryOp::Asinh, self.inner.clone())
    }

    fn acosh(&self) -> PyNlExpr {
        unary(UnaryOp::Acosh, self.inner.clone())
    }

    fn atanh(&self) -> PyNlExpr {
        unary(UnaryOp::Atanh, self.inner.clone())
    }

    /// Gauss error function. Reachable only from here — AMPL has no `erf`
    /// opcode, so no `.nl` round trip can carry it (issue #469).
    fn erf(&self) -> PyNlExpr {
        unary(UnaryOp::Erf, self.inner.clone())
    }

    /// Value of this expression alone at `x`, through the same AD tape a
    /// model uses. `x` must be long enough to cover every variable index
    /// the expression references.
    fn eval(&self, x: Vec<Number>) -> PyResult<Number> {
        let tape = self.checked_tape(x.len(), "eval")?;
        Ok(tape.eval(&x))
    }

    /// Gradient of this expression alone at `x`, length `len(x)`.
    fn gradient<'py>(
        &self,
        py: Python<'py>,
        x: Vec<Number>,
    ) -> PyResult<Bound<'py, PyArray1<Number>>> {
        let tape = self.checked_tape(x.len(), "gradient")?;
        let mut grad = vec![0.0; x.len()];
        tape.gradient_seed(&x, 1.0, &mut grad);
        Ok(grad.into_pyarray_bound(py))
    }

    /// Sorted variable indices this expression references.
    fn variables(&self) -> Vec<usize> {
        Tape::build(&self.inner).variables()
    }

    fn __repr__(&self) -> String {
        let mut count = 0usize;
        const CAP: usize = 64;
        node_count_capped(&self.inner, CAP, &mut count);
        if count > CAP {
            format!("NlExpr(<{CAP}+ nodes>)")
        } else {
            format!("NlExpr({})", render_expression(&self.inner, &[]))
        }
    }
}

impl PyNlExpr {
    /// Tape for this expression, rejecting a variable index `x` cannot
    /// supply. Without the check the tape's forward sweep would index
    /// `x` out of bounds — a panic across the pyo3 boundary rather than a
    /// catchable Python error.
    fn checked_tape(&self, x_len: usize, what: &str) -> PyResult<Tape> {
        let tape = Tape::build(&self.inner);
        if let Some(&max) = tape.variables().iter().max() {
            if max >= x_len {
                return Err(PyValueError::new_err(format!(
                    "{what}: expression references variable {max} but x has length {x_len}"
                )));
            }
        }
        Ok(tape)
    }
}

/// Decode an optional float vector, filling `default` when it is `None`.
fn opt_vec(
    v: Option<&Bound<'_, PyAny>>,
    len: usize,
    default: Number,
    what: &str,
) -> PyResult<Vec<Number>> {
    match v {
        None => Ok(vec![default; len]),
        Some(b) => {
            let mut out = Vec::with_capacity(len);
            for item in b.iter()? {
                out.push(item?.extract::<Number>()?);
            }
            if out.len() != len {
                return Err(PyValueError::new_err(format!(
                    "build_nl_problem: {what} has length {}, expected {len}",
                    out.len()
                )));
            }
            Ok(out)
        }
    }
}

/// Build an evaluable [`crate::PyNlProblem`] from expressions, with no `.nl`
/// file involved (issue #469).
///
/// The returned object is the same `NlProblem` class `read_nl` hands back
/// and supports the same surface — `objective`, `gradient`, `constraints`,
/// `jacobian` / `jacobian_structure`, `hessian` / `hessian_structure`,
/// `hessian_vector_product`, and `variant`.
///
/// Bound vectors default to unbounded (`±1e19`, the `.nl` sentinel) and
/// `x0` to zeros. `var_names` / `con_names` are optional; when given they
/// must match `n` / `len(constraints)`.
#[pyfunction]
#[pyo3(signature = (
    n,
    objective,
    constraints=None,
    x_l=None,
    x_u=None,
    x0=None,
    g_l=None,
    g_u=None,
    minimize=true,
    obj_constant=0.0,
    var_names=None,
    con_names=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn build_nl_problem(
    n: usize,
    objective: &Bound<'_, PyAny>,
    constraints: Option<&Bound<'_, PyAny>>,
    x_l: Option<&Bound<'_, PyAny>>,
    x_u: Option<&Bound<'_, PyAny>>,
    x0: Option<&Bound<'_, PyAny>>,
    g_l: Option<&Bound<'_, PyAny>>,
    g_u: Option<&Bound<'_, PyAny>>,
    minimize: bool,
    obj_constant: Number,
    var_names: Option<Vec<String>>,
    con_names: Option<Vec<String>>,
) -> PyResult<PyNlProblem> {
    let objective = coerce(objective, "build_nl_problem: objective")?;
    let constraints = match constraints {
        None => Vec::new(),
        Some(c) => coerce_all(c, "build_nl_problem: constraints")?,
    };
    let m = constraints.len();

    let parts = NlProblemParts {
        minimize,
        objective,
        obj_constant,
        constraints,
        x_l: opt_vec(x_l, n, -INF, "x_l")?,
        x_u: opt_vec(x_u, n, INF, "x_u")?,
        x0: opt_vec(x0, n, 0.0, "x0")?,
        g_l: opt_vec(g_l, m, -INF, "g_l")?,
        g_u: opt_vec(g_u, m, INF, "g_u")?,
        var_names: var_names.unwrap_or_default(),
        con_names: con_names.unwrap_or_default(),
    };

    let prob = NlProblem::from_expressions(parts)
        .map_err(|e| PyValueError::new_err(format!("build_nl_problem: {e}")))?;
    let tnlp = NlTnlp::try_new(prob)
        .map_err(|e| PyValueError::new_err(format!("build_nl_problem: {e}")))?;
    PyNlProblem::from_tnlp(tnlp, "build_nl_problem")
}

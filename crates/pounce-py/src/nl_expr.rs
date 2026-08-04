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

/// Deepest expression this module will build.
///
/// Everything that consumes an `Expr` — `Tape::build`, the derived
/// `Clone`, and the derived `Drop` — recurses once per level, so a
/// sufficiently deep tree overflows the stack. That is a hard crash, not a
/// Python exception: the interpreter dies with SIGSEGV and no traceback.
/// Measured on Linux with the default 8 MB stack, `Tape::build` survives
/// ~32k levels in a release build but only ~4k in a debug build, and a
/// Python thread with a smaller stack is worse still. Capping at
/// construction is what makes the guarantee hold for all three consumers
/// at once: an expression that cannot be built cannot later be cloned or
/// dropped into a crash.
///
/// 1000 is far below the smallest observed failure and far above anything
/// reachable in practice, because building depth `D` through the operators
/// costs O(D²) anyway (see [`PyNlExpr`]) — a chain long enough to approach
/// the real stack limit takes tens of seconds to build. Wide models are
/// unaffected: [`PyNlExpr::sum`] is one level regardless of term count.
const MAX_DEPTH: u32 = 1000;

/// A node in an expression DAG, and the building block of
/// [`build_nl_problem`].
///
/// **Operands are deep-copied, so `+` chains are quadratic.** Each
/// operator clones its operand subtrees rather than aliasing them, which
/// keeps the semantics obvious — every occurrence is an independent
/// occurrence in the chain rule — but means accumulating in a Python loop
/// is O(N²) in both time and memory:
///
/// ```python
/// e = pounce.NlExpr.const_(0.0)
/// for t in terms:          # O(N^2): each += copies everything built so far
///     e = e + t
///
/// e = pounce.NlExpr.sum(terms)     # O(N), one n-ary node
/// ```
///
/// The difference is not marginal — 200 000 terms through `sum` is
/// instant, while a `+` chain is measured in tens of seconds by 20 000 and
/// hits [`MAX_DEPTH`] long before that. Reach for `sum` whenever the term
/// count is data-driven.
#[pyclass(module = "pounce", name = "NlExpr")]
#[derive(Clone)]
pub struct PyNlExpr {
    pub(crate) inner: Expr,
    /// Nesting depth of `inner`, maintained incrementally (O(1) per
    /// operation) so the [`MAX_DEPTH`] check never has to walk the tree.
    depth: u32,
}

/// An operand decoded from Python: the expression plus its depth, so the
/// consumer can compute its own depth without a walk.
struct Operand {
    expr: Expr,
    depth: u32,
}

impl PyNlExpr {
    /// Wrap an expression whose depth is already known to be in range.
    /// Only for leaves (`Var` / `Const`), where depth is 1 by definition.
    fn leaf(inner: Expr) -> PyNlExpr {
        PyNlExpr { inner, depth: 1 }
    }

    /// Wrap a node built over operands of depth `child_depth`, rejecting
    /// it if that puts the result past [`MAX_DEPTH`].
    fn nested(inner: Expr, child_depth: u32) -> PyResult<PyNlExpr> {
        let depth = child_depth.saturating_add(1);
        if depth > MAX_DEPTH {
            return Err(PyValueError::new_err(format!(
                "NlExpr: expression nesting would reach depth {depth}, past the \
                 limit of {MAX_DEPTH}. Deeper trees overflow the stack when the \
                 expression is taped, copied, or freed — a hard crash rather than \
                 an exception — so they are refused here. If you are accumulating \
                 terms in a loop (`e = e + t`), use NlExpr.sum([...]) instead: it \
                 builds one n-ary node of depth 1 and is O(N) rather than O(N^2)."
            )));
        }
        Ok(PyNlExpr { inner, depth })
    }
}

/// Accept an `NlExpr` or any Python float/int as an expression operand, so
/// `2 * x[0]` and `x[0] ** 2` read the way a modeler expects.
fn coerce(v: &Bound<'_, PyAny>, what: &str) -> PyResult<Operand> {
    if let Ok(e) = v.extract::<PyRef<'_, PyNlExpr>>() {
        return Ok(Operand {
            expr: e.inner.clone(),
            depth: e.depth,
        });
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
        Ok(c) => Ok(Operand {
            expr: Expr::Const(c),
            depth: 1,
        }),
        Err(_) => Err(PyTypeError::new_err(format!(
            "{what}: expected NlExpr or a number, got {}",
            v.get_type().name()?
        ))),
    }
}

fn binary(op: BinOp, a: Operand, b: Operand) -> PyResult<PyNlExpr> {
    PyNlExpr::nested(
        Expr::Binary(op, Box::new(a.expr), Box::new(b.expr)),
        a.depth.max(b.depth),
    )
}

fn unary(op: UnaryOp, a: Operand) -> PyResult<PyNlExpr> {
    PyNlExpr::nested(Expr::Unary(op, Box::new(a.expr)), a.depth)
}

/// Collect a Python iterable of operands, returning them with the deepest
/// operand's depth.
fn coerce_all(items: &Bound<'_, PyAny>, what: &str) -> PyResult<(Vec<Expr>, u32)> {
    let mut out = Vec::new();
    let mut depth = 0;
    for item in items.iter()? {
        let o = coerce(&item?, what)?;
        depth = depth.max(o.depth);
        out.push(o.expr);
    }
    Ok((out, depth))
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
        PyNlExpr::leaf(Expr::Var(index))
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
        PyNlExpr::leaf(Expr::Const(value))
    }

    /// `sum(args)` as a single n-ary node — cheaper to build and to tape
    /// than a left-leaning chain of `+`.
    #[staticmethod]
    fn sum(args: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        let (items, depth) = coerce_all(args, "sum")?;
        PyNlExpr::nested(Expr::Sum(items), depth)
    }

    /// `atan2(y, x)`, the two-argument arctangent. Has no `.nl` writer
    /// path in most frontends, which is one of the reasons this module
    /// exists.
    #[staticmethod]
    fn atan2(y: &Bound<'_, PyAny>, x: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        binary(BinOp::Atan2, coerce(y, "atan2: y")?, coerce(x, "atan2: x")?)
    }

    /// n-ary minimum. Piecewise linear: the derivative follows whichever
    /// operand is currently smallest (ties pick the first) and the second
    /// derivative is identically zero — the standard AD treatment.
    #[staticmethod]
    #[pyo3(signature = (*args))]
    fn min(args: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        let (items, depth) = coerce_all(args, "min")?;
        if items.is_empty() {
            return Err(PyValueError::new_err("min: needs at least one operand"));
        }
        PyNlExpr::nested(Expr::MinList(items), depth)
    }

    /// n-ary maximum; mirrors [`PyNlExpr::min`].
    #[staticmethod]
    #[pyo3(signature = (*args))]
    fn max(args: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        let (items, depth) = coerce_all(args, "max")?;
        if items.is_empty() {
            return Err(PyValueError::new_err("max: needs at least one operand"));
        }
        PyNlExpr::nested(Expr::MaxList(items), depth)
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
        let op = parse_cmp(op)?;
        let (a, b) = (coerce(a, "compare: a")?, coerce(b, "compare: b")?);
        let depth = a.depth.max(b.depth);
        PyNlExpr::nested(Expr::Compare(op, Box::new(a.expr), Box::new(b.expr)), depth)
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
        let cond = coerce(cond, "select: cond")?;
        let then_ = coerce(then_, "select: then_")?;
        let else_ = coerce(else_, "select: else_")?;
        let depth = cond.depth.max(then_.depth).max(else_.depth);
        PyNlExpr::nested(
            Expr::Cond {
                cond: Box::new(cond.expr),
                then_: Box::new(then_.expr),
                else_: Box::new(else_.expr),
            },
            depth,
        )
    }

    /// Logical AND: `1.0` iff both operands are nonzero. Zero derivative.
    #[staticmethod]
    fn logical_and(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        let (a, b) = (coerce(a, "logical_and: a")?, coerce(b, "logical_and: b")?);
        let depth = a.depth.max(b.depth);
        PyNlExpr::nested(Expr::And(Box::new(a.expr), Box::new(b.expr)), depth)
    }

    /// Logical OR: `1.0` iff either operand is nonzero. Zero derivative.
    #[staticmethod]
    fn logical_or(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        let (a, b) = (coerce(a, "logical_or: a")?, coerce(b, "logical_or: b")?);
        let depth = a.depth.max(b.depth);
        PyNlExpr::nested(Expr::Or(Box::new(a.expr), Box::new(b.expr)), depth)
    }

    /// Logical NOT: `1.0` iff the operand is zero. Zero derivative.
    #[staticmethod]
    fn logical_not(a: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        let a = coerce(a, "logical_not: a")?;
        let depth = a.depth;
        PyNlExpr::nested(Expr::Not(Box::new(a.expr)), depth)
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        binary(
            BinOp::Add,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
            coerce(other, "+")?,
        )
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        binary(
            BinOp::Add,
            coerce(other, "+")?,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
        )
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        binary(
            BinOp::Sub,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
            coerce(other, "-")?,
        )
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        binary(
            BinOp::Sub,
            coerce(other, "-")?,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
        )
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        binary(
            BinOp::Mul,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
            coerce(other, "*")?,
        )
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        binary(
            BinOp::Mul,
            coerce(other, "*")?,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
        )
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        binary(
            BinOp::Div,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
            coerce(other, "/")?,
        )
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyNlExpr> {
        binary(
            BinOp::Div,
            coerce(other, "/")?,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
        )
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
        binary(
            BinOp::Pow,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
            coerce(other, "**")?,
        )
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
        binary(
            BinOp::Pow,
            coerce(other, "**")?,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
        )
    }

    fn __neg__(&self) -> PyResult<PyNlExpr> {
        unary(
            UnaryOp::Neg,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
        )
    }

    fn __pos__(&self) -> PyNlExpr {
        self.clone()
    }

    /// Copy support. `Expr` is a plain tree, so a copy is a deep copy
    /// either way — `__copy__` and `__deepcopy__` are the same operation,
    /// and both are what a frontend cloning its own DAG reaches for.
    /// (Pickling is *not* supported: an `NlExpr` has no serialized form,
    /// and pickling one raises `TypeError`. Rebuild from the model source
    /// instead.)
    fn __copy__(&self) -> PyNlExpr {
        self.clone()
    }

    #[pyo3(signature = (_memo=None))]
    fn __deepcopy__(&self, _memo: Option<&Bound<'_, PyAny>>) -> PyNlExpr {
        self.clone()
    }

    /// Nesting depth of this expression. Bounded by `NlExpr.max_depth`;
    /// exposed because it is the number in the error you get when a `+`
    /// chain runs away.
    #[getter]
    fn depth(&self) -> u32 {
        self.depth
    }

    /// The deepest expression this class will build. See the class
    /// docstring for why deep trees are refused rather than allowed to
    /// crash the interpreter.
    #[classattr]
    fn max_depth() -> u32 {
        MAX_DEPTH
    }

    /// Opt out of NumPy's ufunc protocol.
    ///
    /// Without this, `np.array([1.0, 2.0]) * x[0]` silently produces a
    /// `dtype=object` ndarray of `NlExpr` — NumPy broadcasts elementwise
    /// and calls `float.__mul__` per cell — while the forward form
    /// `x[0] * np.array(...)` correctly raises. Setting it to `None`
    /// makes the reflected form raise too, so the two directions agree
    /// and a vectorized expression has to be spelled explicitly
    /// (`NlExpr.sum(c * x[i] for ...)`).
    #[classattr]
    #[allow(non_snake_case)]
    fn __array_ufunc__() -> Option<()> {
        None
    }

    fn __abs__(&self) -> PyResult<PyNlExpr> {
        unary(
            UnaryOp::Abs,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
        )
    }

    fn sqrt(&self) -> PyResult<PyNlExpr> {
        unary(
            UnaryOp::Sqrt,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
        )
    }

    fn exp(&self) -> PyResult<PyNlExpr> {
        unary(
            UnaryOp::Exp,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
        )
    }

    /// Natural logarithm.
    fn log(&self) -> PyResult<PyNlExpr> {
        unary(
            UnaryOp::Log,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
        )
    }

    fn log10(&self) -> PyResult<PyNlExpr> {
        unary(
            UnaryOp::Log10,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
        )
    }

    fn sin(&self) -> PyResult<PyNlExpr> {
        unary(
            UnaryOp::Sin,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
        )
    }

    fn cos(&self) -> PyResult<PyNlExpr> {
        unary(
            UnaryOp::Cos,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
        )
    }

    fn tan(&self) -> PyResult<PyNlExpr> {
        unary(
            UnaryOp::Tan,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
        )
    }

    fn asin(&self) -> PyResult<PyNlExpr> {
        unary(
            UnaryOp::Asin,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
        )
    }

    fn acos(&self) -> PyResult<PyNlExpr> {
        unary(
            UnaryOp::Acos,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
        )
    }

    fn atan(&self) -> PyResult<PyNlExpr> {
        unary(
            UnaryOp::Atan,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
        )
    }

    fn sinh(&self) -> PyResult<PyNlExpr> {
        unary(
            UnaryOp::Sinh,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
        )
    }

    fn cosh(&self) -> PyResult<PyNlExpr> {
        unary(
            UnaryOp::Cosh,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
        )
    }

    fn tanh(&self) -> PyResult<PyNlExpr> {
        unary(
            UnaryOp::Tanh,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
        )
    }

    fn asinh(&self) -> PyResult<PyNlExpr> {
        unary(
            UnaryOp::Asinh,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
        )
    }

    fn acosh(&self) -> PyResult<PyNlExpr> {
        unary(
            UnaryOp::Acosh,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
        )
    }

    fn atanh(&self) -> PyResult<PyNlExpr> {
        unary(
            UnaryOp::Atanh,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
        )
    }

    /// Gauss error function. Reachable only from here — AMPL has no `erf`
    /// opcode, so no `.nl` round trip can carry it (issue #469).
    fn erf(&self) -> PyResult<PyNlExpr> {
        unary(
            UnaryOp::Erf,
            Operand {
                expr: self.inner.clone(),
                depth: self.depth,
            },
        )
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
        match tape.variables().iter().max() {
            Some(&max) if max >= x_len => Err(PyValueError::new_err(format!(
                "{what}: expression references variable {max} but x has length {x_len}"
            ))),
            _ => Ok(tape),
        }
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
    let objective = coerce(objective, "build_nl_problem: objective")?.expr;
    let constraints = match constraints {
        None => Vec::new(),
        Some(c) => coerce_all(c, "build_nl_problem: constraints")?.0,
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

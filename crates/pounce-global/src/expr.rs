//! An ergonomic algebraic-expression builder that compiles to the
//! [`FbbtTape`] the rest of POUNCE already understands (interval forward /
//! reverse, and — here — McCormick relaxation).
//!
//! Build expressions with [`var`]/[`con`] and the usual operators:
//!
//! ```
//! use pounce_global::expr::{var, con};
//! // (x0 − 1)² + (x1 − 2)²
//! let e = (var(0) - con(1.0)).powi(2) + (var(1) - con(2.0)).powi(2);
//! let tape = e.to_tape();
//! assert_eq!(pounce_global::expr::eval(&tape, &[1.0, 2.0]), 0.0);
//! ```

use pounce_nlp::{FbbtOp, FbbtTape};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A symbolic expression tree over the problem variables. Compiles to an
/// [`FbbtTape`] via [`Expr::to_tape`].
#[derive(Clone, Debug)]
pub enum Expr {
    Const(f64),
    Var(usize),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    PowI(Box<Expr>, u32),
    Neg(Box<Expr>),
    Sqrt(Box<Expr>),
    Exp(Box<Expr>),
    Ln(Box<Expr>),
    Abs(Box<Expr>),
    Sin(Box<Expr>),
    Cos(Box<Expr>),
}

/// Reference to problem variable `i` (0-based).
pub fn var(i: usize) -> Expr {
    Expr::Var(i)
}

/// A numeric constant.
pub fn con(v: f64) -> Expr {
    Expr::Const(v)
}

impl Expr {
    /// Integer power `self^n`.
    pub fn powi(self, n: u32) -> Expr {
        Expr::PowI(Box::new(self), n)
    }
    pub fn sqrt(self) -> Expr {
        Expr::Sqrt(Box::new(self))
    }
    pub fn exp(self) -> Expr {
        Expr::Exp(Box::new(self))
    }
    pub fn ln(self) -> Expr {
        Expr::Ln(Box::new(self))
    }
    pub fn abs(self) -> Expr {
        Expr::Abs(Box::new(self))
    }
    pub fn sin(self) -> Expr {
        Expr::Sin(Box::new(self))
    }
    pub fn cos(self) -> Expr {
        Expr::Cos(Box::new(self))
    }

    /// Compile to an [`FbbtTape`] (no common-subexpression sharing — each node
    /// emits its own slot; the relaxation and interval passes are linear-time
    /// regardless).
    pub fn to_tape(&self) -> FbbtTape {
        let mut ops = Vec::new();
        emit(self, &mut ops);
        FbbtTape { ops }
    }
}

fn emit(e: &Expr, ops: &mut Vec<FbbtOp>) -> usize {
    let op = match e {
        Expr::Const(v) => FbbtOp::Const(*v),
        Expr::Var(i) => FbbtOp::Var(*i),
        Expr::Add(a, b) => {
            let (x, y) = (emit(a, ops), emit(b, ops));
            FbbtOp::Add(x, y)
        }
        Expr::Sub(a, b) => {
            let (x, y) = (emit(a, ops), emit(b, ops));
            FbbtOp::Sub(x, y)
        }
        Expr::Mul(a, b) => {
            let (x, y) = (emit(a, ops), emit(b, ops));
            FbbtOp::Mul(x, y)
        }
        Expr::Div(a, b) => {
            let (x, y) = (emit(a, ops), emit(b, ops));
            FbbtOp::Div(x, y)
        }
        Expr::PowI(a, n) => {
            let x = emit(a, ops);
            FbbtOp::PowInt(x, *n)
        }
        Expr::Neg(a) => FbbtOp::Neg(emit(a, ops)),
        Expr::Sqrt(a) => FbbtOp::Sqrt(emit(a, ops)),
        Expr::Exp(a) => FbbtOp::Exp(emit(a, ops)),
        Expr::Ln(a) => FbbtOp::Ln(emit(a, ops)),
        Expr::Abs(a) => FbbtOp::Abs(emit(a, ops)),
        Expr::Sin(a) => FbbtOp::Sin(emit(a, ops)),
        Expr::Cos(a) => FbbtOp::Cos(emit(a, ops)),
    };
    ops.push(op);
    ops.len() - 1
}

/// Evaluate an [`FbbtTape`] at the point `x` (a single forward `f64` sweep).
/// The tape result is the value at the last slot. `Opaque` slots evaluate to
/// `NaN` (the builder above never emits them).
pub fn eval(tape: &FbbtTape, x: &[f64]) -> f64 {
    let mut v: Vec<f64> = Vec::with_capacity(tape.ops.len());
    for op in &tape.ops {
        let r = match *op {
            FbbtOp::Const(c) => c,
            FbbtOp::Var(i) => x[i],
            FbbtOp::Add(a, b) => v[a] + v[b],
            FbbtOp::Sub(a, b) => v[a] - v[b],
            FbbtOp::Mul(a, b) => v[a] * v[b],
            FbbtOp::Div(a, b) => v[a] / v[b],
            FbbtOp::PowInt(a, n) => v[a].powi(n as i32),
            FbbtOp::Neg(a) => -v[a],
            FbbtOp::Sqrt(a) => v[a].sqrt(),
            FbbtOp::Exp(a) => v[a].exp(),
            FbbtOp::Ln(a) => v[a].ln(),
            FbbtOp::Abs(a) => v[a].abs(),
            FbbtOp::Sin(a) => v[a].sin(),
            FbbtOp::Cos(a) => v[a].cos(),
            FbbtOp::Opaque => f64::NAN,
        };
        v.push(r);
    }
    v.last().copied().unwrap_or(0.0)
}

// --- operator overloads (owned and by-reference) -------------------------

macro_rules! bin_op {
    ($Trait:ident, $method:ident, $variant:ident) => {
        impl $Trait for Expr {
            type Output = Expr;
            fn $method(self, rhs: Expr) -> Expr {
                Expr::$variant(Box::new(self), Box::new(rhs))
            }
        }
        impl $Trait<f64> for Expr {
            type Output = Expr;
            fn $method(self, rhs: f64) -> Expr {
                Expr::$variant(Box::new(self), Box::new(Expr::Const(rhs)))
            }
        }
        impl $Trait<Expr> for f64 {
            type Output = Expr;
            fn $method(self, rhs: Expr) -> Expr {
                Expr::$variant(Box::new(Expr::Const(self)), Box::new(rhs))
            }
        }
    };
}

bin_op!(Add, add, Add);
bin_op!(Sub, sub, Sub);
bin_op!(Mul, mul, Mul);
bin_op!(Div, div, Div);

impl Neg for Expr {
    type Output = Expr;
    fn neg(self) -> Expr {
        Expr::Neg(Box::new(self))
    }
}

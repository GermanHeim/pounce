//! NLP traits consumed by the algorithm core — port of `IpNLP.hpp` /
//! `IpIpoptNLP.hpp`.
//!
//! These traits live in `pounce-nlp` (rather than `pounce-algorithm`)
//! so that the concrete [`crate::orig_ipopt_nlp::OrigIpoptNlp`], which
//! wraps a `TNLPAdapter` from this same crate, can implement them
//! without forcing `pounce-nlp` to depend on `pounce-algorithm` (the
//! reverse dependency already exists). `pounce-algorithm` re-exports
//! both traits from its own `ipopt_nlp` module so the rest of the
//! algorithm-side code continues to use the canonical
//! `crate::ipopt_nlp::IpoptNlp` path.

use pounce_common::types::{Index, Number};
use pounce_linalg::{Matrix, SymMatrix, Vector};
use std::rc::Rc;

/// Lower-level NLP interface (post-`TNLPAdapter`). Equality and
/// inequality constraints are already separated; bounds are already
/// classified into `x_l_map` / `x_u_map` / etc.
///
/// This is the equivalent of upstream `Ipopt::NLP`.
pub trait Nlp {
    fn n(&self) -> Index;
    fn m_eq(&self) -> Index;
    fn m_ineq(&self) -> Index;

    fn eval_f(&mut self, x: &dyn Vector) -> Number;
    fn eval_grad_f(&mut self, x: &dyn Vector, g: &mut dyn Vector);
    fn eval_c(&mut self, x: &dyn Vector, c: &mut dyn Vector);
    fn eval_d(&mut self, x: &dyn Vector, d: &mut dyn Vector);
    fn eval_jac_c(&mut self, x: &dyn Vector) -> Rc<dyn Matrix>;
    fn eval_jac_d(&mut self, x: &dyn Vector) -> Rc<dyn Matrix>;
    fn eval_h(
        &mut self,
        x: &dyn Vector,
        obj_factor: Number,
        y_c: &dyn Vector,
        y_d: &dyn Vector,
    ) -> Rc<dyn SymMatrix>;
}

/// Algorithm-side NLP (adds scaling-aware variants and provides the
/// bound expansion matrices `Px_L`, `Px_U`, `Pd_L`, `Pd_U`). Mirrors
/// upstream `Ipopt::IpoptNLP`.
pub trait IpoptNlp: Nlp {
    fn x_l(&self) -> &dyn Vector;
    fn x_u(&self) -> &dyn Vector;
    fn d_l(&self) -> &dyn Vector;
    fn d_u(&self) -> &dyn Vector;

    /// Bound expansion matrices: `Px_L` extracts the
    /// `x` components that have a finite lower bound, etc.
    fn px_l(&self) -> Rc<dyn Matrix>;
    fn px_u(&self) -> Rc<dyn Matrix>;
    fn pd_l(&self) -> Rc<dyn Matrix>;
    fn pd_u(&self) -> Rc<dyn Matrix>;

    /// Fill `x` with the initial primal values (mirrors upstream
    /// `IpoptNLP::GetStartingPoint`'s `init_x` flag). Default impl
    /// leaves `x` at its current contents (typically the zero vector
    /// produced by `make_new`).
    fn get_starting_x(&mut self, _x: &mut dyn Vector) -> bool {
        true
    }

    /// Fill `y_c` / `y_d` with initial multiplier guesses (mirrors
    /// `IpoptNLP::GetStartingPoint`'s `init_lambda` flag). Default
    /// impl leaves them at their current contents (zeros).
    fn get_starting_y(&mut self, _y_c: &mut dyn Vector, _y_d: &mut dyn Vector) -> bool {
        true
    }

    /// Fill `z_l` / `z_u` / `v_l` / `v_u` with initial bound-multiplier
    /// guesses (mirrors `init_z`). Default impl leaves them at zeros.
    #[allow(clippy::too_many_arguments)]
    fn get_starting_z(
        &mut self,
        _z_l: &mut dyn Vector,
        _z_u: &mut dyn Vector,
        _v_l: &mut dyn Vector,
        _v_u: &mut dyn Vector,
    ) -> bool {
        true
    }
}

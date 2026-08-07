//! Shared `(IpoptData, IpoptCq)` fixture for the μ-update unit tests.
//!
//! Both μ strategies need a live [`IpoptCalculatedQuantities`] to read
//! `curr_barrier_error`, `curr_avrg_compl`, `curr_nlp_error` and friends,
//! so the tests that exercise `update_barrier_parameter` end-to-end (as
//! opposed to the scalar helpers) need a real NLP behind the handles.
//! The NLP below is the same 2-variable / 1-equality / 1-inequality mock
//! the `ipopt_cq` tests use, trimmed to what the μ path touches.

use crate::ipopt_cq::{IpoptCalculatedQuantities, IpoptCqHandle};
use crate::ipopt_data::{IpoptData, IpoptDataHandle};
use crate::ipopt_nlp::{IpoptNlp, Nlp};
use crate::iterates_vector::IteratesVector;
use pounce_common::types::{Index, Number};
use pounce_linalg::dense_vector::{DenseVector, DenseVectorSpace};
use pounce_linalg::expansion_matrix::{ExpansionMatrix, ExpansionMatrixSpace};
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace};
use pounce_linalg::{Matrix, SymMatrix, Vector};
use std::cell::RefCell;
use std::rc::Rc;

fn dvec(values: &[Number]) -> DenseVector {
    let space = DenseVectorSpace::new(values.len() as Index);
    let mut v = space.make_new_dense();
    v.values_mut().copy_from_slice(values);
    v
}

fn rcv(values: &[Number]) -> Rc<dyn Vector> {
    Rc::new(dvec(values))
}

/// 2 vars, 1 equality, 1 inequality. Bounds: `x[0] ≥ 0`, `x[1] ≤ 5`,
/// `d ≥ 1`. `f(x) = x0² + x1²`, `c(x) = x0 + x1 - 1`, `d(x) = x0`.
struct MuMockNlp {
    x_l: DenseVector,
    x_u: DenseVector,
    d_l: DenseVector,
    d_u: DenseVector,
    px_l: Rc<dyn Matrix>,
    px_u: Rc<dyn Matrix>,
    pd_l: Rc<dyn Matrix>,
    pd_u: Rc<dyn Matrix>,
}

impl MuMockNlp {
    fn new() -> Self {
        Self {
            x_l: dvec(&[0.0]),
            x_u: dvec(&[5.0]),
            d_l: dvec(&[1.0]),
            d_u: dvec(&[]),
            px_l: Rc::new(ExpansionMatrix::new(ExpansionMatrixSpace::new(
                2,
                1,
                &[0],
                0,
            ))),
            px_u: Rc::new(ExpansionMatrix::new(ExpansionMatrixSpace::new(
                2,
                1,
                &[1],
                0,
            ))),
            pd_l: Rc::new(ExpansionMatrix::new(ExpansionMatrixSpace::new(
                1,
                1,
                &[0],
                0,
            ))),
            pd_u: Rc::new(ExpansionMatrix::new(ExpansionMatrixSpace::new(
                1,
                0,
                &[],
                0,
            ))),
        }
    }
}

impl Nlp for MuMockNlp {
    fn n(&self) -> Index {
        2
    }
    fn m_eq(&self) -> Index {
        1
    }
    fn m_ineq(&self) -> Index {
        1
    }
    fn eval_f(&mut self, x: &dyn Vector) -> Number {
        let xx = x.as_any().downcast_ref::<DenseVector>().unwrap();
        xx.values()[0] * xx.values()[0] + xx.values()[1] * xx.values()[1]
    }
    fn eval_grad_f(&mut self, x: &dyn Vector, g: &mut dyn Vector) {
        let xx = x.as_any().downcast_ref::<DenseVector>().unwrap();
        let gg = g.as_any_mut().downcast_mut::<DenseVector>().unwrap();
        gg.values_mut()[0] = 2.0 * xx.values()[0];
        gg.values_mut()[1] = 2.0 * xx.values()[1];
    }
    fn eval_c(&mut self, x: &dyn Vector, c: &mut dyn Vector) {
        let xx = x.as_any().downcast_ref::<DenseVector>().unwrap();
        let cc = c.as_any_mut().downcast_mut::<DenseVector>().unwrap();
        cc.values_mut()[0] = xx.values()[0] + xx.values()[1] - 1.0;
    }
    fn eval_d(&mut self, x: &dyn Vector, d: &mut dyn Vector) {
        let xx = x.as_any().downcast_ref::<DenseVector>().unwrap();
        let dd = d.as_any_mut().downcast_mut::<DenseVector>().unwrap();
        dd.values_mut()[0] = xx.values()[0];
    }
    fn eval_jac_c(&mut self, _x: &dyn Vector) -> Rc<dyn Matrix> {
        let space = GenTMatrixSpace::new(1, 2, vec![1, 1], vec![1, 2]);
        let mut jac = GenTMatrix::new(space);
        jac.set_values(&[1.0, 1.0]);
        Rc::new(jac)
    }
    fn eval_jac_d(&mut self, _x: &dyn Vector) -> Rc<dyn Matrix> {
        let space = GenTMatrixSpace::new(1, 2, vec![1], vec![1]);
        let mut jac = GenTMatrix::new(space);
        jac.set_values(&[1.0]);
        Rc::new(jac)
    }
    fn eval_h(
        &mut self,
        _x: &dyn Vector,
        _obj_factor: Number,
        _y_c: &dyn Vector,
        _y_d: &dyn Vector,
    ) -> Rc<dyn SymMatrix> {
        unimplemented!("the μ path never asks for the Hessian")
    }
}

impl IpoptNlp for MuMockNlp {
    fn x_l(&self) -> &dyn Vector {
        &self.x_l
    }
    fn x_u(&self) -> &dyn Vector {
        &self.x_u
    }
    fn d_l(&self) -> &dyn Vector {
        &self.d_l
    }
    fn d_u(&self) -> &dyn Vector {
        &self.d_u
    }
    fn px_l(&self) -> Rc<dyn Matrix> {
        self.px_l.clone()
    }
    fn px_u(&self) -> Rc<dyn Matrix> {
        self.px_u.clone()
    }
    fn pd_l(&self) -> Rc<dyn Matrix> {
        self.pd_l.clone()
    }
    fn pd_u(&self) -> Rc<dyn Matrix> {
        self.pd_u.clone()
    }
}

/// Handles seeded at `x = (2, 3)`, `s = (4)`, all bound multipliers at
/// `0.5`, with the given μ. The iterate is deliberately far from
/// optimal, so `curr_barrier_error()` is O(1) — tests that need the
/// "barrier subproblem solved" branch loosen `barrier_tol_factor`
/// rather than hand-tuning an iterate onto the central path.
pub(crate) fn fixture(mu: Number) -> (IpoptDataHandle, IpoptCqHandle) {
    let (x, s, z) = ([2.0, 3.0], 4.0, 0.5);
    let mut data = IpoptData::new();
    data.curr_mu = mu;
    data.curr_tau = 0.99;
    // z_L is the multiplier for x[0] ≥ 0, z_U for x[1] ≤ 5, v_L for d ≥ 1;
    // there is no upper bound on d, hence the empty v_U block.
    data.set_curr(IteratesVector::new(
        rcv(&x),
        rcv(&[s]),
        rcv(&[1.0]),
        rcv(&[1.0]),
        rcv(&[z]),
        rcv(&[z]),
        rcv(&[z]),
        rcv(&[]),
    ));
    let data_handle: IpoptDataHandle = Rc::new(RefCell::new(data));
    let nlp: Rc<RefCell<dyn IpoptNlp>> = Rc::new(RefCell::new(MuMockNlp::new()));
    let cq = IpoptCalculatedQuantities::new(data_handle.clone(), nlp);
    (data_handle, Rc::new(RefCell::new(cq)))
}

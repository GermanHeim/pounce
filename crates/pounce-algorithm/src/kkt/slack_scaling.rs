//! Slack-based symmetric scaling — port of
//! `Algorithm/LinearSolvers/IpSlackBasedTSymScalingMethod.{hpp,cpp}`.
//!
//! Used by the inexact algorithm (and any caller setting
//! `linear_system_scaling=slack-based`). Operates on the augmented KKT
//! system whose row block layout is
//!
//! ```text
//!   [ x  (nx) | s  (ns) | y_c (nc) | y_d (nd) ]
//! ```
//!
//! Per-row factors:
//!
//! * x-rows, y_c-rows, y_d-rows → `1.0`
//! * s-rows → `min(1.0, P_d_L · slack_s_L + P_d_U · slack_s_U)` per
//!   row, i.e. damp the s-row by the active slack value when small.
//!
//! Lives in `pounce-algorithm` rather than `pounce-linsol` because the
//! method needs access to `IpoptCalculatedQuantities` (current slacks)
//! and the `IpoptNlp` bound-expansion matrices, which would otherwise
//! create a circular crate dependency.

use crate::ipopt_cq::IpoptCqHandle;
use pounce_common::types::{Index, Number};
use pounce_linalg::dense_vector::DenseVector;
use pounce_linsol::scaling::TSymScalingMethod;

pub struct SlackBasedTSymScalingMethod {
    cq: IpoptCqHandle,
    /// Mirrors the upstream hard-coded `slack_scale_max = 1.0`. Kept as
    /// a field so the scalar core stays inspectable in tests.
    pub slack_scale_max: Number,
}

impl SlackBasedTSymScalingMethod {
    pub fn new(cq: IpoptCqHandle) -> Self {
        Self {
            cq,
            slack_scale_max: 1.0,
        }
    }
}

impl TSymScalingMethod for SlackBasedTSymScalingMethod {
    fn compute_sym_t_scaling_factors(
        &mut self,
        n: Index,
        _nnz: Index,
        _airn: &[Index],
        _ajcn: &[Index],
        _a: &[Number],
        scaling_factors: &mut [Number],
    ) -> bool {
        let cq_ref = self.cq.borrow();
        let (s_template, nx, ns, nc, nd) = {
            let data = cq_ref.data().borrow();
            let curr = data
                .curr
                .as_ref()
                .unwrap_or_else(|| panic!("SlackBasedTSymScalingMethod: curr iterate missing"));
            (
                curr.s.clone(),
                curr.x.dim(),
                curr.s.dim(),
                curr.y_c.dim(),
                curr.y_d.dim(),
            )
        };
        debug_assert_eq!(n, nx + ns + nc + nd);
        debug_assert_eq!(scaling_factors.len(), n as usize);

        // x-rows: identity.
        for s in scaling_factors[..nx as usize].iter_mut() {
            *s = 1.0;
        }

        // s-rows: tmp = Pd_L * slack_s_L + Pd_U * slack_s_U; then
        // tmp = min(tmp, slack_scale_max) elementwise.
        let curr_slack_s_l = cq_ref.curr_slack_s_l();
        let curr_slack_s_u = cq_ref.curr_slack_s_u();
        let nlp = cq_ref.nlp().borrow();
        let pd_l = nlp.pd_l();
        let pd_u = nlp.pd_u();

        let mut tmp = s_template.make_new();
        pd_l.mult_vector(1.0, &*curr_slack_s_l, 0.0, &mut *tmp);
        pd_u.mult_vector(1.0, &*curr_slack_s_u, 1.0, &mut *tmp);

        let mut bound = tmp.make_new();
        bound.set(self.slack_scale_max);
        tmp.element_wise_min(&*bound);

        let dense = tmp.as_any().downcast_ref::<DenseVector>().unwrap_or_else(|| {
            panic!("SlackBasedTSymScalingMethod: slack vector is not a DenseVector")
        });
        let vals = dense.expanded_values();
        scaling_factors[nx as usize..(nx + ns) as usize].copy_from_slice(&vals);

        // y_c / y_d rows: identity.
        for s in scaling_factors[(nx + ns) as usize..].iter_mut() {
            *s = 1.0;
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipopt_cq::IpoptCalculatedQuantities;
    use crate::ipopt_data::IpoptData;
    use crate::ipopt_nlp::{IpoptNlp, Nlp};
    use crate::iterates_vector::IteratesVector;
    use pounce_common::types::{Index, Number};
    use pounce_linalg::dense_vector::{DenseVector, DenseVectorSpace};
    use pounce_linalg::special_matrix::IdentityMatrix;
    use pounce_linalg::{Matrix, SymMatrix, Vector};
    use std::cell::RefCell;
    use std::rc::Rc;

    fn dvec(values: &[Number]) -> DenseVector {
        let space = DenseVectorSpace::new(values.len() as Index);
        let mut v = space.make_new_dense();
        if !values.is_empty() {
            v.values_mut().copy_from_slice(values);
        }
        v
    }

    fn rcv(values: &[Number]) -> Rc<dyn Vector> {
        Rc::new(dvec(values))
    }

    /// Minimal IpoptNlp stub. All slacks are lower- and upper-bounded by
    /// identity expansion matrices (so `Pd_L = Pd_U = I` of dim `ns`).
    #[derive(Debug)]
    struct StubNlp {
        n_x: Index,
        m_eq: Index,
        m_ineq: Index,
        x_l: DenseVector,
        x_u: DenseVector,
        d_l: DenseVector,
        d_u: DenseVector,
        px_l: Rc<dyn Matrix>,
        px_u: Rc<dyn Matrix>,
        pd_l: Rc<dyn Matrix>,
        pd_u: Rc<dyn Matrix>,
    }

    impl Nlp for StubNlp {
        fn n(&self) -> Index {
            self.n_x
        }
        fn m_eq(&self) -> Index {
            self.m_eq
        }
        fn m_ineq(&self) -> Index {
            self.m_ineq
        }
        fn eval_f(&mut self, _x: &dyn Vector) -> Number {
            0.0
        }
        fn eval_grad_f(&mut self, _x: &dyn Vector, _g: &mut dyn Vector) {}
        fn eval_c(&mut self, _x: &dyn Vector, _c: &mut dyn Vector) {}
        fn eval_d(&mut self, _x: &dyn Vector, _d: &mut dyn Vector) {}
        fn eval_jac_c(&mut self, _x: &dyn Vector) -> Rc<dyn Matrix> {
            unimplemented!()
        }
        fn eval_jac_d(&mut self, _x: &dyn Vector) -> Rc<dyn Matrix> {
            unimplemented!()
        }
        fn eval_h(
            &mut self,
            _x: &dyn Vector,
            _obj_factor: Number,
            _y_c: &dyn Vector,
            _y_d: &dyn Vector,
        ) -> Rc<dyn SymMatrix> {
            unimplemented!()
        }
    }

    impl IpoptNlp for StubNlp {
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

    fn build_cq(
        x: Rc<dyn Vector>,
        s: Rc<dyn Vector>,
        y_c: Rc<dyn Vector>,
        y_d: Rc<dyn Vector>,
        nlp: StubNlp,
    ) -> IpoptCqHandle {
        // The unused multiplier blocks (z_l, z_u, v_l, v_u) are sized
        // arbitrarily; the slack-scaling code never reads them.
        let nx = x.dim();
        let ns = s.dim();
        let z = rcv(&vec![0.0; nx as usize]);
        let v = rcv(&vec![0.0; ns as usize]);
        let iv = IteratesVector::new(x, s, y_c, y_d, z.clone(), z, v.clone(), v);
        let mut data = IpoptData::new();
        data.set_curr(iv);
        let data_handle = Rc::new(RefCell::new(data));
        let nlp_handle: Rc<RefCell<dyn IpoptNlp>> = Rc::new(RefCell::new(nlp));
        let cq = IpoptCalculatedQuantities::new(data_handle, nlp_handle);
        Rc::new(RefCell::new(cq))
    }

    #[test]
    fn slack_scaling_block_layout_with_loose_bounds_is_all_ones() {
        // nx=2, ns=2, nc=0, nd=2.
        // s = [0.5, 1.5], d_l = [0, 0], d_u = [10, 10].
        // slack_s_L = s - d_l = [0.5, 1.5];  slack_s_U = d_u - s = [9.5, 8.5]
        // Pd_L = Pd_U = I  ⇒  tmp = slack_s_L + slack_s_U = [10, 10].
        // Capped at slack_scale_max=1 ⇒ s-rows = [1.0, 1.0].
        let nlp = StubNlp {
            n_x: 2,
            m_eq: 0,
            m_ineq: 2,
            x_l: dvec(&[]),
            x_u: dvec(&[]),
            d_l: dvec(&[0.0, 0.0]),
            d_u: dvec(&[10.0, 10.0]),
            px_l: Rc::new(IdentityMatrix::new(0)),
            px_u: Rc::new(IdentityMatrix::new(0)),
            pd_l: Rc::new(IdentityMatrix::new(2)),
            pd_u: Rc::new(IdentityMatrix::new(2)),
        };
        // Note: px_l/px_u use 0×0 identity since x has no bounds in
        // this fixture; the slack-scaling path never invokes them.
        let cq = build_cq(
            rcv(&[0.0, 0.0]),
            rcv(&[0.5, 1.5]),
            rcv(&[]),
            rcv(&[0.0, 0.0]),
            nlp,
        );

        let mut method = SlackBasedTSymScalingMethod::new(cq);
        let mut sf = vec![0.0; 6];
        assert!(method.compute_sym_t_scaling_factors(6, 0, &[], &[], &[], &mut sf));
        assert_eq!(&sf[0..2], &[1.0, 1.0]);
        assert_eq!(&sf[2..4], &[1.0, 1.0]);
        assert_eq!(&sf[4..6], &[1.0, 1.0]);
    }

    #[test]
    fn slack_scaling_caps_small_slack() {
        // nx=1, ns=2, nc=0, nd=2.
        // d_l = [0, 0], d_u = [0.6, 0.6], s = [0.5, 0.5].
        // slack_s_L = [0.5, 0.5];  slack_s_U = [0.1, 0.1]
        // tmp = [0.6, 0.6]; min(1.0) = [0.6, 0.6].
        let nlp = StubNlp {
            n_x: 1,
            m_eq: 0,
            m_ineq: 2,
            x_l: dvec(&[]),
            x_u: dvec(&[]),
            d_l: dvec(&[0.0, 0.0]),
            d_u: dvec(&[0.6, 0.6]),
            px_l: Rc::new(IdentityMatrix::new(0)),
            px_u: Rc::new(IdentityMatrix::new(0)),
            pd_l: Rc::new(IdentityMatrix::new(2)),
            pd_u: Rc::new(IdentityMatrix::new(2)),
        };
        let cq = build_cq(
            rcv(&[0.0]),
            rcv(&[0.5, 0.5]),
            rcv(&[]),
            rcv(&[0.0, 0.0]),
            nlp,
        );

        let mut method = SlackBasedTSymScalingMethod::new(cq);
        let mut sf = vec![0.0; 5];
        assert!(method.compute_sym_t_scaling_factors(5, 0, &[], &[], &[], &mut sf));
        assert_eq!(sf[0], 1.0);
        assert!((sf[1] - 0.6).abs() < 1e-15);
        assert!((sf[2] - 0.6).abs() < 1e-15);
        assert_eq!(&sf[3..5], &[1.0, 1.0]);
    }

    #[test]
    fn slack_scaling_with_only_lower_bounds_uses_pd_l_only() {
        // nx=1, ns=2, nc=1, nd=2.
        // Only lower bounds present (Pd_U has 0 columns).
        // d_l = [0.1, 0.1], s = [0.4, 0.7].
        // slack_s_L = [0.3, 0.6]; tmp = [0.3, 0.6]; min(1.0) = [0.3, 0.6].
        let nlp = StubNlp {
            n_x: 1,
            m_eq: 1,
            m_ineq: 2,
            x_l: dvec(&[]),
            x_u: dvec(&[]),
            d_l: dvec(&[0.1, 0.1]),
            d_u: dvec(&[]),
            px_l: Rc::new(IdentityMatrix::new(0)),
            px_u: Rc::new(IdentityMatrix::new(0)),
            pd_l: Rc::new(IdentityMatrix::new(2)),
            // Pd_U is 2 rows × 0 cols — slack_s_U has dim 0, so the
            // upper-bound contribution is empty.
            pd_u: Rc::new(zero_matrix(2, 0)),
        };
        let cq = build_cq(
            rcv(&[0.0]),
            rcv(&[0.4, 0.7]),
            rcv(&[0.0]),
            rcv(&[0.0, 0.0]),
            nlp,
        );

        let mut method = SlackBasedTSymScalingMethod::new(cq);
        // Layout: 1 + 2 + 1 + 2 = 6.
        let mut sf = vec![0.0; 6];
        assert!(method.compute_sym_t_scaling_factors(6, 0, &[], &[], &[], &mut sf));
        assert_eq!(sf[0], 1.0);
        assert!((sf[1] - 0.3).abs() < 1e-15);
        assert!((sf[2] - 0.6).abs() < 1e-15);
        assert_eq!(&sf[3..6], &[1.0, 1.0, 1.0]);
    }

    fn zero_matrix(rows: Index, cols: Index) -> pounce_linalg::special_matrix::ZeroMatrix {
        pounce_linalg::special_matrix::ZeroMatrix::new(rows, cols)
    }
}

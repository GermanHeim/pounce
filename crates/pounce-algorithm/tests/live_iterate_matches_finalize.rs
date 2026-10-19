//! The live iterate at the last intermediate-callback fire is the solution
//! `finalize_solution` reports, on both arms, under every layer that sits
//! between the algorithm and the user's TNLP.
//!
//! Upstream `TNLP::get_curr_iterate` (Ipopt 3.14, `IpTNLP.cpp`) reports the
//! iterate in the user's space: fixed variables reinserted (`ResortX`), and
//! their bound multipliers recovered by `TNLPAdapter::ResortBoundMultipliers`
//! as `z_L − z_U = ∇f + Jᵀλ` under `make_parameter`. The interior-point arm
//! installed no materialised iterate at all, so `with_current_user_iterate`
//! read `None` there; the C API's lazy reader left those multipliers at `0`.
//!
//! Model: `min K·[(x0-3)² + (x1-3)² + (x2-1)² + (x3-4)²]`
//! s.t. `g0 = x0 + x1 + x3 ≤ 3.5`, `g1 = x0 − x2 = 0`,
//! `x0` free, `x1 ∈ [-10, 1]` (upper bound active), `x2 ≥ 0`,
//! `x3 = 1` fixed (`x_L = x_U`), so `make_parameter` removes it and its
//! stationarity row `2K(1-4) + λ0` is carried by a bound multiplier.

use pounce_algorithm::application::IpoptApplication;
use pounce_algorithm::intermediate::{with_current, with_current_user_iterate};
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, IterStats, Linearity, NlpInfo, ScalingRequest,
    Solution, SparsityRequest, StartingPoint, TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

const D: [Number; 4] = [2.0, 0.5, 4.0, 3.0];

#[derive(Clone, Debug, Default, PartialEq)]
struct Point {
    x: Vec<Number>,
    z_l: Vec<Number>,
    z_u: Vec<Number>,
    g: Vec<Number>,
    lambda: Vec<Number>,
}

#[derive(Default)]
struct Model {
    /// Objective multiplier: 1000 makes gradient-based scaling engage.
    k: Number,
    x_scaling: bool,
    fires: usize,
    last: Option<Point>,
    /// `(n, m)` of the problem the interior-point algorithm solved.
    solved_dims: Option<(usize, usize)>,
    out: Option<Point>,
    /// `scaled_iterate` of the live iterate at every fire.
    scaled: Vec<Point>,
    /// The unscaled live iterate at every fire.
    unscaled: Vec<Point>,
    /// The interior-point algorithm's own iterate at every fire, scattered
    /// to the TNLP's shape by hand (`NaN` at a column / row it does not
    /// hold), with its objective factor.
    raw: Vec<(Point, Number)>,
}

fn point(u: &pounce_nlp::live_iterate::UserIterate) -> Point {
    Point {
        x: u.x.clone(),
        z_l: u.z_l.clone(),
        z_u: u.z_u.clone(),
        g: u.g.clone(),
        lambda: u.lambda.clone(),
    }
}

fn dense(v: &dyn pounce_linalg::Vector) -> Vec<Number> {
    v.as_any()
        .downcast_ref::<pounce_linalg::DenseVector>()
        .map(|d| d.expanded_values())
        .unwrap_or_default()
}

impl TNLP for Model {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 4,
            m: 3,
            nnz_jac_g: 7,
            nnz_h_lag: 4,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-1e19, -10.0, 0.0, 1.0]);
        b.x_u.copy_from_slice(&[1e19, 1.0, 1e19, 1.0]);
        b.g_l.copy_from_slice(&[-1e19, 0.0, -1e19]);
        b.g_u.copy_from_slice(&[3.5, 0.0, 10.0]);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.0, 0.0, 0.5, 1.0]);
        true
    }
    fn get_scaling_parameters(&mut self, r: ScalingRequest<'_>) -> bool {
        *r.obj_scaling = 3.0;
        *r.use_g_scaling = true;
        r.g_scaling.copy_from_slice(&[0.01, 5.0, 1.0]);
        *r.use_x_scaling = self.x_scaling;
        if self.x_scaling {
            r.x_scaling.copy_from_slice(&D);
        }
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some(
            self.k
                * ((x[0] - 3.0).powi(2)
                    + (x[1] - 3.0).powi(2)
                    + (x[2] - 1.0).powi(2)
                    + (x[3] - 4.0).powi(2)),
        )
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * self.k * (x[0] - 3.0);
        g[1] = 2.0 * self.k * (x[1] - 3.0);
        g[2] = 2.0 * self.k * (x[2] - 1.0);
        g[3] = 2.0 * self.k * (x[3] - 4.0);
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] + x[1] + x[3];
        g[1] = x[0] - x[2];
        g[2] = x[1] + x[3];
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0, 0, 1, 1, 2, 2]);
                jcol.copy_from_slice(&[0, 1, 3, 0, 2, 1, 3]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0]);
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _n: bool,
        of: Number,
        _l: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1, 2, 3]);
                jcol.copy_from_slice(&[0, 1, 2, 3]);
            }
            SparsityRequest::Values { values } => values.copy_from_slice(&[2.0 * self.k * of; 4]),
        }
        true
    }
    fn get_constraints_linearity(&mut self, t: &mut [Linearity]) -> bool {
        t.fill(Linearity::Linear);
        true
    }
    fn get_variables_linearity(&mut self, t: &mut [Linearity]) -> bool {
        t.fill(Linearity::NonLinear);
        true
    }
    fn wants_current_iterate(&self) -> bool {
        true
    }
    fn intermediate_callback(&mut self, _st: IterStats, _d: &IpoptData, _q: &IpoptCq) -> bool {
        self.fires += 1;
        // The dimensions of the problem the algorithm solves (interior-
        // point arm only: the SQP arm installs no context).
        if let Some(dims) = with_current(|c| {
            let nlp = c.nlp.borrow();
            (nlp.n_full_x() as usize, nlp.n_full_g() as usize)
        }) {
            self.solved_dims = Some(dims);
        }
        self.last = with_current_user_iterate(point);
        if let Some((u, sc)) = with_current_user_iterate(|u| {
            (
                point(u),
                point(&pounce_nlp::live_iterate::scaled_iterate(u)),
            )
        }) {
            self.unscaled.push(u);
            self.scaled.push(sc);
        }
        if let Some(raw) = with_current(|c| {
            let curr = c.data.borrow().curr.clone()?;
            let nlp = c.nlp.borrow();
            let (n, m) = (nlp.n_full_x() as usize, nlp.n_full_g() as usize);
            let xv = dense(&*curr.x);
            let x = (0..n)
                .map(|j| match nlp.full_x_to_var_x(j as i32) {
                    Some(k) => xv[k as usize],
                    None => Number::NAN,
                })
                .collect();
            let (yc, yd) = (dense(&*curr.y_c), dense(&*curr.y_d));
            let lambda = (0..m)
                .map(|i| {
                    if let Some(k) = nlp.full_g_to_c_block(i as i32) {
                        yc[k as usize]
                    } else if let Some(k) = nlp.full_g_to_d_block(i as i32) {
                        yd[k as usize]
                    } else {
                        Number::NAN
                    }
                })
                .collect();
            Some((
                Point {
                    x,
                    z_l: nlp.pack_z_l_for_user(&*curr.z_l),
                    z_u: nlp.pack_z_u_for_user(&*curr.z_u),
                    g: Vec::new(),
                    lambda,
                },
                nlp.obj_scaling_factor(),
            ))
        })
        .flatten()
        {
            self.raw.push(raw);
        }
        true
    }
    fn finalize_solution(&mut self, s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.out = Some(Point {
            x: s.x.to_vec(),
            z_l: s.z_l.to_vec(),
            z_u: s.z_u.to_vec(),
            g: s.g.to_vec(),
            lambda: s.lambda.to_vec(),
        });
    }
}

#[derive(Clone, Copy, Debug)]
enum Scaling {
    /// `gradient-based` on a model whose gradient exceeds
    /// `nlp_scaling_max_gradient`, times `obj_scaling_factor = 3`.
    GradientBased,
    /// `user-scaling` with non-unit per-variable factors (`ScalingTnlp`).
    UserX,
    /// `presolve=yes` + `presolve_linear_eq_reduction=yes`: `g2` is
    /// implied by the box and dropped, `g1 = x0 − x2 = 0` eliminates a
    /// column, and the fixed `x3` leaves the problem.
    Presolve,
    /// `Presolve` under `user-scaling` (obj 3, x `D`, g `[0.01, 5, 1]`).
    PresolveUserX,
}

fn solve(algorithm: &str, scaling: Scaling) -> Model {
    solve_with(algorithm, scaling, None)
}

fn solve_with(algorithm: &str, scaling: Scaling, obj_factor: Option<Number>) -> Model {
    let model = match scaling {
        Scaling::GradientBased => Model {
            k: 1000.0,
            ..Default::default()
        },
        Scaling::UserX => Model {
            k: 1.0,
            x_scaling: true,
            ..Default::default()
        },
        Scaling::Presolve => Model {
            k: 1.0,
            ..Default::default()
        },
        Scaling::PresolveUserX => Model {
            k: 1.0,
            x_scaling: true,
            ..Default::default()
        },
    };
    let p = Rc::new(RefCell::new(model));
    let mut a = IpoptApplication::new();
    a.initialize().unwrap();
    let o = a.options_mut();
    o.set_integer_value("print_level", 0, true, false).unwrap();
    o.set_string_value("algorithm", algorithm, true, false)
        .unwrap();
    if let Some(f) = obj_factor {
        o.set_numeric_value("obj_scaling_factor", f, true, false)
            .unwrap();
    }
    match scaling {
        Scaling::GradientBased => {
            o.set_string_value("nlp_scaling_method", "gradient-based", true, false)
                .unwrap();
            o.set_numeric_value("obj_scaling_factor", 3.0, true, false)
                .unwrap();
        }
        Scaling::UserX => {
            o.set_string_value("nlp_scaling_method", "user-scaling", true, false)
                .unwrap();
        }
        Scaling::Presolve | Scaling::PresolveUserX => {
            o.set_string_value("presolve", "yes", true, false).unwrap();
            o.set_string_value("presolve_linear_eq_reduction", "yes", true, false)
                .unwrap();
            if let Scaling::PresolveUserX = scaling {
                o.set_string_value("nlp_scaling_method", "user-scaling", true, false)
                    .unwrap();
            }
        }
    }
    assert_eq!(
        a.optimize_tnlp(p.clone()),
        ApplicationReturnStatus::SolveSucceeded,
        "{algorithm} {scaling:?}"
    );
    if let Scaling::UserX = scaling {
        assert!(a.variable_scaling().is_some(), "x factors not applied");
    }
    Rc::try_unwrap(p).ok().unwrap().into_inner()
}

fn assert_close(what: &str, got: &[Number], want: &[Number]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: length ({got:?} vs {want:?})"
    );
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-9 * (1.0 + w.abs()),
            "{what}[{i}]: live iterate {g:e}, finalize {w:e} ({got:?} vs {want:?})"
        );
    }
}

fn check(algorithm: &str, scaling: Scaling) -> Model {
    let m = solve(algorithm, scaling);
    assert!(m.fires > 1, "{algorithm} {scaling:?}: {} fires", m.fires);
    let last = m
        .last
        .unwrap_or_else(|| panic!("{algorithm} {scaling:?}: no live iterate at the last fire"));
    let out = m.out.expect("finalize_solution");
    // Informative only if the fixed variable's multiplier is not zero:
    // `z_L − z_U = 2K(1 − 4) + λ0`, negative, so z_U carries it.
    assert!(
        out.z_u[3] > 1e-3,
        "{algorithm} {scaling:?}: fixed x3 carries no multiplier: {out:?}"
    );
    assert!(out.z_u[1] > 1e-3, "x1's upper bound inactive: {out:?}");
    assert_close("x", &last.x, &out.x);
    assert_close("z_l", &last.z_l, &out.z_l);
    assert_close("z_u", &last.z_u, &out.z_u);
    assert_close("g", &last.g, &out.g);
    assert_close("lambda", &last.lambda, &out.lambda);
    Model {
        last: Some(last),
        out: Some(out),
        ..m
    }
}

#[test]
fn ipm_live_iterate_equals_finalize_under_gradient_scaling_with_a_fixed_variable() {
    check("interior-point", Scaling::GradientBased);
}

#[test]
fn ipm_live_iterate_equals_finalize_under_user_x_scaling_with_a_fixed_variable() {
    check("interior-point", Scaling::UserX);
}

/// With presolve in the stack the algorithm solves a smaller problem, and
/// installs that problem's iterate; each presolve layer lifts it on its
/// way to the user by its own `finalize_solution`. Unlifted, the user read
/// reduced-space vectors: the wrong length, entries at the wrong indices.
#[test]
fn ipm_live_iterate_equals_finalize_with_a_dropped_row_and_an_eliminated_column() {
    let m = check("interior-point", Scaling::Presolve);
    let (n, rows) = m.solved_dims.expect("an interior-point context");
    assert!(n < 3, "no column eliminated: the algorithm solved n = {n}");
    assert!(rows < 3, "no row dropped: the algorithm solved m = {rows}");
}

#[test]
fn sqp_live_iterate_equals_finalize_with_a_dropped_row_and_an_eliminated_column() {
    check("active-set-sqp", Scaling::Presolve);
}

#[test]
fn sqp_live_iterate_equals_finalize_under_gradient_scaling_with_a_fixed_variable() {
    check("active-set-sqp", Scaling::GradientBased);
}

#[test]
fn sqp_live_iterate_equals_finalize_under_user_x_scaling_with_a_fixed_variable() {
    check("active-set-sqp", Scaling::UserX);
}

/// A TNLP that does not ask for the live iterate pays nothing for it, and
/// reads none, on the interior-point arm (the gate every Python solve,
/// whose bridge fires the callback unconditionally, relies on).
#[test]
fn ipm_installs_no_iterate_for_a_tnlp_that_does_not_want_one() {
    struct NoWant(Model);
    impl TNLP for NoWant {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            self.0.get_nlp_info()
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            self.0.get_bounds_info(b)
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            self.0.get_starting_point(sp)
        }
        fn eval_f(&mut self, x: &[Number], n: bool) -> Option<Number> {
            self.0.eval_f(x, n)
        }
        fn eval_grad_f(&mut self, x: &[Number], n: bool, g: &mut [Number]) -> bool {
            self.0.eval_grad_f(x, n, g)
        }
        fn eval_g(&mut self, x: &[Number], n: bool, g: &mut [Number]) -> bool {
            self.0.eval_g(x, n, g)
        }
        fn eval_jac_g(&mut self, x: Option<&[Number]>, n: bool, m: SparsityRequest<'_>) -> bool {
            self.0.eval_jac_g(x, n, m)
        }
        fn eval_h(
            &mut self,
            x: Option<&[Number]>,
            n: bool,
            of: Number,
            l: Option<&[Number]>,
            nl: bool,
            m: SparsityRequest<'_>,
        ) -> bool {
            self.0.eval_h(x, n, of, l, nl, m)
        }
        fn intermediate_callback(&mut self, st: IterStats, d: &IpoptData, q: &IpoptCq) -> bool {
            self.0.intermediate_callback(st, d, q)
        }
        fn finalize_solution(&mut self, s: Solution<'_>, d: &IpoptData, q: &IpoptCq) {
            self.0.finalize_solution(s, d, q)
        }
    }
    let p = Rc::new(RefCell::new(NoWant(Model {
        k: 1.0,
        ..Default::default()
    })));
    let mut a = IpoptApplication::new();
    a.initialize().unwrap();
    a.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    assert_eq!(
        a.optimize_tnlp(p.clone()),
        ApplicationReturnStatus::SolveSucceeded
    );
    let m = Rc::try_unwrap(p).ok().unwrap().into_inner().0;
    assert!(m.fires > 1);
    assert!(m.last.is_none(), "an iterate was installed: {:?}", m.last);
}

// ---------------------------------------------------------------------
// The scaled iterate (`GetIpoptCurrentIterate(scaled = TRUE)`)
// ---------------------------------------------------------------------

fn assert_rel(what: &str, got: Number, want: Number) {
    assert!(
        (got - want).abs() <= 1e-9 * (1.0 + want.abs()),
        "{what}: {got:e} vs {want:e}"
    );
}

/// Upstream `get_curr_iterate(scaled = true)` is the algorithm's own
/// iterate resorted into the TNLP's shape: `x` as it iterates, its `z` and
/// `y` as they are. Checked against the interior-point iterate read
/// straight out of `IpoptData` at the same fire, under gradient-based
/// scaling (row factors and an objective factor `3·df`) — and at the fixed
/// `x3`, which the algorithm does not hold, upstream's `ResortX` value and
/// `obj·z` (its manual rescale of `ResortBoundMultipliers`' unscaled
/// recovery).
#[test]
fn ipm_scaled_iterate_is_the_algorithms_own_iterate() {
    let m = solve("interior-point", Scaling::GradientBased);
    assert_eq!(m.raw.len(), m.scaled.len());
    assert!(m.raw.len() > 1);
    let mut obj_seen = 1.0;
    for (k, ((raw, obj), (sc, un))) in m
        .raw
        .iter()
        .zip(m.scaled.iter().zip(&m.unscaled))
        .enumerate()
    {
        obj_seen = *obj;
        for j in 0..4 {
            if raw.x[j].is_nan() {
                assert_rel(&format!("fire {k} fixed x[{j}]"), sc.x[j], un.x[j]);
                assert_rel(
                    &format!("fire {k} fixed z_u[{j}]"),
                    sc.z_u[j],
                    obj * un.z_u[j],
                );
                assert_rel(
                    &format!("fire {k} fixed z_l[{j}]"),
                    sc.z_l[j],
                    obj * un.z_l[j],
                );
            } else {
                assert_rel(&format!("fire {k} x[{j}]"), sc.x[j], raw.x[j]);
                assert_rel(&format!("fire {k} z_l[{j}]"), sc.z_l[j], raw.z_l[j]);
                assert_rel(&format!("fire {k} z_u[{j}]"), sc.z_u[j], raw.z_u[j]);
            }
        }
        for i in 0..3 {
            assert_rel(
                &format!("fire {k} lambda[{i}]"),
                sc.lambda[i],
                raw.lambda[i],
            );
        }
    }
    // Informative only if the objective factor is not 1 and the fixed
    // variable carries a multiplier.
    assert!(
        obj_seen > 3.0 * 1.01 || obj_seen < 3.0 * 0.99,
        "df did not engage: {obj_seen}"
    );
    assert!(m.out.unwrap().z_u[3] > 1e-3);
}

/// Under presolve the scaled iterate is the scaled values of the problem
/// the algorithm solved, lifted like the unscaled one: a removed column or
/// row carries factor 1, a multiplier the objective factor. Here only
/// `obj_scaling_factor = 3` scales (`K = 1`: gradient-based does not
/// engage), so every entry is `(x, 3·z, g, 3·λ)` of the unscaled read —
/// kept and removed alike — on both arms.
fn check_presolve_scaled(algorithm: &str) {
    let m = solve_with(algorithm, Scaling::Presolve, Some(3.0));
    assert!(m.scaled.len() > 1);
    for (k, (sc, un)) in m.scaled.iter().zip(&m.unscaled).enumerate() {
        assert_eq!(sc.x.len(), 4);
        assert_eq!(sc.g.len(), 3);
        for j in 0..4 {
            assert_rel(&format!("{algorithm} fire {k} x[{j}]"), sc.x[j], un.x[j]);
            assert_rel(
                &format!("{algorithm} fire {k} z_l[{j}]"),
                sc.z_l[j],
                3.0 * un.z_l[j],
            );
            assert_rel(
                &format!("{algorithm} fire {k} z_u[{j}]"),
                sc.z_u[j],
                3.0 * un.z_u[j],
            );
        }
        for i in 0..3 {
            assert_rel(&format!("{algorithm} fire {k} g[{i}]"), sc.g[i], un.g[i]);
            assert_rel(
                &format!("{algorithm} fire {k} lambda[{i}]"),
                sc.lambda[i],
                3.0 * un.lambda[i],
            );
        }
    }
    let out = m.out.unwrap();
    assert!(out.z_u[3] > 1e-3 && out.lambda.iter().any(|l| l.abs() > 1e-3));
}

#[test]
fn ipm_scaled_iterate_under_presolve_is_lifted_with_the_algorithms_factors() {
    check_presolve_scaled("interior-point");
}

#[test]
fn sqp_scaled_iterate_under_presolve_is_lifted_with_the_algorithms_factors() {
    check_presolve_scaled("active-set-sqp");
}

/// The same under user scaling, where the factors differ per entry: every
/// column's `x̃/x` is `D_j` (the algorithm holds it) or `1` (presolve
/// removed it, or it is the fixed `x3`), its `z̃/z` is `3/(x̃/x)`, every
/// row's `g̃/g` is its own `s_i` or `1` and `λ̃/λ = 3/(g̃/g)` — and both
/// kinds occur. A lift that misaligned the factors would put a neighbour's
/// factor on an entry.
#[test]
fn ipm_scaled_iterate_under_presolve_carries_each_entrys_own_factor() {
    let s = [0.01, 5.0, 1.0];
    let m = solve("interior-point", Scaling::PresolveUserX);
    let (sc, un) = (m.scaled.last().unwrap(), m.unscaled.last().unwrap());
    let (mut kept_col, mut cut_col, mut kept_row, mut cut_row) = (false, false, false, false);
    for j in 0..4 {
        let f = sc.x[j] / un.x[j];
        let kept = (f - D[j]).abs() <= 1e-9 * D[j];
        assert!(kept || (f - 1.0).abs() <= 1e-9, "x[{j}] factor {f}");
        if j < 3 {
            kept_col |= kept;
            cut_col |= !kept;
        }
        for (z, zu) in [(sc.z_l[j], un.z_l[j]), (sc.z_u[j], un.z_u[j])] {
            assert_rel(
                &format!("z[{j}]"),
                z,
                3.0 * zu / if kept { D[j] } else { 1.0 },
            );
        }
    }
    for i in 0..3 {
        // An eliminated equality reads g = 0 exactly; its factor shows in λ.
        let f = if un.g[i].abs() > 1e-12 {
            let f = sc.g[i] / un.g[i];
            assert_rel(
                &format!("lambda[{i}]"),
                sc.lambda[i],
                3.0 * un.lambda[i] / f,
            );
            f
        } else {
            assert_eq!(sc.g[i], 0.0);
            3.0 * un.lambda[i] / sc.lambda[i]
        };
        let kept = (f - s[i]).abs() <= 1e-9 * s[i] && s[i] != 1.0;
        assert!(kept || (f - 1.0).abs() <= 1e-9, "g[{i}] factor {f}");
        kept_row |= kept;
        cut_row |= !kept && s[i] != 1.0;
    }
    assert!(
        kept_col && cut_col,
        "columns: kept {kept_col}, removed {cut_col}"
    );
    assert!(
        kept_row && cut_row,
        "rows: kept {kept_row}, removed {cut_row}"
    );
}

/// Presolve *and* per-variable scaling: the `ScalingTnlp` sits outside
/// the presolve layers, so the algorithm's NLP cannot see its factors and
/// the live iterate reached the user in `x̃ = d ⊙ x` (`x0` read 3.0 against
/// the solution's 1.5). The wrapper now lifts it, as it finalizes.
#[test]
fn ipm_live_iterate_equals_finalize_under_presolve_and_user_x_scaling() {
    check("interior-point", Scaling::PresolveUserX);
}

#[test]
fn sqp_live_iterate_equals_finalize_under_presolve_and_user_x_scaling() {
    check("active-set-sqp", Scaling::PresolveUserX);
}

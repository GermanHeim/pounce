//! `nlp_scaling_method=curvature-based` end-to-end (gh #703).
//!
//! The default `gradient-based` scaling is a point sample: it reads the
//! Jacobian once at x₀ and cannot see a row whose derivative vanishes
//! there, nor a column imbalance that the sample happens not to expose.
//! `curvature-based` derives the factors from the model's quadratic
//! coefficients instead — one joint variable scaling `D` equilibrated
//! across `Q₀ + Σλᵢ Qᵢ`, then a per-row `eᵢ = 1/max(‖D Qᵢ D‖_∞,
//! ‖D aᵢ‖_∞, |bᵢ|)`.
//!
//! What is pinned here:
//!
//! * the **invariance** claim, against a fixture pair that is one problem
//!   in two coordinate systems (see `fixtures/README-qcqp-columns.md`):
//!   on the ill-conditioned twin `curvature-based` returns the
//!   well-conditioned answer to within an ulp and on the same number of
//!   iterations, where `gradient-based` loses five digits;
//! * that it **refuses** a model it is not defined for rather than solving
//!   it unscaled, which is the gh #483 failure this option must not repeat;
//! * that it is **off unless asked for** — the default path is untouched;
//! * that asking for it on a model the convex path would claim actually
//!   *applies* it, by declining that path — the gh#483 bargain, which this
//!   option needs more than any other option that has needed it;
//! * that the pair's *conic* route, which is where these fixtures actually
//!   go by default, agrees on the answer too. That last one is not about
//!   scaling at all — it is the regression pin for the rank test these
//!   fixtures caught in `psd_outer_factor` (see
//!   `rank_does_not_depend_on_the_units_the_columns_are_measured_in`).
//!
//! Every solve through `solve()` passes `solver_selection=nlp`. At this
//! size both fixtures clear the conic guards and would route to the SOCP
//! driver; gh #703 is about the NLP path, and pinning the route is what
//! keeps these tests measuring the thing they name. The conic test below
//! deliberately does the opposite and takes the default route.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

/// A scratch path nobody else in this binary can collide with. The
/// sequence number is not decoration: `cargo test` runs these in parallel
/// and several of them solve the *same* fixture, so a name built from the
/// pid and the fixture alone had two tests writing and then deleting one
/// another's report — a `NotFound` on a file that had just been written.
fn tmp(suffix: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!("pounce_curv_{}_{seq}_{suffix}", std::process::id()));
    p
}

/// Solve `nl` and return `(status, iterations, objective)`.
fn solve(nl: &str, extra: &[&str]) -> (String, u64, f64) {
    let sol = tmp(&format!("{nl}.sol"));
    let json = tmp(&format!("{nl}.json"));
    let out = Command::new(pounce_exe())
        .arg(fixture(nl))
        .arg(&sol)
        .arg("--json-output")
        .arg(&json)
        .arg("print_level=0")
        .arg("solver_selection=nlp")
        .args(extra)
        .output()
        .expect("spawn pounce");
    assert_eq!(
        out.status.code(),
        Some(0),
        "solve of {nl} {extra:?} failed\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let text = std::fs::read_to_string(&json).expect("json report");
    let v: serde_json::Value = serde_json::from_str(&text).expect("json parses");
    let status = v["solution"]["status"].as_str().unwrap_or("?").to_string();
    let iters = v["statistics"]["iteration_count"].as_u64().unwrap_or(0);
    let obj = v["statistics"]["final_objective"]
        .as_f64()
        .unwrap_or(f64::NAN);
    let _ = std::fs::remove_file(&sol);
    let _ = std::fs::remove_file(&json);
    (status, iters, obj)
}

/// The headline claim: the scheme recovers the injected column scaling, so
/// the solve stops depending on which coordinates the model was written in.
#[test]
fn curvature_scaling_is_invariant_to_a_column_imbalance() {
    let (s_well, it_well, obj_well) = solve("qcqp_columns_wellcond.nl", &[]);
    assert_eq!(s_well, "SolveSucceeded");

    // The default loses digits on the ill-conditioned twin: same problem,
    // different coordinates, a visibly different answer.
    let (s_ill, it_ill, obj_ill) = solve("qcqp_columns_illcond.nl", &[]);
    assert_eq!(s_ill, "SolveSucceeded");
    assert!(
        (obj_ill - obj_well).abs() > 1e-6,
        "the fixture pair no longer exercises the failure it was built for: \
         gradient-based got {obj_ill} vs {obj_well}"
    );
    assert!(
        it_ill > it_well,
        "expected the ill-conditioned twin to cost more iterations under \
         the default ({it_ill} vs {it_well})"
    );

    // curvature-based reproduces the well-conditioned answer on the
    // ill-conditioned model — to the last bit, not merely to a tolerance.
    let (s_a, it_a, obj_a) = solve(
        "qcqp_columns_wellcond.nl",
        &["nlp_scaling_method=curvature-based"],
    );
    let (s_b, it_b, obj_b) = solve(
        "qcqp_columns_illcond.nl",
        &["nlp_scaling_method=curvature-based"],
    );
    assert_eq!(s_a, "SolveSucceeded");
    assert_eq!(s_b, "SolveSucceeded");
    // Not bit equality: `D` recovers the injected `c` up to how the Ruiz
    // sweep's factors round, so the two solves agree to a couple of **ulps**
    // rather than exactly. Pinned as a measured distance, not as a
    // tolerance — the claim is that the coordinate system stops mattering,
    // and a regression that made it matter would have to move this number
    // by far more than the two ulps of headroom here.
    let ulps = obj_a.to_bits().abs_diff(obj_b.to_bits());
    assert!(
        ulps <= 2,
        "curvature-based should make the two coordinate systems the same \
         solve to within an ulp: {obj_a} vs {obj_b} ({ulps} ulp)"
    );
    assert_eq!(it_a, it_b, "…including the trajectory length");
    assert!(
        (obj_a - obj_well).abs() < 1e-8,
        "and it should agree with the well-conditioned reference \
         ({obj_a} vs {obj_well})"
    );
    assert!(
        it_b < it_ill,
        "expected fewer iterations than the default on the ill-conditioned \
         twin ({it_b} vs {it_ill})"
    );
}

/// A model with a genuine nonlinearity has no constant `Qᵢ`, so the
/// magnitude envelope this scheme equilibrates does not exist. It must say
/// so rather than accept the option and solve unscaled — the shape of
/// gh #483, which is why the message names the alternatives.
#[test]
fn a_model_without_constant_curvature_is_refused_readably() {
    let sol = tmp("refuse.sol");
    let out = Command::new(pounce_exe())
        .arg(fixture("cresc4.nl"))
        .arg(&sol)
        .arg("print_level=0")
        .arg("nlp_scaling_method=curvature-based")
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(2), "expected a usage-style refusal");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("curvature-based") && err.contains("degree <= 2"),
        "refusal should name the option and the reason, got: {err}"
    );
    assert!(
        err.contains("gradient-based") && err.contains("user-scaling"),
        "refusal should name what to use instead, got: {err}"
    );
}

/// The option is opt-in: without it, the ill-conditioned fixture solves
/// exactly as it did before this feature existed.
#[test]
fn the_default_path_is_untouched() {
    let (s1, it1, obj1) = solve("qcqp_columns_illcond.nl", &[]);
    let (s2, it2, obj2) = solve(
        "qcqp_columns_illcond.nl",
        &["nlp_scaling_method=gradient-based"],
    );
    assert_eq!((s1, it1, obj1.to_bits()), (s2, it2, obj2.to_bits()));
}

/// The same fixture pair on the route they actually take by default.
///
/// `solve()` above pins `solver_selection=nlp` because gh #703 is about the
/// NLP path. But nothing in the corpus was checking where these two models
/// go when nobody pins anything — and by default they are classified as
/// convex QCQPs and reduced to second-order cones. That reduction factors
/// each quadratic row as `Σ_k f_k f_kᵀ = Q`, and the rank it finds is the
/// dimension of the cone it builds.
///
/// It found different ranks for the two twins: 24 on the well-conditioned
/// model and **17** on the ill-conditioned one, because the rank test cut at
/// `1e-12 · max_diag` and the injected column scaling moved `max_diag` by
/// nineteen orders of magnitude. Seven real directions went missing, which
/// makes the feasible set larger than the model's, so the solver
/// legitimately reached a better objective — `-400.65` against the true
/// `-364.21` — reported `SolveSucceeded`, and reported its own constraint
/// violation as `2.66e-15` when the violation of the *actual* constraint
/// was `4.948e+01`, 38% of the right-hand side. Nothing in the suite or in
/// the sweep could see it: the sweep's baseline recorded the wrong answer
/// as the expected one.
///
/// So the pin is the twin property again, one level down: the two
/// coordinate systems are the same problem, so the conic route must return
/// the same objective for both. A rank test that reads the model's units
/// cannot satisfy this.
#[test]
fn the_conic_route_gives_both_twins_the_same_answer() {
    let (s_well, _, obj_well) = solve_default("qcqp_columns_wellcond.nl");
    let (s_ill, _, obj_ill) = solve_default("qcqp_columns_illcond.nl");
    assert_eq!(s_well, "SolveSucceeded");
    assert_eq!(s_ill, "SolveSucceeded");
    let rel = ((obj_ill - obj_well) / obj_well).abs();
    assert!(
        rel < 1e-8,
        "the conic route must not depend on the coordinate system: \
         {obj_ill} vs {obj_well} ({rel:.2e} relative)"
    );
}

/// …and that it *is* the conic route, so the test above cannot be quietly
/// satisfied by a routing change that sends both models to the filter-IPM.
/// That is the gh #690 lesson: a fixture that stops exercising the engine it
/// was built for keeps passing while covering nothing.
#[test]
fn both_twins_take_the_conic_route_by_default() {
    for nl in ["qcqp_columns_wellcond.nl", "qcqp_columns_illcond.nl"] {
        let sol = tmp(&format!("{nl}.route.sol"));
        let out = Command::new(pounce_exe())
            .arg(fixture(nl))
            .arg(&sol)
            .output()
            .expect("spawn pounce");
        let stdout = String::from_utf8_lossy(&out.stdout);
        let _ = std::fs::remove_file(&sol);
        assert!(
            stdout.contains("Problem class: convex QCQP.")
                && stdout.contains("conic interior-point (pounce-convex)"),
            "{nl} must reduce to a cone by default; stdout:\n{stdout}"
        );
    }
}

/// `solve()` with the route left alone. Kept separate rather than
/// parameterizing `solve()`, so that no future edit can drop
/// `solver_selection=nlp` from the gh #703 tests by accident.
fn solve_default(nl: &str) -> (String, u64, f64) {
    solve_default_with(nl, &[])
}

/// As [`solve_default`], with extra options appended — still on the default
/// route, so what is measured includes the routing decision itself.
fn solve_default_with(nl: &str, extra: &[&str]) -> (String, u64, f64) {
    let sol = tmp(&format!("{nl}.dflt.sol"));
    let json = tmp(&format!("{nl}.dflt.json"));
    let out = Command::new(pounce_exe())
        .arg(fixture(nl))
        .arg(&sol)
        .arg("--json-output")
        .arg(&json)
        .arg("print_level=0")
        .args(extra)
        .output()
        .expect("spawn pounce");
    assert_eq!(
        out.status.code(),
        Some(0),
        "default-route solve of {nl} failed\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let text = std::fs::read_to_string(&json).expect("json report");
    let v: serde_json::Value = serde_json::from_str(&text).expect("json parses");
    let status = v["solution"]["status"].as_str().unwrap_or("?").to_string();
    let iters = v["statistics"]["iteration_count"].as_u64().unwrap_or(0);
    let obj = v["statistics"]["final_objective"]
        .as_f64()
        .unwrap_or(f64::NAN);
    let _ = std::fs::remove_file(&sol);
    let _ = std::fs::remove_file(&json);
    (status, iters, obj)
}

/// gh#483, for the third time. `curvature-based` reaches the engine through
/// `TNLP::get_scaling_parameters`, and the convex solvers never call it —
/// they equilibrate internally. So a convex-classified model would accept
/// the option and solve without it.
///
/// This option is the one where that matters most. The models it is defined
/// for are the models with quadratic rows, which is exactly the population
/// `classify_problem` routes to the convex path: of the **47 corpus fixtures
/// the option accepts, 38 classify convex** — including both fixtures gh #703
/// added for it — and 31 of those 38 carry curvature and are rerouted here. Left ungated, the feature would be inert
/// by default on the majority of the models it exists for, and inert
/// *silently* — which is the sentence gh#483 was filed over.
///
/// Under `auto` the fix is to decline the fast path, as it already does for
/// `user-scaling`, a post-optimal request, and a negative
/// `obj_scaling_factor`. The note is required, not decoration: the routing
/// banner changes engine, and a user who reads only the banner should be
/// able to find out why.
#[test]
fn curvature_scaling_declines_the_convex_path_rather_than_being_dropped() {
    let sol = tmp("decline.sol");
    let out = Command::new(pounce_exe())
        .arg(fixture("qcqp_columns_illcond.nl"))
        .arg(&sol)
        .arg("nlp_scaling_method=curvature-based")
        .output()
        .expect("spawn pounce");
    let _ = std::fs::remove_file(&sol);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    // The model still classifies as a convex QCQP — the point is what runs
    // *despite* that, so a routing change that stopped classifying it would
    // make this test vacuous.
    assert!(
        stdout.contains("Problem class: convex QCQP."),
        "fixture no longer classifies as a convex QCQP; stdout:\n{stdout}"
    );
    assert!(
        stdout.contains("NLP filter line-search interior-point (pounce-nlp)"),
        "curvature-based must decline the convex fast path under `auto`; \
         stdout:\n{stdout}"
    );
    assert!(
        stderr.contains("nlp_scaling_method=curvature-based")
            && stderr.contains("routing to the general NLP"),
        "the reroute must say why it rerouted; stderr:\n{stderr}"
    );
}

/// The other half of the same bargain: an *explicit* convex
/// `solver_selection` is respected — the user named the engine — but the
/// option it cannot honor is reported instead of dropped. Silence here is
/// the gh#483 failure with the reroute merely moved one branch over.
#[test]
fn a_forced_convex_solve_warns_that_curvature_scaling_is_skipped() {
    let sol = tmp("forced.sol");
    let out = Command::new(pounce_exe())
        .arg(fixture("qcqp_columns_illcond.nl"))
        .arg(&sol)
        .arg("nlp_scaling_method=curvature-based")
        .arg("solver_selection=socp")
        .output()
        .expect("spawn pounce");
    let _ = std::fs::remove_file(&sol);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stdout.contains("conic interior-point (pounce-convex)"),
        "an explicit solver_selection must still be honored; stdout:\n{stdout}"
    );
    assert!(
        stderr.contains("warning:")
            && stderr.contains("nlp_scaling_method=curvature-based")
            && stderr.contains("skipped"),
        "the skipped option must be reported; stderr:\n{stderr}"
    );
}

/// …and the reroute is conditional on the option, not on the fixture: with
/// no scaling request the same model keeps the conic route it had before
/// gh #703. Guards against "fix the silence by never taking the fast path".
#[test]
fn without_the_option_the_convex_path_is_still_taken() {
    let sol = tmp("noopt.sol");
    let out = Command::new(pounce_exe())
        .arg(fixture("qcqp_columns_illcond.nl"))
        .arg(&sol)
        .output()
        .expect("spawn pounce");
    let _ = std::fs::remove_file(&sol);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stdout.contains("conic interior-point (pounce-convex)"),
        "stdout:\n{stdout}"
    );
    assert!(
        !stderr.contains("curvature-based"),
        "nothing to say when nothing was asked for; stderr:\n{stderr}"
    );
}

/// The limit of that bargain, and the reason the reroute is not
/// unconditional. `curvature-based` accepts any model of degree ≤ 2, and an
/// LP qualifies — with every `Qᵢ` empty. Stage 1's `K̂` then loses its `P̂`
/// block and stage 2 collapses to `eᵢ = 1/max(‖D aᵢ‖_∞, |bᵢ|)`, so what
/// runs is Ruiz equilibration of `[A b]`: a real scaling, but not one that
/// read any curvature, because the model had none.
///
/// `pounce-convex` already equilibrates internally, so declining the fast
/// path here buys a different engine and nothing else. Measured on
/// `lp_israel` (NETLIB, 142×174) before this gate existed:
///
/// | route | iterations |
/// |---|---|
/// | convex, gradient-based (default) | 29 |
/// | NLP, gradient-based | 135 |
/// | NLP, curvature-based | 296 |
///
/// A 10× slowdown, of which the engine switch is 4.7× and a scaling scheme
/// with nothing to read is the other 2.2×. So the request is still
/// *reported* — silence is what gh#483 is about — but it is reported as a
/// note that the fast path is being kept, not paid for with the reroute.
#[test]
fn an_lp_has_no_curvature_to_read_and_keeps_the_convex_path() {
    let sol = tmp("lp_keep.sol");
    let out = Command::new(pounce_exe())
        .arg(fixture("lp_israel.nl"))
        .arg(&sol)
        .arg("nlp_scaling_method=curvature-based")
        .output()
        .expect("spawn pounce");
    let _ = std::fs::remove_file(&sol);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stdout.contains("pounce-convex"),
        "an LP has nothing for the scheme to read, so the fast path stands; \
         stdout:\n{stdout}"
    );
    // Reported, not silent: the gh#483 property is that the user is told what
    // became of the option, which a kept fast path satisfies as well as a
    // reroute does.
    assert!(
        stderr.contains("nlp_scaling_method=curvature-based")
            && stderr.contains("every quadratic coefficient in this model is zero"),
        "the kept fast path must say why it was kept; stderr:\n{stderr}"
    );
}

/// …and keeping it costs the LP nothing at all — same status, same iteration
/// count, same objective bits as a run that never mentioned the option. This
/// is the assertion the gate exists for: before it, this fixture went from
/// 29 iterations to 296 for asking.
#[test]
fn keeping_the_convex_path_leaves_an_lp_bit_for_bit_unchanged() {
    let plain = solve_default("lp_israel.nl");
    let asked = solve_default_with("lp_israel.nl", &["nlp_scaling_method=curvature-based"]);
    assert_eq!(
        plain, asked,
        "asking for a scaling the model has no coefficients for must not \
         move the trajectory"
    );
}

/// A quadratic objective is enough to flip the same fixture family the other
/// way: `convex_qp_share1b` is `lp_share1b` with a `P` bolted on, so it does
/// carry curvature and does get the reroute. Pins the discriminator itself,
/// so a change that made `quadratic` always-true or always-false fails here
/// rather than as a trajectory surprise.
#[test]
fn the_same_model_with_a_quadratic_objective_does_get_the_reroute() {
    for (nl, expect_reroute) in [("lp_share1b.nl", false), ("convex_qp_share1b.nl", true)] {
        let sol = tmp(&format!("{nl}.disc.sol"));
        let out = Command::new(pounce_exe())
            .arg(fixture(nl))
            .arg(&sol)
            .arg("nlp_scaling_method=curvature-based")
            .output()
            .expect("spawn pounce");
        let _ = std::fs::remove_file(&sol);
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert_eq!(
            stdout.contains("pounce-nlp"),
            expect_reroute,
            "{nl}: reroute expectation not met; stdout:\n{stdout}"
        );
    }
}

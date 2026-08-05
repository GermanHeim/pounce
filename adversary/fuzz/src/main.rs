//! Adversary harness for the gh#484 fixes.
//!
//! Two probes, run as subcommands:
//!
//!   `qp <n> [seed]`        — elastic infeasibility-certificate fuzz
//!   `warmstart <n> [seed]` — C warm-start answer-transparency fuzz
//!
//! Both are deterministic given the seed. `qp` also writes every
//! instance to `instances.jsonl` so `runs/*_adjudicate.py` can re-decide
//! feasibility with `scipy.optimize.linprog` — an oracle that has never
//! heard of pounce.

mod instances;
mod qp_probe;
mod rng;
mod warmstart_probe;

use instances::Truth;
use qp_probe::Verdict;
use rng::Rng;
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("qp");
    let count: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(500);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20260805);

    match cmd {
        "qp" => qp_fuzz(count, seed),
        "warmstart" => warmstart_fuzz(count, seed),
        "qp-one" => qp_one(count as u64),
        other => {
            eprintln!("unknown probe {other:?}; expected `qp` or `warmstart`");
            std::process::exit(2);
        }
    }
}

fn qp_fuzz(count: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    let mut jsonl = std::fs::File::create("instances.jsonl").expect("open instances.jsonl");

    let mut n_feasible = 0usize;
    let mut n_infeasible = 0usize;
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    let mut failures: Vec<(u64, Verdict, String, String)> = Vec::new();
    // Completeness on provably-infeasible instances, a quality metric:
    // certified vs merely non-committal.
    let mut certified = 0usize;
    let mut noncommittal = 0usize;

    for k in 0..count {
        let s = seed.wrapping_add(k as u64 * 0x1000_0001);
        let mut r = Rng::new(s);
        let inst = if rng.chance(0.65) {
            instances::feasible(&mut r, s)
        } else {
            instances::infeasible(&mut r, s)
        };

        // Self-check the generator before believing anything it produced.
        // A witness that does not satisfy its own instance would turn
        // every downstream conclusion into noise.
        if let Some(w) = inst.witness.as_ref() {
            let v = inst.violation(w);
            let scale = inst.a.iter().map(|x| x.abs()).fold(1.0_f64, f64::max);
            assert!(
                v <= 1e-9 * scale,
                "generator bug: witness for seed {s} violates its own instance by {v:.3e}"
            );
        }

        writeln!(jsonl, "{}", inst.to_json()).expect("write instance");

        let out = qp_probe::run(&inst);
        if let Ok(path) = std::env::var("ADV_DUMP") {
            use std::io::Write as _;
            let free = (0..inst.n)
                .filter(|&j| {
                    inst.xl[j] <= pounce_common::types::NLP_LOWER_BOUND_INF
                        || inst.xu[j] >= pounce_common::types::NLP_UPPER_BOUND_INF
                })
                .count();
            let neq = (0..inst.m).filter(|&i| inst.bl[i] == inst.bu[i]).count();
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .expect("dump");
            writeln!(
                f,
                "{s}\t{:?}\t{}\t{}\t{}\t{}",
                inst.truth, out.status, inst.kind, free, neq
            )
            .ok();
        }
        *counts.entry(format!("{:?}/{}", inst.truth, out.status)).or_default() += 1;
        match inst.truth {
            Truth::Feasible => n_feasible += 1,
            Truth::Infeasible => {
                n_infeasible += 1;
                if out.status == "Infeasible" {
                    certified += 1;
                } else if out.status != "Optimal" {
                    noncommittal += 1;
                }
            }
        }

        if out.verdict == Verdict::Errored {
            println!("  ERR seed={s}: {}", out.detail);
        }
        match out.verdict {
            Verdict::Ok | Verdict::Weak => {}
            v => failures.push((s, v, out.detail.clone(), inst.proof.clone())),
        }
    }

    println!("=== QP elastic-certificate fuzz ===");
    println!("instances={count} seed={seed} (feasible={n_feasible} infeasible={n_infeasible})");
    println!("--- status distribution (truth/status) ---");
    for (k, v) in &counts {
        println!("  {k:44} {v}");
    }
    println!("--- completeness on provably-infeasible instances ---");
    println!(
        "  certified Infeasible : {certified}/{n_infeasible}{}",
        if n_infeasible > 0 {
            format!(" ({:.1}%)", 100.0 * certified as f64 / n_infeasible as f64)
        } else {
            String::new()
        }
    );
    println!("  non-committal        : {noncommittal}/{n_infeasible}");
    println!("--- invariant violations ---");
    if failures.is_empty() {
        println!("  none");
    } else {
        for (s, v, d, proof) in failures.iter().take(20) {
            println!("  seed={s} {v:?}: {d}");
            if !proof.is_empty() {
                println!("      instance: {proof}");
            }
        }
        if failures.len() > 20 {
            println!("  … and {} more", failures.len() - 20);
        }
    }
    println!(
        "VERDICT: {}",
        if failures.is_empty() {
            "PASS".to_string()
        } else {
            format!("FAIL ({} invariant violations)", failures.len())
        }
    );
    std::process::exit(if failures.is_empty() { 0 } else { 1 });
}

fn warmstart_fuzz(count: usize, seed: u64) {
    use warmstart_probe::Params;
    let mut rng = Rng::new(seed);
    let mut failures: Vec<String> = Vec::new();
    let mut compared = 0usize;
    let mut skipped_baseline = 0usize;
    let mut nonconvex_skipped = 0usize;
    let mut rejected = 0usize;

    for k in 0..count {
        let s = seed.wrapping_add(k as u64 * 0x2000_0003);
        let mut r = Rng::new(s);
        let n = r.int(3, 5);
        // A box translated away from the origin is the adversarial part:
        // the discarded-iterate bug restarted every warm solve at x = 0,
        // which is only catastrophic when 0 is outside the box.
        let lo = if r.chance(0.5) { r.range(0.5, 5.0) } else { r.range(-2.0, 0.2) };
        let hi = lo + r.range(1.0, 5.0);
        // Half the instances are convex, where answer-transparency is a
        // theorem; the other half are nonconvex, where only the weaker
        // feasibility invariant is claimed.
        let convex = r.chance(0.5);
        let p = Params {
            convex,
            n,
            t: (0..n).map(|_| r.range(lo - 2.0, hi + 2.0)).collect(),
            c: (0..n - 1).map(|_| r.range(-2.0, 2.0)).collect(),
            s: r.range(lo * n as f64, hi * n as f64),
            p: lo * lo * r.range(0.5, 1.5),
            lo,
            hi,
        };

        // Baseline: no warm-start call at all.
        let x0: Vec<f64> = (0..n).map(|_| r.range(lo, hi)).collect();
        let base = warmstart_probe::solve(&p, &x0, None);
        if base.status != 0 {
            // Nothing to compare against; the property is about
            // preserving an answer, and here there is none.
            skipped_baseline += 1;
            continue;
        }

        // Working sets to stage. Every one is *valid* (in-range status
        // codes); most are deliberately *wrong*. A hint is a hint: none
        // may change where the solve lands.
        let true_ws = warmstart_probe::working_set_after(&p, &x0);
        let mut sets: Vec<(&str, Vec<i32>, Vec<i32>)> = vec![
            ("all-inactive", vec![0; n], vec![0; 2]),
            ("all-at-lower", vec![1; n], vec![1; 2]),
            ("all-at-upper", vec![2; n], vec![2; 2]),
            // Codes 0..=2 only: Inactive / AtLower / AtUpper. Code 3 is
            // Fixed (variables) / Equality (rows) — a *semantic* claim
            // about the problem, not a claim about the active set, so
            // feeding it for a variable whose bounds differ is a
            // different kind of input and is tested separately below.
            (
                "random-active-set",
                (0..n).map(|_| r.int(0, 2) as i32).collect(),
                (0..2).map(|_| r.int(0, 2) as i32).collect(),
            ),
            (
                "random-with-fixed-code",
                (0..n).map(|_| r.int(0, 3) as i32).collect(),
                (0..2).map(|_| r.int(0, 3) as i32).collect(),
            ),
        ];
        if let Some((b, c)) = true_ws {
            sets.push(("converged", b, c));
        }

        for (tag, b, c) in &sets {
            let warm = warmstart_probe::solve(&p, &x0, Some((b, c)));
            compared += 1;

            // Property 0: the setter's verdict must match the model. A
            // code that asserts something false about the problem —
            // `Fixed` on a variable whose bounds differ, `AtUpper` on a
            // row with no upper bound — has to be rejected, not accepted
            // and silently acted on. This is the defect the first run of
            // this probe found: accepted, TRUE returned, and a *convex*
            // program came back with the wrong optimum.
            let should_accept = warmstart_probe::set_is_structurally_valid(&p, b, c);
            if warm.accepted != should_accept {
                failures.push(format!(
                    "seed={s} ws={tag}: IpoptSetWarmStartWorkingSet returned {} \
                     for a set that is structurally {}",
                    if warm.accepted { "TRUE" } else { "FALSE" },
                    if should_accept { "valid" } else { "INVALID" },
                ));
                continue;
            }
            if !warm.accepted {
                // Rejected, so no working set was staged: the solve is the
                // baseline by construction and there is nothing to compare.
                rejected += 1;
                continue;
            }

            if warm.status != base.status {
                failures.push(format!(
                    "seed={s} ws={tag}: status {} without the call, {} with it \
                     (box [{lo:.3}, {hi:.3}], x0={x0:?})",
                    base.status, warm.status,
                ));
                continue;
            }
            // Invariant 1 — holds regardless of convexity: a converged
            // answer must satisfy the constraints. This alone catches the
            // gh#484 bug, which returned x = 0 on a box excluding it.
            let v = warmstart_probe::violation(&p, &warm.x);
            let vb = warmstart_probe::violation(&p, &base.x);
            if v > 1e-6 && v > vb * 10.0 {
                failures.push(format!(
                    "seed={s} ws={tag}: warm solve reported success at an \
                     infeasible point (violation {v:.3e}; baseline {vb:.3e}); \
                     box [{lo:.3}, {hi:.3}] warm x={:?}",
                    warm.x
                ));
                continue;
            }

            // Invariant 2 — convex instances only, where the minimizer is
            // unique and a working set therefore cannot change it. On a
            // nonconvex program a different working set may steer the SQP
            // to a different local minimum, which is correct behaviour,
            // not a defect; asserting otherwise would be asserting a
            // falsehood.
            if !convex {
                nonconvex_skipped += 1;
                continue;
            }
            let dx = warm
                .x
                .iter()
                .zip(&base.x)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let dobj = (warm.obj - base.obj).abs() / base.obj.abs().max(1.0);
            if dx > 1e-5 || dobj > 1e-7 {
                failures.push(format!(
                    "seed={s} ws={tag}: staging a working set moved the answer \
                     on a CONVEX program (|Δx|∞={dx:.3e}, Δobj_rel={dobj:.3e}); \
                     base x={:?} warm x={:?}",
                    base.x, warm.x
                ));
            }
        }
    }

    println!("=== C warm-start answer-transparency fuzz ===");
    println!("problems={count} seed={seed}");
    println!("comparisons={compared} (skipped {skipped_baseline} whose baseline did not converge)");
    println!("  strict answer-transparency checked on the convex half; {nonconvex_skipped} nonconvex comparisons got the feasibility invariant only");
    println!("  {rejected} structurally-invalid sets correctly rejected by the setter");
    println!("--- property violations ---");
    if failures.is_empty() {
        println!("  none");
    } else {
        for f in failures.iter().take(20) {
            println!("  {f}");
        }
        if failures.len() > 20 {
            println!("  … and {} more", failures.len() - 20);
        }
    }
    println!(
        "VERDICT: {}",
        if failures.is_empty() {
            "PASS".to_string()
        } else {
            format!("FAIL ({} property violations)", failures.len())
        }
    );
    std::process::exit(if failures.is_empty() { 0 } else { 1 });
}

/// Re-solve a single feasible instance under a sweep of solver options.
/// Separates "the tolerance is arithmetically unreachable on this data"
/// from "the certificate logic is wrong": if loosening `feas_tol` by six
/// orders of magnitude still yields `Infeasible`, no tolerance argument
/// survives.
fn qp_one(seed: u64) {
    let mut r = Rng::new(seed);
    // Replay whichever generator produced this seed. `qp_fuzz` decides
    // via a separate stream, so try feasible first and fall back to
    // infeasible when `ADV_KIND=infeasible` says so.
    let infeasible = std::env::var("ADV_KIND").as_deref() == Ok("infeasible");
    let inst = if infeasible {
        instances::infeasible(&mut r, seed)
    } else {
        instances::feasible(&mut r, seed)
    };
    println!("kind={} truth={:?} proof={}", inst.kind, inst.truth, inst.proof);
    println!("seed={seed} n={} m={}", inst.n, inst.m);
    let eq: Vec<usize> = (0..inst.m).filter(|&i| inst.bl[i] == inst.bu[i]).collect();
    println!("equality rows: {eq:?}");
    if let Some(w) = inst.witness.as_ref() {
        println!("witness violation (my arithmetic): {:.3e}", inst.violation(w));
    }
    for ft in [1e-9, 1e-8, 1e-7, 1e-6, 1e-4, 1e-2] {
        let out = qp_probe::run_with(&inst, ft);
        println!("  feas_tol={ft:.0e} -> status={:12} verdict={:?}", out.status, out.verdict);
    }
}

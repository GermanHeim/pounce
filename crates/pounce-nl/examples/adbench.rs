//! Scratch harness: time the three TNLP evaluators in isolation.
//!
//! `cargo run --release -p pounce-nl --example adbench -- model.nl [reps]`
//!
//! Not a regression test — this exists to size AD work against solver work
//! without the solver's own dynamics (restoration, line-search retries)
//! confounding the measurement.

use pounce_nl::nl_reader::{self, NlTnlp};
use pounce_nlp::tnlp::{SparsityRequest, TNLP};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = std::path::PathBuf::from(&args[1]);
    let reps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50);

    let t0 = Instant::now();
    let prob = nl_reader::read_nl_file(&path).expect("read");
    let (n, m) = (prob.n, prob.m);
    let x0 = prob.x0.clone();
    let mut t = NlTnlp::new(prob);
    let info = t.get_nlp_info().expect("info");
    let setup = t0.elapsed().as_secs_f64();

    let nnz_j = info.nnz_jac_g as usize;
    let nnz_h = info.nnz_h_lag as usize;
    println!("n={n} m={m} nnz_jac={nnz_j} nnz_h={nnz_h} setup={setup:.3}s");

    // Perturb off the (often degenerate) start so every branch is live.
    let x: Vec<f64> = x0
        .iter()
        .enumerate()
        .map(|(i, v)| v + 0.01 * ((i % 7) as f64) + 0.001)
        .collect();
    let lambda: Vec<f64> = (0..m).map(|i| 0.5 + 0.01 * ((i % 5) as f64)).collect();

    let mut g = vec![0.0; m];
    let mut jac = vec![0.0; nnz_j];
    let mut hess = vec![0.0; nnz_h];
    let mut grad = vec![0.0; n];

    macro_rules! bench {
        ($name:literal, $body:expr) => {{
            $body; // warm
            let t = Instant::now();
            for _ in 0..reps {
                $body;
            }
            let per = t.elapsed().as_secs_f64() / reps as f64;
            println!("  {:28} {:9.3} ms/call", $name, per * 1000.0);
            per
        }};
    }

    let t_g = bench!("eval_g", {
        t.eval_g(&x, true, &mut g);
    });
    let t_gf = bench!("eval_grad_f", {
        t.eval_grad_f(&x, true, &mut grad);
    });
    let t_j = bench!("eval_jac_g (values)", {
        t.eval_jac_g(Some(&x), true, SparsityRequest::Values { values: &mut jac });
    });
    let t_h = bench!("eval_h (values)", {
        t.eval_h(
            Some(&x),
            true,
            1.0,
            Some(&lambda),
            true,
            SparsityRequest::Values { values: &mut hess },
        );
    });
    println!(
        "  {:28} {:9.3} ms/call",
        "TOTAL",
        (t_g + t_gf + t_j + t_h) * 1000.0
    );
}

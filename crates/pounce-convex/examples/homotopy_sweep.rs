//! Measurement harness for the gh #434 homotopy on/off sweep.
//!
//! gh #434 asks a question that cannot be answered by reading the code: the
//! §4.2 parametric homotopy is a net win on Maros-Mészáros (71/138 against
//! 58/138) but it *loses* seven instances, and on those it is pure cost — it
//! spends the whole budget without reaching `t = 1`. Before any guard can be
//! written, the losses have to be separated from the gains by something
//! observable at runtime, and the issue is explicit that a guard must not be
//! chosen without per-problem data at the shipped code state.
//!
//! This example is the instrument. It solves one QP through
//! [`solve_qp_active_set`] with `use_homotopy` forced on or off — the two arms
//! differing by exactly that one option — and prints a single JSON object with
//! the status, objective, time, and (parsed by the driver from the
//! `POUNCE_HOMOTOPY_DEBUG` trace on stderr) the path's step count and final
//! `t`. Everything else about the solve is the shipped configuration, so the
//! numbers describe the driver users actually get.
//!
//! Run:
//!
//! ```text
//! cargo run -p pounce-convex --release --example homotopy_sweep -- <file.qp> on|off
//! ```
//!
//! The input is the flat text form written by the sweep's `.mat` converter:
//!
//! ```text
//! n <n>
//! c <n floats>
//! P <nnz>                  ; then <nnz> lines of "row col val" (lower triangle)
//! A <m_eq> <nnz>           ; equality matrix
//! b <m_eq floats>          ; then <nnz> lines of "row col val"
//! G <m_ineq> <nnz>         ; inequality matrix
//! h <m_ineq floats>        ; then <nnz> lines of "row col val"
//! lb <n floats>
//! ub <n floats>
//! ```
//!
//! Whitespace-delimited throughout, so the reader is a token stream and the
//! layout above is documentation rather than something the parser enforces.

use pounce_convex::{
    ActiveSetOverrides, NEG_INF, POS_INF, QpOptions, QpProblem, QpStatus, Triplet,
    solve_qp_active_set,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;
use std::time::Instant;

/// Whitespace token stream over the input file.
struct Tokens<'a> {
    it: std::str::SplitAsciiWhitespace<'a>,
}

impl<'a> Tokens<'a> {
    fn new(s: &'a str) -> Self {
        Tokens {
            it: s.split_ascii_whitespace(),
        }
    }

    fn next_tok(&mut self) -> &'a str {
        self.it.next().expect("unexpected end of input")
    }

    /// Consume the expected section tag, so a malformed file fails loudly at
    /// the point of the mismatch rather than silently shifting every field.
    fn tag(&mut self, expect: &str) {
        let got = self.next_tok();
        assert_eq!(got, expect, "expected section tag `{expect}`, got `{got}`");
    }

    fn usize(&mut self) -> usize {
        self.next_tok().parse().expect("expected an integer")
    }

    fn f64(&mut self) -> f64 {
        self.next_tok().parse().expect("expected a float")
    }

    fn floats(&mut self, count: usize) -> Vec<f64> {
        (0..count).map(|_| self.f64()).collect()
    }

    fn triplets(&mut self, nnz: usize) -> Vec<Triplet> {
        (0..nnz)
            .map(|_| {
                let r = self.usize();
                let c = self.usize();
                Triplet::new(r, c, self.f64())
            })
            .collect()
    }
}

/// `1e20` is the Maros-Mészáros infinity constant; map it onto the sentinels
/// `pounce-convex` recognizes so an absent bound is not read as a real one.
fn debound(v: f64) -> f64 {
    if v <= -1e20 {
        NEG_INF
    } else if v >= 1e20 {
        POS_INF
    } else {
        v
    }
}

fn parse(text: &str) -> QpProblem {
    let mut t = Tokens::new(text);
    t.tag("n");
    let n = t.usize();
    t.tag("c");
    let c = t.floats(n);
    t.tag("P");
    let p_nnz = t.usize();
    let p_lower = t.triplets(p_nnz);
    t.tag("A");
    let m_eq = t.usize();
    let a_nnz = t.usize();
    t.tag("b");
    let b = t.floats(m_eq);
    let a = t.triplets(a_nnz);
    t.tag("G");
    let m_ineq = t.usize();
    let g_nnz = t.usize();
    t.tag("h");
    let h = t.floats(m_ineq);
    let g = t.triplets(g_nnz);
    t.tag("lb");
    let lb = t.floats(n).into_iter().map(debound).collect();
    t.tag("ub");
    let ub = t.floats(n).into_iter().map(debound).collect();
    QpProblem {
        n,
        p_lower,
        c,
        a,
        b,
        g,
        h,
        lb,
        ub,
    }
}

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: homotopy_sweep <file.qp> on|off");
        std::process::exit(2);
    }
    let use_homotopy = match args[2].as_str() {
        "on" => true,
        "off" => false,
        other => panic!("arm must be `on` or `off`, got `{other}`"),
    };

    let text = std::fs::read_to_string(&args[1]).expect("cannot read problem file");
    let prob = parse(&text);
    let n = prob.n;
    let m_eq = prob.m_eq();
    let m_ineq = prob.m_ineq();

    // The engine overrides are otherwise the shipped defaults: the point of the
    // sweep is the difference this one option makes to the driver as configured.
    let engine = ActiveSetOverrides {
        use_homotopy: Some(use_homotopy),
        ..Default::default()
    };
    let started = Instant::now();
    let sol = solve_qp_active_set(&prob, &QpOptions::default(), &engine, &mut backend);
    let elapsed = started.elapsed().as_secs_f64();

    let status = match sol.status {
        QpStatus::Optimal => "Optimal",
        QpStatus::OptimalInaccurate => "OptimalInaccurate",
        QpStatus::PrimalInfeasible => "PrimalInfeasible",
        QpStatus::DualInfeasible => "DualInfeasible",
        QpStatus::IterationLimit => "IterationLimit",
        QpStatus::TimeLimit => "TimeLimit",
        QpStatus::NumericalFailure => "NumericalFailure",
    };
    println!(
        "{{\"status\": \"{status}\", \"obj\": {}, \"n\": {n}, \"m_eq\": {m_eq}, \
         \"m_ineq\": {m_ineq}, \"iters\": {}, \"time\": {elapsed}}}",
        sol.obj, sol.iters
    );
}

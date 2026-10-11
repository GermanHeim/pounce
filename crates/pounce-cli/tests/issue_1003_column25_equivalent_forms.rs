//! gh#1003 — two algebraically equivalent models take the same perturbation.
//!
//! Ignored by default: the models are 11 MB each and are not committed. Get
//! them from the gh#999 attachment
//! (<https://github.com/user-attachments/files/33173555/column25.zip>), unzip,
//! and run
//!
//!   POUNCE_COLUMN25_DIR=/path/to/unzipped \
//!     cargo test -p pounce-cli --release --test issue_1003_column25_equivalent_forms -- --ignored
//!
//! `column25_integral.nl` and `column25_derivative.nl` are one collocation of a
//! distillation model; the collocation rows of one are a fixed invertible
//! recombination of the other's. Diagonal equilibration does not undo that
//! recombination, so the two KKT matrices have different smallest pivots, and
//! the gh#540 inertia trust check — which read any sub-floor pivot as making
//! the count noise — sent the integral form to `δ_c` at iteration 37 and the
//! derivative form to `δ_x`. The integral run then left the shared path (88
//! iterations against 95).
//!
//! Both runs reach `inertia excess trusted despite sub-floor pivots` (5 and 3
//! times), which is the branch gh#1003 added and the only one the fixture
//! corpus never reaches — every corpus factor with a sub-floor pivot is a
//! deficit, a flagged factor, or an isolated cluster. The unit tests in
//! `pounce-feral` (`a_graded_excess_count_is_trusted`) pin the rule on a hand
//! built factor; this pins the consequence on the model that found it.

use std::path::PathBuf;
use std::process::Command;

struct Run {
    stdout: String,
    stderr: String,
}

fn run(dir: &str, name: &str) -> Run {
    let nl = PathBuf::from(dir).join(name);
    assert!(nl.exists(), "{} not found", nl.display());
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg("--no-options-file")
        .arg(&nl)
        .env("RUST_LOG", "pounce::linsol=debug")
        .output()
        .expect("run pounce");
    Run {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn field<'a>(out: &'a str, prefix: &str) -> &'a str {
    out.lines()
        .find_map(|l| l.strip_prefix(prefix))
        .unwrap_or_else(|| panic!("no `{prefix}` line:\n{out}"))
        .trim()
}

#[test]
#[ignore = "needs POUNCE_COLUMN25_DIR; see module docs"]
fn equivalent_column25_forms_take_the_same_path() {
    let Ok(dir) = std::env::var("POUNCE_COLUMN25_DIR") else {
        panic!("set POUNCE_COLUMN25_DIR to the unzipped gh#999 attachment");
    };
    let int = run(&dir, "column25_integral.nl");
    let der = run(&dir, "column25_derivative.nl");

    for (name, r) in [("integral", &int), ("derivative", &der)] {
        assert_eq!(
            field(&r.stdout, "EXIT:"),
            "Optimal Solution Found.",
            "{name}:\n{}",
            r.stdout
        );
        assert!(
            !r.stderr.contains("inertia untrustworthy"),
            "{name}: a sub-floor pivot rerouted a trusted count to δ_c"
        );
        assert!(
            r.stderr
                .contains("inertia excess trusted despite sub-floor pivots"),
            "{name}: no longer reaches the gh#1003 branch, so this pins nothing"
        );
    }

    assert_eq!(
        field(&int.stdout, "Number of Iterations....:"),
        field(&der.stdout, "Number of Iterations....:"),
        "the two forms left each other's path"
    );
    let obj = |r: &Run| -> f64 {
        field(&r.stdout, "Objective...............:")
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap()
    };
    let (oi, od) = (obj(&int), obj(&der));
    assert!(
        ((oi - od) / od).abs() < 1e-10,
        "objectives differ: {oi} vs {od}"
    );
    // Ipopt 3.13.2 with MA57 from the same start: 97 iterations, 208.81655.
    assert!((od - 208.81655256813).abs() < 1e-6, "objective {od}");
}

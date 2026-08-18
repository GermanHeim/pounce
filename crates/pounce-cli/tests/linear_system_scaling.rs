//! `linear_system_scaling` values must change the solve, not just parse
//! (#677).
//!
//! `slack-based` was registered — so `OptionsList` accepted it — but the
//! parser routed it to `LinearSystemScalingChoice::None` through a
//! catch-all arm. `mc19` reached the same fallback through a named arm
//! that warns; `slack-based` warned nothing, while a comment directly
//! above the match claimed both did. Setting it was indistinguishable
//! from not setting it.
//!
//! That is not an obscure value: it is what Ipopt's recommended
//! configuration for large collocation NLPs uses, and it appeared in a
//! user's golden configuration — validated across 25 models — reported
//! on #677. The people most likely to set it were the least likely to
//! find out it did nothing.
//!
//! So the test that matters is not "is it accepted" but "does the solve
//! move". #551 puts it directly: "A read site that parses a value and
//! discards it is the same silent no-op this whole line of work exists
//! to kill, and it is indistinguishable from a real fix by inspection.
//! That test is the deliverable, not the line that reads the field."

use std::path::PathBuf;
use std::process::Command;

/// Solve a fixture and return `(status, iteration_count)`.
///
/// `linear_scaling_on_demand=no` is essential: the default is `yes`,
/// which computes scaling only on a factorization that already looks
/// troubled. On a fixture that solves cleanly, every scaling choice is
/// then identical — which is exactly how a test could "pass" against an
/// unimplemented option. Forcing scaling on every factorization is what
/// makes the comparison meaningful.
fn solve(fixture: &str, tag: &str, scaling: &str) -> (String, u64) {
    let dir = std::env::temp_dir().join(format!("pounce_linscale_{tag}"));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let nl = dir.join(fixture);
    let mut src = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    src.push("tests/fixtures");
    src.push(fixture);
    std::fs::copy(&src, &nl).expect("copy fixture");
    let json = dir.join("out.json");

    let out = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")))
        .arg(&nl)
        .arg(dir.join("out.sol"))
        .arg("--json-output")
        .arg(&json)
        .arg(format!("linear_system_scaling={scaling}"))
        .arg("linear_scaling_on_demand=no")
        .arg("print_level=0")
        .output()
        .expect("run pounce");
    assert!(
        json.exists(),
        "no JSON for linear_system_scaling={scaling}; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr),
    );
    let text = std::fs::read_to_string(&json).expect("read json");
    let _ = std::fs::remove_dir_all(&dir);

    // Small hand parse rather than pulling serde into this test.
    let field = |key: &str| -> String {
        let at = text
            .find(&format!("\"{key}\""))
            .unwrap_or_else(|| panic!("no `{key}` in JSON:\n{text}"));
        let rest = &text[at + key.len() + 2..];
        let colon = rest.find(':').expect("colon");
        let tail = rest[colon + 1..].trim_start();
        if let Some(stripped) = tail.strip_prefix('"') {
            stripped[..stripped.find('"').expect("close quote")].to_string()
        } else {
            tail.chars()
                .take_while(|c| c.is_ascii_digit() || *c == '-' || *c == '.' || *c == 'e')
                .collect()
        }
    };
    let iters: u64 = field("iteration_count").parse().expect("iteration_count");
    (field("status"), iters)
}

#[test]
fn slack_based_scaling_changes_the_solve() {
    // cresc4 is sensitive to augmented-system scaling — `ruiz` already
    // moved it before this option existed, which is what makes it a
    // usable probe. A fixture where every choice agrees (airport, csfi2)
    // cannot tell an implemented scaling from an ignored one.
    let (none_status, none_iters) = solve("cresc4.nl", "none", "none");
    let (slack_status, slack_iters) = solve("cresc4.nl", "slack", "slack-based");
    let (ruiz_status, ruiz_iters) = solve("cresc4.nl", "ruiz", "ruiz");

    // All three must still reach the same answer; scaling is a
    // conditioning choice, not a different problem.
    for (name, status) in [
        ("none", &none_status),
        ("slack-based", &slack_status),
        ("ruiz", &ruiz_status),
    ] {
        assert_eq!(
            status, "SolveSucceeded",
            "linear_system_scaling={name} did not solve cresc4",
        );
    }

    // The actual regression guard. Before #677 this assertion failed:
    // `slack-based` produced byte-identical output to `none`, because it
    // *was* `none`.
    assert_ne!(
        slack_iters, none_iters,
        "linear_system_scaling=slack-based took the same {none_iters} iterations as \
         `none` — the option is parsed but not reaching the linear solver",
    );

    // And it is its own method, not an alias for the one that already
    // worked.
    assert_ne!(
        slack_iters, ruiz_iters,
        "slack-based and ruiz agree at {ruiz_iters} iterations, which is suspicious \
         enough to check the wiring before trusting it",
    );
}

/// `mc19` is still unimplemented and still falls back. Pinned so that if
/// someone implements it, this test fails and reminds them to say so —
/// and so the fallback stays deliberate rather than becoming another
/// silent one.
#[test]
fn mc19_still_falls_back_to_no_scaling() {
    let (status, mc19_iters) = solve("cresc4.nl", "mc19", "mc19");
    let (_, none_iters) = solve("cresc4.nl", "mc19none", "none");
    assert_eq!(status, "SolveSucceeded");
    assert_eq!(
        mc19_iters, none_iters,
        "mc19 now differs from `none` — if it was implemented, update this test, \
         the CHANGELOG, and `docs/src/options.md`",
    );
}

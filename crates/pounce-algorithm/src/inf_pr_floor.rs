//! How long a solve has sat at a constraint violation it could not get
//! below — the evidence a locally-infeasible verdict actually rests on.
//!
//! The five *reconstructed* locally-infeasible gates in restoration
//! (`pounce_restoration::resto_inner_solver::run_inner_resto`) all claim
//! "the solve stalled at a violation it could not improve on", and gh#661
//! is what happens when nothing measures that claim: they test only that
//! the recovered violation is *large*, which a diverging restoration
//! satisfies ever more emphatically the worse it gets.
//!
//! The gh#661 divergence guard withholds those gates' verdict when the
//! restoration sub-solve ends far worse than the violation it was
//! *entered* at. That guard needs an exemption for one shape of run: a
//! solve that provably could not get below some floor, and whose
//! restoration then blew up over a handful of iterations at the end. The
//! blow-up is the tail of that trajectory, not a description of it.
//!
//! gh#664 keyed that exemption on `inner_iter_count >= 1000`, which is a
//! proxy for "it ran out of room" and a doubly loose one. A count is not a
//! stall test — that substitution is the same class of error gh#661 fixed,
//! where a *size* stood in for a stall test. And the counter it read is
//! not what its comments claimed: the inner IPM's `iter_count` is seeded
//! from the outer's (`IpRestoMinC_1Nrm.cpp:181`), so `1019` on
//! `issue_508_infeasible_gap_1em2` is 1015 outer iterations plus a
//! *four*-iteration sub-solve, not a sub-solve that ran a thousand times.
//!
//! [`InfPrFloor`] measures the property directly, at the scope where the
//! long trajectory actually lives: the outer solve. It watches the
//! original NLP's scaled primal infeasibility at each outer iterate and
//! counts how many of them sat within [`FLOOR_BAND`] of the floor the
//! count is being measured against. A solve still finding its way down
//! keeps clearing the band and restarting that count; one that is out of
//! room accumulates it.
//!
//! Two choices in that sentence are load-bearing rather than
//! simplifications, and both were arrived at by measuring:
//!
//! *Cumulative, not consecutive.* A real trajectory pinned at a floor
//! does not sit there quietly. `issue_508_infeasible_gap_1em2` returns to
//! `1.0e-2` over and over across 1016 outer iterations while excursions to
//! `9.56e1` break every run in between; simulated over its printed
//! iteration trace, its *longest consecutive* stay is 39 — against 19 for
//! `pooling_rt2stp`, which is feasible and must not be exempted. The
//! consecutive measure does not separate them at all. Time spent at the
//! floor does, by two orders of magnitude: 943 against 7.
//!
//! *Pinned reference, not running minimum.* Measuring the band against
//! the running minimum lets it chase the iterates, so a solve creeping
//! downward by 0.9x per iteration is forever within a decade of its own
//! previous best. Under that reading a 2000-iteration grind — which
//! reduces the violation by eighty-eight orders of magnitude, i.e. is
//! working — accumulates all 2000 and buys the exemption. Pinned, it
//! accumulates one decade's worth and resets.

use pounce_common::types::{Index, Number};

/// How far above the best violation seen so far an iterate may sit and
/// still count as sitting *at* that floor.
///
/// An order of magnitude, matching
/// `pounce_restoration::resto_inner_solver::RESTO_DIVERGENCE_HEADROOM`
/// and for the same reason: a floor a solve keeps returning to is not a
/// fixed point, and a tighter band reads ordinary wander as the solve
/// having left it. The trajectories this must tell apart move by four to
/// twelve orders of magnitude, so the band has room to be generous.
const FLOOR_BAND: Number = 10.0;

/// Running evidence, over one solve, that it found a floor on the
/// constraint violation and could not get below it.
///
/// Fed once per outer iteration from the value already computed for the
/// `inf_pr` column, so it costs no function evaluations. The restoration
/// sub-IPM has its own [`InfPrFloor`] on its own `IpoptData`, measuring
/// its own (restoration) NLP; the guard reads the outer one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InfPrFloor {
    /// The violation the current count is being measured against.
    ///
    /// Deliberately *not* the running minimum. Measuring the band
    /// against the running minimum lets it chase the iterates: a solve
    /// creeping down by 0.9x per iteration is always within a decade of
    /// its own previous best, so a 2000-iteration grind that reduced the
    /// violation by eighty-eight orders of magnitude would have read as
    /// 2000 iterations sitting at a floor. Pinned to where the count
    /// started, the same grind reads 20 — one decade's worth — and then
    /// resets, which is the honest description of a solve that is still
    /// getting somewhere.
    floor: Number,
    /// How many iterates sat within [`FLOOR_BAND`] of `floor`.
    ///
    /// Reset to 1 — not 0 — whenever the solve gets a full band below
    /// `floor`, because the iterate that got there is itself the first
    /// one sitting at the new floor.
    iters_at_floor: Index,
    /// How many iterates were observed at all. Diagnostic: it separates
    /// "this solve demonstrated no floor" from "nothing was sampled",
    /// which read identically from [`Self::iters_at_floor`] alone.
    samples: Index,
}

impl Default for InfPrFloor {
    fn default() -> Self {
        Self {
            floor: Number::INFINITY,
            iters_at_floor: 0,
            samples: 0,
        }
    }
}

impl InfPrFloor {
    /// Record the scaled primal infeasibility at one iterate.
    pub fn observe(&mut self, inf_pr: Number) {
        self.samples += 1;

        // A non-finite iterate is not sitting at anything, and must not
        // be allowed to move `floor` — a `NaN` reaching it would poison
        // every later comparison.
        if !inf_pr.is_finite() {
            return;
        }

        if inf_pr * FLOOR_BAND < self.floor {
            // A full band below the reference: whatever was being
            // measured was not the floor. The count restarts from this
            // iterate, which is the first one at the new one.
            self.floor = inf_pr;
            self.iters_at_floor = 1;
        } else if inf_pr <= self.floor * FLOOR_BAND {
            // Within the band of the floor already found. Note `floor`
            // is left alone: a gain that does not clear the band is
            // wander, not progress, and letting it ratchet the reference
            // down is exactly the chase this field exists to avoid.
            self.iters_at_floor += 1;
        }
        // Otherwise: above the band. Not evidence of a floor, but not
        // evidence against one either — the solve is free to come back,
        // and on the trajectories this exists for it repeatedly does
        // (`issue_508_infeasible_gap_1em2` excurses to `9.56e1` between
        // returns to `1.0e-2`). So the count is held, not reset.
    }

    /// How many iterates sat within an order of magnitude of the floor
    /// this solve settled at. `0` when nothing was observed.
    pub fn iters_at_floor(&self) -> Index {
        self.iters_at_floor
    }

    /// The violation the count is measured against, or `INFINITY` if
    /// nothing was observed. Diagnostic only.
    pub fn floor(&self) -> Number {
        self.floor
    }

    /// How many iterates were observed. Diagnostic only.
    pub fn samples(&self) -> Index {
        self.samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(values: &[Number]) -> InfPrFloor {
        let mut f = InfPrFloor::default();
        for &v in values {
            f.observe(v);
        }
        f
    }

    /// Nothing observed is not evidence of a floor, and must not read as
    /// one: the exemption is granted on a *count*, so an unfed tracker
    /// has to be indistinguishable from a solve that demonstrated
    /// nothing.
    #[test]
    fn an_unobserved_solve_demonstrates_nothing() {
        let f = InfPrFloor::default();
        assert_eq!(f.iters_at_floor(), 0);
        assert_eq!(f.samples(), 0);
        assert!(f.floor().is_infinite());
    }

    /// The shape the exemption exists for, and why the measure is
    /// cumulative rather than a longest-consecutive-run. Measured over
    /// the real outer trajectories, longest-consecutive-stay gives 39 for
    /// `issue_508_infeasible_gap_1em2` against 19 for the *feasible*
    /// `pooling_rt2stp` — no separation at all, because a trajectory
    /// pinned at a floor does not sit there quietly. This fixture bottoms
    /// out at `1.0e-2` early and returns to it across 1016 outer
    /// iterations while excursions to `9.56e1` break every run in
    /// between. Held-not-reset, time at the floor accumulates anyway.
    #[test]
    fn a_floor_returned_to_across_excursions_still_accumulates() {
        let mut vals = vec![2.65e-1, 6.89e-2, 2.01e-2, 1.00e-2];
        for _ in 0..500 {
            vals.push(1.04e-2);
            vals.push(9.56e1); // excursion, well outside the band
        }
        let f = feed(&vals);
        // `2.01e-2` set the reference (a full decade under `2.65e-1`);
        // `1.00e-2` and every return to `1.04e-2` sit inside its band.
        assert_eq!(f.floor(), 2.01e-2);
        assert_eq!(f.iters_at_floor(), 502);
        assert_eq!(f.samples(), 1004);
    }

    /// The counterpart: `pooling_rt2stp` is feasible and must not be
    /// exempted. Its outer solve is 20 iterations long, so even if every
    /// one of them sat at the floor it comes nowhere near evidence of
    /// being out of room. Short solves cannot buy the exemption.
    #[test]
    fn a_short_solve_cannot_accumulate_a_long_floor() {
        let vals: Vec<Number> = (0..20).map(|k| 2.72e-1 * (1.0 + k as Number)).collect();
        assert!(feed(&vals).iters_at_floor() <= 20);
    }

    /// The hole that killed measuring the band against the running
    /// *minimum*: a solve creeping down by 0.9x per iteration is forever
    /// within a decade of its own previous best. Under that reading a
    /// 2000-iteration grind — which reduces the violation by eighty-eight
    /// orders of magnitude, i.e. is working perfectly — accumulated all
    /// 2000 and would have been handed the exemption. Pinned to the
    /// reference it accumulates one decade's worth and resets.
    #[test]
    fn a_slow_steady_grind_downwards_is_not_a_floor() {
        let vals: Vec<Number> = (0..2000).map(|k| 1.0e3 * 0.9_f64.powi(k)).collect();
        let f = feed(&vals);
        assert_eq!(f.samples(), 2000);
        assert!(
            f.iters_at_floor() <= 25,
            "a solve still descending must not accumulate a floor: {}",
            f.iters_at_floor()
        );
    }

    /// Monotone divergence — the failure gh#661 is about — accumulates
    /// exactly one: the opening iterate, trivially the best seen. Nothing
    /// after it is ever within a decade of that, however long it runs.
    #[test]
    fn monotone_divergence_accumulates_nothing_however_long_it_runs() {
        let vals: Vec<Number> = (0..5000).map(|k| 1.0e-3 * 100.0_f64.powi(k)).collect();
        let f = feed(&vals);
        assert_eq!(f.samples(), 5000);
        assert_eq!(f.iters_at_floor(), 1);
    }

    /// Wander inside the band counts and leaves the reference alone; a
    /// drop that clears the band is a new floor and restarts the count at
    /// the iterate that found it.
    #[test]
    fn the_band_holds_wander_and_a_real_drop_restarts_the_count() {
        let f = feed(&[1.0e6, 9.0e5, 3.0e6, 8.0e5]);
        assert_eq!(f.floor(), 1.0e6);
        assert_eq!(f.iters_at_floor(), 4);

        let g = feed(&[1.0e6, 1.0e6, 1.0e6, 1.0e1]);
        assert_eq!(g.floor(), 1.0e1);
        assert_eq!(g.iters_at_floor(), 1);
    }

    /// Non-finite iterates must neither count nor corrupt the reference.
    /// A diverging solve reaches `inf` and `NaN` routinely.
    #[test]
    fn non_finite_iterates_neither_count_nor_corrupt_the_floor() {
        let f = feed(&[5.0, Number::NAN, Number::INFINITY, 5.0]);
        assert_eq!(f.floor(), 5.0);
        assert_eq!(f.iters_at_floor(), 2);
        assert_eq!(f.samples(), 4);
    }
}

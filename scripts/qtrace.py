#!/usr/bin/env python3
"""Q0(b): is a long POUNCE solve stalling, or converging steadily but slowly?

The two answers imply very different work. A stall -- mu frozen while step sizes
collapse -- is what Gondzio multiple centrality correctors fix, and the design
note estimates 332 iterations -> 40-70. Steady slow convergence is not a
corrector problem at all; it puts the fix in the barrier update rule and the
same phase is worth only 1.3-1.5x. Reading the trace is the only way to tell.
"""
import json, sys, statistics

def analyze(path, name):
    d = json.load(open(path))
    it = d.get("iterations") or []
    st = d.get("statistics", {})
    if not it:
        print(f"{name}: no iteration history"); return

    mu = [r["mu"] for r in it]
    ap = [r["alpha_primal"] for r in it]
    ad = [r["alpha_dual"] for r in it]
    ls = [r.get("ls_trials", 0) for r in it]
    reg = [r.get("regularization", 0.0) for r in it]
    pr = [r["inf_pr"] for r in it]
    du = [r["inf_du"] for r in it]

    # per-step mu contraction, skipping the iterations where mu is held fixed
    ratios = [mu[i+1]/mu[i] for i in range(len(mu)-1) if mu[i] > 0]
    frozen = sum(1 for r in ratios if r > 0.999)
    med = statistics.median(ratios) if ratios else float("nan")

    tiny_p = sum(1 for a in ap if a < 1e-4)
    small_p = sum(1 for a in ap if a < 1e-2)
    regged = sum(1 for r in reg if r > 0)

    print(f"=== {name} ===")
    print(f"  iterations              {len(it)}   (status {d.get('solution',{}).get('status')})")
    print(f"  mu   {mu[0]:.3e} -> {mu[-1]:.3e}   median per-step ratio {med:.4f}")
    print(f"  mu frozen (ratio>0.999) {frozen}/{len(ratios)}  = {100*frozen/max(1,len(ratios)):.0f}% of steps")
    print(f"  alpha_primal  median {statistics.median(ap):.3e}   min {min(ap):.2e}")
    print(f"    < 1e-2 on {small_p}/{len(ap)} steps ({100*small_p/len(ap):.0f}%)"
          f"   < 1e-4 on {tiny_p} ({100*tiny_p/len(ap):.0f}%)")
    print(f"  alpha_dual    median {statistics.median(ad):.3e}")
    print(f"  line-search trials      total {sum(ls)}  max {max(ls)}")
    print(f"  regularized iterations  {regged}/{len(it)} ({100*regged/len(it):.0f}%)")
    print(f"  inf_pr {pr[0]:.2e} -> {pr[-1]:.2e}    inf_du {du[0]:.2e} -> {du[-1]:.2e}")
    print(f"  restoration entries     {st.get('restoration_entries', 'n/a')}")

    # Where does the time actually go in iteration space? Split into thirds and
    # report how much mu each third buys -- a stall shows up as a middle or final
    # third that costs many iterations and moves mu almost not at all.
    n = len(mu); t = max(1, n // 3)
    for k, (lo, hi) in enumerate([(0, t), (t, 2*t), (2*t, n)]):
        seg = mu[lo:hi]
        if len(seg) < 2: continue
        decades = 0.0
        if seg[0] > 0 and seg[-1] > 0:
            import math; decades = math.log10(seg[0]/seg[-1])
        print(f"  third {k+1}: iters {lo}-{hi-1} ({hi-lo})  mu drops {decades:.2f} decades"
              f"  ({decades/(hi-lo):.4f} decades/iter)")

    verdict_stall = (frozen > 0.25*len(ratios)) or (small_p > 0.25*len(ap))
    print(f"  --> {'STALL signature' if verdict_stall else 'steady convergence'}")
    print()

for path, name in zip(sys.argv[1::2], sys.argv[2::2]):
    analyze(path, name)

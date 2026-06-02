"""Reaction barriers on a molecular potential energy surface.

A chemical reaction is a walk on a potential energy surface (PES) from one
minimum (a stable molecular state) to another, over a mountain pass. That
pass is an **index-1 saddle point** — the *transition state* — and the
**reaction barrier** is its height above the reactant:

    barrier(reactant -> product) = E(transition state) - E(reactant).

`pounce.reaction_network` does the whole job in one call: it finds the
minima (stable states) with ``find_minima``, the index-1 saddles
(transition states) with ``find_saddles``, then connects each saddle to the
two minima it joins — by sliding downhill from the pass along its unstable
direction — and tabulates the barrier heights.

This example uses the **Müller-Brown potential**, the standard 2-D benchmark
PES for reaction-path methods, which has three minima and two transition
states (Müller & Brown, *Theoret. Chim. Acta* **53**, 75-93 (1979),
doi:10.1007/BF00547608).

Run:  python reaction_barrier.py
"""

import os

os.environ.setdefault("RUST_LOG", "off")  # quiet the harmless solve log

import numpy as np

import pounce

# --- Müller-Brown potential and its analytic derivatives -------------------
_A = np.array([-200.0, -100.0, -170.0, 15.0])
_a = np.array([-1.0, -1.0, -6.5, 0.7])
_b = np.array([0.0, 0.0, 11.0, 0.6])
_c = np.array([-10.0, -10.0, -6.5, 0.7])
_x0 = np.array([1.0, 0.0, -0.5, -1.0])
_y0 = np.array([0.0, 0.5, 1.5, 1.0])


def V(z):
    x, y = z
    dx, dy = x - _x0, y - _y0
    return float(np.sum(_A * np.exp(_a * dx**2 + _b * dx * dy + _c * dy**2)))


def grad(z):
    x, y = z
    dx, dy = x - _x0, y - _y0
    e = _A * np.exp(_a * dx**2 + _b * dx * dy + _c * dy**2)
    return np.array([np.sum(e * (2 * _a * dx + _b * dy)),
                     np.sum(e * (_b * dx + 2 * _c * dy))])


def hess(z):
    x, y = z
    dx, dy = x - _x0, y - _y0
    e = _A * np.exp(_a * dx**2 + _b * dx * dy + _c * dy**2)
    px, py = 2 * _a * dx + _b * dy, _b * dx + 2 * _c * dy
    hxx = np.sum(e * (px * px + 2 * _a))
    hyy = np.sum(e * (py * py + 2 * _c))
    hxy = np.sum(e * (px * py + _b))
    return np.array([[hxx, hxy], [hxy, hyy]])


BOUNDS = [(-1.5, 1.2), (-0.5, 2.2)]


def main():
    # The whole reaction network in one call. Bump widths and heights are
    # "auto" (per-dimension widths from the bounds; amplitude from the local
    # curvature), so no manual sigma/amplitude tuning is needed even though
    # the energy scale here is ~150.
    net = pounce.reaction_network(
        V, [-0.5, 1.4], grad=grad, hess=hess, bounds=BOUNDS,
        n_states=3, n_transition_states=2, dedup=1e-2, seed=0,
        saddle_kw={"max_step": 0.05},
        options={"print_level": 0, "tol": 1e-8},
    )
    print(net.summary())

    # The shared state is the reaction intermediate.
    counts = {}
    for c in net.connections:
        for k in c.minima:
            counts[k] = counts.get(k, 0) + 1
    hub = max(counts, key=counts.get)
    ends = [k for k in counts if k != hub]
    print(f"\nReaction path: state {ends[0]} <=> state {hub} (intermediate) "
          f"<=> state {ends[1]}")

    _maybe_plot(net, hub, ends)


def _maybe_plot(net, hub, ends):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    # --- (1) PES map with states, transition states, and MEPs ---
    xs = np.linspace(-1.5, 1.2, 400)
    ys = np.linspace(-0.5, 2.2, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for k in range(4):
        Z += _A[k] * np.exp(_a[k] * (X - _x0[k])**2
                            + _b[k] * (X - _x0[k]) * (Y - _y0[k])
                            + _c[k] * (Y - _y0[k])**2)
    Z = np.clip(Z, -150, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.contourf(X, Y, Z, levels=40, cmap="viridis")
    ax1.contour(X, Y, Z, levels=20, colors="k", alpha=0.15, linewidths=0.5)
    for c in net.connections:
        ax1.plot(c.path[:, 0], c.path[:, 1], "w-", lw=2)
        ax1.scatter(*c.ts.x, c="red", marker="^", s=160, edgecolors="k", zorder=6)
    for idx, m in enumerate(net.minima):
        ax1.scatter(*m.x, c="white", marker="o", s=140, edgecolors="k", zorder=6)
        ax1.annotate(f"{idx}", m.x, color="k", fontweight="bold",
                     ha="center", va="center", zorder=7)
    ax1.set_title("Müller-Brown PES: o states, ^ transition states, — MEP")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")

    # --- (2) reaction-coordinate energy profile: end0 -> hub -> end1 ---
    full = np.vstack([net.path_between(ends[0], hub),
                      net.path_between(hub, ends[1])[1:]])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(full, axis=0), axis=1))])
    energies = np.array([V(p) for p in full])
    ax2.plot(s, energies, "b-", lw=2)
    for label, e in [(f"state {ends[0]}", net.minima[ends[0]].f),
                     (f"state {hub}", net.minima[hub].f),
                     (f"state {ends[1]}", net.minima[ends[1]].f)]:
        k = int(np.argmin(np.abs(energies - e)))
        ax2.scatter(s[k], energies[k], c="white", edgecolors="b", s=80, zorder=5)
        ax2.annotate(label, (s[k], energies[k]), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=8)
    for c in net.connections:
        k = int(np.argmin(np.abs(energies - c.ts.f)))
        ax2.scatter(s[k], energies[k], c="red", marker="^", s=90, zorder=5)
    ax2.set_title("Reaction-coordinate energy profile")
    ax2.set_xlabel("reaction coordinate (arc length)")
    ax2.set_ylabel("energy")
    ax2.grid(alpha=0.3)

    out = os.path.join(os.path.dirname(__file__), "reaction_barrier.png")
    plt.tight_layout()
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()

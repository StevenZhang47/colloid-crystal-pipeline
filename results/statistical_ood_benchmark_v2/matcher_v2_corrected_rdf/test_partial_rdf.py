#!/usr/bin/env python3
"""
REGRESSION TESTS for the corrected partial RDF. All must pass before any library is
rebuilt or any matcher number is quoted.

  1. ANALYTIC SHELLS   ideal CsCl must show peaks at r/nn = 1.000, 1.155, 1.633, 1.915
                       (8 Cl at a*sqrt3/2; 6 Cs at a; 12 Cs at a*sqrt2; 24 Cl at a*sqrt11/2)
  2. DENSITY NORM      g(r) -> 1 at large r (true RDF, not counts/shell-volume)
  3. SIZE INVARIANCE   same crystal at several supercell sizes -> same curves, tiny distance
  4. SHAPE INVARIANCE  cube vs slab vs cropped sphere -> local shells preserved
                       (this is the one Nicole's finite-domain-in-a-gas-box data needs)
  5. SPECIES CHANNELS  LL / LS / SS normalised separately and correct; typeid 0 = LARGE
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).resolve().parent))
from partial_rdf import partial_rdf  # noqa

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def cscl_lattice(reps, a=1.0):
    """Ideal CsCl. typeid 0 = LARGE (Cl, corner), 1 = SMALL (Cs, body centre)."""
    pts, tid = [], []
    for i in range(reps):
        for j in range(reps):
            for k in range(reps):
                pts.append([i, j, k]); tid.append(0)                     # LARGE
                pts.append([i + .5, j + .5, k + .5]); tid.append(1)      # SMALL
    p = np.array(pts, float) * a
    return p, np.array(tid, int)


def nn_of(p):
    d, _ = cKDTree(p).query(p, k=2)
    return float(np.median(d[:, 1]))


def emd(x, a, b):
    a = np.clip(a, 0, None); b = np.clip(b, 0, None)
    if a.sum() <= 0 or b.sum() <= 0:
        return np.inf
    return float(wasserstein_distance(x, x, a / a.sum(), b / b.sum()))


def peaks_of(r, g, n=4, min_h=0.35):
    idx = [i for i in range(1, len(g) - 1)
           if g[i] > g[i - 1] and g[i] >= g[i + 1] and g[i] > min_h * g.max()]
    idx.sort(key=lambda i: -g[i])
    return sorted(r[i] for i in idx[:n])


def main():
    print("=" * 74)
    print("TEST 1 — ANALYTIC SHELLS (ideal CsCl)")
    p, t = cscl_lattice(12)
    nn = nn_of(p)
    dom = np.ones(len(p), bool)
    r, g = partial_rdf(p, t, dom, nn)
    got = peaks_of(r, g["total"], n=4)
    expect = [1.000, 1.155, 1.633, 1.915]
    tol = 2.5 * (r[1] - r[0])
    ok = all(any(abs(x - e) < tol for x in got) for e in expect[:3])
    check("CsCl shells at 1.000 / 1.155 / 1.633 (+1.915)", ok,
          f"found {[round(x,3) for x in got]}  expect {expect}  tol {tol:.3f}")
    check("first peak at r/nn ~ 1.0 (NOT the v1 bug at ~11.8)",
          abs(r[np.argmax(g['total'])] - 1.0) < 0.10,
          f"argmax at r/nn = {r[np.argmax(g['total'])]:.3f}")

    print("\nTEST 2 — DENSITY NORMALISATION (true RDF, g -> 1 at large r)")
    tail = g["total"][r > 3.0]
    check("g_total -> 1 at large r", 0.6 < tail.mean() < 1.5, f"tail mean = {tail.mean():.3f}")
    for k in ("LL", "LS", "SS"):
        tk = g[k][r > 3.0]
        check(f"g_{k} -> 1 at large r", 0.5 < tk.mean() < 1.7, f"tail mean = {tk.mean():.3f}")

    print("\nTEST 3 — SIZE INVARIANCE (same crystal, different supercells)")
    curves = {}
    for reps in (6, 9, 12, 15):
        pp, tt = cscl_lattice(reps)
        rr, gg = partial_rdf(pp, tt, np.ones(len(pp), bool), nn_of(pp))
        curves[reps] = (rr, gg)
    base_r, base_g = curves[12]
    for reps in (6, 9, 15):
        rr, gg = curves[reps]
        d = np.mean([emd(base_r, gg[k], base_g[k]) for k in ("LL", "LS", "SS")])
        n_ratio = (reps / 12.0) ** 3
        check(f"reps={reps:2d} ({n_ratio:.2f}x particles): partial-RDF distance to reps=12",
              d < 0.05, f"EMD = {d:.4f}  (v1 would explode here)")

    print("\nTEST 4 — SHAPE INVARIANCE (cube vs slab vs cropped sphere)")
    pc, tc = cscl_lattice(14)
    c = pc.mean(0)
    shapes = {}
    shapes["cube"] = (pc, tc)
    m_slab = np.abs(pc[:, 2] - c[2]) < 2.0                    # thin slab
    shapes["slab"] = (pc[m_slab], tc[m_slab])
    rad = np.linalg.norm(pc - c, axis=1)
    m_sph = rad < 5.0                                          # cropped sphere
    shapes["sphere"] = (pc[m_sph], tc[m_sph])
    ref_r, ref_g = partial_rdf(*shapes["cube"][:1], shapes["cube"][1],
                               np.ones(len(shapes["cube"][0]), bool), nn_of(shapes["cube"][0]))
    for name in ("slab", "sphere"):
        pp, tt = shapes[name]
        rr, gg = partial_rdf(pp, tt, np.ones(len(pp), bool), nn_of(pp))
        # compare only the LOCAL shell region (that is what the matcher uses)
        loc = ref_r <= 2.5
        d = np.mean([emd(ref_r[loc], gg[k][loc], ref_g[k][loc]) for k in ("LL", "LS", "SS")])
        check(f"{name:6s} (n={len(pp):5d}) local shells preserved vs cube",
              d < 0.08, f"EMD(local) = {d:.4f}")
        pk = peaks_of(rr, gg["total"], n=2)
        check(f"{name:6s} first shell still at r/nn ~ 1.0", any(abs(x - 1.0) < 0.12 for x in pk),
              f"peaks {[round(x,3) for x in pk]}")

    print("\nTEST 5 — SPECIES CHANNELS + CANONICAL CONVENTION")
    p, t = cscl_lattice(12)
    nn = nn_of(p)
    r, g = partial_rdf(p, t, np.ones(len(p), bool), nn)
    # In CsCl the FIRST neighbour shell is entirely UNLIKE (8 counter-ions at nn).
    # Like-species first appear at a = 1.1547 nn. So:
    check("g_LS has its first peak at r/nn ~ 1.0 (unlike pair = nearest neighbour)",
          abs(r[np.argmax(g['LS'])] - 1.0) < 0.10, f"argmax {r[np.argmax(g['LS'])]:.3f}")
    ll_near = g["LL"][r < 1.08].max()
    check("g_LL is ~0 below r/nn = 1.08 (no like-pairs in the first shell)",
          ll_near < 0.25, f"max g_LL below 1.08 = {ll_near:.3f}")
    check("g_LL first peak at r/nn ~ 1.155 (the lattice constant a)",
          abs(r[np.argmax(g['LL'])] - 1.1547) < 0.10, f"argmax {r[np.argmax(g['LL'])]:.3f}")
    check("g_SS first peak at r/nn ~ 1.155 as well",
          abs(r[np.argmax(g['SS'])] - 1.1547) < 0.10, f"argmax {r[np.argmax(g['SS'])]:.3f}")
    # convention: swapping typeids must swap LL and SS
    r2, g2 = partial_rdf(p, 1 - t, np.ones(len(p), bool), nn)
    check("swapping typeid swaps LL <-> SS (channels are species-resolved, not mixed)",
          np.allclose(g["LL"], g2["SS"], atol=0.05) and np.allclose(g["SS"], g2["LL"], atol=0.05))
    # asymmetric-size sanity: LARGE must be typeid 0 by construction elsewhere; assert guard fires
    try:
        partial_rdf(p, np.full(len(p), 2), np.ones(len(p), bool), nn)
        check("guard rejects non-canonical typeids", False)
    except AssertionError:
        check("guard rejects non-canonical typeids (typeid must be {0=LARGE,1=SMALL})", True)

    print("\n" + "=" * 74)
    if FAIL:
        print(f"FAILED {len(FAIL)} test(s): {FAIL}")
        sys.exit(1)
    print("ALL REGRESSION TESTS PASSED — corrected partial RDF is safe to build on.")


if __name__ == "__main__":
    main()

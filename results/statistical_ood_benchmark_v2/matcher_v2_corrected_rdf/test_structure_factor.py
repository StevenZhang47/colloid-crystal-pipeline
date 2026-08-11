#!/usr/bin/env python3
"""
Regression tests for the repaired structure factor.

These are the five tests named in SK_BUG_DIAGNOSIS_AND_REPAIR_PLAN.md section 2, plus the
direct reproduction of the bug that motivated them. They exist because the identical class
of N-dependent size artifact already slipped through once (the raw-pair-count "RDF"), and
was invisible to every internal benchmark because reference and candidate always had
comparable N. Nicole's 1004-particle domains were the first data with a different N.

    python test_structure_factor.py
"""
from __future__ import annotations

import sys

import numpy as np

from structure_factor import (fibonacci_directions, k_grid, peak_positions,
                              relative_intensities, structure_factor)

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def cscl(reps: int, jitter: float = 0.0, seed: int = 0) -> tuple[np.ndarray, float]:
    """Ideal CsCl (B2): simple-cubic large sublattice + body-centre small sublattice.
    nn (unlike pair) = a*sqrt(3)/2."""
    a = 1.0
    rng = np.random.default_rng(seed)
    c = np.array([[i, j, k] for i in range(reps) for j in range(reps) for k in range(reps)], float)
    pos = np.vstack([c, c + 0.5]) * a
    if jitter:
        pos = pos + rng.normal(scale=jitter, size=pos.shape)
    return pos, a * np.sqrt(3.0) / 2.0


def disordered(n: int, nn: float, seed: int = 0) -> np.ndarray:
    """Ideal gas in a cube at the same number density as the lattice -> no coherence."""
    rng = np.random.default_rng(seed)
    L = (n * (nn ** 3) / 0.6) ** (1.0 / 3.0)
    return rng.uniform(0, L, size=(n, 3))


def main():
    print("=" * 78)
    print("STRUCTURE FACTOR REGRESSION TESTS")
    print("=" * 78)

    # ---------------------------------------------------------------- 0. the bug itself
    print("\n0. reproduce the original bug (random particle subsampling)")
    pos, nn = cscl(15)                                   # 6750 particles
    q = (pos - pos.mean(0)) / nn
    dirs = fibonacci_directions(64)
    kv = k_grid()

    def raw_sk(p):
        return max(float(np.mean(np.abs(np.exp(1j * (p @ (k * dirs).T)).sum(0)) ** 2) / len(p))
                   for k in kv)

    full = raw_sk(q)
    sub2500 = raw_sk(q[np.random.default_rng(1).choice(len(q), 2500, replace=False)])
    sub1004 = raw_sk(q[np.random.default_rng(1).choice(len(q), 1004, replace=False)])
    print(f"      OLD max S(k): all {len(q)} -> {full:.1f} | sub 2500 -> {sub2500:.1f} "
          f"| sub 1004 -> {sub1004:.1f}")
    check("old code: subsampling changes max S(k) by >2x  (the bug is real)",
          full / max(sub1004, 1e-9) > 2.0,
          f"ratio {full / max(sub1004, 1e-9):.1f}x")

    # ------------------------------------------------- 1. same crystal at N=1004/2500/6750
    print("\n1. same CsCl lattice, three system sizes -> curves must agree")
    sizes = {}
    for reps in (8, 11, 15):                            # 1024, 2662, 6750 particles
        p, nnr = cscl(reps)
        k, s = structure_factor(p, nnr)
        sizes[len(p)] = s
        print(f"      N={len(p):5d}  max S_hat={s.max():.3f}  "
              f"peaks={np.round(peak_positions(k, s, 4), 2)}")
    ns = sorted(sizes)
    ref = sizes[ns[-1]]

    def shape_dist(a, b):                               # the matcher's own comparison
        a = np.clip(a, 0, None); b = np.clip(b, 0, None)
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(k, k, a / a.sum(), b / b.sum()))

    d_small = shape_dist(sizes[ns[0]], ref)
    d_mid = shape_dist(sizes[ns[1]], ref)
    print(f"      EMD(N={ns[0]} vs N={ns[-1]}) = {d_small:.4f}"
          f"    EMD(N={ns[1]} vs N={ns[-1]}) = {d_mid:.4f}")
    check("size invariance: EMD between sizes < 0.05", max(d_small, d_mid) < 0.05,
          f"max {max(d_small, d_mid):.4f}")

    amp = [sizes[n].max() for n in ns]
    check("amplitude no longer scales with N (max S_hat within 2x across 6.6x in N)",
          max(amp) / max(min(amp), 1e-9) < 2.0,
          f"max S_hat {np.round(amp, 3)} across N={ns}")

    # ------------------------------------------------------------ 2. no seed dependence
    print("\n2. no dependence on any random seed")
    p, nnr = cscl(11)
    a = structure_factor(p, nnr)[1]
    b = structure_factor(p, nnr)[1]
    perm = np.random.default_rng(7).permutation(len(p))   # particle ORDER must not matter
    c = structure_factor(p[perm], nnr)[1]
    check("bit-identical on repeat call (deterministic directions, no RNG)",
          np.allclose(a, b, atol=0, rtol=0))
    check("invariant to particle ordering", np.allclose(a, c, atol=1e-6),
          f"max |diff| {np.abs(a - c).max():.2e}")

    # -------------------------------------------------------- 3. Bragg positions preserved
    print("\n3. Bragg-peak positions preserved across sizes")
    pk = {}
    for reps in (8, 11, 15):
        p, nnr = cscl(reps)
        k, s = structure_factor(p, nnr)
        pk[len(p)] = peak_positions(k, s, 3)
    ref_pk = pk[ns[-1]]
    ok, worst = True, 0.0
    for n in ns[:-1]:
        m = min(len(pk[n]), len(ref_pk))
        if m == 0:
            ok = False
            continue
        dev = np.abs(pk[n][:m] - ref_pk[:m]).max()
        worst = max(worst, float(dev))
        ok &= dev < 0.25                                 # < ~2.5 k-grid spacings
    print(f"      N={ns[0]}: {np.round(pk[ns[0]], 2)}   N={ns[1]}: {np.round(pk[ns[1]], 2)}"
          f"   N={ns[-1]}: {np.round(ref_pk, 2)}")
    check("peak positions agree across N to < 0.25 (1/nn)", ok, f"worst shift {worst:.3f}")

    # CsCl first Bragg peak: SC sublattice, a = 2*nn/sqrt(3) -> k*nn = 2*pi/a*nn = pi*sqrt(3)/...
    # in reduced units k_reduced = 2*pi*nn/a = 2*pi*(sqrt(3)/2) = 5.441 for the (100) family.
    expect = 2.0 * np.pi * (np.sqrt(3.0) / 2.0)
    got = ref_pk[np.argmin(np.abs(ref_pk - expect))] if len(ref_pk) else np.nan
    print(f"      analytic CsCl (100) at k={expect:.3f} (1/nn); nearest measured {got:.3f}")
    check("strongest peak sits at the analytic CsCl Bragg vector",
          np.isfinite(got) and abs(got - expect) < 0.3, f"|err| {abs(got - expect):.3f}")

    # ---------------------------------------------- 4. relative intensities across sizes
    print("\n4. relative peak intensities agree across sizes")
    rel = {}
    for reps in (8, 11, 15):
        p, nnr = cscl(reps)
        k, s = structure_factor(p, nnr)
        rel[len(p)] = relative_intensities(k, s, ref_pk)
    for n in ns:
        print(f"      N={n:5d}  I/I_max = {np.round(rel[n], 3)}")
    dev = max(float(np.abs(rel[n] - rel[ns[-1]]).max()) for n in ns[:-1])
    check("relative intensities agree across N to < 0.20", dev < 0.20, f"worst {dev:.3f}")

    # ------------------------------------------------------ 5. disordered baseline ~ 0
    print("\n5. disordered material must give S_hat ~ 0")
    for n in (1004, 2500, 6750):
        d = disordered(n, 1.0, seed=3)
        k, s = structure_factor(d, 1.0)
        print(f"      N={n:5d}  max S_hat={s.max():.4f}   mean={s.mean():.4f}")
        check(f"disordered N={n}: max S_hat < 0.05", s.max() < 0.05, f"{s.max():.4f}")

    # --------------------------------------------- 6. thermal noise must not kill the signal
    print("\n6. ordered-vs-disordered separation survives thermal jitter")
    p, nnr = cscl(11)
    k, s_ideal = structure_factor(p, nnr)
    p_th, _ = cscl(11, jitter=0.06)                      # ~6% of a, well inside Lindemann
    _, s_th = structure_factor(p_th, nnr)
    _, s_dis = structure_factor(disordered(len(p), nnr, seed=5), nnr)
    print(f"      max S_hat: ideal {s_ideal.max():.3f}  thermal {s_th.max():.3f}  "
          f"disordered {s_dis.max():.4f}")
    check("thermal crystal stays >10x above the disordered baseline",
          s_th.max() > 10 * max(s_dis.max(), 1e-6),
          f"{s_th.max() / max(s_dis.max(), 1e-6):.0f}x")

    print("\n" + "=" * 78)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"   FAILED: {f}")
    print("=" * 78)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

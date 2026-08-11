#!/usr/bin/env python3
"""Implementation tests (A-F) for radius_normalized_coord_r4_gas_screen_v1.

A uniform-scale invariance · B species-radius sensitivity · C PBC min-image ·
D physical-vs-normalized ordering · E decision boundaries & strict '>' ·
F frozen-reference reproduction against the validated H3 hybrid in sann_gas_screen_v2_final.

Run:  crystal_ml/bin/python3 test_gas_screen.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import radius_normalized_coord_r4_gas_screen as GS

THR = GS.MEAN_R4_STAR_THRESHOLD
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


# ── A. uniform-scale invariance ────────────────────────────────────────────────
def test_A():
    print("\nA. uniform-scale invariance")
    rng = np.random.default_rng(0)
    N, L = 220, 15.0
    pos = rng.uniform(0, L, (N, 3)); R = rng.uniform(0.4, 0.6, N)
    box = np.array([L, L, L]); tid = rng.integers(0, 2, N)
    base = GS.radius_normalized_coord_r4_gas_screen(positions=pos, radii=R, box=box, type_ids=tid)
    for c in (0.5, 1.0, 1.5, 2.0):
        r = GS.radius_normalized_coord_r4_gas_screen(positions=pos * c, radii=R * c, box=box * c,
                                                     type_ids=tid)
        cn_ok = np.array_equal(r.coordination_1p62, base.coordination_1p62)
        mr_ok = np.allclose(r.mean_r4_star, base.mean_r4_star, atol=1e-9, equal_nan=True)
        star_ok = np.allclose(r.r1_star, base.r1_star, atol=1e-9, equal_nan=True)
        dec_ok = np.array_equal(r.remove_mask, base.remove_mask)
        check(f"scale x{c}: CN & mean_r4* & r*_star & decision invariant",
              cn_ok and mr_ok and star_ok and dec_ok,
              f"CN={cn_ok} mean_r4*={mr_ok} r*_star={star_ok} decision={dec_ok}")


# ── B. species-radius sensitivity ──────────────────────────────────────────────
def test_B():
    print("\nB. species-radius sensitivity (invariant to uniform scale, NOT to size ratio)")
    rng = np.random.default_rng(1)
    N, L = 200, 14.0
    pos = rng.uniform(0, L, (N, 3)); box = np.array([L, L, L])
    tid = rng.integers(0, 2, N)
    equal = GS.radius_normalized_coord_r4_gas_screen(positions=pos, radii=np.full(N, 0.5),
                                                     box=box, type_ids=tid)
    R2 = np.where(tid == 0, 0.40, 0.62)
    ratio = GS.radius_normalized_coord_r4_gas_screen(positions=pos, radii=R2, box=box, type_ids=tid)
    changed = (not np.array_equal(equal.coordination_1p62, ratio.coordination_1p62)) or \
              (not np.allclose(equal.mean_r4_star, ratio.mean_r4_star, equal_nan=True))
    check("changing one species' radius changes normalized distances/CN", changed,
          f"CN diff={int((equal.coordination_1p62!=ratio.coordination_1p62).sum())} particles")


# ── C. PBC minimum image ───────────────────────────────────────────────────────
def test_C():
    print("\nC. PBC minimum-image distances")
    L = 10.0; box = np.array([L, L, L])
    # i near one face, j near the opposite face: direct 9.8, min-image 0.2
    pos = np.array([[0.1, 5, 5], [9.9, 5, 5], [0.1, 5.9, 5], [0.1, 5, 5.9], [0.1, 4.1, 5]])
    R = np.full(len(pos), 0.5)
    r = GS.radius_normalized_coord_r4_gas_screen(positions=pos, radii=R, box=box)
    r1_i = r.r1[0]
    check("nearest-neighbour distance uses min image (~0.2, not ~9.8)",
          abs(r1_i - 0.2) < 1e-6, f"r1(particle0)={r1_i:.4f}")
    check("across-boundary neighbour counted in CN", r.coordination_1p62[0] >= 4,
          f"CN(particle0)={int(r.coordination_1p62[0])}")


# ── D. physical-then-normalize ordering ────────────────────────────────────────
def test_D():
    print("\nD. select 4 by PHYSICAL distance, normalize each selected pair")
    box = np.array([1000.0, 1000, 1000])
    ctr = np.array([500.0, 500, 500])
    # A: physically nearest (1.0) but tiny radius -> normalized far ; B: farther (1.2) big radius -> normalized near
    pos = np.array([ctr,
                    ctr + [1.0, 0, 0],    # A
                    ctr + [0, 1.2, 0],    # B
                    ctr + [0, 0, 1.5],    # C
                    ctr + [1.8, 0, 0]])   # D
    R = np.array([0.5, 0.1, 1.0, 0.5, 0.5])
    r = GS.radius_normalized_coord_r4_gas_screen(positions=pos, radii=R, box=box)
    phys = np.array([r.r1[0], r.r2[0], r.r3[0], r.r4[0]])
    phys_sorted = np.all(np.diff(phys) >= -1e-9)
    # A is physical-nearest -> r1 == 1.0 ; its normalization s_A = 1.0/(0.5+0.1)=1.667 (the FAR normalized one)
    a_first = abs(r.r1[0] - 1.0) < 1e-6
    a_star = abs(r.r1_star[0] - (1.0 / 0.6)) < 1e-6
    # if it had (wrongly) sorted by normalized distance, r1_star would be B's 1.2/1.5=0.8
    not_normalized_sort = abs(r.r1_star[0] - 0.8) > 1e-3
    check("neighbours ordered by physical distance", phys_sorted, f"r1..r4={np.round(phys,3)}")
    check("nearest selected = physical-nearest A (r1=1.0)", a_first)
    check("r1_star normalizes A by its OWN pair radius (1.667), not the normalized-nearest",
          a_star and not_normalized_sort, f"r1_star={r.r1_star[0]:.4f}")


# ── E. decision boundaries and strict '>' ──────────────────────────────────────
def test_E():
    print("\nE. decision boundaries (CN gates + strict mean_r4* > threshold)")
    box = np.array([1000.0, 1000, 1000]); ctr = np.array([500.0, 500, 500])

    def one(neighbor_offsets, Rlist):
        pos = np.vstack([ctr] + [ctr + o for o in neighbor_offsets])
        R = np.array([0.5] + list(Rlist))
        return GS.radius_normalized_coord_r4_gas_screen(positions=pos, radii=R, box=box)

    # CN=1 -> obvious_gas remove  (one contact at s=1.0, rest far non-contacts)
    r = one([[1.0, 0, 0], [50, 0, 0], [0, 50, 0], [0, 0, 50]], [0.5, 0.5, 0.5, 0.5])
    check("CN=1 -> remove/obvious_gas", r.coordination_1p62[0] == 1 and r.remove_mask[0]
          and r.roles[0] == GS.ROLE_OBVIOUS_GAS, f"CN={int(r.coordination_1p62[0])} role={r.roles[0]}")

    # CN=4 -> retain/supported regardless of mean_r4*
    r = one([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0]], [0.5] * 4)
    check("CN=4 -> retain/supported_material", r.coordination_1p62[0] == 4 and (not r.remove_mask[0])
          and r.roles[0] == GS.ROLE_SUPPORTED_MATERIAL, f"CN={int(r.coordination_1p62[0])} role={r.roles[0]}")

    # CN=3, low mean_r4* -> retain (compact_ambiguous): 3 contacts @0.9 + 1 non-contact @1.62
    r = one([[0.9, 0, 0], [-0.9, 0, 0], [0, 0.9, 0], [0, 1.62, 0]], [0.5] * 4)
    mr = r.mean_r4_star[0]
    check("CN=3 & mean_r4*<thr -> retain/compact_ambiguous",
          r.coordination_1p62[0] == 3 and (not r.remove_mask[0])
          and r.roles[0] == GS.ROLE_COMPACT_AMBIGUOUS, f"CN=3 mean_r4*={mr:.4f}(<{THR})")

    # CN=3, high mean_r4* -> remove (gas_like_ambiguous): 3 contacts @0.9 + 1 non-contact @2.5
    r = one([[0.9, 0, 0], [-0.9, 0, 0], [0, 0.9, 0], [0, 2.5, 0]], [0.5] * 4)
    mr = r.mean_r4_star[0]
    check("CN=3 & mean_r4*>thr -> remove/gas_like_ambiguous",
          r.coordination_1p62[0] == 3 and r.remove_mask[0]
          and r.roles[0] == GS.ROLE_GAS_LIKE_AMBIGUOUS, f"CN=3 mean_r4*={mr:.4f}(>{THR})")

    # strict '>' invariant on a random frame: role assignment == (mean_r4* > thr) for ambiguous
    rng = np.random.default_rng(3)
    pos = rng.uniform(0, 16, (300, 3)); R = rng.uniform(0.4, 0.6, 300)
    rr = GS.radius_normalized_coord_r4_gas_screen(positions=pos, radii=R, box=np.array([16.0] * 3))
    amb = np.isin(rr.coordination_1p62, [2, 3]) & np.isfinite(rr.mean_r4_star)
    lhs = (rr.roles[amb] == GS.ROLE_GAS_LIKE_AMBIGUOUS)
    rhs = (rr.mean_r4_star[amb] > THR)          # strict; a value == thr must land in retain
    check("ambiguous removal == (mean_r4* > thr) exactly (strict '>', tie retains)",
          np.array_equal(lhs, rhs), f"{int(amb.sum())} ambiguous particles checked")


# ── F. frozen-reference reproduction vs H3 ─────────────────────────────────────
def test_F():
    print("\nF. frozen-reference reproduction vs H3 (sann_gas_screen_v2_final)")
    import pandas as pd
    V2 = Path("/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/sann_gas_screen_v2_final")
    sys.path.insert(0, str(V2))
    import io_utils
    pred = io_utils.read_predictions(V2 / "per_particle_predictions_L.csv")

    # F1: rule equivalence, using the module's frozen constants, over the particles H3 actually
    # made gas predictions on (the amorphous preservation dataset is constraint-only: H3 never
    # scores it, so remove_H3 is a default False there and is not a gas-prediction target).
    evaluated = (pred.dataset != "amorphous").to_numpy()
    cn = pred["CN_1.62"].to_numpy(); mr4 = pred["mean_r4_star"].to_numpy()
    rule = (cn < 2) | (np.isin(cn, [2, 3]) & (mr4 > GS.MEAN_R4_STAR_THRESHOLD))
    h3 = pred["remove_H3_coord+rk"].to_numpy().astype(bool)
    mis = int((rule[evaluated] != h3[evaluated]).sum())
    n_amorph = int((rule[~evaluated] != h3[~evaluated]).sum())
    check(f"F1 rule equivalence to H3 over {int(evaluated.sum())} H3-evaluated particles",
          mis == 0, f"{mis} mismatches; ({n_amorph} amorphous-preservation particles the module "
                    f"classifies but H3 never scored — expected, not a target)")

    # F2: END-TO-END geometry reproduction on finite frames (all particles stored, no dedup)
    import labeled_frames as LF
    sysmap = [("CsCl", "cscl_v4"), ("Th3P4", "th3p4_v4"), ("fluorite_0p69", "fluorite_c1_sr069")]
    total = mism = 0
    for label, dname in sysmap:
        d = LF.finite_copy(label, dname, 100)
        pos = d["pos"]; R = np.array([d["rr"][d["types"][t]] for t in d["tid"]])
        span = float(np.ptp(pos, axis=0).max()); big = np.array([span * 6.0] * 3)
        shifted = pos - pos.min(0) + span     # keep well inside the (huge) box; no PBC wrap artifacts
        res = GS.radius_normalized_coord_r4_gas_screen(positions=shifted, radii=R, box=big,
                                                       type_ids=d["tid"])
        ref = pred[(pred.system == label) &
                   (pred.frame.astype(str) == "100")].sort_values("particle_index")
        if len(ref) != len(pos):
            check(f"F2 {label}: particle-count match", False,
                  f"module {len(pos)} vs stored {len(ref)}"); continue
        h3f = ref["remove_H3_coord+rk"].to_numpy().astype(bool)
        m = int((res.remove_mask != h3f).sum())
        total += len(pos); mism += m
        # characterise any mismatch
        detail = f"{m}/{len(pos)}"
        if m:
            dm = np.flatnonzero(res.remove_mask != h3f)
            near = np.abs(res.mean_r4_star[dm] - THR) < 1e-6
            detail += f"  (near-threshold ties: {int(near.sum())}/{m})"
        check(f"F2 {label}: end-to-end module remove == H3", m == 0, detail)
    check(f"F2 overall end-to-end mismatch fraction",
          mism == 0, f"{mism}/{total} = {mism/max(total,1):.2e}")


def main():
    for t in (test_A, test_B, test_C, test_D, test_E, test_F):
        try:
            t()
        except Exception as e:
            import traceback; traceback.print_exc()
            check(t.__name__ + " (exception)", False, repr(e)[:200])
    npass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{'='*70}\nTOTAL: {npass}/{len(RESULTS)} checks passed")
    fails = [n for n, ok, _ in RESULTS if not ok]
    if fails:
        print("FAILED:", fails)
    print("=" * 70)
    sys.exit(0 if npass == len(RESULTS) else 1)


if __name__ == "__main__":
    main()

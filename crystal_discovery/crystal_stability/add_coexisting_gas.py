#!/usr/bin/env python
"""
Add a coexisting vapour phase around the crystal, so the stability test is done in
crystal-vapour COEXISTENCE rather than sublimation into vacuum.

WHY THIS EXISTS
Enlarging the box to Nicole's box-average volume fraction (0.0087) fixed the pressure
artefact -- our old boxes were 4-19x too dense, which propped up marginal crystals. But it
introduced a different artefact: the crystal then sits in a near-VACUUM. The vapour has
essentially zero chemical potential, so the crystal sublimates regardless of how stable
the lattice is; it only takes longer for stronger crystals. Ni3Sn evaporated outright
(radius of gyration 2.2x, 100% of particles leaving their sites).

Nicole's boxes are not like that: her particles are dispersed at VF 0.0087 and the crystal
coexists with a real vapour. Measured from her final frames:

    fully-crystallised runs   ~2%  of particles in the vapour   (rho ~ 5e-11 - 1e-10 /nm^3)
    partially-crystallised    ~24% of particles in the vapour   (rho ~ 1e-9 /nm^3)
    median gas volume fraction ~0.0033

So "match the volume fraction" and "match the pressure" are NOT the same thing. This script
reproduces the actual thermodynamic state: crystal + coexisting vapour at Nicole's
box-average density.

WHAT IT DOES
  1. keeps the crystal exactly as built (positions, types, diameters, charges untouched);
  2. adds N_gas free particles with the crystal's species ratio, placed at random with a
     hard minimum separation from the crystal and from each other (so nothing starts
     overlapping -> no instant NaN);
  3. sizes the box so the TOTAL (crystal + gas) volume fraction hits the target.

    python add_coexisting_gas.py runs/<name>/first_frame.gsd \
        --gas-fraction 0.24 --target-vf 0.0087
"""
import argparse
from pathlib import Path

import gsd.hoomd
import numpy as np
from scipy.spatial import cKDTree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gsd", type=Path)
    ap.add_argument("--gas-fraction", type=float, default=0.24,
                    help="vapour particles as a fraction of the TOTAL (Nicole median 0.24)")
    ap.add_argument("--target-vf", type=float, default=0.0087,
                    help="total box volume fraction, crystal + gas (Nicole median)")
    ap.add_argument("--min-sep-nn", type=float, default=1.05,
                    help="minimum gas-crystal / gas-gas separation, in units of nn")
    args = ap.parse_args()

    with gsd.hoomd.open(str(args.gsd), "r") as t:
        f = t[0]
        pos = np.asarray(f.particles.position, float)
        tid = np.asarray(f.particles.typeid, int)
        diam = np.asarray(f.particles.diameter, float)
        charge = np.asarray(f.particles.charge, float)
        types = list(f.particles.types)

    n_c = len(pos)
    # crystal's own nearest-neighbour distance sets the exclusion radius
    d, _ = cKDTree(pos).query(pos, k=2)
    nn = float(np.median(d[:, 1]))

    # gas particles keep the crystal's species ratio and per-species diameter/charge
    n_gas = int(round(args.gas_fraction / (1.0 - args.gas_fraction) * n_c))
    frac = {s: float((tid == s).mean()) for s in np.unique(tid)}
    gas_tid = np.concatenate([np.full(int(round(frac[s] * n_gas)), s, int)
                              for s in sorted(frac)])
    n_gas = len(gas_tid)
    proto = {s: (float(diam[tid == s][0]), float(charge[tid == s][0])) for s in np.unique(tid)}
    gas_diam = np.array([proto[s][0] for s in gas_tid])
    gas_charge = np.array([proto[s][1] for s in gas_tid])

    # box sized so TOTAL volume fraction hits the target
    v_tot = float((np.pi / 6.0 * diam ** 3).sum() + (np.pi / 6.0 * gas_diam ** 3).sum())
    L = (v_tot / args.target_vf) ** (1.0 / 3.0)

    pos_c = pos - pos.mean(0)                       # centre the crystal
    half = L / 2.0
    if np.abs(pos_c).max() > half - 2 * nn:
        raise SystemExit("crystal does not fit in the target box; lower --target-vf")

    # WALL EXCLUSION -- do not place gas near the box boundary.
    # The shifted-LJ wall potential diverges as a particle approaches the wall
    # (r -> delta = R - 1). Gas sampled uniformly in [-L/2, L/2] therefore lands ON the
    # wall, receives an essentially infinite force, and is launched: in the first
    # attempt a Ni3Sn run went from max|pos| = 7,357 nm to 322,727,936 nm between frames
    # 0 and 1, while the thermostat happily reported 298 K. Keep gas at least 2.5 * R_max
    # clear of every wall.
    wall_margin = 2.5 * float(diam.max()) / 2.0
    lim = half - wall_margin
    if lim <= np.abs(pos_c).max():
        raise SystemExit("box too small once the wall margin is applied")

    # rejection-sample gas positions: keep clear of the walls, the crystal, and each other
    min_sep = args.min_sep_nn * nn
    tree_c = cKDTree(pos_c)
    placed = []
    rng = np.random.default_rng(5808)
    tries = 0
    while len(placed) < n_gas and tries < 400 * n_gas:
        batch = rng.uniform(-lim, lim, (max(1024, n_gas), 3))     # <-- wall margin applied
        ok = tree_c.query(batch)[0] > min_sep
        batch = batch[ok]
        for p in batch:
            if len(placed) >= n_gas:
                break
            if not placed or cKDTree(np.array(placed)).query(p)[0] > min_sep:
                placed.append(p)
        tries += len(batch) + 1
    if len(placed) < n_gas:
        print(f"  WARNING: placed only {len(placed)}/{n_gas} gas particles")
        gas_tid = gas_tid[:len(placed)]
        gas_diam = gas_diam[:len(placed)]
        gas_charge = gas_charge[:len(placed)]
    gas_pos = np.array(placed)

    all_pos = np.vstack([pos_c, gas_pos])
    all_tid = np.concatenate([tid, gas_tid])
    all_diam = np.concatenate([diam, gas_diam])
    all_q = np.concatenate([charge, gas_charge])

    # sanity: no overlapping pairs anywhere
    dmin = cKDTree(all_pos).query(all_pos, k=2)[0][:, 1].min()
    assert dmin > 0.9 * min_sep, f"overlap in initial condition: {dmin:.1f} nm"

    # assert nothing sits inside the wall exclusion zone
    assert np.abs(gas_pos).max() <= lim + 1e-6, "gas placed inside the wall margin"

    vf_total = v_tot / L ** 3
    vf_gas = float((np.pi / 6.0 * gas_diam ** 3).sum()) / L ** 3

    f2 = gsd.hoomd.Frame()
    f2.particles.N = len(all_pos)
    f2.particles.types = types
    f2.particles.typeid = all_tid.astype(np.uint32)
    f2.particles.position = all_pos.astype(np.float32)
    f2.particles.diameter = all_diam.astype(np.float32)
    f2.particles.charge = all_q.astype(np.float32)
    f2.configuration.box = np.array([L, L, L, 0, 0, 0], dtype=np.float32)
    with gsd.hoomd.open(str(args.gsd), "w") as g:
        g.append(f2)

    print(f"  crystal {n_c} + vapour {len(gas_pos)} = {len(all_pos)} particles   "
          f"box {L:.0f} nm   total VF {vf_total:.5f}   vapour VF {vf_gas:.5f}   "
          f"min sep {dmin:.0f} nm   wall margin {wall_margin:.0f} nm   "
          f"closest particle to wall {half - np.abs(all_pos).max():.0f} nm")


if __name__ == "__main__":
    main()

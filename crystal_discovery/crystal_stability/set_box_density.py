#!/usr/bin/env python
"""
Enlarge the simulation box of a built first_frame.gsd to hit a target volume fraction.

Why: our stability boxes were 4x-19x DENSER than the dataset they are supposed to be
comparable with. Measured box volume fraction (sum of particle volumes / box volume):

    our runs          0.032 - 0.165
    Nicole's dataset  0.0027 - 0.024   (median 0.0087)

LatticeBuilder sizes the box from `lattice_padding`, but the crystal's extent is only
known after the energy-optimised scale factor is chosen, so the padding needed for a
given volume fraction cannot be set in advance. Easiest correct fix: build the lattice as
usual, then rewrite the box.

Only the BOX is changed. Particle positions, types, diameters and charges are untouched,
so the crystal itself is bit-for-bit identical -- we are only giving it more vacuum.
The crystal is re-centred at the origin so the walls stay symmetric about it.

    python set_box_density.py runs/<name>/first_frame.gsd --target-vf 0.0087
"""
import argparse
from pathlib import Path

import gsd.hoomd
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gsd", type=Path)
    ap.add_argument("--target-vf", type=float, default=0.0087,
                    help="target box volume fraction (default 0.0087 = Nicole median)")
    ap.add_argument("--min-vacuum-nn", type=float, default=6.0,
                    help="minimum vacuum gap between crystal edge and wall, in units of nn")
    args = ap.parse_args()

    with gsd.hoomd.open(str(args.gsd), "r") as t:
        f = t[0]
        pos = np.asarray(f.particles.position, float)
        diam = np.asarray(f.particles.diameter, float)
        box = np.asarray(f.configuration.box, float)
        types = list(f.particles.types)
        tid = np.asarray(f.particles.typeid)
        charge = np.asarray(f.particles.charge, float)

    v_part = float((np.pi / 6.0 * diam ** 3).sum())
    L_old = float(box[0])
    vf_old = v_part / L_old ** 3

    L_new = (v_part / args.target_vf) ** (1.0 / 3.0)

    # safety: the box must still comfortably contain the crystal plus vacuum
    pos_c = pos - pos.mean(0)
    extent = 2.0 * float(np.abs(pos_c).max())
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pos_c).query(pos_c, k=2)
    nn = float(np.median(d[:, 1]))
    L_min = extent + 2.0 * args.min_vacuum_nn * nn
    if L_new < L_min:
        L_new = L_min
        print(f"  NOTE: target VF would make the box too small for the crystal; "
              f"clamped to {L_min:.0f} nm ({args.min_vacuum_nn}*nn vacuum each side)")

    vf_new = v_part / L_new ** 3

    f2 = gsd.hoomd.Frame()
    f2.particles.N = len(pos)
    f2.particles.types = types
    f2.particles.typeid = tid
    f2.particles.position = pos_c.astype(np.float32)      # re-centred; geometry unchanged
    f2.particles.diameter = diam.astype(np.float32)
    f2.particles.charge = charge.astype(np.float32)
    f2.configuration.box = np.array([L_new, L_new, L_new, 0, 0, 0], dtype=np.float32)
    with gsd.hoomd.open(str(args.gsd), "w") as g:
        g.append(f2)

    print(f"  box {L_old:.0f} -> {L_new:.0f} nm   "
          f"volume fraction {vf_old:.5f} -> {vf_new:.5f}   "
          f"(crystal extent {extent:.0f} nm, vacuum {(L_new-extent)/2:.0f} nm/side)")


if __name__ == "__main__":
    main()

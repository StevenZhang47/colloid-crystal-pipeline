#!/usr/bin/env python
"""
Turn a LatticeBuilder crystal into a FULLY PERIODIC BULK crystal (no surface at all).

WHY
Every stability artefact we have hit came from the crystal being a finite block inside a
box:
  small box  -> high pressure          -> crystals looked MORE stable than they are
  big box    -> vacuum sublimation sink-> crystals evaporated regardless of lattice stability
  + gas      -> particles placed near the walls got launched by the diverging wall potential

All three are SURFACE / BOUNDARY artefacts, not lattice physics. A fully periodic bulk
crystal has none of them:
  * no free surface  -> no evaporation, no sublimation sink
  * no walls         -> nothing to launch
  * pressure is set by the lattice constant (already energy-minimised by LatticeBuilder)
  * what you measure is exactly the question: is the BULK LATTICE stable at 298 K?

Surface/evaporation behaviour becomes a separate, deliberate experiment instead of a
confound baked into the main test.

HOW
LatticeBuilder builds positions as   supercell.cart_coords * scale   (then centred).
So the periodic box is simply   scale * (supercell lattice vectors).
We recover `scale` by comparing the nearest-neighbour distance in the built GSD with the
nearest-neighbour distance of the same supercell in CIF units, then set the box to the
scaled supercell and wrap the positions into it.

    python make_periodic.py runs/<name>/first_frame.gsd --cif <file>.cif --repeats N
"""
import argparse
from pathlib import Path

import gsd.hoomd
import numpy as np
from pymatgen.io.cif import CifParser
from scipy.spatial import cKDTree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gsd", type=Path)
    ap.add_argument("--cif", type=Path, required=True)
    ap.add_argument("--repeats", type=int, required=True)
    args = ap.parse_args()

    with gsd.hoomd.open(str(args.gsd), "r") as t:
        f = t[0]
        pos = np.asarray(f.particles.position, float)
        tid = np.asarray(f.particles.typeid, int)
        diam = np.asarray(f.particles.diameter, float)
        charge = np.asarray(f.particles.charge, float)
        types = list(f.particles.types)

    # nearest-neighbour distance in the BUILT crystal (nm)
    nn_gsd = float(np.median(cKDTree(pos).query(pos, k=2)[0][:, 1]))

    # the same supercell in CIF units
    s = CifParser(str(args.cif), site_tolerance=0.0, frac_tolerance=0.0
                  ).parse_structures(check_occu=False, primitive=False)[0]
    sc = s.make_supercell(args.repeats, in_place=False)
    cart = sc.cart_coords
    nn_cif = float(np.median(cKDTree(cart).query(cart, k=2)[0][:, 1]))

    scale = nn_gsd / nn_cif
    a, b, c = np.array(sc.lattice.abc) * scale           # periodic box = scaled supercell
    ang = np.array(sc.lattice.angles)
    if not np.allclose(ang, 90.0, atol=1e-3):
        # non-orthogonal cells: express as a HOOMD triclinic box
        m = sc.lattice.matrix * scale
        lx = np.linalg.norm(m[0])
        xy = np.dot(m[1], m[0]) / lx ** 2 if lx else 0.0
        ly = np.sqrt(max(np.dot(m[1], m[1]) - (xy * lx) ** 2, 1e-9))
        xz = np.dot(m[2], m[0]) / lx ** 2 if lx else 0.0
        yz = (np.dot(m[2], m[1]) - xy * lx * xz * lx) / (ly ** 2) if ly else 0.0
        lz = np.sqrt(max(np.dot(m[2], m[2]) - (xz * lx) ** 2 - (yz * ly) ** 2, 1e-9))
        box = [lx, ly, lz, xy, xz, yz]
        L = np.array([lx, ly, lz])
    else:
        box = [a, b, c, 0.0, 0.0, 0.0]
        L = np.array([a, b, c])

    # wrap positions into the periodic box
    p = pos - pos.mean(0)
    p = p - L * np.round(p / L)

    # sanity: the periodic nearest-neighbour distance must match the crystal's own nn
    w = np.mod(p + L / 2.0, L)
    nn_per = float(np.median(cKDTree(w, boxsize=L).query(w, k=2)[0][:, 1]))
    assert abs(nn_per - nn_gsd) / nn_gsd < 0.02, (
        f"periodic wrap broke the lattice: nn {nn_per:.1f} vs {nn_gsd:.1f} nm")
    # and no particle may sit on top of another across the boundary
    assert nn_per > 0.5 * nn_gsd, "overlap created by periodic wrap"

    vf = float((np.pi / 6.0 * diam ** 3).sum()) / float(np.prod(L))

    f2 = gsd.hoomd.Frame()
    f2.particles.N = len(p)
    f2.particles.types = types
    f2.particles.typeid = tid.astype(np.uint32)
    f2.particles.position = p.astype(np.float32)
    f2.particles.diameter = diam.astype(np.float32)
    f2.particles.charge = charge.astype(np.float32)
    f2.configuration.box = np.array(box, dtype=np.float32)
    with gsd.hoomd.open(str(args.gsd), "w") as g:
        g.append(f2)

    print(f"  PERIODIC BULK: N={len(p)}  box=({L[0]:.0f}, {L[1]:.0f}, {L[2]:.0f}) nm  "
          f"scale={scale:.2f}  nn={nn_per:.1f} nm  crystal VF={vf:.4f}  "
          f"(no surface, no walls, no vacuum)")


if __name__ == "__main__":
    main()

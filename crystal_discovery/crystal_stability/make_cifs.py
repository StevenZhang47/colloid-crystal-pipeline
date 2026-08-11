#!/usr/bin/env python
"""
Generate the prototype CIFs that the task14 library is missing, in the form
LatticeBuilder wants: conventional cell, P1, every site explicit, occupancy 1.

LatticeBuilder names particle types after the CIF's ELEMENT SYMBOLS
(Element.from_Z on the atomic numbers), so the symbols must be real elements --
they are only labels for the two colloid sublattices, but they must parse.
That is why zangenite's 'L'/'S' had to be remapped: 'L' is not an element and
pymatgen turns it into a DummySpecies that Element.from_Z rejects.
"""
from pathlib import Path

from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

HERE = Path(__file__).resolve().parent
OUT = HERE / "structures"

# name -> (spacegroup, lattice, species, fractional coords, note)
# Only prototypes whose Wyckoff positions are FULLY FIXED (no free internal x,y,z)
# can be built from the AFLOW label alone. Prototypes with free parameters
# (oP20, oP28, mC28, tP20, most hR10, hP12) need their internal coordinates from
# MP/AFLOW and are handled separately -- see NEEDS_EXTERNAL_COORDS below.
PROTOTYPES = {
    # AB_cF8_225_a_b: rocksalt NaCl B1. Fm-3m. A at 4a, B at 4b.
    "rocksalt_B1": (
        225, Lattice.cubic(5.64), ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        "1:1, each site octahedrally coordinated by 6 unlike",
    ),
    # AB_cF8_216_a_c: zincblende / sphalerite B3. F-43m. A at 4a, B at 4c.
    "zincblende_B3": (
        216, Lattice.cubic(5.4), ["Zn", "S"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        "1:1, each site tetrahedrally coordinated by 4 unlike",
    ),
    # A4B_cP5_221_ac_b: Fe4N / anti-perovskite Lp10. Pm-3m. A at 1a+3c, B at 1b.
    # Covers BOTH the 1:4 (fe4n) and 4:1 (anti_perovskite) list entries -- same
    # framework; which colloid is majority is a design.py assignment, not a new CIF.
    "fe4n_Lp10": (
        221, Lattice.cubic(3.79), ["Fe", "Fe", "N"],
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.5, 0.5]],
        "4:1, N on the cube corner, Fe on face centres (1a+3c)",
    ),
    # AB_hP4_194_c_a: nickeline NiAs B8_1. P6_3/mmc. Ni at 2a, As at 2c.
    "nickeline_B81": (
        194, Lattice.hexagonal(3.6, 5.0), ["Ni", "As"],
        [[0.0, 0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 0.25]],
        "1:1, Ni octahedral / As trigonal prismatic; c/a ~ 1.39",
    ),
    # AB2_cF12_225_a_c: fluorite CaF2 C1. Fm-3m. Ca at 4a, F at 8c.
    "fluorite_C1": (
        225, Lattice.cubic(5.46), ["Ca", "F"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        "1:2, minority on fcc, majority on the 8c simple-cubic sublattice",
    ),
    # Antifluorite: same framework, species swapped -> A2B (2:1). Li2O.
    "antifluorite_C1b": (
        225, Lattice.cubic(4.61), ["O", "Li"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        "2:1, the fluorite framework with the roles of the two species exchanged",
    ),
    # Zangenite: user-supplied Cmmm cell. 'L' (large) -> La, 'S' (small) -> S.
    "zangenite": (
        65, Lattice.orthorhombic(5.09, 9.23, 2.54), ["La", "La", "S"],
        [[0.0, 0.28, 0.0], [0.5, 0.5, 0.0], [0.21, 0.14, 0.5]],
        "Cmmm; La = the LARGE colloid (was 'L'), S = the SMALL colloid (was 'S')",
    ),
}


def main():
    OUT.mkdir(exist_ok=True)
    for name, (spacegroup, lattice, species, coords, note) in PROTOTYPES.items():
        structure = Structure.from_spacegroup(spacegroup, lattice, species, coords)
        counts = structure.composition.get_el_amt_dict()
        path = OUT / f"{name}.cif"
        # symprec=None writes P1 with every site listed explicitly.
        CifWriter(structure, symprec=None).write_file(path)
        ratio = " : ".join(f"{el} {int(n)}" for el, n in counts.items())
        print(f"{name:20s} {structure.num_sites:3d} sites/cell   {ratio:20s}  {note}")

    print(f"\nwrote {len(PROTOTYPES)} CIFs to {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Group library CIFs into structural equivalence classes, by StructureMatcher.

Why this has to be computed rather than hand-written: the self-match test asks whether
a crystal matches "only itself", and that question is meaningless until "itself" is
defined. The library holds several CIFs that are the SAME structure type under different
names, cell settings or species labels:

  * Li3Fe and Ni3Sn (D0_19) are both P6_3/mmc, 8 sites, 1:3, c/a 0.818 vs 0.802
  * three Cu3Au cell settings, three Th3P4, two CsCl, two zangenite
  * fluorite and antifluorite are the same framework with the species swapped

Counting those as distinct prototypes turns a correct match into a reported failure --
which is exactly what happened on the first run, where all six Ni3Sn candidates were
recorded as matching the "wrong" prototype because they landed on the Li3Fe anchor.

Two groupings are emitted, because they answer different questions:

  framework_group   same arrangement of sites, IGNORING which species sits where.
                    This is the structure-type question, and it is the one the
                    prototype-level self-match verdict should use.
  species_group     same arrangement AND the same species on the same sites. This
                    separates fluorite from antifluorite; framework_group does not.

Usage:  python prototype_groups.py        # writes prototype_groups.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pymatgen.analysis.structure_matcher import FrameworkComparator, StructureMatcher
from pymatgen.io.cif import CifParser

ROOT = Path("/projects/rps/gmh4/hockygroup/sz5808")
LIB = ROOT / "crystal_discovery/cif_library"

# Parse the way the BUILDER does, so the grouping describes the structures that were
# actually built rather than pymatgen's site-merged reading of them.
PARSE = dict(site_tolerance=0.0, frac_tolerance=0.0)


def load(path: Path):
    """Parse, and refuse anything the builder could not build either.

    Two library CIFs are unusable and must not enter the grouping: runs/cu3au/Cu3Au.cif
    expands to 96 sites with coincident pairs (spglib then fails outright on it), and the
    L/S zangenite uses placeholder species labels that parse to nonsense atomic numbers.
    Both would otherwise crash or silently pollute the equivalence classes.
    """
    import numpy as np
    from pymatgen.core import Element
    from scipy.spatial import distance_matrix

    s = CifParser(str(path), **PARSE).parse_structures(
        check_occu=False, primitive=False)[0]
    for z in np.unique(s.atomic_numbers):
        Element.from_Z(int(z))                      # raises on placeholder labels
    d = distance_matrix(s.cart_coords, s.cart_coords)
    np.fill_diagonal(d, np.inf)
    if d.min() < 1e-8:
        raise ValueError(f"{int((d < 1e-8).sum() // 2)} coincident site pairs")
    return s


def classes(structures: dict, matcher: StructureMatcher) -> dict:
    """Union-find over pairwise fit(). Names sorted so the label is deterministic."""
    names = sorted(structures)
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if find(a) == find(b):
                continue
            try:
                same = matcher.fit(structures[a], structures[b])
            except Exception:  # noqa: BLE001
                same = False
            if same:
                parent[find(b)] = find(a)
    # label each class by its alphabetically first member
    out = {}
    for n in names:
        root = find(n)
        out.setdefault(root, []).append(n)
    return {n: sorted(members)[0] for root, members in out.items() for n in members}


def main() -> None:
    cif_dir = LIB / "prototypes"
    structures, failed = {}, {}
    for p in sorted(cif_dir.glob("*.cif")):
        try:
            structures[p.stem] = load(p)
        except Exception as exc:  # noqa: BLE001
            failed[p.stem] = f"{type(exc).__name__}: {exc}"[:80]
    if failed:
        print("unparseable, excluded from grouping:")
        for k, v in failed.items():
            print(f"  {k:28s} {v}")
        print()

    framework = classes(structures, StructureMatcher(
        primitive_cell=True, attempt_supercell=True, comparator=FrameworkComparator()))
    species = classes(structures, StructureMatcher(
        primitive_cell=True, attempt_supercell=True))

    rows = []
    for n, s in structures.items():
        try:
            sg, sgn = s.get_space_group_info()
        except Exception:  # noqa: BLE001
            sg, sgn = "undetermined", -1
        a, b, c = s.lattice.abc
        rows.append({"slug": n, "framework_group": framework[n], "species_group": species[n],
                     "spacegroup": sg, "spacegroup_number": sgn, "n_sites": len(s),
                     "c_over_a": round(c / a, 4)})
    df = pd.DataFrame(rows).sort_values(["framework_group", "slug"])
    df.to_csv(LIB / "prototype_groups.csv", index=False)

    print(f"{len(structures)} CIFs -> {df.framework_group.nunique()} framework groups, "
          f"{df.species_group.nunique()} species groups\n")
    print("groups with more than one member (these are NOT distinct prototypes):")
    for g, sub in df.groupby("framework_group"):
        if len(sub) > 1:
            same_species = sub.species_group.nunique() == 1
            note = "" if same_species else "   [species differ -- e.g. fluorite/antifluorite]"
            print(f"  {g:28s} {', '.join(sub.slug)}{note}")
    print(f"\nwrote {LIB / 'prototype_groups.csv'}")


if __name__ == "__main__":
    main()

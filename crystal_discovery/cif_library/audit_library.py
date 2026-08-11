#!/usr/bin/env python3
"""Audit the CIF library against the target prototype list.

Three questions, answered separately because they have different answers:

  1. AMBIGUITY   several prototypes have >1 distinct CIF file.  Are those genuinely
                 different structures, or the same structure written differently?
                 Decided with pymatgen StructureMatcher, not by filename.

  2. COVERAGE    which of the ~30 target prototypes do we already have, and which
                 must be fetched?  Matched on space group + stoichiometry, since
                 prototype names are not standardised across sources.

  3. VALIDITY    for every prototype that has actually been BUILT into a colloidal
                 crystal, is that crystal cohesive or has it exploded?  The failure
                 mode from the anisotropic-scaling fix is a lattice so dilute that
                 nothing interacts, so the test is nearest-neighbour spacing against
                 the steric contact distance.

Read-only.  Writes audit CSVs into the library directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.cif import CifParser

ROOT = Path("/projects/rps/gmh4/hockygroup/sz5808")
CS = ROOT / "crystal_discovery/crystal_stability"
LIB = ROOT / "crystal_discovery/cif_library"

# (prototype, strukturbericht, example, spacegroup_number, LARGE:SMALL) from the
# target list.  Ratio is written minority:majority to match the catalog convention.
TARGETS = [
    # --- AB ---
    ("CsCl", "B2", "CsCl", 221, "1:1"), ("Rocksalt", "B1", "NaCl", 225, "1:1"),
    ("Zinc blende", "B3", "ZnS", 216, "1:1"), ("Wurtzite", "B4", "ZnS", 186, "1:1"),
    ("NiAs", "B8_1", "NiAs", 194, "1:1"), ("WC", "B_h", "WC", 187, "1:1"),
    ("FeSi", "B20", "FeSi", 198, "1:1"), ("MnP", "B31", "MnP", 62, "1:1"),
    ("NaTl", "B32", "NaTl", 227, "1:1"), ("CuAu", "L1_0", "CuAu", 123, "1:1"),
    # --- AB2 / A2B ---
    ("Fluorite", "C1", "CaF2", 225, "1:2"), ("Antifluorite", "C1", "Li2O", 225, "1:2"),
    ("Rutile", "C4", "TiO2", 136, "1:2"), ("Anatase", "C5", "TiO2", 141, "1:2"),
    ("Pyrite", "C2", "FeS2", 205, "1:2"), ("Cuprite", "C3", "Cu2O", 224, "1:2"),
    ("AlB2", "C32", "AlB2", 191, "1:2"), ("CdI2", "C6", "CdI2", 164, "1:2"),
    ("Laves C14", "C14", "MgZn2", 194, "1:2"), ("Laves C15", "C15", "MgCu2", 227, "1:2"),
    ("Laves C36", "C36", "MgNi2", 194, "1:2"), ("MoSi2", "C11b", "MoSi2", 139, "1:2"),
    # --- A3B / AB3 ---
    ("Cu3Au", "L1_2", "Cu3Au", 221, "1:3"), ("Cr3Si", "A15", "Cr3Si", 223, "1:3"),
    ("Ni3Sn", "D0_19", "Ni3Sn", 194, "1:3"), ("Al3Ti", "D0_22", "Al3Ti", 139, "1:3"),
    ("BiF3", "D0_3", "BiF3", 225, "1:3"), ("Cementite", "D0_11", "Fe3C", 62, "1:3"),
    ("ReO3", "D0_9", "ReO3", 221, "1:3"),
    # --- A3B4 ---
    ("Th3P4", "D7_3", "Th3P4", 220, "3:4"), ("Spinel", "H1_1", "MgAl2O4", 227, "3:4"),
]


def contact_validity(first_frame: Path) -> dict:
    """Is the built colloidal crystal cohesive, or blown apart?"""
    import gsd.hoomd
    from scipy.spatial import cKDTree
    with gsd.hoomd.open(str(first_frame), "r") as t:
        f = t[0]
        pos = np.asarray(f.particles.position, float)
        dia = np.asarray(f.particles.diameter, float)
        tid = np.asarray(f.particles.typeid, int)
    md = {int(x): float(dia[tid == x].mean()) for x in np.unique(tid)}
    o = sorted(md, key=lambda x: -md[x])
    contact = (md[o[0]] + md[o[1]]) / 2 if len(o) > 1 else md[o[0]]
    nn = float(np.min(cKDTree(pos).query(pos, k=2)[0][:, 1]))
    ratio = nn / contact
    # cohesive lattices sit just above contact.  A scan that ran to its boundary
    # leaves the cell so dilute that nothing interacts.
    verdict = ("OK_cohesive" if ratio <= 1.25 else
               "SUSPECT_dilute" if ratio <= 2.0 else "EXPLODED_no_contact")
    return {"n_particles": len(pos), "min_nn": round(nn, 1),
            "contact_distance": round(contact, 1), "nn_over_contact": round(ratio, 3),
            "validity": verdict}


def main() -> None:
    cat = pd.read_csv(LIB / "catalog.csv")

    # ---- 1. ambiguity: same structure or genuinely different? ----
    sm = StructureMatcher(primitive_cell=True, attempt_supercell=True)
    structs = {}
    for r in cat[cat.parse_status.eq("OK")].itertuples(index=False):
        structs[r.slug] = CifParser(str(LIB / "prototypes" / f"{r.slug}.cif")
                                    ).parse_structures(primitive=False)[0]
    rows = []
    for key, grp in cat.groupby("prototype_key"):
        slugs = [s for s in grp.slug if s in structs]
        if len(slugs) < 2:
            continue
        ref = slugs[0]
        for other in slugs[1:]:
            same = sm.fit(structs[ref], structs[other])
            rows.append({"prototype": key, "cif_a": ref, "cif_b": other,
                         "structurally_identical": bool(same),
                         "verdict": "duplicate file, same structure" if same
                                    else "GENUINELY DIFFERENT STRUCTURE"})
    amb = pd.DataFrame(rows)
    amb.to_csv(LIB / "audit_ambiguity.csv", index=False)
    pd.set_option("display.width", 220)
    print("=== 1. AMBIGUOUS PROTOTYPES ===")
    print(amb.to_string(index=False) if len(amb) else "  none")

    # ---- 2. coverage against the target list ----
    have = cat[cat.parse_status.eq("OK")]
    cov = []
    for proto, sb, example, sgn, ratio in TARGETS:
        hit = have[(have.spacegroup_number == sgn) & (have.large_small_ratio == ratio)]
        cov.append({"prototype": proto, "strukturbericht": sb, "example": example,
                    "spacegroup_number": sgn, "large_small_ratio": ratio,
                    "have": bool(len(hit)),
                    "matched_cif": ";".join(hit.slug) if len(hit) else "",
                    "n_free_lattice_params": int(hit.n_free_lattice_params.iloc[0]) if len(hit) else None,
                    "requires_anisotropic": bool(hit.requires_anisotropic.iloc[0]) if len(hit) else None})
    cv = pd.DataFrame(cov)
    cv.to_csv(LIB / "audit_coverage.csv", index=False)
    print(f"\n=== 2. COVERAGE: {int(cv.have.sum())}/{len(cv)} targets present ===")
    print(cv[["prototype", "strukturbericht", "spacegroup_number", "large_small_ratio",
              "have", "matched_cif"]].to_string(index=False))

    # ---- 3. validity of everything actually built ----
    val = []
    for run in sorted((CS / "runs").iterdir()):
        ff = run / "first_frame.gsd"
        if not ff.exists():
            continue
        try:
            val.append({"run": run.name, **contact_validity(ff)})
        except Exception as exc:  # noqa: BLE001
            val.append({"run": run.name, "validity": f"ERROR::{type(exc).__name__}"})
    vd = pd.DataFrame(val)
    vd.to_csv(LIB / "audit_build_validity.csv", index=False)
    print(f"\n=== 3. BUILT-CRYSTAL VALIDITY ({len(vd)} runs) ===")
    print(vd.validity.value_counts().to_string())
    bad = vd[~vd.validity.eq("OK_cohesive")]
    if len(bad):
        print("\n  not cohesive:")
        print(bad.to_string(index=False))


if __name__ == "__main__":
    main()

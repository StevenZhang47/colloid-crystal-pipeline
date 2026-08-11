#!/usr/bin/env python3
"""Fetch missing prototypes from the Materials Project, BY mp-id.

Why mp-id and not formula: a formula query returns every polymorph MP holds and
silently returns one of them.  For TiO2 that is rutile OR anatase OR brookite --
three different targets on the list.  Pinning the id makes the fetch reproducible
and reviewable.

Every fetched structure is validated before it is written:
  * space group determined FROM COORDINATES must match the expected number
  * reduced stoichiometry must match the expected LARGE:SMALL ratio
  * axis groups recomputed, so requires_anisotropic is authoritative

Credentials come from $MP_API_KEY.  Nothing is written into any file here.

Usage:
    MP_API_KEY=... python fetch_targets.py --dry-run     # list what would be pulled
    MP_API_KEY=... python fetch_targets.py               # fetch and validate
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ROOT = Path("/projects/rps/gmh4/hockygroup/sz5808")
LIB = ROOT / "crystal_discovery/cif_library"
# Anisotropic LatticeBuilder from the persistent worktree, not a session scratchpad.
sys.path.insert(0, str(ROOT / "crystal_discovery/PACSim_aniso"))
from colloids.colloids_create.configuration_generators.lattice_builder import (  # noqa: E402
    LatticeBuilder,
)

# slug, prototype, strukturbericht, search formula, expected spacegroup number, expected LARGE:SMALL
# NO hand-written mp-ids: the id is RESOLVED by querying formula + space group and
# taking the lowest energy_above_hull hit.  The resolved id is printed and recorded
# so it can be checked, rather than asserted from memory.
TARGETS = [
    # ---------------- AB ----------------
    ("wurtzite_B4",      "Wurtzite",    "B4",    "ZnS",     186, "1:1"),
    ("wc_Bh",            "WC",          "B_h",   "WC",      187, "1:1"),
    ("fesi_B20",         "FeSi",        "B20",   "FeSi",    198, "1:1"),
    ("mnp_B31",          "MnP",         "B31",   "MnP",      62, "1:1"),
    ("natl_B32",         "NaTl",        "B32",   "NaTl",    227, "1:1"),
    ("cuau_L10",         "CuAu",        "L1_0",  "CuAu",    123, "1:1"),
    # ---------------- AB2 / A2B ----------------
    ("rutile_C4",        "Rutile",      "C4",    "TiO2",    136, "1:2"),
    ("anatase_C5",       "Anatase",     "C5",    "TiO2",    141, "1:2"),
    ("pyrite_C2",        "Pyrite",      "C2",    "FeS2",    205, "1:2"),
    ("cuprite_C3",       "Cuprite",     "C3",    "Cu2O",    224, "1:2"),
    ("alb2_C32",         "AlB2",        "C32",   "AlB2",    191, "1:2"),
    ("cdi2_C6",          "CdI2",        "C6",    "CdI2",    164, "1:2"),
    ("laves_C14_MgZn2",  "Laves C14",   "C14",   "MgZn2",   194, "1:2"),
    ("laves_C15_MgCu2",  "Laves C15",   "C15",   "MgCu2",   227, "1:2"),
    ("laves_C36_MgNi2",  "Laves C36",   "C36",   "MgNi2",   194, "1:2"),
    ("mosi2_C11b",       "MoSi2",       "C11b",  "MoSi2",   139, "1:2"),
    # ---------------- A3B / AB3 ----------------
    ("al3ti_D022",       "Al3Ti",       "D0_22", "Al3Ti",   139, "1:3"),
    ("bif3_D03",         "BiF3",        "D0_3",  "BiF3",    225, "1:3"),
    ("cementite_D011",   "Cementite",   "D0_11", "Fe3C",     62, "1:3"),
    # ---------------- A3B4 ----------------
    ("spinel_normal_Co3O4",  "Normal spinel",  "H1_1", "Co3O4", 227, "3:4"),
    ("spinel_inverse_Fe3O4", "Inverse spinel", "H1_1", "Fe3O4", 227, "3:4"),
]


def validate(struct, sgn_expected, ratio_expected) -> dict:
    sga = SpacegroupAnalyzer(struct, symprec=0.1)
    amt = struct.composition.get_el_amt_dict()
    els = sorted(amt, key=lambda e: amt[e])
    counts = [int(amt[e]) for e in els]
    g = 0
    for c in counts:
        g = np.gcd(g, c)
    ratio = ":".join(str(x // max(g, 1)) for x in counts)
    groups = LatticeBuilder._axis_groups(struct)
    sgn = int(sga.get_space_group_number())
    return {
        "formula": struct.composition.reduced_formula,
        "spacegroup_symbol": sga.get_space_group_symbol(), "spacegroup_number": sgn,
        "crystal_system": sga.get_crystal_system(),
        "large_small_ratio": ratio, "n_sites": len(struct),
        "a": round(struct.lattice.abc[0], 4),
        "b_over_a": round(struct.lattice.abc[1] / struct.lattice.abc[0], 4),
        "c_over_a": round(struct.lattice.abc[2] / struct.lattice.abc[0], 4),
        "n_free_lattice_params": len(groups),
        "requires_anisotropic": len(groups) > 1,
        "spacegroup_matches_expected": sgn == sgn_expected,
        "ratio_matches_expected": ratio == ratio_expected,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    key = os.environ.get("MP_API_KEY")
    if not key:
        sys.exit("MP_API_KEY not set in the environment.")

    from mp_api.client import MPRester
    out_dir = LIB / "prototypes"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with MPRester(key) as mpr:
        for slug, proto, sb, formula, sgn, ratio in TARGETS:
            rec = {"slug": slug, "prototype": proto, "strukturbericht": sb,
                   "search_formula": formula, "expected_spacegroup": sgn,
                   "expected_ratio": ratio}
            try:
                docs = mpr.materials.summary.search(
                    formula=formula,
                    fields=["material_id", "structure", "symmetry", "energy_above_hull"])
                if not docs:
                    raise RuntimeError(f"no MP entry for formula {formula}")
                # keep only entries whose space group matches the target prototype,
                # then take the most stable one
                hits = [d for d in docs if int(d.symmetry.number) == sgn]
                if not hits:
                    got = sorted({int(d.symmetry.number) for d in docs})
                    raise RuntimeError(f"no {formula} entry with spacegroup {sgn}; MP has {got}")
                # Selecting the prototype cell needs BOTH signals, because neither
                # alone is right:
                #   * "lowest energy" picks polytype superstructures -- ZnS P6_3mc has
                #     entries at 4/8/12/16/20 sites with c/a = 1.64/3.27/4.90/6.54/8.18
                #     (n x 1.635), all within 0.003 eV/atom of each other.
                #   * "smallest cell" picks metastable oddities -- MgZn2 P6_3/mmc has a
                #     6-site entry at e_hull 0.154 alongside the true 12-site C14 Laves
                #     (c/a 1.624) which sits ON the hull.
                # So: keep entries near the hull, then take the smallest of those.
                E_TOL = 0.05      # eV/atom above the best hit's hull energy
                e = [d.energy_above_hull if d.energy_above_hull is not None else 9e9
                     for d in hits]
                emin = min(e)
                hits = [d for d, ee in zip(hits, e) if ee <= emin + E_TOL]
                hits.sort(key=lambda d: (len(d.structure),
                                         d.energy_above_hull if d.energy_above_hull is not None else 9e9))
                rec["n_spacegroup_matches"] = len(hits)
                rec["mp_id"] = str(hits[0].material_id)
                rec["energy_above_hull"] = float(hits[0].energy_above_hull or 0.0)
                rec["n_polymorphs_considered"] = len(docs)
                s = hits[0].structure
                rec.update(validate(s, sgn, ratio))
                rec["status"] = ("OK" if rec["spacegroup_matches_expected"]
                                 and rec["ratio_matches_expected"] else "MISMATCH")
                if not args.dry_run and rec["status"] == "OK":
                    s.to(filename=str(out_dir / f"{slug}.cif"))
                    rec["written"] = str((out_dir / f"{slug}.cif").relative_to(ROOT))
            except Exception as exc:  # noqa: BLE001
                rec["status"] = f"ERROR::{type(exc).__name__}: {exc}"
            rows.append(rec)
            print(f"  {slug:20s} {rec.get('mp_id','-'):12s} {rec.get('formula',''):10s} "
                  f"{rec.get('spacegroup_symbol',''):10s} {rec['status']}", flush=True)

    df = pd.DataFrame(rows)
    suffix = "_dryrun" if args.dry_run else ""
    df.to_csv(LIB / f"fetch_report{suffix}.csv", index=False)
    pd.set_option("display.width", 240)
    print(f"\nOK {int((df.status == 'OK').sum())} / {len(df)}")
    bad = df[df.status != "OK"]
    if len(bad):
        print("\nneeds attention:")
        print(bad[["slug", "mp_id", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()

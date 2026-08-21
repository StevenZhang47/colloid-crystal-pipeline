#!/usr/bin/env python3
"""One table: every CIF, its built crystal, and whether the matcher can use it.

The pieces live in four places and nothing joined them:
  prototypes/<slug>.cif                          the deduplicated CIF
  ../crystal_stability/runs/lib_<slug>/          the built colloidal crystal
  generated/records/<slug>.json                  how the build went
  catalog.csv                                    crystallography
  ../../pkg/references/prototype_fingerprints.npz  what the matcher actually retrieves against

`usable` is the column that matters: a CIF can be present and still be unusable, either
because it cannot be parsed (coincident sites, placeholder species labels) or because the
colloid pair cannot realise it (the Laves phases need a radius ratio near 1.225 and the
fixed 102/120 nm pair gives 1.176).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

LIB = Path(__file__).resolve().parent
CS = LIB.parent / "crystal_stability"
BANK = LIB.parents[1] / "pkg/references/prototype_fingerprints.npz"


def main():
    cat = pd.read_csv(LIB / "catalog.csv")
    # The prototype grouping is read FROM THE MATCHER BANK, so this index and the thing
    # the matcher actually retrieves against cannot disagree.
    #
    # Do NOT use prototype_groups.csv's framework_group here. That column comes from
    # StructureMatcher with FrameworkComparator, which ignores species entirely and so
    # collapses every FCC superstructure onto plain FCC: it declares CsCl, Cu3Au, Al3Ti,
    # BiF3, CuAu and NaTl to be one prototype. Grouping Cu3Au with Al3Ti would erase the
    # single hardest discrimination in this library.
    in_bank, groups = set(), {}
    if BANK.is_file():
        z = np.load(BANK, allow_pickle=False)
        in_bank = {str(s) for s in z["names"]}
        groups = {str(a).lower(): str(b) for a, b in zip(z["names"], z["prototypes"])}

    rows = []
    for c in cat.itertuples(index=False):
        slug = str(c.slug)
        run = CS / "runs" / f"lib_{slug.lower()}"
        gsd = run / "first_frame.gsd"
        rec_p = LIB / "generated/records" / f"{slug}.json"
        rec = json.loads(rec_p.read_text()) if rec_p.is_file() else {}
        status = str(rec.get("status", "NOT_BUILT"))
        verdict = rec.get("verdict")

        if status.startswith("ERROR"):
            usable, why = False, status.replace("ERROR::", "")
        elif verdict == "cohesive":
            usable, why = True, ""
        elif verdict in ("suspect_dilute", "EXPLODED"):
            usable, why = False, f"{verdict} (nn/contact {rec.get('nn_over_contact')}); " \
                                 f"weakest species held at {rec.get('weakest_kT')} kT"
        else:
            usable, why = False, status

        rows.append({
            "slug": slug,
            "prototype_group": groups.get(slug.lower(), slug),
            "spacegroup": c.spacegroup_symbol, "crystal_system": c.crystal_system,
            "large_small_ratio": c.large_small_ratio, "n_sites": c.n_sites,
            "requires_anisotropic": c.requires_anisotropic,
            "cif": f"cif_library/prototypes/{slug}.cif",
            "built_crystal": str(gsd.relative_to(LIB.parents[1])) if gsd.is_file() else "",
            "n_particles": rec.get("n_particles"),
            "nn_over_contact": rec.get("nn_over_contact"),
            "weakest_kT": rec.get("weakest_kT"),
            "build_verdict": verdict or status,
            "usable": usable,
            "unusable_reason": why,
            "in_matcher_bank": slug.lower() in {b.lower() for b in in_bank},
        })

    df = pd.DataFrame(rows).sort_values(["usable", "prototype_group", "slug"],
                                        ascending=[False, True, True])
    df.to_csv(LIB / "LIBRARY_INDEX.csv", index=False)
    print(f"{len(df)} CIFs   usable {int(df.usable.sum())}   "
          f"in matcher bank {int(df.in_matcher_bank.sum())}\n")
    print("NOT USABLE:")
    for r in df[~df.usable].itertuples(index=False):
        print(f"  {r.slug:30s} {r.unusable_reason[:78]}")
    print(f"\nwrote {LIB/'LIBRARY_INDEX.csv'}")


if __name__ == "__main__":
    main()

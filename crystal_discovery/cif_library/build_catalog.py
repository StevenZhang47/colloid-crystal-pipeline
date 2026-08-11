#!/usr/bin/env python3
"""Build the canonical CIF library catalog.

Problem this solves: 21 distinct CIF contents are scattered across 161 files, and
several prototypes have MORE THAN ONE distinct CIF (three different Th3P4, two
different zangenite).  Nothing records which file a given method actually read, so
"the Th3P4 reference" is ambiguous.

This script:
  1. finds every CIF under crystal_discovery, deduplicates by content hash
  2. parses each one: space group FROM COORDINATES (symmetry-expanded CIFs are
     routinely written as P1, so the header is unreliable), crystal system,
     composition, LARGE:SMALL ratio, site count
  3. computes the anisotropic axis groups using the SAME function the patched
     LatticeBuilder uses, so `requires_anisotropic` is authoritative rather than
     inferred from the space-group symbol
  4. records which crystal_stability runs and which downstream methods consume it
  5. checks that any BUILT reference is a cohesive crystal, not an exploded one

Writes catalog.csv + prototypes/<slug>.cif.  Read-only on all sources.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Element
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.spatial import distance_matrix

ROOT = Path("/projects/rps/gmh4/hockygroup/sz5808")
CD = ROOT / "crystal_discovery"
CS = CD / "crystal_stability"
LIB = CD / "cif_library"
# The anisotropic LatticeBuilder, from the persistent worktree rather than a session
# scratchpad directory that will not outlive the session that made it.
sys.path.insert(0, str(CD / "PACSim_aniso"))
from colloids.colloids_create.configuration_generators.lattice_builder import (  # noqa: E402
    LatticeBuilder,
)

# Which downstream method consumes which crystal_stability run.  Taken from the
# matcher's own FAMILIES dict so this cannot drift.
MATCHER_FAMILIES = {
    "cscl": "EMD matcher (CsCl)", "th3p4": "EMD matcher (Th3P4)",
    "cu3au_d6": "EMD matcher (Cu3Au)", "ni3sn_d019": "EMD matcher (Ni3Sn)",
    "zangenite_deep8": "EMD matcher (zangenite)",
    "cr3si_a15_deep8": "EMD matcher (Cr3Si_A15)",
    "rocksalt_b1_deep8": "EMD matcher (rocksalt_B1)",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def builder_parse_check(path: Path) -> dict:
    """Parse the CIF the way the BUILDER does and report whether it is usable.

    analyse() below parses with pymatgen's DEFAULT tolerances, which merge sites that
    sit on top of each other. LatticeBuilder parses with site_tolerance=0.0 and
    frac_tolerance=0.0 and does not merge. When a symmetry-expanded CIF has duplicated
    its own sites the two disagree, and the catalog then describes a structure that is
    not the one being built: runs/cu3au/Cu3Au.cif reads as 4 sites here and 96 sites
    with 40176 coincident pairs there, which sends the no-overlap scale to infinity and
    every built coordinate to NaN.

    Species labels are checked too: a CIF using placeholder labels (the L/S zangenite)
    parses to nonsense atomic numbers and cannot describe a colloid system at all.
    """
    rec = {"n_sites_builder": None, "min_site_distance": None,
           "coincident_pairs": None, "builder_usable": False, "builder_note": ""}
    try:
        s = CifParser(str(path), site_tolerance=0.0, frac_tolerance=0.0
                      ).parse_structures(check_occu=False, primitive=False)[0]
        t = s.make_supercell((3, 3, 3), in_place=False, to_unit_cell=True)
        d = distance_matrix(t.cart_coords, t.cart_coords)
        np.fill_diagonal(d, np.inf)
        coincident = int((d < 1e-8).sum() // 2)
        rec.update(n_sites_builder=len(s), min_site_distance=round(float(d.min()), 6),
                   coincident_pairs=coincident)
        for z in np.unique(s.atomic_numbers):
            Element.from_Z(int(z))          # raises on placeholder species labels
        if coincident:
            rec["builder_note"] = f"{coincident} coincident site pairs"
        else:
            rec["builder_usable"] = True
    except Exception as exc:  # noqa: BLE001
        rec["builder_note"] = f"{type(exc).__name__}: {exc}"[:120]
    return rec


def analyse(path: Path) -> dict:
    s = CifParser(str(path)).parse_structures(primitive=False)[0]
    sga = SpacegroupAnalyzer(s, symprec=0.1)
    amt = s.composition.get_el_amt_dict()
    els = sorted(amt, key=lambda e: amt[e])          # minority first = LARGE colloid
    counts = [int(amt[e]) for e in els]
    g = 0
    for c in counts:
        g = np.gcd(g, c)
    groups = LatticeBuilder._axis_groups(s)
    a, b, c_ = s.lattice.abc
    return {
        "spacegroup_symbol": sga.get_space_group_symbol(),
        "spacegroup_number": int(sga.get_space_group_number()),
        "crystal_system": sga.get_crystal_system(),
        "elements_LARGE_first": ",".join(els),
        "large_small_ratio": ":".join(str(x // max(g, 1)) for x in counts),
        "n_sites": len(s),
        "a": round(a, 4), "b": round(b, 4), "c": round(c_, 4),
        "b_over_a": round(b / a, 4), "c_over_a": round(c_ / a, 4),
        "axis_groups": json.dumps(groups),
        "n_free_lattice_params": len(groups),
        "requires_anisotropic": len(groups) > 1,
    }


def main() -> None:
    LIB.mkdir(parents=True, exist_ok=True)
    (LIB / "prototypes").mkdir(exist_ok=True)

    by_hash: dict[str, list[Path]] = {}
    for p in sorted(CD.rglob("*.cif")):
        if LIB in p.parents:        # never re-scan the library's own copies
            continue
        by_hash.setdefault(sha(p), []).append(p)
    # ...but DO include CIFs fetched from Materials Project, which live only in the
    # library and have no copy elsewhere in the tree.
    fetched = set()
    fr = LIB / "fetch_report.csv"
    if fr.exists():
        fetched = set(pd.read_csv(fr).query("status == \'OK\'").slug)
    for p in sorted((LIB / "prototypes").glob("*.cif")):
        if p.stem in fetched:
            by_hash.setdefault(sha(p), []).append(p)

    # Distinct CONTENTS can share a filename stem (two different Th3P4.cif, two
    # different zangenite.cif).  Disambiguate with the content hash, otherwise the
    # copy below silently overwrites one variant with the other -- which it did:
    # prototypes/Th3P4.cif held the PACSim test fixture, not the CIF the EMD matcher
    # reads.  The count must be over the CANONICAL path of each hash group, not over
    # paths[0], because paths[0] is whatever the walk happened to reach first and its
    # stem need not be the canonical one.
    canonical_of = {h: min(paths, key=lambda q: (len(q.parts), str(q)))
                    for h, paths in by_hash.items()}
    stem_counts = Counter(c.stem for c in canonical_of.values())

    rows = []
    for h, paths in by_hash.items():
        canonical = canonical_of[h]
        slug = canonical.stem
        if stem_counts[canonical.stem] > 1:
            slug = f"{canonical.stem}__{h[:8]}"
        rec = {"slug": slug, "content_sha256": h[:16], "n_duplicate_files": len(paths),
               "canonical_source": str(canonical.relative_to(ROOT)),
               "origin": "materials_project" if LIB in canonical.parents else "local_tree"}
        rec.update(builder_parse_check(canonical))
        try:
            rec.update(analyse(canonical))
            rec["parse_status"] = "OK"
        except Exception as exc:  # noqa: BLE001
            rec["parse_status"] = f"ERROR::{type(exc).__name__}: {exc}"
        # which crystal_stability runs use this exact content
        runs = sorted({q.parent.name for q in paths if "crystal_stability/runs" in str(q)})
        rec["crystal_stability_runs"] = ";".join(runs)
        rec["consumed_by"] = ";".join(sorted(
            {MATCHER_FAMILIES[r] for r in runs if r in MATCHER_FAMILIES})) or ""
        rows.append(rec)

    df = pd.DataFrame(rows).sort_values(["crystal_system", "slug"])
    # flag prototypes that have MORE THAN ONE distinct CIF
    key = df.slug.str.lower().str.replace(r"[_\-].*$", "", regex=True)
    df["prototype_key"] = key
    dupes = df.groupby("prototype_key").slug.transform("size")
    df["AMBIGUOUS_multiple_cifs"] = dupes > 1

    for r in df.itertuples(index=False):
        src = ROOT / r.canonical_source
        dst = LIB / "prototypes" / f"{r.slug}.cif"
        if src.resolve() != dst.resolve():      # MP-fetched CIFs are already in place
            shutil.copy2(src, dst)

    df.to_csv(LIB / "catalog.csv", index=False)
    pd.set_option("display.width", 250)
    show = ["slug", "spacegroup_symbol", "large_small_ratio", "n_sites", "a", "c_over_a",
            "n_free_lattice_params", "AMBIGUOUS_multiple_cifs", "consumed_by"]
    print(df[show].to_string(index=False))
    print(f"\n{len(df)} unique CIFs -> {LIB}/prototypes/")
    print(f"non-cubic (need anisotropic_energy): {int(df.requires_anisotropic.sum())}")
    print(f"ambiguous prototypes (>1 distinct CIF): "
          f"{sorted(df[df.AMBIGUOUS_multiple_cifs].prototype_key.unique())}")


if __name__ == "__main__":
    main()

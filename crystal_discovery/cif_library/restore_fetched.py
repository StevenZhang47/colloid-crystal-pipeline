#!/usr/bin/env python3
"""Re-fetch the Materials Project CIFs BY the mp-id already recorded in fetch_report.csv.

These 21 CIFs exist nowhere else in the tree, so prototypes/ is their only copy.
Restoring by recorded mp-id (rather than re-running the formula->polymorph selection
in fetch_targets.py) makes the restore exact and independent of any later change to
the selection rule. Each restored file is re-validated against the space group and
stoichiometry recorded at fetch time before it is written.
"""
import os, sys
import pandas as pd
from pathlib import Path
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

LIB = Path(__file__).resolve().parent
key = os.environ["MP_API_KEY"]
rep = pd.read_csv(LIB / "fetch_report.csv").query("status == 'OK'")
from mp_api.client import MPRester

ok = bad = 0
with MPRester(key) as mpr:
    for r in rep.itertuples(index=False):
        doc = mpr.materials.summary.search(material_ids=[r.mp_id],
                                           fields=["material_id", "structure", "symmetry"])[0]
        s = doc.structure
        sgn = int(SpacegroupAnalyzer(s, symprec=0.1).get_space_group_number())
        if sgn != int(r.spacegroup_number) or len(s) != int(r.n_sites):
            print(f"  MISMATCH {r.slug}: sg {sgn} vs {r.spacegroup_number}, "
                  f"{len(s)} vs {r.n_sites} sites"); bad += 1; continue
        s.to(filename=str(LIB / "prototypes" / f"{r.slug}.cif"))
        print(f"  {r.slug:24s} {r.mp_id:10s} sg={sgn} sites={len(s)} OK"); ok += 1
print(f"\nrestored {ok}, mismatched {bad}, of {len(rep)}")
sys.exit(1 if bad else 0)

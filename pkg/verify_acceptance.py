#!/usr/bin/env python3
"""Re-run the acceptance test THROUGH THE SHIPPED CODE PATH.

acceptance_test.py proved the method works. This proves the package implements it. They
are not the same claim: the research script and `crystal_discovery.pipeline` find domains
by slightly different rules (the package requires min_centres=64 and reports every
component; the research script took the largest with a floor of 25), so the packaged
answer has to be checked rather than assumed.

Pass condition, on the 50 held-out validation trajectories plus the two externals:

    every CsCl trajectory       -> retrieved == "CsCl"
    every Th3P4 trajectory      -> retrieved == "Th3P4"
    every amorphous trajectory  -> retrieved is None   (gates reject it)
    steven_example_1/2          -> reported, ground truth is sealed

  python verify_acceptance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pkg/src"))

from crystal_discovery.config import Config      # noqa: E402
from crystal_discovery.pipeline import analyze   # noqa: E402

UP = REPO / "crystal_discovery/preprocessing_pipeline_final/unified_pipeline"
EXTERNAL = {"steven_example_1": UP / "data/test_set/steven_example_1.gsd",
            "steven_example_2": UP / "data/test_set/steven_example_2.gsd"}


def main() -> None:
    cfg = Config.load()
    pop = pd.read_csv(REPO / "DATA/detector_trajectories.csv")
    pop = pop[(pop.dataset == "detector_atlas") & (pop.role == "validation")]
    cases = [(r.trajectory_id, r.label, Path(r.path)) for r in pop.itertuples()]
    cases += [(k, "external", v) for k, v in EXTERNAL.items() if v.is_file()]

    rows = []
    for tid, label, path in cases:
        if not path.is_file():
            print(f"  [missing] {tid}", flush=True); continue
        rep = analyze(path, frame=-1, config=cfg, min_centres=64, top_k=5)
        dom = max(rep.domains, key=lambda d: d.n_centres, default=None)
        if dom is None or dom.retrieval is None:
            rows.append(dict(trajectory=tid, label=label, retrieved=None,
                             rank1=None, distance=None, margin=None,
                             rho=None, note=(dom.error if dom else "no domain")))
            print(f"  {label:10s} {tid[:48]:48s} NO DOMAIN", flush=True); continue
        r = dom.retrieval
        ok = ((label in ("CsCl", "Th3P4") and r.retrieved == label)
              or (label == "amorphous" and r.retrieved is None))
        rows.append(dict(trajectory=tid, label=label, retrieved=r.retrieved,
                         rank1=r.ranking[0].prototype, distance=r.ranking[0].distance,
                         margin=r.margin, rho=dom.ood.rho if dom.ood else None,
                         top5="|".join(m.prototype for m in r.ranking), correct=ok))
        print(f"  {label:10s} {tid[:44]:44s} -> {str(r.retrieved):10s} "
              f"(rank1 {r.ranking[0].prototype}, D={r.ranking[0].distance:.4f}, "
              f"margin={r.margin}) {'ok' if ok else 'MISMATCH' if label!='external' else ''}",
              flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(REPO / "pkg/verify_acceptance.csv", index=False)

    print("\n" + "=" * 78)
    ok_all = True
    for lab, want in (("CsCl", "CsCl"), ("Th3P4", "Th3P4"), ("amorphous", None)):
        s = d[d.label == lab]
        n = int((s.retrieved == want).sum()) if want else int(s.retrieved.isna().sum())
        ok_all &= n == len(s)
        print(f"  {lab:10s} {n}/{len(s)} "
              f"{'-> ' + want if want else 'correctly rejected'}"
              f"   {'PASS' if n == len(s) else 'FAIL'}")
    print("\n  external candidates (ground truth sealed in library_stability.csv):")
    for r in d[d.label == "external"].itertuples():
        print(f"    {r.trajectory:20s} -> {str(r.retrieved):10s} D={r.distance} "
              f"margin={r.margin}\n      top-5: {r.top5}")
    print("\n" + ("ACCEPTANCE TEST PASSES through the packaged code"
                  if ok_all else "ACCEPTANCE TEST FAILS -- package does not reproduce"))
    (REPO / "pkg/verify_acceptance.json").write_text(json.dumps(
        {"passes": bool(ok_all), "n": len(d)}, indent=2) + "\n")


if __name__ == "__main__":
    main()

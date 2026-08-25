#!/usr/bin/env python3
"""Find the trajectories that sit closest to the Th3P4 / amorphous decision boundary.

The validation split happens to contain only easy Th3P4 cases -- distances 0.071 to 0.081
against a 0.1619 gate, so none of them stresses the matcher. This scans every Th3P4 and
amorphous trajectory in the whole detector atlas and reports the hardest cases in both
directions:

  the Th3P4 with the LARGEST distance      most amorphous-looking crystal
  the amorphous with the SMALLEST distance  most crystal-looking disorder

Those two lists bracket the gate, and the margin between them is the real safety factor.

The sealed untouched-test groups are excluded automatically: they carry dataset ==
"untouched_test" in the manifest and this filters to "detector_atlas".
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path("/projects/rps/gmh4/hockygroup/sz5808")
sys.path.insert(0, str(REPO / "pkg/src"))

from crystal_discovery.config import Config      # noqa: E402
from crystal_discovery.pipeline import analyze   # noqa: E402


def main() -> None:
    cfg = Config.load()
    d = pd.read_csv(REPO / "DATA/detector_trajectories.csv")
    d = d[(d.dataset == "detector_atlas") & (d.label.isin(("Th3P4", "amorphous")))]
    print(f"{len(d)} trajectories to scan\n", flush=True)

    rows = []
    for i, r in enumerate(d.itertuples(), 1):
        p = Path(r.path)
        if not p.is_file():
            continue
        try:
            rep = analyze(p, frame=-1, config=cfg, min_centres=64, top_k=3)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {r.trajectory_id}: {exc}", flush=True); continue
        dom = max(rep.domains, key=lambda x: x.n_centres, default=None)
        if dom is None or dom.retrieval is None:
            continue
        q = dom.retrieval
        rows.append(dict(label=r.label, trajectory=r.trajectory_id,
                         group=r.condition_group_id, role=r.role, path=str(p),
                         rank1=q.ranking[0].prototype, distance=q.ranking[0].distance,
                         margin=q.margin, retrieved=q.retrieved,
                         rho=dom.ood.rho if dom.ood else None,
                         n_centres=dom.n_centres))
        if i % 20 == 0:
            print(f"  {i}/{len(d)}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(REPO / "results/statistical_ood_benchmark_v2/"
                      "reference_conditioned_matcher_v1/boundary_scan.csv", index=False)

    d_tau = float(cfg.retrieval["distance_threshold"])
    print(f"\n{'='*100}\nHARDEST Th3P4 (most amorphous-looking) -- gate is D <= {d_tau}\n{'='*100}")
    t = out[out.label == "Th3P4"].nlargest(10, "distance")
    for r in t.itertuples():
        print(f"  D={r.distance:.4f} margin={r.margin:6.3f} rho={r.rho:5.3f} "
              f"-> {str(r.retrieved):8s} [{r.role}] {r.trajectory}\n      {r.path}")
    print(f"\n{'='*100}\nHARDEST amorphous (most crystal-looking)\n{'='*100}")
    a = out[out.label == "amorphous"].nsmallest(10, "distance")
    for r in a.itertuples():
        print(f"  D={r.distance:.4f} margin={r.margin:6.3f} rho={r.rho:5.3f} "
              f"-> {str(r.retrieved):8s} [{r.role}] {r.trajectory}\n      {r.path}")

    th = out[out.label == "Th3P4"]; am = out[out.label == "amorphous"]
    print(f"\nTh3P4     n={len(th):3d}  D {th.distance.min():.4f}-{th.distance.max():.4f}  "
          f"accepted as Th3P4 {int((th.retrieved=='Th3P4').sum())}/{len(th)}")
    print(f"amorphous n={len(am):3d}  D {am.distance.min():.4f}-{am.distance.max():.4f}  "
          f"accepted as anything {int(am.retrieved.notna().sum())}/{len(am)}")
    print(f"\nboundary: worst Th3P4 {th.distance.max():.4f} | "
          f"best amorphous {am.distance.min():.4f} | gap {am.distance.min()/th.distance.max():.2f}x")


if __name__ == "__main__":
    main()

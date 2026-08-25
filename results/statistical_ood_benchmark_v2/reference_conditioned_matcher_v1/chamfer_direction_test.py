#!/usr/bin/env python3
"""Is the one-sided chamfer actually better than the symmetric one? Measure both.

  forward  C -> A   for each CANDIDATE particle, distance to its nearest ANCHOR particle.
                    Asks: is every environment in my candidate explained by this crystal?
  reverse  A -> C   for each ANCHOR particle, distance to its nearest CANDIDATE particle.
                    Asks: are this crystal's sites represented in my candidate?
  symmetric         forward + reverse, the textbook Chamfer distance.

The claim under test is that `reverse` is not merely uninformative but actively harmful,
because a diffuse amorphous cloud covers a large region of SOAP space and therefore has
SOME particle near almost any environment you name -- it passes the reverse test trivially,
and being more disordered makes it pass more easily.

Scored the way the matcher scores: rank-1 distance is the MINIMUM over all anchors.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

R = Path("/projects/rps/gmh4/hockygroup/sz5808")
HERE = R / "results/statistical_ood_benchmark_v2/reference_conditioned_matcher_v1"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(R / "results/motif_pipeline/soap_atlas_core3_phase2_v1/scripts"))
from acceptance_test import soap_of, last_frame_index          # noqa: E402
from matcher_benchmark import SUBSAMPLE, anchor_sources, _sqd  # noqa: E402
from soap_vs_emd_matcher import prototype_of                   # noqa: E402


def main() -> None:
    rng = np.random.default_rng(0)
    anchors = {}
    for name, (path, fi) in anchor_sources("cold").items():
        try:
            r = soap_of(path, fi, "frozen", True)
        except Exception:  # noqa: BLE001
            continue
        if r is None:
            continue
        anchors[name] = [x.astype(np.float32) if len(x) <= SUBSAMPLE
                         else x[rng.choice(len(x), SUBSAMPLE, replace=False)].astype(np.float32)
                         for x in r]
    print(f"{len(anchors)} anchors", flush=True)

    pop = pd.read_csv(R / "DATA/detector_trajectories.csv")
    pop = pop[(pop.dataset == "detector_atlas") & (pop.role == "validation")]
    rows = []
    for lab in ("CsCl", "Th3P4", "amorphous"):
        for p in pop[pop.label == lab].path:
            p = Path(p)
            c = soap_of(p, last_frame_index(p), "frozen", False)
            if c is None:
                continue
            c = [x.astype(np.float32) for x in c]
            F, B, S = {}, {}, {}
            for an, a in anchors.items():
                f = b = 0.0
                for s in (0, 1):
                    d = np.sqrt(np.maximum(_sqd(c[s], a[s]), 0))
                    f += float(d.min(axis=1).mean()); b += float(d.min(axis=0).mean())
                pr = prototype_of(an)
                F[pr] = min(F.get(pr, 9e9), f / 2)
                B[pr] = min(B.get(pr, 9e9), b / 2)
                S[pr] = min(S.get(pr, 9e9), (f + b) / 4)
            rows.append(dict(label=lab, trajectory=p.parent.name,
                             fwd=min(F.values()), fwd_pick=min(F, key=F.get),
                             rev=min(B.values()), rev_pick=min(B, key=B.get),
                             sym=min(S.values()), sym_pick=min(S, key=S.get)))
            print(f"  {lab:10s} {rows[-1]['trajectory'][:44]:44s} "
                  f"fwd {rows[-1]['fwd']:.4f} rev {rows[-1]['rev']:.4f}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(HERE / "chamfer_direction_test.csv", index=False)
    print("\nmean rank-1 distance by label")
    print(d.groupby("label")[["fwd", "rev", "sym"]].mean().round(4).to_string())
    print("\nrank-1 accuracy")
    for lab in ("CsCl", "Th3P4"):
        s = d[d.label == lab]
        print(f"  {lab:10s} " + "   ".join(
            f"{k} {int((s[k + '_pick'] == lab).sum())}/{len(s)}" for k in ("fwd", "rev", "sym")))
    print("\nseparation (the acceptance criterion)")
    for col in ("fwd", "rev", "sym"):
        cry = d[d.label != "amorphous"][col].max()
        amo = d[d.label == "amorphous"][col].min()
        print(f"  {col}: max crystal {cry:.4f}  min amorphous {amo:.4f}  gap {amo/cry:.2f}x  "
              f"{'SEPARATES' if amo > cry else 'OVERLAPS'}")


if __name__ == "__main__":
    main()

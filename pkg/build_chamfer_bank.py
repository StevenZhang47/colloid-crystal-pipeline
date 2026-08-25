#!/usr/bin/env python3
"""Build the reference bank the chamfer matcher needs.

The mean-SOAP matcher only had to ship one 1904-D vector per anchor. Chamfer compares
per-particle environments, so it needs the anchor's whole point cloud -- 41 anchors x
~5000 particles x 952 dimensions is ~800 MB, which cannot live in a git repository.

It does not have to. An ideal lattice has very few DISTINCT environments: every particle
on the same Wyckoff site has the same neighbourhood, so the cloud is a handful of tight
knots plus a surface shell. A greedy cover at radius COVER_R keeps one representative per
knot and discards the duplicates -- on lib_cscl, 2744 particles collapse to 55.

Because the discarded points are all within COVER_R of a kept one, the chamfer distance
computed on the compressed bank can exceed the true value by at most COVER_R, and in
practice by far less because the minimum is taken over many representatives. That claim
is not assumed: verify_chamfer_bank.py recomputes the acceptance test both ways.

  python build_chamfer_bank.py [--out references/chamfer_bank.npz]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "results/statistical_ood_benchmark_v2/reference_conditioned_matcher_v1"))
sys.path.insert(0, str(REPO / "results/motif_pipeline/soap_atlas_core3_phase2_v1/scripts"))

from acceptance_test import soap_of                      # noqa: E402
from matcher_benchmark import anchor_sources             # noqa: E402
from soap_vs_emd_matcher import prototype_of             # noqa: E402

COVER_R = 0.02        # cover radius, ~15% of the 0.13 decision scale
MIN_ENV = 4           # an anchor with fewer distinct environments than this is suspect


def greedy_cover(X: np.ndarray, r: float) -> np.ndarray:
    """Indices of a subset such that every row of X lies within r of one of them."""
    keep = [0]
    d = np.linalg.norm(X - X[0], axis=1)
    while True:
        i = int(np.argmax(d))
        if d[i] <= r:
            break
        keep.append(i)
        d = np.minimum(d, np.linalg.norm(X - X[i], axis=1))
    return np.asarray(keep, dtype=int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "pkg/references/chamfer_bank.npz")
    ap.add_argument("--cover-r", type=float, default=COVER_R)
    a = ap.parse_args()

    store, meta = {}, []
    for name, (path, fi) in sorted(anchor_sources("cold").items()):
        try:
            z = soap_of(path, fi, "frozen", is_anchor=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {name}: {type(exc).__name__}: {exc}", flush=True); continue
        if z is None:
            print(f"  [skip] {name}: too few centres", flush=True); continue
        kept = []
        for s in (0, 1):
            idx = greedy_cover(z[s], a.cover_r)
            # float16 halves the artefact to ~17 MB. Components of an L2-normalised
            # 952-D vector are ~0.03, so the rounding error is ~1e-3 in L2 --
            # three orders below the 0.13 decision scale.
            store[f"{name}::{s}"] = z[s][idx].astype(np.float16)
            kept.append(len(idx))
        if min(kept) < MIN_ENV:
            print(f"  [warn] {name}: only {kept} distinct environments", flush=True)
        meta.append({"anchor": name, "prototype": prototype_of(name),
                     "n_particles": [int(len(z[0])), int(len(z[1]))],
                     "n_environments": [int(kept[0]), int(kept[1])]})
        print(f"  {name:44s} {len(z[0]):5d}/{len(z[1]):5d} -> {kept[0]:4d}/{kept[1]:4d}",
              flush=True)

    store["__meta__"] = np.frombuffer(
        json.dumps({"cover_r": a.cover_r, "source": "cold CIF-ideal anchors, unpreprocessed",
                    "anchors": meta}).encode(), dtype=np.uint8)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **store)
    mb = a.out.stat().st_size / 1e6
    tot = sum(m["n_environments"][0] + m["n_environments"][1] for m in meta)
    print(f"\n{len(meta)} anchors, {tot} environments, {mb:.1f} MB -> {a.out}")


if __name__ == "__main__":
    main()

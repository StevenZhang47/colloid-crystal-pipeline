#!/usr/bin/env python3
"""Rebuild references/chamfer_bank.npz from the CIF-ideal anchor lattices.

SELF-CONTAINED. This imports only `crystal_discovery`, never the research scripts under
results/. It used to reach into acceptance_test -> matcher_benchmark -> soap_vs_emd_matcher
-> build_and_run_matcher, which meant regenerating a SHIPPED artifact pulled in the entire
superseded EMD matcher. A package whose artifacts cannot be rebuilt without the research
tree is not really a package.

WHY THE BANK STORES ENVIRONMENTS, NOT PARTICLES
-----------------------------------------------
Chamfer compares per-particle environments, so it needs each anchor's whole point cloud --
41 anchors x ~5000 particles x 952 dimensions is ~800 MB, which cannot live in a git
repository.

It does not have to. An ideal lattice has very few DISTINCT environments: every particle on
the same Wyckoff site has the same neighbourhood, so the cloud is a handful of tight knots
plus a surface shell. A greedy cover at COVER_R keeps one representative per knot and drops
the duplicates -- lib_cscl's 2744 particles collapse to 55. Because every discarded point
lies within COVER_R of a kept one, the chamfer distance computed on the compressed bank can
exceed the true value by at most COVER_R, and in practice by far less since the minimum is
taken over many representatives. That is checked, not assumed: verify_acceptance.py reruns
the acceptance test through the shipped path and must still score 13/13, 11/11, 26/26.

  python pkg/build_chamfer_bank.py [--runs DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pkg/src"))

from crystal_discovery.descriptors import anchor_environments   # noqa: E402
from crystal_discovery.prototypes import prototype_of           # noqa: E402

DEFAULT_RUNS = REPO / "crystal_discovery/crystal_stability/runs"
COVER_R = 0.02        # ~15% of the 0.13 decision scale
MIN_ENV = 4           # fewer distinct environments than this is suspect, so say so
MIN_PER_SPECIES = 25


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
    ap.add_argument("--runs", type=Path, default=DEFAULT_RUNS,
                    help="directory of lib_*/first_frame.gsd anchor lattices")
    ap.add_argument("--out", type=Path, default=REPO / "pkg/references/chamfer_bank.npz")
    ap.add_argument("--cover-r", type=float, default=COVER_R)
    a = ap.parse_args()

    store, meta = {}, []
    for run in sorted(a.runs.glob("lib_*")):
        gsd = run / "first_frame.gsd"
        if not gsd.is_file():
            continue
        name = run.name[4:]
        try:
            z = anchor_environments(gsd, 0)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {name}: {type(exc).__name__}: {exc}", flush=True); continue
        if min(len(z[0]), len(z[1])) < MIN_PER_SPECIES:
            print(f"  [skip] {name}: too few particles per species", flush=True); continue

        kept = []
        for s in (0, 1):
            idx = greedy_cover(z[s], a.cover_r)
            # float16 halves the artifact to ~17 MB. Components of an L2-normalised 952-D
            # vector are ~0.03, so the rounding error is ~1e-3 in L2 -- two orders below
            # the 0.13 decision scale.
            store[f"{name}::{s}"] = z[s][idx].astype(np.float16)
            kept.append(len(idx))
        if min(kept) < MIN_ENV:
            print(f"  [warn] {name}: only {kept} distinct environments", flush=True)
        meta.append({"anchor": name, "prototype": prototype_of(name),
                     "n_particles": [len(z[0]), len(z[1])],
                     "n_environments": [int(kept[0]), int(kept[1])]})
        print(f"  {name:44s} {len(z[0]):5d}/{len(z[1]):5d} -> {kept[0]:4d}/{kept[1]:4d}",
              flush=True)

    if not meta:
        raise SystemExit(f"no anchors found under {a.runs}")
    store["__meta__"] = np.frombuffer(
        json.dumps({"cover_r": a.cover_r, "source": "CIF-ideal anchors, unpreprocessed",
                    "anchors": meta}).encode(), dtype=np.uint8)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **store)
    tot = sum(m["n_environments"][0] + m["n_environments"][1] for m in meta)
    print(f"\n{len(meta)} anchors, {tot} environments, "
          f"{a.out.stat().st_size / 1e6:.1f} MB -> {a.out}")


if __name__ == "__main__":
    main()

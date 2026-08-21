#!/usr/bin/env python3
"""Build the prototype fingerprint bank the CLI retrieves against.

One 1904-D species-resolved SOAP fingerprint per cohesive library prototype, computed
through the SAME frozen descriptor path the detector uses. Anchors are CIF-ideal lattices
with no gas and no thermal disorder, so they bypass the gas screen and M3 -- running a
gas screen on a freshly built lattice only produces false positives, and on open
frameworks like ReO3 it deletes the entire majority sublattice.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/projects/rps/gmh4/hockygroup/sz5808")
sys.path.insert(0, str(ROOT / "results/statistical_ood_benchmark_v2/reference_conditioned_matcher_v1"))
import soap_vs_emd_matcher as S  # noqa: E402

OUT = Path(__file__).resolve().parent / "references"


def main():
    OUT.mkdir(exist_ok=True)
    admissible = S.cohesive_anchor_runs()
    names, vecs, protos = [], [], []
    for run_dir in sorted((S.CS / "runs").glob("lib_*")):
        if admissible and run_dir.name not in admissible:
            continue
        gsd = run_dir / "first_frame.gsd"
        if not gsd.is_file():
            continue
        slug = run_dir.name[len("lib_"):]
        try:
            fp = S.fingerprint(gsd, 0, is_anchor=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {slug}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if fp is None:
            continue
        names.append(slug)
        protos.append(S.prototype_of(slug))
        vecs.append(fp)
        print(f"  {slug:44s} -> {protos[-1]}", flush=True)

    X = np.vstack(vecs)
    np.savez(OUT / "prototype_fingerprints.npz",
             names=np.array(names), prototypes=np.array(protos), fingerprints=X)
    (OUT / "canonical_prototypes.json").write_text(json.dumps(
        {"n_anchors": len(names), "dimension": int(X.shape[1]),
         "anchors": [{"slug": n, "prototype": p} for n, p in zip(names, protos)]}, indent=2) + "\n")
    print(f"\n{len(names)} anchors, {X.shape[1]}-D -> {OUT/'prototype_fingerprints.npz'}")


if __name__ == "__main__":
    main()

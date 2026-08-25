#!/usr/bin/env python3
"""Where does analysis time go, and what can be recovered without changing the answer?

Levers tested:
  threads      BLAS thread count. Retrieval is two large GEMMs, so this is free speed --
               no approximation at all.
  subsample    cap on candidate centres. The score is a MEDIAN over centres, which is a
               robust statistic: 200 centres estimate it nearly as well as 2000. This is
               an approximation, so it is only worth taking if the reported distance and
               the ranking are unchanged. That is checked, not assumed.
"""
from __future__ import annotations
import os, sys, time
THREADS = int(os.environ.get("PROFILE_THREADS", "8"))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = str(THREADS)

from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pkg/src"))

t_imp = time.perf_counter()
from crystal_discovery.config import Config
from crystal_discovery.descriptors import find_domains
from crystal_discovery.ood import load_model, score_domain
from crystal_discovery.retrieval import load_bank, retrieve, _split_by_species
t_imp = time.perf_counter() - t_imp

FILES = {
    "small (222 centres)":  "crystal_discovery/preprocessing_pipeline_final/unified_pipeline/data/test_set/steven_example_1.gsd",
    "medium (525 centres)": "crystal_discovery/preprocessing_pipeline/nicoles_dataset/phase_diagram/debye5.45_volumefraction.002704544521_seed2/trajectory.gsd",
    "large (4393 centres)": "crystal_discovery/crystal_stability/runs/lib_cscl/first_frame.gsd",
}
ROOT = Path("/projects/rps/gmh4/hockygroup/sz5808")


def main() -> None:
    print(f"threads = {THREADS}   import = {t_imp:.1f} s\n")
    cfg = Config.load()
    t = time.perf_counter(); model = load_model(); bank = load_bank()
    print(f"load model + bank : {time.perf_counter()-t:.2f} s "
          f"({len(bank['envs'])} anchors, {sum(len(v[0])+len(v[1]) for v in bank['envs'].values()):,} envs)\n")

    print(f"{'case':22s} {'read+prep+SOAP':>15s} {'OOD':>7s} {'retrieve':>9s} {'TOTAL':>8s}")
    print("-" * 66)
    doms = {}
    for label, rel in FILES.items():
        p = ROOT / rel
        if not p.is_file():
            print(f"{label:22s} MISSING"); continue
        fi = 0 if "first_frame" in rel else -1
        t = time.perf_counter(); d = find_domains(p, fi, min_centres=64)[0]; t_desc = time.perf_counter()-t
        t = time.perf_counter(); score_domain(d, model); t_ood = time.perf_counter()-t
        t = time.perf_counter(); r = retrieve(d, cfg, top_k=5); t_ret = time.perf_counter()-t
        doms[label] = (d, r)
        print(f"{label:22s} {t_desc:14.2f}s {t_ood:6.2f}s {t_ret:8.2f}s "
              f"{t_desc+t_ood+t_ret:7.2f}s")

    # ---- does capping candidate centres change the answer? --------------------------
    print(f"\n{'='*66}\nSUBSAMPLING candidate centres (median is robust; does it hold?)\n{'='*66}")
    rng = np.random.default_rng(0)
    for label, (d, ref) in doms.items():
        print(f"\n{label}   full: {ref.ranking[0].prototype} D={ref.ranking[0].distance:.4f} "
              f"margin={ref.margin}")
        for cap in (1000, 400, 200, 100):
            if d.n_centres <= cap:
                print(f"   cap {cap:5d}: (already smaller)"); continue
            idx = rng.choice(d.n_centres, cap, replace=False)
            class Sub:  # a view with fewer centres
                soap = d.soap[idx]; species = d.species[idx]
            t = time.perf_counter()
            try:
                r = retrieve(Sub, cfg, top_k=5)
            except ValueError as exc:
                print(f"   cap {cap:5d}: {exc}"); continue
            dt = time.perf_counter() - t
            same = r.ranking[0].prototype == ref.ranking[0].prototype
            print(f"   cap {cap:5d}: {r.ranking[0].prototype:12s} D={r.ranking[0].distance:.4f} "
                  f"(delta {r.ranking[0].distance-ref.ranking[0].distance:+.4f})  "
                  f"margin={r.margin:6.2f}  {dt:.2f}s  rank1 {'same' if same else 'CHANGED'}")


if __name__ == "__main__":
    main()

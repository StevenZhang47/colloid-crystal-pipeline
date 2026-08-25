#!/usr/bin/env python3
"""Would capping SOAP centres speed things up without changing the answer?

On a large domain, read+preprocess+SOAP is ~2.7 s of a ~3.1 s analysis. Computing SOAP on
a random subset of centres would cut that nearly linearly. The retrieval score is a MEDIAN
and tolerates it (measured: 400 centres shift D by <0.0005 and never change rank 1).

But the OOD fingerprint is a MEAN of PCA-projected SOAP per species, fed to a FROZEN
Mahalanobis model calibrated on full domains. A mean over fewer centres is noisier, and
rho is compared against a fixed tau, so the question is whether the shift is small against
the decision it drives. That is what this measures -- on trajectories spanning the whole
rho range, including one that sits near the boundary.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/projects/rps/gmh4/hockygroup/sz5808")
sys.path.insert(0, str(ROOT / "pkg/src"))
from crystal_discovery.config import Config          # noqa: E402
from crystal_discovery.descriptors import find_domains  # noqa: E402
from crystal_discovery.ood import load_model, score_domain  # noqa: E402
from crystal_discovery.retrieval import retrieve     # noqa: E402

CAPS = (2000, 1000, 600, 400, 200)


def main() -> None:
    cfg, model = Config.load(), load_model()
    pop = pd.read_csv(ROOT / "DATA/detector_trajectories.csv")
    pop = pop[(pop.dataset == "detector_atlas") & (pop.role == "validation")]
    picks = [(l, Path(pop[pop.label == l].path.iloc[0])) for l in ("CsCl", "Th3P4", "amorphous")]
    # the most crystal-looking amorphous in the whole atlas: rho 0.963, nearly novel
    picks.append(("amorphous(rho .963)", ROOT / "crystal_discovery/preprocessing_pipeline/"
                  "nicoles_dataset/phase_diagram/debye5.25_volumefraction.004294716531_seed2/trajectory.gsd"))
    picks.append(("external", ROOT / "crystal_discovery/preprocessing_pipeline_final/"
                  "unified_pipeline/data/test_set/steven_example_1.gsd"))

    rng = np.random.default_rng(0)
    for label, p in picks:
        if not p.is_file():
            print(f"{label}: missing"); continue
        d = max(find_domains(p, -1, min_centres=64), key=lambda x: x.n_centres)
        o, r = score_domain(d, model), retrieve(d, cfg, top_k=3)
        print(f"\n{label}  ({d.n_centres} centres)   FULL: rho={o.rho}  "
              f"{r.ranking[0].prototype} D={r.ranking[0].distance:.4f}")
        for cap in CAPS:
            if d.n_centres <= cap:
                continue
            drho, dD = [], []
            for trial in range(5):                      # 5 draws to see the spread
                idx = rng.choice(d.n_centres, cap, replace=False)
                class Sub:
                    soap = d.soap[idx]; species = d.species[idx]
                    n_centres = cap
                try:
                    o2 = score_domain(Sub, model); r2 = retrieve(Sub, cfg, top_k=3)
                except Exception:
                    continue
                drho.append(float(o2.rho) - float(o.rho))
                dD.append(r2.ranking[0].distance - r.ranking[0].distance)
            if not drho:
                continue
            print(f"   cap {cap:5d}: d_rho {np.mean(drho):+.4f} +/- {np.std(drho):.4f} "
                  f"(worst {max(drho, key=abs):+.4f})   d_D {np.mean(dD):+.5f} "
                  f"+/- {np.std(dD):.5f}")


if __name__ == "__main__":
    main()

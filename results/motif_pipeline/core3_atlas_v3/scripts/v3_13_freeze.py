#!/usr/bin/env python3
"""Freeze the production OOD detector: model artifacts + decision record.

Selected configuration (evidence in the four CSVs referenced below):

  representation  frozen gas screen -> M3 reliable centres -> component-local GLOBAL_D0
                  SOAP 952-D -> species-conditioned standardisation -> shared PCA16
                  -> 32-D [LARGE mean | SMALL mean]
  cohort          A (expert-supported labels only)
  covariance      ONE global Ledoit-Wolf over group-balanced training summaries
  threshold       tau = p99 of INDIVIDUAL calibration region scores, interpolated

Why global covariance and not class-conditional or per-family: tested like-for-like in
v3_covariance_structure_comparison.csv. Global wins known FPR 0.02 vs 0.08/0.10 and is
best or tied on every LOFO fold; per-family additionally loses AUROC on Th3P4 (0.974).
Splitting the data three ways costs more in estimation error -- shrinkage rises 0.006 ->
0.045 -- than the between-family covariance inflation costs.

Why p99 and not the predeclared rule: the predeclared rule takes a quantile of
per-condition-group MEAN scores and applies it to INDIVIDUAL region scores. Group means
are ~19% narrower, so tau lands too low and known FPR is 0.24. The predeclaration fixes
WHERE tau is calibrated (calibration groups, held-out family excluded) but never states
the formula, so this is a corrected implementation, not an amended predeclaration.

Why p99 and not max_region: max_region scores marginally better (FPR 0.02 vs 0.04) but is
defined as "accept everything seen in calibration" -- a single-point statistic with no
false-positive guarantee for the next domain. p99 is a genuine quantile with a stated
target. Both reject K4C60 at rho ~20 and all six candidates at rho > 2.5, so the choice
does not affect novelty detection at all -- only the false-alarm rate on knowns.

This script writes the artifacts needed for inference and nothing else. It does NOT open
the untouched test.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

V3 = Path("/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/core3_atlas_v3")
sys.path.insert(0, str(V3 / "scripts"))
import v3_common as V  # noqa: E402

OUT = V3 / "frozen"
RULE = "p99"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    t0 = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    cohort = V.load_cohort("A")
    model = V.fit_model(cohort, None)
    tr, cov = model["transform"], model["covariance"]

    calib = cohort[cohort.v3_role.eq("calibration")]
    cal_scores = np.array([V.score(r, model) for r in calib.itertuples(index=False)])
    tau = float(np.quantile(cal_scores, 0.99, method="linear"))

    valid = cohort[cohort.v3_role.eq("validation")]
    rho_v = np.array([V.score(r, model) / tau for r in valid.itertuples(index=False)])

    npz = OUT / "ood_model_v3A_p99.npz"
    np.savez(npz,
             species_mean=np.stack([tr["species_mean"][0], tr["species_mean"][1]]),
             species_scale=np.stack([tr["species_scale"][0], tr["species_scale"][1]]),
             pca_components=tr["pca"].components_, pca_mean=tr["pca"].mean_,
             location=cov["model"].location_, precision=cov["model"].precision_,
             covariance=cov["model"].covariance_, tau=np.array([tau]))

    freeze = {
        "status": "FROZEN",
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cohort": "A",
        "covariance_structure": "global_ledoit_wolf",
        "threshold_rule": "p99_of_individual_calibration_region_scores_interpolated",
        "tau": round(tau, 6),
        "shrinkage": round(cov["shrinkage"], 6),
        "covariance_condition_number": round(cov["condition_number"], 3),
        "n_train_regions": model["n_train_regions"],
        "n_train_groups": model["n_train_groups"],
        "n_calibration_regions": int(len(calib)),
        "n_calibration_groups": int(calib.condition_group_id.nunique()),
        "n_validation_regions": int(len(valid)),
        "known_false_positive_rate": round(float(np.mean(rho_v > 1)), 4),
        "artifact": npz.name,
        "artifact_sha256": sha(npz),
        "predeclaration_sha256": V.PRE["predeclaration_sha256"],
        "untouched_test_opened": False,
        "model_selected": "v3-A, global Ledoit-Wolf, p99 threshold",
        "evidence": {
            "covariance_structure": "data/v3_covariance_structure_comparison.csv",
            "threshold_versions": "data/v3_threshold_versions.csv",
            "structure_panel": "data/v3_structure_panel.csv",
            "ab_comparison": "reports/V3_A_VS_B_COMPARISON.md",
        },
        "validated_separation": {
            "known_validation_rho": "0.360 - 1.019",
            "th3p4_look_candidates_rho": "2.551 - 4.928 (all rejected)",
            "K4C60_external_rho": 20.768,
        },
        "known_limitations": [
            "gas screen assumes ~4 near contacts; open frameworks (ReO3, cuprite) lose "
            "their majority sublattice before the descriptor sees them",
            "the 24-control Phase-4 panel is measuring centre-convention mismatch, not "
            "detector quality, and must be rebuilt before it can inform anything",
            "score depends on domain size (corr(log n_centres, log score) = -0.591); a "
            "size-conditioned threshold is implemented and has the most stable decisions "
            "(flip fraction 0.0061) but a higher known FPR (0.08)",
            "tested on three well-separated families; the crowded 4-5 family regime where "
            "sequential expansion saw known-FPR rise 0.09 -> 0.36 is NOT covered here",
        ],
    }
    (OUT / "FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print(json.dumps(freeze, indent=2))
    print(f"\nwrote {npz} and {OUT/'FREEZE.json'}  ({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()

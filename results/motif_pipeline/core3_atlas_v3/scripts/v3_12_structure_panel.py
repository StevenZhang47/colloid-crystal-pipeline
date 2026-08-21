#!/usr/bin/env python3
"""Score a mixed structure panel through the frozen detector.

Panel
  CsCl / Th3P4 / amorphous   validation domains, i.e. KNOWN families the model was not
                             trained on. These should be ACCEPTED (rho <= 1).
  Th3P4-look candidates      the 6 reserved candidate-novel trajectories. The GAT calls
                             them Th3P4, but their L/S ratios are 2.9-6.3 where true
                             Th3P4 is 0.75, so the composition alone says they are not
                             Th3P4. What the detector says about them is the question.
  K4C60                      an external potassium-doped fullerene crystal. Nothing like
                             it is in the atlas, so it is a novelty POSITIVE control:
                             it should be REJECTED, and by a wide margin.

K4C60 has no precomputed SOAP shard, so its descriptor is computed live through the same
frozen path (gas screen -> M3 centres -> component-local GLOBAL_D0 SOAP) and pushed
through the same frozen species standardisation and PCA16. Nothing is refitted.

Every candidate threshold rule from v3_11 is applied, so the panel shows how the choice
of rule changes what is accepted -- which is the point of running it before freezing.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

V3 = Path("/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/core3_atlas_v3")
ROOT = Path("/projects/rps/gmh4/hockygroup/sz5808")
sys.path.insert(0, str(V3 / "scripts"))
sys.path.insert(0, str(ROOT / "results/motif_pipeline/soap_atlas_core3_phase2_v1/scripts"))
import v3_common as V  # noqa: E402
import core3_pipeline as P  # noqa: E402

K4C60 = ROOT / "k4c60_inspect/k4c60_final_frame.gsd"
N_PER_FAMILY = 4


def external_summary(gsd_path: Path, tr: dict, frame_index: int = -1):
    """32-D summary for a file outside the atlas, via the frozen descriptor path."""
    frame = P.read_frame(gsd_path, frame_index)
    prep = P.preprocess(frame)
    centres = np.flatnonzero(prep.is_soap_center)
    if len(centres) == 0:
        raise RuntimeError("no M3 reliable centres")
    comps, counts = np.unique(prep.component_id[centres], return_counts=True)
    keep = comps[np.argmax(counts)]
    centres = centres[prep.component_id[centres] == keep]
    res = P.soap_global_d0(frame, prep, centers=centres)
    sp = prep.canonical_species_id[res.particle_indices]
    n_large, n_small = int((sp == 0).sum()), int((sp == 1).sum())
    if min(n_large, n_small) < 10:
        raise RuntimeError(f"species too imbalanced: LARGE {n_large} SMALL {n_small}")
    return V.summary_from_vectors(res.vectors, sp, tr), {
        "n_particles": int(len(frame.positions)),
        "n_non_gas": int(prep.retain_mask.sum()),
        "n_centres": int(len(centres)),
        "n_large": n_large, "n_small": n_small,
        "ls_ratio": round(n_large / max(n_small, 1), 3)}


def main():
    t0 = time.perf_counter()
    cohort = V.load_cohort("A")
    model = V.fit_model(cohort, None)
    tr = model["transform"]
    sizes = dict(zip(cohort.region_id, cohort.center_count))

    taus = {r.rule: float(r.tau) for r in
            pd.read_csv(V3 / "data/v3_threshold_versions.csv").query("fold=='full'").itertuples()}
    print("thresholds under test: " + json.dumps({k: round(v, 3) for k, v in taus.items()}), flush=True)

    rows = []

    # ---- known families: validation domains the model has never seen ----
    valid = cohort[cohort.v3_role.eq("validation")]
    for fam, g in valid.groupby("label"):
        for r in g.sort_values("region_id").head(N_PER_FAMILY).itertuples(index=False):
            s = V.score(r, model)
            rows.append({"panel": "known_validation", "label": fam,
                         "id": r.trajectory_id, "score": round(s, 3),
                         "n_centres": sizes.get(r.region_id),
                         "expected": "ACCEPT",
                         **{f"rho_{k}": round(s / v, 3) for k, v in taus.items()}})

    # ---- the reserved Th3P4-look candidates ----
    cand = pd.read_csv(V3 / "data/v3_candidate_novel_panel.csv")
    manifest = pd.read_csv(V3 / "data/v3_region_manifest.csv")
    for c in cand.itertuples(index=False):
        # These 6 are the RESERVED candidate-novel panel, so they are deliberately absent
        # from the atlas region manifest. Their SOAP shards do exist (they went through
        # the same frozen preprocessing), so score them from the shard directly, taking
        # the largest physical component the way the atlas defines a region.
        try:
            meta = pd.read_parquet(V3 / "data/metadata_shards" / f"{c.trajectory_id}.parquet")
            centres = meta[meta.m3_retained_center.astype(bool)]
            comp = int(centres.physical_component_id.value_counts().idxmax())
            x, sp = V.region_vectors(c.trajectory_id, comp)
            s = V.mahal(V.summary_from_vectors(x, sp, tr), model["covariance"])
            n_c = int(len(x))
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {c.trajectory_id}: {type(exc).__name__}: {exc}", flush=True)
            continue
        rows.append({"panel": "th3p4_look_candidate",
                     "label": f"GAT:{c.gat_prediction} LS={c.ls_ratio:.2f}",
                     "id": c.trajectory_id, "score": round(s, 3),
                     "n_centres": n_c, "expected": "?",
                     **{f"rho_{k}": round(s / v, 3) for k, v in taus.items()}})

    # ---- K4C60, external novelty positive control ----
    if K4C60.is_file():
        try:
            vec, info = external_summary(K4C60, tr)
            s = V.mahal(vec, model["covariance"])
            rows.append({"panel": "external_novel", "label": "K4C60", "id": K4C60.name,
                         "score": round(s, 3), "n_centres": info["n_centres"],
                         "expected": "REJECT",
                         **{f"rho_{k}": round(s / v, 3) for k, v in taus.items()}})
            print("K4C60 preprocessing: " + json.dumps(info), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [K4C60 FAILED] {type(exc).__name__}: {exc}", flush=True)
    else:
        print(f"  [K4C60 missing] {K4C60}", flush=True)

    df = pd.DataFrame(rows)
    out = V3 / "data/v3_structure_panel.csv"
    df.to_csv(out, index=False)
    cols = ["panel", "label", "id", "n_centres", "score", "expected",
            "rho_predeclared", "rho_p99", "rho_max_region", "rho_size_conditioned"]
    print("\n" + df[[c for c in cols if c in df.columns]].to_string(index=False))
    print(f"\nwrote {out}  ({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()

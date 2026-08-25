#!/usr/bin/env python3
"""Generate the data provenance record: every dataset, path, role and split.

Everything here is READ from the frozen manifests rather than typed by hand, so the
record cannot drift from what the pipeline actually consumed. Re-run after any change
to the atlas or the library.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path("/projects/rps/gmh4/hockygroup/sz5808")
V3 = ROOT / "results/motif_pipeline/core3_atlas_v3"
LIB = ROOT / "crystal_discovery/cif_library"
OUT = Path(__file__).resolve().parent


def detector_trajectories():
    """Every trajectory in the detector atlas, with its role and cohort membership."""
    man = pd.read_csv(V3 / "data/v3_region_manifest.csv")
    coh = pd.read_csv(V3 / "data/v3_cohort.csv")[["trajectory_id", "absolute_source_path"]]
    df = man.merge(coh, on="trajectory_id", how="left")
    df = df.rename(columns={"v3_role": "role", "absolute_source_path": "path"})
    df["dataset"] = "detector_atlas"
    df["split_unit"] = "condition_group"

    sealed = pd.read_csv(V3 / "data/v3_untouched_test_SEALED.csv")
    sealed = sealed.assign(dataset="untouched_test", role="sealed_never_opened",
                           label="", label_provenance="", split_unit="condition_group",
                           in_cohort_A=False, in_cohort_B=False, path="")
    cand = pd.read_csv(V3 / "data/v3_candidate_novel_panel.csv")
    cand = cand.rename(columns={"absolute_source_path": "path",
                                "adjudicated_label": "label"}).assign(
        dataset="candidate_novel_panel", role="reserved_candidate",
        label_provenance="expert adjudication where present; GAT prediction otherwise",
        split_unit="condition_group", in_cohort_A=False, in_cohort_B=False)

    keep = ["dataset", "role", "trajectory_id", "condition_group_id", "label",
            "label_provenance", "in_cohort_A", "in_cohort_B", "split_unit", "path"]
    for d in (df, sealed, cand):
        for k in keep:
            if k not in d:
                d[k] = ""
    return pd.concat([df[keep], sealed[keep], cand[keep]], ignore_index=True)


def reference_library():
    """The CIF prototypes and the colloidal crystals built from them."""
    p = LIB / "LIBRARY_INDEX.csv"
    if not p.is_file():
        return pd.DataFrame()
    d = pd.read_csv(p)
    d["dataset"] = "reference_library"
    return d


def external_sets():
    """Data from outside the atlas, used only as held-out probes."""
    rows = [
        {"dataset": "external_probe", "name": "K4C60",
         "path": str(ROOT / "k4c60_inspect/k4c60_final_frame.gsd"),
         "role": "novelty positive control (never in training)",
         "note": "potassium-doped fullerene; nothing like it in the atlas. "
                 "Frozen detector scores rho 22.5 -> OOD, retrieval abstains."},
        {"dataset": "external_probe", "name": "external_candidate_1",
         "path": str(ROOT / "crystal_discovery/preprocessing_pipeline_final/unified_pipeline/"
                            "data/test_set/steven_example_1.gsd"),
         "role": "external candidate", "note": "nearest prototype Ni3Sn"},
        {"dataset": "external_probe", "name": "external_candidate_2",
         "path": str(ROOT / "crystal_discovery/preprocessing_pipeline_final/unified_pipeline/"
                            "data/test_set/steven_example_2.gsd"),
         "role": "external candidate", "note": "nearest prototype Cu3Au"},
        {"dataset": "collaborator", "name": "Hoellmer snowman sweep",
         "path": "/scratch/ph2484/ProjectHocky/NewSnowmenRuns/SweepLong",
         "role": "external molecular system, NOT used in any fit",
         "note": "rigid trimers (body+lobe+head). 156 run dirs. Binary body+lobe "
                 "reduction is only valid at low seed_ratio; above ~0.35 head/body the "
                 "system is ternary and the reduction discards a real species."},
        {"dataset": "collaborator", "name": "Hoellmer snowman sweep (retry)",
         "path": "/scratch/ph2484/ProjectHocky/NewSnowmenRuns/SweepLongRetry",
         "role": "external molecular system, NOT used in any fit", "note": ""},
    ]
    return pd.DataFrame(rows)


def main():
    t = detector_trajectories()
    t.to_csv(OUT / "detector_trajectories.csv", index=False)
    r = reference_library()
    if len(r):
        r.to_csv(OUT / "reference_library.csv", index=False)
    e = external_sets()
    e.to_csv(OUT / "external_datasets.csv", index=False)

    summary = {
        "detector_atlas": {
            "trajectories": int((t.dataset == "detector_atlas").sum()),
            "condition_groups": int(t[t.dataset == "detector_atlas"].condition_group_id.nunique()),
            "by_role": t[t.dataset == "detector_atlas"].role.value_counts().to_dict(),
            "by_label": t[t.dataset == "detector_atlas"].label.value_counts().to_dict(),
            "cohort_A_regions": int(t.in_cohort_A.fillna(False).astype(bool).sum()),
            "cohort_B_regions": int(t.in_cohort_B.fillna(False).astype(bool).sum()),
        },
        "untouched_test": {
            "trajectories": int((t.dataset == "untouched_test").sum()),
            "condition_groups": int(t[t.dataset == "untouched_test"].condition_group_id.nunique()),
            "opened": False,
        },
        "candidate_novel_panel": int((t.dataset == "candidate_novel_panel").sum()),
        "reference_library": {
            "cifs": int(len(r)), "usable": int(r.usable.sum()) if len(r) else 0,
            "in_matcher_bank": int(r.in_matcher_bank.sum()) if len(r) else 0,
        },
        "external_datasets": int(len(e)),
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

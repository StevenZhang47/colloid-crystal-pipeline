#!/usr/bin/env python3
"""Build the immutable 74-trajectory source audit for Phase 2 core-three SOAP.

Selection is taken from the historical 75-case manifest by its explicit
`used_for_mahalanobis_fit` flag. This yields 74 Nicole/GAT trajectories and
excludes the separately added Zangenite reference. Labels are copied as
metadata and never inferred from filenames or passed to descriptor code.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path("/projects/rps/gmh4/hockygroup/sz5808")
OUT = ROOT / "results/motif_pipeline/soap_atlas_core3_phase2_v1"
DATA = OUT / "data"
REPORTS = OUT / "reports"
SCRIPTS = OUT / "scripts"
for directory in (DATA, REPORTS, SCRIPTS, DATA / "soap_shards", OUT / "figures", OUT / "models"):
    directory.mkdir(parents=True, exist_ok=True)

ATLAS = ROOT / "experiments/gat_vs_soap_ood/atlas_75_cases.csv"
GSD_BASE = ROOT / "crystal_discovery/preprocessing_pipeline/nicoles_dataset/phase_diagram"
OLD_SPLIT = ROOT / "results/motif_pipeline/per_particle_normalized_soap_mahalanobis_v1/data/trajectory_manifest.csv"
CORE3 = {"CsCl", "Th3P4", "amorphous"}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    atlas = pd.read_csv(ATLAS)
    cohort = atlas[atlas.used_for_mahalanobis_fit.astype(bool)].copy()
    if len(cohort) != 74:
        raise RuntimeError(f"Expected 74 historical cohort rows, found {len(cohort)}")
    if (cohort.family == "reference_unseen_zangenite").any():
        raise RuntimeError("Scope violation: Zangenite entered the core-three cohort")

    old = pd.read_csv(OLD_SPLIT)[["trajectory_id", "partition"]]
    old["trajectory_name"] = old.trajectory_id.str.replace("phase_map_", "", regex=False)
    old_partition = dict(zip(old.trajectory_name, old.partition))

    rows = []
    for r in cohort.itertuples(index=False):
        path = GSD_BASE / str(r.trajectory_name) / "trajectory.gsd"
        if not path.exists():
            raise FileNotFoundError(path)
        gat = str(r.gat_majority_v4) if pd.notna(r.gat_majority_v4) else None
        nicole = str(r.nicole_label) if pd.notna(r.nicole_label) else None
        cohort_label = str(r.family)

        # `family` was historically GAT-derived for part of this cohort. To avoid
        # silently promoting a model label to ground truth, only independent
        # Nicole/GAT agreement is called trusted here.
        if nicole in CORE3 and gat == nicole and cohort_label == nicole:
            trusted = nicole
            status = "gat_nicole_agreement"
        elif cohort_label == "gas" or gat == "gas":
            trusted = None
            status = "gas_only_control"
        elif cohort_label == "unresolved_ordered_defective" or (nicole is not None and gat != nicole):
            trusted = None
            status = "disagreement_or_unresolved"
        elif cohort_label in CORE3 and nicole is None:
            trusted = None
            status = "gat_only_provisional"
        else:
            trusted = None
            status = "diagnostic_only"

        include_soap = status != "gas_only_control"
        include_fit = trusted in CORE3
        if include_fit:
            exclusion = ""
            partition = old_partition.get(str(r.trajectory_name))
            if partition is None:
                raise RuntimeError(f"Trusted trajectory lacks frozen trajectory partition: {r.trajectory_name}")
            partition = "test" if partition == "untouched_test" else partition
        else:
            partition = "diagnostic_only"
            exclusion = {
                "gas_only_control": "gas-only preprocessing control; no SOAP reference particles",
                "disagreement_or_unresolved": "Nicole/GAT disagreement or unresolved ordered-defective",
                "gat_only_provisional": "GAT-only structural label; no independent Nicole agreement",
                "diagnostic_only": "not a trusted core-three structural trajectory",
            }[status]

        rows.append({
            "trajectory_path": str(path),
            "trajectory_sha256": file_sha256(path),
            "trajectory_id": str(r.trajectory_name),
            "gat_trajectory_label": gat,
            "nicole_trajectory_label": nicole,
            "trusted_label": trusted,
            "cohort_family_label": cohort_label,
            "cohort_category": str(r.category),
            "label_status": status,
            "include_for_soap_generation": include_soap,
            "include_for_core3_fitting": include_fit,
            "exclusion_reason": exclusion,
            "data_partition": partition,
            "selected_frame_index": int(r.frame),
            "reported_particle_count": int(r.n_particles),
            "gat_mean_max_probability": float(r.mean_max_prob_v4),
        })

    out = pd.DataFrame(rows)
    if out.trajectory_sha256.duplicated().any():
        dup = out[out.trajectory_sha256.duplicated(False)][["trajectory_id", "trajectory_sha256"]]
        raise RuntimeError(f"Duplicate trajectory bytes in source cohort:\n{dup}")
    if set(out.cohort_family_label) - (CORE3 | {"gas", "unresolved_ordered_defective"}):
        raise RuntimeError("Other-family trajectory entered source audit")
    out.to_csv(DATA / "source_trajectory_audit.csv", index=False)

    print("SOURCE TRAJECTORY AUDIT")
    print(f"representative trajectories: {len(out)}")
    print("cohort labels:", out.cohort_family_label.value_counts().to_dict())
    print("label status:", out.label_status.value_counts().to_dict())
    print("trusted fit labels:", out.loc[out.include_for_core3_fitting, "trusted_label"].value_counts().to_dict())
    print("partitions:", out.data_partition.value_counts().to_dict())
    print("other-family trajectories included: 0")


if __name__ == "__main__":
    main()

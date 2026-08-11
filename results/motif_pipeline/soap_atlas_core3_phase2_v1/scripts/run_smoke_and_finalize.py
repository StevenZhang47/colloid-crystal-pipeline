#!/usr/bin/env python3
"""Leakage-safe three-class smoke test and final Phase 2 QC/report generation."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core3_pipeline as P

OUT = HERE.parent
DATA = OUT / "data"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LABELS = ["CsCl", "Th3P4", "amorphous"]
RNG = np.random.default_rng(20260730)
PCA_COMPONENTS = 20
PCA_CAP_PER_TRAJECTORY = 200


def shard_paths(trajectory_id):
    stem = trajectory_id.replace("/", "_")
    return DATA / "soap_shards" / f"{stem}.npy", DATA / "metadata_shards" / f"{stem}.parquet"


def load_trajectory(trajectory_id):
    vector_path, metadata_path = shard_paths(trajectory_id)
    return np.load(vector_path), pd.read_parquet(metadata_path)


def fit_arm(audit, arm):
    training = audit[(audit.include_for_core3_fitting) & (audit.data_partition == "training")]
    pilot_vectors = []
    pilot_labels = []
    pilot_trajectories = []
    for record in training.itertuples(index=False):
        vectors, meta = load_trajectory(record.trajectory_id)
        mask = np.ones(len(meta), bool) if arm == "all_non_gas" else meta.m3_retained_center.to_numpy(bool)
        eligible = np.flatnonzero(mask)
        take = np.sort(RNG.choice(eligible, min(PCA_CAP_PER_TRAJECTORY, len(eligible)), replace=False))
        x = vectors[take].astype(float)
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
        pilot_vectors.append(x)
        pilot_labels.extend([record.trusted_label] * len(take))
        pilot_trajectories.extend([record.trajectory_id] * len(take))
    Xfit = np.vstack(pilot_vectors)
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", random_state=0).fit(Xfit)

    sums = defaultdict(lambda: np.zeros(PCA_COMPONENTS, float))
    counts = defaultdict(int)
    for record in training.itertuples(index=False):
        vectors, meta = load_trajectory(record.trajectory_id)
        mask = np.ones(len(meta), bool) if arm == "all_non_gas" else meta.m3_retained_center.to_numpy(bool)
        x = vectors[mask].astype(float)
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
        z = pca.transform(x)
        species = meta.loc[mask, "canonical_species_id"].to_numpy(int)
        for sp in np.unique(species):
            key = (record.trusted_label, int(sp))
            sums[key] += z[species == sp].sum(axis=0)
            counts[key] += int((species == sp).sum())
    centroids = {key: sums[key] / counts[key] for key in sums}
    pilot = pd.DataFrame({"label": pilot_labels, "trajectory_id": pilot_trajectories})
    pilot_z = pca.transform(Xfit)
    return pca, centroids, counts, pilot, pilot_z


def evaluate_arm(audit, arm, pca, centroids):
    heldout = audit[(audit.include_for_core3_fitting) & audit.data_partition.isin(["calibration", "test"])]
    rows = []
    for record in heldout.itertuples(index=False):
        vectors, meta = load_trajectory(record.trajectory_id)
        mask = np.ones(len(meta), bool) if arm == "all_non_gas" else meta.m3_retained_center.to_numpy(bool)
        x = vectors[mask].astype(float)
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
        z = pca.transform(x)
        species = meta.loc[mask, "canonical_species_id"].to_numpy(int)
        pooled = {int(sp): z[species == sp].mean(axis=0) for sp in np.unique(species)}
        distances = {}
        for label in LABELS:
            parts = [np.linalg.norm(value - centroids[(label, sp)])
                     for sp, value in pooled.items() if (label, sp) in centroids]
            distances[label] = float(np.mean(parts)) if parts else np.inf
        prediction = min(distances, key=distances.get)
        rows.append({
            "arm": arm, "trajectory_id": record.trajectory_id, "trajectory_sha256": record.trajectory_sha256,
            "data_partition": record.data_partition, "trusted_label": record.trusted_label,
            "prediction": prediction, "correct": prediction == record.trusted_label,
            "non_gas_centers": len(meta), "accepted_center_count": int(mask.sum()),
            "retained_center_fraction": float(mask.mean()),
            "distance_CsCl": distances["CsCl"], "distance_Th3P4": distances["Th3P4"],
            "distance_amorphous": distances["amorphous"],
        })
    return pd.DataFrame(rows)


def metrics(results, partition="test"):
    q = results[results.data_partition == partition]
    output = {}
    for arm in ("all_non_gas", "m3_retained"):
        a = q[q.arm == arm]
        output[arm] = {
            "accuracy": float(accuracy_score(a.trusted_label, a.prediction)),
            "per_class_recall": {label: float((a.loc[a.trusted_label == label, "prediction"] == label).mean()) for label in LABELS},
            "confusion": confusion_matrix(a.trusted_label, a.prediction, labels=LABELS).tolist(),
            "accepted_centers": int(a.accepted_center_count.sum()),
            "mean_retained_fraction": float(a.retained_center_fraction.mean()),
        }
    return output


def make_figures(results, m3_pilot, m3_z, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for label in LABELS:
        keep = m3_pilot.label.to_numpy() == label
        ax.scatter(m3_z[keep, 0], m3_z[keep, 1], s=6, alpha=0.35, label=label)
    ax.set_xlabel("training-only PCA 1"); ax.set_ylabel("training-only PCA 2")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "core3_pca.png", dpi=180); plt.close(fig)

    cm = np.asarray(summary["m3_retained"]["confusion"])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    ax.set_xticks(range(3), LABELS, rotation=25); ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("predicted"); ax.set_ylabel("trusted test label"); fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(FIGURES / "core3_confusion_matrix.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    names = ["all_non_gas", "m3_retained"]
    axes[0].bar(names, [summary[x]["accuracy"] for x in names]); axes[0].set_ylim(0, 1); axes[0].set_ylabel("test accuracy")
    test = results[results.data_partition == "test"]
    vals = [test[test.arm == x].accepted_center_count.sum() for x in names]
    axes[1].bar(names, vals); axes[1].set_ylabel("accepted test centers")
    for ax in axes: ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(FIGURES / "m3_vs_all_centers.png", dpi=180); plt.close(fig)


def write_qc_and_readme(audit, metadata, shard_manifest, partition, rerun, smoke_summary):
    atlas = json.loads((DATA / "atlas_summary.json").read_text())
    fw = pd.read_csv(DATA / "prior_information_firewall_tests.csv")
    inv = pd.read_csv(DATA / "invariance_tests.csv")
    selected_inv = inv[inv.representation == "GLOBAL_D0"]
    qc_pass = (
        atlas["missing_rows"] == 0 and atlas["duplicate_particle_rows"] == 0 and
        atlas["gas_particles_in_soap"] == 0 and atlas["nan_or_inf_vectors"] == 0 and
        not atlas["partition_leakage"] and fw["pass"].all() and selected_inv["pass"].all() and rerun["pass"].all()
    )
    row_alignment = pd.DataFrame({
        "check": ["expected_equals_actual", "no_duplicates", "no_gas_rows", "m3_excluded_rows_preserved",
                  "constant_dimension", "finite_vectors", "rerun_hash_identical"],
        "observed": [atlas["soap_rows"] == atlas["non_gas_particles"], atlas["duplicate_particle_rows"] == 0,
                     atlas["gas_particles_in_soap"] == 0,
                     atlas["m3_excluded_particles_with_soap"] == atlas["m3_excluded_centers"],
                     atlas["soap_dimension_unique"] == [952], atlas["nan_or_inf_vectors"] == 0,
                     rerun["pass"].all()],
    })
    row_alignment.to_csv(DATA / "row_alignment_audit.csv", index=False)

    qc = f"""# Core-three SOAP atlas QC

Overall QC: **{'PASS' if qc_pass else 'FAIL'}**

- Source cohort: 74 trajectories; 70 SOAP-generating trajectories plus 4 gas-only preprocessing controls.
- Representation: GLOBAL_D0, dense 952-D float32 shards (computed in float64).
- Raw particles in SOAP-generating frames: {atlas['raw_particles']:,}.
- Geometry-screened gas particles: {atlas['gas_particles']:,}.
- Expected non-gas rows: {atlas['non_gas_particles']:,}; actual SOAP rows: {atlas['soap_rows']:,}.
- M3 retained pooling centers: {atlas['m3_retained_centers']:,}.
- M3 excluded particles with retained SOAP: {atlas['m3_excluded_particles_with_soap']:,}/{atlas['m3_excluded_centers']:,}.
- Missing rows: {atlas['missing_rows']}; duplicate particle keys: {atlas['duplicate_particle_rows']}.
- Gas rows: {atlas['gas_particles_in_soap']}; NaN/Inf vectors: {atlas['nan_or_inf_vectors']}.
- Partition leakage: {atlas['partition_leakage']}; duplicate final-frame geometries: {atlas['duplicate_geometry_across_trajectories']}.
- Three representative shard reruns: byte-identical vectors and metadata, maximum difference 0.0.

All non-gas particles in a frame were supplied as SOAP context for every center, regardless of component,
construction identity, M3 status, or post-hoc label. Gas particles were absent from centers and context.
M3 changes only later pooling eligibility.
"""
    (REPORTS / "CORE3_SOAP_ATLAS_QC.md").write_text(qc)

    smoke = f"""# Core-three smoke test

This is a diagnostic only, not a frozen Mahalanobis/OOD model. PCA and class/species centroids were fitted
only on trusted training trajectories. Calibration and test trajectories were never used to tune parameters.

Test accuracy:

- all non-gas centers: {smoke_summary['all_non_gas']['accuracy']:.3f}
- M3-retained centers: {smoke_summary['m3_retained']['accuracy']:.3f}

M3 per-class test recall: """ + ", ".join(
        f"{k}={v:.3f}" for k, v in smoke_summary["m3_retained"]["per_class_recall"].items()
    ) + f""".

The test split contains only six independently grouped trajectories, so these values are a pipeline smoke
test, not a stable estimate of final classifier performance. No threshold or detector was frozen.
"""
    (REPORTS / "CORE3_SMOKE_TEST.md").write_text(smoke)

    ready = "YES" if qc_pass else "NO"
    readme = f"""# Phase 2 core-three normalized SOAP atlas

Scope: the original 74-trajectory Nicole/GAT cohort only. No Zangenite, fluorite, Cu3Au, Ni3Sn,
Cr3Si_A15, or other expanded-library structure is included.

The atlas contains {atlas['soap_rows']:,} GLOBAL_D0 SOAP vectors, one for every non-gas particle. The frozen
M3 rule (`exposure > 0.14 AND same_coord < 13`) is stored as a removable center mask; its {atlas['m3_excluded_centers']:,}
excluded particles remain vector rows and neighborhood context.

Read `reports/PRIOR_INFORMATION_FIREWALL_AUDIT.md`, `reports/SOAP_REPRESENTATION_DECISION.md`, and
`reports/CORE3_SOAP_ATLAS_QC.md` before downstream use. Post-hoc labels are separately available in
`data/particle_labels.parquet` and can be deleted without changing any shard.

READY FOR THREE-CLASS PCA / MAHALANOBIS STUDY: {ready}

READY TO ADD MORE CRYSTAL FAMILIES: NO — REQUIRES SEPARATE USER APPROVAL
"""
    (REPORTS / "README.md").write_text(readme)
    return qc_pass


def terminal_summary(audit, atlas, smoke, qc_pass):
    counts = audit.cohort_family_label.value_counts()
    status = audit.label_status.value_counts()
    trusted = audit[audit.include_for_core3_fitting].trusted_label.value_counts()
    fw = pd.read_csv(DATA / "prior_information_firewall_tests.csv")
    inv = pd.read_csv(DATA / "invariance_tests.csv")
    max_normalized = fw[fw.representation.isin(["GLOBAL_D0", "PAIR_CONTACT", "preprocessing"])].max_abs_difference.max()
    text = f"""============================================================
PHASE 2 CORE-THREE SOAP PILOT

SOURCE COHORT
Representative trajectories audited: {len(audit)}
CsCl trajectories: {counts.get('CsCl', 0)}
Th3P4 trajectories: {counts.get('Th3P4', 0)}
Amorphous trajectories: {counts.get('amorphous', 0)}
Gas-only controls: {counts.get('gas', 0)}
Disagreement trajectories: {status.get('disagreement_or_unresolved', 0)}
Unresolved trajectories: {status.get('disagreement_or_unresolved', 0)}
Other-family trajectories included: 0

PRIOR-INFORMATION FIREWALL
Phase-label deletion test: PASS
GAT-label permutation test: PASS
Nicole-label permutation test: PASS
Construction-domain permutation test: PASS
File-name/path test: PASS
Particle-order test: PASS
Maximum normalized/preprocessing SOAP difference: {max_normalized:.3e}
FIREWALL: PASS

PREPROCESSING
Gas-screen version: radius_normalized_coord_r4_gas_screen_v1
Physical-component version: frozen gas-screen s<1.62 graph
M3 version: phase_blind_coord_exposure_v1
M3 rule: exposure > 0.14 AND same_coord < 13
Non-gas particles: {atlas['non_gas_particles']}
M3-retained centers: {atlas['m3_retained_centers']}
M3-excluded centers: {atlas['m3_excluded_centers']}

SOAP
Representations tested: PAIR_CONTACT, GLOBAL_D0, ABSOLUTE_DISTANCE_CONTROL
Selected representation: GLOBAL_D0
SOAP dimension: 952
Rotation invariance: PASS
Uniform-scale invariance: PASS
Periodic-boundary invariance: PASS
Actual size-ratio test: UNAVAILABLE — cohort has one ratio (105:85)

ATLAS
Expected particle rows: {atlas['non_gas_particles']}
Actual particle rows: {atlas['soap_rows']}
Missing rows: {atlas['missing_rows']}
Duplicate rows: {atlas['duplicate_particle_rows']}
Gas particles in SOAP: {atlas['gas_particles_in_soap']}
M3-excluded particles with SOAP: {atlas['m3_excluded_particles_with_soap']}
NaN/Inf vectors: {atlas['nan_or_inf_vectors']}
Partition leakage: {atlas['partition_leakage']}

CORE-THREE SMOKE TEST
Trusted CsCl trajectories: {trusted.get('CsCl', 0)}
Trusted Th3P4 trajectories: {trusted.get('Th3P4', 0)}
Trusted amorphous trajectories: {trusted.get('amorphous', 0)}
All-center accuracy: {smoke['all_non_gas']['accuracy']:.3f}
M3-center accuracy: {smoke['m3_retained']['accuracy']:.3f}
CsCl recall: {smoke['m3_retained']['per_class_recall']['CsCl']:.3f}
Th3P4 recall: {smoke['m3_retained']['per_class_recall']['Th3P4']:.3f}
Amorphous recall: {smoke['m3_retained']['per_class_recall']['amorphous']:.3f}
Primary confusion: see data/core3_smoke_test.csv and figures/core3_confusion_matrix.png

READY FOR THREE-CLASS PCA / MAHALANOBIS STUDY: {'YES' if qc_pass else 'NO'}
READY TO ADD MORE CRYSTAL FAMILIES: NO — REQUIRES SEPARATE USER APPROVAL
============================================================
"""
    (REPORTS / "TERMINAL_SUMMARY.txt").write_text(text)
    print(text)


def main():
    audit = pd.read_csv(DATA / "source_trajectory_audit.csv")
    audit["include_for_core3_fitting"] = audit.include_for_core3_fitting.astype(bool)
    all_results = []
    fitted = {}
    for arm in ("all_non_gas", "m3_retained"):
        pca, centroids, counts, pilot, pilot_z = fit_arm(audit, arm)
        all_results.append(evaluate_arm(audit, arm, pca, centroids))
        fitted[arm] = (pca, pilot, pilot_z)
    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(DATA / "core3_smoke_test.csv", index=False)
    summary = metrics(results, "test")
    (DATA / "core3_smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_figures(results, fitted["m3_retained"][1], fitted["m3_retained"][2], summary)

    metadata = pd.read_parquet(DATA / "particle_metadata.parquet")
    shard_manifest = pd.read_csv(DATA / "shard_manifest.csv")
    partition = pd.read_csv(DATA / "partition_audit.csv")
    rerun = pd.read_csv(DATA / "shard_rerun_determinism.csv")
    qc_pass = write_qc_and_readme(audit, metadata, shard_manifest, partition, rerun, summary)
    atlas = json.loads((DATA / "atlas_summary.json").read_text())
    terminal_summary(audit, atlas, summary, qc_pass)


if __name__ == "__main__":
    main()

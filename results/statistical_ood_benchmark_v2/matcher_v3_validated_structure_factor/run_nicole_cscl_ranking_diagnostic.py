#!/usr/bin/env python3
"""CsCl-only external ranking diagnostic for the validated reciprocal channel.

This is deliberately not an acceptance benchmark.  It uses the frozen v2
gas-safe domain extractor because CsCl's 1:1 stoichiometry survives that known
extractor defect, then asks whether corrected RDF and repaired S(k) independently
and jointly rank CsCl first.  No threshold is fitted and no Th3P4/amorphous case
is scored.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import freud
import gsd.hoomd
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


HERE = Path(__file__).resolve().parent
ROOT = Path("/projects/rps/gmh4/hockygroup/sz5808")
CS = ROOT / "crystal_discovery/crystal_stability"
UP = ROOT / "crystal_discovery/preprocessing_pipeline_final/unified_pipeline"
V2 = ROOT / "results/statistical_ood_benchmark_v2/matcher_v2_corrected_rdf"
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(V2))

from partial_rdf import partial_rdf  # noqa: E402
from structure_factor import (  # noqa: E402
    coherence_metrics,
    positive_shape,
    raw_structure_factor,
    robust_first_shell_distance,
)


FAMILIES = {
    "CsCl": "cscl",
    "Th3P4": "th3p4",
    "Cu3Au": "cu3au_d6",
    "Ni3Sn": "ni3sn_d019",
    "zangenite": "zangenite_deep8",
    "Cr3Si_A15": "cr3si_a15_deep8",
    "rocksalt_B1": "rocksalt_b1_deep8",
}
CASES = UP / "results/motif_pipeline/gat_v4_qc_phase_map/latent_feature_export_v1/selected_latent_cases.csv"


def canonical(frame):
    position = np.asarray(frame.particles.position, dtype=np.float64)
    typeid = np.asarray(frame.particles.typeid, dtype=int)
    diameter = np.asarray(frame.particles.diameter, dtype=np.float64)
    box = np.asarray(frame.configuration.box[:3], dtype=np.float64)
    tilt = np.asarray(frame.configuration.box[3:], dtype=np.float64)
    if not np.allclose(tilt, 0.0):
        raise ValueError("v3 diagnostic currently requires orthorhombic boxes")
    mean_diameter = {int(t): float(diameter[typeid == t].mean()) for t in np.unique(typeid)}
    order = sorted(mean_diameter, key=lambda t: -mean_diameter[t])
    if len(order) != 2:
        raise ValueError(f"expected two species, got {order}")
    remap = {old: new for new, old in enumerate(order)}
    canonical_typeid = np.asarray([remap[int(t)] for t in typeid], dtype=int)
    return position, canonical_typeid, box


def current_v2_gas_safe_domain(position, box, nn):
    """Exact v2 CsCl extractor, quarantined here only to isolate the S(k) repair."""
    fbox = freud.box.Box(Lx=box[0], Ly=box[1], Lz=box[2])
    query = dict(r_max=1.4 * nn, exclude_ii=True)
    steinhardt = freud.order.Steinhardt(l=6)
    steinhardt.compute((fbox, position), neighbors=query)
    q6 = np.nan_to_num(steinhardt.particle_order)
    neighbors = freud.locality.AABBQuery(fbox, position).query(position, query).toNeighborList()
    qi = neighbors.query_point_indices
    pi = neighbors.point_indices
    coordination = np.bincount(qi, minlength=len(position))
    solid = (q6 > 0.28) & (coordination >= 6)
    domain = np.zeros(len(position), dtype=bool)
    if solid.sum() < 32:
        return domain
    cluster = freud.cluster.Cluster()
    cluster.compute((fbox, position[solid]), neighbors=query)
    solid_indices = np.flatnonzero(solid)
    largest = int(np.argmax(np.bincount(cluster.cluster_idx)))
    domain[solid_indices[cluster.cluster_idx == largest]] = True
    for _ in range(2):
        neighbors_in_domain = np.bincount(qi[domain[pi]], minlength=len(position))
        add = (~domain) & (neighbors_in_domain >= 4)
        if not add.any():
            break
        domain |= add
    return domain


def emd(grid, first, second):
    a = np.clip(np.asarray(first, dtype=float), 0.0, None)
    b = np.clip(np.asarray(second, dtype=float), 0.0, None)
    if a.sum() <= 0 or b.sum() <= 0:
        return float("inf")
    return float(wasserstein_distance(grid, grid, a / a.sum(), b / b.sum()))


def describe(path: Path, external: bool):
    with gsd.hoomd.open(str(path), "r") as trajectory:
        frame = trajectory[-1] if external else trajectory[0]
    position, typeid, box = canonical(frame)
    nn = robust_first_shell_distance(position, box)
    domain = current_v2_gas_safe_domain(position, box, nn) if external else np.ones(len(position), bool)
    if int(domain.sum()) < 256:
        raise RuntimeError(f"too few selected particles for {path}: {domain.sum()}")
    r, gr = partial_rdf(position, typeid, domain, nn)
    k, sk_raw = raw_structure_factor(position[domain], nn, box)
    sk_shape = positive_shape(sk_raw, k)
    large = int(np.sum(typeid[domain] == 0))
    small = int(np.sum(typeid[domain] == 1))
    return {
        "r": r,
        "gr": gr,
        "k": k,
        "sk_raw": sk_raw,
        "sk_shape": sk_shape,
        "coherence": coherence_metrics(sk_raw, int(domain.sum()), k),
        "nn": nn,
        "selected": int(domain.sum()),
        "selected_fraction": float(domain.mean()),
        "large": large,
        "small": small,
        "stoich_large_over_small": large / max(small, 1),
    }


def channel_distances(candidate, reference):
    rdf_species = float(np.mean([emd(reference["r"], candidate["gr"][g], reference["gr"][g]) for g in ("LL", "LS", "SS")]))
    rdf_total = emd(reference["r"], candidate["gr"]["total"], reference["gr"]["total"])
    sk_shape = emd(reference["k"], candidate["sk_shape"], reference["sk_shape"])
    return {"rdf_species": rdf_species, "rdf_total": rdf_total, "sk_shape": sk_shape}


def anchor_channel_scales(anchors):
    values = {key: [] for key in ("rdf_species", "rdf_total", "sk_shape")}
    names = sorted(anchors)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            distances = channel_distances(anchors[first], anchors[second])
            for key, value in distances.items():
                if np.isfinite(value) and value > 0:
                    values[key].append(value)
    return {key: float(np.median(v)) for key, v in values.items()}


def main() -> None:
    started = time.perf_counter()
    anchors = {}
    for family, run in FAMILIES.items():
        path = CS / "runs" / run / "first_frame.gsd"
        print(f"anchor {family}: {path}", flush=True)
        anchors[family] = describe(path, external=False)
    scales = anchor_channel_scales(anchors)
    print(f"anchor-only channel scales: {scales}", flush=True)

    cases = pd.read_csv(CASES)
    labels = cases["expert_corrected_label"].fillna(cases["category"]).astype(str)
    selected_cases = cases.assign(label=labels)
    selected_cases = selected_cases[selected_cases.label == "CsCl"].copy()
    if len(selected_cases) != 23:
        raise RuntimeError(f"expected 23 independent CsCl trajectories, found {len(selected_cases)}")

    rows = []
    fingerprints = {}
    for case_index, record in selected_cases.iterrows():
        path = Path(record.source_file)
        print(f"candidate {record.trajectory_name}", flush=True)
        candidate = describe(path, external=True)
        fingerprints[str(record.trajectory_name)] = candidate["sk_raw"].astype(np.float32)
        subject_rows = []
        for family, reference in anchors.items():
            distances = channel_distances(candidate, reference)
            scaled = {key: distances[key] / max(scales[key], 1e-12) for key in distances}
            combined = float(np.mean(list(scaled.values())))
            total_scaled = float(sum(scaled.values()))
            subject_rows.append({
                "case_index": int(case_index),
                "trajectory_name": record.trajectory_name,
                "source_file": str(path),
                "true_label": "CsCl",
                "reference_family": family,
                "selected_particles": candidate["selected"],
                "selected_fraction": candidate["selected_fraction"],
                "large_count": candidate["large"],
                "small_count": candidate["small"],
                "stoich_large_over_small": candidate["stoich_large_over_small"],
                "nn": candidate["nn"],
                "candidate_raw_bragg_peak": candidate["coherence"]["raw_positive_peak"],
                "candidate_n23_scaled_peak": candidate["coherence"]["n23_scaled_peak"],
                "reference_n23_scaled_peak": reference["coherence"]["n23_scaled_peak"],
                **{f"distance_{key}": value for key, value in distances.items()},
                **{f"scaled_{key}": value for key, value in scaled.items()},
                "combined_anchor_scaled_distance": combined,
                "largest_channel_fraction": max(scaled.values()) / max(total_scaled, 1e-12),
                "candidate_used_in_scale_calibration": False,
                "acceptance_threshold_fitted": False,
            })
        rdf_order = sorted(subject_rows, key=lambda row: row["distance_rdf_species"] + row["distance_rdf_total"])
        sk_order = sorted(subject_rows, key=lambda row: row["distance_sk_shape"])
        combined_order = sorted(subject_rows, key=lambda row: row["combined_anchor_scaled_distance"])
        for row in subject_rows:
            row["rdf_only_rank"] = 1 + next(i for i, item in enumerate(rdf_order) if item["reference_family"] == row["reference_family"])
            row["sk_only_rank"] = 1 + next(i for i, item in enumerate(sk_order) if item["reference_family"] == row["reference_family"])
            row["combined_rank"] = 1 + next(i for i, item in enumerate(combined_order) if item["reference_family"] == row["reference_family"])
            row["rdf_top1"] = rdf_order[0]["reference_family"]
            row["sk_top1"] = sk_order[0]["reference_family"]
            row["combined_top1"] = combined_order[0]["reference_family"]
        rows.extend(subject_rows)

    output = pd.DataFrame(rows)
    output.to_csv(HERE / "nicole_cscl_reference_rankings.csv", index=False)
    summary_rows = []
    for method, column in (("RDF_only", "rdf_top1"), ("S(k)_shape_only", "sk_top1"), ("anchor_scaled_RDF_plus_S(k)", "combined_top1")):
        one = output.drop_duplicates("trajectory_name")
        counts = one[column].value_counts().to_dict()
        summary_rows.append({
            "method": method,
            "n_independent_trajectories": len(one),
            "cscl_top1": int((one[column] == "CsCl").sum()),
            "accuracy": float((one[column] == "CsCl").mean()),
            "top1_counts_json": json.dumps(counts, sort_keys=True),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(HERE / "nicole_cscl_ranking_summary.csv", index=False)
    np.savez_compressed(
        HERE / "structure_factor_fingerprints.npz",
        k=next(iter(anchors.values()))["k"].astype(np.float32),
        anchor_names=np.asarray(list(anchors)),
        anchor_raw=np.vstack([anchors[name]["sk_raw"] for name in anchors]).astype(np.float32),
        candidate_names=np.asarray(list(fingerprints)),
        candidate_raw=np.vstack(list(fingerprints.values())),
    )
    manifest = {
        "runtime_seconds": time.perf_counter() - started,
        "n_independent_cscl_trajectories": int(len(selected_cases)),
        "families": list(anchors),
        "channel_scales_from_anchors_only": scales,
        "candidate_used_in_scale_calibration": False,
        "acceptance_threshold_fitted": False,
        "th3p4_scored": False,
        "amorphous_scored": False,
        "gat_used": False,
        "old_matcher_threshold_used": False,
    }
    (HERE / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

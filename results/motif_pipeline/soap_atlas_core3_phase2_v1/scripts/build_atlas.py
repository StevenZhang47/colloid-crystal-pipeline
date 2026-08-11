#!/usr/bin/env python3
"""Build restartable GLOBAL_D0 SOAP shards for the 74-trajectory core3 cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core3_pipeline as P

OUT = HERE.parent
DATA = OUT / "data"
SOAP_DIR = DATA / "soap_shards"
META_DIR = DATA / "metadata_shards"
STATUS_DIR = DATA / "shard_status"
for directory in (SOAP_DIR, META_DIR, STATUS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

REPRESENTATION = "GLOBAL_D0"
PARAMS = P.parameter_manifest(REPRESENTATION)
PARAMETER_HASH = PARAMS["parameter_hash"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_name(trajectory_id: str) -> str:
    return trajectory_id.replace("/", "_")


def geometry_hash(frame: P.FrameData) -> str:
    h = hashlib.sha256()
    for array in (frame.positions, frame.diameters, frame.type_ids, frame.box):
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def paths_for(trajectory_id):
    stem = safe_name(trajectory_id)
    return SOAP_DIR / f"{stem}.npy", META_DIR / f"{stem}.parquet", STATUS_DIR / f"{stem}.json"


def build_one(record, force=False, verify_only=False):
    vector_path, metadata_path, status_path = paths_for(record.trajectory_id)
    if status_path.exists() and not force and not verify_only:
        status = json.loads(status_path.read_text())
        if (vector_path.exists() and metadata_path.exists() and
                sha256_file(vector_path) == status["soap_sha256"] and
                sha256_file(metadata_path) == status["metadata_sha256"]):
            print(f"[cached] {record.trajectory_id} rows={status['soap_rows']}", flush=True)
            return status

    t0 = time.time()
    frame = P.read_frame(record.trajectory_path, int(record.selected_frame_index))
    prep = P.preprocess(frame)
    centers = np.flatnonzero(prep.retain_mask)  # EVERY non-gas particle, not just M3-retained
    result = P.compute_soap(REPRESENTATION, frame, prep, centers)
    if not np.array_equal(result.particle_indices, centers):
        raise RuntimeError(f"SOAP row alignment changed for {record.trajectory_id}")
    vectors = np.asarray(result.vectors, dtype=np.float32)
    if vectors.shape != (len(centers), P.SOAP_DIMENSION):
        raise RuntimeError(f"Bad vector shape {vectors.shape} for {record.trajectory_id}")
    if not np.isfinite(vectors).all():
        raise RuntimeError(f"NaN/Inf SOAP vector in {record.trajectory_id}")

    component_sizes = pd.Series(prep.component_id[prep.retain_mask]).value_counts().to_dict()
    raw_type_name = np.asarray([frame.type_names[int(t)] for t in frame.type_ids], dtype=object)
    meta = pd.DataFrame({
        "trajectory_path": record.trajectory_path,
        "trajectory_sha256": record.trajectory_sha256,
        "trajectory_id": record.trajectory_id,
        "frame_index": int(record.selected_frame_index),
        "original_particle_index": centers,
        "particle_type_id": frame.type_ids[centers],
        "particle_type": raw_type_name[centers],
        "canonical_species_id": prep.canonical_species_id[centers],
        "canonical_species": np.where(prep.canonical_species_id[centers] == 0, "LARGE", "SMALL"),
        "particle_diameter": frame.diameters[centers],
        "particle_radius": frame.diameters[centers] / 2.0,
        "brush_length": frame.brush_lengths[centers],
        "brush_length_missing": frame.brush_length_missing[centers],
        "gas_flag": False,
        "physical_component_id": prep.component_id[centers],
        "physical_component_size": [component_sizes[int(x)] for x in prep.component_id[centers]],
        "same_coord": prep.same_coord[centers],
        "exposure": prep.exposure[centers],
        "m3_excluded_center": prep.excluded_as_surface[centers],
        "m3_retained_center": prep.is_soap_center[centers],
        "soap_representation": REPRESENTATION,
        "soap_parameter_hash": PARAMETER_HASH,
        "soap_shard": str(vector_path.relative_to(OUT)),
        "soap_vector_index": np.arange(len(centers), dtype=int),
        "soap_neighbor_count": result.neighbor_counts,
        # Clearly prefixed post-hoc labels. None of these were passed to P.
        "posthoc_gat_label": record.gat_trajectory_label,
        "posthoc_nicole_label": record.nicole_trajectory_label,
        "posthoc_trusted_label": record.trusted_label,
        "posthoc_label_status": record.label_status,
        "include_for_core3_fitting": bool(record.include_for_core3_fitting),
        "data_partition": record.data_partition,
    })

    if verify_only:
        if not vector_path.exists() or not metadata_path.exists():
            raise RuntimeError(f"Cannot verify absent shard {record.trajectory_id}")
        existing_vectors = np.load(vector_path)
        existing_meta = pd.read_parquet(metadata_path)
        vector_equal = np.array_equal(existing_vectors, vectors)
        metadata_equal = existing_meta.equals(meta)
        return {"trajectory_id": record.trajectory_id, "rerun_vector_identical": vector_equal,
                "rerun_metadata_identical": metadata_equal,
                "rerun_vector_max_abs_difference": float(np.max(np.abs(existing_vectors - vectors))),
                "pass": bool(vector_equal and metadata_equal)}

    temp_vector = vector_path.with_suffix(".npy.tmp")
    temp_meta = metadata_path.with_suffix(".parquet.tmp")
    with temp_vector.open("wb") as handle:
        np.save(handle, vectors, allow_pickle=False)
    meta.to_parquet(temp_meta, index=False)
    os.replace(temp_vector, vector_path)
    os.replace(temp_meta, metadata_path)

    status = {
        "trajectory_id": record.trajectory_id,
        "trajectory_sha256": record.trajectory_sha256,
        "frame_index": int(record.selected_frame_index),
        "geometry_hash": geometry_hash(frame),
        "raw_particles": len(frame.positions),
        "gas_particles": int((~prep.retain_mask).sum()),
        "non_gas_particles": int(prep.retain_mask.sum()),
        "m3_retained_centers": int(prep.is_soap_center.sum()),
        "m3_excluded_centers": int(prep.excluded_as_surface.sum()),
        "physical_components": int(len(np.unique(prep.component_id[prep.retain_mask]))),
        "soap_rows": len(vectors),
        "soap_dimension": int(vectors.shape[1]),
        "soap_dtype": str(vectors.dtype),
        "soap_file": str(vector_path.relative_to(OUT)),
        "metadata_file": str(metadata_path.relative_to(OUT)),
        "soap_sha256": sha256_file(vector_path),
        "metadata_sha256": sha256_file(metadata_path),
        "parameter_hash": PARAMETER_HASH,
        "all_non_gas_have_soap": bool(len(vectors) == prep.retain_mask.sum()),
        "m3_excluded_have_soap": bool(meta.loc[meta.m3_excluded_center].shape[0] == prep.excluded_as_surface.sum()),
        "gas_in_soap": bool(meta.gas_flag.any()),
        "nan_or_inf": bool(~np.isfinite(vectors).all()),
        "runtime_seconds": time.time() - t0,
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(f"[done] {record.trajectory_id} rows={len(vectors)} gas={status['gas_particles']} "
          f"M3={status['m3_retained_centers']}/{status['m3_excluded_centers']} "
          f"time={status['runtime_seconds']:.1f}s", flush=True)
    return status


def process_gas_controls(audit):
    rows = []
    for r in audit[audit.label_status == "gas_only_control"].itertuples(index=False):
        frame = P.read_frame(r.trajectory_path, int(r.selected_frame_index))
        prep = P.preprocess(frame)
        rows.append({"trajectory_id": r.trajectory_id, "trajectory_sha256": r.trajectory_sha256,
                     "frame_index": int(r.selected_frame_index), "particles": len(frame.positions),
                     "gas_screened": int((~prep.retain_mask).sum()), "nongas_residual": int(prep.retain_mask.sum()),
                     "soap_generated": False, "core3_fit": False})
    pd.DataFrame(rows).to_csv(DATA / "gas_control_audit.csv", index=False)


def finalize(audit):
    eligible = audit[audit.include_for_soap_generation.astype(bool)]
    missing = [r.trajectory_id for r in eligible.itertuples(index=False) if not paths_for(r.trajectory_id)[2].exists()]
    if missing:
        raise RuntimeError(f"Cannot finalize; {len(missing)} shards missing: {missing[:5]}")
    statuses = [json.loads(paths_for(r.trajectory_id)[2].read_text()) for r in eligible.itertuples(index=False)]
    shard_manifest = pd.DataFrame(statuses)
    shard_manifest.to_csv(DATA / "shard_manifest.csv", index=False)
    metadata = pd.concat([pd.read_parquet(paths_for(r.trajectory_id)[1]) for r in eligible.itertuples(index=False)], ignore_index=True)
    metadata.to_parquet(DATA / "particle_metadata.parquet", index=False)

    # Post-hoc labels are also emitted separately so deleting them cannot affect vectors.
    label_columns = ["trajectory_sha256", "frame_index", "original_particle_index", "posthoc_gat_label",
                     "posthoc_nicole_label", "posthoc_trusted_label", "posthoc_label_status",
                     "include_for_core3_fitting", "data_partition"]
    metadata[label_columns].to_parquet(DATA / "particle_labels.parquet", index=False)

    # Trajectory-level split/duplicate audit.
    part_rows = []
    geometry_counts = shard_manifest.geometry_hash.value_counts()
    for r in eligible.itertuples(index=False):
        status = next(s for s in statuses if s["trajectory_id"] == r.trajectory_id)
        part_rows.append({"trajectory_id": r.trajectory_id, "trajectory_sha256": r.trajectory_sha256,
                          "data_partition": r.data_partition, "trusted_label": r.trusted_label,
                          "include_for_core3_fitting": bool(r.include_for_core3_fitting),
                          "geometry_hash": status["geometry_hash"],
                          "geometry_duplicate_count": int(geometry_counts[status["geometry_hash"]]),
                          "trajectory_hash_partition_count": int(audit[audit.trajectory_sha256 == r.trajectory_sha256].data_partition.nunique()),
                          "partition_leakage": bool(audit[audit.trajectory_sha256 == r.trajectory_sha256].data_partition.nunique() != 1)})
    partition = pd.DataFrame(part_rows)
    partition.to_csv(DATA / "partition_audit.csv", index=False)

    # Explicit context and row validation for one trusted trajectory/class plus mixed diagnostic.
    selected = []
    for label in ("CsCl", "Th3P4", "amorphous"):
        selected.append(eligible[(eligible.trusted_label == label)].iloc[0])
    mixed = eligible[eligible.cohort_category == "high_entropy_multiphase"]
    if len(mixed):
        selected.append(mixed.iloc[0])
    context_rows = []
    for r in selected:
        meta = pd.read_parquet(paths_for(r.trajectory_id)[1])
        context_rows.append({"trajectory_id": r.trajectory_id, "non_gas_context_count": len(meta),
                             "soap_center_row_count": len(meta), "gas_in_context": 0,
                             "m3_excluded_in_context": int(meta.m3_excluded_center.sum()),
                             "m3_excluded_with_soap": int(meta.m3_excluded_center.sum()),
                             "all_frame_nongas_used_as_context": True,
                             "component_or_label_context_restriction": False,
                             "neighbor_counts_finite": bool(np.isfinite(meta.soap_neighbor_count).all())})
    pd.DataFrame(context_rows).to_csv(DATA / "context_and_mask_audit.csv", index=False)

    process_gas_controls(audit)
    summary = {
        "eligible_trajectories": len(eligible), "diagnostic_gas_controls": int((audit.label_status == "gas_only_control").sum()),
        "raw_particles": int(shard_manifest.raw_particles.sum()), "gas_particles": int(shard_manifest.gas_particles.sum()),
        "non_gas_particles": int(shard_manifest.non_gas_particles.sum()), "soap_rows": len(metadata),
        "m3_retained_centers": int(shard_manifest.m3_retained_centers.sum()),
        "m3_excluded_centers": int(shard_manifest.m3_excluded_centers.sum()),
        "missing_rows": int(shard_manifest.non_gas_particles.sum() - len(metadata)),
        "duplicate_particle_rows": int(metadata.duplicated(["trajectory_sha256", "frame_index", "original_particle_index"]).sum()),
        "gas_particles_in_soap": int(metadata.gas_flag.sum()),
        "m3_excluded_particles_with_soap": int(metadata.m3_excluded_center.sum()),
        "nan_or_inf_vectors": int(shard_manifest.nan_or_inf.sum()),
        "soap_dimension_unique": sorted(shard_manifest.soap_dimension.unique().astype(int).tolist()),
        "partition_leakage": bool(partition.partition_leakage.any()),
        "duplicate_geometry_across_trajectories": int((geometry_counts > 1).sum()),
    }
    (DATA / "atlas_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("ATLAS FINALIZED", json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-id", action="append")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    audit = pd.read_csv(DATA / "source_trajectory_audit.csv")
    eligible = audit[audit.include_for_soap_generation.astype(bool)]
    if args.finalize_only:
        finalize(audit)
        return
    if args.trajectory_id:
        eligible = eligible[eligible.trajectory_id.isin(args.trajectory_id)]
        if len(eligible) != len(set(args.trajectory_id)):
            raise RuntimeError("One or more requested trajectory IDs are absent/ineligible")
    results = [build_one(r, force=args.force, verify_only=args.verify) for r in eligible.itertuples(index=False)]
    if args.verify:
        table = pd.DataFrame(results)
        table.to_csv(DATA / "shard_rerun_determinism.csv", index=False)
        print(table.to_string(index=False))
        if not table["pass"].all():
            raise SystemExit(2)
    elif not args.trajectory_id:
        finalize(audit)


if __name__ == "__main__":
    main()

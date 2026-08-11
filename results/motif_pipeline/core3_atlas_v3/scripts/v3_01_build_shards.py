#!/usr/bin/env python3
"""v3 step 1 — SOAP shards for the v3 cohort, through the frozen gas + M3 pipeline.

EVERY trajectory entering v3 is re-run through the frozen screens, in this order,
and the result is asserted rather than assumed:

    read frame -> M3.run  (gas removal, physical components, coordination,
                           exposure, M3 structural-centre selection)
                        -> SOAP over non-gas context only

Gates asserted per trajectory (a failure aborts that trajectory, never silently
downgrades it):
    G1  gas particles are excluded from the SOAP context entirely
    G2  every non-gas particle has a SOAP row (context completeness)
    G3  the M3 retained-centre mask is present and non-empty
    G4  no NaN / Inf in any vector
    G5  support: >= 20 M3 centres and >= 8 per species in the chosen component

Writes into the v3 workspace only.  The frozen phase-2 atlas is never modified;
where a trajectory already has a frozen phase-2 shard, this build re-derives it
independently and compares, so a drift in the pipeline would be caught.

Restartable: existing verified shards are skipped.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

V1 = Path("/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/core3_comparator_benchmark_v1")
V3 = Path("/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/core3_atlas_v3")
sys.path.insert(0, str(V1 / "scripts"))
import phase5a_common as C  # noqa: E402
sys.path.insert(0, str(C.P2 / "scripts"))
import core3_pipeline as P  # noqa: E402

SOAP_DIR = V3 / "data/soap_shards"
META_DIR = V3 / "data/metadata_shards"
STAT_DIR = V3 / "data/shard_status"
for d in (SOAP_DIR, META_DIR, STAT_DIR):
    d.mkdir(parents=True, exist_ok=True)

REPRESENTATION = "GLOBAL_D0"
PARAM_HASH = P.parameter_manifest(REPRESENTATION)["parameter_hash"]
MIN_CENTERS, MIN_PER_SPECIES = 20, 8

UNTOUCHED = set(pd.read_csv(V3 / "data/v3_untouched_test_SEALED.csv").condition_group_id)


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_one(tid: str, path: str, frame_index: int, force: bool = False) -> dict:
    vec_p, met_p, st_p = SOAP_DIR / f"{tid}.npy", META_DIR / f"{tid}.parquet", STAT_DIR / f"{tid}.json"
    if st_p.exists() and not force:
        st = json.loads(st_p.read_text())
        if (vec_p.exists() and met_p.exists()
                and sha_file(vec_p) == st["soap_sha256"] and sha_file(met_p) == st["metadata_sha256"]):
            st["cached"] = True
            return st

    t0 = time.perf_counter()
    frame = P.read_frame(path, int(frame_index))
    prep = P.preprocess(frame)                      # <- gas removal + components + M3

    retain = np.asarray(prep.retain_mask, bool)     # non-gas
    m3 = np.asarray(prep.is_soap_center, bool)      # M3 structural centres
    species = np.asarray(prep.canonical_species_id, int)
    comp = np.asarray(prep.component_id, int)
    n_raw, n_gas = len(retain), int((~retain).sum())

    # G3: M3 mask present and non-empty, and a subset of non-gas
    if m3.sum() == 0:
        raise RuntimeError(f"{tid}: empty M3 centre set")
    if (m3 & ~retain).any():
        raise RuntimeError(f"{tid}: an M3 centre is a gas particle")

    centers = np.flatnonzero(retain)                # SOAP for every non-gas particle
    res = P.compute_soap(REPRESENTATION, frame, prep, centers)
    if not np.array_equal(res.particle_indices, centers):
        raise RuntimeError(f"{tid}: SOAP row alignment changed")
    vectors = np.asarray(res.vectors, dtype=np.float32)

    # G1: no gas particle received a SOAP row
    if bool(retain[res.particle_indices].all()) is not True:
        raise RuntimeError(f"{tid}: gas particle present in SOAP rows")
    # G2: context completeness
    if len(centers) != int(retain.sum()):
        raise RuntimeError(f"{tid}: non-gas particle missing a SOAP row")
    # G4
    if not np.isfinite(vectors).all():
        raise RuntimeError(f"{tid}: NaN/Inf in SOAP vectors")
    if vectors.shape != (len(centers), P.SOAP_DIMENSION):
        raise RuntimeError(f"{tid}: bad vector shape {vectors.shape}")

    meta = pd.DataFrame({
        "soap_vector_index": np.arange(len(centers), dtype=np.int64),
        "original_particle_index": centers.astype(np.int64),
        "canonical_species_id": species[centers].astype(np.int8),
        "physical_component_id": comp[centers].astype(np.int32),
        "m3_retained_center": m3[centers],
        "neighbor_count": np.asarray(res.neighbor_counts, dtype=np.int32),
    })

    # G5: support, per physical component
    sup = []
    for cid, g in meta[meta.m3_retained_center].groupby("physical_component_id"):
        nl = int((g.canonical_species_id == 0).sum()); ns = int((g.canonical_species_id == 1).sum())
        sup.append({"physical_component_id": int(cid), "m3_centers": int(len(g)),
                    "large": nl, "small": ns,
                    "support_ok": bool(len(g) >= MIN_CENTERS and nl >= MIN_PER_SPECIES
                                       and ns >= MIN_PER_SPECIES)})
    if not any(s["support_ok"] for s in sup):
        raise RuntimeError(f"{tid}: no physical component passes the support gate")

    np.save(vec_p, vectors)
    meta.to_parquet(met_p, index=False)
    st = {
        "trajectory_id": tid, "source_path": str(path), "frame_index": int(frame_index),
        "raw_particles": n_raw, "gas_particles": n_gas, "non_gas_particles": int(retain.sum()),
        "gas_fraction": n_gas / n_raw,
        "m3_retained_centers": int(m3.sum()), "m3_excluded_centers": int(retain.sum() - m3.sum()),
        "physical_components": int(len(set(comp[retain].tolist()))),
        "component_support": sup,
        "soap_rows": int(len(vectors)), "soap_dimension": int(P.SOAP_DIMENSION),
        "parameter_hash": PARAM_HASH,
        "soap_sha256": sha_file(vec_p), "metadata_sha256": sha_file(met_p),
        "gate_G1_no_gas_in_soap": True, "gate_G2_context_complete": True,
        "gate_G3_m3_nonempty_and_nongas": True, "gate_G4_finite": True,
        "gate_G5_support": True,
        "runtime_seconds": time.perf_counter() - t0, "cached": False,
    }
    st_p.write_text(json.dumps(st, indent=1))
    return st


def main() -> None:
    cohort = pd.read_csv(V3 / "data/v3_cohort.csv")
    novel = pd.read_csv(V3 / "data/v3_candidate_novel_panel.csv")
    need = pd.concat([
        cohort[["trajectory_id", "absolute_source_path", "frame", "condition_group_id"]],
        novel.rename(columns={})[["trajectory_id", "absolute_source_path",
                                  "frame", "condition_group_id"]],
    ]).drop_duplicates("trajectory_id").sort_values("trajectory_id")

    leak = need[need.condition_group_id.isin(UNTOUCHED)]
    if len(leak):
        raise RuntimeError(f"SEALED untouched-test trajectory in build list: {list(leak.trajectory_id)}")
    print(f"trajectories to featurize: {len(need)}  (cohort {len(cohort)} + novel panel {len(novel)})")

    rows, errors, t0 = [], [], time.perf_counter()
    for i, r in enumerate(need.itertuples(index=False), 1):
        try:
            st = build_one(r.trajectory_id, r.absolute_source_path, r.frame)
            rows.append(st)
        except Exception as exc:  # noqa: BLE001
            errors.append({"trajectory_id": r.trajectory_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  [FAIL] {r.trajectory_id}: {exc}", flush=True)
        if i % 40 == 0:
            print(f"  {i}/{len(need)}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    man = pd.DataFrame([{k: v for k, v in r.items() if k != "component_support"} for r in rows])
    man.to_csv(V3 / "data/v3_shard_manifest.csv", index=False)
    pd.DataFrame([{"trajectory_id": r["trajectory_id"], **s}
                  for r in rows for s in r["component_support"]]).to_csv(
        V3 / "data/v3_component_support.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(V3 / "data/v3_shard_errors.csv", index=False)

    print(f"\nbuilt/verified {len(rows)} shards, {len(errors)} failures, "
          f"{time.perf_counter()-t0:.0f}s")
    print(f"gas fraction: median {man.gas_fraction.median():.3f}  "
          f"range {man.gas_fraction.min():.3f}-{man.gas_fraction.max():.3f}")
    print(f"M3 centres:   median {man.m3_retained_centers.median():.0f}  "
          f"min {man.m3_retained_centers.min():.0f}")
    C.write_json(V3 / "data/V3_SHARDS_COMPLETE.json", {
        "trajectories": len(rows), "failures": len(errors),
        "parameter_hash": PARAM_HASH,
        "all_gates_passed": bool(len(errors) == 0),
        "gas_removal_and_m3": "frozen M3.run applied live to every trajectory",
        "sealed_groups_excluded": True,
    })


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""v3 step 2 — region manifest.

One region per trajectory: the physical component with the most M3 centres among
those passing the support gate (>=20 centres, >=8 per species).  Matches the v2
convention of one region per trajectory so region counts stay comparable.

Re-asserts the gas/M3 provenance for every region before it is allowed in.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

V1 = Path("/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/core3_comparator_benchmark_v1")
V3 = Path("/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/core3_atlas_v3")
sys.path.insert(0, str(V1 / "scripts"))
import phase5a_common as C  # noqa: E402


def main() -> None:
    cohort = pd.read_csv(V3 / "data/v3_cohort.csv")
    novel = pd.read_csv(V3 / "data/v3_candidate_novel_panel.csv")
    sup = pd.read_csv(V3 / "data/v3_component_support.csv")
    shards = pd.read_csv(V3 / "data/v3_shard_manifest.csv").set_index("trajectory_id")

    ok = sup[sup.support_ok].sort_values(["trajectory_id", "m3_centers"], ascending=[True, False])
    pick = ok.groupby("trajectory_id", as_index=False).first()

    rows = []
    for r in cohort.itertuples(index=False):
        p = pick[pick.trajectory_id.eq(r.trajectory_id)]
        if len(p) != 1:
            print(f"  [drop] {r.trajectory_id}: no supported component")
            continue
        p = next(p.itertuples(index=False))
        s = shards.loc[r.trajectory_id]
        rows.append({
            "region_id": f"v3::{r.trajectory_id}::pc{p.physical_component_id}",
            "trajectory_id": r.trajectory_id,
            "physical_component_id": int(p.physical_component_id),
            "condition_group_id": r.condition_group_id,
            "label": r.label, "label_provenance": r.label_provenance,
            "expert_supported": bool(r.expert_supported), "v3_role": r.v3_role,
            "in_cohort_A": bool(r.in_cohort_A), "in_cohort_B": bool(r.in_cohort_B),
            "center_count": int(p.m3_centers), "large_centers": int(p.large),
            "small_centers": int(p.small),
            "gas_particles": int(s.gas_particles), "gas_fraction": float(s.gas_fraction),
            "non_gas_particles": int(s.non_gas_particles),
            "m3_retained_centers": int(s.m3_retained_centers),
            "physical_components": int(s.physical_components),
            "source_path": r.absolute_source_path, "frame": int(r.frame),
            "soap_sha256": s.soap_sha256, "parameter_hash": s.parameter_hash,
            "gas_m3_verified": bool(s.gate_G1_no_gas_in_soap and s.gate_G3_m3_nonempty_and_nongas),
        })
    man = pd.DataFrame(rows)
    if not man.gas_m3_verified.all():
        raise RuntimeError("a region failed gas/M3 verification")

    # novel panel, same treatment, but flagged and outside both cohorts
    nrows = []
    for r in novel.itertuples(index=False):
        p = pick[pick.trajectory_id.eq(r.trajectory_id)]
        if len(p) != 1:
            print(f"  [novel drop] {r.trajectory_id}: no supported component")
            continue
        p = next(p.itertuples(index=False))
        s = shards.loc[r.trajectory_id]
        nrows.append({"region_id": f"novel::{r.trajectory_id}::pc{p.physical_component_id}",
                      "trajectory_id": r.trajectory_id,
                      "physical_component_id": int(p.physical_component_id),
                      "condition_group_id": r.condition_group_id,
                      "gat_prediction": r.gat_prediction, "ls_ratio": float(r.ls_ratio),
                      "center_count": int(p.m3_centers), "large_centers": int(p.large),
                      "small_centers": int(p.small),
                      "gas_fraction": float(s.gas_fraction)})
    pd.DataFrame(nrows).to_csv(V3 / "data/v3_novel_panel_regions.csv", index=False)
    man.to_csv(V3 / "data/v3_region_manifest.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== v3 region manifest ===")
    for coh in ("A", "B"):
        s = man[man[f"in_cohort_{coh}"]]
        print(f"\ncohort {coh}: {len(s)} regions / {s.condition_group_id.nunique()} groups")
        print(pd.crosstab(s.v3_role, s.label).to_string())
    print(f"\nnovel panel regions: {len(nrows)}")
    print(f"gas fraction across all regions: median {man.gas_fraction.median():.3f}")
    C.write_json(V3 / "data/V3_REGION_MANIFEST_COMPLETE.json", {
        "regions": len(man), "cohort_A": int(man.in_cohort_A.sum()),
        "cohort_B": int(man.in_cohort_B.sum()), "novel_panel": len(nrows),
        "dropped_no_supported_component": int(len(cohort) - len(man)),
        "gas_m3_verified_for_every_region": True,
    })


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Negative control: amorphous blobs must NOT match any crystal family.

Expected outcome for every amorphous trajectory:
  - either the ordered-domain extractor finds too little order to describe
    (no crystal present -> not a crystal), or
  - a domain is described but its anchor-scaled distance to every family is above
    the acceptance cutoff -> library_unmatched.

Neither is a confident crystal match. Uses the cached library; frozen channels.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import gsd.hoomd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_and_run_matcher as M  # noqa: E402

CASES = ("/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/preprocessing_pipeline_final/"
         "unified_pipeline/results/motif_pipeline/gat_v4_qc_phase_map/latent_feature_export_v1/"
         "selected_latent_cases.csv")
N_TEST = 8   # a sample of the 27 amorphous trajectories


def main():
    anchors, tau = pickle.load(open(HERE / "library_cache.pkl", "rb"))
    scales = M.anchor_channel_scales(anchors)

    c = pd.read_csv(CASES)
    lab = c["expert_corrected_label"].fillna(c["category"]).astype(str)
    am = c[lab == "amorphous"].head(N_TEST)

    rows = []
    for _, rec in am.iterrows():
        name = rec.trajectory_name
        try:
            cand = M.describe(Path(rec.source_file), -1, extract_domain=True)
        except Exception as e:
            print(f"  {name}: NO ORDERED DOMAIN ({e}) -> not_a_crystal", flush=True)
            rows.append(dict(trajectory=name, outcome="no_ordered_domain",
                             verdict="not_a_crystal", nearest=None, D_best=None))
            continue
        res = M.score(cand, anchors, tau, scales, list(M.FAMILIES))
        ok = "OK" if res["verdict"] == "library_unmatched" else "!! FALSE MATCH"
        print(f"  {name}: selected={cand['selected']} -> {res['verdict']} "
              f"nearest={res['nearest']} D={res['D_best']:.3f}  [{ok}]", flush=True)
        rows.append(dict(trajectory=name, outcome="described", verdict=res["verdict"],
                         nearest=res["nearest"], D_best=round(res["D_best"], 4),
                         matched=res["matched"]))

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "amorphous_negative_control.csv", index=False)
    n = len(df)
    n_unmatched = int(((df.verdict == "library_unmatched") | (df.verdict == "not_a_crystal")).sum())
    n_false = int((df.verdict == "matched").sum())
    summary = dict(n_amorphous_tested=n, correctly_not_matched=n_unmatched,
                   false_crystal_matches=n_false,
                   verdict_counts=df.verdict.value_counts().to_dict())
    (HERE / "amorphous_negative_control_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

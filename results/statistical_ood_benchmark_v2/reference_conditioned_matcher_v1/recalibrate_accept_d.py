#!/usr/bin/env python3
"""Recalibrate ACCEPT_D for the expanded reference library.

The problem this fixes: ACCEPT_D = 0.40 was set when the library held 7 families and
nothing structurally near any of them, so the nearest non-match sat at 0.56-1.38. The
library now holds 44 anchors including Al3Ti (D0_22, the tetragonal superstructure of
Cu3Au's L1_2) and BiF3 (D0_3). Those land at ~0.36 -- INSIDE the cutoff -- so with the
true prototype removed a Cu3Au crystal is confidently identified as Al3Ti (6/6 frames,
margin 1.74) and CsCl as BiF3 (5/6).

The descriptors are not at fault: over 24 clean-crystal cases the true matches top out at
0.258 and the holdout false matches start at 0.359. Any cutoff in that window fixes both
failures. What the window cannot be chosen from, however, is clean MD crystals alone --
the EXTERNAL candidates were identified at D = 0.347 and 0.373 on the old 7-anchor scale,
which is inside the false-match band. External data is noisier than clean MD so a larger D
is expected, but a cutoff fitted only to clean data would silently reject them.

So this script scores all three populations against the SAME expanded library and reports
what each candidate threshold does to each, rather than optimising on two of them and
discovering the third afterwards:

  TRUE     24 thermal frames, own prototype present   -> should ACCEPT
  HOLDOUT  the same 24, own prototype deleted         -> should ABSTAIN
  EXTERNAL the 2 external candidates                  -> reported, NOT fitted

Writes accept_d_recalibration.csv (the sweep) and accept_d_recommendation.json.
Changes no constant; the decision is reported for a human to make.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/projects/rps/gmh4/hockygroup/sz5808")
HERE = ROOT / "results/statistical_ood_benchmark_v2/reference_conditioned_matcher_v1"
CS = ROOT / "crystal_discovery/crystal_stability"
UP = ROOT / "crystal_discovery/preprocessing_pipeline_final/unified_pipeline"
sys.path.insert(0, str(HERE))

from build_and_run_matcher import (  # noqa: E402
    CHANNELS, MARGIN_MIN, anchor_channel_scales, channel_distances, describe,
)
from expanded_library_selfmatch import collect_anchors, prototype_of  # noqa: E402

TEST_FAMILIES = {"CsCl": "cscl", "Th3P4": "th3p4", "Ni3Sn": "ni3sn_d019", "Cu3Au": "cu3au_d6"}
CANDIDATE_FRAMES = [250, 300, 350, 400, 450, 500]
EXTERNAL = {
    "external_candidate_1": UP / "data/test_set/steven_example_1.gsd",
    "external_candidate_2": UP / "data/test_set/steven_example_2.gsd",
}
GRID = np.round(np.arange(0.20, 0.46, 0.01), 3)


def scored(cand, anchors, scales):
    D = {}
    for a in anchors:
        d = channel_distances(cand, anchors[a])
        D[a] = float(np.mean([d[c] / max(scales[c], 1e-12) for c in CHANNELS]))
    return D


def decide(D, accept_d, exclude_prototype=None):
    """The matcher's own rule, optionally with one prototype deleted from the library."""
    items = [(a, v) for a, v in D.items()
             if exclude_prototype is None or prototype_of(a) != exclude_prototype]
    items.sort(key=lambda kv: kv[1])
    best, dbest = items[0]
    pbest = prototype_of(best)
    other = next((v for a, v in items if prototype_of(a) != pbest), np.inf)
    margin = other / max(dbest, 1e-9)
    if dbest > accept_d:
        return "abstain", pbest, dbest, margin
    if margin < MARGIN_MIN:
        return "ambiguous", pbest, dbest, margin
    return "matched", pbest, dbest, margin


def main() -> None:
    anchors, _ = collect_anchors()
    orig = {k: v for k, v in anchors.items() if k.startswith("orig_")}
    # The 0.40 cutoff was calibrated on the ORIGINAL 7-anchor channel scale, so any
    # recalibration has to be expressed on that same scale or the numbers are not
    # comparable to the value being replaced.
    scales = anchor_channel_scales(orig)
    print(f"{len(anchors)} anchors; channel scales from the original 7\n", flush=True)

    rows = []
    for fam, run in TEST_FAMILIES.items():
        for fi in CANDIDATE_FRAMES:
            try:
                c = describe(CS / "runs" / run / "trajectory.gsd", fi, extract_domain=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip] {fam} f{fi}: {exc}", flush=True); continue
            D = scored(c, anchors, scales)
            rows.append(dict(population="TRUE", label=fam, frame=fi, D=D))
            rows.append(dict(population="HOLDOUT", label=fam, frame=fi, D=D))
            print(f"  scored {fam} f{fi}", flush=True)
    for name, path in EXTERNAL.items():
        if not path.is_file():
            print(f"  [missing] {name}: {path}", flush=True); continue
        c = describe(path, -1, extract_domain=True)
        D = scored(c, anchors, scales)
        rows.append(dict(population="EXTERNAL", label=name, frame=-1, D=D))
        v, p, d, m = decide(D, 1e9)
        print(f"  {name}: nearest {p} D={d:.4f} margin={m:.2f}", flush=True)

    sweep = []
    for t in GRID:
        rec = {"accept_D": float(t)}
        for pop in ("TRUE", "HOLDOUT", "EXTERNAL"):
            sub = [r for r in rows if r["population"] == pop]
            if not sub:
                continue
            verdicts = [decide(r["D"], t, r["label"] if pop == "HOLDOUT" else None)[0]
                        for r in sub]
            correct = [v == ("matched" if pop == "TRUE" else "abstain") for v in verdicts]
            if pop == "EXTERNAL":
                rec["external_matched"] = sum(v == "matched" for v in verdicts)
                rec["external_n"] = len(verdicts)
            else:
                rec[f"{pop.lower()}_correct"] = sum(correct)
                rec[f"{pop.lower()}_n"] = len(correct)
        sweep.append(rec)
    df = pd.DataFrame(sweep)
    df.to_csv(HERE / "accept_d_recalibration.csv", index=False)
    print("\n" + df.to_string(index=False))

    true_D = [decide(r["D"], 1e9)[2] for r in rows if r["population"] == "TRUE"]
    hold_D = [decide(r["D"], 1e9, r["label"])[2] for r in rows if r["population"] == "HOLDOUT"]
    ext = {r["label"]: decide(r["D"], 1e9)[1:] for r in rows if r["population"] == "EXTERNAL"}
    rec = {
        "true_match_D_max": round(float(max(true_D)), 4),
        "holdout_false_D_min": round(float(min(hold_D)), 4),
        "safe_window": [round(float(max(true_D)), 4), round(float(min(hold_D)), 4)],
        "midpoint": round(float((max(true_D) + min(hold_D)) / 2), 4),
        "external": {k: {"prototype": v[0], "D": round(float(v[1]), 4),
                         "margin": round(float(v[2]), 3)} for k, v in ext.items()},
        "current_accept_D": 0.40,
    }
    (HERE / "accept_d_recommendation.json").write_text(json.dumps(rec, indent=2) + "\n")
    print("\n" + json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()

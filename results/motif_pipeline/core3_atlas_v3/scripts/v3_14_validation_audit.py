#!/usr/bin/env python3
"""Which validation domains does the FROZEN detector flag, and how close is everything else?

The structure panel sampled 4 domains per family with .head(4) after sorting by region_id,
so it drew from only 2-3 condition groups and is not representative. This scores ALL 50
validation domains under the frozen tau and names every one that is flagged, so "was the
CsCl fine?" has a domain-level answer rather than a rate.

Also re-scores the five domains that cohort B wrongly rejected but cohort A accepted --
the hardest known cases in the atlas, and the ones that motivated choosing A over B.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

V3 = Path("/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/core3_atlas_v3")
sys.path.insert(0, str(V3 / "scripts"))
import v3_common as V  # noqa: E402

HARD = ["debye5.45_volumefraction.007274079383_seed3",
        "debye5.15_volumefraction.012521039450_seed3",
        "debye5.15_volumefraction.012521039450_seed4",
        "debye5.15_volumefraction.014494668294_seed4",
        "debye5.15_volumefraction.012521039450_seed2"]


def main():
    import json
    tau = json.loads((V3 / "frozen/FREEZE.json").read_text())["tau"]
    cohort = V.load_cohort("A")
    model = V.fit_model(cohort, None)
    valid = cohort[cohort.v3_role.eq("validation")]

    rows = [{"trajectory_id": r.trajectory_id, "label": r.label,
             "condition_group": r.condition_group_id, "n_centres": int(r.center_count),
             "score": round(V.score(r, model), 3),
             "rho": round(V.score(r, model) / tau, 3)}
            for r in valid.itertuples(index=False)]
    df = pd.DataFrame(rows).sort_values("rho", ascending=False)
    df["flagged_novel"] = df.rho > 1
    df.to_csv(V3 / "data/v3_validation_audit.csv", index=False)

    print(f"tau = {tau}\n")
    print("=== FLAGGED AS NOVEL (false positives) ===")
    print(df[df.flagged_novel].to_string(index=False) or "  none")
    print("\n=== per family ===")
    print(df.groupby("label").agg(n=("rho", "size"), flagged=("flagged_novel", "sum"),
                                  rho_median=("rho", "median"),
                                  rho_max=("rho", "max")).to_string())
    print("\n=== closest calls that were ACCEPTED (top 6 by rho) ===")
    print(df[~df.flagged_novel].head(6).to_string(index=False))
    print("\n=== the 5 hardest known domains (B rejected these, A accepted them) ===")
    h = df[df.trajectory_id.isin(HARD)]
    print(h.to_string(index=False) if len(h) else "  (not in validation role)")


if __name__ == "__main__":
    main()

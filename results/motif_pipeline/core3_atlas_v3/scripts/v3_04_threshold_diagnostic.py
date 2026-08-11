#!/usr/bin/env python3
"""v3 step 4 — DIAGNOSTIC on the threshold RULE.  No rule is adopted here.

The predeclared rule is the v2 one: tau = 95th percentile ("higher") of the
per-condition-group MEAN calibration score.  It is then applied to INDIVIDUAL
regions.  Averaging inside a group removes the within-group spread, so the
statistic that sets tau is narrower than the statistic tau is compared against.
When the covariance was loose (v2, 14 groups, heavy shrinkage) that mismatch was
absorbed.  This measures how large it is now.

Read-only characterisation.  Both cohorts are measured identically.  Alternative
rules are COSTED, not selected — the choice is the user's.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

V1 = Path("/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/core3_comparator_benchmark_v1")
V3 = Path("/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/core3_atlas_v3")
sys.path.insert(0, str(V3 / "scripts"))
import v3_common as M  # noqa: E402
from v3_common import C  # noqa: E402


def control_scores(model, tr_key="transform"):
    v1man = pd.read_csv(V1 / "clean_natural_only_v1/data/comparator_region_manifest.csv")
    ctrl = v1man[v1man.source_kind.eq("phase4_method_a_control")]
    rows = []
    for r in ctrl.itertuples(index=False):
        p = Path(str(r.vector_cache_path))
        if not p.is_absolute():
            p = (V1 / p).resolve()
        z = np.load(p, allow_pickle=False)
        s = M.mahal(M.summary_from_vectors(z["vectors"], z["species_id"].astype(np.int8),
                                           model["transform"]), model["covariance"])
        rows.append({"region_id": r.region_id, "is_clean": "clean" in str(r.region_id), "score": s})
    return pd.DataFrame(rows)


def main() -> None:
    out, rules = [], []
    for coh in ("A", "B"):
        cohort = M.load_cohort(coh)
        model = M.fit_model(cohort)
        tau_pre, cal = M.threshold(cohort, model)
        val = cohort[cohort.v3_role.eq("validation")]
        vs = np.array([M.score(r, model) for r in val.itertuples(index=False)])
        ctl = control_scores(model)
        gm = cal.groupby("condition_group_id").score.mean()

        dim = 2 * M.PCA_DIM
        out.append({
            "cohort": coh, "summary_dimension": dim, "sqrt_dim_reference": float(np.sqrt(dim)),
            "tau_predeclared": tau_pre,
            "calibration_region_scores_p50": float(np.median(cal.score)),
            "calibration_region_scores_p95": float(np.percentile(cal.score, 95)),
            "calibration_region_scores_max": float(cal.score.max()),
            "calibration_group_means_p50": float(gm.median()),
            "calibration_group_means_p95": float(np.percentile(gm, 95)),
            "calibration_group_means_max": float(gm.max()),
            "within_group_sd_median": float(cal.groupby("condition_group_id").score.std().median()),
            "between_group_sd": float(gm.std()),
            "validation_scores_p50": float(np.median(vs)),
            "validation_scores_p95": float(np.percentile(vs, 95)),
            "clean_control_scores_p50": float(ctl[ctl.is_clean].score.median()),
            "shrinkage": model["covariance"]["shrinkage"],
            "cov_condition": model["covariance"]["condition_number"],
        })

        # ---- alternative rules, COSTED not selected ----
        cands = {
            "PREDECLARED_p95_of_group_means": tau_pre,
            "p95_of_calibration_region_scores": float(np.percentile(cal.score, 95)),
            "p99_of_calibration_region_scores": float(np.percentile(cal.score, 99)),
            "max_calibration_region_score": float(cal.score.max()),
            "max_group_mean_times_1.5": float(gm.max() * 1.5),
            "chi2_like_sqrt_dim": float(np.sqrt(dim)),
        }
        for name, t in cands.items():
            rules.append({
                "cohort": coh, "rule": name, "tau": t,
                "validation_false_positive_rate": float((vs > t).mean()),
                "clean_controls_accepted": int((ctl[ctl.is_clean].score <= t).sum()),
                "clean_controls_total": int(ctl.is_clean.sum()),
                "perturbed_controls_rejected": int((ctl[~ctl.is_clean].score > t).sum()),
                "perturbed_controls_total": int((~ctl.is_clean).sum()),
            })

    o = pd.DataFrame(out); r = pd.DataFrame(rules)
    o.to_csv(V3 / "data/v3_threshold_diagnostic.csv", index=False)
    r.to_csv(V3 / "data/v3_threshold_rule_options.csv", index=False)
    pd.set_option("display.width", 250)
    print("=== score-scale diagnostic ===")
    print(o.T.to_string())
    print("\n=== threshold rules, COSTED NOT SELECTED ===")
    print(r.to_string(index=False))
    C.write_json(V3 / "data/V3_THRESHOLD_DIAGNOSTIC.json", {
        "kind": "READ_ONLY_DIAGNOSTIC", "rule_selected": False,
        "predeclared_rule_retained_for_the_comparison": True,
    })


if __name__ == "__main__":
    main()

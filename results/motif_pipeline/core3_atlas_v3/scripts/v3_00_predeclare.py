#!/usr/bin/env python3
"""v3 step 0 — PREDECLARATION.  Runs before any v3 model exists.

Fixes, and hashes, everything that must not be chosen after seeing results:
  * the two cohorts (v3-A expert-only, v3-B expert + qualified provisional)
  * the condition-group hard split, IDENTICAL for both cohorts
  * the fitting configuration (PCA16, group-balanced)
  * the five comparison metrics and their decision rules

Firewalls asserted here and re-asserted downstream:
  * the six untouched-test condition groups are excluded from every cohort
  * the six inverted-ratio trajectories are excluded and reserved as the
    candidate-novel panel
  * calibration and validation groups are FULLY expert-labelled, so the two
    cohorts differ ONLY in training content
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V1 = Path("/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/core3_comparator_benchmark_v1")
V2X = Path("/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/core3_comparator_benchmark_v2_expanded")
V3 = Path("/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/core3_atlas_v3")
sys.path.insert(0, str(V1 / "scripts"))
import phase5a_common as C  # noqa: E402

UNTOUCHED_TEST_GROUPS = {
    "debye5.4_volumefraction.010890058823", "debye5.3_volumefraction.024130533489",
    "debye5.5_volumefraction.002704544521", "debye5.5_volumefraction.005575710117",
    "debye5.5_volumefraction.008655804247", "debye5.5_volumefraction.017591158913",
}

# ---- predeclared configuration -------------------------------------------
GAT_CONF_MIN = 0.911925        # conservative option from the adjudicated calibration
INVERTED_RATIO_MAX = 2.0       # LARGE/SMALL M3-centre ratio above this = candidate novel
PCA_DIM = 16                   # user-specified
CENTER_CAP = 128               # centres per (family, group, species) PCA stratum
GROUP_BALANCE_R = 4            # region summaries contributed by EVERY training group
CALIB_FRACTION = 0.40          # of the expert-pure groups, per family
VALID_FRACTION = 0.30
SEED = C.SEED


def stable_rank(name: str) -> str:
    return hashlib.sha256(f"v3split::{SEED}::{name}".encode()).hexdigest()


def main() -> None:
    cen = pd.read_csv(V2X / "data/v3_trajectory_census_adjudicated.csv")
    probe = pd.read_csv(V2X / "data/v3_candidate_frozen_soap_probe.csv")[
        ["trajectory_id", "large_centers", "small_centers", "global_lw_rho"]]
    cen = cen.drop(columns=[c for c in ("large_centers", "small_centers", "global_lw_rho")
                            if c in cen.columns]).merge(probe, on="trajectory_id", how="left")
    cen["ls_ratio"] = cen.large_centers / cen.small_centers.clip(lower=1)

    # ---- panels held out of both cohorts ----
    sealed = cen[cen.condition_group_id.isin(UNTOUCHED_TEST_GROUPS)]
    novel = cen[(cen.ls_ratio > INVERTED_RATIO_MAX) & cen.ls_ratio.notna()
                & ~cen.condition_group_id.isin(UNTOUCHED_TEST_GROUPS)]
    if len(novel) != 6:
        raise RuntimeError(f"candidate-novel panel expected 6, got {len(novel)}")

    # ---- eligible pool ----
    pool = cen[
        ~cen.condition_group_id.isin(UNTOUCHED_TEST_GROUPS)
        & ~cen.trajectory_id.isin(novel.trajectory_id)
        & cen.category.isin(["NICOLE_GAT_AGREEMENT", "EXPERT_ADJUDICATED_USABLE",
                             "GAT_ONLY_HIGH_CONFIDENCE_CANDIDATE"])
        & cen.gat_mean_max_probability.ge(GAT_CONF_MIN)
    ].copy()
    pool["expert_supported"] = pool.category.ne("GAT_ONLY_HIGH_CONFIDENCE_CANDIDATE")
    pool["label"] = pool.adjudicated_label.fillna("").where(
        pool.adjudicated_label.notna() & pool.adjudicated_label.ne(""), pool.gat_prediction)
    pool["label_provenance"] = np.where(
        pool.category.eq("EXPERT_ADJUDICATED_USABLE"), "steven_expert_adjudication",
        np.where(pool.expert_supported, "nicole_gat_agreement", "gat_provisional"))
    if not set(pool.label) <= set(C.CLASSES):
        raise RuntimeError(f"unexpected labels: {set(pool.label) - set(C.CLASSES)}")

    # ---- condition-group hard split ----
    grp = pool.groupby("condition_group_id").agg(
        n_regions=("trajectory_id", "size"),
        n_expert=("expert_supported", "sum"),
        dominant_family=("label", lambda s: s.value_counts().idxmax()),
        n_families=("label", "nunique"))
    grp["fully_expert"] = grp.n_regions == grp.n_expert

    role = pd.Series("train", index=grp.index, name="v3_role")
    for fam, sub in grp[grp.fully_expert].groupby("dominant_family"):
        order = sorted(sub.index, key=stable_rank)
        n_cal = int(np.ceil(CALIB_FRACTION * len(order)))
        n_val = int(np.ceil(VALID_FRACTION * len(order)))
        for g in order[:n_cal]:
            role[g] = "calibration"
        for g in order[n_cal:n_cal + n_val]:
            role[g] = "validation"
    grp["v3_role"] = role

    # firewall: calibration and validation must be 100% expert
    bad = grp[grp.v3_role.isin(["calibration", "validation"]) & ~grp.fully_expert]
    if len(bad):
        raise RuntimeError(f"provisional labels reached calib/valid: {list(bad.index)}")

    pool = pool.merge(grp[["v3_role", "dominant_family"]], left_on="condition_group_id",
                      right_index=True)
    pool["in_cohort_A"] = pool.expert_supported
    pool["in_cohort_B"] = pool.expert_supported | pool.v3_role.eq("train")
    # provisional may ONLY appear in training
    if (pool.in_cohort_B & ~pool.expert_supported & pool.v3_role.ne("train")).any():
        raise RuntimeError("provisional trajectory outside training in cohort B")

    # ---- reports ----
    pd.set_option("display.width", 220)
    summary = {}
    for coh in ("A", "B"):
        s = pool[pool[f"in_cohort_{coh}"]]
        t = s.groupby("v3_role").agg(regions=("trajectory_id", "size"),
                                     groups=("condition_group_id", "nunique"))
        print(f"\n=== cohort v3-{coh} ===")
        print(t.to_string())
        print(pd.crosstab(s.v3_role, s.label).to_string())
        summary[coh] = {r: {"regions": int(t.regions[r]), "groups": int(t.groups[r])}
                        for r in t.index}
        summary[coh]["provisional_regions"] = int((~s.expert_supported).sum())

    # identical split check
    a = pool[pool.in_cohort_A]; b = pool[pool.in_cohort_B]
    for r in ("calibration", "validation"):
        ga = set(a[a.v3_role.eq(r)].trajectory_id); gb = set(b[b.v3_role.eq(r)].trajectory_id)
        if ga != gb:
            raise RuntimeError(f"cohorts disagree on {r} membership")
    print("\ncalibration and validation membership identical across cohorts: OK")

    pool.to_csv(V3 / "data/v3_cohort.csv", index=False)
    grp.reset_index().to_csv(V3 / "data/v3_group_split.csv", index=False)
    novel[["trajectory_id", "absolute_source_path", "condition_group_id", "frame",
           "adjudicated_label", "gat_prediction", "ls_ratio", "global_lw_rho"]].to_csv(
        V3 / "data/v3_candidate_novel_panel.csv", index=False)
    sealed[["trajectory_id", "condition_group_id"]].to_csv(
        V3 / "data/v3_untouched_test_SEALED.csv", index=False)

    predecl = {
        "status": "PREDECLARED_BEFORE_ANY_V3_MODEL_EXISTS",
        "cohorts": summary,
        "configuration": {
            "pca_dimension": PCA_DIM, "center_cap_per_stratum": CENTER_CAP,
            "group_balance_regions_per_training_group": GROUP_BALANCE_R,
            "covariance": "LedoitWolf on [LARGE|SMALL] region-mean summaries",
            "summary_dimension": 2 * PCA_DIM,
            "architecture": "hard condition-group split, NO group cross-fitting",
            "gat_confidence_min": GAT_CONF_MIN, "seed": SEED,
        },
        "split": {"calibration_fraction_of_expert_pure": CALIB_FRACTION,
                  "validation_fraction_of_expert_pure": VALID_FRACTION,
                  "assignment": "deterministic sha256 rank of condition_group_id, per dominant family"},
        "panels_reserved": {
            "untouched_test_groups": sorted(UNTOUCHED_TEST_GROUPS),
            "untouched_test_trajectories": int(len(sealed)),
            "candidate_novel_trajectories": sorted(novel.trajectory_id),
        },
        "metrics": {
            "M1_lofo": ("strict leave-one-family-out. For each held-out family f: fit PCA and "
                        "covariance on TRAIN groups excluding f; set tau on CALIBRATION groups "
                        "excluding f; score VALIDATION regions of trained families (negatives) "
                        "and all f regions from train+validation (positives). Report AUROC, "
                        "average precision, TPR at tau, FPR at tau."),
            "M2_known_false_positive_rate": ("full model, no family held out. Fraction of "
                                             "expert-labelled VALIDATION regions with rho > 1."),
            "M3_clean_control_retention": ("the 24 frozen Phase-4 Method-A controls scored under "
                                           "each model; report acceptance of the 8 clean controls "
                                           "and rejection of the perturbed ones."),
            "M4_threshold_stability": ("1000 bootstrap resamples of the calibration GROUPS; "
                                       "report tau 5-95%, its relative width, and the fraction of "
                                       "validation regions whose accept/reject decision flips."),
            "M5_context_shift": ("the five center-set variants from the v2 failure analysis "
                                 "(frozen M3 centres, all non-gas centres, random species-matched, "
                                 "paired full natural component, cropped-with-natural-context) "
                                 "re-scored under each model; report rho ordering and stability."),
        },
        "decision_rule": ("NO model selection in this run. The untouched test stays sealed until "
                          "the comparison report is complete and reviewed."),
    }
    blob = json.dumps(predecl, sort_keys=True).encode()
    predecl["predeclaration_sha256"] = hashlib.sha256(blob).hexdigest()
    C.write_json(V3 / "data/V3_PREDECLARATION.json", predecl)
    print(f"\npredeclaration sha256 = {predecl['predeclaration_sha256']}")


if __name__ == "__main__":
    main()

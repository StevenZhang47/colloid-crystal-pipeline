#!/usr/bin/env python3
"""Global vs class-conditional vs per-family covariance, tested like-for-like.

WHY THIS RUN EXISTS

Three ways to model "known" have been tried in this project, never against each other
under one protocol:

  GLOBAL             one mean, one covariance over ALL training regions   (v3 incumbent)
  CLASS-CONDITIONAL  one mean per family, ONE covariance shared between them
  PER-FAMILY         one mean AND one covariance per family, plus per-family thresholds

Previous comparisons were not like-for-like:
  * v2 compared per-family against a GLOBAL model whose covariance was near-degenerate,
    so global accepted almost everything (7/8 clean AND 6/16 perturbed). Per-family's
    0/16 perturbation acceptance was measured against a broken opponent.
  * the class-conditional follow-up used a TWO-family atlas (CsCl + amorphous) whose
    centres sit 10-27x apart. With that much separation the three models cannot
    disagree, and indeed they matched to 4 decimal places.

The regime where pooling should actually break is a CROWDED atlas, and the sequential
expansion walked into it: known-FPR went 0.091 -> 0.357 as families were added and
stayed there. The reason to suspect the covariance is that a global Sigma is fitted over
data whose families are far apart, so BETWEEN-family separation is absorbed into Sigma
and inflates the acceptance region along exactly the directions that separate families.
Class-conditional removes that by construction; per-family removes it and additionally
lets each family have its own shape.

WHAT IS HELD IDENTICAL

Everything except the covariance structure:
  * the same cohort-A rows and the same hard condition-group split;
  * ONE representation, `fit_transform(train)` -- species-conditioned standardisation and
    PCA16 -- fitted once per fold on the same training rows and shared by all 3 models;
  * the same group-balanced sampling (every training group contributes GROUP_BALANCE_R);
  * the same threshold RULE applied to each model's own score scale.

WHAT IS REPORTED

Both threshold rules, because they are not interchangeable and the choice is open:
  predeclared  95th percentile of per-condition-group MEAN calibration scores
  p99_region   99th percentile of INDIVIDUAL calibration region scores

and for each model x rule: known FPR on validation, per-family FPR, and strict
leave-one-family-out AUROC / TPR / FPR. No model is selected here and the untouched test
is not opened.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

V3 = Path("/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/core3_atlas_v3")
sys.path.insert(0, str(V3 / "scripts"))
import v3_common as C  # noqa: E402

FAMILIES = ["CsCl", "Th3P4", "amorphous"]


# ---------------------------------------------------------------- model fitting
def _balanced_rows(df: pd.DataFrame, tr: dict) -> np.ndarray:
    """Group-balanced summaries: every condition group contributes exactly R rows.

    Identical to v3_common.fit_covariance's sampling, so the only thing that differs
    between the three models below is what is done WITH these rows.
    """
    out = []
    for _, g in df.groupby("condition_group_id", sort=True):
        g = g.sort_values("region_id")
        s = np.vstack([C.summary(r, tr) for r in g.itertuples(index=False)])
        out.append(s[np.arange(C.GROUP_BALANCE_R) % len(s)])
    return np.vstack(out) if out else np.empty((0, 32))


def fit_global(train, tr):
    m = LedoitWolf().fit(_balanced_rows(train, tr))
    return {"kind": "global", "lw": m, "shrinkage": float(m.shrinkage_)}


def fit_class_conditional(train, tr):
    """One mean per family; ONE covariance fitted on within-family residuals.

    Pooling the residuals rather than the raw rows is the whole point: Sigma then
    describes how a domain varies AROUND its own family, not how families differ from
    each other. It also keeps the parameter count affordable -- a 32x32 covariance has
    528 free parameters and no single family here has enough units to support one.
    """
    means, residuals = {}, []
    for fam in sorted(train.label.unique()):
        X = _balanced_rows(train[train.label.eq(fam)], tr)
        if len(X) == 0:
            continue
        means[fam] = X.mean(axis=0)
        residuals.append(X - means[fam])
    m = LedoitWolf(assume_centered=True).fit(np.vstack(residuals))
    return {"kind": "class_conditional", "means": means, "lw": m,
            "shrinkage": float(m.shrinkage_)}


def fit_per_family(train, tr):
    """A separate mean AND covariance per family."""
    models = {}
    for fam in sorted(train.label.unique()):
        X = _balanced_rows(train[train.label.eq(fam)], tr)
        if len(X) < 4:
            continue
        m = LedoitWolf().fit(X)
        models[fam] = {"lw": m, "n_rows": int(len(X)), "shrinkage": float(m.shrinkage_)}
    return {"kind": "per_family", "families": models}


FITTERS = {"global": fit_global, "class_conditional": fit_class_conditional,
           "per_family": fit_per_family}


# ---------------------------------------------------------------- scoring
def _m(vec, lw, mean=None):
    d = np.asarray(vec, float) - (lw.location_ if mean is None else mean)
    return float(np.sqrt(max(float(d @ lw.precision_ @ d), 0.0)))


def score_parts(vec, model) -> dict:
    """Per-family scores where the model has them; {'_all': s} for the global model."""
    if model["kind"] == "global":
        return {"_all": _m(vec, model["lw"])}
    if model["kind"] == "class_conditional":
        return {f: _m(vec, model["lw"], mean=mu) for f, mu in model["means"].items()}
    return {f: _m(vec, sub["lw"]) for f, sub in model["families"].items()}


# ---------------------------------------------------------------- thresholds
def thresholds(model, calib_scores: pd.DataFrame, rule: str) -> dict:
    """One tau per score channel, using the SAME rule for every model.

    calib_scores has columns: region_id, condition_group_id, channel, score.
    The global model has a single channel; the other two have one per family, and a
    per-family threshold is what lets rho be comparable across channels.
    """
    tau = {}
    for ch, g in calib_scores.groupby("channel"):
        # A per-family channel must be calibrated on calibration domains OF THAT FAMILY.
        # Calibrating tau_CsCl on every calibration domain includes amorphous domains
        # scored against the CsCl mean -- distances that are enormous by construction --
        # which inflates tau_CsCl until the CsCl channel accepts anything. Because rho is
        # a MIN over channels, one inflated channel makes the whole detector permissive:
        # the first run of this script returned FPR 0.00 and TPR 0.00 together, i.e. it
        # rejected nothing at all while still ranking perfectly (AUROC 1.0).
        if ch != "_all":
            g = g[g.label.eq(ch)]
            if len(g) < 2:
                tau[ch] = float("inf")
                continue
        if rule == "predeclared":
            v = g.groupby("condition_group_id").score.mean().to_numpy()
            tau[ch] = float(np.quantile(v, 0.95, method="higher"))
        elif rule == "p99_region":
            tau[ch] = float(np.quantile(g.score.to_numpy(), 0.99, method="higher"))
        else:
            raise ValueError(rule)
    return tau


def rho_of(vec, model, tau) -> float:
    """rho = min over channels of score/tau. Below 1 = accepted as known."""
    parts = score_parts(vec, model)
    return min(parts[ch] / tau[ch] for ch in parts if ch in tau and tau[ch] > 0)


# ---------------------------------------------------------------- one fold
def run_fold(cohort, exclude_family, rows):
    train = cohort[cohort.v3_role.eq("train")]
    calib = cohort[cohort.v3_role.eq("calibration")]
    valid = cohort[cohort.v3_role.eq("validation")]
    if exclude_family:
        train = train[train.label.ne(exclude_family)]
        calib = calib[calib.label.ne(exclude_family)]

    tr = C.fit_transform(train)                      # ONE representation for all models
    cache = {r.region_id: C.summary(r, tr) for r in cohort.itertuples(index=False)}

    for name, fitter in FITTERS.items():
        model = fitter(train, tr)
        cal = pd.DataFrame([
            {"region_id": r.region_id, "condition_group_id": r.condition_group_id,
             "label": r.label, "channel": ch, "score": s}
            for r in calib.itertuples(index=False)
            for ch, s in score_parts(cache[r.region_id], model).items()])

        for rule in ("predeclared", "p99_region"):
            tau = thresholds(model, cal, rule)
            if exclude_family is None:
                neg = valid
                fp = sum(rho_of(cache[r.region_id], model, tau) > 1
                         for r in neg.itertuples(index=False))
                per_fam = {}
                for fam, g in neg.groupby("label"):
                    k = sum(rho_of(cache[r.region_id], model, tau) > 1
                            for r in g.itertuples(index=False))
                    per_fam[fam] = f"{k}/{len(g)}"
                rows.append({"fold": "full", "model": name, "rule": rule,
                             "n_validation": len(neg), "known_fpr": fp / len(neg),
                             "per_family_fpr": json.dumps(per_fam),
                             "shrinkage": model.get("shrinkage"),
                             "n_channels": len(tau)})
            else:
                pos = cohort[cohort.label.eq(exclude_family)
                             & cohort.v3_role.isin(["train", "validation"])]
                neg = valid[valid.label.ne(exclude_family)]
                rp = [rho_of(cache[r.region_id], model, tau) for r in pos.itertuples(index=False)]
                rn = [rho_of(cache[r.region_id], model, tau) for r in neg.itertuples(index=False)]
                y = np.r_[np.ones(len(rp)), np.zeros(len(rn))]
                rows.append({"fold": exclude_family, "model": name, "rule": rule,
                             "n_positive": len(rp), "n_negative": len(rn),
                             "auroc": float(roc_auc_score(y, np.r_[rp, rn])),
                             "tpr_at_tau": float(np.mean(np.array(rp) > 1)),
                             "fpr_at_tau": float(np.mean(np.array(rn) > 1)),
                             "median_rho_positive": float(np.median(rp)),
                             "median_rho_negative": float(np.median(rn)),
                             "shrinkage": model.get("shrinkage"),
                             "n_channels": len(tau)})
            print(f"  {str(exclude_family or 'full'):10s} {name:18s} {rule:12s} "
                  f"-> {rows[-1]}", flush=True)


def main():
    t0 = time.perf_counter()
    cohort = C.load_cohort("A")
    rows = []
    print("=== full model (known false-positive rate) ===", flush=True)
    run_fold(cohort, None, rows)
    print("\n=== strict leave-one-family-out ===", flush=True)
    for fam in FAMILIES:
        run_fold(cohort, fam, rows)
    df = pd.DataFrame(rows)
    out = V3 / "data/v3_covariance_structure_comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({time.perf_counter()-t0:.0f}s)")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

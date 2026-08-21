#!/usr/bin/env python3
"""Candidate threshold rules for the FROZEN global model, with stability.

Context. The covariance-structure run settled the model: a single global Ledoit-Wolf
Gaussian beats class-conditional and per-family on this atlas, because splitting the data
three ways costs more in estimation error (shrinkage 0.045 vs 0.006) than the
between-family inflation costs. What is NOT settled is the threshold rule, and two
defects are known:

  1. the predeclared rule takes a quantile of per-condition-group MEAN scores and then
     applies it to INDIVIDUAL region scores. Group means are ~19% narrower (sigma_b=0.762,
     sigma_w=0.716, n~3.68), so tau lands too low and known FPR is 0.24.

  2. `np.quantile(v, 0.99, method="higher")` on 70 values returns index ceil(0.99*69)=69,
     i.e. THE MAXIMUM. The "p99" that scored 0.02 in the previous run was really "the
     single largest calibration score" -- a one-point statistic, and a bad thing to freeze.

So this sweeps rules with EXPLICIT quantile semantics and measures how much each one moves
when the calibration groups are resampled. A rule that is only good because one domain
happened to score high is not a rule.

Rules
  predeclared        p95 of per-group means, method="higher"  (the hashed rule, for reference)
  p90/p95/p975/p99   interpolated quantiles of INDIVIDUAL calibration region scores
  max_region         the largest individual calibration score
  size_conditioned   log(score) regressed on log(n_centres) over calibration; tau becomes a
                     FUNCTION of domain size rather than one number. Motivated by the
                     measured corr(log n_centres, log score) = -0.591: small domains score
                     systematically higher, so a flat fence penalises them.

Reported per rule: known FPR on validation, per-family FPR, strict LOFO TPR/FPR, and
1000-bootstrap resampling of the CALIBRATION GROUPS giving tau's 5-95% spread, its
relative width, and the fraction of validation decisions that flip. Nothing is selected
here and the untouched test stays sealed.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

V3 = Path("/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/core3_atlas_v3")
sys.path.insert(0, str(V3 / "scripts"))
import v3_common as C  # noqa: E402

FAMILIES = ["CsCl", "Th3P4", "amorphous"]
RNG = np.random.default_rng(20260803)
N_BOOT = 1000


def size_fit(scores, sizes):
    """log(score) = a + b*log(n). Returns (a, b, residual sd)."""
    x, y = np.log(np.asarray(sizes, float)), np.log(np.asarray(scores, float))
    b, a = np.polyfit(x, y, 1)
    return a, b, float(np.std(y - (a + b * x), ddof=2))


def tau_of(rule, scores, groups, sizes):
    """Return either a float tau, or a callable tau(n_centres) for the size rule."""
    s = np.asarray(scores, float)
    if rule == "predeclared":
        gm = pd.Series(s).groupby(pd.Series(groups).values).mean().to_numpy()
        return float(np.quantile(gm, 0.95, method="higher"))
    if rule == "max_region":
        return float(s.max())
    if rule.startswith("p"):
        q = {"p90": 0.90, "p95": 0.95, "p975": 0.975, "p99": 0.99}[rule]
        return float(np.quantile(s, q, method="linear"))   # EXPLICIT: interpolated
    if rule == "size_conditioned":
        a, b, sd = size_fit(s, sizes)
        z = 2.326                                          # one-sided 99%
        return lambda n: float(np.exp(a + b * np.log(n) + z * sd))
    raise ValueError(rule)


def rho(score, size, tau):
    t = tau(size) if callable(tau) else tau
    return score / t if t > 0 else np.inf


def fold_scores(cohort, exclude_family):
    train = cohort[cohort.v3_role.eq("train")]
    calib = cohort[cohort.v3_role.eq("calibration")]
    if exclude_family:
        train = train[train.label.ne(exclude_family)]
        calib = calib[calib.label.ne(exclude_family)]
    model = C.fit_model(cohort, exclude_family)
    sc = {r.region_id: C.score(r, model) for r in cohort.itertuples(index=False)}
    return model, calib, sc


def main():
    t0 = time.perf_counter()
    cohort = C.load_cohort("A")
    sizes = dict(zip(cohort.region_id, cohort.center_count))
    valid = cohort[cohort.v3_role.eq("validation")]
    rules = ["predeclared", "p90", "p95", "p975", "p99", "max_region", "size_conditioned"]
    rows = []

    # ---------------- full model: known FPR + bootstrap stability ----------------
    model, calib, sc = fold_scores(cohort, None)
    cal_s = np.array([sc[r] for r in calib.region_id])
    cal_g = calib.condition_group_id.to_numpy()
    cal_n = np.array([sizes[r] for r in calib.region_id])
    groups = np.unique(cal_g)

    for rule in rules:
        tau = tau_of(rule, cal_s, cal_g, cal_n)
        dec = {r.region_id: rho(sc[r.region_id], sizes[r.region_id], tau) > 1
               for r in valid.itertuples(index=False)}
        fp = sum(dec.values())
        per_fam = {f: f"{sum(dec[r.region_id] for r in g.itertuples(index=False))}/{len(g)}"
                   for f, g in valid.groupby("label")}

        # bootstrap the CALIBRATION GROUPS, as the predeclaration specifies for M4
        taus, flips = [], []
        for _ in range(N_BOOT):
            pick = RNG.choice(groups, size=len(groups), replace=True)
            idx = np.concatenate([np.flatnonzero(cal_g == q) for q in pick])
            tb = tau_of(rule, cal_s[idx], cal_g[idx], cal_n[idx])
            taus.append(tb(np.median(cal_n)) if callable(tb) else tb)
            flips.append(np.mean([(rho(sc[r.region_id], sizes[r.region_id], tb) > 1) != dec[r.region_id]
                                  for r in valid.itertuples(index=False)]))
        taus = np.array(taus)
        t_ref = tau(np.median(cal_n)) if callable(tau) else tau
        rows.append({"fold": "full", "rule": rule, "tau": round(float(t_ref), 4),
                     "known_fpr": fp / len(valid), "per_family_fpr": json.dumps(per_fam),
                     "tau_p5": round(float(np.percentile(taus, 5)), 4),
                     "tau_p95": round(float(np.percentile(taus, 95)), 4),
                     "tau_relative_width": round(float((np.percentile(taus, 95) -
                                                        np.percentile(taus, 5)) / max(t_ref, 1e-9)), 4),
                     "decision_flip_fraction": round(float(np.mean(flips)), 4)})
        print(f"  full {rule:18s} tau={t_ref:7.3f} FPR={fp/len(valid):.3f} "
              f"width={rows[-1]['tau_relative_width']:.3f} flips={rows[-1]['decision_flip_fraction']:.4f}",
              flush=True)

    # ---------------- strict LOFO per rule ----------------
    for fam in FAMILIES:
        model, calib, sc = fold_scores(cohort, fam)
        cal_s = np.array([sc[r] for r in calib.region_id])
        cal_g = calib.condition_group_id.to_numpy()
        cal_n = np.array([sizes[r] for r in calib.region_id])
        pos = cohort[cohort.label.eq(fam) & cohort.v3_role.isin(["train", "validation"])]
        neg = valid[valid.label.ne(fam)]
        for rule in rules:
            tau = tau_of(rule, cal_s, cal_g, cal_n)
            rp = np.array([rho(sc[r.region_id], sizes[r.region_id], tau) for r in pos.itertuples(index=False)])
            rn = np.array([rho(sc[r.region_id], sizes[r.region_id], tau) for r in neg.itertuples(index=False)])
            y = np.r_[np.ones(len(rp)), np.zeros(len(rn))]
            rows.append({"fold": fam, "rule": rule, "n_positive": len(rp), "n_negative": len(rn),
                         "auroc": round(float(roc_auc_score(y, np.r_[rp, rn])), 4),
                         "tpr_at_tau": round(float(np.mean(rp > 1)), 4),
                         "fpr_at_tau": round(float(np.mean(rn > 1)), 4)})
            print(f"  {fam:10s} {rule:18s} TPR={rows[-1]['tpr_at_tau']:.3f} "
                  f"FPR={rows[-1]['fpr_at_tau']:.3f} AUROC={rows[-1]['auroc']}", flush=True)

    df = pd.DataFrame(rows)
    out = V3 / "data/v3_threshold_versions.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""v3 step 3 — the predeclared comparison of v3-A and v3-B.

Five metrics, all fixed in V3_PREDECLARATION.json before either model existed:

  M1 LOFO                    strict leave-one-family-out
  M2 known false-positive    full model, expert validation regions
  M3 clean-control retention 24 frozen Phase-4 Method-A controls
  M4 threshold stability     1000 bootstraps of the calibration GROUPS
  M5 context shift           centre-set variants

Because calibration and validation membership are identical across cohorts, every
evaluation set here is byte-identical between A and B.  Only the model differs.

NO model is selected.  The untouched test is not opened.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

V1 = Path("/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/core3_comparator_benchmark_v1")
V3 = Path("/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/core3_atlas_v3")
sys.path.insert(0, str(V3 / "scripts"))
import v3_common as M  # noqa: E402
from v3_common import C  # noqa: E402

N_BOOT = 1000
COHORTS = ("A", "B")


# ---------------------------------------------------------------- M1
def m1_lofo(coh: str, cohort: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    out, per_region = [], []
    for fam in M.CLASSES:
        model = M.fit_model(cohort, exclude_family=fam)
        tau, cal_scores = M.threshold(cohort, model, exclude_family=fam)
        # positives: held-out-family regions from calibration + validation (all expert)
        pos = cohort[cohort.label.eq(fam) & cohort.v3_role.isin(["calibration", "validation"])]
        # negatives: other-family validation regions
        neg = cohort[cohort.label.ne(fam) & cohort.v3_role.eq("validation")]
        if not pos.expert_supported.all() or not neg.expert_supported.all():
            raise RuntimeError("non-expert region in a LOFO evaluation set")
        sp = np.array([M.score(r, model) for r in pos.itertuples(index=False)])
        sn = np.array([M.score(r, model) for r in neg.itertuples(index=False)])
        y = np.r_[np.ones(len(sp)), np.zeros(len(sn))]
        s = np.r_[sp, sn]
        out.append({
            "cohort": coh, "holdout_family": fam, "threshold": tau,
            "n_train_regions": model["n_train_regions"], "n_train_groups": model["n_train_groups"],
            "n_positive": len(sp), "n_negative": len(sn),
            "auroc": float(roc_auc_score(y, s)), "average_precision": float(average_precision_score(y, s)),
            "tpr_at_tau": float((sp > tau).mean()), "fpr_at_tau": float((sn > tau).mean()),
            "median_rho_positive": float(np.median(sp / tau)),
            "median_rho_negative": float(np.median(sn / tau)),
            "shrinkage": model["covariance"]["shrinkage"],
            "cov_condition": model["covariance"]["condition_number"],
            "pca_explained_variance": model["transform"]["explained_variance"],
        })
        for r, v in zip(pos.itertuples(index=False), sp):
            per_region.append({"cohort": coh, "holdout_family": fam, "region_id": r.region_id,
                               "role": "positive_heldout", "label": r.label,
                               "score": v, "rho": v / tau})
        for r, v in zip(neg.itertuples(index=False), sn):
            per_region.append({"cohort": coh, "holdout_family": fam, "region_id": r.region_id,
                               "role": "negative_indist", "label": r.label,
                               "score": v, "rho": v / tau})
        print(f"  [{coh}] holdout {fam:9s}  AUROC {out[-1]['auroc']:.3f}  "
              f"AP {out[-1]['average_precision']:.3f}  TPR {out[-1]['tpr_at_tau']:.3f}  "
              f"FPR {out[-1]['fpr_at_tau']:.3f}", flush=True)
    return out, per_region


# ---------------------------------------------------------------- M2/M4
def m2_m4(coh: str, cohort: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    model = M.fit_model(cohort)
    tau, cal = M.threshold(cohort, model)
    val = cohort[cohort.v3_role.eq("validation")]
    vs = pd.DataFrame([{"region_id": r.region_id, "condition_group_id": r.condition_group_id,
                        "label": r.label, "score": M.score(r, model)}
                       for r in val.itertuples(index=False)])
    vs["rho"] = vs.score / tau
    vs["rejected"] = vs.rho > 1.0
    vs["cohort"] = coh

    m2 = {"cohort": coh, "threshold": tau, "n_validation": len(vs),
          "n_false_positive": int(vs.rejected.sum()),
          "known_false_positive_rate": float(vs.rejected.mean()),
          "median_rho": float(vs.rho.median()), "max_rho": float(vs.rho.max()),
          "n_train_regions": model["n_train_regions"], "n_train_groups": model["n_train_groups"],
          "shrinkage": model["covariance"]["shrinkage"],
          "cov_condition": model["covariance"]["condition_number"],
          "pca_explained_variance": model["transform"]["explained_variance"]}
    for fam in M.CLASSES:
        q = vs[vs.label.eq(fam)]
        m2[f"fpr_{fam}"] = float(q.rejected.mean()) if len(q) else None

    # M4 — bootstrap the calibration GROUPS
    gm = cal.groupby("condition_group_id").score.mean()
    groups = gm.index.to_numpy()
    rng = np.random.default_rng(M.SEED)
    taus = np.array([M.higher95(gm.loc[rng.choice(groups, len(groups), replace=True)].to_numpy())
                     for _ in range(N_BOOT)])
    flips = np.array([float(((vs.score.to_numpy()[:, None] > t).mean(0)
                             != vs.rejected.to_numpy().mean()).mean()) for t in [taus]])
    dec = (vs.score.to_numpy()[None, :] > taus[:, None])
    flip_frac = float((dec != vs.rejected.to_numpy()[None, :]).mean())
    m4 = {"cohort": coh, "tau_frozen": tau, "tau_p5": float(np.percentile(taus, 5)),
          "tau_p50": float(np.percentile(taus, 50)), "tau_p95": float(np.percentile(taus, 95)),
          "tau_relative_width": float((np.percentile(taus, 95) - np.percentile(taus, 5)) / tau),
          "n_calibration_groups": int(len(groups)),
          "validation_decision_flip_fraction": flip_frac,
          "n_bootstrap": N_BOOT}
    print(f"  [{coh}] M2 FPR {m2['known_false_positive_rate']:.4f}  "
          f"tau {tau:.4f}  M4 width {m4['tau_relative_width']:.3f}  "
          f"flips {flip_frac:.4f}", flush=True)
    return m2, m4, vs


# ---------------------------------------------------------------- M3
def m3_controls(coh: str, cohort: pd.DataFrame) -> pd.DataFrame:
    model = M.fit_model(cohort)
    tau, _ = M.threshold(cohort, model)
    v1man = pd.read_csv(V1 / "clean_natural_only_v1/data/comparator_region_manifest.csv")
    ctrl = v1man[v1man.source_kind.eq("phase4_method_a_control")].copy()
    if len(ctrl) != 24:
        raise RuntimeError(f"expected 24 Phase-4 controls, got {len(ctrl)}")
    rows = []
    for r in ctrl.itertuples(index=False):
        p = Path(str(r.vector_cache_path))
        if not p.is_absolute():
            p = (V1 / p).resolve()
        if C.sha256_file(p) != str(r.vector_cache_sha256):
            raise RuntimeError(f"control cache hash mismatch: {r.region_id}")
        z = np.load(p, allow_pickle=False)
        s = M.mahal(M.summary_from_vectors(z["vectors"], z["species_id"].astype(np.int8),
                                           model["transform"]), model["covariance"])
        sub = str(r.region_id).split("::")[-1].split("::m3r")[0]
        rows.append({"cohort": coh, "region_id": r.region_id, "true_family": r.true_family,
                     "is_clean": "clean" in str(r.region_id), "score": s, "rho": s / tau,
                     "accepted": s <= tau, "center_count": int(r.center_count)})
    df = pd.DataFrame(rows)
    cl = df[df.is_clean]
    print(f"  [{coh}] M3 clean accepted {int(cl.accepted.sum())}/{len(cl)}  "
          f"perturbed rejected {int((~df[~df.is_clean].accepted).sum())}/{len(df)-len(cl)}", flush=True)
    return df


# ---------------------------------------------------------------- M5
def m5_context(coh: str, cohort: pd.DataFrame) -> pd.DataFrame:
    """Centre-set / context shift on every validation region, identical sets per cohort."""
    model = M.fit_model(cohort)
    tau, _ = M.threshold(cohort, model)
    rng = np.random.default_rng(M.SEED)
    val = cohort[cohort.v3_role.eq("validation")]
    rows = []
    for r in val.itertuples(index=False):
        meta = pd.read_parquet(V3 / "data/metadata_shards" / f"{r.trajectory_id}.parquet")
        shard = np.load(V3 / "data/soap_shards" / f"{r.trajectory_id}.npy", mmap_mode="r")
        comp = meta.physical_component_id.to_numpy(int) == r.physical_component_id
        m3 = meta.m3_retained_center.to_numpy(bool)
        sp_all = meta.canonical_species_id.to_numpy(np.int8)

        def sc(mask):
            idx = np.flatnonzero(mask)
            return M.mahal(M.summary_from_vectors(np.asarray(shard[idx]), sp_all[idx],
                                                  model["transform"]), model["covariance"])

        base = sc(comp & m3)
        variants = {"frozen_M3_centers": base,
                    "all_non_gas_centers_in_component": sc(comp),
                    "all_non_gas_centers_whole_frame": sc(np.ones(len(meta), bool))}
        # random species-matched subset of the same size, median of 25 draws
        want = {0: r.large_centers, 1: r.small_centers}
        draws = []
        for _ in range(25):
            pick = []
            for sid, n in want.items():
                pool = np.flatnonzero(comp & (sp_all == sid))
                if len(pool):
                    pick.append(rng.choice(pool, min(n, len(pool)), replace=False))
            idx = np.concatenate(pick)
            draws.append(M.mahal(M.summary_from_vectors(np.asarray(shard[idx]), sp_all[idx],
                                                        model["transform"]), model["covariance"]))
        variants["random_species_matched_median_of_25"] = float(np.median(draws))
        for name, s in variants.items():
            rows.append({"cohort": coh, "region_id": r.region_id, "label": r.label,
                         "variant": name, "score": s, "rho": s / tau, "accepted": s <= tau,
                         "rho_ratio_to_frozen": s / base})
    df = pd.DataFrame(rows)
    g = df.groupby("variant").agg(median_rho=("rho", "median"), accept_rate=("accepted", "mean"))
    print(f"  [{coh}] M5 accept rate by variant: "
          + "  ".join(f"{k.split('_')[0]}:{v:.2f}" for k, v in g.accept_rate.items()), flush=True)
    return df


def main() -> None:
    t0 = time.perf_counter()
    # Per-cohort checkpoints: a kill mid-run must not discard the finished cohort.
    for coh in COHORTS:
        ck = V3 / f"data/_ckpt_{coh}"
        if (ck / "done.json").exists():
            print(f"=== cohort v3-{coh}: cached ===", flush=True)
            continue
        ck.mkdir(parents=True, exist_ok=True)
        cohort = M.load_cohort(coh)
        print(f"\n=== cohort v3-{coh}: {len(cohort)} regions / "
              f"{cohort.condition_group_id.nunique()} groups ===", flush=True)
        print("M1 strict LOFO", flush=True)
        a, b = m1_lofo(coh, cohort)
        pd.DataFrame(a).to_csv(ck / "m1.csv", index=False)
        pd.DataFrame(b).to_csv(ck / "m1_per_region.csv", index=False)
        print("M2 + M4", flush=True)
        m2, m4, vs = m2_m4(coh, cohort)
        pd.DataFrame([m2]).to_csv(ck / "m2.csv", index=False)
        pd.DataFrame([m4]).to_csv(ck / "m4.csv", index=False)
        vs.to_csv(ck / "validation.csv", index=False)
        print("M3 Phase-4 controls", flush=True)
        m3_controls(coh, cohort).to_csv(ck / "m3.csv", index=False)
        print("M5 context shift", flush=True)
        m5_context(coh, cohort).to_csv(ck / "m5.csv", index=False)
        (ck / "done.json").write_text(json.dumps({"cohort": coh}))

    def gather(name):
        return pd.concat([pd.read_csv(V3 / f"data/_ckpt_{c}/{name}.csv") for c in COHORTS],
                         ignore_index=True)

    gather("m1").to_csv(V3 / "data/v3_m1_lofo.csv", index=False)
    gather("m1_per_region").to_csv(V3 / "data/v3_m1_lofo_per_region.csv", index=False)
    gather("m2").to_csv(V3 / "data/v3_m2_false_positive.csv", index=False)
    gather("m4").to_csv(V3 / "data/v3_m4_threshold_stability.csv", index=False)
    gather("validation").to_csv(V3 / "data/v3_validation_scores.csv", index=False)
    gather("m3").to_csv(V3 / "data/v3_m3_clean_controls.csv", index=False)
    gather("m5").to_csv(V3 / "data/v3_m5_context_shift.csv", index=False)

    C.write_json(V3 / "data/V3_COMPARISON_COMPLETE.json", {
        "status": "complete", "cohorts": list(COHORTS),
        "predeclaration_sha256": M.PRE["predeclaration_sha256"],
        "model_selected": False, "untouched_test_opened": False,
        "runtime_seconds": time.perf_counter() - t0,
    })
    print(f"\ndone in {time.perf_counter()-t0:.0f}s — no model selected, untouched test sealed")


if __name__ == "__main__":
    main()

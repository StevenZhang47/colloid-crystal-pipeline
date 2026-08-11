#!/usr/bin/env python3
"""Shared v3 fitting and scoring.  One transform + one covariance per model —
no group cross-fitting, because the hard condition-group split makes it unnecessary.

Model = (species moments, PCA16, Ledoit-Wolf on 32-D region summaries).
Score = Mahalanobis distance of the region summary.  rho = score / tau.

Group balance is applied at BOTH fitting stages:
  * PCA sample: every (family, condition_group, species) stratum contributes
    exactly CENTER_CAP centres, quota-split evenly across the regions in it.
  * covariance: every training condition group contributes exactly
    GROUP_BALANCE_R region summaries, by deterministic cyclic tiling.  A group
    with 8 seeds therefore carries no more weight than a group with 1.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

V1 = Path("/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/core3_comparator_benchmark_v1")
V3 = Path("/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/core3_atlas_v3")
sys.path.insert(0, str(V1 / "scripts"))
import phase5a_common as C  # noqa: E402

PRE = json.loads((V3 / "data/V3_PREDECLARATION.json").read_text())
PCA_DIM = PRE["configuration"]["pca_dimension"]
CENTER_CAP = PRE["configuration"]["center_cap_per_stratum"]
GROUP_BALANCE_R = PRE["configuration"]["group_balance_regions_per_training_group"]
SEED = PRE["configuration"]["seed"]
CLASSES = tuple(C.CLASSES)


# ---------------------------------------------------------------- loading
@lru_cache(maxsize=96)   # ~430x952 float64 per region = 3.3 MB; keep the cache under ~0.4 GB
def region_vectors(trajectory_id: str, component_id: int) -> tuple[np.ndarray, np.ndarray]:
    """L2-normalised SOAP rows for the M3 centres of one physical component."""
    meta = pd.read_parquet(V3 / "data/metadata_shards" / f"{trajectory_id}.parquet")
    q = meta.m3_retained_center.to_numpy(bool) & (
        meta.physical_component_id.to_numpy(int) == int(component_id))
    sel = meta.loc[q].sort_values("soap_vector_index")
    rows = sel.soap_vector_index.to_numpy(int)
    x = np.asarray(np.load(V3 / "data/soap_shards" / f"{trajectory_id}.npy", mmap_mode="r")[rows],
                   dtype=np.float64)
    if not np.isfinite(x).all():
        raise RuntimeError(f"non-finite vectors for {trajectory_id}/pc{component_id}")
    return C.l2_rows(x), sel.canonical_species_id.to_numpy(np.int8)


def vectors_for(record) -> tuple[np.ndarray, np.ndarray]:
    return region_vectors(str(record.trajectory_id), int(record.physical_component_id))


# ---------------------------------------------------------------- fitting
def stratum_sample(train: pd.DataFrame, family: str, group: str, species_id: int) -> np.ndarray:
    """CENTER_CAP centres for one stratum, quota-split evenly across its regions."""
    sub = train[train.label.eq(family) & train.condition_group_id.eq(group)].sort_values("region_id")
    recs = list(sub.itertuples(index=False))
    base, rem = divmod(CENTER_CAP, len(recs))
    pieces = []
    for i, rec in enumerate(recs):
        x, sp = vectors_for(rec)
        q = x[sp == species_id]
        quota = base + int(i < rem)
        if len(q) == 0:
            continue
        take = C.stable_indices(len(q), min(quota, len(q)), "v3transform", family, group,
                                str(species_id), rec.region_id)
        pieces.append(q[take])
    if not pieces:
        return np.empty((0, C.SOAP_DIMENSION))
    s = np.vstack(pieces)
    if len(s) > CENTER_CAP:
        s = s[C.stable_indices(len(s), CENTER_CAP, "v3transform-final", family, group, str(species_id))]
    return s


def fit_transform(train: pd.DataFrame) -> dict:
    by_species = {0: [], 1: []}
    strata = train[["label", "condition_group_id"]].drop_duplicates().sort_values(
        ["label", "condition_group_id"])
    for fam, grp in strata.itertuples(index=False):
        for sid in (0, 1):
            s = stratum_sample(train, fam, grp, sid)
            if len(s):
                by_species[sid].append(s)
    mean = np.vstack([np.vstack(by_species[s]).mean(0) for s in (0, 1)])
    scale = np.maximum(np.vstack([np.vstack(by_species[s]).std(0, ddof=0) for s in (0, 1)]), 1e-10)
    cond = []
    for sid in (0, 1):
        cond.extend((x - mean[sid]) / scale[sid] for x in by_species[sid])
    sample = np.vstack(cond)
    pca = PCA(n_components=PCA_DIM, svd_solver="randomized", random_state=SEED, iterated_power=5)
    pca.fit(sample)
    return {"species_mean": mean, "species_scale": scale, "pca": pca,
            "n_strata": len(strata), "pca_sample_rows": int(len(sample)),
            "explained_variance": float(pca.explained_variance_ratio_.sum())}


def summary(record, tr: dict) -> np.ndarray:
    """[LARGE mean | SMALL mean] in PCA space — 2 * PCA_DIM."""
    x, sp = vectors_for(record)
    z = np.empty_like(x)
    for sid in (0, 1):
        q = sp == sid
        z[q] = (x[q] - tr["species_mean"][sid]) / tr["species_scale"][sid]
    p = tr["pca"].transform(z)
    return np.concatenate([p[sp == 0].mean(0), p[sp == 1].mean(0)])


def summary_from_vectors(x: np.ndarray, sp: np.ndarray, tr: dict) -> np.ndarray:
    x = C.l2_rows(np.asarray(x, dtype=np.float64))
    z = np.empty_like(x)
    for sid in (0, 1):
        q = sp == sid
        if q.any():
            z[q] = (x[q] - tr["species_mean"][sid]) / tr["species_scale"][sid]
    p = tr["pca"].transform(z)
    return np.concatenate([p[sp == 0].mean(0), p[sp == 1].mean(0)])


def fit_covariance(train: pd.DataFrame, tr: dict) -> dict:
    """Ledoit-Wolf, with every training condition group contributing equally."""
    rows, provenance = [], []
    for grp, g in train.groupby("condition_group_id", sort=True):
        g = g.sort_values("region_id")
        s = np.vstack([summary(r, tr) for r in g.itertuples(index=False)])
        idx = np.arange(GROUP_BALANCE_R) % len(s)      # cyclic tiling = exact equal weight
        rows.append(s[idx])
        provenance.append({"condition_group_id": grp, "n_regions": len(s),
                           "contributed": GROUP_BALANCE_R})
    X = np.vstack(rows)
    m = LedoitWolf().fit(X)
    return {"model": m, "n_rows": int(len(X)), "n_groups": len(provenance),
            "shrinkage": float(m.shrinkage_),
            "condition_number": float(np.linalg.cond(m.covariance_)),
            "provenance": provenance}


def mahal(vec: np.ndarray, cov: dict) -> float:
    d = np.asarray(vec, float) - cov["model"].location_
    return float(np.sqrt(max(float(d @ cov["model"].precision_ @ d), 0.0)))


def fit_model(cohort: pd.DataFrame, exclude_family: str | None = None) -> dict:
    train = cohort[cohort.v3_role.eq("train")]
    if exclude_family:
        train = train[train.label.ne(exclude_family)]
    tr = fit_transform(train)
    cov = fit_covariance(train, tr)
    return {"transform": tr, "covariance": cov, "exclude_family": exclude_family,
            "n_train_regions": int(len(train)),
            "n_train_groups": int(train.condition_group_id.nunique())}


def score(record, model: dict) -> float:
    return mahal(summary(record, model["transform"]), model["covariance"])


def higher95(v) -> float:
    return float(np.quantile(np.asarray(v, float), 0.95, method="higher"))


def threshold(cohort: pd.DataFrame, model: dict, exclude_family: str | None = None) -> tuple[float, pd.DataFrame]:
    """tau = 95th percentile ('higher') of per-condition-group mean scores on CALIBRATION."""
    cal = cohort[cohort.v3_role.eq("calibration")]
    if exclude_family:
        cal = cal[cal.label.ne(exclude_family)]
    if not cal.expert_supported.all():
        raise RuntimeError("non-expert region reached threshold calibration")
    rows = [{"region_id": r.region_id, "condition_group_id": r.condition_group_id,
             "label": r.label, "score": score(r, model)} for r in cal.itertuples(index=False)]
    df = pd.DataFrame(rows)
    gm = df.groupby("condition_group_id").score.mean()
    return higher95(gm.to_numpy()), df


def load_cohort(which: str) -> pd.DataFrame:
    c = pd.read_csv(V3 / "data/v3_region_manifest.csv")
    c = c[c[f"in_cohort_{which}"]].copy()
    if which == "A" and not c.expert_supported.all():
        raise RuntimeError("cohort A contains a non-expert region")
    bad = c[~c.expert_supported & c.v3_role.ne("train")]
    if len(bad):
        raise RuntimeError(f"provisional region outside training: {list(bad.region_id)}")
    return c

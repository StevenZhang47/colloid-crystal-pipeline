#!/usr/bin/env python3
"""The acceptance test for the matcher. Pass this and the matcher is finished.

THE STANDARD (set by the project, not by me)
--------------------------------------------
Take any trajectory out of the labelled OOD atlas and the answer must be right or absent:

    CsCl trajectory       -> ranks CsCl first, and is accepted
    Th3P4 trajectory      -> ranks Th3P4 first, and is accepted
    amorphous trajectory  -> matches NOTHING

and the two external candidates must come out correctly.

This is a stronger test than the 14-family prototype benchmark because the amorphous
trajectories are a real null. "Matches nothing" is not checkable without one: a matcher
that accepts everything scores perfectly on crystals alone.

Everything is scored on ONE number:

    gap = min(D over amorphous) / max(D over correctly-ranked crystals)

  gap > 1  a single ACCEPT_D accepts every CsCl and Th3P4 and rejects every amorphous
           frame. The window is [max crystal D, min amorphous D] and the pipeline is done.
  gap < 1  no threshold separates them; `n_amorphous_accepted` counts what leaks in at
           the tightest threshold that still keeps all the crystals.

ANCHORS ARE NOT PREPROCESSED
----------------------------
The library anchors are built from CIFs and are perfectly ordered, so there is no gas to
screen and no unreliable centre to drop -- every particle the screen would remove is a
false positive. They are read whole.

CANDIDATE preprocessing is the variable under test, because the 14-family benchmark showed
the matcher's failures track how much of the candidate survives it (zincblende_b3: 2048
particles -> 168) rather than tracking anything structural:

    frozen     gas screen + M3 reliable centres, the production path
    gas_only   gas screen only. M3 excludes a centre when exposure > 0.14 AND
               same_coord < 13; in an open framework nothing ever reaches 13 contacts, so
               that clause is vacuous and exposure alone deletes most of the crystal.
    none       every particle, exactly like the anchors

EXTERNAL CANDIDATES
-------------------
steven_example_1/2 are REPORTED, never scored. Their true prototypes live in
crystal_stability/library_stability.csv, which is sealed and is not read here -- the
verdicts are printed for a human to check against it.

  python acceptance_test.py --prep frozen|gas_only|none [--anchors cold|warm]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/projects/rps/gmh4/hockygroup/sz5808")
HERE = ROOT / "results/statistical_ood_benchmark_v2/reference_conditioned_matcher_v1"
CS = ROOT / "crystal_discovery/crystal_stability"
UP = ROOT / "crystal_discovery/preprocessing_pipeline_final/unified_pipeline"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "results/motif_pipeline/soap_atlas_core3_phase2_v1/scripts"))

import core3_pipeline as C3  # noqa: E402
from matcher_benchmark import (  # noqa: E402
    N_PROJ, SUBSAMPLE, anchor_sources, m_soap_chamfer, m_soap_mean, m_soap_meanstd,
    m_soap_mmd, m_soap_sw, _self_k, _sqd,
)
from soap_vs_emd_matcher import anchor_prep, prototype_of  # noqa: E402


def _chamfer_nn(c, a, s_):
    """Per-candidate-particle distance to its nearest anchor environment."""
    return np.sqrt(np.maximum(_sqd(c[s_], a[s_]), 0)).min(axis=1)


def m_chamfer_median(c, a):
    """Chamfer with the MEDIAN instead of the mean.

    A clean CsCl crystal has median nearest-anchor distance 0.117 but p99 = 0.726: about
    1% of its particles -- defects, and surface particles M3 did not catch -- sit far from
    every anchor environment and inflate the plain mean, pushing down the lower edge of
    the acceptance window. Amorphous is uniformly far rather than mostly-close-with-
    outliers, so a robust centre should tighten the crystal side without helping it.
    """
    return float(np.mean([np.median(_chamfer_nn(c, a, s_)) for s_ in (0, 1)]))


def m_chamfer_t90(c, a):
    """Chamfer over the closest 90% of candidate particles; same motive, softer trim."""
    out = []
    for s_ in (0, 1):
        nn = np.sort(_chamfer_nn(c, a, s_))
        out.append(float(nn[:max(1, int(0.9 * len(nn)))].mean()))
    return float(np.mean(out))

SPLIT = "validation"          # never used to fit anything in the matcher
MIN_CENTERS = 25
EXTERNAL = {"steven_example_1": UP / "data/test_set/steven_example_1.gsd",
            "steven_example_2": UP / "data/test_set/steven_example_2.gsd"}


def load_population() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "DATA/detector_trajectories.csv")
    d = d[(d.dataset == "detector_atlas") & (d.role == SPLIT)]
    return d[["trajectory_id", "label", "path"]].reset_index(drop=True)


def centers_for(frame, mode: str):
    """Anchors are always read whole; this applies only to candidates."""
    if mode == "none":
        return anchor_prep(frame), np.arange(len(frame.positions))
    prep = C3.preprocess(frame)
    sel = prep.retain_mask if mode == "gas_only" else prep.is_soap_center
    centers = np.flatnonzero(sel)
    if len(centers) == 0:
        return prep, centers
    comps, counts = np.unique(prep.component_id[centers], return_counts=True)
    return prep, centers[prep.component_id[centers] == comps[np.argmax(counts)]]


def soap_of(gsd_path: Path, frame_index, mode: str, is_anchor: bool):
    frame = C3.read_frame(gsd_path, frame_index)
    if is_anchor:
        prep, centers = anchor_prep(frame), np.arange(len(frame.positions))
    else:
        prep, centers = centers_for(frame, mode)
    if len(centers) == 0:
        return None
    res = C3.soap_global_d0(frame, prep, centers=centers)
    v = res.vectors
    v = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
    sp = prep.canonical_species_id[res.particle_indices]
    out = [v[sp == s] for s in (0, 1)]
    if any(len(x) < MIN_CENTERS for x in out):
        return None
    return out


def last_frame_index(path: Path) -> int:
    import gsd.hoomd
    with gsd.hoomd.open(str(path), "r") as t:
        return len(t) - 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", choices=["frozen", "gas_only", "none"], required=True)
    ap.add_argument("--anchors", choices=["cold", "warm"], default="cold")
    a = ap.parse_args()
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)

    print(f"=== anchors ({a.anchors}, never preprocessed) ===", flush=True)
    anchors = {}
    for name, (path, fi) in anchor_sources(a.anchors).items():
        try:
            r = soap_of(path, fi, a.prep, is_anchor=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {name}: {type(exc).__name__}: {exc}", flush=True); continue
        if r is None:
            print(f"  [skip] {name}: too few centres", flush=True); continue
        anchors[name] = [x if len(x) <= SUBSAMPLE
                         else x[rng.choice(len(x), SUBSAMPLE, replace=False)] for x in r]
    print(f"{len(anchors)} anchors\n", flush=True)

    pop = load_population()
    cases = [(r.trajectory_id, r.label, Path(r.path)) for r in pop.itertuples()]
    cases += [(k, "external", v) for k, v in EXTERNAL.items() if v.is_file()]
    print(f"=== candidates (prep={a.prep}) ===", flush=True)

    cands, dropped = {}, []
    for tid, label, path in cases:
        if not path.is_file():
            dropped.append((tid, label, "missing file")); continue
        try:
            r = soap_of(path, last_frame_index(path), a.prep, is_anchor=False)
        except Exception as exc:  # noqa: BLE001
            dropped.append((tid, label, f"{type(exc).__name__}: {exc}")); continue
        if r is None:
            dropped.append((tid, label, "too few centres")); continue
        cands[tid] = (label, [x if len(x) <= SUBSAMPLE
                              else x[rng.choice(len(x), SUBSAMPLE, replace=False)] for x in r])
        print(f"  {label:10s} {tid[:52]:52s} {len(r[0])}L {len(r[1])}S", flush=True)
    print(f"\n{len(cands)} scored, {len(dropped)} dropped\n", flush=True)

    pool = np.concatenate([x[s] for x in anchors.values() for s in (0, 1)])
    pool = pool[rng.choice(len(pool), min(3000, len(pool)), replace=False)]
    med = float(np.median(np.sqrt(_sqd(pool, pool))[np.triu_indices(len(pool), 1)]))
    gamma = 1.0 / (2 * med ** 2)
    proj = rng.normal(size=(pool.shape[1], N_PROJ)); proj /= np.linalg.norm(proj, axis=0)
    memo: dict = {}
    methods = {
        "soap_mean": lambda c, x, ck, ak: m_soap_mean(c, x),
        "soap_meanstd": lambda c, x, ck, ak: m_soap_meanstd(c, x),
        "soap_mmd": lambda c, x, ck, ak: m_soap_mmd(c, x, gamma, memo, ck, ak),
        "soap_sw": lambda c, x, ck, ak: m_soap_sw(c, x, proj),
        "soap_chamfer": lambda c, x, ck, ak: m_soap_chamfer(c, x),
        "chamfer_median": lambda c, x, ck, ak: m_chamfer_median(c, x),
        "chamfer_t90": lambda c, x, ck, ak: m_chamfer_t90(c, x),
    }

    rows = []
    for tid, (label, c) in cands.items():
        for mname, fn in methods.items():
            D = {}
            for an, x in anchors.items():
                p = prototype_of(an)
                v = fn(c, x, ("c", tid), ("a", an))
                D[p] = min(D.get(p, np.inf), v)
            order = sorted(D, key=D.get)
            rows.append(dict(method=mname, trajectory=tid, label=label,
                             rank1=order[0], D_rank1=round(D[order[0]], 5),
                             rank2=order[1] if len(order) > 1 else None,
                             D_rank2=round(D[order[1]], 5) if len(order) > 1 else np.nan,
                             margin=round(D[order[1]] / max(D[order[0]], 1e-9), 3)
                             if len(order) > 1 else np.nan,
                             correct=(label in ("CsCl", "Th3P4") and order[0] == label)))
        print(f"  scored {tid[:60]}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / f"acceptance_{a.anchors}_{a.prep}.csv", index=False)
    report(df, a, dropped, time.perf_counter() - t0)


def report(df, a, dropped, secs) -> None:
    print(f"\n{'='*92}\nACCEPTANCE TEST   anchors={a.anchors}  candidate_prep={a.prep}\n{'='*92}")
    hdr = (f"{'method':14s} {'CsCl':>9s} {'Th3P4':>9s} {'max D cryst':>12s} "
           f"{'min D amorph':>12s} {'gap':>6s} {'amorph in':>10s}")
    print(hdr); print("-" * len(hdr))
    summary = {}
    for m, g in df.groupby("method"):
        res = {}
        for lab in ("CsCl", "Th3P4"):
            sub = g[g.label == lab]
            res[lab] = f"{int(sub.correct.sum())}/{len(sub)}"
        good = g[(g.label.isin(("CsCl", "Th3P4"))) & g.correct]
        amo = g[g.label == "amorphous"]
        dc = float(good.D_rank1.max()) if len(good) else np.nan
        da = float(amo.D_rank1.min()) if len(amo) else np.nan
        gap = da / dc if dc and dc > 0 else np.nan
        leak = int((amo.D_rank1 <= dc).sum()) if len(amo) else 0
        summary[m] = {"CsCl_rank1": res["CsCl"], "Th3P4_rank1": res["Th3P4"],
                      "max_D_crystal": round(dc, 5), "min_D_amorphous": round(da, 5),
                      "gap_ratio": round(gap, 3),
                      "n_amorphous_accepted": leak, "n_amorphous": len(amo),
                      "accept_D_window": [round(dc, 5), round(da, 5)] if gap > 1 else None,
                      "amorphous_rank1_prototypes":
                          amo.rank1.value_counts().to_dict() if len(amo) else {},
                      "external": {r.trajectory: {"rank1": r.rank1, "D": r.D_rank1,
                                                  "margin": r.margin}
                                   for r in g[g.label == "external"].itertuples()}}
        print(f"{m:14s} {res['CsCl']:>9s} {res['Th3P4']:>9s} {dc:12.4f} {da:12.4f} "
              f"{gap:6.2f} {leak:5d}/{len(amo):<4d}")

    print("\nEXTERNAL CANDIDATES (ground truth is sealed in library_stability.csv "
          "and was not read; verify by hand)")
    for m, g in df.groupby("method"):
        for r in g[g.label == "external"].itertuples():
            w = summary[m]["accept_D_window"]
            verdict = ("accept" if w and r.D_rank1 <= w[0]
                       else "in-window" if w and r.D_rank1 <= w[1] else "reject/unknown")
            print(f"  {m:14s} {r.trajectory:20s} -> {r.rank1:16s} "
                  f"D={r.D_rank1:.4f} margin={r.margin:.2f}  [{verdict}]")

    if dropped:
        print(f"\nDROPPED ({len(dropped)}):")
        for tid, lab, why in dropped[:20]:
            print(f"  {lab:10s} {tid[:52]:52s} {why}")

    out = {"anchors": a.anchors, "candidate_prep": a.prep, "split": SPLIT,
           "runtime_seconds": round(secs, 1),
           "dropped": [{"trajectory": t, "label": l, "reason": w} for t, l, w in dropped],
           "per_method": summary}
    (HERE / f"acceptance_{a.anchors}_{a.prep}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote acceptance_{a.anchors}_{a.prep}.csv / .json")


if __name__ == "__main__":
    main()

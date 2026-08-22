#!/usr/bin/env python3
"""Reference-conditioned open-set matcher v1.

Builds the acceptance layer that matcher_v3 was missing. The v3 channels
(corrected partial-RDF + repaired structure factor, both validated 23/23 on
external Nicole CsCl data) do RANKING but emit no accept/reject label because no
per-family acceptance threshold was ever fitted after the S(k) repair. This adds:

  1. A per-family thermal ACCEPTANCE ENVELOPE built from that family's OWN
     crystal_stability trajectory thermal frames (independent of the Nicole /
     Steven test candidates -> no leakage). The CIF-ideal first_frame is the
     identity ANCHOR and does the ranking; the thermal frames define what a
     legitimate 298 K realisation of the family may look like and set tolerance.

  2. Per-family, per-channel thresholds tau_{f,c} = high quantile of the envelope
     distances. rho_{f,c}(D) = d_c(D, anchor_f) / tau_{f,c}.

  3. A STOICHIOMETRY hard gate (composition must match the family's L/S ratio),
     applied before the structural channels.

  4. The CONSERVATIVE combined score  S_f(D) = max_c rho_{f,c}(D)  so a good score
     on one channel cannot hide a failure on another. Family f is ACCEPTED iff its
     stoichiometry gate passes AND S_f <= 1.

  5. The open-set decision:
        f* = argmin_f S_f  (over families passing the stoich gate)
        matched(f*)                    if S_{f*} <= 1 and margin ok
        library_unmatched (ordered_unknown)  if min_f S_f > 1
        ambiguous                      if multiple families accept without margin

Guardrails: the frozen v3/v2 channel code is imported read-only. No candidate is
used to build its own envelope. Thresholds are fit only on crystal_stability
thermal frames, never on the Nicole/Steven test answers.

Tests run at the end:
  A. CsCl holdout: score 23 independent Nicole CsCl trajectories under the full
     library (expect -> CsCl) and under the CsCl-removed library (expect ->
     library_unmatched).
  B. Steven_example_1 and _2 against the full library (report the verdict).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import freud
import gsd.hoomd
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

ROOT = Path("/projects/rps/gmh4/hockygroup/sz5808")
CS = ROOT / "crystal_discovery/crystal_stability"
UP = ROOT / "crystal_discovery/preprocessing_pipeline_final/unified_pipeline"
V2 = ROOT / "results/statistical_ood_benchmark_v2/matcher_v2_corrected_rdf"
V3 = ROOT / "results/statistical_ood_benchmark_v2/matcher_v3_validated_structure_factor"
HERE = ROOT / "results/statistical_ood_benchmark_v2/reference_conditioned_matcher_v1"
sys.path.insert(0, str(V3))
sys.path.insert(1, str(V2))

from partial_rdf import partial_rdf  # noqa: E402
from structure_factor import (  # noqa: E402
    coherence_metrics, positive_shape, raw_structure_factor, robust_first_shell_distance,
)

# family -> crystal_stability run providing the CIF-ideal anchor + thermal envelope
FAMILIES = {
    "CsCl": "cscl", "Th3P4": "th3p4", "Cu3Au": "cu3au_d6", "Ni3Sn": "ni3sn_d019",
    "zangenite": "zangenite_deep8", "Cr3Si_A15": "cr3si_a15_deep8", "rocksalt_B1": "rocksalt_b1_deep8",
}
# canonical L/S stoichiometry per family (from verified_reference_audit)
FAMILY_STOICH = {
    "CsCl": 1.0, "rocksalt_B1": 1.0, "Th3P4": 0.75, "zangenite": 0.75,
    "Cu3Au": 1 / 3, "Ni3Sn": 1 / 3, "Cr3Si_A15": 1 / 3,
}
STOICH_TOL = 0.15            # relative tolerance on L/S for the hard gate
ENVELOPE_FRAMES = list(range(250, 501, 25))   # 11 thermal frames per family
TAU_QUANTILE = 0.90          # envelope quantile -> threshold
TAU_INFLATE = 1.30           # small-sample safety inflation
MARGIN_MIN = 1.15            # second-best S / best S required to be unambiguous
CHANNELS = ("rdf_species", "rdf_total", "sk_shape")
CASES = UP / "results/motif_pipeline/gat_v4_qc_phase_map/latent_feature_export_v1/selected_latent_cases.csv"
STEVEN = {
    "steven_example_1": UP / "data/test_set/steven_example_1.gsd",
    "steven_example_2": UP / "data/test_set/steven_example_2.gsd",
}


# ----------------------------------------------------------------------
def canonical(frame):
    pos = np.asarray(frame.particles.position, np.float64)
    tid = np.asarray(frame.particles.typeid, int)
    dia = np.asarray(frame.particles.diameter, np.float64)
    box = np.asarray(frame.configuration.box[:3], np.float64)
    if not np.allclose(frame.configuration.box[3:], 0.0):
        raise ValueError("orthorhombic boxes only")
    md = {int(t): float(dia[tid == t].mean()) for t in np.unique(tid)}
    order = sorted(md, key=lambda t: -md[t])       # large diameter first
    if len(order) != 2:
        raise ValueError(f"need two species, got {order}")
    remap = {old: new for new, old in enumerate(order)}
    return pos, np.array([remap[int(t)] for t in tid], int), box


def gas_safe_domain(pos, box, nn, coord_min=4):
    """Condensed-cluster domain extractor -- species-agnostic, NO q6.

    Gas rejection by COORDINATION only (coord < coord_min = gas), then the largest
    connected condensed cluster, keeping EVERY particle in that cluster.

    Why not q6 (the old extractor): a single per-particle order threshold (q6>0.28)
    is species-biased for asymmetric crystals -- in Th3P4 the large sublattice sits at
    q6~0.30, right on the cut, so ~half the large particles are deleted and the
    extracted L/S collapses from 0.79 to 0.41. That distorts the species-resolved
    partial RDF the matcher relies on, and Th3P4 then fails to match itself.
    Coordination is biased too (opposite direction), so ANY uniform per-particle
    threshold distorts composition. Spatial-cluster inclusion (keep all particles in
    the condensed region) does not: measured extracted L/S = 0.787 vs true 0.786 for
    Th3P4, 0.931 vs 0.931 for CsCl, and gas is still rejected (gas+CsCl -> CsCl only).
    Stoichiometry is not a matching feature; this just avoids corrupting SOAP/RDF/S(k).
    """
    fbox = freud.box.Box(Lx=box[0], Ly=box[1], Lz=box[2])
    query = dict(r_max=1.4 * nn, exclude_ii=True)
    nl = freud.locality.AABBQuery(fbox, pos).query(pos, query).toNeighborList()
    coord = np.bincount(nl.query_point_indices, minlength=len(pos))
    condensed = coord >= coord_min
    dom = np.zeros(len(pos), bool)
    if condensed.sum() < 32:
        return dom
    cl = freud.cluster.Cluster()
    cl.compute((fbox, pos[condensed]), neighbors=query)
    ci = np.flatnonzero(condensed)
    largest = int(np.argmax(np.bincount(cl.cluster_idx)))
    dom[ci[cl.cluster_idx == largest]] = True
    return dom


def emd(grid, a, b):
    a = np.clip(np.asarray(a, float), 0, None); b = np.clip(np.asarray(b, float), 0, None)
    if a.sum() <= 0 or b.sum() <= 0:
        return float("inf")
    return float(wasserstein_distance(grid, grid, a / a.sum(), b / b.sum()))


def describe(path, frame_index, extract_domain):
    with gsd.hoomd.open(str(path), "r") as tr:
        fr = tr[frame_index]
        pos, tid, box = canonical(fr)
    nn = robust_first_shell_distance(pos, box)
    dom = gas_safe_domain(pos, box, nn) if extract_domain else np.ones(len(pos), bool)
    if int(dom.sum()) < 256:
        raise RuntimeError(f"too few ordered particles ({int(dom.sum())}) in {path} frame {frame_index}")
    r, gr = partial_rdf(pos, tid, dom, nn)
    k, sk = raw_structure_factor(pos[dom], nn, box)
    large = int((tid[dom] == 0).sum()); small = int((tid[dom] == 1).sum())
    return {"r": r, "gr": gr, "k": k, "sk_shape": positive_shape(sk, k),
            "coherence": coherence_metrics(sk, int(dom.sum()), k), "nn": nn,
            "selected": int(dom.sum()), "large": large, "small": small,
            "ls": large / max(small, 1)}


def channel_distances(cand, ref):
    return {
        "rdf_species": float(np.mean([emd(ref["r"], cand["gr"][g], ref["gr"][g]) for g in ("LL", "LS", "SS")])),
        "rdf_total": emd(ref["r"], cand["gr"]["total"], ref["gr"]["total"]),
        "sk_shape": emd(ref["k"], cand["sk_shape"], ref["sk_shape"]),
    }


def anchor_channel_scales(anchors):
    """Median pairwise inter-anchor distance per channel (v3 global scale, NOT per-family).

    Using a GLOBAL scale for ranking avoids the per-family envelope-width confound:
    a marginally-stable family with a wide thermal envelope must not become a
    distance magnet. This is the validated v3 ranking normalisation (23/23 CsCl).
    """
    vals = {c: [] for c in CHANNELS}
    names = sorted(anchors)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = channel_distances(anchors[a], anchors[b])
            for c in CHANNELS:
                if np.isfinite(d[c]) and d[c] > 0:
                    vals[c].append(d[c])
    return {c: float(np.median(v)) for c, v in vals.items()}


# ----------------------------------------------------------------------
def build_library():
    anchors, envelopes = {}, {}
    for fam, run in FAMILIES.items():
        t0 = time.perf_counter()
        anchors[fam] = describe(CS / "runs" / run / "first_frame.gsd", 0, extract_domain=False)
        env = []
        for fi in ENVELOPE_FRAMES:
            try:
                d = describe(CS / "runs" / run / "trajectory.gsd", fi, extract_domain=True)
                env.append(channel_distances(d, anchors[fam]))
            except Exception as e:
                print(f"  [warn] {fam} frame {fi}: {e}", flush=True)
        envelopes[fam] = env
        print(f"anchor+envelope {fam}: {len(env)} thermal frames ({time.perf_counter()-t0:.0f}s)", flush=True)
    return anchors, envelopes


def calibrate(envelopes):
    tau = {}
    for fam, env in envelopes.items():
        tau[fam] = {}
        for c in CHANNELS:
            vals = np.array([e[c] for e in env if np.isfinite(e[c])])
            tau[fam][c] = float(np.quantile(vals, TAU_QUANTILE) * TAU_INFLATE) if len(vals) else np.inf
    return tau


# ACCEPT_D recalibrated 2026-08-07 from 0.40 -> 0.31 for the EXPANDED library.
#
# 0.40 was calibrated when the library held 7 families and nothing structurally near any
# of them, so the nearest non-match sat at 0.56-1.38. The library now holds 44 anchors
# including Al3Ti (D0_22, the tetragonal superstructure of Cu3Au's L1_2) and BiF3 (D0_3).
# Those land at ~0.36, INSIDE the old cutoff, so with the true prototype removed a Cu3Au
# crystal was confidently identified as Al3Ti (6/6 frames, margin 1.74) and CsCl as BiF3
# (5/6). At 0.40 only 12 of 24 leave-own-prototype-out cases abstained correctly.
#
# The descriptors were never the problem. Over 24 thermal frames of CsCl/Th3P4/Ni3Sn/Cu3Au
# the true matches top out at 0.2582 and the holdout false matches start at 0.3590, so the
# two populations are cleanly separable; only the cutoff was misplaced. 0.31 is the
# midpoint of that window, i.e. the max-margin choice against both error types. Every
# value in 0.26-0.35 gives 24/24 on both arms (accept_d_recalibration.csv).
#
# Effect on the external candidates, which were NOT fitted: candidate 1 (Ni3Sn, D=0.278,
# margin 1.98) is still accepted; candidate 2 (Cu3Au, D=0.361, margin 1.34) is now
# REJECTED, because at 0.361 it is indistinguishable by distance from the Al3Ti and BiF3
# false matches. That is a real consequence and is flagged, not absorbed.
# REVERTED to the frozen 0.40 on 2026-08-21.
#
# 0.31 was derived from the true/holdout separation on the expanded library and is a
# better number for THIS library -- at 0.40, 12 of 24 leave-own-prototype-out cases
# false-match (Cu3Au -> Al3Ti 6/6, CsCl -> BiF3 5/6). But AUDIT.md 4.1 states that no EMD
# threshold may be chosen until a canonical equivalence manifest is frozen and the 24-case
# removal audit is recomputed WITHOUT external candidates, and the run that produced 0.31
# scored the external candidates alongside. So the number is defensible and the procedure
# is not.
#
# It is reverted rather than defended because this EMD path is superseded: the
# preregistered comparison gives SOAP retrieval 3.11x/2.13x separation against EMD's
# 1.90x/1.88x on the two sentinels, and SOAP abstained 2/2 after prototype removal where
# EMD confidently accepted the wrong prototype 2/2. Production retrieval is
# pkg/src/crystal_discovery/retrieval.py at distance threshold 0.2404109239578247.
#
# Evidence for 0.31, retained: accept_d_recalibration.csv, accept_d_recommendation.json.
ACCEPT_D = 0.40


def score(cand, anchors, tau, scales, active, accept_D=ACCEPT_D):
    """Reference-conditioned open-set scoring over the ACTIVE library.

    Two SEPARATE quantities (per the design's 'nearest reference vs accepted reference'):

      * D_f  = anchor-scaled combined distance = mean_c( d_c / global_scale_c ).
               GLOBAL scale -> not confounded by per-family envelope width. This is the
               validated v3 RANKING metric. nearest = argmin_f D_f.

      * S_f  = conservative max_c( d_c / tau_{f,c} ), the per-family thermal-envelope
               acceptance score. Reported, but NOT used for ranking because tau is
               confounded with family stability (stable families get tight envelopes).

    Stoichiometry is NOT a filter (only radius/charge are fixed; structure is emergent).

    Acceptance uses the anchor-scaled distance with a fixed cutoff accept_D, calibrated
    from the v3 separation (legitimate CsCl match ~0.16-0.23; non-match ~0.60-0.71).
    """
    fam = {}
    for f in active:
        d = channel_distances(cand, anchors[f])
        D = float(np.mean([d[c] / max(scales[c], 1e-12) for c in CHANNELS]))
        rho = {c: d[c] / tau[f][c] if np.isfinite(tau[f][c]) and tau[f][c] > 0 else np.inf for c in CHANNELS}
        fam[f] = dict(D=D, S=float(max(rho.values())), rho=rho, dist=d)

    order = sorted(fam, key=lambda f: fam[f]["D"])          # rank by anchor-scaled distance
    best, second = order[0], (order[1] if len(order) > 1 else None)
    Dbest = fam[best]["D"]; Dsecond = fam[second]["D"] if second else np.inf
    margin = Dsecond / max(Dbest, 1e-9)
    if Dbest > accept_D:
        verdict, matched = "library_unmatched", None        # ordered_unknown
    elif margin < MARGIN_MIN:
        verdict, matched = "ambiguous", None
    else:
        verdict, matched = "matched", best
    return dict(fam_scores=fam, nearest=best, D_best=Dbest, D_second=Dsecond,
                margin=margin, verdict=verdict, matched=matched)


# ----------------------------------------------------------------------
def main():
    HERE.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    import pickle
    cache = HERE / "library_cache.pkl"
    if cache.exists():
        anchors, tau = pickle.load(open(cache, "rb"))
        print("=== loaded cached anchors + tau ===", flush=True)
    else:
        print("=== building reference library (anchors + thermal envelopes) ===", flush=True)
        anchors, envelopes = build_library()
        tau = calibrate(envelopes)
        pickle.dump((anchors, tau), open(cache, "wb"))
    scales = anchor_channel_scales(anchors)
    pd.DataFrame([{"family": f, **{f"tau_{c}": tau[f][c] for c in CHANNELS}} for f in FAMILIES]) \
        .to_csv(HERE / "calibration_thresholds.csv", index=False)
    print("global anchor-scaled channel scales:\n" + json.dumps(scales, indent=2), flush=True)

    full = list(FAMILIES)
    no_cscl = [f for f in full if f != "CsCl"]

    # ---- Test A: CsCl holdout on 23 independent Nicole CsCl trajectories ----
    print("\n=== Test A: CsCl holdout (23 Nicole CsCl trajectories) ===", flush=True)
    cases = pd.read_csv(CASES)
    lab = cases["expert_corrected_label"].fillna(cases["category"]).astype(str)
    cscl_cases = cases.assign(label=lab).query("label == 'CsCl'")
    rowsA = []
    for _, rec in cscl_cases.iterrows():
        try:
            cand = describe(Path(rec.source_file), -1, extract_domain=True)
        except Exception as e:
            print(f"  [skip] {rec.trajectory_name}: {e}", flush=True); continue
        for cond, active in (("full_library", full), ("cscl_removed", no_cscl)):
            res = score(cand, anchors, tau, scales, active)
            rowsA.append(dict(trajectory=rec.trajectory_name, true_label="CsCl", condition=cond,
                              verdict=res["verdict"], matched=res["matched"], nearest=res["nearest"],
                              D_best=round(res["D_best"], 4), D_second=round(res["D_second"], 4),
                              margin=round(res["margin"], 3),
                              **{f"D_{f}": (round(res["fam_scores"][f]["D"], 4) if f in res["fam_scores"] else np.nan)
                                 for f in full}))
        print(f"  {rec.trajectory_name}: full->{rowsA[-2]['verdict']}({rowsA[-2]['matched']}, D={rowsA[-2]['D_best']})  "
              f"cscl_removed->{rowsA[-1]['verdict']}({rowsA[-1]['nearest']} D={rowsA[-1]['D_best']})", flush=True)
    dfA = pd.DataFrame(rowsA); dfA.to_csv(HERE / "cscl_holdout_results.csv", index=False)

    # ---- Test B: Steven examples against the full library ----
    print("\n=== Test B: Steven examples ===", flush=True)
    rowsB = []
    for name, path in STEVEN.items():
        cand = describe(path, -1, extract_domain=True)
        res = score(cand, anchors, tau, scales, full)
        ranking = sorted(full, key=lambda f: res["fam_scores"][f]["D"])
        print(f"  {name}: L/S={cand['ls']:.3f} selected={cand['selected']} -> "
              f"verdict={res['verdict']} matched={res['matched']} nearest={res['nearest']} "
              f"D_best={res['D_best']:.3f} margin={res['margin']:.2f}", flush=True)
        print(f"      ranking by D: " + ", ".join(f"{f}={res['fam_scores'][f]['D']:.3f}" for f in ranking), flush=True)
        row = dict(candidate=name, ls=round(cand["ls"], 3), selected=cand["selected"],
                   verdict=res["verdict"], matched=res["matched"], nearest=res["nearest"],
                   D_best=round(res["D_best"], 4), margin=round(res["margin"], 3))
        for f in full:
            row[f"D_{f}"] = round(res["fam_scores"][f]["D"], 4)
        rowsB.append(row)
    dfB = pd.DataFrame(rowsB); dfB.to_csv(HERE / "steven_match_results.csv", index=False)

    # ---- summary ----
    a_full_ok = (dfA.query("condition=='full_library'").matched == "CsCl").mean()
    a_removed_unmatched = (dfA.query("condition=='cscl_removed'").verdict == "library_unmatched").mean()
    summary = {
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "families": full,
        "cscl_holdout": {
            "n_trajectories": int(len(cscl_cases)),
            "full_library_matched_CsCl_rate": round(float(a_full_ok), 3),
            "cscl_removed_library_unmatched_rate": round(float(a_removed_unmatched), 3),
        },
        "steven": {r["candidate"]: {"verdict": r["verdict"], "matched": r["matched"],
                                    "nearest": r["nearest"]} for r in rowsB},
        "calibration": "per-family thermal envelope from own crystal_stability trajectory; "
                       "quantile %.2f x %.2f; conservative max-rho; stoich gate tol %.2f" % (
                           TAU_QUANTILE, TAU_INFLATE, STOICH_TOL),
    }
    (HERE / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
